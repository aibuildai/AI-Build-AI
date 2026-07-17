"""Standalone inference for the anomalous-diffusion OOD detector.

Reads the test trajectories of every evaluation instance under DATA_DIR, loads
best_model.pth, and writes one output/<instance>/predictions.npz per instance
with (ood_scores, model_predictions, alpha_predictions) -- the exact submission
that scored mean AUROC 0.820 (g +0.5838, surpass_sota).

The out-of-distribution score is family- and instance-specific: probability mass
on the in-distribution families for the four-family instances, a Mahalanobis
distance for `acfls_ou`, and a CDF-calibrated blend of embedding-density,
centroid, and physics scores for the remaining five-family instances. Building
the density banks needs the training corpus, so it is regenerated with
`andi-datasets` on first run (a few minutes, cached under ./_generated).

Run:  python inference.py --input /path/to/data --output ./output
      # or set DATA_DIR / OUTPUT_DIR in the environment
"""
import os
import json
import argparse

import numpy as np
import torch
from sklearn.covariance import LedoitWolf

import config as C
from model import Net, physics, infer
from train import generate_corpus, load_train_arrays

HERE = os.path.dirname(os.path.abspath(__file__))


# ---- out-of-distribution scoring primitives ---------------------------------

def bank_radii(bank, ks=C.KNN_KS, chunk=1024):
    """Per-point local scale: cosine deficit to the k-th nearest bank neighbour."""
    bt = torch.from_numpy(bank).cuda()
    values = {k: [] for k in ks}
    with torch.inference_mode():
        for s in range(0, len(bank), chunk):
            sim = bt[s:s + chunk] @ bt.T
            row = torch.arange(len(sim), device="cuda")
            sim[row, s + row] = -2
            top = sim.topk(max(ks), dim=1).values
            for k in ks:
                values[k].append((1 - top[:, k - 1]).clamp_min(1e-5).cpu())
    return {k: torch.cat(v).numpy() for k, v in values.items()}


def density_scores(q, bank, radii, ks=C.KNN_KS, chunk=2048):
    """Raw cosine-neighbour, scale-normalized ratio, and log-ratio density scores."""
    bt = torch.from_numpy(bank).cuda()
    rt = {k: torch.from_numpy(v).cuda() for k, v in radii.items()}
    raw = {k: [] for k in ks}
    ratio = {k: [] for k in ks}
    logratio = {k: [] for k in ks}
    with torch.inference_mode():
        for s in range(0, len(q), chunk):
            sim, idx = (torch.from_numpy(q[s:s + chunk]).cuda() @ bt.T).topk(max(ks), dim=1)
            deficit = (1 - sim).clamp_min(1e-6)
            for k in ks:
                scale = rt[k][idx[:, :k]].clamp_min(1e-5)
                r = deficit[:, :k] / scale
                raw[k].append(sim[:, :k].mean(1).cpu())
                ratio[k].append((-r.mean(1)).cpu())
                logratio[k].append((-torch.log(r).mean(1)).cpu())
    unpack = lambda d: {k: torch.cat(v).numpy() for k, v in d.items()}
    return unpack(raw), unpack(ratio), unpack(logratio)


def empirical_cdf(values, reference):
    ref = np.sort(np.asarray(reference))
    return np.searchsorted(ref, values, side="right").astype(np.float64) / len(ref)


def model_cdf(values, pred, ref_values, ref_pred):
    """Per-predicted-family empirical CDF."""
    values = np.asarray(values); pred = np.asarray(pred)
    ref_values = np.asarray(ref_values); ref_pred = np.asarray(ref_pred)
    out = np.empty(len(values), np.float64)
    for cls in np.unique(pred):
        take = pred == cls
        out[take] = empirical_cdf(values[take], ref_values[ref_pred == cls])
    return out


def conditional_cdf(values, pred, alpha, ref_values, ref_pred, ref_alpha, width):
    """Empirical CDF conditioned on predicted family AND exponent bin."""
    values = np.asarray(values); pred = np.asarray(pred)
    bins = np.floor(np.asarray(alpha) / width).astype(np.int64)
    ref_values = np.asarray(ref_values); ref_pred = np.asarray(ref_pred)
    ref_bins = np.floor(np.asarray(ref_alpha) / width).astype(np.int64)
    out = np.empty(len(values), np.float64)
    for cls in np.unique(pred):
        class_ref = ref_pred == cls
        for b in np.unique(bins[pred == cls]):
            take = (pred == cls) & (bins == b)
            reference = ref_values[class_ref & (ref_bins == b)]
            if len(reference) < 100:
                reference = ref_values[class_ref]
            out[take] = empirical_cdf(values[take], reference)
    return out


def local_scores(emb, phy, stats):
    """Best centroid alignment and best physics-typicality over exponent bins."""
    centroid, typical = [], []
    z = (phy - stats["pm"]) / stats["ps"]
    for center, mu, sd in stats["bins"]:
        centroid.append(emb @ center)
        typical.append(-np.mean(((z - mu) / sd) ** 2, axis=1))
    return np.max(np.stack(centroid, 1), 1), np.max(np.stack(typical, 1), 1)


# ---- reference statistics built once from the training corpus ---------------

def build_references(model, tx, ta, P, pm, ps, vx):
    """Density banks, per-family Gaussians and exponent-bin stats, and the
    validation-derived CDF references the instance scoring calibrates against."""
    banks, gaussian, model_stats, validation = {}, {}, {}, {}
    for yi, m in enumerate(C.MODELS):
        sel = np.linspace(0, C.TRAIN_PER_MODEL - 1, C.BANK_PER_MODEL, dtype=int)
        e, _, _ = infer(model, tx[yi][sel], pm, ps, views=1)
        banks[m] = e
        pmodel = P[yi * C.TRAIN_PER_MODEL:(yi + 1) * C.TRAIN_PER_MODEL]
        fit = LedoitWolf().fit(pmodel)
        gaussian[m] = (fit.location_.astype(np.float32), fit.precision_.astype(np.float32))
        z = (pmodel - pm) / ps
        bins = []
        bin_id = np.floor(ta[yi] / C.ALPHA_BIN_WIDTH).astype(int)
        selected_bins = np.floor(ta[yi][sel] / C.ALPHA_BIN_WIDTH).astype(int)
        for b in np.unique(bin_id):
            take = bin_id == b
            etake = selected_bins == b
            center = e[etake].mean(0)
            center /= np.linalg.norm(center) + 1e-8
            bins.append((center.astype(np.float32), z[take].mean(0).astype(np.float32),
                         np.maximum(z[take].std(0), .25).astype(np.float32)))
        model_stats[m] = {"pm": pm, "ps": ps, "bins": bins}
        validation[m] = infer(model, vx[yi], pm, ps, views=C.TTA_VIEWS)

    all_bank = np.concatenate([banks[m] for m in C.MODELS])
    radii = bank_radii(all_bank)
    combined = {"pm": pm, "ps": ps, "bins": sum([model_stats[m]["bins"] for m in C.MODELS], [])}
    vemb = np.concatenate([validation[m][0] for m in C.MODELS])
    vprob = np.concatenate([validation[m][1] for m in C.MODELS])
    valpha = np.concatenate([validation[m][2] for m in C.MODELS])
    vpred = np.argmax(vprob, axis=1)
    vphy = physics(np.concatenate(vx))
    vraw, vratio, vlog = density_scores(vemb, all_bank, radii)
    vcent, vtyp = local_scores(vemb, vphy, combined)
    references = {"knn": vraw[20], "centroid": vcent, "physics": vtyp,
                  **{"raw" + str(k): vraw[k] for k in vraw},
                  **{"ratio" + str(k): vratio[k] for k in vratio},
                  **{"log" + str(k): vlog[k] for k in vlog}}
    return banks, all_bank, radii, gaussian, model_stats, references, vraw, vpred, valpha


def score_instance(name, emb, prob, alpha, ptest, ids, cols, banks, all_bank,
                   radii, gaussian, model_stats, references, vraw, vpred, valpha):
    """The out-of-distribution score for one instance (higher = more in-distribution)."""
    # Four-family instances: probability mass on the in-distribution families.
    if len(ids) == 4:
        return prob[:, cols].sum(1).astype(np.float64)
    # acfls_ou: negative minimum Mahalanobis distance to the family Gaussians.
    if name == "acfls_ou":
        distances = []
        for m in ids:
            mu, precision = gaussian[m]
            q = ptest - mu
            distances.append(np.einsum("ni,ij,nj->n", q, precision, q))
        return -np.min(np.stack(distances, 1), 1).astype(np.float64)
    # Remaining five-family instances: CDF-calibrated density blends.
    traw, tratio, tlog = density_scores(emb, all_bank, radii)
    local = {"pm": model_stats[C.MODELS[0]]["pm"], "ps": model_stats[C.MODELS[0]]["ps"],
             "bins": sum([model_stats[m]["bins"] for m in ids], [])}
    raw_cent, raw_typ = local_scores(emb, ptest, local)
    ck = empirical_cdf(traw[20], references["knn"])
    cc = empirical_cdf(raw_cent, references["centroid"])
    cp = empirical_cdf(raw_typ, references["physics"])
    cr20 = empirical_cdf(traw[20], references["raw20"])
    cd20 = empirical_cdf(tratio[20], references["ratio20"])
    if name == "acfls_dbm":
        return (ck + cc + cp) / 3
    # The three calibrated instances add an exponent-conditional residual.
    global_pred = np.argmax(prob, axis=1)
    if name == "acfls_cbm":
        selected = .65 * cr20 + .35 * cd20
        baseline, k, w = selected, 50, .25
        joint = conditional_cdf(traw[k], global_pred, alpha, vraw[k], vpred, valpha, .4)
        model_score = model_cdf(traw[k], global_pred, vraw[k], vpred)
        return baseline + w * (model_score - joint)
    if name == "acfls_sinai":
        selected = .5 * cr20 + .5 * cd20
        baseline, k, w = selected, 5, .90
        joint = conditional_cdf(traw[k], global_pred, alpha, vraw[k], vpred, valpha, .2)
        model_score = model_cdf(traw[k], global_pred, vraw[k], vpred)
        return baseline + w * (model_score - joint)
    if name == "acfls_tsm":
        baseline = empirical_cdf(traw[5], references["raw5"])
        k, w = 5, .45
        joint = conditional_cdf(traw[k], global_pred, alpha, vraw[k], vpred, valpha, .4)
        model_score = model_cdf(traw[k], global_pred, vraw[k], vpred)
        return baseline + w * (model_score - joint)
    return traw[20]  # unreached for the defined instances; safe default


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=os.environ.get("DATA_DIR"))
    ap.add_argument("--output", default=os.environ.get("OUTPUT_DIR", os.path.join(HERE, "output")))
    ap.add_argument("--weights", default=os.path.join(HERE, "best_model.pth"))
    args = ap.parse_args()

    data = args.input
    if data is None:
        raise SystemExit("set --input or DATA_DIR to the instance data directory")
    # Accept either <data>/<instance>/ or <data>/problem/data/<instance>/.
    if not os.path.isdir(os.path.join(data, C.INSTANCES[0])) and \
       os.path.isdir(os.path.join(data, "problem", "data")):
        data = os.path.join(data, "problem", "data")
    os.makedirs(args.output, exist_ok=True)

    torch.set_float32_matmul_precision("high")
    checkpoint = torch.load(args.weights, map_location="cpu", weights_only=False)
    pm = checkpoint["mean"]
    ps = checkpoint["std"]

    cache = os.path.join(HERE, "_generated")
    generate_corpus(cache)
    tx, ta, _, vx = load_train_arrays(cache)
    P = physics(np.concatenate(tx))

    model = Net(P.shape[1]).cuda()
    model.load_state_dict(checkpoint["model"])
    model.eval()

    refs = build_references(model, tx, ta, P, pm, ps, vx)
    banks, all_bank, radii, gaussian, model_stats, references, vraw, vpred, valpha = refs

    for name in C.INSTANCES:
        with open(os.path.join(data, name, "instance_info.json")) as f:
            info = json.load(f)
        xt = np.load(os.path.join(data, name, "x_test.npy")).reshape(-1, 2, C.TRAJ_LEN).astype(np.float32)
        emb, prob, alpha = infer(model, xt, pm, ps)
        ids = info["id_models"]
        cols = [C.MODELS.index(m) for m in ids]
        pred = np.argmax(prob[:, cols], 1).astype(np.int64)
        ptest = physics(xt)
        score = score_instance(name, emb, prob, alpha, ptest, ids, cols, banks,
                               all_bank, radii, gaussian, model_stats, references,
                               vraw, vpred, valpha)
        dest = os.path.join(args.output, name)
        os.makedirs(dest, exist_ok=True)
        np.savez_compressed(os.path.join(dest, "predictions.npz"),
                            ood_scores=np.asarray(score, dtype=np.float64),
                            model_predictions=pred,
                            alpha_predictions=alpha.astype(np.float64))
        print("wrote", dest, flush=True)


if __name__ == "__main__":
    main()

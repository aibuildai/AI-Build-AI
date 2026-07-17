"""Model + feature extraction for Protein Variant Effect Prediction.

Pipeline per assay (instance):
  1. Reconstruct the wild-type sequence from the variant records.
  2. ESM-2 (frozen) masked-marginal log-prob table over the sequence.
  3. Per-position structural features from the assay's PDB.
  4. Assemble per-variant features: site-independent one-hot substitution
     indicators + aggregated ESM / structural / biophysical descriptors.
  5. Fit a LightGBM (L2 + L1) ensemble and a Ridge member on the same matrix,
     then choose {mean(L2,L1)} vs {0.5*GBM + 0.5*Ridge} by train-only holdout
     Spearman.

`fit_predict_lgbm` also returns the fitted head so train.py can serialize a
checkpoint; `predict_with_bundle` reloads a checkpoint bundle and reproduces the
predictions with NO retraining, NO ESM, and NO GPU (all feature state is baked
into the bundle).
"""
import re
import numpy as np
from scipy import sparse

import config

# --- amino-acid alphabet + physico-chemical constants (feature extraction) ---
AA = "ACDEFGHIKLMNPQRSTVWY"
AA_IDX = {a: i for i, a in enumerate(AA)}
AA3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
}
# Kyte-Doolittle hydropathy
KD = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5,
    "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8,
    "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}
# residue volume (A^3)
VOL = {
    "A": 88.6, "R": 173.4, "N": 114.1, "D": 111.1, "C": 108.5, "Q": 143.8,
    "E": 138.4, "G": 60.1, "H": 153.2, "I": 166.7, "L": 166.7, "K": 168.6,
    "M": 162.9, "F": 189.9, "P": 112.7, "S": 89.0, "T": 116.1, "W": 227.8,
    "Y": 193.6, "V": 140.0,
}
CHARGE = {a: 0.0 for a in AA}
CHARGE.update({"D": -1.0, "E": -1.0, "K": 1.0, "R": 1.0, "H": 0.5})

try:
    from Bio.Align import substitution_matrices
    _B62 = substitution_matrices.load("BLOSUM62")

    def blosum62(a, b):
        try:
            return float(_B62[a, b])
        except Exception:
            return 0.0
except Exception:
    def blosum62(a, b):
        return 4.0 if a == b else -1.0

VAR_RE = re.compile(r"^([A-Z])(\d+)([A-Z])$")


# --------------------------------------------------------------------------- #
# variant / data parsing
# --------------------------------------------------------------------------- #
def parse_variant(v):
    """Return list of (wt_aa, pos, mut_aa)."""
    out = []
    for m in v.split(","):
        m = m.strip()
        mm = VAR_RE.match(m)
        if mm:
            out.append((mm.group(1), int(mm.group(2)), mm.group(3)))
    return out


def read_tsv(path, has_score):
    variants, nmut = [], []
    scores = [] if has_score else None
    with open(path) as f:
        f.readline()  # header
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if not parts or parts[0] == "":
                continue
            variants.append(parts[0])
            nmut.append(int(parts[1]))
            if has_score:
                scores.append(float(parts[2]))
    return variants, np.asarray(nmut, dtype=np.float32), (
        np.asarray(scores, dtype=np.float32) if has_score else None
    )


def reconstruct_wt(variants_list):
    """Reconstruct WT map {pos: aa} from all single-mutation records."""
    wt = {}
    for variants in variants_list:
        for v in variants:
            for (w, p, mt) in parse_variant(v):
                wt[p] = w
    return wt


# --------------------------------------------------------------------------- #
# PDB structural features
# --------------------------------------------------------------------------- #
def compute_pdb_features(pdb_path, wt_map):
    """Per-position structural features keyed by VARIANT position:
    [sasa, bfactor, n_contacts, depth]. Aligns the reconstructed WT sequence to
    the PDB residues to fix numbering offset. Missing -> NaN (mean-imputed)."""
    from Bio.PDB import PDBParser
    from Bio.PDB.SASA import ShrakeRupley

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("x", pdb_path)
    model = structure[0]

    best_res = []
    for chain in model:
        res = [r for r in chain if r.id[0] == " " and r.resname in AA3TO1]
        if len(res) > len(best_res):
            best_res = res
    residues = best_res
    if not residues:
        return {}

    pdb_seq = "".join(AA3TO1[r.resname] for r in residues)
    ps = sorted(wt_map)
    wt_seq = "".join(wt_map[p] for p in ps)
    offset = _best_offset(wt_seq, pdb_seq)

    try:
        ShrakeRupley().compute(model, level="R")
    except Exception:
        pass

    ca_coords = []
    for r in residues:
        if "CA" in r:
            ca_coords.append(r["CA"].get_coord())
        else:
            ca_coords.append(np.array([np.nan, np.nan, np.nan], dtype=np.float32))
    ca = np.asarray(ca_coords, dtype=np.float32)
    centroid = np.nanmean(ca, axis=0)

    n = len(residues)
    valid_ca = ~np.isnan(ca[:, 0])
    contacts = np.zeros(n, dtype=np.float32)
    if valid_ca.sum() > 1:
        cav = ca[valid_ca]
        d = np.sqrt(((cav[:, None, :] - cav[None, :, :]) ** 2).sum(-1))
        cnt = ((d < 10.0) & (d > 0)).sum(1).astype(np.float32)  # CA<10A ~ heavy<8A packing
        contacts[valid_ca] = cnt

    depth = np.linalg.norm(ca - centroid[None, :], axis=1).astype(np.float32)

    feats = {}
    for i, r in enumerate(residues):
        wi = i - offset
        if wi < 0 or wi >= len(ps):
            continue
        pos = ps[wi]
        sasa = getattr(r, "sasa", np.nan)
        try:
            bfac = np.mean([a.get_bfactor() for a in r])
        except Exception:
            bfac = np.nan
        feats[pos] = np.array(
            [sasa if sasa is not None else np.nan, bfac, contacts[i], depth[i]],
            dtype=np.float32,
        )
    return feats


def _best_offset(wt_seq, pdb_seq):
    """Integer offset o with pdb index = wt index + o, maximizing matches."""
    best_o, best_score = 0, -1
    Lw, Lp = len(wt_seq), len(pdb_seq)
    for o in range(-5, min(30, Lp) + 1):
        score = 0
        for wi in range(Lw):
            pi = wi + o
            if 0 <= pi < Lp and pdb_seq[pi] == wt_seq[wi]:
                score += 1
        if score > best_score:
            best_score, best_o = score, o
    return best_o


# --------------------------------------------------------------------------- #
# ESM-2 zero-shot masked-marginal log-probs
# --------------------------------------------------------------------------- #
_ESM = {}


def get_esm():
    if "model" not in _ESM:
        import torch
        from transformers import AutoTokenizer, EsmForMaskedLM
        name = config.ESM_MODEL_NAME
        tok = AutoTokenizer.from_pretrained(name)
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        model = EsmForMaskedLM.from_pretrained(name, torch_dtype=dtype)
        if torch.cuda.is_available():
            model = model.cuda()
        model.eval()
        _ESM.update(model=model, tok=tok, torch=torch)
    return _ESM["model"], _ESM["tok"], _ESM["torch"]


def esm_logprob_table(wt_seq):
    """Return masked-marginal log-prob table [L, 20] over the AA alphabet."""
    model, tok, torch = get_esm()
    enc = tok(wt_seq, return_tensors="pt")
    dev = next(model.parameters()).device
    input_ids = enc["input_ids"].to(dev)
    attn = enc["attention_mask"].to(dev)
    L = len(wt_seq)
    mask_id = tok.mask_token_id
    aa_token_ids = torch.tensor([tok.convert_tokens_to_ids(a) for a in AA], device=dev)

    table = np.zeros((L, 20), dtype=np.float32)
    bs = 32
    base = input_ids[0]
    for start in range(0, L, bs):
        end = min(start + bs, L)
        rows = end - start
        batch = base.unsqueeze(0).repeat(rows, 1).clone()
        for j, i in enumerate(range(start, end)):
            batch[j, i + 1] = mask_id  # +1 for CLS
        bmask = attn.repeat(rows, 1)
        with torch.no_grad():
            out = model(input_ids=batch, attention_mask=bmask).logits
        for j, i in enumerate(range(start, end)):
            lp = torch.log_softmax(out[j, i + 1].float(), dim=-1)
            table[i] = lp[aa_token_ids].cpu().numpy()
    return table


def esm_position_stats(esm_table):
    """Per-position entropy + per-AA rank from the log-prob table [L,20]."""
    if esm_table is None:
        return None
    lp = esm_table.astype(np.float64)
    m = lp.max(axis=1, keepdims=True)
    ex = np.exp(lp - m)
    p = ex / ex.sum(axis=1, keepdims=True)
    entropy = -(p * np.log(p + 1e-12)).sum(axis=1)
    order = np.argsort(-lp, axis=1)
    aa_rank = np.empty_like(lp)
    rows = np.arange(lp.shape[0])[:, None]
    aa_rank[rows, order] = np.arange(20)[None, :]
    return {"entropy": entropy.astype(np.float32), "aa_rank": aa_rank.astype(np.float32)}


# --------------------------------------------------------------------------- #
# feature assembly
# --------------------------------------------------------------------------- #
def build_features(variants, nmut, wt_map, struct_feats, esm_table, esm_pos_index,
                   onehot_vocab, esm_pos_stats=None):
    """Return a CSR matrix: [one-hot substitution indicators | dense aggregates]."""
    N = len(variants)

    rows, cols = [], []
    for r, v in enumerate(variants):
        for (w, p, mt) in parse_variant(v):
            c = onehot_vocab.get((p, mt))
            if c is not None:
                rows.append(r)
                cols.append(c)
    onehot = sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(N, len(onehot_vocab)), dtype=np.float32,
    )

    have_pos = esm_table is not None and esm_pos_stats is not None
    n_pos_cols = 20 if have_pos else 0
    dense = np.zeros((N, 20 + n_pos_cols), dtype=np.float32)
    if struct_feats:
        sf = np.stack(list(struct_feats.values()))
        struct_mean = np.nanmean(sf, axis=0)
        struct_mean = np.where(np.isnan(struct_mean), 0.0, struct_mean)
    else:
        struct_mean = np.zeros(4, dtype=np.float32)

    for r, v in enumerate(variants):
        muts = parse_variant(v)
        if not muts:
            continue
        s_stack, esm_scores = [], []
        pp_entropy, pp_rank, pp_wtlp, pp_mutlp, pp_delta = [], [], [], [], []
        d_kd, d_vol, d_charge, blos = [], [], [], []
        pro_gly = 0.0
        for (w, p, mt) in muts:
            sfv = struct_feats.get(p)
            s_stack.append(sfv if sfv is not None else struct_mean)
            if esm_table is not None and p in esm_pos_index:
                ei = esm_pos_index[p]
                lp = esm_table[ei]
                if w in AA_IDX and mt in AA_IDX:
                    wlp = float(lp[AA_IDX[w]])
                    mlp = float(lp[AA_IDX[mt]])
                    esm_scores.append(mlp - wlp)
                    if have_pos:
                        pp_entropy.append(float(esm_pos_stats["entropy"][ei]))
                        pp_rank.append(float(esm_pos_stats["aa_rank"][ei, AA_IDX[mt]]))
                        pp_wtlp.append(wlp)
                        pp_mutlp.append(mlp)
                        pp_delta.append(mlp - wlp)
            d_kd.append(KD.get(mt, 0) - KD.get(w, 0))
            d_vol.append(VOL.get(mt, 0) - VOL.get(w, 0))
            d_charge.append(CHARGE.get(mt, 0) - CHARGE.get(w, 0))
            blos.append(blosum62(w, mt))
            if mt in ("P", "G") or w in ("P", "G"):
                pro_gly += 1.0

        s_arr = np.stack(s_stack)
        f = []
        f.extend(np.nanmean(s_arr, axis=0).tolist())
        f.extend(np.nanmin(s_arr, axis=0).tolist())
        f.extend(np.nanmax(s_arr, axis=0).tolist())  # 12 struct
        f.extend([
            float(np.sum(esm_scores)) if esm_scores else 0.0,
            float(np.mean(esm_scores)) if esm_scores else 0.0,
            float(np.min(esm_scores)) if esm_scores else 0.0,
        ])  # 3 esm
        f.append(float(np.sum(d_kd)))
        f.append(float(np.sum(d_vol)))
        f.append(float(np.sum(d_charge)))
        f.append(float(np.mean(blos)) if blos else 0.0)
        f.append(pro_gly)  # 5 biophys -> 20 total

        if have_pos:
            for vals in (pp_entropy, pp_rank, pp_wtlp, pp_mutlp, pp_delta):
                if vals:
                    f.extend([float(np.mean(vals)), float(np.min(vals)),
                              float(np.max(vals)), float(np.sum(vals))])
                else:
                    f.extend([0.0, 0.0, 0.0, 0.0])
        dense[r] = np.array(f, dtype=np.float32)

    dense = np.nan_to_num(dense, nan=0.0, posinf=0.0, neginf=0.0)
    dense = np.hstack([dense, nmut.reshape(-1, 1)]).astype(np.float32)
    return sparse.hstack([onehot, sparse.csr_matrix(dense)], format="csr")


def _spearman(a, b):
    from scipy.stats import rankdata
    a = rankdata(a)
    b = rankdata(b)
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    return 0.0 if denom == 0 else float((a * b).sum() / denom)


# --------------------------------------------------------------------------- #
# model: LightGBM (L2+L1) + Ridge blend
# --------------------------------------------------------------------------- #
def fit_predict_lgbm(Xtr, ytr, Xte, seed=config.SEED, fast_gb1=False):
    """Fit L2+L1 LightGBM boosters + a Ridge member; select {mean(L2,L1)} vs
    {0.5*GBM + 0.5*Ridge} by train-only-holdout Spearman.

    Returns (pred_test, boosters, holdout_spearman, head) where `head` carries the
    fitted Ridge model + blend decision so a checkpoint can be serialized.
    fast_gb1: force_col_wise + reuse L2 rounds for L1 (gb1 speed)."""
    import lightgbm as lgb

    rng = np.random.RandomState(seed)
    n = Xtr.shape[0]
    idx = rng.permutation(n)
    n_val = max(200, int(0.1 * n))
    val_idx, sub_idx = idx[:n_val], idx[n_val:]
    Xsub, ysub = Xtr[sub_idx], ytr[sub_idx]
    Xval, yval = Xtr[val_idx], ytr[val_idx]

    params_common = dict(config.LGBM_PARAMS)
    if fast_gb1:
        params_common["force_col_wise"] = True

    gbm_val_preds, gbm_te_preds, boosters = [], [], []
    l2_best_rounds = None
    for obj in ("regression", "regression_l1"):
        params = dict(params_common, objective=obj)
        if fast_gb1 and obj == "regression_l1" and l2_best_rounds is not None:
            best_rounds = l2_best_rounds
            m = lgb.train(params, lgb.Dataset(Xsub, label=ysub), num_boost_round=best_rounds)
        else:
            round_cap = config.LGBM_ROUND_CAP_GB1 if fast_gb1 else config.LGBM_ROUND_CAP
            dsub = lgb.Dataset(Xsub, label=ysub)
            dval = lgb.Dataset(Xval, label=yval, reference=dsub)
            m = lgb.train(
                params, dsub, num_boost_round=round_cap, valid_sets=[dval],
                callbacks=[lgb.early_stopping(config.LGBM_EARLY_STOPPING_ROUNDS, verbose=False)],
            )
            best_rounds = m.best_iteration if m.best_iteration and m.best_iteration > 0 else 500
            if obj == "regression":
                l2_best_rounds = best_rounds
        gbm_val_preds.append(m.predict(Xval))
        mf = lgb.train(params, lgb.Dataset(Xtr, label=ytr), num_boost_round=best_rounds)
        gbm_te_preds.append(mf.predict(Xte))
        boosters.append(mf)

    gbm_val = np.mean(gbm_val_preds, axis=0)
    gbm_te = np.mean(gbm_te_preds, axis=0)

    from sklearn.linear_model import Ridge
    ridge_val = ridge_te = best_alpha = None
    ridge_model = None
    try:
        best_ridge_sp = -2.0
        for alpha in config.RIDGE_ALPHAS:
            rr = Ridge(alpha=alpha, random_state=seed)
            rr.fit(Xsub, ysub)
            pv = rr.predict(Xval)
            sp = _spearman(pv, yval)
            if sp > best_ridge_sp:
                best_ridge_sp, best_alpha, ridge_val = sp, alpha, pv
        rr = Ridge(alpha=best_alpha, random_state=seed)
        rr.fit(Xtr, ytr)
        ridge_te = rr.predict(Xte)
        ridge_model = rr
    except Exception:
        ridge_val = ridge_te = None

    sp_gbm = _spearman(gbm_val, yval)
    chosen, sp_chosen, pred = "gbm", sp_gbm, gbm_te
    if ridge_val is not None and ridge_te is not None:
        blend_val = 0.5 * gbm_val + 0.5 * ridge_val
        sp_blend = _spearman(blend_val, yval)
        if sp_blend > sp_gbm:
            chosen, sp_chosen, pred = "blend", sp_blend, 0.5 * gbm_te + 0.5 * ridge_te

    head = {"ridge_model": ridge_model, "chosen": chosen, "best_alpha": best_alpha}
    return pred, boosters, sp_chosen, head


def predict_with_bundle(bundle, variants, nmut):
    """Reproduce an instance's predictions from a saved checkpoint bundle.

    Uses ONLY the state baked into the bundle (fitted boosters + ridge + blend
    choice + the train-derived one-hot vocab, ESM log-prob table/stats, PDB
    structural features, WT map) -- so NO retraining, NO ESM forward pass, and NO
    GPU are needed at inference time."""
    Xte = build_features(
        variants, nmut, bundle["wt_map"], bundle["struct_feats"],
        bundle["esm_table"], bundle["esm_pos_index"],
        bundle["onehot_vocab"], bundle["esm_pos_stats"],
    )
    gbm = np.mean([bo.predict(Xte) for bo in bundle["boosters"]], axis=0)
    if bundle["chosen"] == "blend" and bundle["ridge_model"] is not None:
        return 0.5 * gbm + 0.5 * bundle["ridge_model"].predict(Xte)
    return gbm

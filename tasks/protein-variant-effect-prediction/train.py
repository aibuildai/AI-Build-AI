"""Train the Protein Variant Effect Prediction solution.

    python train.py [--data-dir DIR] [--output-dir DIR]

For each of the 11 DMS assays it builds ESM-2 + structure + biophysics features,
fits the LightGBM(L2+L1)+Ridge blend, writes per-instance predictions, and
serializes a single `best_model.pkl` holding every fitted head plus the feature
state needed to reproduce predictions at inference time.

--data-dir must contain one subdirectory per instance, each with train.tsv and
test.tsv (variant<TAB>n_mut<TAB>score) and one .pdb structure file. Defaults to
the staged task data.

The run is deterministic (seed in config.py) up to LightGBM multithread float
noise, which does not move the ranking-based Spearman/g.
"""
import argparse
import json
import os
import pickle
import time

import numpy as np

import config
import model as M

DEFAULT_DATA_DIR = "/data4/han/aba_v2.5_bench/naturebench_staged/s41592-025-02776-2/public/problem/data"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def process_instance(instance, data_dir, output_dir):
    t0 = time.time()
    data_path = os.path.join(data_dir, instance)
    out_path = os.path.join(output_dir, instance)
    os.makedirs(out_path, exist_ok=True)

    tr_var, tr_nmut, tr_score = M.read_tsv(os.path.join(data_path, "train.tsv"), True)
    te_var, te_nmut, _ = M.read_tsv(os.path.join(data_path, "test.tsv"), False)
    log(f"{instance}: train={len(tr_var)} test={len(te_var)}")

    wt_map = M.reconstruct_wt([tr_var, te_var])
    ps = sorted(wt_map)

    struct_feats = {}
    try:
        pdbs = [f for f in os.listdir(data_path) if f.endswith(".pdb")]
        if pdbs:
            struct_feats = M.compute_pdb_features(os.path.join(data_path, pdbs[0]), wt_map)
            log(f"{instance}: struct feats for {len(struct_feats)}/{len(ps)} positions")
    except Exception as e:
        log(f"{instance}: PDB feature failure: {e}")

    esm_table = None
    esm_pos_index = {}
    esm_pos_stats = None
    try:
        wt_seq = "".join(wt_map[p] for p in ps)
        esm_table = M.esm_logprob_table(wt_seq)
        esm_pos_index = {p: i for i, p in enumerate(ps)}
        esm_pos_stats = M.esm_position_stats(esm_table)
        log(f"{instance}: ESM table {esm_table.shape}")
    except Exception as e:
        log(f"{instance}: ESM failure: {e}")

    onehot_vocab = {}
    for v in tr_var:
        for (w, p, mt) in M.parse_variant(v):
            key = (p, mt)
            if key not in onehot_vocab:
                onehot_vocab[key] = len(onehot_vocab)

    Xtr = M.build_features(tr_var, tr_nmut, wt_map, struct_feats, esm_table,
                           esm_pos_index, onehot_vocab, esm_pos_stats)
    Xte = M.build_features(te_var, te_nmut, wt_map, struct_feats, esm_table,
                           esm_pos_index, onehot_vocab, esm_pos_stats)
    log(f"{instance}: X shape train={Xtr.shape} test={Xte.shape}")

    pred, boosters, holdout_sp, head = M.fit_predict_lgbm(
        Xtr, tr_score, Xte, fast_gb1=(instance == "gb1"))

    with open(os.path.join(out_path, "predictions.tsv"), "w") as f:
        f.write("variant\tscore\n")
        for v, s in zip(te_var, pred):
            f.write(f"{v}\t{float(s)}\n")

    n_features = int(Xtr.shape[1])
    del Xtr, Xte

    bundle = {
        "boosters": boosters,
        "ridge_model": head["ridge_model"],
        "chosen": head["chosen"],
        "best_alpha": head["best_alpha"],
        "wt_map": wt_map,
        "struct_feats": struct_feats,
        "esm_table": esm_table,
        "esm_pos_index": esm_pos_index,
        "esm_pos_stats": esm_pos_stats,
        "onehot_vocab": onehot_vocab,
        "n_features": n_features,
    }
    log(f"{instance}: done in {time.time()-t0:.1f}s (holdout Spearman={holdout_sp:.4f})")
    return bundle, holdout_sp


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default=os.environ.get("DATA_DIR", DEFAULT_DATA_DIR))
    ap.add_argument("--output-dir", default=os.environ.get("OUTPUT_DIR", "./output"))
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    t_start = time.time()
    all_models, holdout_scores = {}, {}
    for inst in config.INSTANCES:
        bundle, holdout_sp = process_instance(inst, args.data_dir, args.output_dir)
        all_models[inst] = bundle
        holdout_scores[inst] = holdout_sp

    with open(os.path.join(args.output_dir, "best_model.pkl"), "wb") as f:
        pickle.dump(all_models, f, protocol=pickle.HIGHEST_PROTOCOL)

    mean_score = float(np.mean(list(holdout_scores.values()))) if holdout_scores else None
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump({
            "score": mean_score,
            "per_instance_holdout_spearman": holdout_scores,
            "instances_done": list(all_models.keys()),
        }, f, indent=2)
    log(f"ALL DONE in {time.time()-t_start:.1f}s mean_holdout_spearman={mean_score}")


if __name__ == "__main__":
    main()

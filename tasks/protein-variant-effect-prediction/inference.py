"""Standalone inference for Protein Variant Effect Prediction.

    python inference.py --input /path/to/instances --output ./output
                        [--checkpoint best_model.pkl]

--input must contain one subdirectory per instance, each with a test.tsv
(variant<TAB>n_mut[<TAB>score]). For every instance present in both --input and
the checkpoint, writes --output/<instance>/predictions.tsv (variant<TAB>score).

Loads the trained checkpoint and predicts using ONLY the state baked into it
(fitted LightGBM+Ridge heads + one-hot vocab + ESM log-prob table + PDB
structural features + WT map). No retraining, no ESM forward pass, no GPU, and
the raw training data are NOT required.
"""
import argparse
import os
import pickle

import config
import model as M


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="dir with one subdir per instance (test.tsv inside)")
    ap.add_argument("--output", default="./output", help="where to write <instance>/predictions.tsv")
    ap.add_argument("--checkpoint", default=os.path.join(os.path.dirname(__file__), "best_model.pkl"))
    args = ap.parse_args()

    with open(args.checkpoint, "rb") as f:
        ckpt = pickle.load(f)
    os.makedirs(args.output, exist_ok=True)

    for inst in config.INSTANCES:
        test_path = os.path.join(args.input, inst, "test.tsv")
        if inst not in ckpt or not os.path.exists(test_path):
            print(f"[skip] {inst}: {'no checkpoint' if inst not in ckpt else 'no test.tsv'}", flush=True)
            continue
        te_var, te_nmut, _ = M.read_tsv(test_path, False)
        pred = M.predict_with_bundle(ckpt[inst], te_var, te_nmut)

        out_dir = os.path.join(args.output, inst)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "predictions.tsv"), "w") as f:
            f.write("variant\tscore\n")
            for v, s in zip(te_var, pred):
                f.write(f"{v}\t{float(s)}\n")
        print(f"[ok]   {inst}: {len(te_var)} predictions", flush=True)


if __name__ == "__main__":
    main()

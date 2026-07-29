"""Final count-space ensemble for the Chicken Counting task.

The Counting Score depends ONLY on each density map's sum (the per-image count), so
the ensemble is done in COUNT space, then re-materialized as density maps.

Method (aggregator across-node blend):
  Members are four CSRNet-style density regressors that cluster into families by
  near-duplicate per-image counts: A = {node_24, node_25} (r=0.999), B = {node_19,
  node_20} (r=0.999). node_16 was excluded (weakest, drags every blend it enters).

  Blended per-image count (the "w_24dom" winner, robust across nearby weights):

      count = 0.6 * c24 + 0.25 * c25 + 0.15 * (c19 + c20) / 2

  Spatial maps come from node_24's per-image unit-normalized maps (best spatial prior)
  rescaled so each map's sum equals the blended count:

      pred_map_i = (node24_map_i / sum(node24_map_i)) * count_i

This keeps the best-calibrated member (node_24, 0.9389) dominant, adds its family-mate
node_25, and a light 0.15 dose of family B to shave per-image variance -> 0.9398,
beating every individual node.

Usage:
  python inference.py --output ./output
     -> writes ./output/chicken_counting/predictions.npy  (100, 180, 320) float32

The member per-image counts (c24, c25, c19, c20) and node_24's density maps are the
bundled `members/` artifacts; there is no image forward pass here -- this is the final
count-space fusion of the members that were trained upstream.
"""
import os
import argparse
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MEMBERS_DIR = os.path.join(SCRIPT_DIR, "members")
INSTANCE = "chicken_counting"


def _load(name):
    return np.load(os.path.join(MEMBERS_DIR, name))


def run(output_dir):
    c24 = _load("counts_24.npy")
    c25 = _load("counts_25.npy")
    c19 = _load("counts_19.npy")
    c20 = _load("counts_20.npy")
    base = _load("node24_maps.npy").astype(np.float32)   # (100, 180, 320)

    # w_24dom blended per-image count
    count = 0.6 * c24 + 0.25 * c25 + 0.15 * (c19 + c20) / 2.0

    # re-materialize onto node_24's unit-sum spatial maps
    base_sum = base.reshape(base.shape[0], -1).sum(1)
    unit = base / base_sum[:, None, None]
    pred = np.clip((unit * count[:, None, None]).astype(np.float32), 0, None)

    out_inst = os.path.join(output_dir, INSTANCE)
    os.makedirs(out_inst, exist_ok=True)
    np.save(os.path.join(out_inst, "predictions.npy"), pred)
    print(f"Wrote {out_inst}/predictions.npy  shape={pred.shape} "
          f"count_mean={float(count.mean()):.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default=os.path.join(SCRIPT_DIR, "output"),
                    help="output dir; predictions written to <output>/chicken_counting/")
    args = ap.parse_args()
    run(args.output)


if __name__ == "__main__":
    main()

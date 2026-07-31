#!/usr/bin/env python
"""Standalone inference for RT-1 action-conditioned adaptation (candidate 1).

FINAL configuration = fully fine-tuned checkpoint (best_model.pth in this
directory) + training-free initial-noise truncation (FreeAction,
arXiv:2509.24241) stacked on top at inference time. The truncation was
certified head-to-head against the plain frozen protocol on the full frozen
24-episode validation list (see results.json protocol dict).

Usage:
  python inference.py --input <test_dir_or_data_dir> --output <submission.csv path>

Self-contained: loads the base Cosmos-Predict2.5-2B action-cond pipeline, then
replaces the DiT weights with the fine-tuned consolidated state from
best_model.pth located in the SAME directory as this file. Generation =
official chunked autoregressive rollout (guidance 0, 35 steps, 13-frame
chunks, 1 latent conditional frame, action_scaler 20) with the noise
truncation defined by TRUNC below applied to the seeded initial latent noise.
"""
import os

WM_ROOT = os.environ.get("WM_ROOT", ".")

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("HF_HOME", os.path.join(WM_ROOT, "hf_cache"))
os.environ.setdefault("IMAGEIO_FFMPEG_EXE", os.environ.get("FFMPEG_BIN", "ffmpeg"))

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = os.path.join(WM_ROOT, "cosmos-predict2.5")
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import torch  # noqa: E402

torch.set_num_threads(4)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RELEASED_CKPT = (
    os.path.join(WM_ROOT, "checkpoints/Cosmos-Predict2.5-2B/robot/action-cond/"
                             "38c6c645-7d41-4560-8eeb-6f4ddc0e6574_ema_bf16.pt")
)
EXPERIMENT = (
    "cosmos_predict2p5_2B_reason_embeddings_action_conditioned_"
    "rectified_flow_bridge_13frame_256x320"
)
ACTION_CONFIG_FILE = (
    "cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py"
)
NEG_PROMPT = (
    "The video captures a series of frames showing ugly scenes, static with no motion, "
    "motion blur, over-saturation, shaky footage, low resolution, grainy texture, "
    "pixelated images, poorly lit areas, underexposed and overexposed scenes, poor color "
    "balance, washed out colors, choppy sequences, jerky movements, low frame rate, "
    "artifacting, color banding, unnatural transitions, outdated special effects, fake "
    "elements, unconvincing visuals, poorly edited content, jump cuts, visual noise, and "
    "flickering. Overall, the video is of poor quality."
)

# ---- training-free inference enhancement: initial-noise truncation --------
# Certified on the full frozen 24-ep list (see attempt_8/results.json):
# tau sweep {1.5, 1.0, 0.75, 0.65, 0.6, 0.5} + action-scaled variants; fixed
# tau=0.6 won at PSNR 23.871 vs 23.385 baseline replication (+0.487 dB), with
# pixel-std improved (55.6 vs 52.1, min ep 41.0). tau_mode "fixed": truncate
# every chunk's initial noise to |z| <= tau via the deterministic inverse-CDF
# map (preserves the seeded noise field ordering).
TRUNC = {"tau_mode": "fixed", "tau": 0.6}

NOISE_STATE = {"tau": None}


def truncate_noise(z, tau):
    from torch.special import ndtr, ndtri
    zf = z.float()
    t = torch.as_tensor(float(tau), device=z.device, dtype=torch.float32)
    lo = ndtr(-t)
    hi = ndtr(t)
    u = ndtr(zf).clamp(1e-7, 1 - 1e-7)
    z2 = ndtri((lo + u * (hi - lo)).clamp(1e-7, 1 - 1e-7))
    return z2.to(z.dtype)


def install_noise_patch():
    from cosmos_predict2._src.imaginaire.utils import misc as _misc
    orig = _misc.arch_invariant_rand

    def patched(shape, dtype, device, seed):
        z = orig(shape, dtype, device, seed)
        if NOISE_STATE["tau"] is not None:
            z = truncate_noise(z, NOISE_STATE["tau"])
        return z

    _misc.arch_invariant_rand = patched


def chunk_tau(act_chunk, trunc):
    mode = trunc.get("tau_mode", "none")
    if mode == "none":
        return None
    if mode == "fixed":
        return float(trunc["tau"])
    if mode == "action":
        m = float(np.linalg.norm(act_chunk, axis=1).mean())
        mu, sg = float(trunc["mu"]), float(trunc["sigma"])
        s = 1.0 / (1.0 + np.exp(-(m - mu) / sg))
        return float(trunc["tau_min"] + (trunc["tau_max"] - trunc["tau_min"]) * s)
    raise ValueError(mode)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def chunked_rollout(v2w, first_frame_rgb, actions_scaled, cfg):
    import torchvision
    chunk = cfg["chunk_size"]
    img = first_frame_rgb
    futures = []
    n = len(actions_scaled)
    for i in range(0, n, chunk):
        act_chunk = actions_scaled[i:i + chunk]
        if act_chunk.shape[0] != chunk:
            pad = np.zeros((chunk - act_chunk.shape[0], act_chunk.shape[1]), dtype=act_chunk.dtype)
            act_chunk = np.concatenate([act_chunk, pad], 0)
        NOISE_STATE["tau"] = chunk_tau(act_chunk, TRUNC)
        img_tensor = torchvision.transforms.functional.to_tensor(img).unsqueeze(0)
        num_video_frames = act_chunk.shape[0] + 1
        vid_input = torch.cat(
            [img_tensor, torch.zeros_like(img_tensor).repeat(num_video_frames - 1, 1, 1, 1)], 0)
        vid_input = (vid_input * 255.0).to(torch.uint8).unsqueeze(0).permute(0, 2, 1, 3, 4)
        video = v2w.generate_vid2world(
            prompt="", input_path=vid_input,
            action=torch.from_numpy(act_chunk).float(),
            guidance=cfg["guidance"], num_video_frames=num_video_frames,
            num_latent_conditional_frames=cfg["num_latent_conditional_frames"],
            resolution="256,320", seed=i, negative_prompt=NEG_PROMPT, num_steps=cfg["num_steps"],
        )
        NOISE_STATE["tau"] = None
        vnorm = (video - (-1)) / 2.0
        vclamp = (torch.clamp(vnorm[0], 0, 1) * 255).to(torch.uint8).permute(1, 2, 3, 0).cpu().numpy()
        futures.append(vclamp[1:])
        img = vclamp[-1]
    return np.concatenate(futures, 0)


def find_test_episodes(input_path):
    """Accept either the test dir directly or the data dir (containing test/)."""
    p = Path(input_path)
    if (p / "test").is_dir():
        p = p / "test"
    eps = []
    for d in sorted(p.iterdir()):
        if d.is_dir() and (d / "first_frame.png").exists() and (d / "actions.npy").exists():
            eps.append(d)
    return eps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    # Optional data-parallel sharding (defaults regenerate everything in one call).
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()

    ckpt_path = os.path.join(SCRIPT_DIR, "best_model.pth")
    use_ema_from_latest = False
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(SCRIPT_DIR, "checkpoint_latest.pth")
        use_ema_from_latest = True
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    protocol = ck.get("protocol", {})
    net_state = ck.get("ema_state") if use_ema_from_latest and ck.get("ema_state") else ck["net_state"]
    log(f"loaded checkpoint: {ckpt_path} (step {ck.get('step')}, "
        f"{'EMA-from-latest' if use_ema_from_latest else 'best'})")
    log(f"noise truncation config: {TRUNC}")

    gen_cfg = {
        "chunk_size": int(protocol.get("chunk_size", cfg["chunk_size"])),
        "guidance": float(protocol.get("guidance", cfg["guidance"])),
        "num_steps": int(protocol.get("num_steps", cfg["num_steps"])),
        "num_latent_conditional_frames": int(protocol.get(
            "num_latent_conditional_frames", cfg["num_latent_conditional_frames"])),
    }
    action_scaler = float(protocol.get("action_scaler", cfg["action_scaler"]))
    gripper_scale = float(protocol.get("gripper_scale", cfg["gripper_scale"]))
    scaler_vec = np.array([action_scaler] * 6 + [gripper_scale], dtype=np.float32)

    try:
        from cosmos_oss.init import init_environment
        init_environment()
    except Exception as e:
        log(f"init_environment skipped: {e}")
    from cosmos_predict2._src.predict2.inference.video2world import Video2WorldInference
    log("loading base model...")
    v2w = Video2WorldInference(
        experiment_name=EXPERIMENT, ckpt_path=RELEASED_CKPT, s3_credential_path="",
        context_parallel_size=1, config_file=ACTION_CONFIG_FILE,
    )
    model = v2w.model
    net = model.net
    dt = next(net.parameters()).dtype
    load_sd = {
        k: (v.to(dt) if torch.is_tensor(v) and v.is_floating_point() else v)
        for k, v in net_state.items() if not k.endswith("_extra_state")
    }
    missing, unexpected = net.load_state_dict(load_sd, strict=False)
    missing_real = [m for m in missing if not m.endswith("_extra_state")]
    log(f"fine-tuned weights loaded: missing={len(missing_real)} unexpected={len(unexpected)}")
    if len(missing_real) > 0:
        log(f"WARNING: missing non-extra-state keys, e.g. {missing_real[:5]}")
    net.eval()
    del ck, net_state

    install_noise_patch()

    eps = find_test_episodes(args.input)
    log(f"found {len(eps)} test episodes")
    if args.num_shards > 1:
        eps = eps[args.shard_index::args.num_shards]
        log(f"shard {args.shard_index}/{args.num_shards}: handling {len(eps)} episodes")
    out_csv = Path(args.output)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pred_dir = out_csv.parent / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    from PIL import Image
    rows = []
    nontrivial = 0
    for d in eps:
        eid = d.name
        first = np.asarray(Image.open(d / "first_frame.png").convert("RGB"))
        if first.shape[0] != 256 or first.shape[1] != 320:
            first = np.asarray(Image.fromarray(first).resize((320, 256), Image.BILINEAR))
        actions = np.load(d / "actions.npy").astype(np.float32) * scaler_vec
        meta = json.load(open(d / "meta.json"))
        n_future = int(meta.get("num_future_frames", len(actions)))
        with torch.no_grad():
            pred = chunked_rollout(v2w, first, actions, gen_cfg)
        pred = pred[:n_future]
        if len(pred) < n_future:
            pred = np.concatenate([pred, np.repeat(pred[-1:], n_future - len(pred), 0)], 0)
        pred = pred.astype(np.uint8)
        if pred.std() > 1.0:
            nontrivial += 1
        npz_path = pred_dir / f"{eid}.npz"
        np.savez_compressed(npz_path, frames=pred)
        rows.append((eid, os.path.relpath(npz_path, out_csv.parent)))
        log(f"  {eid}: pred {pred.shape} std={pred.std():.2f}")

    csv_target = out_csv if args.num_shards == 1 else Path(str(out_csv) + f".shard{args.shard_index}")
    with open(csv_target, "w", newline="") as f:
        w = csv.writer(f)
        if args.num_shards == 1:
            w.writerow(["episode_id", "path"])
        for eid, rel in rows:
            w.writerow([eid, rel])
    log(f"wrote {csv_target} with {len(rows)} rows; non-trivial predictions: {nontrivial}/{len(rows)}")
    if nontrivial == 0 and len(rows) > 0:
        log("ERROR: all predictions degenerate!")
        sys.exit(2)


if __name__ == "__main__":
    main()

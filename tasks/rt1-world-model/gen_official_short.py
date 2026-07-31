#!/usr/bin/env python
"""Generate predictions for the official IRASim RT-1 short-trajectory protocol.

For each of the 4,799 official test clips ({episode_id}_{cam}_{start}.mp4):
  conditioning = episode frame[start], actions = episode actions[start:start+15]
  generate 15 future frames with the released fine-tuned checkpoint (+
  FreeAction noise truncation), then write:
    pred_sample_videos/{name}.mp4   16 frames (decoded cond frame + 15 pred), 4 fps
    pred_sample_frames/{name}/000001..000015.png   the 15 predicted frames only
  matching the official GT layout (16-frame mp4 for PSNR/SSIM, 15-frame PNG dir
  for FVD/FID).

Resumable: clips whose mp4 + 15 PNGs already exist are skipped.
Shardable: --shard-index i --num-shards n takes clips[i::n].
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import winner_infer as W  # noqa: E402  (sets env + sys.path for the cosmos repo)

import cv2  # noqa: E402
import torch  # noqa: E402
import torchvision  # noqa: E402
import imageio  # noqa: E402

RT1 = os.environ.get("RT1_DATA", os.path.join(os.environ.get("WM_ROOT", "."), "data/rt1"))
CKPT = os.environ.get("FINETUNED_CKPT",
        os.path.join(os.environ.get("WM_ROOT", "."), "finetuned/best_model.pth"))
GT_VIDEOS = f"{RT1}/evaluation_videos/test_sample_videos"
EP_INDEX_CACHE = os.path.join(SCRIPT_DIR, "ep_index_test.json")


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def build_ep_index():
    """episode_id -> {video: path, ann: path} for the test split."""
    if os.path.exists(EP_INDEX_CACHE):
        return json.load(open(EP_INDEX_CACHE))
    import glob
    idx = {}
    for f in glob.glob(f"{RT1}/annotation/test/*.json"):
        d = json.load(open(f))
        vp = d["videos"][0]["video_path"] if isinstance(d["videos"], list) else d["videos"]
        idx[str(d["episode_id"])] = {"video": f"{RT1}/{vp}", "ann": f}
    json.dump(idx, open(EP_INDEX_CACHE, "w"))
    return idx


def read_frame(video_path, frame_idx):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, bgr = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"cannot read frame {frame_idx} of {video_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if rgb.shape[0] != 256 or rgb.shape[1] != 320:
        rgb = cv2.resize(rgb, (320, 256), interpolation=cv2.INTER_LINEAR)
    return rgb


def rollout16(v2w, first_frame_rgb, actions_scaled, cfg):
    """Same as winner_infer.chunked_rollout but also returns the decoded
    conditioning frame (vclamp[0] of the first chunk) for the 16-frame mp4."""
    chunk = cfg["chunk_size"]
    img = first_frame_rgb
    futures = []
    dec0 = None
    n = len(actions_scaled)
    for i in range(0, n, chunk):
        act_chunk = actions_scaled[i:i + chunk]
        if act_chunk.shape[0] != chunk:
            pad = np.zeros((chunk - act_chunk.shape[0], act_chunk.shape[1]), dtype=act_chunk.dtype)
            act_chunk = np.concatenate([act_chunk, pad], 0)
        W.NOISE_STATE["tau"] = W.chunk_tau(act_chunk, W.TRUNC)
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
            resolution="256,320", seed=i, negative_prompt=W.NEG_PROMPT,
            num_steps=cfg["num_steps"],
        )
        W.NOISE_STATE["tau"] = None
        vnorm = (video - (-1)) / 2.0
        vclamp = (torch.clamp(vnorm[0], 0, 1) * 255).to(torch.uint8).permute(1, 2, 3, 0).cpu().numpy()
        if dec0 is None:
            dec0 = vclamp[0]
        futures.append(vclamp[1:])
        img = vclamp[-1]
    return dec0, np.concatenate(futures, 0)


def write_outputs(name, frames16, out_videos, out_frames):
    vdir = Path(out_videos); vdir.mkdir(parents=True, exist_ok=True)
    fdir = Path(out_frames) / name
    fdir.mkdir(parents=True, exist_ok=True)
    wr = imageio.get_writer(str(vdir / f"{name}.mp4"), fps=4, codec="libx264",
                            quality=9, pixelformat="yuv420p", macro_block_size=1)
    for f in frames16:
        wr.append_data(f)
    wr.close()
    # PNG dir: the 15 PREDICTED frames only, matching GT test_sample_frames
    for j, f in enumerate(frames16[1:], start=1):
        cv2.imwrite(str(fdir / f"{j:06d}.png"), cv2.cvtColor(f, cv2.COLOR_RGB2BGR))


def clip_done(name, out_videos, out_frames):
    v = Path(out_videos) / f"{name}.mp4"
    d = Path(out_frames) / name
    return v.exists() and d.is_dir() and len(list(d.glob("*.png"))) == 15


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.path.join(SCRIPT_DIR, "results"))
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    out_videos = os.path.join(args.out_dir, "pred_sample_videos")
    out_frames = os.path.join(args.out_dir, "pred_sample_frames")
    status = os.path.join(args.out_dir, f"status_shard{args.shard_index}.txt")
    os.makedirs(args.out_dir, exist_ok=True)

    clips = sorted(n[:-4] for n in os.listdir(GT_VIDEOS) if n.endswith(".mp4"))
    log(f"official clip list: {len(clips)} clips")
    clips = clips[args.shard_index::args.num_shards]
    log(f"shard {args.shard_index}/{args.num_shards}: {len(clips)} clips")
    if args.limit:
        clips = clips[:args.limit]
        log(f"limit: {len(clips)} clips")

    todo = [c for c in clips if not clip_done(c, out_videos, out_frames)]
    log(f"already done: {len(clips) - len(todo)}, to generate: {len(todo)}")
    if not todo:
        Path(status).write_text("ALL_DONE\n")
        return

    ep_index = build_ep_index()
    log(f"episode index: {len(ep_index)} test episodes")

    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    protocol = ck.get("protocol", {})
    net_state = ck["net_state"]
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
    log(f"gen_cfg: {gen_cfg}, scaler: {action_scaler}/{gripper_scale}, trunc: {W.TRUNC}")

    try:
        from cosmos_oss.init import init_environment
        init_environment()
    except Exception as e:
        log(f"init_environment skipped: {e}")
    from cosmos_predict2._src.predict2.inference.video2world import Video2WorldInference
    log("loading base model...")
    v2w = Video2WorldInference(
        experiment_name=W.EXPERIMENT, ckpt_path=W.RELEASED_CKPT, s3_credential_path="",
        context_parallel_size=1, config_file=W.ACTION_CONFIG_FILE,
    )
    net = v2w.model.net
    dt = next(net.parameters()).dtype
    load_sd = {k: (v.to(dt) if torch.is_tensor(v) and v.is_floating_point() else v)
               for k, v in net_state.items() if not k.endswith("_extra_state")}
    missing, unexpected = net.load_state_dict(load_sd, strict=False)
    missing_real = [m for m in missing if not m.endswith("_extra_state")]
    log(f"fine-tuned weights loaded: missing={len(missing_real)} unexpected={len(unexpected)}")
    net.eval()
    del ck, net_state
    W.install_noise_patch()

    ann_cache = {}
    t0 = time.time()
    for k, name in enumerate(todo):
        ep_id, cam, start = name.rsplit("_", 2)
        start = int(start)
        info = ep_index[ep_id]
        if ep_id not in ann_cache:
            ann_cache.clear()
            ann_cache[ep_id] = np.array(json.load(open(info["ann"]))["action"], dtype=np.float32)
        actions = ann_cache[ep_id][start:start + 15] * scaler_vec
        assert actions.shape == (15, 7), f"{name}: bad action slice {actions.shape}"
        first = read_frame(info["video"], start)
        with torch.no_grad():
            dec0, pred = rollout16(v2w, first, actions, gen_cfg)
        frames16 = np.concatenate([dec0[None], pred[:15]], 0).astype(np.uint8)
        assert frames16.shape[0] == 16
        write_outputs(name, frames16, out_videos, out_frames)
        el = time.time() - t0
        eta_h = el / (k + 1) * (len(todo) - k - 1) / 3600
        Path(status).write_text(f"{k+1}/{len(todo)} last={name} "
                                f"avg={el/(k+1):.1f}s eta={eta_h:.1f}h\n")
        if (k + 1) % 10 == 0:
            log(f"{k+1}/{len(todo)} avg {el/(k+1):.1f}s/clip eta {eta_h:.1f}h")
    Path(status).write_text("ALL_DONE\n")
    log("shard complete")


if __name__ == "__main__":
    main()

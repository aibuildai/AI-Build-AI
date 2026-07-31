#!/usr/bin/env python
"""FULL fine-tune of Cosmos-Predict2.5-2B robot/action-cond on RT-1.

Design:
- ALL DiT parameters trainable (attn, MLP, adaLN, norms, embedders, final layer).
- FSDP2 (fully_shard) full-shard across all GPUs, fp32 sharded master
  params + bf16 compute (MixedPrecisionPolicy), selective activation checkpointing.
- Flow-matching loss via the repo's own model.training_step() -> identical
  parameterization / timestep sampling to the released training objective.
- EMA over the sharded fp32 params (local-shard EMA; evaluated weights are EMA).
- Frozen eval protocol = official action-conditioned inference example:
  guidance 0, 35 steps, 13-frame chunks (12 actions), 1 latent conditional frame,
  action_scaler 20.0 (recorded in protocol dict). Fixed held-out validation
  episode list from config.
- Actions strictly from annotation JSON (videos/action.npy is a known all-zeros trap).

Launch: torchrun --nproc_per_node=<n> train.py --config <config.yaml>
(also runs single-process without torchrun; FSDP mesh of size 1).
"""
import os

WM_ROOT = os.environ.get("WM_ROOT", ".")

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("HF_HOME", os.path.join(WM_ROOT, "hf_cache"))
os.environ.setdefault("IMAGEIO_FFMPEG_EXE", os.environ.get("FFMPEG_BIN", "ffmpeg"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import copy
import json
import math
import random
import signal
import sys
import time
from pathlib import Path

import numpy as np
import yaml

REPO = os.path.join(WM_ROOT, "cosmos-predict2.5")
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

torch.set_num_threads(4)

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

STOP = {"flag": False}


def log(msg, rank=None):
    r = f"[r{rank}]" if rank is not None else ""
    print(f"[{time.strftime('%H:%M:%S')}]{r} {msg}", flush=True)


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def decode_video_rgb(path):
    """Decode full mp4 to uint8 (T,H,W,3) RGB."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr[:, :, ::-1])  # BGR -> RGB
    cap.release()
    return np.ascontiguousarray(np.stack(frames, 0))


class RT1WindowDataset(torch.utils.data.Dataset):
    """13-frame windows (1 cond frame + 12 predicted, 12 actions) from RT-1 train."""

    def __init__(self, data_dir, episode_ids, chunk_actions, scaler_vec, seed=42):
        self.video_dir = Path(data_dir) / "train" / "videos"
        self.ann_dir = Path(data_dir) / "train" / "annotation"
        self.ids = episode_ids
        self.chunk = chunk_actions  # 12
        self.scaler = scaler_vec.astype(np.float32)
        self.seed = seed
        self.epoch = 0
        # audit: actions from annotation JSON must be non-zero (videos/action.npy trap)
        tot = 0.0
        for eid in self.ids[:5]:
            a = np.asarray(json.load(open(self.ann_dir / f"{eid}.json"))["action"], np.float32)
            tot += float(np.abs(a).mean())
        assert tot > 1e-4, "annotation JSON actions look all-zero; data audit failed"

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        eid = self.ids[idx]
        ann = json.load(open(self.ann_dir / f"{eid}.json"))
        actions = np.asarray(ann["action"], np.float32)  # (T-1, 7)
        n_act = actions.shape[0]
        rng = np.random.default_rng((self.seed + self.epoch * 1_000_003 + idx * 97) % (2**31))
        start = int(rng.integers(0, max(n_act - self.chunk, 0) + 1))
        frames = decode_video_rgb(self.video_dir / eid / "rgb.mp4")  # (T,H,W,3)
        n_frames = frames.shape[0]
        # window frames start .. start+12 (13 frames), actions start .. start+11
        idxs = np.clip(np.arange(start, start + self.chunk + 1), 0, n_frames - 1)
        win = frames[idxs]  # (13,H,W,3)
        act = np.zeros((self.chunk, 7), np.float32)
        avail = min(self.chunk, max(n_act - start, 0))
        act[:avail] = actions[start:start + avail]
        act *= self.scaler  # same scaling as inference conditioning
        video = torch.from_numpy(win).permute(3, 0, 1, 2).contiguous()  # (C,T,H,W) uint8
        return {"video": video, "action": torch.from_numpy(act)}


# --------------------------------------------------------------------------
# generation (identical to the official chunked-rollout inference protocol)
# --------------------------------------------------------------------------
def chunked_rollout(v2w, first_frame_rgb, actions_scaled, gen_cfg):
    import torchvision

    chunk = gen_cfg["chunk_size"]
    img = first_frame_rgb
    futures = []
    n = len(actions_scaled)
    for i in range(0, n, chunk):
        act_chunk = actions_scaled[i:i + chunk]
        if act_chunk.shape[0] != chunk:
            pad = np.zeros((chunk - act_chunk.shape[0], act_chunk.shape[1]), dtype=act_chunk.dtype)
            act_chunk = np.concatenate([act_chunk, pad], 0)
        img_tensor = torchvision.transforms.functional.to_tensor(img).unsqueeze(0)
        num_video_frames = act_chunk.shape[0] + 1
        vid_input = torch.cat(
            [img_tensor, torch.zeros_like(img_tensor).repeat(num_video_frames - 1, 1, 1, 1)], 0)
        vid_input = (vid_input * 255.0).to(torch.uint8).unsqueeze(0).permute(0, 2, 1, 3, 4)
        video = v2w.generate_vid2world(
            prompt="", input_path=vid_input,
            action=torch.from_numpy(act_chunk).float(),
            guidance=gen_cfg["guidance"], num_video_frames=num_video_frames,
            num_latent_conditional_frames=gen_cfg["num_latent_conditional_frames"],
            resolution="256,320", seed=i, negative_prompt=NEG_PROMPT,
            num_steps=gen_cfg["num_steps"],
        )
        vnorm = (video - (-1)) / 2.0
        vclamp = (torch.clamp(vnorm[0], 0, 1) * 255).to(torch.uint8).permute(1, 2, 3, 0).cpu().numpy()
        futures.append(vclamp[1:])
        img = vclamp[-1]
    return np.concatenate(futures, 0)


# --------------------------------------------------------------------------
# EMA over local FSDP shards
# --------------------------------------------------------------------------
class ShardEMA:
    def __init__(self, params, decay):
        self.decay = decay
        self.params = [p for p in params if p.requires_grad]
        self.shadow = [self._local(p).detach().clone().float() for p in self.params]

    @staticmethod
    def _local(p):
        return p._local_tensor if hasattr(p, "_local_tensor") else p.data

    @torch.no_grad()
    def update(self):
        d = self.decay
        for p, s in zip(self.params, self.shadow):
            s.mul_(d).add_(self._local(p).float(), alpha=1.0 - d)

    @torch.no_grad()
    def swap_in(self):
        self.backup = [self._local(p).detach().clone() for p in self.params]
        for p, s in zip(self.params, self.shadow):
            self._local(p).copy_(s.to(self._local(p).dtype))

    @torch.no_grad()
    def swap_out(self):
        for p, b in zip(self.params, self.backup):
            self._local(p).copy_(b)
        self.backup = None


# --------------------------------------------------------------------------
# checkpoint helpers (consolidated full state dicts, rank0 writes)
# --------------------------------------------------------------------------
def strip_wrap(sd):
    return {k.replace("._checkpoint_wrapped_module", "").replace("_checkpoint_wrapped_module.", ""): v
            for k, v in sd.items()}


def full_model_sd(net, dtype=torch.bfloat16):
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
    sd = get_model_state_dict(net, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
    out = {}
    for k, v in strip_wrap(sd).items():
        # Only cast floating-point tensors. Non-float tensors (notably transformer_engine
        # `_extra_state` pickled byte blobs) must be kept verbatim or TE's
        # set_extra_state() crashes on reload ("unsupported ScalarType BFloat16").
        if torch.is_tensor(v) and v.is_floating_point():
            out[k] = v.to(dtype)
        else:
            out[k] = v
    return out


def full_optim_sd(net, optim):
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_optimizer_state_dict
    return get_optimizer_state_dict(net, optim,
                                    options=StateDictOptions(full_state_dict=True, cpu_offload=True))


def load_optim_sd(net, optim, osd):
    from torch.distributed.checkpoint.state_dict import StateDictOptions, set_optimizer_state_dict
    set_optimizer_state_dict(net, optim, optim_state_dict=osd,
                             options=StateDictOptions(full_state_dict=True, broadcast_from_rank0=False))


def load_model_full_sd(net, sd):
    from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict
    # Drop TE `_extra_state` pickled byte blobs (casting them corrupts unpickling;
    # the wrapped modules already carry correct extra state) and cast only
    # floating-point tensors to fp32. strict=False tolerates the dropped keys.
    clean = {k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v)
             for k, v in sd.items() if not k.endswith("_extra_state")}
    set_model_state_dict(net, clean,
                         options=StateDictOptions(full_state_dict=True,
                                                  broadcast_from_rank0=False, strict=False))


# --------------------------------------------------------------------------
# evaluation on frozen protocol
# --------------------------------------------------------------------------
def evaluate(v2w, model, ep_ids, data_dir, gen_cfg, scaler_vec, rank, tm,
             compute_static=False):
    net = model.net
    was_training = net.training
    net.eval()
    video_dir = Path(data_dir) / "train" / "videos"
    ann_dir = Path(data_dir) / "train" / "annotation"
    per_ep = {}
    static_psnrs = []
    for eid in ep_ids:
        frames = decode_video_rgb(video_dir / eid / "rgb.mp4")
        ann = json.load(open(ann_dir / f"{eid}.json"))
        actions = np.asarray(ann["action"], np.float32) * scaler_vec
        gt = frames[1:]
        t0 = time.time()
        with torch.no_grad():
            pred = chunked_rollout(v2w, frames[0], actions, gen_cfg)
        n = min(len(pred), len(gt))
        pred, gt_n = pred[:n], gt[:n]
        if rank == 0:
            sc = tm.score_episode(pred, gt_n)
            sc["pixel_std"] = float(pred.astype(np.float32).std())
            sc["frames_scored"] = n
            per_ep[eid] = sc
            if compute_static:
                static = np.repeat(frames[0:1], n, 0)
                static_psnrs.append(tm.psnr(static, gt_n))
            log(f"  eval ep {eid}: psnr={sc['psnr']:.3f} ssim={sc['ssim']:.3f} "
                f"l2={sc['latent_l2']:.3f} std={sc['pixel_std']:.1f} ({time.time()-t0:.0f}s)", rank)
    agg = None
    if rank == 0:
        agg = {
            "psnr": float(np.mean([v["psnr"] for v in per_ep.values()])),
            "ssim": float(np.mean([v["ssim"] for v in per_ep.values()])),
            "latent_l2": float(np.mean([v["latent_l2"] for v in per_ep.values()])),
            "pixel_std": float(np.mean([v["pixel_std"] for v in per_ep.values()])),
            "n_episodes": len(per_ep),
        }
        if compute_static and static_psnrs:
            agg["static_null_psnr"] = float(np.mean(static_psnrs))
    if dist.is_initialized():
        obj = [agg]
        dist.broadcast_object_list(obj, src=0)
        agg = obj[0]
    net.train(was_training)
    torch.cuda.empty_cache()
    return agg, per_ep


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = cfg["data_dir"]

    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    if world > 1 and not dist.is_initialized():
        dist.init_process_group("nccl")
    is0 = rank == 0

    seed = int(cfg.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed + rank)  # rank offset only for training noise diversity

    signal.signal(signal.SIGTERM, lambda *a: STOP.__setitem__("flag", True))

    def jlog(row):
        if is0:
            with open(out_dir / "training_progress.jsonl", "a") as f:
                f.write(json.dumps(row) + "\n")

    # ---- frozen protocol ----
    protocol = {
        "num_steps": int(cfg["num_steps"]),
        "guidance": float(cfg["guidance"]),
        "chunk_size": int(cfg["chunk_size"]),
        "num_latent_conditional_frames": int(cfg["num_latent_conditional_frames"]),
        "action_scaler": float(cfg["action_scaler"]),
        "gripper_scale": float(cfg["gripper_scale"]),
        "val_episodes": list(cfg["val_episodes"]),
    }
    gen_cfg = {k: protocol[k] for k in
               ["num_steps", "guidance", "chunk_size", "num_latent_conditional_frames"]}
    scaler_vec = np.array([protocol["action_scaler"]] * 6 + [protocol["gripper_scale"]],
                          np.float32)
    val_ids = [str(e) for e in protocol["val_episodes"]]

    # official task metric (dynamic import from the data dir, if provided there)
    sys.path.insert(0, data_dir)
    try:
        import task_metric as tm
    except ImportError:
        tm = None
        log("task_metric module not found in data_dir; metric scoring disabled. "
            "Validation/final evals are skipped and best-model selection falls "
            "back to (negative) training loss.", rank)

    # ---- dataset ----
    all_ids = sorted(os.listdir(Path(data_dir) / "train" / "videos"))
    train_ids = [e for e in all_ids if e not in set(val_ids)]
    if cfg.get("test_mode", False):
        cap = int(cfg.get("test_max_samples") or 100)
        train_ids = train_ids[:cap]
    if int(cfg["batch_size"]) != 1:
        raise ValueError(
            "batch_size must be 1 (per GPU): the action-conditioning input path "
            "(_get_data_batch_input) pairs exactly one action sequence with the "
            "batch. Scale the effective batch via world size and grad_accum_steps.")
    ds = RT1WindowDataset(data_dir, train_ids, int(cfg["chunk_size"]), scaler_vec, seed=seed)
    if world > 1:
        sampler = torch.utils.data.DistributedSampler(ds, num_replicas=world, rank=rank,
                                                      shuffle=True, seed=seed)
    else:
        sampler = torch.utils.data.RandomSampler(ds, generator=torch.Generator().manual_seed(seed))
    # persistent_workers must stay False: workers re-fork each epoch so they see
    # the updated ds.epoch used for window-start sampling.
    loader = torch.utils.data.DataLoader(
        ds, batch_size=int(cfg["batch_size"]), sampler=sampler,
        num_workers=int(cfg.get("num_workers", 2)), pin_memory=True, drop_last=True,
        persistent_workers=False)
    log(f"train episodes: {len(train_ids)}, val episodes: {len(val_ids)}", rank)

    # ---- model ----
    try:
        from cosmos_oss.init import init_environment
        init_environment()
    except Exception as e:
        log(f"init_environment skipped: {e}", rank)
    from cosmos_predict2._src.predict2.inference.video2world import Video2WorldInference
    from cosmos_predict2._src.predict2.networks.selective_activation_checkpoint import SACConfig
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

    log("loading base model...", rank)
    v2w = Video2WorldInference(
        experiment_name=EXPERIMENT, ckpt_path=RELEASED_CKPT, s3_credential_path="",
        context_parallel_size=1, config_file=ACTION_CONFIG_FILE,
    )
    model = v2w.model
    if hasattr(model, "net_ema") and model.net_ema is not None:
        model.net_ema = None  # free; we keep our own shard EMA
        torch.cuda.empty_cache()
    net = model.net

    # resume: load consolidated model state BEFORE wrapping (plain module names)
    resume_ck = None
    rc = cfg.get("resume_checkpoint")
    if rc and os.path.exists(rc):
        log(f"resuming from {rc}", rank)
        resume_ck = torch.load(rc, map_location="cpu", weights_only=False)
        missing, unexpected = net.load_state_dict(
            {k: v for k, v in resume_ck["net_state"].items()
             if not k.endswith("_extra_state")}, strict=False)
        log(f"resume model load: missing={len(missing)} unexpected={len(unexpected)}", rank)

    # full unfreeze + fp32 master params
    net.to(torch.float32)
    for p in net.parameters():
        p.requires_grad_(True)

    # selective activation checkpointing
    if cfg.get("gradient_checkpointing", True):
        net.enable_selective_checkpoint(SACConfig(mode=cfg.get("sac_mode", "mm_only")), net.blocks)

    # FSDP2 full shard (works with world size 1 as well)
    mesh = init_device_mesh("cuda", (world,))
    # cast_forward_inputs=False is REQUIRED: the net uses the wan-fp32 strategy
    # (timesteps + emb_B_T_D must stay fp32 across module boundaries; each block
    # asserts emb.dtype==fp32 and upcasts adaln via fp32 autocast). The repo pipeline
    # already casts x to bf16 itself (denoise(): xt.to(tensor_kwargs)), so implicit
    # input casting is unnecessary. The default cast_forward_inputs=True casts the
    # fp32 emb -> bf16 at each block boundary, tripping minimal_v4_dit.py:1105.
    mp = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16, reduce_dtype=torch.float32, cast_forward_inputs=False)
    net.fully_shard(mesh, mp_policy=mp)
    net = fully_shard(net, mesh=mesh, reshard_after_forward=True, mp_policy=mp)
    model.net = net
    net.train()
    torch.cuda.empty_cache()
    log(f"FSDP wrapped; mem={torch.cuda.memory_allocated()/2**30:.1f}GB", rank)

    # ---- optimizer ----
    decay, no_decay = [], []
    for n_, p in net.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if (p.ndim < 2 or "norm" in n_.lower()) else decay).append(p)
    lr = float(cfg["learning_rate"])
    optim = torch.optim.AdamW(
        [{"params": decay, "weight_decay": float(cfg["weight_decay"])},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=lr, betas=(0.9, 0.95), eps=1e-8, fused=False)
    warmup = int(cfg.get("warmup_steps", 100))

    gstep = 0
    best_metric = -1e9
    if resume_ck is not None:
        gstep = int(resume_ck.get("step", 0))
        best_metric = float(resume_ck.get("best_metric", -1e9))
        if "optim_state" in resume_ck and resume_ck["optim_state"] is not None:
            try:
                load_optim_sd(net, optim, resume_ck["optim_state"])
                log("optimizer state resumed", rank)
            except Exception as e:
                log(f"optimizer resume failed ({e}); fresh optimizer", rank)
        log(f"resumed at step {gstep}, best_metric {best_metric:.3f}", rank)
    # optional override: seed the best-model threshold from a prior run
    # (checkpoint_latest.pth may have been written before the final eval updated
    # it); null/absent = always improvable
    if cfg.get("initial_best_metric") is not None:
        best_metric = float(cfg["initial_best_metric"])
        log(f"initial_best_metric override: {best_metric:.3f}", rank)

    ema = ShardEMA(net.parameters(), float(cfg.get("ema_decay", 0.999)))
    if resume_ck is not None and "ema_state" in resume_ck and resume_ck["ema_state"] is not None:
        # Restore EMA shadow shards DIRECTLY from the consolidated EMA state dict.
        # Never route through module.load_state_dict / set_model_state_dict here:
        # TE `_extra_state` pickled blobs break under casting, and strict=False
        # inserts _EXTRA_STATE placeholders that TE's set_extra_state cannot handle.
        from torch.distributed.tensor import distribute_tensor
        esd = resume_ck["ema_state"]
        named = [(n_.replace("._checkpoint_wrapped_module", "")
                    .replace("_checkpoint_wrapped_module.", ""), p)
                 for n_, p in net.named_parameters() if p.requires_grad]
        assert len(named) == len(ema.shadow), (len(named), len(ema.shadow))
        n_hit = 0
        for (nm, p), shadow in zip(named, ema.shadow):
            full = esd.get(nm)
            if full is None or not torch.is_tensor(full):
                continue
            full = full.detach().to(torch.float32)
            if hasattr(p, "_local_tensor"):
                dt = distribute_tensor(full.cuda(), p.device_mesh, p.placements)
                shadow.copy_(dt._local_tensor)
                del dt
            else:
                shadow.copy_(full.to(shadow.device))
            n_hit += 1
        log(f"EMA state resumed ({n_hit}/{len(ema.shadow)} tensors)", rank)
    if resume_ck is not None:
        del resume_ck

    def save_ckpt(path, with_optim=True):
        # resume checkpoint: keep fp32 master weights and EMA verbatim so that
        # save -> resume round trips do not quantize through bf16
        t0 = time.time()
        net_sd = full_model_sd(net, dtype=torch.float32)
        ema.swap_in()
        ema_sd = full_model_sd(net, dtype=torch.float32)
        ema.swap_out()
        osd = full_optim_sd(net, optim) if with_optim else None
        if is0:
            tmp = str(path) + ".tmp"
            torch.save({"net_state": net_sd, "ema_state": ema_sd, "optim_state": osd,
                        "step": gstep, "best_metric": best_metric,
                        "config": cfg, "protocol": protocol}, tmp)
            os.replace(tmp, path)
            log(f"saved {path} ({time.time()-t0:.0f}s)", rank)
        if dist.is_initialized():
            dist.barrier()

    def save_ema_model(score, tag):
        ema.swap_in()
        ema_sd = full_model_sd(net)
        ema.swap_out()
        if is0:
            tmp = str(out_dir / "best_model.pth") + ".tmp"
            torch.save({"net_state": ema_sd, "step": gstep, "score": score,
                        "score_tag": tag, "config": cfg, "protocol": protocol}, tmp)
            os.replace(tmp, out_dir / "best_model.pth")
            log(f"saved best_model.pth ({tag} score {score:.3f})", rank)
        if dist.is_initialized():
            dist.barrier()

    def save_best(score, tag="full_val"):
        nonlocal best_metric
        best_metric = score
        save_ema_model(score, tag)

    t_start = time.time()

    # ---- baseline gate (zero-shot, official protocol) ----
    n_gate = int(cfg.get("gate_episodes", 0))
    if cfg.get("run_baseline_gate", False) and n_gate > 0 and gstep == 0 and tm is not None:
        gate_ids = val_ids[:n_gate]
        log(f"running zero-shot baseline gate on {len(gate_ids)} episodes...", rank)
        agg, _ = evaluate(v2w, model, gate_ids, data_dir, gen_cfg, scaler_vec, rank, tm,
                          compute_static=True)
        if is0:
            log(f"ZERO-SHOT GATE: psnr={agg['psnr']:.3f} (window 14.0-17.0), "
                f"static_null={agg.get('static_null_psnr', float('nan')):.3f}", rank)
            jlog({"event": "baseline_gate", "step": 0, **agg,
                  "elapsed_seconds": time.time() - t_start})

    # ---- training loop ----
    max_steps = cfg.get("max_train_steps")
    if cfg.get("test_mode", False):
        max_steps = min(int(max_steps or 8), 8)
    budget = cfg.get("train_time_budget_seconds")
    accum = int(cfg.get("grad_accum_steps", 4))
    val_interval = int(cfg.get("val_interval_steps", 10**9))
    ck_int_steps = int(cfg.get("checkpoint_interval_steps", 500))
    ck_int_sec = float(cfg.get("checkpoint_interval_seconds", 900))
    log_int = int(cfg.get("log_interval_steps", 20))
    subset_frac = float(cfg.get("val_subset_fraction", 0.3))
    grad_clip = float(cfg.get("grad_clip", 1.0))

    loss_hist = []
    # initial_best_subset (optional): seed the subset-val threshold from a prior
    # run so the mid-run insurance snapshot only overwrites best_model.pth when
    # the EMA actually beats it; null/absent = always improvable.
    init_sub = cfg.get("initial_best_subset")
    best_subset = {"psnr": float(init_sub) if init_sub is not None else -1e9}
    last_ck = time.time()
    micro = 0
    done = False
    epoch = 0
    log("starting training loop...", rank)
    while not done:
        if world > 1:
            sampler.set_epoch(epoch)
        ds.epoch = epoch
        for batch in loader:
            if done:
                break
            video = batch["video"].cuda(non_blocking=True)  # (1,C,T,H,W) uint8
            # (12,7); batch_size==1 is enforced at startup
            action = batch["action"][0].cuda(non_blocking=True)
            db = v2w._get_data_batch_input(
                video=video, prompt="", num_conditional_frames=int(
                    cfg["num_latent_conditional_frames"]),
                negative_prompt=NEG_PROMPT, use_neg_prompt=False, action=action)
            _, loss = model.training_step(db, gstep)
            loss = loss.mean()
            # skip FSDP gradient reduce-scatter on non-final micro-steps
            if accum > 1 and hasattr(net, "set_requires_gradient_sync"):
                net.set_requires_gradient_sync((micro + 1) % accum == 0)
            (loss / accum).backward()
            loss_hist.append(float(loss.detach().float()))
            micro += 1
            if micro % accum == 0:
                gn = torch.nn.utils.clip_grad_norm_(
                    [p for p in net.parameters() if p.requires_grad], grad_clip)
                # lr warmup (+ optional cosine decay toward lr_decay_steps)
                # lr_schedule_offset lets a resumed run open a FRESH schedule
                # window: warmup and cosine fraction are computed on the step
                # count relative to the offset (default 0 = legacy behavior).
                sch_off = int(cfg.get("lr_schedule_offset", 0))
                rel_step = max(gstep - sch_off, 0)
                cur_lr = lr * min(1.0, (rel_step + 1) / max(warmup, 1))
                dec_T = cfg.get("lr_decay_steps")
                if dec_T:
                    frac = min(1.0, max(0.0, rel_step / float(dec_T)))
                    mlr = float(cfg.get("min_lr_ratio", 0.1))
                    cur_lr *= mlr + (1.0 - mlr) * 0.5 * (1.0 + math.cos(math.pi * frac))
                for g in optim.param_groups:
                    g["lr"] = cur_lr
                optim.step()
                optim.zero_grad(set_to_none=True)
                ema.update()
                gstep += 1

                if gstep % log_int == 0 or gstep <= 3:
                    el = time.time() - t_start
                    tl = float(np.mean(loss_hist[-log_int * accum:]))
                    gn_f = float(gn.full_tensor()) if hasattr(gn, "full_tensor") else float(gn)
                    if is0:
                        log(f"step {gstep} loss {tl:.4f} gn {gn_f:.2f} lr {cur_lr:.2e} "
                            f"mem {torch.cuda.max_memory_allocated()/2**30:.1f}GB "
                            f"{gstep/el:.3f} steps/s", rank)
                    jlog({"step": gstep, "epoch": epoch, "elapsed_seconds": el,
                          "train_loss": tl, "grad_norm": gn_f, "learning_rate": cur_lr,
                          "steps_per_second": gstep / el})

                if val_interval and gstep % val_interval == 0 and tm is not None:
                    n_sub = max(2, int(round(len(val_ids) * subset_frac)))
                    ema.swap_in()
                    try:
                        agg, _ = evaluate(v2w, model, val_ids[:n_sub], data_dir, gen_cfg,
                                          scaler_vec, rank, tm)
                    finally:
                        ema.swap_out()
                    jlog({"event": "val_subset", "step": gstep,
                          "elapsed_seconds": time.time() - t_start, **(agg or {})})
                    if is0:
                        log(f"[val subset {n_sub}] psnr={agg['psnr']:.3f}", rank)
                    # insurance: keep best_model.pth fresh mid-run (EMA weights) when
                    # the subset metric improves; does NOT touch best_metric (which is
                    # full-24-ep scale) -- final full eval still decides the real best.
                    if cfg.get("save_best_on_subset", False) and agg is not None:
                        if agg["psnr"] > best_subset["psnr"]:
                            best_subset["psnr"] = agg["psnr"]
                            save_ema_model(agg["psnr"], f"subset{n_sub}")

                if gstep % ck_int_steps == 0 or (time.time() - last_ck) > ck_int_sec:
                    save_ckpt(out_dir / "checkpoint_latest.pth")
                    last_ck = time.time()

                if max_steps is not None and gstep >= int(max_steps):
                    done = True
                if budget is not None and (time.time() - t_start) > float(budget):
                    done = True
            # propagate stop flags consistently across ranks
            flag = torch.tensor([1.0 if (STOP["flag"] or done) else 0.0], device="cuda")
            if dist.is_initialized():
                dist.all_reduce(flag, op=dist.ReduceOp.MAX)
            if flag.item() > 0:
                done = True
        epoch += 1
        if max_steps is None and budget is None and epoch >= int(cfg.get("num_epochs", 1)):
            done = True

    save_ckpt(out_dir / "checkpoint_latest.pth")

    # ---- final eval on frozen list (EMA weights) ----
    results = {}
    if cfg.get("run_eval", True) and tm is None:
        # no task metric available: fall back to (negative) recent training loss
        score = -float(np.mean(loss_hist[-100:])) if loss_hist else -1e9
        log(f"task_metric unavailable; best-model selection falls back to "
            f"-train_loss = {score:.4f}", rank)
        jlog({"event": "final_val", "step": gstep, "score_metric": "neg_train_loss",
              "score": score, "elapsed_seconds": time.time() - t_start})
        if score > best_metric:
            save_best(score, tag="neg_train_loss")
        if is0:
            results = {"score": score, "score_metric": "neg_train_loss",
                       "step": gstep, "protocol": protocol,
                       "elapsed_seconds": time.time() - t_start}
            with open(out_dir / "results.json", "w") as f:
                json.dump(results, f, indent=2)
    elif cfg.get("run_eval", True):
        n_final = int(cfg.get("final_eval_episodes", len(val_ids)))
        final_ids = val_ids[:n_final]
        log(f"final eval on {len(final_ids)} episodes (EMA weights)...", rank)
        ema.swap_in()
        try:
            agg, per_ep = evaluate(v2w, model, final_ids, data_dir, gen_cfg, scaler_vec,
                                   rank, tm, compute_static=True)
        finally:
            ema.swap_out()
        jlog({"event": "final_val", "step": gstep,
              "elapsed_seconds": time.time() - t_start, **(agg or {})})
        score = agg["psnr"]
        if score > best_metric:
            save_best(score)
        else:
            log(f"score {score:.3f} did not beat best {best_metric:.3f}; best_model kept", rank)
        if is0:
            results = {
                "score": score,
                "psnr": score,
                "ssim": agg["ssim"],
                "latent_l2": agg["latent_l2"],
                "pixel_std": agg["pixel_std"],
                "static_null_psnr": agg.get("static_null_psnr"),
                "best_metric": max(best_metric, score),
                "step": gstep,
                "episodes": per_ep,
                "protocol": protocol,
                "elapsed_seconds": time.time() - t_start,
            }
            with open(out_dir / "results.json", "w") as f:
                json.dump(results, f, indent=2)
            log(f"FINAL: psnr={score:.3f} ssim={agg['ssim']:.3f} l2={agg['latent_l2']:.3f}", rank)

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    log("done.", rank)


if __name__ == "__main__":
    main()

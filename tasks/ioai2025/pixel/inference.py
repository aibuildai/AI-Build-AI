"""
Pixel Efficiency -- ExhaustiveSlidingWindow-PseudoLabelAgreement.

No trainable model. Frozen openai/clip-vit-large-patch14 (fp16, cuda:0) is used as the
label-free objective. For each of the 698 images we brute-force a dense grid of candidate
axis-aligned crop boxes (area <= 3136 px), black out everything outside the box, and score
the masked 224x224 image with CLIP zero-shot over the exact grader label set
sorted(9 classes) + ['other']. We select the box whose masked-CLIP prediction is confident
in the SAME animal class the full (unmasked) image predicts (pseudo-label agreement).

This reproduces the grader pipeline exactly: inputs are already 224x224 so CLIP's
do_resize(shortest_edge=224) and do_center_crop(224) are no-ops, and manual
rescale(/255) + normalize(image_mean/std) is bit-identical (verified to ~2e-7) to feeding
the masked PIL image through CLIPImageProcessor. Features are the CLIP projection
embeddings (visual_projection(vision_model(x).pooler_output)) which reproduce
model.logits_per_image (transformers 5.x get_image_features returns encoder output, not
the projected embedding -- so we call the projection explicitly).
"""
import os
import io
import json
import time
import numpy as np
from PIL import Image
import torch

DATA_DIR = os.environ.get("DATA_DIR", os.path.join(os.path.dirname(__file__), "data"))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", os.path.join(os.path.dirname(__file__), "output"))

INSTANCES = ["pixel"]

MODEL_ID = "openai/clip-vit-large-patch14"
DEVICE = "cuda:0"
AREA_CAP = 3136                 # 6.25% of 224x224
STRIDE = 8            # (retained for reference; coarse/refine use their own strides)
STRIDE_COARSE = 16    # coarse full-canvas grid
STRIDE_REFINE = 4     # local refinement grid
REFINE_RADIUS = 12    # +/- pixels around coarse winner top-left
BATCH = 1024
LAMBDA_OTHER = 0.5
FLOOR = 0.30                    # min prob[y_hat] for the primary (agreement) branch
TIE_TOL = 0.02                 # J-tie tolerance -> prefer highest raw prob[y_hat]

# Elongated + square candidate shapes, all area <= 3136.
SHAPES = [(56, 56), (44, 71), (71, 44), (49, 64), (64, 49), (40, 78), (78, 40)]

# CLIP zero-shot prompt ensemble.
TEMPLATES = [
    "a photo of a {}.",
    "a close-up photo of a {}.",
    "a photo of a {}, a type of animal.",
    "a blurry photo of a {}.",
    "a photo of one {}.",
    "a bright photo of a {}.",
    "a cropped photo of a {}.",
]

IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)
IMAGE_STD = (0.26862954, 0.26130258, 0.27577711)


def build_text_bank(model, tokenizer, labels):
    """Prompt-ensembled, L2-normalized text embeddings, one per label. Returns (L, D) fp16."""
    bank = []
    with torch.no_grad():
        for lb in labels:
            texts = [t.format(lb) for t in TEMPLATES]
            tin = tokenizer(texts, padding=True, return_tensors="pt").to(DEVICE)
            tout = model.text_model(input_ids=tin["input_ids"], attention_mask=tin["attention_mask"])
            tf = model.text_projection(tout.pooler_output)          # (n_templates, D)
            tf = tf / tf.norm(dim=-1, keepdim=True)
            e = tf.mean(0)
            e = e / e.norm()
            bank.append(e)
    return torch.stack(bank).half()                                 # (L, D)


def build_coarse_boxes():
    """Coarse full-canvas grid over ALL shapes (stride STRIDE_COARSE).

    Returns (boxes int64 (B,4), shape_id int64 (B,)) where shape_id indexes SHAPES.
    """
    boxes = []
    shape_ids = []
    for sid, (h, w) in enumerate(SHAPES):
        for top in range(0, 224 - h + 1, STRIDE_COARSE):
            for left in range(0, 224 - w + 1, STRIDE_COARSE):
                boxes.append((top, left, top + h, left + w))
                shape_ids.append(sid)
        # Ensure the extreme edge is reachable even if not on the stride grid.
        last_t = 224 - h
        last_l = 224 - w
        if last_t % STRIDE_COARSE != 0:
            for left in range(0, 224 - w + 1, STRIDE_COARSE):
                boxes.append((last_t, left, last_t + h, left + w))
                shape_ids.append(sid)
        if last_l % STRIDE_COARSE != 0:
            for top in range(0, 224 - h + 1, STRIDE_COARSE):
                boxes.append((top, last_l, top + h, last_l + w))
                shape_ids.append(sid)
    return np.asarray(boxes, dtype=np.int64), np.asarray(shape_ids, dtype=np.int64)


def _neighbor_shape_ids(sid):
    """Winning shape + its two nearest-area neighbor shapes (by |area diff|), as ids."""
    aw = SHAPES[sid][0] * SHAPES[sid][1]
    order = sorted(range(len(SHAPES)),
                   key=lambda j: abs(SHAPES[j][0] * SHAPES[j][1] - aw))
    # order[0] is sid itself (area diff 0); take it plus next two.
    return order[:3]


def build_refine_boxes(sid, t_star, l_star):
    """Local stride-REFINE grid around coarse winner top-left, for winning shape and its
    two nearest-area neighbors. Coordinates clamped in-range. Returns int64 (B,4)."""
    boxes = []
    for s in _neighbor_shape_ids(sid):
        h, w = SHAPES[s]
        t_lo = max(0, t_star - REFINE_RADIUS)
        t_hi = min(224 - h, t_star + REFINE_RADIUS)
        l_lo = max(0, l_star - REFINE_RADIUS)
        l_hi = min(224 - w, l_star + REFINE_RADIUS)
        if t_hi < t_lo or l_hi < l_lo:
            continue
        for top in range(t_lo, t_hi + 1, STRIDE_REFINE):
            for left in range(l_lo, l_hi + 1, STRIDE_REFINE):
                boxes.append((top, left, top + h, left + w))
    if not boxes:
        return np.zeros((0, 4), dtype=np.int64)
    return np.asarray(boxes, dtype=np.int64)


def keep_masks_from_boxes(boxes_t):
    """Boolean keep-mask (B,224,224) for a set of boxes on GPU."""
    rows = torch.arange(224, device=DEVICE).view(1, 224, 1)
    cols = torch.arange(224, device=DEVICE).view(1, 1, 224)
    top = boxes_t[:, 0].view(-1, 1, 1)
    left = boxes_t[:, 1].view(-1, 1, 1)
    bot = boxes_t[:, 2].view(-1, 1, 1)
    right = boxes_t[:, 3].view(-1, 1, 1)
    return (rows >= top) & (rows < bot) & (cols >= left) & (cols < right)


def image_features(model, pixel_values):
    """CLIP projected + L2-normalized image embeddings for a normalized pixel batch."""
    feats = model.visual_projection(model.vision_model(pixel_values=pixel_values).pooler_output)
    return feats / feats.norm(dim=-1, keepdim=True)


def main():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    from transformers import CLIPModel, CLIPTokenizer

    for instance in INSTANCES:
        data_path = os.path.join(DATA_DIR, instance)
        output_path = os.path.join(OUTPUT_DIR, instance)
        os.makedirs(output_path, exist_ok=True)

        idxs = open(os.path.join(data_path, "index.txt")).read().split()
        _smoke_n = os.environ.get("PIX_SMOKE_N")            # test-only subset cap; unset => all 698
        if _smoke_n:
            idxs = idxs[:int(_smoke_n)]
        classes = json.load(open(os.path.join(data_path, "classes.json")))["classes"]
        labels = sorted(classes) + ["other"]                       # exact grader label set
        n_anim = len(classes)                                      # first n_anim entries are animals
        other_idx = len(labels) - 1

        model = CLIPModel.from_pretrained(MODEL_ID, torch_dtype=torch.float16).to(DEVICE).eval()
        tokenizer = CLIPTokenizer.from_pretrained(MODEL_ID)
        text_bank = build_text_bank(model, tokenizer, labels)      # (L, D)
        scale = model.logit_scale.exp()

        mean = torch.tensor(IMAGE_MEAN, device=DEVICE).view(1, 3, 1, 1).half()
        std = torch.tensor(IMAGE_STD, device=DEVICE).view(1, 3, 1, 1).half()

        # Coarse box set (fixed across images): precompute boxes, shape ids, and keep-masks once.
        coarse_boxes, coarse_sids = build_coarse_boxes()
        coarse_boxes_t = torch.from_numpy(coarse_boxes).to(DEVICE)
        coarse_keep = keep_masks_from_boxes(coarse_boxes_t)        # (Bc,224,224) bool
        n_coarse = len(coarse_boxes)

        def score_boxes(gsq, keep_masks, y_hat):
            """Batched J / p[y_hat] / best-animal-prob for a set of keep-masks. Returns
            (J (n,), pyhat (n,), anim (n,)) float tensors on GPU."""
            n = keep_masks.shape[0]
            J_all = torch.empty(n, device=DEVICE)
            pyhat_all = torch.empty(n, device=DEVICE)
            anim_all = torch.empty(n, device=DEVICE)
            with torch.no_grad():
                for s in range(0, n, BATCH):
                    kmask = keep_masks[s:s + BATCH].unsqueeze(1)    # (b,1,224,224)
                    masked = gsq.unsqueeze(0) * kmask               # (b,3,224,224)
                    masked = (masked - mean) / std
                    feats = image_features(model, masked)
                    probs = (scale * feats.half() @ text_bank.T).softmax(-1).float()  # (b,L)
                    e = s + probs.shape[0]
                    pyhat_all[s:e] = probs[:, y_hat]
                    J_all[s:e] = probs[:, y_hat] - LAMBDA_OTHER * probs[:, other_idx]
                    anim_all[s:e] = probs[:, :n_anim].max(1).values
            return J_all, pyhat_all, anim_all

        # Resumable output: if a partial submission.jsonl exists, keep valid lines and skip
        # those idxs. Assemble the final ordered file at the end.
        sub_file = os.path.join(output_path, "submission.jsonl")
        done = {}                                                  # idx -> record dict
        if os.path.exists(sub_file):
            with open(sub_file) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        if isinstance(rec, dict) and "idx" in rec and "coordinates" in rec:
                            done[rec["idx"]] = rec
                    except Exception:
                        pass
        idx_set = set(idxs)
        done = {i: r for i, r in done.items() if i in idx_set}     # drop stale ids
        print(f"resume: {len(done)} idxs already present, {len(idxs) - len(done)} remaining",
              flush=True)

        # Open append handle for incremental writes of newly decided images.
        out_f = open(sub_file, "a")

        # Diagnostics accumulators.
        chosen_pyhat = []
        n_fallback = 0
        n_agree = 0
        shape_hist = {}
        pos_hist = {}

        t0 = time.time()
        n_new = 0

        for k, idx in enumerate(idxs):
            if idx in done:
                continue
            img = np.asarray(Image.open(os.path.join(data_path, "images", idx + ".png")).convert("RGB"),
                             dtype=np.uint8)
            g = torch.from_numpy(img).to(DEVICE).permute(2, 0, 1).unsqueeze(0).half() / 255.0  # (1,3,224,224)

            with torch.no_grad():
                full_feat = image_features(model, (g - mean) / std)
                full_probs = (scale * full_feat.half() @ text_bank.T).softmax(-1)[0]           # (L,)
            y_hat = int(full_probs[:n_anim].argmax().item())        # pseudo-label over 9 animals

            gsq = g[0]                                             # (3,224,224)

            # STAGE 1 (coarse): score the full-canvas stride-16 grid over all shapes.
            Jc, pyc, anc = score_boxes(gsq, coarse_keep, y_hat)
            ci = int(Jc.argmax().item())                           # coarse winner (by J)
            sid_star = int(coarse_sids[ci])
            t_star = int(coarse_boxes[ci, 0])
            l_star = int(coarse_boxes[ci, 1])

            # STAGE 2 (refine): local stride-4 grid around coarse winner (winning + 2 neighbor shapes).
            ref_boxes = build_refine_boxes(sid_star, t_star, l_star)
            if len(ref_boxes):
                ref_boxes_t = torch.from_numpy(ref_boxes).to(DEVICE)
                ref_keep = keep_masks_from_boxes(ref_boxes_t)
                Jr, pyr, anr = score_boxes(gsq, ref_keep, y_hat)
                all_boxes = np.concatenate([coarse_boxes, ref_boxes], axis=0)
                J_all = torch.cat([Jc, Jr])
                pyhat_all = torch.cat([pyc, pyr])
                anim_all = torch.cat([anc, anr])
            else:
                all_boxes = coarse_boxes
                J_all, pyhat_all, anim_all = Jc, pyc, anc

            # Primary (agreement) branch: argmax J, tie-break within TIE_TOL by highest p[y_hat].
            jmax = J_all.max()
            near = J_all >= (jmax - TIE_TOL)
            cand_p = torch.where(near, pyhat_all, torch.full_like(pyhat_all, -1.0))
            gi = int(cand_p.argmax().item())
            best_pyhat = float(pyhat_all[gi].item())
            best_box = all_boxes[gi]
            # Chosen box's top-1 == y_hat?  When it clears the floor, J[gi] > 0 requires
            # p[y_hat] to dominate 'other'; agreement is measured against the search probs.
            best_is_agree = (best_pyhat >= float(anim_all[gi].item()) - 1e-6)

            # Fallback: box with globally highest prob over the 9 animals.
            fb_i = int(anim_all.argmax().item())
            fb_box = all_boxes[fb_i]

            used_fallback = best_pyhat < FLOOR
            box = fb_box if used_fallback else best_box
            if used_fallback:
                n_fallback += 1
            elif best_is_agree:
                n_agree += 1

            top_i, left_i, bot_i, right_i = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
            # Hard-enforce validity (defensive; construction already guarantees it).
            top_i = max(0, min(223, top_i)); left_i = max(0, min(223, left_i))
            bot_i = max(top_i + 1, min(224, bot_i)); right_i = max(left_i + 1, min(224, right_i))
            assert (bot_i - top_i) * (right_i - left_i) <= AREA_CAP, (idx, box)
            assert 0 <= top_i < bot_i <= 224 and 0 <= left_i < right_i <= 224

            rec = {"idx": idx, "coordinates": [[top_i, left_i], [bot_i, right_i]]}
            done[idx] = rec
            out_f.write(json.dumps(rec) + "\n")
            out_f.flush()
            n_new += 1

            chosen_pyhat.append(best_pyhat)
            shape_hist[(bot_i - top_i, right_i - left_i)] = shape_hist.get((bot_i - top_i, right_i - left_i), 0) + 1
            cy, cx = (top_i + bot_i) // 2, (left_i + right_i) // 2
            qk = (cy // 56, cx // 56)
            pos_hist[qk] = pos_hist.get(qk, 0) + 1

            if n_new % 50 == 0:
                el = time.time() - t0
                print(f"[{n_new} new / {len(done)} total] {el:.1f}s  {el/max(1,n_new):.2f}s/img  "
                      f"fallback={n_fallback} agree={n_agree}", flush=True)

        out_f.close()

        # Finalize: rewrite submission.jsonl in canonical index.txt order, exactly one line
        # per idx. Assert full coverage before overwriting the (possibly appended) file.
        missing = [i for i in idxs if i not in done]
        assert not missing, f"missing {len(missing)} idxs, e.g. {missing[:5]}"
        with open(sub_file, "w") as f:
            for i in idxs:
                f.write(json.dumps(done[i]) + "\n")
        with open(sub_file) as f:
            n_lines = sum(1 for _ in f)
        assert n_lines == len(idxs), (n_lines, len(idxs))
        print(f"submission finalized: {n_lines} lines", flush=True)

        # Self-diagnostics for the next Reviser (computed over images decided THIS run).
        cp = np.asarray(chosen_pyhat, dtype=np.float64) if chosen_pyhat else np.zeros(1)
        diag = {
            "n_images": len(idxs),
            "n_lines_written": len(done),
            "n_new_this_run": n_new,
            "sec_per_image": (time.time() - t0) / max(1, n_new),
            "chosen_pyhat_mean": float(cp.mean()),
            "chosen_pyhat_median": float(np.median(cp)),
            "chosen_pyhat_p10": float(np.percentile(cp, 10)),
            "chosen_pyhat_frac_below_0.5": float((cp < 0.5).mean()),
            "fallback_rate": n_fallback / max(1, n_new),
            "crop_agreement_rate": n_agree / max(1, n_new),
            "shape_hist": {f"{h}x{w}": c for (h, w), c in sorted(shape_hist.items())},
            "pos_hist_56grid": {f"{r},{c}": v for (r, c), v in sorted(pos_hist.items())},
        }
        with open(os.path.join(output_path, "diagnostics.json"), "w") as f:
            json.dump(diag, f, indent=2)
        print("DIAGNOSTICS", json.dumps(diag), flush=True)

    # Sentinel required by the grading contract.
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "best_model.pkl"), "wb") as f:
        f.write(b"no-trainable-model: exhaustive CLIP crop search (sentinel)\n")


if __name__ == "__main__":
    main()

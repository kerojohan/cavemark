"""
app.py — CaveMark Gradio Space for Hugging Face
Wraps detect_cave.py pipeline to work in-memory (no disk I/O).
"""

import cv2
import numpy as np
import gradio as gr

from detect_cave import (
    preprocess_image,
    compute_valid_region,
    compute_ir_depth,
    generate_candidates,
    select_best_candidate,
    grabcut_refine,
    refine_mask,
)


# ──────────────────────────────────────────────────────────────────────────────
# In-memory draw helpers (mirrors draw_result but returns numpy arrays)
# ──────────────────────────────────────────────────────────────────────────────

def _draw_result_arrays(gray_u8, refined_mask, scores,
                        weight_map, profile_norm,
                        all_candidates, all_scores):
    h, w = gray_u8.shape

    # ── Main result overlay ───────────────────────────────────────────────────
    vis = cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR)

    dil_r = max(5, int(min(h, w) * 0.025))
    dil_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*dil_r+1, 2*dil_r+1))
    dil_mask  = cv2.dilate(refined_mask, dil_k)
    ring_mask = cv2.bitwise_and(dil_mask, cv2.bitwise_not(refined_mask))
    ring_overlay = vis.copy()
    ring_overlay[ring_mask > 0] = (30, 160, 255)
    cv2.addWeighted(ring_overlay, 0.28, vis, 0.72, 0, vis)

    overlay = vis.copy()
    overlay[refined_mask > 0] = (100, 210, 60)
    cv2.addWeighted(overlay, 0.35, vis, 0.65, 0, vis)

    contours, _ = cv2.findContours(refined_mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, contours, -1, (0, 255, 80), 2)

    score_val = scores.get("total", 0.0)
    label = f"cave entrance  score={score_val:.2f}"
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        x, y, bw, bh = cv2.boundingRect(cnt)
        tx, ty = x + 5, max(y - 12, 25)
    else:
        tx, ty = 10, 30

    fs = max(0.55, min(w, h) / 900)
    th = max(1, int(fs * 2))
    cv2.putText(vis, label, (tx+2, ty+2), cv2.FONT_HERSHEY_SIMPLEX,
                fs, (0, 0, 0), th+2)
    cv2.putText(vis, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                fs, (0, 255, 120), th)

    result_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)

    # ── Mask ──────────────────────────────────────────────────────────────────
    mask_rgb = cv2.cvtColor(refined_mask, cv2.COLOR_GRAY2RGB)

    # ── Debug: valid region ───────────────────────────────────────────────────
    dv = cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR)
    for ch in range(3):
        c = dv[:, :, ch].astype(np.float32)
        if ch == 2:
            c = c * weight_map + 180 * (1.0 - weight_map)
        else:
            c = c * weight_map
        dv[:, :, ch] = np.clip(c, 0, 255).astype(np.uint8)
    for col in range(w - 1):
        y1 = h - 1 - int(profile_norm[col]     * 59)
        y2 = h - 1 - int(profile_norm[col + 1] * 59)
        cv2.line(dv, (col, y1), (col+1, y2), (0, 255, 255), 1)
    cv2.putText(dv, "valid region (red=penalised)", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    valid_rgb = cv2.cvtColor(dv, cv2.COLOR_BGR2RGB)

    # ── Debug: candidates ─────────────────────────────────────────────────────
    dc = cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR)
    colours = [(255,80,0),(0,80,255),(200,0,200),(0,200,200),
               (200,200,0),(0,160,80),(128,128,255),(255,128,128)]
    indexed = sorted(range(len(all_candidates)),
                     key=lambda i: all_scores[i]["total"])
    for rank, i in enumerate(indexed):
        col = colours[i % len(colours)]
        cl, _ = cv2.findContours(all_candidates[i], cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(dc, cl, -1, col, 1)
        if rank >= len(indexed) - 5 and cl:
            c0 = max(cl, key=cv2.contourArea)
            M = cv2.moments(c0)
            if M["m00"] > 0:
                cx_m = int(M["m10"] / M["m00"])
                cy_m = int(M["m01"] / M["m00"])
                cv2.putText(dc, f"{all_scores[i]['total']:.2f}", (cx_m, cy_m),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)
    cv2.drawContours(dc, contours, -1, (255, 255, 255), 2)
    cv2.putText(dc,
                f"{len(all_candidates)} candidates (white=best, {score_val:.2f})",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cands_rgb = cv2.cvtColor(dc, cv2.COLOR_BGR2RGB)

    return result_rgb, mask_rgb, valid_rgb, cands_rgb


# ──────────────────────────────────────────────────────────────────────────────
# Full in-memory pipeline
# ──────────────────────────────────────────────────────────────────────────────

def _process_array(img_rgb: np.ndarray):
    """Run the full CaveMark pipeline on a numpy RGB array."""
    # Convert to grayscale
    gray_u8 = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    gray_f32 = gray_u8.astype(np.float32) / 255.0
    h, w = gray_u8.shape

    proc = preprocess_image(gray_u8, gray_f32)
    wmap, lc, rc, pn, actual_lc, actual_rc = compute_valid_region(gray_f32)
    depth_map = compute_ir_depth(gray_f32)

    candidates = generate_candidates(proc, gray_f32, h, w, lc, rc)

    if not candidates:
        blank = np.zeros((h, w), np.uint8)
        blank_rgb = cv2.cvtColor(blank, cv2.COLOR_GRAY2RGB)
        vis_rgb = cv2.cvtColor(cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR),
                               cv2.COLOR_BGR2RGB)
        info = "No cave entrance candidates found."
        return vis_rgb, blank_rgb, blank_rgb, blank_rgb, info

    best_mask, scores, all_sc = select_best_candidate(
        candidates, gray_f32, wmap, lc, rc, depth_map=depth_map
    )

    # Solidity filter
    if scores.get("solidity", 1.0) < 0.65 and np.count_nonzero(best_mask) > 100:
        _is_dark_void = scores.get("mean_inside", 1.0) < 0.15
        mask_weights = wmap[best_mask > 0]
        w_thresh = np.percentile(mask_weights, 50 if _is_dark_void else 60)
        high_w = ((best_mask > 0) & (wmap >= w_thresh)).astype(np.uint8) * 255
        sk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        high_w = cv2.morphologyEx(high_w, cv2.MORPH_CLOSE, sk)
        high_w = cv2.morphologyEx(high_w, cv2.MORPH_OPEN, sk)
        n_hw, labels_hw, stats_hw, centroids_hw = cv2.connectedComponentsWithStats(
            high_w, 8)
        if n_hw > 1:
            valid_comps = []
            for ci in range(1, n_hw):
                cx_ci   = centroids_hw[ci, 0]
                area_ci = stats_hw[ci, cv2.CC_STAT_AREA]
                if lc <= cx_ci <= rc and area_ci >= np.count_nonzero(best_mask) * 0.10:
                    valid_comps.append((ci, area_ci))
            if valid_comps:
                best_ci = max(valid_comps, key=lambda x: x[1])[0]
                best_mask = ((labels_hw == best_ci) * 255).astype(np.uint8)
            else:
                largest = 1 + np.argmax(stats_hw[1:, cv2.CC_STAT_AREA])
                candidate_hw = ((labels_hw == largest) * 255).astype(np.uint8)
                if np.count_nonzero(candidate_hw) >= np.count_nonzero(best_mask) * 0.15:
                    best_mask = candidate_hw

    # Post-selection expansion
    pre_expansion_mask = best_mask.copy()
    best_area_frac = np.count_nonzero(best_mask) / (h * w)
    if best_area_frac < 0.25:
        orig_mean = float(gray_f32[best_mask > 0].mean())
        br_size = max(9, int(min(h, w) * 0.02) | 1)
        br_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (br_size, br_size))
        reach_r = max(15, int(min(h, w) * 0.04))
        reach_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                             (2*reach_r+1, 2*reach_r+1))
        base_pct = min(50, max(30, int(scores.get("area_frac", 0.1) * 100 * 4)))
        relax_thr = int(np.percentile(proc["denoised"], base_pct))
        _, relax_dark = cv2.threshold(proc["denoised"], relax_thr, 255,
                                       cv2.THRESH_BINARY_INV)
        relax_dark = cv2.morphologyEx(relax_dark, cv2.MORPH_CLOSE, br_k)
        n_rd, labels_rd, _, _ = cv2.connectedComponentsWithStats(relax_dark, 8)
        seed_reach = cv2.dilate(best_mask, reach_k)
        overlap_labels = set(np.unique(labels_rd[seed_reach > 0])) - {0}
        if overlap_labels:
            expanded = np.zeros_like(best_mask)
            for lb in overlap_labels:
                expanded[labels_rd == lb] = 255
            clip_lc = actual_lc if actual_lc > lc else lc
            clip_rc = actual_rc if actual_rc < rc else rc
            if clip_lc > int(w * 0.05):
                expanded[:, :clip_lc] = 0
            if clip_rc < int(w * 0.95):
                expanded[:, clip_rc+1:] = 0
            n_exp, labels_exp, stats_exp, _ = cv2.connectedComponentsWithStats(
                expanded, 8)
            if n_exp > 1:
                largest_exp = 1 + np.argmax(stats_exp[1:, cv2.CC_STAT_AREA])
                expanded = ((labels_exp == largest_exp) * 255).astype(np.uint8)
            exp_area_frac = np.count_nonzero(expanded) / (h * w)
            exp_mean = float(gray_f32[expanded > 0].mean())
            if (exp_area_frac <= 0.40
                    and exp_area_frac > best_area_frac * 0.8
                    and exp_mean < orig_mean + 0.15):
                best_mask = expanded
                best_area_frac = exp_area_frac

    # GrabCut
    pre_gc = np.count_nonzero(best_mask) / (h * w)
    pre_exp_frac = np.count_nonzero(pre_expansion_mask) / (h * w)
    use_conservative = (pre_gc > pre_exp_frac * 1.3)
    gc_result = grabcut_refine(
        gray_u8, best_mask,
        conservative_mask=pre_expansion_mask if use_conservative else None,
        expand_ratio=2.5,
    )
    if np.count_nonzero(gc_result) > 0:
        best_mask = gc_result

    refined = refine_mask(best_mask, gray_f32)

    result_rgb, mask_rgb, valid_rgb, cands_rgb = _draw_result_arrays(
        gray_u8, refined, scores, wmap, pn, candidates, all_sc
    )

    final_area = np.count_nonzero(refined) / (h * w)
    info = (
        f"**Score:** {scores['total']:.2f}  |  "
        f"**Area:** {final_area*100:.1f}%  |  "
        f"**Contrast:** {scores['contrast']:.2f}  |  "
        f"**IR depth:** {scores['ir_depth']:.2f}  |  "
        f"**Darkness:** {scores['dark']:.2f}  |  "
        f"**Texture mult:** {scores['texture_mult']:.2f}  |  "
        f"**Candidates:** {len(candidates)}"
    )
    return result_rgb, mask_rgb, valid_rgb, cands_rgb, info


# ──────────────────────────────────────────────────────────────────────────────
# Gradio interface
# ──────────────────────────────────────────────────────────────────────────────

def detect(image):
    if image is None:
        return None, None, None, None, "No image provided."
    return _process_array(image)


with gr.Blocks(title="CaveMark — Cave Entrance Detector") as demo:
    gr.Markdown(
        """
# CaveMark — Automatic Cave Entrance Detector

Classical computer vision pipeline (OpenCV + NumPy) that locates cave entrances
in IR/NIR monochrome imagery — **no deep learning required**.

Upload an IR or NIR image from a trail camera, security camera or similar sensor.
The pipeline runs: preprocess → valid-region → IR-depth → candidates → score →
expand → GrabCut → refine → visualise.
        """
    )

    with gr.Row():
        inp = gr.Image(label="Input image", type="numpy")
        btn = gr.Button("Detect cave entrance", variant="primary")

    info_box = gr.Markdown(label="Detection summary")

    with gr.Row():
        out_result = gr.Image(label="Result overlay")
        out_mask   = gr.Image(label="Binary mask")

    with gr.Row():
        out_valid = gr.Image(label="Valid-region weight map")
        out_cands = gr.Image(label="Candidate scoring debug")

    btn.click(
        fn=detect,
        inputs=inp,
        outputs=[out_result, out_mask, out_valid, out_cands, info_box],
    )

    gr.Examples(
        examples=[
            ["examples/background.png"],
            ["examples/background2.png"],
            ["examples/background3.png"],
            ["examples/background4.png"],
            ["examples/background5.png"],
            ["examples/background6.png"],
            ["examples/background7.png"],
            ["examples/background8.png"],
        ],
        inputs=inp,
        outputs=[out_result, out_mask, out_valid, out_cands, info_box],
        fn=detect,
        cache_examples=True,
    )

    gr.Markdown(
        """
---
**How it works:** [GitHub repo](https://github.com/kerojohan/cavemark) · MIT License
        """
    )

if __name__ == "__main__":
    demo.launch()

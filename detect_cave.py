#!/usr/bin/env python3
"""
detect_cave.py — Automatic cave entrance detector for IR/NIR imagery.

Usage:
    python detect_cave.py                      # batch: all jpg/png in current dir
    python detect_cave.py img1.png img2.png    # specific images

v4 — Improved pipeline (opencv + numpy only, no external models):
  - IR physics depth map (darkness × multi-scale local uniformity) as new signal
  - Texture gate: penalises textured rock/vegetation masquerading as voids
  - Vertical centroid gate: suppresses top-of-frame artefacts
  - GrabCut boundary refinement after candidate selection
  - Contour smoothing in refine_mask (wrap-around Gaussian)
  - Amber dilation ring in result visualisation
"""

import cv2
import numpy as np
import os
import sys
import glob


# ──────────────────────────────────────────────────────────────────────────────
# 1. LOAD
# ──────────────────────────────────────────────────────────────────────────────

def load_image(path: str):
    """Load image → (gray_u8, gray_f32 [0..1])."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {path}")
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    gray_u8 = gray.astype(np.uint8)
    gray_f32 = gray_u8.astype(np.float32) / 255.0
    return gray_u8, gray_f32


# ──────────────────────────────────────────────────────────────────────────────
# 2. PREPROCESS
# ──────────────────────────────────────────────────────────────────────────────

def preprocess_image(gray_u8, gray_f32):
    """
    Gentle preprocessing:
      - Median denoise
      - Very-large-blur background illumination estimate
      - Division normalisation (preserves cave darkness relative to local bg)
    """
    h, w = gray_u8.shape

    denoised = cv2.medianBlur(gray_u8, 5)

    # Background: huge blur (40% of min dimension)
    bg_k = max(3, int(min(h, w) * 0.40) | 1)
    background = cv2.GaussianBlur(denoised.astype(np.float32),
                                   (bg_k, bg_k), 0)
    background = np.clip(background, 10.0, 255.0)

    # Division normalisation
    corrected_f = denoised.astype(np.float32) / background
    corrected_u8 = np.clip(corrected_f * 170, 0, 255).astype(np.uint8)

    return {
        "denoised": denoised,
        "background": background,
        "corrected_u8": corrected_u8,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 3. VALID REGION
# ──────────────────────────────────────────────────────────────────────────────

def compute_valid_region(gray_f32):
    """
    Soft weight map based on horizontal illumination profile.
    Uses 80th percentile per column, smoothed.
    Returns (weight_map, left_col, right_col, profile_norm).
    """
    h, w = gray_f32.shape

    col_profile = np.percentile(gray_f32, 80, axis=0).astype(np.float32)
    smooth_k = max(5, int(w * 0.08) | 1)
    profile_smooth = cv2.GaussianBlur(
        col_profile.reshape(1, -1), (smooth_k, 1), 0
    ).flatten()

    pmax = max(profile_smooth.max(), 1e-6)
    profile_norm = profile_smooth / pmax

    drop_thresh = 0.45
    actual_left_col = 0
    for c in range(w):
        if profile_norm[c] >= drop_thresh:
            actual_left_col = c
            break
    actual_right_col = w - 1
    for c in range(w - 1, -1, -1):
        if profile_norm[c] >= drop_thresh:
            actual_right_col = c
            break

    left_col  = min(actual_left_col,  int(w * 0.30))
    right_col = max(actual_right_col, int(w * 0.70))

    # Soft weight map
    weight_row = np.ones(w, dtype=np.float32)
    for c in range(w):
        if c < left_col:
            weight_row[c] = 0.3 + 0.7 * c / max(left_col, 1)
        elif c > right_col:
            weight_row[c] = 0.3 + 0.7 * (w - 1 - c) / max(w - 1 - right_col, 1)
    weight_row *= np.clip(profile_norm / drop_thresh, 0.3, 1.0)
    weight_row = np.clip(weight_row, 0.0, 1.0)

    weight_map = np.tile(weight_row, (h, 1))
    return weight_map, left_col, right_col, profile_norm, actual_left_col, actual_right_col


# ──────────────────────────────────────────────────────────────────────────────
# 4. IR PHYSICS DEPTH
# ──────────────────────────────────────────────────────────────────────────────

def compute_ir_depth(gray_f32):
    """
    Fast IR physics depth map: darkness × local_uniformity at 3 scales.

    Cave voids absorb all IR → near-black AND very uniform.
    Textured surfaces (rock, vegetation) can appear dark but non-uniform.
    Returns a depth map in [0..1]; higher = more likely to be a deep cavity.
    """
    h, w = gray_f32.shape
    darkness = 1.0 - gray_f32
    depths = []
    for base_k in [15, 31, 61]:
        ksize = max(3, min(base_k, min(h, w) // 3) | 1)
        mean_l  = cv2.GaussianBlur(gray_f32, (ksize, ksize), 0)
        mean_sq = cv2.GaussianBlur(gray_f32 * gray_f32, (ksize, ksize), 0)
        var_l   = np.clip(mean_sq - mean_l * mean_l, 0.0, None)
        std_l   = np.sqrt(var_l)
        # uniformity: 1 when perfectly uniform, drops toward 0 with high relative std
        denom      = np.clip(mean_l + 0.05, 0.05, None)
        uniformity = 1.0 - np.clip(std_l / denom, 0.0, 1.0)
        depths.append(darkness * uniformity)
    return np.mean(depths, axis=0).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# 5. GENERATE CANDIDATES
# ──────────────────────────────────────────────────────────────────────────────

def _extract_components(binary, min_area):
    """Extract connected components ≥ min_area after morphological cleaning."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  k)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    result = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            result.append(((labels == i) * 255).astype(np.uint8))
    return result


def _extract_components_heavy(binary, min_area, h, w):
    """Extract components with HEAVY morphological bridging (large closing).
    Bridges fragmented dark spots that belong to the same cave entrance."""
    close_size = max(15, int(min(h, w) * 0.04) | 1)
    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                         (close_size, close_size))
    bridged = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_k)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    bridged = cv2.morphologyEx(bridged, cv2.MORPH_OPEN, k)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bridged, 8)
    result = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            result.append(((labels == i) * 255).astype(np.uint8))
    return result


def generate_candidates(proc, gray_f32, h, w, left_col=0, right_col=None):
    """
    Multi-strategy candidate generation:
      A. Multi-level thresholding with standard cleaning
      B. Multi-level thresholding with heavy bridging
      C. Iterative seed-growth from darkest pixels
      D. Otsu thresholding
      E. Adaptive threshold intersected with a dark base
      F. Valid-zone-only masking (lateral shadows masked out)
    """
    denoised     = proc["denoised"]
    corrected_u8 = proc["corrected_u8"]

    candidates = []
    min_area = int(h * w * 0.008)  # 0.8%

    # ── A. Multi-level standard thresholding ──────────────────────────────────
    for pct in [10, 15, 20, 25, 30, 35, 40]:
        thr = int(np.percentile(denoised, pct))
        _, binary = cv2.threshold(denoised, thr, 255, cv2.THRESH_BINARY_INV)
        candidates += _extract_components(binary, min_area)

    # Same on corrected
    for pct in [15, 25, 35, 45]:
        thr = int(np.percentile(corrected_u8, pct))
        _, binary = cv2.threshold(corrected_u8, thr, 255, cv2.THRESH_BINARY_INV)
        candidates += _extract_components(binary, min_area)

    # ── B. Multi-level with HEAVY bridging ────────────────────────────────────
    for pct in [10, 15, 20, 25, 30, 35]:
        thr = int(np.percentile(denoised, pct))
        _, binary = cv2.threshold(denoised, thr, 255, cv2.THRESH_BINARY_INV)
        candidates += _extract_components_heavy(binary, min_area, h, w)

    # ── C. Iterative seed-growth ──────────────────────────────────────────────
    p1 = int(np.percentile(denoised, 1))
    _, seed = cv2.threshold(denoised, max(p1, 3), 255, cv2.THRESH_BINARY_INV)
    seed_k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (max(7, int(min(h, w) * 0.03) | 1),
         max(7, int(min(h, w) * 0.03) | 1))
    )
    seed = cv2.morphologyEx(seed, cv2.MORPH_CLOSE, seed_k)

    for pct in [5, 10, 15, 20, 25, 30, 35, 40, 50]:
        thr = int(np.percentile(denoised, pct))
        _, dark_level = cv2.threshold(denoised, thr, 255, cv2.THRESH_BINARY_INV)
        grow_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        grown = cv2.dilate(seed, grow_k, iterations=2)
        grown = cv2.bitwise_and(grown, dark_level)
        grown = cv2.morphologyEx(grown, cv2.MORPH_CLOSE, seed_k)
        seed = cv2.bitwise_or(seed, grown)
        candidates += _extract_components(grown, min_area)

    # ── D. Otsu ───────────────────────────────────────────────────────────────
    _, th_otsu = cv2.threshold(denoised, 0, 255,
                               cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    candidates += _extract_components(th_otsu, min_area)
    candidates += _extract_components_heavy(th_otsu, min_area, h, w)

    # ── E. Adaptive + dark base ───────────────────────────────────────────────
    block = max(11, int(min(h, w) * 0.15) | 1)
    th_adapt = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=block, C=10
    )
    med_val = int(np.median(denoised))
    _, dark_base = cv2.threshold(denoised, med_val, 255, cv2.THRESH_BINARY_INV)
    combined = cv2.bitwise_and(th_adapt, dark_base)
    candidates += _extract_components_heavy(combined, min_area, h, w)

    # ── F. Valid-zone-only thresholding ──────────────────────────────────────
    if right_col is None:
        right_col = w - 1
    if left_col > 10 or right_col < w - 11:
        masked_den = denoised.copy()
        masked_den[:, :left_col] = 255
        masked_den[:, right_col+1:] = 255
        for pct in [10, 15, 20, 25, 30, 35, 40]:
            thr = int(np.percentile(denoised, pct))
            _, binary = cv2.threshold(masked_den, thr, 255, cv2.THRESH_BINARY_INV)
            candidates += _extract_components(binary, min_area)
            candidates += _extract_components_heavy(binary, min_area, h, w)

    # ── Deduplicate (IoU > 0.80) ──────────────────────────────────────────────
    unique = []
    for cand in candidates:
        cand_nz = np.count_nonzero(cand)
        is_dup = False
        for ref in unique:
            inter = np.count_nonzero(cand & ref)
            union = cand_nz + np.count_nonzero(ref) - inter
            if union > 0 and inter / union > 0.80:
                is_dup = True
                break
        if not is_dup:
            unique.append(cand)

    return unique


# ──────────────────────────────────────────────────────────────────────────────
# 6. SCORE A CANDIDATE
# ──────────────────────────────────────────────────────────────────────────────

def score_candidate(mask, gray_f32, weight_map, left_col, right_col,
                    darkest5_mask, depth_map=None):
    """
    Multi-criteria scoring with MULTIPLICATIVE gates.

    Key design:
      - Contrast vs surround is the primary additive signal
      - IR physics depth rewards dark AND uniform regions (true voids)
      - Texture gate (multiplicative) penalises textured rock/vegetation
      - Vertical gate (multiplicative) suppresses top-frame artefacts
      - Area and solidity are MULTIPLICATIVE — wrong size/shape kills score
    """
    h, w = gray_f32.shape
    img_area = h * w
    mask_bool = mask.astype(bool)
    area = int(mask_bool.sum())
    if area < 10:
        return {"total": -1.0}

    # ── Geometry ──────────────────────────────────────────────────────────────
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {"total": -1.0}
    cnt = max(contours, key=cv2.contourArea)
    cnt_area = cv2.contourArea(cnt)
    hull_area = cv2.contourArea(cv2.convexHull(cnt))
    solidity = cnt_area / hull_area if hull_area > 0 else 0.0
    x, y, bw, bh = cv2.boundingRect(cnt)
    aspect = min(bw, bh) / max(bw, bh) if max(bw, bh) > 0 else 0.0
    area_frac = area / img_area

    # ── Intensity ─────────────────────────────────────────────────────────────
    vals_inside = gray_f32[mask_bool]
    mean_inside = float(vals_inside.mean())
    std_inside  = float(vals_inside.std())
    darkness    = 1.0 - mean_inside

    # ── 1. Contrast vs wide surround (ADDITIVE, primary) ─────────────────────
    ring_width = max(40, int(min(h, w) * 0.08))
    dil_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (ring_width, ring_width))
    dilated = cv2.dilate(mask, dil_k)
    ring = dilated.astype(bool) & (~mask_bool)
    if ring.sum() > 100:
        mean_outside = float(gray_f32[ring].mean())
    else:
        mean_outside = float(gray_f32.mean())
    contrast = mean_outside - mean_inside
    contrast_score = np.clip(contrast / 0.25, 0.0, 1.0)

    # ── 2. Darkness score (ADDITIVE) ──────────────────────────────────────────
    dark_score = np.clip(darkness / 0.7, 0.0, 1.0)

    # ── 3. Darkness enrichment (ADDITIVE) ─────────────────────────────────────
    darkest5_bool = darkest5_mask.astype(bool)
    total_darkest = max(darkest5_bool.sum(), 1)
    contained_frac = float((mask_bool & darkest5_bool).sum()) / total_darkest
    enrichment = contained_frac / max(area_frac, 0.001)
    enrichment_score = np.clip((enrichment - 1.0) / 8.0, 0.0, 1.0)

    # ── 4. Distance-transform depth (ADDITIVE) ───────────────────────────────
    dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    max_dist = float(dist_transform.max())
    ref_dist = min(h, w) * 0.12
    depth_score = np.clip(max_dist / ref_dist, 0.0, 1.0)

    # ── 5. IR physics depth (ADDITIVE) ────────────────────────────────────────
    # Cave voids are dark AND spatially uniform; textured surfaces score lower
    if depth_map is not None:
        ir_depth_score = float(np.clip(depth_map[mask_bool].mean() / 0.5, 0.0, 1.0))
    else:
        ir_depth_score = dark_score * 0.5   # fallback without depth map

    # ── 6. Boundary gradient (ADDITIVE) ───────────────────────────────────────
    grad_x = cv2.Sobel(gray_f32, cv2.CV_32F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(gray_f32, cv2.CV_32F, 0, 1, ksize=5)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    thin_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    contour_ring = cv2.dilate(mask, thin_k) - cv2.erode(mask, thin_k)
    contour_bool = contour_ring.astype(bool)
    if contour_bool.sum() > 0:
        gradient_score = np.clip(float(grad_mag[contour_bool].mean()) / 0.10,
                                  0.0, 1.0)
    else:
        gradient_score = 0.0

    # ── 7. Valid region alignment (ADDITIVE) ──────────────────────────────────
    valid_score = float(weight_map[mask_bool].mean())

    # ── 8. Aspect ratio (ADDITIVE) ────────────────────────────────────────────
    aspect_score = 1.0 if aspect >= 0.15 else aspect / 0.15

    # ── 9. Position (ADDITIVE, very mild centre bias) ─────────────────────────
    cx = x + bw / 2.0
    cy = y + bh / 2.0
    dist_x = abs(cx / w - 0.5) * 2
    dist_y = abs(cy / h - 0.5) * 2
    position_score = 1.0 - 0.10 * dist_x - 0.05 * dist_y

    # ── Additive base score ───────────────────────────────────────────────────
    # Weights sum to 1.0:
    # 0.24+0.14+0.06+0.12+0.10+0.09+0.06+0.03+0.04+0.12 = 1.00
    additive = (
        0.24 * contrast_score
      + 0.14 * dark_score
      + 0.06 * enrichment_score
      + 0.12 * depth_score          # distance-transform depth
      + 0.10 * ir_depth_score       # IR physics depth (darkness × uniformity)
      + 0.09 * gradient_score
      + 0.06 * valid_score
      + 0.03 * aspect_score
      + 0.04 * position_score
      + 0.12 * 1.0                  # base
    )

    # ── MULTIPLICATIVE GATES ──────────────────────────────────────────────────

    # Area gate: 8%–28% ideal; large blobs (>28%) are usually outdoor
    # shadows, not cave entrances — penalise them more steeply than before.
    if area_frac < 0.005:
        area_mult = 0.05
    elif area_frac < 0.02:
        area_mult = 0.05 + 0.20 * (area_frac - 0.005) / 0.015
    elif area_frac < 0.04:
        area_mult = 0.25 + 0.25 * (area_frac - 0.02) / 0.02
    elif area_frac < 0.08:
        area_mult = 0.50 + 0.50 * (area_frac - 0.04) / 0.04
    elif area_frac <= 0.28:
        area_mult = 1.0
    elif area_frac <= 0.45:
        area_mult = 1.0 - 0.80 * (area_frac - 0.28) / 0.17
    else:
        area_mult = max(0.05, 0.20 - 0.15 * (area_frac - 0.45) / 0.55)

    # Solidity gate: very non-convex (donut, tentacles) → penalised
    if solidity >= 0.45:
        solidity_mult = 1.0
    elif solidity >= 0.25:
        solidity_mult = 0.4 + 0.6 * (solidity - 0.25) / 0.20
    else:
        solidity_mult = 0.4

    # Texture gate: cave voids are dark AND uniform; textured regions penalised.
    # std_inside > 0.10 starts the ramp; above 0.22 caps at 0.60.
    if std_inside <= 0.10:
        texture_mult = 1.0
    elif std_inside <= 0.22:
        texture_mult = 1.0 - 0.40 * (std_inside - 0.10) / 0.12
    else:
        texture_mult = 0.60

    # Vertical gate: entrances in the top quarter of the frame are unlikely.
    # Penalises bright sky patches and illuminated ceiling artefacts.
    if cy / h < 0.25:
        vert_gate = 0.75
    elif cy / h < 0.35:
        vert_gate = 0.75 + 0.25 * (cy / h - 0.25) / 0.10
    else:
        vert_gate = 1.0

    # Lateral penalty
    lateral_pen = 1.0
    if int(cx) < left_col or int(cx) > right_col:
        if valid_score < 0.5:
            lateral_pen = 0.4

    # Border contact penalty: true cave entrances are contained within the frame.
    # Dark patches that bleed to an image edge are likely cast shadows or vegetation
    # extending beyond the visible area, not a cave void.
    bp = 3  # border margin in pixels
    border_touches = int(
        (mask[:bp, :] > 0).any()       # top
        + (mask[-bp:, :] > 0).any()    # bottom
        + (mask[:, :bp] > 0).any()     # left
        + (mask[:, -bp:] > 0).any()    # right
    )
    if border_touches == 0:
        border_mult = 1.0
    elif border_touches == 1:
        border_mult = 0.60
    else:
        border_mult = 0.35

    total = (additive * area_mult * solidity_mult
             * texture_mult * vert_gate * lateral_pen * border_mult)

    return {
        "total":          round(float(total), 4),
        "additive":       round(float(additive), 3),
        "contrast":       round(float(contrast_score), 3),
        "dark":           round(float(dark_score), 3),
        "enrichment":     round(float(enrichment_score), 3),
        "depth":          round(float(depth_score), 3),
        "ir_depth":       round(float(ir_depth_score), 3),
        "texture":        round(float(std_inside), 3),
        "texture_mult":   round(float(texture_mult), 3),
        "vert_gate":      round(float(vert_gate), 3),
        "area_mult":      round(float(area_mult), 3),
        "area_frac":      round(float(area_frac), 4),
        "solidity":       round(float(solidity), 3),
        "sol_mult":       round(float(solidity_mult), 3),
        "gradient":       round(float(gradient_score), 3),
        "valid_score":    round(float(valid_score), 3),
        "mean_inside":    round(float(mean_inside), 3),
        "mean_outside":   round(float(mean_outside), 3),
        "border_touches": border_touches,
        "border_mult":    round(float(border_mult), 3),
    }


# ──────────────────────────────────────────────────────────────────────────────
# 7. SELECT BEST
# ──────────────────────────────────────────────────────────────────────────────

def select_best_candidate(candidates, gray_f32, weight_map,
                          left_col, right_col, depth_map=None):
    """Score all candidates, return (best_mask, best_scores, all_scores)."""
    if not candidates:
        return None, {}, []

    p5 = np.percentile(gray_f32, 5)
    darkest5_mask = (gray_f32 <= p5).astype(np.uint8) * 255

    all_scores = []
    for cand in candidates:
        sc = score_candidate(cand, gray_f32, weight_map, left_col, right_col,
                             darkest5_mask, depth_map=depth_map)
        all_scores.append(sc)

    best_idx = max(range(len(all_scores)),
                   key=lambda i: all_scores[i]["total"])
    return candidates[best_idx], all_scores[best_idx], all_scores


# ──────────────────────────────────────────────────────────────────────────────
# 8. GRABCUT REFINE
# ──────────────────────────────────────────────────────────────────────────────

def grabcut_refine(gray_u8, mask, conservative_mask=None, expand_ratio=2.5):
    """
    Refine mask boundary using GrabCut (OpenCV graph-cut, no extra deps).

    If conservative_mask is provided (the pre-expansion baseline), it is used
    as definite FG so GrabCut anchors on the known-good core and can include
    additional dark interior pixels without over-trimming.

    Without conservative_mask: eroded core is definite FG.
    With conservative_mask: conservative_mask is definite FG; extra pixels in
    mask become probable FG, letting GrabCut decide which dark interior areas
    (e.g. cave floor below a rock band) are genuinely part of the entrance.
    """
    h, w = gray_u8.shape
    area = np.count_nonzero(mask)
    if area < 200:
        return mask

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask
    cnt = max(contours, key=cv2.contourArea)
    x, y, bw, bh = cv2.boundingRect(cnt)
    if bw < 5 or bh < 5:
        return mask

    # Expand bounding rectangle
    ex = int(bw * (expand_ratio - 1) / 2)
    ey = int(bh * (expand_ratio - 1) / 2)
    x1 = max(0, x - ex);  y1 = max(0, y - ey)
    x2 = min(w, x + bw + ex);  y2 = min(h, y + bh + ey)
    if x2 - x1 < 5 or y2 - y1 < 5:
        return mask

    gc_mask = np.full((h, w), cv2.GC_BGD, dtype=np.uint8)

    # Probable FG: dilated mask clipped to expanded rect
    dil_r = max(5, int(min(bw, bh) * 0.10))
    dil_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*dil_r+1, 2*dil_r+1))
    prob_fg = cv2.dilate(mask, dil_k)
    prob_fg[:y1, :] = 0;  prob_fg[y2:, :] = 0
    prob_fg[:, :x1] = 0;  prob_fg[:, x2:] = 0
    gc_mask[prob_fg > 0] = cv2.GC_PR_FGD

    if conservative_mask is not None and np.count_nonzero(conservative_mask) >= 10:
        # Definite FG: the pre-expansion mask (known-good entrance core)
        gc_mask[conservative_mask > 0] = cv2.GC_FGD
    else:
        # Definite FG: eroded core of input mask
        ero_r = max(3, int(min(bw, bh) * 0.08))
        ero_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                           (2*ero_r+1, 2*ero_r+1))
        core = cv2.erode(mask, ero_k)
        gc_mask[core > 0] = cv2.GC_FGD

    # Probable BG: inside expanded rect but well away from mask
    far_r = max(7, int(min(bw, bh) * 0.20))
    far_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*far_r+1, 2*far_r+1))
    far_dil = cv2.dilate(mask, far_k)
    in_rect = np.zeros((h, w), np.uint8)
    in_rect[y1:y2, x1:x2] = 255
    prob_bg = cv2.bitwise_and(in_rect, cv2.bitwise_not(far_dil))
    gc_mask[prob_bg > 0] = cv2.GC_PR_BGD

    if (gc_mask == cv2.GC_FGD).sum() < 10:
        return mask

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    rect = (x1, y1, x2 - x1, y2 - y1)

    try:
        vis3 = cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR)
        cv2.grabCut(vis3, gc_mask, rect, bgd_model, fgd_model,
                    3, cv2.GC_INIT_WITH_MASK)
        result = np.where(
            (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD),
            255, 0
        ).astype(np.uint8)

        if np.count_nonzero(result) < area * 0.25:
            return mask

        # Keep the largest component that overlaps the original mask.
        # (Guards against GrabCut fragmenting into many tiny pieces.)
        n_comp, labels, stats, _ = cv2.connectedComponentsWithStats(result, 8)
        if n_comp > 2:
            overlap_ids = np.unique(labels[mask > 0])
            overlap_ids = overlap_ids[overlap_ids != 0]
            if len(overlap_ids) > 0:
                keep_id = overlap_ids[
                    np.argmax(stats[overlap_ids, cv2.CC_STAT_AREA])
                ]
                result = ((labels == keep_id) * 255).astype(np.uint8)

        if np.count_nonzero(result) < area * 0.25:
            return mask

        return result

    except Exception:
        return mask


# ──────────────────────────────────────────────────────────────────────────────
# 9. REFINE MASK
# ──────────────────────────────────────────────────────────────────────────────

def refine_mask(mask, gray_f32):
    """Close gaps, fill holes, smooth boundary, keep largest component."""
    h, w = gray_f32.shape
    orig_area = np.count_nonzero(mask)

    cs = max(11, int(min(h, w) * 0.02) | 1)
    ck = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cs, cs))
    refined = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, ck)

    # Fill interior holes safely using a 1-px black border.
    bordered = np.zeros((h + 2, w + 2), np.uint8)
    bordered[1:-1, 1:-1] = refined
    flood = bordered.copy()
    pad = np.zeros((h + 4, w + 4), np.uint8)
    cv2.floodFill(flood, pad, (0, 0), 255)
    holes = cv2.bitwise_not(flood)[1:-1, 1:-1]
    refined = cv2.bitwise_or(refined, holes)

    # Smooth
    sk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, sk)
    refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN,  sk)

    # Largest component only
    n, labels, stats, _ = cv2.connectedComponentsWithStats(refined, 8)
    if n > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        refined = ((labels == largest) * 255).astype(np.uint8)

    # ── Contour smoothing (wrap-around Gaussian) ──────────────────────────────
    cnts, _ = cv2.findContours(refined, cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_NONE)
    if cnts:
        main_cnt = max(cnts, key=cv2.contourArea)
        pts = main_cnt.reshape(-1, 2).astype(np.float32)
        n_pts = len(pts)
        if n_pts > 30:
            sigma = min(15.0, max(4.0, n_pts / 120.0))
            ksize = max(3, int(6 * sigma) | 1)
            pad_n = ksize // 2
            padded = np.concatenate([pts[-pad_n:], pts, pts[:pad_n]], axis=0)
            kernel = cv2.getGaussianKernel(ksize, sigma).flatten()
            sx = np.convolve(padded[:, 0], kernel, mode='valid')[:n_pts]
            sy = np.convolve(padded[:, 1], kernel, mode='valid')[:n_pts]
            sx = np.clip(sx, 0, w - 1)
            sy = np.clip(sy, 0, h - 1)
            smooth_cnt = (np.stack([sx, sy], axis=1)
                          .astype(np.int32).reshape(-1, 1, 2))
            smooth_mask = np.zeros_like(refined)
            cv2.fillPoly(smooth_mask, [smooth_cnt], 255)
            # Safety: don't shrink more than 30%
            if np.count_nonzero(smooth_mask) >= np.count_nonzero(refined) * 0.70:
                refined = smooth_mask

    # Safety: revert if refinement bloated the mask beyond 2× original
    if np.count_nonzero(refined) > max(orig_area * 2, h * w * 0.50):
        return mask

    return refined


# ──────────────────────────────────────────────────────────────────────────────
# 10. DRAW RESULT
# ──────────────────────────────────────────────────────────────────────────────

def draw_result(gray_u8, refined_mask, scores,
                out_path, mask_path, debug_valid_path,
                weight_map, profile_norm,
                debug_cands_path, all_candidates, all_scores):
    """Save result overlay, mask, and debug images."""
    h, w = gray_u8.shape

    # ── Main result ───────────────────────────────────────────────────────────
    vis = cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR)

    # Amber dilation ring — buffer zone around detected entrance
    dil_r = max(5, int(min(h, w) * 0.025))
    dil_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*dil_r+1, 2*dil_r+1))
    dil_mask  = cv2.dilate(refined_mask, dil_k)
    ring_mask = cv2.bitwise_and(dil_mask, cv2.bitwise_not(refined_mask))
    ring_overlay = vis.copy()
    ring_overlay[ring_mask > 0] = (30, 160, 255)   # amber (BGR)
    cv2.addWeighted(ring_overlay, 0.28, vis, 0.72, 0, vis)

    # Green cave entrance overlay
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
                fs, (0,0,0), th+2)
    cv2.putText(vis, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                fs, (0,255,120), th)

    cv2.imwrite(out_path, vis)
    cv2.imwrite(mask_path, refined_mask)

    # ── Debug: valid region ───────────────────────────────────────────────────
    dv = cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR)
    for ch in range(3):
        c = dv[:,:,ch].astype(np.float32)
        if ch == 2:
            c = c * weight_map + 180 * (1.0 - weight_map)
        else:
            c = c * weight_map
        dv[:,:,ch] = np.clip(c, 0, 255).astype(np.uint8)
    for c in range(w - 1):
        y1 = h - 1 - int(profile_norm[c]     * 59)
        y2 = h - 1 - int(profile_norm[c + 1] * 59)
        cv2.line(dv, (c, y1), (c+1, y2), (0,255,255), 1)
    cv2.putText(dv, "valid region (red=penalised)", (10,25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
    cv2.imwrite(debug_valid_path, dv)

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
                cx_m = int(M["m10"]/M["m00"])
                cy_m = int(M["m01"]/M["m00"])
                sc_v = all_scores[i]["total"]
                cv2.putText(dc, f"{sc_v:.2f}", (cx_m, cy_m),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)
    cv2.drawContours(dc, contours, -1, (255,255,255), 2)
    cv2.putText(dc,
                f"{len(all_candidates)} candidates (white=best, {score_val:.2f})",
                (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)
    cv2.imwrite(debug_cands_path, dc)


# ──────────────────────────────────────────────────────────────────────────────
# 11. PROCESS ONE IMAGE
# ──────────────────────────────────────────────────────────────────────────────

def process_image(input_path, output_dir):
    """Full pipeline for one image."""
    bn = os.path.splitext(os.path.basename(input_path))[0]
    out_r  = os.path.join(output_dir, f"{bn}_result.png")
    out_m  = os.path.join(output_dir, f"{bn}_mask.png")
    out_dv = os.path.join(output_dir, f"{bn}_debug_valid.png")
    out_dc = os.path.join(output_dir, f"{bn}_debug_candidates.png")

    gray_u8, gray_f32 = load_image(input_path)
    h, w = gray_u8.shape
    print(f"  [{bn}] loaded {w}x{h}")

    proc      = preprocess_image(gray_u8, gray_f32)
    wmap, lc, rc, pn, actual_lc, actual_rc = compute_valid_region(gray_f32)
    depth_map = compute_ir_depth(gray_f32)
    print(f"  [{bn}] valid cols {lc}–{rc} (actual {actual_lc}–{actual_rc}, of {w})")

    candidates = generate_candidates(proc, gray_f32, h, w, lc, rc)
    print(f"  [{bn}] {len(candidates)} unique candidates")

    if not candidates:
        print(f"  [{bn}] WARNING: no candidates")
        blank = np.zeros((h, w), np.uint8)
        cv2.imwrite(out_m, blank)
        cv2.imwrite(out_r, cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR))
        return [out_r, out_m]

    best_mask, scores, all_sc = select_best_candidate(
        candidates, gray_f32, wmap, lc, rc, depth_map=depth_map
    )
    print(f"  [{bn}] best score {scores['total']:.3f}  "
          f"area={scores['area_frac']*100:.1f}%  "
          f"add={scores['additive']:.2f}  "
          f"contrast={scores['contrast']:.2f}  "
          f"texture={scores['texture']:.2f}(×{scores['texture_mult']:.2f})  "
          f"ir_depth={scores['ir_depth']:.2f}  "
          f"depth={scores['depth']:.2f}  "
          f"area_m={scores['area_mult']:.2f}  "
          f"sol={scores['solidity']:.2f}(×{scores['sol_mult']:.2f})  "
          f"border={scores['border_touches']}(×{scores['border_mult']:.2f})  "
          f"in={scores['mean_inside']:.2f}±{scores['texture']:.2f}  "
          f"out={scores['mean_outside']:.2f}")

    # ── Solidity filter ───────────────────────────────────────────────────────
    # Non-convex candidate (e.g. entrance merged with lateral IR shadow):
    # keep only the well-illuminated weight-map portion.
    if scores.get("solidity", 1.0) < 0.65 and np.count_nonzero(best_mask) > 100:
        _is_dark_void = scores.get("mean_inside", 1.0) < 0.15
        mask_weights = wmap[best_mask > 0]
        # Dark voids use 50th pct (gentler); others use 60th pct
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
                cx_ci  = centroids_hw[ci, 0]
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

    # ── Post-selection expansion ──────────────────────────────────────────────
    # Grow selected mask into connected dark pixels at a relaxed threshold.
    # Captures the full entrance when the initial candidate covers only the core.
    best_area_frac = np.count_nonzero(best_mask) / (h * w)
    if best_area_frac < 0.25:
        relax_pct = min(50, max(30, int(scores.get("area_frac", 0.1) * 100 * 4)))
        relax_thr = int(np.percentile(proc["denoised"], relax_pct))
        _, relax_dark = cv2.threshold(proc["denoised"], relax_thr, 255,
                                       cv2.THRESH_BINARY_INV)
        br_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                          (max(9, int(min(h,w)*0.02)|1),
                                           max(9, int(min(h,w)*0.02)|1)))
        relax_dark = cv2.morphologyEx(relax_dark, cv2.MORPH_CLOSE, br_k)
        n_rd, labels_rd, _, _ = cv2.connectedComponentsWithStats(relax_dark, 8)
        overlap_labels = set(np.unique(labels_rd[best_mask > 0])) - {0}
        if overlap_labels:
            expanded = np.zeros_like(best_mask)
            for lb in overlap_labels:
                expanded[labels_rd == lb] = 255
            # Clip to valid columns to prevent re-introducing lateral dark zones
            if lc > int(w * 0.05):
                expanded[:, :lc] = 0
            if rc < int(w * 0.95):
                expanded[:, rc+1:] = 0
            n_exp, labels_exp, stats_exp, _ = cv2.connectedComponentsWithStats(
                expanded, 8)
            if n_exp > 1:
                largest_exp = 1 + np.argmax(stats_exp[1:, cv2.CC_STAT_AREA])
                expanded = ((labels_exp == largest_exp) * 255).astype(np.uint8)
            exp_area_frac = np.count_nonzero(expanded) / (h * w)
            if exp_area_frac <= 0.40 and exp_area_frac > best_area_frac * 0.8:
                exp_mean  = float(gray_f32[expanded > 0].mean())
                orig_mean = float(gray_f32[best_mask > 0].mean())
                # Reject expansion if centroid drifted far — guards against
                # bridging to a disconnected dark zone (e.g. vegetation corner).
                orig_pts = np.argwhere(best_mask > 0).astype(np.float32)
                exp_pts  = np.argwhere(expanded > 0).astype(np.float32)
                orig_cy_m, orig_cx_m = orig_pts.mean(axis=0)
                exp_cy_m,  exp_cx_m  = exp_pts.mean(axis=0)
                centroid_shift = (np.sqrt((exp_cx_m - orig_cx_m)**2
                                         + (exp_cy_m - orig_cy_m)**2)
                                  / min(h, w))
                if exp_mean < orig_mean + 0.15 and centroid_shift <= 0.20:
                    print(f"  [{bn}] expanded {best_area_frac*100:.1f}% → "
                          f"{exp_area_frac*100:.1f}%")
                    best_mask = expanded

    # ── GrabCut boundary refinement ───────────────────────────────────────────
    pre_gc = np.count_nonzero(best_mask) / (h * w)
    gc_result = grabcut_refine(gray_u8, best_mask, expand_ratio=2.0)
    post_gc = np.count_nonzero(gc_result) / (h * w)
    if post_gc > 0:
        print(f"  [{bn}] grabcut {pre_gc*100:.1f}% → {post_gc*100:.1f}%")
        best_mask = gc_result

    refined = refine_mask(best_mask, gray_f32)

    # Hard-clip final mask to valid illumination columns.
    # Removes penumbra/vignetting zones from the result even when the initial
    # candidate or GrabCut extended into them.
    if lc > int(w * 0.05):
        refined[:, :lc] = 0
    if rc < int(w * 0.95):
        refined[:, rc + 1:] = 0

    draw_result(gray_u8, refined, scores,
                out_r, out_m, out_dv,
                wmap, pn,
                out_dc, candidates, all_sc)

    final_area = np.count_nonzero(refined) / (h * w)
    print(f"  [{bn}] final area {final_area*100:.1f}%")
    outputs = [out_r, out_m, out_dv, out_dc]
    for p in outputs:
        print(f"  [{bn}] saved: {os.path.basename(p)}")
    return outputs


# ──────────────────────────────────────────────────────────────────────────────
# 12. MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) >= 2:
        # One or more explicit image paths
        for img_path in sys.argv[1:]:
            out_dir = os.path.dirname(os.path.abspath(img_path)) or "."
            print(f"Processing: {os.path.basename(img_path)}")
            process_image(img_path, out_dir)
            print()
    else:
        # Batch mode: process all jpg/png in the script's directory
        cwd = os.path.dirname(os.path.abspath(__file__))
        patterns = ["*.jpg","*.jpeg","*.png","*.JPG","*.JPEG","*.PNG"]
        found = []
        for pat in patterns:
            found.extend(glob.glob(os.path.join(cwd, pat)))
        suffixes = ("_result.png","_mask.png","_debug_valid.png",
                    "_debug_candidates.png")
        inputs = sorted(set(
            f for f in found
            if not any(os.path.basename(f).endswith(s) for s in suffixes)
        ))
        if not inputs:
            print("No input images found.")
            sys.exit(1)
        print(f"Found {len(inputs)} input image(s):")
        for p in inputs:
            print(f"  {os.path.basename(p)}")
        print()
        all_out = []
        for img in inputs:
            print(f"Processing: {os.path.basename(img)}")
            all_out += process_image(img, cwd)
            print()
        print("─" * 60)
        print(f"Done. {len(all_out)} output files:")
        for p in all_out:
            print(f"  {os.path.basename(p)}")


if __name__ == "__main__":
    main()

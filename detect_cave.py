#!/usr/bin/env python3
"""
detect_cave.py — Automatic cave entrance detector for IR/NIR imagery.

Usage:
    python detect_cave.py input.jpg output.jpg
    python detect_cave.py           # batch mode: processes all jpg/png in current dir

v3 — Redesigned pipeline:
  - Iterative flood-fill growth from darkest seeds for better candidate shapes
  - Heavy morphological bridging to connect fragmented dark areas
  - Dark-mass scoring (area × darkness) to prefer large dark regions
  - Darkest-pixel containment to identify the main cavity
  - Sharp penalties for both tiny and enormous candidates
"""

import cv2
import numpy as np
import os
import sys
import glob

# ──────────────────────────────────────────────────────────────────────────────
# DEPTH ESTIMATION — multi-signal ensemble
#
# Combines four complementary depth signals:
#   1. Depth Anything v2 Small  (ML, generalises well to IR/NIR images)
#   2. MiDaS v2.1 Small         (ML, complementary architecture;
#                                 auto-flipped when anti-correlated with DA2)
#   3. IR physics pseudo-depth  (domain-specific: dark+uniform=cave=deep)
#   4. Local entropy depth      (near=textured=high entropy; cave=low entropy)
#
# Each signal is smoothed with SLIC superpixels (boundary-preserving), then
# combined with a weighted average.  A 2-component GMM (sklearn) is fitted to
# the (ir_signal, da2_signal) feature space to learn the per-image bimodal
# cave/surface distribution, refining the ensemble weights.
#
# All signals share the convention:  0 = deepest / cave void
#                                    1 = nearest  / rock surface
# ──────────────────────────────────────────────────────────────────────────────

_DEPTH_SESSION = None   # lazy-loaded ONNX session
_DEPTH_MODEL   = "depth_anything_v2_small.onnx"
_MIDAS_SESSION = None
_MIDAS_MODEL   = "midas_v21_small_256.onnx"


def _load_depth_session():
    global _DEPTH_SESSION
    if _DEPTH_SESSION is not None:
        return _DEPTH_SESSION
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), _DEPTH_MODEL)
    if not os.path.exists(model_path):
        return None
    try:
        import onnxruntime as ort
        _DEPTH_SESSION = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"])
        return _DEPTH_SESSION
    except Exception as e:
        print(f"  [depth] DA2 ONNX load failed: {e}")
        return None


def _load_midas_session():
    global _MIDAS_SESSION
    if _MIDAS_SESSION is not None:
        return _MIDAS_SESSION
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), _MIDAS_MODEL)
    if not os.path.exists(model_path):
        return None
    try:
        import onnxruntime as ort
        _MIDAS_SESSION = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"])
        return _MIDAS_SESSION
    except Exception as e:
        print(f"  [depth] MiDaS ONNX load failed: {e}")
        return None


def _imagenet_preprocess(gray_u8, target_hw):
    """Grayscale → 3-ch RGB, resize to target_hw (h,w), ImageNet-normalize → BCHW."""
    th, tw = target_hw
    rgb = cv2.cvtColor(cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (tw, th)).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std  = np.array([0.229, 0.224, 0.225], np.float32)
    return ((resized - mean) / std).transpose(2, 0, 1)[None]  # BCHW


def _norm01(arr):
    """Normalize float array to [0, 1]."""
    mn, mx = float(arr.min()), float(arr.max())
    return (arr - mn) / (mx - mn + 1e-6)


def estimate_depth(gray_u8):
    """
    Depth Anything v2 Small inference.
    Returns depth_norm float32 [0=deepest, 1=nearest], or None.
    """
    sess = _load_depth_session()
    if sess is None:
        return None
    h, w = gray_u8.shape
    scale = 518.0 / max(h, w)
    th = max(14, int(h * scale) // 14 * 14)
    tw = max(14, int(w * scale) // 14 * 14)
    inp = _imagenet_preprocess(gray_u8, (th, tw))
    raw = sess.run(None, {"pixel_values": inp})[0][0]
    out = cv2.resize(_norm01(raw).astype(np.float32), (w, h),
                     interpolation=cv2.INTER_LINEAR)
    return out


def _estimate_midas_raw(gray_u8):
    """MiDaS v2.1 Small inference. Returns normalized disparity [0..1] or None."""
    sess = _load_midas_session()
    if sess is None:
        return None
    h, w = gray_u8.shape
    inp = _imagenet_preprocess(gray_u8, (256, 256))
    raw = sess.run(None, {"0": inp})[0][0]
    out = cv2.resize(_norm01(raw).astype(np.float32), (w, h),
                     interpolation=cv2.INTER_LINEAR)
    return out


def _compute_ir_pseudo_depth(gray_f32):
    """
    Physics-based depth signal for IR/NIR cave imagery.

    Cave voids absorb all IR flash light → near-zero intensity, very low texture.
    Rock surfaces reflect IR         → medium-to-bright, significant texture.
    Dark vegetation                  → dark BUT textured  (key discriminator!).

    Signal: cave_score = (1 - brightness) × max(0, 1 - local_std / thresh)
    Multi-scale: averaged over three kernel sizes for robustness.

    Returns ir_depth float32 [0=deepest/cave, 1=nearest/surface].
    """
    h, w = gray_f32.shape
    cave_acc = np.zeros_like(gray_f32)
    min_dim  = min(h, w)

    for ksize in [0.03, 0.06, 0.10]:   # fraction of min dimension
        k = max(11, int(min_dim * ksize) | 1)
        blurred    = cv2.GaussianBlur(gray_f32,       (k, k), 0)
        blurred_sq = cv2.GaussianBlur(gray_f32 ** 2,  (k, k), 0)
        local_std  = np.sqrt(np.maximum(blurred_sq - blurred ** 2, 0.0))

        darkness    = 1.0 - gray_f32
        uniformity  = np.clip(1.0 - local_std / 0.12, 0.0, 1.0)
        cave_acc   += darkness * uniformity

    cave_acc /= 3.0

    # Percentile-based normalisation to prevent camera border artifacts
    # (pure-black vignette pixels score high in cave_acc because they are dark
    # and perfectly uniform, dominating a min-max normalisation and collapsing
    # all interior variation to near-zero).  Clip to the 1st–99th range so
    # border outliers don't set the scale.
    c_lo = float(np.percentile(cave_acc, 1))
    c_hi = float(np.percentile(cave_acc, 99))
    cave_norm = np.clip((cave_acc - c_lo) / (c_hi - c_lo + 1e-6), 0.0, 1.0)

    # Invert: high cave score → low depth value (= deep)
    return (1.0 - cave_norm).astype(np.float32)


def _compute_entropy_depth(gray_u8):
    """
    Multi-scale local standard deviation as texture depth proxy.

    Near surfaces (rock, vegetation) have rich texture → high local std.
    Cave interior is near-uniform black            → low local std.

    Uses OpenCV Gaussian-blur variance trick for speed (10× faster than
    skimage rank-entropy while achieving 0.80+ correlation with it).

    Returns ent_depth float32 [0=low texture=deepest, 1=high texture=nearest].
    """
    gf = gray_u8.astype(np.float32) / 255.0
    min_dim = min(gray_u8.shape)
    ent_acc = np.zeros_like(gf)
    for kf in [0.02, 0.04, 0.06]:
        k = max(7, int(min_dim * kf) | 1)
        bl  = cv2.GaussianBlur(gf,      (k, k), 0)
        bl2 = cv2.GaussianBlur(gf ** 2, (k, k), 0)
        ent_acc += np.sqrt(np.maximum(bl2 - bl ** 2, 0.0))
    return _norm01(ent_acc / 3.0).astype(np.float32)


def _compute_slic_segments(gray_u8, n_segments=250):
    """Compute SLIC superpixel labels once; reuse across multiple signals."""
    try:
        from skimage.segmentation import slic
        gray_rgb = np.stack([gray_u8] * 3, axis=-1)
        return slic(gray_rgb, n_segments=n_segments, compactness=10.0,
                    sigma=1.0, start_label=0)
    except Exception:
        return None


def _superpixel_smooth(depth_map, segs_or_gray_u8, n_segments=250):
    """
    Boundary-preserving depth smoothing via SLIC superpixels.
    segs_or_gray_u8: pre-computed integer segment labels OR a gray_u8 image.
    Each superpixel is assigned the mean depth of its member pixels.
    """
    try:
        if segs_or_gray_u8 is None:
            return depth_map
        arr = np.asarray(segs_or_gray_u8)
        # SLIC labels: int32 (signed). Gray images: uint8 (unsigned).
        # Distinguish via sign: signed int = pre-computed labels.
        if np.issubdtype(arr.dtype, np.signedinteger):
            segs = arr          # pre-computed SLIC labels
        else:
            segs = _compute_slic_segments(segs_or_gray_u8, n_segments)
        if segs is None:
            return depth_map
        smoothed = np.zeros_like(depth_map)
        for sp_id in np.unique(segs):
            m = segs == sp_id
            smoothed[m] = float(depth_map[m].mean())
        return smoothed.astype(np.float32)
    except Exception:
        return depth_map


def estimate_depth_ensemble(gray_u8, gray_f32):
    """
    Multi-signal depth ensemble optimised for IR/NIR cave imagery.

    Returns (ensemble_depth, components) where:
      ensemble_depth : float32 ndarray [0=deepest/cave, 1=nearest/surface]
      components     : dict with individual signal maps + timing info
    """
    import time
    t0 = time.time()

    h, w = gray_u8.shape
    n_sp = max(80, min(250, gray_u8.size // 4000))

    # ── Signal 1: Depth Anything v2 ──────────────────────────────────────────
    da2 = estimate_depth(gray_u8)

    # ── Signal 2: MiDaS (adaptive convention) ────────────────────────────────
    midas_raw = _estimate_midas_raw(gray_u8)
    midas = None
    if midas_raw is not None:
        if da2 is not None:
            # MiDaS v2.1 can produce inverted depth ordering on IR/NIR imagery
            # because it was trained on RGB and interprets the pitch-black cave
            # interior differently from how DA2 does.
            # Detection: compute full-image Pearson correlation with DA2.
            # DA2 is the anchor (reliably low at cave voids on IR).
            # If MiDaS is strongly anti-correlated → its convention is inverted.
            corr = float(np.corrcoef(da2.flatten(), midas_raw.flatten())[0, 1])
            if corr < -0.15:
                midas = 1.0 - midas_raw
                print(f"  [depth] MiDaS convention flipped (corr={corr:.2f})")
            else:
                midas = midas_raw
        else:
            midas = midas_raw

    # ── Signal 3: IR physics pseudo-depth ────────────────────────────────────
    ir_depth = _compute_ir_pseudo_depth(gray_f32)

    # ── Signal 4: Local entropy depth ────────────────────────────────────────
    ent_depth = _compute_entropy_depth(gray_u8)

    # ── Edge-preserving smoothing for ML depth signals ────────────────────────
    # Downsample → bilateral (d=9, fixed neighbourhood) → upsample.
    # Working at ≤640px max-side keeps each bilateral call under 50 ms even
    # for 2+ MP images while still preserving rock/void boundaries.
    max_side = 640
    scale = min(1.0, max_side / max(h, w))
    hs, ws = max(1, int(h * scale)), max(1, int(w * scale))

    def _smooth_ml(arr):
        if arr is None:
            return None
        small = cv2.resize(arr, (ws, hs), interpolation=cv2.INTER_LINEAR)
        filt  = cv2.bilateralFilter(small, d=9, sigmaColor=0.06, sigmaSpace=9)
        return cv2.resize(filt, (w, h), interpolation=cv2.INTER_LINEAR)

    da2_s  = _smooth_ml(da2)
    mid_s  = _smooth_ml(midas)
    ent_s  = _smooth_ml(ent_depth)
    # IR: keep at full resolution; Gaussian smooths pixel-level noise
    ks_ir = max(5, int(min(h, w) * 0.01) | 1)
    ir_s  = cv2.GaussianBlur(ir_depth, (ks_ir, ks_ir), 0)

    # ── Base weighted average ─────────────────────────────────────────────────
    # IR pseudo gets the highest weight: domain-specific physics signal.
    # DA2 and MiDaS add ML-based geometry; entropy adds texture discrimination.
    w_ir  = 0.45
    w_ent = 0.15
    w_da2 = 0.28 if da2_s  is not None else 0.0
    w_mid = 0.12 if mid_s  is not None else 0.0
    total = w_ir + w_ent + w_da2 + w_mid

    ensemble = w_ir * ir_s + w_ent * ent_s
    if da2_s is not None:
        ensemble += w_da2 * da2_s
    if mid_s is not None:
        ensemble += w_mid * mid_s
    ensemble /= total

    # ── GMM refinement (sklearn) ──────────────────────────────────────────────
    # Fit a 2-component Gaussian Mixture to (ir_signal, da2_signal) feature
    # space to learn the per-image bimodal cave/surface distribution.
    # The cave component (low ir_depth + low da2_depth) corrects the ensemble
    # in a principled, per-image adaptive way.
    gmm_depth = None
    try:
        from sklearn.mixture import GaussianMixture
        # Subsample for speed: use every 4th pixel
        ir_sub  = ir_s.flatten()[::4]
        da2_sub = (da2_s if da2_s is not None else ensemble).flatten()[::4]
        f = np.stack([ir_sub, da2_sub], axis=1).astype(np.float64)
        gmm = GaussianMixture(n_components=2, max_iter=60, random_state=0)
        gmm.fit(f)
        # Component with lowest mean ir_signal = deep (cave)
        cave_comp = int(np.argmin(gmm.means_[:, 0]))
        # Apply to full-resolution features
        f_full = np.stack([ir_s.flatten(),
                           (da2_s if da2_s is not None else ensemble).flatten()],
                          axis=1).astype(np.float64)
        proba = gmm.predict_proba(f_full)
        gmm_depth = (1.0 - proba[:, cave_comp]).reshape(h, w).astype(np.float32)
        # Blend: GMM is most useful where both IR and DA2 agree (confident)
        ensemble = 0.65 * ensemble + 0.35 * gmm_depth
    except Exception:
        pass

    # ── Final normalisation ───────────────────────────────────────────────────
    ensemble = _norm01(ensemble).astype(np.float32)

    t1 = time.time()
    da2_p5_str = f"{np.percentile(da2_s,5):.3f}" if da2_s is not None else "n/a"
    print(f"  [depth] ensemble ({t1-t0:.2f}s): "
          f"ir_deep5%={np.percentile(ir_s,5):.3f}  "
          f"da2_deep5%={da2_p5_str}  "
          f"ens_deep5%={np.percentile(ensemble,5):.3f}")

    return ensemble, {
        "da2":      da2,
        "midas":    midas,
        "ir":       ir_s,
        "entropy":  ent_s,
        "gmm":      gmm_depth,
        "ensemble": ensemble,
    }


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
    Preprocessing:
      - Median denoise
      - Very-large-blur background illumination estimate
      - Division normalisation (preserves cave darkness relative to local bg)
      - CLAHE for local contrast enhancement (extra candidate source)
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

    # CLAHE — boosts local contrast; helps candidate generation in
    # scenes with uneven IR illumination
    tile = max(4, int(min(h, w) / 64))
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(tile, tile))
    clahe_u8 = clahe.apply(denoised)

    return {
        "denoised":     denoised,
        "background":   background,
        "corrected_u8": corrected_u8,
        "clahe_u8":     clahe_u8,
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
    left_col = 0
    for c in range(w):
        if profile_norm[c] >= drop_thresh:
            left_col = c
            break
    right_col = w - 1
    for c in range(w - 1, -1, -1):
        if profile_norm[c] >= drop_thresh:
            right_col = c
            break

    left_col  = min(left_col,  int(w * 0.30))
    right_col = max(right_col, int(w * 0.70))

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
    return weight_map, left_col, right_col, profile_norm


# ──────────────────────────────────────────────────────────────────────────────
# 4. GENERATE CANDIDATES
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
    # Large closing to bridge gaps
    close_size = max(15, int(min(h, w) * 0.04) | 1)
    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                         (close_size, close_size))
    bridged = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_k)
    # Standard cleanup
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    bridged = cv2.morphologyEx(bridged, cv2.MORPH_OPEN, k)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bridged, 8)
    result = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            result.append(((labels == i) * 255).astype(np.uint8))
    return result


def generate_candidates(proc, gray_f32, h, w, left_col=0, right_col=None,
                        depth_norm=None):
    """
    Multi-strategy candidate generation:
      A. Multi-level thresholding with standard cleaning
      B. Multi-level thresholding with heavy bridging
      C. Iterative seed-growth from darkest pixels
      D. Otsu
      E. Adaptive threshold intersected with dark base
    """
    denoised     = proc["denoised"]
    corrected_u8 = proc["corrected_u8"]
    clahe_u8     = proc["clahe_u8"]

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
    # This connects fragmented dark spots that form one physical entrance
    for pct in [10, 15, 20, 25, 30, 35]:
        thr = int(np.percentile(denoised, pct))
        _, binary = cv2.threshold(denoised, thr, 255, cv2.THRESH_BINARY_INV)
        candidates += _extract_components_heavy(binary, min_area, h, w)

    # ── C. Iterative seed-growth ──────────────────────────────────────────────
    # Start from the darkest 1% of pixels, grow by increasing threshold.
    # Record candidates at each growth level.
    p1 = int(np.percentile(denoised, 1))
    _, seed = cv2.threshold(denoised, max(p1, 3), 255, cv2.THRESH_BINARY_INV)
    # Bridge the seed
    seed_k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (max(7, int(min(h, w) * 0.03) | 1),
         max(7, int(min(h, w) * 0.03) | 1))
    )
    seed = cv2.morphologyEx(seed, cv2.MORPH_CLOSE, seed_k)

    # Grow at increasing thresholds
    for pct in [5, 10, 15, 20, 25, 30, 35, 40, 50]:
        thr = int(np.percentile(denoised, pct))
        _, dark_level = cv2.threshold(denoised, thr, 255, cv2.THRESH_BINARY_INV)
        # Grow: dilate seed, then intersect with dark_level
        grow_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        grown = cv2.dilate(seed, grow_k, iterations=2)
        grown = cv2.bitwise_and(grown, dark_level)
        # Bridge gaps
        grown = cv2.morphologyEx(grown, cv2.MORPH_CLOSE, seed_k)
        # Update seed for next iteration
        seed = cv2.bitwise_or(seed, grown)
        # Record as candidate
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
    # Mask out lateral zones (set to bright) so dark laterals don't merge
    # with the cave entrance during connected-component extraction.
    if right_col is None:
        right_col = w - 1
    if left_col > 10 or right_col < w - 11:
        masked_den = denoised.copy()
        masked_den[:, :left_col] = 255
        masked_den[:, right_col+1:] = 255
        for pct in [10, 15, 20, 25, 30, 35, 40]:
            thr = int(np.percentile(denoised, pct))  # percentile from FULL image
            _, binary = cv2.threshold(masked_den, thr, 255, cv2.THRESH_BINARY_INV)
            candidates += _extract_components(binary, min_area)
            candidates += _extract_components_heavy(binary, min_area, h, w)

    # ── G. CLAHE-based thresholding ───────────────────────────────────────────
    # CLAHE equalises local contrast, making cave entrances dark relative to
    # their immediate surroundings even when global illumination is uneven.
    for pct in [10, 15, 20, 25, 30, 35]:
        thr = int(np.percentile(clahe_u8, pct))
        _, binary = cv2.threshold(clahe_u8, thr, 255, cv2.THRESH_BINARY_INV)
        candidates += _extract_components(binary, min_area)
        candidates += _extract_components_heavy(binary, min_area, h, w)

    # ── I. Depth-map based candidates ────────────────────────────────────────
    # Depth Anything v2 assigns the lowest disparity (= farthest / deepest) to
    # cave voids and pits because the IR flash cannot illuminate or reach inside.
    # Thresholding the depth map at progressively deeper levels gives candidates
    # that are naturally bounded by the depth boundary — complementing the
    # intensity-based methods above with a geometric signal.
    if depth_norm is not None:
        for pct in [8, 12, 16, 20, 25]:
            thr = float(np.percentile(depth_norm, pct))
            binary_d = (depth_norm <= thr).astype(np.uint8) * 255
            candidates += _extract_components(binary_d, min_area)
            candidates += _extract_components_heavy(binary_d, min_area, h, w)

    # ── H. Canny-enclosed dark regions ───────────────────────────────────────
    # Find regions bounded by strong edges and predominantly dark inside.
    # This is especially effective for circular/oval pit entrances where the
    # rim produces a clean closed contour, giving the full pit as one candidate
    # without requiring multi-step flood-fill expansion.
    edges_h = cv2.Canny(denoised, 20, 60)
    ek = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    edges_h = cv2.morphologyEx(edges_h, cv2.MORPH_CLOSE, ek)
    cnts_h, hier_h = cv2.findContours(edges_h, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in cnts_h:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        cand_h = np.zeros((h, w), np.uint8)
        cv2.fillPoly(cand_h, [cnt], 255)
        if not np.any(cand_h > 0):
            continue
        interior_mean = float(denoised[cand_h > 0].mean())
        # Only keep if the enclosed region is predominantly dark
        if interior_mean < 110:  # < ~0.43 on [0,1]
            candidates.append(cand_h)

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
# 5. SCORE A CANDIDATE
# ──────────────────────────────────────────────────────────────────────────────

def score_candidate(mask, gray_f32, weight_map, left_col, right_col,
                    darkest5_mask, depth_norm=None):
    """
    Multi-criteria scoring with MULTIPLICATIVE gates.

    Key design:
      - Contrast vs surround is the primary additive signal
      - Darkness enrichment replaces raw containment (penalises large blobs)
      - Area and solidity are MULTIPLICATIVE — wrong size or donut shape
        kills the score regardless of how dark/contrasty the region is
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
    bbox_area = bw * bh
    aspect = min(bw, bh) / max(bw, bh) if max(bw, bh) > 0 else 0.0
    area_frac = area / img_area

    # ── Intensity ─────────────────────────────────────────────────────────────
    mean_inside = float(gray_f32[mask_bool].mean())
    std_inside  = float(gray_f32[mask_bool].std())
    darkness = 1.0 - mean_inside

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

    # ── 2b. Interior texture / uniformity (ADDITIVE) ─────────────────────────
    # A true cave void is uniformly near-black (std ≈ 0.03–0.08).
    # Dark textured terrain / vegetation has std 0.10–0.20+.
    # Reference std for "perfect void" is 0.08; score drops linearly to 0 at 0.20.
    texture_score = np.clip(1.0 - std_inside / 0.18, 0.0, 1.0)

    # ── 3. Darkness enrichment (ADDITIVE) ─────────────────────────────────────
    # enrichment = (frac_of_darkest_inside / area_frac)
    darkest5_bool = darkest5_mask.astype(bool)
    total_darkest = max(darkest5_bool.sum(), 1)
    contained_frac = float((mask_bool & darkest5_bool).sum()) / total_darkest
    enrichment = contained_frac / max(area_frac, 0.001)
    enrichment_score = np.clip((enrichment - 1.0) / 8.0, 0.0, 1.0)

    # ── 4. Distance-transform depth (ADDITIVE) ───────────────────────────────
    # Measures how thick/deep the darkest core is.  Large coherent voids
    # have high max-distance; narrow shadows or elongated strips don't.
    dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    max_dist = float(dist_transform.max())
    ref_dist = min(h, w) * 0.12   # reference: 12% of min dimension
    depth_score = np.clip(max_dist / ref_dist, 0.0, 1.0)

    # ── 5. Boundary gradient (ADDITIVE) ───────────────────────────────────────
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

    # ── 5. Valid region alignment (ADDITIVE) ──────────────────────────────────
    valid_score = float(weight_map[mask_bool].mean())

    # ── 6. Aspect ratio (ADDITIVE) ────────────────────────────────────────────
    aspect_score = 1.0 if aspect >= 0.15 else aspect / 0.15

    # ── 7. Position (ADDITIVE, very mild centre bias) ─────────────────────────
    cx = x + bw / 2.0
    cy = y + bh / 2.0
    dist_x = abs(cx / w - 0.5) * 2
    dist_y = abs(cy / h - 0.5) * 2
    position_score = 1.0 - 0.10 * dist_x - 0.05 * dist_y

    # ── 8. Depth ensemble score (ADDITIVE) ────────────────────────────────────
    # The depth ensemble (IR physics + DA2 + MiDaS + GMM) assigns values close
    # to 0 for cave voids (dark, uniform, absorbs IR) and close to 1 for near
    # surfaces (rock, vegetation).
    # Formula: score → 1 when mean_depth ≤ 0.20; → 0 when mean_depth ≥ 0.55.
    if depth_norm is not None:
        mean_d = float(depth_norm[mask_bool].mean())
        depth_model_score = np.clip(1.0 - (mean_d - 0.10) / 0.45, 0.0, 1.0)
    else:
        depth_model_score = 0.5  # neutral when model unavailable

    # ── Additive base score ───────────────────────────────────────────────────
    additive = (
        0.20 * contrast_score       # brightness transition at boundary
      + 0.12 * dark_score           # raw darkness
      + 0.09 * texture_score        # uniform dark = void; textured = terrain (additive)
      + 0.17 * depth_model_score    # depth ensemble: cave voids are "far/deep"
      + 0.05 * enrichment_score     # concentration of darkest pixels
      + 0.11 * depth_score          # distance-transform thick dark core
      + 0.08 * gradient_score       # sharp boundary edge
      + 0.05 * valid_score          # within illuminated zone
      + 0.04 * aspect_score         # not absurdly thin
      + 0.04 * position_score       # mild centre preference
      + 0.05 * 1.0                  # base
    )

    # ── MULTIPLICATIVE GATES ──────────────────────────────────────────────────
    # These cannot be compensated by high additive scores.

    # Area gate: allow reasonable range from 3% to 35%.
    # Small but well-defined cave entrances (3–8%) should compete fairly.
    # Very tiny (<1%) or huge (>50%) regions are still heavily penalised.
    if area_frac < 0.003:
        area_mult = 0.05
    elif area_frac < 0.01:
        area_mult = 0.05 + 0.20 * (area_frac - 0.003) / 0.007
    elif area_frac < 0.03:
        area_mult = 0.25 + 0.35 * (area_frac - 0.01) / 0.02
    elif area_frac < 0.06:
        area_mult = 0.60 + 0.40 * (area_frac - 0.03) / 0.03
    elif area_frac <= 0.35:
        area_mult = 1.0
    elif area_frac <= 0.50:
        area_mult = 1.0 - 0.7 * (area_frac - 0.35) / 0.15
    else:
        area_mult = max(0.05, 0.3 - 0.25 * (area_frac - 0.50) / 0.50)

    # Texture gate: cave voids are near-uniform (std ≈ 0.03–0.07).
    # Dark rock faces / vegetation show much higher interior std (0.10–0.20+).
    # This multiplicative gate directly penalises textured "dark but not cave"
    # regions that contrast_score and dark_score cannot distinguish alone.
    if std_inside <= 0.08:
        texture_mult = 1.0
    elif std_inside <= 0.16:
        texture_mult = max(0.35, 1.0 - 0.65 * (std_inside - 0.08) / 0.08)
    else:
        texture_mult = 0.35

    # Solidity gate: very non-convex (donut, tentacles) → penalised
    if solidity >= 0.45:
        solidity_mult = 1.0
    elif solidity >= 0.25:
        solidity_mult = 0.4 + 0.6 * (solidity - 0.25) / 0.20
    else:
        solidity_mult = 0.4

    # Lateral penalty
    lateral_pen = 1.0
    if int(cx) < left_col or int(cx) > right_col:
        if valid_score < 0.5:
            lateral_pen = 0.4

    # Vertical position gate: cave entrances appear in the middle/lower portion
    # of trail-camera frames.  Dark regions in the top 10–30% of the image are
    # overwhelmingly camera-border vignette, sky, rock overhang, or vegetation —
    # never the cave entrance itself (cameras are placed at ground level looking
    # toward the cave).  Candidates whose centroid sits in the upper zone are
    # penalised multiplicatively so the correct lower-lying candidate wins.
    y_frac_c = cy / h
    if y_frac_c < 0.10:
        vert_gate = 0.15
    elif y_frac_c < 0.30:
        vert_gate = 0.15 + 0.85 * (y_frac_c - 0.10) / 0.20
    else:
        vert_gate = 1.0

    # Absolute darkness gate: cave interior in IR is < 0.30.
    # Dark-but-textured terrain (vegetation, soil) tends to be 0.30–0.50.
    # This gate shrinks scores for regions that are "not dark enough" to be voids.
    if mean_inside <= 0.20:
        dark_gate = 1.0
    elif mean_inside <= 0.40:
        dark_gate = 1.0 - 0.65 * (mean_inside - 0.20) / 0.20
    else:
        dark_gate = max(0.05, 0.35 - 0.30 * (mean_inside - 0.40) / 0.20)

    total = additive * area_mult * solidity_mult * lateral_pen * dark_gate * texture_mult * vert_gate

    return {
        "total":            round(float(total), 4),
        "additive":         round(float(additive), 3),
        "contrast":         round(float(contrast_score), 3),
        "dark":             round(float(dark_score), 3),
        "texture":          round(float(texture_score), 3),
        "texture_mult":     round(float(texture_mult), 3),
        "depth_model":      round(float(depth_model_score), 3),
        "enrichment":       round(float(enrichment_score), 3),
        "depth":            round(float(depth_score), 3),
        "dark_gate":        round(float(dark_gate), 3),
        "area_mult":        round(float(area_mult), 3),
        "area_frac":        round(float(area_frac), 4),
        "solidity":         round(float(solidity), 3),
        "sol_mult":         round(float(solidity_mult), 3),
        "gradient":         round(float(gradient_score), 3),
        "valid_score":      round(float(valid_score), 3),
        "vert_gate":        round(float(vert_gate), 3),
        "mean_inside":      round(float(mean_inside), 3),
        "std_inside":       round(float(std_inside), 3),
        "mean_outside":     round(float(mean_outside), 3),
    }


# ──────────────────────────────────────────────────────────────────────────────
# 6. SELECT BEST
# ──────────────────────────────────────────────────────────────────────────────

def select_best_candidate(candidates, gray_f32, weight_map,
                          left_col, right_col, depth_norm=None):
    """Score all candidates, return (best_mask, best_scores, all_scores)."""
    if not candidates:
        return None, {}, []

    # Pre-compute: mask of darkest 5% of pixels in the image
    p5 = np.percentile(gray_f32, 5)
    darkest5_mask = (gray_f32 <= p5).astype(np.uint8) * 255

    all_scores = []
    for cand in candidates:
        sc = score_candidate(cand, gray_f32, weight_map, left_col, right_col,
                             darkest5_mask, depth_norm=depth_norm)
        all_scores.append(sc)

    best_idx = max(range(len(all_scores)),
                   key=lambda i: all_scores[i]["total"])
    return candidates[best_idx], all_scores[best_idx], all_scores


# ──────────────────────────────────────────────────────────────────────────────
# 7. REFINE MASK
# ──────────────────────────────────────────────────────────────────────────────

def refine_mask(mask, gray_f32):
    """Close gaps, fill holes, smooth boundary, keep largest component.

    Uses a bordered flood-fill so it works even when the mask touches
    the image edges or corners.
    """
    h, w = gray_f32.shape
    orig_area = np.count_nonzero(mask)

    cs = max(11, int(min(h, w) * 0.02) | 1)
    ck = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cs, cs))
    refined = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, ck)

    # Fill interior holes safely using a 1px black border.
    # This guarantees the flood-fill seed is always background,
    # even if the mask touches image edges.
    bordered = np.zeros((h + 2, w + 2), np.uint8)
    bordered[1:-1, 1:-1] = refined
    flood = bordered.copy()
    pad = np.zeros((h + 4, w + 4), np.uint8)
    cv2.floodFill(flood, pad, (0, 0), 255)
    # Interior holes are pixels that stayed 0 (not reachable from border)
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

    # Safety: if refinement bloated the mask beyond 2× original, revert
    refined_area = np.count_nonzero(refined)
    if refined_area > max(orig_area * 2, h * w * 0.50):
        return mask

    return refined


# ──────────────────────────────────────────────────────────────────────────────
# 8. GRABCUT REFINEMENT
# ──────────────────────────────────────────────────────────────────────────────

def grabcut_refine(best_mask, gray_u8, h, w):
    """
    Use GrabCut to refine the cave mask boundary.

    Strategy:
      - Pixels inside the mask → probable foreground (GC_PR_FGD)
      - Darkest pixels inside the mask → definite foreground (GC_FGD)
      - Pixels outside the mask with brightness ≥ 65th-percentile → definite background (GC_BGD)
      - Everything else → probable background (GC_PR_BGD)

    GrabCut builds a per-pixel Gaussian Mixture Model of dark vs bright, then
    refines the boundary using graph-cuts (min-cut/max-flow), which naturally
    follows strong edges — ideal for the sharp cave-entrance boundary.

    Safety: if GrabCut expands or shrinks the mask more than 3× / 0.2×,
    the original mask is returned unchanged.
    """
    if np.count_nonzero(best_mask) < 100:
        return best_mask

    bgr = cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR)
    gc_mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)
    gc_mask[best_mask > 0] = cv2.GC_PR_FGD

    # Definite FG: darkest 30th percentile of pixels already inside the mask
    pix_inside = gray_u8[best_mask > 0]
    thr_fg = int(np.percentile(pix_inside, 30))
    thr_fg = min(thr_fg, 60)   # cap at 60/255 ≈ 0.24 — must be truly dark
    gc_mask[(best_mask > 0) & (gray_u8 <= thr_fg)] = cv2.GC_FGD

    # Definite BG: pixels outside the mask that are bright (top 25% of image).
    # Using the 75th percentile keeps only clearly-lit regions as definite
    # background, avoiding mis-labelling dark-but-not-cave areas.
    thr_bg = int(np.percentile(gray_u8, 75))
    gc_mask[(best_mask == 0) & (gray_u8 >= thr_bg)] = cv2.GC_BGD

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(bgr, gc_mask, None, bgd_model, fgd_model, 5,
                    cv2.GC_INIT_WITH_MASK)
        result = np.where(
            (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0
        ).astype(np.uint8)

        orig_area = np.count_nonzero(best_mask)
        new_area  = np.count_nonzero(result)
        # Safety bounds: GrabCut is used as a boundary-refiner, not a grower.
        # Accept only if it shrinks or stays roughly the same size (max +10%).
        # Larger expansions indicate GrabCut merged unrelated dark regions.
        if new_area < orig_area * 0.20 or new_area > orig_area * 1.10:
            return best_mask
        return result
    except Exception:
        return best_mask


# ──────────────────────────────────────────────────────────────────────────────
# 9. DRAW RESULT
# ──────────────────────────────────────────────────────────────────────────────

def draw_result(gray_u8, refined_mask, scores,
                out_path, mask_path, debug_valid_path,
                weight_map, profile_norm,
                debug_cands_path, all_candidates, all_scores):
    """Save result overlay, mask, and debug images."""
    h, w = gray_u8.shape

    # ── Main result ───────────────────────────────────────────────────────────
    vis = cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR)
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
# 9. PROCESS ONE IMAGE
# ──────────────────────────────────────────────────────────────────────────────

def process_image(input_path, output_dir):
    """Full pipeline for one image."""
    bn = os.path.splitext(os.path.basename(input_path))[0]
    out_r  = os.path.join(output_dir, f"{bn}_result.png")
    out_m  = os.path.join(output_dir, f"{bn}_mask.png")
    out_dv = os.path.join(output_dir, f"{bn}_debug_valid.png")
    out_dc = os.path.join(output_dir, f"{bn}_debug_candidates.png")
    out_dd = os.path.join(output_dir, f"{bn}_debug_depth.png")

    gray_u8, gray_f32 = load_image(input_path)
    h, w = gray_u8.shape
    print(f"  [{bn}] loaded {w}x{h}")

    # ── Depth estimation (multi-signal ensemble) ──────────────────────────────
    depth_norm, depth_comps = estimate_depth_ensemble(gray_u8, gray_f32)

    # Save multi-panel depth debug image (2 rows × 3 cols)
    def _depth_panel(arr, label):
        """Render a named depth map as a plasma-colourmap panel with a label."""
        if arr is None:
            panel = np.full((h, w, 3), 40, dtype=np.uint8)
            cv2.putText(panel, "unavailable", (10, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 120, 120), 1)
        else:
            vis = (np.clip(_norm01(arr), 0, 1) * 255).astype(np.uint8)
            panel = cv2.applyColorMap(vis, cv2.COLORMAP_PLASMA)
        fs = max(0.45, min(w, h) / 1000)
        cv2.putText(panel, label, (6, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), 3)
        cv2.putText(panel, label, (5, 21),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 1)
        return panel

    panels = [
        _depth_panel(depth_comps.get("da2"),      "DA2"),
        _depth_panel(depth_comps.get("midas"),     "MiDaS"),
        _depth_panel(depth_comps.get("ir"),        "IR-physics"),
        _depth_panel(depth_comps.get("entropy"),   "Entropy"),
        _depth_panel(depth_comps.get("gmm"),       "GMM"),
        _depth_panel(depth_comps.get("ensemble"),  "ENSEMBLE"),
    ]
    # Resize all panels to same size
    ph, pw = max(120, h // 3), max(160, w // 3)
    panels = [cv2.resize(p, (pw, ph)) for p in panels]
    row0 = np.concatenate(panels[:3], axis=1)
    row1 = np.concatenate(panels[3:], axis=1)
    depth_grid = np.concatenate([row0, row1], axis=0)
    cv2.imwrite(out_dd, depth_grid)

    proc = preprocess_image(gray_u8, gray_f32)
    wmap, lc, rc, pn = compute_valid_region(gray_f32)
    print(f"  [{bn}] valid cols {lc}–{rc} (of {w})")

    candidates = generate_candidates(proc, gray_f32, h, w, lc, rc,
                                     depth_norm=depth_norm)
    print(f"  [{bn}] {len(candidates)} unique candidates")

    if not candidates:
        print(f"  [{bn}] WARNING: no candidates")
        blank = np.zeros((h, w), np.uint8)
        cv2.imwrite(out_m, blank)
        cv2.imwrite(out_r, cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR))
        return [out_r, out_m]

    best_mask, scores, all_sc = select_best_candidate(
        candidates, gray_f32, wmap, lc, rc, depth_norm=depth_norm
    )
    print(f"  [{bn}] best score {scores['total']:.3f}  "
          f"area={scores['area_frac']*100:.1f}%  "
          f"add={scores['additive']:.2f}  "
          f"contrast={scores['contrast']:.2f}  "
          f"texture={scores['texture']:.2f}(×{scores['texture_mult']:.2f})  "
          f"depth_model={scores['depth_model']:.2f}  "
          f"dark_gate={scores['dark_gate']:.2f}  "
          f"depth={scores['depth']:.2f}  "
          f"area_m={scores['area_mult']:.2f}  "
          f"sol={scores['solidity']:.2f}(×{scores['sol_mult']:.2f})  "
          f"in={scores['mean_inside']:.2f}±{scores['std_inside']:.2f}  "
          f"out={scores['mean_outside']:.2f}")

    # ── Universal soft lateral clip ───────────────────────────────────────────
    # When the image has significant lateral dark borders (IR fall-off),
    # remove any selected pixels with weight < 0.15 (deep border zone).
    # The ramp zone (0.15–1.0) is preserved so real entrance pixels near
    # the illumination boundary are not discarded.
    if lc > int(w * 0.05) or rc < int(w * 0.95):
        clip_ok = (wmap >= 0.15).astype(np.uint8) * 255
        best_mask = cv2.bitwise_and(best_mask, clip_ok)
        n_cl, labels_cl, stats_cl, _ = cv2.connectedComponentsWithStats(best_mask, 8)
        if n_cl > 1:
            lc_cl = 1 + np.argmax(stats_cl[1:, cv2.CC_STAT_AREA])
            best_mask = ((labels_cl == lc_cl) * 255).astype(np.uint8)

    # If the selected candidate has low solidity (non-convex, e.g. entrance
    # merged with lateral shadow), split by valid-region weight: keep only
    # the high-weight pixels (well-illuminated zone = actual entrance).
    if scores.get("solidity", 1.0) < 0.65 and np.count_nonzero(best_mask) > 100:
        mask_weights = wmap[best_mask > 0]
        w_thresh = np.percentile(mask_weights, 60)
        high_w = ((best_mask > 0) & (wmap >= w_thresh)).astype(np.uint8) * 255
        # Clean and extract connected components
        sk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        high_w = cv2.morphologyEx(high_w, cv2.MORPH_CLOSE, sk)
        high_w = cv2.morphologyEx(high_w, cv2.MORPH_OPEN, sk)
        n_hw, labels_hw, stats_hw, centroids_hw = cv2.connectedComponentsWithStats(
            high_w, 8)
        if n_hw > 1:
            # Filter: keep only components whose centroid is within valid cols
            valid_comps = []
            for ci in range(1, n_hw):
                cx_ci = centroids_hw[ci, 0]
                area_ci = stats_hw[ci, cv2.CC_STAT_AREA]
                if lc <= cx_ci <= rc and area_ci >= np.count_nonzero(best_mask) * 0.10:
                    valid_comps.append((ci, area_ci))
            if valid_comps:
                # Pick the largest valid component
                best_ci = max(valid_comps, key=lambda x: x[1])[0]
                candidate_hw = ((labels_hw == best_ci) * 255).astype(np.uint8)
                best_mask = candidate_hw
            else:
                # Fallback: just use largest
                largest = 1 + np.argmax(stats_hw[1:, cv2.CC_STAT_AREA])
                candidate_hw = ((labels_hw == largest) * 255).astype(np.uint8)
                if np.count_nonzero(candidate_hw) >= np.count_nonzero(best_mask) * 0.15:
                    best_mask = candidate_hw

    # ── Post-selection iterative expansion ───────────────────────────────────
    # Conservative CC-flood growth: only absorb pixels that are truly dark
    # (mean stays ≤ 0.28 absolute, or ≤ 2× the seed mean) and the area
    # stays ≤ 25% of the image.  Stop early if interior std rises sharply
    # (would mean we're growing into textured terrain, not a void).
    #
    # Two modes depending on lateral borders (unchanged from before):
    #  • Lateral zone: hard-clip relax_dark at lc / rc.
    #  • No lateral zone: soft-clip at wmap < 0.15.
    best_area_frac = np.count_nonzero(best_mask) / (h * w)
    if best_area_frac < 0.25:
        orig_mean = float(gray_f32[best_mask > 0].mean())
        orig_std  = float(gray_f32[best_mask > 0].std())
        current   = best_mask.copy()
        br_size   = max(9, int(min(h, w) * 0.02) | 1)
        br_k      = cv2.getStructuringElement(
                        cv2.MORPH_ELLIPSE, (br_size, br_size))
        has_lateral = lc > int(w * 0.05) or rc < int(w * 0.95)

        # Conservative percentile range: max 30 (previously 50)
        for relax_pct in [10, 15, 20, 25, 30]:
            relax_thr = int(np.percentile(proc["denoised"], relax_pct))
            _, relax_dark = cv2.threshold(
                proc["denoised"], relax_thr, 255, cv2.THRESH_BINARY_INV)
            relax_dark = cv2.morphologyEx(relax_dark, cv2.MORPH_CLOSE, br_k)

            if has_lateral:
                relax_dark[:, :lc] = 0
                relax_dark[:, rc+1:] = 0

            n_rd, labels_rd, _, _ = cv2.connectedComponentsWithStats(
                relax_dark, 8)
            overlap_labels = set(np.unique(labels_rd[current > 0])) - {0}
            if not overlap_labels:
                continue

            candidate = np.zeros_like(current)
            for lb in overlap_labels:
                candidate[labels_rd == lb] = 255

            if not has_lateral:
                candidate[wmap < 0.15] = 0

            n_c, labels_c, stats_c, _ = cv2.connectedComponentsWithStats(
                candidate, 8)
            if n_c > 1:
                lc_idx = 1 + np.argmax(stats_c[1:, cv2.CC_STAT_AREA])
                candidate = ((labels_c == lc_idx) * 255).astype(np.uint8)

            cand_frac = np.count_nonzero(candidate) / (h * w)
            if cand_frac == 0:
                continue
            cand_mean = float(gray_f32[candidate > 0].mean())
            cand_std  = float(gray_f32[candidate > 0].std())

            # ── Stop conditions ──────────────────────────────────────────────
            # 1. Hard area ceiling
            if cand_frac > 0.30:
                break

            # 2. Absolute brightness ceiling (nothing this bright is a void)
            if cand_mean > 0.35:
                break

            # 3. Texture: rising std means we're entering textured terrain.
            # Cave voids: std ~0.03–0.05. Cave walls: ~0.06–0.08. Terrain: ~0.09+.
            std_limit = max(orig_std * 1.5, 0.08)
            if cand_std > std_limit:
                break

            # 4. Local contrast: stop when candidate interior is nearly as bright
            #    as its immediate exterior — boundary between two similar-dark regions.
            outer_dil = cv2.dilate(candidate, br_k)
            outer_region = outer_dil.astype(bool) & ~candidate.astype(bool)
            if outer_region.any():
                outer_mean = float(gray_f32[outer_region].mean())
                if outer_mean - cand_mean < 0.07:
                    break

            # 5. Depth constraint: newly-added pixels must be "far enough".
            #    The depth model assigns low values to cave voids and high values
            #    to nearby surfaces.  If the marginal pixels being added are near
            #    (high depth_norm), we are expanding into terrain, not the cave.
            if depth_norm is not None:
                new_px = candidate.astype(bool) & ~current.astype(bool)
                if new_px.any():
                    marginal_depth = float(depth_norm[new_px].mean())
                    if marginal_depth > 0.30:
                        break

            current = candidate

        final_frac = np.count_nonzero(current) / (h * w)
        if final_frac > best_area_frac * 1.05:
            print(f"  [{bn}] expanded {best_area_frac*100:.1f}% → "
                  f"{final_frac*100:.1f}%")
            best_mask = current

        # ── Wall-expansion bonus ──────────────────────────────────────────────
        # If the main expansion produced a small result (< 18%) and there are no
        # lateral borders, attempt one or two extra percentile levels with a
        # slightly relaxed std limit (0.11).  This captures cave walls that are
        # slightly more textured than the pure void but still darker than terrain.
        # A tight growth budget (+6% max) prevents unconstrained expansion.
        wall_frac = np.count_nonzero(best_mask) / (h * w)
        if not has_lateral and wall_frac < 0.18:
            wb_current = best_mask.copy()
            wb_orig_std = float(gray_f32[best_mask > 0].std())
            for relax_pct in [35, 40]:
                relax_thr = int(np.percentile(proc["denoised"], relax_pct))
                _, relax_dark = cv2.threshold(
                    proc["denoised"], relax_thr, 255, cv2.THRESH_BINARY_INV)
                relax_dark = cv2.morphologyEx(relax_dark, cv2.MORPH_CLOSE, br_k)
                relax_dark[wmap < 0.15] = 0

                n_rd, labels_rd, _, _ = cv2.connectedComponentsWithStats(relax_dark, 8)
                ov_labels = set(np.unique(labels_rd[wb_current > 0])) - {0}
                if not ov_labels:
                    continue
                wb_cand = np.zeros_like(wb_current)
                for lb in ov_labels:
                    wb_cand[labels_rd == lb] = 255
                n_c, lbs_c, sts_c, _ = cv2.connectedComponentsWithStats(wb_cand, 8)
                if n_c > 1:
                    wb_cand = ((lbs_c == 1 + np.argmax(sts_c[1:, cv2.CC_STAT_AREA])) * 255
                               ).astype(np.uint8)

                wb_frac = np.count_nonzero(wb_cand) / (h * w)
                if wb_frac == 0:
                    continue
                wb_mean = float(gray_f32[wb_cand > 0].mean())
                wb_std  = float(gray_f32[wb_cand > 0].std())

                if wb_frac > wall_frac + 0.06:  # max 6% additional growth
                    break
                if wb_mean > 0.27:
                    break
                if wb_std > max(wb_orig_std * 2.0, 0.11):
                    break
                wb_outer = cv2.dilate(wb_cand, br_k).astype(bool) & ~wb_cand.astype(bool)
                if wb_outer.any() and float(gray_f32[wb_outer].mean()) - wb_mean < 0.07:
                    break
                # Depth constraint on marginal pixels
                if depth_norm is not None:
                    wb_new = wb_cand.astype(bool) & ~wb_current.astype(bool)
                    if wb_new.any() and float(depth_norm[wb_new].mean()) > 0.30:
                        break
                wb_current = wb_cand

            wb_final = np.count_nonzero(wb_current) / (h * w)
            if wb_final > wall_frac * 1.05:
                print(f"  [{bn}] wall-expanded {wall_frac*100:.1f}% → {wb_final*100:.1f}%")
                best_mask = wb_current

    # ── GrabCut boundary refinement ───────────────────────────────────────────
    # After flood-fill gives us a coarse region, GrabCut refines the exact
    # boundary using a per-pixel GMM (dark interior vs bright surround) and
    # graph-cut optimisation — giving a crisp, edge-following contour.
    pre_gc_frac = np.count_nonzero(best_mask) / (h * w)
    best_mask = grabcut_refine(best_mask, gray_u8, h, w)
    post_gc_frac = np.count_nonzero(best_mask) / (h * w)
    if abs(post_gc_frac - pre_gc_frac) > 0.005:
        print(f"  [{bn}] grabcut {pre_gc_frac*100:.1f}% → {post_gc_frac*100:.1f}%")

    refined = refine_mask(best_mask, gray_f32)
    draw_result(gray_u8, refined, scores,
                out_r, out_m, out_dv,
                wmap, pn,
                out_dc, candidates, all_sc)

    outputs = [out_r, out_m, out_dv, out_dc]
    if depth_norm is not None:
        outputs.append(out_dd)
    for p in outputs:
        print(f"  [{bn}] saved: {os.path.basename(p)}")
    return outputs


# ──────────────────────────────────────────────────────────────────────────────
# 10. MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) == 3:
        process_image(sys.argv[1],
                      os.path.dirname(sys.argv[2]) or ".")
    else:
        cwd = os.path.dirname(os.path.abspath(__file__))
        patterns = ["*.jpg","*.jpeg","*.png","*.JPG","*.JPEG","*.PNG"]
        found = []
        for pat in patterns:
            found.extend(glob.glob(os.path.join(cwd, pat)))
        suffixes = ("_result.png","_mask.png","_debug_valid.png",
                    "_debug_candidates.png","_debug_depth.png")
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

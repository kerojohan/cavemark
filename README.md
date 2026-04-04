# CaveMark

Automatic cave entrance detector for IR/NIR monochrome imagery — no deep learning required.

CaveMark uses a classical computer vision pipeline (OpenCV + NumPy) to locate cave entrances in images captured by trail cameras, security cameras or NIR-equipped sensors in low-light or no-light conditions.

---

## How it works

```
Load → Preprocess → Valid Region → IR Depth → Candidates → Score → Expand → GrabCut → Refine → Visualise
```

1. **Preprocess** — median denoise + large-blur background estimation + division normalisation (corrects uneven IR illumination)
2. **Valid region** — horizontal illumination profile (80th percentile per column) builds a soft weight map that suppresses lateral dark borders caused by the camera's IR flash fall-off
3. **IR physics depth** — per-pixel `darkness × local_uniformity` at three blur scales; cave voids absorb all IR and are spatially uniform, so they score high; textured rock/vegetation scores low even when dark
4. **Candidate generation** — six complementary strategies:
   - Multi-level thresholding (standard and heavy morphological bridging)
   - Iterative seed-growth from the darkest pixels
   - Otsu thresholding
   - Adaptive threshold intersected with a dark base
   - Valid-zone-only masking (lateral shadows masked before connected-component extraction)
5. **Scoring** — multiplicative gates prevent wrong-size, wrong-shape, or textured blobs from winning regardless of darkness:
   - Area gate: ideal 8 %–28 % of image; hard penalty below 2 % or above 45 %
   - Solidity gate: very non-convex shapes penalised
   - **Texture gate** *(new)*: high internal pixel std-dev penalises textured regions
   - **Vertical gate** *(new)*: centroid in top 25 % of frame penalised
   - Additive components: contrast vs. surround, darkness, enrichment of darkest pixels, distance-transform depth, IR physics depth, boundary gradient, valid-region alignment, aspect ratio, centroid position
6. **Post-selection expansion** — grows the selected mask into connected dark pixels at a relaxed threshold; captures pit entrances where the initial candidate covers only the darkest core
7. **GrabCut refinement** *(new)* — OpenCV graph-cut sharpens the boundary after expansion; eroded core = definite FG, dilated shell = probable FG, outer ring = probable BG
8. **Refine** — morphological close + bordered flood-fill to fill interior holes + contour smoothing (wrap-around Gaussian, σ capped at 15)

---

## Requirements

```
python >= 3.8
opencv-python
numpy
```

Install with:

```bash
pip install opencv-python numpy
```

---

## Usage

### Single image

```bash
python detect_cave.py input.jpg output.jpg
```

### Multiple images

```bash
python detect_cave.py img1.png img2.png img3.png
```

### Batch mode

Run without arguments to process every `.jpg` / `.png` in the current directory (output files are excluded automatically):

```bash
python detect_cave.py
```

---

## Output files

For each input image `NAME.png` four files are produced:

| File | Description |
|------|-------------|
| `NAME_result.png` | Original image with amber dilation ring, green overlay, contour and score label |
| `NAME_mask.png` | Binary mask (white = cave entrance) |
| `NAME_debug_valid.png` | Valid-region weight map + illumination profile curve |
| `NAME_debug_candidates.png` | All scored candidates with their scores; best candidate in white |

---

## Examples

Three test images are included in the repository root (`background.png`, `background2.png`, `background3.png`). Run the detector:

```bash
python detect_cave.py
```

### background.png — horizontal slot entrance (1728 × 1296)

A cave entrance visible as a dark horizontal slot between a rock ceiling and a gravel floor. The image has strong IR flash fall-off on both lateral edges.

| Result | Mask |
|--------|------|
| ![result](examples/background_result.png) | ![mask](examples/background_mask.png) |

Score: **0.83** — area 7.7 %, contrast 0.61, IR depth 1.00

Valid-region weight map (lateral dark borders suppressed):

![valid region](examples/background_debug_valid.png)

---

### background2.png — large vegetated entrance (1920 × 1080)

A wide cave entrance surrounded by vegetation. Uniform IR illumination — no lateral correction needed.

| Result | Mask |
|--------|------|
| ![result](examples/background2_result.png) | ![mask](examples/background2_mask.png) |

Score: **0.96** — area 16.8 %, contrast 0.92, IR depth 1.00

---

### background3.png — vertical pit entrance (1024 × 576)

A circular pit entrance viewed from above. The darkest core is only the deepest part; the full visible pit is recovered by post-selection expansion (8.9 % → 20.2 %) then refined by GrabCut.

| Result | Mask |
|--------|------|
| ![result](examples/background3_result.png) | ![mask](examples/background3_mask.png) |

Score: **1.00** — area 19.5 %, contrast 1.00, IR depth 1.00

Candidate scoring debug view:

![candidates](examples/background3_debug_candidates.png)

---

## Scoring details

```
final_score = additive × area_mult × solidity_mult × texture_mult × vert_gate × lateral_pen

additive =
    0.24 × contrast_score        # brightness drop vs. surroundings (primary)
  + 0.14 × dark_score            # raw mean darkness
  + 0.06 × enrichment_score      # concentration of darkest 5 % of pixels
  + 0.12 × depth_score           # distance-transform depth (penalises narrow shadows)
  + 0.10 × ir_depth_score        # IR physics depth (darkness × local uniformity)
  + 0.09 × gradient_score        # edge sharpness at boundary
  + 0.06 × valid_score           # alignment with illuminated zone
  + 0.03 × aspect_score          # not absurdly thin
  + 0.04 × position_score        # mild centre preference
  + 0.12                         # base
```

---

## Limitations

- Assumes a single dominant cave entrance per image
- Struggles when the entrance is brighter than its surroundings (e.g. back-lit scenes)
- Very thin or fragmented entrances may score lower than large dark shadows; tweak `min_area` and threshold percentiles if needed

---

## License

MIT

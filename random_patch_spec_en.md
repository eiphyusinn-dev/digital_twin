# Random Patch Extraction Specification

> **Note**: This specification was created with the assistance of AI. Its purpose is to convey the overall strategy and algorithm intent. Implementation details (folder structure, filename conventions, config structure, etc.) may be adjusted at the implementer's discretion.

---

## 0. Key Differences from Current Script

Summary of changes from the current `patch_split_preprocess_seg.py`.

| # | Item | Current | This Spec |
|---|------|---------|-----------|
| 1 | **Patch extraction method** | Fixed grid (stride=112) | Random coordinates (defect-centered offset / fully random) |
| 2 | **Histogram normalization timing** | Applied to full image as-is (no crop) | 2 switchable patterns: after work-area crop or after patch splitting |
| 3 | **FDA timing** | After patch splitting only | Same 2 switchable patterns as above |
| 4 | **OK/NG classification** | Defect px >= 10 -> NG, otherwise -> OK | 3-way: NG (>=30% and >=200px) / OK (defect px=0) / Excluded |
| 5 | **OK image source** | Generated from defect-free areas of defect images | Generated from a dedicated defect-free CG image (1 image) |
| 6 | **Defect color** | Hardcoded `BGR=[61,61,204]` | Configurable (separate CG/Real, Real red/blue switching) |
| 7 | **Semantic seg filename** | `instance_segmentation_{idx}.png` | `post_semantic_segmentation_{idx}.png` |
| 8 | **Input folders** | Single directory | Recursive search across 12 folders |
| 9 | **Train/Val split** | None (managed externally) | Built-in 80/20 split by source image |
| 10 | **NG:OK ratio control** | None (save all generated patches) | Ratio specified in config |
| 11 | **CoordConv** | Filename contains `_r{r}_c{c}_` | OFF in random mode, no coordinate pattern |

*Items **1-4 are major logic changes**; 5-11 are configuration/structure extensions.*

---

## 1. Background & Objective

CG (Omniverse) images have completely identical textures, differing only at defect locations.
With the current fixed-grid splitting (224x224, stride=112), OK patches are nearly identical repetitions, leading to a **high risk of overfitting**.

### Countermeasure
Add a random patch extraction mode to ensure diversity in training data.

---

## 2. Overall Strategy

| Item | Description |
|------|-------------|
| Target | **Train + Val** (CG images only) |
| Val (CG) | Random extraction (NG source images separated from train) |
| Test | Fixed grid maintained (same conditions as inference for final evaluation) |
| Real images | Fixed grid maintained (textures differ across images, ensuring natural diversity) |
| CoordConv | **OFF** in random mode (`coordconv.enabled: false`) |
| train.py | No changes needed (`class_weights` already supported by `_create_criterion`) |

---

## 3. Input Data

### 3-1. CG Images (with defects): ~3,000 images

Stored across multiple folders. Each folder contains a subfolder structure.

```
input/cg_image/
├── _defects.pos4_20260216_postL/
│   ├── post_rgb.png/
│   │   ├── post_rgb_0001.png
│   │   ├── post_rgb_0002.png
│   │   └── ...
│   ├── post_semantic_segmentation.png/
│   │   ├── post_semantic_segmentation_0001.png
│   │   ├── post_semantic_segmentation_0002.png
│   │   └── ...
│   └── bounding_box_2d_tight.npy/     <- Not used
├── _defects.pos4_20260216_postM/
│   └── (same structure)
├── _defects.pos4_20260216_postS/
├── _defects.pos4_20260219_post/
├── _defects.pos4_20260220_post1/
├── _defects.pos4_20260220_post2/
├── _defects.pos4_20260223_post1/
├── _defects.pos4_20260223_post2/
├── _defects.pos4_20260223_post3/
├── _defects.pos4_20260223_post4/
├── _defects.pos4_20260224_post1/
└── _defects.pos4_20260224_post2/
```

- **Filename mapping**: `post_rgb_XXXX.png` -> `post_semantic_segmentation_XXXX.png` (in a separate subfolder within the same parent folder)
- **Defect color**: BGR=[61, 61, 204] (for CG, Omniverse semantic segmentation color)
  - Same as the existing `DEFECT_BGR` in current scripts
- Each image has essentially 1 class / 1 region of defect (no connected component analysis needed)

### 3-2. CG Image (defect-free): 1 image

- **Path**: `input/cg_image/post_rgb_original.png`
- Composited by transplanting normal regions from 2 images with different defect locations (already created)

### 3-3. Real Images (future support)

```
real_rgb/                              (tentative directory name)
├── 12_1_-5.png
├── 12_1_-10.png
└── ...

real_semantic_segmentation/            (tentative directory name)
├── 12_1_-5_semantic_segmentation.png
├── 12_1_-10_semantic_segmentation.png
└── ...
```

- **Filename mapping**: `{name}.png` -> `{name}_semantic_segmentation.png`
- **Red** RGB=(255, 0, 0) / BGR=[0, 0, 255]: Definite defect
- **Blue** RGB=(0, 0, 255) / BGR=[255, 0, 0]: Uncertain defect
- Switchable via config: "red only as NG" or "both red + blue as NG"
- Real image patch splitting uses **fixed grid** (existing script)

---

## 4. Preprocessing Pipeline

Histogram normalization and FDA can be switched between the following 2 patterns **via config**.

### Pattern 1: Apply after cropping (recommended)

```
Load original image
-> Crop work area using background mask (exclude background as much as possible)
-> Apply histogram normalization
-> Restore black background using background mask
-> Random patch extraction
```

### Pattern 2: Apply after patch splitting

```
Load original image
-> Apply background mask
-> Random patch extraction (224x224)
-> Apply histogram normalization to each patch
```

### Config settings

```yaml
preprocessing:
  hist_norm:
    enabled: false
    mode: "crop_first"     # "crop_first" (Pattern 1) or "patch_first" (Pattern 2)
  fda:
    enabled: false
    mode: "crop_first"     # same as above
```

---

## 5. Classification Rules & OK/NG Generation Overview

### OK/NG Patch Sources

OK and NG patches are generated from **separate source images**.

| Patch Type | Source | Details |
|-----------|--------|---------|
| **OK patches** | Dedicated defect-free CG image (1 image) | No defects exist, so any crop is OK -> See Section 6 |
| **NG patches** | CG images with defects (~3,000 images) | Random crop around defect centroid -> See Section 7 |

### Classification Rule for NG Patch Extraction

When extracting from defect images, the random offset causes varying defect coverage, so the following criteria are applied:

| Classification | Condition | Action |
|----------------|-----------|--------|
| **NG (defect)** | >= 30% of total defect pixels are within the patch **AND** defect pixels in patch >= 200px | Save |
| **Excluded** | Does not meet the above | Do not save (retry) |

**Purpose of exclusion**: Prevent noise from ambiguous patches that contain small amounts of defect, avoiding boundary case contamination in training.

---

## 6. OK Patch Generation Specification

| Item | Value |
|------|-------|
| Source | Defect-free CG image **1 image** (`input/cg_image/post_rgb_original.png`) |
| Patch size | 224x224 |
| Extraction method | Fully random coordinates |
| Work area condition | Non-black pixels in patch >= **90%** (adjustable 50%-90% via config) |
| Number of patches | Calculated from NG patch count (based on NG:OK ratio) |

### Algorithm

```
1. Load defect-free CG image
2. Load background mask (JSON polygon)
3. Apply preprocessing (Pattern 1 or Pattern 2, as per config)
4. Loop until target count is reached:
   a. Sample random coordinates (x, y) (0 <= x <= W-224, 0 <= y <= H-224)
   b. Get background mask for patch region
   c. work_area_ratio = non-zero pixels in mask / (224 x 224)
   d. work_area_ratio >= min_work_area_ratio -> save patch
   * Since random sampling may produce patches that don't meet the condition,
     sampling attempts are capped at 20x the target count (to prevent infinite loops).
     If the cap is reached before meeting the target, a warning log is issued
     and processing continues with patches collected so far.
```

---

## 7. NG Patch Generation Specification

| Item | Value |
|------|-------|
| Source | CG images with defects (~3,000 images, across 12 folders) |
| Patch size | 224x224 |
| Extraction method | Random offset centered on defect centroid |
| NG condition | >= 30% of total defect pixels within patch **AND** defect pixels in patch >= **200px** |
| Exclusion condition | Does not meet NG condition -> not saved |
| Patches per image | **10 patches** per defect (that meet NG condition) |

### Algorithm

```
For each CG defect image:
1. Load image
2. Apply preprocessing (Pattern 1 or Pattern 2, as per config)
3. Load corresponding semantic segmentation mask
4. Detect defect pixels (create mask with BGR=[61,61,204])
5. Calculate defect pixel centroid (cx, cy)
   - ys, xs = np.where(defect_mask > 0)
   - cx = int(xs.mean()), cy = int(ys.mean())
6. Record total defect pixel count: total_defect_pixels
7. Repeat until patches_per_defect patches meeting NG condition are collected:
   a. Generate random offset (dx, dy)
      - Range: [-patch_size//2, +patch_size//2] (= [-112, +112])
   b. Calculate patch top-left coordinates:
      - patch_x = clamp(cx - 112 + dx, 0, W - 224)
      - patch_y = clamp(cy - 112 + dy, 0, H - 224)
   c. Count defect pixels within patch:
      - overlap = count_nonzero(defect_mask[patch_y:patch_y+224, patch_x:patch_x+224])
   d. Decision:
      - overlap / total_defect_pixels >= 0.3 AND overlap >= 200 -> save as NG patch
      - Otherwise -> exclude (do not save)
   e. Retry limit: max 50 attempts per patch
```

---

## 8. Train/Val Data Split

### NG Images (with defects) Split

| Item | Value |
|------|-------|
| Split unit | **Per source image** (not per patch) |
| Ratio | train: 80% / val: 20% (adjustable via config) |
| Reason | Patches from the same defect spanning train/val would cause data leakage |

```
Example: ~3,000 images
  Train source images: ~2,400 -> 10 patches each -> ~24,000 NG patches
  Val source images:   ~600   -> 10 patches each -> ~6,000 NG patches
```

### OK Images (defect-free) Split

| Item | Value |
|------|-------|
| Split | **Not needed** |
| Reason | Textures are completely identical, so any position yields equivalent patches |
| Method | Generate separate random patches for train and val from the same defect-free CG image |

### Estimate (Train + Val total)

- Train NG patches: ~24,000 -> Train OK patches: ~16,000 -> **Train total: ~40,000 patches**
- Val NG patches: ~6,000 -> Val OK patches: ~4,000 -> **Val total: ~10,000 patches**

---

## 9. NG:OK Ratio Control

| Item | Value |
|------|-------|
| Target ratio | NG:OK = 6:4 to 7:3 (specified in config) |
| Adjustment method | Up to the implementer (patch count control, WeightedSampler, etc. - whichever is easier) |
| One possible approach | (1) Generate all NG patches -> (2) Calculate required OK count from NG count -> (3) Generate OK patches |

- Training class_weights can be adjusted via existing `_create_criterion` (config.yaml: `class_weights: [1.5, 1.0]`, etc.)

---

## 10. Real Image Defect Color Switching Specification

Used when processing Real images with fixed-grid splitting (existing script).

### Config settings

```yaml
defect_colors:
  cg:
    # For CG images (Omniverse semantic segmentation color)
    colors:
      - name: "defect"
        bgr: [61, 61, 204]
        enabled: true

  real:
    # For Real images (manual annotation colors)
    colors:
      - name: "definite_defect"     # Definite defect
        bgr: [0, 0, 255]           # Red (BGR)
        enabled: true
      - name: "uncertain_defect"    # Uncertain defect
        bgr: [255, 0, 0]           # Blue (BGR)
        enabled: true              # Set to false to treat only red as NG
```

### Behavior

- OR-combine pixels from all colors with `enabled: true` to generate defect mask
- `uncertain_defect.enabled: false` -> Only red is treated as defect, blue is ignored
- `uncertain_defect.enabled: true` -> Both red and blue are treated as defect

---

## 11. Filename Convention

### Random Patches (training data)

- OK patches: `cg_ok_rnd{index:05d}_OK.png`
  - Example: `cg_ok_rnd00042_OK.png`
- NG patches: `cg_{image_id}_rnd{index:02d}_NG.png`
  - Example: `cg_0301_rnd07_NG.png` (source image 0301, random patch #7)

**Important**: Does not contain `_r{digit}_c{digit}_` pattern -> Compatible with CoordConv OFF

### Output Directory

```
patches_dataset/random_patches/
├── train/
│   ├── OK/
│   │   ├── cg_ok_rnd00001_OK.png
│   │   └── ... (~16,000 images)
│   └── NG/
│       ├── cg_0301_rnd01_NG.png
│       └── ... (~24,000 images)
└── val/
    ├── OK/
    │   ├── cg_ok_rnd00001_OK.png
    │   └── ... (~4,000 images)
    └── NG/
        ├── cg_2801_rnd01_NG.png
        └── ... (~6,000 images)
```

---

## 12. config.yaml Additional Section

```yaml
random_patch:
  enabled: false                  # Set to true to enable
  patch_size: 224

  # OK settings
  ok_source_image: "input/cg_image/post_rgb_original.png"
  min_work_area_ratio: 0.9        # Minimum work area ratio (0.5-0.9)

  # NG settings (recursively search multiple folders)
  ng_source_dirs:                 # Specify multiple directories as a list
    - "input/cg_image/_defects.pos4_20260216_postL"
    - "input/cg_image/_defects.pos4_20260216_postM"
    - "input/cg_image/_defects.pos4_20260216_postS"
    - "input/cg_image/_defects.pos4_20260219_post"
    - "input/cg_image/_defects.pos4_20260220_post1"
    - "input/cg_image/_defects.pos4_20260220_post2"
    - "input/cg_image/_defects.pos4_20260223_post1"
    - "input/cg_image/_defects.pos4_20260223_post2"
    - "input/cg_image/_defects.pos4_20260223_post3"
    - "input/cg_image/_defects.pos4_20260223_post4"
    - "input/cg_image/_defects.pos4_20260224_post1"
    - "input/cg_image/_defects.pos4_20260224_post2"
  rgb_subdir: "post_rgb.png"                          # RGB subdirectory name within each folder
  seg_subdir: "post_semantic_segmentation.png"        # Semantic seg subdirectory name within each folder
  patches_per_defect: 10
  min_defect_overlap: 0.3         # Minimum ratio of defect pixels within patch
  min_defect_pixels_in_patch: 200 # Minimum defect pixel count in patch

  # Train/Val split
  train_val_split: 0.8            # Train ratio for NG source images (remainder is val)
  random_seed: 42                 # For split reproducibility

  # Ratio
  ng_ok_ratio: [6, 4]             # NG:OK (adjustment method up to implementer)

  # Preprocessing
  mask_json_path: "mask/cg_mask_label.json"
  preprocessing:
    hist_norm:
      enabled: false
      mode: "crop_first"          # "crop_first" or "patch_first"
    fda:
      enabled: false
      mode: "crop_first"          # "crop_first" or "patch_first"

  # Output
  output_dir: "patches_dataset/random_patches"

# Defect color settings (used for both CG and Real)
defect_colors:
  cg:
    colors:
      - name: "defect"
        bgr: [61, 61, 204]
        enabled: true
  real:
    colors:
      - name: "definite_defect"
        bgr: [0, 0, 255]
        enabled: true
      - name: "uncertain_defect"
        bgr: [255, 0, 0]
        enabled: true               # false -> only red is NG
```

---

## 13. Files to Modify/Create

| File | Change Type | Description |
|------|-------------|-------------|
| `patch_splitting_utils/random_patch_extractor.py` | **New** | Random patch extraction script |
| `config.yaml` | **Add** | `random_patch:` + `defect_colors:` sections |
| `patch_splitting_utils/patch_split_preprocess_seg.py` | **Modify** | Load defect color from config (remove hardcoded DEFECT_BGR) |
| `dataset.py` | **Minor (optional)** | Add warning log when CoordConv ON + filename has no coordinates |
| `train.py` | **No changes** | -- |

### Existing Code Reuse

- `utils/preprocessing.py` -> `BackgroundMasking`, `HistogramNormalization`, `FDATransform`
- `patch_split_preprocess_seg.py` -> Reference defect mask loading logic

---

## 14. Prerequisites

| # | Item | Status |
|---|------|--------|
| 1 | Defect-free CG image: `input/cg_image/post_rgb_original.png` | Created |
| 2 | Real image annotation colors: Red=(255,0,0), Blue=(0,0,255) | Confirmed |
| 3 | CG defect images: ~3,000 across 12 folders | Location confirmed |

---

## 15. Verification Steps

1. **Patch generation**: Run `python -m patch_splitting_utils.random_patch_extractor`
2. **Count check**: Verify OK/NG folder file counts are close to the configured ratio
3. **Visual inspection**:
   - NG patches should have visible defects
   - OK patches should not be entirely black background
   - OK patches should show variation (different extraction positions)
   - "Excluded" patches should not be saved
4. **Training test**: Set `coordconv.enabled: false`, change `dataset.root_dir` to random patches output directory, and run training
5. **Val/Test**: Verify that validation/test with fixed-grid patches works correctly

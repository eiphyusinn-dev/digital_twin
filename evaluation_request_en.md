# FDA + Histogram Normalization Effectiveness Comparison - Task Request

## Objective

Conduct a quantitative comparison to determine the optimal combination of FDA and histogram normalization for the training pipeline.

**Key decisions to make:**
1. **FDA only vs FDA + Histogram Normalization** - which is more effective?
2. **Apply to cropped images vs Apply after patch splitting** - which is more effective?

---

## Prerequisites

### Image Data (delivered as `input/` folder zipped)

```
input/
├── _defects.pos4_work/
│   ├── semantic_segmentation_0000.png   ← Background mask (shared for CG & Real)
│   └── ...（Reference data from Omniverse output）
├── cg_image/                            ← CG images: 10
│   ├── post_rgb_0004.png
│   ├── post_rgb_0007.png
│   ├── ...
│   └── post_rgb_0222.png
└── real_image/                          ← Real images: 10
    ├── 12_1_20.png
    ├── 13_2_15.png                      ← ★ FDA reference image
    ├── ...
    └── 17_1_-25.png
```

### FDA Settings
- **Direction**: CG → Real only (apply Real's low-frequency spectrum to CG)
- **Reference**: `input/real_image/13_2_15.png` (fixed, single image)
- **Beta values**: **0.003, 0.005, 0.01** (3 points)

### Histogram Normalization
- Apply histogram normalization **after** FDA (order: FDA → Histogram Normalization)
- **Please pick 2 candidates from past experiments**
  - Examples: LAB CLAHE (L channel only), RGB all-channel CLAHE, standard Histogram Equalization, etc.

### Patch Splitting Specification (used in Evaluation 2)
- Patch size: 224 x 224
- Overlap: 50% (stride = 112)
- Black background patches: exclude (filter out patches that are almost entirely black)
- Reference implementation: `process_single_image` method in `patch_split_preprocess_seg.py`

---

## Evaluation Patterns

### Evaluation 1: Cropped Image Level (9 patterns)

**Preprocessing flow:**
```
Original image (5328x4608) → Background masking → Cropping (~2621x3069) → FDA (± Histogram) → Evaluate
```

| # | Method | Beta |
|---|--------|------|
| 1 | FDA only | 0.003 |
| 2 | FDA only | 0.005 |
| 3 | FDA only | 0.01 |
| 4 | FDA + Histogram Candidate A | 0.003 |
| 5 | FDA + Histogram Candidate A | 0.005 |
| 6 | FDA + Histogram Candidate A | 0.01 |
| 7 | FDA + Histogram Candidate B | 0.003 |
| 8 | FDA + Histogram Candidate B | 0.005 |
| 9 | FDA + Histogram Candidate B | 0.01 |

**Metric computation:** Compare as whole sets (per-image metrics not required)
- JS/EMD: Mean histogram of 10 Real images vs Mean histogram of 10 FDA-processed CG images
- MMD: 10 Real images (set) vs 10 FDA-processed CG images (set)
- t-SNE: Plot all 20 images for visualization

### Evaluation 2: After Patch Splitting (9 patterns)

**Preprocessing flow:**
```
Original image → Background masking → Cropping → Paste onto black background (restore to 5328x4608)
→ Patch splitting (224x224, 50% overlap) → FDA (± Histogram) per matching coordinate pair → Evaluate
```

※ **"Matching coordinate pair"**: Use the Real image's (row, col) patch as the FDA reference, and apply FDA to the corresponding CG image's (row, col) patch

| # | Method | Beta |
|---|--------|------|
| 1 | FDA only | 0.003 |
| 2 | FDA only | 0.005 |
| 3 | FDA only | 0.01 |
| 4 | FDA + Histogram Candidate A | 0.003 |
| 5 | FDA + Histogram Candidate A | 0.005 |
| 6 | FDA + Histogram Candidate A | 0.01 |
| 7 | FDA + Histogram Candidate B | 0.003 |
| 8 | FDA + Histogram Candidate B | 0.005 |
| 9 | FDA + Histogram Candidate B | 0.01 |

**Metric computation:** Compare per matching coordinate pair
- For each coordinate (row, col): compute metrics for Real_patch vs FDA-processed CG_patch
- Per image: mean of all valid patches
- Overall: mean across all 10 CG images

---

## Evaluation Metrics

| Metric | Type | Existing Implementation | Notes |
|--------|------|------------------------|-------|
| JS (Jensen-Shannon) | Histogram distance | Implemented | `scipy.spatial.distance.jensenshannon` |
| EMD (Earth Mover's Distance) | Histogram distance | **Needs implementation** | Can be implemented with `scipy.stats.wasserstein_distance` |
| MMD (Maximum Mean Discrepancy) | Deep feature distance | Implemented | ResNet feature-based |
| t-SNE | Visualization | Implemented | ResNet feature-based |

- Since this is a Real vs CG comparison, **entropy computation is not needed**
- KL Divergence is asymmetric (results depend on which is the reference), so **not needed for this evaluation**

### EMD Implementation Reference

Implementation example in `run_fda_verification.py` from the `fda-verification-tools` repository:

```python
from scipy.stats import wasserstein_distance

def compute_histogram_distances(hist_a, hist_b):
    bins = np.arange(len(hist_a))
    distances = {}
    distances['JS'] = float(jensenshannon(hist_a, hist_b))
    distances['EMD'] = float(wasserstein_distance(bins, bins, hist_a, hist_b))
    return distances
```

- Input: histograms normalized to probability distributions (255 bins)
- Computed per RGB channel
- Lower values indicate more similar distributions

---

## Reference Repositories & Existing Implementations

| Repository | GitHub URL | Key Reference Files |
|-----------|-----------|---------------------|
| Digital-Twin-Construction-Technology-Verification-for-Image-Inspection-Environment | https://github.com/eiphyusinn-dev/Digital-Twin-Construction-Technology-Verification-for-Image-Inspection-Environment | `patch_split_preprocess_seg.py` (patch splitting), `utils/preprocessing.py` (FDA, histogram normalization, background masking), `utils/fda_utils.py` (`FDA_source_to_target_np()`) |
| fda-verification-tools | https://github.com/gw-okada/fda-verification-tools | `run_fda_verification.py` (comprehensive JS, EMD, MMD, t-SNE implementation; reference for EMD) |
| FDA_and_Histogram_Equalization | https://github.com/mllh-gw/FDA_and_Histogram_Equalization | `MMD_fda_evaluation.py` (MMD evaluation), `tsne_fda_evaluation.py` (t-SNE visualization) |

---

## Output Requirements

Format is up to the implementer, but must include at minimum:
- **Metrics summary table** for all 18 patterns (in a format that allows numerical comparison)
- **t-SNE visualizations** (before/after comparison)
- **Analysis notes** on which combination performed best

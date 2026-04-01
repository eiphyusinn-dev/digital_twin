# Bug Report: Increased Valid Patch Count in crop_first + HistNorm Mode

## Issue Found

Experiments 33-36 (crop_first + HistNorm ON) have a different total patch count compared to all other experiments.

| Experiment Group | Total Patches | Difference |
|---|---|---|
| exp1-32 (all except crop_first+HistNorm) | 59,466 | — |
| exp33-36 (crop_first + HistNorm ON) | 61,268 | **+1,802** |

Since the same 106 evaluation images are used, the total patch count should be identical across all experiments.

## Root Cause

In `inference_by_patch.py`, the `_apply_hist_crop_first` method does **not re-apply the background mask after HistNorm (CLAHE)**.

### Current Processing Order (crop_first mode)
```
Image → Background mask → crop_first HistNorm → Patch splitting → Valid patch check
```

### What Happens
1. Background masking sets pixels outside the product area to black (0,0,0)
2. crop_first extracts the bounding box and applies CLAHE
3. CLAHE enhances contrast in the L channel, which **turns some black boundary pixels into small non-zero values**
4. The valid patch filter (`np.count_nonzero(patch) < patch_size² × 0.1`) now passes boundary patches that were previously excluded
5. This results in +1,802 extra patches being evaluated

### Note: patch_first mode is NOT affected
In patch_first mode, the valid patch check happens **before** HistNorm is applied, so this issue does not occur (exp16, 23-25: 59,466 patches, correct).

## Deviation from Original Specification

This issue was anticipated and a fix was previously requested in the following Slack thread:
https://globalwalkerstemp.slack.com/archives/C06LTBASPMG/p1773110563886459?thread_ts=1772771005.127109&cid=C06LTBASPMG

> :memo: [Correction 1] Evaluation 1: Cropped Image Unit (9 patterns)
> [Before]: Original -> Background Masking -> Cropping -> FDA (± Histogram) -> Evaluation
> [After]: Original -> Background Masking -> Cropping -> FDA (± Histogram) -> :o:️ Re-masking (Apply cropped seg_mask) -> Evaluation

It appears that this **Re-masking step was not implemented**.

## Action Required

1. **Confirm**: Verify that the `_apply_hist_crop_first` method is missing the Re-masking step
2. **Fix**: Add background mask re-application after HistNorm processing
3. **Re-run**: Re-run inference and evaluation for exp33-36 with the fixed code, and verify that the total patch count becomes 59,466

## Relevant Code Locations

- `inference_by_patch.py` → `_apply_hist_crop_first` method (needs Re-masking after HistNorm)
- `inference_by_patch.py:172` → Valid patch check: `np.count_nonzero(patch) < (self.patch_size * self.patch_size * 0.1)`

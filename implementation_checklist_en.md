# Training Pipeline - Implementation Status Checklist

---

## 0. Script Classification

> We have organized the scripts into production pipeline and development/verification tools.
> Please confirm whether this classification is correct.

### Production Pipeline (Execution Order)

| Order | Script | Role |
|-------|--------|------|
| 1 | `patch_splitting_utils/patch_split_complete.py` | Preprocessing + patch splitting -> NG/OK folder classification |
| 2 | `train.py` | Model training |
| 3 | `inference_by_patch.py` | Patch-based inference + visualization |
| 4 | `evaluate.py` (**Not implemented**) | Evaluation metrics: inference results vs Ground Truth |

### Internal Scripts (Not executed directly, but required for production)

| Script | Role |
|--------|------|
| `model.py` | ConvNeXt-V2 model definition (CoordConv supported) |
| `dataset.py` | DataLoader + preprocessing pipeline |
| `config.yaml` | Centralized configuration management |
| `utils/preprocessing.py` | FDA / Histogram normalization / Background masking |
| `utils/fda_utils.py` | FDA helper functions |

### Development / Verification Tools (Not used in production pipeline)

| Script | Purpose | Notes |
|--------|---------|-------|
| `inference.py` | Full-image inference (no patch splitting) | Old version. Not used in production |
| `vis_dataloader.py` | Preprocessing pipeline visualization | Useful for data verification. Can be kept |
| `coordconv_dataloader_check.py` | CoordConv coordinate verification | Verification complete. No longer needed |
| `patch_split_bbox.py` | Bounding box based patch splitting | Experimental. Not used in production |
| `patch_split_seg.py` | Segmentation based patch splitting | Experimental. Not used in production |
| `patch_split_no_preprocessing.py` | Patch splitting without preprocessing | Experimental. Not used in production |

---

## 1. Overall Summary

| Cat | Feature | Status | Config Key | Current Setting | Main Files |
|-----|---------|--------|------------|-----------------|------------|
| 2 | FDA | Implemented | `use_fda` | **OFF** | `utils/preprocessing.py:17-71`, `dataset.py:154-159` |
| 2,3 | Background Masking | Implemented | `use_bg_masking` | **ON** | `utils/preprocessing.py:90-143`, `dataset.py:147-151` |
| 2,3 | Histogram Normalization | Implemented | `use_hist_norm` | **OFF** | `utils/preprocessing.py:73-88`, `dataset.py:187-188` |
| 2,3 | CoordConv | Implemented | `coordconv.enabled` | **ON** | `model.py:98-157`, `dataset.py:113-134` |
| 4 | MixUp | Implemented | `mixup.enabled` | **OFF** | `train.py:305-334` |
| 4 | CutMix | Implemented | `cutmix.enabled` | **OFF** | `train.py:336-369` |
| 4 | RandAugment | Implemented | `randaugment.enabled` | **ON** | `dataset.py:245-261` |
| 4 | Color Jitter | Implemented | `color_jitter.enabled` | **ON** | `dataset.py:235-243` |
| 4 | Dropout Enhancement (DropPath) | Implemented | `drop_path_rate` | 0.1 | `model.py:68-121` |
| 5 | Frozen Backbone | Implemented | `freeze_backbone` | **ON** | `train.py:434-436` |
| 5 | Loss Weight Adjustment | Implemented | `class_weights` | [1.0, 1.0] | `train.py:84-89` |
| - | Patch Splitting (224x224) | Implemented | - | stride 112 | `patch_split_complete.py` |
| - | Patch-level Inference | Implemented | - | - | `inference_by_patch.py` |
| - | Work-level Evaluation (OR aggregation) | Implemented | - | threshold 0.7 | `inference_by_patch.py:218` |
| - | Patch-level Evaluation Metrics | **Not implemented** | - | - | `evaluate.py` (new script) |
| - | Recall / Precision Evaluation | **Not implemented** | - | - | `evaluate.py` (new script) |

### Legend
- **Cat 2**: CG image preprocessing
- **Cat 3**: Real image preprocessing
- **Cat 4**: Data Augmentation
- **Cat 5**: Domain Adaptation / Training strategy

---

## 2. Remaining Tasks: Evaluation

### 2-1. train.py validate() - Improvement Recommended (Low Priority)

The `validate()` method in `train.py` (`train.py:227-259`) only calculates **Accuracy**.

```python
# train.py:249-254 (current)
total += targets.size(0)
correct += predicted.eq(targets).sum().item()
# ...
val_acc = 100. * correct / total  # Accuracy only
```

validate() is a process that periodically checks performance on validation data during training (e.g., detecting overfitting). Currently it only calculates Accuracy, but adding Recall would enable monitoring the "miss rate" during training.

However, **the top priority is evaluate.py (post-inference evaluation)**. Modifying validate() can be addressed if time permits.

### 2-2. evaluate.py (New Script Required)

We propose creating `evaluate.py` as a separate script to separate inference from evaluation.

**Reasons:**
- The current project already separates scripts by function (patch_split -> train -> inference). Evaluation follows the same pattern
- If inference results are saved, evaluation can be re-run with different thresholds without re-running inference (saves GPU time)
- NVIDIA TAO Toolkit itself provides `inference` and `evaluate` as separate commands, consistent with industry standards

#### Input

| Data | Source | Description |
|------|--------|-------------|
| Inference results | `inference_by_patch.py` | NG probability and coordinates for all patches (requires modification, see below) |
| Ground Truth | `patch_split_complete.py` | NG/OK labels per patch (reuse existing script) |

#### Patch-level Evaluation

For each patch (224x224):

| | Predicted NG | Predicted OK |
|---|-------------|-------------|
| **Actual NG** | TP | FN (miss) |
| **Actual OK** | FP (false alarm) | TN |

-> Calculate Recall, Precision, F1-score

#### Work-level Evaluation (Full Image)

OR aggregation: If any patch in the image is NG -> the entire image is classified as NG (same logic as `inference_by_patch.py:218`)

Calculate TP/FP/TN/FN at the image level -> Recall, Precision, F1-score

#### Output

- Confusion Matrix
- Patch-level / Work-level Recall, Precision, F1
- Evaluation results by threshold (current default threshold: 0.7)
- ROC curve / AUC (can be implemented using scikit-learn's `roc_curve()`, `auc()`)

**Impact**: There is currently no mechanism to verify the requirement "Recall 100% (zero misses)". This will be achieved through evaluate.py.

---

### 2-3. Proposed Modification to inference_by_patch.py

To perform patch-level evaluation in evaluate.py, the output of `inference_by_patch.py` needs to be extended.

**Current Processing Flow:**

`inference_by_patch.py:209` calculates the NG probability for each patch, and `:211` performs threshold classification:

```python
# inference_by_patch.py:209
ng_conf = prob[0].item()  # <- NG probability for each patch (0-1). Already calculated

# inference_by_patch.py:211
if ng_conf >= self.threshold:  # <- Classified as NG if >= threshold (0.7)
    ng_patches.append(...)     # <- Only NG patches are saved. OK patch info is discarded
```

**Problem:** OK patch information (including probability values) is discarded, making it impossible for evaluate.py to evaluate all patches.

**Proposal:** Keep all patches' `(row_idx, col_idx, ng_probability)` in a list and add a feature to export them in CSV or JSON format.

- `ng_probability` is simply saving the value already calculated at `:209`
- `predicted_label` (NG/OK) is mechanically determined by the threshold, so it is more flexible to calculate it on the evaluate.py side while varying the threshold (enables threshold tuning)

---

### 2-4. Ground Truth Generation Method

For generating Ground Truth of test data, **reuse `patch_split_complete.py` as-is**.

**Reasons:**
1. The labeling logic (`patch_split_complete.py:200-205`) is already tested
   - Defect pixel count in the segmentation mask >= 10 -> NG
2. Same patch size (224) and stride (112) ensure coordinate alignment
3. Output filename `{base}_r{row}_c{col}_{label}.png` allows extraction of (row, col, label)
4. Preprocessing (FDA, etc.) does not affect label determination, so it can be applied to test data as well

**Flow:**
```
Test real images -> Run patch_split_complete.py -> Generate NG/OK folders -> evaluate.py reads as ground truth
```

---

## 3. Patch Splitting Variant Comparison

There are 4 types of scripts in `patch_splitting_utils/`.
**Only `patch_split_complete.py` is used in production.** The other 3 were created experimentally.

| Script | BG Mask | Hist Norm | FDA | Label Method | Output Dir | Notes |
|--------|---------|-----------|-----|-------------|------------|-------|
| `patch_split_complete.py` | Yes | Yes | Yes | segmentation | `patches_dataset_bgmask_histnorm_fda/` | **Production use** |
| `patch_split_bbox.py` | Yes | No | No | bbox | `patches_dataset_by_bbox/` | Experimental. Not used |
| `patch_split_seg.py` | Yes | No | No | segmentation | `patches_dataset/` | Experimental. Not used |
| `patch_split_no_preprocessing.py` | No | No | No | segmentation | `patches_dataset_unmasked/` | Experimental. Not used |

**Notes:**
- All variants share: patch size 224x224, stride 112 (50% overlap)
- `patch_split_complete.py` is the most feature-complete, but since preprocessing is also performed on the DataLoader (`dataset.py`) side, be careful of double application

---

## 4. Training / Inference Preprocessing Consistency

| Preprocessing | Training | inference_by_patch.py | Consistency |
|--------------|----------|----------------------|-------------|
| Background Masking | Yes | Yes | Match |
| Histogram Normalization | Yes (if enabled) | Yes (if enabled) | Match |
| FDA | Yes (CG only) | - | **No issue** *1 |
| ImageNet Normalization *2 | Yes | Yes | Match |
| CoordConv | Yes | Yes | Match |

***1 FDA not applied during inference**: FDA is a process applied only to CG images. Since inference is performed only on real images, not applying FDA during inference is not a problem.

***2 ImageNet Normalization**: A mandatory, fixed process to match the input specification of the pretrained model (ConvNeXt-V2). Normalizes each RGB channel using the ImageNet dataset mean (R=0.485, G=0.456, B=0.406) and standard deviation (R=0.229, G=0.224, B=0.225). Cannot be toggled ON/OFF.

---

## 5. Additional Notes

### Regarding Mask JSON Files

Since CG and real image camera angles now match, only `mask/cg_mask_label.json` is used.
`mask/real_mask_label.json` was created when the camera angles were different and is no longer needed.

Alternatively, the segmentation mask output from Omniverse Replicator
(e.g., `_defects.pos4_work/semantic_segmentation_0000.png`) can also be used as a background mask.

---

## 6. Confirmation Requests for Developer

> The following is organized based on reading the current code. Please confirm whether our understanding is correct.

1. **Script classification (Section 0)**: Is the classification of "Production / Internal / Development" correct?
2. **Create evaluate.py as a new script**: Is it acceptable to separate inference and evaluation? (Separated design)
3. **Proposed modification to inference_by_patch.py**: `inference_by_patch.py:209` already calculates the NG probability for each patch, but only NG patches are saved and OK patches are discarded (`:211`). If we add a feature to export all patches' results (row, col, ng_probability) in CSV or JSON, evaluate.py can perform patch-level evaluation while varying the threshold. Is this approach acceptable?
4. **Reuse `patch_split_complete.py` as-is** for generating test data Ground Truth: Is this acceptable? (Since the labeling logic is identical)

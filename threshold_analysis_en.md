# Threshold Analysis — Visualization Script Task

## 1. Background & Purpose

Current evaluation uses a **fixed threshold of 0.7** for NG classification, but **Patch Level Recall is low** (e.g., 42.5% for Exp15).
The threshold 0.7 may not be optimal. We want to **visualize how metrics change across different thresholds** and analyze the **confidence score distribution** to find a better threshold.

- **No need to re-run inference** — use existing `patch_prob_csv` data + GT patch data
- Target: experiments previously presented to the customer

---

## 2. Graphs to Create (per Experiment)

### Graph 1: Confidence Distribution Histogram by GT Label

Display a histogram of each patch's confidence score (`ng_probability`), **color-coded by ground truth label (NG vs OK)**.
Output **2 graphs: Patch Level and Work Level**.

#### Patch Level Histogram
- X-axis: `ng_probability` (0.0 to 1.0)
- Y-axis: Number of patches (frequency)
- Overlay **GT:NG patches** and **GT:OK patches** with different colors (semi-transparent)
- Draw a vertical dashed line at current threshold 0.7
- Title example: `Exp15 - Patch Confidence Distribution (Patch Level)`

```
Count
  |  # = GT:NG patches (actual defects)
  |  o = GT:OK patches (normal)
  |
  |o
  |o                                          #
  |oo                                    #    #
  |oo                                    #    #
  |ooo                              #    #    #
  |oooo                        #    #    #    #
  |ooooo                  #    #    #    #    #
  |ooooooo           #    #    #    #    #    #
  +--+--+--+--+--+--+--+--+--+--+---> ng_probability
  0.0  0.1  0.2  0.3  0.4  0.5  0.6  0.7  0.8  0.9  1.0
                                      :
                                  Threshold 0.7 (dashed)
```

#### Work Level Histogram
- X-axis: **Maximum ng_probability per image** (0.0 to 1.0)
- Y-axis: Number of images (frequency)
- Color-code **GT:NG images** (containing at least one NG patch) vs **GT:OK images** (all patches OK)
- At Work Level, an image is classified as NG if any patch's confidence >= threshold, so use max value
- Draw a vertical dashed line at current threshold 0.7
- Title example: `Exp15 - Max Confidence Distribution (Work Level)`

#### What this graph reveals

| Pattern | Meaning |
|---------|---------|
| NG patches concentrated at right (0.8-1.0) | Model detects defects with high confidence -> threshold 0.7 is sufficient |
| **NG patches concentrated around 0.5-0.7** | **Threshold 0.7 causes many misses (FN) -> should lower threshold** |
| OK patches concentrated at left (0.0-0.2) | Model correctly assigns low scores to normal patches -> few false positives |
| NG and OK distributions overlap significantly | Model's discriminative ability is inherently low (hard to fix by threshold tuning alone) |

---

### Graph 2: Precision / Recall / F1 vs Threshold Curve

Vary the threshold from **0.0 to 1.0 in 0.01 increments** and plot Precision, Recall, and F1 as line graphs.
Output **2 graphs: Patch Level and Work Level**.

- X-axis: Threshold (0.0 to 1.0)
- Y-axis: Metric value (0.0 to 1.0)
- **Precision (red), Recall (blue), F1 (green)** — 3 lines
- Mark the threshold where F1 is maximized with a star marker, and display the threshold value and F1 score as text on the graph
- Draw a vertical dashed line at current threshold 0.7
- Title example: `Exp15 - Precision/Recall/F1 vs Threshold (Patch Level)`

```
1.0 -+
     |  === Recall (blue)
     |  --- Precision (red)
     |  ... F1 (green)
     |
     |========================\
0.8 -+                         \===
     |               ...........\...
     |          .../...          \
0.6 -+       ./...                \==
     |     /.            * Best F1
     |   /.              (threshold=0.XX, F1=0.XX)
0.4 -+--/-------------------------------------
     |/
0.2 -+-------------
     |
0.0 -+--+--+--+--+--+--+--+--+--+---> Threshold
     0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
                                :
                            Threshold 0.7 (dashed)
```

#### Calculation Method
```
Vary threshold from 0.0 to 1.0 in 0.01 increments (101 points)
For each threshold:
  ng_probability >= threshold -> predict NG
  ng_probability <  threshold -> predict OK
Compare with GT to compute TP/FP/FN/TN
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1        = 2 * Precision * Recall / (Precision + Recall)
```

---

## 3. Additional Output: Threshold Metrics CSV

In addition to graphs, save metrics for each threshold as a **CSV file** (for numerical verification).

### Filename: `threshold_metrics.csv`

```csv
threshold,patch_tp,patch_fp,patch_tn,patch_fn,patch_precision,patch_recall,patch_f1,patch_accuracy,work_tp,work_fp,work_tn,work_fn,work_precision,work_recall,work_f1,work_accuracy
0.00,...
0.01,...
...
1.00,...
```

---

## 4. Target Experiments

### 4-1. Fixed Grid Method (case2)

Google Drive: https://drive.google.com/drive/u/0/folders/1k4PYwAmhLDHTrnUhBjFn9Bpx4LXtJ0uW

| Exp | Preprocessing | Evaluation Results Folder | Note |
|-----|--------------|--------------------------|------|
| Exp1 | FDA (b=0.005), No HistNorm | `exp1_results/new_evaluation_results/` | |
| Exp2 | FDA (b=0.005), No HistNorm | `exp2_results/new_evaluation_results/` | |
| Exp3 | FDA (b=0.005), No HistNorm | `exp3_results/new_evaluation_results/` | |
| Exp4 | FDA (b=0.005), No HistNorm | `exp4_results/new_evaluation_results/` | |
| Exp5 | FDA (b=0.005), No HistNorm | `exp5_results/new_evaluation_results/` | |
| Exp6 | FDA (b=0.001), Clahe-LAB | `exp6_results/whole_img_histnorm_evaluation_results/` | whole_image mode |
| Exp7 | FDA (b=0.001), Clahe-LAB | `exp7_results/whole_img_histnorm_evaluation_results/` | whole_image mode |
| Exp8 | FDA (b=0.001), Clahe-LAB | `exp8_results/whole_img_histnorm_evaluation_results/` | whole_image mode |

### 4-2. Random Patch Method (case2)

Google Drive: https://drive.google.com/drive/folders/1CrbINN5k6uHUK_Cp97CcGGQy8G2HVNUL

| Exp | Preprocessing | Dataset | Evaluation Results Folder | Note |
|-----|--------------|---------|--------------------------|------|
| Exp3 | None | Full | `exp3/new_evaluation_results/` | |
| Exp6 | None | Full | `exp6/new_evaluation_results/` | |
| Exp9 | None | Subset | `exp9/new_evaluation_results/` | |
| Exp15 | FDA (patch_first) | Subset | `exp15/new_evaluation_results/` | Best model |
| Exp16 | FDA + HistNorm (patch_first) | Subset | `exp16/new_evaluation_results/` | |
| Exp17 | None | Subset | `exp17/new_evaluation_results/` | |

### Evaluation Dataset (case2, shared across all experiments)
- **106 real images**
- Red (93 images) = NG (defects)
- Blue (11 images) = Excluded
- Real OK (13 images) = OK (normal)
- Image source: 20260302kizu (93 images) + 20260304ok (13 images)

---

## 5. Required Data

Each experiment requires the following data:

| Data | Description | Location |
|------|-------------|----------|
| `patch_prob_csv/` | Per-patch `ng_probability` (continuous 0-1 values) | `all_annotated_dir/patch_prob_csv/` within each exp |
| GT Patches | Ground truth labels (NG/OK) | NG/OK folders (`{image_name}_r{row}_c{col}_{NG\|OK}.png`) |

### patch_prob_csv Format
```csv
filename,row_idx,col_idx,x,y,ng_probability
12_1_0.png,0,18,2016,0,0.42195454239845276
12_1_0.png,0,19,2128,0,0.07127857953310013
...
```

### GT Patch Filename Convention
```
GT_DIR/
├── NG/
│   ├── 12_1_0_r5_c20_NG.png
│   └── ...
└── OK/
    ├── 12_1_0_r0_c18_OK.png
    └── ...
```

Regex: `(.+)_r(\d+)_c(\d+)_(NG|OK)$`

### Matching CSV Predictions with GT
Match by `filename` + `row_idx` + `col_idx`:
- CSV: `filename=12_1_0.png, row_idx=5, col_idx=20` -> GT: `NG/12_1_0_r5_c20_NG.png`

---

## 6. Script Reference

Please create a script based on the following reference implementation.
Reuse the logic from `evaluate.py`'s `PatchEvaluator` class (GT loading, metrics calculation).

### Usage (Command Line)
```bash
python visualize_threshold_analysis.py \
    --inference_csvs /path/to/exp15/all_annotated_dir/patch_prob_csv/ \
    --ground_truth /path/to/gt_patches/ \
    --output_dir /path/to/exp15/new_evaluation_results/threshold_analysis/ \
    --exp_name "Exp15"
```

### Script Structure (Reference)

```python
#!/usr/bin/env python3
"""
Threshold Analysis Visualization Script
- Graph 1: Confidence distribution histogram by GT label (Patch Level + Work Level)
- Graph 2: Precision/Recall/F1 vs Threshold curve (Patch Level + Work Level)
- CSV: Per-threshold metrics
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
import re
import json


def load_inference_csvs(csv_dir):
    """Load all CSVs from patch_prob_csv directory"""
    # Same logic as evaluate.py PatchEvaluator.load_inference_csvs
    csv_files = list(Path(csv_dir).glob("patch_probs_*.csv"))
    all_data = {}
    for f in csv_files:
        key = f.stem.replace("patch_probs_", "")
        all_data[key] = pd.read_csv(f).to_dict('records')
    return all_data


def load_ground_truth(gt_dir):
    """Load labels from GT directory (NG/OK subfolders)"""
    # Same logic as evaluate.py PatchEvaluator.load_ground_truth
    filename_re = re.compile(r"(.+)_r(\d+)_c(\d+)_(NG|OK)$")
    gt_data = defaultdict(dict)
    for label in ["NG", "OK"]:
        label_dir = Path(gt_dir) / label
        if not label_dir.exists():
            continue
        for img_file in label_dir.glob("*.png"):
            match = filename_re.match(img_file.stem)
            if match:
                base_name, r, c, _ = match.groups()
                gt_data[base_name][(int(r), int(c))] = label
    return gt_data


def pair_predictions_with_gt(inference_data, ground_truth):
    """Pair inference results with GT, return list of (probability, gt_label)"""
    patch_pairs = []  # [(ng_prob, is_ng), ...]
    image_data = defaultdict(lambda: {"max_prob": 0.0, "has_ng": False})

    for image_name, patches in inference_data.items():
        if image_name not in ground_truth:
            continue
        gt_img = ground_truth[image_name]
        for p in patches:
            r, c, prob = p['row_idx'], p['col_idx'], p['ng_probability']
            gt_label = gt_img.get((r, c))
            if gt_label is None:
                continue
            is_ng = 1 if gt_label == "NG" else 0
            patch_pairs.append((prob, is_ng))
            # Work Level: max confidence and NG presence per image
            image_data[image_name]["max_prob"] = max(image_data[image_name]["max_prob"], prob)
            if is_ng:
                image_data[image_name]["has_ng"] = True

    work_pairs = [(v["max_prob"], 1 if v["has_ng"] else 0) for v in image_data.values()]
    return patch_pairs, work_pairs


def plot_confidence_histogram(pairs, output_path, title, xlabel_label):
    """Graph 1: Confidence distribution histogram by GT label"""
    ng_probs = [p for p, label in pairs if label == 1]
    ok_probs = [p for p, label in pairs if label == 0]

    plt.figure(figsize=(10, 6))
    bins = np.arange(0, 1.05, 0.05)  # 0.05 step, 20 bins
    plt.hist(ok_probs, bins=bins, alpha=0.6, label=f"GT: OK ({len(ok_probs)})", color="blue")
    plt.hist(ng_probs, bins=bins, alpha=0.6, label=f"GT: NG ({len(ng_probs)})", color="red")
    plt.axvline(x=0.7, color='black', linestyle='--', linewidth=1.5, label="Current Threshold (0.7)")

    plt.xlabel(xlabel_label)
    plt.ylabel("Count")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def compute_metrics_at_threshold(pairs, threshold):
    """Compute TP/FP/TN/FN/Precision/Recall/F1 at a given threshold"""
    tp = fp = tn = fn = 0
    for prob, is_ng in pairs:
        pred = 1 if prob >= threshold else 0
        if is_ng == 1:
            if pred == 1: tp += 1
            else: fn += 1
        else:
            if pred == 1: fp += 1
            else: tn += 1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy}


def plot_threshold_curve(pairs, output_path, title):
    """Graph 2: Precision/Recall/F1 vs Threshold curve"""
    thresholds = np.arange(0.0, 1.01, 0.01)
    precisions, recalls, f1s = [], [], []

    for t in thresholds:
        m = compute_metrics_at_threshold(pairs, t)
        precisions.append(m["precision"])
        recalls.append(m["recall"])
        f1s.append(m["f1"])

    # Find best F1 point
    best_idx = np.argmax(f1s)
    best_threshold = thresholds[best_idx]
    best_f1 = f1s[best_idx]

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, precisions, 'r-', label="Precision", linewidth=1.5)
    plt.plot(thresholds, recalls, 'b-', label="Recall", linewidth=1.5)
    plt.plot(thresholds, f1s, 'g-', label="F1", linewidth=1.5)
    plt.axvline(x=0.7, color='black', linestyle='--', linewidth=1, label="Current Threshold (0.7)")
    plt.plot(best_threshold, best_f1, 'g*', markersize=15, label=f"Best F1={best_f1:.3f} @ threshold={best_threshold:.2f}")

    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return best_threshold, best_f1


def save_threshold_metrics_csv(patch_pairs, work_pairs, output_path):
    """Save per-threshold metrics as CSV"""
    thresholds = np.arange(0.0, 1.01, 0.01)
    rows = []
    for t in thresholds:
        pm = compute_metrics_at_threshold(patch_pairs, t)
        wm = compute_metrics_at_threshold(work_pairs, t)
        rows.append({
            "threshold": round(t, 2),
            "patch_tp": pm["tp"], "patch_fp": pm["fp"],
            "patch_tn": pm["tn"], "patch_fn": pm["fn"],
            "patch_precision": round(pm["precision"], 4),
            "patch_recall": round(pm["recall"], 4),
            "patch_f1": round(pm["f1"], 4),
            "patch_accuracy": round(pm["accuracy"], 4),
            "work_tp": wm["tp"], "work_fp": wm["fp"],
            "work_tn": wm["tn"], "work_fn": wm["fn"],
            "work_precision": round(wm["precision"], 4),
            "work_recall": round(wm["recall"], 4),
            "work_f1": round(wm["f1"], 4),
            "work_accuracy": round(wm["accuracy"], 4),
        })
    pd.DataFrame(rows).to_csv(output_path, index=False)


def main():
    parser = argparse.ArgumentParser(description="Threshold Analysis Visualization")
    parser.add_argument('--inference_csvs', required=True, help="Path to patch_prob_csv directory")
    parser.add_argument('--ground_truth', required=True, help="Path to GT directory (with NG/OK subfolders)")
    parser.add_argument('--output_dir', required=True, help="Output directory")
    parser.add_argument('--exp_name', default="Experiment", help="Experiment name (for graph titles)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    print(f"Loading inference CSVs from: {args.inference_csvs}")
    inference_data = load_inference_csvs(args.inference_csvs)
    print(f"Loading ground truth from: {args.ground_truth}")
    ground_truth = load_ground_truth(args.ground_truth)

    # 2. Pair predictions with GT
    patch_pairs, work_pairs = pair_predictions_with_gt(inference_data, ground_truth)
    print(f"Matched: {len(patch_pairs)} patches, {len(work_pairs)} images")

    # 3. Graph 1: Confidence distribution histograms
    plot_confidence_histogram(
        patch_pairs,
        output_dir / "confidence_histogram_patch_level.png",
        f"{args.exp_name} - Patch Confidence Distribution (Patch Level)",
        "ng_probability"
    )
    plot_confidence_histogram(
        work_pairs,
        output_dir / "confidence_histogram_work_level.png",
        f"{args.exp_name} - Max Confidence Distribution (Work Level)",
        "Max ng_probability per image"
    )

    # 4. Graph 2: Threshold curves
    best_t_patch, best_f1_patch = plot_threshold_curve(
        patch_pairs,
        output_dir / "threshold_curve_patch_level.png",
        f"{args.exp_name} - Precision/Recall/F1 vs Threshold (Patch Level)"
    )
    best_t_work, best_f1_work = plot_threshold_curve(
        work_pairs,
        output_dir / "threshold_curve_work_level.png",
        f"{args.exp_name} - Precision/Recall/F1 vs Threshold (Work Level)"
    )

    # 5. Threshold metrics CSV
    save_threshold_metrics_csv(patch_pairs, work_pairs, output_dir / "threshold_metrics.csv")

    # Summary
    print(f"\n{'='*50}")
    print(f"Results saved to: {output_dir}")
    print(f"Best F1 (Patch Level): {best_f1_patch:.3f} @ threshold={best_t_patch:.2f}")
    print(f"Best F1 (Work Level):  {best_f1_work:.3f} @ threshold={best_t_work:.2f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
```

---

## 7. Output Files (per Experiment)

```
threshold_analysis/
├── confidence_histogram_patch_level.png    <- Graph 1 (patch level)
├── confidence_histogram_work_level.png     <- Graph 1 (image level)
├── threshold_curve_patch_level.png         <- Graph 2 (patch level)
├── threshold_curve_work_level.png          <- Graph 2 (image level)
└── threshold_metrics.csv                   <- Per-threshold metrics
```

Total: **5 files x 14 experiments = 70 files**

---

## 8. Execution Steps

### Step 1: Create the Script
Create `visualize_threshold_analysis.py` based on the reference code in Section 6.

### Step 2: Run for Each Experiment

#### Fixed Grid (case2) Exp1-5
```bash
# Example for Exp1
python visualize_threshold_analysis.py \
    --inference_csvs ./exp1_results/all_annotated_dir/patch_prob_csv/ \
    --ground_truth /path/to/gt_patches/ \
    --output_dir ./exp1_results/new_evaluation_results/threshold_analysis/ \
    --exp_name "Fixed Grid Exp1"
```

#### Fixed Grid (case2) Exp6-8 (whole_image HistNorm)
```bash
# Example for Exp6
# NOTE: For Exp6-8, use the patch_prob_csv from the re-inference with whole_image HistNorm
#       See "Section 9: Confirmation Points" for CSV location
python visualize_threshold_analysis.py \
    --inference_csvs ./exp6_results/<histnorm_patch_prob_csv_folder>/ \
    --ground_truth /path/to/gt_patches/ \
    --output_dir ./exp6_results/whole_img_histnorm_evaluation_results/threshold_analysis/ \
    --exp_name "Fixed Grid Exp6 (whole_img HistNorm)"
```

#### Random Patch (case2) Exp3,6,9,15,16,17
```bash
# Example for Exp15
python visualize_threshold_analysis.py \
    --inference_csvs ./exp15/all_annotated_dir/patch_prob_csv/ \
    --ground_truth /path/to/gt_patches/ \
    --output_dir ./exp15/new_evaluation_results/threshold_analysis/ \
    --exp_name "Random Patch Exp15"
```

### Step 3: Verify Output
- Confirm that each `threshold_analysis/` folder contains 5 files
- Open `threshold_metrics.csv` and verify that metrics at threshold 0.7 match the existing `evaluation_results_threshold_0.7.json`

---

## 9. Confirmation Points

- [ ] Is `all_annotated_dir/patch_prob_csv/` the correct location for patch_prob_csv?
- [ ] Is the GT patch path the same for all experiments? (case2 evaluation dataset should be shared)
- [ ] Do the script outputs at threshold 0.7 match the existing `evaluation_results_threshold_0.7.json` values?

### Important: Fixed Grid Exp6-8 (whole_image HistNorm)

Exp6-8 used HistNorm (Clahe-LAB) during training, and inference was re-run with whole_image mode HistNorm applied to inference images (`whole_img_histnorm_evaluation_results/`).

HistNorm is an inference preprocessing step — applying it changes the model input, which means **ng_probability values themselves change**. The evaluation results confirm that predictions are different:

| Version | Exp6 TP | FP | Recall |
|---|---|---|---|
| Without HistNorm (new_evaluation_results) | 474 | 52 | 0.406 |
| With HistNorm (whole_img_histnorm_evaluation_results) | 517 | 89 | 0.443 |

**Action Required:**
- [ ] Is the `patch_prob_csv` from the whole_image HistNorm re-inference saved?
  - If YES -> Please provide the CSV location
  - If NO -> Need to re-run inference with `--save_patch_probs` flag for threshold analysis

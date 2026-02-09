# NVIDIA TAO Toolkit Constraint Investigation Report

## 1. Background & Objective

This report summarizes the constraints of adopting NVIDIA TAO Toolkit's pretrained model (`convnextv2_large_trainable_v1.0`) for the development of an automotive parts anomaly detection AI using the ConvNeXt-V2 Large model, and addresses the following questions.

### Answer Summary

| Question | Answer |
|----------|--------|
| Are we required to use TAO Toolkit? | **No.** Weights can be extracted from TAO and used for training/inference in a PyTorch environment |
| Are there constraints that prevent us from achieving our goals? | Some constraints exist when training within TAO (e.g., CoordConv). No constraints in a PyTorch environment |
| How can we run it outside of TAO? | Extract weights → Train in PyTorch → Deploy via ONNX/TensorRT conversion |
| Is there a risk that the model architecture is TAO-proprietary and incompatible? | **Low.** The architecture is considered identical to the official Meta/FAIR implementation |

---

## 2. Conclusion: Comparison of Feasible Approaches

**All gap mitigation measures can be implemented while leveraging TAO's pretrained weights.** The recommended approach is Approach (3): "Extract TAO weights → Train and infer in a PyTorch environment."

### List of Approaches

| | Approach | Overview |
|---|----------|----------|
| **(1)** | TAO training → ONNX export → Inference outside TAO | Train with TAO, export to ONNX, and run inference with ONNX Runtime or TensorRT |
| **(2)** | TAO training → Inference in TAO environment | Deploy TAO Docker environment to production and use TAO end-to-end |
| **(3)** | Extract TAO weights → Train and infer in PyTorch | Use only TAO's pretrained weights and build the training pipeline in PyTorch |

**Key Point**: Approaches (1) and (2) use TAO's training pipeline = development within TAO's constraints. Approach (3) borrows only the weights from TAO and allows free development.

### Approach Comparison Summary

| Aspect | (1) TAO Training → ONNX Inference | (2) TAO Training → TAO Inference | (3) TAO Weights → PyTorch ★Recommended |
|--------|:-:|:-:|:-:|
| Confirmed measure implementation rate | 75% (9/12) | 75% (9/12) | **100%** (12/12) |
| Production TAO dependency | None | **Yes** | None |
| Customization flexibility | Low | Low | **High** |
| Commercial license | ✅ | ✅ | ✅ |
| Initial setup cost | Low | Low | Medium |
| Inference speed optimization | ◎ (TensorRT) | ○ | ◎ (TensorRT capable) |
| Ease of production deployment | ◎ | △ | ◎ |

*Note: The basis for the measure implementation rate is described in "3-2. TAO Implementation Feasibility of Gap Mitigation Measures." Two items requiring verification are excluded from the calculation.*

---

## 3. Details

### 3-1. NVIDIA TAO Toolkit vs. Standard PyTorch Approach Comparison

| Item | TAO Toolkit | Standard PyTorch (timm, etc.) |
|------|:-----------:|:----------------------------:|
| **Training pipeline** | Executed via TAO-specific commands (`tao model classification_pyt train`). Controlled by YAML configuration files | Freely built with Python scripts |
| **Model architecture** | ConvNeXt-V2 architecture is considered identical to the official Meta/FAIR version (7×7 Depthwise Conv, LayerNorm, GRN, 4-stage hierarchy). No TAO-specific architectural modifications are believed to exist (detailed below) | Same architecture |
| **Pretraining data** | NVImageNet (NVIDIA proprietary dataset, **commercially licensable**) | ImageNet-1K/22K (CC-BY-NC, **non-commercial**) |
| **Customization scope** | Within the range of parameters exposed via YAML configuration | Fully flexible at the code level |
| **Input channels** | 1ch or 3ch only (**4ch or more not supported**) | Arbitrary (configurable via `in_chans` parameter) |
| **Loss function** | CrossEntropyLoss / LabelSmoothLoss. class_weight may be configurable (requires verification) | Fully flexible (custom loss functions possible) |
| **Augmentation** | Predefined options (MixUp, CutMix, RandAugment, Color Jitter, etc.) | Fully flexible (Albumentations, etc. also available) |
| **Weight file format** | `.pth` (standard PyTorch format) → Exportable to ONNX/TensorRT | `.pth` (same format) |
| **Inference environment** | TAO Docker environment, or convert to ONNX/TensorRT for standalone inference | PyTorch / ONNX Runtime / TensorRT all supported |
| **License** | Commercially licensable. However, **requires use on NVIDIA GPU-equipped environments** | Meta official weights are CC-BY-NC (non-commercial) |
| **Runtime environment** | NVIDIA GPU + TAO Docker required | NVIDIA GPU recommended but not required |

#### Model Architecture Compatibility

TAO's ConvNeXt-V2 implementation contains explicit references in its source code docstrings to the official Meta/FAIR paper (arXiv:2301.00808) and the GitHub repository (facebookresearch/ConvNeXt-V2), confirming it is **an implementation based on the official architecture**.

##### Confirmed Architecture Matches

| Component | Meta/FAIR Official | TAO Implementation |
|-----------|:-----------------:|:-----------------:|
| Block structure (DWConv→LN→FC→GELU→GRN→FC) | ✓ | ✓ |
| GlobalResponseNorm (GRN) | ✓ | ✓ |
| LayerNorm2d (channels_first) | ✓ | ✓ |
| Stem (4x4 Conv + LN) | ✓ | ✓ |
| DropPath (Stochastic Depth) | ✓ | ✓ |
| Model size configuration (depths/dims) | ✓ | ✓ |

Since all the above components match, **the TAO version's architecture is considered identical to the Meta/FAIR official version**. However, TAO uses its own framework (backbone_v2), and the presence of TAO-specific wrapper layers has not been verified. Within the scope of our investigation, no TAO-specific NAS modifications or architecture changes have been identified.

The only confirmed difference is the dataset used for pretraining (NVImageNet vs. ImageNet).

##### Technical Feasibility of Weight Extraction

Extracting TAO weights and loading them into a PyTorch environment (timm, etc.) is considered technically feasible. However, since there may be differences between TAO's internal key names and those in the timm/facebookresearch official implementation, key name remapping and numerical consistency verification will be required at load time. Since the architectures are considered identical, remapping is expected to be mechanically straightforward.

---

### 3-2. TAO Implementation Feasibility of Gap Mitigation Measures

#### Confirmed Measures (✅ Adopted)

| # | Measure | Implementable within TAO? | Implementation Method / Notes |
|---|---------|:-:|------|
| 2-1 | FDA (Fourier Domain Adaptation) | ✅ | Implemented as preprocessing before TAO input. Model-independent |
| 3-1 | Histogram normalization | ✅ | Implemented as preprocessing before TAO input. Model-independent |
| 3-5 | Background masking | ✅ | Implemented as preprocessing before TAO input. Model-independent |
| 3-7 | CoordConv (concatenate patch coordinates to head layer) | **❌ Not possible** | TAO is believed to have no hook points for inserting custom processing between Backbone and Head. 4ch input layer is also not possible |
| 4-1 | MixUp | ✅ | Natively supported via `train_cfg.augments: BatchMixup` |
| 4-2 | CutMix | ✅ | Natively supported via `train_cfg.augments: BatchCutMix` |
| 4-4 | RandAugment | ✅ | Natively supported via `random_aug.enable: True` |
| 4-5 | Color Jitter | ✅ | Natively supported via `random_color` configuration |
| 4-7 | Enhanced Dropout | ⚠️ Requires verification | ConvNeXt-V2 uses DropPath (Stochastic Depth) instead of traditional Dropout. Although the mechanism differs, the purpose is the same — "preventing overfitting." DropPath achieves regularization by randomly skipping layers during training. The intensity may be configurable via `custom_args.drop_path_rate` in TAO |
| 5-1 | Frozen Backbone | ✅ | Natively supported via `freeze_backbone: True` |
| 5-7 | Class weight in loss function | ⚠️ Requires verification | The `class_weight` parameter exists in MMPretrain's CrossEntropyLoss, but runtime verification is needed to confirm it is correctly passed through TAO's YAML parser |
| 6-1 | Threshold tuning | ✅ | Post-inference processing, implemented outside TAO. Model-independent |

#### Backup Measures (△ Conditionally Adopted)

Backup measures are alternative candidates in case confirmed measure 2-1 (FDA) does not achieve the expected results.

| # | Measure | Implementable within TAO? | Notes |
|---|---------|:-:|------|
| 2-7 | Grayscale conversion | ✅ | Implemented as preprocessing |
| 3-4 | Grayscale conversion | ✅ | Implemented as preprocessing |
| 3-6 | Edge intensity image conversion | **❌ Not possible** | Intended to be used in conjunction with 5-8 multi-channel input; not feasible due to TAO's input channel limitation (3ch max) |
| 5-4 | LayerNorm adaptation (train only γ/β) | **❌ Not possible** | TAO's `freeze_backbone` is all-or-nothing; per-parameter freeze control is not believed to be available |
| 5-8 | Multi-channel input | **❌ Not possible** | TAO limits input channels to 1ch or 3ch. 4ch or more is not officially supported |

#### Summary

**Confirmed Measures (12 items):**

- **Implementable within TAO**: **9 items** (including preprocessing and natively supported augmentation)
- **Not implementable within TAO**: **1 item** (CoordConv)
- **Requires verification**: **2 items** (Enhanced Dropout, Class Weight in loss)

**Backup Measures (5 items):**

- **Implementable within TAO**: **2 items** (Grayscale conversion ×2)
- **Not implementable within TAO**: **3 items** (Edge intensity image, LayerNorm adaptation, Multi-channel input)

#### Patch-Based Inference

TAO does not include sliding window inference functionality. TAO inference operates on a per-image basis: "1 image → 1 classification result." Overlapping patch division should be implemented outside TAO as follows:

1. Use a preprocessing script to crop the original image with overlap (e.g., 224×224) and save as individual images
2. Feed individual images into TAO (for both training and inference)
3. Use a post-processing script to aggregate inference results from each patch

---

### 3-3. Detailed Description of Each Approach

#### (1) TAO Training → ONNX Export → Inference Outside TAO

```
[TAO Docker] Preprocessing script → TAO train → TAO export (.onnx) → [Production] Inference with ONNX Runtime or TensorRT
```

| Item | Details |
|------|---------|
| **Advantages** | Full utilization of TAO's training capabilities; TAO not required in production; ONNX/TensorRT is lightweight and fast |
| **Disadvantages** | Training customization is constrained by TAO limitations (CoordConv not possible) |
| **Production requirements** | ONNX Runtime (pip install) or TensorRT Runtime + NVIDIA GPU |
| **Confirmed measure implementation rate** | **75%** (9/12) *2 items requiring verification excluded* |

#### (2) TAO Training → Inference in TAO Environment

```
[TAO Docker] Preprocessing script → TAO train → [Production also TAO Docker] TAO inference
```

| Item | Details |
|------|---------|
| **Advantages** | Consistent environment from training to inference; no conversion step required |
| **Disadvantages** | TAO Docker environment required in production (large footprint, NVIDIA GPU required); system integration through Docker is cumbersome; same measure constraints as (1) |
| **Production requirements** | TAO Docker + NVIDIA GPU |
| **Confirmed measure implementation rate** | **75%** (9/12) *2 items requiring verification excluded* |

#### (3) Extract TAO Weights → Train and Infer in PyTorch

```
[Development env] Extract backbone from TAO weights → Train with PyTorch scripts → Infer with .pth or ONNX
```

| Item | Details |
|------|---------|
| **Advantages** | All measures can be implemented; fully customizable at the code level; flexible training while leveraging TAO's commercially licensed weights |
| **Disadvantages** | Initial work required for weight extraction and remapping (considered technically straightforward); training script development cost |
| **Production requirements** | PyTorch / ONNX Runtime / TensorRT (any) |
| **Confirmed measure implementation rate** | **100%** (12/12) |

---

### 3-4. Detailed Implementation of Approach (3)

#### TAO Weight Extraction Procedure

Since TAO PyTorch backend checkpoints are in standard `.pth` format (PyTorch `state_dict`), weights can be extracted with code similar to the following:

```python
import torch

# Load TAO checkpoint
tao_ckpt = torch.load('tao_convnextv2_large.pth', map_location='cpu')
state_dict = tao_ckpt['state_dict']  # or tao_ckpt['model']

# Extract backbone only (remap key name prefixes)
backbone_weights = {}
for k, v in state_dict.items():
    if k.startswith('backbone.'):
        new_key = k.replace('backbone.', '')
        backbone_weights[new_key] = v

# Load into PyTorch ConvNeXt-V2 model
model.load_state_dict(backbone_weights, strict=False)
```

**Note**: Since there may be discrepancies between TAO's internal key names and those in timm / facebookresearch official implementations, key name mapping (remapping) and numerical consistency verification must be performed at load time. Since the architectures are considered identical, remapping is expected to be mechanically straightforward; however, the actual key structure needs to be confirmed by loading the `.pth` file.

#### Difficulty of Modifying facebookresearch/ConvNeXt-V2 Scripts

GitHub: https://github.com/facebookresearch/ConvNeXt-V2

**Conclusion: The modifications are not considered particularly difficult. Only the following 3 changes are required.**

##### Repository Structure

```
ConvNeXt-V2/
├── models/
│   └── convnextv2.py          # Model definition (modification target)
├── main_finetune.py            # Fine-tuning script (modification target)
├── engine_finetune.py          # Training loop
├── datasets.py                 # Data loader (modification target)
├── optim_factory.py            # Optimizer
└── TRAINING.md                 # Training recipe
```

##### Modification 1: Model Definition (models/convnextv2.py)

The ConvNeXt-V2 class accepts `in_chans` (number of input channels), `num_classes` (number of classes), and `drop_path_rate` as constructor arguments, so switching to 2-class classification can be done simply by specifying arguments.

```python
# Existing definition (no changes needed)
class ConvNeXtV2(nn.Module):
    def __init__(self, in_chans=3, num_classes=1000,
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768],
                 drop_path_rate=0., head_init_scale=1.):
```

To add CoordConv or multi-channel input, edit this file directly and add processing to the forward() method. This is standard PyTorch model modification without the black-box constraints of TAO.

##### Modification 2: Fine-tuning Script (main_finetune.py)

The existing script targets ImageNet, but switching to 2-class classification is expected to be possible simply by changing the following command-line arguments:

```bash
python main_finetune.py \
  --model convnextv2_large \
  --nb_classes 2 \                    # Change to 2 classes (OK/NG)
  --drop_path 0.1 \                   # DropPath rate setting
  --finetune /path/to/tao_weights.pth \  # Specify TAO weights
  --data_path /path/to/dataset \      # ImageFolder format data
  --input_size 224 \
  --batch_size 32 \
  --epochs 100 \
  --mixup 0.8 \                       # MixUp (built-in)
  --cutmix 1.0 \                      # CutMix (built-in)
  --smoothing 0.1                     # Label Smoothing (built-in)
```

The checkpoint loading logic already includes a mechanism to automatically detect and exclude head layer size mismatches (1000 classes → 2 classes).

##### Modification 3: Data Loader / Preprocessing Addition

FDA and histogram normalization can be added as custom Transforms in `datasets.py`, or pre-processed images can be generated offline and placed in ImageFolder format. The dataset structure follows the standard ImageNet format:

```
dataset/
├── train/
│   ├── ok/        # Normal images
│   └── ng/        # Anomalous images
└── val/
    ├── ok/
    └── ng/
```

##### Measures Requiring Additional Modifications

| Measure | Modification Details | Difficulty |
|---------|---------------------|:----------:|
| Class Weight in loss | Change `criterion` in `engine_finetune.py` to `nn.CrossEntropyLoss(weight=tensor([1.0, 3.0]))` | Low (1 line) |
| CoordConv (Head concatenation) | Add coordinate feature concatenation to `forward_head()` in `convnextv2.py` | Medium (10-20 lines) |
| Multi-channel input | Change `in_chans` in the stem layer of `convnextv2.py` | Low (1 line) |
| Frozen Backbone | Add backbone parameter freeze logic to `main_finetune.py` | Low (5 lines) |
| Staged Fine-tuning | Run the training script twice (1st: synthetic + real images, 2nd: real images only) | Low (configuration change only) |

##### Environment Setup Notes

The INSTALL.md in the facebookresearch/ConvNeXt-V2 repository mentions installing MinkowskiEngine (a sparse convolution library) and apex, but **these are required only for FCMAE pretraining and are not needed when performing fine-tuning only**. The only requirements are:

```bash
pip install torch torchvision timm tensorboardX
```

#### Alternative Method Using timm (Simpler)

Instead of the facebookresearch scripts, using the timm library allows for an even simpler implementation. timm natively supports ConvNeXt-V2 Large.

```python
import timm

# Create model (2 classes, DropPath rate 0.1)
model = timm.create_model(
    'convnextv2_large',
    pretrained=False,
    num_classes=2,
    drop_path_rate=0.1
)

# Load TAO weights (key name remapping may be required)
model.load_state_dict(tao_backbone_weights, strict=False)
```

When using timm, the training loop must be written manually, but this can be handled with a standard PyTorch training loop (or PyTorch Lightning, etc.). timm also includes a train.py script that allows training to be executed directly from the command line.

#### Recommended Implementation Path

Considering ease of implementation, the following order of approach is considered practical:

1. **Start by modifying `main_finetune.py` from facebookresearch/ConvNeXt-V2**
   - Built-in recipes for MixUp/CutMix/DropPath/EMA are already in place
   - Basic operation with `--nb_classes 2 --finetune /path/to/tao_weights.pth`
   - Additional modifications like Class Weight require only a few lines of changes in `engine_finetune.py`

2. **Switch to timm-based approach if customization becomes complex**
   - timm is more flexible for modifications involving model architecture, such as CoordConv and multi-channel input

---

## 4. Recommendation

### Recommended Approach

**We recommend Approach (3): "Extract TAO weights → Train and infer in a PyTorch environment."**

### Rationale

**(a) Highest measure implementation rate**

With Approaches (1) and (2), CoordConv (a confirmed measure) cannot be implemented due to TAO constraints, limiting the confirmed measure implementation rate to 75% (9 out of 12). Furthermore, 3 out of 5 backup measures (edge intensity image, multi-channel input, LayerNorm adaptation) are also not feasible within TAO, severely restricting alternative options if FDA does not achieve the expected results.

With Approach (3), all confirmed and backup measures (17 items total) can be implemented, providing flexibility for future measure additions.

**(b) Limited initial setup cost**

The additional cost of Approach (3) is TAO weight key name remapping (one-time effort) and training script development. Using the official facebookresearch/ConvNeXt-V2 scripts as a base, only 3 main modifications are required (model definition, fine-tuning script, data loader), all within the scope of standard PyTorch development.

**(c) Leverages commercially licensed weights**

By extracting TAO's pretrained weights (trained on NVImageNet, commercially licensable) as the backbone and fine-tuning in a PyTorch environment, the CC-BY-NC license restriction of Meta's official ImageNet weights can be avoided.

**(d) High flexibility in inference environment**

The trained model can be converted to ONNX/TensorRT, eliminating the need to deploy TAO Docker in the production environment. The inference pipeline can also be freely built with Python scripts.

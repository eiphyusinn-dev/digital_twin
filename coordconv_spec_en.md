# CoordConv Implementation Specification

## Table of Contents

1. [What is CoordConv](#1-what-is-coordconv)
2. [Implementation Approach for This Project](#2-implementation-approach-for-this-project)
3. [Architecture Changes Overview](#3-architecture-changes-overview)
4. [Detailed Changes and Code Examples per File](#4-detailed-changes-and-code-examples-per-file)
5. [Prerequisites and Notes](#5-prerequisites-and-notes)
6. [Verification Steps](#6-verification-steps)

---

## 1. What is CoordConv

### Background: Spatial Information Lost During Patch Splitting

In this pipeline, a large workpiece image is split into **224x224 pixel patches**,
and each patch is classified as either "OK (normal)" or "NG (defective)".

However, once the image is split into patches, **"where on the workpiece this patch is looking at"**
is lost. The model cannot distinguish between a corner patch and a center patch.

Examples where this becomes problematic:
- **Corners and edges** of the workpiece are prone to false positives due to lighting reflections and shape effects
- When **defect patterns tend to appear at specific positions**, spatial information can serve as a useful cue for classification

### The CoordConv Concept

**CoordConv** is a technique that explicitly tells the model "where it is currently looking."

The original paper (Uber, 2018: "An Intriguing Failing of Convolutional Neural Networks and the CoordConv Solution")
proposed adding X/Y coordinate gradient images as additional channels to the input image.

In this project, instead of adding input channels,
we adopt an approach of **concatenating patch coordinates to the Backbone output feature vector before passing it to the Head** (details in the next section).

---

## 2. Implementation Approach for This Project

### Comparison of Approaches

| Approach | Description | Pros | Cons |
|----------|-------------|------|------|
| **A. Multi-channel input** | Add coordinate channels to input image (3ch → 5ch) | Provides spatially fine-grained coordinate information | Requires changes to Backbone input layer. Breaks compatibility with pretrained weights |
| **B. Feature vector concatenation (adopted)** | Concatenate 2D coordinates to the 1536-dim Backbone output vector | No changes to Backbone (frozen part). Simple implementation | Only patch-level coarse position (not pixel-level) |

### Rationale

- Currently `freeze_backbone: True` — the Backbone is frozen (weights are not updated)
- Approach A would require changing the input layer (3ch → 5ch), breaking compatibility with TAO pretrained weights
- Approach B keeps the **Backbone completely unchanged**, only modifying the Head input dimension
- Patch-level "approximate location" is sufficient; pixel-level precision is unnecessary

### Coordinate Representation

Patch positions are represented as **normalized 2D coordinates (row_norm, col_norm)**.

```
row_norm = patch row index / (total rows - 1)    → 0.0 (top) to 1.0 (bottom)
col_norm = patch column index / (total cols - 1)  → 0.0 (left) to 1.0 (right)
```

Example: 5 rows x 8 columns grid

```
(0.00, 0.00)  (0.00, 0.14)  ...  (0.00, 1.00)   ← top row
(0.25, 0.00)  (0.25, 0.14)  ...  (0.25, 1.00)
(0.50, 0.00)  (0.50, 0.14)  ...  (0.50, 1.00)   ← center
(0.75, 0.00)  (0.75, 0.14)  ...  (0.75, 1.00)
(1.00, 0.00)  (1.00, 0.14)  ...  (1.00, 1.00)   ← bottom row
```

---

## 3. Architecture Changes Overview

### Before (Current)

```
Input image (batch, 3, 224, 224)
    │
    ▼
┌──────────────────────────────┐
│  Backbone (forward_features)     │  ← Frozen (freeze_backbone=True)
│  - Stem + Stage 0-3              │
│  - Global Average Pooling        │
│  - LayerNorm                     │
│  Output: (batch, 1536)           │
└──────────────────────────────┘
    │
    ▼
┌──────────────────────────────┐
│  Head: Linear(1536 → 2)         │  ← Trainable
│  Output: (batch, 2) = [NG prob, OK prob] │
└──────────────────────────────┘
```

### After (With CoordConv)

```
Input image (batch, 3, 224, 224)        Patch coords (batch, 2)
    │                                       │  ← [row_norm, col_norm]
    ▼                                       │
┌──────────────────────────────┐            │
│  Backbone (forward_features)     │  ← Frozen (no changes)  │
│  Output: (batch, 1536)           │            │
└──────────────────────────────┘            │
    │                                       │
    ▼                                       ▼
┌──────────────────────────────────────────────┐
│  torch.cat([features, coords], dim=1)            │
│  → (batch, 1536 + 2) = (batch, 1538)            │
└──────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────┐
│  Head: Linear(1538 → 2)         │  ← Trainable (input dim changed)
│  Output: (batch, 2) = [NG prob, OK prob] │
└──────────────────────────────┘
```

**Key points**:
- Backbone remains completely identical (stays frozen)
- Only the Head input dimension changes **1536 → 1538**
- Coordinate tensors need to be passed through Dataset → DataLoader → Model

---

## 4. Detailed Changes and Code Examples per File

### 4-1. config.yaml

Add CoordConv-related settings.

```yaml
# --- New section ---
# CoordConv Configuration
coordconv:
  enabled: false          # Set to true to enable CoordConv
  grid_rows: 5            # Total rows in patch grid (match patch splitting settings)
  grid_cols: 8            # Total columns in patch grid (match patch splitting settings)
```

> **How to determine grid_rows / grid_cols**:
> Match these to the grid dimensions generated by the patch splitting script (`patch_split_bbox.py`).
> They can be calculated from original image size, patch_size, and stride:
> ```
> grid_rows = len(range(0, H - patch_size + 1, stride)) + (possibly +1 for edge correction)
> grid_cols = len(range(0, W - patch_size + 1, stride)) + (possibly +1 for edge correction)
> ```
> The most reliable approach is to check the maximum r/c values in the actual patch filenames.

---

### 4-2. model.py

#### Summary of Changes

- `ConvNeXtV2.__init__()`: Add `use_coordconv` parameter. When True, increase Head input dimension by +2
- `ConvNeXtV2.forward()`: Accept coordinate tensor, concatenate with Backbone output before passing to Head
- `create_model()`: Accept `use_coordconv`

#### Code Example

```python
# === Changes to ConvNeXtV2 class ===

class ConvNeXtV2(nn.Module):
    def __init__(self, in_chans=3, num_classes=1000,
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768],
                 drop_path_rate=0., head_init_scale=1.,
                 use_coordconv=False          # ★ New parameter
                 ):
        super().__init__()
        self.depths = depths
        self.use_coordconv = use_coordconv    # ★ Store flag

        # ... (downsample_layers, stages, norm definitions remain unchanged) ...

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)

        # ★ Conditionally set Head input dimension
        head_in_dim = dims[-1] + 2 if use_coordconv else dims[-1]
        self.head = nn.Linear(head_in_dim, num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    # forward_features remains unchanged
    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1]))  # (batch, 1536)

    # ★ Add coords parameter to forward signature
    def forward(self, x, coords=None):
        x = self.forward_features(x)         # (batch, 1536)

        if self.use_coordconv and coords is not None:
            # coords shape: (batch, 2)
            x = torch.cat([x, coords], dim=1)  # (batch, 1538)

        x = self.head(x)                     # (batch, 2)
        return x
```

#### Changes to create_model

```python
def convnextv2_large(**kwargs):
    model = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    return model

def create_model(num_classes: int = 1000, **kwargs) -> ConvNeXtV2:
    """Create ConvNeXtV2 model with specified parameters."""
    supported_kwargs = {}
    for key, value in kwargs.items():
        if key in ['in_chans', 'drop_path_rate', 'head_init_scale',
                    'use_coordconv']:        # ★ Add use_coordconv
            supported_kwargs[key] = value

    return convnextv2_large(num_classes=num_classes, **supported_kwargs)
```

#### Note on TAO Weight Loading

When CoordConv is enabled, the Head weight shape changes:
- Normal: `head.weight` shape = `(2, 1536)`
- CoordConv: `head.weight` shape = `(2, 1538)`

The existing `load_tao_weights()` skips keys with shape mismatches (model.py:175-180),
so **Head weights will be automatically skipped and remain randomly initialized for training**.
This is consistent with the `freeze_backbone: True` (train Head only) approach, so no issues arise.

---

### 4-3. dataset.py

#### Summary of Changes

- `ClientCustomDataset.__init__()`: Load CoordConv settings, store grid size
- `ClientCustomDataset.__getitem__()`: Parse row/col from filename, return normalized coordinates
- Return type changes from `(image, label)` to `(image, label, coords)`

#### Coordinate Parsing from Filenames

The patch splitting script (`patch_split_bbox.py:131`) saves files in the following format:

```
{original_name}_r{row_index}_c{col_index}_{label}.png
Example: cg_1_7000_0_r2_c3_NG.png → row=2, col=3
```

Row/col can be extracted using the regex `_r(\d+)_c(\d+)_`.

#### Code Example

```python
import re  # ★ Add to file imports

class ClientCustomDataset(Dataset):
    def __init__(self,
                 split_path: Path,
                 root_dir: Path,
                 config: Dict,
                 transform: Optional[Callable] = None,
                 class_names: Optional[List[str]] = None):
        # ... (existing initialization remains unchanged) ...

        # ★ Load CoordConv settings
        self.coordconv_cfg = config.get('coordconv', {})
        self.use_coordconv = self.coordconv_cfg.get('enabled', False)
        if self.use_coordconv:
            self.grid_rows = self.coordconv_cfg.get('grid_rows', 1)
            self.grid_cols = self.coordconv_cfg.get('grid_cols', 1)

        # ... (remaining existing initialization) ...

    def _parse_patch_coords(self, filepath: str) -> Tuple[float, float]:
        """
        Extract patch grid coordinates from filename and normalize to 0-1.

        Filename format: {name}_r{row}_c{col}_{label}.png
        Example: cg_1_7000_0_r2_c3_NG.png → row=2, col=3

        Returns:
            (row_norm, col_norm): Each value in range 0.0 to 1.0
        """
        filename = Path(filepath).stem  # Filename without extension
        match = re.search(r'_r(\d+)_c(\d+)_', filename)

        if match:
            row = int(match.group(1))
            col = int(match.group(2))
            # Normalize to 0-1 (avoid division by zero when grid_rows/cols is 1)
            row_norm = row / max(self.grid_rows - 1, 1)
            col_norm = col / max(self.grid_cols - 1, 1)
            return (row_norm, col_norm)
        else:
            # Default to center when coordinates not found in filename
            return (0.5, 0.5)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # ... (existing background masking and FDA processing remain unchanged) ...

        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']

        # ★ CoordConv: Return coordinate tensor as additional output
        if self.use_coordconv:
            row_norm, col_norm = self._parse_patch_coords(img_path)
            coords = torch.tensor([row_norm, col_norm], dtype=torch.float32)
            return image, label, coords

        return image, label
```

> **Note on default value (0.5, 0.5)**:
>
> If images with filenames that don't contain grid coordinates (e.g., older data formats) are mixed in,
> a center value of (0.5, 0.5) is returned to prevent errors.
> However, CoordConv will have no effect for such images.
> Ensure all training images have `_r{row}_c{col}_` in their filenames before enabling CoordConv.

---

### 4-4. train.py

#### Summary of Changes

The following 3 areas need to be modified:

1. **`main()`**: Pass `use_coordconv` during model creation. Backbone freeze logic adjustment
2. **`train_epoch()`**: Extract coordinates from DataLoader, pass to model
3. **`validate()`**: Same as above

#### Change 1: main() — Model Creation

```python
def main():
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # ★ Pass use_coordconv
    model = create_model(
        model_name=config['model']['name'],
        num_classes=config['classes']['num_classes'],
        in_chans=config['model']['in_chans'],
        drop_path_rate=config['model']['drop_path_rate'],
        use_coordconv=config.get('coordconv', {}).get('enabled', False)  # ★ Added
    )

    # ... (TAO weight loading remains unchanged) ...

    # ★ Backbone freeze logic: do not freeze head parameters
    if config['training'].get('freeze_backbone'):
        for name, param in model.named_parameters():
            if 'head' not in name:
                param.requires_grad = False
    # ↑ Existing code works as-is ('head' in name still matches even with changed input dim)

    # ... (rest remains unchanged) ...
```

#### Change 2: train_epoch()

```python
def train_epoch(self) -> Dict[str, float]:
    self.model.train()
    running_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")

    # ★ Check if CoordConv is enabled
    use_coordconv = self.config.get('coordconv', {}).get('enabled', False)

    for batch in pbar:
        # ★ Branch DataLoader unpacking
        if use_coordconv:
            data, targets, coords = batch
            data = data.to(self.device)
            targets = targets.to(self.device)
            coords = coords.to(self.device)       # (batch, 2)
        else:
            data, targets = batch
            data = data.to(self.device)
            targets = targets.to(self.device)
            coords = None

        self.optimizer.zero_grad()

        # ... (MixUp/CutMix application remains unchanged) ...

        if self.scaler:
            with autocast('cuda'):
                outputs = self.model(data, coords=coords)  # ★ Pass coords
                # ... (loss computation remains unchanged) ...
        else:
            outputs = self.model(data, coords=coords)       # ★ Pass coords
            # ... (loss computation remains unchanged) ...

        # ... (rest remains unchanged) ...
```

#### Change 3: validate()

```python
def validate(self) -> Dict[str, float]:
    self.model.eval()
    running_loss, correct, total = 0.0, 0, 0

    # ★ Check if CoordConv is enabled
    use_coordconv = self.config.get('coordconv', {}).get('enabled', False)

    with torch.no_grad():
        for batch in self.val_loader:
            # ★ Branch DataLoader unpacking
            if use_coordconv:
                data, targets, coords = batch
                coords = coords.to(self.device)
            else:
                data, targets = batch
                coords = None

            data, targets = data.to(self.device), targets.to(self.device)
            outputs = self.model(data, coords=coords)    # ★ Pass coords

            # ... (rest remains unchanged) ...
```

#### Note on MixUp/CutMix Compatibility

MixUp shuffles and blends images within a batch.
**Coordinates must also be blended with the same shuffle order and ratio**.

```python
def apply_mixup(self, data, targets, coords=None):
    """Apply MixUp augmentation to batch."""
    # ... (existing lam computation unchanged) ...

    index = torch.randperm(data.size(0)).to(self.device)
    mixed_data = lam * data + (1 - lam) * data[index]
    targets_a, targets_b = targets, targets[index]

    # ★ Blend coordinates with the same ratio
    mixed_coords = None
    if coords is not None:
        mixed_coords = lam * coords + (1 - lam) * coords[index]

    return mixed_data, targets_a, targets_b, lam, mixed_coords  # ★ Return coords too
```

For CutMix, coordinates also need blending based on the cut area ratio.
Since coordinates are patch-level values (only 2 dimensions),
**weighted averaging by cut area ratio** is appropriate:

```python
def apply_cutmix(self, data, targets, coords=None):
    # ... (existing bbox computation unchanged) ...

    data[:, :, bbx1:bbx2, bby1:bby2] = data[index, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))
    targets_a, targets_b = targets, targets[index]

    # ★ Blend coordinates by area ratio
    mixed_coords = None
    if coords is not None:
        mixed_coords = lam * coords + (1 - lam) * coords[index]

    return data, targets_a, targets_b, lam, mixed_coords  # ★ Return coords too
```

---

### 4-5. inference_by_patch.py

#### Summary of Changes

- `InferenceEngine.__init__()`: Load CoordConv settings
- `InferenceEngine._load_model()`: Pass `use_coordconv` to model creation
- `InferenceEngine.predict_full_image()`: Normalize patch (x, y) pixel coordinates and pass to model

#### Coordinate Normalization (Inference)

During inference, patches are split dynamically, so normalization uses **pixel coordinates instead of filenames**.

```
row_norm = y / max(H - patch_size, 1)    # y: patch top-left y coordinate, H: original image height
col_norm = x / max(W - patch_size, 1)    # x: patch top-left x coordinate, W: original image width
```

This is equivalent to training-time normalization:
- Training: `row / (grid_rows - 1)` → top row=0, bottom row=1
- Inference: `y / (H - patch_size)` → top (y=0)=0, bottom (y=H-224)=1

#### Code Example

> **Note**: The current `inference_by_patch.py` has been updated to accept a `full_config` parameter.
> The following code examples are aligned with the latest API.

```python
class InferenceEngine:
    def __init__(self,
                 model_path: str,
                 model_config: Dict,
                 full_config: Dict,           # ← Current API
                 device: Optional[str] = None,
                 threshold: float = 0.5,
                 patch_size: int = 224,
                 stride: int = 112):
        self.model_config = model_config
        self.full_config = full_config
        self.threshold = threshold
        self.patch_size = patch_size
        self.stride = stride

        # ★ Load CoordConv settings from full_config
        self.use_coordconv = full_config.get('coordconv', {}).get('enabled', False)

        # ... (device setup, background masking initialization, etc. remain unchanged) ...

    def _load_model(self, model_path: str) -> ConvNeXtV2:
        model = create_model(
            model_name=self.model_config.get('model_name', 'convnextv2_large'),
            num_classes=self.model_config.get('num_classes', 2),
            in_chans=self.model_config.get('in_chans', 3),
            use_coordconv=self.use_coordconv      # ★ Added
        )
        checkpoint = torch.load(model_path, map_location='cpu')
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        return model

    def predict_full_image(self, image_path: str, save_preview: bool = True) -> Dict:
        filename = os.path.basename(image_path)
        original_img_bgr = cv2.imread(image_path)
        if original_img_bgr is None:
            raise ValueError(f"Could not load {image_path}")

        img_rgb = cv2.cvtColor(original_img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]

        # ... (background masking processing remains unchanged) ...
        patches = self.get_patches(img_rgb)  # Varies depending on masking setting

        ng_patches = []
        max_ng_confidence = 0.0
        start_time = time.time()

        # ★ Pre-compute normalization denominators
        max_y = max(h - self.patch_size, 1)
        max_x = max(w - self.patch_size, 1)

        batch_size = 16
        for i in range(0, len(patches), batch_size):
            batch_data = patches[i:i+batch_size]
            batch_tensors = []
            batch_coords = []                     # ★ Coordinate batch

            for p in batch_data:
                transformed = self.transform(image=p['img'])['image']
                batch_tensors.append(transformed)

                # ★ Normalize coordinates
                if self.use_coordconv:
                    row_norm = p['y'] / max_y
                    col_norm = p['x'] / max_x
                    batch_coords.append([row_norm, col_norm])

            input_batch = torch.stack(batch_tensors).to(self.device)

            # ★ Create coordinate tensor
            coords = None
            if self.use_coordconv and batch_coords:
                coords = torch.tensor(batch_coords, dtype=torch.float32).to(self.device)

            with torch.no_grad():
                outputs = self.model(input_batch, coords=coords)  # ★ Pass coords
                probs = F.softmax(outputs, dim=1)

            # ... (result processing remains unchanged) ...
```

#### Changes to main()

The current `main()` passes the full config via `model_config=config, full_config=config`.
CoordConv settings are automatically loaded within `__init__()` via `full_config`,
so **no changes to main() are required**.

```python
def main():
    # ... (argparse remains unchanged) ...

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # ★ No changes needed: full_config=config means coordconv settings are loaded in __init__
    engine = InferenceEngine(
        model_path=args.model,
        model_config=config,
        full_config=config,
        device=device,
        threshold=args.threshold or config['inference']['threshold']
    )
    # ... (rest remains unchanged) ...
```

---

### Summary of File Changes

| File | What to Change | Nature of Change |
|------|----------------|------------------|
| `config.yaml` | Add `coordconv` section | Configuration |
| `model.py` | `ConvNeXtV2.__init__()`, `forward()`, `create_model()` | Coordinate concatenation, Head dim change |
| `dataset.py` | `__init__()`, `__getitem__()`, new `_parse_patch_coords()` | Coordinate parsing and return |
| `train.py` | `main()`, `train_epoch()`, `validate()`, `apply_mixup()`, `apply_cutmix()` | Coordinate passing |
| `inference_by_patch.py` | `__init__()`, `_load_model()`, `predict_full_image()` | Coordinate normalization and passing |
| `patch_split_bbox.py` | **No changes needed** | Filenames already contain `_r{row}_c{col}_` |

---

## 5. Prerequisites and Notes

### Prerequisites

1. **Training data filenames must contain `_r{row}_c{col}_`**
   - Automatically included when generated by `patch_split_bbox.py`
   - Example: `cg_1_7000_0_r2_c3_NG.png`
   - If not included, coordinates default to (0.5, 0.5) and CoordConv has no effect

2. **`grid_rows` / `grid_cols` in config.yaml must match the actual grid**
   - If mismatched, normalization will be incorrect (values may exceed the 0-1 range)

### Important Notes

1. **Checkpoints with CoordConv enabled are not compatible with CoordConv-disabled models**
   - Head weight shapes differ (1536x2 vs 1538x2)
   - Models trained with CoordConv must be inferred with CoordConv enabled

2. **When `enabled: false`, behavior is completely identical to existing pipeline**
   - Dataset returns `(image, label)` (no coordinates)
   - Model `forward()` operates normally with `coords=None`
   - Full compatibility with existing training results and checkpoints is maintained

3. **Coordinate blending when using MixUp/CutMix**
   - Coordinates must be blended with the same ratio as the data (see Section 4-4)
   - Blended coordinates represent the "weighted average of two patch positions"

---

## 6. Verification Steps

### Step 1: CoordConv OFF (Baseline Check)

```yaml
# config.yaml
coordconv:
  enabled: false
```

- Verify behavior is completely identical to existing pipeline
- Confirm the full train → inference pipeline works

### Step 2: CoordConv ON (Basic Functionality Check)

```yaml
# config.yaml
coordconv:
  enabled: true
  grid_rows: 5    # Match actual patch grid
  grid_cols: 8    # Match actual patch grid
```

Verification points:
- Confirm DataLoader returns `(image, label, coords)` (3 values)
- `coords` shape should be `(batch_size, 2)`
- `coords` values should be in the 0.0 to 1.0 range
- Model output shape should remain `(batch_size, 2)`

Example debug code:

```python
# Temporarily add at the beginning of train_epoch() for verification
for batch in self.train_loader:
    if use_coordconv:
        data, targets, coords = batch
        print(f"data.shape: {data.shape}")       # (8, 3, 224, 224)
        print(f"targets.shape: {targets.shape}")  # (8,)
        print(f"coords.shape: {coords.shape}")    # (8, 2)
        print(f"coords min: {coords.min()}, max: {coords.max()}")  # 0.0 to 1.0

        data = data.to(self.device)
        coords = coords.to(self.device)
        outputs = self.model(data, coords=coords)
        print(f"outputs.shape: {outputs.shape}")  # (8, 2)
    break  # Check only 1 batch
```

### Step 3: Effectiveness Evaluation

| Experiment | Setting | Purpose |
|------------|---------|---------|
| Baseline | `coordconv.enabled: false` | Performance without CoordConv |
| CoordConv | `coordconv.enabled: true` | Performance with CoordConv |

Comparison metrics: Recall / Precision (patch-level and workpiece-level)

---

## Appendix: Quick Reference for Changes

### One-line Summary of Changes per File

| File | Essence of Change |
|------|-------------------|
| config.yaml | Add `coordconv: {enabled, grid_rows, grid_cols}` |
| model.py | Change `head = Linear(1536→2)` to `Linear(1538→2)`, cat coords in forward() |
| dataset.py | Extract `_r{row}_c{col}_` from filename via regex, normalize to 0-1, and return |
| train.py | Extract coords from DataLoader, pass via `model(data, coords=coords)` |
| inference_by_patch.py | Normalize patch (x,y) pixel coords, pass via `model(batch, coords=coords)` |

# Random Patch Extraction Script — Modification Guide

> Target files: `patch_splitting_utils/random_patch_extractor.py`, `config.yaml`
> Modifications: (1) Fix preprocessing order, (2) FDA reference image coordinate matching

---

## (1) Change preprocessing order to FDA → Histogram Normalization

When both FDA and histogram normalization are applied, **FDA must be applied first, followed by histogram normalization**.
Currently the order is reversed (histogram → FDA). Please fix the following 2 locations.

### Fix ① `_apply_preprocessing_crop_first()` (L335-349)

**Before:**
```python
        # STEP 2: Apply histogram normalization to cropped product area only
        if self.hist_norm and self.hist_norm_mode == 'crop_first':
            try:
                cropped_image = self.hist_norm(image=cropped_image)['image']
                print("    Applied histogram normalization to cropped product area")
            except Exception as e:
                print(f"    Warning: Histogram normalization failed: {e}")

        # STEP 3: Apply FDA to cropped product area only
        if self.fda_transform and self.fda_mode == 'crop_first':
            try:
                cropped_image = self.fda_transform(image=cropped_image)['image']
                print("    Applied FDA to cropped product area")
            except Exception as e:
                print(f"    Warning: FDA failed: {e}")
```

**After:**
```python
        # STEP 2: Apply FDA to cropped product area only
        if self.fda_transform and self.fda_mode == 'crop_first':
            try:
                cropped_image = self.fda_transform(image=cropped_image)['image']
                print("    Applied FDA to cropped product area")
            except Exception as e:
                print(f"    Warning: FDA failed: {e}")

        # STEP 3: Apply histogram normalization to cropped product area only
        if self.hist_norm and self.hist_norm_mode == 'crop_first':
            try:
                cropped_image = self.hist_norm(image=cropped_image)['image']
                print("    Applied histogram normalization to cropped product area")
            except Exception as e:
                print(f"    Warning: Histogram normalization failed: {e}")
```

### Fix ② `_apply_preprocessing_patch_first()` (L370-388)

**Before:**
```python
    def _apply_preprocessing_patch_first(self, patch: np.ndarray) -> np.ndarray:
        """Apply preprocessing in patch_first mode: preprocess individual patch."""
        processed_patch = patch.copy()

        # Apply histogram normalization
        if self.hist_norm and self.hist_norm_mode == 'patch_first':
            try:
                processed_patch = self.hist_norm(image=processed_patch)['image']
            except Exception as e:
                print(f"    Warning: Histogram normalization failed: {e}")

        # Apply FDA
        if self.fda_transform and self.fda_mode == 'patch_first':
            try:
                processed_patch = self.fda_transform(image=processed_patch)['image']
            except Exception as e:
                print(f"    Warning: FDA failed: {e}")

        return processed_patch
```

**After:**
```python
    def _apply_preprocessing_patch_first(self, patch: np.ndarray, x: int = 0, y: int = 0) -> np.ndarray:
        """Apply preprocessing in patch_first mode: preprocess individual patch."""
        processed_patch = patch.copy()

        # Apply FDA (must be applied BEFORE histogram normalization)
        if self.fda_transform and self.fda_mode == 'patch_first':
            try:
                processed_patch = self.fda_transform(image=processed_patch)['image']
            except Exception as e:
                print(f"    Warning: FDA failed: {e}")

        # Apply histogram normalization
        if self.hist_norm and self.hist_norm_mode == 'patch_first':
            try:
                processed_patch = self.hist_norm(image=processed_patch)['image']
            except Exception as e:
                print(f"    Warning: Histogram normalization failed: {e}")

        return processed_patch
```

> **Note**: The `x`, `y` parameters added to `_apply_preprocessing_patch_first` are used in modification (2) below.

---

## (2) FDA Reference Image Coordinate Matching

### Background

FDA transfers the low-frequency components (color tone, brightness tendency) from a reference image (Real image) to a source image (CG image).
For correct FDA application, **CG and Real images must be paired by the same region** before applying FDA.

The correct approach is:
- **crop_first**: Apply the same mask → crop to the Real image, then apply FDA between the two cropped images (same size, same region)
- **patch_first**: Extract a patch from the Real image at the same coordinates, then apply FDA between the two patches

Currently, the implementation reuses a single fixed 224×224 patch (`reaLimage_cropped_patch/5_7000_25.png`) as the FDA reference for all CG images. This needs to be changed to use a **full-size Real image** and extract the appropriate region from it.

### config.yaml Change

Change `reference_path` to a full-size Real image path (no need for separate patch and full-size paths).

**Before:**
```yaml
    fda:
      enabled: false
      mode: "crop_first"
      beta_limit : 0.001
      reference_path: "reaLimage_cropped_patch/5_7000_25.png"
```

**After:**
```yaml
    fda:
      enabled: false
      mode: "crop_first"
      beta_limit: 0.001
      reference_path: "input/real_image/13_2_15.png"    # Full-size Real image for both modes
```

### Fix ③ `_init_preprocessing()` (L169-190)

The `FDATransform` class caches a single reference image at init time and resizes it to match the input size on every `apply()` call. This means it cannot accept different reference images for each patch.

To solve this, we replace `FDATransform` with loading the full-size Real reference image into memory and calling the `FDA_source_to_target_np` function directly, after extracting the appropriate region.

**Before:**
```python
        # FDA
        fda_config = preprocessing_config.get('fda', {})
        self.fda_transform = None
        self.fda_mode = fda_config.get('mode', 'crop_first')

        if fda_config.get('enabled', False):
            reference_path = fda_config.get('reference_path')
            if reference_path and os.path.exists(reference_path):
                try:
                    self.fda_transform = FDATransform(
                        reference_images_path=reference_path,
                        beta_limit=fda_config.get('beta_limit', 0.1),
                        always_apply=True
                    )
                    print(f"FDA: Enabled ({self.fda_mode}) with reference: {reference_path}")
                except Exception as e:
                    print(f"Warning: Could not initialize FDA: {e}")
                    self.fda_transform = None
            else:
                print(f"FDA enabled but reference path not found: {reference_path}, disabling FDA")
        else:
            print("FDA: Disabled")
```

**After:**
```python
        # FDA
        fda_config = preprocessing_config.get('fda', {})
        self.fda_enabled = False
        self.fda_mode = fda_config.get('mode', 'crop_first')
        self.fda_beta = fda_config.get('beta_limit', 0.001)
        self.fda_reference_fullsize = None

        if fda_config.get('enabled', False):
            reference_path = fda_config.get('reference_path')
            if reference_path and os.path.exists(reference_path):
                try:
                    ref_img = cv2.imread(reference_path)
                    self.fda_reference_fullsize = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
                    self.fda_enabled = True
                    print(f"FDA: Enabled ({self.fda_mode}) with reference: {reference_path} "
                          f"({ref_img.shape[1]}x{ref_img.shape[0]})")
                except Exception as e:
                    print(f"Warning: Could not load FDA reference: {e}")
            else:
                print(f"FDA enabled but reference path not found: {reference_path}, disabling FDA")
        else:
            print("FDA: Disabled")
```

Add the FDA function import near the top of the file:

```python
# Add near existing import lines
from utils.fda_utils import FDA_source_to_target_np
```

Also add the following helper method to the `RandomPatchExtractor` class:

```python
    def _apply_fda(self, src_image: np.ndarray, ref_image: np.ndarray) -> np.ndarray:
        """Apply FDA: transfer low-frequency style from ref_image to src_image.

        Both images must be RGB uint8 with the same dimensions.
        """
        src = src_image.astype(np.float32).transpose(2, 0, 1) / 255.0
        trg = ref_image.astype(np.float32).transpose(2, 0, 1) / 255.0
        result = FDA_source_to_target_np(src, trg, L=self.fda_beta)
        return np.clip(result.transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8)
```

### Fix ④ `_apply_preprocessing_crop_first()` (L335-349)

In crop_first mode, apply the **same mask → crop** to the Real reference image before FDA.

> Note: Combined with Fix ① (order swap), replace the entire STEP 2-3 section with the following.

**Before:** (STEP 2 and STEP 3)
```python
        # STEP 2: Apply histogram normalization to cropped product area only
        if self.hist_norm and self.hist_norm_mode == 'crop_first':
            try:
                cropped_image = self.hist_norm(image=cropped_image)['image']
                print("    Applied histogram normalization to cropped product area")
            except Exception as e:
                print(f"    Warning: Histogram normalization failed: {e}")

        # STEP 3: Apply FDA to cropped product area only
        if self.fda_transform and self.fda_mode == 'crop_first':
            try:
                cropped_image = self.fda_transform(image=cropped_image)['image']
                print("    Applied FDA to cropped product area")
            except Exception as e:
                print(f"    Warning: FDA failed: {e}")
```

**After:**
```python
        # STEP 2: Apply FDA with coordinate-matched Real reference
        if self.fda_enabled and self.fda_mode == 'crop_first' and self.fda_reference_fullsize is not None:
            try:
                # Apply same mask + crop to Real reference image
                ref_h, ref_w = self.fda_reference_fullsize.shape[:2]
                if self.bg_masking:
                    ref_mask = self.bg_masking.get_mask(ref_h, ref_w)
                    ref_masked = cv2.bitwise_and(
                        self.fda_reference_fullsize, self.fda_reference_fullsize, mask=ref_mask)
                    ref_cropped = ref_masked[y:y+h_crop, x:x+w_crop]
                else:
                    ref_cropped = self.fda_reference_fullsize[y:y+h_crop, x:x+w_crop]

                cropped_image = self._apply_fda(cropped_image, ref_cropped)
                print("    Applied FDA (crop-matched) to cropped product area")
            except Exception as e:
                print(f"    Warning: FDA failed: {e}")

        # STEP 3: Apply histogram normalization to cropped product area only
        if self.hist_norm and self.hist_norm_mode == 'crop_first':
            try:
                cropped_image = self.hist_norm(image=cropped_image)['image']
                print("    Applied histogram normalization to cropped product area")
            except Exception as e:
                print(f"    Warning: Histogram normalization failed: {e}")
```

> **Note**: `x, y, w_crop, h_crop` are variables already computed in STEP 1.

### Fix ⑤ `_apply_preprocessing_patch_first()` (L370-388)

In patch_first mode, extract a patch from the full-size Real image at the **same coordinates (x, y)** before FDA.

> Note: This is the final version combined with Fix ② (order swap + parameter addition).

**Before:**
```python
    def _apply_preprocessing_patch_first(self, patch: np.ndarray) -> np.ndarray:
        """Apply preprocessing in patch_first mode: preprocess individual patch."""
        processed_patch = patch.copy()

        # Apply histogram normalization
        if self.hist_norm and self.hist_norm_mode == 'patch_first':
            try:
                processed_patch = self.hist_norm(image=processed_patch)['image']
            except Exception as e:
                print(f"    Warning: Histogram normalization failed: {e}")

        # Apply FDA
        if self.fda_transform and self.fda_mode == 'patch_first':
            try:
                processed_patch = self.fda_transform(image=processed_patch)['image']
            except Exception as e:
                print(f"    Warning: FDA failed: {e}")

        return processed_patch
```

**After:**
```python
    def _apply_preprocessing_patch_first(self, patch: np.ndarray, x: int = 0, y: int = 0) -> np.ndarray:
        """Apply preprocessing in patch_first mode: preprocess individual patch.

        Args:
            patch: Source CG patch (RGB, uint8)
            x, y: Patch coordinates in the original full-size image (for FDA coordinate matching)
        """
        processed_patch = patch.copy()

        # Apply FDA with coordinate-matched Real reference patch
        if self.fda_enabled and self.fda_mode == 'patch_first' and self.fda_reference_fullsize is not None:
            try:
                # Extract Real patch at same coordinates
                ref_patch = self.fda_reference_fullsize[y:y+self.patch_size, x:x+self.patch_size]
                processed_patch = self._apply_fda(processed_patch, ref_patch)
            except Exception as e:
                print(f"    Warning: FDA failed: {e}")

        # Apply histogram normalization
        if self.hist_norm and self.hist_norm_mode == 'patch_first':
            try:
                processed_patch = self.hist_norm(image=processed_patch)['image']
            except Exception as e:
                print(f"    Warning: Histogram normalization failed: {e}")

        return processed_patch
```

### Fix ⑥ Pass coordinates to callers (2 locations)

Pass the patch extraction coordinates to `_apply_preprocessing_patch_first`.

**`generate_ok_patches()` L462-463:**

Before:
```python
            if self.hist_norm_mode == 'patch_first' or self.fda_mode == 'patch_first':
                patch = self._apply_preprocessing_patch_first(patch)
```

After:
```python
            if self.hist_norm_mode == 'patch_first' or self.fda_mode == 'patch_first':
                patch = self._apply_preprocessing_patch_first(patch, x, y)
```

**`_generate_ng_patches_for_image()` L594-595:**

Before:
```python
                if self.hist_norm_mode == 'patch_first' or self.fda_mode == 'patch_first':
                    patch = self._apply_preprocessing_patch_first(patch)
```

After:
```python
                if self.hist_norm_mode == 'patch_first' or self.fda_mode == 'patch_first':
                    patch = self._apply_preprocessing_patch_first(patch, patch_x, patch_y)
```

> **Note**: `x, y` and `patch_x, patch_y` are variables already computed during patch extraction in each method.

---

## Reference

### FDA Function Location

- `utils/fda_utils.py` → `FDA_source_to_target_np(src, trg, L=beta)`
  - `src`: Source image (C, H, W) float32 [0, 1]
  - `trg`: Target (reference) image (C, H, W) float32 [0, 1]
  - `L`: Beta parameter (ratio of low-frequency components to transfer)

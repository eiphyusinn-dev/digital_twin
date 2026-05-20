# PatchCore Implementation Comparison Analysis

## Overview
This document compares the original PatchCore implementation with the client version (ad.patch.core.toyota.v2) to identify why the client version produces worse segmentation results despite using the same dataset and backbone.

## Critical Differences Found

### 1. Feature Extraction and Patching Mechanism ⚠️ **CRITICAL**

**Original Implementation:**
- Uses `PatchMaker` class with proper `torch.nn.Unfold` for patch extraction
- Implements sophisticated patchify/unpatchify operations with proper padding and stride
- Uses `NetworkFeatureAggregator` with forward hooks to extract features from specific layers
- Multiple feature layers are properly interpolated to match spatial dimensions before concatenation
- Uses `Preprocessing` and `Aggregator` modules to standardize feature dimensions across layers
- Features are properly reshaped and permuted for multi-scale feature fusion

**Client Implementation:**
- Uses simple adaptive average pooling (`AdaptiveAvgPool2d`) instead of proper patching
- No `PatchMaker` or `torch.nn.Unfold` operations
- Features are simply averaged with a 3x3 kernel and concatenated without proper spatial alignment
- Missing proper interpolation between multi-scale features (except for ConvNeXt v2 which has some interpolation)
- No standardization of feature dimensions across different backbone layers
- Features are concatenated directly without proper spatial alignment

**Impact:** This is the **MOST CRITICAL** difference. The original PatchCore's sophisticated patching mechanism ensures proper spatial alignment and feature fusion, which is essential for accurate anomaly localization. The client's simple pooling approach loses spatial information and leads to poor segmentation.

### 2. Nearest Neighbor Search Method ⚠️ **HIGH IMPACT**

**Original Implementation:**
- Uses FAISS (Facebook AI Similarity Search) for efficient nearest neighbor search
- Supports GPU acceleration for FAISS operations
- Uses `NearestNeighbourScorer` with proper FAISS integration
- Efficient for large-scale feature banks
- Uses `ConcatMerger` to properly merge multi-layer features

**Client Implementation:**
- Uses `torch.cdist` for brute-force distance computation
- No FAISS integration
- Less efficient for large datasets
- Simple distance computation without optimization

**Impact:** While both methods compute distances correctly, FAISS is more efficient and may handle edge cases better. However, this is less critical than the patching mechanism.

### 3. Coreset Sampling Implementation ⚠️ **MEDIUM IMPACT**

**Original Implementation:**
- Uses `GreedyCoresetSampler` or `ApproximateGreedyCoresetSampler`
- Proper greedy coreset selection with full distance matrix computation
- Uses dimensionality projection (Linear layer) for efficiency
- Well-tested and optimized implementation

**Client Implementation:**
- Uses `get_coreset_idx_randomp` with sparse random projection from sklearn
- Different implementation of greedy coreset selection
- Has fallback mechanism with random projection for memory issues
- Uses sklearn's `SparseRandomProjection` instead of learned projection

**Impact:** Both implement greedy coreset sampling, but the original uses a learned projection while the client uses random projection. This could lead to different subset selection and affect performance.

### 4. Anomaly Score Computation ⚠️ **MEDIUM IMPACT**

**Original Implementation:**
- Uses mean of k-nearest neighbor distances as the base anomaly score
- Implements reweighting mechanism based on nearest neighbor distances
- Computes patch-level scores and aggregates to image-level scores
- Uses `PatchMaker.score()` to aggregate patch scores (max pooling)

**Client Implementation:**
- Computes multiple scoring methods (max, top5 mean, softmax, percentile)
- Implements custom reweighting mechanism similar to original
- More complex scoring with multiple aggregation strategies
- Uses different reweighting formula

**Impact:** The client's multiple scoring methods could actually be beneficial, but the underlying feature quality is poor due to the patching issue, so the scores are based on inferior features.

### 5. Segmentation Map Generation ⚠️ **HIGH IMPACT**

**Original Implementation:**
- Uses `RescaleSegmentor` with proper interpolation to original image size
- Applies Gaussian smoothing with sigma=4 for refined segmentation maps
- Interpolates patch scores back to original image dimensions
- Uses scipy.ndimage.gaussian_filter for smoothing

**Client Implementation:**
- Uses bilinear interpolation to resize segmentation maps
- Implements native Gaussian blur with custom kernel
- Different smoothing parameters (kernel_size=21, sigma=4.0)
- Interpolates to input image size instead of fixed size

**Impact:** The interpolation method and smoothing parameters differ, which affects the final segmentation quality. However, this is secondary to the feature extraction issue.

### 6. Feature Normalization ⚠️ **MEDIUM IMPACT**

**Original Implementation:**
- Normalization is handled through the `Preprocessing` module
- Uses `MeanMapper` with adaptive average pooling to normalize feature dimensions
- Features are normalized to a target dimension during preprocessing

**Client Implementation:**
- Applies L2 normalization explicitly during training and inference
- Normalizes patch library after coreset sampling
- Normalizes test patches before distance computation

**Impact:** Different normalization approaches could lead to different distance scales, but both are valid approaches.

### 7. Backbone Loading ⚠️ **LOW IMPACT**

**Original Implementation:**
- Uses timm's pretrained models directly
- Supports a wide range of backbones through timm
- Standard pretrained weights from timm

**Client Implementation:**
- Loads custom checkpoints from local directories
- Implements custom feature extractors for specific architectures (ConvNeXt, ViT)
- Uses custom checkpoint loading with prefix adjustments

**Impact:** If the checkpoints are the same, this should not affect results. However, custom implementations may have subtle differences.

### 8. Data Preprocessing Pipeline ⚠️ **LOW IMPACT**

**Original Implementation:**
- Uses MVTec dataset with specific transforms
- Resize to 256, then CenterCrop to 224
- Standard ImageNet normalization
- Handles ground truth masks properly

**Client Implementation:**
- Simple resize to SIZE (224) with BICUBIC interpolation
- Standard ImageNet normalization
- Custom dataset class for OK/NG images

**Impact:** The original uses Resize(256) + CenterCrop(224) while the client uses direct Resize(224). This could lead to slight differences in input images, but is unlikely to cause major performance differences.

### 9. Spatial Feature Alignment ⚠️ **CRITICAL**

**Original Implementation:**
- Multi-scale features are interpolated to match the spatial dimensions of the first layer
- Uses bilinear interpolation with `align_corners=False`
- Features are properly reshaped and permuted before interpolation
- Ensures all feature maps have the same spatial resolution before concatenation

**Client Implementation:**
- For ConvNeXt v2, there is some interpolation to align features
- For other backbones (WideResNet), no proper spatial alignment
- Features are simply pooled to the same size without proper interpolation
- Missing the sophisticated spatial alignment from the original

**Impact:** This is another **CRITICAL** difference. The original ensures proper spatial alignment of multi-scale features, which is essential for accurate anomaly localization. The client's approach loses spatial correspondence between features from different layers.

## Root Cause Analysis

The primary reason for worse segmentation results in the client version is:

### **Missing Proper Patching Mechanism**

The original PatchCore uses a sophisticated patching mechanism (`PatchMaker` with `torch.nn.Unfold`) that:
1. Extracts overlapping patches from feature maps with proper stride and padding
2. Maintains spatial correspondence between patches and original image locations
3. Properly handles multi-scale feature fusion with spatial alignment
4. Ensures that anomaly scores can be accurately localized back to image regions

The client version replaces this with simple adaptive pooling, which:
1. Loses spatial information by averaging over regions
2. Does not maintain proper patch-to-image correspondence
3. Fails to properly align multi-scale features
4. Results in poor localization accuracy for segmentation

### **Secondary Issues**

1. **Spatial Feature Alignment:** Missing proper interpolation between multi-scale features
2. **Coreset Sampling:** Different projection method (random vs learned)
3. **Nearest Neighbor Search:** Using brute-force instead of FAISS (less critical)

## Recommendations

### Immediate Actions (High Priority)

1. **Implement Proper Patching Mechanism**
   - Add `PatchMaker` class with `torch.nn.Unfold` operations
   - Replace adaptive pooling with proper patchify/unpatchify
   - Ensure proper spatial correspondence between patches and image regions

2. **Implement Spatial Feature Alignment**
   - Add proper interpolation between multi-scale features
   - Use the original's approach for feature reshaping and permutation
   - Ensure all feature maps have matching spatial dimensions before concatenation

3. **Use Original's Feature Extraction Pipeline**
   - Implement `NetworkFeatureAggregator` with forward hooks
   - Use the original's preprocessing and aggregation modules
   - Ensure consistent feature dimensions across layers

### Medium Priority

4. **Consider Using FAISS**
   - Replace `torch.cdist` with FAISS for efficiency
   - This may improve performance for large datasets

5. **Align Coreset Sampling**
   - Consider using the original's `GreedyCoresetSampler`
   - Or ensure the random projection approach is properly validated

### Low Priority

6. **Standardize Data Preprocessing**
   - Consider using Resize(256) + CenterCrop(224) like the original
   - Ensure consistent preprocessing between both versions

## Code References

### Original Implementation Files
- `original_patchcore/src/patchcore/patchcore.py` - Main PatchCore class
- `original_patchcore/src/patchcore/common.py` - Feature aggregation, scoring, segmentation
- `original_patchcore/src/patchcore/sampler.py` - Coreset sampling
- `original_patchcore/bin/run_patchcore.py` - Training/inference script

### Client Implementation Files
- `ad.patch.core.toyota.v2/models.py` - Client's PatchCore implementation
- `ad.patch.core.toyota.v2/data.py` - Data loading
- `ad.patch.core.toyota.v2/utils.py` - Utility functions including coreset sampling
- `ad.patch.core.toyota.v2/train_test_patchcore.py` - Training/inference script

## Conclusion

The client version's segmentation results are worse primarily due to the missing proper patching mechanism and spatial feature alignment. The original PatchCore's sophisticated approach to patch extraction and multi-scale feature fusion is essential for accurate anomaly localization. Simply replacing adaptive pooling with the original's `PatchMaker` and implementing proper spatial alignment should significantly improve the client version's performance.

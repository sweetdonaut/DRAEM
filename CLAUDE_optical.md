# Optical Dataset Guidelines

## Overview
This document contains specific guidelines for working with the optical dataset in the DRAEM project.

## Key Files
- `train_DRAEM_optical.py` - Training script for optical data
- `test_DRAEM_optical.py` - Testing script for optical data
- `train_optical.sh` - Shell script for training execution
- `test_optical.sh` - Shell script for testing execution

## Working Principles
- FOCUS on optical-specific requirements and adjustments
- UNDERSTAND the data characteristics before making changes
- MAINTAIN compatibility with the original DRAEM architecture
- TEST changes incrementally to ensure functionality

## Communication Style
- ASK for clarification when optical data specifics are unclear
- PROVIDE brief status updates during long operations
- SUGGEST optimizations only when directly relevant to the task
- AVOID over-engineering solutions

## Dataset Specifications
- Input format: 976x176 float32 TIFF files
- Target: Anomaly detection on optical images
- Processing: Split into 128x128 patches with overlap
  - Height (976): 8 splits
  - Width (176): 2 splits  
  - Total: 16 patches per image with overlapping regions

## Technical Details
- Patch extraction: `extract_patches_976x176` in test_DRAEM_optical_slide.py
- Parameters: stride_h=121, stride_w=48 → 16 patches (8×2)
- Overlap: 7 pixels (height), 80 pixels (width)

### Phase 1 Complete ✓
- Created OpticalDatasetSlide (5,536 patches)
- Trained 128x128 model successfully
- Implemented patch inference in `test_DRAEM_optical_slide.py`
- Visualization: Original | Reconstructed | Heatmap per patch

### Phase 2 Complete ✓
- Implemented heatmap fusion (max & average methods)
- Created 3-panel visualization: Origin | Reconstruct | Heatmap
- Production-ready test_optical.sh with fusion options
- Ready for deployment and evaluation
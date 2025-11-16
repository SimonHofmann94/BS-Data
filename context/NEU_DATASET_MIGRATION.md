# NEU Surface Defect Database Migration

## Overview

Successfully migrated the project from **Severstal Steel Defect Classification** (multi-label) to **NEU Surface Defect Database** (single-label) with minimal changes to the existing architecture.

---

## Dataset Information

### NEU Surface Defect Database
- **Source**: Kaggle (`kaustubhdikshit/neu-surface-defect-database`)
- **Classes**: 6 defect types (single-label classification)
  1. `crazing` (Cr)
  2. `inclusion` (In)
  3. `patches` (Pa)
  4. `pitted_surface` (PS)
  5. `rolled-in_scale` (RS)
  6. `scratches` (Sc)
- **Images**: 1,800 grayscale images (200x200 pixels)
- **Distribution**: Perfectly balanced - 300 samples per class
- **Splits**:
  - Train: 1,440 images (240 per class)
  - Validation: 360 images (60 per class)
  - Test: 360 images (same as validation, no separate test set provided)

### Key Differences from Severstal
| Aspect | Severstal | NEU |
|--------|-----------|-----|
| Task | Multi-label | Single-label |
| Classes | 5 (no_defect + 4 defects) | 6 (defect types) |
| Image Size | 256x1600 | 200x200 |
| Label Source | JSON annotations | Filename prefix |
| Balance | Highly imbalanced | Perfectly balanced |

---

## Changes Made

### 1. Dataset Setup (`code/additional/setup_neu_dataset.py`)

Created script to:
- Download NEU dataset from Kaggle using `kagglehub`
- Copy images to `data/images/`
- Create split files (`data/splits/train.txt`, `val.txt`, `test.txt`)
- Verify class distribution

**Usage**:
```bash
./venv/bin/python3 code/additional/setup_neu_dataset.py
```

### 2. Dataset Class (`code/core/data/dataset.py`)

**Modified**: `SeverstalFullImageDataset` class

**Changes**:
- Updated docstring to reflect NEU dataset
- Changed `NUM_CLASSES` from 5 to 6
- Replaced `DEFECT_CLASS_TO_IDX` mapping with NEU class names
- Modified `_load_label()` to extract class from filename instead of JSON
- Updated dummy tensor size from (3, 256, 1600) to (3, 200, 200)

**Key Logic**:
```python
# Extract class from filename (e.g., "crazing_49.jpg" -> crazing)
for class_name, class_idx in self.CLASS_TO_IDX.items():
    if img_name.startswith(class_name):
        label[class_idx] = 1.0
        return label  # One-hot encoded
```

### 3. Loss Function (`code/core/losses/ce_loss.py`)

**Added**: `CrossEntropyLossWrapper` for single-label classification

**Features**:
- Wraps `torch.nn.CrossEntropyLoss`
- Handles both class indices and one-hot encoded labels
- Consistent interface with existing loss functions
- Registered as `"cross_entropy"` in loss registry

### 4. Configuration (`config/train_config.yaml`)

**Modified sections**:

```yaml
model:
  num_classes: 6  # Changed from 5

loss:
  type: "cross_entropy"  # Changed from focal_loss
  weight: null
  reduction: "mean"

data:
  image_size: [200, 200]  # Changed from [256, 1600]
  num_classes: 6  # Changed from 5
  class_names: ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
  batch_size: 32  # Increased from 24 (smaller images)
  
augmentation:
  brightness_contrast:
    brightness: 0.2  # Reduced from 0.3
    contrast: 0.2    # Reduced from 0.3
  defect_blackout:
    enabled: false  # Disabled (not applicable for NEU)
```

### 5. Loss Registry (`code/core/losses/registry.py`)

**Added**: Registration for `CrossEntropyLossWrapper`

```python
registry.register(
    "cross_entropy",
    CrossEntropyLossWrapper,
    description="Cross-Entropy Loss for single-label classification"
)
```

---

## Files Modified

### Core Changes
1. `code/core/data/dataset.py` - Dataset loading logic
2. `config/train_config.yaml` - Configuration parameters
3. `code/core/losses/ce_loss.py` - **NEW**: CrossEntropy loss wrapper
4. `code/core/losses/registry.py` - Loss registration

### Setup Scripts (New)
1. `code/additional/download_neu_dataset.py` - Dataset download helper
2. `code/additional/setup_neu_dataset.py` - **MAIN SETUP SCRIPT**
3. `code/additional/test_neu_dataset.py` - Dataset loading test
4. `code/additional/test_training_setup.py` - Full pipeline test

### Data Files (Generated)
1. `data/images/*.jpg` - 1,800 images (200x200)
2. `data/splits/train.txt` - 1,440 image names
3. `data/splits/val.txt` - 360 image names
4. `data/splits/test.txt` - 360 image names

---

## Unchanged Components

The following components **work without modification**:
- ✅ Model architecture (`convnext_tiny_cbam`)
- ✅ CBAM attention mechanism
- ✅ Training pipeline (`code/train.py`)
- ✅ Data loaders (`code/core/data/loaders.py`)
- ✅ Split loading logic (`code/core/data/splitting.py`)
- ✅ Metrics computation (`code/core/training/metrics.py`)
- ✅ Trainer orchestrator (`code/core/training/trainer.py`)
- ✅ Model registry
- ✅ Augmentation pipeline (minus blackout)

**This demonstrates the modularity and extensibility of the architecture!**

---

## Verification

All tests passed:

```bash
# 1. Dataset loading
./venv/bin/python3 code/additional/test_neu_dataset.py
# ✓ 1440 samples loaded, one-hot labels correct

# 2. Full pipeline test
./venv/bin/python3 code/additional/test_training_setup.py
# ✓ Dataset, model, loss, forward pass all working
```

---

## Training

### Quick Start

```bash
# Using venv python directly
./venv/bin/python3 code/train.py

# Or with custom settings
./venv/bin/python3 code/train.py \
  data.batch_size=32 \
  training.num_epochs=50 \
  optimizer.lr=0.001
```

### Expected Results

With balanced classes (300 samples each), expect:
- **Faster convergence** than imbalanced Severstal
- **Higher accuracy** (no class imbalance issues)
- **Simpler loss function** (CrossEntropy vs Focal Loss)
- **Smaller memory footprint** (200x200 vs 256x1600)

### Hyperparameters

Recommended settings for NEU:
```yaml
batch_size: 32        # Small images fit more per batch
learning_rate: 0.001  # Standard for balanced data
epochs: 50-100        # Balanced data converges faster
loss: cross_entropy   # Standard for single-label
```

---

## Design Decisions

### 1. Why keep `SeverstalFullImageDataset` name?
- **Minimal changes principle**: Renaming would require changes across multiple files
- Class is flexible enough to handle both datasets
- Clear documentation indicates NEU usage

### 2. Why CrossEntropy instead of Focal Loss?
- NEU dataset is **perfectly balanced** (300 samples per class)
- Focal Loss designed for **imbalanced** datasets
- CrossEntropy is simpler and more appropriate
- Focal Loss can still be used if desired (one-hot labels work)

### 3. Why one-hot encoding for single-label?
- **Consistency** with existing pipeline
- Dataset returns one-hot, loss function converts to indices
- Allows easy switching between loss functions
- Minimal code changes

### 4. Why disable blackout augmentation?
- Blackout designed for **wide-strip defects** (Severstal)
- NEU defects are **localized patterns** in square images
- Not meaningful to black out entire defect regions

---

## Future Improvements

Optional enhancements (not required for basic functionality):

1. **Test Set Split**: Currently using validation as test. Could re-split 80/10/10.
2. **Class-wise Metrics**: Track per-defect-type performance
3. **Confusion Matrix**: Visualize classification errors
4. **Data Augmentation**: Add rotation/scaling for small datasets
5. **Ensemble Methods**: Combine multiple models for better accuracy

---

## Summary

✅ **Migration Complete** with minimal changes:
- 4 files modified
- 1 new loss function added
- 4 helper scripts created
- All existing architecture preserved

✅ **Ready for Training**:
```bash
./venv/bin/python3 code/train.py
```

✅ **Verification**: All tests passing

The modular design made switching datasets straightforward, demonstrating the power of the registry pattern and clean architecture!

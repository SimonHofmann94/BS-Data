"""
Setup NEU Surface Defect Database for the project.
- Download dataset
- Copy images to data/images/
- Create split files (train.txt, val.txt, test.txt) - no test set provided, so we'll use val as test
- Check image dimensions
"""
import os
import kagglehub
from pathlib import Path
import shutil
from PIL import Image
from collections import defaultdict

# Project paths
PROJECT_ROOT = Path("/Users/ema/projects/DL4SE-simon-convnext-cbam-run")
DATA_DIR = PROJECT_ROOT / "data"
IMAGES_DIR = DATA_DIR / "images"
SPLITS_DIR = DATA_DIR / "splits"

# Download dataset
print("Downloading NEU Surface Defect Database...")
dataset_path = Path(kagglehub.dataset_download("kaustubhdikshit/neu-surface-defect-database"))
neu_det_path = dataset_path / "NEU-DET"

print(f"\nDataset downloaded to: {dataset_path}")

# Class mapping: 6 classes
# Classes: crazing, inclusion, patches, pitted_surface, rolled-in_scale, scratches
classes = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
print(f"\nClasses ({len(classes)}): {classes}")

# Check image dimensions
print("\nChecking image dimensions...")
sample_img = None
for class_name in classes:
    train_class_dir = neu_det_path / "train" / "images" / class_name
    if train_class_dir.exists():
        imgs = list(train_class_dir.glob("*.jpg"))
        if imgs:
            sample_img = imgs[0]
            break

if sample_img:
    img = Image.open(sample_img)
    print(f"Sample image: {sample_img.name}")
    print(f"Dimensions: {img.size} (width x height)")
    print(f"Mode: {img.mode}")

# Create directories
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
SPLITS_DIR.mkdir(parents=True, exist_ok=True)

# Copy images and create splits
all_images = []
class_images = {class_name: [] for class_name in classes}

print("\nCopying images and organizing by class...")
for split in ["train", "validation"]:
    split_dir = neu_det_path / split / "images"
    for class_name in classes:
        class_dir = split_dir / class_name
        if not class_dir.exists():
            continue
        
        images = list(class_dir.glob("*.jpg"))
        print(f"  {split}/{class_name}: {len(images)} images")
        
        for img_path in images:
            # Copy image
            dest_path = IMAGES_DIR / img_path.name
            shutil.copy2(img_path, dest_path)
            
            # Organize by class for proper splitting
            all_images.append(img_path.name)
            class_images[class_name].append(img_path.name)

# Create proper 70/15/15 split (stratified by class)
print("\nCreating stratified 70/15/15 split...")
train_images = []
val_images = []
test_images = []

import random
random.seed(42)

for class_name, images in class_images.items():
    random.shuffle(images)
    n = len(images)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)
    
    train_images.extend(images[:n_train])
    val_images.extend(images[n_train:n_train+n_val])
    test_images.extend(images[n_train+n_val:])

# Save split files
print("\nSaving split files...")
with open(SPLITS_DIR / "train.txt", "w") as f:
    f.write("\n".join(sorted(train_images)))
print(f"  train.txt: {len(train_images)} images")

with open(SPLITS_DIR / "val.txt", "w") as f:
    f.write("\n".join(sorted(val_images)))
print(f"  val.txt: {len(val_images)} images")

with open(SPLITS_DIR / "test.txt", "w") as f:
    f.write("\n".join(sorted(test_images)))
print(f"  test.txt: {len(test_images)} images")

# Count class distribution
print("\nClass distribution:")
class_counts = defaultdict(lambda: {"train": 0, "val": 0, "test": 0})

for img_name in train_images:
    for class_name in classes:
        if class_name in img_name:
            class_counts[class_name]["train"] += 1
            break

for img_name in val_images:
    for class_name in classes:
        if class_name in img_name:
            class_counts[class_name]["val"] += 1
            break

for img_name in test_images:
    for class_name in classes:
        if class_name in img_name:
            class_counts[class_name]["test"] += 1
            break

for class_name in classes:
    train_count = class_counts[class_name]["train"]
    val_count = class_counts[class_name]["val"]
    test_count = class_counts[class_name]["test"]
    total = train_count + val_count + test_count
    print(f"  {class_name}: {train_count} train, {val_count} val, {test_count} test (total: {total})")

print("\n✓ NEU dataset setup complete!")
print(f"Images copied to: {IMAGES_DIR}")
print(f"Splits saved to: {SPLITS_DIR}")

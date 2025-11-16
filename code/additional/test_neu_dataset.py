"""
Test NEU dataset loading.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from core.data.dataset import SeverstalFullImageDataset
from torchvision import transforms
import numpy as np

# Test dataset loading
print("Testing NEU dataset loading...")

# Load train split
splits_dir = Path("data/splits")
with open(splits_dir / "train.txt", "r") as f:
    train_images = [line.strip() for line in f if line.strip()]

print(f"\nTrain split: {len(train_images)} images")
print(f"Sample images: {train_images[:3]}")

# Create dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = SeverstalFullImageDataset(
    img_dir="data/images",
    ann_dir="data/annotations",  # Not used
    image_names=train_images,
    transform=transform,
    num_classes=6
)

print(f"\nDataset size: {len(dataset)}")

# Test loading a sample
print("\nTesting sample loading...")
img, label = dataset[0]
print(f"Image shape: {img.shape}")
print(f"Label shape: {label.shape}")
print(f"Label: {label}")
print(f"Label sum (should be 1.0 for single-label): {label.sum()}")

# Check class distribution
print("\nChecking class distribution...")
class_counts = np.zeros(6)
for i in range(min(100, len(dataset))):  # Sample first 100
    _, label = dataset[i]
    class_counts += label.numpy()

print(f"Class counts (first 100 samples):")
class_names = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
for i, name in enumerate(class_names):
    print(f"  {name}: {int(class_counts[i])}")

print("\n✓ Dataset loading successful!")

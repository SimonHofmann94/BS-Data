"""
Quick test to verify the NEU dataset setup works with the training pipeline.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from omegaconf import OmegaConf
from core.data.dataset import SeverstalFullImageDataset
from core.data.loaders import create_dataloaders
from core.losses.registry import get_registry
from core.models.registry import get_registry as get_model_registry
from torchvision import transforms

# Load config
config_path = Path("config/train_config.yaml")
cfg = OmegaConf.load(config_path)

print("=" * 60)
print("NEU Dataset Setup Test")
print("=" * 60)

# 1. Test dataset loading
print("\n1. Testing dataset loading...")
splits_dir = Path(cfg.data.splits_dir)
with open(splits_dir / "train.txt", "r") as f:
    train_images = [line.strip() for line in f if line.strip()]

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = SeverstalFullImageDataset(
    img_dir=cfg.data.img_dir,
    ann_dir=cfg.data.ann_dir,
    image_names=train_images,
    transform=transform,
    num_classes=cfg.data.num_classes
)

print(f"✓ Train dataset: {len(train_dataset)} samples")

# 2. Test model instantiation
print("\n2. Testing model instantiation...")
model_registry = get_model_registry()
model = model_registry.get(
    cfg.model.name,
    num_classes=cfg.model.num_classes,
    pretrained=cfg.model.pretrained
)
print(f"✓ Model: {cfg.model.name}")
print(f"  Input shape expected: {cfg.data.image_size}")
print(f"  Output classes: {cfg.model.num_classes}")

# 3. Test loss function
print("\n3. Testing loss function...")
loss_registry = get_registry()
loss_fn = loss_registry.get(
    cfg.loss.type,
    num_classes=cfg.data.num_classes,
    **{k: v for k, v in cfg.loss.items() if k != 'type'}
)
print(f"✓ Loss function: {cfg.loss.type}")

# 4. Test forward pass
print("\n4. Testing forward pass...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Get a batch
batch_size = 4
images = []
labels = []
for i in range(batch_size):
    img, label = train_dataset[i]
    images.append(img)
    labels.append(label)

images = torch.stack(images).to(device)
labels = torch.stack(labels).to(device)

print(f"  Batch shape: {images.shape}")
print(f"  Labels shape: {labels.shape}")

# Forward pass
with torch.no_grad():
    outputs = model(images)
    loss = loss_fn(outputs, labels)

print(f"  Output shape: {outputs.shape}")
print(f"  Loss: {loss.item():.4f}")

print("\n" + "=" * 60)
print("✓ All tests passed! Ready for training.")
print("=" * 60)

print("\nTo start training, run:")
print("  ./venv/bin/python3 code/train.py")

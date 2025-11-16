"""
Dataset implementation for NEU Surface Defect Database.

Loads 200x200 grayscale images for single-label classification.
"""

import os
import json
import numpy as np
from typing import Optional, Tuple
from PIL import Image
import torch
from torch.utils.data import Dataset
import logging

logger = logging.getLogger(__name__)


class SeverstalFullImageDataset(Dataset):
    """
    Dataset for NEU Surface Defect Database with single-label classification.
    
    Loads 200x200 images. Class is determined from filename.
    
    6-Class Setup (single-label):
        - Index 0: crazing
        - Index 1: inclusion
        - Index 2: patches
        - Index 3: pitted_surface
        - Index 4: rolled-in_scale
        - Index 5: scratches
    
    Args:
        img_dir: Directory containing images
        ann_dir: Directory containing annotations (NOT USED for NEU)
        image_names: List of image filenames to load
        transform: Torchvision transforms
        num_classes: Number of classes (6 defect types)
    """
    
    CLASS_TO_IDX = {
        "crazing": 0,
        "inclusion": 1,
        "patches": 2,
        "pitted_surface": 3,
        "rolled-in_scale": 4,
        "scratches": 5,
    }
    NUM_CLASSES = 6
    
    def __init__(
        self,
        img_dir: str,
        ann_dir: str,
        image_names: list,
        transform: Optional = None,
        num_classes: int = 6,
        verbose: bool = False
    ):
        self.img_dir = img_dir
        self.ann_dir = ann_dir  # Not used for NEU dataset
        self.image_names = image_names
        self.transform = transform
        self.num_classes = num_classes
        self.verbose = verbose
        
        # Load image-label pairs
        self.samples = []
        self._load_samples()
        
        logger.info(
            f"Loaded {len(self.samples)} samples from {len(image_names)} images"
        )
    
    def _load_samples(self) -> None:
        """Load all image-label pairs into memory."""
        for img_name in self.image_names:
            img_path = os.path.join(self.img_dir, img_name)
            
            if not os.path.exists(img_path):
                logger.warning(f"Image not found: {img_path}")
                continue
            
            # Load label
            label = self._load_label(img_name)
            if label is None:
                continue
            
            self.samples.append({
                "image_name": img_name,
                "image_path": img_path,
                "label": label
            })
    
    def _load_label(self, img_name: str) -> Optional[np.ndarray]:
        """
        Load label for an image from filename.
        
        Returns a 6-element one-hot vector for single-label classification.
        Class is extracted from the filename prefix (e.g., 'crazing_49.jpg' -> crazing)
        
        Returns:
            One-hot encoded label vector
        """
        label = np.zeros(self.num_classes, dtype=np.float32)
        
        # Extract class from filename
        # Format: classname_number.jpg (e.g., 'crazing_49.jpg', 'pitted_surface_123.jpg')
        for class_name, class_idx in self.CLASS_TO_IDX.items():
            if img_name.startswith(class_name):
                label[class_idx] = 1.0
                return label
        
        # If no class matched, log warning
        logger.warning(f"Could not determine class for image: {img_name}")
        return label
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample.
        
        Args:
            idx: Sample index
        
        Returns:
            Tuple of (image_tensor, label_tensor)
        """
        sample = self.samples[idx]
        
        try:
            # Load image
            image_pil = Image.open(sample["image_path"]).convert("RGB")
            
            # Apply transforms
            if self.transform:
                image_tensor = self.transform(image_pil)
            else:
                image_tensor = torch.from_numpy(
                    np.array(image_pil).transpose(2, 0, 1)
                ).float() / 255.0
            
            # Get label
            label_tensor = torch.from_numpy(sample["label"]).float()
            
            return image_tensor, label_tensor
            
        except Exception as e:
            logger.error(
                f"Error loading sample {idx}: {sample['image_name']} - {e}"
            )
            # Return dummy tensors (200x200 for NEU dataset)
            return torch.zeros(3, 200, 200), torch.zeros(self.num_classes)


if __name__ == "__main__":
    # Test dataset loading
    print("Dataset implementation ready for use")

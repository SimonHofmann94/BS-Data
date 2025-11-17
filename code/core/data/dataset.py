"""
Dataset implementation for Ball Screw Drive Surface Defect Dataset.

Loads 150x150 RGB images for binary classification.
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
    Dataset for Ball Screw Drive Surface Defect Dataset with binary classification.
    
    Loads 150x150 images. Class is determined from filename prefix.
    
    Binary Classification:
        - Index 0: no_defect (filename starts with 'N ')
        - Index 1: defect (filename starts with 'P ' - pitting)
    
    Args:
        img_dir: Directory containing images
        ann_dir: Directory containing annotations (NOT USED)
        image_names: List of image filenames to load
        transform: Torchvision transforms
        num_classes: Number of classes (2: no_defect/defect)
    """
    
    CLASS_TO_IDX = {
        "no_defect": 0,
        "defect": 1,
    }
    NUM_CLASSES = 2
    
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
        
        Returns a 2-element one-hot vector for binary classification.
        Class is extracted from the filename prefix:
        - 'N ' prefix -> no_defect (index 0)
        - 'P ' prefix -> defect/pitting (index 1)
        
        Returns:
            One-hot encoded label vector
        """
        label = np.zeros(self.num_classes, dtype=np.float32)
        
        # Extract class from filename prefix
        if img_name.startswith('N '):
            label[0] = 1.0  # no_defect
            return label
        elif img_name.startswith('P '):
            label[1] = 1.0  # defect (pitting)
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
            # Return dummy tensors (150x150 for Ball Screw Drive dataset)
            return torch.zeros(3, 150, 150), torch.zeros(self.num_classes)


if __name__ == "__main__":
    # Test dataset loading
    print("Dataset implementation ready for use")

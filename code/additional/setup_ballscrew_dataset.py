#!/usr/bin/env python3
"""
Setup script for Ball Screw Drive Surface Defect Dataset.

Creates stratified train/val/test splits (70/15/15) for binary classification.
- N (no defect): 11,075 images
- P (pitting defect): 10,760 images

Total: 21,835 images (150x150 pixels)
"""

import os
import random
from pathlib import Path
from collections import defaultdict
import json


def create_splits(img_dir, splits_dir, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Create stratified splits for binary classification.
    
    Args:
        img_dir: Directory containing all images
        splits_dir: Directory to save split files
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    
    # Verify ratios sum to 1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    # Get all image files
    img_dir = Path(img_dir)
    all_images = sorted([f.name for f in img_dir.glob("*.png")])
    
    print(f"Total images found: {len(all_images)}")
    
    # Separate by class (binary: N=no defect, P=pitting)
    classes = defaultdict(list)
    for img_name in all_images:
        # Extract class from first character (N or P)
        if img_name.startswith('N '):
            classes['no_defect'].append(img_name)
        elif img_name.startswith('P '):
            classes['defect'].append(img_name)
        else:
            print(f"Warning: Unknown class for {img_name}")
    
    print(f"\nClass distribution:")
    for class_name, images in classes.items():
        print(f"  {class_name}: {len(images)} images")
    
    # Shuffle each class
    for class_name in classes:
        random.shuffle(classes[class_name])
    
    # Create stratified splits
    train_set = []
    val_set = []
    test_set = []
    
    for class_name, images in classes.items():
        n_total = len(images)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        # n_test gets the remainder to ensure all images are used
        
        train_set.extend(images[:n_train])
        val_set.extend(images[n_train:n_train + n_val])
        test_set.extend(images[n_train + n_val:])
    
    # Shuffle the final sets
    random.shuffle(train_set)
    random.shuffle(val_set)
    random.shuffle(test_set)
    
    # Print split statistics
    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_set)} ({len(train_set)/len(all_images)*100:.1f}%)")
    print(f"  Val:   {len(val_set)} ({len(val_set)/len(all_images)*100:.1f}%)")
    print(f"  Test:  {len(test_set)} ({len(test_set)/len(all_images)*100:.1f}%)")
    print(f"  Total: {len(train_set) + len(val_set) + len(test_set)}")
    
    # Verify per-class distribution in splits
    def count_classes(image_list):
        counts = defaultdict(int)
        for img in image_list:
            if img.startswith('N '):
                counts['no_defect'] += 1
            elif img.startswith('P '):
                counts['defect'] += 1
        return counts
    
    print(f"\nPer-class distribution:")
    for split_name, split_data in [("Train", train_set), ("Val", val_set), ("Test", test_set)]:
        counts = count_classes(split_data)
        print(f"  {split_name}:")
        for class_name, count in counts.items():
            print(f"    {class_name}: {count}")
    
    # Create splits directory if it doesn't exist
    splits_dir = Path(splits_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)
    
    # Save splits to text files
    with open(splits_dir / 'train.txt', 'w') as f:
        f.write('\n'.join(train_set))
    
    with open(splits_dir / 'val.txt', 'w') as f:
        f.write('\n'.join(val_set))
    
    with open(splits_dir / 'test.txt', 'w') as f:
        f.write('\n'.join(test_set))
    
    # Save metadata
    metadata = {
        'dataset': 'Ball Screw Drive Surface Defect Dataset',
        'total_images': len(all_images),
        'num_classes': 2,
        'classes': ['no_defect', 'defect'],
        'class_names': ['no_defect', 'defect'],
        'image_size': [150, 150],
        'split_ratios': {
            'train': train_ratio,
            'val': val_ratio,
            'test': test_ratio
        },
        'split_sizes': {
            'train': len(train_set),
            'val': len(val_set),
            'test': len(test_set)
        },
        'class_distribution': {
            'train': dict(count_classes(train_set)),
            'val': dict(count_classes(val_set)),
            'test': dict(count_classes(test_set)),
            'total': {
                'no_defect': len(classes['no_defect']),
                'defect': len(classes['defect'])
            }
        },
        'seed': seed
    }
    
    with open(splits_dir / 'split_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nSplits saved to: {splits_dir}")
    print(f"  - train.txt ({len(train_set)} images)")
    print(f"  - val.txt ({len(val_set)} images)")
    print(f"  - test.txt ({len(test_set)} images)")
    print(f"  - split_metadata.json")
    
    return train_set, val_set, test_set


if __name__ == '__main__':
    # Get project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    
    img_dir = project_root / 'data' / 'images'
    splits_dir = project_root / 'data' / 'splits'
    
    print("=" * 60)
    print("Ball Screw Drive Dataset Setup")
    print("=" * 60)
    print(f"Image directory: {img_dir}")
    print(f"Splits directory: {splits_dir}")
    print()
    
    create_splits(
        img_dir=img_dir,
        splits_dir=splits_dir,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42
    )
    
    print("\n" + "=" * 60)
    print("Setup complete!")
    print("=" * 60)

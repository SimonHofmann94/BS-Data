"""
Download and explore NEU Surface Defect Database from Kaggle.
"""
import os
import kagglehub
from pathlib import Path
import shutil

# Download dataset
print("Downloading NEU Surface Defect Database...")
dataset_path = kagglehub.dataset_download("kaustubhdikshit/neu-surface-defect-database")
print(f"Dataset downloaded to: {dataset_path}")

# Explore structure
print("\nDataset structure:")
for root, dirs, files in os.walk(dataset_path):
    level = root.replace(dataset_path, '').count(os.sep)
    indent = ' ' * 2 * level
    print(f"{indent}{os.path.basename(root)}/")
    subindent = ' ' * 2 * (level + 1)
    for file in files[:5]:  # Show first 5 files per directory
        print(f"{subindent}{file}")
    if len(files) > 5:
        print(f"{subindent}... and {len(files) - 5} more files")

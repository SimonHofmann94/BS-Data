#!/usr/bin/env python3
"""
Setup script to ensure training data is extracted and ready.
Handles Git LFS pointer detection and nested zip structures.
"""

import os
import zipfile
import shutil
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def is_lfs_pointer(file_path: Path) -> bool:
    """Check if a file is a Git LFS pointer instead of actual content."""
    try:
        with open(file_path, 'rb') as f:
            header = f.read(100)
            # LFS pointers start with "version https://git-lfs.github.com/spec"
            return b'git-lfs.github.com' in header
    except Exception:
        return False


def setup_data():
    project_root = Path(__file__).parent.parent.parent
    zip_path = project_root / "data" / "images" / "training_data.zip"
    extract_to = project_root / "data" / "images"

    logger.info(f"📂 Project root: {project_root}")
    logger.info(f"📦 Zip path: {zip_path}")
    logger.info(f"📁 Extract to: {extract_to}")

    # Check if images are already present
    existing_images = []
    if extract_to.exists():
        existing_images = [f for f in os.listdir(extract_to) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    if len(existing_images) > 10:  # More than 10 images = likely already extracted
        logger.info(f"✅ Data already present ({len(existing_images)} images found). Skipping extraction.")
        _validate_prefixes(extract_to)
        return

    # Check if zip file exists
    if not zip_path.exists():
        logger.error(f"❌ Zip file not found at {zip_path}")
        logger.error("   Make sure to run: git lfs pull")
        return

    # Check if it's an LFS pointer instead of actual file
    if is_lfs_pointer(zip_path):
        logger.error(f"❌ {zip_path} is a Git LFS pointer, not the actual file!")
        logger.error("   Run: git lfs pull")
        logger.error("   Or:  git lfs fetch --all && git lfs checkout")
        return

    # Check file size (LFS pointers are tiny, ~130 bytes)
    zip_size = zip_path.stat().st_size
    logger.info(f"� Zip file size: {zip_size / (1024*1024):.2f} MB")
    
    if zip_size < 1000:  # Less than 1KB = definitely a pointer
        logger.error(f"❌ Zip file is only {zip_size} bytes - likely an LFS pointer!")
        logger.error("   Run: git lfs pull")
        return

    # Extract the zip
    logger.info(f"📂 Extracting {zip_path}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # List contents first to understand structure
            namelist = zip_ref.namelist()
            logger.info(f"   Zip contains {len(namelist)} items")
            
            # Show first few items for debugging
            for name in namelist[:5]:
                logger.info(f"   - {name}")
            if len(namelist) > 5:
                logger.info(f"   ... and {len(namelist) - 5} more")
            
            # Extract to a temp location first
            temp_extract = extract_to / "_temp_extract"
            if temp_extract.exists():
                shutil.rmtree(temp_extract)
            temp_extract.mkdir(parents=True)
            
            zip_ref.extractall(temp_extract)
            logger.info("✅ Extraction completed to temp folder.")

        # Find where the images actually are (handle nested folders)
        images_found = list(temp_extract.rglob("*.jpg")) + list(temp_extract.rglob("*.png")) + list(temp_extract.rglob("*.jpeg"))
        logger.info(f"   Found {len(images_found)} images in extracted content")

        if not images_found:
            logger.error("❌ No images found in zip file!")
            shutil.rmtree(temp_extract)
            return

        # Move images to the target directory
        moved_count = 0
        for img_path in images_found:
            dest_path = extract_to / img_path.name
            if not dest_path.exists():
                shutil.move(str(img_path), str(dest_path))
                moved_count += 1
        
        logger.info(f"✅ Moved {moved_count} images to {extract_to}")

        # Cleanup temp folder
        shutil.rmtree(temp_extract)
        logger.info("🧹 Cleaned up temp folder.")

    except zipfile.BadZipFile:
        logger.error(f"❌ {zip_path} is not a valid zip file!")
        logger.error("   This usually means Git LFS didn't download the actual file.")
        logger.error("   Run: git lfs pull")
        return
    except Exception as e:
        logger.error(f"❌ Failed to extract: {e}")
        raise

    _validate_prefixes(extract_to)


def _validate_prefixes(extract_to: Path):
    """Validate that images have the expected N/P prefixes."""
    images = [f for f in os.listdir(extract_to) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    
    if not images:
        logger.warning("⚠️ No images found!")
        return

    valid_count = sum(1 for f in images if f.startswith('N ') or f.startswith('P '))
    invalid_count = len(images) - valid_count

    if valid_count == 0:
        logger.error("❌ CRITICAL: No images with 'N ' or 'P ' prefix found!")
        logger.error("   Examples of first 5 filenames:")
        for f in images[:5]:
            logger.error(f"     - '{f}'")
    elif invalid_count > 0:
        logger.warning(f"⚠️ {invalid_count}/{len(images)} images lack valid prefix (will be skipped)")
    else:
        logger.info(f"✅ All {len(images)} images have valid prefixes.")


if __name__ == "__main__":
    setup_data()

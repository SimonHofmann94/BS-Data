
import os
import zipfile
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_data():
    project_root = Path(__file__).parent.parent.parent
    zip_path = project_root / "data" / "images" / "training_data.zip"
    extract_to = project_root / "data" / "images"

    # Check if images are already present
    # We check for at least a reasonable number of images, or simply if the dir is non-empty
    # But since the user might have some other files there, let's look for known prefixes or just check entry count
    existing_images = []
    if extract_to.exists():
        existing_images = [f for f in os.listdir(extract_to) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    if len(existing_images) > 0:
        logger.info(f"✅ Data seems to be already present ({len(existing_images)} images found). Skipping unzip.")
    else:
        if not zip_path.exists():
            logger.error(f"❌ Zip file not found at {zip_path}. Cannot set up data.")
            return

        logger.info(f"📂 Unzipping {zip_path} to {extract_to}...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
            logger.info("✅ Unzip completed.")
        except Exception as e:
            logger.error(f"❌ Failed to unzip: {e}")
            raise e

    # Validation: Check for required prefixes
    # Dataset expects 'N ' or 'P '
    # We'll just scan the directory again
    images = [f for f in os.listdir(extract_to) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    if not images:
        logger.warning("⚠️ No images found after setup!")
        return

    valid_prefix_count = sum(1 for f in images if f.startswith('N ') or f.startswith('P '))
    invalid_count = len(images) - valid_prefix_count

    if valid_prefix_count == 0:
        logger.error("❌ CRITICAL: No images found with required prefixes 'N ' or 'P '!")
        logger.error("   The training script WILL FAIL to load labels.")
        logger.error("   Please ensure images in the zip file are named like 'N 01.jpg' or 'P 01.jpg'.")
    elif invalid_count > 0:
        logger.warning(f"⚠️ Found {invalid_count} images without 'N ' or 'P ' prefix. These will be skipped by the loader.")
        logger.info(f"✅ Found {valid_prefix_count} valid images.")
    else:
        logger.info(f"✅ All {len(images)} images appear to have valid prefixes.")

if __name__ == "__main__":
    setup_data()

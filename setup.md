# Runpod Setup Guide

## Prerequisites
- Runpod instance with GPU (recommended: RTX 3090 or better)
- PyTorch template or base Ubuntu image

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/DL4SE-project-UPM.git
cd DL4SE-project-UPM

# 2. Make setup script executable and run
chmod +x setup.sh
./setup.sh
```

That's it! The script handles everything automatically.

## What `setup.sh` Does
1. **Git LFS** – Pulls large files (training data zip)
2. **Virtual Environment** – Creates and activates `venv/`
3. **Dependencies** – Installs from `requirements.txt`
4. **Data Extraction** – Unzips training images
5. **Training** – Runs hyperparameter sweep

## Command Options

| Command | Description |
|---------|-------------|
| `./setup.sh` | Full training run |
| `./setup.sh --dry-run` | Preview without training |
| `./setup.sh --quick-test` | Fast sanity check (2 epochs, 1 config) |

## Results
Results are saved to: `code/experiments/results/hparam_summaries/`

## Troubleshooting

**Git LFS not installed:**
```bash
apt-get update && apt-get install -y git-lfs
git lfs install && git lfs pull
```

**Missing images after setup:**
Check that `data/images/training_data.zip` exists and re-run:
```bash
python code/utils/setup_data.py
```

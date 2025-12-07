#!/bin/bash
set -e  # Exit on error

echo "=================================================================="
echo "🚀 RUNPOD SETUP AND TRAINING WORKFLOW"
echo "=================================================================="

# Navigate to project root (where this script is located)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
echo "📂 Working directory: $(pwd)"

# ---------------------------------------------------------------------------
# Step 1: Git LFS Pull (ensure large files are downloaded)
# ---------------------------------------------------------------------------
echo ""
echo "[1/4] Pulling Git LFS files..."
if command -v git-lfs &> /dev/null; then
    git lfs pull
    echo "✅ Git LFS pull completed."
else
    echo "⚠️  git-lfs not found. Attempting to install..."
    apt-get update && apt-get install -y git-lfs
    git lfs install
    git lfs pull
    echo "✅ Git LFS installed and files pulled."
fi

# ---------------------------------------------------------------------------
# Step 2: Create and Activate Virtual Environment
# ---------------------------------------------------------------------------
echo ""
echo "[2/4] Setting up Python virtual environment..."
VENV_DIR="venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment in ./$VENV_DIR..."
    python3 -m venv "$VENV_DIR"
    echo "✅ Virtual environment created."
else
    echo "✅ Virtual environment already exists."
fi

# Activate the virtual environment
source "$VENV_DIR/bin/activate"
echo "✅ Virtual environment activated: $(which python)"

# ---------------------------------------------------------------------------
# Step 3: Install Dependencies
# ---------------------------------------------------------------------------
echo ""
echo "[3/4] Installing dependencies from requirements.txt..."
pip install --upgrade pip
pip install -r requirements.txt
echo "✅ Dependencies installed."

# ---------------------------------------------------------------------------
# Step 4: Setup Data (Unzip if needed)
# ---------------------------------------------------------------------------
echo ""
echo "[4/4] Setting up training data..."
python code/utils/setup_data.py

# ---------------------------------------------------------------------------
# Step 5: Run Hyperparameter Sweep
# ---------------------------------------------------------------------------
echo ""
echo "=================================================================="
echo "🏃 Starting Hyperparameter Sweep..."
echo "=================================================================="
# Pass all arguments to the python script (e.g., --dry-run, --quick-test)
python code/experiments/run_hparam_sweep.py "$@"

echo ""
echo "=================================================================="
echo "✅ WORKFLOW COMPLETED"
echo "=================================================================="

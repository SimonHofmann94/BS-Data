#!/usr/bin/env python3
r"""
Interactive prediction script for Ball Screw Drive defect classification.

Usage:
    # Predict on a single image
    python code/predict.py --image "data/images/N (1).png"
    
    # Predict on multiple images
    python code/predict.py --image "data/images/N (1).png" "data/images/P (500).png"
    
    # Use a specific experiment
    python code/predict.py --image "data/images/N (1).png" --experiment experiment_20251116_230211
    
    # Save visualization instead of displaying
    python code/predict.py --image "data/images/N (1).png" --save
"""

import os
import sys
import argparse
from pathlib import Path
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json

# Add code directory to path
sys.path.insert(0, str(Path(__file__).parent))

from core.models.registry import get_registry as get_model_registry
from core.augmentation import get_augmentation_pipeline


def load_latest_experiment(results_dir: Path):
    """Find the most recent experiment directory."""
    exp_dirs = sorted([
        d for d in results_dir.iterdir() 
        if d.is_dir() and d.name.startswith('experiment_')
    ], key=lambda x: x.name, reverse=True)
    
    if not exp_dirs:
        raise FileNotFoundError(f"No experiment directories found in {results_dir}")
    
    return exp_dirs[0]


def load_model_and_config(experiment_dir: Path, device: torch.device):
    """Load model and configuration from experiment directory."""
    
    # Load config
    results_file = experiment_dir / "results.json"
    if not results_file.exists():
        raise FileNotFoundError(f"Results file not found: {results_file}")
    
    with open(results_file, 'r') as f:
        config = json.load(f)
    
    print(f"📋 Loaded config from: {experiment_dir.name}")
    print(f"   Model: {config['model']}")
    print(f"   Best Epoch: {config['best_epoch']}")
    print(f"   Test F1: {config['test_metrics']['f1_macro']:.4f}")
    
    # Load model
    model_registry = get_model_registry()
    model_config = config["config"]["model"]
    
    model = model_registry.get(
        model_config["name"],
        num_classes=model_config["num_classes"],
        pretrained=False,  # Don't need pretrained weights, we're loading checkpoint
        cbam_stages=model_config.get("cbam_stages", [])
    )
    
    # Load checkpoint
    checkpoint_path = experiment_dir / "model_final.pt"
    if not checkpoint_path.exists():
        # Try finding any .pt file
        pt_files = list(experiment_dir.glob("*.pt"))
        if not pt_files:
            raise FileNotFoundError(f"No model checkpoint found in {experiment_dir}")
        checkpoint_path = pt_files[0]
    
    print(f"🧠 Loading model from: {checkpoint_path.name}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    print("✅ Model loaded successfully")
    
    return model, config


def preprocess_image(image_path: Path, image_size: tuple):
    """Load and preprocess an image for prediction."""
    
    # Load image
    img = Image.open(image_path).convert('RGB')
    
    # Get preprocessing transform (resize + normalize, no augmentation)
    transform = get_augmentation_pipeline(
        image_size=image_size,
        augmentation_config={}  # No augmentation for inference
    )
    
    # Transform
    img_tensor = transform(img)
    
    return img, img_tensor


def predict_single_image(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    device: torch.device,
    class_names: list
):
    """Make prediction on a single image."""
    
    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        logits = model(image_tensor)
    
    # For binary classification: use softmax
    probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]
    predicted_class_idx = int(np.argmax(probabilities))
    predicted_class = class_names[predicted_class_idx]
    confidence = float(probabilities[predicted_class_idx])
    
    return {
        'predicted_class': predicted_class,
        'predicted_idx': predicted_class_idx,
        'confidence': confidence,
        'probabilities': {class_names[i]: float(probabilities[i]) for i in range(len(class_names))},
        'raw_logits': logits.cpu().numpy()[0].tolist()
    }


def visualize_prediction(
    image: Image.Image,
    image_path: Path,
    prediction: dict,
    class_names: list,
    save_path: Path = None
):
    """Visualize prediction with original image and prediction details."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot original image
    ax1.imshow(image)
    ax1.axis('off')
    ax1.set_title(f'Original Image\n{image_path.name}', fontsize=12, fontweight='bold')
    
    # Plot prediction
    ax2.axis('off')
    
    # Determine prediction status
    predicted_class = prediction['predicted_class']
    confidence = prediction['confidence']
    
    # Color based on prediction (green for no_defect, red for defect)
    color = 'green' if predicted_class == 'no_defect' else 'red'
    
    # Create prediction text
    pred_text = f"""
PREDICTION RESULTS
{'='*40}

Predicted Class: {predicted_class.upper()}
Confidence: {confidence*100:.2f}%

Class Probabilities:
"""
    
    for class_name in class_names:
        prob = prediction['probabilities'][class_name]
        bar = '█' * int(prob * 20)
        pred_text += f"  {class_name:15s}: {prob*100:5.2f}% {bar}\n"
    
    # Ground truth (if filename has label)
    if image_path.name.startswith('N '):
        true_class = 'no_defect'
        correct = (predicted_class == 'no_defect')
    elif image_path.name.startswith('P '):
        true_class = 'defect'
        correct = (predicted_class == 'defect')
    else:
        true_class = 'unknown'
        correct = None
    
    if true_class != 'unknown':
        pred_text += f"\nGround Truth: {true_class.upper()}\n"
        if correct:
            pred_text += "Result: [CORRECT]\n"
        else:
            pred_text += "Result: [INCORRECT]\n"
    
    # Add box with prediction
    bbox_props = dict(boxstyle='round,pad=1', facecolor=color, alpha=0.2, edgecolor=color, linewidth=2)
    ax2.text(0.5, 0.5, pred_text, transform=ax2.transAxes,
             fontsize=11, verticalalignment='center', horizontalalignment='center',
             family='monospace', bbox=bbox_props)
    
    # Add title based on correctness
    if correct is not None:
        title_color = 'green' if correct else 'red'
        title = 'Correct Prediction' if correct else 'Incorrect Prediction'
        ax2.set_title(title, fontsize=14, fontweight='bold', color=title_color)
    else:
        ax2.set_title('Prediction', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"💾 Saved visualization to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Predict defects in Ball Screw Drive images')
    parser.add_argument('--image', '-i', type=str, nargs='+', required=True,
                       help='Path(s) to image file(s) to predict')
    parser.add_argument('--experiment', '-e', type=str, default=None,
                       help='Experiment directory name (default: use latest)')
    parser.add_argument('--save', '-s', action='store_true',
                       help='Save visualization instead of displaying')
    parser.add_argument('--output_dir', '-o', type=str, default='predictions',
                       help='Output directory for saved predictions (default: predictions/)')
    
    args = parser.parse_args()
    
    # Setup device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("💻 Using GPU")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("💻 Using Apple Silicon GPU (MPS)")
    else:
        device = torch.device("cpu")
        print("💻 Using CPU")
    
    # Find experiment directory
    project_root = Path(__file__).parent.parent
    results_dir = project_root / "code" / "experiments" / "results"
    
    if args.experiment:
        experiment_dir = results_dir / args.experiment
        if not experiment_dir.exists():
            print(f"❌ Experiment directory not found: {experiment_dir}")
            return
    else:
        print("🔍 Looking for latest experiment...")
        experiment_dir = load_latest_experiment(results_dir)
    
    print(f"📂 Using experiment: {experiment_dir.name}\n")
    
    # Load model
    model, config = load_model_and_config(experiment_dir, device)
    
    # Get class names and image size from config
    class_names = config['config']['data']['class_names']
    image_size = tuple(config['config']['data']['image_size'])
    
    print(f"\n📸 Image size: {image_size}")
    print(f"🏷️  Classes: {class_names}\n")
    
    # Setup output directory if saving
    if args.save:
        output_dir = project_root / args.output_dir
        output_dir.mkdir(exist_ok=True)
        print(f"💾 Saving predictions to: {output_dir}\n")
    
    # Process each image
    for image_path_str in args.image:
        image_path = Path(image_path_str)
        
        if not image_path.exists():
            # Try relative to project root
            image_path = project_root / image_path_str
            if not image_path.exists():
                print(f"❌ Image not found: {image_path_str}")
                continue
        
        print(f"🔮 Predicting: {image_path.name}")
        
        # Preprocess
        try:
            image, image_tensor = preprocess_image(image_path, image_size)
        except Exception as e:
            print(f"❌ Error loading image: {e}")
            continue
        
        # Predict
        prediction = predict_single_image(model, image_tensor, device, class_names)
        
        # Print results
        print(f"   Predicted: {prediction['predicted_class']} ({prediction['confidence']*100:.2f}%)")
        for cls_name in class_names:
            prob = prediction['probabilities'][cls_name]
            print(f"     {cls_name}: {prob*100:.2f}%")
        
        # Visualize
        if args.save:
            save_path = output_dir / f"{image_path.stem}_prediction.png"
        else:
            save_path = None
        
        visualize_prediction(image, image_path, prediction, class_names, save_path)
        print()
    
    print("✅ Prediction complete!")


if __name__ == "__main__":
    main()

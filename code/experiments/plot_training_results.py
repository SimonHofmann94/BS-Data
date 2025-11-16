#!/usr/bin/env python3
"""
Script to visualize training results from experiment results.json file.
Creates a comprehensive figure with key insights from the training run.
"""

import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def load_results(results_path):
    """Load results.json file."""
    with open(results_path, 'r') as f:
        return json.load(f)


def plot_training_results(results, save_path=None):
    """
    Create a comprehensive visualization of training results.
    
    Args:
        results: Dictionary containing experiment results
        save_path: Optional path to save the figure
    """
    # Extract data
    epochs = results['training_history']['epoch']
    train_loss = results['training_history']['train_loss']
    val_loss = results['training_history']['val_loss']
    
    # Extract validation metrics history
    val_f1_macro = [m['f1_macro'] for m in results['metrics_history']['val_metrics']]
    val_precision_macro = [m['precision_macro'] for m in results['metrics_history']['val_metrics']]
    val_recall_macro = [m['recall_macro'] for m in results['metrics_history']['val_metrics']]
    val_accuracy = [m['accuracy'] for m in results['metrics_history']['val_metrics']]
    
    # Get class names
    class_names = results['config']['data']['class_names']
    
    # Get best epoch metrics (per-class)
    best_epoch = results['best_epoch']
    best_metrics = results['best_val_metrics']
    
    # Extract per-class F1 scores from best epoch
    per_class_f1 = [best_metrics[f'{cls}_f1'] for cls in class_names]
    per_class_precision = [best_metrics[f'{cls}_precision'] for cls in class_names]
    per_class_recall = [best_metrics[f'{cls}_recall'] for cls in class_names]
    
    # Get test metrics
    test_metrics = results['test_metrics']
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Loss curves (top left, larger)
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(epochs, train_loss, 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, val_loss, 'r-', label='Val Loss', linewidth=2)
    ax1.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.7, label=f'Best Epoch ({best_epoch})')
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Loss', fontsize=11)
    ax1.set_title('Training and Validation Loss', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 2. F1-Macro progression (top right)
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.plot(epochs, val_f1_macro, 'g-', linewidth=2)
    ax2.axvline(x=best_epoch, color='orange', linestyle='--', alpha=0.7)
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('F1-Macro', fontsize=11)
    ax2.set_title('Validation F1-Macro', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.05])
    
    # 3. Precision/Recall progression (middle left)
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(epochs, val_precision_macro, 'b-', label='Precision', linewidth=2)
    ax3.plot(epochs, val_recall_macro, 'r-', label='Recall', linewidth=2)
    ax3.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.7)
    ax3.set_xlabel('Epoch', fontsize=11)
    ax3.set_ylabel('Score', fontsize=11)
    ax3.set_title('Precision & Recall (Macro)', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 1.05])
    
    # 4. Per-class F1 scores at best epoch (middle center-right, larger)
    ax4 = fig.add_subplot(gs[1, 1:])
    x_pos = np.arange(len(class_names))
    bars = ax4.bar(x_pos, per_class_f1, color='steelblue', alpha=0.8)
    ax4.set_xlabel('Class', fontsize=11)
    ax4.set_ylabel('F1 Score', fontsize=11)
    ax4.set_title(f'Per-Class F1 Scores (Best Epoch: {best_epoch})', fontsize=12, fontweight='bold')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(class_names, rotation=45, ha='right', fontsize=9)
    ax4.set_ylim([0, 1.05])
    ax4.grid(True, alpha=0.3, axis='y')
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    # 5. Per-class precision/recall comparison (bottom left)
    ax5 = fig.add_subplot(gs[2, 0])
    x_pos = np.arange(len(class_names))
    width = 0.35
    bars1 = ax5.bar(x_pos - width/2, per_class_precision, width, label='Precision', color='skyblue', alpha=0.8)
    bars2 = ax5.bar(x_pos + width/2, per_class_recall, width, label='Recall', color='lightcoral', alpha=0.8)
    ax5.set_xlabel('Class', fontsize=11)
    ax5.set_ylabel('Score', fontsize=11)
    ax5.set_title(f'Per-Class Precision vs Recall (Epoch {best_epoch})', fontsize=12, fontweight='bold')
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(class_names, rotation=45, ha='right', fontsize=9)
    ax5.legend(fontsize=9)
    ax5.set_ylim([0, 1.05])
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. Test vs Validation comparison (bottom center)
    ax6 = fig.add_subplot(gs[2, 1])
    metrics_to_compare = ['F1-Macro', 'Precision', 'Recall', 'Accuracy']
    val_scores = [best_metrics['f1_macro'], best_metrics['precision_macro'], 
                  best_metrics['recall_macro'], best_metrics['accuracy']]
    test_scores = [test_metrics['f1_macro'], test_metrics['precision_macro'],
                   test_metrics['recall_macro'], test_metrics['accuracy']]
    
    x_pos = np.arange(len(metrics_to_compare))
    width = 0.35
    bars1 = ax6.bar(x_pos - width/2, val_scores, width, label='Validation', color='steelblue', alpha=0.8)
    bars2 = ax6.bar(x_pos + width/2, test_scores, width, label='Test', color='orange', alpha=0.8)
    ax6.set_ylabel('Score', fontsize=11)
    ax6.set_title('Validation vs Test Performance', fontsize=12, fontweight='bold')
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels(metrics_to_compare, fontsize=10)
    ax6.legend(fontsize=9)
    ax6.set_ylim([0, 1.05])
    ax6.grid(True, alpha=0.3, axis='y')
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    # 7. Summary statistics (bottom right)
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.axis('off')
    
    # Prepare summary text
    summary_text = f"""
EXPERIMENT SUMMARY
{'='*30}

Model: {results['model']}
Loss: {results['loss']}
Device: {results['device'].upper()}

Parameters:
  Total: {results['num_model_parameters']:,}
  Trainable: {results['num_trainable_parameters']:,}
  Frozen: {results['num_model_parameters'] - results['num_trainable_parameters']:,}

Dataset Split:
  Train: {results['split_info']['train']['total_samples']} samples
  Val: {results['split_info']['val']['total_samples']} samples
  Test: {results['split_info']['test']['total_samples']} samples

Best Epoch: {best_epoch}/{results['config']['training']['num_epochs']}

Best Val Metrics:
  F1-Macro: {best_metrics['f1_macro']:.4f}
  Accuracy: {best_metrics['accuracy']:.4f}

Test Metrics:
  F1-Macro: {test_metrics['f1_macro']:.4f}
  Accuracy: {test_metrics['accuracy']:.4f}
"""
    
    ax7.text(0.05, 0.95, summary_text, transform=ax7.transAxes,
             fontsize=9, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # Main title
    fig.suptitle(f"Training Results: {results['experiment_name']}", 
                 fontsize=14, fontweight='bold', y=0.98)
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize training results from results.json')
    parser.add_argument('results_path', type=str, nargs='?', 
                       help='Path to results.json file')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output path for the figure (default: saves in same directory as results.json)')
    
    args = parser.parse_args()
    
    # If no path provided, try to find the most recent experiment
    if args.results_path is None:
        experiments_dir = Path(__file__).parent / 'results'
        if experiments_dir.exists():
            # Find all experiment directories
            exp_dirs = sorted([d for d in experiments_dir.iterdir() if d.is_dir() and d.name.startswith('experiment_')],
                            key=lambda x: x.name, reverse=True)
            if exp_dirs:
                results_path = exp_dirs[0] / 'results.json'
                print(f"Using most recent experiment: {results_path}")
            else:
                print("No experiment directories found. Please provide a results.json path.")
                return
        else:
            print("No experiments directory found. Please provide a results.json path.")
            return
    else:
        results_path = Path(args.results_path)
    
    if not results_path.exists():
        print(f"Error: {results_path} not found")
        return
    
    # Load results
    results = load_results(results_path)
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = results_path.parent / 'training_visualization.png'
    
    # Create visualization
    plot_training_results(results, save_path=output_path)


if __name__ == '__main__':
    main()

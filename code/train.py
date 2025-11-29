"""
Main training entry point.

This is the clean, minimal script that users interact with.
All complexity is abstracted into modular components.

Usage:
    python code/train.py
    python code/train.py model.name=convnext_tiny_cbam loss.type=focal_loss
    python code/train.py data.batch_size=8 training.num_epochs=50
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

# Add code directory to path
sys.path.insert(0, str(Path(__file__).parent))

from core.models.registry import get_registry as get_model_registry
from core.losses.registry import get_registry as get_loss_registry
from core.augmentation import get_augmentation_pipeline
from core.data import SeverstalFullImageDataset, StratifiedSplitter, create_dataloaders
from core.training import Trainer
import torch.optim as optim

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def prepare_experiment_directory(cfg: DictConfig, project_root: Path) -> Path:
    """Create (or reuse) the folder that will store artifacts for this run."""
    base_dir = Path(cfg.experiment.save_dir)
    if not base_dir.is_absolute():
        base_dir = project_root / base_dir
    base_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if cfg.experiment.name:
        sanitized = "".join(
            ch if ch.isalnum() or ch in ("-", "_", ".", "=") else "_"
            for ch in str(cfg.experiment.name)
        )
        dir_name = f"experiment_{sanitized}_{timestamp}"
    else:
        dir_name = f"experiment_{timestamp}"

    experiment_dir = base_dir / dir_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Persist location inside the config for downstream scripts (predict, analysis, etc.)
    cfg.experiment.output_dir = str(experiment_dir)

    logger.info(f"Artifacts will be stored in: {experiment_dir}")
    return experiment_dir


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seed set to {seed}")


def setup_device() -> torch.device:
    """Setup compute device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f}GB")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Apple Silicon GPU (MPS)")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
    
    return device


def calculate_split_statistics(
    train_labels: np.ndarray,
    val_labels: np.ndarray,
    test_labels: np.ndarray,
    class_names: list
) -> dict:
    """
    Calculate detailed statistics for train/val/test splits.
    
    Args:
        train_labels: Label matrix for train set (N_train, C)
        val_labels: Label matrix for val set (N_val, C)
        test_labels: Label matrix for test set (N_test, C)
        class_names: List of class names
    
    Returns:
        Dictionary with split statistics
    """
    def get_class_counts(labels, class_names):
        """Get count of samples for each class."""
        counts = {}
        for i, class_name in enumerate(class_names):
            counts[class_name] = int(labels[:, i].sum())
        return counts
    
    split_info = {
        "train": {
            "total_samples": int(len(train_labels)),
            "class_counts": get_class_counts(train_labels, class_names)
        },
        "val": {
            "total_samples": int(len(val_labels)),
            "class_counts": get_class_counts(val_labels, class_names)
        },
        "test": {
            "total_samples": int(len(test_labels)),
            "class_counts": get_class_counts(test_labels, class_names)
        },
        "total": {
            "total_samples": int(len(train_labels) + len(val_labels) + len(test_labels)),
            "class_counts": get_class_counts(
                np.vstack([train_labels, val_labels, test_labels]),
                class_names
            )
        }
    }
    
    return split_info


def load_data(cfg: DictConfig, device: torch.device) -> tuple:
    """
    Load and prepare data.
    
    Args:
        cfg: Configuration
        device: Compute device
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    
    logger.info("\n" + "="*60)
    logger.info("Loading Data")
    logger.info("="*60)
    
    # Get image and annotation directories
    project_root = Path(__file__).parent.parent
    img_dir = project_root / cfg.data.img_dir
    ann_dir = project_root / cfg.data.ann_dir
    
    logger.info(f"Image directory: {img_dir}")
    logger.info(f"Annotation directory: {ann_dir}")
    
    # Get list of image files
    all_image_files = sorted([
        f for f in os.listdir(img_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])
    
    logger.info(f"Found {len(all_image_files)} images")
    
    # Get all labels
    dataset_full = SeverstalFullImageDataset(
        img_dir=str(img_dir),
        ann_dir=str(ann_dir),
        image_names=all_image_files,
        transform=None,
        num_classes=cfg.data.num_classes
    )
    
    # Extract labels
    all_labels = np.array([
        sample["label"] for sample in dataset_full.samples
    ])
    all_image_names = np.array([
        sample["image_name"] for sample in dataset_full.samples
    ])
    
    logger.info(f"Label matrix shape: {all_labels.shape}")
    
    # Stratified split
    split_strategy = cfg.data.split_strategy
    if split_strategy == "stratified_70_15_15":
        split_ratios = (0.7, 0.15, 0.15)
    elif split_strategy == "stratified_80_10_10":
        split_ratios = (0.8, 0.1, 0.1)
    else:
        raise ValueError(f"Unknown split strategy: {split_strategy}")
    
    logger.info(f"Using split strategy: {split_strategy} {split_ratios}")
    
    # Initialize splitter
    splitter = StratifiedSplitter(random_state=cfg.experiment.seed)
    
    # Try to load existing splits first
    splits_dir = project_root / cfg.data.splits_dir
    use_saved_splits = cfg.data.get('use_saved_splits', True)
    
    if use_saved_splits:
        logger.info(f"Attempting to load splits from {splits_dir}")
        loaded_splits = splitter.load_splits(str(splits_dir), all_image_names)
        
        if loaded_splits is not None:
            train_idx, val_idx, test_idx = loaded_splits
            logger.info("✅ Using saved splits")
        else:
            logger.info("No saved splits found. Creating new splits...")
            train_idx, val_idx, test_idx = splitter.split(
                all_labels,
                split_ratios=split_ratios
            )
            # Save the newly created splits
            splitter.save_splits(
                train_idx, val_idx, test_idx,
                all_image_names,
                save_dir=str(splits_dir),
                split_name=split_strategy
            )
            logger.info("✅ New splits created and saved")
    else:
        logger.info("Creating new splits (use_saved_splits=False)")
        train_idx, val_idx, test_idx = splitter.split(
            all_labels,
            split_ratios=split_ratios
        )
    
    # Create subsets
    train_image_names = all_image_names[train_idx].tolist()
    val_image_names = all_image_names[val_idx].tolist()
    test_image_names = all_image_names[test_idx].tolist()
    
    # Setup augmentations - ONLY for training!
    train_transform = get_augmentation_pipeline(
        image_size=tuple(cfg.data.image_size),
        augmentation_config=OmegaConf.to_container(cfg.augmentation)
    )
    
    # Validation and test transforms - NO augmentation, only normalization
    eval_transform = get_augmentation_pipeline(
        image_size=tuple(cfg.data.image_size),
        augmentation_config={}  # Empty config = only ToTensor + Normalize
    )
    
    # Create datasets
    train_dataset = SeverstalFullImageDataset(
        img_dir=str(img_dir),
        ann_dir=str(ann_dir),
        image_names=train_image_names,
        transform=train_transform,  # With augmentation
        num_classes=cfg.data.num_classes
    )
    
    val_dataset = SeverstalFullImageDataset(
        img_dir=str(img_dir),
        ann_dir=str(ann_dir),
        image_names=val_image_names,
        transform=eval_transform,  # NO augmentation
        num_classes=cfg.data.num_classes
    )
    
    test_dataset = SeverstalFullImageDataset(
        img_dir=str(img_dir),
        ann_dir=str(ann_dir),
        image_names=test_image_names,
        transform=eval_transform,  # NO augmentation
        num_classes=cfg.data.num_classes
    )
    
    logger.info(f"Train set: {len(train_dataset)} samples")
    logger.info(f"Val set: {len(val_dataset)} samples")
    logger.info(f"Test set: {len(test_dataset)} samples")
    
    # Calculate split statistics for results.json
    split_info = calculate_split_statistics(
        train_labels=all_labels[train_idx],
        val_labels=all_labels[val_idx],
        test_labels=all_labels[test_idx],
        class_names=cfg.data.class_names
    )
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset,
        val_dataset,
        test_dataset,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory
    )
    
    return train_loader, val_loader, test_loader, split_info


def build_model(cfg: DictConfig, device: torch.device) -> torch.nn.Module:
    """Build model from configuration."""
    
    logger.info("\n" + "="*60)
    logger.info("Building Model")
    logger.info("="*60)
    
    model_registry = get_model_registry()
    
    logger.info(f"Model: {cfg.model.name}")
    logger.info(f"Available models: {list(model_registry.list_models().keys())}")
    
    model = model_registry.get(
        cfg.model.name,
        num_classes=cfg.model.num_classes,
        pretrained=cfg.model.pretrained,
        cbam_stages=cfg.model.cbam_stages
    )
    
    # Freeze backbone if requested
    if cfg.training.freeze_backbone:
        logger.info("Freezing backbone weights")
        model.freeze_backbone(freeze=True)
    
    if cfg.training.freeze_early_stages is not None:
        logger.info(f"Freezing first {cfg.training.freeze_early_stages} stages")
        model.freeze_early_stages(num_stages=cfg.training.freeze_early_stages)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    return model


def build_loss(cfg: DictConfig) -> torch.nn.Module:
    """Build loss function from configuration."""
    
    logger.info("\n" + "="*60)
    logger.info("Building Loss Function")
    logger.info("="*60)
    
    loss_registry = get_loss_registry()
    
    logger.info(f"Loss: {cfg.loss.type}")
    logger.info(f"Available losses: {list(loss_registry.list_losses().keys())}")
    
    # Build loss parameters based on loss type
    loss_params = {
        "num_classes": cfg.data.num_classes,
        "reduction": cfg.loss.reduction
    }
    
    # Add type-specific parameters
    if cfg.loss.type == "focal_loss":
        # Convert alpha from OmegaConf to native Python type
        alpha = cfg.loss.alpha
        if isinstance(alpha, (list, tuple)) or OmegaConf.is_list(alpha):
            alpha = list(alpha)  # Convert OmegaConf ListConfig to Python list
            logger.info(f"Converted alpha from config: {alpha}")
        
        loss_params["alpha"] = alpha
        loss_params["gamma"] = cfg.loss.gamma
        
        # Add optional Class-Balanced Loss parameters
        if hasattr(cfg.loss, 'use_effective_num'):
            loss_params['use_effective_num'] = cfg.loss.use_effective_num
        if hasattr(cfg.loss, 'beta'):
            loss_params['beta'] = cfg.loss.beta
            
    elif cfg.loss.type == "cross_entropy":
        # CrossEntropy-specific parameters
        if hasattr(cfg.loss, 'weight') and cfg.loss.weight is not None:
            loss_params['weight'] = cfg.loss.weight
    
    elif cfg.loss.type == "bce_with_logits":
        # BCE-specific parameters
        if hasattr(cfg.loss, 'pos_weight') and cfg.loss.pos_weight is not None:
            loss_params['pos_weight'] = cfg.loss.pos_weight
    
    loss_fn = loss_registry.get(cfg.loss.type, **loss_params)
    
    logger.info(f"Loss configuration: {loss_fn.log_info()}")
    
    return loss_fn


def build_optimizer(cfg: DictConfig, model: torch.nn.Module) -> torch.optim.Optimizer:
    """Build optimizer from configuration."""
    
    logger.info("\n" + "="*60)
    logger.info("Building Optimizer")
    logger.info("="*60)
    
    if cfg.optimizer.type.lower() == "adamw":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=cfg.optimizer.lr,
            weight_decay=cfg.optimizer.weight_decay,
            betas=cfg.optimizer.betas
        )
    elif cfg.optimizer.type.lower() == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=cfg.optimizer.lr,
            weight_decay=cfg.optimizer.weight_decay
        )
    elif cfg.optimizer.type.lower() == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=cfg.optimizer.lr,
            weight_decay=cfg.optimizer.weight_decay,
            momentum=0.9
        )
    else:
        raise ValueError(f"Unknown optimizer: {cfg.optimizer.type}")
    
    logger.info(f"Optimizer: {cfg.optimizer.type}")
    logger.info(f"Learning rate: {cfg.optimizer.lr}")
    logger.info(f"Weight decay: {cfg.optimizer.weight_decay}")
    
    return optimizer


def main(cfg: DictConfig) -> None:
    """Main training pipeline."""
    
    logger.info("\n" + "="*70)
    logger.info("SEVERSTAL DEFECT CLASSIFICATION - TRAINING PIPELINE")
    logger.info("="*70)
    project_root = Path(__file__).parent.parent
    
    # Print configuration
    logger.info("\nConfiguration:")
    logger.info(OmegaConf.to_yaml(cfg))
    
    # Set seed
    set_seed(cfg.experiment.seed)
    
    # Setup device
    device = setup_device()
    
    # Load data
    train_loader, val_loader, test_loader, split_info = load_data(cfg, device)
    
    # Build model
    model = build_model(cfg, device)
    
    # Build loss
    loss_fn = build_loss(cfg)
    
    # Build optimizer
    optimizer = build_optimizer(cfg, model)
    
    # Prepare experiment directory for artifacts (mirrors Hydra's run dir behavior)
    experiment_dir = prepare_experiment_directory(cfg, project_root)

    # Persist the resolved config alongside the run for reproducibility
    config_snapshot_path = experiment_dir / "config.yaml"
    with open(config_snapshot_path, "w") as f:
        f.write(OmegaConf.to_yaml(cfg))
    logger.info(f"Saved resolved config to {config_snapshot_path}")

    # Create trainer
    logger.info("\n" + "="*60)
    logger.info("Initializing Trainer")
    logger.info("="*60)
    
    # Convert OmegaConf to dict for JSON serialization
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        scheduler=None,  # Will be created in train()
        device=device,
        class_names=cfg.data.class_names,
        config=config_dict,  # Pass config for saving
        split_info=split_info,  # Pass split statistics
        experiment_dir=experiment_dir,
        save_checkpoints=cfg.training.get("save_checkpoints", True)
    )
    
    # Train
    results = trainer.train(
        num_epochs=cfg.training.num_epochs,
        early_stopping_patience=cfg.training.early_stopping_patience,
        warmup_epochs=cfg.training.warmup_epochs,
        log_interval=cfg.training.log_interval,
        threshold=cfg.training.threshold
    )
    
    logger.info("\n" + "="*70)
    logger.info("TRAINING COMPLETED SUCCESSFULLY")
    logger.info("="*70)
    logger.info(f"Experiment directory: {trainer.experiment_dir}")
    
    # Generate visualization plot
    logger.info("\n" + "="*70)
    logger.info("GENERATING VISUALIZATION")
    logger.info("="*70)
    try:
        from experiments.plot_training_results import plot_training_results
        
        results_file = Path(trainer.experiment_dir) / "results.json"
        viz_file = Path(trainer.experiment_dir) / "training_visualization.png"
        
        if results_file.exists():
            import json
            with open(results_file, 'r') as f:
                results_data = json.load(f)
            
            plot_training_results(results_data, save_path=str(viz_file))
            logger.info(f"✅ Visualization saved to: {viz_file}")
        else:
            logger.warning(f"Results file not found: {results_file}")
    except Exception as e:
        logger.exception("Failed to generate visualization")
        logger.info("You can manually generate it with:")
        logger.info(f"  ./venv/bin/python3 code/experiments/plot_training_results.py {results_file}")


if __name__ == "__main__":
    # Load config using Hydra
    config_dir = Path(__file__).parent.parent / "config"
    
    try:
        with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
            # Honor CLI overrides, e.g., "optimizer.lr=1e-4 experiment.name=my_run"
            cfg = compose(config_name="train_config", overrides=sys.argv[1:])
            main(cfg)
    except Exception as e:
        logger.exception("An error occurred during the training pipeline.")
        sys.exit(1)
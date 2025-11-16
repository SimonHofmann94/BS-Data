"""
CrossEntropyLoss wrapper for single-label classification.

Provides a consistent interface with other loss functions.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
import logging

from .base import BaseLoss

logger = logging.getLogger(__name__)


class CrossEntropyLossWrapper(BaseLoss):
    """
    Wrapper for CrossEntropyLoss for single-label classification.
    
    CrossEntropyLoss combines LogSoftmax and NLLLoss in a numerically stable way.
    Suitable for single-label multi-class classification.
    
    Args:
        num_classes: Number of classes
        weight: Class weights (shape: (C,) or None).
               If None, uniform weights are used.
        reduction: 'mean', 'sum', or 'none' (default: 'mean')
    
    Example:
        >>> loss_fn = CrossEntropyLossWrapper(num_classes=6)
        >>> logits = torch.randn(32, 6)
        >>> targets = torch.randint(0, 6, (32,))  # Class indices
        >>> loss = loss_fn(logits, targets)
    """
    
    def __init__(
        self,
        num_classes: int,
        weight: Optional[torch.Tensor] = None,
        reduction: str = 'mean'
    ):
        super().__init__(num_classes)
        self.reduction = reduction
        self.weight = weight
        
        # Create underlying loss function
        self.ce_loss = nn.CrossEntropyLoss(
            weight=weight,
            reduction=reduction
        )
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss.
        
        Args:
            predictions: Model predictions (logits) of shape (B, C)
            targets: Target labels. Can be:
                    - Class indices of shape (B,) - standard format
                    - One-hot encoded of shape (B, C) - will be converted to indices
        
        Returns:
            Scalar loss value
        """
        # Handle one-hot encoded targets
        if targets.dim() == 2 and targets.shape[1] == self.num_classes:
            # Convert one-hot to class indices
            targets = torch.argmax(targets, dim=1)
        
        return self.ce_loss(predictions, targets)
    
    def get_params(self) -> Dict[str, Any]:
        """Get loss hyperparameters for logging."""
        weight_val = None
        if self.weight is not None:
            weight_val = self.weight.cpu().numpy().tolist()
        
        return {
            "weight": weight_val,
            "reduction": self.reduction
        }


if __name__ == "__main__":
    # Test CE loss
    B, C = 32, 6
    
    # Create synthetic data
    logits = torch.randn(B, C)
    
    # Test with class indices
    targets_indices = torch.randint(0, C, (B,))
    
    # Test with one-hot encoding
    targets_onehot = torch.zeros(B, C)
    targets_onehot[torch.arange(B), targets_indices] = 1.0
    
    # Create loss function
    loss_fn = CrossEntropyLossWrapper(
        num_classes=C,
        weight=None,
        reduction='mean'
    )
    
    # Compute loss with indices
    loss1 = loss_fn(logits, targets_indices)
    print(f"Loss (indices): {loss1.item():.4f}")
    
    # Compute loss with one-hot
    loss2 = loss_fn(logits, targets_onehot)
    print(f"Loss (one-hot): {loss2.item():.4f}")
    
    # Should be the same
    assert torch.allclose(loss1, loss2), "Losses should match!"
    
    print(f"Loss params: {loss_fn.get_params()}")
    print(f"Loss info: {loss_fn.log_info()}")

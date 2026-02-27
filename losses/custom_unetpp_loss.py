import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.filters import sobel

class MultiClassDiceLoss(nn.Module):
    """
    Computes Dice Loss for multi-class segmentation.
    Returns both the total loss and per-class Dice scores.
    """
    def __init__(self, smooth=1e-6):
        super(MultiClassDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        # preds and targets: (N, C, H, W)
        N, C, H, W = preds.shape
        preds = preds.view(N, C, -1)
        targets = targets.view(N, C, -1)
        intersection = (preds * targets).sum(-1)
        denominator = preds.sum(-1) + targets.sum(-1)
        dice = (2. * intersection + self.smooth) / (denominator + self.smooth)
        return 1 - dice.mean(), dice.mean(dim=0)


class CustomUNetPlusPlusLoss(nn.Module):
    """
    Custom loss for CustomUNetPlusPlus with optional deep supervision.
    Combines Dice loss, Focal loss, Boundary loss, and Presence loss.
    """
    def __init__(self, alpha=0.5, gamma=2.0, smooth=1e-6, class_weights=None, weights=None, deep_supervision_weight=0.2):
        """
        Args:
            alpha (float): Weight for Dice loss in the segmentation loss.
            gamma (float): Gamma parameter for Focal loss.
            smooth (float): Smoothing factor for Dice loss.
            class_weights (torch.Tensor): Class weights for segmentation loss.
            weights (dict): Weights for different loss components (dice, ce, boundary, presence).
            deep_supervision_weight (float): Weight for deep supervision losses.
        """
        super(CustomUNetPlusPlusLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
        self.class_weights = class_weights
        self.weights = weights if weights is not None else {
            'dice': 1.5,
            'ce': 1.0,
            'boundary': 0.1,
            'presence': 0.2
        }
        self.deep_supervision_weight = deep_supervision_weight
        self.dice_loss_fn = MultiClassDiceLoss(smooth=smooth)
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        self.bce_loss = nn.BCEWithLogitsLoss()

    def focal_loss(self, preds, targets):
        """
        Compute Focal loss for segmentation.
        """
        bce_loss = F.binary_cross_entropy_with_logits(preds, targets, reduction="none")
        pt = torch.exp(-bce_loss)
        focal_loss = (1 - pt) ** self.gamma * bce_loss
        if self.class_weights is not None:
            focal_loss *= self.class_weights.view(1, -1, 1, 1).to(focal_loss.device)
        return focal_loss.mean()

    def boundary_loss(self, preds, targets):
        """
        Compute boundary loss using Sobel operator.
        """
        pred_edges = sobel(torch.sigmoid(preds))
        target_edges = sobel(targets)
        intersection = (pred_edges * target_edges).sum(dim=(2, 3))
        union = pred_edges.sum(dim=(2, 3)) + target_edges.sum(dim=(2, 3))
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

    def presence_loss(self, preds, targets):
        """
        Compute presence loss for class presence detection.
        """
        return F.binary_cross_entropy(preds, targets.float())

    def forward(self, outputs, targets, deep_supervision=False):
        """
        Compute the loss for CustomUNetPlusPlus.

        Args:
            outputs (tuple): Model outputs. If deep supervision is enabled, includes deep supervision outputs.
            targets (torch.Tensor): Ground truth segmentation masks.
            deep_supervision (bool): Whether deep supervision is enabled.

        Returns:
            total_loss (torch.Tensor): Combined loss.
            loss_details (dict): Individual loss components for logging.
        """
        if deep_supervision:
            pred_masks, pred_boundaries, pred_presence, deep_outs = outputs
        else:
            pred_masks, pred_boundaries, pred_presence = outputs

        # Compute segmentation losses
        dice_loss, per_class_dice = self.dice_loss_fn(torch.sigmoid(pred_masks), targets)
        focal_loss = self.focal_loss(pred_masks, targets)
        ce_loss = self.ce_loss(pred_masks, targets.argmax(dim=1))  # For multi-class segmentation

        # Compute boundary and presence losses
        loss_boundary = self.boundary_loss(pred_boundaries, targets)
        target_presence = (targets.sum(dim=(2, 3)) > 0).float()
        loss_presence = self.presence_loss(pred_presence, target_presence)

        # Combine losses
        total_loss = (
            self.weights['dice'] * (self.alpha * dice_loss + (1 - self.alpha) * focal_loss) +
            self.weights['ce'] * ce_loss +
            self.weights['boundary'] * loss_boundary +
            self.weights['presence'] * loss_presence
        )

        # Add deep supervision losses if enabled
        if deep_supervision:
            deep_supervision_loss = 0
            for deep_out in deep_outs:
                deep_dice_loss, _ = self.dice_loss_fn(torch.sigmoid(deep_out), targets)
                deep_ce_loss = self.ce_loss(deep_out, targets.argmax(dim=1))
                deep_supervision_loss += deep_dice_loss + deep_ce_loss
            
            # Synchronize across GPUs before averaging for DDP consistency
            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(deep_supervision_loss)
                deep_supervision_loss /= torch.distributed.get_world_size()
            
            deep_supervision_loss /= len(deep_outs)  # Average over all deep supervision outputs
            total_loss += self.deep_supervision_weight * deep_supervision_loss

        # Return total loss and individual loss components for logging
        loss_details = {
            'dice_loss': dice_loss.item(),
            'focal_loss': focal_loss.item(),
            'ce_loss': ce_loss.item(),
            'boundary_loss': loss_boundary.item(),
            'presence_loss': loss_presence.item(),
            'total_loss': total_loss.item()
        }
        return total_loss, loss_details
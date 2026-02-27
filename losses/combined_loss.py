import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.filters import sobel

class MultiClassDiceLoss(nn.Module):
    """
    Computes Dice Loss for multi-class segmentation.
    Returns both the total loss and the per-class Dice Score.
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

class CorrectedAdvancedCombinedLoss(nn.Module):
    """
    Combined loss for segmentation integrating focal loss and Dice loss.
    Applied to the segmentation output (without directly including the boundary loss).
    """
    def __init__(self, alpha=0.5, gamma=2.0, smooth=1e-6, class_weights=None):
        super(CorrectedAdvancedCombinedLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
        self.dice_loss = MultiClassDiceLoss(smooth=smooth)
        self.class_weights = class_weights

    def focal_loss(self, preds, targets):
        # Apply BCE with logits, which combines sigmoid + binary cross entropy.
        bce_loss = F.binary_cross_entropy_with_logits(preds, targets, reduction="none")
        pt = torch.exp(-bce_loss)
        focal_loss = (1 - pt) ** self.gamma * bce_loss
        if self.class_weights is not None:
            focal_loss *= self.class_weights.view(1, -1, 1, 1).to(focal_loss.device)
        return focal_loss.mean()

    def forward(self, preds, targets):
        # preds are the raw logits from the segmentation branch.
        focal = self.focal_loss(preds, targets)
        dice, per_class_dice = self.dice_loss(torch.sigmoid(preds), targets)
        loss_main = self.alpha * dice + (1 - self.alpha) * focal
        return loss_main, per_class_dice

class BoundaryLoss(nn.Module):
    """
    Boundary loss for segmentation, computed using the Sobel operator.
    Applied to the output of the model's boundary branch.
    """
    def __init__(self, epsilon=1e-6):
        super(BoundaryLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, preds, targets):
        # Apply sigmoid on boundary predictions to obtain probabilities.
        pred_edges = sobel(torch.sigmoid(preds))
        target_edges = sobel(targets)
        intersection = (pred_edges * target_edges).sum(dim=(2, 3))
        union = pred_edges.sum(dim=(2, 3)) + target_edges.sum(dim=(2, 3))
        dice = (2. * intersection + self.epsilon) / (union + self.epsilon)
        return 1 - dice.mean()

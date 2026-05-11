import random
import numpy as np
import torch
import torch.nn as nn

DATA_ROOT = "/home/abyhu/parallelproj/ai4mars-dataset-merged-0.6/ai4mars-dataset-merged-0.6"
NUM_CLASSES = 5
IGNORE_INDEX = 255
BATCH_SIZE = 8          # 8 may OOM at 512x512; drop to 2 if needed
EPOCHS = 100
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.05
SEED = 42
IMG_SIZE = 512

# Inverse-frequency proxy weights based on observed IoU after 60 epochs:
# soil≈0.77, bedrock≈0.71, sand≈0.36, big_rock≈0.19, rover_track≈0.30
CLASS_WEIGHTS = [1.0, 1.0, 2.5, 5.0, 2.0]

CLASS_NAMES = ["soil", "bedrock", "sand", "big_rock", "rover_track"]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_iou(preds, labels, num_classes, ignore_index):
    preds = torch.argmax(preds.float(), dim=1)
    ious = []
    for cls in range(num_classes):
        pred_inds = preds == cls
        target_inds = labels == cls
        valid_mask = labels != ignore_index
        pred_inds = pred_inds & valid_mask
        target_inds = target_inds & valid_mask
        intersection = (pred_inds & target_inds).long().sum().item()
        union = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection
        if union == 0:
            ious.append(float("nan"))
        else:
            ious.append(float(intersection) / float(max(union, 1)))
    valid_ious = [iou for iou in ious if not np.isnan(iou)]
    miou = sum(valid_ious) / len(valid_ious) if valid_ious else 0.0
    return miou, ious


class DiceLoss(nn.Module):
    """
    Soft Dice loss averaged over all classes (handles class imbalance better than CE alone).
    Pixels with targets == ignore_index are masked out before computing Dice.
    Works inside torch.autocast — logits are cast to float32 internally.
    Reference: Sudre et al., MICCAI Deep Learning Workshop 2017.
    """

    def __init__(self, num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX, smooth=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Force FP32 — softmax and accumulation are numerically sensitive
        probs = torch.softmax(logits.float(), dim=1)  # (B, C, H, W)

        valid = (targets != self.ignore_index)          # (B, H, W)
        tgt = targets.clone()
        tgt[~valid] = 0                                  # safe index for scatter

        # One-hot encode then zero out ignored pixels
        one_hot = torch.zeros_like(probs)
        one_hot.scatter_(1, tgt.unsqueeze(1), 1.0)
        valid_f = valid.unsqueeze(1).float()
        probs = probs * valid_f
        one_hot = one_hot * valid_f

        # Per-class Dice over batch + spatial dims, then mean across classes
        dims = (0, 2, 3)
        intersection = (probs * one_hot).sum(dim=dims)
        cardinality = probs.sum(dim=dims) + one_hot.sum(dim=dims)
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice.mean()


class CombinedLoss(nn.Module):
    """
    0.5 * CrossEntropyLoss + 0.5 * DiceLoss
    CE handles per-pixel accuracy; Dice handles class imbalance (big_rock, rover_track).
    Both terms use ignore_index=255 to exclude unlabeled pixels.
    class_weights: optional list/tensor of per-class weights for the CE term.
    """

    def __init__(self, num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX, dice_weight=0.5, class_weights=None):
        super().__init__()
        weight_tensor = torch.tensor(class_weights, dtype=torch.float32) if class_weights is not None else None
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index, weight=weight_tensor)
        self.dice = DiceLoss(num_classes=num_classes, ignore_index=ignore_index)
        self.w = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return (1.0 - self.w) * self.ce(logits, targets) + self.w * self.dice(logits, targets)

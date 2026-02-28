"""
model/losses.py

Combined loss for splat enhancement training: L1 + LPIPS + Gram matrix.

- L1: Pixel-level reconstruction. Preferred over L2 because it produces
  sharper results (L2 biases toward blurry averages).
- LPIPS: Perceptual similarity using VGG features. Captures structural
  quality that pixel metrics miss.
- Gram matrix: Style loss computed on VGG feature maps. Captures texture
  statistics, which is specifically useful for restoring skin texture
  from the waxy/plastic look of Gaussian splat renders.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import lpips
from torchvision import models


class VGGFeatureExtractor(nn.Module):
    """Extract intermediate VGG19 features for Gram matrix computation.

    Uses layers from different depths to capture both fine texture
    (early layers) and broader style patterns (deeper layers).
    """

    def __init__(self):
        super().__init__()
        vgg = models.vgg19(pretrained=True).features.eval()

        # Feature blocks at different scales
        # relu1_1, relu2_1, relu3_1, relu4_1
        self.blocks = nn.ModuleList([
            vgg[:2],    # relu1_1: 64 channels, captures edges/texture
            vgg[2:7],   # relu2_1: 128 channels, captures patterns
            vgg[7:12],  # relu3_1: 256 channels, captures larger structures
            vgg[12:21], # relu4_1: 512 channels, captures style
        ])

        for p in self.parameters():
            p.requires_grad = False

        # ImageNet normalization
        self.register_buffer("mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        """Extract features. Input should be in [-1, 1] range."""
        # Convert from [-1, 1] to [0, 1] then normalize for ImageNet
        x = (x + 1) / 2
        x = (x - self.mean) / self.std

        features = []
        for block in self.blocks:
            x = block(x)
            features.append(x)
        return features


def gram_matrix(features):
    """Compute Gram matrix from a feature map.

    The Gram matrix captures correlations between feature channels,
    encoding texture/style information independent of spatial layout.

    Args:
        features: tensor of shape (B, C, H, W)

    Returns:
        Gram matrix of shape (B, C, C), normalized by spatial dimensions
    """
    B, C, H, W = features.shape
    # Reshape to (B, C, H*W)
    f = features.view(B, C, -1)
    # Gram = F * F^T, normalized
    G = torch.bmm(f, f.transpose(1, 2))
    return G / (C * H * W)


class CombinedLoss(nn.Module):
    """
    L1 + LPIPS + Gram matrix loss.

    Args:
        lambda_l1: Weight for L1 pixel loss (default 1.0)
        lambda_lpips: Weight for LPIPS perceptual loss (default 0.5)
        lambda_gram: Weight for Gram matrix style loss (default 0.01)
    """

    def __init__(self, lambda_l1=1.0, lambda_lpips=0.5, lambda_gram=0.01):
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_lpips = lambda_lpips
        self.lambda_gram = lambda_gram

        # LPIPS with VGG backbone
        self.lpips_fn = lpips.LPIPS(net="vgg").cuda()
        self.lpips_fn.requires_grad_(False)

        # VGG features for Gram matrix
        self.vgg = VGGFeatureExtractor().cuda()

    def forward(self, pred, target):
        """
        Compute combined loss.

        Args:
            pred: predicted enhanced image, (B, 3, H, W) in [-1, 1]
            target: ground truth clean image, (B, 3, H, W) in [-1, 1]

        Returns:
            total_loss, dict of individual loss values for logging
        """
        # L1 pixel loss
        loss_l1 = F.l1_loss(pred, target)

        # LPIPS perceptual loss
        loss_lpips = self.lpips_fn(pred, target).mean()

        # Gram matrix style loss (force float32 to avoid fp16 overflow)
        with torch.amp.autocast("cuda", enabled=False):
            pred_features = self.vgg(pred.float())
            target_features = self.vgg(target.float())

            loss_gram = 0
            for pf, tf in zip(pred_features, target_features):
                loss_gram += F.l1_loss(gram_matrix(pf), gram_matrix(tf))
            loss_gram = loss_gram / len(pred_features)

        # Combined
        total = (self.lambda_l1 * loss_l1
                 + self.lambda_lpips * loss_lpips
                 + self.lambda_gram * loss_gram)

        losses = {
            "total": total.item(),
            "l1": loss_l1.item(),
            "lpips": loss_lpips.item(),
            "gram": loss_gram.item(),
        }

        return total, losses

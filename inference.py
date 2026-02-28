"""
inference.py

Inference script for the splat enhancement model.
Runs the trained enhancer on the test set, computes LPIPS metrics,
and produces visual comparison grids (degraded | enhanced | ground truth).

Adapted from the eval loop in train_pix2pix_turbo.py.
"""

import os
import json
import argparse
import torch
import lpips
import numpy as np
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF
from tqdm import tqdm

from model.architecture import SplatEnhancer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run inference and evaluation")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pkl)")
    parser.add_argument("--dataset_folder", type=str, required=True,
                        help="Path to dataset with test_A/ and test_B/")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Directory for metrics and comparison images")
    parser.add_argument("--mixed_precision", action="store_true",
                        help="Use fp16 for inference")
    parser.add_argument("--num_comparisons", type=int, default=10,
                        help="Number of comparison grid images to save")
    return parser.parse_args()


def load_image(path):
    """Load image and normalize to [-1, 1] tensor."""
    img = Image.open(path).convert("RGB")
    t = TF.to_tensor(img)
    t = TF.normalize(t, mean=[0.5], std=[0.5])
    return t


def tensor_to_pil(t):
    """Convert [-1, 1] tensor to PIL image."""
    t = (t.clamp(-1, 1) + 1) / 2
    t = (t * 255).byte()
    return Image.fromarray(t.permute(1, 2, 0).cpu().numpy())


def make_comparison_grid(degraded, enhanced, ground_truth):
    """Create side-by-side comparison: degraded | enhanced | ground truth."""
    w, h = degraded.size
    grid = Image.new("RGB", (w * 3, h))
    grid.paste(degraded, (0, 0))
    grid.paste(enhanced, (w, 0))
    grid.paste(ground_truth, (w * 2, 0))
    return grid


def main():
    args = parse_args()
    os.makedirs(os.path.join(args.output_dir, "comparisons"), exist_ok=True)

    # --- Load model ---
    print(f"Loading checkpoint: {args.checkpoint}")
    model = SplatEnhancer(pretrained_path=args.checkpoint)
    model.set_eval()

    if args.mixed_precision:
        model = model.half()

    # --- LPIPS metric ---
    net_lpips = lpips.LPIPS(net="vgg").cuda()
    net_lpips.requires_grad_(False)

    # --- Get test images ---
    test_a_dir = os.path.join(args.dataset_folder, "test_A")
    test_b_dir = os.path.join(args.dataset_folder, "test_B")
    img_names = sorted([
        f for f in os.listdir(test_a_dir)
        if f.endswith((".png", ".jpg", ".jpeg"))
    ])
    print(f"Running inference on {len(img_names)} test images")

    # --- Run inference ---
    lpips_scores = []

    for i, name in enumerate(tqdm(img_names, desc="Inference")):
        # Load paired images
        input_t = load_image(os.path.join(test_a_dir, name)).unsqueeze(0).cuda()
        target_t = load_image(os.path.join(test_b_dir, name)).unsqueeze(0).cuda()

        if args.mixed_precision:
            input_t = input_t.half()

        # Forward pass
        with torch.no_grad():
            output_t = model(input_t)

        # LPIPS (needs float32)
        with torch.no_grad():
            score = net_lpips(output_t.float(), target_t.float()).item()
            lpips_scores.append(score)

        # Save comparison grids
        if i < args.num_comparisons:
            degraded_pil = tensor_to_pil(input_t[0].float())
            enhanced_pil = tensor_to_pil(output_t[0].float())
            gt_pil = tensor_to_pil(target_t[0].float())

            grid = make_comparison_grid(degraded_pil, enhanced_pil, gt_pil)
            grid.save(os.path.join(
                args.output_dir, "comparisons", f"comparison_{i:03d}.png"))

    # --- Metrics ---
    metrics = {
        "lpips_mean": float(np.mean(lpips_scores)),
        "lpips_std": float(np.std(lpips_scores)),
        "lpips_median": float(np.median(lpips_scores)),
        "num_test_images": len(img_names),
        "checkpoint": args.checkpoint,
    }

    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nResults:")
    print(f"  LPIPS mean:   {metrics['lpips_mean']:.4f}")
    print(f"  LPIPS std:    {metrics['lpips_std']:.4f}")
    print(f"  LPIPS median: {metrics['lpips_median']:.4f}")
    print(f"\nMetrics saved to {metrics_path}")
    print(f"Comparison grids saved to {args.output_dir}/comparisons/")


if __name__ == "__main__":
    main()

"""
data/prepare_data.py

Applies Gaussian splat degradation to pre-processed RGBA images
and splits into train/test pairs.

Degradation uses face-parsing-driven regional blur with anisotropic Gaussian
variation masks to simulate real Gaussian splat rendering artifacts.

Expects rgba_512/ directory to already contain RGBA PNGs with background
removed. Run RMBG preprocessing separately before this script.
"""

import argparse
import torch
import numpy as np
import cv2
import os
import random
from PIL import Image
from transformers import (
    SegformerImageProcessor,
    SegformerForSemanticSegmentation
)



def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare degraded/clean paired dataset")
    parser.add_argument("--output_dir", type=str, default="data",
                        help="Output directory for dataset")
    parser.add_argument("--num_images", type=int, default=500,
                        help="Number of RGBA images to use")
    parser.add_argument("--num_train", type=int, default=450)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


# =========================================================================
# DEGRADATION PARAMS - all tuneable from one place
# =========================================================================
DEGRADE_PARAMS = {
    # Geometric warp
    "warp_strength": 0.9,
    "warp_scale": 30,

    # Per-region blur strength
    "skin_blur_strength": 0.5,
    "eye_blur_strength": 0.8,
    "nose_blur_strength": 0.6,
    "mouth_blur_strength": 0.9,
    "hair_blur_strength": 0.6,

    # Heavy blur for eyes/nose/mouth
    "nuke_kernel": 101,
    "nuke_sigma": 40.0,

    # Variation mask sizes per region
    "skin_points": 200,
    "skin_radius": (20, 50),
    "eye_points": 200,
    "eye_radius": (5, 10),
    "nose_points": 200,
    "nose_radius": (6, 12),
    "mouth_points": 200,
    "mouth_radius": (8, 8),
    "hair_points": 150,
    "hair_radius": (25, 60),

    # Mouth overall blur (flat blur on top of variation pass)
    "mouth_overall_blur_kernel": 41,
    "mouth_overall_blur_sigma": 15.0,
    "mouth_overall_blur_strength": 0.5,

    # Bilateral surface blur
    "bilateral_d": 12,
    "bilateral_sigma_color": 100,
    "bilateral_sigma_space": 100,
    "bilateral_strength": 0.45,

    # Brightness variation
    "brightness_intensity": 15,

    # Floor / ceiling clamps
    "global_floor": 40,
    "global_ceiling": 230,
    "mouth_floor": 60,
    "nose_floor": 50,

    # Overall blur
    "overall_blur_kernel": 15,
    "overall_blur_sigma": 6.0,
    "overall_blur_strength": 0.20,

    # Colour
    "desaturation": 0.2,

    # Downscale + upscale
    "downscale_factor": 2,

    # Noise
    "noise_strength": 4.5,
}


# =========================================================================
# TRAIN/TEST SPLIT (ground truths)
# =========================================================================
def create_ground_truths(output_dir, num_images=500, num_train=450, seed=42):
    train_b = os.path.join(output_dir, "train_B")
    test_b = os.path.join(output_dir, "test_B")
    rgba_dir = os.path.join(output_dir, "rgba_512")

    if os.path.exists(train_b) and len(os.listdir(train_b)) >= num_train:
        print(f"Split already done: {len(os.listdir(train_b))} train, "
              f"{len(os.listdir(test_b))} test")
        return

    os.makedirs(train_b, exist_ok=True)
    os.makedirs(test_b, exist_ok=True)

    indices = list(range(num_images))
    random.seed(seed)
    random.shuffle(indices)

    for j, i in enumerate(indices):
        img_rgba = Image.open(os.path.join(rgba_dir, f"{i:05d}.png"))
        white_bg = Image.new("RGB", img_rgba.size, (255, 255, 255))
        white_bg.paste(img_rgba, mask=img_rgba.split()[3])
        if j < num_train:
            white_bg.save(os.path.join(train_b, f"{j:05d}.png"))
        else:
            white_bg.save(os.path.join(test_b, f"{(j - num_train):05d}.png"))
    print("Split done!")


# =========================================================================
# VARIATION MASK (anisotropic Gaussian blobs)
# =========================================================================
def make_variation_mask(h, w, n_points=120, radius_range=(30, 70)):
    """Generate a patchy variation mask using randomly placed anisotropic
    Gaussian blobs. Mimics the organic egg-shaped patches seen in real
    Gaussian splat renders where each projected 3D Gaussian has different
    scale on each axis plus rotation."""
    points = np.random.rand(n_points, 2) * np.array([w, h])
    values = np.random.uniform(0.0, 1.0, n_points)

    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    mask = np.zeros((h, w), dtype=np.float32)
    total_weight = np.zeros((h, w), dtype=np.float32)

    for i, (px, py) in enumerate(points):
        r1 = np.random.uniform(radius_range[0], radius_range[1])
        r2 = r1 * np.random.uniform(0.3, 0.7)
        angle = np.random.uniform(0, np.pi)

        cos_a = np.cos(angle)
        sin_a = np.sin(angle)

        dx = xx - px
        dy = yy - py

        rx = dx * cos_a + dy * sin_a
        ry = -dx * sin_a + dy * cos_a

        dist_sq = (rx / r1)**2 + (ry / r2)**2
        weight = np.exp(-dist_sq / 2)

        mask += weight * values[i]
        total_weight += weight

    mask /= (total_weight + 1e-6)
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-6)

    # Posterize into distinct patch levels
    mask = np.round(mask * 5) / 5
    mask = cv2.GaussianBlur(mask, (5, 5), 1.0)

    return mask


# =========================================================================
# DEGRADATION PIPELINE
# =========================================================================
def degrade_image(rgba_path, processor, face_parser, params=None):
    if params is None:
        params = DEGRADE_PARAMS

    img_rgba = Image.open(rgba_path)
    img_rgb = img_rgba.convert("RGB")
    img_np = np.array(img_rgb).astype(np.float32)
    alpha = np.array(img_rgba.split()[3])
    alpha_norm = (alpha / 255.0)[:, :, None]
    h, w = 512, 512

    # Face parsing
    inputs = processor(images=img_rgb, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = face_parser(**inputs)
    logits = torch.nn.functional.interpolate(
        outputs.logits, size=(h, w), mode="bilinear", align_corners=False
    )
    seg = logits.argmax(dim=1)[0].cpu().numpy()

    # Region masks from face parsing
    skin_region = np.isin(seg, [1]).astype(np.float32) * 255
    skin_region = cv2.GaussianBlur(skin_region, (21, 21), 0) / 255.0

    eye_region = np.isin(seg, [4, 5, 6, 7]).astype(np.float32) * 255
    eye_region = cv2.GaussianBlur(eye_region, (21, 21), 0) / 255.0

    nose_region = np.isin(seg, [2]).astype(np.float32) * 255
    nose_region = cv2.GaussianBlur(nose_region, (21, 21), 0) / 255.0

    mouth_region = np.isin(seg, [10, 11, 12]).astype(np.float32) * 255
    mouth_region = cv2.GaussianBlur(mouth_region, (21, 21), 0) / 255.0

    hair_region = np.isin(seg, [13]).astype(np.float32) * 255
    hair_region = cv2.GaussianBlur(hair_region, (21, 21), 0) / 255.0

    # Blur sources
    large_soft = cv2.GaussianBlur(img_np, (25, 25), 6.0)
    nk = params["nuke_kernel"]
    ns = params["nuke_sigma"]
    nuke_blur = cv2.GaussianBlur(img_np, (nk, nk), ns)

    # Geometric warp
    ws = params["warp_scale"]
    dx = np.random.randn(h // ws, w // ws).astype(np.float32)
    dy = np.random.randn(h // ws, w // ws).astype(np.float32)
    dx = cv2.resize(dx, (w, h), interpolation=cv2.INTER_CUBIC) * params["warp_strength"]
    dy = cv2.resize(dy, (w, h), interpolation=cv2.INTER_CUBIC) * params["warp_strength"]
    map_x = (np.arange(w)[None, :] + dx).astype(np.float32)
    map_y = (np.arange(h)[:, None] + dy).astype(np.float32)
    img_warped = cv2.remap(img_np, map_x, map_y, cv2.INTER_LINEAR)
    large_soft = cv2.remap(large_soft, map_x, map_y, cv2.INTER_LINEAR)
    nuke_blur = cv2.remap(nuke_blur, map_x, map_y, cv2.INTER_LINEAR)

    # Variation masks per region
    skin_var = make_variation_mask(h, w,
        n_points=params["skin_points"], radius_range=params["skin_radius"])
    skin_var = cv2.GaussianBlur(skin_var, (31, 31), 8.0)

    eyes_var = make_variation_mask(h, w,
        n_points=params["eye_points"], radius_range=params["eye_radius"])
    eyes_var = cv2.GaussianBlur(eyes_var, (31, 31), 8.0)

    nose_var = make_variation_mask(h, w,
        n_points=params["nose_points"], radius_range=params["nose_radius"])
    nose_var = cv2.GaussianBlur(nose_var, (31, 31), 8.0)

    mouth_var = make_variation_mask(h, w,
        n_points=params["mouth_points"], radius_range=params["mouth_radius"])
    mouth_var = cv2.GaussianBlur(mouth_var, (31, 31), 8.0)

    hair_var = make_variation_mask(h, w,
        n_points=params["hair_points"], radius_range=params["hair_radius"])
    hair_var = cv2.GaussianBlur(hair_var, (31, 31), 8.0)

    # Cutouts: variation * region
    sm = (skin_var * skin_region)[:, :, None]
    em = (eyes_var * eye_region)[:, :, None]
    nm = (nose_var * nose_region)[:, :, None]
    mm = (mouth_var * mouth_region)[:, :, None]
    hm = (hair_var * hair_region)[:, :, None]

    # Region blurs from warped image
    degraded = img_warped.copy()
    degraded = (degraded * (1 - sm * params["skin_blur_strength"])
                + large_soft * (sm * params["skin_blur_strength"]))
    degraded = (degraded * (1 - em * params["eye_blur_strength"])
                + nuke_blur * (em * params["eye_blur_strength"]))
    degraded = (degraded * (1 - nm * params["nose_blur_strength"])
                + nuke_blur * (nm * params["nose_blur_strength"]))
    degraded = (degraded * (1 - mm * params["mouth_blur_strength"])
                + nuke_blur * (mm * params["mouth_blur_strength"]))

    # Mouth overall blur (flat blur on top of variation pass)
    mok = params["mouth_overall_blur_kernel"]
    mos = params["mouth_overall_blur_sigma"]
    mouth_extra = cv2.GaussianBlur(degraded, (mok, mok), mos)
    mr2 = mouth_region[:, :, None]
    degraded = (degraded * (1 - mr2 * params["mouth_overall_blur_strength"])
                + mouth_extra * (mr2 * params["mouth_overall_blur_strength"]))

    # Hair blur
    degraded = (degraded * (1 - hm * params["hair_blur_strength"])
                + large_soft * (hm * params["hair_blur_strength"]))

    # Bilateral surface blur
    bilateral = cv2.bilateralFilter(
        degraded.clip(0, 255).astype(np.uint8),
        params["bilateral_d"],
        params["bilateral_sigma_color"],
        params["bilateral_sigma_space"]
    ).astype(np.float32)
    degraded = (degraded * (1 - params["bilateral_strength"])
                + bilateral * params["bilateral_strength"])

    # Brightness variation
    bright_mask = make_variation_mask(h, w, n_points=150, radius_range=(15, 40))
    bright_mask = cv2.GaussianBlur(bright_mask, (31, 31), 8.0)
    degraded += (bright_mask[:, :, None] - 0.5) * params["brightness_intensity"]

    # Floor and ceiling clamps
    degraded = np.maximum(degraded, params["global_floor"])
    mr = mouth_region[:, :, None]
    degraded = np.maximum(degraded, params["mouth_floor"]) * mr + degraded * (1 - mr)
    nr = nose_region[:, :, None]
    degraded = np.maximum(degraded, params["nose_floor"]) * nr + degraded * (1 - nr)
    degraded = np.minimum(degraded, params["global_ceiling"])

    # Overall blur
    ok = params["overall_blur_kernel"]
    os_ = params["overall_blur_sigma"]
    overall = cv2.GaussianBlur(degraded, (ok, ok), os_)
    degraded = (degraded * (1 - params["overall_blur_strength"])
                + overall * params["overall_blur_strength"])

    # Desaturate
    gray = np.mean(degraded, axis=2, keepdims=True)
    degraded = degraded * (1 - params["desaturation"]) + gray * params["desaturation"]

    # Downscale + upscale (resolution loss)
    df = params["downscale_factor"]
    small = cv2.resize(degraded, (w // df, h // df),
                       interpolation=cv2.INTER_LINEAR)
    degraded = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

    # Noise injection
    noise = np.random.randn(h, w, 3).astype(np.float32) * params["noise_strength"]
    degraded += noise

    # Composite on white
    white = np.ones_like(img_np) * 255
    out = np.clip(degraded, 0, 255) * alpha_norm + white * (1 - alpha_norm)

    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


# =========================================================================
# BATCH DEGRADATION
# =========================================================================
def batch_degrade(output_dir, num_images=500, num_train=450, seed=42):
    train_a = os.path.join(output_dir, "train_A")
    test_a = os.path.join(output_dir, "test_A")
    rgba_dir = os.path.join(output_dir, "rgba_512")

    os.makedirs(train_a, exist_ok=True)
    os.makedirs(test_a, exist_ok=True)

    processor = SegformerImageProcessor.from_pretrained("jonathandinu/face-parsing")
    face_parser = SegformerForSemanticSegmentation.from_pretrained(
        "jonathandinu/face-parsing")
    face_parser = face_parser.to("cuda").eval()

    indices = list(range(num_images))
    random.seed(seed)
    random.shuffle(indices)

    for j, i in enumerate(indices):
        path = os.path.join(rgba_dir, f"{i:05d}.png")
        degraded = degrade_image(path, processor, face_parser)

        if j < num_train:
            degraded.save(os.path.join(train_a, f"{j:05d}.png"))
        else:
            degraded.save(os.path.join(test_a, f"{(j - num_train):05d}.png"))

        if j % 25 == 0:
            print(f"{j}/{num_images}")

    print("Done!")


# =========================================================================
# MAIN
# =========================================================================
if __name__ == "__main__":
    args = parse_args()

    print("Step 1: Creating ground truth train/test split...")
    create_ground_truths(args.output_dir, args.num_images,
                         args.num_train, args.seed)

    print("\nStep 2: Applying degradation pipeline...")
    batch_degrade(args.output_dir, args.num_images,
                  args.num_train, args.seed)

    print("\nDataset ready!")
    print(f"  train_A: {len(os.listdir(os.path.join(args.output_dir, 'train_A')))} degraded")
    print(f"  train_B: {len(os.listdir(os.path.join(args.output_dir, 'train_B')))} ground truth")
    print(f"  test_A:  {len(os.listdir(os.path.join(args.output_dir, 'test_A')))} degraded")
    print(f"  test_B:  {len(os.listdir(os.path.join(args.output_dir, 'test_B')))} ground truth")
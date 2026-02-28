"""
train.py

Training script for the splat enhancement model.
Adapted from pix2pix-turbo's train_pix2pix_turbo.py with the following changes:
- Removed GAN discriminator (unnecessary for paired restoration with small dataset)
- Removed wandb/FID tracking (simplified logging to console)
- Removed accelerate multi-GPU (single GPU training)
- Replaced L2 with L1 (sharper results for restoration)
- Added Gram matrix style loss (per ELITE/task spec, restores texture)
- Kept CLIP similarity loss from original (semantic regularizer)

The core training flow is preserved from the original:
    encode input -> UNet at t=999 -> scheduler step -> decode with skips
    -> compute loss -> backward -> optimizer step
"""

import os
import gc
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF
from tqdm.auto import tqdm
import clip

from model.architecture import SplatEnhancer
from model.losses import CombinedLoss


# ---------------------------------------------------------------------------
# Dataset - follows PairedDataset from training_utils.py
# ---------------------------------------------------------------------------
class PairedDataset(Dataset):
    """Paired dataset for (degraded, ground_truth) image pairs.

    Follows the same folder structure as pix2pix-turbo:
        dataset_folder/train_A/  (degraded inputs)
        dataset_folder/train_B/  (clean ground truths)

    Images are normalized to [-1, 1] for both input and output,
    matching what the VAE encoder expects.
    """

    def __init__(self, dataset_folder, split="train"):
        super().__init__()
        if split == "train":
            self.input_folder = os.path.join(dataset_folder, "train_A")
            self.output_folder = os.path.join(dataset_folder, "train_B")
        elif split == "test":
            self.input_folder = os.path.join(dataset_folder, "test_A")
            self.output_folder = os.path.join(dataset_folder, "test_B")

        self.img_names = sorted([
            f for f in os.listdir(self.input_folder)
            if f.endswith((".png", ".jpg", ".jpeg"))
        ])
        print(f"Loaded {len(self.img_names)} {split} pairs")

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        name = self.img_names[idx]
        input_img = Image.open(
            os.path.join(self.input_folder, name)).convert("RGB")
        output_img = Image.open(
            os.path.join(self.output_folder, name)).convert("RGB")

        # Both normalized to [-1, 1] as the VAE expects
        input_t = TF.to_tensor(input_img)
        input_t = TF.normalize(input_t, mean=[0.5], std=[0.5])

        output_t = TF.to_tensor(output_img)
        output_t = TF.normalize(output_t, mean=[0.5], std=[0.5])

        return {"input": input_t, "target": output_t}


# ---------------------------------------------------------------------------
# Argument parsing - subset of training_utils.parse_args_paired_training
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train splat enhancement model")

    # Data
    parser.add_argument("--dataset_folder", type=str, required=True,
                        help="Path to dataset with train_A/, train_B/, etc.")
    # Model
    parser.add_argument("--lora_rank_unet", type=int, default=8)
    parser.add_argument("--lora_rank_vae", type=int, default=4)
    parser.add_argument("--pretrained_path", type=str, default=None,
                        help="Resume from checkpoint")

    # Training - defaults from pix2pix-turbo training_utils.py
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--max_train_steps", type=int, default=None,
                        help="Override num_epochs with fixed step count")
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed_precision", action="store_true",
                        help="Use fp16 mixed precision (recommended for T4)")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Enable gradient checkpointing to save VRAM")
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--num_workers", type=int, default=4)

    # Loss weights - L1/LPIPS/Gram per task spec, CLIP from original
    parser.add_argument("--lambda_l1", type=float, default=1.0)
    parser.add_argument("--lambda_lpips", type=float, default=0.5)
    parser.add_argument("--lambda_gram", type=float, default=0.01)
    parser.add_argument("--lambda_clipsim", type=float, default=5.0,
                        help="CLIP similarity loss weight (from pix2pix-turbo)")

    # Output
    parser.add_argument("--output_dir", type=str, default="output",
                        help="Directory for checkpoints and logs")
    parser.add_argument("--checkpoint_every", type=int, default=200)
    parser.add_argument("--eval_every", type=int, default=100)

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Validation - adapted from the eval loop in train_pix2pix_turbo.py
# ---------------------------------------------------------------------------
@torch.no_grad()
def validate(model, dl_val, criterion, net_clip, clip_renorm, prompt_text):
    """Run validation and return average losses."""
    model.set_eval()
    losses = {"l1": [], "lpips": [], "gram": [], "clipsim": []}

    for batch in dl_val:
        x_src = batch["input"].cuda()
        x_tgt = batch["target"].cuda()

        x_pred = model(x_src)

        # L1 + LPIPS + Gram
        _, batch_losses = criterion(x_pred.float(), x_tgt.float())
        losses["l1"].append(batch_losses["l1"])
        losses["lpips"].append(batch_losses["lpips"])
        losses["gram"].append(batch_losses["gram"])

        # CLIP similarity (from train_pix2pix_turbo.py eval loop)
        x_pred_renorm = clip_renorm(x_pred * 0.5 + 0.5)
        x_pred_renorm = F.interpolate(
            x_pred_renorm, (224, 224), mode="bilinear", align_corners=False)
        caption_tokens = clip.tokenize(
            [prompt_text], truncate=True).to(x_pred.device)
        clipsim, _ = net_clip(x_pred_renorm, caption_tokens)
        losses["clipsim"].append(clipsim.mean().item())

    model.set_train()
    return {k: np.mean(v) for k, v in losses.items()}


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)

    # --- Model ---
    print("Initializing model...")
    model = SplatEnhancer(
        lora_rank_unet=args.lora_rank_unet,
        lora_rank_vae=args.lora_rank_vae,
        pretrained_path=args.pretrained_path,
    )
    model.set_train()

    if args.gradient_checkpointing:
        model.unet.enable_gradient_checkpointing()

    # Count trainable parameters
    trainable_params = model.get_trainable_params()
    n_params = sum(p.numel() for p in trainable_params)
    print(f"Trainable parameters: {n_params:,}")

    # --- CLIP model (from train_pix2pix_turbo.py) ---
    net_clip, _ = clip.load("ViT-B/32", device="cuda")
    net_clip.requires_grad_(False)
    net_clip.eval()
    # ImageNet renormalization for CLIP (from train_pix2pix_turbo.py)
    clip_renorm = transforms.Normalize(
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711))
    prompt_text = "a high quality photograph of a person"

    # --- Loss ---
    criterion = CombinedLoss(
        lambda_l1=args.lambda_l1,
        lambda_lpips=args.lambda_lpips,
        lambda_gram=args.lambda_gram,
    )

    # --- Data ---
    dataset_train = PairedDataset(args.dataset_folder, split="train")
    dl_train = DataLoader(
        dataset_train, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers, pin_memory=True)

    dataset_val = PairedDataset(args.dataset_folder, split="test")
    dl_val = DataLoader(
        dataset_val, batch_size=1, shuffle=False, num_workers=0)

    # --- Optimizer (same as train_pix2pix_turbo.py) ---
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # --- Mixed precision ---
    scaler = None
    if args.mixed_precision:
        scaler = torch.amp.GradScaler("cuda")
        print("Using fp16 mixed precision")

    # --- Calculate total steps ---
    steps_per_epoch = len(dl_train)
    if args.max_train_steps is not None:
        total_steps = args.max_train_steps
        total_epochs = (total_steps // steps_per_epoch) + 1
    else:
        total_epochs = args.num_epochs
        total_steps = total_epochs * steps_per_epoch

    print(f"Training for {total_steps} steps ({total_epochs} epochs)")
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Batch size: {args.batch_size}")
    print(f"Loss weights: L1={args.lambda_l1}, LPIPS={args.lambda_lpips}, "
          f"Gram={args.lambda_gram}, CLIP={args.lambda_clipsim}")

    # --- Training loop (adapted from train_pix2pix_turbo.py) ---
    global_step = 0
    progress_bar = tqdm(total=total_steps, desc="Training")

    for epoch in range(total_epochs):
        for step, batch in enumerate(dl_train):
            if global_step >= total_steps:
                break

            x_src = batch["input"].cuda()
            x_tgt = batch["target"].cuda()

            # Forward + loss computation
            if scaler is not None:
                with torch.amp.autocast("cuda"):
                    x_pred = model(x_src)
                    loss, loss_dict = criterion(
                        x_pred.float(), x_tgt.float())

                    # CLIP similarity (from train_pix2pix_turbo.py L181-187)
                    if args.lambda_clipsim > 0:
                        x_pred_renorm = clip_renorm(x_pred * 0.5 + 0.5)
                        x_pred_renorm = F.interpolate(
                            x_pred_renorm, (224, 224),
                            mode="bilinear", align_corners=False)
                        caption_tokens = clip.tokenize(
                            [prompt_text], truncate=True).to(x_pred.device)
                        clipsim, _ = net_clip(x_pred_renorm, caption_tokens)
                        loss_clipsim = (1 - clipsim.mean() / 100)
                        loss = loss + loss_clipsim * args.lambda_clipsim
                        loss_dict["clipsim"] = loss_clipsim.item()

                # Backward with gradient scaling
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    trainable_params, args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                x_pred = model(x_src)
                loss, loss_dict = criterion(x_pred.float(), x_tgt.float())

                # CLIP similarity (from train_pix2pix_turbo.py L181-187)
                if args.lambda_clipsim > 0:
                    x_pred_renorm = clip_renorm(x_pred * 0.5 + 0.5)
                    x_pred_renorm = F.interpolate(
                        x_pred_renorm, (224, 224),
                        mode="bilinear", align_corners=False)
                    caption_tokens = clip.tokenize(
                        [prompt_text], truncate=True).to(x_pred.device)
                    clipsim, _ = net_clip(x_pred_renorm, caption_tokens)
                    loss_clipsim = (1 - clipsim.mean() / 100)
                    loss = loss + loss_clipsim * args.lambda_clipsim
                    loss_dict["clipsim"] = loss_clipsim.item()

                # Backward (same pattern as train_pix2pix_turbo.py L188-193)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    trainable_params, args.max_grad_norm)
                optimizer.step()

            # Logging
            global_step += 1
            progress_bar.update(1)
            progress_bar.set_postfix(
                l1=f"{loss_dict['l1']:.4f}",
                lpips=f"{loss_dict['lpips']:.4f}",
                gram=f"{loss_dict['gram']:.4f}",
                clip=f"{loss_dict.get('clipsim', 0):.4f}",
            )

            # Checkpoint (same pattern as train_pix2pix_turbo.py L254-256)
            if global_step % args.checkpoint_every == 0:
                ckpt_path = os.path.join(
                    args.output_dir, "checkpoints",
                    f"model_{global_step}.pkl")
                model.save_model(ckpt_path)

            # Validation (same pattern as train_pix2pix_turbo.py L259-301)
            if global_step % args.eval_every == 0:
                val_losses = validate(
                    model, dl_val, criterion,
                    net_clip, clip_renorm, prompt_text)
                print(f"\n[Step {global_step}] Val - "
                      f"L1: {val_losses['l1']:.4f}, "
                      f"LPIPS: {val_losses['lpips']:.4f}, "
                      f"Gram: {val_losses['gram']:.4f}, "
                      f"CLIP: {val_losses['clipsim']:.2f}")
                gc.collect()
                torch.cuda.empty_cache()

        if global_step >= total_steps:
            break

    # Save final checkpoint
    final_path = os.path.join(
        args.output_dir, "checkpoints", "model_final.pkl")
    model.save_model(final_path)
    print(f"\nTraining complete. Final model saved to {final_path}")

    progress_bar.close()


if __name__ == "__main__":
    main()

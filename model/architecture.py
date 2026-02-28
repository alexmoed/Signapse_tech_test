"""
model/architecture.py

Single-step diffusion image enhancer based on pix2pix-turbo.
Adapts SD-Turbo for paired image-to-image enhancement with:
- VAE encoder (frozen) encodes degraded input
- UNet processes latent at fixed t=999 (single step, not iterative)
- VAE decoder reconstructs enhanced output with skip connections from encoder
- LoRA adapters on UNet + VAE decoder for parameter-efficient fine-tuning

Trainable: LoRA weights, skip convolutions, UNet conv_in
Frozen: VAE encoder, text encoder, all base weights
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from peft import LoraConfig


def make_1step_sched():
    """Single-step DDPM scheduler at t=999."""
    sched = DDPMScheduler.from_pretrained("stabilityai/sd-turbo", subfolder="scheduler")
    sched.set_timesteps(1, device="cuda")
    sched.alphas_cumprod = sched.alphas_cumprod.cuda()
    return sched


def vae_encoder_fwd(self, sample):
    """Modified VAE encoder that stores intermediate activations for skip connections.

    At each downsampling stage, the activation is saved to self.current_down_blocks.
    These get passed to the decoder to preserve spatial detail that would otherwise
    be lost through the latent bottleneck.
    """
    sample = self.conv_in(sample)
    l_blocks = []
    for down_block in self.down_blocks:
        l_blocks.append(sample)
        sample = down_block(sample)
    sample = self.mid_block(sample)
    sample = self.conv_norm_out(sample)
    sample = self.conv_act(sample)
    sample = self.conv_out(sample)
    self.current_down_blocks = l_blocks
    return sample


def vae_decoder_fwd(self, sample, latent_embeds=None):
    """Modified VAE decoder that accepts skip connections from encoder.

    Each up_block receives the corresponding encoder activation (reversed order)
    through a learned 1x1 conv. The gamma parameter controls skip strength.
    During training, dropout is applied to skip connections to prevent
    the decoder from relying too heavily on encoder shortcuts.
    """
    sample = self.conv_in(sample)
    upscale_dtype = next(iter(self.up_blocks.parameters())).dtype
    sample = self.mid_block(sample, latent_embeds)
    sample = sample.to(upscale_dtype)

    if not self.ignore_skip:
        skip_convs = [self.skip_conv_1, self.skip_conv_2,
                      self.skip_conv_3, self.skip_conv_4]
        for idx, up_block in enumerate(self.up_blocks):
            skip_in = skip_convs[idx](
                self.incoming_skip_acts[::-1][idx] * self.gamma)
            skip_in = nn.functional.dropout(skip_in, p=self.skip_dropout, training=True)
            sample = sample + skip_in
            sample = up_block(sample, latent_embeds)
    else:
        for idx, up_block in enumerate(self.up_blocks):
            sample = up_block(sample, latent_embeds)

    if latent_embeds is None:
        sample = self.conv_norm_out(sample)
    else:
        sample = self.conv_norm_out(sample, latent_embeds)
    sample = self.conv_act(sample)
    sample = self.conv_out(sample)
    return sample


class SplatEnhancer(nn.Module):
    """
    Single-step diffusion enhancer for Gaussian splat renders.

    Adapts SD-Turbo following the pix2pix-turbo architecture for paired
    image-to-image translation. A degraded Gaussian splat render goes in,
    an enhanced photorealistic image comes out in a single forward pass.

    Flow:
        degraded -> VAE encode (+ store skip activations)
                 -> UNet denoise at t=999
                 -> scheduler step
                 -> VAE decode (with encoder skip connections)
                 -> enhanced output

    Args:
        lora_rank_unet: LoRA rank for UNet adapters (default 8)
        lora_rank_vae: LoRA rank for VAE decoder adapters (default 4)
        prompt: Fixed text prompt for CLIP conditioning
    """

    def __init__(self, lora_rank_unet=8, lora_rank_vae=4,
                 prompt="a high quality photograph of a person",
                 pretrained_path=None):
        super().__init__()

        # --- Text encoder + tokenizer (frozen) ---
        self.tokenizer = AutoTokenizer.from_pretrained(
            "stabilityai/sd-turbo", subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(
            "stabilityai/sd-turbo", subfolder="text_encoder").cuda()
        self.text_encoder.requires_grad_(False)

        # Pre-encode the fixed prompt once, reuse every forward pass
        tokens = self.tokenizer(
            prompt, max_length=self.tokenizer.model_max_length,
            padding="max_length", truncation=True, return_tensors="pt"
        ).input_ids.cuda()
        self.register_buffer("prompt_embeds", self.text_encoder(tokens)[0])

        # --- Scheduler ---
        self.sched = make_1step_sched()
        self.register_buffer("timesteps",
                             torch.tensor([999], device="cuda").long())

        # --- VAE with skip connections ---
        vae = AutoencoderKL.from_pretrained(
            "stabilityai/sd-turbo", subfolder="vae")

        # Monkey-patch encoder/decoder with skip-aware versions
        vae.encoder.forward = vae_encoder_fwd.__get__(
            vae.encoder, vae.encoder.__class__)
        vae.decoder.forward = vae_decoder_fwd.__get__(
            vae.decoder, vae.decoder.__class__)

        # 1x1 convs bridging encoder activations to decoder
        # Channels match the VAE's internal feature map sizes
        vae.decoder.skip_conv_1 = nn.Conv2d(512, 512, kernel_size=1,
                                            bias=False).cuda()
        vae.decoder.skip_conv_2 = nn.Conv2d(256, 512, kernel_size=1,
                                            bias=False).cuda()
        vae.decoder.skip_conv_3 = nn.Conv2d(128, 512, kernel_size=1,
                                            bias=False).cuda()
        vae.decoder.skip_conv_4 = nn.Conv2d(128, 256, kernel_size=1,
                                            bias=False).cuda()
        vae.decoder.ignore_skip = False
        vae.decoder.gamma = 1
        vae.decoder.skip_dropout = 0.3

        # --- UNet ---
        unet = UNet2DConditionModel.from_pretrained(
            "stabilityai/sd-turbo", subfolder="unet")

        if pretrained_path is not None:
            # Load from checkpoint
            sd = torch.load(pretrained_path, map_location="cpu")
            unet_lora_config = LoraConfig(
                r=sd["rank_unet"], init_lora_weights="gaussian",
                target_modules=sd["unet_lora_target_modules"])
            vae_lora_config = LoraConfig(
                r=sd["rank_vae"], init_lora_weights="gaussian",
                target_modules=sd["vae_lora_target_modules"])
            vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
            _sd_vae = vae.state_dict()
            for k in sd["state_dict_vae"]:
                _sd_vae[k] = sd["state_dict_vae"][k]
            vae.load_state_dict(_sd_vae)
            unet.add_adapter(unet_lora_config)
            _sd_unet = unet.state_dict()
            for k in sd["state_dict_unet"]:
                _sd_unet[k] = sd["state_dict_unet"][k]
            unet.load_state_dict(_sd_unet)
        else:
            # Init skip convs near zero so they don't disrupt the
            # pretrained decoder at the start of training
            nn.init.constant_(vae.decoder.skip_conv_1.weight, 1e-5)
            nn.init.constant_(vae.decoder.skip_conv_2.weight, 1e-5)
            nn.init.constant_(vae.decoder.skip_conv_3.weight, 1e-5)
            nn.init.constant_(vae.decoder.skip_conv_4.weight, 1e-5)

            # LoRA on VAE decoder (includes skip convs)
            target_modules_vae = [
                "conv1", "conv2", "conv_in", "conv_shortcut",
                "conv", "conv_out",
                "skip_conv_1", "skip_conv_2", "skip_conv_3", "skip_conv_4",
                "to_k", "to_q", "to_v", "to_out.0",
            ]
            vae_lora_config = LoraConfig(
                r=lora_rank_vae, init_lora_weights="gaussian",
                target_modules=target_modules_vae)
            vae.add_adapter(vae_lora_config, adapter_name="vae_skip")

            # LoRA on UNet
            target_modules_unet = [
                "to_k", "to_q", "to_v", "to_out.0",
                "conv", "conv1", "conv2", "conv_shortcut", "conv_out",
                "proj_in", "proj_out", "ff.net.2", "ff.net.0.proj",
            ]
            unet_lora_config = LoraConfig(
                r=lora_rank_unet, init_lora_weights="gaussian",
                target_modules=target_modules_unet)
            unet.add_adapter(unet_lora_config)

            self.target_modules_unet = target_modules_unet
            self.target_modules_vae = target_modules_vae

        unet.to("cuda")
        vae.to("cuda")
        self.unet = unet
        self.vae = vae
        self.lora_rank_unet = lora_rank_unet
        self.lora_rank_vae = lora_rank_vae

    def set_train(self):
        """Enable training mode. Only LoRA, skip convs, and conv_in
        are trainable. Everything else stays frozen."""
        self.unet.train()
        self.vae.train()

        # Freeze everything first
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)

        # Unfreeze LoRA params in UNet
        for n, p in self.unet.named_parameters():
            if "lora" in n:
                p.requires_grad = True
        # Unfreeze UNet conv_in (adapts to our input domain)
        self.unet.conv_in.requires_grad_(True)

        # Unfreeze LoRA params in VAE
        for n, p in self.vae.named_parameters():
            if "lora" in n:
                p.requires_grad = True
        # Unfreeze skip convolutions
        self.vae.decoder.skip_conv_1.requires_grad_(True)
        self.vae.decoder.skip_conv_2.requires_grad_(True)
        self.vae.decoder.skip_conv_3.requires_grad_(True)
        self.vae.decoder.skip_conv_4.requires_grad_(True)

    def set_eval(self):
        """Freeze everything for inference."""
        self.unet.eval()
        self.vae.eval()
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)

    def get_trainable_params(self):
        """Return list of trainable parameters for the optimizer."""
        params = []
        for n, p in self.unet.named_parameters():
            if "lora" in n:
                params.append(p)
        params += list(self.unet.conv_in.parameters())
        for n, p in self.vae.named_parameters():
            if "lora" in n and "vae_skip" in n:
                params.append(p)
        params += list(self.vae.decoder.skip_conv_1.parameters())
        params += list(self.vae.decoder.skip_conv_2.parameters())
        params += list(self.vae.decoder.skip_conv_3.parameters())
        params += list(self.vae.decoder.skip_conv_4.parameters())
        return params

    def forward(self, x):
        """
        Single-step enhancement.

        Args:
            x: degraded input images, tensor of shape (B, 3, 512, 512)
               in range [-1, 1]

        Returns:
            Enhanced images, tensor of shape (B, 3, 512, 512) in range [-1, 1]
        """
        # Expand prompt embedding to match batch size
        prompt_embeds = self.prompt_embeds.expand(x.shape[0], -1, -1)

        # VAE encode (stores skip activations internally)
        encoded = self.vae.encode(x).latent_dist.sample()
        encoded = encoded * self.vae.config.scaling_factor

        # UNet single-step denoise at t=999
        model_pred = self.unet(
            encoded, self.timesteps,
            encoder_hidden_states=prompt_embeds
        ).sample

        # Scheduler step
        x_denoised = self.sched.step(
            model_pred, self.timesteps, encoded,
            return_dict=True
        ).prev_sample
        x_denoised = x_denoised.to(model_pred.dtype)

        # VAE decode with skip connections from encoder
        self.vae.decoder.incoming_skip_acts = \
            self.vae.encoder.current_down_blocks
        output = self.vae.decode(
            x_denoised / self.vae.config.scaling_factor
        ).sample

        return output.clamp(-1, 1)

    def save_model(self, path):
        """Save only the trainable weights (LoRA + skip convs + conv_in).
        Keeps checkpoint small since base SD-Turbo weights aren't saved."""
        sd = {
            "unet_lora_target_modules": self.target_modules_unet,
            "vae_lora_target_modules": self.target_modules_vae,
            "rank_unet": self.lora_rank_unet,
            "rank_vae": self.lora_rank_vae,
            "state_dict_unet": {
                k: v for k, v in self.unet.state_dict().items()
                if "lora" in k or "conv_in" in k
            },
            "state_dict_vae": {
                k: v for k, v in self.vae.state_dict().items()
                if "lora" in k or "skip" in k
            },
        }
        torch.save(sd, path)
        print(f"Saved checkpoint to {path}")
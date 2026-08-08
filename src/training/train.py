# src/training/train.py
# HiDDeN training loop with checkpointing.
# Based on Zhu et al. 2018, "HiDDeN: Hiding Data with Deep Networks".

import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.hidden_watermark import HiDDeN
from src.training.dataset import WatermarkDataset


def bit_accuracy(logits: torch.Tensor, message: torch.Tensor) -> float:
    """Fraction of bits correctly recovered from the decoder logits."""
    preds = (torch.sigmoid(logits) > 0.5).float()
    return (preds == message).float().mean().item()


def psnr_batch(a: torch.Tensor, b: torch.Tensor) -> float:
    """Mean PSNR over a batch of [B,3,H,W] tensors in [0,1]."""
    mse = ((a - b) ** 2).mean(dim=[1, 2, 3])
    mse = torch.clamp(mse, min=1e-10)
    return (10 * torch.log10(1.0 / mse)).mean().item()


def train(
    image_dirs,
    msg_len: int = 48,
    image_size: int = 128,
    batch_size: int = 12,
    total_steps: int = 50000,
    lr: float = 1e-3,
    lambda_image: float = 0.7,      # image reconstruction weight
    lambda_message: float = 1.0,    # message recovery weight
    lambda_adv: float = 1e-3,       # adversary weight (small — helps imperceptibility)
    checkpoint_dir: str = "checkpoints",
    log_every: int = 100,
    save_every: int = 5000,
    resume_from: str | None = None,
    device: str | None = None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device={device}")

    os.makedirs(checkpoint_dir, exist_ok=True)

    dataset = WatermarkDataset(image_dirs, msg_len=msg_len, image_size=image_size)
    print(f"[train] dataset size = {len(dataset)} images")

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=(device == "cuda"), drop_last=True,
    )

    model = HiDDeN(msg_len=msg_len).to(device)
    opt_g = torch.optim.Adam(
        list(model.encoder.parameters()) + list(model.decoder.parameters()), lr=lr
    )
    opt_d = torch.optim.Adam(model.adversary.parameters(), lr=lr)

    start_step = 0
    if resume_from and os.path.exists(resume_from):
        ckpt = torch.load(resume_from, map_location=device)
        model.load_state_dict(ckpt["model"])
        opt_g.load_state_dict(ckpt["opt_g"])
        opt_d.load_state_dict(ckpt["opt_d"])
        start_step = ckpt["step"]
        print(f"[train] resumed from step {start_step}")

    model.train()
    step = start_step
    t0 = time.time()
    running = {"img": 0.0, "msg": 0.0, "adv": 0.0, "bit_acc": 0.0, "psnr": 0.0, "n": 0}

    while step < total_steps:
        for image, message in loader:
            if step >= total_steps:
                break
            image = image.to(device)
            message = message.to(device)

            # --- Train adversary: classify real vs encoded ---
            with torch.no_grad():
                encoded, _, _ = model(image, message)
            d_real = model.adversary(image)
            d_fake = model.adversary(encoded.detach())
            loss_d = F.binary_cross_entropy_with_logits(
                d_real, torch.ones_like(d_real)
            ) + F.binary_cross_entropy_with_logits(
                d_fake, torch.zeros_like(d_fake)
            )
            opt_d.zero_grad()
            loss_d.backward()
            opt_d.step()

            # --- Train encoder + decoder ---
            encoded, attacked, decoded_logits = model(image, message)
            loss_img = F.mse_loss(encoded, image)
            loss_msg = F.binary_cross_entropy_with_logits(decoded_logits, message)
            d_fake_for_g = model.adversary(encoded)
            loss_adv = F.binary_cross_entropy_with_logits(
                d_fake_for_g, torch.ones_like(d_fake_for_g)
            )
            loss_g = (
                lambda_image * loss_img
                + lambda_message * loss_msg
                + lambda_adv * loss_adv
            )
            opt_g.zero_grad()
            loss_g.backward()
            opt_g.step()

            # --- Log ---
            with torch.no_grad():
                acc = bit_accuracy(decoded_logits, message)
                psnr = psnr_batch(encoded, image)
            running["img"] += loss_img.item()
            running["msg"] += loss_msg.item()
            running["adv"] += loss_adv.item()
            running["bit_acc"] += acc
            running["psnr"] += psnr
            running["n"] += 1

            if step % log_every == 0 and step > 0:
                n = running["n"]
                elapsed = time.time() - t0
                its = step / max(elapsed, 1e-6)
                print(
                    f"[{step:>6}/{total_steps}] "
                    f"img={running['img']/n:.4f}  "
                    f"msg={running['msg']/n:.4f}  "
                    f"adv={running['adv']/n:.4f}  "
                    f"bit_acc={running['bit_acc']/n:.3f}  "
                    f"psnr={running['psnr']/n:.2f}  "
                    f"({its:.1f} it/s)"
                )
                running = {k: 0.0 for k in running}
                running["n"] = 0

            if step % save_every == 0 and step > 0:
                ckpt_path = os.path.join(checkpoint_dir, f"hidden_step_{step}.pt")
                torch.save({
                    "step": step,
                    "model": model.state_dict(),
                    "opt_g": opt_g.state_dict(),
                    "opt_d": opt_d.state_dict(),
                    "msg_len": msg_len,
                    "image_size": image_size,
                }, ckpt_path)
                # Always keep a "latest" symlink-style copy
                torch.save({
                    "step": step,
                    "model": model.state_dict(),
                    "opt_g": opt_g.state_dict(),
                    "opt_d": opt_d.state_dict(),
                    "msg_len": msg_len,
                    "image_size": image_size,
                }, os.path.join(checkpoint_dir, "hidden_latest.pt"))
                print(f"[save] {ckpt_path}")

            step += 1

    # Final save
    final_path = os.path.join(checkpoint_dir, "hidden_final.pt")
    torch.save({
        "step": step,
        "model": model.state_dict(),
        "opt_g": opt_g.state_dict(),
        "opt_d": opt_d.state_dict(),
        "msg_len": msg_len,
        "image_size": image_size,
    }, final_path)
    print(f"[done] saved final checkpoint to {final_path}")
# src/training/train_v2.py
# Training loop for HiDDeN v2 (src/hidden_v2.py).
#
# Run (Colab or locally):
#   python -m src.training.train_v2 --data data/coco --out checkpoints
#
# Only one loss: message recovery. Image quality is fixed by construction
# (the residual is scaled to target_psnr), so there is nothing to balance.
#
# Curriculum (both end at `ramp_end`):
#   - distortions: none for `warmup_steps` (the networks first learn to
#     communicate at all), then severity ramps 0 -> 1;
#   - quality: starts at a loose `start_psnr` (strong, easy-to-read watermark)
#     and tightens linearly to `target_psnr`. Starting directly at 40 dB keeps
#     the model near chance for thousands of steps.
#
# Checkpoints: hidden_v2_latest.pt (resume after a Colab disconnect),
#              hidden_v2_best.pt   (best validation accuracy at full severity),
#              hidden_v2_final.pt.

import argparse
import math
import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from src.hidden_v2 import HiDDeNv2, Distortions
from src.training.dataset import WatermarkDataset


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _severity(step: int, warmup_steps: int, ramp_end: int) -> float:
    if step < warmup_steps:
        return 0.0
    return min(1.0, (step - warmup_steps) / max(1, ramp_end - warmup_steps))


def _psnr(step: int, start_psnr: float, target_psnr: float, ramp_end: int) -> float:
    return start_psnr + (target_psnr - start_psnr) * min(1.0, step / max(1, ramp_end))


def _lr(step: int, total_steps: int, base_lr: float) -> float:
    """Cosine decay to 2% of base_lr; function of step so resuming is exact."""
    return base_lr * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * step / total_steps)))


@torch.no_grad()
def validate(model: HiDDeNv2, loader: DataLoader, device: str, ecc_t: int) -> dict:
    """
    Per-distortion results at full severity on held-out images:
      bit_acc   — fraction of bits recovered
      decodable — fraction of images with <= ecc_t bit errors, i.e. images whose
                  product ID the error-correcting code would recover
    """
    model.eval()
    gen = torch.Generator().manual_seed(0)
    stats = {name: [0.0, 0.0, 0] for name in Distortions.NAMES}
    psnr_sum, n_img = 0.0, 0
    for image, _ in loader:
        image = image.to(device)
        message = torch.randint(0, 2, (image.size(0), model.msg_len),
                                generator=gen).float().to(device)
        for name in Distortions.NAMES:
            watermarked, logits, _ = model(image, message, 1.0, name)
            errors = ((logits > 0).float() != message).sum(dim=1)
            stats[name][0] += (1 - errors.float() / model.msg_len).sum().item()
            stats[name][1] += (errors <= ecc_t).float().sum().item()
            stats[name][2] += image.size(0)
        mse = ((watermarked - image) ** 2).mean(dim=(1, 2, 3)).clamp(min=1e-10)
        psnr_sum += (10 * torch.log10(1 / mse)).sum().item()
        n_img += image.size(0)
    model.train()
    out = {name: {"bit_acc": a / n, "decodable": d / n} for name, (a, d, n) in stats.items()}
    out["_mean_bit_acc"] = sum(v["bit_acc"] for v in out.values()) / len(Distortions.NAMES)
    out["_psnr"] = psnr_sum / n_img
    return out


def _print_validation(step: int, val: dict) -> None:
    print(f"\n[val @ {step}] PSNR {val['_psnr']:.2f} dB   mean bit acc {val['_mean_bit_acc']:.3f}")
    print(f"  {'distortion':<13}{'bit acc':>9}{'decodable':>11}")
    for name in Distortions.NAMES:
        print(f"  {name:<13}{val[name]['bit_acc']:>9.3f}{val[name]['decodable']:>10.0%}")
    print()


def train(
    image_dirs,
    msg_len: int = 63,
    image_size: int = 128,       # resolution the encoder / decoder work at
    train_size: int = 256,       # resolution of the "full-size" training images
    target_psnr: float = 40.0,
    start_psnr: float = 30.0,
    batch_size: int = 32,
    total_steps: int = 40000,
    lr: float = 1e-3,
    warmup_steps: int = 2000,
    ramp_end: int = 15000,
    ecc_t: int = 6,
    val_images: int = 256,
    val_every: int = 2000,
    log_every: int = 100,
    checkpoint_dir: str = "checkpoints",
    resume_from: str | None = None,
    num_workers: int = 2,
    device: str | None = None,
):
    device = device or _pick_device()
    print(f"[train_v2] device={device}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    dataset = WatermarkDataset(image_dirs, msg_len=msg_len, image_size=train_size)
    dataset.paths.sort()
    n_val = min(val_images, len(dataset) // 5)
    train_set = Subset(dataset, range(len(dataset) - n_val))
    val_set = Subset(dataset, range(len(dataset) - n_val, len(dataset)))
    print(f"[train_v2] {len(train_set)} training / {len(val_set)} validation images")

    loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                        num_workers=num_workers, pin_memory=(device == "cuda"),
                        drop_last=True, persistent_workers=num_workers > 0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers)

    model = HiDDeNv2(msg_len=msg_len, image_size=image_size,
                     target_psnr=target_psnr).to(device)
    params = list(model.encoder.parameters()) + list(model.decoder.parameters())
    opt = torch.optim.Adam(params, lr=lr)

    step, best_acc = 0, 0.0
    if resume_from and os.path.exists(resume_from):
        ckpt = torch.load(resume_from, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        step, best_acc = ckpt["step"], ckpt.get("best_acc", 0.0)
        print(f"[train_v2] resumed from step {step}")

    def save(name: str, **extra) -> None:
        torch.save({
            "arch": "hidden_v2",
            "step": step,
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "msg_len": msg_len,
            "image_size": image_size,
            "target_psnr": target_psnr,
            "best_acc": best_acc,
            **extra,
        }, os.path.join(checkpoint_dir, name))

    model.train()
    t0, start_step = time.time(), step
    running = {"loss": 0.0, "acc": 0.0, "n": 0}

    while step < total_steps:
        for image, message in loader:
            if step >= total_steps:
                break
            image, message = image.to(device), message.to(device)
            severity = _severity(step, warmup_steps, ramp_end)
            for group in opt.param_groups:
                group["lr"] = _lr(step, total_steps, lr)

            psnr = _psnr(step, start_psnr, target_psnr, ramp_end)
            _, logits, _ = model(image, message, severity, psnr=psnr)
            loss = F.binary_cross_entropy_with_logits(logits, message)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            step += 1

            running["loss"] += loss.item()
            running["acc"] += ((logits > 0).float() == message).float().mean().item()
            running["n"] += 1

            if step % log_every == 0:
                n = running["n"]
                its = (step - start_step) / max(time.time() - t0, 1e-6)
                print(f"[{step:>6}/{total_steps}] loss={running['loss']/n:.4f}  "
                      f"bit_acc={running['acc']/n:.3f}  severity={severity:.2f}  "
                      f"psnr={psnr:.1f}  "
                      f"({its:.1f} it/s)")
                running = {"loss": 0.0, "acc": 0.0, "n": 0}

            if step % val_every == 0 or step == total_steps:
                val = validate(model, val_loader, device, ecc_t)
                _print_validation(step, val)
                # "best" only counts once severity and quality are at their final values
                if step >= ramp_end and val["_mean_bit_acc"] > best_acc:
                    best_acc = val["_mean_bit_acc"]
                    save("hidden_v2_best.pt", val=val)
                    print(f"[save] new best ({best_acc:.3f}) -> hidden_v2_best.pt")
                save("hidden_v2_latest.pt", val=val)

    save("hidden_v2_final.pt")
    print(f"[done] {os.path.join(checkpoint_dir, 'hidden_v2_final.pt')}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train HiDDeN v2")
    p.add_argument("--data", nargs="+", required=True, help="image folder(s)")
    p.add_argument("--out", default="checkpoints")
    p.add_argument("--steps", type=int, default=40000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--target-psnr", type=float, default=40.0)
    p.add_argument("--start-psnr", type=float, default=30.0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup-steps", type=int, default=2000)
    p.add_argument("--ramp-end", type=int, default=15000)
    p.add_argument("--val-every", type=int, default=2000)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--resume", default=None)
    p.add_argument("--device", default=None)
    a = p.parse_args()
    train(a.data, total_steps=a.steps, batch_size=a.batch_size,
          target_psnr=a.target_psnr, start_psnr=a.start_psnr, lr=a.lr, warmup_steps=a.warmup_steps,
          ramp_end=a.ramp_end, val_every=a.val_every, log_every=a.log_every,
          checkpoint_dir=a.out, resume_from=a.resume, num_workers=a.workers,
          device=a.device)

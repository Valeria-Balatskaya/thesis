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
# Checkpoints: hidden_v2_latest.pt (resume after a Colab disconnect; written
#              every --save-every steps and at every validation),
#              hidden_v2_best.pt   (best validation accuracy at full severity),
#              hidden_v2_final.pt.
#
# v3 options (all default to the v2 behaviour):
#   several --data folders with --data-weights (e.g. COCO + ABO products, 50/50);
#     validation holds out the last images of EACH folder and reports per source
#   --mask-type nvf --mask-floor 0.1 --zero-white --mask-renorm   texture-adaptive embedding,
#     applied inside training so the network learns to live with the mask
#   --headroom 2                pull pure white/black in by 2 levels before embedding
#   --distortions v3            adds brightness/levels (white clipping)
#   --overwrite-prob 0.15       re-embed a random second message; decoder must return the first
#
# v4 (key-conditioned) options:
#   --key-dim 64                encoder writes / decoder reads with a secret key (random per image
#                               in training; any 64-bit key works at deployment)
#   --wrong-key-weight 0.5      with a WRONG key the decoder must output nothing (logits -> 0)
#   --other-key-prob 0.3        someone protects the image with the same model but another key;
#                               our key must still read our message
#   --init-from hidden_v3_best.pt   warm start: new key layers start at zero = v3 exactly

import argparse
import math
import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, ConcatDataset, WeightedRandomSampler

from src.hidden_v2 import HiDDeNv2, Distortions, random_keys, load_warm_start
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


def _round_ste_eval(x: torch.Tensor) -> torch.Tensor:
    return torch.round(x * 255) / 255


@torch.no_grad()
def validate_sources(model: HiDDeNv2, loaders: dict, device: str, ecc_t: int) -> dict:
    """validate() per data source; the combined entry averages the sources."""
    per = {name: validate(model, loader, device, ecc_t) for name, loader in loaders.items()}
    if len(per) == 1:
        return next(iter(per.values()))
    names = model.distort.names
    keys = [k for k in next(iter(per.values())) if not k.startswith("_")]
    out = {n: {k: sum(v[n][k] for v in per.values()) / len(per) for k in ("bit_acc", "decodable")}
           for n in keys}
    out["_mean_bit_acc"] = sum(v["_mean_bit_acc"] for v in per.values()) / len(per)
    out["_psnr"] = sum(v["_psnr"] for v in per.values()) / len(per)
    out["_sources"] = {k: {"mean_bit_acc": v["_mean_bit_acc"], "psnr": v["_psnr"],
                           "mean_decodable": sum(v[n]["decodable"] for n in names) / len(names)}
                       for k, v in per.items()}
    return out


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
    stats = {name: [0.0, 0.0, 0] for name in model.distort.names}
    psnr_sum, n_img = 0.0, 0
    if model.key_dim:
        stats.update({"wrong_key": [0.0, 0.0, 0], "other_key_overwrite": [0.0, 0.0, 0]})
    for image, _ in loader:
        image = image.to(device)
        message = torch.randint(0, 2, (image.size(0), model.msg_len),
                                generator=gen).float().to(device)
        key = (torch.randint(0, 2, (image.size(0), model.key_dim), generator=gen).float().to(device)
               if model.key_dim else None)

        def add(name, logits):
            errors = ((logits > 0).float() != message).sum(dim=1)
            stats[name][0] += (1 - errors.float() / model.msg_len).sum().item()
            stats[name][1] += (errors <= ecc_t).float().sum().item()
            stats[name][2] += image.size(0)

        for name in model.distort.names:
            watermarked, logits, _ = model(image, message, 1.0, name, key=key)
            add(name, logits)
        if model.key_dim:
            # wrong key on a clean watermarked image: "decodable" here = a false read
            clean = _round_ste_eval(watermarked)
            add("wrong_key", model.decode(clean, random_keys(image.size(0), model.key_dim, device, avoid=key)))
            # someone re-protects with another key at full strength; our key reads
            other = random_keys(image.size(0), model.key_dim, device, avoid=key)
            second = torch.randint(0, 2, message.shape, generator=gen).float().to(device)
            over = _round_ste_eval(model.watermark(clean, second, key=other))
            add("other_key_overwrite", model.decode(over, key))
        mse = ((watermarked - image) ** 2).mean(dim=(1, 2, 3)).clamp(min=1e-10)
        psnr_sum += (10 * torch.log10(1 / mse)).sum().item()
        n_img += image.size(0)
    model.train()
    out = {name: {"bit_acc": a / n, "decodable": d / n} for name, (a, d, n) in stats.items()}
    out["_mean_bit_acc"] = sum(out[n]["bit_acc"] for n in model.distort.names) / len(model.distort.names)
    out["_psnr"] = psnr_sum / n_img
    return out


def _print_validation(step: int, val: dict) -> None:
    print(f"\n[val @ {step}] PSNR {val['_psnr']:.2f} dB   mean bit acc {val['_mean_bit_acc']:.3f}")
    print(f"  {'distortion':<13}{'bit acc':>9}{'decodable':>11}")
    for name in [k for k in val if not k.startswith("_")]:
        print(f"  {name:<13}{val[name]['bit_acc']:>9.3f}{val[name]['decodable']:>10.0%}")
    for src, v in val.get("_sources", {}).items():
        print(f"  [{src}] PSNR {v['psnr']:.2f} dB  bit acc {v['mean_bit_acc']:.3f}  "
              f"decodable {v['mean_decodable']:.0%}")
    print()


def train(
    image_dirs,
    msg_len: int = 63,
    image_size: int = 128,       # resolution the encoder / decoder work at
    train_size: int = 256,       # resolution of the "full-size" training images
    target_psnr: float = 40.0,
    start_psnr: float = 30.0,
    chroma_weight: float = 1.0,
    mask_floor: float = 1.0,
    mask_type: str = "flat",
    zero_white: bool = False,
    mask_renorm: bool = False,
    headroom: int = 0,
    distortion_set: str = "v2",
    overwrite_prob: float = 0.0,
    data_weights: list | None = None,
    save_every: int = 2000,
    key_dim: int = 0,
    wrong_key_weight: float = 0.5,
    other_key_prob: float = 0.0,
    init_from: str | None = None,
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

    # One dataset per folder; the last images of each folder are held out, so
    # every source is validated separately (e.g. COCO vs ABO product photos).
    train_parts, val_loaders = [], {}
    per_folder_val = max(1, val_images // len(image_dirs))
    for d in image_dirs:
        ds = WatermarkDataset([d], msg_len=msg_len, image_size=train_size)
        ds.paths.sort()
        n_val = min(per_folder_val, len(ds) // 5)
        train_parts.append(Subset(ds, range(len(ds) - n_val)))
        val_loaders[os.path.basename(os.path.normpath(d))] = DataLoader(
            Subset(ds, range(len(ds) - n_val, len(ds))), batch_size=batch_size,
            shuffle=False, num_workers=num_workers)
    train_set = ConcatDataset(train_parts)
    for d, part in zip(image_dirs, train_parts):
        print(f"[train_v2] {d}: {len(part)} training images")
    print(f"[train_v2] {len(train_set)} training / "
          f"{sum(len(l.dataset) for l in val_loaders.values())} validation images")

    if data_weights:
        # Sample each source with the given share, whatever its size
        assert len(data_weights) == len(image_dirs), "one --data-weights value per --data folder"
        weights = torch.cat([torch.full((len(p),), w / len(p)) for p, w in zip(train_parts, data_weights)])
        sampler = WeightedRandomSampler(weights, num_samples=len(train_set), replacement=True)
        loader = DataLoader(train_set, batch_size=batch_size, sampler=sampler,
                            num_workers=num_workers, pin_memory=(device == "cuda"),
                            drop_last=True, persistent_workers=num_workers > 0)
    else:
        loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, pin_memory=(device == "cuda"),
                            drop_last=True, persistent_workers=num_workers > 0)

    model = HiDDeNv2(msg_len=msg_len, image_size=image_size, target_psnr=target_psnr,
                     chroma_weight=chroma_weight, mask_floor=mask_floor,
                     mask_type=mask_type, zero_white=zero_white, mask_renorm=mask_renorm,
                     headroom=headroom, distortion_set=distortion_set,
                     key_dim=key_dim).to(device)
    params = list(model.encoder.parameters()) + list(model.decoder.parameters())
    opt = torch.optim.Adam(params, lr=lr)

    step, best_acc = 0, 0.0
    if init_from and not (resume_from and os.path.exists(resume_from)):
        fresh = load_warm_start(model, torch.load(init_from, map_location=device,
                                                  weights_only=False)["model"])
        print(f"[train_v2] warm start from {init_from}; new/extended: {len(fresh)} tensors")
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
            "chroma_weight": chroma_weight,
            "mask_floor": mask_floor,
            "mask_type": mask_type,
            "zero_white": zero_white,
            "mask_renorm": mask_renorm,
            "headroom": headroom,
            "distortion_set": distortion_set,
            "overwrite_prob": overwrite_prob,
            "key_dim": key_dim,
            "wrong_key_weight": wrong_key_weight,
            "other_key_prob": other_key_prob,
            "init_from": init_from,
            "data": [str(d) for d in image_dirs],
            "data_weights": data_weights,
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
            # overwrite augmentation only once the networks communicate at all
            ow = overwrite_prob if step >= warmup_steps else 0.0
            key = random_keys(image.size(0), key_dim, device) if key_dim else None
            ok = other_key_prob if step >= warmup_steps else 0.0
            _, logits, _ = model(image, message, severity, psnr=psnr, overwrite_prob=ow,
                                 key=key, other_key_prob=ok)
            loss = F.binary_cross_entropy_with_logits(logits, message)
            if key_dim and wrong_key_weight > 0:
                # a wrong key must give no information: logits pushed to 0 (p = 0.5)
                wrong = random_keys(image.size(0), key_dim, device, avoid=key)
                wrong_logits = model.decode(model.last_attacked, wrong)
                loss = loss + wrong_key_weight * F.binary_cross_entropy_with_logits(
                    wrong_logits, torch.full_like(message, 0.5))
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

            if step % save_every == 0 and step % val_every != 0:
                save("hidden_v2_latest.pt")

            if step % val_every == 0 or step == total_steps:
                val = validate_sources(model, val_loaders, device, ecc_t)
                _print_validation(step, val)
                # "best" only counts once severity and quality are at their final values
                score = val["_mean_bit_acc"]
                if key_dim:
                    # v4: also reward reading through another key's overwrite, and
                    # penalise any decodable read with a wrong key (= false positive)
                    score = (0.7 * score + 0.3 * val["other_key_overwrite"]["bit_acc"]
                             - val["wrong_key"]["decodable"])
                if step >= ramp_end and score > best_acc:
                    best_acc = score
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
    p.add_argument("--chroma-weight", type=float, default=1.0,
                   help="colour changes count this many times against the quality budget")
    p.add_argument("--mask-floor", type=float, default=1.0,
                   help="watermark amplitude on flat areas (1.0 = no masking)")
    p.add_argument("--mask-type", default="flat", choices=["flat", "nvf", "none"])
    p.add_argument("--zero-white", action="store_true", help="no watermark on pure-white pixels")
    p.add_argument("--mask-renorm", action="store_true",
                   help="move the energy removed from flat areas into textured ones")
    p.add_argument("--headroom", type=int, default=0, help="levels kept free at pure white/black")
    p.add_argument("--distortions", default="v2", choices=["v2", "v3"])
    p.add_argument("--overwrite-prob", type=float, default=0.0)
    p.add_argument("--data-weights", type=float, nargs="+", default=None,
                   help="sampling share per --data folder, e.g. 0.5 0.5")
    p.add_argument("--save-every", type=int, default=2000)
    p.add_argument("--key-dim", type=int, default=0, help="> 0: key-conditioned model (v4)")
    p.add_argument("--wrong-key-weight", type=float, default=0.5)
    p.add_argument("--other-key-prob", type=float, default=0.0)
    p.add_argument("--init-from", default=None, help="warm start from a v2/v3 checkpoint")
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
          target_psnr=a.target_psnr, start_psnr=a.start_psnr,
          chroma_weight=a.chroma_weight, mask_floor=a.mask_floor, mask_type=a.mask_type,
          zero_white=a.zero_white, mask_renorm=a.mask_renorm, headroom=a.headroom,
          distortion_set=a.distortions, overwrite_prob=a.overwrite_prob,
          data_weights=a.data_weights, save_every=a.save_every,
          key_dim=a.key_dim, wrong_key_weight=a.wrong_key_weight,
          other_key_prob=a.other_key_prob, init_from=a.init_from,
          lr=a.lr, warmup_steps=a.warmup_steps,
          ramp_end=a.ramp_end, val_every=a.val_every, log_every=a.log_every,
          checkpoint_dir=a.out, resume_from=a.resume, num_workers=a.workers,
          device=a.device)

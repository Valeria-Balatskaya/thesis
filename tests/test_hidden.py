import os
import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

from src.hidden_watermark import HiDDeN
from src.metrics import compute
from src import attacks

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

IMAGES = ["baboon", "airplane", "peppers", "splash", "house"]
SELLER_ID = "SELLER_ID:0042"
MSG_LEN = 30
IMG_SIZE = 128
EVAL_SIZE = 512
BATCH_SIZE = 2
EPOCHS = 5
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTDIR = "output/hidden"
os.makedirs(OUTDIR, exist_ok=True)


def load_img(path: str, size: int = IMG_SIZE) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((size, size))
    arr = np.asarray(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def bits_from_text(text: str, n: int = MSG_LEN) -> torch.Tensor:
    bits = []
    for byte in text.encode("utf-8"):
        for i in range(7, -1, -1):
            bits.append((byte >> i) & 1)
    bits = (bits + [0] * n)[:n]
    return torch.tensor(bits, dtype=torch.float32)


def save_tensor_image_resized(x: torch.Tensor, path: str, out_size=(EVAL_SIZE, EVAL_SIZE)):
    arr = (x.squeeze(0).detach().cpu().permute(1, 2, 0).numpy().clip(0, 1) * 255).astype(np.uint8)
    img = Image.fromarray(arr).resize(out_size, Image.Resampling.BICUBIC)
    img.save(path)


def bit_accuracy(logits: torch.Tensor, target: torch.Tensor) -> float:
    pred = (torch.sigmoid(logits) > 0.5).float()
    return float((pred == target).float().mean().item())


images = torch.stack([load_img(f"data/sipi/{name}.png") for name in IMAGES])
messages = torch.stack([bits_from_text(SELLER_ID) for _ in IMAGES])
loader = DataLoader(TensorDataset(images, messages), batch_size=BATCH_SIZE, shuffle=True)

model = HiDDeN(MSG_LEN).to(DEVICE)
optimizer_encdec = torch.optim.Adam(list(model.encoder.parameters()) + list(model.decoder.parameters()), lr=LR)
optimizer_adv = torch.optim.Adam(model.adversary.parameters(), lr=LR)

criterion_msg = nn.BCEWithLogitsLoss()
criterion_img = nn.MSELoss()
criterion_adv = nn.BCEWithLogitsLoss()

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0
    epoch_acc = 0.0
    batches = 0
    for x, m in loader:
        x = x.to(DEVICE)
        m = m.to(DEVICE)

        encoded = model.encoder(x, m)
        real_logits = model.adversary(x)
        fake_logits = model.adversary(encoded.detach())
        adv_real = criterion_adv(real_logits, torch.ones_like(real_logits))
        adv_fake = criterion_adv(fake_logits, torch.zeros_like(fake_logits))
        loss_adv = 0.5 * (adv_real + adv_fake)
        optimizer_adv.zero_grad()
        loss_adv.backward()
        optimizer_adv.step()

        encoded, attacked, decoded_logits = model(x, m)
        image_loss = criterion_img(encoded, x)
        message_loss = criterion_msg(decoded_logits, m)
        fool_loss = criterion_adv(model.adversary(encoded), torch.ones_like(real_logits))
        loss = message_loss + 8.0 * image_loss + 0.01 * fool_loss

        optimizer_encdec.zero_grad()
        loss.backward()
        optimizer_encdec.step()

        epoch_loss += float(loss.item())
        epoch_acc += bit_accuracy(decoded_logits.detach(), m)
        batches += 1

    print(f"epoch {epoch+1}/{EPOCHS} loss={epoch_loss/batches:.4f} bit_acc={epoch_acc/batches:.4f}")

torch.save(model.state_dict(), f"{OUTDIR}/hidden_model.pt")

rows = []
model.eval()
with torch.no_grad():
    for name in IMAGES:
        original_path = f"data/sipi/{name}.png"
        stego_path = f"{OUTDIR}/stego_{name}.png"
        jpeg_path = f"{OUTDIR}/jpeg70_{name}.png"
        resize_path = f"{OUTDIR}/resize75_{name}.png"

        x = load_img(original_path).unsqueeze(0).to(DEVICE)
        m = bits_from_text(SELLER_ID).unsqueeze(0).to(DEVICE)
        encoded = model.encoder(x, m)
        save_tensor_image_resized(encoded, stego_path, out_size=(EVAL_SIZE, EVAL_SIZE))

        attacks.jpeg_compress(stego_path, jpeg_path, quality=70)
        attacks.resize_attack(stego_path, resize_path, scale=0.75)

        jpeg_tensor = load_img(jpeg_path).unsqueeze(0).to(DEVICE)
        resize_tensor = load_img(resize_path).unsqueeze(0).to(DEVICE)

        jpeg_acc = bit_accuracy(model.decoder(jpeg_tensor), m)
        resize_acc = bit_accuracy(model.decoder(resize_tensor), m)
        clean_acc = bit_accuracy(model.decoder(encoded), m)

        stego_metrics = compute(original_path, stego_path)
        jpeg_metrics = compute(original_path, jpeg_path)
        resize_metrics = compute(original_path, resize_path)

        rows.append([
            name,
            round(clean_acc, 4),
            round(jpeg_acc, 4),
            round(resize_acc, 4),
            round(stego_metrics['PSNR_dB'], 4),
            round(stego_metrics['SSIM'], 6),
            round(jpeg_metrics['PSNR_dB'], 4),
            round(resize_metrics['PSNR_dB'], 4),
        ])

csv_path = f"{OUTDIR}/hidden_results.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "image",
        "clean_bit_acc",
        "jpeg70_bit_acc",
        "resize75_bit_acc",
        "stego_psnr_db",
        "stego_ssim",
        "jpeg70_psnr_db",
        "resize75_psnr_db",
    ])
    writer.writerows(rows)

print("saved", csv_path)
for row in rows:
    print(row)
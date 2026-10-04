# src/hidden_inference.py
# Load a trained HiDDeN checkpoint and use it to embed / recover messages
# on real images (not training data).

import hashlib
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from src.hidden_watermark import HiDDeN


def _seller_to_bits(seller_id: str, msg_len: int) -> torch.Tensor:
    """Deterministically turn a seller_id string into a msg_len-bit vector."""
    h = hashlib.sha256(seller_id.encode()).digest()
    bits = []
    for byte in h:
        for i in range(7, -1, -1):
            bits.append((byte >> i) & 1)
            if len(bits) == msg_len:
                return torch.tensor(bits, dtype=torch.float32)
    return torch.tensor(bits[:msg_len], dtype=torch.float32)


def _ber(a: torch.Tensor, b: torch.Tensor) -> float:
    """Bit error rate between two binary tensors."""
    return (a != b).float().mean().item()


class HiddenModel:
    """
    Wraps a trained HiDDeN checkpoint for embedding and decoding.
    Works on arbitrary image sizes by processing at the model's native
    training size, then resizing back to original dimensions.
    """

    def __init__(self, checkpoint_path: str, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.msg_len = ckpt.get("msg_len", 48)
        self.image_size = ckpt.get("image_size", 128)

        self.model = HiDDeN(msg_len=self.msg_len).to(self.device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()

        self.to_tensor = transforms.ToTensor()

    @torch.no_grad()
    def embed(self, image_path: str, seller_id: str, output_path: str,
              mode: str = "residual", strength: float = 1.0) -> dict:
        """
        Embed seller_id into the image.

        mode="residual" (default): run the encoder at the model's native size,
            keep only the watermark residual (encoded - input), upsample that
            residual to the original resolution and add it to the untouched
            full-resolution image. Image detail is preserved (~40 dB PSNR).
        mode="legacy": the original approach — the encoder's 128px output is
            upsampled and *replaces* the image. Destroys detail (~20-29 dB PSNR);
            kept only so the thesis can report the before/after comparison.
        """
        img = Image.open(image_path).convert("RGB")
        orig_size = img.size  # (W, H)

        img_resized = img.resize((self.image_size, self.image_size), Image.BICUBIC)
        x = self.to_tensor(img_resized).unsqueeze(0).to(self.device)

        message = _seller_to_bits(seller_id, self.msg_len).unsqueeze(0).to(self.device)
        encoded = self.model.encoder(x, message)

        if mode == "legacy":
            encoded_np = (encoded.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255)
            encoded_np = encoded_np.clip(0, 255).astype(np.uint8)
            out = Image.fromarray(encoded_np).resize(orig_size, Image.BICUBIC)
        elif mode == "residual":
            residual = (encoded - x).squeeze(0).permute(1, 2, 0).cpu().numpy()
            # Upsample each channel as float so the low-amplitude signal isn't quantised
            residual_full = np.stack([
                np.array(Image.fromarray(residual[:, :, c].astype(np.float32), "F")
                         .resize(orig_size, Image.BICUBIC))
                for c in range(3)
            ], axis=2)
            base = np.array(img, dtype=np.float32)
            out_np = np.clip(base + residual_full * 255.0 * strength, 0, 255)
            out = Image.fromarray(out_np.astype(np.uint8))
        else:
            raise ValueError(f"unknown embed mode: {mode}")

        out.save(output_path, format="PNG")
        return {"seller_id": seller_id, "msg_len": self.msg_len, "mode": mode}

    @torch.no_grad()
    def decode(self, image_path: str) -> torch.Tensor:
        img = Image.open(image_path).convert("RGB")
        img_resized = img.resize((self.image_size, self.image_size), Image.BICUBIC)
        x = self.to_tensor(img_resized).unsqueeze(0).to(self.device)
        logits = self.model.decoder(x).squeeze(0)
        bits = (torch.sigmoid(logits) > 0.5).float().cpu()
        return bits

    @torch.no_grad()
    def identify(self, image_path: str, seller_ids: list[str]) -> dict:
        """
        Given a list of candidate seller_ids, return the one whose
        expected message is closest (lowest BER) to what the decoder recovers.
        """
        recovered = self.decode(image_path)
        scores = []
        for seller_id in seller_ids:
            expected = _seller_to_bits(seller_id, self.msg_len)
            ber = _ber(recovered, expected)
            scores.append({"seller_id": seller_id, "ber": ber})
        scores.sort(key=lambda s: s["ber"])
        return {"match": scores[0], "ranking": scores}
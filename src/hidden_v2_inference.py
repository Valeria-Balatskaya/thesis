# src/hidden_v2_inference.py
# Use a trained HiDDeN v2 checkpoint on real image files.
#
# Unlike src/hidden_inference.py (v1), detection here is blind: read_product()
# returns the product ID directly from the suspect image — no original image
# and no list of candidates needed.

import numpy as np
import torch
from PIL import Image

from src.hidden_v2 import HiDDeNv2
from src.payload import encode_product_id, decode_payload, PAYLOAD_BITS


class HiddenV2Model:
    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        self.device = device
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if ckpt.get("arch") != "hidden_v2":
            raise ValueError(f"{checkpoint_path} is not a HiDDeN v2 checkpoint "
                             "(use src.hidden_inference.HiddenModel for v1)")
        self.msg_len = ckpt["msg_len"]
        self.image_size = ckpt["image_size"]
        self.target_psnr = ckpt["target_psnr"]
        self.model = HiDDeNv2(self.msg_len, self.image_size, self.target_psnr).to(device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()

    def _load(self, image_path: str) -> torch.Tensor:
        arr = np.array(Image.open(image_path).convert("RGB"), dtype=np.float32) / 255
        return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def embed_bits(self, image_path: str, bits, output_path: str,
                   strength: float = 1.0) -> None:
        """Embed msg_len raw bits. strength scales the residual (1.0 = target_psnr)."""
        message = torch.tensor(np.asarray(bits), dtype=torch.float32,
                               device=self.device).view(1, self.msg_len)
        out = self.model.watermark(self._load(image_path), message, strength)
        out = (out.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).round()
        Image.fromarray(out.astype(np.uint8)).save(output_path, format="PNG")

    @torch.no_grad()
    def decode_bits(self, image_path: str) -> np.ndarray:
        logits = self.model.decode(self._load(image_path)).squeeze(0)
        return (logits > 0).cpu().numpy().astype(np.uint8)

    # ─── product-ID payload (BCH protected, see src/payload.py) ───

    def embed_product(self, image_path: str, product_id: int, output_path: str,
                      strength: float = 1.0) -> dict:
        if self.msg_len != PAYLOAD_BITS:
            raise ValueError(f"checkpoint embeds {self.msg_len} bits, payload needs {PAYLOAD_BITS}")
        self.embed_bits(image_path, encode_product_id(product_id), output_path, strength)
        return {"product_id": product_id, "output_path": output_path}

    def read_product(self, image_path: str) -> dict:
        """Blind detection. product_id is None when no valid watermark is found."""
        bits = self.decode_bits(image_path)
        return {**decode_payload(bits), "bits": bits}

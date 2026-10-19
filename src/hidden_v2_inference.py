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


# Original aspect ratios tried when a suspect looks centre-cropped
ASPECT_RATIOS = [(4, 3), (3, 4), (3, 2), (2, 3), (16, 9), (9, 16)]


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
        self.model = HiDDeNv2(self.msg_len, self.image_size, self.target_psnr,
                              ckpt.get("chroma_weight", 1.0),
                              ckpt.get("mask_floor", 1.0)).to(device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()

    def _load(self, image_path: str) -> torch.Tensor:
        arr = np.array(Image.open(image_path).convert("RGB"), dtype=np.float32) / 255
        return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def embed_bits(self, image_path: str, bits, output_path: str,
                   strength: float = 1.0, mask_floor: float | None = None) -> None:
        """
        Embed msg_len raw bits. strength scales the residual (1.0 = target_psnr).
        mask_floor < 1 weakens the watermark on flat areas (see hidden_v2.texture_mask);
        None uses whatever the checkpoint was trained with.
        """
        message = torch.tensor(np.asarray(bits), dtype=torch.float32,
                               device=self.device).view(1, self.msg_len)
        out = self.model.watermark(self._load(image_path), message, strength,
                                   mask_floor=mask_floor)
        out = (out.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).round()
        Image.fromarray(out.astype(np.uint8)).save(output_path, format="PNG")

    @torch.no_grad()
    def _decode_tensor(self, x: torch.Tensor) -> np.ndarray:
        logits = self.model.decode(x).squeeze(0)
        return (logits > 0).cpu().numpy().astype(np.uint8)

    def decode_bits(self, image_path: str) -> np.ndarray:
        return self._decode_tensor(self._load(image_path))

    # ─── product-ID payload (BCH protected, see src/payload.py) ───

    def embed_product(self, image_path: str, product_id: int, output_path: str,
                      strength: float = 1.0, mask_floor: float | None = None) -> dict:
        if self.msg_len != PAYLOAD_BITS:
            raise ValueError(f"checkpoint embeds {self.msg_len} bits, payload needs {PAYLOAD_BITS}")
        self.embed_bits(image_path, encode_product_id(product_id), output_path, strength,
                        mask_floor)
        return {"product_id": product_id, "output_path": output_path}

    def read_product(self, image_path: str, try_aspects: bool = True) -> dict:
        """
        Blind detection. product_id is None when no valid watermark is found.

        The watermark is position-dependent, so a centre crop to another aspect
        ratio (marketplace thumbnails) misaligns it. If the direct read fails
        and try_aspects is set, the image is padded back to common original
        aspect ratios and read again. Each extra attempt is another chance of a
        false positive: worst case len(ASPECT_RATIOS) + 1 times the single-read rate.
        """
        x = self._load(image_path)
        bits = self._decode_tensor(x)
        result = {**decode_payload(bits), "bits": bits, "aspect": None}
        if result["product_id"] is not None or not try_aspects:
            return result
        _, _, h, w = x.shape
        for aw, ah in ASPECT_RATIOS:
            # Smallest canvas of that aspect ratio that contains the image
            new_w, new_h = max(w, round(h * aw / ah)), max(h, round(w * ah / aw))
            if (new_w, new_h) == (w, h):
                continue
            canvas = torch.full((1, 3, new_h, new_w), 0.5, device=x.device)
            top, left = (new_h - h) // 2, (new_w - w) // 2
            canvas[:, :, top:top + h, left:left + w] = x
            attempt = decode_payload(self._decode_tensor(canvas))
            if attempt["product_id"] is not None:
                return {**attempt, "bits": bits, "aspect": f"{aw}:{ah}"}
        return result

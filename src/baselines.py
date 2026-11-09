# src/baselines.py
# Pretrained robust watermarks as baselines, behind one interface (Part 4, roadmap 4).
#
#   TrustMark           Bui et al., Adobe (pip install trustmark). 100 encoded bits =
#                       61 data bits + BCH_5 (the package's default), variant "Q"
#                       (default). Native decision: the package's BCH decode succeeds.
#   Watermark Anything  Sander et al., Meta (github.com/facebookresearch/watermark-anything,
#                       MIT weights wam_mit.pth). 32 bits, no error correction; it
#                       predicts a watermark mask and a message per pixel. Native
#                       decision: predicted watermarked area >= 50% of the image
#                       (as in the paper's detection experiments), message = average
#                       over the predicted mask.
#   Ours (HiDDeN v3/v4) 63 bits = 20-bit copy ID + 10-bit HMAC tag + BCH(63,30,t=6).
#                       Native decision: BCH decodes and the HMAC tag checks.
#
# Every system: embed(src, dst, i, strength) writes a watermarked PNG carrying message i;
# read_rgb(rgb) -> {"ok": exact message for that i recovered, "detected": native
# "watermark present" decision, "bit_acc": raw bit accuracy}. "ok" needs the expected
# message, so read_rgb takes i as well.
# Install the optional baselines: pip install -r requirements-baselines.txt and
# bash scripts/get_baselines.sh (clones WAM + downloads its weights into third_party/).

import os
import sys

import numpy as np
import torch
from PIL import Image

from src.payload import encode_product_id, decode_payload

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WAM_DIR = os.path.join(ROOT, "third_party", "watermark-anything")


def _rng_bits(i: int, n: int, salt: int) -> np.ndarray:
    return np.random.default_rng(1_000_003 * salt + i).integers(0, 2, n).astype(np.uint8)


class Ours:
    name = "ours"

    def __init__(self, checkpoint: str, strength: float = 1.0, mask_floor=None, label: str | None = None):
        from src.hidden_v2_inference import HiddenV2Model
        self.model = HiddenV2Model(checkpoint)
        self.default_strength, self.mask_floor = strength, mask_floor
        self.name = label or os.path.basename(checkpoint).replace("hidden_", "").replace("_best.pt", "")
        self.bits = 63

    def copy_id(self, i):
        return 1000 + i

    def embed(self, src, dst, i, strength=None):
        self.model.embed_product(src, self.copy_id(i), dst,
                                 strength=self.default_strength if strength is None else strength,
                                 mask_floor=self.mask_floor)

    def read_rgb(self, rgb, i):
        x = torch.from_numpy(rgb.astype(np.float32) / 255).permute(2, 0, 1).unsqueeze(0)
        bits = self.model._decode_tensor(x.to(self.model.device))
        got = decode_payload(bits)["product_id"]
        return {"ok": got == self.copy_id(i), "detected": got is not None,
                "bit_acc": float((bits == encode_product_id(self.copy_id(i))).mean())}

    def read_path(self, path, i):
        """Blind read as deployed: includes the aspect-ratio retry of read_product."""
        r = self.model.read_product(path)
        return {"ok": r["product_id"] == self.copy_id(i), "detected": r["product_id"] is not None,
                "bit_acc": float((r["bits"] == encode_product_id(self.copy_id(i))).mean())}


class TrustMarkBaseline:
    name = "trustmark"

    def __init__(self, model_type: str = "Q", strength: float = 1.0):
        from trustmark import TrustMark
        self.tm = TrustMark(verbose=False, model_type=model_type, device="cpu")
        self.default_strength = strength
        self.data_bits = self.tm.schemaCapacity()          # 61 with BCH_5
        self.bits = 100
        self.name = f"trustmark_{model_type}"

    def message(self, i):
        return "".join(map(str, _rng_bits(i, self.data_bits, 7)))

    def embed(self, src, dst, i, strength=None):
        img = Image.open(src).convert("RGB")
        out = self.tm.encode(img, self.message(i), MODE="binary",
                             WM_STRENGTH=self.default_strength if strength is None else strength)
        out.convert("RGB").save(dst, format="PNG")

    def read_rgb(self, rgb, i):
        img = Image.fromarray(rgb)
        secret, detected, _ = self.tm.decode(img, MODE="binary")
        # raw 100-bit accuracy against the encoded codeword
        enc = np.asarray(self.tm.ecc.encode_binary([self.message(i)])).reshape(-1)[:self.bits] > 0.5
        sub = self.tm.get_the_image_for_processing(img)
        sub = sub.resize((self.tm.model_resolution_dec, self.tm.model_resolution_dec), Image.BILINEAR)
        from torchvision import transforms
        t = transforms.ToTensor()(sub).unsqueeze(0) * 2.0 - 1.0
        with torch.no_grad():
            raw = (self.tm.decoder.decoder(t.to(self.tm.decoder.device)) > 0).cpu().numpy().reshape(-1)
        return {"ok": bool(detected) and secret == self.message(i), "detected": bool(detected),
                "bit_acc": float((raw[:len(enc)] == enc).mean())}

    def read_path(self, path, i):
        return self.read_rgb(np.asarray(Image.open(path).convert("RGB")), i)


class WAMBaseline:
    name = "wam"

    def __init__(self, scaling_w: float | None = None, weights: str = "wam_mit.pth"):
        if not os.path.isdir(WAM_DIR):
            raise ImportError("Watermark Anything not found: run bash scripts/get_baselines.sh")
        sys.path.insert(0, WAM_DIR)
        from notebooks.inference_utils import load_model_from_checkpoint
        from watermark_anything.data.metrics import msg_predict_inference
        from watermark_anything.data.transforms import default_transform, unnormalize_img
        self._predict, self._tf, self._unnorm = msg_predict_inference, default_transform, unnormalize_img
        cwd = os.getcwd()
        os.chdir(WAM_DIR)                      # params.json refers to configs/ relatively
        try:
            self.wam = load_model_from_checkpoint("checkpoints/params.json",
                                                  os.path.join("checkpoints", weights)).eval()
        finally:
            os.chdir(cwd)
        self.default_strength = self.wam.scaling_w if scaling_w is None else scaling_w
        self.bits = 32

    def message(self, i):
        return _rng_bits(i, self.bits, 11)

    @torch.no_grad()
    def embed(self, src, dst, i, strength=None):
        img = Image.open(src).convert("RGB")
        x = self._tf(img).unsqueeze(0)
        self.wam.scaling_w = self.default_strength if strength is None else strength
        out = self.wam.embed(x, torch.tensor(self.message(i), dtype=torch.float32).view(1, -1))
        y = self._unnorm(out["imgs_w"]).clamp(0, 1)
        arr = (y[0].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
        Image.fromarray(arr).save(dst, format="PNG")

    @torch.no_grad()
    def read_rgb(self, rgb, i):
        x = self._tf(Image.fromarray(rgb)).unsqueeze(0)
        preds = self.wam.detect(x)["preds"]
        mask = torch.sigmoid(preds[:, :1])
        area = float((mask > 0.5).float().mean())
        bits = self._predict(preds[:, 1:], mask).reshape(-1).numpy().astype(np.uint8)
        acc = float((bits == self.message(i)).mean())
        return {"ok": area >= 0.5 and acc == 1.0, "detected": area >= 0.5, "bit_acc": acc}

    def read_path(self, path, i):
        return self.read_rgb(np.asarray(Image.open(path).convert("RGB")), i)

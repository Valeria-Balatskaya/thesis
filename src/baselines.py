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
# Peel-off / order proof (src/defence.py) use two more methods per system:
#   foreign_read(rgb) -> the system's own message if its native decision says "present", else None
#   residual(rgb, message, strength=None) -> float32 HxWx3 in [0,1] units: what embedding that
#       message on this image adds (embed(rgb) - rgb), at the system's default strength
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

    def foreign_read(self, rgb):
        secret, detected, _ = self.tm.decode(Image.fromarray(rgb), MODE="binary")
        return secret if detected and secret else None

    def residual(self, rgb, message, strength=None):
        out = self.tm.encode(Image.fromarray(rgb), message, MODE="binary",
                             WM_STRENGTH=self.default_strength if strength is None else strength)
        return (np.asarray(out.convert("RGB"), dtype=np.float32) - rgb.astype(np.float32)) / 255

    def remove(self, src, dst):
        """TrustMark-RM: the package's learned watermark remover (an attack on any system)."""
        out = self.tm.remove_watermark(Image.open(src).convert("RGB"))
        out.convert("RGB").save(dst, format="PNG")


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

    @torch.no_grad()
    def foreign_read(self, rgb):
        x = self._tf(Image.fromarray(rgb)).unsqueeze(0)
        preds = self.wam.detect(x)["preds"]
        mask = torch.sigmoid(preds[:, :1])
        if float((mask > 0.5).float().mean()) < 0.5:
            return None
        return self._predict(preds[:, 1:], mask).reshape(-1).numpy().astype(np.uint8)

    @torch.no_grad()
    def residual(self, rgb, message, strength=None):
        x = self._tf(Image.fromarray(rgb)).unsqueeze(0)
        self.wam.scaling_w = self.default_strength if strength is None else strength
        out = self.wam.embed(x, torch.tensor(np.asarray(message), dtype=torch.float32).view(1, -1))
        y = self._unnorm(out["imgs_w"]).clamp(0, 1)[0].permute(1, 2, 0).numpy()
        return (y - rgb.astype(np.float32) / 255).astype(np.float32)


class VideoSealBaseline:
    """
    Meta Video Seal 1.0 (Fernandez et al., 2024; pip videoseal, image mode, weights
    y_256b_img.pth). 256 bits, no error correction, luminance-only embedder with a JND
    attenuation map. Default strength = the card's scaling_w (0.2).
    Native decision (no ECC, so no "exact message" in the strict sense): the expected
    256-bit message matches in >= VS_MIN_MATCH bits, i.e. a one-sided binomial p-value
    <= 1e-6 against random bits (the same budget as our candidate verification). This
    needs the expected message, like looking up one registered ID.
    Install: pip install --no-deps videoseal==1.0.1 pycocotools lpips calflops PyWavelets
    (its pinned timm==0.9.16 is not needed; it runs with timm 1.0.x). The config file
    configs/attenuation.yaml lives in third_party/videoseal/ (scripts/get_baselines.sh).
    """
    name = "videoseal"
    VS_DIR = os.path.join(ROOT, "third_party", "videoseal")

    def __init__(self, strength: float | None = None):
        from pathlib import Path
        import types
        from scipy.stats import binom
        # videoseal imports PyAV only for its video augmentations; PyAV's bundled libav
        # clashes with OpenCV's on macOS, so a stub stands in (image mode never uses it)
        stub = "av" not in sys.modules
        if stub:
            sys.modules["av"] = types.ModuleType("av")
        try:
            import videoseal
            from videoseal.utils.cfg import setup_model_from_model_card  # noqa: F401 (imports av)
        finally:
            if stub:                          # diffusers checks importlib.find_spec("av") later
                sys.modules.pop("av", None)
        card = Path(os.path.dirname(videoseal.__file__)) / "cards" / "videoseal_1.0.yaml"
        cwd = os.getcwd()
        os.chdir(self.VS_DIR)                   # card refers to configs/ and ckpts/ relatively
        try:
            self.vs = videoseal.load(card).eval()
        finally:
            os.chdir(cwd)
        self.default_strength = float(self.vs.blender.scaling_w) if strength is None else strength
        self.bits = 256
        # smallest k with P(Binomial(256, 0.5) >= k) <= 1e-6
        self.min_match = int(next(k for k in range(129, 257) if binom.sf(k - 1, 256, 0.5) <= 1e-6))

    def message(self, i):
        return _rng_bits(i, self.bits, 13)

    @torch.no_grad()
    def embed(self, src, dst, i, strength=None):
        x = torch.from_numpy(np.asarray(Image.open(src).convert("RGB"), dtype=np.float32) / 255)
        x = x.permute(2, 0, 1).unsqueeze(0)
        self.vs.blender.scaling_w = self.default_strength if strength is None else strength
        m = torch.tensor(self.message(i), dtype=torch.float32).view(1, -1)
        y = self.vs.embed(x, m, is_video=False)["imgs_w"].clamp(0, 1)
        arr = (y[0].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
        Image.fromarray(arr).save(dst, format="PNG")

    @torch.no_grad()
    def decode_bits(self, rgb):
        x = torch.from_numpy(rgb.astype(np.float32) / 255).permute(2, 0, 1).unsqueeze(0)
        preds = self.vs.detect(x, is_video=False)["preds"]
        return (preds[0, 1:].reshape(-1) > 0).numpy().astype(np.uint8)

    def read_rgb(self, rgb, i):
        bits = self.decode_bits(rgb)
        match_ = int((bits == self.message(i)).sum())
        ok = match_ >= self.min_match
        return {"ok": ok, "detected": ok, "bit_acc": match_ / self.bits}

    def read_path(self, path, i):
        return self.read_rgb(np.asarray(Image.open(path).convert("RGB")), i)

    # Blind "present" decision for peel-off: Video Seal has no ECC and no detection head
    # in image mode, but the decoder's mean |logit| separates marked images (3.6-11.6 on
    # 10 photos, also after JPEG q70) from clean ones (0.2-1.6). Threshold measured on
    # the negatives in tests/test_peel_off.py.
    PRESENT_LOGIT = 3.0

    @torch.no_grad()
    def foreign_read(self, rgb):
        x = torch.from_numpy(rgb.astype(np.float32) / 255).permute(2, 0, 1).unsqueeze(0)
        logits = self.vs.detect(x, is_video=False)["preds"][0, 1:].reshape(-1)
        if float(logits.abs().mean()) < self.PRESENT_LOGIT:
            return None
        return (logits > 0).numpy().astype(np.uint8)

    @torch.no_grad()
    def residual(self, rgb, message, strength=None):
        x = torch.from_numpy(rgb.astype(np.float32) / 255).permute(2, 0, 1).unsqueeze(0)
        self.vs.blender.scaling_w = self.default_strength if strength is None else strength
        m = torch.tensor(np.asarray(message), dtype=torch.float32).view(1, -1)
        y = self.vs.embed(x, m, is_video=False)["imgs_w"].clamp(0, 1)[0].permute(1, 2, 0).numpy()
        return (y - rgb.astype(np.float32) / 255).astype(np.float32)

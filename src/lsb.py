import numpy as np
from PIL import Image


def text_to_bits(text: str) -> list[int]:
    """Convert a UTF-8 string to a flat list of bits."""
    bits = []
    for byte in text.encode("utf-8"):
        for i in range(7, -1, -1):
            bits.append((byte >> i) & 1)
    return bits


def bits_to_text(bits: list[int]) -> str:
    """Reconstruct a UTF-8 string from a flat list of bits."""
    byte_array = bytearray()
    for i in range(0, len(bits), 8):
        byte = 0
        for b in bits[i:i+8]:
            byte = (byte << 1) | b
        byte_array.append(byte)
    return byte_array.decode("utf-8", errors="replace")


def embed(image_path: str, message: str, output_path: str, k: int = 1) -> None:
    """
    Embed message into image using k-LSB substitution.
    Implements Eq. (4) from Chan & Cheng 2004:
        x'_li = x_li - (x_li mod 2^k) + m'_i
    k=1 gives worst-case PSNR ~48.13 dB (Table 1, Chan & Cheng 2004).
    """
    img = np.array(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    bits = text_to_bits(message)

    # Prepend a 32-bit header storing message length in bits
    length_bits = [(len(bits) >> (31 - i)) & 1 for i in range(32)]
    payload = length_bits + bits

    flat = img.flatten()
    if len(payload) > len(flat):
        raise ValueError(
            f"Message too long: need {len(payload)} pixels, image has {len(flat)}"
        )

    # Eq. (4): replace k LSBs of each chosen pixel with message bits
    mask = 0xFF ^ ((1 << k) - 1)           # zero out k LSBs
    for idx, bit in enumerate(payload):
        flat[idx] = (flat[idx] & mask) | bit  # k=1 case: replace 1 LSB

    stego = flat.reshape(img.shape)
    Image.fromarray(stego, "RGB").save(output_path, format="PNG")
    print(f"[embed] saved stego image → {output_path}")


def extract(stego_path: str, k: int = 1) -> str:
    """
    Extract message from stego image using k-LSB extraction.
    Implements Eq. (5) from Chan & Cheng 2004:
        m'_i = x'_li mod 2^k
    No original image required (blind detection).
    """
    img = np.array(Image.open(stego_path).convert("RGB"), dtype=np.uint8)
    flat = img.flatten()

    # Read 32-bit length header first
    header = [int(flat[i]) & 1 for i in range(32)]
    msg_len = 0
    for b in header:
        msg_len = (msg_len << 1) | b

    # Read message bits
    bits = [int(flat[32 + i]) & 1 for i in range(msg_len)]
    return bits_to_text(bits)
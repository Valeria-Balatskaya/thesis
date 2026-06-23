from pathlib import Path
import os
from PIL import Image, ImageDraw

base = Path('data/sipi/baboon.png')
if not base.exists():
    raise FileNotFoundError(str(base))

orig = Image.open(base).convert('RGB')
lsb  = Image.open('results/stego_lsb_baboon.png').convert('RGB')
dct_path = Path('results/dct/dct_baboon_a0.015_n300.png')
if dct_path.exists():
    dct = Image.open(dct_path).convert('RGB')
else:
    dct = Image.open('results/dct/dct_baboon.png').convert('RGB')

w, h = orig.size
canvas = Image.new('RGB', (w * 3, h + 60), 'white')
canvas.paste(orig, (0, 0))
canvas.paste(lsb, (w, 0))
canvas.paste(dct, (w * 2, 0))

d = ImageDraw.Draw(canvas)
d.text((w // 2 - 40, h + 15), 'Original', fill='black')
d.text((w + w // 2 - 20, h + 15), 'LSB', fill='black')
d.text((2 * w + w // 2 - 20, h + 15), 'DCT', fill='black')

os.makedirs('output', exist_ok=True)
canvas.save('output/baboon_compare.png')
print('Saved output/baboon_compare.png')
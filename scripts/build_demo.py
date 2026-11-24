# scripts/build_demo.py
# Build the TraceMark demo registry (called by scripts/build_demo.sh).
#
#   TRACEMARK_DATA=~/tracemark_demo python scripts/build_demo.py            (never app/data)
#
# Registers 5 products (3 of our own photos, 2 ABO), each with 2 copies: the published one ("Public listing") and
# a second one that was never published ("Partner channel"). The copy IDs are the ones the real captures in
# data/real_world carry (70xx published, 75xx decoy), so the REAL screenshot / WhatsApp copy / phone photo from
# that dataset are readable by this registry. Then it writes the suspects to <data>/suspects/:
#   1_real_mac_screenshot.png      real Mac screenshot (Chrome 100 %)        tote bag 7011   expect watermark_aligned
#   2_real_whatsapp.jpeg           real WhatsApp Desktop copy                phone case 7013 expect watermark
#   3_real_iphone_photo.jpg        real iPhone photo of the screen           pudding 7016    expect watermark_aligned
#   4_crop_and_rotation.png        simulated: 60 % off-centre crop + 8 deg   OPPO box 7003   expect watermark_aligned
#   5_our_model_on_top_2x.png      thief re-marks with our model at 2x       Samsung 7007    expect watermark_peeled (+ order proof)
#   6_unrelated_photo.png          a photo that was never registered         -               expect none
# SIFT retrieval needs the original, so the registry holds the clean originals (as in production).

import csv
import os
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

DATA = os.environ.get("TRACEMARK_DATA")
if not DATA or Path(DATA).resolve() == Path("app/data").resolve():
    sys.exit("set TRACEMARK_DATA to a demo folder (never app/data)")

from PIL import Image, ImageOps

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass

from app import guard, store
from src import modern_attacks as ma

RW = "data/real_world"
# (published copy id, title, source photo)
PRODUCTS = [
    (7003, "OPPO Reno15 F5 - retail box", None),
    (7007, "Samsung Galaxy A56 5G - phone and box", None),
    (7011, "Tan leather tote bag", None),
    (7013, "Phone case, butterfly dress", None),
    (7016, "Organic vanilla instant pudding", None),
]
manifest = {int(r["copy_id"]): r for r in csv.DictReader(open(f"{RW}/manifest.csv")) if r["kind"] == "protected"}

store.init_db()
seller = store.create_seller("Demo Shop", "demo@tracemark.example")
db = sqlite3.connect(store.DB_PATH if hasattr(store, "DB_PATH") else str(Path(DATA) / "tracemark.db"))


def next_copy_id(cid):
    """The copies table is AUTOINCREMENT: the next inserted copy gets seq + 1 (ids must ascend)."""
    db.execute("UPDATE sqlite_sequence SET seq = ? WHERE name = 'copies'", (cid - 1,))
    db.commit()


products = {}
for cid, title, _ in PRODUCTS:                                  # published copies 70xx, ascending
    next_copy_id(cid)
    p = guard.protect(seller["id"], title, manifest[cid]["source"], None)
    got = p["copies"][0]["id"]
    assert got == cid, f"copy id {got} != {cid}"
    products[cid] = p
    print(f"product {p['id']}: {title}: copy {got}")
for cid, title, _ in PRODUCTS:                                  # never-published copies 75xx, ascending
    next_copy_id(cid + 500)
    c = guard.create_copy(products[cid]["id"], "Partner channel (never published)")
    assert c["id"] == cid + 500
    print(f"  second copy {c['id']}")
db.close()

out = Path(DATA) / "suspects"
out.mkdir(exist_ok=True)


def real(kind, cid, name):
    src = next(Path(f"{RW}/{kind}").glob(f"{cid}_1.*"))
    img = ImageOps.exif_transpose(Image.open(src)).convert("RGB")
    if name.endswith((".jpg", ".jpeg")):
        img.thumbnail((2400, 2400))                              # the iPhone original is 24 MP: keep the upload light
        img.save(out / name, quality=92)
    else:
        img.save(out / name)
    print("suspect", name, "<-", src)


real("screenshot_100", 7011, "1_real_mac_screenshot.png")
real("whatsapp", 7013, "2_real_whatsapp.jpeg")
real("phone_photo_front", 7016, "3_real_iphone_photo.jpg")

pub = lambda cid: str(Path(DATA) / "copies" / f"p{products[cid]['id']}_c{cid}.png")
tmp = str(out / "_tmp.png")
ma.offcentre_crop(pub(7003), tmp, keep=0.6)
ma.rotate(tmp, str(out / "4_crop_and_rotation.png"), degrees=8.0)
os.remove(tmp)
print("suspect 4_crop_and_rotation.png <- copy 7003")

model = guard.pipeline().model                                  # the thief: our public model on our published copy, 2x
model.embed_product(pub(7007), 500_123, str(out / "5_our_model_on_top_2x.png"), strength=2.0 * guard.STRENGTH,
                    mask_floor=guard.MASK_FLOOR)
print("suspect 5_our_model_on_top_2x.png <- copy 7007 + thief id 500123 at 2x")

neg = sorted(Path("data/coco_sample").glob("*.jpg"))[3]
ImageOps.exif_transpose(Image.open(neg)).convert("RGB").save(out / "6_unrelated_photo.png")
print("suspect 6_unrelated_photo.png <-", neg)
print("done:", DATA)

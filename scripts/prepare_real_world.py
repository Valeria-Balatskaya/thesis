# scripts/prepare_real_world.py
# Prepare the images for the real-world capture dataset (docs/real_world_protocol.md).
#
#   python scripts/prepare_real_world.py            # v3 set + controls (+ v4 if the checkpoint exists)
#   python scripts/prepare_real_world.py --add v4   # later: add the v4 set, keep everything else
#
# Writes data/real_world/ (git-ignored):
#   originals/<item>.png        clean originals, longer side 1200 px (registered, never published)
#   to_capture/<item>.png       what you capture: protected copies, and the clean controls
#   to_capture/viewer.html      full-screen viewer, one image at a time (arrow keys)
#   to_capture/shop.html        marketplace-style grid page (like /demo/shop, but unattacked)
#   to_capture/listing_<k>.html product detail page per item
#   to_capture/key.html         thumbnails with copy IDs (to rename unidentified captures)
#   manifest.csv                item, kind, model, copy_id, decoy copy ids, source photo, set
#
# Items: per model 20 photos (10 own + 10 ABO evaluation split), DIFFERENT photos per
# model, so every capture can be identified by its picture alone. 10 controls (5 + 5):
# registered with issued copies that are never published; you capture the CLEAN
# original. Any watermark verdict on a control is a false positive.
# Every product also gets one unpublished second copy (decoy) so "which copy leaked"
# is tested too (as in the app: one product, several channels).

import argparse
import csv
import html
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from PIL import Image

from src.hidden_v2_inference import HiddenV2Model

ROOT = "data/real_world"
SIDE = 1200
MODELS = {"v3": ("checkpoints/hidden_v3_best.pt", 7000), "v4": ("checkpoints/hidden_v4_best.pt", 8000)}
CONTROL_BASE = 9000
FIELDS = ["item", "kind", "model", "copy_id", "decoy_copy_id", "source", "set", "file"]


def photo_lists():
    own = sorted(f for f in os.listdir("data/products_png") if f.lower().endswith(".png"))
    abo = sorted(f for f in os.listdir("data/abo_sample") if f.lower().endswith((".jpg", ".png")))
    # spread over each folder (own photos come in runs of the same product)
    pick = lambda xs, k: [xs[round(j * (len(xs) - 1) / (k - 1))] for j in range(k)]
    own, abo = pick(own, 25), pick(abo, 25)
    # slots: v3 own 0-9 / abo 0-9, v4 own 10-19 / abo 10-19, controls own 20-24 / abo 20-24
    return [os.path.join("data/products_png", f) for f in own], [os.path.join("data/abo_sample", f) for f in abo]


def save_original(src, item):
    p = f"{ROOT}/originals/{item}.png"
    img = Image.open(src).convert("RGB")
    img.thumbnail((SIDE, SIDE), Image.LANCZOS)
    img.save(p, format="PNG")
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--add", choices=list(MODELS), help="add one model's set to an existing manifest")
    args = ap.parse_args()
    for sub in ("originals", "to_capture"):
        os.makedirs(f"{ROOT}/{sub}", exist_ok=True)
    manifest = f"{ROOT}/manifest.csv"
    rows = list(csv.DictReader(open(manifest))) if os.path.exists(manifest) else []
    have = {r["model"] for r in rows if r["kind"] == "protected"}
    own, abo = photo_lists()

    todo = [args.add] if args.add else [m for m in MODELS if os.path.exists(MODELS[m][0])]
    for k, name in enumerate(MODELS):
        if name not in todo or name in have:
            continue
        ckpt, base = MODELS[name]
        if not os.path.exists(ckpt):
            sys.exit(f"{ckpt} missing")
        model = HiddenV2Model(ckpt)
        slot = 10 * k                                   # v3 -> photos 0-9, v4 -> 10-19
        for j, (tag, src) in enumerate([("own", p) for p in own[slot:slot + 10]]
                                       + [("abo", p) for p in abo[slot:slot + 10]]):
            item = f"{name}_{j + 1:02d}"
            orig = save_original(src, item)
            cid = base + j
            model.embed_product(orig, cid, f"{ROOT}/to_capture/{item}.png", strength=1.0)
            rows.append({"item": item, "kind": "protected", "model": name, "copy_id": cid,
                         "decoy_copy_id": base + 500 + j, "source": src, "set": tag,
                         "file": f"to_capture/{item}.png"})
        print(f"{name}: 20 protected copies")

    if not any(r["kind"] == "control" for r in rows):
        for j, (tag, src) in enumerate([("own", p) for p in own[20:25]] + [("abo", p) for p in abo[20:25]]):
            item = f"ctrl_{j + 1:02d}"
            orig = save_original(src, item)
            Image.open(orig).save(f"{ROOT}/to_capture/{item}.png")
            rows.append({"item": item, "kind": "control", "model": "none",
                         "copy_id": CONTROL_BASE + j, "decoy_copy_id": CONTROL_BASE + 500 + j,
                         "source": src, "set": tag, "file": f"to_capture/{item}.png"})
        print("controls: 10 clean originals (registered, never protected copies published)")

    rows.sort(key=lambda r: (r["kind"] != "protected", r["item"]))
    with open(manifest, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    write_pages(rows)
    print(f"{len(rows)} items -> {ROOT}/to_capture/ (open viewer.html), manifest {manifest}")


def write_pages(rows):
    # capture order is shuffled once (fixed seed) so controls are not all at the end
    import random
    order = sorted(rows, key=lambda r: r["item"])
    random.Random(7).shuffle(order)
    items = [os.path.basename(r["file"]) for r in order]
    js_items = ",".join(f'"{i}"' for i in items)
    open(f"{ROOT}/to_capture/viewer.html", "w").write(f"""<!doctype html><html><head><meta charset="utf-8">
<title>Capture viewer</title><style>
html,body{{margin:0;height:100%;background:#fff;font:14px -apple-system,sans-serif;color:#555}}
#bar{{position:fixed;top:0;left:0;right:0;padding:6px 12px;background:#f2f2f2;border-bottom:1px solid #ddd;z-index:2}}
#wrap{{position:absolute;top:34px;bottom:0;left:0;right:0;display:flex;align-items:center;justify-content:center;overflow:auto}}
img.fit{{max-width:96vw;max-height:calc(100vh - 50px)}} img.actual{{width:auto;height:auto}}
</style></head><body>
<div id="bar">← → next / previous · F fit-to-window / actual size · <b id="pos"></b> · <span id="name"></span> · mode <b id="mode"></b></div>
<div id="wrap"><img id="im"></div>
<script>
const items=[{js_items}];
let k=+(location.hash.slice(1)||0), fit=true;
function show(){{k=(k+items.length)%items.length; const im=document.getElementById('im');
 im.src=items[k]; im.className=fit?'fit':'actual'; location.hash=k;
 document.getElementById('pos').textContent=(k+1)+' / '+items.length;
 document.getElementById('name').textContent=items[k].replace('.png','');
 document.getElementById('mode').textContent=fit?'fit':'actual size (100%)';}}
document.onkeydown=e=>{{if(e.key==='ArrowRight')k++; else if(e.key==='ArrowLeft')k--; else if(e.key==='f'||e.key==='F')fit=!fit; else return; show();}};
show();
</script></body></html>""")
    # key.html: every item with its copy ID, to rename captures the sorter could not identify
    tiles = "".join(f'<figure><img src="{os.path.basename(r["file"])}"><figcaption><b>{r["copy_id"]}</b> '
                    f'{r["item"]}</figcaption></figure>' for r in sorted(rows, key=lambda r: r["item"]))
    open(f"{ROOT}/to_capture/key.html", "w").write(
        '<!doctype html><html><head><meta charset="utf-8"><title>Capture key</title><style>'
        'body{font:14px sans-serif;margin:16px}div{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:12px}'
        'img{width:100%;height:120px;object-fit:contain;background:#eee}figure{margin:0}</style></head><body>'
        '<h1>Copy IDs</h1><p>Rename an unidentified capture to <code>&lt;copy ID&gt;_x.ext</code>.</p>'
        f'<div>{tiles}</div></body></html>')
    style = """body{font-family:sans-serif;margin:0;background:#f3f3f3}header{background:#232f3e;color:#fff;padding:16px 24px;font-size:22px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:16px;padding:24px}
.card{background:#fff;padding:12px;border-radius:6px;display:flex;flex-direction:column;gap:6px}
.card img{width:100%;height:170px;object-fit:contain}.card span{color:#b12704;font-size:18px}
button{background:#ffd814;border:0;padding:8px 18px;border-radius:99px}
.page{display:grid;grid-template-columns:480px 1fr;gap:32px;padding:32px;max-width:1100px}
.page img{width:480px;background:#fff}.page h1{font-size:24px;margin:0 0 8px}.price{color:#b12704;font-size:28px}
.thumbs{display:flex;gap:8px;margin-top:10px}.thumbs div{width:60px;height:60px;background:#ddd;border-radius:4px}
nav a{margin-right:12px}"""
    cards = "".join(f'<div class="card"><img src="{html.escape(i)}"><b>Super Deal #{n + 1}</b>'
                    f'<span>${19 + 7 * n}.99</span><button>Buy now</button></div>' for n, i in enumerate(items))
    open(f"{ROOT}/to_capture/shop.html", "w").write(
        f'<!doctype html><html><head><meta charset="utf-8"><title>MegaDeals</title><style>{style}</style></head>'
        f'<body><header>MegaDeals — unbeatable prices</header><div class="grid">{cards}</div></body></html>')
    for n, i in enumerate(items):
        prev, nxt = f"listing_{(n - 1) % len(items) + 1:02d}.html", f"listing_{(n + 1) % len(items) + 1:02d}.html"
        open(f"{ROOT}/to_capture/listing_{n + 1:02d}.html", "w").write(
            f'<!doctype html><html><head><meta charset="utf-8"><title>Listing {n + 1}</title><style>{style}</style></head>'
            f'<body><header>MegaDeals — unbeatable prices</header><nav style="padding:8px 32px">'
            f'<a href="{prev}">← previous</a><a href="{nxt}">next →</a> listing {n + 1} / {len(items)}</nav>'
            f'<div class="page"><div><img src="{html.escape(i)}"><div class="thumbs"><div></div><div></div><div></div></div></div>'
            f'<div><h1>Premium product, limited offer #{n + 1}</h1><div class="price">${19 + 7 * n}.99</div>'
            f'<p>★★★★☆ 1,284 ratings</p><p>Free delivery tomorrow. In stock.</p><button>Add to basket</button>'
            f'<p style="margin-top:24px;color:#555">High quality, durable materials. Ships from and sold by MegaDeals. '
            f'30-day returns.</p></div></div></body></html>')


if __name__ == "__main__":
    main()

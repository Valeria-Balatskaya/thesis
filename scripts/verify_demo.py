# scripts/verify_demo.py: scan every demo suspect through POST /api/scan and compare with the expected verdict.
# Exit code 1 if any suspect does not give what docs/demo_script.md promises. Nothing is recorded (record=false).
import argparse
import json
import sys
import time
import urllib.request
import uuid
from pathlib import Path

EXPECT = {   # file -> (acceptable verdicts, expected product title fragment, expected copy id or None)
    "1_real_mac_screenshot.png": ({"watermark_aligned", "watermark"}, "tote", 7011),
    "2_real_whatsapp.jpeg": ({"watermark"}, "Phone case", 7013),
    "3_real_iphone_photo.jpg": ({"watermark_aligned"}, "pudding", 7016),
    "4_crop_and_rotation.png": ({"watermark_aligned"}, "OPPO", 7003),
    "5_our_model_on_top_2x.png": ({"watermark_peeled"}, "Samsung", 7007),
    "6_unrelated_photo.png": ({"none"}, None, None),
}


def post_file(url, path):
    boundary = uuid.uuid4().hex
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{path.name}\"\r\n"
            f"Content-Type: application/octet-stream\r\n\r\n").encode() + path.read_bytes() + \
           (f"\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"record\"\r\n\r\nfalse\r\n--{boundary}--\r\n").encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    return json.loads(urllib.request.urlopen(req, timeout=600).read())["result"]


ap = argparse.ArgumentParser()
ap.add_argument("--base", default="http://127.0.0.1:8765")
ap.add_argument("--suspects", required=True)
a = ap.parse_args()
bad = 0
print(f"{'suspect':<28}{'verdict':<20}{'product':<38}{'copy':<6}{'s':<7}extra")
for name, (ok_verdicts, title, copy) in EXPECT.items():
    t = time.time()
    r = post_file(f"{a.base}/api/scan", Path(a.suspects) / name)
    got_title = (r["product"] or {}).get("title", "")
    got_copy = (r["copy"] or {}).get("id")
    extra = ""
    order = r.get("order_proof") or []
    if order:
        o = order[0]
        extra = f"order proof: our_copy_first={o['our_copy_first']} z={o['z']} t={o['t']} (thief: {o['system']})"
    ok = r["verdict"] in ok_verdicts and (title is None or title.lower() in got_title.lower()) and got_copy == copy
    if name.startswith("5") and not (order and order[0]["our_copy_first"]):
        ok = False
    bad += not ok
    print(f"{name:<28}{r['verdict']:<20}{got_title[:36]:<38}{str(got_copy):<6}{time.time() - t:<7.1f}{extra}  {'OK' if ok else '<-- UNEXPECTED'}")
print("DEMO OK: every suspect gives the expected verdict" if not bad else f"DEMO CHECK FAILED: {bad} unexpected")
sys.exit(1 if bad else 0)

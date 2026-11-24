# TraceMark demo: 10 steps, 2 minutes

Everything runs from a **separate** data folder (`~/tracemark_demo`), on **port 8765**. The app on port 8000 and `app/data/` are never touched.

## Before you start (once, ~1 min)

```bash
bash scripts/build_demo.sh            # first time only: builds the registry and checks all 6 suspects (verified 2026-11-24: all OK)
bash scripts/build_demo.sh --reset    # rebuild from scratch (deletes ~/tracemark_demo, only if it carries the demo marker)
bash scripts/run_demo.sh              # serves http://127.0.0.1:8765
```

- Browser: `http://127.0.0.1:8765`, full screen, zoom so the result panel fits.
- Finder: open `~/tracemark_demo/suspects/` next to the browser (you drag files onto the grey drop box).
- Between rehearsals, empty the sightings log: `curl -X DELETE http://127.0.0.1:8765/api/detections`.
- The demo registry holds 1 seller, 5 products, 10 copies and 0 sightings. Each product has a published copy ("Public listing", ID 70xx) and a second copy that was never published ("Partner channel", ID 75xx).

## The 10 steps

A "scan step" is two actions: drag the file onto the box, press **Scan**. Leave "Where you found it" empty and the "Save to the sightings log" box ticked.

| # | Time | Do | Expected on screen | Say |
|---|---|---|---|---|
| 1 | 0:00 | Open **Overview** | 5 protected products, 10 traceable copies, the four-step flow | "Every product image I publish carries an invisible ID. If a copy turns up somewhere else, TraceMark reads the ID back and tells me whose image it is and which copy leaked. Four steps: blind read, retrieve, align, verify." |
| 2 | 0:15 | **Catalogue** → click **Tan leather tote bag** | Two copies: #7011 "Public listing", #7511 "Partner channel (never published)" | "Every sales channel gets its own copy with its own ID. If one leaks, the ID says which channel. The change is hard to see: about 1 colour-difference unit on average (CIEDE2000, 1 = just noticeable), less on flat backgrounds." *(Optional: drag the compare slider.)* |
| 3 | 0:30 | **Scan a suspect** → `1_real_mac_screenshot.png` | **Traced: watermark read after alignment**, tote bag, copy #7011. Stage strip: 1 fails, 2 finds the picture, 3 aligns, 4 verifies | "This is a real Mac screenshot of a listing page. The blind decoder fails on a screenshot, so the service finds my original by its visual features, warps the screenshot back into the original's frame and reads the ID again. Copy #7011, 'Public listing'." |
| 4 | 0:45 | `2_real_whatsapp.jpeg` | **Traced: watermark read directly**, phone case, #7013, about 0.1 s | "A picture forwarded through WhatsApp: recompressed, but the ID is read straight away, no original needed." |
| 5 | 0:55 | `3_real_iphone_photo.jpg` | **Traced after alignment**, vanilla pudding, #7016 | "A photo of a screen taken with an iPhone, with perspective and moiré. Still traced after alignment." |
| 6 | 1:05 | `4_crop_and_rotation.png` | **Traced after alignment**, OPPO box, #7003 | "Cropped to a third of the area and rotated: simulated here, not real. The picture match undoes it." |
| 7 | 1:15 | `5_our_model_on_top_2x.png` (about 7 s) | **Traced: watermark read after removing a mark on top**, Samsung, #7007 | "The thief now adds their own watermark with the same public model at double strength, which blinds the normal decoder. The service finds that mark, subtracts it and reads mine underneath." |
| 8 | 1:30 | Click **Open evidence report**, scroll to *How it was established* and *Registration record* | "Order proof: ... t = 0.95 ... z = 7.4 ... our copy came first." a registration record (#2 in a fresh demo), its signature and the hash chain **verified** | "It also answers who was first: the thief's mark sits on my copy, not on my clean original (z = 7.4, threshold 4). And my registration is in a signed, hash-chained log. This is evidence for a dispute, not a court verdict." |
| 9 | 1:45 | **Scan a suspect** → `6_unrelated_photo.png` | **No protected image found** | "An unrelated photo: no accusation. In the real-world test, 0 of 50 control scans were flagged." |
| 10 | 1:55 | **Overview** | 5 sightings listed as unauthorised (no address given), 5 traced | "Every sighting is logged with a report. 97 of 99 real captures were traced in the real-world test; the two misses were small, low-detail pictures." |

Measured scan times on this Mac (CPU): 1.6 s, 0.1 s, 1.3 s, 1.2 s, **6.9 s**, 1.1 s. The whole run is about 1:50 plus talking pauses.

## Expected verdicts (`scripts/verify_demo.py` checks these through the same `/api/scan`)

| Suspect | Product / copy | Verdict | Where it comes from |
|---|---|---|---|
| `1_real_mac_screenshot.png` | tote bag, #7011 | `watermark_aligned` | real capture, `data/real_world/screenshot_100` |
| `2_real_whatsapp.jpeg` | phone case, #7013 | `watermark` | real capture, WhatsApp Desktop |
| `3_real_iphone_photo.jpg` | pudding, #7016 | `watermark_aligned` | real capture, iPhone photo of the screen (downscaled to 2,400 px) |
| `4_crop_and_rotation.png` | OPPO box, #7003 | `watermark_aligned` | simulated: 60 % off-centre crop + 8° |
| `5_our_model_on_top_2x.png` | Samsung, #7007 | `watermark_peeled` + order proof z = 7.43 | simulated: our model, thief ID 500123, 2x strength |
| `6_unrelated_photo.png` | none | `none` | COCO photo, never registered |

## If something goes wrong

- **A scan shows a different verdict:** run `bash scripts/build_demo.sh --reset`; it re-checks all six and says which one deviates.
- **Step 7 takes long:** it is the only scan that also builds the order proof (25 decoy copies); say so, then continue.
- **Browser upload of the iPhone file fails:** the suspect is already a downscaled JPEG; if it still fails, skip step 5.

## Claims to keep honest

- Real captures: one device, one session, 20 products; blind read works for messengers, screenshots and photos need the alignment step.
- Steps 3-5 are **real captures**; steps 6 and 7 are **simulated** attacks. Say which is which.
- Step 7 shows the owner-side evidence for one thief strength (2x). The adaptive-attack study is still running; do not claim robustness against a thief who changes the image first.
- Do not describe the order proof or the registry as legal proof.

## Likely questions

- *What if the thief removes the watermark?* "We tested JPEG, screenshots, crops and AI regeneration (SD VAE). Diffusion img2img is a real limit: at strength 0.3 only 7 of 30 photos are still traced, at 0.5 none. That is in the thesis as a limit."
- *What if the thief puts a different watermark on top?* "Industry watermarks (TrustMark, WAM, Video Seal) never blocked ours in the tests; our own model at 2x is the case you just saw."
- *Is it visible?* "About 1 on a colour-difference scale where 1 is just noticeable, lower on flat backgrounds."

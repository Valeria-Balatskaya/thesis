# Real-world capture protocol

Goal: test the system on **real** screenshots, phone photos and messenger round trips instead of
simulations (objective 1.2.3 robustness, 1.4 evaluation). Every result so far comes from simulated
attacks (`src/ecommerce_attacks.py`, `src/modern_attacks.py`). This dataset checks whether those
simulations were realistic.

## What is already prepared

`python scripts/prepare_real_world.py` has created `data/real_world/` (git-ignored):

| Folder / file | Content |
|---|---|
| `to_capture/` | the images you capture: **20 v3-protected copies** (`v3_01…20`), **20 v4 copies** (`v4_01…20`, added once v4 is trained), **10 controls** (`ctrl_01…10`) |
| `to_capture/viewer.html` | full-screen viewer, one image at a time, in a fixed shuffled order |
| `to_capture/listing_01.html …` | one marketplace product page per image (photo at 480 px, price, text) |
| `to_capture/shop.html` | marketplace grid page in the style of `/demo/shop` (photos about 170 px high) |
| `originals/` | clean originals. The test registers them. **Don't capture these.** |
| `manifest.csv` | which file is which copy ID, model, own/ABO photo |

- **Photos:** each model uses different photos: 10 own + 10 ABO (evaluation split, never used in
  training). So every capture can be identified by its picture alone.
- **Controls:** clean, never-watermarked originals of registered products. The decoder must find
  nothing in them. Any ID read from a control is a false positive.
- **Why not the live `/demo/shop`:** it shows *attacked* copies, and it needs products in the app's
  own database. `shop.html` uses the same layout with the unattacked images.

> **v4:** if `checkpoints/hidden_v4_best.pt` exists when you start, run
> `python scripts/prepare_real_world.py --add v4` first. Then you capture all 50 images in one go.
> Otherwise capture the 30 v3 + control images now and the 20 v4 images later. The sorting script
> doesn't care about order.

## Before you start (5 min)

- [ ] Make one inbox folder per capture type, e.g. `~/captures/screenshot_100/`. Captures can have any file name.
- [ ] Make macOS screenshots land in the inbox: **Cmd+Shift+5 → Options → Save to → Other Location…**
      Pick the current inbox. Change it each time you switch capture type.
- [ ] Turn off **Cmd+Shift+5 → Options → Show Floating Thumbnail**. Otherwise a quick next screenshot is delayed.
- [ ] Browser: Chrome or Safari, zoom 100% (Cmd+0), no extensions that change colours. Night Shift / True Tone **off** for the phone-photo sessions. Note in `--notes` if they were on.
- [ ] Don't edit, crop or convert the captures afterwards. Real thieves do, but the attack suite covers editing separately.

## Capture types (tick when done)

Numbers are for the full set of 50 (20 v3 + 20 v4 + 10 controls). With v3 only, it is 30.

| ✓ | Type (folder name) | How | Count | Time |
|---|---|---|---|---|
| [ ] | `screenshot_100` | open `viewer.html`, press **F** for *actual size*. **Cmd+Shift+4**, drag a rectangle roughly around the photo (a bit of white border is fine), **→** for the next | 50 | 15 min |
| [ ] | `screenshot_zoomed` | same, but browser zoom **50%** (Cmd+− until 50%), actual size | 50 | 15 min |
| [ ] | `page_screenshot` | open `listing_01.html`, **Cmd+Shift+4, then Space**, click the browser window (whole window incl. page text). Click **next →** | 50 | 15 min |
| [ ] | `phone_photo_front` | `viewer.html` in *fit* mode, browser full screen (Ctrl+Cmd+F). iPhone camera **straight on**, about 40–60 cm, photo fills about ⅔ of the frame. Default camera settings (HEIC is fine) | 50 | 20 min |
| [ ] | `phone_photo_angle` | same, but from about **30° to the side** and slightly above | 50 (min. 20) | 20 min |
| [ ] | `whatsapp` | WhatsApp Desktop → chat **"Message yourself"** → attach all images **as Photos (not Document)**, standard quality (not HD). After they arrive: select them → right-click → Save As… into the inbox. Note in `--notes` whether HD was off | 50 | 10 min |
| [ ] | `telegram` | Telegram Desktop → **Saved Messages** → attach the images, keep **"Compress images" ticked** (= sent as photo). Then right-click each one → Save As… | 50 | 15 min |
| [ ] | `shop_grid_crop` *(optional)* | open `shop.html`, **Cmd+Shift+4** around **one card's photo** (about 170 px, like a search grid) | 20 | 10 min |
| [ ] | `instagram` *(optional)* | **private** account: post the images as carousels (max. 20 per post). Get the server's copies with *Settings → Your activity → Download your information* (the posts' JPEGs), or take phone screenshots of each post (then use folder `instagram_screenshot`) | 10–20 | 30 min |
| [ ] | `print_photo` *(optional)* | print 10 images (10×15 cm photo paper or A4 office printer, note which). Photograph each with the iPhone, flat on a table, daylight | 10 | 30 min |

Instagram: ABO images are CC BY 4.0 (Amazon Berkeley Objects). Use a private account, or credit
"Amazon Berkeley Objects (CC BY 4.0)" and delete the posts after capturing.

### Filing the captures (after each type, 1 min)

```bash
python scripts/sort_captures.py screenshot_100 ~/captures/screenshot_100 --device "MacBook Pro M1 (Retina)" --app "Chrome 131"
python scripts/sort_captures.py phone_photo_front ~/captures/phone_photo_front --device "iPhone 15" --notes "True Tone off"
python scripts/sort_captures.py whatsapp ~/captures/whatsapp --app "WhatsApp Desktop 2.24, standard quality"
```

- The script identifies each capture by its picture (feature matching against the 50 originals).
  The watermark is not used, so it can't bias the results.
- It copies each capture, unchanged, to `data/real_world/<type>/<copy_id>_<n>.<ext>` and logs it in
  `data/real_world/captures.csv`: type, copy ID, size, device, app, notes, original name and time.
- Captures it can't identify safely go to `data/real_world/<type>/_unidentified/`. Look at each one,
  rename it to `<copy_id>_x.<ext>` (copy IDs are in `manifest.csv`, or open `to_capture/key.html` to see every image with its copy ID), move it up one folder, and run
  the same command with `--manual`. Most likely cases: a very small or very blurred capture.
- **Extra negatives (recommended):** put about 10 of your own unrelated photos into the inbox,
  renamed `x_<anything>.jpg`. They must never come back with an ID.

## Evaluate

```bash
python tests/test_real_world.py              # ~2 min for 500 captures
python tests/test_real_world.py --informed   # also the ownership check on "same picture" results (slower)
```

- Output per capture type and model: blind read %, full-pipeline %, wrong copies, verdicts, and
  false positives on controls / `x_` photos, all with 95% Wilson intervals.
- Files: `output/real_world/summary.csv`, `rows.csv`, and the chart `real_world.png`.

## What to compare it with (for the thesis)

| Real capture | Closest simulation (`tests/test_v3_vs_v2.py`) |
|---|---|
| `screenshot_100`, `screenshot_zoomed` | `native_screenshot`, `screenshot` |
| `page_screenshot` | `page_screenshot`; real Chrome captures in `tests/test_in_the_wild.py` |
| `phone_photo_*` | `screen_photo`, `perspective` |
| `whatsapp`, `telegram` | `social_recompress`, `jpeg_q70` |
| `instagram` | `instagram_filter` + `social_recompress` |
| `print_photo` | `print_photograph` |
| `shop_grid_crop` | `in_the_wild` "search grid 200 px" (expected to fail: below the ~200 px limit) |

If a real capture type is clearly worse than its simulation, the simulation was too gentle. That is a
thesis finding in itself, not a failure of the protocol.

## Checklist summary

- [ ] (when v4 exists) `python scripts/prepare_real_world.py --add v4`
- [ ] screenshot_100 · [ ] screenshot_zoomed · [ ] page_screenshot
- [ ] phone_photo_front · [ ] phone_photo_angle
- [ ] whatsapp · [ ] telegram
- [ ] optional: shop_grid_crop · instagram · print_photo
- [ ] 10 unrelated `x_` photos in at least two of the types
- [ ] every type filed with `sort_captures.py`, `_unidentified/` empty
- [ ] `python tests/test_real_world.py`, then send me the console output

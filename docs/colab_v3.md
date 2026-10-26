# Training v3 / v4 in Google Colab — step by step

Everything needed is already in the repository; nothing has to be uploaded by hand.

## Before you start (5 minutes)

1. The notebook uses the `main` branch (cell 3, `BRANCH = 'main'`).
2. Have about **300 MB free in Google Drive** (training images ~200 MB, checkpoints ~5 MB each).
3. On the Mac, keep the computer awake while it trains: in Terminal run `caffeinate -d`
   (stop it later with Ctrl+C). Plug in the charger and don't close the lid.

## Run it

1. Open <https://colab.research.google.com> → **File → Open notebook → GitHub** → paste
   `Valeria-Balatskaya/thesis`, choose branch `main`,
   open `notebooks/train_hidden_v2.ipynb`.
2. **Runtime → Change runtime type → T4 GPU → Save.**
3. **Runtime → Run all.** When asked, allow access to Google Drive.
4. What happens, cell by cell:

| Cell | What it does | Time |
|---|---|---|
| 2 | checks the GPU | seconds |
| 3 | mounts Drive, sets `RUN = 'v3'` and the v3 options | seconds |
| 4 | downloads the code from GitHub | ~10 s |
| 5 | downloads COCO (5,000 photos) and the 2,500 ABO product photos **into your Drive** | ~10 min the first time, ~0 later |
| 7 | trains 40,000 steps, prints a validation table every 2,000 steps | **~5.5–6 h on a T4** |
| 9 | downloads `hidden_v3_best.pt` to your computer | seconds |

## While it trains

- The table printed every 2,000 steps shows, per distortion, `decodable` = share of images
  whose product ID would be recovered. Until step 15,000 the attacks get stronger
  ("severity" ramps to 1.0), so numbers dip — that is expected. Judge the model on the tables
  **after step 15,000**, especially the line `[abo_train] ... decodable`.
- A free Colab session can disconnect (idle timeout, or the ~12 h limit). Then simply
  **Runtime → Run all** again: data is already in Drive and training resumes from
  `MyDrive/thesis_checkpoints_v3/hidden_v2_latest.pt` (at most 2,000 steps are lost).
- If a run looks hopeless (all `decodable` near 0% after step 20,000), stop it and send me
  the last two validation tables.

## After training

1. Put the downloaded `hidden_v3_best.pt` into `checkpoints/` in the project on the Mac.
   (If cell 9 fails, take it from Drive: `MyDrive/thesis_checkpoints_v3/hidden_v2_best.pt`
   and rename it to `hidden_v3_best.pt`.)
2. The app picks it up automatically on the next start.
3. Evaluate (each takes minutes, not hours):

```bash
python tests/test_hidden_v2_eval.py checkpoints/hidden_v3_best.pt data/products_png
python tests/test_ecommerce_vs_regular.py        # edit CHECKPOINT at the top to v3
python tests/test_texture_masks.py               # idem
python tests/test_defence_benchmark.py data/products_png 35 1.0
```

## What the v3 options mean

| Option | Why |
|---|---|
| `--chroma-weight 8` | colour changes cost 8× in the quality budget → no green/purple tint |
| `--mask-type nvf --mask-floor 0.1` | supervisor's request: embed mainly where there is texture (noise visibility function, Voloshynovskiy et al. 1999) |
| `--zero-white` | nothing at all on pure-white background |
| `--mask-renorm` | the energy taken off flat areas goes into textured areas (keeps robustness) |
| `--distortions v3` | adds brightness/levels changes (the weakest attack on ABO photos) |
| `--overwrite-prob 0.15` | 15% of steps: someone re-protects the image; the decoder must keep OUR message |
| `--data-weights 0.5 0.5` | each batch is half COCO, half Amazon product photos |

Not used: `--headroom 2`. With `--zero-white` nothing is embedded on pure white, so the
255 → 253 shift would only change the background without helping (tests/test_texture_masks.py).

## v4 — the secret-key model (after v3)

v4 makes the watermark depend on a secret key: it is written and read with the key, so someone
using the same model file without the key writes into a different channel and cannot erase
ours. It starts from your v3 model (already in Drive) and only has to learn the key part.

1. Check that `MyDrive/thesis_checkpoints_v3/hidden_v2_best.pt` exists in Google Drive
   (it is the v3 run's best checkpoint; the notebook stops with a message if it is missing).
2. Open the notebook as above. Cell 3 already says `RUN = 'v4'`.
3. **Runtime → Run all.** 20,000 steps, about **3.5 h on a T4**; resumes after a disconnect.
4. In every validation table, watch two extra lines:
   - `wrong_key ... decodable` must fall to **0%**: a wrong key gives no ID at all.
   - `other_key_overwrite ... decodable` should **rise**: after someone overwrites with another
     key, our key still reads our ID.
5. The last cell downloads `hidden_v4_best.pt`. Put it in `checkpoints/`; the app then uses it
   automatically (it prefers v4 over v3). Evaluate with:

```bash
python tests/test_v4_keys.py checkpoints/hidden_v4_best.pt
python tests/test_v3_vs_v2.py 30 v4=checkpoints/hidden_v4_best.pt
```

The key is derived from the server secret (`WATERMARK_MODEL_KEY`, else `WATERMARK_KEY`). Change
it once before real use and keep it secret: anyone with both the model file and the key can
write in our channel.

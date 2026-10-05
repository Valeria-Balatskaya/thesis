# Watermarking for E-commerce Image Traceability

Bachelor's thesis project: steganography and digital watermarking to trace stolen
product images back to the seller/product they came from.

## Setup (macOS / Linux)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Setup (Windows)

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Running

```bash
# Web app (http://127.0.0.1:8000)
uvicorn app.main:app --reload

# Benchmarks / tests
python -m pytest tests/
```

The benchmarks are standalone scripts: `python tests/test_deployed_benchmark.py`.

## Model checkpoints

Checkpoints are git-ignored and must be copied into `checkpoints/` manually;
without them the benchmarks and the web app stop with `FileNotFoundError`.

- `checkpoints/hidden_final.pt` — HiDDeN v1 (used by the current app and benchmarks)
- `checkpoints/hidden_v2_best.pt` — HiDDeN v2. Train it with
  `notebooks/train_hidden_v2.ipynb` in Google Colab, then evaluate it with
  `python tests/test_hidden_v2_eval.py`

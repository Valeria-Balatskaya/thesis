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

The trained HiDDeN checkpoint (`checkpoints/hidden_final.pt`) is git-ignored and
must be copied over manually (or retrained via `src/training/train.py`).

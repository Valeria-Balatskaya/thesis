#!/usr/bin/env bash
# Fetch Watermark Anything (Meta, MIT-licensed weights trained on SA-1B) into
# third_party/ for tests/test_baselines.py. Pinned to the commit the thesis used.
set -euo pipefail
cd "$(dirname "$0")/.."
COMMIT=2c08af04d037d5667c02f6ddebbda9ff04581c3e
mkdir -p third_party
if [ ! -d third_party/watermark-anything ]; then
  git clone -q https://github.com/facebookresearch/watermark-anything.git third_party/watermark-anything
fi
git -C third_party/watermark-anything checkout -q "$COMMIT"
W=third_party/watermark-anything/checkpoints/wam_mit.pth
[ -s "$W" ] || curl -L -o "$W" https://dl.fbaipublicfiles.com/watermark_anything/wam_mit.pth   # 378 MB
pip install -r requirements-baselines.txt
# Meta Video Seal 1.0 (image mode). Installed without its pins (timm==0.9.16 would break
# WAM); its config file and weights go into third_party/videoseal (weights on first use)
pip install --no-deps videoseal==1.0.1
mkdir -p third_party/videoseal/configs
VS=third_party/videoseal/configs/attenuation.yaml
[ -s "$VS" ] || curl -sfL -o "$VS" https://raw.githubusercontent.com/facebookresearch/videoseal/main/configs/attenuation.yaml
echo "baselines ready"

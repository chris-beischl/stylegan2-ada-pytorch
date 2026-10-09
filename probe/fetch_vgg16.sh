#!/bin/bash
# PROBE: fetch NVIDIA's VGG16 LPIPS feature network for projector_batch.py into .cache/vgg16.pt
# (once per checkout; compute nodes then need no network). Prints the path.
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
url=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt
sha=b437eb095feaeb0b83eb3fa11200ebca4548ee39a07fb944a417ddc516cc07c3
dst=$repo/.cache/vgg16.pt
if [ ! -f "$dst" ]; then
    mkdir -p "$repo/.cache"
    curl -fsSL --retry 3 -o "$dst.part" "$url"
    mv "$dst.part" "$dst"
fi
if ! echo "$sha  $dst" | sha256sum -c --quiet; then
    echo "error: checksum mismatch: $dst" >&2
    exit 1
fi
echo "$dst"

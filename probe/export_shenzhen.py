"""Spike S5: export Shenzhen through PROBE's plugin into a dataset_tool.py-style zip.

Run with PROBE's environment (read-only on PROBE and the raw data):
    ../PROBE/.venv/bin/python probe/export_shenzhen.py ROOT OUT.zip [SIZE]
Pixels equal PROBE's samples: PROBE yields uint8 / 255, so round(x * 255) restores them exactly.
"""
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from probe.data.shenzhen import ShenzhenDataset

root, out = Path(sys.argv[1]), Path(sys.argv[2])
size = int(sys.argv[3]) if len(sys.argv) > 3 else 256
ds = ShenzhenDataset(ShenzhenDataset.Config(name="shenzhen", image_size=size)).load(root)
out.parent.mkdir(parents=True, exist_ok=True)
labels = []
with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_STORED) as zf:
    for i in range(len(ds)):
        item = ds[i]
        arr = np.rint(item.sample.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        assert np.array_equal(arr.astype(np.float32) / 255.0, item.sample.permute(1, 2, 0).numpy())
        name = f"{item.unit_ids['image_id']}.png"
        buf = io.BytesIO()
        Image.fromarray(arr, "RGB").save(buf, format="png", compress_level=0, optimize=False)
        zf.writestr(name, buf.getvalue())
        labels.append([name, int(item.label)])
    zf.writestr("dataset.json", json.dumps({"labels": labels}))
print(f"{len(labels)} images, class counts {np.bincount([l for _, l in labels]).tolist()} -> {out}")

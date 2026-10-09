"""Second half of probe/train_gan.sh: export G_ema of the last snapshot and write backend.json."""
import json
import pickle
import subprocess
import sys
from pathlib import Path

import torch

import legacy

outdir = Path(sys.argv[1])
(run_dir,) = sorted((outdir / "run").iterdir())  # train.py creates exactly one numbered run directory
snapshot = sorted(run_dir.glob("network-snapshot-*.pkl"))[-1]
with open(snapshot, "rb") as f:
    G = legacy.load_network_pkl(f)["G_ema"].eval().requires_grad_(False).cpu()
with open(outdir / "G_ema.pkl", "wb") as f:
    pickle.dump({"G_ema": G}, f)

repo = Path(__file__).resolve().parent.parent
git = lambda *a: subprocess.run(["git", *a], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()
stats = [json.loads(line) for line in open(run_dir / "stats.jsonl")]
info = {
    "commit": git("rev-parse", "HEAD"),
    "dirty": bool(git("status", "--porcelain")),
    "snapshot": snapshot.name,
    "kimg": int(snapshot.stem.split("-")[-1]),
    "sec_per_kimg": sum(s["Timing/sec_per_kimg"]["mean"] for s in stats[1:]) / max(len(stats) - 1, 1),  # tick 0 includes setup
    "augment_p": stats[-1]["Progress/augment"]["mean"],
    "training_options": json.loads((run_dir / "training_options.json").read_text()),
    "generator": {"img_resolution": G.img_resolution, "img_channels": G.img_channels,
                  "c_dim": G.c_dim, "z_dim": G.z_dim, "w_dim": G.w_dim, "num_ws": G.mapping.num_ws},
    "torch": torch.__version__,
}
(outdir / "backend.json").write_text(json.dumps(info, indent=2))
print(f"G_ema of {snapshot.name} -> {outdir / 'G_ema.pkl'}")

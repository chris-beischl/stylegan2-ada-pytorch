"""Check projector_batch.py: batch vs. single targets, time per image, spread of 'sample' starts.

    python probe/spike_project.py G_ema.pkl TARGET.png... [--steps N]
"""
import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import PIL.Image
import torch

from projector_batch import project_batch

p = argparse.ArgumentParser()
p.add_argument("network")
p.add_argument("targets", nargs="+")
p.add_argument("--steps", type=int, default=1000)
p.add_argument("--vgg16", default=str(Path(__file__).resolve().parent.parent / ".cache" / "vgg16.pt"))
args = p.parse_args()
device = torch.device("cuda")
with open(args.network, "rb") as f:
    G = pickle.load(f)["G_ema"].to(device)
imgs = torch.stack([torch.from_numpy(np.array(PIL.Image.open(t).convert("RGB"))).permute(2, 0, 1) for t in args.targets])
labels = [int(Path(t).stem.rsplit("_", 1)[-1]) for t in args.targets]
c = torch.nn.functional.one_hot(torch.tensor(labels), G.c_dim).float()


def run(targets, cc, **kw):
    torch.cuda.synchronize()
    t0 = time.time()
    w, d = project_batch(G, targets, cc, seed=0, device=device, vgg16_path=args.vgg16, num_steps=args.steps, **kw)
    torch.cuda.synchronize()
    return w, d, time.time() - t0


n = len(labels)
w_b, d_b, t_b = run(imgs, c, batch_size=n)
w_s, d_s, t_s = run(imgs, c, batch_size=1)  # same call, one target per optimisation: same random draws
print(f"steps {args.steps}; batch_size {n}: {t_b / n:.1f} s/image; batch_size 1: {t_s / n:.1f} s/image")
print(f"dist batch_size {n}: {d_b.numpy().round(4)}; batch_size 1: {d_s.numpy().round(4)}")
print(f"batch_size {n} vs 1: max |w diff| {(w_b - w_s).abs().max():.2e}, max |w| {w_b.abs().max():.2f}")
for bs in (16, 32):
    k = bs // n
    _, d, t = run(imgs.repeat(k, 1, 1, 1), c.repeat(k, 1), batch_size=bs, init="sample")
    print(f"batch {bs} (init sample): {t:.1f} s, {t / bs:.2f} s/image, peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB, dist mean {d.mean():.3f}")
w5, d5, _ = run(imgs[:1].repeat(5, 1, 1, 1), c[:1].repeat(5, 1), batch_size=5, init="sample")
pd = torch.cdist(w5, w5)
print(f"5 'sample' projections of one target: pairwise w distance {pd[pd > 0].min():.2f}-{pd.max():.2f}, dist {d5.numpy().round(3)}")

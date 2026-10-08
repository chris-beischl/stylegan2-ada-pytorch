"""Spike S9: gradient from a fixed loss through G.synthesis to a path of broadcast w.

Runs in this repo's env (compiled ops) or in PROBE's venv with only this repo on
PYTHONPATH (--ref forces the reference ops, so no extension build is attempted):
    python probe/spike_grad.py --network PKL --out grad.npz [--ref]
Compare two outputs with: python probe/spike_grad.py --compare A.npz B.npz
"""
import argparse
import sys
import time

import numpy as np
import torch


def run(args):
    from torch_utils.ops import bias_act, upfirdn2d
    if args.ref:
        bias_act._init = lambda: False
        upfirdn2d._init = lambda: False
    import pickle

    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False  # compare like with like across envs
    torch.backends.cudnn.allow_tf32 = False
    with open(args.network, "rb") as f:  # G-only pickle from probe/export_generator.py
        G = pickle.load(f)["G_ema"].eval().requires_grad_(False).to(device)

    gen = torch.Generator().manual_seed(0)
    z = torch.randn([2, G.z_dim], generator=gen).to(device)
    c = torch.zeros([2, G.c_dim], device=device)
    c[:, args.label] = 1
    w_ends = G.mapping(z, c)[:, 0, :]  # [2, w_dim]
    t = torch.linspace(0, 1, args.T, device=device)[1:-1, None]  # interior points, as PLAN optimises
    path = (w_ends[:1] + t * (w_ends[1:] - w_ends[:1])).detach().requires_grad_(True)
    R = torch.randn([path.shape[0], G.img_channels, G.img_resolution, G.img_resolution], generator=gen).to(device)

    def step():
        img = G.synthesis(path.unsqueeze(1).repeat([1, G.mapping.num_ws, 1]), noise_mode="const")
        loss = (img * R).mean()
        (g,) = torch.autograd.grad(loss, path)
        return loss, g

    loss, g = step()
    for _ in range(2):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for _ in range(args.iters):
        step()
    torch.cuda.synchronize()
    ms = (time.time() - t0) / args.iters * 1e3
    ops = "reference" if args.ref else ("compiled" if bias_act._init() and upfirdn2d._init() else "reference (build failed)")
    print(f"torch {torch.__version__}, python {sys.version.split()[0]}, ops {ops}")
    print(f"batch {path.shape[0]} (T={args.T}): {ms:.0f} ms per forward+backward, "
          f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB; loss {loss.item():.6e}, |grad| {g.norm().item():.6e}")
    np.savez(args.out, grad=g.cpu().numpy(), loss=loss.item())


def compare(a, b):
    A, B = np.load(a), np.load(b)
    d = np.abs(A["grad"] - B["grad"])
    print(f"loss {A['loss']:.8e} vs {B['loss']:.8e}; grad max abs diff {d.max():.3e}, "
          f"rel {d.max() / np.abs(A['grad']).max():.3e}, cosine "
          f"{(A['grad'] * B['grad']).sum() / np.linalg.norm(A['grad']) / np.linalg.norm(B['grad']):.8f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--network")
    p.add_argument("--out")
    p.add_argument("--ref", action="store_true")
    p.add_argument("--T", type=int, default=50)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--label", type=int, default=1)
    p.add_argument("--compare", nargs=2)
    args = p.parse_args()
    compare(*args.compare) if args.compare else run(args)

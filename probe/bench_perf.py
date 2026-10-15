"""Performance study (probe/perf-handover.md): projection and PLAN path steps as PROBE runs them.

Run with PROBE's venv, this checkout on the path (as S9; nothing is written in PROBE):
    ../PROBE/.venv/bin/python probe/bench_perf.py proj --gan DIR --n 32 --batch 32 --steps 100 --out X.npz
    ../PROBE/.venv/bin/python probe/bench_perf.py path --gan DIR --points 48
    ../PROBE/.venv/bin/python probe/bench_perf.py compare A.npz B.npz
DIR is a GAN store entry (G_ema.pkl + meta.json). Determinism settings are PROBE's. The switches
--ops compiled and --amp vgg use the fork's options (STYLEGAN2ADA_PREBUILT_OPS, vgg_bf16); --tf32,
--cl and --compile are applied here. --ops compiled and --amp vgg need the patches of commit 96f9e9f
(reverted after the study); on later commits they fall back to the reference ops (ops_loaded false)
or fail. One JSON line per run goes to stdout and --log.
"""
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # before CUDA starts, as PROBE

import argparse
import json
import pickle
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
PROBE = REPO.parent / "PROBE"
ROOTS = {"shenzhen": "/vol/miltank/users/beischlc/datasets/shenzhen/ChinaSet_AllFiles",
         "aptos2019": "/vol/miltank/users/beischlc/datasets/aptos2019"}


def setup(args):
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = args.tf32 in ("matmul", "all")
    torch.backends.cudnn.allow_tf32 = args.tf32 != "off"  # torch's default (on) is PROBE's
    sys.path.append(str(REPO))
    if args.ops == "ref":
        os.environ["STYLEGAN2ADA_REFERENCE_OPS"] = "1"
    else:  # the .so built in this checkout; no nvcc/ninja in PROBE's venv
        os.environ["STYLEGAN2ADA_PREBUILT_OPS"] = "1"
    with open(Path(args.gan) / "G_ema.pkl", "rb") as f:
        G = pickle.load(f)["G_ema"].eval().requires_grad_(False).cuda()
    if args.compile:
        G.synthesis = torch.compile(G.synthesis, mode=args.compile)
    return G


def autocast(on):
    return torch.autocast("cuda", dtype=torch.bfloat16, enabled=on)


def members(args, n):
    """n members of the GAN, evenly spaced in meta.json order, through PROBE's plugin: uint8 targets, labels."""
    from probe.plugins.datasets.aptos2019 import Aptos2019Dataset
    from probe.plugins.datasets.shenzhen import ShenzhenDataset
    meta = json.loads((Path(args.gan) / "meta.json").read_text())
    cls = {"shenzhen": ShenzhenDataset, "aptos2019": Aptos2019Dataset}[meta["dataset"]["name"]]
    ds = cls(cls.Config(name=cls.name, image_size=256)).load(Path(ROOTS[cls.name]))
    index = {image_id: i for i, (_, _, image_id) in enumerate(ds._items)}
    picked = meta["members"][::max(1, len(meta["members"]) // n)][:n]  # spread over the classes
    items = [ds[index[m]] for m in picked]
    x = torch.stack([(it.sample * 255).round().to(torch.uint8) for it in items])
    return x, [int(it.label) for it in items], meta["dataset"]["name"]


def peak_gb():
    return torch.cuda.max_memory_allocated() / 2**30


def proj(args, G):
    import projector_batch as pb
    targets, labels, dataset = members(args, args.n)
    c = torch.nn.functional.one_hot(torch.tensor(labels), G.c_dim).float()
    if args.cl:
        features = pb._features
        pb._features = lambda vgg, img, bf16=False: features(vgg, img.contiguous(memory_format=torch.channels_last), bf16)
        jit_load = torch.jit.load
        torch.jit.load = lambda *a, **k: jit_load(*a, **k).to(memory_format=torch.channels_last)
    if args.compile:  # project_batch deep-copies G, which would drop the compiled module
        pb.copy = types.SimpleNamespace(deepcopy=lambda g: g)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.time()
    with autocast(args.amp == "all"):
        w, dist = pb.project_batch(G, targets, c, seed=0, device=torch.device("cuda"),
                                   vgg16_path=str(REPO / ".cache" / "vgg16.pt"), num_steps=args.steps,
                                   init="sample", batch_size=args.batch, vgg_bf16=args.amp == "vgg")
    torch.cuda.synchronize()
    s = time.time() - t0
    if args.out:
        np.savez(args.out, w=w.numpy(), dist=dist.numpy())
    return dict(dataset=dataset, n=args.n, batch=args.batch, steps=args.steps, s_total=round(s, 1),
                s_per_target=round(s / args.n, 3), s_per_target_1000=round(s / args.n / args.steps * 1000, 2),
                peak_gb=round(peak_gb(), 2), dist_mean=round(float(dist.mean()), 4),
                dist_median=round(float(dist.median()), 4))


def path(args, G):
    """One PLAN chunk step: G.synthesis + both ResNet-18 + gradient w.r.t. the points (PROBE code)."""
    from torchvision.models import resnet18
    from probe.methods.defenses.plan import _image_losses
    torch.manual_seed(0)  # same classifiers in every run, so gradients compare across runs
    phi_id =resnet18(num_classes=args.n_ids).eval().requires_grad_(False).cuda()
    phi_class = resnet18(num_classes=G.c_dim).eval().requires_grad_(False).cuda()
    if args.cl:
        phi_id, phi_class = (m.to(memory_format=torch.channels_last) for m in (phi_id, phi_class))
    gen = torch.Generator().manual_seed(0)
    z = torch.randn([2, G.z_dim], generator=gen).cuda()
    c = torch.zeros([2, G.c_dim], device="cuda")
    c[:, 1] = 1
    ends = G.mapping(z, c)[:, 0]
    t = torch.linspace(0, 1, args.points + 2, device="cuda")[1:-1, None]
    w = (ends[:1] + t * (ends[1:] - ends[:1])).detach().requires_grad_(True)
    gen_ = types.SimpleNamespace(synthesis=G.synthesis, num_ws=G.num_ws)

    def step():
        with autocast(args.amp == "all"):
            l_id, l_class = _image_losses(gen_, phi_id, phi_class, w, 1)
        (g,) = torch.autograd.grad(l_id + l_class, w)
        return g

    g = step()
    for _ in range(2):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for _ in range(args.iters):
        step()
    torch.cuda.synchronize()
    ms = (time.time() - t0) / args.iters * 1e3
    if args.out:
        np.savez(args.out, grad=g.cpu().numpy())
    return dict(points=args.points, n_ids=args.n_ids, ms_per_step=round(ms, 1),
                ms_per_point=round(ms / args.points, 2), peak_gb=round(peak_gb(), 2))


def compare(a, b):
    A, B = np.load(a), np.load(b)
    k = "w" if "w" in A else "grad"
    d = np.abs(A[k] - B[k])
    out = dict(key=k, bitwise_equal=bool(np.array_equal(A[k], B[k])), max_abs=float(d.max()),
               max_rel=float(d.max() / np.abs(A[k]).max()))
    if k == "w":
        out.update(w_dist_mean=float(np.linalg.norm(A[k] - B[k], axis=1).mean()),
                   dist_a=float(A["dist"].mean()), dist_b=float(B["dist"].mean()),
                   dist_diff_mean=float((B["dist"] - A["dist"]).mean()),
                   dist_diff_std=float((B["dist"] - A["dist"]).std()))
    else:
        out["cosine"] = float((A[k] * B[k]).sum() / np.linalg.norm(A[k]) / np.linalg.norm(B[k]))
    print(json.dumps(out))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["proj", "path", "compare"])
    p.add_argument("files", nargs="*")
    p.add_argument("--gan")
    p.add_argument("--ops", choices=["ref", "compiled"], default="ref")
    p.add_argument("--tf32", choices=["off", "cudnn", "matmul", "all"], default="cudnn")  # cudnn = PROBE now
    p.add_argument("--amp", choices=["none", "vgg", "all"], default="none")  # bf16 autocast
    p.add_argument("--cl", action="store_true", help="channels_last for VGG16 / ResNet-18")
    p.add_argument("--compile", default="", help="torch.compile mode for G.synthesis, e.g. default")
    p.add_argument("--n", type=int, default=32)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--points", type=int, default=48)
    p.add_argument("--n-ids", type=int, default=463)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--out")
    p.add_argument("--log")
    p.add_argument("--tag", default="")
    args = p.parse_args()
    if args.mode == "compare":
        compare(*args.files)
        sys.exit()
    G = setup(args)
    res = dict(tag=args.tag, mode=args.mode, ops=args.ops, tf32=args.tf32, amp=args.amp, cl=args.cl,
               compile=args.compile, **(proj if args.mode == "proj" else path)(args, G))
    from torch_utils.ops import bias_act, upfirdn2d
    res["ops_loaded"] = bias_act._plugin is not None and upfirdn2d._plugin is not None  # no silent fallback
    print(json.dumps(res), flush=True)
    if args.log:
        with open(args.log, "a") as f:
            f.write(json.dumps(res) + "\n")

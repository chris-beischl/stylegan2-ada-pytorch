# Spike: StyleGAN2-ADA for PROBE/PLAN under torch 2.14

Goal: find out, time-boxed (1-2 days of GPU-session work), whether this
code base trains, generates and projects a **label-conditioned 256 x 256**
StyleGAN2-ADA on our stack (Python 3.12, torch 2.14.1 + CUDA 13, as in
PROBE), what it costs, and which minimal patches it needs. The code may be
thrown away; the findings are the result. Cluster rules: `CLAUDE.md`.

Why these properties (PLAN, Pennisi et al. 2023, and its official
`privacy_path.py`):
- StyleGAN2-ADA, label-conditioned, trained on the members (one GAN per
  member set in PROBE), images 256 x 256;
- paths in W space: one w per point, broadcast to all `G.mapping.num_ws`
  layers, synthesised with `G.synthesis(w, noise_mode='const')`;
- GAN projections of real images (5 per identity for ϕ_id; possibly the
  path endpoints, open question in PROBE §15.1).

## Findings before the spike (static, 2026-10-08, no code run)

| # | Finding | Planned handling |
|---|---|---|
| F1 | `dnnlib/util.py` imports `distutils.util.strtobool`; distutils is gone in Python 3.12 | replace with a local `strtobool` (`# PROBE:` comment) |
| F2 | `torch_utils/ops/grid_sample_gradfix.py` enables itself only for torch 1.7-1.9; otherwise falls back to `F.grid_sample`. ADA's geometric augmentation + R1 needs the double backward through `grid_sample` (`training_loop.py:131` enables it "to avoid errors with the augmentation pipe") | S4: test the double backward on torch 2.14; if missing, port NVIDIA's newer version from github.com/NVlabs/stylegan3 (same licence, supports torch >= 1.11) |
| F3 | `torch_utils/ops/conv2d_gradfix.py` likewise 1.7-1.9 only; fallback is plain `conv2d` | correct, only slower (weight gradients in R1/path-length passes); keep unless S4/S6 show a problem |
| F4 | Custom CUDA ops (`bias_act`, `upfirdn2d`) compile at run time with `nvcc`; the login node has no `nvcc`, no `/usr/local/cuda`, no module system | S3: check the GPU node; else reference implementation (slow) or `nvcc` as pip wheels in this repo's `.venv` (no system changes) |
| F5 | `projector.py` computes the average w with `G.mapping(z, None)`: fails for a conditional G (`c_dim > 0`) | S8: patch to take a class label (`--class`), average w per class |
| F6 | `projector.py` downloads VGG16 (LPIPS) from `https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt`; `--metrics` (FID) would download Inception | pin URL + SHA-256 for VGG16; spike runs use `--metrics=none` |
| F7 | `projector.py` imports `imageio` at top level (video only) | add `imageio` to the environment, or leave video off |
| F8 | Pillow constants (`LANCZOS`, `BOX`), NumPy 2 removed aliases | checked: none of the removed names are used |

## Steps

Every step: record commands, versions, timings and outcomes in
`probe/progress.md`; patches on branch `probe` with `# PROBE: <why>`.

**S0 Preconditions (inside the GPU job).** `nvidia-smi` (GPU type, driver,
free memory), `echo $CUDA_VISIBLE_DEVICES $SLURM_CPUS_PER_TASK`, memory
cap, `df -h .`, remaining job time. `which nvcc`, `ls /usr/local/cuda*`.

**S1 Environment.** New `pyproject.toml` in this repo (not in PROBE):
Python 3.12, `torch==2.14.1` (same build as PROBE: PyPI, cu130), numpy,
pillow, click, requests, tqdm, ninja, imageio; scipy only if metrics are
needed. `uv sync`; check `torch.cuda.is_available()`, device name.
Decide later (S10) whether PROBE and the GAN can share one torch version;
a separate env is the default either way (licence, isolation).

**S2 Python 3.12.** Patch F1. `uv run python train.py --help` must work.

**S3 Custom ops.** With `TORCH_EXTENSIONS_DIR=$PWD/.torch_extensions`,
run a tiny forward/backward through `bias_act` and `upfirdn2d` on the GPU.
Outcomes: compiled (record nvcc version) / fell back (record the warning).
If it falls back and S6 shows the reference ops are too slow, try `nvcc`
from pip wheels matching CUDA 13 inside `.venv` (set `CUDA_HOME`), never a
system install. Record the speed difference.

**S4 Gradfix.** Tiny test: `x.requires_grad_()`, `y = F.grid_sample(x,
grid)`, `g = autograd.grad(y.sum(), x, create_graph=True)`, then
`g[0].sum().backward()`. Works -> F2 needs nothing. Fails -> port
StyleGAN3's `grid_sample_gradfix.py` (and `conv2d_gradfix.py` if useful),
rerun the test with the gradfix enabled.

**S5 Data.** Export Shenzhen at 256 x 256 with labels (normal/TB) into
`out/data/shenzhen256.zip` in `dataset_tool.py` format (PNG files +
`dataset.json` with `{"labels": [[file, label], ...]}`), read through
PROBE's `shenzhen` plugin so the pixels equal what PROBE trains on (run
with PROBE's environment, read-only on PROBE and the raw data; a throwaway
script in `probe/`). Members only would be one PROBE split; for the spike
all 662 images are fine.

**S6 Training run.** `uv run python train.py --outdir=out/train
--data=out/data/shenzhen256.zip --cond=1 --mirror=0 --gpus=1
--cfg=paper256 --metrics=none --kimg=200 --snap=12 --seed=0
--workers=<= cpus`. `--mirror=0`: chest X-rays are not left/right
symmetric. Record: sec/kimg, GPU memory, augment probability p, losses,
whether `fakes*.png` look like X-rays of both classes. Stop early if the
time budget says so. Compare `--cfg=paper256` and `auto` only if cheap.

**S7 Determinism.** Two short runs (e.g. `--kimg=10`) with the same seed:
compare the final snapshot's parameters (max abs difference) and the
logged losses. Expected: not bitwise equal (atomics, ADA's adaptive p) ->
PROBE records the GAN as `deterministic = False`.

**S8 Generation and projection.** `generate.py --class 0/1
--noise-mode const` from the S6 snapshot. Patch F5, then project a few
real images with their class (`--num-steps` default 1000 and fewer);
record time per projection and the LPIPS/pixel distance to the target.
PLAN needs 5 projections per identity: cost x 5 x n_train per member set.

**S9 Gradient through G in PROBE's environment** (replaces the PLAN path
smoke test; PLAN's code belongs in PROBE, D-51). Load the S6 snapshot with
PROBE's `.venv/bin/python` and only this repo on `PYTHONPATH` (no install,
nothing written in PROBE; custom ops forced to the reference
implementation, so no build is attempted). Backpropagate a fixed dummy loss
from `G.synthesis` to a path of T = 50 broadcast w (`noise_mode='const'`),
compare the gradient with the same computation in this repo's env
(compiled ops), and time a forward/backward step for both.

**S10 Report and decisions** (in `probe/progress.md`, for the user and
PROBE's planning):
- runs: yes/no; required patches (list, each with reason);
- custom ops: compiled or reference, speed;
- cost: sec/kimg; projections/s; path optimisation/s; estimated GPU hours
  for one PLAN cell (GAN on n_train members + projections + paths);
- determinism outcome;
- environment: separate `.venv` (default) or shared torch version;
- interface for PROBE (D-51): what `train.py` needs as a subprocess (data
  zip, seeds, options, outputs), what PROBE imports to load G, packaging
  of this repo as a git dependency — to be planned as a PROBE PR.

## Acceptance

The spike is done when S10 is written with measured numbers, every patch
is committed on `probe` with a reason, `out/` holds only what S10 refers
to, and no process or job of ours is left running.

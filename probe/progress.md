# Progress (fork)

Handoff between sessions in this repository. Update at the end of every
session. Plan: `probe/SPIKE.md`. Rules: `CLAUDE.md`.

## Status (2026-10-08, evening session in GPU job 286359)

- Fork of NVIDIA upstream `d72cc7d` on github.com/chris-beischl; branch
  `probe` for our work, `main` untouched.
- S0-S5 done; S6 under way (training runs with compiled custom ops).
- S0: node `atlas`, partition `universe`, 1x A40 (46 GB), driver 595.91.07
  (CUDA 13.2), 2 CPUs, job RAM cap 8 GB (cgroup), 16 h limit. No `nvcc`, no
  `/usr/local/cuda`; gcc/g++ 13.3 present; 27 TB free on the volume.
- S1: `pyproject.toml` (not a package), torch 2.14.1+cu130 like PROBE,
  numpy 2.5.3, scipy 1.18.1 (needed by `training/augment.py`), psutil
  (needed by `training_loop.py`), imageio, ninja, click, requests, tqdm.
  Group `cuda-build`: nvcc/crt/nvvm 13.0.88, cccl 13.0.85 as pip wheels.
- The venv must use a uv-managed Python (system 3.12.3 has no `Python.h`,
  which the extension build needs). Installed repo-local:
  `UV_PYTHON_INSTALL_DIR=$PWD/.uv-python uv python install --no-bin 3.12`
  (3.12.15; without `--no-bin` uv also puts a `python3.12` link into
  `~/.local/bin`, removed again). Recreate the env with
  `UV_PYTHON_INSTALL_DIR=$PWD/.uv-python uv sync --group cuda-build --python .uv-python/cpython-3.12.15-linux-x86_64-gnu/bin/python3.12`.
- Every run: `source probe/env.sh` (PYTHONPATH, TORCH_EXTENSIONS_DIR,
  CUDA_HOME = the wheel's `nvidia/cu13`, arch from nvidia-smi, threads,
  and `.cuda_link/libcudart.so` -> the wheel's `libcudart.so.13` on
  `LIBRARY_PATH`, since the wheel has no unversioned `.so` for `-lcudart`).

## Next step

S6 longer run (200 kimg), then S7-S10.

## Patches on `probe` (each: file, reason, step)

- `dnnlib/util.py`: local `strtobool` (F1; distutils removed in 3.12; it
  only imported because setuptools ships a distutils shim). S2.
- `torch_utils/misc.py`: `InfiniteSampler` calls `super().__init__()`;
  torch >= 2.2 `Sampler.__init__` takes no `data_source` (TypeError). S6.
- `torch_utils/ops/upfirdn2d.py`: set `_inited = True` (upstream bug: a
  failed build was retried on every call; `bias_act.py` already does it). S3.
- `torch_utils/ops/{conv2d,grid_sample}_gradfix.py`: warn once. Other
  modules' `catch_warnings()` reset the warning registry, so the fallback
  warning repeated on every conv (35 MB log in 15 min). S6.
- `torch_utils/custom_ops.py`: use the module returned by
  `cpp_extension.load()`; torch 2.14 no longer makes it importable by name
  (`ModuleNotFoundError` after a successful build). S3.
- New: `pyproject.toml`, `uv.lock`, `probe/env.sh`, `probe/spike_ops.py`
  (S3/S4 test), `probe/export_shenzhen.py` (S5).
- F2 (grid_sample double backward) needs nothing: works natively in torch
  2.14 (S4). F3 (conv2d_gradfix) falls back to plain conv2d as planned.

## Measurements

| Item | Value | Notes |
|---|---|---|
| GPU / driver / CUDA | A40 46 GB / 595.91.07 / 13.2 (torch cu130) | S0 |
| nvcc available | no system nvcc; pip wheel nvcc 13.0.88 works | S0, S3 |
| custom ops | compiled (after 3 env fixes + custom_ops patch); bias_act+upsample fwd/bwd 3.4 ms vs ref 7.6 ms, identical outputs | S3 |
| grid_sample double backward | works natively | S4 |
| sec/kimg (256 px, cond, batch) | | S6 |
| GPU memory | | S6 |
| bitwise reproducible | | S7 |
| time per projection | | S8 |
| time per path step (T = 50) | | S9 |

## S5 data

`probe/export_shenzhen.py` (run with PROBE's `.venv/bin/python`, not
`uv run --project`, which reinstalls PROBE into its venv): all 662 images
through PROBE's `shenzhen` plugin at 256 px (bilinear resize, RGB),
exact uint8 round trip asserted; `out/data/shenzhen256.zip` (131 MB,
326 normal / 336 TB). 92 s for the export on 2 CPUs.

## S6 runs

| Run | Ops | kimg | sec/kimg | GPU mem | CPU RAM | Notes |
|---|---|---|---|---|---|---|
| 00000 | reference | 8 | 92 (tick 1) | 4.5 GB | 3.7 GB | GPU util ~64 %; tick 2 (119) slowed by a parallel compile |

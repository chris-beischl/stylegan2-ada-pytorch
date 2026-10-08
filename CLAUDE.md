# stylegan2-ada-pytorch — PROBE fork

Personal fork (github.com/chris-beischl/stylegan2-ada-pytorch) of NVIDIA's
official StyleGAN2-ADA (github.com/NVlabs/stylegan2-ada-pytorch, commit
d72cc7d). It provides the GAN for the PLAN defense in PROBE
(`/vol/miltank/users/beischlc/Documents/PROBE`, see its `docs/design.md`
§15.1 and `docs/progress.md`). PLAN (Pennisi et al., MICCAI 2023) used
exactly this code base: a label-conditioned StyleGAN2-ADA at 256 x 256.

Current phase: the **spike** in `probe/SPIKE.md` (does it run under
torch 2.14 / CUDA 13, at what cost, with which patches). Handoff between
sessions: `probe/progress.md`. Read both before doing anything.

## Licence (why this is a separate repository)

NVIDIA Source Code License (`LICENSE.txt`): non-commercial research use
only; redistribution must keep the licence and notices. Works that remain
separable from the code or merely call its interfaces are not derivative
works (§1). So: never copy or move code from this repository into PROBE.
Decided 2026-10-08 (PROBE D-51): until PROBE is published, PROBE uses this
fork as an optional, commit-pinned git dependency. GAN training runs here as
a subprocess (this repo's own env); PROBE imports the modules by name only to
load the trained generator pickle and backpropagate through `G.synthesis`.
PLAN's logic (paths, classifiers, losses) lives in PROBE, never here. After
publication PLAN becomes an external PROBE plugin package.

## Branches and diffs

- `main` mirrors NVIDIA's upstream unchanged. Never commit to it.
- `probe` holds all our changes. Keep the diff against `main` minimal and
  reviewable: patch only what is needed, never reformat or restyle upstream
  files (no ruff/black over upstream code), keep NVIDIA's headers.
- New files of ours live in `probe/` (docs, helper scripts, CLI glue) or
  are clearly named; every patch to an upstream file gets a short comment
  `# PROBE: <why>` and a line in `probe/progress.md`.
- Commit messages explain why. Push only when the user asks.

## Cluster safety (shared Slurm cluster; nothing may break)

- **Jobs:** never submit (`sbatch`, `srun`), cancel (`scancel`) or modify
  Slurm jobs unless the user explicitly asks in the current session. Work
  only inside the allocation this session runs in.
- **GPU:** use only the GPU(s) Slurm assigned (`CUDA_VISIBLE_DEVICES`);
  always `train.py --gpus=1` unless the allocation has more and the user
  agreed. Check `nvidia-smi` first; never touch other users' processes.
- **CPU and memory:** set `OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK`; keep data
  loader workers <= allocated CPUs; check the job's memory cap
  (`cat /sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup)/memory.max`, or the
  parent cgroup) before loading datasets.
- **Software:** no system or global installs (no sudo, no `pip install
  --user`, no conda base changes, no module changes). Only `uv` with this
  repository's own `.venv`; CUDA tools, if needed, as pip wheels inside it.
  Compiled extensions go to `TORCH_EXTENSIONS_DIR=<repo>/.torch_extensions`.
- **Disk:** write only inside this repository (`out/`, gitignored) or the
  session scratchpad; check free space (`df -h .`) before training; keep
  snapshots sparse (`--snap`) and delete spike snapshots that are not
  needed. Datasets under `/vol/miltank/users/beischlc/datasets/` are
  read-only: never write there.
- **Time:** keep every run well inside the job's time limit (`squeue -j
  $SLURM_JOB_ID -o %L`); no background processes left running when a
  session ends.
- **Network:** only PyPI (via uv) and the files the spike lists (NVIDIA's
  VGG16 for projection); record URL and SHA-256 of every download.
- **PROBE:** read only from this session. Changes to PROBE are made in a
  PROBE session (its own CLAUDE.md, tests and review rules apply).

## Commands

- `uv sync` — environment (after the spike creates `pyproject.toml`)
- `uv run python train.py --help`, `generate.py`, `projector.py`,
  `dataset_tool.py` — NVIDIA's entry points (see README.md)

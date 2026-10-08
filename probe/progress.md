# Progress (fork)

Handoff between sessions in this repository. Update at the end of every
session. Plan: `probe/SPIKE.md`. Rules: `CLAUDE.md`.

## Status (2026-10-08)

- Fork of NVIDIA upstream `d72cc7d` on github.com/chris-beischl; branch
  `probe` created for our work, `main` untouched.
- Static review done (findings F1-F8 in `probe/SPIKE.md`); nothing
  installed, compiled or run yet. No pyproject/uv env yet (S1).
- Login node has no GPU, no `nvcc`, no `/usr/local/cuda`, no `module`.

## Next step

The user restarts Claude Code inside a Slurm job with a GPU. Then: S0
(record the node), S1-S10 in order.

## Patches on `probe` (each: file, reason, step)

(none yet)

## Measurements

| Item | Value | Notes |
|---|---|---|
| GPU / driver / CUDA | | S0 |
| nvcc available | | S0, S3 |
| custom ops | compiled / reference | S3 |
| grid_sample double backward | | S4 |
| sec/kimg (256 px, cond, batch) | | S6 |
| GPU memory | | S6 |
| bitwise reproducible | | S7 |
| time per projection | | S8 |
| time per path step (T = 50) | | S9 |

# Handover: GPU performance of the fork as PROBE uses it (2026-10-15)

From a PROBE session, at the user's request. Read `CLAUDE.md` (rules, cluster
safety) and `probe/progress.md` (patches F-1 … F-5, measurements S0-S10)
first. This file is the task; record results in `probe/progress.md`.

## Goal

The user will rerun all PLAN experiments (PROBE data seeds 0-4, Shenzhen and
APTOS 2019), so numerics may change now; afterwards they must not. Find out
which changes make the GAN work faster per GPU-hour, **measure** them, and
deliver (a) patches on `probe` where the change belongs in this repo and
(b) a list of PROBE-side changes for a PROBE session (never edit PROBE from
here). The output is an answer with numbers, not a refactoring.

## Where the time goes (PROBE paper protocol, data seed 0, one A40 each)

Per PLAN cell, from PROBE's `defense_report.timings_s`:

| Step | Shenzhen (463 members) | APTOS (577 members) | Runs in |
|---|---|---|---|
| GAN training, `--cfg=paper256 --cond=1 --mirror=0`, 1 000 kimg | 8.6 h | 7.5 h (26.2 s/kimg) | this repo's env, compiled ops (`probe/train_gan.sh`) |
| Projections: 5 per member, 1 000 steps, `projector_batch.project_batch`, batch 32, `init='sample'` | 9.4 h (2 315 targets, 14.7 s each) | 11.8 h (2 885 targets) | PROBE's process, **reference ops** (F-2) |
| Path optimisation: 231 / 288 paths x 100 Adam steps, 48 interior points per step through `G.synthesis` + two ResNet-18 | 3.2 h (0.49 s/step) | 4.0 h | PROBE's process, reference ops |
| Rendering the paths (11 550 / 14 400 images) and downstream training | ~1 h | ~1.5 h | PROBE |

Over the remaining runs, projections and GAN training dominate (each
~20 h per data seed for both datasets), then paths. PROBE will store
trained GANs (already, D-52) and is adding stores for models and released
data, so each GAN and each projection set is computed once per data seed.

Known numbers (`probe/progress.md`): compiled vs reference ops: training
28 vs 92 s/kimg (3.3x, hence training runs here); `G.synthesis` fwd+bwd at
48 points 319 vs 392 ms (1.23x); projection at batch 32 3.84 vs 4.29 s/img
per 300 steps (1.12x). Compiled ops are not bitwise equal to the reference
(`--use_fast_math`, max rel diff 2.6e-3).

## How PROBE calls this repo (read-only references in PROBE)

- `src/probe/gan/stylegan2_ada.py`: `train_gan` (subprocess
  `probe/train_gan.sh`, options incl. `--allow-tf32`, default off),
  `load_generator` (sets `STYLEGAN2ADA_REFERENCE_OPS=1`, `pickle.load`s
  `G_ema.pkl`), `project` (imports `projector_batch`, checks VGG16 SHA-256).
- `src/probe/methods/defenses/plan.py`: `optimise_path` (Adam on the path
  interior, chunks of `chunk` points through `G.synthesis(ws,
  noise_mode='const')`, gradients via `torch.autograd.grad`), `render`.
- PROBE runs with `torch.use_deterministic_algorithms(True, warn_only=True)`,
  `cudnn.deterministic=True`, `cudnn.benchmark=False`,
  `CUBLAS_WORKSPACE_CONFIG=:4096:8`; matmul TF32 off, cuDNN TF32 at torch's
  default (on). Projection and paths are bitwise reproducible under this.
- PROBE pins this fork's commit (`113de6a`); a new commit needs a PROBE
  change (pin, GAN-store key includes the commit).

## Candidates to measure (on one A40, as in the jobs)

For projection, path step and training, report time, peak memory, and the
effect on results (projection LPIPS distance; for training, sec/kimg and a
look at samples; never claim equality without a check).

1. **Compiled ops inside PROBE's process.** PROBE's venv cannot compile
   (F-2). Can PROBE load the extensions built here (`.torch_extensions/`,
   same torch 2.14.1+cu130 and cp312) without nvcc, e.g. by pointing
   `TORCH_EXTENSIONS_DIR` at them? What would break (Python 3.12.15 vs
   3.12.3, CUDA arch, determinism under `use_deterministic_algorithms`)?
   Expected gain is only 1.1-1.25x (above); worth it only if cheap and safe.
2. **TF32 / reduced precision** for `G.synthesis` + VGG16 in projection and
   paths (`torch.backends.cuda.matmul.allow_tf32`, cuDNN TF32, autocast
   bf16 where G is fp32), and `--allow-tf32=1` for training. Effect on LPIPS
   distance and samples.
3. **Batch size.** Projection throughput at batch 32 / 64 / 96 / 128
   (memory on 46 GB); `G.synthesis` fwd+bwd per point at 48 / 96 / 144
   points. This decides whether PROBE should optimise several paths at once
   (only pays if per-point throughput grows beyond 48 points).
4. **torch.compile / CUDA graphs / channels_last** for `G.synthesis` in
   PROBE's process (with reference ops, the graph is plain PyTorch).
5. **Projection steps.** LPIPS distance vs step (100 … 1 000) for the
   trained 1 000 kimg GANs; does it plateau early? This is a method choice
   (PLAN's paper setting), so report it; the user decides.
6. **GAN training.** `--workers`, `--batch`, `--nobench` interaction, data
   loading on the node's CPUs (`OMP_NUM_THREADS`), GPU utilisation. Training
   is not reproducible anyway (S7).

Trained 1 000 kimg generators for testing are in PROBE's GAN store
(`/vol/miltank/users/beischlc/Documents/PROBE/runs/gans/<key>/G_ema.pkl`;
read-only), member images via `out/data/` zips or `probe/export_shenzhen.py`.

## Constraints

- Cluster safety as in `CLAUDE.md`: GPU benchmarks only inside a GPU
  allocation the user gives you; no job submission without asking.
- Licence: nothing from this repo is copied into PROBE.
- Patches minimal, `# PROBE: <why>`, listed in `probe/progress.md`; commit on
  `probe` with a message that explains why; push only when asked.
- Every change that alters numbers PROBE sees must be selectable or pinned
  so PROBE can hash it (as `batch_size` in F-4).

## Deliverable

A section "Performance (2026-10)" in `probe/progress.md`: a table per
candidate (setting, time, memory, result change), a recommendation per
candidate (adopt / skip / user decides), and the list of PROBE-side changes
with expected savings in GPU-hours for 5 data seeds x 2 datasets.

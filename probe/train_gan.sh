#!/bin/bash
# PROBE: training entry point for PROBE's GAN backend (PROBE design D-51). Contract:
#   probe/train_gan.sh --outdir=DIR [train.py options, e.g. --data=ZIP --cond=1 --seed=N --kimg=K]
# DIR must not contain a previous run. Runs train.py with --outdir=DIR/run (log: DIR/train.log),
# then writes DIR/G_ema.pkl (generator only, from the last snapshot; load with pickle.load(f)["G_ema"])
# and DIR/backend.json (commit, dirty flag, options, kimg, sec/kimg). Non-zero exit on failure.
# Needs a synced env in this checkout: uv sync --group cuda-build (see probe/progress.md).
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
outdir=""
args=()
for a in "$@"; do
    case $a in
        --outdir=*) outdir=${a#--outdir=} ;;
        *) args+=("$a") ;;
    esac
done
if [ -z "$outdir" ]; then
    echo "usage: probe/train_gan.sh --outdir=DIR [train.py options]" >&2
    exit 2
fi
mkdir -p "$outdir"
outdir=$(cd "$outdir" && pwd)
if [ -e "$outdir/run" ] || [ -e "$outdir/G_ema.pkl" ]; then
    echo "error: $outdir already holds a run" >&2
    exit 2
fi
source "$repo/probe/env.sh"
cd "$repo"
uv run --no-sync python train.py --outdir="$outdir/run" "${args[@]}" >> "$outdir/train.log" 2>&1
uv run --no-sync python probe/finish_training.py "$outdir" >> "$outdir/train.log" 2>&1

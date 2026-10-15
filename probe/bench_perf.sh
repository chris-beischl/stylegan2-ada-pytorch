#!/bin/bash
# PROBE: performance study (probe/perf-handover.md), run inside a GPU job: bash probe/bench_perf.sh PHASE...
# Phases: proj (projection switches, batch), path (PLAN chunk step), steps (LPIPS dist vs steps),
# quality (1000 steps per candidate), verify (fork patches, combination, fixed-seed paths, compile),
# train (train.py options, 12 kimg each). Results: out/perf/*.jsonl, logs in out/perf/.
set -uo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo"
py=$repo/../PROBE/.venv/bin/python
gans=$repo/../PROBE/runs/gans
sz=$gans/2c06ef60c44821f59e11ff1d13029d22d9ab6a98f33f47baa4cf465b52771663  # Shenzhen seed 0, 463 members
ap=$gans/f617292c63ce7fa754d939b78422150e0433654aaf58f6901a44ed2e4d36a00e  # APTOS seed 0, 577 members
out=out/perf
mkdir -p $out/npz
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2}
export TORCHINDUCTOR_CACHE_DIR=$repo/$out/inductor TRITON_CACHE_DIR=$repo/$out/triton  # not /tmp, ~/.triton
b() { $py probe/bench_perf.py "$@" 2>&1 | grep -v "^Setting up\|UserWarning\|warnings.warn"; }

for phase in "$@"; do case $phase in
proj)
    l=$out/proj.jsonl
    b proj --gan $sz --tag base --log $l --out $out/npz/p_base.npz
    b proj --gan $sz --tag base2 --log $l --out $out/npz/p_base2.npz
    b proj --gan $sz --tag compiled --ops compiled --log $l --out $out/npz/p_compiled.npz
    b proj --gan $sz --tag compiled2 --ops compiled --log $l --out $out/npz/p_compiled2.npz
    b proj --gan $sz --tag tf32off --tf32 off --log $l --out $out/npz/p_tf32off.npz
    b proj --gan $sz --tag tf32all --tf32 all --log $l --out $out/npz/p_tf32all.npz
    b proj --gan $sz --tag ampvgg --amp vgg --log $l --out $out/npz/p_ampvgg.npz
    b proj --gan $sz --tag ampvgg2 --amp vgg --log $l --out $out/npz/p_ampvgg2.npz
    b proj --gan $sz --tag ampvgg_cl --amp vgg --cl --log $l --out $out/npz/p_ampvgg_cl.npz
    b proj --gan $sz --tag cl --cl --log $l --out $out/npz/p_cl.npz
    b proj --gan $sz --tag ampall --amp all --log $l --out $out/npz/p_ampall.npz
    b proj --gan $sz --tag compile --compile default --log $l --out $out/npz/p_compile.npz
    for n in 64 96 128; do b proj --gan $sz --tag batch$n --n $n --batch $n --log $l; done
    for t in base2 compiled compiled2 tf32off tf32all ampvgg ampvgg2 ampvgg_cl cl ampall compile; do
        echo "base vs $t: $($py probe/bench_perf.py compare $out/npz/p_base.npz $out/npz/p_$t.npz 2>&1 | tail -1)"
    done | tee $out/proj_compare.txt ;;
path)
    l=$out/path.jsonl
    for p in 48 96 144; do b path --gan $sz --points $p --tag base --log $l; done
    b path --gan $sz --tag base48 --log $l --out $out/npz/g_base.npz
    b path --gan $sz --tag base48b --log $l --out $out/npz/g_base2.npz
    b path --gan $sz --tag compiled --ops compiled --log $l --out $out/npz/g_compiled.npz
    b path --gan $sz --tag compiled2 --ops compiled --log $l --out $out/npz/g_compiled2.npz
    for p in 96 144; do b path --gan $sz --points $p --tag compiled --ops compiled --log $l; done
    b path --gan $sz --tag tf32all --tf32 all --log $l --out $out/npz/g_tf32all.npz
    b path --gan $sz --tag ampall --amp all --log $l --out $out/npz/g_ampall.npz
    b path --gan $sz --tag cl --cl --log $l --out $out/npz/g_cl.npz
    b path --gan $sz --tag compile --compile default --log $l --out $out/npz/g_compile.npz
    for t in base2 compiled compiled2 tf32all ampall cl compile; do
        echo "base vs $t: $($py probe/bench_perf.py compare $out/npz/g_base.npz $out/npz/g_$t.npz 2>&1 | tail -1)"
    done | tee $out/path_compare.txt ;;
steps)
    for g in sz ap; do for s in 100 200 300 500 1000; do
        b proj --gan ${!g} --steps $s --tag steps_$g --log $out/steps.jsonl --out $out/npz/s_${g}_$s.npz
    done; done ;;
quality)  # 1000 steps with the candidate switches; baseline = steps phase
    for g in sz ap; do for o in "ampvgg --amp vgg" "compiled --ops compiled"; do set -- $o
        b proj --gan ${!g} --steps 1000 --tag q_$1_$g ${@:2} --log $out/steps.jsonl --out $out/npz/q_${g}_$1.npz
        echo "$g 1000 steps base vs $1: $($py probe/bench_perf.py compare $out/npz/s_${g}_1000.npz $out/npz/q_${g}_$1.npz 2>&1 | tail -1)" | tee -a $out/quality_compare.txt
    done; done ;;
verify)  # the fork patches reproduce the monkeypatched runs bitwise; combination; fixed-seed paths; compile
    l=$out/verify.jsonl
    c() { echo "$1 vs $2: $($py probe/bench_perf.py compare $out/npz/$1.npz $out/npz/$2.npz 2>&1 | tail -1)" | tee -a $out/verify_compare.txt; }
    b proj --gan $sz --tag v_base --log $l --out $out/npz/v_base.npz; c p_base v_base
    b proj --gan $sz --tag v_ampvgg --amp vgg --log $l --out $out/npz/v_ampvgg.npz; c p_ampvgg v_ampvgg
    b proj --gan $sz --tag v_compiled --ops compiled --log $l --out $out/npz/v_compiled.npz; c p_compiled v_compiled
    for g in sz ap; do
        b proj --gan ${!g} --steps 1000 --tag v_both_$g --ops compiled --amp vgg --log $l --out $out/npz/v_both_$g.npz
        c s_${g}_1000 v_both_$g
    done
    for t in base base2 compiled compiled2; do
        o=""; [[ $t == compiled* ]] && o="--ops compiled"
        b path --gan $sz --tag v_$t $o --log $l --out $out/npz/vg_$t.npz
    done
    b path --gan $sz --tag v_tf32all --tf32 all --log $l --out $out/npz/vg_tf32all.npz
    for t in base2 compiled compiled2 tf32all; do c vg_base vg_$t; done
    # Triton builds its launcher against Python.h, which PROBE's system Python lacks: borrow 3.12 headers.
    export CPATH=$repo/.uv-python/cpython-3.12.15-linux-x86_64-gnu/include/python3.12
    b proj --gan $sz --tag v_compile --compile default --log $l --out $out/npz/v_compile.npz; c p_base v_compile
    b path --gan $sz --tag v_compile --compile default --log $l --out $out/npz/vg_compile.npz; c vg_base vg_compile
    unset CPATH ;;
train)
    source probe/env.sh
    w=$(( $(nproc) < 4 ? $(nproc) : 4 ))  # as PROBE's train_gan
    t() {  # tag, extra train.py options
        local d=$out/train/$1; shift
        local sel=""  # cgroup-constrained jobs show only our GPU; otherwise select it
        [ "$(nvidia-smi -L | wc -l)" -gt 1 ] && sel="-i $CUDA_VISIBLE_DEVICES"
        nvidia-smi $sel --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits -l 2 > $d.smi &
        local smi=$!
        uv run --no-sync python train.py --outdir=$d --data=out/data/shenzhen256.zip --cond=1 --mirror=0 \
            --gpus=1 --cfg=paper256 --metrics=none --kimg=12 --snap=0 --seed=0 "$@" > $d.log 2>&1
        kill $smi
        find $d -name 'network-snapshot-*.pkl' -delete  # 307 MB each, not needed
        echo "$(basename $d) $*: $(grep -o 'sec/kimg [0-9.]*' $d.log | tail -n +2 | awk '{s+=$2;n++} END {printf "%.1f sec/kimg (ticks 1-%d)", s/n, n}');" \
             "GPU util $(awk -F, '{s+=$1;n++} END {printf "%.0f", s/n}' $d.smi) %, max mem $(sort -t, -k2 -n $d.smi | tail -1 | cut -d, -f2) MiB" | tee -a $out/train.txt
    }
    mkdir -p $out/train
    t base --workers=$w
    t tf32 --workers=$w --allow-tf32=1
    t nhwc --workers=$w --nhwc=1
    t nobench --workers=$w --nobench=1
    t workers1 --workers=1
    t tf32_nhwc --workers=$w --allow-tf32=1 --nhwc=1
    t batch128 --workers=$w --batch=128 ;;
*) echo "unknown phase $phase" >&2; exit 2 ;;
esac; done

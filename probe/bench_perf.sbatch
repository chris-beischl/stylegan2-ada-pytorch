#!/bin/bash
# PROBE: the performance study as one job (probe/perf-handover.md); resources as PROBE's PLAN jobs.
# Submit from the checkout: sbatch probe/bench_perf.sbatch
#SBATCH --job-name=fork-perf
#SBATCH --partition=universe
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=out/perf/job-%j.log
set -u
echo "host $(hostname), CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, cpus $SLURM_CPUS_PER_TASK, $(date)"
nvidia-smi -L
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
cat /sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup)/memory.max
df -h . | tail -1
time bash probe/bench_perf.sh proj path steps quality train
echo "done $(date)"

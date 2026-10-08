# PROBE: environment for running this fork on a GPU node. Usage: source probe/env.sh
# Builds bias_act/upfirdn2d with nvcc from pip wheels (uv sync --group cuda-build); the
# nodes have no system CUDA toolkit. The venv must use the uv-managed Python in
# .uv-python/ (the system Python 3.12 lacks Python.h), see probe/progress.md.
_repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONPATH=$_repo
export TORCH_EXTENSIONS_DIR=$_repo/.torch_extensions
export DNNLIB_CACHE_DIR=$_repo/.cache/dnnlib  # downloads (VGG16 for projection) stay in the repo
export UV_PYTHON_INSTALL_DIR=$_repo/.uv-python
export CUDA_HOME=$_repo/.venv/lib/python3.12/site-packages/nvidia/cu13
export TORCH_CUDA_ARCH_LIST=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-$(nproc)}
export MAX_JOBS=$OMP_NUM_THREADS
# The wheel ships only libcudart.so.13; torch's extension build links -lcudart.
mkdir -p "$_repo/.cuda_link"
ln -sf "$CUDA_HOME/lib/libcudart.so.13" "$_repo/.cuda_link/libcudart.so"
export LIBRARY_PATH=$_repo/.cuda_link${LIBRARY_PATH:+:$LIBRARY_PATH}
unset _repo

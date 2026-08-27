#!/bin/bash
# M1 task 1.6 (estimator A): MDC2 33-pulsar Stage C run with the HD<->CURN
# correlation path coordinate eps SAMPLED, so one posterior spans both models.
#
# Read the Bayes factor afterwards with:
#   JAX_PLATFORMS=cpu python scripts/lnb_savage_dickey.py \
#       --results outputs/mdc2_stageC_path_sampled/mdc2_stageC_path_sampled_results.nc \
#       --out outputs/mdc2_stageC_path_sampled/lnb_savage_dickey.json
#
# Submit:  sbatch workflows/ng15_sgwb_demo/slurm_scripts/mdc2_path_sampled.sh
# Resume after a timeout: set resume = true in the config and resubmit.

#SBATCH --job-name=mdc2_path_sampled
#SBATCH --account=oz022
# The OzSTAR job_submit plugin canonicalizes any GPU request to milan-gpu.
#SBATCH --partition=milan-gpu
# 4 GPUs matched to num_chains=4 so chains run in parallel; NUTS chains are
# sequential per device.
#SBATCH --gres=gpu:4
# The ridge Stage C runs took ~9.5 h at these settings. This adds one dimension
# whose posterior may pile against a prior boundary, which can slow mixing, so
# the budget is generous. Checkpointing is on (interval 250 draws), so a timeout
# now costs the last segment rather than the whole run.
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --export=ALL
#SBATCH --chdir=/fred/oz022/tkimpson/Argus/workflows/ng15_sgwb_demo
#SBATCH --output=/fred/oz022/tkimpson/Argus/workflows/ng15_sgwb_demo/outputs/logfiles/mdc2_path_sampled_%j.out

# NB: no `set -e` -- sourcing ~/.bashrc / conda init returns non-zero in a
# non-interactive shell, which under `set -e` aborts before any output.

ROOT=/fred/oz022/tkimpson/Argus/workflows/ng15_sgwb_demo
REPO_PY=/fred/oz022/tkimpson/Argus/python
CONFIG="${ROOT}/configs/mdc2_stage_c_path_sampled.ini"

if [ ! -e "${ROOT}/data/mdc2_all" ]; then
    echo "ERROR: ${ROOT}/data/mdc2_all not found (symlink to the ingested feathers)." >&2
    exit 1
fi
if [ ! -f "${ROOT}/data/stage_a_empirical_priors.json" ]; then
    echo "ERROR: ${ROOT}/data/stage_a_empirical_priors.json not found." >&2
    exit 1
fi

mkdir -p "${ROOT}/outputs/logfiles"

source ~/.bashrc
conda activate /fred/oz022/tkimpson/conda_envs/Argus

# argus is pip-installed EDITABLE pointing at this checkout, so PYTHONPATH is
# belt-and-braces here rather than the fix it is for a worktree. The provenance
# lines below prove which code actually ran.
export PYTHONPATH="${REPO_PY}:${PYTHONPATH}"

echo "=== env check ==="
which python
python -c "import jax, flax; print('jax', jax.__version__, 'flax', flax.__version__)"
python -c "import argus.gravitational_waves as gw; print('argus from:', gw.__file__); print('correlation_path available:', hasattr(gw, 'correlation_path'))"
python -c "import argus.checkpointing as cp; print('checkpointing available:', hasattr(cp, 'save_checkpoint'))"
git -C /fred/oz022/tkimpson/Argus rev-parse --short HEAD
git -C /fred/oz022/tkimpson/Argus rev-parse --abbrev-ref HEAD
nvidia-smi -L

echo "=== running task 1.6 (eps sampled) ==="
time python -u "${ROOT}/run_analysis.py" "${CONFIG}"

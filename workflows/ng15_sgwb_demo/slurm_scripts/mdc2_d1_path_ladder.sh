#!/bin/bash
# POSITIVE CONTROL: the path-sampling ladder on MDC2 open dataset 1b.
#
# 2b is a published non-detection (arXiv:1912.12939 reports B = 1.1-2.6 there against a
# threshold of 3; we measured lnB = 0.175 +/- 0.017, consistent). 1b has a smaller
# injected amplitude but no per-pulsar red noise, and was strongly detected in the same
# paper: B = 40 to infinity. A working pipeline must return a decisive answer here.
# See notes/mdc2_literature_check.md.
#
# One array task per rung, each a Stage C run with eps FIXED. The rung values
# form a uniform 5-point grid so the ladder can be coarsened twice, which is what
# lets lnb_path_sampling.py estimate its own discretisation error by Romberg
# extrapolation. Refine (to 9 or 13 rungs, keeping n-1 divisible by 4) if the
# residual is above tolerance.
#
# Read the Bayes factor afterwards with:
#   JAX_PLATFORMS=cpu python scripts/lnb_path_sampling.py \
#       --from-runs 'outputs/mdc2_d1_ladder_eps*/mdc2_d1_ladder_eps*_results.nc' \
#       --evaluate configs/mdc2_d1_ladder_eps000.ini \
#       --out outputs/lnb_path_sampling_d1.json
# (--evaluate needs any one rung's config: it supplies the data and model, and
#  each rung's own eps is read from its results file.)
#
# Submit:  sbatch workflows/ng15_sgwb_demo/slurm_scripts/mdc2_path_ladder.sh

#SBATCH --job-name=mdc2_d1_ladder
#SBATCH --account=oz022
#SBATCH --partition=milan-gpu
#SBATCH --gres=gpu:4
# Half the draws of task 1.6 (1000/1000 vs 2000/1500): a rung needs the posterior
# MEAN of one scalar to modest precision, not a well-resolved posterior for every
# parameter. Checkpointing is on, so a timeout costs one segment.
#SBATCH --time=16:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --array=0-4
#SBATCH --export=ALL
#SBATCH --chdir=/fred/oz022/tkimpson/Argus/workflows/ng15_sgwb_demo
#SBATCH --output=/fred/oz022/tkimpson/Argus/workflows/ng15_sgwb_demo/outputs/logfiles/mdc2_d1_ladder_%A_%a.out

ROOT=/fred/oz022/tkimpson/Argus/workflows/ng15_sgwb_demo
REPO_PY=/fred/oz022/tkimpson/Argus/python

# Uniform 5-point grid on [0, 1]. Both endpoints are required: the integral is
# ln Z(1) - ln Z(0), so a ladder that misses one is not a Bayes factor.
EPS_VALUES=(0.0 0.25 0.5 0.75 1.0)
EPS_TAGS=(000 025 050 075 100)

IDX="${SLURM_ARRAY_TASK_ID:-0}"
EPS="${EPS_VALUES[$IDX]}"
TAG="${EPS_TAGS[$IDX]}"

if [ -z "${EPS}" ]; then
    echo "ERROR: no eps value for array index ${IDX}" >&2
    exit 1
fi

if [ ! -f "${ROOT}/data/stage_a_d1_empirical_priors.json" ]; then
    echo "ERROR: ${ROOT}/data/stage_a_d1_empirical_priors.json not found." >&2
    echo "Run the d1 Stage A array, then scripts/extract_stage_a.py --run-prefix mdc2_d1_stageA_" >&2
    exit 1
fi
if [ ! -e "${ROOT}/data/mdc2_d1_all" ]; then
    echo "ERROR: ${ROOT}/data/mdc2_d1_all not found." >&2
    exit 1
fi

mkdir -p "${ROOT}/outputs/logfiles"

# Derive this rung's config from the template.
CONFIG="${ROOT}/configs/mdc2_d1_ladder_eps${TAG}.ini"
sed -e "s/__EPS_VALUE__/${EPS}/" -e "s/__EPS_TAG__/${TAG}/" \
    "${ROOT}/configs/mdc2_d1_ladder_rung.ini.template" > "${CONFIG}"

source ~/.bashrc
conda activate /fred/oz022/tkimpson/conda_envs/Argus
export PYTHONPATH="${REPO_PY}:${PYTHONPATH}"

echo "=== env check (rung ${IDX}: eps=${EPS}) ==="
which python
python -c "import argus.gravitational_waves as gw; print('argus from:', gw.__file__); print('correlation_path available:', hasattr(gw, 'correlation_path'))"
git -C /fred/oz022/tkimpson/Argus rev-parse --short HEAD
nvidia-smi -L
grep -E "orf_path|orf_epsilon_value|output_id" "${CONFIG}"

echo "=== running ladder rung eps=${EPS} ==="
time python -u "${ROOT}/run_analysis.py" "${CONFIG}"

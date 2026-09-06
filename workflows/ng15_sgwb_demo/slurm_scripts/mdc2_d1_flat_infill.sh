#!/bin/bash
# INFILL: the three interior rungs of the FLAT-prior path-sampling ladder on 1b.
#
# The removal test (slurm_scripts/mdc2_d1_flat_ladder.sh) ran the two ENDPOINTS with
# flat per-pulsar red-noise priors and confirmed the diagnosis outright: the pivot
# log-PSD moved from -9.5 (posterior = prior, sd/prior_sd = 0.95-0.99) to -6.47 with
# sd/prior_sd = 0.21 at eps=1, against the injected -6.908. Per-pulsar red noise came
# back constrained, not prior-dominated (sd 0.218 = 9.4% of prior sd, 0/33 railing).
#
# Those two runs cannot give a Bayes factor -- ln B = integral over eps of the
# integrand, and two points do not integrate a path. This job adds eps = 0.25, 0.5,
# 0.75 to complete the frozen uniform 5-point grid, whose spacing is what lets
# lnb_path_sampling.py coarsen twice (5 -> 3 -> 2) and estimate its own discretisation
# error by Romberg extrapolation. lnb_path_sampling.py enforces MIN_RUNGS = 5.
#
# The endpoints are NOT re-run: outputs/mdc2_d1_flat_eps000 and _eps100 are already
# rungs 1 and 5, produced from this same template with the same priors. Only the
# interior three are missing.
#
# Identical in every respect to the endpoint job except the eps values and the
# resource request, which is trimmed to what the endpoints actually used (9.4 GB peak
# of 32 GB; 2h23m and 1h30m of 16 h) to schedule sooner in a deep queue. Resource
# requests do not touch the numerics; --gres=gpu:4 is held fixed because num_chains
# is 4.
#
# Read the Bayes factor afterwards, once all five rungs exist:
#   JAX_PLATFORMS=cpu python scripts/lnb_path_sampling.py \
#       --from-runs 'outputs/mdc2_d1_flat_eps*/mdc2_d1_flat_eps*_results.nc' \
#       --evaluate configs/mdc2_d1_flat_eps000.ini \
#       --batch-size 100 \
#       --out outputs/lnb_path_sampling_d1_flat.json
# (--evaluate needs any one rung's config: it supplies the data and model, and each
#  rung's own eps is read from its results file. --batch-size is required or the
#  integrand evaluation is OOM-killed. Budget ~30 min and ~19 GB for five rungs.)
#
# Submit:  sbatch workflows/ng15_sgwb_demo/slurm_scripts/mdc2_d1_flat_infill.sh

#SBATCH --job-name=mdc2_d1_flat_infill
#SBATCH --account=oz022
#SBATCH --partition=milan-gpu
#SBATCH --gres=gpu:4
#SBATCH --time=08:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --array=0-2
#SBATCH --export=ALL
#SBATCH --chdir=/fred/oz022/tkimpson/Argus/workflows/ng15_sgwb_demo
#SBATCH --output=/fred/oz022/tkimpson/Argus/workflows/ng15_sgwb_demo/outputs/logfiles/mdc2_d1_flat_infill_%A_%a.out

ROOT=/fred/oz022/tkimpson/Argus/workflows/ng15_sgwb_demo
REPO_PY=/fred/oz022/tkimpson/Argus/python

# The three INTERIOR rungs. The endpoints (0.0, 1.0) already exist; see the header.
EPS_VALUES=(0.25 0.5 0.75)
EPS_TAGS=(025 050 075)

IDX="${SLURM_ARRAY_TASK_ID:-0}"
EPS="${EPS_VALUES[$IDX]}"
TAG="${EPS_TAGS[$IDX]}"

if [ -z "${EPS}" ]; then
    echo "ERROR: no eps value for array index ${IDX}" >&2
    exit 1
fi

if [ ! -e "${ROOT}/data/mdc2_d1_all" ]; then
    echo "ERROR: ${ROOT}/data/mdc2_d1_all not found." >&2
    exit 1
fi

# The endpoints must survive untouched for the ladder to be complete and consistent.
for END_TAG in 000 100; do
    if [ ! -f "${ROOT}/outputs/mdc2_d1_flat_eps${END_TAG}/mdc2_d1_flat_eps${END_TAG}_results.nc" ]; then
        echo "ERROR: endpoint rung eps${END_TAG} missing; run mdc2_d1_flat_ladder.sh first." >&2
        exit 1
    fi
done

mkdir -p "${ROOT}/outputs/logfiles"

# Derive this rung's config from the SAME template the endpoints used.
CONFIG="${ROOT}/configs/mdc2_d1_flat_eps${TAG}.ini"
sed -e "s/__EPS_VALUE__/${EPS}/" -e "s/__EPS_TAG__/${TAG}/" \
    "${ROOT}/configs/mdc2_d1_flat_rung.ini.template" > "${CONFIG}"

# Fail loudly rather than silently reverting to empirical priors: empirical_priors_path
# takes precedence over red_noise_prior in get_pulsar_noise_priors, so a stray copy of
# that line would put this rung on a different model from the endpoints and quietly
# corrupt the integral.
if grep -qE "^\s*empirical_priors_path\s*=\s*\S" "${CONFIG}"; then
    echo "ERROR: ${CONFIG} still sets empirical_priors_path; the ladder is void." >&2
    exit 1
fi
if ! grep -qE "^\s*red_noise_prior\s*=\s*flat\s*$" "${CONFIG}"; then
    echo "ERROR: ${CONFIG} does not set red_noise_prior = flat." >&2
    exit 1
fi

source ~/.bashrc
conda activate /fred/oz022/tkimpson/conda_envs/Argus
export PYTHONPATH="${REPO_PY}:${PYTHONPATH}"

echo "=== env check (flat infill rung ${IDX}: eps=${EPS}) ==="
which python
python -c "import argus.gravitational_waves as gw; print('argus from:', gw.__file__); print('correlation_path available:', hasattr(gw, 'correlation_path'))"
git -C /fred/oz022/tkimpson/Argus rev-parse --short HEAD
nvidia-smi -L
grep -E "orf_path|orf_epsilon_value|red_noise_prior|empirical|output_id" "${CONFIG}"

echo "=== running flat infill rung eps=${EPS} ==="
time python -u "${ROOT}/run_analysis.py" "${CONFIG}"

#!/bin/bash
# NULL CALIBRATION: the full 5-rung flat-prior ladder on MDC2 1b under a SCRAMBLED sky.
#
# The flat-prior ladder returned lnB(HD/CURN) = 3.043 +/- 0.015 with the pivot log-PSD
# recovered onto the injected -6.908 (outputs/lnb_path_sampling_d1_flat.json). That number
# does not yet distinguish "we detect the Hellings-Downs correlation" from "flat red-noise
# priors let the GW claim any common power going". This ladder is the falsification test:
# same residuals, same epochs, same per-pulsar noise, same priors, same NUTS settings --
# only the sky geometry destroyed.
#
#   lnB ~ 0 with a flat, near-zero integrand => 3.043 measures correlation. Null passes.
#   lnB ~ 3                                  => flat priors manufacture HD evidence. STOP;
#                                               re-examine the amplitude result before
#                                               running 2b or scaling to M3.
#
# ONE scramble, run COLD (full warmup). This is a falsification check, not a false-alarm
# probability -- N=1 gives a point, not a distribution, and the estimator's +/- 0.015 is
# its own numerical error, saying nothing about scramble-to-scramble scatter. The ensemble
# (null-calibration spec tasks 4.3-4.7) is a separate, much larger decision that this run's
# outcome informs.
#
# All five rungs are run fresh rather than reusing the existing eps000. At eps=0 the model
# never touches the ORF (correlation_path interpolates toward the identity), so
# mdc2_d1_null_eps000 MUST reproduce mdc2_d1_flat_eps000 (integrand 4.695 +/- 0.061) within
# MC error -- a free check that the override landed. Array tasks run concurrently, so the
# fifth rung costs no wall clock.
#
# Read the Bayes factor afterwards with the WRAPPER, not the estimator directly:
#   JAX_PLATFORMS=cpu python scripts/lnb_scrambled.py \
#       --scramble-npz data/scrambles/mdc2_d1_scrambles.npz --scramble-index 0 \
#       --from-runs 'outputs/mdc2_d1_null_eps*/mdc2_d1_null_eps*_results.nc' \
#       --evaluate configs/mdc2_d1_null_eps000.ini \
#       --batch-size 100 \
#       --out outputs/lnb_path_sampling_d1_null.json
# lnb_path_sampling.py --evaluate REBUILDS the data from the config and would integrate
# against the TRUE ORF, returning a well-formed and meaningless number. --batch-size is
# required or the integrand evaluation is OOM-killed (~30 min, ~19 GB for five rungs).
#
# Submit:  sbatch workflows/ng15_sgwb_demo/slurm_scripts/mdc2_d1_null_ladder.sh

#SBATCH --job-name=mdc2_d1_null
#SBATCH --account=oz022
#SBATCH --partition=milan-gpu
#SBATCH --gres=gpu:4
# Sized from the flat ladder's measured usage (9.4 GB peak of 32 GB; 1h30m-2h23m of 16 h).
# gpu:4 is fixed by num_chains = 4. Checkpointing is on, so a timeout costs one segment
# rather than the run.
#SBATCH --time=08:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --array=0-4
#SBATCH --export=ALL
#SBATCH --chdir=/fred/oz022/tkimpson/Argus/workflows/ng15_sgwb_demo
#SBATCH --output=/fred/oz022/tkimpson/Argus/workflows/ng15_sgwb_demo/outputs/logfiles/mdc2_d1_null_%A_%a.out

ROOT=/fred/oz022/tkimpson/Argus/workflows/ng15_sgwb_demo
REPO_PY=/fred/oz022/tkimpson/Argus/python
SCRAMBLE_NPZ="${ROOT}/data/scrambles/mdc2_d1_scrambles.npz"
SCRAMBLE_INDEX=0

# The frozen uniform 5-point grid. Uniform spacing is what lets lnb_path_sampling.py
# coarsen twice (5 -> 3 -> 2) and estimate its own discretisation error by Romberg
# extrapolation; it enforces MIN_RUNGS = 5.
EPS_VALUES=(0.0 0.25 0.5 0.75 1.0)
EPS_TAGS=(000 025 050 075 100)

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

# Without the scramble this job is a duplicate of the flat ladder, not a null.
if [ ! -f "${SCRAMBLE_NPZ}" ]; then
    echo "ERROR: ${SCRAMBLE_NPZ} not found; generate it with scripts/sky_scrambles.py." >&2
    exit 1
fi

mkdir -p "${ROOT}/outputs/logfiles"

# Derive this rung's config from the template.
CONFIG="${ROOT}/configs/mdc2_d1_null_eps${TAG}.ini"
sed -e "s/__EPS_VALUE__/${EPS}/" -e "s/__EPS_TAG__/${TAG}/" \
    "${ROOT}/configs/mdc2_d1_null_rung.ini.template" > "${CONFIG}"

# Inherited from the flat ladder scripts and kept deliberately: empirical_priors_path takes
# precedence over red_noise_prior in get_pulsar_noise_priors, so a stray copy of that line
# would silently change the noise model out from under the null and make it incomparable to
# the flat ladder it is the control for.
if grep -qE "^\s*empirical_priors_path\s*=\s*\S" "${CONFIG}"; then
    echo "ERROR: ${CONFIG} still sets empirical_priors_path; the null is not comparable." >&2
    exit 1
fi
if ! grep -qE "^\s*red_noise_prior\s*=\s*flat\s*$" "${CONFIG}"; then
    echo "ERROR: ${CONFIG} does not set red_noise_prior = flat." >&2
    exit 1
fi

source ~/.bashrc
conda activate /fred/oz022/tkimpson/conda_envs/Argus
export PYTHONPATH="${REPO_PY}:${PYTHONPATH}"

echo "=== env check (null rung ${IDX}: eps=${EPS}) ==="
which python
python -c "import argus.gravitational_waves as gw; print('argus from:', gw.__file__); print('correlation_path available:', hasattr(gw, 'correlation_path'))"
git -C /fred/oz022/tkimpson/Argus rev-parse --short HEAD
nvidia-smi -L
grep -E "orf_path|orf_epsilon_value|red_noise_prior|empirical|output_id" "${CONFIG}"

echo "=== running scrambled rung eps=${EPS} ==="
time python -u "${ROOT}/run_scrambled.py" "${CONFIG}" \
    --scramble-npz "${SCRAMBLE_NPZ}" --scramble-index "${SCRAMBLE_INDEX}"

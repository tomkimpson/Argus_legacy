#!/bin/bash
# REMOVAL TEST: 1b at the two ladder endpoints with FLAT per-pulsar red-noise priors.
#
# The 1b positive control failed under the frozen two-stage procedure -- the pivot
# log-PSD posterior came back as its prior (sd 95-99% of prior sd, centre drifting away
# from the injected -6.908), giving lnB = 0.053 +/- 0.004 on a dataset the IPTA detect at
# B = 40 to infinity. The diagnosis is that Stage A's single-pulsar fits absorb the GWB
# into per-pulsar red noise, and Stage C then imposes those posteriors as priors.
# See notes/PROBLEM_empirical_priors_absorb_gwb.md.
#
# This tests that diagnosis by removal. configs/mdc2_d1_flat_rung.ini.template is the
# empirical ladder template with the empirical priors deleted and flat Uniforms in their
# place; everything else is held identical.
#
#   Amplitude moves from ~-9.5 toward -6.9  => diagnosis and fix confirmed together.
#   It does not                             => mechanism is wrong; next suspects are the
#                                              GW prior parameterization and the Stage A
#                                              white-noise treatment.
#
# TWO rungs only (eps = 0 and 1). This is a diagnostic on the amplitude posterior, NOT a
# Bayes factor -- two endpoints cannot integrate the path, so do not run
# lnb_path_sampling.py on these. If the amplitude moves, the full 5-rung ladder follows.
#
# Read the result with (note --log10-a: the script defaults to dataset2/2b's -14.886,
# and 1b's injected amplitude is -15.18046 from group1_gw_parameters.json "dataset1"):
#   JAX_PLATFORMS=cpu python scripts/check_mdc2_truth.py \
#       --run mdc2_d1_flat_eps000 --run mdc2_d1_flat_eps100 \
#       --log10-a -15.18045606445813 \
#       --out outputs/mdc2_d1_flat_truth_gate.json
# Then compare against the empirical-prior runs at the same rungs
# (outputs/mdc2_d1_ladder_eps000, ..._eps100). The number that matters is the pivot
# log-PSD posterior sd against the N(-9.0, 1.333) prior sd: the failure signature was
# sd/prior_sd ~ 1 with the centre drifting away from the injected -6.908.
#
# Submit:  sbatch workflows/ng15_sgwb_demo/slurm_scripts/mdc2_d1_flat_ladder.sh

#SBATCH --job-name=mdc2_d1_flat
#SBATCH --account=oz022
#SBATCH --partition=milan-gpu
#SBATCH --gres=gpu:4
# Same budget as the empirical rungs, which converged well inside it. Wide priors can
# cost tree depth, but max_tree_depth is capped at 8 and checkpointing is on, so a
# timeout costs one segment rather than the run.
#SBATCH --time=16:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --array=0-1
#SBATCH --export=ALL
#SBATCH --chdir=/fred/oz022/tkimpson/Argus/workflows/ng15_sgwb_demo
#SBATCH --output=/fred/oz022/tkimpson/Argus/workflows/ng15_sgwb_demo/outputs/logfiles/mdc2_d1_flat_%A_%a.out

ROOT=/fred/oz022/tkimpson/Argus/workflows/ng15_sgwb_demo
REPO_PY=/fred/oz022/tkimpson/Argus/python

# The two ladder ENDPOINTS. eps=0 is CURN (no cross-correlations), eps=1 is full
# Hellings-Downs. Both are needed: the whole point is whether the amplitude is recovered
# with and without the cross-correlations available to break the GW<->red-noise
# degeneracy that flat priors make explicit.
EPS_VALUES=(0.0 1.0)
EPS_TAGS=(000 100)

IDX="${SLURM_ARRAY_TASK_ID:-0}"
EPS="${EPS_VALUES[$IDX]}"
TAG="${EPS_TAGS[$IDX]}"

if [ -z "${EPS}" ]; then
    echo "ERROR: no eps value for array index ${IDX}" >&2
    exit 1
fi

# No empirical-priors guard here: their ABSENCE is the point of this run.
if [ ! -e "${ROOT}/data/mdc2_d1_all" ]; then
    echo "ERROR: ${ROOT}/data/mdc2_d1_all not found." >&2
    exit 1
fi

mkdir -p "${ROOT}/outputs/logfiles"

# Derive this rung's config from the template.
CONFIG="${ROOT}/configs/mdc2_d1_flat_eps${TAG}.ini"
sed -e "s/__EPS_VALUE__/${EPS}/" -e "s/__EPS_TAG__/${TAG}/" \
    "${ROOT}/configs/mdc2_d1_flat_rung.ini.template" > "${CONFIG}"

# Fail loudly rather than silently reverting to empirical priors: empirical_priors_path
# takes precedence over red_noise_prior in get_pulsar_noise_priors, so a stray copy of
# that line would make this run a duplicate of the one it is meant to be a control for.
if grep -qE "^\s*empirical_priors_path\s*=\s*\S" "${CONFIG}"; then
    echo "ERROR: ${CONFIG} still sets empirical_priors_path; the removal test is void." >&2
    exit 1
fi
if ! grep -qE "^\s*red_noise_prior\s*=\s*flat\s*$" "${CONFIG}"; then
    echo "ERROR: ${CONFIG} does not set red_noise_prior = flat." >&2
    exit 1
fi

source ~/.bashrc
conda activate /fred/oz022/tkimpson/conda_envs/Argus
export PYTHONPATH="${REPO_PY}:${PYTHONPATH}"

echo "=== env check (flat rung ${IDX}: eps=${EPS}) ==="
which python
python -c "import argus.gravitational_waves as gw; print('argus from:', gw.__file__); print('correlation_path available:', hasattr(gw, 'correlation_path'))"
git -C /fred/oz022/tkimpson/Argus rev-parse --short HEAD
nvidia-smi -L
grep -E "orf_path|orf_epsilon_value|red_noise_prior|empirical|output_id" "${CONFIG}"

echo "=== running flat-prior rung eps=${EPS} ==="
time python -u "${ROOT}/run_analysis.py" "${CONFIG}"

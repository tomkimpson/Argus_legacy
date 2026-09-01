#!/bin/bash
# Stage A for MDC2 open dataset 1b — the POSITIVE CONTROL.
#
# Why 1b: dataset 2b, which M1 used, is a published non-detection. The IPTA MDC2
# results paper (arXiv:1912.12939) reports Bayes factors of only 1.1-2.6 for it under
# every search method, against their own detection threshold of 3. Dataset 1b has a
# SMALLER injected amplitude (0.66e-15 vs 1.3e-15) but NO per-pulsar red noise, and was
# strongly detected there: B = 40 (free WN) to infinity (fixed WN), with a real amplitude
# measurement 0.7 +0.4/-0.3 e-15 against 0.66 injected.
#
# So 1b is the dataset on which a working pipeline MUST return a decisive answer, and it
# carries an independent published number to check against. See
# notes/mdc2_literature_check.md.
#
# Same single-pulsar noise characterization as the 2b Stage A, on the 1b feathers, with
# run directories prefixed mdc2_d1_stageA_ so the two datasets' results cannot be mixed.
#
# NOTE: 1b has no injected red noise, so these runs are expected to return posteriors
# consistent with negligible red noise -- possibly railing against the prior floor
# (log10_sigma_p_min = -20). extract_stage_a.py flags railed pulsars; CHECK ITS REPORT
# before building empirical priors, because a Normal fitted to a railed posterior is not
# a meaningful prior.
#
# Submit:  sbatch workflows/ng15_sgwb_demo/slurm_scripts/mdc2_d1_stage_a.sh

#SBATCH --job-name=mdc2_d1_stage_a
#SBATCH --account=oz022
#SBATCH --partition=milan-gpu
#SBATCH --gres=gpu:1
# 2 chains run sequentially on the single GPU; 1:30 was sufficient for the 2b Stage A.
#SBATCH --time=1:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --export=ALL
#SBATCH --array=0-32
#SBATCH --chdir=/fred/oz022/tkimpson/Argus/workflows/ng15_sgwb_demo
#SBATCH --output=/fred/oz022/tkimpson/Argus/workflows/ng15_sgwb_demo/outputs/logfiles/mdc2_d1_stage_a_%A_%a.out

ROOT=/fred/oz022/tkimpson/Argus/workflows/ng15_sgwb_demo
REPO_PY=/fred/oz022/tkimpson/Argus/python

mapfile -t PSRS < <(ls -d "${ROOT}"/data/mdc2_d1_singles/*/ | xargs -n1 basename | sort)
if [ "${#PSRS[@]}" -eq 0 ]; then
    echo "ERROR: no staged pulsar directories under ${ROOT}/data/mdc2_d1_singles/" >&2
    echo "Run: scripts/stage_mdc2.py --feather-dir data/mdc2_d1_all --output-dir data/mdc2_d1_singles" >&2
    exit 1
fi
if [ "${SLURM_ARRAY_TASK_ID}" -ge "${#PSRS[@]}" ]; then
    echo "ERROR: array task ${SLURM_ARRAY_TASK_ID} >= ${#PSRS[@]} staged pulsars" >&2
    exit 1
fi
PSR="${PSRS[$SLURM_ARRAY_TASK_ID]}"

PSR_DIR="${ROOT}/data/mdc2_d1_singles/${PSR}/"
PSR_NOISE="${ROOT}/data/mdc2_d1_singles/${PSR}/psr_noise.json"
# Derived config must live OUTSIDE the run's own output dir (run_inference copies the
# config into it) but inside the workflow tree, with ABSOLUTE paths.
RUN="${ROOT}/outputs/derived_configs/d1_stage_a_${PSR}.ini"

mkdir -p "${ROOT}/outputs/derived_configs" "${ROOT}/outputs/logfiles"

sed -e "s|^data_path = .*|data_path = ${PSR_DIR}|" \
    -e "s|^noise_params_path = .*|noise_params_path = ${PSR_NOISE}|" \
    -e "s|^output_id = .*|output_id = mdc2_d1_stageA_${PSR}|" \
    "${ROOT}/configs/mdc2_stage_a.ini" > "${RUN}"

echo "=== Stage A (d1) task ${SLURM_ARRAY_TASK_ID}: ${PSR} ==="
grep -E "^data_path|^noise_params_path|^output_id" "${RUN}"

source ~/.bashrc
conda activate /fred/oz022/tkimpson/conda_envs/Argus
export PYTHONPATH="${REPO_PY}:${PYTHONPATH}"

echo "=== env check ==="
which python
python -c "import argus.prior_models as pm; print('argus from:', pm.__file__)"
git -C /fred/oz022/tkimpson/Argus rev-parse --short HEAD
nvidia-smi -L

echo "=== running Stage A (d1): ${PSR} ==="
time python -u "${ROOT}/run_analysis.py" "${RUN}"

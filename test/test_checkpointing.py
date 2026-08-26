"""Checkpointing and resumption of long NUTS runs.

Argus previously wrote results only after sampling finished, so a run that hit its
walltime lost everything -- a routine outcome for a multi-day 68-pulsar array run on a
contended GPU partition. These tests pin the three properties that make checkpointing
worth having: an interrupted run leaves usable output, a resume continues the same
chain rather than restarting it, and a resume against a changed configuration is
refused rather than silently mixing incompatible draws.
"""

import json
import os
import pickle
from unittest.mock import patch

import arviz as az
import numpy as np
import pytest

from argus import checkpointing
from argus import workflow

# --------------------------------------------------------------------------
# Settings and segmentation
# --------------------------------------------------------------------------


def _config(text):
    import configparser

    config = configparser.ConfigParser()
    config.read_string(text)
    return config


def test_checkpointing_is_off_by_default():
    """A config with no [Checkpointing] section keeps the previous behaviour."""
    settings = checkpointing.get_checkpoint_settings(_config("[Data]\nx = 1\n"))
    assert settings["enabled"] is False
    assert settings["resume"] is False


def test_enabling_without_an_interval_is_an_error():
    """Enabling checkpointing without saying how often is a configuration mistake."""
    config = _config("[Checkpointing]\nenabled = true\n")
    with pytest.raises(ValueError, match="interval"):
        checkpointing.get_checkpoint_settings(config)


@pytest.mark.parametrize(
    "total, interval, expected",
    [
        (1000, 250, [250, 250, 250, 250]),
        (1000, 300, [300, 300, 300, 100]),
        (100, 250, [100]),
        (100, 0, [100]),
        (7, 3, [3, 3, 1]),
    ],
)
def test_segments_sum_to_the_requested_draws(total, interval, expected):
    """However the interval divides, the run must produce exactly what was asked for."""
    sizes = checkpointing.segment_sizes(total, interval)
    assert sizes == expected
    assert sum(sizes) == total


# --------------------------------------------------------------------------
# The fingerprint gate
# --------------------------------------------------------------------------


def test_fingerprint_is_stable_across_key_order():
    """Rewriting a config must not invalidate a checkpoint."""
    first = checkpointing.run_fingerprint({"a": 1, "b": {"x": 2, "y": 3}})
    second = checkpointing.run_fingerprint({"b": {"y": 3, "x": 2}, "a": 1})
    assert first == second


def test_fingerprint_changes_with_the_model():
    """A moved prior bound must invalidate it."""
    base = {"priors": {"log10_ha": ["Uniform", -17.0, -11.0]}}
    moved = {"priors": {"log10_ha": ["Uniform", -16.0, -11.0]}}
    assert checkpointing.run_fingerprint(base) != checkpointing.run_fingerprint(moved)


def test_missing_checkpoint_returns_none(tmp_path):
    """A first run and a resume share a code path, so absence is not an error."""
    assert checkpointing.load_checkpoint(str(tmp_path), "run", "abc") is None


def test_resume_refuses_a_fingerprint_mismatch(tmp_path):
    """The property that makes a resume safe: incompatible draws are never mixed."""
    idata = _dummy_inference_data(draws=5)
    checkpointing.save_checkpoint(
        str(tmp_path),
        "run",
        {"dummy": np.zeros(3)},
        idata,
        "fingerprint-A",
        {
            "draws_completed": 5,
            "draws_target": 10,
            "segments_completed": 1,
            "segments_total": 2,
        },
    )
    with pytest.raises(ValueError, match="fingerprint"):
        checkpointing.load_checkpoint(str(tmp_path), "run", "fingerprint-B")


def test_resume_refuses_when_the_partial_posterior_is_missing(tmp_path):
    """Half a checkpoint is not a checkpoint."""
    idata = _dummy_inference_data(draws=5)
    checkpointing.save_checkpoint(
        str(tmp_path),
        "run",
        {"dummy": np.zeros(3)},
        idata,
        "fp",
        {
            "draws_completed": 5,
            "draws_target": 10,
            "segments_completed": 1,
            "segments_total": 2,
        },
    )
    _, partial_path = checkpointing.checkpoint_paths(str(tmp_path), "run")
    os.remove(partial_path)
    with pytest.raises(ValueError, match="incomplete"):
        checkpointing.load_checkpoint(str(tmp_path), "run", "fp")


def test_round_trip_preserves_state_and_progress(tmp_path):
    state = {"step_size": np.array([0.1, 0.2]), "counter": 7}
    idata = _dummy_inference_data(draws=5)
    checkpointing.save_checkpoint(
        str(tmp_path),
        "run",
        state,
        idata,
        "fp",
        {
            "draws_completed": 5,
            "draws_target": 10,
            "segments_completed": 1,
            "segments_total": 2,
        },
    )
    loaded = checkpointing.load_checkpoint(str(tmp_path), "run", "fp")
    np.testing.assert_array_equal(
        loaded["sampler_state"]["step_size"], state["step_size"]
    )
    assert loaded["progress"]["draws_completed"] == 5
    assert loaded["inference_data"].posterior.sizes["draw"] == 5


# --------------------------------------------------------------------------
# The partial marker
# --------------------------------------------------------------------------


def test_partial_output_is_labelled(tmp_path):
    """Output from an incomplete run must be distinguishable from a finished one."""
    idata = _dummy_inference_data(draws=5)
    checkpointing.save_checkpoint(
        str(tmp_path),
        "run",
        {},
        idata,
        "fp",
        {
            "draws_completed": 5,
            "draws_target": 10,
            "segments_completed": 1,
            "segments_total": 2,
        },
    )
    _, partial_path = checkpointing.checkpoint_paths(str(tmp_path), "run")
    reloaded = az.from_netcdf(partial_path)
    assert checkpointing.is_complete(reloaded) is False
    assert "5/10" in checkpointing.describe_progress(reloaded)


def test_completed_run_is_labelled_complete(tmp_path):
    idata = _dummy_inference_data(draws=10)
    checkpointing.save_checkpoint(
        str(tmp_path),
        "run",
        {},
        idata,
        "fp",
        {
            "draws_completed": 10,
            "draws_target": 10,
            "segments_completed": 2,
            "segments_total": 2,
        },
    )
    _, partial_path = checkpointing.checkpoint_paths(str(tmp_path), "run")
    reloaded = az.from_netcdf(partial_path)
    assert checkpointing.is_complete(reloaded) is True


def test_results_without_a_marker_are_treated_as_complete():
    """Every result predating checkpointing is complete by construction."""
    assert checkpointing.is_complete(_dummy_inference_data(draws=3)) is True


# --------------------------------------------------------------------------
# End-to-end: segmentation, resume, and equivalence
# --------------------------------------------------------------------------


STAGE_A_TEMPLATE = """
[Data]
data_path = {psr_dir}
excluded_psrs = __NONE__

[NUTS]
num_samples = {num_samples}
num_warmup = 10
num_chains = 1
target_accept_prob = 0.9
max_tree_depth = 5
dense_mass = true
seed = 1234

[PriorModel]
log10_ha_fixed = true
log10_ha_value = -20.0
log10_gamma_a_fixed = true
log10_gamma_a_value = {log10_gamma_a_value}

spin_injections_path =
red_noise_prior = flat

log10_gamma_p_min = -12.0
log10_gamma_p_max = -6.0
log10_sigma_p_min = -20.0
log10_sigma_p_max = -12.0

noise_params_path = {noise_json}

efac_min = 0.5
efac_max = 2.0
log10_equad_min = -8.0
log10_equad_max = -6.0

[Checkpointing]
enabled = {enabled}
interval = {interval}
resume = {resume}

[Logging]
level = INFO
enable_file_logging = false

[Output]
output_id = ckpt_smoke
base_dir = {{output_id}}
"""


@pytest.fixture
def staged_pulsar(tmp_path):
    """A one-pulsar directory and noise JSON, as scripts/stage_mdc2.py produces."""
    feather_src = os.path.abspath("test/data/test_pulsar.feather")
    if not os.path.exists(feather_src):
        pytest.skip(f"test feather not found: {feather_src}")

    psr_dir = tmp_path / "J9999+9999"
    psr_dir.mkdir()
    os.symlink(feather_src, psr_dir / "J9999+9999.feather")
    noise_json = psr_dir / "psr_noise.json"
    noise_json.write_text(json.dumps({"J9999+9999": {"efac": 1.0, "equad": -7.0}}))
    return psr_dir, noise_json


def _write_config(tmp_path, staged, name, **overrides):
    psr_dir, noise_json = staged
    values = {
        "psr_dir": psr_dir,
        "noise_json": noise_json,
        "num_samples": 12,
        "enabled": "false",
        "interval": 0,
        "resume": "false",
        "log10_gamma_a_value": -8.5,
    }
    values.update(overrides)
    path = tmp_path / f"{name}.ini"
    path.write_text(STAGE_A_TEMPLATE.format(**values))
    return str(path)


def _run(config_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    with patch("argus.io_manager.setup_output_directory", return_value=output_dir):
        return workflow.run_inference(config_path, use_gw=True)


def test_segmented_run_matches_an_unsegmented_one(tmp_path, staged_pulsar):
    """Task 3.3: segmenting must not change the chain.

    Same seed, same model, same total draws -- one run in a single shot, the other in
    three segments. The draws must be identical, not merely statistically similar:
    continuing from the stored state is the same Markov chain, so anything less would
    mean the resume path perturbs the sampler.
    """
    single = _write_config(tmp_path, staged_pulsar, "single", num_samples=12)
    segmented = _write_config(
        tmp_path,
        staged_pulsar,
        "segmented",
        num_samples=12,
        enabled="true",
        interval=4,
    )

    _run(single, str(tmp_path / "out_single"))
    _run(segmented, str(tmp_path / "out_segmented"))

    single_idata = az.from_netcdf(
        os.path.join(tmp_path, "out_single", "ckpt_smoke_results.nc")
    )
    segmented_idata = az.from_netcdf(
        os.path.join(tmp_path, "out_segmented", "ckpt_smoke_results.nc")
    )

    assert segmented_idata.posterior.sizes["draw"] == 12
    for name in ("log10_γp", "log10_σp"):
        np.testing.assert_allclose(
            np.asarray(segmented_idata.posterior[name].values),
            np.asarray(single_idata.posterior[name].values),
            rtol=1e-10,
        )


def test_interrupted_run_leaves_usable_partial_output(tmp_path, staged_pulsar):
    """An interruption after the first checkpoint must leave draws on disk."""
    config_path = _write_config(
        tmp_path,
        staged_pulsar,
        "interrupted",
        num_samples=12,
        enabled="true",
        interval=4,
    )
    output_dir = str(tmp_path / "out_interrupted")

    # Fail partway through, after at least one checkpoint has been written.
    real_save = checkpointing.save_checkpoint
    calls = {"n": 0}

    def failing_save(*args, **kwargs):
        result = real_save(*args, **kwargs)
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("simulated walltime kill")
        return result

    with patch.object(checkpointing, "save_checkpoint", side_effect=failing_save):
        with pytest.raises(RuntimeError, match="walltime"):
            _run(config_path, output_dir)

    _, partial_path = checkpointing.checkpoint_paths(output_dir, "ckpt_smoke")
    assert os.path.exists(partial_path)

    partial = az.from_netcdf(partial_path)
    assert partial.posterior.sizes["draw"] == 8
    assert checkpointing.is_complete(partial) is False


def test_resume_completes_an_interrupted_run(tmp_path, staged_pulsar):
    """And resuming it must finish the same chain, matching the unsegmented result."""
    output_dir = str(tmp_path / "out_resume")

    interrupted = _write_config(
        tmp_path,
        staged_pulsar,
        "resume_first",
        num_samples=12,
        enabled="true",
        interval=4,
    )

    real_save = checkpointing.save_checkpoint
    calls = {"n": 0}

    def failing_save(*args, **kwargs):
        result = real_save(*args, **kwargs)
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("simulated walltime kill")
        return result

    with patch.object(checkpointing, "save_checkpoint", side_effect=failing_save):
        with pytest.raises(RuntimeError):
            _run(interrupted, output_dir)

    resumed = _write_config(
        tmp_path,
        staged_pulsar,
        "resume_second",
        num_samples=12,
        enabled="true",
        interval=4,
        resume="true",
    )
    _run(resumed, output_dir)

    final = az.from_netcdf(os.path.join(output_dir, "ckpt_smoke_results.nc"))
    assert final.posterior.sizes["draw"] == 12
    assert checkpointing.is_complete(final) is True

    single = _write_config(tmp_path, staged_pulsar, "resume_reference", num_samples=12)
    _run(single, str(tmp_path / "out_reference"))
    reference = az.from_netcdf(
        os.path.join(tmp_path, "out_reference", "ckpt_smoke_results.nc")
    )
    for name in ("log10_γp", "log10_σp"):
        np.testing.assert_allclose(
            np.asarray(final.posterior[name].values),
            np.asarray(reference.posterior[name].values),
            rtol=1e-10,
        )


def test_resume_against_a_changed_model_is_refused(tmp_path, staged_pulsar):
    """Task 3.2: a resume must not silently mix draws from two different posteriors."""
    output_dir = str(tmp_path / "out_mismatch")

    first = _write_config(
        tmp_path,
        staged_pulsar,
        "mismatch_first",
        num_samples=8,
        enabled="true",
        interval=4,
    )
    _run(first, output_dir)

    changed = _write_config(
        tmp_path,
        staged_pulsar,
        "mismatch_second",
        num_samples=8,
        enabled="true",
        interval=4,
        resume="true",
        log10_gamma_a_value=-7.5,  # a different model
    )
    with pytest.raises(ValueError, match="fingerprint"):
        _run(changed, output_dir)


def test_disabled_checkpointing_writes_no_checkpoint(tmp_path, staged_pulsar):
    """With checkpointing off the run must behave exactly as it did before."""
    config_path = _write_config(tmp_path, staged_pulsar, "plain", num_samples=8)
    output_dir = str(tmp_path / "out_plain")
    _run(config_path, output_dir)

    state_path, partial_path = checkpointing.checkpoint_paths(output_dir, "ckpt_smoke")
    assert not os.path.exists(state_path)
    assert not os.path.exists(partial_path)

    results = az.from_netcdf(os.path.join(output_dir, "ckpt_smoke_results.nc"))
    assert checkpointing.is_complete(results) is True


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _dummy_inference_data(draws):
    return az.from_dict(
        posterior={"theta": np.arange(draws, dtype=float).reshape(1, draws)}
    )


def test_checkpointing_overhead_is_bounded(tmp_path, staged_pulsar):
    """Task 3.5: the protection must not cost much of what it protects.

    Overhead is per *segment*, not per draw: each segment writes the accumulated
    posterior and the sampler state once. So the cost is set by how often you
    checkpoint, and the tolerance below is expressed that way. On a real array run a
    segment is hours, so a few seconds of writing is negligible; here the run is
    seconds, which makes this a deliberately pessimistic bound.
    """
    import time

    def timed(config_path, output_dir):
        start = time.perf_counter()
        _run(config_path, output_dir)
        return time.perf_counter() - start

    plain = _write_config(tmp_path, staged_pulsar, "overhead_plain", num_samples=24)
    checkpointed = _write_config(
        tmp_path,
        staged_pulsar,
        "overhead_ckpt",
        num_samples=24,
        enabled="true",
        interval=8,
    )

    baseline = timed(plain, str(tmp_path / "out_overhead_plain"))
    protected = timed(checkpointed, str(tmp_path / "out_overhead_ckpt"))

    per_segment = (protected - baseline) / 3.0
    # Measured at ~0.0 s per segment: the kernel object is reused across segments, so
    # JAX does not recompile, and the only real cost is writing the accumulated
    # posterior once per segment. The bound is left loose because that write scales
    # with the posterior's size, which is far larger on a 68-pulsar run than here.
    assert per_segment < 5.0, (
        f"checkpoint overhead {per_segment:.2f} s per segment "
        f"(baseline {baseline:.2f} s, checkpointed {protected:.2f} s)"
    )
    print(
        f"\ncheckpoint overhead: {per_segment:.2f} s per segment "
        f"(baseline {baseline:.2f} s, checkpointed {protected:.2f} s)"
    )

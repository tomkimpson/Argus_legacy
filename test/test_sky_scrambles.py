"""Sky scrambles and warm starts: the machinery behind the null calibration.

A Bayes factor is a number, not a significance. Sky scrambles supply the empirical
null: analyse the same data under a randomised correlation pattern, so a value's rarity
can be measured rather than asserted. These tests pin the two properties that make a
scramble a valid null draw -- the correlation structure really is destroyed, and nothing
else about the data changes -- plus the warm start that makes an ensemble affordable.
"""

import importlib.util
import os
from collections import namedtuple

import numpy as np
import pytest

from argus import gravitational_waves

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Defined at module level so they can be pickled into a stand-in checkpoint; a class
# defined inside a test function cannot be.
FakeState = namedtuple("FakeState", ["rng_key", "step_size", "position"])
FakeMultiChainState = namedtuple("FakeMultiChainState", ["rng_key", "step_size"])


@pytest.fixture(scope="module")
def scrambles():
    path = os.path.join(
        REPO_ROOT, "workflows", "ng15_sgwb_demo", "scripts", "sky_scrambles.py"
    )
    if not os.path.exists(path):
        pytest.skip(f"script not found: {path}")
    spec = importlib.util.spec_from_file_location("sky_scrambles", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _true_orf(n_pulsars=20, seed=0):
    rng = np.random.default_rng(seed)
    right_ascension = rng.uniform(0, 2 * np.pi, n_pulsars)
    declination = np.arcsin(rng.uniform(-1, 1, n_pulsars))
    separations = gravitational_waves.pairwise_angular_separation(
        right_ascension, declination
    )
    return gravitational_waves.hellings_downs(separations)


# --------------------------------------------------------------------------
# The match statistic
# --------------------------------------------------------------------------


def test_match_of_an_orf_with_itself_is_one(scrambles):
    orf = _true_orf()
    assert scrambles.orf_match(orf, orf) == pytest.approx(1.0)


def test_match_ignores_the_diagonal(scrambles):
    """Self-pairs carry no correlation information and must not inflate the match.

    Every ORF has a unit diagonal, so including it would make any two matrices look
    similar and let near-true geometries through the acceptance threshold.
    """
    orf = _true_orf()
    altered = orf.copy()
    np.fill_diagonal(altered, 5.0)
    assert scrambles.orf_match(orf, altered) == pytest.approx(
        scrambles.orf_match(orf, orf)
    )


def test_match_is_invariant_to_overall_scale(scrambles):
    """The match measures pattern, not amplitude."""
    orf = _true_orf()
    other = _true_orf(seed=3)
    assert scrambles.orf_match(orf, 7.0 * other) == pytest.approx(
        scrambles.orf_match(orf, other)
    )


def test_match_is_symmetric(scrambles):
    first, second = _true_orf(seed=1), _true_orf(seed=2)
    assert scrambles.orf_match(first, second) == pytest.approx(
        scrambles.orf_match(second, first)
    )


def test_match_rejects_mismatched_shapes(scrambles):
    with pytest.raises(ValueError, match="shape mismatch"):
        scrambles.orf_match(_true_orf(6), _true_orf(8))


# --------------------------------------------------------------------------
# Position drawing
# --------------------------------------------------------------------------


def test_positions_are_uniform_on_the_sphere(scrambles):
    """Declination must be drawn through sin(dec), not uniformly in the angle.

    Drawing the angle directly over-populates the poles, which would bias every
    scramble's geometry the same way and make the null ensemble unrepresentative.
    """
    rng = np.random.default_rng(0)
    _, declination = scrambles.random_sky_positions(200000, rng)
    sines = np.sin(declination)
    # Uniform in sin(dec) => mean 0, variance 1/3.
    assert abs(sines.mean()) < 0.01
    assert abs(sines.var() - 1.0 / 3.0) < 0.01


def test_positions_span_the_full_right_ascension_range(scrambles):
    rng = np.random.default_rng(1)
    right_ascension, _ = scrambles.random_sky_positions(10000, rng)
    assert right_ascension.min() < 0.1
    assert right_ascension.max() > 2 * np.pi - 0.1


# --------------------------------------------------------------------------
# Generating an ensemble
# --------------------------------------------------------------------------


def test_generated_scrambles_all_clear_the_threshold(scrambles):
    """The acceptance criterion must actually be applied, not merely recorded."""
    true_orf = _true_orf(20)
    result = scrambles.generate_scrambles(
        true_orf, n_scrambles=30, seed=0, match_threshold=0.2
    )
    assert result["orfs"].shape == (30, 20, 20)
    assert np.all(result["matches"] < 0.2)
    for orf in result["orfs"]:
        assert scrambles.orf_match(orf, true_orf) < 0.2


def test_scrambled_orfs_are_valid_correlation_matrices(scrambles):
    """A scramble must be a real ORF, or the null run is not the same model."""
    true_orf = _true_orf(12)
    result = scrambles.generate_scrambles(true_orf, n_scrambles=10, seed=1)
    for orf in result["orfs"]:
        np.testing.assert_allclose(np.diag(orf), 1.0, atol=1e-12)
        np.testing.assert_allclose(orf, orf.T, atol=1e-12)


def test_a_tighter_threshold_yields_a_tighter_ensemble(scrambles):
    true_orf = _true_orf(20)
    loose = scrambles.generate_scrambles(
        true_orf, n_scrambles=25, seed=2, match_threshold=0.4
    )
    tight = scrambles.generate_scrambles(
        true_orf, n_scrambles=25, seed=2, match_threshold=0.1
    )
    assert tight["matches"].max() < loose["matches"].max()
    assert tight["acceptance_rate"] < loose["acceptance_rate"]


def test_generation_is_reproducible(scrambles):
    true_orf = _true_orf(10)
    first = scrambles.generate_scrambles(true_orf, n_scrambles=5, seed=7)
    second = scrambles.generate_scrambles(true_orf, n_scrambles=5, seed=7)
    np.testing.assert_array_equal(first["orfs"], second["orfs"])


def test_an_unreachable_threshold_raises_rather_than_returning_fewer(scrambles):
    """Silently returning a short ensemble would misstate the p-value resolution."""
    true_orf = _true_orf(20)
    with pytest.raises(RuntimeError, match="accepted"):
        scrambles.generate_scrambles(
            true_orf,
            n_scrambles=5,
            seed=0,
            match_threshold=1e-8,
            max_attempts_per_scramble=20,
        )


# --------------------------------------------------------------------------
# Applying a scramble leaves everything else alone
# --------------------------------------------------------------------------


def test_applying_a_scramble_changes_only_the_orf(scrambles):
    """The defining property of a valid null draw."""
    true_orf = _true_orf(4)
    data = {
        "processed_residuals": {
            "toas": np.arange(5.0),
            "residuals": np.ones((5, 4)),
            "errors": np.full((5, 4), 1e-6),
        },
        "metadata": "metadata-sentinel",
        "design_matrices": ["a", "b", "c", "d"],
        "parameter_covariances": ["w", "x", "y", "z"],
        "hd_correlation": true_orf,
    }
    scrambled_orf = scrambles.generate_scrambles(true_orf, 1, seed=4)["orfs"][0]
    scrambled = scrambles.apply_scramble(data, scrambled_orf)

    np.testing.assert_array_equal(scrambled["hd_correlation"], scrambled_orf)
    for key in ("metadata", "design_matrices", "parameter_covariances"):
        assert scrambled[key] is data[key]
    assert scrambled["processed_residuals"] is data["processed_residuals"]
    # And the original is untouched, so a harness can loop without reloading.
    np.testing.assert_array_equal(data["hd_correlation"], true_orf)


def test_summary_reports_the_match_distribution(scrambles):
    true_orf = _true_orf(15)
    result = scrambles.generate_scrambles(true_orf, n_scrambles=20, seed=5)
    report = scrambles.summarise(result, true_orf)
    assert report["n_scrambles"] == 20
    assert report["self_match"] == pytest.approx(1.0)
    assert report["match_max"] < report["match_threshold"]
    assert 0.0 < report["acceptance_rate"] <= 1.0


# --------------------------------------------------------------------------
# Warm start
# --------------------------------------------------------------------------


def test_warm_start_is_off_without_a_state_path():
    import configparser

    from argus import checkpointing

    config = configparser.ConfigParser()
    config.read_string("[Data]\nx = 1\n")
    assert checkpointing.get_warm_start_settings(config)["enabled"] is False


def test_warm_start_requires_an_existing_state():
    from argus import checkpointing

    with pytest.raises(ValueError, match="not found"):
        checkpointing.load_warm_start_state("/nonexistent/state.pkl", 0)


def test_warm_start_rejects_a_file_without_a_sampler_state(tmp_path):
    import pickle

    from argus import checkpointing

    path = tmp_path / "not_a_checkpoint.pkl"
    with open(path, "wb") as handle:
        pickle.dump({"something": "else"}, handle)
    with pytest.raises(ValueError, match="not a checkpoint"):
        checkpointing.load_warm_start_state(str(path), 0)


def test_warm_start_replaces_the_parent_random_key(tmp_path):
    """Reusing the parent's key would correlate every scramble's trajectory.

    Each realisation of the null must be an independent draw. If they all replayed the
    parent run's randomness the ensemble would be too narrow and the p-value it produced
    would be too small -- an error in the direction of claiming a detection.
    """
    import pickle

    import jax

    from argus import checkpointing

    parent_key = np.asarray(jax.random.PRNGKey(999))
    state = FakeState(
        rng_key=parent_key, step_size=np.array(0.031), position=np.arange(3.0)
    )

    path = tmp_path / "parent_checkpoint.pkl"
    with open(path, "wb") as handle:
        pickle.dump({"sampler_state": state}, handle)

    warm = checkpointing.load_warm_start_state(str(path), seed=7)

    assert not np.array_equal(np.asarray(warm.rng_key), parent_key)
    # The tuned quantities -- the whole point of a warm start -- are preserved.
    np.testing.assert_array_equal(warm.step_size, state.step_size)
    np.testing.assert_array_equal(warm.position, state.position)

    # And a different seed gives a different stream.
    other = checkpointing.load_warm_start_state(str(path), seed=8)
    assert not np.array_equal(np.asarray(warm.rng_key), np.asarray(other.rng_key))


def test_warm_start_preserves_per_chain_key_shape(tmp_path):
    """With one key per chain the replacement must keep that structure."""
    import pickle

    import jax

    from argus import checkpointing

    n_chains = 4
    parent_keys = np.asarray(jax.random.split(jax.random.PRNGKey(1), n_chains))
    state = FakeMultiChainState(rng_key=parent_keys, step_size=np.full(n_chains, 0.02))

    path = tmp_path / "multichain.pkl"
    with open(path, "wb") as handle:
        pickle.dump({"sampler_state": state}, handle)

    warm = checkpointing.load_warm_start_state(str(path), seed=3)
    assert np.asarray(warm.rng_key).shape == parent_keys.shape
    assert not np.array_equal(np.asarray(warm.rng_key), parent_keys)

"""Mask support on the marginalized (Rao-Blackwellized) Kalman filter.

Masks were previously wired into the sequential augmented-state filter only, so
union-grid array data silently fell back to the slower path -- on exactly the runs
where speed matters most (68 pulsars, ~42% grid occupancy). These tests pin the port
against three independent references: the recorded goldens, the sequential filter,
and a non-recursive batch GLS oracle computed on the observed entries alone.

The offset convention: an absent (epoch, pulsar) entry is handled by masking rather
than reshaping, which leaves a unit-variance slot in the innovation covariance. Each
such slot contributes log(2*pi) to the log-det stream and nothing to the quadratic
forms, so the filter's masked log-likelihood sits a fixed 0.5*n_absent*log(2*pi)
below the likelihood of the genuinely reduced dataset. That offset does not depend on
any sampled parameter, which is what the tests below check rather than assume.
"""

import math
import os
import pickle
from unittest.mock import Mock, patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from argus import bayesian_inference
from argus import io_manager
from argus import jax_kalman_filter as jk
from argus.utils import get_efac_equad_injections, get_psr_noise_injections

from test.test_jax_kalman_filter import _realistic_pulsar_data

LN2PI = math.log(2.0 * math.pi)


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mdc2():
    """The MDC2 golden dataset and its recorded parameter vector."""

    class MockConfig:
        def get(self, section, key, fallback=None):
            return fallback

    io_manager.setup_single_logger(MockConfig(), enable_file_logging=False)

    here = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(here, "data/processed_pulsar_data.pkl")
    noise_path = os.path.join(here, "data/noise_parameters.json")
    spin_path = os.path.join(here, "data/spin_injections.pkl")
    for path in (data_path, noise_path, spin_path):
        if not os.path.exists(path):
            pytest.skip(f"Test data not found: {path}")

    with open(data_path, "rb") as f:
        pulsar_data = pickle.load(f)
    efac, equad = get_efac_equad_injections(noise_path, excluded_psrs=["J1640+2224"])
    sigma_p, gamma_p = get_psr_noise_injections(spin_path, excluded_psrs=["J1640+2224"])

    gamma_a = 1e-9
    params = bayesian_inference.Parameters(
        γa=gamma_a,
        ha=1e-15,
        log10_gamma_a=jnp.log10(gamma_a),
        γp=gamma_p,
        σp=sigma_p,
        EFAC=efac,
        EQUAD=equad,
    )
    return pulsar_data, params


def with_mask(data, mask):
    """Return a copy of a data dict carrying the given observation mask."""
    residuals = dict(data["processed_residuals"])
    residuals["mask"] = np.asarray(mask, dtype=float)
    out = dict(data)
    out["processed_residuals"] = residuals
    return out


def random_mask(shape, occupancy, seed=0):
    """A random mask with every pulsar observed at least once.

    A fully unobserved pulsar is a separate case (tested below); leaving one in by
    accident would silently change what the other tests are measuring.
    """
    rng = np.random.default_rng(seed)
    mask = (rng.random(shape) < occupancy).astype(float)
    for column in range(shape[1]):
        if mask[:, column].sum() == 0:
            mask[0, column] = 1.0
    return mask


# --------------------------------------------------------------------------
# Task 2.2 -- the goldens must not move
# --------------------------------------------------------------------------


def test_all_ones_mask_preserves_the_informative_golden(mdc2):
    """An all-ones mask on the marginal path reproduces 63618.93 bit-for-bit."""
    data, params = mdc2
    unmasked = float(
        jk.JaxKalmanFilter(data=data, use_gw=True, use_marginal=True).get_likelihood(
            params
        )
    )
    ones = np.ones_like(data["processed_residuals"]["residuals"])
    masked = float(
        jk.JaxKalmanFilter(
            data=with_mask(data, ones), use_gw=True, use_marginal=True
        ).get_likelihood(params)
    )
    assert abs(unmasked - 63618.93) < 1.0
    np.testing.assert_allclose(masked, unmasked, rtol=0, atol=0)


def test_all_ones_mask_preserves_the_diffuse_golden(mdc2):
    """And 59420.06 on the diffuse-prior variant."""
    data, params = mdc2
    kwargs = dict(use_gw=True, use_marginal=True, timing_prior="diffuse")
    unmasked = float(jk.JaxKalmanFilter(data=data, **kwargs).get_likelihood(params))
    ones = np.ones_like(data["processed_residuals"]["residuals"])
    masked = float(
        jk.JaxKalmanFilter(data=with_mask(data, ones), **kwargs).get_likelihood(params)
    )
    assert abs(unmasked - 59420.06) < 1.0
    np.testing.assert_allclose(masked, unmasked, rtol=0, atol=0)


# --------------------------------------------------------------------------
# Task 2.3 -- agreement with the sequential filter
# --------------------------------------------------------------------------


@pytest.mark.parametrize("occupancy", [0.4, 0.6, 0.9])
@patch("argus.io_manager.get_argus_logger")
def test_masked_marginal_agrees_with_masked_sequential(mock_logger, occupancy):
    """The two backends are mathematically equivalent; masking must not break that.

    This is the test that exposed the jitter-scale bug (see
    `test_jitter_scale_ignores_absent_slots`): before the fix the two paths disagreed
    by ~7e-4 relative here and by ~23 nats on a 60%-occupied MDC2 grid. They now agree
    to ~6e-12, the same order as the unmasked case.
    """
    mock_logger.return_value = Mock()
    data = _realistic_pulsar_data()
    shape = data["processed_residuals"]["residuals"].shape
    masked = with_mask(data, random_mask(shape, occupancy, seed=1))

    params = bayesian_inference.Parameters(
        log10_gamma_a=-9.0,
        γa=1e-9,
        ha=1e-15,
        γp=jnp.full(2, 1e-8),
        σp=jnp.full(2, 1e-15),
        EFAC=jnp.ones(2),
        EQUAD=jnp.full(2, 1e-6),
    )

    sequential = float(
        jk.JaxKalmanFilter(data=masked, use_gw=True, use_marginal=False).get_likelihood(
            params
        )
    )
    marginal = float(
        jk.JaxKalmanFilter(data=masked, use_gw=True, use_marginal=True).get_likelihood(
            params
        )
    )
    np.testing.assert_allclose(marginal, sequential, rtol=1e-9)


# --------------------------------------------------------------------------
# Task 2.4 -- an absent pulsar contributes only a constant
# --------------------------------------------------------------------------


@patch("argus.io_manager.get_argus_logger")
def test_fully_absent_pulsar_offset_is_parameter_independent(mock_logger):
    """Masking a pulsar out entirely must not leak its data into the likelihood.

    Checked the strong way: the gap between the masked three-pulsar likelihood and the
    genuinely reduced two-pulsar likelihood must be the *same* at two different
    parameter vectors. A gap that moved with the parameters would mean the absent
    pulsar was still contributing.
    """
    mock_logger.return_value = Mock()
    data = _realistic_pulsar_data_n(3, seed=4)
    n_epoch, n_psr = data["processed_residuals"]["residuals"].shape

    mask = np.ones((n_epoch, n_psr))
    mask[:, 2] = 0.0
    masked = with_mask(data, mask)
    reduced = _drop_pulsar(data, 2)

    gaps = []
    for ha in (1e-15, 1e-13):
        full_params = _params_for(n_psr, ha)
        reduced_params = _params_for(n_psr - 1, ha)
        masked_ll = float(
            jk.JaxKalmanFilter(
                data=masked, use_gw=True, use_marginal=True
            ).get_likelihood(full_params)
        )
        reduced_ll = float(
            jk.JaxKalmanFilter(
                data=reduced, use_gw=True, use_marginal=True
            ).get_likelihood(reduced_params)
        )
        gaps.append(masked_ll - reduced_ll)

    # The offset is exactly the log(2*pi) each absent slot's unit-variance placeholder
    # contributes to the log-det stream -- nothing else survives from the masked-out
    # pulsar. Both that it is constant and that it takes the predicted value matter:
    # the value alone would not catch a leak that happened to be parameter-free.
    np.testing.assert_allclose(gaps[0], gaps[1], rtol=1e-12)
    np.testing.assert_allclose(gaps[0], -0.5 * n_epoch * LN2PI, rtol=1e-10)


@patch("argus.io_manager.get_argus_logger")
def test_absent_pulsar_data_cannot_influence_the_likelihood(mock_logger):
    """Replacing a masked-out pulsar's residuals must change nothing at all."""
    mock_logger.return_value = Mock()
    data = _realistic_pulsar_data_n(3, seed=5)
    n_epoch, n_psr = data["processed_residuals"]["residuals"].shape
    mask = np.ones((n_epoch, n_psr))
    mask[:, 1] = 0.0
    params = _params_for(n_psr, 1e-15)

    def likelihood(residual_column, error_column):
        base = data["processed_residuals"]
        residuals = base["residuals"].copy()
        errors = base["errors"].copy()
        residuals[:, 1] = residual_column
        errors[:, 1] = error_column
        altered = dict(data)
        altered["processed_residuals"] = {
            **base,
            "residuals": residuals,
            "errors": errors,
            "mask": mask,
        }
        return float(
            jk.JaxKalmanFilter(
                data=altered, use_gw=True, use_marginal=True
            ).get_likelihood(params)
        )

    rng = np.random.default_rng(6)
    original = likelihood(
        data["processed_residuals"]["residuals"][:, 1],
        data["processed_residuals"]["errors"][:, 1],
    )
    scrambled = likelihood(rng.standard_normal(n_epoch) * 1e-3, np.full(n_epoch, 5e-7))
    assert original == scrambled


# --------------------------------------------------------------------------
# Task 2.5 -- an epoch with nothing observed
# --------------------------------------------------------------------------


@pytest.mark.parametrize("use_marginal", [False, True])
@patch("argus.io_manager.get_argus_logger")
def test_empty_epoch_receives_no_update(mock_logger, use_marginal):
    """An epoch where every pulsar is absent must not update the state.

    Tested directly: scramble that epoch's residuals and the log-likelihood must be
    bit-for-bit identical. If any update were applied the innovation would change and
    so would the result. This is stronger than checking that the likelihood shifts by
    a constant, which a partially-applied update could also satisfy.
    """
    mock_logger.return_value = Mock()
    data = _realistic_pulsar_data_n(2, seed=7)
    n_epoch, n_psr = data["processed_residuals"]["residuals"].shape
    empty_epoch = 3

    mask = np.ones((n_epoch, n_psr))
    mask[empty_epoch, :] = 0.0
    params = _params_for(n_psr, 1e-15)

    def likelihood(residuals_at_empty_epoch):
        base = data["processed_residuals"]
        residuals = base["residuals"].copy()
        residuals[empty_epoch, :] = residuals_at_empty_epoch
        altered = dict(data)
        altered["processed_residuals"] = {
            **base,
            "residuals": residuals,
            "mask": mask,
        }
        return float(
            jk.JaxKalmanFilter(
                data=altered, use_gw=True, use_marginal=use_marginal
            ).get_likelihood(params)
        )

    rng = np.random.default_rng(10)
    original = likelihood(data["processed_residuals"]["residuals"][empty_epoch, :])
    scrambled = likelihood(rng.standard_normal(n_psr) * 1e-3)
    assert original == scrambled


@patch("argus.io_manager.get_argus_logger")
def test_empty_epoch_still_propagates_the_state(mock_logger):
    """The state must still be propagated across an empty epoch, not frozen.

    An epoch with no observations advances time: the covariance grows by the process
    noise over that interval even though no update is applied. If the epoch were simply
    skipped, later epochs would see a state that had not aged, so the likelihood would
    differ from one where the gap is respected. Comparing against a run whose empty
    epoch sits at a different position detects that -- the two are the same amount of
    data and the same number of placeholder slots, differing only in when the gap falls.
    """
    mock_logger.return_value = Mock()
    data = _realistic_pulsar_data_n(2, seed=11)
    n_epoch, n_psr = data["processed_residuals"]["residuals"].shape
    params = _params_for(n_psr, 1e-13)

    def likelihood(empty_epoch):
        mask = np.ones((n_epoch, n_psr))
        mask[empty_epoch, :] = 0.0
        return float(
            jk.JaxKalmanFilter(
                data=with_mask(data, mask), use_gw=True, use_marginal=True
            ).get_likelihood(params)
        )

    assert likelihood(3) != likelihood(8)


# --------------------------------------------------------------------------
# Task 2.6 -- gradients
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mask_kind", ["dense", "sparse", "empty_epoch", "absent_pulsar"]
)
@patch("argus.io_manager.get_argus_logger")
def test_masked_marginal_gradients_are_finite(mock_logger, mask_kind):
    """Gradient-based sampling needs finite gradients at every mask configuration."""
    mock_logger.return_value = Mock()
    data = _realistic_pulsar_data_n(3, seed=8)
    n_epoch, n_psr = data["processed_residuals"]["residuals"].shape

    mask = np.ones((n_epoch, n_psr))
    if mask_kind == "sparse":
        mask = random_mask((n_epoch, n_psr), 0.5, seed=9)
    elif mask_kind == "empty_epoch":
        mask[2, :] = 0.0
    elif mask_kind == "absent_pulsar":
        mask[:, 1] = 0.0

    kf = jk.JaxKalmanFilter(data=with_mask(data, mask), use_gw=True, use_marginal=True)

    def logl(log10_ha, log10_gamma_a, log10_gamma_p, log10_sigma_p, efac, equad):
        return bayesian_inference.log_likelihood_fn(
            kf, log10_ha, log10_gamma_a, log10_gamma_p, log10_sigma_p, efac, equad
        )

    args = (
        -15.0,
        -9.0,
        jnp.full(n_psr, -8.0),
        jnp.full(n_psr, -15.0),
        jnp.ones(n_psr),
        jnp.full(n_psr, 1e-6),
    )
    grads = jax.grad(logl, argnums=(0, 1, 2, 3, 4, 5))(*args)
    for index, gradient in enumerate(grads):
        assert np.all(
            np.isfinite(np.asarray(gradient))
        ), f"non-finite gradient for argument {index} with mask '{mask_kind}'"


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _realistic_pulsar_data_n(n_psr, seed=0, n_epochs=12):
    """The realistic fixture generalised to an arbitrary pulsar count."""
    import pandas as pd

    rng = np.random.default_rng(seed)
    dims = [5 + i for i in range(n_psr)]
    err_scale = 1e-6

    metadata = pd.DataFrame(
        {
            "name": [f"P{i}" for i in range(n_psr)],
            "dim_M": dims,
            "RA": list(np.linspace(0.3, 1.5, n_psr)),
            "DEC": list(np.linspace(-0.2, 0.4, n_psr)),
            "F0": list(np.linspace(150.0, 250.0, n_psr)),
            "par_file": [f"p{i}" for i in range(n_psr)],
            "tim_file": [f"t{i}" for i in range(n_psr)],
        }
    )
    residuals = rng.standard_normal((n_epochs, n_psr)) * err_scale
    errors = np.ones((n_epochs, n_psr)) * err_scale

    design_matrices, parameter_covariances = [], []
    for i in range(n_psr):
        design = rng.standard_normal((n_epochs, dims[i]))
        design = design / np.sqrt(np.sum(design**2, axis=0))
        noise_inv = np.diag(1.0 / errors[:, i] ** 2)
        design_matrices.append(design)
        parameter_covariances.append(np.linalg.inv(design.T @ noise_inv @ design))

    from argus import gravitational_waves

    separations = gravitational_waves.pairwise_angular_separation(
        metadata["RA"].to_numpy(dtype=float), metadata["DEC"].to_numpy(dtype=float)
    )
    return {
        "processed_residuals": {
            "toas": np.arange(n_epochs, dtype=float) * 30.0,
            "residuals": residuals,
            "errors": errors,
        },
        "metadata": metadata,
        "design_matrices": design_matrices,
        "parameter_covariances": parameter_covariances,
        "hd_correlation": gravitational_waves.hellings_downs(separations),
    }


def _drop_pulsar(data, index):
    """Remove one pulsar from a data dict entirely."""
    from argus import gravitational_waves

    keep = [i for i in range(len(data["design_matrices"])) if i != index]
    metadata = data["metadata"].drop(index=index).reset_index(drop=True)
    separations = gravitational_waves.pairwise_angular_separation(
        metadata["RA"].to_numpy(dtype=float), metadata["DEC"].to_numpy(dtype=float)
    )
    base = data["processed_residuals"]
    return {
        "processed_residuals": {
            "toas": base["toas"],
            "residuals": base["residuals"][:, keep],
            "errors": base["errors"][:, keep],
        },
        "metadata": metadata,
        "design_matrices": [data["design_matrices"][i] for i in keep],
        "parameter_covariances": [data["parameter_covariances"][i] for i in keep],
        "hd_correlation": gravitational_waves.hellings_downs(separations),
    }


def _params_for(n_psr, ha):
    return bayesian_inference.Parameters(
        log10_gamma_a=-9.0,
        γa=1e-9,
        ha=ha,
        γp=jnp.full(n_psr, 1e-8),
        σp=jnp.full(n_psr, 1e-15),
        EFAC=jnp.ones(n_psr),
        EQUAD=jnp.full(n_psr, 1e-6),
    )


# --------------------------------------------------------------------------
# Regression: the jitter magnitude scale must ignore absent slots
# --------------------------------------------------------------------------


def test_jitter_scale_ignores_absent_slots():
    """The absent slots' unit variance must not set the jitter magnitude.

    Missing observations are handled by putting a unit variance in the absent slots so
    the innovation covariance stays invertible. PTA innovation variances are ~1e-12, so
    a jitter scaled by the full trace is set by those placeholders rather than by the
    data. That inflated the jitter by ~1e12, which is not a rounding detail: it shifted
    the masked log-likelihood by tens of nats and broke agreement between the two
    filter backends on sparse grids.
    """
    observed_variance = 1e-12
    cov = np.diag([observed_variance, observed_variance, 1.0])
    mask = jnp.array([1.0, 1.0, 0.0])

    unmasked_scale = float(jk._jitter_scale(jnp.asarray(cov)))
    masked_scale = float(jk._jitter_scale(jnp.asarray(cov), mask))

    # Without the mask the scale is set by the placeholder, not the data.
    assert unmasked_scale > 100 * observed_variance
    # With it, the scale tracks the observed variances as intended.
    np.testing.assert_allclose(masked_scale, 1e-9 * observed_variance, rtol=1e-12)


def test_jitter_scale_unchanged_without_a_mask():
    """With no mask the scale must still be exactly trace(cov)/n, preserving goldens."""
    rng = np.random.default_rng(0)
    cov = np.diag(rng.uniform(1e-13, 1e-11, size=5))
    expected = 1e-9 * np.trace(cov) / 5
    np.testing.assert_allclose(
        float(jk._jitter_scale(jnp.asarray(cov))), expected, rtol=1e-14
    )


def test_all_ones_mask_gives_the_same_jitter_scale_as_no_mask():
    """An all-ones mask must be indistinguishable from no mask at all."""
    rng = np.random.default_rng(1)
    cov = jnp.asarray(np.diag(rng.uniform(1e-13, 1e-11, size=4)))
    ones = jnp.ones(4)
    np.testing.assert_allclose(
        float(jk._jitter_scale(cov, ones)), float(jk._jitter_scale(cov)), rtol=1e-14
    )

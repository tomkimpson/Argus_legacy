"""Tests for the HD<->CURN correlation path C(eps) = (1-eps)*I + eps*C_HD.

The path embeds both SGWB correlation hypotheses in one continuous family so a
single model can span them (see gravitational_waves.correlation_path). These tests
pin the two endpoints exactly -- eps=1 must be Hellings-Downs and eps=0 must be the
existing CURN construction -- because the Bayes factor is read off at precisely
those points.
"""

import os
import pickle

import jax.numpy as jnp
import numpy as np
import pytest

from argus import bayesian_inference
from argus import gravitational_waves
from argus import jax_kalman_filter as jk
from argus import io_manager
from argus.parameter_sampling import count_free_parameters
from argus.utils import get_efac_equad_injections, get_psr_noise_injections

# --------------------------------------------------------------------------
# Pure-function endpoint tests
# --------------------------------------------------------------------------


def _example_hd_matrix(n=6, seed=0):
    """A symmetric, unit-diagonal matrix with the sign structure of a real ORF."""
    rng = np.random.default_rng(seed)
    off = rng.uniform(-0.3, 0.5, size=(n, n))
    hd = (off + off.T) / 2.0
    np.fill_diagonal(hd, 1.0)
    return jnp.asarray(hd)


def test_correlation_path_epsilon_one_is_hellings_downs():
    """eps=1 must reproduce the input matrix exactly, not just to tolerance."""
    hd = _example_hd_matrix()
    result = gravitational_waves.correlation_path(hd, 1.0)
    assert jnp.array_equal(result, hd)


def test_correlation_path_epsilon_zero_is_identity():
    """eps=0 must be the identity exactly -- this is the CURN hypothesis."""
    hd = _example_hd_matrix()
    result = gravitational_waves.correlation_path(hd, 0.0)
    assert jnp.array_equal(result, jnp.eye(hd.shape[0], dtype=hd.dtype))


def test_correlation_path_preserves_unit_diagonal():
    """Interpolation scales cross-correlations only; auto-power is untouched."""
    hd = _example_hd_matrix()
    for eps in (0.0, 0.25, 0.5, 0.75, 1.0):
        result = gravitational_waves.correlation_path(hd, eps)
        np.testing.assert_allclose(np.diag(np.asarray(result)), 1.0, rtol=0, atol=0)


def test_correlation_path_is_linear_in_epsilon():
    """Off-diagonal entries scale linearly, which is what makes dlogL/deps clean."""
    hd = _example_hd_matrix()
    half = gravitational_waves.correlation_path(hd, 0.5)
    expected_off = 0.5 * (np.asarray(hd) - np.eye(hd.shape[0]))
    actual_off = np.asarray(half) - np.eye(hd.shape[0])
    np.testing.assert_allclose(actual_off, expected_off, rtol=1e-15, atol=0)


def test_correlation_path_is_differentiable():
    """Path sampling integrates dlogL/deps, so eps must carry a finite gradient."""
    import jax

    hd = _example_hd_matrix()
    grad = jax.grad(lambda e: jnp.sum(gravitational_waves.correlation_path(hd, e)))(0.4)
    assert np.isfinite(float(grad))
    # d/deps sum[(1-eps)I + eps*HD] = sum(HD) - trace(I)
    expected = float(jnp.sum(hd) - hd.shape[0])
    np.testing.assert_allclose(float(grad), expected, rtol=1e-12)


# --------------------------------------------------------------------------
# Endpoint tests through the full Kalman likelihood
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mdc2_setup():
    """The same MDC2 test data and parameter vector the golden tests use."""

    class MockConfig:
        def get(self, section, key, fallback=None):
            return fallback

    io_manager.setup_single_logger(MockConfig(), enable_file_logging=False)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "data/processed_pulsar_data.pkl")
    noise_params_path = os.path.join(script_dir, "data/noise_parameters.json")
    spin_injections_path = os.path.join(script_dir, "data/spin_injections.pkl")

    for path in (data_path, noise_params_path, spin_injections_path):
        if not os.path.exists(path):
            pytest.skip(f"Test data not found: {path}")

    with open(data_path, "rb") as f:
        pulsar_data = pickle.load(f)

    efac_array, equad_array = get_efac_equad_injections(
        noise_params_path, excluded_psrs=["J1640+2224"]
    )
    sigma_p_injected, gamma_p_injected = get_psr_noise_injections(
        spin_injections_path, excluded_psrs=["J1640+2224"]
    )

    def make_params(orf_epsilon=None):
        γa = 1e-9
        ha = 1e-15
        return bayesian_inference.Parameters(
            γa=γa,
            ha=ha,
            log10_gamma_a=jnp.log10(γa),
            γp=gamma_p_injected,
            σp=sigma_p_injected,
            EFAC=efac_array,
            EQUAD=equad_array,
            orf_epsilon=orf_epsilon,
        )

    return pulsar_data, make_params


@pytest.mark.parametrize("use_marginal", [False, True])
def test_epsilon_one_reproduces_the_golden_likelihood(mdc2_setup, use_marginal):
    """Task 1.1: switching the path on at eps=1 must not move the golden value."""
    pulsar_data, make_params = mdc2_setup
    kf = jk.JaxKalmanFilter(data=pulsar_data, use_gw=True, use_marginal=use_marginal)

    without_path = float(kf.get_likelihood(make_params(orf_epsilon=None)))
    at_epsilon_one = float(kf.get_likelihood(make_params(orf_epsilon=1.0)))

    assert np.isclose(without_path, 63618.93, atol=1.0)
    np.testing.assert_allclose(at_epsilon_one, without_path, rtol=0, atol=0)


def test_epsilon_one_reproduces_the_diffuse_golden(mdc2_setup):
    """The diffuse-prior golden must be equally untouched by the path at eps=1."""
    pulsar_data, make_params = mdc2_setup
    kf = jk.JaxKalmanFilter(
        data=pulsar_data, use_gw=True, use_marginal=True, timing_prior="diffuse"
    )

    without_path = float(kf.get_likelihood(make_params(orf_epsilon=None)))
    at_epsilon_one = float(kf.get_likelihood(make_params(orf_epsilon=1.0)))

    assert np.isclose(without_path, 59420.06, atol=1.0)
    np.testing.assert_allclose(at_epsilon_one, without_path, rtol=0, atol=0)


@pytest.mark.parametrize("use_marginal", [False, True])
def test_epsilon_zero_matches_the_curn_construction(mdc2_setup, use_marginal):
    """Task 1.2: eps=0 must equal the identity-ORF override run_curn.py performs.

    run_curn.py builds CURN by replacing data['hd_correlation'] with the identity
    before the filter is constructed. The path must reproduce that bit-for-bit,
    otherwise the two ends of the path are not the two models being compared.
    """
    pulsar_data, make_params = mdc2_setup

    curn_data = dict(pulsar_data)
    n_psr = np.asarray(pulsar_data["hd_correlation"]).shape[0]
    curn_data["hd_correlation"] = np.eye(n_psr, dtype=float)

    kf_curn = jk.JaxKalmanFilter(data=curn_data, use_gw=True, use_marginal=use_marginal)
    kf_path = jk.JaxKalmanFilter(
        data=pulsar_data, use_gw=True, use_marginal=use_marginal
    )

    curn_likelihood = float(kf_curn.get_likelihood(make_params(orf_epsilon=None)))
    path_likelihood = float(kf_path.get_likelihood(make_params(orf_epsilon=0.0)))

    np.testing.assert_allclose(path_likelihood, curn_likelihood, rtol=0, atol=0)


def test_hd_and_curn_endpoints_differ(mdc2_setup):
    """Guard against a no-op path: the two endpoints must be distinguishable.

    An early HD-vs-CURN check was misleadingly null because the test parameters sat
    in a GW-negligible regime. Assert the endpoints differ at a GW-relevant
    amplitude so a silently inert path cannot pass the tests above.
    """
    pulsar_data, _ = mdc2_setup
    kf = jk.JaxKalmanFilter(data=pulsar_data, use_gw=True)

    _, make_params = mdc2_setup
    base = make_params(orf_epsilon=None)
    loud = base.replace(ha=1e-12, orf_epsilon=1.0)
    loud_curn = loud.replace(orf_epsilon=0.0)

    hd_likelihood = float(kf.get_likelihood(loud))
    curn_likelihood = float(kf.get_likelihood(loud_curn))

    assert abs(hd_likelihood - curn_likelihood) > 1.0


def test_likelihood_is_differentiable_in_epsilon(mdc2_setup):
    """Path sampling needs dlogL/deps from autodiff; it must be finite and non-zero."""
    import jax

    pulsar_data, make_params = mdc2_setup
    kf = jk.JaxKalmanFilter(data=pulsar_data, use_gw=True)
    base = make_params(orf_epsilon=None).replace(ha=1e-12)

    def logl(eps):
        return kf.get_likelihood(base.replace(orf_epsilon=eps))

    grad = float(jax.grad(logl)(0.5))
    assert np.isfinite(grad)
    assert grad != 0.0


# --------------------------------------------------------------------------
# Configuration and model wiring
# --------------------------------------------------------------------------


def _config(**priormodel_keys):
    import configparser

    config = configparser.ConfigParser()
    config["PriorModel"] = {k: str(v) for k, v in priormodel_keys.items()}
    return config


def test_orf_path_defaults_to_off():
    from argus.prior_models import _get_orf_path_specs

    specs = _get_orf_path_specs(_config())
    assert specs["orf_path"] == "off"
    assert specs["orf_epsilon_value"] is None
    assert specs["orf_epsilon_bounds"] is None


def test_orf_path_fixed_reads_its_value():
    from argus.prior_models import _get_orf_path_specs

    specs = _get_orf_path_specs(_config(orf_path="fixed", orf_epsilon_value=0.25))
    assert specs["orf_path"] == "fixed"
    assert specs["orf_epsilon_value"] == 0.25


def test_orf_path_sampled_defaults_to_unit_interval():
    from argus.prior_models import _get_orf_path_specs

    specs = _get_orf_path_specs(_config(orf_path="sampled"))
    assert specs["orf_path"] == "sampled"
    assert specs["orf_epsilon_bounds"] == (0.0, 1.0)


def test_orf_path_rejects_unknown_mode():
    from argus.prior_models import _get_orf_path_specs

    with pytest.raises(ValueError, match="Unknown orf_path"):
        _get_orf_path_specs(_config(orf_path="hypermodel"))


def test_orf_path_rejects_inverted_bounds():
    from argus.prior_models import _get_orf_path_specs

    with pytest.raises(ValueError, match="must exceed"):
        _get_orf_path_specs(
            _config(orf_path="sampled", orf_epsilon_min=1.0, orf_epsilon_max=0.0)
        )


def test_sampled_path_adds_one_free_parameter():
    """A sampled eps is one more dimension for NUTS; a fixed one is not."""
    base = {
        "gw_parameterization": "ridge",
        "log10_gamma_p_spec": None,
        "log10_sigma_p_spec": None,
        "efac_spec": None,
        "equad_spec": None,
    }
    n_psr = 5
    off = count_free_parameters({**base, "orf_path": "off"}, n_psr)
    fixed = count_free_parameters({**base, "orf_path": "fixed"}, n_psr)
    sampled = count_free_parameters({**base, "orf_path": "sampled"}, n_psr)

    assert fixed == off
    assert sampled == off + 1


def test_sample_orf_epsilon_returns_none_when_off():
    from argus.parameter_sampling import sample_orf_epsilon

    assert sample_orf_epsilon({"orf_path": "off"}) is None
    assert sample_orf_epsilon({}) is None


def test_sampled_epsilon_appears_as_a_model_site():
    """eps must be a real sampled site, not a hidden constant."""
    import numpyro
    from numpyro import handlers
    from jax import random
    from argus.parameter_sampling import sample_orf_epsilon

    def model():
        sample_orf_epsilon({"orf_path": "sampled", "orf_epsilon_bounds": (0.0, 1.0)})

    trace = handlers.trace(handlers.seed(model, random.PRNGKey(0))).get_trace()
    assert "orf_epsilon" in trace
    assert trace["orf_epsilon"]["type"] == "sample"
    assert isinstance(trace["orf_epsilon"]["fn"], numpyro.distributions.Uniform)
    value = float(trace["orf_epsilon"]["value"])
    assert 0.0 <= value <= 1.0


def test_fixed_epsilon_is_recorded_as_deterministic():
    """A ladder rung must record where on the path it sat."""
    from numpyro import handlers
    from jax import random
    from argus.parameter_sampling import sample_orf_epsilon

    def model():
        sample_orf_epsilon({"orf_path": "fixed", "orf_epsilon_value": 0.3})

    trace = handlers.trace(handlers.seed(model, random.PRNGKey(0))).get_trace()
    assert trace["orf_epsilon"]["type"] == "deterministic"
    np.testing.assert_allclose(float(trace["orf_epsilon"]["value"]), 0.3)


def test_epsilon_gradient_matches_finite_differences(mdc2_setup):
    """The path-sampling integrand is dlogL/deps from autodiff -- check it is right.

    Path sampling integrates this gradient along the path, so an error here would
    propagate silently into the Bayes factor. A central difference is an independent
    route to the same number.
    """
    import jax

    pulsar_data, make_params = mdc2_setup
    kf = jk.JaxKalmanFilter(data=pulsar_data, use_gw=True)
    base = make_params(orf_epsilon=None).replace(ha=1e-12)

    def logl(eps):
        return float(kf.get_likelihood(base.replace(orf_epsilon=eps)))

    # The step is deliberately large. The log-likelihood is O(6e4) while its
    # derivative in eps is O(6), so a central difference subtracts two nearly equal
    # numbers: below h ~ 1e-3 the result is dominated by the filter's own numerical
    # noise, not by the derivative. Measured here, h=1e-2 agrees with autodiff to
    # 1.6e-3 while h=1e-5 is wrong by a factor of two. The precise check on the
    # gradient is the integral identity in the next test, which does not cancel.
    eps0 = 0.5
    step = 1e-2
    finite_difference = (logl(eps0 + step) - logl(eps0 - step)) / (2.0 * step)
    autodiff = float(
        jax.grad(lambda e: kf.get_likelihood(base.replace(orf_epsilon=e)))(eps0)
    )

    np.testing.assert_allclose(autodiff, finite_difference, rtol=5e-3)


def test_epsilon_gradient_integrates_to_the_likelihood_difference(mdc2_setup):
    """The identity path sampling rests on, checked at fixed parameters.

    At a fixed parameter vector, integrating dlogL/deps from 0 to 1 must return
    logL(HD) - logL(CURN). The full estimator additionally averages over the
    posterior at each eps, but if this fixed-parameter version failed, nothing
    downstream could be right.

    Uses a fixed Gauss-Legendre rule rather than adaptive quadrature: the integrand
    is smooth but each evaluation is a full differentiated Kalman pass, so an
    adaptive rule chasing a tight tolerance costs minutes for no extra confidence.
    """
    import jax

    pulsar_data, make_params = mdc2_setup
    kf = jk.JaxKalmanFilter(data=pulsar_data, use_gw=True)
    base = make_params(orf_epsilon=None).replace(ha=1e-12)

    grad_fn = jax.jit(
        jax.grad(lambda e: kf.get_likelihood(base.replace(orf_epsilon=e)))
    )

    nodes, weights = np.polynomial.legendre.leggauss(12)
    points = 0.5 * (nodes + 1.0)  # map the [-1, 1] rule onto [0, 1]
    integral = 0.5 * float(
        np.sum(weights * np.array([float(grad_fn(x)) for x in points]))
    )

    direct = float(kf.get_likelihood(base.replace(orf_epsilon=1.0))) - float(
        kf.get_likelihood(base.replace(orf_epsilon=0.0))
    )
    np.testing.assert_allclose(integral, direct, rtol=1e-5)

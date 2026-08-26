"""Validation of the HD-vs-CURN Bayes-factor estimators against analytic answers.

Both estimators are validated on the same synthetic problem, which has a closed
form: a posterior on the correlation-path coordinate eps proportional to
exp(k*eps) on [0, 1], under a uniform prior. Then

    ln B(HD/CURN) = ln p(eps=1|d) - ln p(eps=0|d) = k

exactly, for any k. Varying k walks the estimators from the easy regime (k ~ 0,
both endpoints well populated) into the regime that breaks boundary density
estimation (large k, no draws anywhere near eps=0) -- which is the regime the real
68-D problem is expected to sit in, and the reason the reliability gate exists.
"""

import importlib.util
import math
import os

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT_DIR = os.path.join(REPO_ROOT, "workflows", "ng15_sgwb_demo", "scripts")


def _load_script(name):
    path = os.path.join(SCRIPT_DIR, f"{name}.py")
    if not os.path.exists(path):
        pytest.skip(f"script not found: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def sd():
    return _load_script("lnb_savage_dickey")


def draw_tilted_uniform(k, n_chain=4, n_draw=4000, seed=0):
    """Exact i.i.d. draws from p(eps) ∝ exp(k*eps) on [0, 1] by inverse CDF."""
    rng = np.random.default_rng(seed)
    u = rng.uniform(size=(n_chain, n_draw))
    if abs(k) < 1e-12:
        return u
    return np.log1p(u * (math.expm1(k))) / k


# ---------------------------------------------------------------------------
# Density estimators at the boundary
# ---------------------------------------------------------------------------


def test_reflection_kde_recovers_a_flat_density(sd):
    """A uniform sample has density 1 everywhere, endpoints included."""
    rng = np.random.default_rng(1)
    u = rng.uniform(size=20000)
    for x in (0.0, 0.5, 1.0):
        value = math.exp(sd.reflection_kde_logpdf(u, x, 0.05))
        assert abs(value - 1.0) < 0.1, f"at {x}: {value}"


def test_beta_kde_recovers_a_flat_density(sd):
    rng = np.random.default_rng(2)
    u = rng.uniform(size=20000)
    for x in (0.0, 0.5, 1.0):
        value = math.exp(sd.beta_kde_logpdf(u, x, 0.01))
        assert abs(value - 1.0) < 0.1, f"at {x}: {value}"


def test_uncorrected_kde_would_halve_the_boundary_density(sd):
    """Motivates the boundary correction: without it the answer is wrong by ln 2 per end.

    A plain Gaussian KDE loses the mass that reflection puts back, so it reports
    about half the true density at an endpoint. Both endpoints are used, so the
    errors would partly cancel -- but only for a symmetric posterior, which is
    exactly what the real problem is not.
    """
    rng = np.random.default_rng(3)
    u = rng.uniform(size=20000)
    bandwidth = 0.05
    n = u.size
    z = (0.0 - u) / bandwidth
    plain = np.sum(np.exp(-0.5 * z**2) / math.sqrt(2 * math.pi)) / (n * bandwidth)
    assert 0.4 < plain < 0.6
    corrected = math.exp(sd.reflection_kde_logpdf(u, 0.0, bandwidth))
    assert abs(corrected - 1.0) < 0.1


# ---------------------------------------------------------------------------
# Savage-Dickey against the analytic answer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("k", [0.0, 1.0, 2.0, 3.0])
def test_savage_dickey_recovers_the_analytic_ln_bayes_factor(sd, k):
    """Task 1.3: the estimator must return k on a posterior built to give k."""
    eps = draw_tilted_uniform(k, seed=10 + int(k))
    res = sd.analyse(eps, (0.0, 1.0), n_bootstrap=100, seed=0)

    assert res[
        "reliable"
    ], f"unexpectedly unreliable at k={k}: {res['failed_diagnostics']}"
    tolerance = max(3.0 * res["uncert"], 0.25)
    assert (
        abs(res["ln_bayes_factor"] - k) < tolerance
    ), f"k={k}: got {res['ln_bayes_factor']:.3f} +/- {res['uncert']:.3f}"


def test_savage_dickey_sign_follows_the_evidence(sd):
    """A posterior piled at CURN must give a negative ln B, not just a small one."""
    eps = draw_tilted_uniform(-2.0, seed=99)
    res = sd.analyse(eps, (0.0, 1.0), n_bootstrap=100, seed=0)
    assert res["ln_bayes_factor"] < 0
    assert abs(res["ln_bayes_factor"] - (-2.0)) < max(3.0 * res["uncert"], 0.25)


def test_savage_dickey_handles_a_non_unit_interval(sd):
    """Bounds other than [0, 1] must be rescaled, not silently assumed."""
    eps_unit = draw_tilted_uniform(1.5, seed=7)
    res_unit = sd.analyse(eps_unit, (0.0, 1.0), n_bootstrap=100, seed=0)
    # Same shape of posterior stretched onto [0, 2]; the density ratio between the
    # endpoints is unchanged by a common rescaling.
    res_wide = sd.analyse(2.0 * eps_unit, (0.0, 2.0), n_bootstrap=100, seed=0)
    assert abs(res_unit["ln_bayes_factor"] - res_wide["ln_bayes_factor"]) < 0.05


# ---------------------------------------------------------------------------
# The reliability gate
# ---------------------------------------------------------------------------


def test_gate_fails_when_an_endpoint_is_unpopulated(sd):
    """The failure mode that matters: a strong signal empties the CURN endpoint.

    At large k essentially no draws land near eps=0, so the density there is pure
    kernel extrapolation. The estimator must say so rather than return a number.
    """
    eps = draw_tilted_uniform(30.0, seed=5)
    res = sd.analyse(eps, (0.0, 1.0), n_bootstrap=100, seed=0)
    assert not res["reliable"]
    assert "endpoint_occupancy" in res["failed_diagnostics"]


def test_gate_reports_the_named_failing_diagnostic(sd):
    """The spec requires the failing diagnostic be named, not just a boolean."""
    eps = draw_tilted_uniform(30.0, seed=6)
    res = sd.analyse(eps, (0.0, 1.0), n_bootstrap=100, seed=0)
    assert isinstance(res["failed_diagnostics"], list)
    assert res["failed_diagnostics"]
    assert all(isinstance(name, str) for name in res["failed_diagnostics"])


def test_unreliable_result_still_carries_its_diagnostics(sd):
    """A refusal must be diagnosable, so the raw value and diagnostics stay."""
    eps = draw_tilted_uniform(30.0, seed=8)
    res = sd.analyse(eps, (0.0, 1.0), n_bootstrap=100, seed=0)
    assert not res["reliable"]
    assert np.isfinite(res["ln_bayes_factor"])
    assert "endpoint_occupancy_curn" in res["diagnostics"]
    assert res["diagnostics"]["endpoint_occupancy_curn"] < sd.MIN_ENDPOINT_OCCUPANCY


def test_reliable_result_is_json_serialisable(sd):
    """The report is written to JSON, so every field must survive round-tripping."""
    import json

    eps = draw_tilted_uniform(1.0, seed=11)
    res = sd.analyse(eps, (0.0, 1.0), n_bootstrap=50, seed=0)
    round_tripped = json.loads(json.dumps(res))
    assert round_tripped["ln_bayes_factor"] == pytest.approx(res["ln_bayes_factor"])
    assert round_tripped["reliable"] == res["reliable"]


def test_zero_bandwidth_extrapolation_reduces_the_smoothing_bias(sd):
    """Locks in the bias correction: Silverman's own bandwidth over-smooths.

    A kernel density estimate flattens the density it estimates, which pulls the
    endpoint ratio towards zero. The h -> 0 extrapolation removes most of that.
    Measured over an ensemble of realisations at k=2, the raw estimate at
    Silverman's bandwidth is biased low by ~0.11 nats while the extrapolated one is
    biased by ~0.04 -- a real correction, several times the sampling uncertainty.

    The claim is about *bias over realisations*, not about every realisation: the
    extrapolation trades bias for variance, so on any single seed it can land
    further from the truth. That trade is worth making here because the bias is
    systematic and would not shrink with more draws, whereas the variance does.
    Asserting it per-seed would be asserting something untrue.
    """
    k = 2.0
    n_seeds = 6
    raw, extrapolated = [], []
    for seed in range(n_seeds):
        eps = draw_tilted_uniform(k, seed=200 + seed)
        res = sd.analyse(eps, (0.0, 1.0), n_bootstrap=10, seed=0)
        raw.append(
            float(
                np.mean([e["raw_at_silverman"] for e in res["per_estimator"].values()])
            )
        )
        extrapolated.append(res["ln_bayes_factor"])

    raw_bias = float(np.mean(raw)) - k
    extrapolated_bias = float(np.mean(extrapolated)) - k

    assert (
        raw_bias < -0.05
    ), f"expected the raw estimate to be biased low, got {raw_bias:+.4f}"
    assert abs(extrapolated_bias) < abs(raw_bias) / 2.0, (
        f"extrapolation did not halve the bias: raw {raw_bias:+.4f}, "
        f"extrapolated {extrapolated_bias:+.4f}"
    )


# ---------------------------------------------------------------------------
# Path sampling (estimator B) against the same analytic answer
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ps():
    return _load_script("lnb_path_sampling")


def draw_cubic_tilted(a, n_chain=4, n_draw=4000, seed=0):
    """Draws from p(eps) ∝ exp(a*eps^3) on [0, 1] by numerical inverse CDF.

    This is the posterior implied by ln Z(eps) = a*eps^3 under a uniform prior on
    eps, which is the *same* synthetic problem the path-sampling rungs below
    represent -- viewed from the single-run side instead of the ladder side. Both
    estimators must return ln B = ln Z(1) - ln Z(0) = a.

    Chosen over the simpler exp(k*eps) because its integrand 3*a*eps^2 is curved,
    so the trapezoid rule is not exact and the discretisation diagnostics are
    actually exercised.
    """
    grid = np.linspace(0.0, 1.0, 200001)
    density = np.exp(a * grid**3)
    cdf = np.cumsum(density)
    cdf -= cdf[0]
    cdf /= cdf[-1]
    rng = np.random.default_rng(seed)
    u = rng.uniform(size=(n_chain, n_draw))
    return np.interp(u, cdf, grid)


def make_ladder(a, eps_grid, n_chain=4, n_draw=1000, noise=0.5, seed=0):
    """Synthetic ladder rungs for ln Z(eps) = a*eps^3.

    The path-sampling integrand is E[dlogL/deps] = dlnZ/deps = 3*a*eps^2 at each
    rung; the per-draw values scatter around it with MCMC noise.
    """
    rng = np.random.default_rng(seed)
    rungs = []
    for eps in eps_grid:
        mean = 3.0 * a * eps**2
        integrand = rng.normal(mean, noise, size=(n_chain, n_draw))
        rungs.append({"eps": float(eps), "integrand": integrand, "path": f"eps={eps}"})
    return rungs


@pytest.mark.parametrize("a", [0.0, 1.0, 3.0])
def test_path_sampling_recovers_the_analytic_ln_bayes_factor(ps, a):
    """Task 1.4: the ladder integral must return a on a problem built to give a."""
    eps_grid = np.linspace(0.0, 1.0, 9)
    rungs = make_ladder(a, eps_grid, seed=int(a) + 1)
    res = ps.analyse(rungs)

    assert res["reliable"], f"unreliable at a={a}: {res['failed_diagnostics']}"
    tolerance = max(3.0 * res["uncert"], 0.1)
    assert (
        abs(res["ln_bayes_factor"] - a) < tolerance
    ), f"a={a}: got {res['ln_bayes_factor']:.4f} +/- {res['uncert']:.4f}"


def test_path_sampling_sign_follows_the_evidence(ps):
    eps_grid = np.linspace(0.0, 1.0, 9)
    res = ps.analyse(make_ladder(-2.0, eps_grid, seed=4))
    assert res["ln_bayes_factor"] < 0
    assert abs(res["ln_bayes_factor"] + 2.0) < max(3.0 * res["uncert"], 0.1)


def test_path_sampling_survives_where_savage_dickey_fails(ps, sd):
    """The reason estimator B exists: no boundary density, so no boundary failure.

    At a=10 the posterior on eps is crushed against the HD end, which empties the
    CURN endpoint and makes estimator A refuse. The ladder has no such step -- each
    rung is sampled at its own fixed eps, so eps=0 is as well sampled as any other.
    """
    strong = 10.0
    sd_res = sd.analyse(draw_cubic_tilted(strong, seed=31), (0.0, 1.0), n_bootstrap=50)
    assert not sd_res["reliable"]
    assert "endpoint_occupancy" in sd_res["failed_diagnostics"]

    ps_res = ps.analyse(make_ladder(strong, np.linspace(0.0, 1.0, 13), seed=31))
    assert ps_res["reliable"], ps_res["failed_diagnostics"]
    assert abs(ps_res["ln_bayes_factor"] - strong) < max(3.0 * ps_res["uncert"], 0.2)


def test_the_two_estimators_agree_on_the_shared_synthetic_case(ps, sd):
    """Task 1.4: A and B must agree where both are reliable.

    This is the bake-off in miniature. The two estimators share only the definition
    of ln B; their machinery is completely different (boundary density estimate
    versus quadrature of a posterior expectation), so agreement is real evidence.
    """
    a = 2.0
    sd_res = sd.analyse(draw_cubic_tilted(a, seed=41), (0.0, 1.0), n_bootstrap=100)
    ps_res = ps.analyse(make_ladder(a, np.linspace(0.0, 1.0, 9), seed=41))

    assert sd_res["reliable"], sd_res["failed_diagnostics"]
    assert ps_res["reliable"], ps_res["failed_diagnostics"]

    combined = math.hypot(sd_res["uncert"], ps_res["uncert"])
    gap = abs(sd_res["ln_bayes_factor"] - ps_res["ln_bayes_factor"])
    assert gap < max(3.0 * combined, 0.2), (
        f"A={sd_res['ln_bayes_factor']:.4f}+/-{sd_res['uncert']:.4f}  "
        f"B={ps_res['ln_bayes_factor']:.4f}+/-{ps_res['uncert']:.4f}"
    )
    for res, truth_gap in ((sd_res, a), (ps_res, a)):
        assert abs(res["ln_bayes_factor"] - truth_gap) < max(3.0 * res["uncert"], 0.25)


# ---------------------------------------------------------------------------
# Path-sampling reliability gate
# ---------------------------------------------------------------------------


def test_gate_fails_on_a_ladder_too_coarse_for_the_curvature(ps):
    """A three-rung ladder cannot resolve a curved integrand; it must say so."""
    res = ps.analyse(
        make_ladder(6.0, [0.0, 0.5, 1.0], n_draw=20000, noise=0.05, seed=2)
    )
    assert not res["reliable"]
    assert "ladder_too_short" in res["failed_diagnostics"] or (
        "discretisation_error" in res["failed_diagnostics"]
    )


def test_gate_fails_when_the_ladder_misses_an_endpoint(ps):
    """Integrating from 0.2 to 1 is not ln B; the estimator must not pretend it is."""
    res = ps.analyse(make_ladder(2.0, np.linspace(0.2, 1.0, 9), seed=3))
    assert not res["reliable"]
    assert "ladder_endpoints" in res["failed_diagnostics"]


def test_gate_fails_on_a_badly_mixed_rung(ps):
    """One rung with a stuck integrand poisons the integral; ESS must catch it."""
    eps_grid = np.linspace(0.0, 1.0, 9)
    rungs = make_ladder(2.0, eps_grid, seed=5)
    # Replace one rung's draws with a highly autocorrelated sequence of the same
    # mean and variance -- indistinguishable by mean alone, obvious by ESS.
    stuck = rungs[4]["integrand"]
    rungs[4]["integrand"] = np.repeat(stuck[:, :10], stuck.shape[1] // 10, axis=1)
    res = ps.analyse(rungs)
    assert not res["reliable"]
    assert "integrand_ess" in res["failed_diagnostics"]


def test_duplicate_rungs_are_rejected(ps):
    """Two runs at the same eps is a staging mistake, not a finer ladder."""
    rungs = make_ladder(1.0, [0.0, 0.25, 0.5, 0.5, 1.0], seed=6)
    with pytest.raises(ValueError, match="Duplicate eps"):
        ps.analyse(rungs)


def test_refinement_points_at_the_most_curved_interval(ps):
    """The ladder length is left open by design; the estimator must advise on it."""
    eps_grid = np.linspace(0.0, 1.0, 7)
    res = ps.analyse(make_ladder(5.0, eps_grid, n_draw=20000, noise=0.05, seed=7))
    suggestion = ps.suggest_refinement(res)
    # The integrand 3*a*eps^2 curves most steeply near eps=1, so that is where
    # extra rungs buy the most.
    assert suggestion["worst_interval"][1] > 0.5
    assert len(suggestion["suggested_new_eps"]) == 2


def test_path_sampling_result_is_json_serialisable(ps):
    import json

    res = ps.analyse(make_ladder(1.0, np.linspace(0.0, 1.0, 5), seed=8))
    round_tripped = json.loads(json.dumps(res))
    assert round_tripped["ln_bayes_factor"] == pytest.approx(res["ln_bayes_factor"])


def test_romberg_is_used_on_a_uniform_ladder(ps):
    """A uniform ladder that can be coarsened twice gets the higher-order treatment."""
    res = ps.analyse(make_ladder(2.0, np.linspace(0.0, 1.0, 9), seed=12))
    assert res["diagnostics"]["quadrature_method"] == "romberg"
    assert res["diagnostics"]["romberg"] is not None


def test_non_uniform_ladder_falls_back_without_pretending(ps):
    """A refined (non-uniform) ladder cannot support Romberg; no correction is faked."""
    grid = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
    res = ps.analyse(make_ladder(2.0, grid, seed=13))
    assert res["diagnostics"]["quadrature_method"] == "trapezoid"
    assert res["diagnostics"]["quadrature_correction"] == 0.0
    assert res["diagnostics"]["romberg"] is None


def _peaked_ladder(grid, amplitude=4.0, width=0.12, noise=0.02, seed=0):
    """Rungs for a narrow, strongly non-polynomial integrand.

    The polynomial test problems above are integrated exactly by the extrapolation,
    so they cannot exercise the discretisation gate at all. This one can: a Gaussian
    bump of width 0.12 is invisible to a five-rung ladder and well resolved by
    seventeen.
    """
    rng = np.random.default_rng(seed)
    return [
        {
            "eps": float(e),
            "integrand": rng.normal(
                amplitude * math.exp(-(((e - 0.5) / width) ** 2)),
                noise,
                size=(4, 2000),
            ),
            "path": str(e),
        }
        for e in grid
    ]


def _peaked_truth(amplitude=4.0, width=0.12):
    from scipy.special import erf

    return (
        amplitude
        * width
        * math.sqrt(math.pi)
        / 2.0
        * (erf(0.5 / width) - erf(-0.5 / width))
    )


def test_discretisation_gate_fires_on_an_under_resolved_integrand(ps):
    """The gate must catch a ladder too coarse for the shape of the integrand."""
    res = ps.analyse(_peaked_ladder(np.linspace(0.0, 1.0, 5), seed=3))
    assert not res["reliable"]
    assert "discretisation_error" in res["failed_diagnostics"]
    assert abs(res["ln_bayes_factor"] - _peaked_truth()) > 0.1


def test_refining_the_ladder_clears_the_discretisation_gate(ps):
    """And must clear once the ladder resolves it, recovering the analytic integral."""
    res = ps.analyse(_peaked_ladder(np.linspace(0.0, 1.0, 17), seed=3))
    assert res["reliable"], res["failed_diagnostics"]
    assert abs(res["ln_bayes_factor"] - _peaked_truth()) < 0.05

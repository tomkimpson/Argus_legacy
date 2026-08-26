#!/usr/bin/env python
"""Estimator A — HD-vs-CURN Bayes factor by generalised Savage-Dickey on the
correlation path. Pure CPU post-processing of a single results ``.nc``.

Why this exists
---------------
The learned harmonic mean (``logz_lhm.py``) reproduces a trusted nested-sampling
anchor at 2-D and works at 18-D, but goes degenerate at 68-D — the dimensionality
wall that left M1 without a calibrated Bayes factor. This estimator sidesteps
per-model evidences entirely.

The correlation path (``gravitational_waves.correlation_path``) embeds both
hypotheses in one model,

    C(eps) = (1 - eps) * I + eps * C_HD ,

so CURN is the point eps=0 and HD is the point eps=1 of a single posterior. Every
other parameter, prior and data input is shared, so the nuisance parameters
integrate out of the ratio and the Bayes factor is a ratio of *marginal* densities
of one scalar:

    ln B(HD/CURN) = [ln p(eps=1|d) - ln p(eps=1)] - [ln p(eps=0|d) - ln p(eps=0)]

This is the generalised Savage-Dickey density ratio: ordinary Savage-Dickey
compares a model with a nested null, here both models are points in a common
extension, so both endpoints appear. With the uniform prior the code enforces, the
two prior terms are equal and cancel exactly.

One run, one scalar, no evidence needed. The cost is that everything now rests on
a density estimate at the *boundary* of the sampled interval — which is exactly the
tail-estimation failure mode that killed LHM, relocated rather than removed. So
this script does not simply report a number: it reports whether the number can be
believed, from four diagnostics, and refuses the estimate when they fail. The
path-sampling estimator (``lnb_path_sampling.py``) has no boundary-density step and
is the fallback when they do.

Diagnostics (all must pass for ``reliable: true``)
--------------------------------------------------
* ``endpoint_occupancy`` — the fraction of draws within a neighbourhood of each
  endpoint. Near zero means the density there is extrapolation, not estimation.
  This is the direct analogue of LHM's max-weight-fraction blow-up.
* ``bandwidth_plateau`` — ln B must be flat across a bandwidth sweep. A trend means
  the answer is a bandwidth choice, not a measurement.
* ``estimator_agreement`` — two independent boundary-corrected density estimators
  (reflection KDE and beta-kernel KDE) must agree within their combined uncertainty.
* ``fold_agreement`` — the two halves of the chains must agree.

Run (CPU):
    JAX_PLATFORMS=cpu python workflows/ng15_sgwb_demo/scripts/lnb_savage_dickey.py \
        --results outputs/mdc2_stage_c_path/mdc2_stage_c_path_results.nc \
        --out outputs/mdc2_stage_c_path/lnb_savage_dickey.json
"""

import argparse
import json
import math

import numpy as np
from scipy.special import betaln, logsumexp

# Diagnostic tolerances. Deliberately conservative: this estimator's whole purpose
# is to fail loudly rather than return a plausible wrong number.
MIN_ENDPOINT_OCCUPANCY = 0.005  # >=0.5% of draws within the endpoint neighbourhood
ENDPOINT_NEIGHBOURHOOD = 0.05  # fraction of the eps range counted as "near" an end
MAX_FIT_RESIDUAL_SIGMA = 3.0  # h^2 model must describe the sweep this well
MAX_EXTRAPOLATION_DRIFT_SIGMA = 3.0  # short vs full ladder must agree this well
MAX_ESTIMATOR_DISAGREEMENT_SIGMA = 3.0
MAX_FOLD_DISAGREEMENT_SIGMA = 3.0

# Bandwidth ladder, as multiples of Silverman's rule. Weighted towards small
# bandwidths: Silverman minimises integrated squared error over the whole density,
# but this estimator reads the density at two boundary points, where over-smoothing
# is the dominant error. Measured on the analytic test problem, Silverman's own
# bandwidth is biased low in |ln B| by ~0.15 nats, which the extrapolation removes.
BANDWIDTH_FACTORS = (0.25, 0.4, 0.6, 1.0, 1.5)
SHORT_LADDER = 3  # the smallest-bandwidth rungs, for the stability check
DEFAULT_N_BOOTSTRAP = 200
DEFAULT_N_BLOCKS = 8


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_epsilon(nc_path, site="orf_epsilon"):
    """Load the correlation-path coordinate from a results ``.nc``.

    Returns
    -------
    eps : ndarray, shape (n_chain, n_draw)
    bounds : tuple[float, float]
        The prior support (low, high). Read from the sampled site's recorded
        bounds when present, else assumed to be the unit interval.
    """
    import arviz as az

    idata = az.from_netcdf(nc_path)
    post = idata.posterior
    if site not in post.data_vars:
        raise ValueError(
            f"{nc_path} has no '{site}' site. This estimator needs a run with "
            f"orf_path = sampled; found: {sorted(post.data_vars)}"
        )
    eps = np.asarray(post[site].values, dtype=float)
    if eps.ndim != 2:
        raise ValueError(f"'{site}' should be scalar per draw, got shape {eps.shape}")

    low = float(post.attrs.get("orf_epsilon_min", 0.0))
    high = float(post.attrs.get("orf_epsilon_max", 1.0))
    if not high > low:
        raise ValueError(f"Bad eps bounds ({low}, {high}) in {nc_path}")

    outside = np.sum((eps < low - 1e-9) | (eps > high + 1e-9))
    if outside:
        raise ValueError(
            f"{outside} draws of '{site}' lie outside the prior support "
            f"[{low}, {high}] — the run and this estimator disagree about the model."
        )
    return eps, (low, high)


# ---------------------------------------------------------------------------
# Boundary-corrected density estimation on a bounded interval
# ---------------------------------------------------------------------------


def _to_unit(x, low, high):
    return (x - low) / (high - low)


def silverman_bandwidth(u):
    """Silverman's rule on the unit-scaled samples, floored away from zero."""
    n = u.size
    spread = min(
        np.std(u, ddof=1), (np.percentile(u, 75) - np.percentile(u, 25)) / 1.349
    )
    if not np.isfinite(spread) or spread <= 0:
        spread = np.std(u, ddof=1)
    if not np.isfinite(spread) or spread <= 0:
        spread = 1e-3
    return float(max(0.9 * spread * n ** (-0.2), 1e-4))


def reflection_kde_logpdf(u_samples, u_eval, bandwidth):
    """Log density at ``u_eval`` by Gaussian KDE with reflection at 0 and 1.

    Reflecting the sample across each boundary cancels the leading boundary bias of
    a plain KDE, which would otherwise halve the density exactly at an endpoint —
    precisely where this estimator reads its answer.
    """
    u_samples = np.asarray(u_samples, dtype=float)
    n = u_samples.size
    images = np.concatenate([u_samples, -u_samples, 2.0 - u_samples])
    z = (u_eval - images) / bandwidth
    log_kernels = -0.5 * z**2 - 0.5 * math.log(2.0 * math.pi)
    return float(logsumexp(log_kernels) - math.log(n * bandwidth))


def beta_kde_logpdf(u_samples, u_eval, bandwidth):
    """Log density at ``u_eval`` by a Chen (1999) beta-kernel estimator.

    The beta kernel has support exactly [0, 1] and its shape adapts as the
    evaluation point approaches a boundary, so it is free of boundary bias by
    construction rather than by correction. Independent of the reflection KDE in
    its failure modes, which is what makes the agreement check meaningful.
    """
    u_samples = np.asarray(u_samples, dtype=float)
    n = u_samples.size
    a = u_eval / bandwidth + 1.0
    b = (1.0 - u_eval) / bandwidth + 1.0
    # Guard the log at the sample endpoints; a draw exactly at 0 or 1 has measure
    # zero in exact arithmetic but can appear after rounding.
    eps_guard = 1e-300
    safe = np.clip(u_samples, eps_guard, 1.0 - 1e-16)
    log_kernels = (a - 1.0) * np.log(safe) + (b - 1.0) * np.log1p(-safe) - betaln(a, b)
    return float(logsumexp(log_kernels) - math.log(n))


DENSITY_ESTIMATORS = {
    "reflection_kde": reflection_kde_logpdf,
    "beta_kde": beta_kde_logpdf,
}


# ---------------------------------------------------------------------------
# The estimator
# ---------------------------------------------------------------------------


def log_bayes_factor(u_samples, bandwidth, estimator):
    """ln B(HD/CURN) from the marginal density of eps at its two endpoints.

    The prior is uniform on the sampled interval, so ln p(eps=1) = ln p(eps=0) and
    the prior terms cancel; only the posterior densities remain.
    """
    logpdf = DENSITY_ESTIMATORS[estimator]
    return logpdf(u_samples, 1.0, bandwidth) - logpdf(u_samples, 0.0, bandwidth)


def extrapolate_to_zero_bandwidth(bandwidths, values):
    """Fit ``v(h) = v0 + c h^2`` and return ``(v0, residual_rms, curvature)``.

    A kernel density estimate smooths the density it estimates, which flattens the
    ratio between the two endpoints and biases ln B towards zero. The leading bias
    of a second-order kernel is O(h^2), so the zero-bandwidth intercept of that fit
    is the bias-corrected estimate. The residual says whether the ladder is actually
    in the asymptotic regime where that model holds.
    """
    h2 = np.asarray(bandwidths, dtype=float) ** 2
    values = np.asarray(values, dtype=float)
    design = np.vstack([np.ones_like(h2), h2]).T
    coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
    residuals = values - design @ coefficients
    return (
        float(coefficients[0]),
        float(np.sqrt(np.mean(residuals**2))),
        float(coefficients[1]),
    )


def ladder_estimate(u_samples, base_bandwidth, estimator, factors=BANDWIDTH_FACTORS):
    """ln B extrapolated to zero bandwidth over the bandwidth ladder."""
    bandwidths = [base_bandwidth * f for f in factors]
    values = [log_bayes_factor(u_samples, h, estimator) for h in bandwidths]
    v0, residual, curvature = extrapolate_to_zero_bandwidth(bandwidths, values)
    return {
        "ln_bayes_factor": v0,
        "fit_residual": residual,
        "curvature": curvature,
        "bandwidths": bandwidths,
        "values": values,
    }


def block_bootstrap_uncertainty(u_blocks, base_bandwidth, estimator, n_bootstrap, rng):
    """Uncertainty on the extrapolated ln B by resampling contiguous blocks of draws.

    Blocks rather than individual draws, because MCMC draws are autocorrelated and a
    naive bootstrap would understate the spread. The whole extrapolation is redone
    inside each resample, so the reported uncertainty covers the ladder fit too.
    """
    n_blocks = len(u_blocks)
    values = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n_blocks, size=n_blocks)
        resampled = np.concatenate([u_blocks[i] for i in idx])
        values.append(
            ladder_estimate(resampled, base_bandwidth, estimator)["ln_bayes_factor"]
        )
    return float(np.std(values, ddof=1))


def endpoint_occupancy(u_samples, width=ENDPOINT_NEIGHBOURHOOD):
    """Fraction of draws within ``width`` of each endpoint.

    The single most important diagnostic: a density read where no draws live is an
    extrapolation of the kernel, not a property of the posterior.
    """
    near_zero = float(np.mean(u_samples <= width))
    near_one = float(np.mean(u_samples >= 1.0 - width))
    return near_zero, near_one


def analyse(
    eps, bounds, n_bootstrap=DEFAULT_N_BOOTSTRAP, n_blocks=DEFAULT_N_BLOCKS, seed=0
):
    """Full Savage-Dickey analysis: bandwidth ladder x two estimators x folds."""
    low, high = bounds
    n_chain, n_draw = eps.shape
    u = _to_unit(eps, low, high)
    u_flat = u.reshape(-1)
    rng = np.random.default_rng(seed)

    # Contiguous blocks for the bootstrap, spread across chains.
    per_chain = max(1, n_blocks // max(n_chain, 1))
    u_blocks = [
        block
        for chain in u
        for block in np.array_split(chain, per_chain)
        if block.size > 0
    ]

    base_bw = silverman_bandwidth(u_flat)
    occ_zero, occ_one = endpoint_occupancy(u_flat)

    per_estimator = {}
    for name in DENSITY_ESTIMATORS:
        full = ladder_estimate(u_flat, base_bw, name)
        short = ladder_estimate(
            u_flat, base_bw, name, factors=BANDWIDTH_FACTORS[:SHORT_LADDER]
        )
        sigma = block_bootstrap_uncertainty(u_blocks, base_bw, name, n_bootstrap, rng)
        per_estimator[name] = {
            "ln_bayes_factor": full["ln_bayes_factor"],
            "uncert": sigma,
            "fit_residual": full["fit_residual"],
            "extrapolation_drift": abs(
                full["ln_bayes_factor"] - short["ln_bayes_factor"]
            ),
            "raw_at_silverman": full["values"][BANDWIDTH_FACTORS.index(1.0)],
            "bandwidths": full["bandwidths"],
            "values": full["values"],
        }

    # Fold agreement: first half of the chains against the second half.
    half = max(1, n_chain // 2)
    fold_gap = 0.0
    for name in DENSITY_ESTIMATORS:
        fold_a = ladder_estimate(u[:half].reshape(-1), base_bw, name)
        fold_b = ladder_estimate(u[half:].reshape(-1), base_bw, name)
        fold_gap = max(
            fold_gap, abs(fold_a["ln_bayes_factor"] - fold_b["ln_bayes_factor"])
        )

    ref = per_estimator["reflection_kde"]
    beta = per_estimator["beta_kde"]
    combined_sigma = math.hypot(ref["uncert"], beta["uncert"])
    estimator_gap = abs(ref["ln_bayes_factor"] - beta["ln_bayes_factor"])

    # The reported value averages the two estimators: they are independent
    # treatments of the same boundary problem, and are only reported when they
    # agree. Their residual difference is folded into the uncertainty as a
    # systematic rather than discarded.
    ln_b = 0.5 * (ref["ln_bayes_factor"] + beta["ln_bayes_factor"])
    uncert = math.hypot(max(ref["uncert"], beta["uncert"]), estimator_gap / 2.0)

    max_fit_residual = max(ref["fit_residual"], beta["fit_residual"])
    max_drift = max(ref["extrapolation_drift"], beta["extrapolation_drift"])

    failures = []
    if min(occ_zero, occ_one) < MIN_ENDPOINT_OCCUPANCY:
        failures.append("endpoint_occupancy")
    if max_fit_residual > MAX_FIT_RESIDUAL_SIGMA * max(combined_sigma, 1e-12):
        failures.append("bandwidth_model_fit")
    if max_drift > MAX_EXTRAPOLATION_DRIFT_SIGMA * max(combined_sigma, 1e-12):
        failures.append("extrapolation_stability")
    if estimator_gap > MAX_ESTIMATOR_DISAGREEMENT_SIGMA * max(combined_sigma, 1e-12):
        failures.append("estimator_agreement")
    if fold_gap > MAX_FOLD_DISAGREEMENT_SIGMA * max(uncert, 1e-12):
        failures.append("fold_agreement")

    return {
        "estimator": "savage_dickey",
        "n_chain": int(n_chain),
        "n_draw": int(n_draw),
        "eps_bounds": [low, high],
        "ln_bayes_factor": ln_b,
        "uncert": uncert,
        "reliable": len(failures) == 0,
        "failed_diagnostics": failures,
        "diagnostics": {
            "endpoint_occupancy_curn": occ_zero,
            "endpoint_occupancy_hd": occ_one,
            "bandwidth_silverman": base_bw,
            "bandwidth_model_residual": max_fit_residual,
            "extrapolation_drift": max_drift,
            "extrapolation_correction": abs(
                ln_b - 0.5 * (ref["raw_at_silverman"] + beta["raw_at_silverman"])
            ),
            "estimator_gap": estimator_gap,
            "fold_gap": fold_gap,
            "eps_mean": float(np.mean(u_flat) * (high - low) + low),
            "eps_median": float(np.median(u_flat) * (high - low) + low),
        },
        "per_estimator": per_estimator,
    }


def print_report(res):
    print("\n=== Savage-Dickey ln B(HD/CURN) ===")
    print(
        f"  chains={res['n_chain']}  draws/chain={res['n_draw']}  "
        f"eps in [{res['eps_bounds'][0]}, {res['eps_bounds'][1]}]"
    )
    d = res["diagnostics"]
    print(f"  eps posterior: mean={d['eps_mean']:.4f} median={d['eps_median']:.4f}")
    print(f"\n  bandwidth ladder (ln B at each h, extrapolated to h -> 0):")
    for name, entry in res["per_estimator"].items():
        rungs = "  ".join(f"{v:.4f}" for v in entry["values"])
        print(
            f"    {name:>15}: {rungs}   -> {entry['ln_bayes_factor']:.4f} "
            f"+/- {entry['uncert']:.4f}"
        )
    print("\n  diagnostics:")
    print(
        f"    endpoint occupancy   : CURN {d['endpoint_occupancy_curn']:.4f}  "
        f"HD {d['endpoint_occupancy_hd']:.4f}  "
        f"(floor {MIN_ENDPOINT_OCCUPANCY})"
    )
    print(f"    h^2 model residual   : {d['bandwidth_model_residual']:.4f}")
    print(f"    extrapolation drift  : {d['extrapolation_drift']:.4f}")
    print(f"    extrapolation applied: {d['extrapolation_correction']:.4f}")
    print(f"    estimator gap        : {d['estimator_gap']:.4f}")
    print(f"    fold gap             : {d['fold_gap']:.4f}")
    if res["reliable"]:
        print(
            f"\n  -> ln B(HD/CURN) = {res['ln_bayes_factor']:.4f} "
            f"+/- {res['uncert']:.4f}   [reliable]"
        )
    else:
        print(f"\n  -> NOT USABLE: failed {', '.join(res['failed_diagnostics'])}")
        print(
            f"     (raw value {res['ln_bayes_factor']:.4f} +/- {res['uncert']:.4f} "
            f"is reported for diagnosis only, not as a result)"
        )


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--results", required=True, help="Path to results .nc (sampled eps)."
    )
    p.add_argument("--site", default="orf_epsilon", help="Name of the eps site.")
    p.add_argument("--n-bootstrap", type=int, default=DEFAULT_N_BOOTSTRAP)
    p.add_argument("--n-blocks", type=int, default=DEFAULT_N_BLOCKS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="Write JSON summary here.")
    args = p.parse_args()

    eps, bounds = load_epsilon(args.results, site=args.site)
    res = analyse(
        eps,
        bounds,
        n_bootstrap=args.n_bootstrap,
        n_blocks=args.n_blocks,
        seed=args.seed,
    )
    res["results_path"] = args.results
    print_report(res)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)
        print(f"\n  wrote {args.out}")

    return 0 if res["reliable"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

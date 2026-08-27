#!/usr/bin/env python
"""Estimator B — HD-vs-CURN Bayes factor by path sampling (thermodynamic
integration) along the correlation path. Post-processing of a ladder of runs.

Why this exists
---------------
Estimator A (``lnb_savage_dickey.py``) gets the Bayes factor from one run, but it
reads it off a density estimate at the *boundary* of the sampled interval. When the
data strongly favour Hellings-Downs, no draws land near eps=0 and that density is
extrapolation — the same tail-estimation failure that killed the learned harmonic
mean, relocated rather than removed. This estimator has no tail-density step at all.

The correlation path C(eps) = (1 - eps)*I + eps*C_HD connects CURN (eps=0) to HD
(eps=1) through a family of models that share every other parameter and prior. Write
Z(eps) for the evidence of the model at fixed eps. Then

    d ln Z(eps) / d eps = E_{p(θ | d, eps)} [ ∂ ln L(θ, eps) / ∂ eps ]

(the standard path-sampling identity: differentiating ln Z under the integral gives
a posterior expectation, because the prior does not depend on eps). Integrating
along the path,

    ln B(HD/CURN) = ln Z(1) - ln Z(0) = ∫₀¹ E_eps[ ∂ ln L / ∂ eps ] d eps

Each rung of the ladder is an independent NUTS run at fixed eps (``orf_path =
fixed``), so the ladder is embarrassingly parallel across GPUs. The integrand comes
from JAX autodiff of the Kalman likelihood — no finite differences.

Cost versus robustness: this needs one array run per rung where estimator A needs
one in total. It buys an estimator whose error is *discretisation* error, which is
measurable by refining the ladder and shrinks predictably, rather than a density
estimate that can fail silently.

Two modes
---------
``--from-runs``  Each rung's ``.nc`` already stores the per-draw integrand under a
                 recorded site (written by the run harness). Pure CPU, no JAX.
``--evaluate``   Compute the integrand from stored posteriors by differentiating the
                 Kalman likelihood at each draw, given the run's config. Needs the
                 data and JAX. This is the normal path: recording the integrand
                 during sampling would nest a gradient inside NUTS's own.

Run (CPU, integrand already stored):
    JAX_PLATFORMS=cpu python workflows/ng15_sgwb_demo/scripts/lnb_path_sampling.py \
        --from-runs outputs/mdc2_path_ladder/eps_*.nc \
        --out outputs/mdc2_path_ladder/lnb_path_sampling.json
"""

import argparse
import glob
import json
import math

import numpy as np

# Diagnostic tolerances.
MIN_RUNGS = 5  # below this the ladder cannot be halved to estimate its own error
MIN_INTEGRAND_ESS = 50.0  # per-rung effective sample size of the integrand

# Discretisation tolerance, in nats, on the *residual* error after extrapolation. Absolute rather than relative to the MCMC uncertainty: quadrature
# error only matters if it could change a decision made with this number, and the
# decisions here are threshold tests at the nat scale (a decisive ln B is >= 3).
# A relative gate would fail a perfectly accurate ladder whenever the chains happen
# to be long, which is backwards.
#
# Caveat measured on a deliberately peaked test integrand: the Romberg residual is a
# leading-order estimate and can understate the true quadrature error by roughly an
# order of magnitude when the ladder badly under-resolves a narrow feature. It still
# fires well before the error reaches the tolerance (at the coarsest ladder tested,
# residual 0.13 against a true error of 0.28), but the residual should be watched for
# *stability* as the ladder is refined, not read as an exact error bar. That is what
# ``suggest_refinement`` is for.
MAX_RESIDUAL_DISCRETISATION = 0.1
INTEGRAND_SITE = "d_loglik_d_eps"

# numpy renamed trapz -> trapezoid in 2.0; the pinned environment predates that.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_rung(nc_path, integrand_site=INTEGRAND_SITE, eps_site="orf_epsilon"):
    """Load one ladder rung: its eps value and the per-draw integrand.

    Returns
    -------
    dict with ``eps``, ``integrand`` (n_chain, n_draw) and the source path.
    """
    import arviz as az

    idata = az.from_netcdf(nc_path)
    post = idata.posterior

    if eps_site not in post.data_vars:
        raise ValueError(f"{nc_path} has no '{eps_site}' site — not a ladder rung.")
    eps_values = np.asarray(post[eps_site].values, dtype=float)
    eps = float(np.unique(np.round(eps_values, 12)).item()) if eps_values.size else None
    if eps is None:
        raise ValueError(f"{nc_path}: could not read a fixed eps value.")

    if integrand_site not in post.data_vars:
        raise ValueError(
            f"{nc_path} has no '{integrand_site}' site. Either re-run with the "
            f"integrand recorded, or use --evaluate to compute it from the posterior."
        )
    integrand = np.asarray(post[integrand_site].values, dtype=float)
    return {"eps": eps, "integrand": integrand, "path": nc_path}


# ---------------------------------------------------------------------------
# Computing the integrand from a stored posterior
# ---------------------------------------------------------------------------

# Physical parameter sites the Kalman likelihood needs. All are recorded as
# deterministics by numpyro_model in every prior mode (flat, hierarchical,
# empirical), so a rung's .nc carries them regardless of how it was configured.
PHYSICAL_SITES = (
    "log10_ha",
    "log10_gamma_a",
    "log10_γp",
    "log10_σp",
    "efac",
    "equad",
)


DEFAULT_BATCH_SIZE = 100


def evaluate_integrand(
    nc_path, config_path, thin=1, eps_site="orf_epsilon", batch_size=DEFAULT_BATCH_SIZE
):
    """Compute the per-draw integrand ∂ ln L / ∂eps from a stored posterior.

    Done after the fact rather than during sampling on purpose. Recording the
    integrand inside the model would put a ``jax.grad`` under NUTS's own gradient —
    higher-order autodiff on the most expensive part of the run, for a quantity
    NUTS never needs. Here each draw costs one extra gradient evaluation, paid once,
    on whatever hardware is free.

    Draws are processed in batches of ``batch_size``. A single ``vmap`` over the whole
    posterior would hold one differentiated Kalman pass per draw in memory at once —
    at 33 pulsars and 183 epochs that is enough to be killed by the OOM reaper at a few
    hundred draws, and the full-array posteriors are larger still. Batching keeps the
    peak independent of how long the run was.

    Returns the same rung dict shape as ``load_rung``.
    """
    import configparser
    import logging

    import arviz as az
    import jax

    from argus import bayesian_inference, io_manager, utils, workflow

    config = configparser.ConfigParser()
    if not config.read(config_path):
        raise ValueError(f"Could not read config {config_path}")
    # Config paths are written relative to the config file, exactly as the run
    # harness reads them; resolve them the same way so this can be invoked from
    # anywhere rather than only from the workflow directory.
    config = utils.resolve_config_paths(config, config_path)

    # The Kalman filter logs through the package's own logger, which the run harness
    # normally initialises; do it here so this is usable outside a full run.
    io_manager.setup_single_logger(config, enable_file_logging=False)
    logger = logging.getLogger("lnb_path_sampling")
    _, kalman_filter = workflow.setup_data_and_kalman_filter(
        config, logger, use_gw=True
    )

    idata = az.from_netcdf(nc_path)
    post = idata.posterior

    missing = [site for site in PHYSICAL_SITES if site not in post.data_vars]
    if missing:
        raise ValueError(
            f"{nc_path} is missing physical sites {missing}; it cannot be "
            f"re-evaluated. Present: {sorted(post.data_vars)}"
        )
    if eps_site not in post.data_vars:
        raise ValueError(f"{nc_path} has no '{eps_site}' site — not a ladder rung.")

    eps_values = np.asarray(post[eps_site].values, dtype=float)
    unique_eps = np.unique(np.round(eps_values, 12))
    if unique_eps.size != 1:
        raise ValueError(
            f"{nc_path} has {unique_eps.size} distinct eps values; a ladder rung "
            f"must hold eps fixed (orf_path = fixed)."
        )
    eps = float(unique_eps.item())

    n_chain = post.sizes["chain"]
    n_draw = post.sizes["draw"]
    flat = {}
    for site in PHYSICAL_SITES:
        arr = np.asarray(post[site].values)
        arr = arr.reshape(n_chain * n_draw, *arr.shape[2:])
        flat[site] = arr[::thin]

    def d_loglik_d_eps(
        eps_value, log10_ha, log10_gamma_a, log10_gp, log10_sp, efac, equad
    ):
        def loglik(e):
            return bayesian_inference.log_likelihood_fn(
                kalman_filter,
                log10_ha,
                log10_gamma_a,
                log10_gp,
                log10_sp,
                efac,
                equad,
                orf_epsilon=e,
            )

        return jax.grad(loglik)(eps_value)

    batched = jax.jit(jax.vmap(d_loglik_d_eps, in_axes=(None, 0, 0, 0, 0, 0, 0)))

    n_draws = flat[PHYSICAL_SITES[0]].shape[0]
    if batch_size is None or batch_size <= 0:
        batch_size = n_draws
    pieces = []
    for start in range(0, n_draws, batch_size):
        stop = min(start + batch_size, n_draws)
        pieces.append(
            np.asarray(
                batched(eps, *[flat[site][start:stop] for site in PHYSICAL_SITES]),
                dtype=float,
            )
        )
    integrand = np.concatenate(pieces) if pieces else np.zeros(0)

    n_kept = integrand.size // n_chain
    integrand = integrand[: n_chain * n_kept].reshape(n_chain, n_kept)
    return {"eps": eps, "integrand": integrand, "path": nc_path}


# ---------------------------------------------------------------------------
# Per-rung statistics
# ---------------------------------------------------------------------------


def effective_sample_size(x):
    """ESS of a (n_chain, n_draw) array by the initial-positive-sequence estimator.

    The integrand is an MCMC average, so its uncertainty depends on autocorrelation,
    not on the raw draw count. A rung whose integrand mixes badly is the one that
    will dominate the error budget, so it has to be visible.
    """
    x = np.atleast_2d(np.asarray(x, dtype=float))
    n_chain, n_draw = x.shape
    if n_draw < 4:
        return float(x.size)

    centred = x - x.mean(axis=1, keepdims=True)
    if not np.any(centred):
        return float(x.size)

    # Autocorrelation via FFT, averaged over chains.
    n_pad = 1
    while n_pad < 2 * n_draw:
        n_pad *= 2
    spectrum = np.fft.rfft(centred, n=n_pad, axis=1)
    acov = np.fft.irfft(spectrum * np.conjugate(spectrum), n=n_pad, axis=1)[:, :n_draw]
    acov = acov.mean(axis=0) / n_draw
    if acov[0] <= 0:
        return float(x.size)
    rho = acov / acov[0]

    # Geyer's initial positive sequence on successive pairs.
    total = 0.0
    for t in range(1, n_draw - 1, 2):
        pair = rho[t] + rho[t + 1]
        if pair <= 0:
            break
        total += pair
    tau = 1.0 + 2.0 * total
    return float(max(x.size / max(tau, 1e-12), 1.0))


def rung_statistics(rung):
    """Posterior mean of the integrand at one rung, with its MCMC uncertainty."""
    integrand = rung["integrand"]
    mean = float(np.mean(integrand))
    ess = effective_sample_size(integrand)
    sigma = float(np.std(integrand, ddof=1) / math.sqrt(max(ess, 1.0)))
    return {
        "eps": rung["eps"],
        "mean": mean,
        "uncert": sigma,
        "ess": ess,
        "n_draws": int(integrand.size),
        "path": rung["path"],
    }


# ---------------------------------------------------------------------------
# Quadrature along the path
# ---------------------------------------------------------------------------


def trapezoid_integral(eps, values, sigmas=None):
    """Trapezoidal integral and its propagated uncertainty."""
    eps = np.asarray(eps, dtype=float)
    values = np.asarray(values, dtype=float)
    integral = float(_trapezoid(values, eps))
    if sigmas is None:
        return integral, None
    # Trapezoid weights: w_i = (eps_{i+1} - eps_{i-1}) / 2 at the interior.
    weights = np.zeros_like(eps)
    weights[0] = (eps[1] - eps[0]) / 2.0
    weights[-1] = (eps[-1] - eps[-2]) / 2.0
    weights[1:-1] = (eps[2:] - eps[:-2]) / 2.0
    uncert = float(np.sqrt(np.sum((weights * np.asarray(sigmas)) ** 2)))
    return integral, uncert


def simpson_integral(eps, values):
    """Simpson's rule. Used only on ladders Romberg cannot handle.

    Note that on a *uniform* grid Simpson is algebraically identical to the
    Richardson extrapolation of the trapezoid rule, S = (4*T_h - T_2h)/3, so
    comparing the two there would estimate an error of exactly zero. That identity
    is why the uniform case uses Romberg (one extrapolation level further) instead.
    """
    from scipy.integrate import simpson

    eps = np.asarray(eps, dtype=float)
    values = np.asarray(values, dtype=float)
    if eps.size < 3:
        return trapezoid_integral(eps, values)[0]
    return float(simpson(y=values, x=eps))


def is_uniform_grid(eps):
    """True when the rungs are equally spaced, to within rounding."""
    eps = np.asarray(eps, dtype=float)
    if eps.size < 3:
        return True
    spacings = np.diff(eps)
    return bool(np.allclose(spacings, spacings[0], rtol=1e-9, atol=1e-12))


def romberg_from_ladder(eps, values):
    """Romberg integration using the ladder, its half and its quarter.

    The trapezoid rule on a uniform grid has an error expansion in even powers of
    the spacing, so successive Richardson extrapolations kill successive terms:

        T(h), T(2h), T(4h)  ->  S(h), S(2h)  ->  R(h)

    The gap between the last two levels is the standard estimate of what remains.
    Needs a uniform ladder that can be coarsened twice, i.e. (n_rungs - 1) divisible
    by 4.

    Returns
    -------
    dict with ``integral``, ``residual``, and the intermediate levels, or None when
    the ladder cannot support it.
    """
    eps = np.asarray(eps, dtype=float)
    values = np.asarray(values, dtype=float)
    n = eps.size
    if n < 5 or (n - 1) % 4 != 0 or not is_uniform_grid(eps):
        return None

    t_h = trapezoid_integral(eps, values)[0]
    t_2h = trapezoid_integral(eps[::2], values[::2])[0]
    t_4h = trapezoid_integral(eps[::4], values[::4])[0]

    s_h = (4.0 * t_h - t_2h) / 3.0
    s_2h = (4.0 * t_2h - t_4h) / 3.0
    r_h = (16.0 * s_h - s_2h) / 15.0

    return {
        "integral": float(r_h),
        "residual": float(abs(r_h - s_h)),
        "trapezoid_h": float(t_h),
        "trapezoid_2h": float(t_2h),
        "trapezoid_4h": float(t_4h),
        "simpson_h": float(s_h),
        "simpson_2h": float(s_2h),
    }


def analyse(rungs, endpoints=(0.0, 1.0)):
    """Full path-sampling analysis over a ladder of rungs.

    On a uniform ladder that can be coarsened twice, the integral is Romberg
    extrapolated: the trapezoid rule's error expansion in even powers of the
    spacing is peeled off two terms at a time, and the gap between the last two
    levels is the residual. On any other ladder no correction is attempted and the
    residual is the trapezoid-Simpson gap, which is a genuine (if cruder) estimate
    there because the two rules are not algebraically related off a uniform grid.
    """
    stats = sorted((rung_statistics(r) for r in rungs), key=lambda s: s["eps"])
    eps = np.array([s["eps"] for s in stats])
    values = np.array([s["mean"] for s in stats])
    sigmas = np.array([s["uncert"] for s in stats])

    if len(set(eps.tolist())) != len(eps):
        raise ValueError(f"Duplicate eps values in the ladder: {eps.tolist()}")

    trapezoid, mcmc_uncert = trapezoid_integral(eps, values, sigmas)
    romberg = romberg_from_ladder(eps, values)

    if romberg is not None:
        integral = romberg["integral"]
        residual_discretisation = romberg["residual"]
        quadrature_method = "romberg"
        quadrature_correction = integral - trapezoid
    else:
        simpson_value = simpson_integral(eps, values)
        integral = trapezoid
        residual_discretisation = abs(simpson_value - trapezoid)
        quadrature_method = "trapezoid"
        quadrature_correction = 0.0

    # Fold the residual in as a systematic rather than reporting MCMC error alone.
    total_uncert = math.hypot(mcmc_uncert or 0.0, residual_discretisation)

    min_ess = float(min(s["ess"] for s in stats))
    covers_endpoints = math.isclose(
        eps[0], endpoints[0], abs_tol=1e-9
    ) and math.isclose(eps[-1], endpoints[1], abs_tol=1e-9)

    failures = []
    if len(eps) < MIN_RUNGS:
        failures.append("ladder_too_short")
    if not covers_endpoints:
        failures.append("ladder_endpoints")
    if residual_discretisation > MAX_RESIDUAL_DISCRETISATION:
        failures.append("discretisation_error")
    if min_ess < MIN_INTEGRAND_ESS:
        failures.append("integrand_ess")

    return {
        "estimator": "path_sampling",
        "n_rungs": int(len(eps)),
        "eps_grid": eps.tolist(),
        "ln_bayes_factor": integral,
        "uncert": total_uncert,
        "reliable": len(failures) == 0,
        "failed_diagnostics": failures,
        "diagnostics": {
            "mcmc_uncert": mcmc_uncert,
            "quadrature_method": quadrature_method,
            "trapezoid": trapezoid,
            "quadrature_correction": quadrature_correction,
            "residual_discretisation": residual_discretisation,
            "min_integrand_ess": min_ess,
            "covers_endpoints": bool(covers_endpoints),
            "romberg": romberg,
        },
        "rungs": stats,
    }


def print_report(res):
    print("\n=== Path-sampling ln B(HD/CURN) ===")
    print(
        f"  rungs={res['n_rungs']}  eps grid: "
        f"{', '.join(f'{e:.3f}' for e in res['eps_grid'])}"
    )
    print(f"\n  {'eps':>7} {'<dlogL/deps>':>16} {'uncert':>10} {'ESS':>9}")
    for rung in res["rungs"]:
        print(
            f"  {rung['eps']:7.3f} {rung['mean']:16.4f} {rung['uncert']:10.4f} "
            f"{rung['ess']:9.0f}"
        )
    d = res["diagnostics"]
    print("\n  diagnostics:")
    print(
        f"    quadrature           : {d['quadrature_method']} "
        f"(raw trapezoid {d['trapezoid']:.4f}, "
        f"correction {d['quadrature_correction']:+.4f})"
    )
    print(
        f"    residual after fix   : {d['residual_discretisation']:.4f} "
        f"(ceiling {MAX_RESIDUAL_DISCRETISATION})"
    )
    print(
        f"    min integrand ESS    : {d['min_integrand_ess']:.0f} "
        f"(floor {MIN_INTEGRAND_ESS:.0f})"
    )
    print(f"    covers endpoints     : {d['covers_endpoints']}")
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


def suggest_refinement(res, target_uncert=0.5):
    """Where to add rungs next, from the measured curvature of the integrand.

    The design leaves the ladder length open, to be decided from the integrand.
    This turns that into a concrete recommendation: the interval contributing the
    most curvature is the one worth splitting.
    """
    eps = np.array(res["eps_grid"])
    values = np.array([r["mean"] for r in res["rungs"]])
    if eps.size < 3:
        return {"suggestion": "add rungs: at least three are needed to see curvature"}

    second_difference = np.abs(np.diff(values, n=2))
    widths = eps[2:] - eps[:-2]
    weight = second_difference * widths**2
    worst = int(np.argmax(weight))
    midpoints = [
        0.5 * (eps[worst] + eps[worst + 1]),
        0.5 * (eps[worst + 1] + eps[worst + 2]),
    ]
    return {
        "worst_interval": [float(eps[worst]), float(eps[worst + 2])],
        "suggested_new_eps": [float(m) for m in midpoints],
        "meets_target": bool(res["uncert"] <= target_uncert),
        "target_uncert": target_uncert,
    }


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--from-runs",
        nargs="+",
        required=True,
        help="Ladder rung .nc files (globs allowed); one per fixed eps.",
    )
    p.add_argument(
        "--evaluate",
        default=None,
        metavar="CONFIG",
        help="Compute the integrand from each rung's posterior using this config "
        "(needs the data and JAX). Without it, the integrand must already be "
        "stored in each .nc under --integrand-site.",
    )
    p.add_argument(
        "--thin",
        type=int,
        default=1,
        help="Use every Nth draw when evaluating the integrand (--evaluate only).",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Draws per vmap batch when evaluating the integrand. Caps peak memory; "
        "0 or negative means one batch (only safe for small posteriors).",
    )
    p.add_argument("--integrand-site", default=INTEGRAND_SITE)
    p.add_argument("--eps-site", default="orf_epsilon")
    p.add_argument("--target-uncert", type=float, default=0.5)
    p.add_argument("--out", default=None, help="Write JSON summary here.")
    args = p.parse_args()

    paths = []
    for pattern in args.from_runs:
        matched = sorted(glob.glob(pattern))
        paths.extend(matched or [pattern])
    if not paths:
        raise SystemExit("No ladder rungs matched.")

    if args.evaluate:
        rungs = [
            evaluate_integrand(
                path,
                args.evaluate,
                thin=args.thin,
                eps_site=args.eps_site,
                batch_size=args.batch_size,
            )
            for path in paths
        ]
    else:
        rungs = [
            load_rung(path, integrand_site=args.integrand_site, eps_site=args.eps_site)
            for path in paths
        ]
    res = analyse(rungs)
    res["refinement"] = suggest_refinement(res, target_uncert=args.target_uncert)
    print_report(res)
    print(f"\n  next rungs to add: {res['refinement'].get('suggested_new_eps')}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)
        print(f"\n  wrote {args.out}")

    return 0 if res["reliable"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

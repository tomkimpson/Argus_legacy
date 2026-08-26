#!/usr/bin/env python
"""T3.4 — Bayesian evidence (logZ) by the learned harmonic mean (LHM) estimator,
reusing an existing NUTS posterior. Pure CPU post-processing of a results ``.nc``.

Why this exists
---------------
NUTS gives a posterior but no evidence. Nested sampling is parked as too slow
(``notes/DECISION_nested_sampling_parked.md``). This computes logZ from the NUTS
posterior we already pay for, via the re-targeted ("learned") harmonic mean of
McEwen et al. 2021 (arXiv:2111.12720), which fixes the classic harmonic mean's
infinite-variance pathology by importance-sampling against a normalised target
``phi`` that is *contained inside* the posterior bulk:

    1/Z = E_post[ phi(z) / (pi(z) L(z)) ],
    logZ = ln(N_test) - logsumexp_i[ ln phi(z_i) - ln(pi(z_i) L(z_i)) ].

Why it is directly comparable to the blackjax-NS anchor
-------------------------------------------------------
The NUTS run and ``run_blackjax_nested_sampling`` share the SAME ``numpyro_model``
(``bayesian_inference.py``), whose free sample sites are unit Gaussians
(``*_prime``/``*_raw``); the physical params are deterministic transforms. So:
  * the prior in latent space is an isotropic unit Gaussian (Jacobian-free), and
  * the ``.nc`` ``log_likelihood/likelihood`` group is exactly the Kalman
    ``numpyro.factor("likelihood", .)`` value per draw.
Working in that latent space makes this logZ directly comparable to the NS anchor
logZ = 63780.97 +/- 0.16 (``outputs/mdc2_blackjax_ns_demo/*_evidence.json``) with
no unit/Jacobian ambiguity. That anchor is the validation GATE (``--expected-logz``).

Target
------
phi = N(mu_hat, s^2 * Sigma_hat) fit on a TRAIN fold, with shrinkage 0<s<1 so phi
is strictly inside the posterior (lighter tails => finite, low variance). Full
covariance captures the curved log10_ha<->log10_gamma_a ridge. The estimate is
computed on a disjoint TEST fold. A shrinkage sweep + 2-fold-swap probe reliability.

Run (CPU, no GPU/JAX needed):
    JAX_PLATFORMS=cpu python workflows/ng15_sgwb_demo/scripts/logz_lhm.py \
        --results outputs/mdc2_smoke_wide/mdc2_smoke_wide_results.nc \
        --expected-logz 63780.97477 --expected-uncert 0.15854

Report a Bayes factor (two runs):
    ... --results outputs/ng15_real/ng15_real_results.nc \
        --compare-to outputs/ng15_curn/ng15_curn_results.nc
"""

import argparse
import json
import math

import arviz as az
import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.special import logsumexp

LN2PI = math.log(2.0 * math.pi)

# Reliability thresholds. These were already applied implicitly when choosing the
# reported shrinkage; they are named here so a run can say *which* check failed
# instead of returning a bare number (or a nan) that looks like a result.
MIN_ESS_FRACTION = 0.1  # effective sample size of the importance weights
MAX_WEIGHT_FRACTION = 0.1  # no single draw may dominate the importance sum
MAX_VOV_RELATIVE = 0.5  # McEwen's variance-of-variance relative error
MIN_HEALTHY_SHRINK_VALUES = 2  # a plateau needs at least two points
MAX_PLATEAU_SPREAD = 1.0  # nats, across the healthy shrinkage values
MAX_MATCHED_WEIGHT_FRACTION = 0.25  # looser, for the cancelling lnB difference


def assess_reliability(res):
    """Name the diagnostics a single-model LHM estimate fails, if any.

    The estimator is known to break down as dimension grows: at 68-D on MDC2 the
    importance weights collapse onto a single draw, no shrinkage value is healthy,
    and the reported logZ is meaningless. Before this gate existed that failure
    surfaced as a nan, which is indistinguishable at a glance from a bug. Now it
    surfaces as an explicit refusal naming the cause.
    """
    failures = []
    if not np.isfinite(res["log_Z_mean"]) or not np.isfinite(res["log_Z_uncert"]):
        failures.append("non_finite_estimate")
    if res["min_ess_frac"] < MIN_ESS_FRACTION:
        failures.append("effective_sample_size")
    if res["max_weight_frac"] > MAX_WEIGHT_FRACTION:
        failures.append("importance_weight_degeneracy")
    if res["max_vov_rel"] > MAX_VOV_RELATIVE:
        failures.append("variance_of_variance")
    if res["n_healthy_shrink"] < MIN_HEALTHY_SHRINK_VALUES:
        failures.append("no_shrinkage_plateau")
    elif (
        not np.isfinite(res["plateau_spread"])
        or res["plateau_spread"] > MAX_PLATEAU_SPREAD
    ):
        failures.append("plateau_not_flat")
    return failures


def load_latent_and_loglik(nc_path):
    """Load the unit-Gaussian latent vectors and per-draw log-likelihood from a ``.nc``.

    Returns
    -------
    z : ndarray, shape (n_chain, n_draw, ndim)
        Concatenated ``*_prime``/``*_raw`` sample sites in a fixed sorted key order.
    loglik : ndarray, shape (n_chain, n_draw)
        The ``log_likelihood/likelihood`` group (Kalman ``numpyro.factor`` value).
    names : list[str]
        The latent site names, in the column order used to build ``z``.
    """
    idata = az.from_netcdf(nc_path)
    post = idata.posterior
    # Latent sites of numpyro_model are exactly the reparameterised unit-Gaussian
    # variables; physical params (deterministics) carry no prior mass and are excluded.
    latent_vars = sorted(v for v in post.data_vars if v.endswith(("_prime", "_raw")))
    if not latent_vars:
        raise ValueError(f"No latent (*_prime / *_raw) sites found in {nc_path}")

    n_chain = post.sizes["chain"]
    n_draw = post.sizes["draw"]
    cols = []
    for v in latent_vars:
        arr = np.asarray(post[v].values)  # (chain, draw, *event)
        arr = arr.reshape(n_chain, n_draw, -1)  # scalars -> (.,.,1)
        cols.append(arr)
    z = np.concatenate(cols, axis=2)  # (chain, draw, ndim)

    if not hasattr(idata, "log_likelihood"):
        raise ValueError(f"{nc_path} has no log_likelihood group")
    loglik = np.asarray(idata.log_likelihood["likelihood"].values)  # (chain, draw)
    if loglik.shape != (n_chain, n_draw):
        raise ValueError(
            f"log_likelihood shape {loglik.shape} != posterior ({n_chain},{n_draw})"
        )
    return z, loglik, latent_vars


def log_prior_unit_gaussian(z):
    """log pi(z) for an isotropic unit Gaussian, z shape (..., ndim)."""
    ndim = z.shape[-1]
    return -0.5 * np.sum(z**2, axis=-1) - 0.5 * ndim * LN2PI


def fit_target(z_train, shrink, jitter=1e-8):
    """Fit phi = N(mu, s^2 Sigma) on the train fold. Returns (mu, chol_lower, logdet)."""
    ndim = z_train.shape[1]
    mu = np.mean(z_train, axis=0)
    if ndim == 1:
        cov = np.array([[np.var(z_train[:, 0], ddof=1)]])
    else:
        cov = np.cov(z_train, rowvar=False)
    cov = (shrink**2) * np.atleast_2d(cov) + jitter * np.eye(ndim)
    chol = cholesky(cov, lower=True)  # raises LinAlgError if not PD
    logdet = 2.0 * np.sum(np.log(np.diag(chol)))
    return mu, chol, logdet


def log_target(z, mu, chol, logdet):
    """ln phi(z) for phi = N(mu, Sigma) given lower-Cholesky, z shape (n, ndim)."""
    ndim = z.shape[1]
    diff = (z - mu).T  # (ndim, n)
    sol = solve_triangular(chol, diff, lower=True)  # (ndim, n)
    quad = np.sum(sol**2, axis=0)  # (n,)
    return -0.5 * quad - 0.5 * ndim * LN2PI - 0.5 * logdet


def lhm_estimate(z_train, z_test, logpost_test, shrink):
    """Learned-harmonic-mean logZ on one train/test split for a fixed shrinkage.

    Returns a dict with logZ, sigma (delta-method SE), ESS, max-weight fraction, and
    the variance-of-variance relative error (McEwen reliability check).
    """
    mu, chol, logdet = fit_target(z_train, shrink)
    logphi = log_target(z_test, mu, chol, logdet)
    lw = logphi - logpost_test  # ln r_i ; 1/Z = mean(exp(lw))
    n = lw.size

    A = logsumexp(lw)  # ln sum r_i
    logZ = math.log(n) - A  # ln N - logsumexp(lw)

    # Moments via a numerically-stable scaled copy r_s = exp(lw - c).
    c = float(np.max(lw))
    rs = np.exp(lw - c)
    sum_rs = rs.sum()
    ess = (sum_rs**2) / np.sum(rs**2)
    max_w = float(np.max(rs) / sum_rs)
    mean_rs = rs.mean()
    dev = rs - mean_rs
    m2 = np.mean(dev**2)
    m4 = np.mean(dev**4)
    kurt = m4 / m2**2 if m2 > 0 else np.inf
    vov_rel = math.sqrt(max(kurt - 1.0, 0.0) / n)  # rel. error on the variance itself

    # Delta-method SE on logZ = -ln(mean_r): SE(mean_r)/mean_r, scale-invariant.
    var_r_over_meanr2 = m2 / mean_rs**2  # = Var(r)/mean(r)^2 (scale cancels)
    sigma = math.sqrt(var_r_over_meanr2 / n)

    return {
        "logZ": logZ,
        "sigma": sigma,
        "ess": float(ess),
        "ess_frac": float(ess / n),
        "max_weight_frac": max_w,
        "vov_rel_error": vov_rel,
        "n_test": int(n),
        "shrink": shrink,
    }


def chain_folds(n_chain):
    """Two disjoint (train, test) chain-index splits (first/second half, then swapped)."""
    half = max(1, n_chain // 2)
    a = list(range(half))
    b = list(range(half, n_chain)) or [n_chain - 1]
    return [(a, b), (b, a)]


def combine_folds(fold_results):
    """Combine the two swapped-fold estimates: inverse-variance mean + swap spread."""
    logZs = np.array([r["logZ"] for r in fold_results])
    sig = np.array([r["sigma"] for r in fold_results])
    w = 1.0 / np.clip(sig, 1e-12, None) ** 2
    logZ_mean = float(np.sum(w * logZs) / np.sum(w))
    stat_err = float(1.0 / math.sqrt(np.sum(w)))
    swap_spread = float(np.max(logZs) - np.min(logZs))
    # Report the larger of the statistical error and half the swap spread.
    uncert = max(stat_err, swap_spread / 2.0)
    return logZ_mean, uncert, swap_spread


def run_one(nc_path, shrink_grid):
    """Full LHM analysis of one posterior: shrinkage sweep x 2-fold swap."""
    z, loglik, names = load_latent_and_loglik(nc_path)
    n_chain, n_draw, ndim = z.shape
    logpost = log_prior_unit_gaussian(z) + loglik  # (chain, draw)

    folds = chain_folds(n_chain)
    sweep = []
    for s in shrink_grid:
        fold_res = []
        for train_ch, test_ch in folds:
            z_tr = z[train_ch].reshape(-1, ndim)
            z_te = z[test_ch].reshape(-1, ndim)
            lp_te = logpost[test_ch].reshape(-1)
            fold_res.append(lhm_estimate(z_tr, z_te, lp_te, s))
        logZ_mean, uncert, swap_spread = combine_folds(fold_res)
        sweep.append(
            {
                "shrink": s,
                "logZ": logZ_mean,
                "uncert": uncert,
                "swap_spread": swap_spread,
                "min_ess_frac": min(r["ess_frac"] for r in fold_res),
                "max_weight_frac": max(r["max_weight_frac"] for r in fold_res),
                "max_vov_rel": max(r["vov_rel_error"] for r in fold_res),
                "folds": fold_res,
            }
        )

    # Plateau pick: among shrink values with healthy diagnostics, the logZ should be
    # flat. Choose the healthiest (highest min-ESS) as the reported estimate.
    healthy = [
        e for e in sweep if e["min_ess_frac"] >= 0.1 and e["max_weight_frac"] <= 0.1
    ]
    chosen = max(healthy or sweep, key=lambda e: e["min_ess_frac"])

    # Plateau flatness: spread of logZ across healthy shrink values.
    if len(healthy) >= 2:
        lz = np.array([e["logZ"] for e in healthy])
        plateau_spread = float(lz.max() - lz.min())
    else:
        plateau_spread = float("nan")

    result = {
        "results_path": nc_path,
        "ndim": int(ndim),
        "n_chain": int(n_chain),
        "n_draw": int(n_draw),
        "latent_sites": names,
        "log_Z_mean": chosen["logZ"],
        "log_Z_uncert": chosen["uncert"],
        "chosen_shrink": chosen["shrink"],
        "n_healthy_shrink": len(healthy),
        "plateau_spread": plateau_spread,
        "min_ess_frac": chosen["min_ess_frac"],
        "max_weight_frac": chosen["max_weight_frac"],
        "max_vov_rel": chosen["max_vov_rel"],
        "swap_spread": chosen["swap_spread"],
        "sweep": [{k: v for k, v in e.items() if k != "folds"} for e in sweep],
    }

    result["failed_diagnostics"] = assess_reliability(result)
    result["reliable"] = len(result["failed_diagnostics"]) == 0
    return result


def print_report(res):
    print(f"\n=== LHM logZ : {res['results_path']} ===")
    print(f"  ndim={res['ndim']}  chains={res['n_chain']}  draws/chain={res['n_draw']}")
    print(f"  latent sites: {', '.join(res['latent_sites'])}")
    print(
        f"\n  {'shrink':>7} {'logZ':>14} {'uncert':>9} {'swap':>8} "
        f"{'ESSfrac':>8} {'maxw':>7} {'vov':>7}"
    )
    for e in res["sweep"]:
        print(
            f"  {e['shrink']:7.2f} {e['logZ']:14.4f} {e['uncert']:9.4f} "
            f"{e['swap_spread']:8.4f} {e['min_ess_frac']:8.3f} "
            f"{e['max_weight_frac']:7.3f} {e['max_vov_rel']:7.3f}"
        )
    if res.get("reliable", True):
        print(
            f"\n  -> log_Z = {res['log_Z_mean']:.4f} +/- {res['log_Z_uncert']:.4f} "
            f"(shrink={res['chosen_shrink']:.2f}, {res['n_healthy_shrink']} healthy "
            f"shrink values, plateau spread={res['plateau_spread']:.4f})   [reliable]"
        )
    else:
        print(f"\n  -> NOT USABLE: failed " f"{', '.join(res['failed_diagnostics'])}")
        print(
            f"     (raw value {res['log_Z_mean']:.4f} +/- {res['log_Z_uncert']:.4f} "
            f"at shrink={res['chosen_shrink']:.2f} is reported for diagnosis only, "
            f"not as a result)"
        )


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--results", required=True, help="Path to results .nc (HD run).")
    p.add_argument(
        "--compare-to",
        default=None,
        help="Second results .nc (e.g. CURN) -> report lnB = logZ1 - logZ2.",
    )
    p.add_argument(
        "--shrink-grid",
        type=float,
        nargs="+",
        default=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    )
    p.add_argument(
        "--expected-logz",
        type=float,
        default=None,
        help="Anchor logZ for the validation GATE (prints PASS/FAIL).",
    )
    p.add_argument("--expected-uncert", type=float, default=0.0)
    p.add_argument(
        "--n-sigma",
        type=float,
        default=3.0,
        help="GATE tolerance in sigma (default 3).",
    )
    p.add_argument("--out", default=None, help="Write JSON summary here.")
    args = p.parse_args()

    res1 = run_one(args.results, args.shrink_grid)
    print_report(res1)

    payload = res1
    verdict = None

    if args.expected_logz is not None:
        d = res1["log_Z_mean"] - args.expected_logz
        combined = math.sqrt(res1["log_Z_uncert"] ** 2 + args.expected_uncert**2)
        tol = args.n_sigma * combined
        verdict = "PASS" if abs(d) <= tol else "FAIL"
        print(
            f"\n  GATE vs anchor {args.expected_logz:.4f} +/- {args.expected_uncert:.4f}:"
            f"  delta={d:+.4f}  tol(+/-{args.n_sigma:g}sigma)={tol:.4f}  -> {verdict}"
        )
        payload = {
            **res1,
            "gate": {
                "expected_logz": args.expected_logz,
                "expected_uncert": args.expected_uncert,
                "delta": d,
                "tol": tol,
                "n_sigma": args.n_sigma,
                "verdict": verdict,
            },
        }

    if args.compare_to is not None:
        res2 = run_one(args.compare_to, args.shrink_grid)
        print_report(res2)

        # Bayes factor at MATCHED shrinkage. When the two models share most
        # dimensions (here 16 of 18 red-noise/hierarchical params are common;
        # only the GWB params + the ORF differ), the LHM's per-model difficulty
        # is largely common and cancels in the difference, so lnB(s) is far more
        # stable than either absolute logZ. We therefore report lnB from the
        # matched-shrinkage plateau, not from two independently-chosen shrinks.
        sweep1 = {e["shrink"]: e["logZ"] for e in res1["sweep"]}
        sweep2 = {e["shrink"]: e["logZ"] for e in res2["sweep"]}
        maxw1 = {e["shrink"]: e["max_weight_frac"] for e in res1["sweep"]}
        maxw2 = {e["shrink"]: e["max_weight_frac"] for e in res2["sweep"]}
        print(f"\n  === Bayes factor (model1 vs model2), matched shrinkage ===")
        print(f"  {'shrink':>7} {'logZ1':>12} {'logZ2':>12} {'lnB':>9}")
        lnb_plateau = []
        for s in args.shrink_grid:
            b = sweep1[s] - sweep2[s]
            # A shrink is on the plateau only if BOTH models are non-degenerate
            # there (no single draw dominating the importance sum).
            healthy = (
                maxw1[s] <= MAX_MATCHED_WEIGHT_FRACTION
                and maxw2[s] <= MAX_MATCHED_WEIGHT_FRACTION
            )
            flag = "" if healthy else "  (degenerate, excluded)"
            if healthy:
                lnb_plateau.append(b)
            print(f"  {s:7.2f} {sweep1[s]:12.4f} {sweep2[s]:12.4f} {b:9.4f}{flag}")

        # Every shrinkage degenerate leaves nothing to average. Before this guard
        # that produced a bare nan from the mean of an empty array -- which is how
        # the 68-D MDC2 Stage C failure first surfaced. Refuse instead.
        lnb_failures = []
        if not lnb_plateau:
            lnb_failures.append("no_matched_shrinkage_plateau")
            lnB = float("nan")
            sigB = float("nan")
        else:
            lnb_arr = np.array(lnb_plateau)
            lnB = float(lnb_arr.mean())
            sigB = float(max(lnb_arr.std(), 0.02))  # plateau spread as the error
            if len(lnb_arr) < MIN_HEALTHY_SHRINK_VALUES:
                lnb_failures.append("matched_plateau_too_short")
            if not np.isfinite(lnB):
                lnb_failures.append("non_finite_estimate")

        # Both single-model estimates feed the difference. Their own degeneracies
        # largely cancel at matched shrinkage, which is the whole point of the
        # matched construction, so they are recorded but do not on their own veto
        # the Bayes factor.
        lnb_reliable = not lnb_failures

        if lnb_reliable:
            print(
                f"\n  lnB = logZ1 - logZ2 = {lnB:+.3f} +/- {sigB:.3f}  "
                f"(matched-shrinkage plateau over {len(lnb_plateau)} values)   "
                f"[reliable]"
            )
            print(f"  odds favouring model1 ~ e^lnB = {math.exp(lnB):.1f} : 1")
            print(
                f"  (Kass-Raftery 2lnB: <2 bare, 2-6 positive, 6-10 strong, "
                f">10 very strong; here 2lnB={2*lnB:.1f})"
            )
        else:
            print(f"\n  lnB NOT USABLE: failed {', '.join(lnb_failures)}")
            print(
                "     every shrinkage value was degenerate in at least one model, "
                "so there is no plateau to read a Bayes factor from. This is the "
                "expected high-dimensional failure of the learned harmonic mean; "
                "use the correlation-path estimators instead "
                "(lnb_savage_dickey.py / lnb_path_sampling.py)."
            )

        payload = {
            "model1_path": res1["results_path"],
            "model2_path": res2["results_path"],
            "lnB_mean": lnB,
            "lnB_uncert": sigB,
            "reliable": lnb_reliable,
            "failed_diagnostics": lnb_failures,
            "n_matched_plateau": len(lnb_plateau),
            "odds_model1": math.exp(lnB) if lnb_reliable else None,
            "matched_shrink_lnB": {
                str(s): sweep1[s] - sweep2[s] for s in args.shrink_grid
            },
            "model1": res1,
            "model2": res2,
        }

    if args.out:
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\n  wrote {args.out}")

    if verdict == "FAIL":
        raise SystemExit(1)
    if not payload.get("reliable", True):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

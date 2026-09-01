#!/usr/bin/env python
"""Extract Stage A single-pulsar posteriors into the Stage B/C input artifacts.

Reads every ``outputs/mdc2_stageA_<PSR>/mdc2_stageA_<PSR>_results.nc`` produced
by the Stage A array job (slurm_scripts/mdc2_stage_a.sh) and writes:

1. ``data/stage_a_medians.pkl`` — DataFrame(psr, optimal_sigma, optimal_gamma)
   with 10**median(log10_σp / log10_γp), rows sorted by pulsar name. This is
   the ``spin_injections_path`` pickle that FIXES red noise in Stage B
   (utils.get_psr_noise_injections preserves row order, and the data loader
   sorts its feather glob, so sorted rows are what keeps the arrays aligned).
2. ``data/stage_a_empirical_priors.json`` — per-pulsar Normal-prior (loc, scale)
   for log10_γp and log10_ratio = log10_σp − log10_γp, consumed by
   ``empirical_priors_path`` in Stage C (prior_models.get_empirical_noise_priors).
3. ``outputs/stage_a_summary/`` — a PASS/FAIL health table (markdown) and a
   summary plot of the per-pulsar posteriors with the MDC2 truth-file power-law
   red noise (rn_log10_A, rn_spec_ind) in a side panel. The truth is POWER-LAW
   while Argus fits an OU process, so that panel is qualitative only — γ/σ are
   effective parameters, not directly comparable.

Health gates per pulsar (the evidence mechanism for dropping e.g. J1640+2224):
r_hat < --rhat-max on log10_γp/log10_σp, ESS > --ess-min, posterior median more
than 2 posterior-std from both prior edges (not railed), no NaN. With
``--drop-failing`` the artifacts exclude failing pulsars and the matching
``excluded_psrs`` string for the Stage B/C configs is printed.

Pure CPU post-processing. Run from anywhere:

    JAX_PLATFORMS=cpu python workflows/ng15_sgwb_demo/scripts/extract_stage_a.py
"""

import argparse
import glob
import json
import os
import subprocess
from datetime import datetime, timezone

import arviz as az
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)  # workflows/ng15_sgwb_demo
REPO = os.path.dirname(os.path.dirname(ROOT))

# Default run-directory prefix. Overridable so a second dataset's Stage A runs can
# live alongside the first in outputs/ without their results being mixed together.
RUN_PREFIX = "mdc2_stageA_"

# Prior edges of configs/mdc2_stage_a.ini, used for the rail check.
DEFAULT_GAMMA_RANGE = (-12.0, -6.0)
DEFAULT_SIGMA_RANGE = (-20.0, -12.0)

DEFAULT_TRUTH = os.path.join(
    REPO, "workflows", "data", "IPTA_MockDataChallenge2", "group1_psr_noise.json"
)


def _git_sha():
    """Best-effort short git SHA of the repo for artifact provenance."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO, text=True
        ).strip()
    except Exception:
        return "unknown"


def load_run(nc_path):
    """Load one Stage A results file -> dict of flattened posterior draws + health.

    Parameters
    ----------
    nc_path : str
        Path to a ``*_results.nc`` ArviZ NetCDF.

    Returns
    -------
    dict
        Draws for log10_gamma, log10_sigma, log10_ratio plus r_hat/ess per
        parameter and the divergence fraction.
    """
    idata = az.from_netcdf(nc_path)
    post = idata.posterior

    log10_gamma = post["log10_γp"].values  # (chain, draw, 1)
    log10_sigma = post["log10_σp"].values
    # Per-draw ratio, NOT median-of-ratio-of-medians
    log10_ratio = log10_sigma - log10_gamma

    rhat = az.rhat(idata, var_names=["log10_γp", "log10_σp"])
    ess = az.ess(idata, var_names=["log10_γp", "log10_σp"])

    diverging = 0.0
    if hasattr(idata, "sample_stats") and "diverging" in idata.sample_stats:
        diverging = float(np.mean(idata.sample_stats["diverging"].values))

    return {
        "log10_gamma": log10_gamma.reshape(-1),
        "log10_sigma": log10_sigma.reshape(-1),
        "log10_ratio": log10_ratio.reshape(-1),
        "rhat_gamma": float(rhat["log10_γp"].values.max()),
        "rhat_sigma": float(rhat["log10_σp"].values.max()),
        "ess_gamma": float(ess["log10_γp"].values.min()),
        "ess_sigma": float(ess["log10_σp"].values.min()),
        "divergence_frac": diverging,
    }


def health_check(run, rhat_max, ess_min, gamma_range, sigma_range):
    """Apply the per-pulsar health gates. Returns (passed, list-of-failure-reasons)."""
    reasons = []

    for name in ("log10_gamma", "log10_sigma", "log10_ratio"):
        if not np.all(np.isfinite(run[name])):
            reasons.append(f"non-finite draws in {name}")

    for par in ("gamma", "sigma"):
        if run[f"rhat_{par}"] > rhat_max:
            reasons.append(f"r_hat({par}) = {run[f'rhat_{par}']:.3f} > {rhat_max}")
        if run[f"ess_{par}"] < ess_min:
            reasons.append(f"ESS({par}) = {run[f'ess_{par}']:.0f} < {ess_min}")

    for par, (lo, hi) in (("gamma", gamma_range), ("sigma", sigma_range)):
        draws = run[f"log10_{par}"]
        med, std = float(np.median(draws)), float(np.std(draws))
        if med - 2.0 * std < lo:
            reasons.append(f"log10_{par} railed at low prior edge {lo}")
        if med + 2.0 * std > hi:
            reasons.append(f"log10_{par} railed at high prior edge {hi}")

    return (len(reasons) == 0), reasons


def summary_plot(results, truth, out_png):
    """Per-pulsar (γ_p, σ_p) posteriors + qualitative power-law truth panel."""
    psrs = sorted(results)
    y = np.arange(len(psrs))

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(13, 0.28 * len(psrs) + 2),
        sharey=True,
        gridspec_kw={"width_ratios": [3, 3, 2]},
    )
    for ax, par, label in (
        (axes[0], "log10_gamma", r"$\log_{10}\gamma_p$"),
        (axes[1], "log10_sigma", r"$\log_{10}\sigma_p$"),
    ):
        med = [np.median(results[p][par]) for p in psrs]
        lo = [np.percentile(results[p][par], 16) for p in psrs]
        hi = [np.percentile(results[p][par], 84) for p in psrs]
        ax.errorbar(
            med,
            y,
            xerr=[np.array(med) - np.array(lo), np.array(hi) - np.array(med)],
            fmt="o",
            ms=3,
            lw=1,
            color="C0",
        )
        ax.set_xlabel(label)
        ax.grid(alpha=0.2)

    # Truth panel: power-law (rn_log10_A, rn_spec_ind) — QUALITATIVE only; the
    # fitted OU (γ, σ) are effective parameters of a different spectral model.
    ax = axes[2]
    rn_a = [truth[p]["rn_log10_A"] if p in truth else np.nan for p in psrs]
    ax.plot(rn_a, y, "s", ms=3, color="C3")
    ax.set_xlabel(r"truth $\log_{10}A_{\rm rn}$ (power-law)")
    ax.set_title(
        "power-law truth vs OU effective\nparams — qualitative only", fontsize=8
    )
    ax.grid(alpha=0.2)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(psrs, fontsize=6)
    fig.suptitle("M1 Stage A: single-pulsar OU red-noise posteriors (16–84%)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main():
    """Parse command-line arguments and build the Stage B/C artifacts."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results-dir", default=os.path.join(ROOT, "outputs"))
    ap.add_argument("--truth", default=DEFAULT_TRUTH)
    ap.add_argument(
        "--out-pickle", default=os.path.join(ROOT, "data", "stage_a_medians.pkl")
    )
    ap.add_argument(
        "--out-json",
        default=os.path.join(ROOT, "data", "stage_a_empirical_priors.json"),
    )
    ap.add_argument(
        "--report-dir", default=os.path.join(ROOT, "outputs", "stage_a_summary")
    )
    ap.add_argument(
        "--run-prefix",
        default=RUN_PREFIX,
        help="Prefix of the Stage A run directories to collect "
        f"(default: {RUN_PREFIX!r}).",
    )
    ap.add_argument("--rhat-max", type=float, default=1.05)
    ap.add_argument("--ess-min", type=float, default=200.0)
    ap.add_argument(
        "--gamma-range",
        nargs=2,
        type=float,
        default=list(DEFAULT_GAMMA_RANGE),
        metavar=("MIN", "MAX"),
        help="log10_gamma_p prior edges (rail check)",
    )
    ap.add_argument(
        "--sigma-range",
        nargs=2,
        type=float,
        default=list(DEFAULT_SIGMA_RANGE),
        metavar=("MIN", "MAX"),
        help="log10_sigma_p prior edges (rail check)",
    )
    ap.add_argument(
        "--drop-failing",
        action="store_true",
        help="exclude health-gate failures from the output artifacts "
        "(default: any failure aborts without writing)",
    )
    args = ap.parse_args()

    run_dirs = sorted(glob.glob(os.path.join(args.results_dir, args.run_prefix + "*")))
    run_dirs = [d for d in run_dirs if os.path.isdir(d)]
    if not run_dirs:
        raise SystemExit(
            f"No {args.run_prefix}* run directories under {args.results_dir} — has the "
            "Stage A array job finished?"
        )

    with open(args.truth) as f:
        truth = json.load(f)

    results, missing = {}, []
    for run_dir in run_dirs:
        tag = os.path.basename(run_dir)
        psr = tag[len(args.run_prefix) :]
        nc_path = os.path.join(run_dir, f"{tag}_results.nc")
        if not os.path.exists(nc_path):
            missing.append(psr)
            continue
        results[psr] = load_run(nc_path)

    if missing:
        print(
            f"WARNING: {len(missing)} run(s) with no results.nc: {', '.join(missing)}"
        )

    # --- health gates ---
    failures = {}
    print(
        f"\n{'pulsar':<14} {'rhat_g':>7} {'rhat_s':>7} {'ess_g':>7} {'ess_s':>7} "
        f"{'div%':>6}  verdict"
    )
    print("-" * 72)
    for psr in sorted(results):
        run = results[psr]
        ok, reasons = health_check(
            run,
            args.rhat_max,
            args.ess_min,
            tuple(args.gamma_range),
            tuple(args.sigma_range),
        )
        if not ok:
            failures[psr] = reasons
        print(
            f"{psr:<14} {run['rhat_gamma']:>7.3f} {run['rhat_sigma']:>7.3f} "
            f"{run['ess_gamma']:>7.0f} {run['ess_sigma']:>7.0f} "
            f"{100 * run['divergence_frac']:>5.1f}%  "
            f"{'PASS' if ok else 'FAIL: ' + '; '.join(reasons)}"
        )

    if failures and not args.drop_failing:
        raise SystemExit(
            f"\n{len(failures)} pulsar(s) failed the health gates (see table). "
            "Re-run with --drop-failing to write artifacts without them, or fix "
            "the failing runs first."
        )

    kept = sorted(p for p in results if p not in failures)
    if failures:
        dropped = sorted(failures)
        print(f"\nDropping {len(dropped)} pulsar(s): {', '.join(dropped)}")
        print("Set this in the Stage B/C configs ([Data] section):")
        print(f"  excluded_psrs = {','.join(dropped)}")

    # --- markdown report ---
    os.makedirs(args.report_dir, exist_ok=True)
    table_path = os.path.join(args.report_dir, "stage_a_health.md")
    with open(table_path, "w") as f:
        f.write("# M1 Stage A health table\n\n")
        f.write(
            f"Generated {datetime.now(timezone.utc).isoformat()} "
            f"(git {_git_sha()})\n\n"
        )
        f.write(
            "| pulsar | rhat γ | rhat σ | ESS γ | ESS σ | div % | "
            "median log10γ | median log10σ | verdict |\n"
        )
        f.write("|---|---|---|---|---|---|---|---|---|\n")
        for psr in sorted(results):
            run = results[psr]
            verdict = (
                "PASS" if psr not in failures else "FAIL: " + "; ".join(failures[psr])
            )
            f.write(
                f"| {psr} | {run['rhat_gamma']:.3f} | {run['rhat_sigma']:.3f} "
                f"| {run['ess_gamma']:.0f} | {run['ess_sigma']:.0f} "
                f"| {100 * run['divergence_frac']:.1f} "
                f"| {np.median(run['log10_gamma']):.2f} "
                f"| {np.median(run['log10_sigma']):.2f} | {verdict} |\n"
            )
        if missing:
            f.write(f"\nMissing results: {', '.join(missing)}\n")

    summary_plot(results, truth, os.path.join(args.report_dir, "stage_a_summary.png"))

    # --- Stage B pickle: rows sorted by pulsar name (alignment contract) ---
    df = pd.DataFrame(
        {
            "psr": kept,
            "optimal_sigma": [
                10.0 ** float(np.median(results[p]["log10_sigma"])) for p in kept
            ],
            "optimal_gamma": [
                10.0 ** float(np.median(results[p]["log10_gamma"])) for p in kept
            ],
        }
    )
    os.makedirs(os.path.dirname(args.out_pickle), exist_ok=True)
    df.to_pickle(args.out_pickle)

    # --- Stage C empirical priors JSON ---
    priors = {
        "_meta": {
            "generated": datetime.now(timezone.utc).isoformat(),
            "git_sha": _git_sha(),
            "source_runs": [args.run_prefix + p for p in kept],
            "dropped": sorted(failures),
            "note": "loc/scale are posterior mean/std of the Stage A single-pulsar "
            "runs; scales are inflated at load time via empirical_prior_inflation",
        }
    }
    for psr in kept:
        run = results[psr]
        priors[psr] = {
            "log10_gamma_p": {
                "loc": float(np.mean(run["log10_gamma"])),
                "scale": float(np.std(run["log10_gamma"])),
            },
            "log10_ratio": {
                "loc": float(np.mean(run["log10_ratio"])),
                "scale": float(np.std(run["log10_ratio"])),
            },
        }
    with open(args.out_json, "w") as f:
        json.dump(priors, f, indent=2)

    print(f"\nKept {len(kept)}/{len(results)} pulsars.")
    print(f"Stage B pickle          -> {args.out_pickle}")
    print(f"Stage C empirical JSON  -> {args.out_json}")
    print(f"Health table            -> {table_path}")
    print(
        f"Summary plot            -> {os.path.join(args.report_dir, 'stage_a_summary.png')}"
    )


if __name__ == "__main__":
    main()

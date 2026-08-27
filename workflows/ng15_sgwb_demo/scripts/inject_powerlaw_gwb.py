#!/usr/bin/env python
"""Inject a synthetic SGWB (+ optional red noise + white noise) into the aligned NG15 feathers.

This is the Stage-2 de-risking injector (TASKS.md T2.1). Argus recovers the
gravitational-wave background (GWB) with a **single-corner Ornstein-Uhlenbeck (OU)**
model -- there is no power-law spectral model anywhere in the code. Before spending
production GPU time on real data we need to know: *can that OU model absorb a signal of a
different spectral shape without biasing the recovered amplitude?* This script builds the
synthetic datasets that test it, on the real NG15 sampling geometry (the 6 epoch-aligned
feathers from T1.5: 6 pulsars x 78 epochs on a shared ~30-day grid).

Two injection modes:

* ``--mode powerlaw`` (the real test): an HD-correlated **true power law**
  ``P(f) = A^2/(12 pi^2) (f/f_yr)^-gamma f_yr^-3`` (``log10_A_gw=-14.6``, ``gamma=13/3``),
  generated as a frequency-domain Fourier-sum Gaussian process. This is a spectral shape
  the OU recovery model *cannot* represent exactly -- the point of the gate.

* ``--mode ou`` (the control): a GWB drawn from Argus's **own** OU generative model
  (``jax_kalman_filter._compute_sigma_matrix`` + ``model.get_F_block``/``get_Q_block``),
  so an OU->OU recovery is self-consistent and should return the injected truth.

**Framing (this is not a claim that the universe's SGWB is a power law).** Neither shape is
"the truth": the power law is the field's reference parameterization (and how NG15 quotes its
headline amplitude); an OU turnover is at least as physically motivated (environmental
coupling -> low-frequency turnover). A PTA only constrains the spectrum over ~1 decade
(~2-60 nHz), so the robust observable is the band-referenced amplitude at a pivot frequency,
not the spectral index. The injector therefore records the injected cross-PSD at the sampling
frequencies (plus pivot values at ``f=1/yr`` and a band-centroid) into ``injection_truth.json``
so the downstream comparison (T2.4/T4.1) is shape-agnostic. ``log10_A_gw``/``gamma`` are
secondary metadata.

The injected residual is a **pure synthetic** series (GWB + optional per-pulsar red noise +
optional white noise) that *replaces* the real residual; every other feather field (TOAs,
errors, raw design matrix, RA/DEC, F0, distance) is kept identical, so the stock GWB loader
sees the exact real geometry (same ``Gamma_HD``, same timing-model marginalization).

Defaults (all overridable on the CLI):
  * red noise OFF   -- isolates the pure OU-vs-power-law GWB question for the gate run.
                       ``--red-noise`` needs explicit per-pulsar (log10_gamma_p, log10_sigma_p),
                       which are NOT in ng15_psr_noise.json.
  * white noise ON  -- drawn from ng15_psr_noise.json (R=(efac*sigma)^2+equad^2), matching the
                       fixed recovery noise. ``--no-white-noise`` to disable.
  * OU control amp  -- log10_gamma_a=-8.5, log10_ha=-14.35 (approximately matches the power-law
                       injection variance over this baseline); nudge by comparing printed RMS.

CPU only, numpy only (the F/Q blocks are reimplemented in numpy: the JAX versions in
``model.py`` silently return float32 unless x64 is enabled). No library edits.

Examples
--------
    JAX_PLATFORMS=cpu python workflows/ng15_sgwb_demo/scripts/inject_powerlaw_gwb.py \
        --mode powerlaw --overwrite
    JAX_PLATFORMS=cpu python workflows/ng15_sgwb_demo/scripts/inject_powerlaw_gwb.py \
        --mode ou --overwrite
"""

import argparse
import glob
import json
import os
import sys
import types

import numpy as np

# Add the repo's python/ dir to sys.path so ``argus`` imports standalone (mirrors the
# sibling data-prep scripts). This file is workflows/ng15_sgwb_demo/scripts/, so walk up
# four dirnames to reach the repo root.
_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.append(os.path.join(_PROJECT_ROOT, "python"))

from argus import gravitational_waves  # noqa: E402
from argus.data_loader import LoadWidebandPulsarData  # noqa: E402

SEC_PER_DAY = 86400.0
DAYS_PER_YEAR = 365.25
SEC_PER_YEAR = DAYS_PER_YEAR * SEC_PER_DAY
F_YR = (
    1.0 / SEC_PER_YEAR
)  # Julian-year reference frequency, matches enterprise const.fyr

EXPECTED_NEPOCH = 78
EXPECTED_NPSR = 6

# Default injected power-law (the NANOGrav 15yr reference SGWB).
DEFAULT_LOG10_A_GW = -14.6
DEFAULT_GAMMA = 13.0 / 3.0
DEFAULT_N_FREQ = 30

# Default OU-control truth: gamma_a corner tau_c = 1/gamma_a ~ 10 yr (interior to the
# ~(-10,-8) log10_gamma_a prior); log10_ha chosen to roughly match the power-law variance
# over this ~6.3 yr baseline. Nudge by comparing the printed per-pulsar RMS of both modes.
DEFAULT_LOG10_HA = -14.35
DEFAULT_LOG10_GAMMA_A = -8.5

DEFAULT_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
)
DEFAULT_ALIGNED_DIR = os.path.join(DEFAULT_DATA_DIR, "aligned")
DEFAULT_NOISE_JSON = os.path.join(DEFAULT_DATA_DIR, "ng15_psr_noise.json")


# --------------------------------------------------------------------------------------
# numpy reimplementations of the OU state-space blocks (exact copies of model.py:17-61,
# which are JAX and would return float32 without jax_enable_x64).
# --------------------------------------------------------------------------------------
def f_block_np(gamma, dt):
    """2x2 OU state-transition block (numpy port of ``model.get_F_block``)."""
    neg = -gamma * dt
    f12 = -np.expm1(neg) / gamma
    return np.array([[1.0, f12], [0.0, np.exp(neg)]])


def q_block_np(gamma, dt):
    """2x2 unit-noise OU process-noise block (numpy port of ``model.get_Q_block``).

    Matches the corrected ``get_Q_block`` (q11 divided by ``gamma**2`` -- the exact
    integrated-OU position variance ``int_0^dt [(1-e^{-g t})/g]^2 dt``).
    """
    neg = -gamma * dt
    neg2 = -2.0 * gamma * dt
    ome = -np.expm1(neg)
    ome2 = -np.expm1(neg2)
    q11 = (dt - 2.0 * ome / gamma + ome2 / (2.0 * gamma)) / gamma**2
    q12 = (ome - ome2 / 2.0) / gamma**2
    q22 = ome2 / (2.0 * gamma)
    return np.array([[q11, q12], [q12, q22]])


# --------------------------------------------------------------------------------------
# Loading / HD
# --------------------------------------------------------------------------------------
def load_aligned(aligned_dir):
    """Load every aligned ``*.feather`` in loader (``sorted(glob)``) order.

    Returns
    -------
    dict with keys:
        objs   : list of LoadWidebandPulsarData (one per pulsar, sorted-glob order)
        names  : list of pulsar names in the same order
        toas   : list of (nepoch,) TOA arrays (seconds), one per pulsar
        errs   : list of (nepoch,) TOA-error arrays (seconds), one per pulsar
        f0     : list of spin frequencies (Hz) or None
        Gamma  : (Npsr, Npsr) Hellings-Downs matrix (unit diagonal), same order
    """
    files = sorted(glob.glob(os.path.join(aligned_dir, "*.feather")))
    if not files:
        raise FileNotFoundError(
            f"No *.feather in {aligned_dir}; run build_aligned_feathers.py first (T1.5)."
        )
    objs = [LoadWidebandPulsarData.read_feather(f) for f in files]

    nepochs = {int(np.asarray(o.toas).size) for o in objs}
    if len(nepochs) != 1:
        raise SystemExit(
            f"*** Aligned feathers are ragged ({sorted(nepochs)} TOAs); the GWB path needs "
            "an identical epoch count per pulsar. Re-run build_aligned_feathers.py. ***"
        )

    names = [o.name for o in objs]
    toas = [np.asarray(o.toas, dtype=float) for o in objs]
    errs = [np.asarray(o.toaerrs, dtype=float) for o in objs]
    f0 = [getattr(o, "F0", None) for o in objs]

    ra = np.array([float(o.RA) for o in objs])
    dec = np.array([float(o.DEC) for o in objs])
    theta = gravitational_waves.pairwise_angular_separation(ra, dec)
    Gamma = np.asarray(gravitational_waves.hellings_downs(theta), dtype=float)

    return {
        "objs": objs,
        "names": names,
        "toas": toas,
        "errs": errs,
        "f0": f0,
        "Gamma": Gamma,
    }


def hd_cholesky(Gamma):
    """Return a lower-triangular square root of the HD matrix, with a PD guard.

    The Hellings-Downs ORF can be numerically indefinite; jitter the diagonal if the
    smallest eigenvalue is non-positive so the Cholesky (used for every correlated draw)
    does not crash. Returns ``(L, min_eig)``.
    """
    min_eig = float(np.linalg.eigvalsh(Gamma).min())
    G = Gamma
    if min_eig <= 0:
        G = Gamma + (abs(min_eig) + 1e-12) * np.eye(Gamma.shape[0])
    return np.linalg.cholesky(G), min_eig


# --------------------------------------------------------------------------------------
# Signal generators (each returns a (Npsr, nepoch) residual matrix, seconds)
# --------------------------------------------------------------------------------------
def powerlaw_psd(freqs, log10_A, gamma):
    """Residual power-law PSD ``P(f)=A^2/(12 pi^2)(f/f_yr)^-gamma f_yr^-3`` [s^3]."""
    A = 10.0**log10_A
    return (A**2 / (12.0 * np.pi**2)) * (freqs / F_YR) ** (-gamma) * F_YR**-3


def ou_psd(freqs, log10_ha, log10_gamma_a):
    """OU residual PSD ``S_r(f)=sigma_a2/((2 pi f)^2 (gamma_a^2+(2 pi f)^2))`` [s^3].

    ``sigma_a2=(ha^2/12) gamma_a``. This is the same quantity ``powerlaw_psd`` returns for
    the other shape, which is the point: a PTA constrains the spectrum over about a decade,
    so the comparable observable between an OU and a power-law injection is the PSD at a
    pivot frequency, not the spectral index. Recording it for both makes the downstream
    comparison shape-agnostic.
    """
    w = 2.0 * np.pi * np.asarray(freqs, dtype=float)
    gamma_a = 10.0**log10_gamma_a
    sigma_a2 = (10.0**log10_ha) ** 2 / 12.0 * gamma_a
    return sigma_a2 / (w**2 * (gamma_a**2 + w**2))


def inject_powerlaw_gwb(toas, freqs, log10_A, gamma, L_hd, t0, rng):
    """HD-correlated true-power-law GWB via a frequency-domain Fourier-sum GP.

    For each frequency ``f_k`` draw sin/cos coefficient 6-vectors, each
    ``~ N(0, (P(f_k)/T) Gamma_HD)`` (sin and cos get equal power, per enterprise), where
    ``T = 1/freqs[0]`` is the baseline. Each pulsar's residual is evaluated at its *own*
    TOAs: ``r_n(t) = sum_k [a_k[n] sin(2 pi f_k (t-t0)) + b_k[n] cos(2 pi f_k (t-t0))]``.

    Parameters
    ----------
    toas : list of (nepoch,) arrays, seconds (one per pulsar)
    freqs : (N_f,) sampling frequencies, Hz
    L_hd : (Npsr, Npsr) lower-triangular HD square root
    t0 : float, global time origin subtracted before the trig (numerical hygiene)
    rng : numpy Generator

    Returns
    -------
    (Npsr, nepoch) residual matrix (seconds) and the injected (N_f,) PSD P(f_k).
    """
    npsr = L_hd.shape[0]
    T = 1.0 / freqs[0]
    P = powerlaw_psd(freqs, log10_A, gamma)  # (N_f,)
    scale = np.sqrt(P / T)  # per-mode coeff std, (N_f,)

    # Correlated coefficients: a = scale * (Z @ L^T) gives Cov(a_k) = (P_k/T) Gamma.
    Za = rng.standard_normal((freqs.size, npsr))
    Zb = rng.standard_normal((freqs.size, npsr))
    a = scale[:, None] * (Za @ L_hd.T)  # (N_f, Npsr)
    b = scale[:, None] * (Zb @ L_hd.T)

    res = np.empty((npsr, toas[0].size))
    two_pi_f = 2.0 * np.pi * freqs  # (N_f,)
    for n in range(npsr):
        ang = np.outer(two_pi_f, toas[n] - t0)  # (N_f, nepoch)
        res[n] = a[:, n] @ np.sin(ang) + b[:, n] @ np.cos(ang)
    return res, P


def inject_ou_gwb(t_shared, log10_ha, log10_gamma_a, Gamma, L_hd, rng):
    """HD-correlated OU GWB, forward-simulating Argus's own generative model.

    State is interleaved ``[r_0, a_0, r_1, a_1, ...]`` (r even, a odd), matching
    ``F_gw = kron(I, F_block(gamma_a))`` and ``Q_gw = kron(sigma_a2, Q_block(gamma_a))``.
    ``sigma_a2 = (ha^2/12) gamma_a Gamma`` and the initial a-state covariance is
    ``ha^2 Gamma / 24`` (``jax_kalman_filter._initialize_kalman_filter``). The residual is
    ``-r`` (the GW state enters the observation with coefficient -1).

    Uses a single ``dt`` per step from the shared (mean) epoch grid, exactly as the joint
    filter does. Noise is drawn via the Kronecker square root
    ``w = kron(L_A, L_B) z`` with ``L_A = chol(sigma_a2)``, ``L_B = chol(Q_block)``.
    """
    npsr = Gamma.shape[0]
    ha = 10.0**log10_ha
    gamma_a = 10.0**log10_gamma_a
    sigma_a2_scale = ha**2 / 12.0 * gamma_a  # sigma_a2 = scale * Gamma
    L_A = np.sqrt(sigma_a2_scale) * L_hd  # chol((ha^2/12) gamma_a Gamma)

    nepoch = t_shared.size
    dt_array = np.diff(t_shared)

    # State vector (2*Npsr,): r at even indices, a at odd indices.
    x = np.zeros(2 * npsr)
    a0 = (np.sqrt(ha**2 / 24.0) * L_hd) @ rng.standard_normal(
        npsr
    )  # ~N(0, ha^2 Gamma/24)
    x[1::2] = a0

    r = np.empty((nepoch, npsr))
    r[0] = x[0::2]
    for k, dt in enumerate(dt_array):
        Fb = f_block_np(gamma_a, dt)  # (2,2)
        Lb = np.linalg.cholesky(q_block_np(gamma_a, dt))  # (2,2)
        # F_full = kron(I, Fb): apply Fb to each pulsar's (r,a) pair.
        x2 = x.reshape(npsr, 2) @ Fb.T  # (Npsr, 2)
        # w = kron(L_A, Lb) z: draw z as (Npsr,2), form Lb z^T then mix pulsars by L_A.
        z = rng.standard_normal((npsr, 2))
        w = L_A @ (z @ Lb.T)  # (Npsr, 2) = L_A[n,m] (Lb z_m)
        x = (x2 + w).reshape(2 * npsr)
        r[k + 1] = x[0::2]
    return -r.T  # (Npsr, nepoch)


def inject_red_noise(t_shared, gamma_p, sigma_p, f0, rng):
    """Per-pulsar OU spin red noise (uncorrelated across pulsars); residual += dphi/f0.

    Same OU form as the GW state but on the spin block ``(dphi, dphi_dot=df)``:
    ``d(df) = -gamma_p df dt + chi_p``, ``<chi_p^2> = sigma_p^2``, entering the residual as
    ``+dphi/f0``. Initial ``df ~ N(0, sigma_p^2/(2 gamma_p))``, ``dphi ~ 0``.
    """
    npsr = len(gamma_p)
    nepoch = t_shared.size
    dt_array = np.diff(t_shared)
    res = np.zeros((npsr, nepoch))
    for n in range(npsr):
        if f0[n] is None:
            raise SystemExit(
                f"*** Red noise requested but pulsar index {n} has no F0; cannot map "
                "dphi -> residual. ***"
            )
        g, s2 = gamma_p[n], sigma_p[n] ** 2
        x = np.array(
            [0.0, rng.standard_normal() * np.sqrt(s2 / (2.0 * g))]
        )  # (dphi, df)
        dphi = np.empty(nepoch)
        dphi[0] = x[0]
        for k, dt in enumerate(dt_array):
            Fb = f_block_np(g, dt)
            Lb = np.linalg.cholesky(q_block_np(g, dt) * s2)
            x = Fb @ x + Lb @ rng.standard_normal(2)
            dphi[k + 1] = x[0]
        res[n] = dphi / f0[n]
    return res


def inject_white_noise(errs, efac, equad, rng):
    """Per-TOA white noise ``n ~ N(0, (efac*sigma)^2 + equad^2)`` per pulsar."""
    res = []
    for n, sigma in enumerate(errs):
        var = (efac[n] * sigma) ** 2 + equad[n] ** 2
        res.append(rng.standard_normal(sigma.size) * np.sqrt(var))
    return res  # list of (nepoch,) arrays


# --------------------------------------------------------------------------------------
# White-noise params (indexed by name -- NOT positional; see PLAN pitfall #1)
# --------------------------------------------------------------------------------------
def load_white_noise(noise_json, names):
    """Return (efac, equad) arrays aligned to ``names`` order (equad delogged to seconds).

    ``ng15_psr_noise.json`` stores ``equad`` as log10 seconds and its key order is not
    guaranteed to match the sorted-glob pulsar order, so index by name explicitly.
    """
    with open(noise_json) as f:
        params = json.load(f)
    efac, equad = [], []
    for name in names:
        if name not in params:
            raise KeyError(
                f"{name} not in {noise_json}; keys={sorted(params)}. "
                "White-noise injection needs an entry per pulsar (or use --no-white-noise)."
            )
        efac.append(float(params[name]["efac"]))
        equad.append(10.0 ** float(params[name]["equad"]))
    return np.array(efac), np.array(equad)


# --------------------------------------------------------------------------------------
# Write / verify / truth sidecar
# --------------------------------------------------------------------------------------
def write_injected_feathers(objs, residuals, out_dir, overwrite):
    """Round-trip each pulsar with its injected residual (build_aligned_object pattern).

    Keeps every real field (TOAs, errors, raw design matrix, RA/DEC, distance, F0) and only
    swaps ``residuals``, so ``M_scaled``/``P_eps`` are recomputed identically.
    """
    os.makedirs(out_dir, exist_ok=True)
    for obj, res in zip(objs, residuals):
        path = os.path.join(out_dir, f"{obj.name}.feather")
        if os.path.exists(path) and not overwrite:
            raise SystemExit(f"*** {path} exists; pass --overwrite to replace it. ***")
        ns = types.SimpleNamespace(
            toas=np.asarray(obj.toas, dtype=float),
            toaerrs=np.asarray(obj.toaerrs, dtype=float),
            residuals=np.asarray(res, dtype=float),
            fitpars=obj.fitpars,
            Mmat=np.asarray(obj.M_matrix, dtype=float),
            name=obj.name,
            _raj=obj.RA,
            _decj=obj.DEC,
            _pdist=(obj.distance_kpc, obj.distance_err_kpc),
        )
        new = LoadWidebandPulsarData(ns)
        new.save_feather(path, F0=getattr(obj, "F0", None))
        print(f"  wrote {path}")


def write_truth_sidecar(out_dir, truth):
    """Write ``injection_truth.json`` (the raw ingredients for the shape-agnostic compare)."""
    path = os.path.join(out_dir, "injection_truth.json")
    with open(path, "w") as f:
        json.dump(truth, f, indent=2)
    print(f"  wrote {path}")


def verify(out_dir, n_pulsars):
    """Assert the stock GWB path consumes the injected feathers cleanly (T2.1 gate)."""
    print("\nVerifying get_processed_residuals(mode='gwb') on the injected dir ...")
    data = LoadWidebandPulsarData.get_processed_residuals(out_dir, mode="gwb")
    pr = data["processed_residuals"]
    res = np.asarray(pr["residuals"])
    errs = np.asarray(pr["errors"])
    hd = np.asarray(data["hd_correlation"])
    nepoch = res.shape[0]
    assert res.shape == (nepoch, n_pulsars), f"residuals shape {res.shape}"
    assert errs.shape == (nepoch, n_pulsars), f"errors shape {errs.shape}"
    assert hd.shape == (n_pulsars, n_pulsars), f"hd shape {hd.shape}"
    assert np.allclose(np.diag(hd), 1.0), "HD diagonal is not unit"
    assert np.all(np.isfinite(res)) and np.all(np.isfinite(errs)), "non-finite output"
    print(
        f"  OK: residuals {res.shape}, errors {errs.shape}, HD {hd.shape} (unit diagonal)."
    )


def print_rms(names, residuals):
    """Print per-pulsar injected residual RMS (a physical sanity check)."""
    print("\nInjected residual RMS per pulsar:")
    for name, res in zip(names, residuals):
        rms = float(np.sqrt(np.mean(np.asarray(res) ** 2)))
        print(f"  {name:<12} {rms * 1e9:8.1f} ns  ({rms * 1e6:.4f} us)")


# --------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------
def run(args):
    data = load_aligned(args.aligned_dir)
    objs, names, toas, errs, f0, Gamma = (
        data["objs"],
        data["names"],
        data["toas"],
        data["errs"],
        data["f0"],
        data["Gamma"],
    )
    npsr = len(objs)
    nepoch = toas[0].size
    print(f"Loaded {npsr} pulsars ({nepoch} epochs each): {', '.join(names)}")
    if npsr != EXPECTED_NPSR or nepoch != EXPECTED_NEPOCH:
        print(
            f"  NOTE: expected {EXPECTED_NPSR} pulsars x {EXPECTED_NEPOCH} epochs "
            f"(T1.5); got {npsr} x {nepoch}."
        )

    L_hd, min_eig = hd_cholesky(Gamma)
    print(
        f"HD matrix min eigenvalue = {min_eig:.3e} "
        f"({'PD' if min_eig > 0 else 'jittered to PD'})"
    )

    # Shared (mean) epoch grid -> single dt sequence for the OU forward-sim; global origin
    # + baseline for the power-law Fourier grid.
    toa_stack = np.vstack(toas)  # (Npsr, nepoch)
    t_shared = toa_stack.mean(axis=0)  # (nepoch,)
    t0 = float(toa_stack.min())
    T_span = float(toa_stack.max() - toa_stack.min())
    print(
        f"Baseline T_span = {T_span / SEC_PER_YEAR:.2f} yr; "
        f"f_1 = {1.0 / T_span:.3e} Hz"
    )

    rng = np.random.default_rng(args.seed)

    truth = {
        "mode": args.mode,
        "seed": args.seed,
        "pulsars": names,
        "nepoch": nepoch,
        "T_span_yr": T_span / SEC_PER_YEAR,
        "f_yr_hz": F_YR,
        "red_noise": bool(args.red_noise),
        "white_noise": not args.no_white_noise,
    }

    # ---- GWB signal ----
    if args.mode == "powerlaw":
        freqs = np.arange(1, args.n_freq + 1) / T_span
        gwb, P = inject_powerlaw_gwb(
            toas, freqs, args.log10_A_gw, args.gamma, L_hd, t0, rng
        )
        f_band = 1.0 / (5.0 * SEC_PER_YEAR)  # band-centroid pivot (~1/5yr, in-band)
        truth.update(
            {
                "log10_A_gw": args.log10_A_gw,
                "gamma": args.gamma,
                "n_freq": args.n_freq,
                "freqs_hz": freqs.tolist(),
                "psd_at_freqs_s3": P.tolist(),
                "pivot_psd_s3": {
                    "f_yr": float(
                        powerlaw_psd(np.array([F_YR]), args.log10_A_gw, args.gamma)[0]
                    ),
                    "f_band_1over5yr": float(
                        powerlaw_psd(np.array([f_band]), args.log10_A_gw, args.gamma)[0]
                    ),
                },
            }
        )
        out_dir = args.out_dir or os.path.join(DEFAULT_DATA_DIR, "inject_powerlaw")
    else:  # ou
        gwb = inject_ou_gwb(
            t_shared, args.log10_ha, args.log10_gamma_a, Gamma, L_hd, rng
        )
        sigma_a2_diag = (10.0**args.log10_ha) ** 2 / 12.0 * 10.0**args.log10_gamma_a
        f_band = 1.0 / (5.0 * SEC_PER_YEAR)  # band-centroid pivot (~1/5yr, in-band)
        truth.update(
            {
                "log10_ha": args.log10_ha,
                "log10_gamma_a": args.log10_gamma_a,
                "sigma_a2_diag": sigma_a2_diag,
                # Recorded for the OU shape too, in the same units and at the same
                # pivots as the power-law branch. Without this the two injections of a
                # matched pair cannot be compared at all: their native parameters
                # (log10_A, gamma) and (log10_ha, log10_gamma_a) do not correspond, and
                # the band-referenced PSD is the only common ground.
                "pivot_psd_s3": {
                    "f_yr": float(
                        ou_psd(np.array([F_YR]), args.log10_ha, args.log10_gamma_a)[0]
                    ),
                    "f_band_1over5yr": float(
                        ou_psd(np.array([f_band]), args.log10_ha, args.log10_gamma_a)[0]
                    ),
                },
                "note": "OU residual PSD S_r(f)=sigma_a2_diag/((2 pi f)^2 (gamma_a^2+(2 pi f)^2)); "
                "sigma_a2_diag=(ha^2/12) gamma_a.",
            }
        )
        out_dir = args.out_dir or os.path.join(DEFAULT_DATA_DIR, "inject_ou")

    residuals = [gwb[n].copy() for n in range(npsr)]

    # ---- optional per-pulsar red noise ----
    if args.red_noise:
        if args.log10_gamma_p is None or args.log10_sigma_p is None:
            raise SystemExit(
                "*** --red-noise needs --log10-gamma-p and --log10-sigma-p (one value, "
                "broadcast to all pulsars, or N comma-separated values). ***"
            )
        gamma_p = _broadcast(args.log10_gamma_p, npsr, "log10-gamma-p")
        sigma_p = _broadcast(args.log10_sigma_p, npsr, "log10-sigma-p")
        gamma_p = 10.0**gamma_p
        sigma_p = 10.0**sigma_p
        rn = inject_red_noise(t_shared, gamma_p, sigma_p, f0, rng)
        for n in range(npsr):
            residuals[n] += rn[n]
        truth["red_noise_log10_gamma_p"] = np.log10(gamma_p).tolist()
        truth["red_noise_log10_sigma_p"] = np.log10(sigma_p).tolist()

    # ---- optional white noise ----
    if not args.no_white_noise:
        efac, equad = load_white_noise(args.noise_json, names)
        wn = inject_white_noise(errs, efac, equad, rng)
        for n in range(npsr):
            residuals[n] += wn[n]
        truth["white_noise_efac"] = efac.tolist()
        truth["white_noise_log10_equad"] = np.log10(equad).tolist()

    print_rms(names, residuals)

    print(f"\nWriting injected feathers to: {out_dir}")
    write_injected_feathers(objs, residuals, out_dir, args.overwrite)
    write_truth_sidecar(out_dir, truth)
    verify(out_dir, npsr)
    print(
        f"\nT2.1 ({args.mode}) complete: injected feathers built and consumed by the GWB path."
    )


def _broadcast(spec, npsr, label):
    """Parse a scalar or comma-separated list into an (npsr,) float array."""
    vals = [float(v) for v in str(spec).split(",")]
    if len(vals) == 1:
        return np.full(npsr, vals[0])
    if len(vals) != npsr:
        raise SystemExit(
            f"*** --{label} needs 1 or {npsr} values, got {len(vals)}. ***"
        )
    return np.array(vals)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--mode",
        choices=["powerlaw", "ou"],
        default="powerlaw",
        help="Injected GWB spectral shape (default: powerlaw)",
    )
    p.add_argument(
        "--aligned-dir",
        default=DEFAULT_ALIGNED_DIR,
        help="Dir of epoch-aligned input feathers (default: data/aligned)",
    )
    p.add_argument(
        "--out-dir", default=None, help="Output dir (default: data/inject_<mode>)"
    )
    p.add_argument(
        "--noise-json",
        default=DEFAULT_NOISE_JSON,
        help="Per-pulsar white-noise JSON (default: data/ng15_psr_noise.json)",
    )
    # Power-law params.
    p.add_argument(
        "--log10-A-gw",
        type=float,
        default=DEFAULT_LOG10_A_GW,
        help=f"Power-law log10 amplitude (default: {DEFAULT_LOG10_A_GW})",
    )
    p.add_argument(
        "--gamma",
        type=float,
        default=DEFAULT_GAMMA,
        help="Power-law spectral index gamma (default: 13/3)",
    )
    p.add_argument(
        "--n-freq",
        type=int,
        default=DEFAULT_N_FREQ,
        help=f"Number of Fourier modes (default: {DEFAULT_N_FREQ})",
    )
    # OU-control params.
    p.add_argument(
        "--log10-ha",
        type=float,
        default=DEFAULT_LOG10_HA,
        help=f"OU strain amplitude log10 (default: {DEFAULT_LOG10_HA})",
    )
    p.add_argument(
        "--log10-gamma-a",
        type=float,
        default=DEFAULT_LOG10_GAMMA_A,
        help=f"OU corner log10 gamma_a (default: {DEFAULT_LOG10_GAMMA_A})",
    )
    # Optional per-pulsar red noise (off by default).
    p.add_argument(
        "--red-noise",
        action="store_true",
        help="Inject per-pulsar OU red noise (needs --log10-gamma-p/-sigma-p)",
    )
    p.add_argument(
        "--log10-gamma-p",
        default=None,
        help="Red-noise log10 gamma_p (scalar or N comma-separated)",
    )
    p.add_argument(
        "--log10-sigma-p",
        default=None,
        help="Red-noise log10 sigma_p (scalar or N comma-separated)",
    )
    # White noise (on by default).
    p.add_argument(
        "--no-white-noise", action="store_true", help="Do not add white noise"
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed (recorded in injection_truth.json; default: 0)",
    )
    p.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing injected feathers"
    )
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()

"""Bayesian inference module for pulsar timing array analysis.

This module provides the main interface for performing Bayesian parameter estimation
on pulsar timing array data. It serves as the orchestration layer that coordinates
prior model specifications, parameter sampling, and NUTS inference.

The module handles parameters like:
- Gravitational wave background amplitude (ha) and spectral index (γa)
- Pulsar-specific red noise parameters (γp, σp)
- White noise parameters (EFAC, EQUAD)

The implementation uses the Hellings-Downs correlation pattern for the
gravitational wave background and models pulsar red noise as an
Ornstein-Uhlenbeck process.
"""

import jax
import jax.numpy as jnp
from flax import struct
import tensorflow_probability.substrates.jax as tfp
import jax.random as random
import numpyro
import arviz as az
from numpyro.infer import MCMC, NUTS
import hashlib
import os
import time

import numpy as np

from . import checkpointing

from .parameter_sampling import (
    sample_gw_parameters,
    sample_orf_epsilon,
    sample_cw_parameters,
    sample_chi_parameters,
    sample_pulsar_noise_parameters,
    sample_measurement_noise_parameters,
    count_free_parameters,
)

jax.config.update("jax_enable_x64", True)
tfpd = tfp.distributions


@struct.dataclass
class Parameters:
    """Define a struct to store the parameters of the Kalman filter model."""

    # GW parameters
    log10_gamma_a: float  # log10(γa) - log10 of GW spectral index
    γa: float  # s⁻¹ - GW spectral index (derived from log10_gamma_a)
    ha: float  # GWB amplitude

    # Pulsar parameters for the OU process
    γp: jnp.ndarray  # Pulsar-specific gamma values
    σp: jnp.ndarray  # Pulsar-specific sigma values

    # Measurement noise parameters
    EFAC: jnp.ndarray  # Error factors
    EQUAD: jnp.ndarray  # Extra quadrature noise

    # Correlation-path coordinate: None disables the path and uses the Hellings-Downs
    # matrix as supplied; 0 gives CURN (identity ORF), 1 gives Hellings-Downs.
    orf_epsilon: float = None


@struct.dataclass
class CWParameters:
    """Parameter struct for the CW signal model and per-pulsar noise."""

    # CW source parameters
    alpha_gw: float  # Source right ascension (radians)
    delta_gw: float  # Source declination (radians)
    f_gw: float  # GW frequency (Hz)
    h0: float  # Strain amplitude
    cos_iota: float  # Cosine of inclination angle
    psi: float  # Polarization angle (radians)
    Phi0: float  # Initial GW phase (radians)

    # Per-pulsar phase parameters (phase reparameterization of pulsar term)
    chi: jnp.ndarray  # Per-pulsar phase offsets in [0, 2pi), shape (Npsr,)

    # Pulsar noise parameters (same as GWB mode)
    gamma_p: jnp.ndarray  # Per-pulsar OU damping rates
    sigma_p: jnp.ndarray  # Per-pulsar OU driving amplitudes
    EFAC: jnp.ndarray  # Per-pulsar error scale factors
    EQUAD: jnp.ndarray  # Per-pulsar quadrature noise


def display_prior_summary(prior_specs, n_pulsars, logger=None):
    """Display a readable summary of all prior distributions.

    Parameters
    ----------
    prior_specs : dict
        Dictionary containing prior distributions from get_prior_model_specs()
    n_pulsars : int
        Number of pulsars (for vector parameter information)
    logger : logging.Logger, optional
        Logger object for output. If None, gets the centralized argus logger.
    """
    if logger is None:
        from argus.io_manager import get_argus_logger

        logger = get_argus_logger()

    def log_or_print(message):
        logger.info(message)

    log_or_print("\n" + "=" * 60)
    log_or_print("PRIOR SPECIFICATIONS SUMMARY")
    log_or_print("=" * 60)

    # CW parameters (if CW mode)
    cw_specs = prior_specs.get("cw_specs")
    if cw_specs is not None:
        log_or_print("\n--- Continuous Wave Source Parameters ---")
        for param_name, transform_key, spec_key in [
            ("log10(h₀)", "log10_h0_transform_params", "log10_h0_spec"),
            ("α_gw (RA)", "alpha_gw_transform_params", "alpha_gw_spec"),
            ("sin(δ_gw)", "sin_delta_gw_transform_params", "sin_delta_gw_spec"),
            ("log10(f_gw)", "log10_f_gw_transform_params", "log10_f_gw_spec"),
            ("cos(ι)", "cos_iota_transform_params", "cos_iota_spec"),
            ("ψ", "psi_transform_params", "psi_spec"),
            ("Φ₀", "Phi0_transform_params", "Phi0_spec"),
        ]:
            tp = cw_specs.get(transform_key)
            if tp is not None:
                log_or_print(
                    f"{param_name}: Uniform({tp['min']:.3f}, {tp['max']:.3f}) [reparameterized]"
                )
            else:
                spec = cw_specs.get(spec_key)
                if spec is not None:
                    log_or_print(f"{param_name}: FIXED at {float(spec):.4f}")
                else:
                    log_or_print(f"{param_name}: Not configured")

        chi_tp = cw_specs.get("chi_transform_params")
        if chi_tp is not None:
            log_or_print(
                f"χ (per-pulsar phase): Uniform({chi_tp['min']:.3f}, {chi_tp['max']:.3f}) "
                f"× {n_pulsars} pulsars [phase reparameterization]"
            )
    elif prior_specs.get("gw_parameterization") == "ridge":
        # Ridge parameterization (issue #109)
        log_or_print("\n--- Gravitational Wave Background Parameters (RIDGE) ---")
        psd_tp = prior_specs["log10_pivot_psd_transform_params"]
        ga_tp = prior_specs["log10_gamma_a_transform_params"]
        log_or_print(
            "Sampling basis: pivot log-PSD + log10(γ_a); log10(h_a) derived "
            "(straightens the h_a<->γ_a ridge)"
        )
        log_or_print(
            f"  - log10(pivot PSD): Uniform({psd_tp['min']:.1f}, {psd_tp['max']:.1f}) "
            "via log10_pivot_psd_prime ~ N(0,1)"
        )
        log_or_print(
            f"  - log10(γ_a): Uniform({ga_tp['min']:.1f}, {ga_tp['max']:.1f}) "
            "via log10_gamma_a_prime ~ N(0,1)"
        )
    else:
        # GW background parameters
        log_or_print("\n--- Gravitational Wave Background Parameters ---")

        # log10_ha parameter
        ha_spec = prior_specs["log10_ha_spec"]
        ha_transform = prior_specs["log10_ha_transform_params"]

        if ha_transform is not None:
            # Reparameterized case
            log_or_print("log10(h_a): REPARAMETERIZED for better NUTS sampling")
            log_or_print("  - Sampling: log10_ha_prime ~ N(0, 1)")
            log_or_print(
                f"  - Transform: log10_ha = {ha_transform['mean']:.2f} + log10_ha_prime * {ha_transform['std']:.3f}"
            )
            log_or_print(
                f"  - Equivalent to: Uniform({ha_transform['min']:.1f}, {ha_transform['max']:.1f})"
            )
        elif isinstance(ha_spec, tfpd.Distribution):
            # Direct distribution case (backward compatibility)
            if hasattr(ha_spec, "low"):
                log_or_print(
                    f"log10(h_a): Uniform({float(ha_spec.low):.1f}, {float(ha_spec.high):.1f})"
                )
            else:
                log_or_print(f"log10(h_a): {type(ha_spec).__name__} distribution")
        else:
            # Fixed value case
            log_or_print(f"log10(h_a): FIXED at {float(ha_spec):.1f}")

        # log10_gamma_a parameter
        log10_gamma_spec = prior_specs["log10_gamma_a_spec"]
        if isinstance(log10_gamma_spec, tfpd.Distribution):
            log_or_print(
                f"log10(γ_a): Uniform({float(log10_gamma_spec.low):.1f}, {float(log10_gamma_spec.high):.1f})"
            )
        else:
            log_or_print(f"log10(γ_a): FIXED at {float(log10_gamma_spec):.1f}")

    # Pulsar red noise parameters
    log_or_print(f"\n--- Pulsar Red Noise Parameters ({n_pulsars} pulsars) ---")

    # log10_gamma_p parameter - check for empirical/hierarchical modeling
    gamma_p_spec = prior_specs["log10_gamma_p_spec"]
    hierarchical_specs = prior_specs.get("hierarchical_specs")
    empirical_specs = prior_specs.get("empirical_specs")

    if empirical_specs is not None:
        gamma_loc = empirical_specs["gamma_loc"]
        gamma_scale = empirical_specs["gamma_scale"]
        log_or_print("log10(γ_p): EMPIRICAL per-pulsar Normal priors")
        log_or_print(
            f"  - loc range: [{float(jnp.min(gamma_loc)):.2f}, {float(jnp.max(gamma_loc)):.2f}]"
        )
        log_or_print(
            f"  - scale range: [{float(jnp.min(gamma_scale)):.3f}, {float(jnp.max(gamma_scale)):.3f}] (inflation applied)"
        )
    elif hierarchical_specs and hierarchical_specs.get("hierarchical_noise", False):
        # Hierarchical modeling case
        mean_spec = hierarchical_specs["log10_gamma_p_mean_spec"]
        std_spec = hierarchical_specs["log10_gamma_p_std_spec"]
        log_or_print("log10(γ_p): HIERARCHICAL modeling")
        log_or_print(
            f"  - Population mean: Uniform({float(mean_spec.low):.1f}, {float(mean_spec.high):.1f})"
        )
        log_or_print(
            f"  - Population std: Uniform({float(std_spec.low):.1f}, {float(std_spec.high):.1f})"
        )
        log_or_print("  - Individual pulsars: Normal(population_mean, population_std)")
    elif isinstance(gamma_p_spec, tfpd.Distribution):
        log_or_print(
            f"log10(γ_p): Uniform({float(gamma_p_spec.low[0]):.1f}, {float(gamma_p_spec.high[0]):.1f}) for each pulsar"
        )
    elif gamma_p_spec is not None:
        # jnp.min/max also handle length-1 arrays (n=1 runs), where float() would raise
        if hasattr(gamma_p_spec, "__len__"):
            log_or_print(
                f"log10(γ_p): FIXED at individual values (range: {float(jnp.min(gamma_p_spec)):.2f} to {float(jnp.max(gamma_p_spec)):.2f})"
            )
        else:
            log_or_print(f"log10(γ_p): FIXED at {float(gamma_p_spec):.2f}")
    else:
        log_or_print("log10(γ_p): ERROR - None value encountered")

    # log10_sigma_p parameter - check for empirical/hierarchical modeling
    sigma_p_spec = prior_specs["log10_sigma_p_spec"]
    if empirical_specs is not None:
        ratio_loc = empirical_specs["ratio_loc"]
        ratio_scale = empirical_specs["ratio_scale"]
        log_or_print("log10(σ_p): EMPIRICAL log-ratio parameterization")
        log_or_print("  - log10(σ_p) = log10(γ_p) + log10(ratio)")
        log_or_print(
            f"  - ratio loc range: [{float(jnp.min(ratio_loc)):.2f}, {float(jnp.max(ratio_loc)):.2f}]"
        )
        log_or_print(
            f"  - ratio scale range: [{float(jnp.min(ratio_scale)):.3f}, {float(jnp.max(ratio_scale)):.3f}] (inflation applied)"
        )
    elif hierarchical_specs and hierarchical_specs.get(
        "log_ratio_parameterization", False
    ):
        # Check if the required specs exist before accessing them
        if (
            "log10_ratio_mean_spec" in hierarchical_specs
            and "log10_ratio_std_spec" in hierarchical_specs
        ):
            # Log-ratio parameterization case
            mean_spec = hierarchical_specs["log10_ratio_mean_spec"]
            std_spec = hierarchical_specs["log10_ratio_std_spec"]
            log_or_print("log10(σ_p): LOG-RATIO parameterization")
            log_or_print("  - log10(σ_p) = log10(γ_p) + log10(ratio)")
            log_or_print(
                f"  - Ratio mean: Uniform({float(mean_spec.low):.1f}, {float(mean_spec.high):.1f})"
            )
            log_or_print(
                f"  - Ratio std: Uniform({float(std_spec.low):.1f}, {float(std_spec.high):.1f})"
            )
            log_or_print("  - Individual ratios: Normal(ratio_mean, ratio_std)")
        else:
            # Fallback: hierarchical settings enabled but specs not created (likely due to fixed params)
            log_or_print(
                "log10(σ_p): FIXED (hierarchical settings detected but overridden by fixed parameters)"
            )
    elif isinstance(sigma_p_spec, tfpd.Distribution):
        log_or_print(
            f"log10(σ_p): Uniform({float(sigma_p_spec.low[0]):.1f}, {float(sigma_p_spec.high[0]):.1f}) for each pulsar"
        )
    elif sigma_p_spec is not None:
        # jnp.min/max also handle length-1 arrays (n=1 runs), where float() would raise
        if hasattr(sigma_p_spec, "__len__"):
            log_or_print(
                f"log10(σ_p): FIXED at individual values (range: {float(jnp.min(sigma_p_spec)):.2f} to {float(jnp.max(sigma_p_spec)):.2f})"
            )
        else:
            log_or_print(f"log10(σ_p): FIXED at {float(sigma_p_spec):.2f}")
    else:
        log_or_print("log10(σ_p): ERROR - None value encountered")

    # Measurement noise parameters
    log_or_print(f"\n--- Measurement Noise Parameters ({n_pulsars} pulsars) ---")

    # EFAC parameter
    efac_spec = prior_specs["efac_spec"]
    if isinstance(efac_spec, tfpd.Distribution):
        log_or_print(
            f"EFAC: Uniform({float(efac_spec.low[0]):.2f}, {float(efac_spec.high[0]):.2f}) for each pulsar"
        )
    elif efac_spec is not None:
        # jnp.min/max also handle length-1 arrays (n=1 runs), where float() would raise
        if hasattr(efac_spec, "__len__"):
            log_or_print(
                f"EFAC: FIXED at individual values (range: {float(jnp.min(efac_spec)):.3f} to {float(jnp.max(efac_spec)):.3f})"
            )
        else:
            log_or_print(f"EFAC: FIXED at {float(efac_spec):.3f}")
    else:
        log_or_print("EFAC: ERROR - None value encountered")

    # EQUAD parameter
    equad_spec = prior_specs["equad_spec"]
    if isinstance(equad_spec, dict) and equad_spec.get("use_log10", False):
        # log10(EQUAD) parameterization
        log10_equad_spec = equad_spec["log10_equad_spec"]
        log10_low = float(log10_equad_spec.low[0])
        log10_high = float(log10_equad_spec.high[0])
        log_or_print(
            f"EQUAD: log10(EQUAD) ~ Uniform({log10_low:.1f}, {log10_high:.1f}) for each pulsar"
        )
    elif isinstance(equad_spec, tfpd.Distribution):
        # Regular uniform distribution
        log_or_print(
            f"EQUAD: Uniform({float(equad_spec.low[0]):.2e}, {float(equad_spec.high[0]):.2e}) for each pulsar"
        )
    elif equad_spec is not None:
        # jnp.min/max also handle length-1 arrays (n=1 runs), where float() would raise
        if hasattr(equad_spec, "__len__"):
            log_or_print(
                f"EQUAD: FIXED at individual values (range: {float(jnp.min(equad_spec)):.2e} to {float(jnp.max(equad_spec)):.2e})"
            )
        else:
            log_or_print(f"EQUAD: FIXED at {float(equad_spec):.2e}")
    else:
        log_or_print("EQUAD: ERROR - None value encountered")

    log_or_print("=" * 60)


def log_likelihood_fn(
    kalman_filter,
    log10_ha,
    log10_gamma_a,
    log10_γp,
    log10_σp,
    efac,
    equad,
    orf_epsilon=None,
):
    """Calculate log likelihood for NumPyro sampling.

    Parameters
    ----------
    kalman_filter : object
        Kalman filter instance with get_likelihood method
    log10_ha : float
        Log10 of GW amplitude
    log10_gamma_a : float
        Log10 of GW spectral index
    log10_γp : jax.Array
        Log10 of pulsar gamma values
    log10_σp : jax.Array
        Log10 of pulsar sigma values
    efac : jax.Array
        EFAC values
    equad : jax.Array
        EQUAD values
    orf_epsilon : float, jax.Array or None
        Correlation-path coordinate. None (default) leaves the overlap reduction
        function as supplied; 0 gives CURN, 1 gives Hellings-Downs.

    Returns
    -------
    float
        Log likelihood value
    """
    ha = 10.0**log10_ha
    γa = 10.0**log10_gamma_a
    γp = 10.0**log10_γp
    σp = 10.0**log10_σp

    params = Parameters(
        log10_gamma_a=log10_gamma_a,
        γa=γa,
        ha=ha,
        γp=γp,
        σp=σp,
        EFAC=efac,
        EQUAD=equad,
        orf_epsilon=orf_epsilon,
    )

    return kalman_filter.get_likelihood(params)


def numpyro_model(kalman_filter, prior_specs, n_pulsars):
    """NumPyro model definition for Bayesian inference with parameter standardization.

    This function defines the NumPyro probabilistic model using standardized
    parameter transformations for better NUTS sampling in high-dimensional spaces.

    Parameters
    ----------
    kalman_filter : object
        JAX Kalman filter with get_likelihood method
    prior_specs : dict
        Dictionary containing prior distributions from get_prior_model_specs()
    n_pulsars : int
        Number of pulsars
    """
    # Sample parameters using specialized functions
    log10_ha, log10_gamma_a, γa = sample_gw_parameters(prior_specs)
    orf_epsilon = sample_orf_epsilon(prior_specs)
    log10_γp, log10_σp = sample_pulsar_noise_parameters(prior_specs, n_pulsars)
    efac, equad = sample_measurement_noise_parameters(prior_specs, n_pulsars)

    # Calculate log likelihood
    log_likelihood = log_likelihood_fn(
        kalman_filter,
        log10_ha,
        log10_gamma_a,
        log10_γp,
        log10_σp,
        efac,
        equad,
        orf_epsilon=orf_epsilon,
    )

    # Add likelihood to the model
    numpyro.factor("likelihood", log_likelihood)


def cw_log_likelihood_fn(
    kalman_filter,
    log10_h0,
    alpha_gw,
    delta_gw,
    log10_f_gw,
    cos_iota,
    psi,
    Phi0,
    chi,
    log10_γp,
    log10_σp,
    efac,
    equad,
):
    """Calculate CW log likelihood for NumPyro sampling.

    Parameters
    ----------
    kalman_filter : CWKalmanFilter
        CW Kalman filter instance.
    log10_h0 : float
        Log10 of strain amplitude.
    alpha_gw : float
        Source right ascension (radians).
    delta_gw : float
        Source declination (radians).
    log10_f_gw : float
        Log10 of GW frequency (Hz).
    cos_iota : float
        Cosine of inclination angle.
    psi : float
        Polarization angle (radians).
    Phi0 : float
        Initial GW phase (radians).
    chi : jax.Array
        Per-pulsar phase parameters, shape (Npsr,).
    log10_γp : jax.Array
        Log10 of pulsar gamma values.
    log10_σp : jax.Array
        Log10 of pulsar sigma values.
    efac : jax.Array
        EFAC values.
    equad : jax.Array
        EQUAD values.

    Returns
    -------
    float
        Log likelihood value.
    """
    h0 = 10.0**log10_h0
    f_gw = 10.0**log10_f_gw
    gamma_p = 10.0**log10_γp
    sigma_p = 10.0**log10_σp

    params = CWParameters(
        alpha_gw=alpha_gw,
        delta_gw=delta_gw,
        f_gw=f_gw,
        h0=h0,
        cos_iota=cos_iota,
        psi=psi,
        Phi0=Phi0,
        chi=chi,
        gamma_p=gamma_p,
        sigma_p=sigma_p,
        EFAC=efac,
        EQUAD=equad,
    )

    return kalman_filter.get_likelihood(params)


def numpyro_model_cw(kalman_filter, prior_specs, n_pulsars):
    """NumPyro model for CW signal inference.

    Parameters
    ----------
    kalman_filter : CWKalmanFilter
        CW Kalman filter instance.
    prior_specs : dict
        Dictionary containing prior distributions.
    n_pulsars : int
        Number of pulsars.
    """
    # Sample CW source parameters
    log10_h0, alpha_gw, delta_gw, log10_f_gw, cos_iota, psi, Phi0 = (
        sample_cw_parameters(prior_specs)
    )

    # Sample per-pulsar phase parameters (for phase-reparameterized pulsar term)
    chi = sample_chi_parameters(prior_specs, n_pulsars)

    # Sample noise parameters (shared with GWB mode)
    log10_γp, log10_σp = sample_pulsar_noise_parameters(prior_specs, n_pulsars)
    efac, equad = sample_measurement_noise_parameters(prior_specs, n_pulsars)

    # Calculate CW log likelihood
    log_likelihood = cw_log_likelihood_fn(
        kalman_filter,
        log10_h0,
        alpha_gw,
        delta_gw,
        log10_f_gw,
        cos_iota,
        psi,
        Phi0,
        chi,
        log10_γp,
        log10_σp,
        efac,
        equad,
    )

    numpyro.factor("likelihood", log_likelihood)


def setup_nuts_kernel(prior_specs, n_pulsars, config):
    """Set up NUTS kernel with optimized parameters.

    Parameters
    ----------
    prior_specs : dict
        Prior specifications dictionary
    n_pulsars : int
        Number of pulsars
    config : configparser.ConfigParser
        Configuration object

    Returns
    -------
    tuple
        (nuts_kernel, nuts_info) where nuts_info contains diagnostic information
    """
    # Get NUTS parameters from config with optimized defaults for high-dimensional sampling
    target_accept_prob = config.getfloat(
        "NUTS", "target_accept_prob", fallback=0.95
    )  # More conservative for high-dim
    max_tree_depth = config.getint("NUTS", "max_tree_depth", fallback=10)
    dense_mass = config.getboolean("NUTS", "dense_mass", fallback=False)

    # Optional per-block dense mass matrix (NumPyro's list-of-tuples form). When present and
    # non-empty, this OVERRIDES the boolean above: the named latent sites get a dense sub-matrix
    # and every other site keeps its diagonal block. Sites within a group are comma-separated;
    # ';' separates independent groups. Used to learn the log10_ha<->log10_gamma_a ridge
    # correlation (2x2 dense block over log10_ha_prime, log10_gamma_a_prime) without the cost of a
    # full ~142x142 dense matrix. A mass-matrix choice never changes the target posterior.
    if config.has_option("NUTS", "dense_mass_blocks"):
        raw = config.get("NUTS", "dense_mass_blocks").strip()
        if raw:
            dense_mass = [
                tuple(s.strip() for s in group.split(",") if s.strip())
                for group in raw.split(";")
                if group.strip()
            ]

    # Handle step_size - only set if explicitly provided in config
    nuts_kwargs = {
        "target_accept_prob": target_accept_prob,
        "max_tree_depth": max_tree_depth,
        "adapt_step_size": True,
        "adapt_mass_matrix": True,
        "dense_mass": dense_mass,
    }

    # Only add step_size if explicitly set in config
    if config.has_option("NUTS", "step_size"):
        step_size = config.getfloat("NUTS", "step_size")
        nuts_kwargs["step_size"] = step_size
        print(f"Using custom step size: {step_size}")

    # Count total number of free parameters for diagnostics
    total_params = count_free_parameters(prior_specs, n_pulsars)

    nuts_info = {
        "total_params": total_params,
        "target_accept_prob": target_accept_prob,
        "max_tree_depth": max_tree_depth,
        "dense_mass": dense_mass,
    }

    # Set up NUTS kernel with optimizations
    def model_fn():
        return numpyro_model(None, prior_specs, n_pulsars)  # Will be bound later

    kernel = NUTS(model_fn, **nuts_kwargs)

    return kernel, nuts_info


def print_nuts_diagnostics(prior_specs, nuts_info, config):
    """Print NUTS sampling diagnostics and parameter information.

    Parameters
    ----------
    prior_specs : dict
        Prior specifications
    nuts_info : dict
        NUTS diagnostic information
    config : configparser.ConfigParser
        Configuration object
    """
    n_pulsars = len([spec for spec in prior_specs.keys() if "pulsar" in spec])
    num_samples = config.getint("NUTS", "num_samples", fallback=2000)
    num_warmup = config.getint("NUTS", "num_warmup", fallback=2000)
    num_chains = config.getint("NUTS", "num_chains", fallback=2)

    print("Running NumPyro NUTS inference...")
    print(
        f"NUTS parameters: {num_samples} samples, {num_warmup} warmup, {num_chains} chains"
    )
    print(
        f"Target accept prob: {nuts_info['target_accept_prob']} (optimized for high-dimensional sampling)"
    )
    print(f"Dense mass matrix: {nuts_info['dense_mass']}")
    print(f"Max tree depth: {nuts_info['max_tree_depth']}")
    print(f"Total free parameters: {nuts_info['total_params']}")

    # Check if empirical/hierarchical modeling is enabled
    hierarchical_specs = prior_specs.get("hierarchical_specs")
    if prior_specs.get("empirical_specs") is not None:
        print("Empirical per-pulsar red noise priors (no population hyperparameters)")
    elif hierarchical_specs:
        hier_gamma = hierarchical_specs.get("hierarchical_noise", False)
        log_ratio = hierarchical_specs.get("log_ratio_parameterization", False)
        if hier_gamma or log_ratio:
            print("Advanced modeling enabled for pulsar noise parameters")
            if hier_gamma and log_ratio:
                print("γp hierarchical + σp via log-ratio parameterization")
                print(
                    f"Effective dimensionality: 4 hyperparameters + {2*n_pulsars} constrained parameters"
                )
                print("σp = γp + ratio (reduces parameter correlations)")
            elif hier_gamma:
                print("γp uses hierarchical priors, σp fixed")
                print(
                    f"Effective dimensionality: 2 hyperparameters + {n_pulsars} constrained parameters"
                )
            elif log_ratio:
                print("σp via log-ratio parameterization, γp independent")
                print(
                    f"Effective dimensionality: 2 hyperparameters + {2*n_pulsars} parameters"
                )

    if nuts_info["total_params"] > 10:
        print(
            "High-dimensional parameter space detected - using aggressive NUTS tuning"
        )


def run_nuts_sampling(
    kalman_filter,
    config,
    n_pulsars,
    sigma_p_array,
    gamma_p_array,
    efac_array,
    equad_array,
    mode="gwb",
):
    """Run NumPyro NUTS inference with optimizations for high-dimensional sampling.

    Parameters
    ----------
    kalman_filter : object
        JAX Kalman filter with get_likelihood method
    config : configparser.ConfigParser
        Configuration object
    n_pulsars : int
        Number of pulsars
    sigma_p_array : jnp.ndarray
        Pulsar red noise sigma values
    gamma_p_array : jnp.ndarray
        Pulsar red noise gamma values
    efac_array : jnp.ndarray
        EFAC values
    equad_array : jnp.ndarray
        EQUAD values
    mode : str
        Signal model mode: 'gwb' or 'cw'.

    Returns
    -------
    arviz.InferenceData
        ArviZ InferenceData object containing MCMC results
    """
    from .prior_models import get_prior_model_specs

    # Get prior model distributions
    prior_specs = get_prior_model_specs(
        config,
        n_pulsars,
        sigma_p_array,
        gamma_p_array,
        efac_array,
        equad_array,
        mode=mode,
    )

    # Get NUTS parameters from config
    num_samples = config.getint("NUTS", "num_samples", fallback=2000)
    num_warmup = config.getint("NUTS", "num_warmup", fallback=2000)
    num_chains = config.getint("NUTS", "num_chains", fallback=2)

    # Set up NUTS kernel
    kernel, nuts_info = setup_nuts_kernel(prior_specs, n_pulsars, config)

    # Print diagnostics
    print_nuts_diagnostics(prior_specs, nuts_info, config)

    # Create the actual model function bound to the Kalman filter
    if mode == "cw":

        def bound_model():
            return numpyro_model_cw(kalman_filter, prior_specs, n_pulsars)

    else:

        def bound_model():
            return numpyro_model(kalman_filter, prior_specs, n_pulsars)

    # Create NUTS kernel with bound model
    kernel = NUTS(
        bound_model,
        **{
            "target_accept_prob": nuts_info.get("target_accept_prob", 0.95),
            "max_tree_depth": nuts_info.get("max_tree_depth", 10),
            "adapt_step_size": True,
            "adapt_mass_matrix": True,
            "dense_mass": nuts_info.get("dense_mass", False),
        },
    )

    # Set up MCMC sampler
    # Use parallel chains across devices when multiple GPUs are available
    chain_method = "sequential"
    if num_chains > 1:
        import jax

        n_devices = jax.local_device_count()
        if n_devices >= num_chains:
            chain_method = "parallel"
            print(f"Running {num_chains} chains in parallel across {n_devices} devices")
        else:
            print(
                f"Running {num_chains} chains sequentially ({n_devices} device(s) available)"
            )

    seed = config.getint("NUTS", "seed", fallback=42)
    rng_key = random.PRNGKey(seed)

    settings = checkpointing.get_checkpoint_settings(config)
    warm_start = checkpointing.get_warm_start_settings(config)

    if warm_start["enabled"]:
        # Skip adaptation entirely and continue from the parent run's tuned state.
        # Warmup is roughly half the wall-clock of a run, and every scramble in the
        # null ensemble shares this run's geometry, so re-adapting for each is the
        # largest avoidable cost in that campaign.
        state = checkpointing.load_warm_start_state(
            warm_start["state_path"], warm_start["seed"]
        )
        print(
            f"Warm start from {warm_start['state_path']} "
            f"(seed {warm_start['seed']}); skipping warmup."
        )
        sampler = MCMC(
            kernel,
            num_samples=num_samples,
            num_warmup=0,
            num_chains=num_chains,
            chain_method=chain_method,
            progress_bar=True,
        )
        sampler.post_warmup_state = state
        sampler.run(sampler.post_warmup_state.rng_key)
        sampler.print_summary()
        return az.from_numpyro(sampler)

    if not settings["enabled"]:
        sampler = MCMC(
            kernel,
            num_samples=num_samples,
            num_warmup=num_warmup,
            num_chains=num_chains,
            chain_method=chain_method,
            progress_bar=True,
        )
        sampler.run(rng_key)
        sampler.print_summary()
        return az.from_numpyro(sampler)

    return _run_nuts_with_checkpointing(
        kernel=kernel,
        rng_key=rng_key,
        num_samples=num_samples,
        num_warmup=num_warmup,
        num_chains=num_chains,
        chain_method=chain_method,
        settings=settings,
        config=config,
        fingerprint_spec={
            "mode": mode,
            "n_pulsars": int(n_pulsars),
            "num_samples": int(num_samples),
            "num_warmup": int(num_warmup),
            "num_chains": int(num_chains),
            "seed": int(seed),
            "nuts": {
                key: nuts_info.get(key)
                for key in (
                    "target_accept_prob",
                    "max_tree_depth",
                    "dense_mass",
                )
            },
            "priors": _prior_fingerprint(prior_specs),
            "data": _data_fingerprint(kalman_filter),
        },
    )


def _data_fingerprint(kalman_filter):
    """Identify the data and correlation structure the run is conditioned on.

    Without this a resume would be judged only on the config, and two runs whose
    overlap reduction functions differ -- an HD run and a sky-scrambled null, say --
    would look interchangeable, because the ORF reaches the filter through the data
    rather than through any config key. Concatenating their draws would be silent and
    undetectable, which is exactly what the fingerprint exists to prevent.
    """
    orf = np.asarray(kalman_filter.hellings_downs_matrix)
    return {
        "n_pulsars": int(kalman_filter.Npsr),
        "n_epochs": int(np.asarray(kalman_filter.jax_data).shape[0]),
        "m_sum": int(kalman_filter.M_sum),
        "orf_sha": hashlib.sha256(
            np.ascontiguousarray(orf, dtype=np.float64).tobytes()
        ).hexdigest()[:16],
        "data_sha": hashlib.sha256(
            np.ascontiguousarray(
                np.asarray(kalman_filter.jax_data), dtype=np.float64
            ).tobytes()
        ).hexdigest()[:16],
    }


def _prior_fingerprint(prior_specs):
    """A comparable summary of the prior specification for the resume check.

    Distribution objects do not serialise usefully, so each is reduced to its type and
    support. That is enough to catch the mistakes that matter — a moved prior bound, a
    parameter switched between fixed and sampled, a different parameterization — while
    staying stable across irrelevant details.
    """
    summary = {}
    for name, spec in sorted(prior_specs.items()):
        if isinstance(spec, dict):
            summary[name] = {
                key: (float(value) if isinstance(value, (int, float)) else str(value))
                for key, value in sorted(spec.items())
                if not isinstance(value, dict)
            }
        elif hasattr(spec, "low") and hasattr(spec, "high"):
            # Bounds may be arrays (one entry per pulsar), so summarise rather than
            # coerce to a scalar.
            summary[name] = [
                type(spec).__name__,
                np.asarray(spec.low).ravel().tolist(),
                np.asarray(spec.high).ravel().tolist(),
            ]
        elif isinstance(spec, (int, float, str, bool)) or spec is None:
            summary[name] = spec
        else:
            summary[name] = type(spec).__name__
    return summary


def _run_nuts_with_checkpointing(
    kernel,
    rng_key,
    num_samples,
    num_warmup,
    num_chains,
    chain_method,
    settings,
    config,
    fingerprint_spec,
):
    """Run NUTS in segments, saving state and accumulated draws after each.

    Warmup is paid once, in the first segment. Later segments start from the previous
    segment's final state via ``post_warmup_state``, so the chain continues rather than
    restarting — the concatenated draws are the same chain a single uninterrupted run
    would have produced.
    """
    output_dir = config.get("Checkpointing", "directory", fallback=None)
    if not output_dir:
        raise ValueError(
            "Checkpointing is enabled but no 'directory' is set in [Checkpointing]."
        )
    os.makedirs(output_dir, exist_ok=True)
    output_id = config.get("Checkpointing", "output_id", fallback="run")

    fingerprint = checkpointing.run_fingerprint(fingerprint_spec)
    resumed = None
    if settings["resume"]:
        resumed = checkpointing.load_checkpoint(output_dir, output_id, fingerprint)
        if resumed is None:
            print("No checkpoint found; starting from the beginning.")

    sizes = checkpointing.segment_sizes(num_samples, settings["interval"])
    accumulated = None
    draws_done = 0
    first_segment = 0

    if resumed is not None:
        accumulated = resumed["inference_data"]
        draws_done = resumed["progress"]["draws_completed"]
        first_segment = resumed["progress"]["segments_completed"]
        print(
            f"Resuming from checkpoint: {draws_done}/{num_samples} draws per chain "
            f"({first_segment}/{len(sizes)} segments)."
        )
        if first_segment >= len(sizes):
            print("Checkpoint is already complete; nothing to do.")
            return checkpointing.mark_partial(
                accumulated, True, draws_done, num_samples
            )

    segments = [accumulated] if accumulated is not None else []
    state = resumed["sampler_state"] if resumed is not None else None

    for index in range(first_segment, len(sizes)):
        size = sizes[index]
        sampler = MCMC(
            kernel,
            num_samples=size,
            num_warmup=num_warmup,
            num_chains=num_chains,
            chain_method=chain_method,
            progress_bar=True,
        )
        print(
            f"Segment {index + 1}/{len(sizes)}: {size} draws per chain "
            f"({draws_done}/{num_samples} done)."
        )
        if state is None:
            sampler.run(rng_key)
        else:
            # Setting post_warmup_state makes run() skip adaptation and continue the
            # chain from the stored state, which is what makes a resume a continuation
            # rather than a restart.
            sampler.post_warmup_state = state
            sampler.run(sampler.post_warmup_state.rng_key)

        state = sampler.last_state
        draws_done += size
        segments.append(az.from_numpyro(sampler))
        combined = checkpointing.concat_segments(segments)
        segments = [combined]

        checkpointing.save_checkpoint(
            output_dir,
            output_id,
            checkpointing.numpy_state(state),
            combined,
            fingerprint,
            {
                "draws_completed": draws_done,
                "draws_target": num_samples,
                "segments_completed": index + 1,
                "segments_total": len(sizes),
            },
        )

    sampler.print_summary()
    return checkpointing.mark_partial(segments[0], True, draws_done, num_samples)


def _jaxns_results_to_arviz(results, num_posterior_samples=10000):
    """Convert jaxns nested sampling results to ArviZ InferenceData.

    jaxns produces weighted samples. We resample to get unweighted
    posterior samples compatible with ArviZ and downstream diagnostics.

    Parameters
    ----------
    results : jaxns.NestedSamplerResults
        Results from jaxns nested sampling.
    num_posterior_samples : int
        Number of unweighted posterior samples to generate.

    Returns
    -------
    arviz.InferenceData
        ArviZ InferenceData object with posterior samples.
    """
    # Get weighted samples and log posterior mass
    samples = results.samples
    log_dp_mean = results.log_dp_mean

    # Resample to get unweighted posterior samples
    key = jax.random.PRNGKey(0)
    indices = jax.random.categorical(key, log_dp_mean, shape=(num_posterior_samples,))

    posterior_dict = {}
    for param_name, values in samples.items():
        resampled = values[indices]
        # ArviZ expects shape (chains, draws, ...) — single chain for nested sampling
        posterior_dict[param_name] = jnp.expand_dims(resampled, axis=0)

    # Add derived delta_gw from sin_delta_gw if present
    if "sin_delta_gw" in posterior_dict:
        posterior_dict["delta_gw"] = jnp.arcsin(posterior_dict["sin_delta_gw"])

    # Convert JAX arrays to numpy for ArviZ
    import numpy as np

    posterior_np = {k: np.asarray(v) for k, v in posterior_dict.items()}

    inf_data = az.from_dict(posterior=posterior_np)
    return inf_data


def run_nested_sampling(
    kalman_filter,
    config,
    n_pulsars,
    sigma_p_array,
    gamma_p_array,
    efac_array,
    equad_array,
    mode="cw",
):
    """Run jaxns nested sampling inference for CW signal analysis.

    Nested sampling handles multimodal posteriors natively and computes
    Bayesian evidence as a byproduct, making it suitable for CW pulsar-term
    searches where NUTS gets trapped in local modes.

    Parameters
    ----------
    kalman_filter : CWKalmanFilter
        CW Kalman filter instance.
    config : configparser.ConfigParser
        Configuration object.
    n_pulsars : int
        Number of pulsars.
    sigma_p_array : jnp.ndarray
        Pulsar red noise sigma values.
    gamma_p_array : jnp.ndarray
        Pulsar red noise gamma values.
    efac_array : jnp.ndarray
        EFAC values.
    equad_array : jnp.ndarray
        EQUAD values.
    mode : str
        Signal model mode. Currently only 'cw' is supported.

    Returns
    -------
    tuple
        (arviz.InferenceData, (log_Z_mean, log_Z_uncert))
    """
    if mode != "cw":
        raise NotImplementedError(
            "Nested sampling is currently only supported for CW mode. "
            "Use NUTS for GWB inference."
        )

    from jaxns import Model, NestedSampler
    from .prior_models import get_prior_model_specs
    from .parameter_sampling import build_jaxns_cw_prior_model

    # Build prior specs (same as NUTS path)
    prior_specs = get_prior_model_specs(
        config,
        n_pulsars,
        sigma_p_array,
        gamma_p_array,
        efac_array,
        equad_array,
        mode=mode,
    )

    # Build jaxns prior model
    prior_model_fn = build_jaxns_cw_prior_model(prior_specs, n_pulsars)

    # Build log-likelihood wrapper that calls existing cw_log_likelihood_fn
    def log_likelihood(
        log10_h0,
        alpha_gw,
        delta_gw,
        log10_f_gw,
        cos_iota,
        psi,
        Phi0,
        chi,
        log10_gamma_p,
        log10_sigma_p,
        efac,
        equad,
    ):
        return cw_log_likelihood_fn(
            kalman_filter,
            log10_h0,
            alpha_gw,
            delta_gw,
            log10_f_gw,
            cos_iota,
            psi,
            Phi0,
            chi,
            log10_gamma_p,
            log10_sigma_p,
            efac,
            equad,
        )

    # Create jaxns Model
    model = Model(prior_model=prior_model_fn, log_likelihood=log_likelihood)

    # Read nested sampler config
    max_samples = config.getint("NestedSampler", "max_samples", fallback=100000)
    num_live_points = config.getint("NestedSampler", "num_live_points", fallback=1000)
    s = config.getint("NestedSampler", "s", fallback=5)
    k = config.getint("NestedSampler", "k", fallback=0)
    num_posterior_samples = config.getint(
        "NestedSampler", "num_posterior_samples", fallback=10000
    )

    print("Running jaxns nested sampling...")
    print(f"  max_samples: {max_samples}")
    print(f"  num_live_points: {num_live_points}")
    print(f"  s (num slices): {s}")
    print(f"  k (phantom samples): {k}")
    print(f"  Model dimensions: {model.U_ndims}")

    # Create and run sampler
    sampler = NestedSampler(
        model=model,
        max_samples=max_samples,
        num_live_points=num_live_points,
        s=s,
        k=k,
    )

    rng_key = jax.random.PRNGKey(42)
    termination_reason, state = sampler(rng_key)
    results = sampler.to_results(termination_reason=termination_reason, state=state)

    # Log evidence
    log_Z_mean = float(results.log_Z_mean)
    log_Z_uncert = float(results.log_Z_uncert)
    print(f"\nLog-evidence: {log_Z_mean:.2f} +/- {log_Z_uncert:.2f}")

    # Print summary
    sampler.summary(results)

    # Convert to ArviZ
    inf_data = _jaxns_results_to_arviz(results, num_posterior_samples)

    return inf_data, (log_Z_mean, log_Z_uncert)


def test_likelihood_performance(kalman_filter, config, n_pulsars, logger):
    """Test likelihood evaluation performance using known parameter values.

    This function runs a single likelihood evaluation using the same parameter
    values as in test_likelihood_value to provide users with timing and
    likelihood value information before running the full inference.

    Parameters
    ----------
    kalman_filter : object
        Kalman filter object
    config : configparser.ConfigParser
        Configuration object
    n_pulsars : int
        Number of pulsars
    logger : logging.Logger
        Logger object

    Returns
    -------
    float
        The computed log likelihood value
    """
    from argus.utils import get_noise_parameters

    logger.info("=== Likelihood Performance Test ===")
    logger.info("Testing likelihood evaluation with known parameter values...")

    # Get noise parameters using the common function
    efac_array, equad_array, sigma_p_array, gamma_p_array = get_noise_parameters(config)

    # Set test parameter values
    γa_test = 1e-9
    ha_test = 1e-15

    # If noise parameters are None, create test arrays with reasonable values
    if gamma_p_array is None:
        gamma_p_array = jnp.full(n_pulsars, 1e-8)  # Default gamma_p test value
    if sigma_p_array is None:
        sigma_p_array = jnp.full(n_pulsars, 1e-15)  # Default sigma_p test value
    if efac_array is None:
        efac_array = jnp.ones(n_pulsars)  # Default EFAC test value
    if equad_array is None:
        equad_array = jnp.full(n_pulsars, 1e-7)  # Default EQUAD test value

    # Create parameter object
    test_params = Parameters(
        log10_gamma_a=jnp.log10(γa_test),
        γa=γa_test,
        ha=ha_test,
        γp=gamma_p_array,
        σp=sigma_p_array,
        EFAC=efac_array,
        EQUAD=equad_array,
    )

    logger.info(f"Test parameters: γa={γa_test}, ha={ha_test}")
    logger.info(f"Number of pulsars: {n_pulsars}")

    # Time the likelihood evaluation (first time)
    logger.info("Performing for the first time a likelihood evaluation...")
    start_time = time.perf_counter()

    log_likelihood = kalman_filter.get_likelihood(test_params)
    # Ensure computation is complete before stopping timer
    log_likelihood.block_until_ready()

    end_time = time.perf_counter()
    duration1 = end_time - start_time

    # Time the likelihood evaluation (second time)
    logger.info("Performing timed for the second time a likelihood evaluation...")
    start_time = time.perf_counter()

    log_likelihood = kalman_filter.get_likelihood(test_params)
    # Ensure computation is complete before stopping timer
    log_likelihood.block_until_ready()

    end_time = time.perf_counter()
    duration2 = end_time - start_time

    # Log results
    logger.info(
        f"Likelihood evaluation completed in {duration1:.4f} seconds the first time"
    )
    logger.info(
        f"Likelihood evaluation completed in {duration2:.4f} seconds the second time"
    )
    logger.info(f"Log likelihood value: {float(log_likelihood)}")
    logger.info("=== End Likelihood Performance Test ===")

    return float(log_likelihood)

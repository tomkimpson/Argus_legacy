"""Prior model specifications for Bayesian inference.

This module provides functionality for defining and creating prior distributions
for gravitational wave background and pulsar noise parameters used in
pulsar timing array analysis.
"""

import json

import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp

tfpd = tfp.distributions


def get_gw_parameter_priors(config):
    """Extract gravitational wave parameter prior distributions from config.

    Parameters
    ----------
    config : ConfigParser
        Configuration object containing prior model settings

    Returns
    -------
    dict
        Dictionary containing GW parameter prior distributions:
        - log10_ha_spec: Prior distribution for log10(ha)
        - log10_ha_transform_params: Transformation parameters for reparameterization
        - log10_gamma_a_spec: Prior distribution for log10(γa)

    Notes
    -----
    ``gw_parameterization`` (fallback ``direct``) selects the sampling basis:

    - ``direct``: independent reparameterized priors on log10_ha and
      log10_gamma_a (the original behavior; unchanged when the key is absent).
    - ``ridge``: sample the band-referenced pivot log-PSD and log10_gamma_a as
      independent coordinates, deriving log10_ha deterministically. The pivot
      log-PSD is the direction the data actually constrains, so this decouples
      it from the flat along-ridge direction and straightens the curved
      log10_ha<->log10_gamma_a ridge that stalls NUTS chains (issue #109).
      Reads log10_pivot_psd_{min,max}, gw_pivot_freq_hz (fallback 1/(5 yr)),
      and log10_gamma_a_{min,max}.
    """

    def _reparam(min_val, max_val):
        """N(0,1)-reparameterized uniform: returns (mean, std) for the 3-sigma map."""
        return (min_val + max_val) / 2.0, (max_val - min_val) / 6.0

    gw_parameterization = (
        config.get("PriorModel", "gw_parameterization", fallback="direct")
        .strip()
        .lower()
    )
    if gw_parameterization == "ridge":
        import math

        psd_min = config.getfloat("PriorModel", "log10_pivot_psd_min")
        psd_max = config.getfloat("PriorModel", "log10_pivot_psd_max")
        ga_min = config.getfloat("PriorModel", "log10_gamma_a_min")
        ga_max = config.getfloat("PriorModel", "log10_gamma_a_max")
        f_piv = config.getfloat(
            "PriorModel", "gw_pivot_freq_hz", fallback=1.0 / (5.0 * 365.25 * 86400.0)
        )
        psd_mean, psd_std = _reparam(psd_min, psd_max)
        ga_mean, ga_std = _reparam(ga_min, ga_max)
        return {
            "gw_parameterization": "ridge",
            # log10_ha is derived; keep the direct-mode keys present (None) so
            # count_free_parameters / display code that reads them stay happy.
            "log10_ha_spec": None,
            "log10_ha_transform_params": None,
            "log10_gamma_a_spec": None,
            "log10_pivot_psd_transform_params": {
                "mean": psd_mean,
                "std": psd_std,
                "min": psd_min,
                "max": psd_max,
            },
            "log10_gamma_a_transform_params": {
                "mean": ga_mean,
                "std": ga_std,
                "min": ga_min,
                "max": ga_max,
            },
            # angular pivot frequency w = 2*pi*f_piv, used in the ha inversion
            "gw_pivot_w": 2.0 * math.pi * f_piv,
            **_get_orf_path_specs(config),
        }

    # Helper function to create prior spec based on fixed/sampled setting
    def get_prior_spec(param_name):
        is_fixed = config.getboolean("PriorModel", f"{param_name}_fixed")
        if is_fixed:
            return config.getfloat("PriorModel", f"{param_name}_value")
        else:
            min_val = config.getfloat("PriorModel", f"{param_name}_min")
            max_val = config.getfloat("PriorModel", f"{param_name}_max")
            return tfpd.Uniform(min_val, max_val)

    # Handle log10_ha with reparameterization for better NUTS sampling
    log10_ha_fixed = config.getboolean("PriorModel", "log10_ha_fixed")
    if log10_ha_fixed:
        # Fixed value - no reparameterization needed
        log10_ha_spec = config.getfloat("PriorModel", "log10_ha_value")
        log10_ha_transform_params = None
    else:
        # Reparameterize U(a,b) -> N(0,1) for better NUTS sampling
        min_val = config.getfloat("PriorModel", "log10_ha_min")
        max_val = config.getfloat("PriorModel", "log10_ha_max")

        # Calculate improved transformation parameters: log10_ha = mean + log10_ha_prime * std
        # Use 3-sigma rule for better convergence
        mean = (min_val + max_val) / 2.0
        std = (max_val - min_val) / 6.0  # 3-sigma rule: 99.7% of samples within range

        # Use N(0,1) for log10_ha_prime, store transformation parameters
        log10_ha_spec = tfpd.Normal(0.0, 1.0)  # log10_ha_prime ~ N(0,1)
        log10_ha_transform_params = {
            "mean": mean,
            "std": std,
            "min": min_val,
            "max": max_val,
        }

    log10_gamma_a_spec = get_prior_spec("log10_gamma_a")

    return {
        "gw_parameterization": "direct",
        "log10_ha_spec": log10_ha_spec,
        "log10_ha_transform_params": log10_ha_transform_params,
        "log10_gamma_a_spec": log10_gamma_a_spec,
        **_get_orf_path_specs(config),
    }


def _get_orf_path_specs(config):
    """Parse the correlation-path settings from ``[PriorModel]``.

    The correlation path embeds the Hellings-Downs and CURN hypotheses in one
    continuous family ``C(ε) = (1 - ε)·I + ε·C_HD`` (see
    ``gravitational_waves.correlation_path``). ``orf_path`` selects the mode:

    - ``off`` (fallback): no path. The Hellings-Downs matrix is used as supplied
      and the likelihood is byte-identical to the pre-path behavior.
    - ``fixed``: ``ε`` is held at ``orf_epsilon_value`` — one rung of the path
      sampling ladder (estimator B).
    - ``sampled``: ``ε`` is sampled under ``Uniform(orf_epsilon_min,
      orf_epsilon_max)``, fallback bounds (0, 1) — the single-run generalised
      Savage-Dickey route (estimator A).

    A genuinely uniform prior is used rather than the N(0,1) reparameterization
    the other parameters take, because the Bayes factor is read from the prior and
    posterior densities *at the endpoints*: a uniform prior makes the prior terms
    cancel, and bounds keep ``C(ε)`` positive definite (outside [0, 1] the
    interpolation can over-weight the negative Hellings-Downs off-diagonals).

    Returns
    -------
    dict
        Keys ``orf_path``, ``orf_epsilon_value`` and ``orf_epsilon_bounds``.
        With the path off, the latter two are None.
    """
    orf_path = config.get("PriorModel", "orf_path", fallback="off").strip().lower()

    if orf_path == "off":
        return {
            "orf_path": "off",
            "orf_epsilon_value": None,
            "orf_epsilon_bounds": None,
        }

    if orf_path == "fixed":
        return {
            "orf_path": "fixed",
            "orf_epsilon_value": config.getfloat("PriorModel", "orf_epsilon_value"),
            "orf_epsilon_bounds": None,
        }

    if orf_path == "sampled":
        low = config.getfloat("PriorModel", "orf_epsilon_min", fallback=0.0)
        high = config.getfloat("PriorModel", "orf_epsilon_max", fallback=1.0)
        if not high > low:
            raise ValueError(
                f"orf_epsilon_max ({high}) must exceed orf_epsilon_min ({low})."
            )
        return {
            "orf_path": "sampled",
            "orf_epsilon_value": None,
            "orf_epsilon_bounds": (low, high),
        }

    raise ValueError(
        f"Unknown orf_path '{orf_path}'. Expected 'off', 'fixed' or 'sampled'."
    )


def get_pulsar_noise_priors(config, n_pulsars, sigma_p_array, gamma_p_array):
    """Extract pulsar red noise parameter prior distributions from config.

    Parameters
    ----------
    config : ConfigParser
        Configuration object containing prior model settings
    n_pulsars : int
        Number of pulsars
    sigma_p_array : array
        Array of pulsar red noise sigma values
    gamma_p_array : array
        Array of pulsar red noise gamma values

    Returns
    -------
    dict
        Dictionary containing pulsar noise parameter prior distributions:
        - log10_gamma_p_spec: Prior distribution for log10(γp)
        - log10_sigma_p_spec: Prior distribution for log10(σp)
        - hierarchical_specs: Hierarchical modeling prior distributions
        - empirical_specs: Per-pulsar empirical Normal priors (or None)

    Notes
    -----
    Red-noise prior precedence:
    1. ``spin_injections_path`` non-empty -> parameters FIXED.
    2. ``empirical_priors_path`` non-empty -> per-pulsar Normal priors on
       (log10_γp, log10_ratio) from single-pulsar posteriors (Stage C of the
       two-stage noise procedure, issue #111).
    3. ``red_noise_prior = flat`` -> independent Uniform priors per pulsar,
       intended for single-pulsar (n=1) noise-characterization runs where the
       shared population hyperpriors are degenerate.
    4. otherwise -> hierarchical modeling with shared hyperpriors (default).
    """
    # Check if spin_injections_path is provided to determine if red noise parameters should be fixed
    try:
        spin_injections_path = config.get("PriorModel", "spin_injections_path")
        # If path is provided and not empty, fix red noise parameters
        log10_gamma_p_fixed = bool(spin_injections_path.strip())
        log10_sigma_p_fixed = bool(spin_injections_path.strip())
        print(
            f"Red noise parameters fixed via spin_injections_path: {log10_gamma_p_fixed}"
        )
    except Exception:
        # If no spin_injections_path, sample from priors
        log10_gamma_p_fixed = False
        log10_sigma_p_fixed = False
        print(
            "No spin_injections_path provided, sampling red noise parameters from priors"
        )

    if not (log10_gamma_p_fixed or log10_sigma_p_fixed):
        empirical_priors_path = config.get(
            "PriorModel", "empirical_priors_path", fallback=""
        ).strip()
        if empirical_priors_path:
            excluded_psrs = [
                psr.strip()
                for psr in config.get("Data", "excluded_psrs", fallback="").split(",")
                if psr.strip()
            ]
            inflation = config.getfloat(
                "PriorModel", "empirical_prior_inflation", fallback=1.0
            )
            empirical_specs = get_empirical_noise_priors(
                empirical_priors_path, excluded_psrs, inflation
            )
            n_loaded = len(empirical_specs["psr_names"])
            if n_loaded != n_pulsars:
                raise ValueError(
                    f"Empirical priors file {empirical_priors_path} yields {n_loaded} "
                    f"pulsars after exclusions but the loaded data has {n_pulsars}. "
                    "Pulsar sets must match exactly (sorted by name)."
                )
            print(
                f"Using empirical per-pulsar red noise priors from: {empirical_priors_path} "
                f"(scale inflation x{inflation})"
            )
            return {
                "log10_gamma_p_spec": None,
                "log10_sigma_p_spec": None,
                "hierarchical_specs": None,
                "empirical_specs": empirical_specs,
            }

        red_noise_prior = (
            config.get("PriorModel", "red_noise_prior", fallback="hierarchical")
            .strip()
            .lower()
        )
        if red_noise_prior == "flat":
            print(
                "Using flat (independent Uniform) per-pulsar red noise priors "
                "(red_noise_prior = flat)"
            )
            log10_gamma_p_spec = tfpd.Uniform(
                low=jnp.full(
                    n_pulsars, config.getfloat("PriorModel", "log10_gamma_p_min")
                ),
                high=jnp.full(
                    n_pulsars, config.getfloat("PriorModel", "log10_gamma_p_max")
                ),
            )
            log10_sigma_p_spec = tfpd.Uniform(
                low=jnp.full(
                    n_pulsars, config.getfloat("PriorModel", "log10_sigma_p_min")
                ),
                high=jnp.full(
                    n_pulsars, config.getfloat("PriorModel", "log10_sigma_p_max")
                ),
            )
            return {
                "log10_gamma_p_spec": log10_gamma_p_spec,
                "log10_sigma_p_spec": log10_sigma_p_spec,
                "hierarchical_specs": None,
                "empirical_specs": None,
            }

    # Always use hierarchical modeling and log-ratio parameterization
    hierarchical_specs = create_hierarchical_priors(config)

    # Handle gamma_p specification
    if log10_gamma_p_fixed:
        if config.has_option("PriorModel", "log10_gamma_p_value"):
            # Check if value is a string (for 'injected'/'default') or a number
            gamma_p_value_str = config.get("PriorModel", "log10_gamma_p_value")
            if gamma_p_value_str.lower() in ["injected", "default"]:
                # Use injected values
                log10_gamma_p_spec = jnp.log10(gamma_p_array)
                print(f"Using injected gamma_p values: {gamma_p_value_str}")
            else:
                # Use explicit fixed value from config
                gamma_p_fixed_value = config.getfloat(
                    "PriorModel", "log10_gamma_p_value"
                )
                log10_gamma_p_spec = jnp.full(n_pulsars, gamma_p_fixed_value)
                print(f"Using fixed gamma_p value: {gamma_p_fixed_value}")
        else:
            # Use injected values (legacy approach)
            log10_gamma_p_spec = jnp.log10(gamma_p_array)
            print("Using injected gamma_p values (legacy mode)")
    else:
        log10_gamma_p_spec = None  # Will be handled hierarchically

    # Handle sigma_p specification
    if log10_sigma_p_fixed:
        if config.has_option("PriorModel", "log10_sigma_p_value"):
            # Check if value is a string (for 'injected'/'default') or a number
            sigma_p_value_str = config.get("PriorModel", "log10_sigma_p_value")
            if sigma_p_value_str.lower() in ["injected", "default"]:
                # Use injected values
                log10_sigma_p_spec = jnp.log10(sigma_p_array)
                print(f"Using injected sigma_p values: {sigma_p_value_str}")
            else:
                # Use explicit fixed value from config
                sigma_p_fixed_value = config.getfloat(
                    "PriorModel", "log10_sigma_p_value"
                )
                log10_sigma_p_spec = jnp.full(n_pulsars, sigma_p_fixed_value)
                print(f"Using fixed sigma_p value: {sigma_p_fixed_value}")
        else:
            # Use injected values (legacy approach)
            log10_sigma_p_spec = jnp.log10(sigma_p_array)
            print("Using injected sigma_p values (legacy mode)")
    else:
        log10_sigma_p_spec = None  # Will be derived from log-ratio

    return {
        "log10_gamma_p_spec": log10_gamma_p_spec,
        "log10_sigma_p_spec": log10_sigma_p_spec,
        "hierarchical_specs": hierarchical_specs,
        "empirical_specs": None,
    }


def get_empirical_noise_priors(empirical_priors_path, excluded_psrs=[], inflation=1.0):
    """Load per-pulsar empirical red noise priors from a JSON file.

    The file is produced by Stage A single-pulsar noise runs (see
    workflows/ng15_sgwb_demo/scripts/extract_stage_a.py) and maps each pulsar
    name to Normal-prior (loc, scale) pairs for log10_γp and the log10 ratio
    σp/γp:

        {"J0030+0451": {"log10_gamma_p": {"loc": -8.1, "scale": 0.2},
                        "log10_ratio":   {"loc": -6.3, "scale": 0.3}}, ...}

    Keys starting with "_" (e.g. "_meta") are ignored. Pulsars are sorted by
    name to match the data loader's sorted-glob ordering.

    Parameters
    ----------
    empirical_priors_path : str
        Path to the empirical priors JSON file
    excluded_psrs : list of str, optional
        Pulsar names to exclude (substring match, same semantics as
        utils.get_efac_equad_injections)
    inflation : float, optional
        Multiplicative inflation factor applied to the prior scales
        (single-pulsar posteriors can be overconfident). Default 1.0.

    Returns
    -------
    dict
        {"gamma_loc", "gamma_scale", "ratio_loc", "ratio_scale"} as JAX arrays
        of shape (n_pulsars,), plus "psr_names" (sorted list of str).
    """
    with open(empirical_priors_path, "r") as f:
        empirical_priors = json.load(f)

    psr_names = sorted(
        psr
        for psr in empirical_priors
        if not psr.startswith("_")
        and not any(excluded_psr in psr for excluded_psr in excluded_psrs)
    )
    if not psr_names:
        raise ValueError(f"No pulsars left in {empirical_priors_path} after exclusions")

    gamma_loc = jnp.array(
        [empirical_priors[psr]["log10_gamma_p"]["loc"] for psr in psr_names]
    )
    gamma_scale = inflation * jnp.array(
        [empirical_priors[psr]["log10_gamma_p"]["scale"] for psr in psr_names]
    )
    ratio_loc = jnp.array(
        [empirical_priors[psr]["log10_ratio"]["loc"] for psr in psr_names]
    )
    ratio_scale = inflation * jnp.array(
        [empirical_priors[psr]["log10_ratio"]["scale"] for psr in psr_names]
    )

    return {
        "gamma_loc": gamma_loc,
        "gamma_scale": gamma_scale,
        "ratio_loc": ratio_loc,
        "ratio_scale": ratio_scale,
        "psr_names": psr_names,
    }


def get_measurement_noise_priors(config, n_pulsars, efac_array, equad_array):
    """Extract measurement noise parameter prior distributions from config.

    Parameters
    ----------
    config : ConfigParser
        Configuration object containing prior model settings
    n_pulsars : int
        Number of pulsars
    efac_array : array or None
        Array of EFAC values, or None if not provided
    equad_array : array or None
        Array of EQUAD values, or None if not provided

    Returns
    -------
    dict
        Dictionary containing measurement noise parameter prior distributions:
        - efac_spec: Prior distribution for EFAC
        - equad_spec: Prior distribution for EQUAD
    """
    # Check if noise_params_path is provided to determine if EFAC/EQUAD should be fixed
    try:
        noise_params_path = config.get("PriorModel", "noise_params_path")
        # If path is provided and not empty, fix EFAC/EQUAD parameters
        efac_equad_fixed = bool(noise_params_path.strip())
        print(f"EFAC/EQUAD parameters fixed via noise_params_path: {efac_equad_fixed}")
    except Exception:
        # If no noise_params_path, sample from priors
        efac_equad_fixed = False
        print("No noise_params_path provided, sampling EFAC/EQUAD from priors")

    if efac_equad_fixed and efac_array is not None and equad_array is not None:
        efac_spec = efac_array
        equad_spec = equad_array
    else:
        # Create prior distributions using the number of pulsars to determine shape
        efac_spec = tfpd.Uniform(
            low=jnp.full(n_pulsars, config.getfloat("PriorModel", "efac_min")),
            high=jnp.full(n_pulsars, config.getfloat("PriorModel", "efac_max")),
        )

        # Use log10(EQUAD) uniform prior - transformation handled in numpyro model
        log10_equad_spec = tfpd.Uniform(
            low=jnp.full(n_pulsars, config.getfloat("PriorModel", "log10_equad_min")),
            high=jnp.full(n_pulsars, config.getfloat("PriorModel", "log10_equad_max")),
        )
        equad_spec = {"log10_equad_spec": log10_equad_spec, "use_log10": True}

    return {"efac_spec": efac_spec, "equad_spec": equad_spec}


def create_hierarchical_priors(config):
    """Create hierarchical modeling prior distributions.

    Always creates hierarchical priors for both gamma_p and sigma_p (via log-ratio)
    parameterization to improve MCMC sampling efficiency.

    Parameters
    ----------
    config : ConfigParser
        Configuration object containing hyperprior ranges

    Returns
    -------
    dict
        Hierarchical prior distributions dictionary
    """
    hierarchical_specs = {
        "hierarchical_noise": True,
        "log_ratio_parameterization": True,
        "log10_gamma_p_mean_spec": tfpd.Uniform(
            config.getfloat("PriorModel", "log10_gamma_p_mean_min"),
            config.getfloat("PriorModel", "log10_gamma_p_mean_max"),
        ),
        "log10_gamma_p_std_spec": tfpd.Uniform(
            config.getfloat("PriorModel", "log10_gamma_p_std_min"),
            config.getfloat("PriorModel", "log10_gamma_p_std_max"),
        ),
        "log10_ratio_mean_spec": tfpd.Uniform(
            config.getfloat("PriorModel", "log10_ratio_mean_min"),
            config.getfloat("PriorModel", "log10_ratio_mean_max"),
        ),
        "log10_ratio_std_spec": tfpd.Uniform(
            config.getfloat("PriorModel", "log10_ratio_std_min"),
            config.getfloat("PriorModel", "log10_ratio_std_max"),
        ),
    }

    return hierarchical_specs


def _make_reparameterized_prior(config, section, param_name):
    """Create a reparameterized prior from config settings.

    Returns (spec, transform_params) tuple. If fixed, transform_params is None.
    """
    is_fixed = config.getboolean(section, f"{param_name}_fixed")
    if is_fixed:
        return config.getfloat(section, f"{param_name}_value"), None
    else:
        min_val = config.getfloat(section, f"{param_name}_min")
        max_val = config.getfloat(section, f"{param_name}_max")
        mean = (min_val + max_val) / 2.0
        std = (max_val - min_val) / 6.0
        return None, {"mean": mean, "std": std, "min": min_val, "max": max_val}


def get_cw_parameter_priors(config):
    """Extract CW source parameter prior distributions from config.

    Parameters
    ----------
    config : ConfigParser
        Configuration object containing [CWModel] section.

    Returns
    -------
    dict
        Dictionary containing CW parameter prior specifications.
    """
    section = "CWModel"
    cw_specs = {}

    # log10_h0: strain amplitude
    spec, tp = _make_reparameterized_prior(config, section, "log10_h0")
    cw_specs["log10_h0_spec"] = spec
    cw_specs["log10_h0_transform_params"] = tp

    # alpha_gw: source RA
    spec, tp = _make_reparameterized_prior(config, section, "alpha_gw")
    cw_specs["alpha_gw_spec"] = spec
    cw_specs["alpha_gw_transform_params"] = tp

    # sin_delta_gw: for isotropic sky coverage (sample in sin(delta), convert to delta)
    spec, tp = _make_reparameterized_prior(config, section, "sin_delta_gw")
    cw_specs["sin_delta_gw_spec"] = spec
    cw_specs["sin_delta_gw_transform_params"] = tp
    # Also store delta_gw_spec for fixed case (direct declination)
    if config.getboolean(section, "sin_delta_gw_fixed"):
        cw_specs["delta_gw_spec"] = config.getfloat(section, "delta_gw_value")
    else:
        cw_specs["delta_gw_spec"] = None

    # log10_f_gw: GW frequency
    spec, tp = _make_reparameterized_prior(config, section, "log10_f_gw")
    cw_specs["log10_f_gw_spec"] = spec
    cw_specs["log10_f_gw_transform_params"] = tp

    # cos_iota: inclination
    spec, tp = _make_reparameterized_prior(config, section, "cos_iota")
    cw_specs["cos_iota_spec"] = spec
    cw_specs["cos_iota_transform_params"] = tp

    # psi: polarization angle
    spec, tp = _make_reparameterized_prior(config, section, "psi")
    cw_specs["psi_spec"] = spec
    cw_specs["psi_transform_params"] = tp

    # Phi0: initial phase
    spec, tp = _make_reparameterized_prior(config, section, "Phi0")
    cw_specs["Phi0_spec"] = spec
    cw_specs["Phi0_transform_params"] = tp

    # Per-pulsar phase parameters (phase reparameterization of pulsar term)
    include_pulsar_term = config.getboolean(
        section, "include_pulsar_term", fallback=False
    )
    phase_parameterization = config.getboolean(
        section, "phase_parameterization", fallback=True
    )

    if include_pulsar_term and phase_parameterization:
        import math

        chi_min = 0.0
        chi_max = 2.0 * math.pi
        mean = (chi_min + chi_max) / 2.0
        std = (chi_max - chi_min) / 6.0
        cw_specs["chi_transform_params"] = {
            "mean": mean,
            "std": std,
            "min": chi_min,
            "max": chi_max,
        }
    else:
        cw_specs["chi_transform_params"] = None

    return cw_specs


def get_prior_model_specs(
    config,
    n_pulsars,
    sigma_p_array,
    gamma_p_array,
    efac_array,
    equad_array,
    mode="gwb",
):
    """Create prior model distributions based on config settings.

    Parameters
    ----------
    config : ConfigParser
        Configuration object containing prior model settings
    n_pulsars : int
        Number of pulsars
    sigma_p_array : array
        Array of pulsar red noise sigma values
    gamma_p_array : array
        Array of pulsar red noise gamma values
    efac_array : array
        Array of EFAC values
    equad_array : array
        Array of EQUAD values
    mode : str
        Signal model mode: 'gwb' or 'cw'.

    Returns
    -------
    dict
        Dictionary containing all prior distributions.
    """
    print(f"Getting prior model specs (mode={mode})...")

    # Pulsar noise and measurement noise priors are shared between modes
    pulsar_noise_specs = get_pulsar_noise_priors(
        config, n_pulsars, sigma_p_array, gamma_p_array
    )
    measurement_noise_specs = get_measurement_noise_priors(
        config, n_pulsars, efac_array, equad_array
    )

    result = {
        "log10_gamma_p_spec": pulsar_noise_specs["log10_gamma_p_spec"],
        "log10_sigma_p_spec": pulsar_noise_specs["log10_sigma_p_spec"],
        "efac_spec": measurement_noise_specs["efac_spec"],
        "equad_spec": measurement_noise_specs["equad_spec"],
        "hierarchical_specs": pulsar_noise_specs["hierarchical_specs"],
        "empirical_specs": pulsar_noise_specs.get("empirical_specs"),
    }

    if mode == "cw":
        # CW-specific priors (no GWB amplitude/spectral index)
        cw_specs = get_cw_parameter_priors(config)
        result["cw_specs"] = cw_specs
        # Dummy GWB entries for backward compatibility with count_free_parameters
        result["log10_ha_spec"] = None
        result["log10_ha_transform_params"] = None
        result["log10_gamma_a_spec"] = None
    else:
        # GWB-specific priors (direct or ridge parameterization). Copy every key
        # get_gw_parameter_priors returns so the ridge-mode extras
        # (gw_parameterization, *_transform_params, gw_pivot_w) pass through.
        result.update(get_gw_parameter_priors(config))

    return result

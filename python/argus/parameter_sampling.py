"""Parameter sampling functions for NumPyro models.

This module provides functions for sampling parameters from their priors
in NumPyro models, including support for hierarchical modeling and
reparameterization techniques for improved NUTS sampling.
"""

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import tensorflow_probability.substrates.jax as tfp

tfpd = tfp.distributions


# Single source of truth for the CW source scalars. Each entry is
# (name, fixed_spec_key, derived):
#   - name           -> the "<name>_transform_params" / "<name>_spec" keys and
#                        the sample-site name; preserved exactly so results
#                        parsing (utils.corner_plot, count_free_parameters) keeps
#                        keying off the same names.
#   - fixed_spec_key -> spec key holding the fixed value when not reparameterized.
#   - derived        -> (det_name, fn) when the sampled quantity is transformed
#                        into a second deterministic (sin_delta_gw -> delta_gw).
CW_SCALAR_PARAMS = [
    ("log10_h0", "log10_h0_spec", None),
    ("alpha_gw", "alpha_gw_spec", None),
    ("sin_delta_gw", "delta_gw_spec", ("delta_gw", jnp.arcsin)),
    ("log10_f_gw", "log10_f_gw_spec", None),
    ("cos_iota", "cos_iota_spec", None),
    ("psi", "psi_spec", None),
    ("Phi0", "Phi0_spec", None),
]


def _sample_cw_scalar_numpyro(cw_specs, name, fixed_spec_key, derived):
    """Sample (or fix) one CW scalar in the numpyro path.

    Reparameterized: x_prime ~ Normal(0,1), x = mean + std * x_prime, both
    registered as deterministics. Fixed: a single deterministic at the fixed
    value. The ``derived`` transform (sin_delta_gw -> delta_gw) is applied only
    to the sampled value; when fixed, the spec already holds the derived
    quantity. Returns the final value used by the likelihood.
    """
    tp = cw_specs[f"{name}_transform_params"]
    if tp is not None:
        prime = numpyro.sample(f"{name}_prime", dist.Normal(0.0, 1.0))
        value = numpyro.deterministic(name, tp["mean"] + prime * tp["std"])
        if derived is not None:
            det_name, fn = derived
            value = numpyro.deterministic(det_name, fn(value))
        return value
    out_name = derived[0] if derived is not None else name
    return numpyro.deterministic(out_name, cw_specs[fixed_spec_key])


def sample_gw_parameters(prior_specs):
    """Sample gravitational wave parameters from their priors.

    Parameters
    ----------
    prior_specs : dict
        Prior distributions dictionary

    Returns
    -------
    tuple
        (log10_ha, log10_gamma_a, γa) values
    """
    # Ridge parameterization (issue #109): sample the band-referenced pivot
    # log-PSD and log10_gamma_a as independent coordinates, then derive log10_ha
    # deterministically. The pivot log-PSD is the direction the data constrains,
    # so this decouples it from the flat along-ridge direction and straightens
    # the curved log10_ha<->log10_gamma_a ridge that stalls NUTS chains. The
    # likelihood still sees (log10_ha, log10_gamma_a); only the prior/sampling
    # basis changes (uniform on (pivot log-PSD, log10_gamma_a)).
    if prior_specs.get("gw_parameterization") == "ridge":
        psd_tp = prior_specs["log10_pivot_psd_transform_params"]
        ga_tp = prior_specs["log10_gamma_a_transform_params"]

        log10_pivot_psd_prime = numpyro.sample(
            "log10_pivot_psd_prime", dist.Normal(0.0, 1.0)
        )
        log10_pivot_psd = numpyro.deterministic(
            "log10_pivot_psd", psd_tp["mean"] + log10_pivot_psd_prime * psd_tp["std"]
        )

        log10_gamma_a_prime = numpyro.sample(
            "log10_gamma_a_prime", dist.Normal(0.0, 1.0)
        )
        log10_gamma_a = numpyro.deterministic(
            "log10_gamma_a", ga_tp["mean"] + log10_gamma_a_prime * ga_tp["std"]
        )
        γa = numpyro.deterministic("γa", 10.0**log10_gamma_a)

        # Invert S_r(f_piv) = (ha^2/12) * ga / (w^2 (ga^2 + w^2)) for log10_ha:
        #   log10_ha = 0.5[log10(12) + log10_S_r + 2 log10(w)
        #                  + log10(ga^2 + w^2) - log10_gamma_a]
        w = prior_specs["gw_pivot_w"]
        log10_ha = numpyro.deterministic(
            "log10_ha",
            0.5
            * (
                jnp.log10(12.0)
                + log10_pivot_psd
                + 2.0 * jnp.log10(w)
                + jnp.log10(γa**2 + w**2)
                - log10_gamma_a
            ),
        )
        return log10_ha, log10_gamma_a, γa

    # Handle log10_ha: either fixed value or reparameterized sampling
    if prior_specs["log10_ha_transform_params"] is not None:
        # Sample log10_ha_prime ~ N(0,1) and transform to log10_ha for efficient NUTS sampling
        transform_params = prior_specs["log10_ha_transform_params"]
        log10_ha_prime = numpyro.sample("log10_ha_prime", dist.Normal(0.0, 1.0))
        log10_ha = numpyro.deterministic(
            "log10_ha",
            transform_params["mean"] + log10_ha_prime * transform_params["std"],
        )
    else:
        # Fixed value (delta prior)
        log10_ha = numpyro.deterministic("log10_ha", prior_specs["log10_ha_spec"])

    # Handle log10_gamma_a: either fixed value or reparameterized sampling
    if isinstance(prior_specs["log10_gamma_a_spec"], tfpd.Distribution):
        # Use reparameterization for efficient NUTS sampling
        low = prior_specs["log10_gamma_a_spec"].low
        high = prior_specs["log10_gamma_a_spec"].high
        mean = (low + high) / 2.0
        std = (high - low) / 6.0  # 3-sigma rule

        log10_gamma_a_prime = numpyro.sample(
            "log10_gamma_a_prime", dist.Normal(0.0, 1.0)
        )
        log10_gamma_a = numpyro.deterministic(
            "log10_gamma_a", mean + log10_gamma_a_prime * std
        )
        γa = numpyro.deterministic("γa", 10.0**log10_gamma_a)
    else:
        # Fixed value
        log10_gamma_a = numpyro.deterministic(
            "log10_gamma_a", prior_specs["log10_gamma_a_spec"]
        )
        γa = numpyro.deterministic("γa", 10.0**log10_gamma_a)

    return log10_ha, log10_gamma_a, γa


def sample_orf_epsilon(prior_specs):
    """Sample (or fix) the correlation-path coordinate ``ε``.

    ``ε`` interpolates the overlap reduction function between the identity (CURN,
    ``ε = 0``) and Hellings-Downs (``ε = 1``); see
    ``gravitational_waves.correlation_path``. Returns None when the path is off,
    which leaves the likelihood on its original code path.

    Parameters
    ----------
    prior_specs : dict
        Prior distributions dictionary.

    Returns
    -------
    float, jax.Array or None
        The value of ``ε`` to hand to the filter, or None when the path is off.
    """
    orf_path = prior_specs.get("orf_path", "off")

    if orf_path == "off":
        return None

    if orf_path == "fixed":
        # One rung of the path-sampling ladder: ε is a constant, recorded as a
        # deterministic so every run's outputs say where on the path they sit.
        return numpyro.deterministic(
            "orf_epsilon", jnp.asarray(prior_specs["orf_epsilon_value"])
        )

    if orf_path == "sampled":
        low, high = prior_specs["orf_epsilon_bounds"]
        return numpyro.sample("orf_epsilon", dist.Uniform(low, high))

    raise ValueError(f"Unknown orf_path '{orf_path}'.")


def sample_hierarchical_gamma_parameters(hierarchical_specs, n_pulsars):
    """Sample hierarchical gamma parameters with gradient balancing.

    Parameters
    ----------
    hierarchical_specs : dict
        Hierarchical modeling specifications
    n_pulsars : int
        Number of pulsars

    Returns
    -------
    jax.Array
        Sampled log10_γp values
    """
    # Hierarchical modeling for log10_gamma_p with gradient balancing
    log10_gamma_p_mean_raw = numpyro.sample(
        "log10_gamma_p_mean_raw", dist.Normal(0.0, 1.0)
    )
    log10_gamma_p_std_raw = numpyro.sample(
        "log10_gamma_p_std_raw", dist.Normal(0.0, 1.0)
    )

    # Transform to appropriate ranges with balanced gradients
    mean_low = hierarchical_specs["log10_gamma_p_mean_spec"].low
    mean_high = hierarchical_specs["log10_gamma_p_mean_spec"].high
    std_low = hierarchical_specs["log10_gamma_p_std_spec"].low
    std_high = hierarchical_specs["log10_gamma_p_std_spec"].high

    # Apply gradient-balanced transforms
    log10_gamma_p_mean = numpyro.deterministic(
        "log10_gamma_p_mean",
        (mean_low + mean_high) / 2.0
        + log10_gamma_p_mean_raw * (mean_high - mean_low) / 6.0,
    )
    log10_gamma_p_std = numpyro.deterministic(
        "log10_gamma_p_std",
        (std_low + std_high) / 2.0 + log10_gamma_p_std_raw * (std_high - std_low) / 6.0,
    )

    # Sample individual pulsar parameters with scaled gradients
    log10_γp_raw = numpyro.sample(
        "log10_γp_raw", dist.Normal(jnp.zeros(n_pulsars), jnp.ones(n_pulsars))
    )
    log10_γp = numpyro.deterministic(
        "log10_γp",
        log10_gamma_p_mean + log10_γp_raw * log10_gamma_p_std / jnp.sqrt(n_pulsars),
    )

    return log10_γp


def sample_reparameterized_parameters(prior_spec, param_name, n_pulsars):
    """Sample parameters using reparameterization for efficient NUTS sampling.

    Parameters
    ----------
    prior_spec : tfpd.Distribution
        Uniform distribution specification
    param_name : str
        Name of the parameter for sampling
    n_pulsars : int
        Number of pulsars

    Returns
    -------
    jax.Array
        Sampled parameter values
    """
    # Reparameterize uniform distribution using Normal(0,1) + affine transformation
    low = prior_spec.low
    high = prior_spec.high
    mean = (low + high) / 2.0
    std = (high - low) / 6.0  # 3-sigma rule

    param_standardized = numpyro.sample(
        f"{param_name}_standardized",
        dist.Normal(jnp.zeros(n_pulsars), jnp.ones(n_pulsars) / jnp.sqrt(n_pulsars)),
    )
    param_values = numpyro.deterministic(param_name, mean + param_standardized * std)

    return param_values


def sample_log_ratio_parameters(hierarchical_specs, log10_γp, n_pulsars):
    """Sample log-ratio parameters for sigma_p derivation.

    Parameters
    ----------
    hierarchical_specs : dict
        Hierarchical modeling specifications
    log10_γp : jax.Array
        Log10 gamma_p values
    n_pulsars : int
        Number of pulsars

    Returns
    -------
    jax.Array
        Derived log10_σp values
    """
    # Log-ratio parameterization: σp derived from γp + ratio with gradient balancing
    log10_ratio_mean_raw = numpyro.sample("log10_ratio_mean_raw", dist.Normal(0.0, 1.0))
    log10_ratio_std_raw = numpyro.sample("log10_ratio_std_raw", dist.Normal(0.0, 1.0))

    # Transform to appropriate ranges with balanced gradients
    mean_low = hierarchical_specs["log10_ratio_mean_spec"].low
    mean_high = hierarchical_specs["log10_ratio_mean_spec"].high
    std_low = hierarchical_specs["log10_ratio_std_spec"].low
    std_high = hierarchical_specs["log10_ratio_std_spec"].high

    # Apply gradient-balanced transforms
    log10_ratio_mean = numpyro.deterministic(
        "log10_ratio_mean",
        (mean_low + mean_high) / 2.0
        + log10_ratio_mean_raw * (mean_high - mean_low) / 6.0,
    )
    log10_ratio_std = numpyro.deterministic(
        "log10_ratio_std",
        (std_low + std_high) / 2.0 + log10_ratio_std_raw * (std_high - std_low) / 6.0,
    )

    # Sample individual ratio parameters with scaled gradients
    log10_ratio_raw = numpyro.sample(
        "log10_ratio_raw", dist.Normal(jnp.zeros(n_pulsars), jnp.ones(n_pulsars))
    )
    log10_ratio = numpyro.deterministic(
        "log10_ratio",
        log10_ratio_mean + log10_ratio_raw * log10_ratio_std / jnp.sqrt(n_pulsars),
    )

    # Derive log10_σp deterministically from γp + ratio
    log10_σp = numpyro.deterministic("log10_σp", log10_γp + log10_ratio)

    return log10_σp


def sample_empirical_noise_parameters(empirical_specs, n_pulsars):
    """Sample pulsar red noise parameters from per-pulsar empirical priors.

    Empirical mode (Stage C of the two-stage noise procedure, issue #111):
    each pulsar gets an independent Normal prior on log10_γp and on the
    log-ratio σp/γp, with (loc, scale) taken from single-pulsar posteriors.
    No population hyperparameters are sampled. The latent sites reuse the
    hierarchical-mode names (log10_γp_raw, log10_ratio_raw) as unit Gaussians
    so downstream consumers (corner plots, logz_lhm.py) work unchanged.

    Parameters
    ----------
    empirical_specs : dict
        Output of prior_models.get_empirical_noise_priors
    n_pulsars : int
        Number of pulsars

    Returns
    -------
    tuple
        (log10_γp, log10_σp) values
    """
    log10_γp_raw = numpyro.sample(
        "log10_γp_raw", dist.Normal(jnp.zeros(n_pulsars), jnp.ones(n_pulsars))
    )
    log10_γp = numpyro.deterministic(
        "log10_γp",
        empirical_specs["gamma_loc"] + log10_γp_raw * empirical_specs["gamma_scale"],
    )

    log10_ratio_raw = numpyro.sample(
        "log10_ratio_raw", dist.Normal(jnp.zeros(n_pulsars), jnp.ones(n_pulsars))
    )
    log10_ratio = numpyro.deterministic(
        "log10_ratio",
        empirical_specs["ratio_loc"] + log10_ratio_raw * empirical_specs["ratio_scale"],
    )

    log10_σp = numpyro.deterministic("log10_σp", log10_γp + log10_ratio)

    return log10_γp, log10_σp


def sample_pulsar_noise_parameters(prior_specs, n_pulsars):
    """Sample pulsar red noise parameters.

    Dispatches between empirical per-pulsar priors, independent flat priors,
    fixed injected values, and the default hierarchical modeling with log-ratio
    parameterization (see prior_models.get_pulsar_noise_priors for precedence).

    Parameters
    ----------
    prior_specs : dict
        Prior distributions dictionary
    n_pulsars : int
        Number of pulsars

    Returns
    -------
    tuple
        (log10_γp, log10_σp) values
    """
    empirical_specs = prior_specs.get("empirical_specs")
    if empirical_specs is not None:
        return sample_empirical_noise_parameters(empirical_specs, n_pulsars)

    hierarchical_specs = prior_specs.get("hierarchical_specs")

    # Handle log10_gamma_p - flat (Distribution), fixed, or hierarchical
    if isinstance(prior_specs["log10_gamma_p_spec"], tfpd.Distribution):
        # Flat per-pulsar Uniform priors (red_noise_prior = flat)
        log10_γp = sample_reparameterized_parameters(
            prior_specs["log10_gamma_p_spec"], "log10_γp", n_pulsars
        )
    elif prior_specs["log10_gamma_p_spec"] is not None:
        # Fixed value (from injections)
        log10_γp = numpyro.deterministic("log10_γp", prior_specs["log10_gamma_p_spec"])
    else:
        # Always use hierarchical modeling
        log10_γp = sample_hierarchical_gamma_parameters(hierarchical_specs, n_pulsars)

    # Handle log10_sigma_p - flat (Distribution), fixed, or log-ratio
    if isinstance(prior_specs["log10_sigma_p_spec"], tfpd.Distribution):
        # Flat per-pulsar Uniform priors (red_noise_prior = flat)
        log10_σp = sample_reparameterized_parameters(
            prior_specs["log10_sigma_p_spec"], "log10_σp", n_pulsars
        )
    elif prior_specs["log10_sigma_p_spec"] is not None:
        # Fixed value (from injections)
        log10_σp = numpyro.deterministic("log10_σp", prior_specs["log10_sigma_p_spec"])
    else:
        # Always use log-ratio parameterization
        log10_σp = sample_log_ratio_parameters(hierarchical_specs, log10_γp, n_pulsars)

    return log10_γp, log10_σp


def sample_measurement_noise_parameters(prior_specs, n_pulsars):
    """Sample measurement noise parameters from their priors.

    Parameters
    ----------
    prior_specs : dict
        Prior distributions dictionary
    n_pulsars : int
        Number of pulsars

    Returns
    -------
    tuple
        (efac, equad) values
    """
    # Handle EFAC: either fixed value or reparameterized sampling
    if isinstance(prior_specs["efac_spec"], tfpd.Distribution):
        # Use reparameterization for efficient NUTS sampling
        low = prior_specs["efac_spec"].low
        high = prior_specs["efac_spec"].high
        mean = (low + high) / 2.0
        std = (high - low) / 6.0  # 3-sigma rule

        efac_standardized = numpyro.sample(
            "efac_standardized", dist.Normal(jnp.zeros(n_pulsars), jnp.ones(n_pulsars))
        )
        efac = numpyro.deterministic("efac", mean + efac_standardized * std)
    else:
        # Fixed value
        efac = numpyro.deterministic("efac", prior_specs["efac_spec"])

    # Handle EQUAD: either fixed value or reparameterized sampling
    if isinstance(prior_specs["equad_spec"], dict) and prior_specs["equad_spec"].get(
        "use_log10", False
    ):
        # log10(EQUAD) parameterization with reparameterization
        log10_equad_spec = prior_specs["equad_spec"]["log10_equad_spec"]
        low = log10_equad_spec.low
        high = log10_equad_spec.high
        mean = (low + high) / 2.0
        std = (high - low) / 6.0  # 3-sigma rule

        log10_equad_prime = numpyro.sample(
            "log10_equad_prime", dist.Normal(jnp.zeros(n_pulsars), jnp.ones(n_pulsars))
        )
        log10_equad = numpyro.deterministic(
            "log10_equad", mean + log10_equad_prime * std
        )
        equad = numpyro.deterministic("equad", 10.0**log10_equad)
    elif isinstance(prior_specs["equad_spec"], tfpd.Distribution):
        # Regular distribution with reparameterization
        low = prior_specs["equad_spec"].low
        high = prior_specs["equad_spec"].high
        mean = (low + high) / 2.0
        std = (high - low) / 6.0  # 3-sigma rule

        equad_standardized = numpyro.sample(
            "equad_standardized", dist.Normal(jnp.zeros(n_pulsars), jnp.ones(n_pulsars))
        )
        equad = numpyro.deterministic("equad", mean + equad_standardized * std)
    else:
        # Fixed value
        equad = numpyro.deterministic("equad", prior_specs["equad_spec"])

    return efac, equad


def sample_cw_parameters(prior_specs):
    """Sample continuous wave source parameters from their priors.

    Parameters
    ----------
    prior_specs : dict
        Prior distributions dictionary containing CW parameter specs.

    Returns
    -------
    tuple
        (log10_h0, alpha_gw, delta_gw, log10_f_gw, cos_iota, psi, Phi0)
    """
    cw_specs = prior_specs["cw_specs"]

    # Each CW source scalar is reparameterized (or fixed) identically; drive them
    # from the shared CW_SCALAR_PARAMS spec. Order matches the documented return
    # tuple (log10_h0, alpha_gw, delta_gw, log10_f_gw, cos_iota, psi, Phi0) and
    # the sample-site PRNG order, so seeded draws are unchanged.
    return tuple(
        _sample_cw_scalar_numpyro(cw_specs, *spec) for spec in CW_SCALAR_PARAMS
    )


def sample_chi_parameters(prior_specs, n_pulsars):
    """Sample per-pulsar phase parameters for phase-reparameterized pulsar term.

    When phase parameterization is active, samples chi^(n) ~ Uniform(0, 2pi)
    for each pulsar using Normal(0,1) -> affine reparameterization.
    When inactive, returns zeros (no pulsar term phase contribution).

    Parameters
    ----------
    prior_specs : dict
        Prior distributions dictionary containing CW parameter specs.
    n_pulsars : int
        Number of pulsars.

    Returns
    -------
    jax.Array
        Per-pulsar phase parameters, shape (n_pulsars,).
    """
    cw_specs = prior_specs.get("cw_specs", {})

    if cw_specs.get("chi_transform_params") is not None:
        tp = cw_specs["chi_transform_params"]
        chi_prime = numpyro.sample(
            "chi_prime",
            dist.Normal(jnp.zeros(n_pulsars), jnp.ones(n_pulsars)),
        )
        chi = numpyro.deterministic("chi", tp["mean"] + chi_prime * tp["std"])
    else:
        chi = numpyro.deterministic("chi", jnp.zeros(n_pulsars))

    return chi


def count_free_parameters(prior_specs, n_pulsars):
    """Count the total number of free (non-fixed) parameters for NUTS sampling.

    Parameters
    ----------
    prior_specs : dict
        Prior distributions dictionary
    n_pulsars : int
        Number of pulsars

    Returns
    -------
    int
        Total number of free parameters
    """
    count = 0

    # CW parameters (if CW mode)
    cw_specs = prior_specs.get("cw_specs")
    if cw_specs is not None:
        # Count each CW parameter that has transform_params (i.e., is sampled)
        for name, _, _ in CW_SCALAR_PARAMS:
            if cw_specs.get(f"{name}_transform_params") is not None:
                count += 1
        # Per-pulsar phase parameters (phase reparameterization)
        if cw_specs.get("chi_transform_params") is not None:
            count += n_pulsars
    elif prior_specs.get("gw_parameterization") == "ridge":
        # Ridge mode: two free GW coordinates (log10_pivot_psd_prime,
        # log10_gamma_a_prime); log10_ha is derived deterministically.
        count += 2
    else:
        # GWB parameters
        # GW amplitude parameter - free if reparameterization is used
        if prior_specs["log10_ha_transform_params"] is not None:
            count += 1

        # GW spectral index parameter - free if it's a distribution (not fixed)
        if isinstance(prior_specs["log10_gamma_a_spec"], tfpd.Distribution):
            count += 1

    # Correlation-path coordinate (free only when sampled)
    if prior_specs.get("orf_path") == "sampled":
        count += 1

    # Pulsar red noise parameters
    if prior_specs.get("empirical_specs") is not None:
        # Empirical per-pulsar priors: gamma_raw + ratio_raw, no hyperparameters
        count += 2 * n_pulsars
    else:
        # Count gamma_p parameters
        if isinstance(prior_specs["log10_gamma_p_spec"], tfpd.Distribution):
            count += n_pulsars  # Flat priors: one per pulsar
        elif prior_specs["log10_gamma_p_spec"] is None:
            # Hierarchical modeling: 2 hyperparameters + n_pulsars individual parameters
            count += 2  # log10_gamma_p_mean and log10_gamma_p_std
            count += n_pulsars  # Individual pulsar gamma parameters
        # If fixed, no parameters to count

        # Count sigma_p parameters
        if isinstance(prior_specs["log10_sigma_p_spec"], tfpd.Distribution):
            count += n_pulsars  # Flat priors: one per pulsar
        elif prior_specs["log10_sigma_p_spec"] is None:
            # Log-ratio parameterization: 2 hyperparameters + n_pulsars ratio parameters
            count += 2  # log10_ratio_mean and log10_ratio_std
            count += n_pulsars  # Individual pulsar ratio parameters (σp derived deterministically)
        # If fixed, no parameters to count

    # Measurement noise parameters
    if isinstance(prior_specs["efac_spec"], tfpd.Distribution):
        count += n_pulsars  # One per pulsar

    # Handle EQUAD - can be either regular distribution or log10 parameterization
    if isinstance(prior_specs["equad_spec"], dict) and prior_specs["equad_spec"].get(
        "use_log10", False
    ):
        count += n_pulsars  # log10(EQUAD) parameters
    elif isinstance(prior_specs["equad_spec"], tfpd.Distribution):
        count += n_pulsars  # Regular EQUAD parameters

    return count


def build_jaxns_cw_prior_model(prior_specs, n_pulsars):
    """Build a jaxns-compatible prior model generator for CW inference.

    Creates a generator function that yields jaxns Prior objects using native
    TFP distributions. Unlike the NUTS path, no Normal(0,1) reparameterization
    is applied — jaxns samples the prior directly via unit hypercube mapping.

    Parameters
    ----------
    prior_specs : dict
        Prior specifications dictionary (same format as used by NUTS path).
    n_pulsars : int
        Number of pulsars.

    Returns
    -------
    callable
        Generator function compatible with jaxns.Model.
    """
    from jaxns.framework.prior import Prior

    if prior_specs.get("empirical_specs") is not None:
        raise NotImplementedError(
            "Empirical per-pulsar red noise priors are not supported in the "
            "jaxns nested-sampling path; use NUTS (sampler = numpyro)."
        )

    cw_specs = prior_specs["cw_specs"]
    hierarchical_specs = prior_specs.get("hierarchical_specs")

    def prior_model():
        # --- CW source parameters (7 scalars) ---
        # Driven from the shared CW_SCALAR_PARAMS spec. jaxns samples each prior
        # directly as Uniform(min, max) (no Normal(0,1) reparameterization); the
        # sin_delta_gw -> delta_gw transform is applied to the sampled value.
        # Prior names match the NUTS path exactly so results parsing is unaffected.
        cw_vals = {}
        for name, fixed_spec_key, derived in CW_SCALAR_PARAMS:
            tp = cw_specs[f"{name}_transform_params"]
            out_name = derived[0] if derived is not None else name
            if tp is not None:
                value = yield Prior(
                    tfpd.Uniform(low=tp["min"], high=tp["max"]),
                    name=name,
                )
                if derived is not None:
                    value = derived[1](value)
            else:
                value = jnp.asarray(cw_specs[fixed_spec_key])
            cw_vals[out_name] = value

        log10_h0 = cw_vals["log10_h0"]
        alpha_gw = cw_vals["alpha_gw"]
        delta_gw = cw_vals["delta_gw"]
        log10_f_gw = cw_vals["log10_f_gw"]
        cos_iota = cw_vals["cos_iota"]
        psi = cw_vals["psi"]
        Phi0 = cw_vals["Phi0"]

        # --- Per-pulsar chi parameters ---
        if cw_specs.get("chi_transform_params") is not None:
            tp = cw_specs["chi_transform_params"]
            chi = yield Prior(
                tfpd.Uniform(
                    low=jnp.full(n_pulsars, tp["min"]),
                    high=jnp.full(n_pulsars, tp["max"]),
                ),
                name="chi",
            )
        else:
            chi = jnp.zeros(n_pulsars)

        # --- Pulsar noise parameters ---

        # log10_gamma_p: hierarchical or fixed
        if isinstance(prior_specs["log10_gamma_p_spec"], tfpd.Distribution):
            # Fallback: direct uniform prior
            spec = prior_specs["log10_gamma_p_spec"]
            log10_gamma_p = yield Prior(
                tfpd.Uniform(
                    low=jnp.full(n_pulsars, spec.low),
                    high=jnp.full(n_pulsars, spec.high),
                ),
                name="log10_gamma_p",
            )
        elif prior_specs["log10_gamma_p_spec"] is not None:
            # Fixed value (from injections)
            log10_gamma_p = jnp.asarray(prior_specs["log10_gamma_p_spec"])
        else:
            # Hierarchical modeling
            gp_mean_spec = hierarchical_specs["log10_gamma_p_mean_spec"]
            log10_gamma_p_mean = yield Prior(
                tfpd.Uniform(low=gp_mean_spec.low, high=gp_mean_spec.high),
                name="log10_gamma_p_mean",
            )
            gp_std_spec = hierarchical_specs["log10_gamma_p_std_spec"]
            log10_gamma_p_std = yield Prior(
                tfpd.Uniform(low=gp_std_spec.low, high=gp_std_spec.high),
                name="log10_gamma_p_std",
            )
            log10_gamma_p = yield Prior(
                tfpd.Normal(
                    loc=jnp.full(n_pulsars, log10_gamma_p_mean),
                    scale=jnp.full(n_pulsars, log10_gamma_p_std),
                ),
                name="log10_gamma_p",
            )

        # log10_sigma_p: log-ratio or fixed
        if isinstance(prior_specs["log10_sigma_p_spec"], tfpd.Distribution):
            # Fallback: direct uniform prior
            spec = prior_specs["log10_sigma_p_spec"]
            log10_sigma_p = yield Prior(
                tfpd.Uniform(
                    low=jnp.full(n_pulsars, spec.low),
                    high=jnp.full(n_pulsars, spec.high),
                ),
                name="log10_sigma_p",
            )
        elif prior_specs["log10_sigma_p_spec"] is not None:
            # Fixed value (from injections)
            log10_sigma_p = jnp.asarray(prior_specs["log10_sigma_p_spec"])
        else:
            # Log-ratio parameterization
            ratio_mean_spec = hierarchical_specs["log10_ratio_mean_spec"]
            log10_ratio_mean = yield Prior(
                tfpd.Uniform(low=ratio_mean_spec.low, high=ratio_mean_spec.high),
                name="log10_ratio_mean",
            )
            ratio_std_spec = hierarchical_specs["log10_ratio_std_spec"]
            log10_ratio_std = yield Prior(
                tfpd.Uniform(low=ratio_std_spec.low, high=ratio_std_spec.high),
                name="log10_ratio_std",
            )
            log10_ratio = yield Prior(
                tfpd.Normal(
                    loc=jnp.full(n_pulsars, log10_ratio_mean),
                    scale=jnp.full(n_pulsars, log10_ratio_std),
                ),
                name="log10_ratio",
            )
            log10_sigma_p = log10_gamma_p + log10_ratio

        # --- Measurement noise parameters ---

        # EFAC
        if isinstance(prior_specs["efac_spec"], tfpd.Distribution):
            spec = prior_specs["efac_spec"]
            efac = yield Prior(
                tfpd.Uniform(
                    low=jnp.full(n_pulsars, spec.low),
                    high=jnp.full(n_pulsars, spec.high),
                ),
                name="efac",
            )
        else:
            efac = jnp.asarray(prior_specs["efac_spec"])

        # EQUAD
        if isinstance(prior_specs["equad_spec"], dict) and prior_specs[
            "equad_spec"
        ].get("use_log10", False):
            log10_equad_spec = prior_specs["equad_spec"]["log10_equad_spec"]
            log10_equad = yield Prior(
                tfpd.Uniform(
                    low=jnp.full(n_pulsars, log10_equad_spec.low),
                    high=jnp.full(n_pulsars, log10_equad_spec.high),
                ),
                name="log10_equad",
            )
            equad = 10.0**log10_equad
        elif isinstance(prior_specs["equad_spec"], tfpd.Distribution):
            spec = prior_specs["equad_spec"]
            equad = yield Prior(
                tfpd.Uniform(
                    low=jnp.full(n_pulsars, spec.low),
                    high=jnp.full(n_pulsars, spec.high),
                ),
                name="equad",
            )
        else:
            equad = jnp.asarray(prior_specs["equad_spec"])

        return (
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

    return prior_model

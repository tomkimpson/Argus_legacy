"""Module which holds all functions which are related to the properties of the gravitational wave."""

import numpy as np
import jax
import jax.numpy as jnp


def pairwise_angular_separation(ra_rad, dec_rad):
    """Compute the pairwise angular separations for a set of celestial coordinates in radians.

    This function takes arrays of right ascension (RA) and declination (Dec), both in radians,
    and returns an NxN matrix of angular separations, where N is the length of the input arrays.
    Each entry (i, j) in the output is the angular separation between the coordinate pair
    (ra_rad[i], dec_rad[i]) and (ra_rad[j], dec_rad[j]).

    Parameters
    ----------
    ra_rad : numpy.ndarray
        1D array of right ascensions in radians, of length N.
    dec_rad : numpy.ndarray
        1D array of declinations in radians, of length N.

    Returns
    -------
    sep_rad : numpy.ndarray
        NxN matrix (2D array) of pairwise angular separations in radians.

    Notes
    -----
    The spherical distance formula used is:

        cos(theta) = sin(dec1) * sin(dec2)
                    + cos(dec1) * cos(dec2) * cos(ra1 - ra2)

    where (ra1, dec1) and (ra2, dec2) are coordinate pairs in radians.

    """
    # Reshape for broadcasting
    ra1 = ra_rad[:, None]
    ra2 = ra_rad[None, :]
    dec1 = dec_rad[:, None]
    dec2 = dec_rad[None, :]

    # Spherical distance formula:
    #   cos(theta) = sin(dec1)*sin(dec2) + cos(dec1)*cos(dec2)*cos(ra1 - ra2)
    cos_sep = np.sin(dec1) * np.sin(dec2) + np.cos(dec1) * np.cos(dec2) * np.cos(
        ra1 - ra2
    )

    # Clip values to avoid floating-point errors outside [-1, 1] when taking arccos
    cos_sep = np.clip(cos_sep, -1.0, 1.0)

    # Compute separation in radians
    sep_rad = np.arccos(cos_sep)

    # A pulsar's separation from itself is exactly zero by definition. Rounding in
    # sin^2 + cos^2 can leave the diagonal cosine just below 1.0, so arccos returns a
    # tiny (~1e-5 rad) angle instead of 0; that then slips past the ``np.isclose``
    # self-pair test in ``hellings_downs`` and the diagonal is set to ~0.5 (the HD
    # curve) rather than the 1.0 auto-correlation. Pinning the diagonal to 0 fixes it
    # robustly regardless of array size (only surfaced with the full NG15 array).
    np.fill_diagonal(sep_rad, 0.0)

    return sep_rad


def hellings_downs(θ):
    """Compute the Hellings–Downs function for an angle θ (in radians).

    Parameters
    ----------
    θ : np.ndarray or float
        Angular separation between pulsars in radians

    Returns
    -------
    np.ndarray or float
        Hellings-Downs correlation values
    """
    # Handle the vector case first
    if isinstance(θ, np.ndarray):
        mask = np.isclose(θ, 0.0)
        x = np.zeros_like(θ)
        # Only compute (1-cos(θ))/2 for non-zero angles
        x[~mask] = (1 - np.cos(θ[~mask])) / 2.0

        result = np.zeros_like(θ)
        result[mask] = 1.0
        # Only compute HD function for non-zero angles
        result[~mask] = (3 / 2) * x[~mask] * np.log(x[~mask]) - x[~mask] / 4 + 0.5

        return result
    else:
        # Handle scalar input
        # Special case for θ = 0 (autocorrelation): x = 0 leads to 0 * log(0) = nan
        # The correct limit as θ → 0 is HD(0) = 1
        if np.isclose(θ, 0.0):
            return 1.0
        x = (1 - np.cos(θ)) / 2.0
        return (3 / 2) * x * np.log(x) - x / 4 + 0.5


def correlation_path(hd_correlation_matrix, epsilon):
    """Interpolate between uncorrelated (CURN) and Hellings-Downs correlation.

    Embeds the two competing SGWB correlation hypotheses in a single continuous
    family::

        C(ε) = (1 - ε) · I + ε · C_HD

    so that ``ε = 0`` is CURN (identity overlap reduction function) exactly and
    ``ε = 1`` is Hellings-Downs exactly, with every other model input shared. A
    Bayes factor between the two hypotheses can then be obtained either from the
    posterior density of ``ε`` at its endpoints (generalised Savage-Dickey) or by
    integrating ``∂ log L / ∂ε`` along a ladder of fixed ``ε`` (path sampling).

    The Hellings-Downs matrix has a unit diagonal (a pulsar's correlation with
    itself is 1), so the interpolation leaves the diagonal at 1 for every ``ε``:
    only the cross-correlations are scaled. ``ε`` therefore moves power between
    the correlated and uncorrelated hypotheses without changing the GW auto-power.

    Parameters
    ----------
    hd_correlation_matrix : jax.Array
        Hellings-Downs correlation matrix, shape (Npsr, Npsr), unit diagonal.
    epsilon : float or jax.Array
        Interpolation coordinate. 0 gives the identity, 1 gives the input matrix.
        Traceable, so it may be a sampled parameter.

    Returns
    -------
    jax.Array
        The interpolated correlation matrix, shape (Npsr, Npsr).
    """
    identity = jnp.eye(
        hd_correlation_matrix.shape[0], dtype=hd_correlation_matrix.dtype
    )
    return (1.0 - epsilon) * identity + epsilon * hd_correlation_matrix


# =============================================================================
# Continuous Wave (CW) signal model functions
# All implemented in JAX for autodiff/JIT compatibility
# =============================================================================


def gw_propagation_direction(alpha_gw, delta_gw):
    """Compute the GW propagation direction unit vector.

    Parameters
    ----------
    alpha_gw : float
        Right ascension of the GW source in radians.
    delta_gw : float
        Declination of the GW source in radians.

    Returns
    -------
    jax.Array
        Unit vector n_hat of shape (3,) pointing in the GW propagation direction.
    """
    return -jnp.array(
        [
            jnp.cos(alpha_gw) * jnp.cos(delta_gw),
            jnp.sin(alpha_gw) * jnp.cos(delta_gw),
            jnp.sin(delta_gw),
        ]
    )


def pulsar_direction(ra, dec):
    """Compute the pulsar direction unit vector.

    Parameters
    ----------
    ra : float
        Right ascension of the pulsar in radians.
    dec : float
        Declination of the pulsar in radians.

    Returns
    -------
    jax.Array
        Unit vector q_hat of shape (3,).
    """
    return jnp.array(
        [
            jnp.cos(ra) * jnp.cos(dec),
            jnp.sin(ra) * jnp.cos(dec),
            jnp.sin(dec),
        ]
    )


def polarization_vectors(alpha_gw, delta_gw):
    """Compute the orthonormal polarization basis vectors m_hat and l_hat.

    Parameters
    ----------
    alpha_gw : float
        Right ascension of the GW source in radians.
    delta_gw : float
        Declination of the GW source in radians.

    Returns
    -------
    tuple[jax.Array, jax.Array]
        (m_hat, l_hat), each of shape (3,).
    """
    m_hat = jnp.array(
        [
            jnp.sin(alpha_gw),
            -jnp.cos(alpha_gw),
            0.0,
        ]
    )
    l_hat = jnp.array(
        [
            -jnp.cos(alpha_gw) * jnp.sin(delta_gw),
            -jnp.sin(alpha_gw) * jnp.sin(delta_gw),
            jnp.cos(delta_gw),
        ]
    )
    return m_hat, l_hat


def polarization_tensors(alpha_gw, delta_gw, psi):
    """Compute the plus and cross polarization tensors rotated by angle psi.

    Parameters
    ----------
    alpha_gw : float
        Right ascension of the GW source in radians.
    delta_gw : float
        Declination of the GW source in radians.
    psi : float
        Polarization angle in radians.

    Returns
    -------
    tuple[jax.Array, jax.Array]
        (e_plus, e_cross), each of shape (3, 3).
    """
    m_hat, l_hat = polarization_vectors(alpha_gw, delta_gw)

    # Outer products
    mm = jnp.outer(m_hat, m_hat)
    ll = jnp.outer(l_hat, l_hat)
    ml = jnp.outer(m_hat, l_hat)
    lm = jnp.outer(l_hat, m_hat)

    cos2psi = jnp.cos(2.0 * psi)
    sin2psi = jnp.sin(2.0 * psi)

    e_plus = (mm - ll) * cos2psi + (ml + lm) * sin2psi
    e_cross = (mm - ll) * sin2psi - (ml + lm) * cos2psi

    return e_plus, e_cross


def antenna_pattern_single(pulsar_ra, pulsar_dec, alpha_gw, delta_gw, psi):
    """Compute antenna pattern functions F_plus and F_cross for a single pulsar.

    Parameters
    ----------
    pulsar_ra : float
        Right ascension of the pulsar in radians.
    pulsar_dec : float
        Declination of the pulsar in radians.
    alpha_gw : float
        Right ascension of the GW source in radians.
    delta_gw : float
        Declination of the GW source in radians.
    psi : float
        Polarization angle in radians.

    Returns
    -------
    tuple[float, float]
        (F_plus, F_cross) antenna pattern values.
    """
    n_hat = gw_propagation_direction(alpha_gw, delta_gw)
    q_hat = pulsar_direction(pulsar_ra, pulsar_dec)
    e_plus, e_cross = polarization_tensors(alpha_gw, delta_gw, psi)

    # Denominator with numerical guard: clip to avoid divergence
    denominator = 1.0 + jnp.dot(n_hat, q_hat)
    denominator = jnp.maximum(denominator, 1e-10)

    # Quadratic form: q^i q^j e_ij = q^T @ e @ q
    numerator_plus = q_hat @ e_plus @ q_hat
    numerator_cross = q_hat @ e_cross @ q_hat

    F_plus = numerator_plus / (2.0 * denominator)
    F_cross = numerator_cross / (2.0 * denominator)

    return F_plus, F_cross


def compute_antenna_patterns(
    pulsar_ra_array, pulsar_dec_array, alpha_gw, delta_gw, psi
):
    """Compute antenna pattern functions for all pulsars (vectorized).

    Parameters
    ----------
    pulsar_ra_array : jax.Array
        Right ascensions of pulsars in radians, shape (Npsr,).
    pulsar_dec_array : jax.Array
        Declinations of pulsars in radians, shape (Npsr,).
    alpha_gw : float
        Right ascension of the GW source in radians.
    delta_gw : float
        Declination of the GW source in radians.
    psi : float
        Polarization angle in radians.

    Returns
    -------
    tuple[jax.Array, jax.Array]
        (F_plus, F_cross), each of shape (Npsr,).
    """
    vmapped_fn = jax.vmap(
        lambda ra, dec: antenna_pattern_single(ra, dec, alpha_gw, delta_gw, psi)
    )
    return vmapped_fn(pulsar_ra_array, pulsar_dec_array)


def cw_timing_residual(t, f_gw, h0, cos_iota, Phi0, F_plus, F_cross):
    """Compute the Earth-term-only CW timing residual for a single observation.

    Parameters
    ----------
    t : float
        Observation time in seconds.
    f_gw : float
        Gravitational wave frequency in Hz.
    h0 : float
        Strain amplitude.
    cos_iota : float
        Cosine of the inclination angle.
    Phi0 : float
        Initial GW phase in radians.
    F_plus : float
        Plus antenna pattern function value.
    F_cross : float
        Cross antenna pattern function value.

    Returns
    -------
    float
        GW-induced timing residual in seconds.
    """
    Omega = 2.0 * jnp.pi * f_gw
    phase = Omega * t + Phi0

    # Integrated Earth-term GW responses
    Delta_s_plus = h0 * (1.0 + cos_iota**2) / (2.0 * Omega) * jnp.sin(phase)
    Delta_s_cross = -h0 * cos_iota / Omega * jnp.cos(phase)

    return F_plus * Delta_s_plus + F_cross * Delta_s_cross


def _cw_earth_term(toas, f_gw, h0, cos_iota, Phi0, F_plus, F_cross):
    """Shared CW waveform core: frequency, polarization amplitudes, Earth term.

    Both pulsar-term variants (distance-based and phase-parameterized) reuse the
    same angular frequency, polarization amplitudes, and Earth-term residual;
    they differ only in how the pulsar-term phase is formed.

    Returns
    -------
    tuple
        (Omega, amp_plus, amp_cross, earth_term). ``earth_term`` has the shape
        of ``toas``; the amplitudes and Omega are reused for the pulsar term.
    """
    Omega = 2.0 * jnp.pi * f_gw
    amp_plus = h0 * (1.0 + cos_iota**2) / (2.0 * Omega)
    amp_cross = -h0 * cos_iota / Omega

    phase_e = Omega * toas + Phi0
    earth_term = F_plus * amp_plus * jnp.sin(phase_e) + F_cross * amp_cross * jnp.cos(
        phase_e
    )

    return Omega, amp_plus, amp_cross, earth_term


def compute_cw_signal_single_pulsar(
    toas,
    f_gw,
    h0,
    cos_iota,
    Phi0,
    F_plus,
    F_cross,
    pulsar_distance=0.0,
    geometric_factor=0.0,
):
    """Compute CW timing residuals for all observation times of a single pulsar.

    Computes Earth term, and optionally the pulsar term if pulsar_distance > 0.
    The pulsar term uses the same waveform evaluated at the retarded time
    t_p = t - tau_a, where tau_a = d_a * (1 + n_hat . q_hat) / c.

    Parameters
    ----------
    toas : jax.Array
        Observation times for this pulsar, shape (nobs,).
    f_gw : float
        Gravitational wave frequency in Hz.
    h0 : float
        Strain amplitude.
    cos_iota : float
        Cosine of the inclination angle.
    Phi0 : float
        Initial GW phase in radians.
    F_plus : float
        Plus antenna pattern function value for this pulsar.
    F_cross : float
        Cross antenna pattern function value for this pulsar.
    pulsar_distance : float, optional
        Pulsar distance in seconds (d/c). Default 0.0 (Earth-term only).
    geometric_factor : float, optional
        Geometric delay factor (1 + n_hat . q_hat). Default 0.0.

    Returns
    -------
    jax.Array
        CW timing residuals, shape (nobs,).
    """
    Omega, amp_plus, amp_cross, earth_term = _cw_earth_term(
        toas, f_gw, h0, cos_iota, Phi0, F_plus, F_cross
    )

    # Pulsar term: subtract signal at retarded time t_p = t - tau_a
    # tau_a = pulsar_distance * geometric_factor
    # When pulsar_distance=0, pulsar term contribution is zero (Earth-term only)
    tau_a = pulsar_distance * geometric_factor
    phase_p = Omega * (toas - tau_a) + Phi0
    pulsar_term = F_plus * amp_plus * jnp.sin(phase_p) + F_cross * amp_cross * jnp.cos(
        phase_p
    )

    # Use pulsar_distance as a switch: when 0, no pulsar term subtracted
    has_pulsar_term = jnp.where(pulsar_distance > 0.0, 1.0, 0.0)
    return earth_term - has_pulsar_term * pulsar_term


def compute_cw_signal_single_pulsar_phase(
    toas, f_gw, h0, cos_iota, Phi0, F_plus, F_cross, chi
):
    """Compute CW timing residuals using phase-parameterized pulsar term.

    Instead of computing the pulsar term from physical distance, uses a
    per-pulsar phase parameter chi that absorbs the distance-dependent delay:
    chi = Omega * (1 + n_hat . q_hat) * d mod 2pi  (arXiv 2410.10087).

    This eliminates the highly multimodal likelihood surface caused by
    distance-dependent phase oscillations, enabling efficient NUTS sampling.

    Parameters
    ----------
    toas : jax.Array
        Observation times for this pulsar, shape (nobs,).
    f_gw : float
        Gravitational wave frequency in Hz.
    h0 : float
        Strain amplitude.
    cos_iota : float
        Cosine of the inclination angle.
    Phi0 : float
        Initial GW phase in radians.
    F_plus : float
        Plus antenna pattern function value for this pulsar.
    F_cross : float
        Cross antenna pattern function value for this pulsar.
    chi : float
        Per-pulsar phase parameter in [0, 2pi), replacing the
        distance-based pulsar term delay.

    Returns
    -------
    jax.Array
        CW timing residuals (earth - pulsar), shape (nobs,).
    """
    Omega, amp_plus, amp_cross, earth_term = _cw_earth_term(
        toas, f_gw, h0, cos_iota, Phi0, F_plus, F_cross
    )

    # Pulsar term with phase reparameterization
    phase_p = Omega * toas + Phi0 - chi
    pulsar_term = F_plus * amp_plus * jnp.sin(phase_p) + F_cross * amp_cross * jnp.cos(
        phase_p
    )

    return earth_term - pulsar_term

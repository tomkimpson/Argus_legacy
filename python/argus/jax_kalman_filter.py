"""Module which implements JAX-based Kalman filter algorithm."""

import numpy as np

from argus import gravitational_waves
from argus.model import get_F, get_Q, precompute_R_matrices, precompute_H_matrix
from functools import partial
import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.linalg import block_diag
from typing import Tuple


# Get the centralized logger (will be initialized by workflow)
def get_logger():
    """Get the centralized logger instance."""
    from argus.io_manager import get_argus_logger

    return get_argus_logger()


@partial(jax.jit, static_argnums=(2, 3))
def compute_predicted_state(F_list, x, gw_size, spin_size):
    """Compute the predicted state vector by applying transition matrices to state blocks.

    Args:
        F_list: Tuple of (F_gw, F_spin) transition matrices for GW and spin components
        x: Current state vector containing GW, spin and timing components
        gw_size: Size of gravitational wave state block
        spin_size: Size of spin state block

    Returns
    -------
        jax.Array: Predicted state vector with same structure as input, computed by:
            - Applying F_gw transition to GW states
            - Applying F_spin transition to spin states
            - Keeping timing states unchanged

    Note:
        The state vector x is assumed to have structure [x_gw, x_spin, x_timing]
        where each component has size determined by gw_size and spin_size parameters.
    """
    F_gw, F_spin = F_list
    x_gw = x[:gw_size]
    x_spin = x[gw_size : gw_size + spin_size]
    x_timing = x[gw_size + spin_size :]
    return jnp.vstack([F_gw @ x_gw, F_spin @ x_spin, x_timing])


@partial(jax.jit, static_argnums=(3, 4))
def compute_predicted_covariance(
    P: jax.Array,
    F_list: Tuple[jax.Array, jax.Array],
    Q_list: Tuple[jax.Array, ...],
    gw_size: int,
    spin_size: int,
) -> jax.Array:
    """Compute predicted covariance matrix in one operation.

    Args:
        P: Full covariance matrix
        F_list: Tuple of (F_gw, F_spin) transition matrices
        Q_list: Tuple of (Q_gw, Q_spin, Q_timing) process noise matrices
        gw_size: Size of GW block
        spin_size: Size of spin block

    Returns
    -------
        jax.Array: Combined predicted covariance matrix

    Note:
        Computing the predicted covariance by slicing the matrix into blocks and doing
        individual matrix products is significantly faster than doing the full matrix
        multiplication FPF^T + Q. This is because the block structure allows us to avoid
        many unnecessary multiplications with zero elements.
    """
    F1, F2 = F_list
    Q1, Q2 = Q_list

    # Extract blocks directly from P
    P1 = P[:gw_size, :gw_size]
    P2 = P[gw_size : gw_size + spin_size, gw_size : gw_size + spin_size]
    P3 = P[gw_size + spin_size :, gw_size + spin_size :]
    P4 = P[:gw_size, gw_size : gw_size + spin_size]
    P5 = P[gw_size : gw_size + spin_size, gw_size + spin_size :]
    P6 = P[:gw_size, gw_size + spin_size :]

    # Compute individual blocks
    PF1 = F1 @ P1 @ F1.T + Q1
    PF2 = F2 @ P2 @ F2.T + Q2
    PF4 = F1 @ P4 @ F2.T
    PF5 = F2 @ P5
    PF6 = F1 @ P6

    # Assemble full matrix
    return jnp.block([[PF1, PF4, PF6], [PF4.T, PF2, PF5], [PF6.T, PF5.T, P3]])


def _log_likelihood(y: jax.Array, cov: jax.Array) -> jax.Array:
    """Calculate the log likelihood given innovation and innovation covariance.

    Args:
        y: Innovation term (measurement residual), shape (n,)
        cov: Innovation covariance matrix, shape (n,n)

    Returns
    -------
        float: Log likelihood value
    """
    # Robustness against a near-singular innovation covariance. A global sampler (e.g. nested
    # sampling) explores prior tails where `cov` can go numerically near-singular; then a raw
    # slogdet runs to -inf and, if the innovation misses the near-zero-variance direction, the
    # likelihood spikes to a spurious huge positive value that the sampler locks onto. We
    # symmetrise and add a jitter scaled to the matrix magnitude (negligible when `cov` is
    # well-conditioned, so gradient-guided samplers and the golden likelihood are unchanged),
    # and reject any residually non-positive-definite `cov` rather than return garbage.
    n = cov.shape[0]
    cov = 0.5 * (cov + cov.T)
    jitter = 1e-9 * (jnp.trace(cov) / n)
    cov = cov + jitter * jnp.eye(n)
    sign, logdet = jnp.linalg.slogdet(2.0 * jnp.pi * cov)
    quadratic_term = y.T @ jnp.linalg.solve(cov, y)
    ll = -0.5 * (logdet + quadratic_term)
    return jnp.where(sign > 0, ll, -jnp.inf)


@partial(jax.jit, static_argnums=(4,))
def _predict(
    x: jax.Array, P: jax.Array, F_list: tuple, Q_list: tuple, dim_x: int
) -> tuple[jax.Array, jax.Array]:
    """Predict the next state and covariance.

    Args:
        x: Current state vector
        P: Current covariance matrix
        F_list: Tuple of state transition matrices
        Q_list: Tuple of process noise matrices
        dim_x: Dimension of the state vector

    Returns
    -------
        tuple: (predicted state, predicted covariance)
    """
    xp = compute_predicted_state(F_list, x, dim_x, dim_x)
    Pp = compute_predicted_covariance(P, F_list, Q_list, dim_x, dim_x)
    return xp, Pp


def _update(
    xp: jax.Array,
    Pp: jax.Array,
    H: jax.Array,
    R: jax.Array,
    z: jax.Array,
    mask: jax.Array = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Update the state and covariance with a new observation.

    Args:
        xp: Predicted state vector
        Pp: Predicted covariance matrix
        H: Observation matrix, shape (Npsr, nx)
        R: Observation noise covariance, shape (Npsr, Npsr)
        z: Observation, shape (Npsr,)
        mask: Optional per-pulsar observation mask, shape (Npsr,), with 1.0 for a
            pulsar observed at this epoch and 0.0 for one that is absent. When None
            (the default) every pulsar is treated as observed, which reproduces the
            original all-pulsars-present update bit-for-bit.

    Returns
    -------
        tuple: (updated state, updated covariance, innovation, innovation covariance)

    Note:
        Missing observations are handled by masking rather than by reshaping, so the
        matrix dimensions stay static (required for `jax.jit`/`lax.scan`). Each row of
        H touches only its own pulsar's states and all cross-pulsar coupling lives in
        Pp, so zeroing the absent pulsars' rows of H (and the corresponding rows/cols
        of R, replaced by unit diagonal to keep S invertible) yields exactly the
        Kalman update conditioned on only the present pulsars — the absent pulsars'
        states are propagated by _predict and simply not updated this epoch.
    """
    if mask is None:
        H_eff = H
        R_eff = R
        y = z[:, None] - H_eff @ xp
    else:
        # M = diag(mask). H_eff = M H zeroes absent rows; R_eff = M R M + (I - M) keeps
        # the present block untouched and puts a unit diagonal on absent slots so S is
        # invertible and block-decoupled. Zeroing the absent innovations keeps their
        # contribution to the state/covariance update and the likelihood at zero.
        H_eff = mask[:, None] * H
        R_eff = (mask[:, None] * mask[None, :]) * R + jnp.diag(1.0 - mask)
        y = mask[:, None] * (z[:, None] - H_eff @ xp)

    S = H_eff @ Pp @ H_eff.T + R_eff
    # Use solve instead of inv for better performance and numerical stability
    # K = Pp @ H.T @ Sinv becomes K = Pp @ H.T @ solve(S, I)
    K = Pp @ H_eff.T @ jnp.linalg.solve(S, jnp.eye(S.shape[0]))
    x = xp + K @ y

    # Following FilterPy https://github.com/rlabbe/filterpy/blob/master/filterpy/kalman/EKF.py by using
    # Joseph form for numerically stable update of the covariance matrix
    # P = (I-KH)P(I-KH)' + KRK' which is more numerically stable
    # and works for non-optimal K vs the equation
    # P = (I-KH)P usually seen in the literature.
    I_KH = jnp.eye(len(xp)) - K @ H_eff
    P = I_KH @ Pp @ I_KH.T + K @ R_eff @ K.T

    # Optional: enforce symmetry for numerical stability
    # P = 0.5 * (P + P.T)

    return x, P, y, S


def _compute_sigma_matrix(h2, γa, Γ):
    return (h2 / 12) * γa * Γ


def _effective_orf(hellings_downs_matrix, orf_epsilon):
    """Resolve the overlap reduction function actually used by the filter.

    With the correlation path switched off (``orf_epsilon is None``) the static
    Hellings-Downs matrix is returned unchanged, so the default likelihood path is
    untouched. Otherwise the matrix is interpolated towards the identity, which is
    what lets a single model span the HD and CURN hypotheses (see
    ``gravitational_waves.correlation_path``).
    """
    if orf_epsilon is None:
        return hellings_downs_matrix
    return gravitational_waves.correlation_path(hellings_downs_matrix, orf_epsilon)


def _initialize_kalman_filter(nx, Npsr, P_eps, σa2, γa, σp2, γp):
    """Initialize the state vector (x0) and covariance matrix (P0).

    This function sets up the initial conditions for the Kalman filter based on
    the assumed structure of the state vector and prior knowledge about the
    system noise properties (GW, spin noise, measurement noise).

    The state vector `x` is assumed to be structured block-wise:
    `x = [GW states (2*Npsr), Spin states (2*Npsr), Epsilon states (approx. 10*Npsr)]`

    Args:
        nx: Total dimension of the state vector.
        Npsr: Number of pulsars in the array.
        P_eps: Initial covariance matrix for the epsilon (measurement white noise)
               states block. Shape depends on epsilon state definition, e.g., (Npsr, Npsr).
               Represents initial uncertainty associated with terms like EFAC/EQUAD.
        h2: Squared characteristic strain amplitude (h_c^2) of the expected GW background.
            Used to calculate the stationary variance of the GW 'a' state component.
        γa: Damping constant (1 / correlation time) for the Ornstein-Uhlenbeck (OU)
            process modeling the GW 'a' state component.

    Returns
    -------
        tuple[jax.Array, jax.Array]: A tuple containing:
            - x0: Initial state vector, shape (nx, 1). Initialized to zeros, assuming
                  states represent perturbations around a known mean (or zero).
            - P0: Initial state covariance matrix, shape (nx, nx). Constructed by
                  combining covariance blocks for GW, Spin, and Epsilon states.
    """
    # Initialize the states
    x0 = jnp.zeros(
        (nx, 1)
    )  # Initialize as column vector. jnp.zeros(nx) # Initial state vector. δφ=0,δf=0, etc. As all the states are effecitvely perturbations, this is a reasonable guess.

    # Initialize the covariance matrices

    ## 1. The GW block "r/a"
    P_GW = jnp.zeros((Npsr * 2, Npsr * 2))

    # 1.1 Set diagonal variances for 'r' states (indices 0, 2, 4, ...)
    # Set P[2n, 2n] = 1e-40 (very small initial variance)
    r_indices = jnp.arange(0, Npsr * 2, 2)
    P_GW = P_GW.at[r_indices, r_indices].set(1e-40)

    # 1.2 Set the P_aa block (indices 1, 3, 5, ...)
    # Sets P[2n+1, 2m+1] = P_aa_init[n, m]
    P_aa_init = σa2 / (2.0 * γa)
    P_GW = P_GW.at[1::2, 1::2].set(P_aa_init)

    ## 2. The spin block "phi / f "
    P_spin = jnp.zeros((Npsr * 2, Npsr * 2))

    # 2.1 Set diagonal variances for 'phi' states (indices 0, 2, 4, ...)
    # Set P[2n, 2n] = 1e-40
    phi_indices = jnp.arange(0, Npsr * 2, 2)
    P_spin = P_spin.at[phi_indices, phi_indices].set(1e-40)

    # 2.2 Set diagonal variances for 'f' states (indices 1, 3, 5, ...)
    # Eq: Var(f) = sigma2_spin[n] / (2 * gamma_spin[n])
    # This is element-wise calculation resulting in a vector of length Npsr
    spin_variance_values = σp2 / (2.0 * γp)
    f_indices = jnp.arange(1, Npsr * 2, 2)
    P_spin = P_spin.at[f_indices, f_indices].set(spin_variance_values)

    P0 = block_diag(P_GW, P_spin, P_eps)

    return x0, P0


def _initialize_dynamic_kalman_filter(Npsr, σa2, γa, σp2, γp):
    """Initialize the dynamic-only state (x0) and covariance (P0) for the marginalized filter.

    Identical GW/spin blocks as `_initialize_kalman_filter`, but WITHOUT the trailing
    epsilon (timing-model) block. The epsilon parameters are marginalized analytically
    (Rao-Blackwellization) rather than carried as state, so the propagated state is the
    dynamic block only: `x = [GW states (2*Npsr), spin states (2*Npsr)]`, dim `4*Npsr`.

    Returns
    -------
        tuple[jax.Array, jax.Array]: (x0_dyn shape (4*Npsr, 1), P0_dyn shape (4*Npsr, 4*Npsr)).
    """
    x0 = jnp.zeros((4 * Npsr, 1))

    # GW block "r/a": position 'r' states get a tiny prior variance; 'a' states the OU stationary block.
    P_GW = jnp.zeros((Npsr * 2, Npsr * 2))
    r_indices = jnp.arange(0, Npsr * 2, 2)
    P_GW = P_GW.at[r_indices, r_indices].set(1e-40)
    P_aa_init = σa2 / (2.0 * γa)
    P_GW = P_GW.at[1::2, 1::2].set(P_aa_init)

    # Spin block "phi/f": 'phi' states tiny prior variance; 'f' states the OU stationary variance.
    P_spin = jnp.zeros((Npsr * 2, Npsr * 2))
    phi_indices = jnp.arange(0, Npsr * 2, 2)
    P_spin = P_spin.at[phi_indices, phi_indices].set(1e-40)
    spin_variance_values = σp2 / (2.0 * γp)
    f_indices = jnp.arange(1, Npsr * 2, 2)
    P_spin = P_spin.at[f_indices, f_indices].set(spin_variance_values)

    P0 = block_diag(P_GW, P_spin)
    return x0, P0


@partial(jax.jit, static_argnums=(3,))
def _predict_dynamic_cov(P, F_list, Q_list, gw_size):
    """Predict the dynamic-only (2-block) covariance: [[F1 P1 F1'+Q1, F1 P4 F2'], [., F2 P2 F2'+Q2]].

    Two-block version of `compute_predicted_covariance` (no timing block). Avoids a
    zero-width `jnp.block` element that would arise from calling the 3-block helper on a
    state with an empty epsilon slice.
    """
    F1, F2 = F_list
    Q1, Q2 = Q_list
    P1 = P[:gw_size, :gw_size]
    P2 = P[gw_size:, gw_size:]
    P4 = P[:gw_size, gw_size:]
    PF1 = F1 @ P1 @ F1.T + Q1
    PF2 = F2 @ P2 @ F2.T + Q2
    PF4 = F1 @ P4 @ F2.T
    return jnp.block([[PF1, PF4], [PF4.T, PF2]])


@partial(jax.jit, static_argnums=(3,))
def _predict_xi(Xi, F_gw, F_spin, gw_size):
    """Propagate the state->epsilon sensitivity Ξ = ∂x/∂β through the dynamics.

    Ξ has the same block structure as the dynamic state: `[GW rows; spin rows]`,
    shape (4*Npsr, M_sum). Since β does not drive the dynamics, Ξ_pred = F Ξ_filt.
    """
    return jnp.vstack([F_gw @ Xi[:gw_size], F_spin @ Xi[gw_size:]])


def _update_marginal(xp, Pp, Xi_pred, H_dyn, H_eps, R, z):
    """Joseph update of the dynamic state plus epsilon sensitivity/information accumulation.

    Runs the standard Joseph update on the dynamic state `x` (dim 4*Npsr) treating the
    static timing parameters β symbolically, and returns the per-epoch contributions to
    the accumulated epsilon information used to marginalize β at the end.

    Args:
        xp, Pp: predicted dynamic state (4*Npsr, 1) and covariance (4*Npsr, 4*Npsr).
        Xi_pred: predicted sensitivity Ξ_pred = ∂x_pred/∂β, shape (4*Npsr, M_sum).
        H_dyn: dynamic columns of H for this epoch, shape (Npsr, 4*Npsr).
        H_eps: epsilon (timing design) columns of H, shape (Npsr, M_sum).
        R: measurement noise covariance, shape (Npsr, Npsr).
        z: observation, shape (Npsr,).

    Returns
    -------
        tuple: (x, P, Xi, dA, db, dc, dL) where
            x, P, Xi   : updated dynamic state, covariance, sensitivity;
            dA = Ψ' S⁻¹ Ψ (M_sum, M_sum), db = Ψ' S⁻¹ ỹ (M_sum, 1), dc = ỹ' S⁻¹ ỹ (scalar),
            dL = logdet(2π S) (or +inf if S is not positive definite, forcing logL -> -inf).
    """
    # β=0 innovation and its sensitivity to β. S is β-independent (it never sees a mean).
    y0 = z[:, None] - H_dyn @ xp
    S = H_dyn @ Pp @ H_dyn.T + R
    Psi = H_eps + H_dyn @ Xi_pred

    # Same symmetrise + magnitude-scaled jitter + PD guard as `_log_likelihood`, applied once
    # so the log-det stream, the gain and the epsilon quadratics all use a single stable S.
    n = S.shape[0]
    S = 0.5 * (S + S.T)
    jitter = 1e-9 * (jnp.trace(S) / n)
    S = S + jitter * jnp.eye(n)
    sign, logdet = jnp.linalg.slogdet(2.0 * jnp.pi * S)
    Sinv = jnp.linalg.solve(S, jnp.eye(n))

    # Joseph update of the dynamic state (identical form to `_update`).
    K = Pp @ H_dyn.T @ Sinv
    x = xp + K @ y0
    I_KH = jnp.eye(len(xp)) - K @ H_dyn
    P = I_KH @ Pp @ I_KH.T + K @ R @ K.T

    # Sensitivity update: Ξ_filt = (I - K H_dyn) Ξ_pred - K H_eps.
    Xi = I_KH @ Xi_pred - K @ H_eps

    # Epsilon information contributions for this epoch.
    Sinv_y0 = Sinv @ y0
    Sinv_Psi = Sinv @ Psi
    dA = Psi.T @ Sinv_Psi
    db = Psi.T @ Sinv_y0
    dc = (y0.T @ Sinv_y0)[0, 0]
    dL = jnp.where(sign > 0, logdet, jnp.inf)

    return x, P, Xi, dA, db, dc, dL


@partial(jax.jit, static_argnames=("Npsr", "M_sum"))
def _precompute_transition_matrices(γa, γp, σa2, σp2, dt_array, Npsr, M_sum):
    """Precompute all F and Q matrices for all timesteps.

    Args:
        γa: GW damping parameter
        γp: Pulsar damping parameters
        σa2: GW noise variance matrix
        σp2: Pulsar noise variance parameters
        dt_array: Array of time differences
        Npsr: Number of pulsars
        M_sum: Sum of timing model dimensions

    Returns
    -------
        tuple: (F_matrices, Q_matrices) where each is a tuple of (gw_matrices, spin_matrices)
    """
    # Use vmap to vectorize over all timesteps
    vectorized_get_F = jax.vmap(lambda dt: get_F(γa, γp, dt, Npsr, M_sum))
    vectorized_get_Q = jax.vmap(lambda dt: get_Q(γa, σa2, γp, σp2, dt))

    F_gw_all, F_spin_all = vectorized_get_F(dt_array)
    Q_gw_all, Q_spin_all = vectorized_get_Q(dt_array)

    return (F_gw_all, F_spin_all), (Q_gw_all, Q_spin_all)


@jax.named_call
@partial(jax.jit, static_argnames=("Npsr", "M_sum", "dim_x", "n_states"))
def _run_kalman_filter_scan(
    θ,
    data,
    data_errors,
    H_matrices,
    Npsr,
    M_sum,
    hellings_downs_matrix,
    dt_array,
    dim_x,
    n_states,
    P_eps,
    mask_matrices,
):
    """Run the Kalman filter algorithm over all observations and return a log likelihood.

    `mask_matrices` has shape (nepoch, Npsr): entry (t, n) is 1.0 if pulsar n is
    observed at epoch t and 0.0 if it is absent. An all-ones mask reproduces the
    original every-pulsar-present likelihood exactly.
    """
    Γ = _effective_orf(hellings_downs_matrix, θ.orf_epsilon)
    σa2 = _compute_sigma_matrix(θ.ha**2, θ.γa, Γ)

    x0, P0 = _initialize_kalman_filter(n_states, Npsr, P_eps, σa2, θ.γa, θ.σp**2, θ.γp)

    # Precompute the R matrix for this parameter set and these data errors
    R_matrices = precompute_R_matrices(data_errors, θ.EFAC, θ.EQUAD)

    # Precompute all F and Q matrices for all timesteps
    # We need matrices for indices 0 to len(data)-2 (corresponding to dt_array)
    dt_indices = jnp.arange(len(data) - 1)
    F_matrices, Q_matrices = _precompute_transition_matrices(
        θ.γa, θ.γp, σa2, θ.σp**2, dt_array[dt_indices], Npsr, M_sum
    )

    # First update
    x, P, y, S = _update(
        xp=x0,
        Pp=P0,
        H=H_matrices[0, :, :],
        R=R_matrices[0, :, :],
        z=data[0],
        mask=mask_matrices[0],
    )
    ll0 = _log_likelihood(y, S)

    def step(carry, inputs):
        x, P = carry
        z, R, H, mask, F_gw, F_spin, Q_gw, Q_spin = inputs

        # Use precomputed matrices
        F = (F_gw, F_spin)
        Q = (Q_gw, Q_spin)

        x_predict, P_predict = _predict(x, P, F, Q, dim_x)

        x_new, P_new, y, S = _update(x_predict, P_predict, H, R, z, mask)
        ll = _log_likelihood(y, S)
        return (x_new, P_new), ll

    # Pack inputs for scan - include precomputed matrices
    F_gw_all, F_spin_all = F_matrices
    Q_gw_all, Q_spin_all = Q_matrices

    inputs = (
        data[1:],
        R_matrices[1:],
        H_matrices[1:],
        mask_matrices[1:],
        F_gw_all,
        F_spin_all,
        Q_gw_all,
        Q_spin_all,
    )

    # Run scan loop
    (xf, Pf), ll_arr = lax.scan(step, (x, P), inputs)

    total_ll = ll0 + jnp.sum(ll_arr)
    return total_ll[0][0]


@jax.named_call
@partial(jax.jit, static_argnames=("Npsr", "M_sum", "dim_x", "n_states", "diffuse"))
def _run_kalman_filter_marginal(
    θ,
    data,
    data_errors,
    H_matrices,
    Npsr,
    M_sum,
    hellings_downs_matrix,
    dt_array,
    dim_x,
    n_states,
    P_eps_inv,
    diffuse=False,
):
    """Marginalized (Rao-Blackwellized) Kalman filter log likelihood.

    Mathematically equivalent to `_run_kalman_filter_scan` but marginalizes the static
    linearized timing-model parameters β (dim `M_sum`) analytically instead of carrying
    them as state. The recursion therefore propagates only the `4*Npsr` dynamic (GW+spin)
    state, cutting the per-epoch O(d^3) Joseph update from d = 4*Npsr + M_sum to d = 4*Npsr.

    The result equals the full augmented-state marginal likelihood in exact arithmetic
    (Gaussian integration commutes), so it reproduces the golden likelihood. Because β is
    static and enters only the measurement linearly, every filtered mean is affine in β and
    every covariance is β-independent; we track the sensitivity Ξ = ∂x/∂β and accumulate

        A = Σ Ψ_k' S_k⁻¹ Ψ_k,  b = Σ Ψ_k' S_k⁻¹ ỹ_k,  c = Σ ỹ_k' S_k⁻¹ ỹ_k,  L = Σ logdet(2π S_k),

    with Ψ_k = H_eps_k + H_dyn_k Ξ_pred_k the innovation sensitivity and ỹ_k the β=0
    innovation. With Λ = P_eps⁻¹ + A the marginal likelihood is

        logL = -0.5 [ c + L - b' Λ⁻¹ b + logdet(Λ) - logdet(P_eps⁻¹) ].

    `P_eps_inv` is the prior precision block P_eps⁻¹ (= Σ_n M_n' N_n⁻¹ M_n, block-diagonal
    over pulsars); it is passed in rather than P_eps so we never invert the (potentially
    ill-conditioned) full prior covariance.

    If `diffuse` is True the timing-model prior is taken to be flat/improper
    (P_eps⁻¹ → 0), the community-standard PTA treatment (van Haasteren & Levin), which
    fully projects the timing-model subspace out of the data. The formula collapses to

        Λ = A,   logL = -0.5 [ c + L - b' A⁻¹ b + logdet(A) ],

    i.e. `P_eps_inv` is dropped from Λ and the `logdet(P_eps⁻¹)` term is discarded. The
    latter is parameter-independent, so the diffuse marginal likelihood is defined only up
    to an additive constant (harmless for posteriors and for Bayes factors between models
    sharing the same timing model, where it cancels). `P_eps_inv` is then unused.
    """
    Γ = _effective_orf(hellings_downs_matrix, θ.orf_epsilon)
    σa2 = _compute_sigma_matrix(θ.ha**2, θ.γa, Γ)

    x0, P0 = _initialize_dynamic_kalman_filter(Npsr, σa2, θ.γa, θ.σp**2, θ.γp)

    R_matrices = precompute_R_matrices(data_errors, θ.EFAC, θ.EQUAD)

    dt_indices = jnp.arange(len(data) - 1)
    F_matrices, Q_matrices = _precompute_transition_matrices(
        θ.γa, θ.γp, σa2, θ.σp**2, dt_array[dt_indices], Npsr, M_sum
    )

    # Split the (precomputed) H matrices once into dynamic (GW+spin) and epsilon columns.
    n_dyn = 4 * Npsr
    H_dyn_all = H_matrices[:, :, :n_dyn]
    H_eps_all = H_matrices[:, :, n_dyn:]

    # Epoch 0 is update-only against the prior: Ξ_pred = 0 => Ξ_filt = -K0 H_eps0.
    Xi0 = jnp.zeros((n_dyn, M_sum))
    x, P, Xi, A, b, c, L = _update_marginal(
        xp=x0,
        Pp=P0,
        Xi_pred=Xi0,
        H_dyn=H_dyn_all[0],
        H_eps=H_eps_all[0],
        R=R_matrices[0],
        z=data[0],
    )

    def step(carry, inputs):
        x, P, Xi, A, b, c, L = carry
        z, R, H_dyn, H_eps, F_gw, F_spin, Q_gw, Q_spin = inputs

        F = (F_gw, F_spin)
        Q = (Q_gw, Q_spin)

        x_pred = compute_predicted_state(F, x, dim_x, dim_x)
        P_pred = _predict_dynamic_cov(P, F, Q, dim_x)
        Xi_pred = _predict_xi(Xi, F_gw, F_spin, dim_x)

        x, P, Xi, dA, db, dc, dL = _update_marginal(
            x_pred, P_pred, Xi_pred, H_dyn, H_eps, R, z
        )
        return (x, P, Xi, A + dA, b + db, c + dc, L + dL), None

    F_gw_all, F_spin_all = F_matrices
    Q_gw_all, Q_spin_all = Q_matrices

    inputs = (
        data[1:],
        R_matrices[1:],
        H_dyn_all[1:],
        H_eps_all[1:],
        F_gw_all,
        F_spin_all,
        Q_gw_all,
        Q_spin_all,
    )

    # Accumulate inside the carry (no stacked outputs) to avoid materialising a
    # (T, M_sum, M_sum) array of per-epoch A contributions.
    (x, P, Xi, A, b, c, L), _ = lax.scan(step, (x, P, Xi, A, b, c, L), inputs)

    if diffuse:
        # Flat/improper prior limit P_eps⁻¹ → 0: Λ = A, and the logdet(P_eps⁻¹) term is
        # dropped (a parameter-independent constant). A can be weakly conditioned when the
        # timing params are poorly constrained, so mirror the `_log_likelihood` /
        # `_update_marginal` policy: symmetrise, add magnitude-scaled jitter, and reject a
        # residually non-PD matrix by returning -inf.
        n = A.shape[0]
        Lambda = 0.5 * (A + A.T)
        jitter = 1e-9 * (jnp.trace(Lambda) / n)
        Lambda = Lambda + jitter * jnp.eye(n)
        sign, logdet_Lambda = jnp.linalg.slogdet(Lambda)
        bLb = (b.T @ jnp.linalg.solve(Lambda, b))[0, 0]
        logL = -0.5 * (c + L - bLb + logdet_Lambda)
        return jnp.where(sign > 0, logL, -jnp.inf)

    # Analytic marginalization of β with prior precision P_eps_inv.
    Lambda = P_eps_inv + A
    _, logdet_Lambda = jnp.linalg.slogdet(Lambda)
    _, logdet_P_eps_inv = jnp.linalg.slogdet(P_eps_inv)
    bLb = (b.T @ jnp.linalg.solve(Lambda, b))[0, 0]

    logL = -0.5 * (c + L - bLb + logdet_Lambda - logdet_P_eps_inv)
    return logL


class JaxKalmanFilter:
    """A class to implement the linear Kalman filter on scalar inputs using JAX.

    Args:
        df_psr: DataFrame containing pulsar information including:
            - dim_M: integer, number of design parameters for that pulsar
            - F0: pulsar spin frequency
        observations: Dictionary containing 'toas', 'residuals', and 'errors' arrays from the data loader
        Peps: The uncertainty matrix for the epsilon states
        hd_correlation_matrix: Precomputed Hellings-Downs correlation matrix
        pulsar_design_matrices: Design matrices for each pulsar
        use_gw: If True, include GW terms in measurement equation. Default True.
    """

    def __init__(
        self,
        data: dict,
        use_gw: bool = True,
        use_marginal: bool | None = None,
        timing_prior: str = "informative",
        prior_scale: float = 1.0,
    ):
        """Initialize the class.

        Args:
            data: Loaded pulsar data dictionary (see data_loader).
            use_gw: If True, include GW terms in the measurement equation. Default True.
            use_marginal: If True, use the marginalized (Rao-Blackwellized) filter
                that analytically integrates out the timing-model parameters instead of
                carrying them as state; set False for the original sequential augmented-state
                filter. The two are mathematically equivalent (identical log likelihood to
                <1 nat on the MDC2 golden dataset), but the marginal path is faster (~1.4x on
                A100, ~2x on CPU) because the propagated state shrinks from 4*Npsr + M_sum to
                4*Npsr, cutting the per-epoch O(d^3) update. The default (None) selects the
                marginal filter unless the data carry a per-epoch observation mask, which
                only the sequential filter supports — masked data fall back to the
                sequential path with a warning. Explicitly requesting use_marginal=True
                together with a mask raises NotImplementedError.
            timing_prior: Prior on the linearized timing-model parameters β. "informative"
                (default) uses the data-matched GLS prior P_eps = (MᵀN⁻¹M)⁻¹ and reproduces
                the golden likelihood. "diffuse" takes the flat/improper limit P_eps⁻¹ → 0
                (the community-standard PTA treatment), fully projecting the timing-model
                subspace out of the data; only supported on the marginal backend.
            prior_scale: Multiplicative scale α on the informative prior covariance
                (P_eps → α·P_eps, P_eps⁻¹ → P_eps⁻¹/α). Default 1.0 reproduces the golden
                likelihood; large α weakens the prior toward the diffuse limit. Ignored when
                timing_prior="diffuse".
        """
        get_logger().info("Initializing JaxKalmanFilter...")

        if timing_prior not in ("informative", "diffuse"):
            raise ValueError(
                f"timing_prior must be 'informative' or 'diffuse', got {timing_prior!r}"
            )
        self.timing_prior = timing_prior

        observations = data["processed_residuals"]

        # Per-epoch observation mask (nepoch, Npsr): 1.0 present, 0.0 absent. Absent when
        # not supplied by the data loader -> all-ones, i.e. the original fully-observed
        # behaviour (backward compatible with MDC2 and the intersection-aligned feathers).
        self.mask = observations.get("mask", None)

        # Resolve the filter backend. Masked epochs are only wired into the sequential
        # augmented-state filter; the marginalized filter has no masked-update path yet.
        if use_marginal is None:
            use_marginal = self.mask is None
            if self.mask is not None:
                get_logger().warning(
                    "Data carry a missing-observation mask: falling back to the "
                    "sequential augmented-state filter (use_marginal=False); the "
                    "marginalized filter does not yet support masked epochs."
                )
        elif use_marginal and self.mask is not None:
            raise NotImplementedError(
                "use_marginal=True is not supported for data with a missing-observation "
                "mask: the marginalized filter has no masked-update path. Omit "
                "use_marginal (auto-fallback) or pass use_marginal=False."
            )
        if timing_prior == "diffuse" and not use_marginal:
            raise ValueError(
                "timing_prior='diffuse' is only supported on the marginalized filter "
                "(use_marginal=True); the augmented-state filter cannot represent a flat "
                "timing-model prior without hitting the near-singular P0 pathology."
            )
        df_psr = data["metadata"]
        pulsar_design_matrices = data["design_matrices"]
        P_eps_matrices = data["parameter_covariances"]
        hd_correlation_matrix = data["hd_correlation"]

        alpha = prior_scale  # informative-prior covariance scale (α → ∞ ≈ diffuse)
        Peps = alpha * block_diag(*P_eps_matrices)

        # Prior precision block P_eps⁻¹ for the marginalized filter. Built by per-pulsar
        # inversion so it is exactly the inverse of the block-diagonal `Peps` above (the
        # augmented filter's prior), while staying well-conditioned (small per-pulsar blocks)
        # and never inverting the full M_sum×M_sum prior covariance.
        self.use_marginal = use_marginal
        self.P_eps_inv = (
            block_diag(*[jnp.linalg.inv(jnp.asarray(pc)) for pc in P_eps_matrices])
            / alpha
        )

        # Store observations and Peps
        self.observations = observations
        self.P_eps = Peps

        # Extract the observations using dictionary keys
        self.toa = self.observations["toas"]
        self.data = self.observations["residuals"]
        self.data_errors = self.observations["errors"]
        self.t_diffs = np.diff(self.toa)

        # Initialize model parameters from df_psr
        self.Npsr = int(len(df_psr))
        get_logger().info(f"Number of pulsars: {self.Npsr}")
        self.use_gw = use_gw

        if not self.use_gw:
            get_logger().info(
                "Initializing null GW model - GW states present but not used in measurements"
            )

        # Calculate state dimensions
        self.M = df_psr["dim_M"].values.astype(int)  # array of integers
        self.M_sum = self.M.sum()
        # Total state dimension: for each pulsar, two state variables from spin noise,
        # two from GW noise, and dim_M extra parameters
        self.nx = self.Npsr * (2 + 2) + self.M_sum

        # Store correlation and design matrices
        self.hd_correlation_matrix = hd_correlation_matrix
        self.pulsar_design_matrices = pulsar_design_matrices

        # Calculate timing parameter start indices
        self.M_start_indices = np.cumsum([0] + [m for m in self.M]) + 4 * self.Npsr

        # Store pulsar frequencies
        self.f0 = df_psr["F0"].values
        get_logger().info(f"Pulsar frequencies: {self.f0}")

        get_logger().info(f"Total number of observations: {len(self.data)}")
        get_logger().info(f"Starting dt (days): {self.t_diffs[0]/86400}")
        get_logger().info(f"Ending dt (days): {self.t_diffs[-1]/86400}")
        get_logger().info(f"The errors at t=1 are: {self.data_errors[0,:]}")

        # Precompute the observation matrices
        self.Hmat = precompute_H_matrix(
            self.Npsr,
            self.nx,
            self.M_start_indices,
            self.pulsar_design_matrices,
            self.use_gw,
            self.f0,
        )

        # Convert to JAX arrays for faster processing
        self._prepare_jax_arrays()

    def _prepare_jax_arrays(self):
        """Convert numpy arrays to JAX arrays and verify they are 64-bit."""
        # Convert observations and related data
        self.jax_data = jnp.array(self.data)
        self.jax_data_errors = jnp.array(self.data_errors)
        self.jax_t_diffs = jnp.array(self.t_diffs)

        # Build the observation mask as float64 (ones where no mask was supplied).
        if self.mask is None:
            self.jax_mask_matrices = jnp.ones_like(self.jax_data)
        else:
            self.jax_mask_matrices = jnp.asarray(self.mask, dtype=jnp.float64)

        # Convert H matrices
        self.jax_H_matrices = jnp.array(self.Hmat)

        # Convert hellings downs matrix
        self.hellings_downs_matrix = jnp.array(self.hd_correlation_matrix)

        # Verify all floating-point arrays are 64-bit
        float_arrays = [
            ("jax_data", self.jax_data),
            ("jax_data_errors", self.jax_data_errors),
            ("jax_t_diffs", self.jax_t_diffs),
            ("jax_H_matrices", self.jax_H_matrices),
            ("hellings_downs_matrix", self.hellings_downs_matrix),
            ("jax_mask_matrices", self.jax_mask_matrices),
        ]

        for name, arr in float_arrays:
            if arr.dtype != jnp.float64:
                raise ValueError(
                    f"{name} is {arr.dtype}, expected {jnp.float64}. The Kalman filter requires floats at standard precision for numerical stability."
                )

    def get_likelihood(self, θ):
        """Run the Kalman filter algorithm over all observations and return a log likelihood."""
        if self.use_marginal:
            return _run_kalman_filter_marginal(
                θ=θ,
                data=self.jax_data,
                data_errors=self.jax_data_errors,
                H_matrices=self.jax_H_matrices,
                Npsr=self.Npsr,
                M_sum=self.M_sum,
                hellings_downs_matrix=self.hellings_downs_matrix,
                dt_array=self.jax_t_diffs,
                dim_x=2 * self.Npsr,
                n_states=self.nx,
                P_eps_inv=self.P_eps_inv,
                diffuse=(self.timing_prior == "diffuse"),
            )
        return _run_kalman_filter_scan(
            θ=θ,
            data=self.jax_data,
            data_errors=self.jax_data_errors,
            H_matrices=self.jax_H_matrices,
            Npsr=self.Npsr,
            M_sum=self.M_sum,
            hellings_downs_matrix=self.hellings_downs_matrix,
            dt_array=self.jax_t_diffs,
            dim_x=2 * self.Npsr,
            n_states=self.nx,
            P_eps=self.P_eps,
            mask_matrices=self.jax_mask_matrices,
        )

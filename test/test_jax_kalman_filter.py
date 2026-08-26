"""Unit tests for jax_kalman_filter module."""

import pytest
import numpy as np
import jax.numpy as jnp
from unittest.mock import Mock, patch
from argus import jax_kalman_filter, bayesian_inference


class TestComputePredictedState:
    """Tests for compute_predicted_state function."""

    def test_basic_state_prediction(self):
        """Test basic state prediction."""
        # Setup simple transition matrices
        F_gw = jnp.eye(4)  # Identity for GW states
        F_spin = jnp.eye(4)  # Identity for spin states
        F_list = (F_gw, F_spin)

        # Current state: 4 GW + 4 spin + 2 timing = 10 total
        x = jnp.arange(10).reshape(-1, 1).astype(float)

        x_pred = jax_kalman_filter.compute_predicted_state(
            F_list, x, gw_size=4, spin_size=4
        )

        # With identity matrices, prediction should equal input
        assert jnp.allclose(x_pred, x)

    def test_timing_states_unchanged(self):
        """Test that timing states remain unchanged."""
        F_gw = jnp.array([[1.0, 0.1], [0.0, 0.9]])  # 2x2
        F_spin = jnp.array([[1.0, 0.2], [0.0, 0.8]])  # 2x2
        F_list = (F_gw, F_spin)

        x = jnp.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])  # 6 states

        x_pred = jax_kalman_filter.compute_predicted_state(
            F_list, x, gw_size=2, spin_size=2
        )

        # Timing states (last 2) should be unchanged
        assert jnp.allclose(x_pred[4:], x[4:])


class TestComputePredictedCovariance:
    """Tests for compute_predicted_covariance function."""

    def test_basic_covariance_prediction(self):
        """Test basic covariance prediction."""
        # Simple test with identity matrices
        P = jnp.eye(6)
        F_gw = jnp.eye(2)
        F_spin = jnp.eye(2)
        Q_gw = jnp.zeros((2, 2))
        Q_spin = jnp.zeros((2, 2))

        P_pred = jax_kalman_filter.compute_predicted_covariance(
            P, (F_gw, F_spin), (Q_gw, Q_spin), gw_size=2, spin_size=2
        )

        # With identity F and zero Q, should get identity P back
        assert jnp.allclose(P_pred, P)

    def test_process_noise_addition(self):
        """Test that process noise is added correctly."""
        P = jnp.zeros((6, 6))
        F_gw = jnp.eye(2)
        F_spin = jnp.eye(2)
        Q_gw = jnp.eye(2) * 0.1
        Q_spin = jnp.eye(2) * 0.2

        P_pred = jax_kalman_filter.compute_predicted_covariance(
            P, (F_gw, F_spin), (Q_gw, Q_spin), gw_size=2, spin_size=2
        )

        # GW block should have Q_gw
        assert jnp.allclose(P_pred[:2, :2], Q_gw)
        # Spin block should have Q_spin
        assert jnp.allclose(P_pred[2:4, 2:4], Q_spin)


class TestLogLikelihood:
    """Tests for _log_likelihood function."""

    def test_basic_likelihood(self):
        """Test basic log likelihood calculation."""
        y = jnp.array([[0.1], [0.2]])
        cov = jnp.eye(2)

        ll = jax_kalman_filter._log_likelihood(y, cov)

        # Should return a (1, 1) array, not a scalar
        assert ll.shape == (1, 1)
        assert jnp.isfinite(ll)
        # For non-zero innovation, likelihood should be negative
        assert ll[0, 0] < 0

    def test_zero_innovation(self):
        """Test likelihood with zero innovation."""
        y = jnp.zeros((2, 1))
        cov = jnp.eye(2)

        ll = jax_kalman_filter._log_likelihood(y, cov)

        # Log likelihood of zero innovation should be relatively high
        assert ll > -10


class TestUpdate:
    """Tests for _update function."""

    def test_basic_update(self):
        """Test basic Kalman filter update step."""
        xp = jnp.zeros((4, 1))
        Pp = jnp.eye(4)
        H = jnp.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        R = jnp.eye(2) * 0.01
        z = jnp.array([0.1, 0.2])

        x, P, y, S = jax_kalman_filter._update(xp, Pp, H, R, z)

        # Updated state should be closer to measurement
        assert not jnp.allclose(x, xp)
        # Innovation should be non-zero
        assert not jnp.allclose(y, 0.0)
        # Updated covariance should be smaller
        assert jnp.trace(P) < jnp.trace(Pp)

    def test_joseph_form_symmetry(self):
        """Test that Joseph form maintains symmetry."""
        xp = jnp.ones((3, 1))
        Pp = jnp.eye(3)
        H = jnp.array([[1.0, 0.5, 0.0]])
        R = jnp.array([[0.01]])
        z = jnp.array([1.5])

        x, P, y, S = jax_kalman_filter._update(xp, Pp, H, R, z)

        # P should remain symmetric
        assert jnp.allclose(P, P.T)


class TestJaxKalmanFilterInitialization:
    """Tests for JaxKalmanFilter initialization."""

    @patch("argus.io_manager.get_argus_logger")
    def test_basic_initialization(self, mock_logger, sample_pulsar_data):
        """Test basic Kalman filter initialization."""
        mock_logger.return_value = Mock()

        kf = jax_kalman_filter.JaxKalmanFilter(data=sample_pulsar_data, use_gw=True)

        assert kf.Npsr == 2
        assert kf.use_gw is True
        assert kf.nx > 0

    @patch("argus.io_manager.get_argus_logger")
    def test_no_gw_initialization(self, mock_logger, sample_pulsar_data):
        """Test initialization without GW."""
        mock_logger.return_value = Mock()

        kf = jax_kalman_filter.JaxKalmanFilter(data=sample_pulsar_data, use_gw=False)

        assert kf.use_gw is False

    @patch("argus.io_manager.get_argus_logger")
    def test_jax_array_conversion(self, mock_logger, sample_pulsar_data):
        """Test that numpy arrays are converted to JAX arrays."""
        mock_logger.return_value = Mock()

        kf = jax_kalman_filter.JaxKalmanFilter(data=sample_pulsar_data, use_gw=True)

        # Check JAX array types
        assert isinstance(kf.jax_data, jnp.ndarray)
        assert isinstance(kf.jax_data_errors, jnp.ndarray)
        assert isinstance(kf.jax_t_diffs, jnp.ndarray)

    @patch("argus.io_manager.get_argus_logger")
    def test_float64_precision(self, mock_logger, sample_pulsar_data):
        """Test that arrays are 64-bit precision."""
        mock_logger.return_value = Mock()

        kf = jax_kalman_filter.JaxKalmanFilter(data=sample_pulsar_data, use_gw=True)

        # All arrays should be float64
        assert kf.jax_data.dtype == jnp.float64
        assert kf.jax_data_errors.dtype == jnp.float64
        assert kf.jax_H_matrices.dtype == jnp.float64


class TestGetLikelihood:
    """Tests for get_likelihood method."""

    @patch("argus.io_manager.get_argus_logger")
    def test_likelihood_computation(
        self, mock_logger, sample_pulsar_data, sample_noise_parameters
    ):
        """Test basic likelihood computation."""
        mock_logger.return_value = Mock()

        kf = jax_kalman_filter.JaxKalmanFilter(data=sample_pulsar_data, use_gw=True)

        # Create test parameters
        params = bayesian_inference.Parameters(
            log10_gamma_a=-9.0,
            γa=1e-9,
            ha=1e-15,
            γp=sample_noise_parameters["gamma_p"],
            σp=sample_noise_parameters["sigma_p"],
            EFAC=sample_noise_parameters["efac"],
            EQUAD=sample_noise_parameters["equad"],
        )

        ll = kf.get_likelihood(params)

        # Likelihood should be a scalar
        assert ll.shape == ()
        # Should be finite
        assert jnp.isfinite(ll)


class TestMaskedUpdate:
    """Tests for the missing-observation mask in _update."""

    def test_all_ones_mask_matches_no_mask(self):
        """An all-ones mask must reproduce the no-mask update bit-for-bit."""
        xp = jnp.zeros((4, 1))
        Pp = jnp.array(
            [
                [1.0, 0.5, 0.1, 0.0],
                [0.5, 1.0, 0.0, 0.2],
                [0.1, 0.0, 1.0, 0.3],
                [0.0, 0.2, 0.3, 1.0],
            ]
        )
        H = jnp.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
        R = jnp.eye(2) * 0.01
        z = jnp.array([0.1, 0.2])

        x0, P0, y0, S0 = jax_kalman_filter._update(xp, Pp, H, R, z)
        x1, P1, y1, S1 = jax_kalman_filter._update(xp, Pp, H, R, z, mask=jnp.ones(2))

        assert jnp.array_equal(x0, x1)
        assert jnp.array_equal(P0, P1)
        assert jnp.array_equal(y0, y1)
        assert jnp.array_equal(S0, S1)

    def test_all_absent_update_is_noop(self):
        """An all-zero mask must leave state/covariance unchanged (no observation)."""
        xp = jnp.array([[1.0], [2.0], [3.0], [4.0]])
        Pp = jnp.eye(4) + 0.1
        H = jnp.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
        R = jnp.eye(2) * 0.01
        z = jnp.array([5.0, 6.0])

        x, P, y, S = jax_kalman_filter._update(xp, Pp, H, R, z, mask=jnp.zeros(2))

        # No pulsar observed -> no update.
        assert jnp.allclose(x, xp)
        assert jnp.allclose(P, Pp)
        # Innovations zeroed; S is the identity block so it stays invertible.
        assert jnp.allclose(y, 0.0)
        assert jnp.allclose(S, jnp.eye(2))

    def test_partial_mask_matches_reduced_dimension_update(self):
        """Masking pulsar 1 must equal the update that only ever saw pulsar 0."""
        xp = jnp.zeros((2, 1))
        Pp = jnp.array([[1.0, 0.5], [0.5, 1.0]])
        H = jnp.eye(2)
        R = jnp.diag(jnp.array([0.1, 0.1]))
        z = jnp.array([1.0, 2.0])

        # Full 2-D update with pulsar 1 masked absent.
        x_m, P_m, y_m, S_m = jax_kalman_filter._update(
            xp, Pp, H, R, z, mask=jnp.array([1.0, 0.0])
        )

        # Genuine 1-measurement update using only pulsar 0's row.
        H_r = jnp.array([[1.0, 0.0]])
        R_r = jnp.array([[0.1]])
        z_r = jnp.array([1.0])
        x_r, P_r, y_r, S_r = jax_kalman_filter._update(xp, Pp, H_r, R_r, z_r)

        # The masked full update and the reduced update give the same state + covariance.
        assert jnp.allclose(x_m, x_r)
        assert jnp.allclose(P_m, P_r)


class TestMaskedFilterEquivalence:
    """Filter-level tests that masking is the correct missing-observation treatment."""

    @pytest.mark.parametrize("use_marginal", [False, True])
    @patch("argus.io_manager.get_argus_logger")
    def test_all_ones_mask_matches_default(
        self, mock_logger, use_marginal, sample_pulsar_data, sample_noise_parameters
    ):
        """Threading an explicit all-ones mask through the scan changes nothing.

        Checked on both backends: masks used to be wired into the sequential filter
        only, so this comparison had to pin use_marginal=False. Now the marginalized
        filter supports them too and must be equally unaffected.
        """
        mock_logger.return_value = Mock()

        params = bayesian_inference.Parameters(
            log10_gamma_a=-9.0,
            γa=1e-9,
            ha=1e-15,
            γp=sample_noise_parameters["gamma_p"],
            σp=sample_noise_parameters["sigma_p"],
            EFAC=sample_noise_parameters["efac"],
            EQUAD=sample_noise_parameters["equad"],
        )

        kf_default = jax_kalman_filter.JaxKalmanFilter(
            data=sample_pulsar_data, use_gw=True, use_marginal=use_marginal
        )
        ll_default = kf_default.get_likelihood(params)

        # Same data, but with an explicit all-ones mask supplied by the loader.
        residuals = dict(sample_pulsar_data["processed_residuals"])
        residuals["mask"] = np.ones_like(residuals["residuals"])
        data_masked = dict(sample_pulsar_data)
        data_masked["processed_residuals"] = residuals

        kf_masked = jax_kalman_filter.JaxKalmanFilter(
            data=data_masked, use_gw=True, use_marginal=use_marginal
        )
        ll_masked = kf_masked.get_likelihood(params)

        assert jnp.allclose(ll_default, ll_masked, rtol=0, atol=0)

    @patch("argus.io_manager.get_argus_logger")
    def test_appending_absent_epochs_adds_only_constant(
        self, mock_logger, sample_pulsar_data, sample_noise_parameters
    ):
        """Extra epochs where every pulsar is absent add no information.

        Absent epochs are appended at the END so no existing time step is subdivided
        (the filter's covariance propagation is not required to compose over sub-steps,
        which is a separate pre-existing property). A correct mask then leaves the
        likelihood unchanged apart from the fixed log(2pi) contribution the identity
        innovation slot adds for each absent (epoch, pulsar) pair.
        """
        mock_logger.return_value = Mock()

        params = bayesian_inference.Parameters(
            log10_gamma_a=-9.0,
            γa=1e-9,
            ha=1e-15,
            γp=sample_noise_parameters["gamma_p"],
            σp=sample_noise_parameters["sigma_p"],
            EFAC=sample_noise_parameters["efac"],
            EQUAD=sample_noise_parameters["equad"],
        )

        base = sample_pulsar_data["processed_residuals"]
        nepoch, npsr = base["residuals"].shape

        residuals = dict(base)
        residuals["mask"] = np.ones((nepoch, npsr))
        data_base = dict(sample_pulsar_data)
        data_base["processed_residuals"] = residuals
        ll_base = jax_kalman_filter.JaxKalmanFilter(
            data=data_base, use_gw=True
        ).get_likelihood(params)

        # Append K all-absent epochs beyond the last real TOA.
        K = 3
        toas = base["toas"]
        dt = toas[1] - toas[0]
        extra_toas = toas[-1] + dt * np.arange(1, K + 1)
        new_toas = np.concatenate([toas, extra_toas])
        new_res = np.vstack([base["residuals"], np.zeros((K, npsr))])
        new_err = np.vstack([base["errors"], base["errors"][:K]])
        new_design = [
            np.vstack([dm, np.zeros((K, dm.shape[1]))])
            for dm in sample_pulsar_data["design_matrices"]
        ]
        new_mask = np.vstack([np.ones((nepoch, npsr)), np.zeros((K, npsr))])

        data_ext = dict(sample_pulsar_data)
        data_ext["processed_residuals"] = {
            "toas": new_toas,
            "residuals": new_res,
            "errors": new_err,
            "mask": new_mask,
        }
        data_ext["design_matrices"] = new_design
        ll_ext = jax_kalman_filter.JaxKalmanFilter(
            data=data_ext, use_gw=True
        ).get_likelihood(params)

        offset = 0.5 * (K * npsr) * np.log(2.0 * np.pi)
        np.testing.assert_allclose(ll_ext + offset, ll_base, rtol=1e-8, atol=1e-6)

    @patch("argus.io_manager.get_argus_logger")
    @pytest.mark.parametrize("use_marginal", [False, True])
    def test_absent_pulsar_data_is_ignored(
        self, mock_logger, use_marginal, sample_pulsar_data, sample_noise_parameters
    ):
        """The likelihood must not depend on a pulsar's data where it is masked absent.

        This is the core end-to-end correctness property of the mask: a pulsar marked
        absent at every epoch contributes no data-dependent information, so arbitrarily
        changing its residuals (and errors) leaves the likelihood bit-for-bit unchanged.
        """
        mock_logger.return_value = Mock()

        params = bayesian_inference.Parameters(
            log10_gamma_a=-9.0,
            γa=1e-9,
            ha=1e-15,
            γp=sample_noise_parameters["gamma_p"],
            σp=sample_noise_parameters["sigma_p"],
            EFAC=sample_noise_parameters["efac"],
            EQUAD=sample_noise_parameters["equad"],
        )

        base = sample_pulsar_data["processed_residuals"]
        nepoch, npsr = base["residuals"].shape
        mask = np.ones((nepoch, npsr))
        mask[:, 1] = 0.0  # pulsar 1 absent everywhere

        def ll_with(res_col1, err_col1):
            res = base["residuals"].copy()
            err = base["errors"].copy()
            res[:, 1] = res_col1
            err[:, 1] = err_col1
            data = dict(sample_pulsar_data)
            data["processed_residuals"] = {
                "toas": base["toas"],
                "residuals": res,
                "errors": err,
                "mask": mask,
            }
            return jax_kalman_filter.JaxKalmanFilter(
                data=data, use_gw=True, use_marginal=use_marginal
            ).get_likelihood(params)

        ll_a = ll_with(base["residuals"][:, 1], base["errors"][:, 1])
        ll_b = ll_with(np.random.randn(nepoch) * 1e-3, np.full(nepoch, 5e-7))

        assert jnp.array_equal(ll_a, ll_b)


class TestMaskBackendSelection:
    """Backend resolution: both filters support masks, so the fast one is the default."""

    @staticmethod
    def _with_mask(sample_pulsar_data):
        residuals = dict(sample_pulsar_data["processed_residuals"])
        residuals["mask"] = np.ones_like(residuals["residuals"])
        data = dict(sample_pulsar_data)
        data["processed_residuals"] = residuals
        return data

    @patch("argus.io_manager.get_argus_logger")
    def test_default_without_mask_is_marginal(self, mock_logger, sample_pulsar_data):
        mock_logger.return_value = Mock()
        kf = jax_kalman_filter.JaxKalmanFilter(data=sample_pulsar_data, use_gw=True)
        assert kf.use_marginal is True

    @patch("argus.io_manager.get_argus_logger")
    def test_mask_selects_marginal(self, mock_logger, sample_pulsar_data):
        """A mask no longer forces the slower sequential filter.

        Masked data used to fall back to the sequential path, which gave up the
        marginal filter's speedup on exactly the runs that need it most (union-grid
        array data, where most of the grid is unobserved).
        """
        mock_logger.return_value = Mock()
        kf = jax_kalman_filter.JaxKalmanFilter(
            data=self._with_mask(sample_pulsar_data), use_gw=True
        )
        assert kf.use_marginal is True

    @patch("argus.io_manager.get_argus_logger")
    def test_explicit_marginal_with_mask_succeeds(
        self, mock_logger, sample_pulsar_data
    ):
        """Explicitly requesting the marginal filter with a mask is now supported."""
        mock_logger.return_value = Mock()
        kf = jax_kalman_filter.JaxKalmanFilter(
            data=self._with_mask(sample_pulsar_data),
            use_gw=True,
            use_marginal=True,
        )
        assert kf.use_marginal is True

    @patch("argus.io_manager.get_argus_logger")
    def test_sequential_backend_still_selectable(self, mock_logger, sample_pulsar_data):
        """The sequential path remains available as the cross-check backend."""
        mock_logger.return_value = Mock()
        kf = jax_kalman_filter.JaxKalmanFilter(
            data=self._with_mask(sample_pulsar_data),
            use_gw=True,
            use_marginal=False,
        )
        assert kf.use_marginal is False

    @patch("argus.io_manager.get_argus_logger")
    def test_diffuse_rejects_a_fully_unobserved_pulsar(
        self, mock_logger, sample_pulsar_data
    ):
        """Under a flat timing prior, a pulsar with no data leaves Lambda singular.

        On the informative path such a pulsar only shifts the likelihood by a
        constant, so it is allowed. In the diffuse limit Lambda = A, so its
        timing-model block has no information at all and the jitter — not the data —
        would set the log-determinant. Refused rather than silently meaningless.
        """
        mock_logger.return_value = Mock()
        data = self._with_mask(sample_pulsar_data)
        data["processed_residuals"]["mask"][:, 0] = 0.0
        with pytest.raises(ValueError, match="zero epochs"):
            jax_kalman_filter.JaxKalmanFilter(
                data=data, use_gw=True, timing_prior="diffuse"
            )


class TestPrecomputeTransitionMatrices:
    """Tests for _precompute_transition_matrices function."""

    def test_matrix_precomputation(self):
        """Test precomputation of F and Q matrices."""
        γa = 1e-9
        γp = jnp.array([1e-8, 2e-8])
        σa2 = jnp.eye(2) * 1e-30
        σp2 = jnp.array([1e-30, 2e-30])
        dt_array = jnp.array([1.0, 2.0, 3.0])
        Npsr = 2
        M_sum = 10

        F_matrices, Q_matrices = jax_kalman_filter._precompute_transition_matrices(
            γa, γp, σa2, σp2, dt_array, Npsr, M_sum
        )

        F_gw_all, F_spin_all = F_matrices
        Q_gw_all, Q_spin_all = Q_matrices

        # Should have matrices for each time step
        assert F_gw_all.shape[0] == 3
        assert F_spin_all.shape[0] == 3
        assert Q_gw_all.shape[0] == 3
        assert Q_spin_all.shape[0] == 3


class TestInitializeKalmanFilter:
    """Tests for _initialize_kalman_filter function."""

    def test_initialization_shapes(self):
        """Test that initialization produces correct shapes."""
        nx = 20
        Npsr = 2
        P_eps = jnp.eye(12)  # Timing parameters covariance
        σa2 = jnp.eye(2) * 1e-30
        γa = 1e-9
        σp2 = jnp.array([1e-30, 2e-30])
        γp = jnp.array([1e-8, 2e-8])

        x0, P0 = jax_kalman_filter._initialize_kalman_filter(
            nx, Npsr, P_eps, σa2, γa, σp2, γp
        )

        # Check shapes
        assert x0.shape == (nx, 1)
        assert P0.shape == (nx, nx)

    def test_initialization_values(self):
        """Test initial values are reasonable."""
        nx = 10
        Npsr = 1
        P_eps = jnp.eye(6)
        σa2 = jnp.array([[1e-30]])
        γa = 1e-9
        σp2 = jnp.array([1e-30])
        γp = jnp.array([1e-8])

        x0, P0 = jax_kalman_filter._initialize_kalman_filter(
            nx, Npsr, P_eps, σa2, γa, σp2, γp
        )

        # Initial state should be zero
        assert jnp.allclose(x0, 0.0)

        # Initial covariance should be positive definite
        eigenvalues = jnp.linalg.eigvalsh(P0)
        assert jnp.all(eigenvalues >= 0)

    def test_gw_block_structure(self):
        """Test GW block has correct structure."""
        nx = 10
        Npsr = 1
        P_eps = jnp.eye(6)
        σa2 = jnp.array([[1e-30]])
        γa = 1e-9
        σp2 = jnp.array([1e-30])
        γp = jnp.array([1e-8])

        x0, P0 = jax_kalman_filter._initialize_kalman_filter(
            nx, Npsr, P_eps, σa2, γa, σp2, γp
        )

        # GW 'r' states should have very small variance
        assert P0[0, 0] < 1e-30

        # GW 'a' states should have variance proportional to σa2 / (2*γa)
        expected_var_a = σa2[0, 0] / (2.0 * γa)
        assert jnp.isclose(P0[1, 1], expected_var_a, rtol=0.1)


def _realistic_pulsar_data(seed=0):
    """Build a small, well-conditioned pulsar-data dict for equivalence testing.

    Mirrors the structure the real data loader produces: unit-scaled design matrices
    and a GLS timing-model prior ``P_eps = (Mᵀ N⁻¹ M)⁻¹`` consistent with the design and
    the TOA errors. This is the regime the filter actually runs in. (The generic
    `sample_pulsar_data` fixture uses an arbitrary O(1) timing prior with 1e-6-scale
    residuals — a physically inconsistent configuration whose extreme dynamic range makes
    it a poor equivalence test, even though both backends are individually well-defined.)
    """
    import pandas as pd

    rng = np.random.default_rng(seed)
    n_epochs, n_psr = 12, 2
    dims = [5, 6]
    err_scale = 1e-6  # ~microsecond TOA errors

    metadata = pd.DataFrame(
        {
            "name": ["A", "B"],
            "dim_M": dims,
            "RA": [0.5, 1.2],
            "DEC": [0.3, -0.1],
            "F0": [200.0, 150.0],
            "par_file": ["a", "b"],
            "tim_file": ["a", "b"],
        }
    )
    residuals = rng.standard_normal((n_epochs, n_psr)) * err_scale
    errors = np.ones((n_epochs, n_psr)) * err_scale

    design_matrices, parameter_covariances = [], []
    for i in range(n_psr):
        M = rng.standard_normal((n_epochs, dims[i]))
        M = M / np.sqrt(np.sum(M**2, axis=0))  # unit-norm columns (as data_loader does)
        Ninv = np.diag(1.0 / errors[:, i] ** 2)
        design_matrices.append(M)
        parameter_covariances.append(np.linalg.inv(M.T @ Ninv @ M))  # GLS prior

    return {
        "processed_residuals": {
            "toas": np.linspace(0, 1000, n_epochs) * 86400,
            "residuals": residuals,
            "errors": errors,
        },
        "metadata": metadata,
        "design_matrices": design_matrices,
        "parameter_covariances": parameter_covariances,
        "hd_correlation": np.array([[1.0, 0.5], [0.5, 1.0]]),
    }


class TestMarginalFilter:
    """Tests for the marginalized (Rao-Blackwellized) timing-model filter.

    The marginal filter integrates the static timing-model parameters out of the
    propagated state analytically instead of augmenting the state with them. It is
    mathematically equivalent to the default sequential filter, so the two must return
    the same log likelihood; the marginal path just propagates a smaller state.
    """

    def _params(self):
        return bayesian_inference.Parameters(
            log10_gamma_a=-9.0,
            γa=1e-9,
            ha=1e-15,
            γp=jnp.full(2, 1e-8),
            σp=jnp.full(2, 1e-15),
            EFAC=jnp.ones(2),
            EQUAD=jnp.full(2, 1e-6),
        )

    @patch("argus.io_manager.get_argus_logger")
    def test_matches_sequential(self, mock_logger):
        """Marginal and sequential backends must agree on the log likelihood."""
        mock_logger.return_value = Mock()
        data = _realistic_pulsar_data()

        kf_seq = jax_kalman_filter.JaxKalmanFilter(
            data=data, use_gw=True, use_marginal=False
        )
        kf_marg = jax_kalman_filter.JaxKalmanFilter(
            data=data, use_gw=True, use_marginal=True
        )

        ll_seq = kf_seq.get_likelihood(self._params())
        ll_marg = kf_marg.get_likelihood(self._params())

        assert ll_marg.shape == ()
        assert jnp.isfinite(ll_marg)
        assert jnp.isclose(
            ll_marg, ll_seq, rtol=1e-6, atol=1e-4
        ), f"marginal {ll_marg} != sequential {ll_seq}"

    @patch("argus.io_manager.get_argus_logger")
    def test_null_gw_matches_sequential(self, mock_logger):
        """Equivalence must also hold with GW terms disabled (use_gw=False)."""
        mock_logger.return_value = Mock()
        data = _realistic_pulsar_data()

        kf_seq = jax_kalman_filter.JaxKalmanFilter(
            data=data, use_gw=False, use_marginal=False
        )
        kf_marg = jax_kalman_filter.JaxKalmanFilter(
            data=data, use_gw=False, use_marginal=True
        )

        ll_seq = kf_seq.get_likelihood(self._params())
        ll_marg = kf_marg.get_likelihood(self._params())

        assert jnp.isclose(
            ll_marg, ll_seq, rtol=1e-6, atol=1e-4
        ), f"marginal {ll_marg} != sequential {ll_seq}"

    @patch("argus.io_manager.get_argus_logger")
    def test_P_eps_inv_is_prior_inverse(self, mock_logger, sample_pulsar_data):
        """P_eps_inv must be the exact inverse of the augmented filter's prior block,
        with per-pulsar blocks in the same order as the epsilon columns of H."""
        mock_logger.return_value = Mock()

        kf = jax_kalman_filter.JaxKalmanFilter(
            data=sample_pulsar_data, use_gw=True, use_marginal=True
        )

        M_sum = kf.M_sum
        assert kf.P_eps_inv.shape == (M_sum, M_sum)
        # P_eps_inv @ P_eps == I  (validates block ordering + inverse relationship)
        identity = kf.P_eps_inv @ kf.P_eps
        assert jnp.allclose(identity, jnp.eye(M_sum), atol=1e-6)


def _batch_gls_diffuse_loglik(data, params, use_gw=True):
    """Independent, non-recursive batch reference for the diffuse-prior log likelihood.

    Builds the joint measurement covariance ``C = cov(z)`` implied by the linear-Gaussian
    state-space model (β = 0), the stacked timing design ``G`` mapping β to the measurement
    mean, and the β = 0 residual ``r = z``, then evaluates the flat-prior marginal likelihood
    in closed form via the standard GLS / G-matrix projection

        logL = -0.5 [ r'C⁻¹r - (G'C⁻¹r)'(G'C⁻¹G)⁻¹(G'C⁻¹r) + logdet(2πC) + logdet(G'C⁻¹G) ].

    This shares the model's F/Q/H/R/P0 builders with the filter but computes the likelihood
    through entirely different (batch, non-recursive) algebra — no Kalman recursion and no
    A/b/c/L accumulation — so it is an independent oracle for the marginalization math. The
    additive-constant convention matches the filter's diffuse output (the (M/2)ln(2π) term of
    the proper improper-prior marginal is omitted in both).
    """
    kf = jax_kalman_filter.JaxKalmanFilter(data=data, use_gw=use_gw, use_marginal=True)
    Npsr, M_sum = kf.Npsr, int(kf.M_sum)
    n_dyn = 4 * Npsr
    T = len(kf.jax_data)

    σa2 = jax_kalman_filter._compute_sigma_matrix(
        params.ha**2, params.γa, kf.hellings_downs_matrix
    )
    _, P0 = jax_kalman_filter._initialize_dynamic_kalman_filter(
        Npsr, σa2, params.γa, params.σp**2, params.γp
    )
    R_all = np.asarray(
        jax_kalman_filter.precompute_R_matrices(
            kf.jax_data_errors, params.EFAC, params.EQUAD
        )
    )
    dt_indices = jnp.arange(T - 1)
    (F_gw, F_spin), (Q_gw, Q_spin) = jax_kalman_filter._precompute_transition_matrices(
        params.γa, params.γp, σa2, params.σp**2, kf.jax_t_diffs[dt_indices], Npsr, M_sum
    )

    # Full dynamic transition / process-noise matrices (block-diagonal in [GW; spin]).
    from scipy.linalg import block_diag as _bd

    F = [_bd(np.asarray(F_gw[k]), np.asarray(F_spin[k])) for k in range(T - 1)]
    Q = [_bd(np.asarray(Q_gw[k]), np.asarray(Q_spin[k])) for k in range(T - 1)]
    P0 = np.asarray(P0)

    H = np.asarray(kf.jax_H_matrices)  # (T, Npsr, nx)
    H_dyn = H[:, :, :n_dyn]
    H_eps = H[:, :, n_dyn:]

    # Marginal state covariances Σ_k = cov(x_k).
    Sigma = [None] * T
    Sigma[0] = P0
    for k in range(1, T):
        Sigma[k] = F[k - 1] @ Sigma[k - 1] @ F[k - 1].T + Q[k - 1]

    # Joint measurement covariance C: cov(z_j, z_k) = H_dyn_j Σ_j Φ(k,j)' H_dyn_k' (+ R_k δ).
    C = np.zeros((T * Npsr, T * Npsr))
    for j in range(T):
        Phi = np.eye(n_dyn)  # Φ(j,j) = I
        for k in range(j, T):
            if k > j:
                Phi = F[k - 1] @ Phi  # Φ(k,j) = F_{k-1} … F_j
            block = H_dyn[j] @ Sigma[j] @ Phi.T @ H_dyn[k].T
            if k == j:
                block = block + R_all[k]
            C[j * Npsr : (j + 1) * Npsr, k * Npsr : (k + 1) * Npsr] = block
            C[k * Npsr : (k + 1) * Npsr, j * Npsr : (j + 1) * Npsr] = block.T

    G = np.vstack([H_eps[k] for k in range(T)])  # (T*Npsr, M_sum)
    r = np.asarray(kf.jax_data).reshape(-1)  # (T*Npsr,)

    Cinv_r = np.linalg.solve(C, r)
    Cinv_G = np.linalg.solve(C, G)
    A = G.T @ Cinv_G  # G' C⁻¹ G
    bvec = G.T @ Cinv_r  # G' C⁻¹ r
    rCr = r @ Cinv_r
    quad = bvec @ np.linalg.solve(A, bvec)
    _, logdet_2piC = np.linalg.slogdet(2.0 * np.pi * C)
    _, logdet_A = np.linalg.slogdet(A)

    return -0.5 * (rCr - quad + logdet_2piC + logdet_A)


class TestDiffuseFilter:
    """Tests for the diffuse (flat/improper) timing-model prior on the marginal filter.

    The diffuse limit P_eps⁻¹ → 0 fully projects the timing-model subspace out of the data
    (the community-standard PTA treatment). It is validated two ways: (1) internal
    consistency — the informative marginal filter must converge to it as the prior scale
    α → ∞ (up to a known additive constant), and (2) an independent, non-recursive batch
    GLS / G-matrix reference computed with numpy.
    """

    def _params(self):
        return bayesian_inference.Parameters(
            log10_gamma_a=-9.0,
            γa=1e-9,
            ha=1e-15,
            γp=jnp.full(2, 1e-8),
            σp=jnp.full(2, 1e-15),
            EFAC=jnp.ones(2),
            EQUAD=jnp.full(2, 1e-6),
        )

    @patch("argus.io_manager.get_argus_logger")
    def test_diffuse_requires_marginal(self, mock_logger):
        """Diffuse prior is only supported on the marginal backend; misuse must raise."""
        mock_logger.return_value = Mock()
        data = _realistic_pulsar_data()

        with pytest.raises(ValueError, match="marginal"):
            jax_kalman_filter.JaxKalmanFilter(
                data=data, use_marginal=False, timing_prior="diffuse"
            )
        with pytest.raises(ValueError, match="timing_prior"):
            jax_kalman_filter.JaxKalmanFilter(data=data, timing_prior="bogus")

    @patch("argus.io_manager.get_argus_logger")
    def test_diffuse_finite_and_shape(self, mock_logger):
        """Diffuse log likelihood must be a finite scalar."""
        mock_logger.return_value = Mock()
        data = _realistic_pulsar_data()

        kf = jax_kalman_filter.JaxKalmanFilter(data=data, timing_prior="diffuse")
        ll = kf.get_likelihood(self._params())
        assert ll.shape == ()
        assert jnp.isfinite(ll)

    @patch("argus.io_manager.get_argus_logger")
    def test_diffuse_matches_large_alpha(self, mock_logger):
        """As α → ∞ the informative marginal filter converges to the diffuse one.

        Exact limit relation (P_eps⁻¹(α) = P_eps⁻¹_base / α):

            diffuse = informative(α) - 0.5·logdet(P_eps⁻¹_base) + 0.5·M_sum·ln(α).
        """
        mock_logger.return_value = Mock()
        data = _realistic_pulsar_data()
        params = self._params()

        kf_diffuse = jax_kalman_filter.JaxKalmanFilter(
            data=data, timing_prior="diffuse"
        )
        kf_base = jax_kalman_filter.JaxKalmanFilter(data=data, prior_scale=1.0)
        ll_diffuse = kf_diffuse.get_likelihood(params)
        _, logdet_Pinv_base = jnp.linalg.slogdet(kf_base.P_eps_inv)
        M_sum = int(kf_diffuse.M_sum)

        def predicted(alpha):
            kf_a = jax_kalman_filter.JaxKalmanFilter(data=data, prior_scale=alpha)
            ll_a = kf_a.get_likelihood(params)
            return ll_a - 0.5 * logdet_Pinv_base + 0.5 * M_sum * jnp.log(alpha)

        err_small = abs(float(predicted(1e3) - ll_diffuse))
        err_large = abs(float(predicted(1e6) - ll_diffuse))

        # Convergence: the α = 1e6 estimate is closer than α = 1e3, and matches tightly.
        assert err_large < err_small
        assert (
            err_large < 1e-3
        ), f"diffuse {ll_diffuse} vs informative(1e6) off by {err_large}"

    @patch("argus.io_manager.get_argus_logger")
    def test_diffuse_matches_batch_gls(self, mock_logger):
        """Diffuse log likelihood must match the independent batch GLS / G-matrix oracle."""
        mock_logger.return_value = Mock()
        data = _realistic_pulsar_data()
        params = self._params()

        kf = jax_kalman_filter.JaxKalmanFilter(data=data, timing_prior="diffuse")
        ll_filter = float(kf.get_likelihood(params))
        ll_batch = float(_batch_gls_diffuse_loglik(data, params, use_gw=True))

        assert np.isclose(
            ll_filter, ll_batch, rtol=1e-5, atol=1e-3
        ), f"filter {ll_filter} != batch GLS {ll_batch}"

    @patch("argus.io_manager.get_argus_logger")
    def test_diffuse_matches_batch_gls_null_gw(self, mock_logger):
        """Batch-GLS agreement must also hold with GW terms disabled."""
        mock_logger.return_value = Mock()
        data = _realistic_pulsar_data()
        params = self._params()

        kf = jax_kalman_filter.JaxKalmanFilter(
            data=data, use_gw=False, timing_prior="diffuse"
        )
        ll_filter = float(kf.get_likelihood(params))
        ll_batch = float(_batch_gls_diffuse_loglik(data, params, use_gw=False))

        assert np.isclose(
            ll_filter, ll_batch, rtol=1e-5, atol=1e-3
        ), f"filter {ll_filter} != batch GLS {ll_batch}"

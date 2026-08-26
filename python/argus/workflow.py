"""Workflow orchestration and high-level functions for the argus package."""

import os

from argus import (
    data_loader,
    jax_kalman_filter,
    cw_kalman_filter,
    bayesian_inference,
    utils,
    prior_models,
)
from argus import io_manager
from argus import checkpointing


def setup_data_and_kalman_filter(config, logger, use_gw, signal_model="gwb"):
    """Load and process data, initialize Kalman filter.

    Args:
        config: Configuration object
        logger: Logger object
        use_gw (bool): Whether to include gravitational wave model (GWB mode only)
        signal_model (str): Signal model mode: 'gwb' or 'cw'

    Returns
    -------
        tuple: (pulsar_data, KF)
    """
    logger.info(f"Loading and processing data (mode={signal_model})...")
    data_path = config.get("Data", "data_path")
    excluded_psrs = config.get("Data", "excluded_psrs").split(",")
    pulsar_data = data_loader.LoadWidebandPulsarData.get_processed_residuals(
        data_path,
        excluded_psrs=[psr.strip() for psr in excluded_psrs if psr.strip()],
        mode=signal_model,
    )

    if signal_model == "cw":
        include_pulsar_term = config.getboolean(
            "CWModel", "include_pulsar_term", fallback=False
        )
        phase_parameterization = config.getboolean(
            "CWModel", "phase_parameterization", fallback=True
        )

        # Override pulsar distances from JSON file if provided
        # Only needed for distance-based pulsar term (not phase parameterization)
        if include_pulsar_term and not phase_parameterization:
            distance_file = config.get("CWModel", "pulsar_distances_path", fallback="")
            if distance_file.strip():
                import json
                import os

                if not os.path.isabs(distance_file):
                    config_dir = os.path.dirname(
                        os.path.abspath(config.get("Data", "data_path"))
                    )
                    distance_file = os.path.join(config_dir, distance_file)
                with open(distance_file) as f:
                    dist_data = json.load(f)
                metadata = pulsar_data["metadata"]
                for idx, row in metadata.iterrows():
                    psr_name = row["name"]
                    if psr_name in dist_data:
                        metadata.at[idx, "distance_kpc"] = dist_data[psr_name][
                            "distance_kpc"
                        ]
                logger.info(f"Loaded pulsar distances from {distance_file}")

        logger.info(
            f"Initializing CW per-pulsar Kalman filter (pulsar_term={include_pulsar_term}, phase_param={phase_parameterization})..."
        )
        KF = cw_kalman_filter.CWKalmanFilter(
            data=pulsar_data,
            include_pulsar_term=include_pulsar_term,
            phase_parameterization=phase_parameterization,
        )
    else:
        timing_prior = config.get(
            "PriorModel", "timing_prior", fallback="informative"
        ).strip()
        prior_scale = config.getfloat("PriorModel", "prior_scale", fallback=1.0)
        logger.info(
            f"Initializing joint GWB Kalman filter "
            f"(timing_prior={timing_prior}, prior_scale={prior_scale})..."
        )
        KF = jax_kalman_filter.JaxKalmanFilter(
            data=pulsar_data,
            use_gw=use_gw,
            timing_prior=timing_prior,
            prior_scale=prior_scale,
        )

    return pulsar_data, KF


def run_inference(config_path, use_gw=True, timestamp=None):
    """
    Run Bayesian inference on pulsar timing data.

    Args:
        config_path (str): Path to configuration file
        use_gw (bool): Whether to include gravitational wave model
        timestamp (str): Optional timestamp to use for output directory

    Returns
    -------
        str: Output directory path
    """
    # Load configuration and resolve any relative paths
    config = utils.load_config(config_path)
    config = utils.resolve_config_paths(config, config_path)

    # Determine signal model mode from config
    signal_model = config.get("Data", "signal_model", fallback="gwb").strip().lower()

    # Get output_id from config
    output_id = io_manager.get_output_id_from_config(config, timestamp)

    # Setup output directory
    output_dir = io_manager.setup_output_directory(
        config, use_gw, timestamp, config_path
    )

    # Setup centralized logging - file logging controlled by config
    logger = io_manager.setup_single_logger(config, output_dir)

    # Copy config file to output directory
    io_manager.copy_config_file(config_path, output_dir, logger)

    # Checkpointing writes alongside the run's other outputs. Injected here rather than
    # required in the config file so an existing config gains checkpointing by setting
    # `enabled` alone, without having to restate paths the harness already knows.
    if not config.has_section("Checkpointing"):
        config.add_section("Checkpointing")
    config.set("Checkpointing", "directory", output_dir)
    config.set("Checkpointing", "output_id", output_id)

    # Setup data and Kalman filter
    pulsar_data, KF = setup_data_and_kalman_filter(
        config, logger, use_gw, signal_model=signal_model
    )

    # Get noise parameters
    efac_array, equad_array, sigma_p_array, gamma_p_array = utils.get_noise_parameters(
        config
    )

    # Get prior model specifications and display them
    n_pulsars = len(pulsar_data["metadata"])
    prior_specs = prior_models.get_prior_model_specs(
        config,
        n_pulsars,
        sigma_p_array,
        gamma_p_array,
        efac_array,
        equad_array,
        mode=signal_model,
    )

    # Display prior summary
    bayesian_inference.display_prior_summary(prior_specs, n_pulsars, logger)

    # Test likelihood performance with known parameters
    logger.info("Performing likelihood performance test...")
    if signal_model == "cw":
        logger.info("Skipping GWB likelihood test in CW mode")
    else:
        bayesian_inference.test_likelihood_performance(KF, config, n_pulsars, logger)

    # Select sampler and run inference
    sampler_method = config.get("Data", "sampler", fallback="nuts").strip().lower()
    log_evidence = None

    if sampler_method in ("nested", "jaxns"):
        logger.info(f"Running jaxns nested sampling (mode={signal_model})...")
        results, log_evidence = bayesian_inference.run_nested_sampling(
            KF,
            config,
            len(pulsar_data["metadata"]),
            sigma_p_array,
            gamma_p_array,
            efac_array,
            equad_array,
            mode=signal_model,
        )
        logger.info(
            f"Bayesian evidence: log_Z = {log_evidence[0]:.2f} +/- {log_evidence[1]:.2f}"
        )
    else:
        logger.info(f"Running NUMPYRO NUTS inference (mode={signal_model})...")
        results = bayesian_inference.run_nuts_sampling(
            KF,
            config,
            len(pulsar_data["metadata"]),
            sigma_p_array,
            gamma_p_array,
            efac_array,
            equad_array,
            mode=signal_model,
        )

    # Save results
    if not checkpointing.is_complete(results):
        logger.warning(
            f"Sampling did not complete ({checkpointing.describe_progress(results)}). "
            f"Results are saved and marked partial; they must not be used for a "
            f"reported result without the usual convergence checks."
        )
    results_path = io_manager.save_numpyro_results(
        results, output_dir, output_id, logger
    )

    # Save evidence if nested sampling was used
    if log_evidence is not None:
        import json

        evidence_path = os.path.join(output_dir, f"{output_id}_evidence.json")
        with open(evidence_path, "w") as f:
            json.dump(
                {"log_Z_mean": log_evidence[0], "log_Z_uncert": log_evidence[1]}, f
            )
        logger.info(f"Evidence saved to {evidence_path}")

    # Create plots and diagnostics
    logger.info("Creating corner plot and diagnostics...")

    # Create corner plot
    try:
        plot_path = utils.corner_plot(results_path, output_dir)
        if plot_path:
            logger.info(f"Corner plot saved to {plot_path}")

    except Exception as e:
        logger.error(f"Error creating corner plot: {e}")

    # Run diagnostics
    try:
        logger.info("Running MCMC diagnostics...")
        utils.diagnostics(results_path, output_dir)
        logger.info("MCMC diagnostics completed")
    except Exception as e:
        logger.error(f"Error running diagnostics: {e}")

    return output_dir

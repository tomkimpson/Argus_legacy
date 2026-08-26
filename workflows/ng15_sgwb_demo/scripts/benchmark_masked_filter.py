#!/usr/bin/env python
"""Benchmark the two Kalman backends on masked (union-grid) data.

Masked data used to fall back to the sequential augmented-state filter, because the
marginalized filter had no masked-update path. That fallback was reached only on real
array data -- precisely the runs where the cost matters -- so the size of what it gave
up is worth measuring rather than assuming.

Reports, at several grid occupancies: the log-likelihood from each backend (they are
mathematically equivalent, so this doubles as an agreement check) and the wall-clock
per likelihood evaluation after compilation.

Run:
    JAX_PLATFORMS=cpu python workflows/ng15_sgwb_demo/scripts/benchmark_masked_filter.py
    python workflows/ng15_sgwb_demo/scripts/benchmark_masked_filter.py --gpu
"""

import argparse
import json
import os
import pickle
import time

import jax.numpy as jnp
import numpy as np


def build_mask(shape, occupancy, seed=0):
    """Random mask at the requested occupancy, every pulsar observed at least once."""
    if occupancy >= 1.0:
        return np.ones(shape)
    rng = np.random.default_rng(seed)
    mask = (rng.random(shape) < occupancy).astype(float)
    for column in range(shape[1]):
        if mask[:, column].sum() == 0:
            mask[0, column] = 1.0
    return mask


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        default=None,
        help="Pickled processed pulsar data (default: the MDC2 test dataset).",
    )
    parser.add_argument(
        "--occupancy",
        type=float,
        nargs="+",
        default=[0.416, 0.6, 1.0],
        help="Grid occupancies to test. 0.416 is the NG15 68-pulsar union grid.",
    )
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--gpu", action="store_true", help="Do not force JAX to CPU.")
    parser.add_argument("--out", default=None, help="Write JSON results here.")
    args = parser.parse_args()

    if not args.gpu:
        os.environ.setdefault("JAX_PLATFORMS", "cpu")

    from argus import bayesian_inference, io_manager
    from argus import jax_kalman_filter as jk
    from argus.utils import get_efac_equad_injections, get_psr_noise_injections

    class MinimalConfig:
        def get(self, section, key, fallback=None):
            return fallback

    io_manager.setup_single_logger(MinimalConfig(), enable_file_logging=False)

    repo_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
    test_data = os.path.join(repo_root, "test", "data")
    data_path = args.data or os.path.join(test_data, "processed_pulsar_data.pkl")

    with open(data_path, "rb") as f:
        pulsar_data = pickle.load(f)
    efac, equad = get_efac_equad_injections(
        os.path.join(test_data, "noise_parameters.json"),
        excluded_psrs=["J1640+2224"],
    )
    sigma_p, gamma_p = get_psr_noise_injections(
        os.path.join(test_data, "spin_injections.pkl"), excluded_psrs=["J1640+2224"]
    )
    gamma_a = 1e-9
    params = bayesian_inference.Parameters(
        γa=gamma_a,
        ha=1e-15,
        log10_gamma_a=jnp.log10(gamma_a),
        γp=gamma_p,
        σp=sigma_p,
        EFAC=efac,
        EQUAD=equad,
    )

    base = pulsar_data["processed_residuals"]
    shape = base["residuals"].shape
    print(f"dataset: {shape[1]} pulsars x {shape[0]} epochs   ({data_path})")
    print(
        f"\n  {'occupancy':>10} {'sequential':>14} {'marginal':>14} {'rel diff':>10} "
        f"{'seq ms':>9} {'marg ms':>9} {'speedup':>8}"
    )

    results = []
    for occupancy in args.occupancy:
        residuals = dict(base)
        residuals["mask"] = build_mask(shape, occupancy)
        data = dict(pulsar_data)
        data["processed_residuals"] = residuals

        measured = {}
        for use_marginal in (False, True):
            kf = jk.JaxKalmanFilter(data=data, use_gw=True, use_marginal=use_marginal)
            value = float(kf.get_likelihood(params))  # compile
            start = time.perf_counter()
            for _ in range(args.repeats):
                float(kf.get_likelihood(params))
            elapsed = (time.perf_counter() - start) / args.repeats
            measured[use_marginal] = (value, elapsed)

        seq_value, seq_time = measured[False]
        marg_value, marg_time = measured[True]
        relative = abs(marg_value - seq_value) / abs(seq_value)
        results.append(
            {
                "occupancy": occupancy,
                "sequential_loglik": seq_value,
                "marginal_loglik": marg_value,
                "relative_difference": relative,
                "sequential_seconds": seq_time,
                "marginal_seconds": marg_time,
                "speedup": seq_time / marg_time,
            }
        )
        print(
            f"  {occupancy:10.3f} {seq_value:14.4f} {marg_value:14.4f} "
            f"{relative:10.2e} {seq_time * 1e3:9.1f} {marg_time * 1e3:9.1f} "
            f"{seq_time / marg_time:7.2f}x"
        )

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"data_path": data_path, "results": results}, f, indent=2)
        print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()

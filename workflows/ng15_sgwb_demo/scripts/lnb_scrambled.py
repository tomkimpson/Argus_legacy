#!/usr/bin/env python
"""Path-sampling Bayes factor for a SKY-SCRAMBLED ladder.

Why this wrapper exists
-----------------------
``lnb_path_sampling.py --evaluate`` does not read the integrand out of the rung files; it
recomputes it, rebuilding the data and the Kalman filter from the config via
``workflow.setup_data_and_kalman_filter``. That reload goes through the stock data loader,
so it picks up the TRUE overlap reduction function -- even when every rung in
``--from-runs`` was sampled under a scrambled one. The result would be a well-formed,
fully "reliable" number computed against the wrong correlation matrix, and nothing in the
pipeline would flag it: the ORF fingerprint in bayesian_inference guards *resume*, not the
estimator.

So the scramble override has to be installed here too. This wrapper does exactly that and
then hands over to the estimator unchanged.

Why a wrapper rather than a flag on the estimator
-------------------------------------------------
The production evidence procedure is FROZEN (openspec sgwb/model-selection: "any change to
it after freezing SHALL invalidate downstream results until they are re-run"). Adding a
scramble flag to lnb_path_sampling.py would put the 3.043 in question. Patching the data
seam from outside leaves that file byte-identical, which is also the right description of
what a null is: the same estimator, the same model, one substituted matrix.

Every argument is passed straight through to lnb_path_sampling.main().

    JAX_PLATFORMS=cpu python scripts/lnb_scrambled.py \
        --scramble-npz data/scrambles/mdc2_d1_scrambles.npz --scramble-index 0 \
        --from-runs 'outputs/mdc2_d1_null_eps*/mdc2_d1_null_eps*_results.nc' \
        --evaluate configs/mdc2_d1_null_eps000.ini \
        --batch-size 100 \
        --out outputs/lnb_path_sampling_d1_null.json

Exit status is the estimator's own: 0 only if it reports reliable.
"""

import argparse
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(HERE)), "python"))


def _load(name):
    """Import a sibling script by path (they are scripts, not package modules)."""
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--scramble-npz", required=True)
    parser.add_argument("--scramble-index", type=int, default=0)
    args, passthrough = parser.parse_known_args()

    scrambles = _load("sky_scrambles")
    info = scrambles.install_scramble_override(args.scramble_npz, args.scramble_index)
    print(
        f"[SCRAMBLE] integrand will be evaluated under scramble {info['index']} of "
        f"{info['n_scrambles']} (match {info['match']:.4f} against the true ORF).\n"
    )

    # The estimator parses sys.argv itself; hand it everything except our two flags.
    sys.argv = [sys.argv[0]] + passthrough
    return _load("lnb_path_sampling").main()


if __name__ == "__main__":
    raise SystemExit(main())

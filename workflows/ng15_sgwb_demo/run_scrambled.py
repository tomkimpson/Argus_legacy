"""Null-calibration driver — run the production pipeline with the Hellings-Downs
inter-pulsar correlation replaced by a SKY-SCRAMBLED one.

A Bayes factor is a number, not a significance, and lnB(HD/CURN) = 3.043 on MDC2 1b does
not by itself say whether the evidence came from the correlation pattern or from flat
red-noise priors letting the GW claim any common power going. Scrambling the sky destroys
the pattern and leaves everything else -- residuals, epochs, per-pulsar noise, priors,
NUTS settings -- untouched, so a ladder run this way answers that question directly.

The ONLY difference from run_analysis.py is a runtime override, applied with NO library
edit (per repo convention, and following run_curn.py, which does the same thing with the
identity matrix): the data loader's ``hd_correlation`` is replaced by an accepted scramble
read from the .npz that sky_scrambles.py writes. The ORF reaches the filter through the
data dict rather than through any config key, so there is nothing to set in the .ini.

    python run_scrambled.py configs/mdc2_d1_null_eps100.ini \
        --scramble-npz data/scrambles/mdc2_d1_scrambles.npz --scramble-index 0

The SAME override must be installed when the Bayes factor is computed -- see
scripts/lnb_scrambled.py. lnb_path_sampling.py rebuilds the data from the config and would
otherwise evaluate the integrand against the true ORF, giving a plausible but meaningless
number that nothing downstream would catch.
"""

import os
import sys
import argparse
import importlib.util
from datetime import datetime

import jax

project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.append(os.path.join(project_root, "python"))

from argus import utils, workflow  # noqa: E402

jax.config.update("jax_enable_x64", True)


def load_sky_scrambles():
    """Import scripts/sky_scrambles.py by path (it is a script, not a package module)."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "scripts", "sky_scrambles.py")
    spec = importlib.util.spec_from_file_location("sky_scrambles", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description="Run inference under a scrambled ORF.")
    parser.add_argument("config", type=str, help="Path to the configuration file")
    parser.add_argument("--scramble-npz", required=True, help="Archive from sky_scrambles.py")
    parser.add_argument("--scramble-index", type=int, default=0)
    args = parser.parse_args()

    print("=== NG15 SGWB WORKFLOW (NULL — sky-scrambled ORF) ===")
    print(f"JAX version: {jax.__version__}")
    print("Default device:", jax.default_backend())
    print("Note: HD correlation replaced by a scrambled-sky ORF => the null.\n")

    utils.check_gpu_availability()

    scrambles = load_sky_scrambles()
    info = scrambles.install_scramble_override(args.scramble_npz, args.scramble_index)
    print(
        f"[SCRAMBLE] installed index {info['index']} of {info['n_scrambles']}, "
        f"match {info['match']:.4f} against the true ORF."
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = workflow.run_inference(
        config_path=args.config, use_gw=True, timestamp=timestamp
    )
    print(f"\nNull inference complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()

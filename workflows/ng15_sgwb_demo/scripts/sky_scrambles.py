#!/usr/bin/env python
"""Sky scrambles: an empirical null distribution for the SGWB detection statistic.

Why this exists
---------------
A Bayes factor on its own is a number, not a significance. To say how often a value
that large arises with no inter-pulsar correlation present, the statistic has to be
recomputed on realisations where that correlation has been destroyed but everything
else -- each pulsar's noise, its observing epochs, its residuals -- is untouched.

Sky scrambles do exactly that: draw random sky positions for the pulsars, rebuild the
overlap reduction function from them, and analyse the same data under the wrong
correlation pattern. In a time-domain state-space likelihood the ORF is just an input
matrix (``data['hd_correlation']``), so a scramble enters at the same seam the CURN run
already uses -- no change to the model, the priors, or the data.

The alternative the field also uses, phase shifts, randomises the phases of Fourier
coefficients. There is no clean analogue of that here, and inventing one would need its
own validation, so scrambles are the natural choice for this pipeline.

Match threshold
---------------
A random sky draw can land close to the true geometry by chance, and such a realisation
does not test the null -- it re-tests Hellings-Downs. Scrambles are therefore accepted
only when their normalised overlap with the true ORF,

    match = |sum_{i<j} G_scr(ij) G_HD(ij)| / sqrt( sum_{i<j} G_scr(ij)^2 * sum_{i<j} G_HD(ij)^2 )

is below a threshold (0.2 by convention). Self-pairs are excluded: the diagonal is 1 for
every ORF, carries no correlation information, and would otherwise dominate the match
and make every scramble look similar to the truth.

Run:
    python workflows/ng15_sgwb_demo/scripts/sky_scrambles.py \
        --data-path <aligned feather dir> --n-scrambles 100 \
        --out outputs/scrambles/scrambles.npz
"""

import argparse
import json
import os

import numpy as np

DEFAULT_MATCH_THRESHOLD = 0.2
DEFAULT_MAX_ATTEMPTS_PER_SCRAMBLE = 500


def orf_match(first, second):
    """Normalised overlap between two ORF matrices, over distinct pairs only.

    Returns a value in [0, 1]: 1 for identical correlation patterns (up to scale) and 0
    for orthogonal ones.
    """
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    if first.shape != second.shape:
        raise ValueError(f"shape mismatch: {first.shape} vs {second.shape}")

    upper = np.triu_indices(first.shape[0], k=1)
    a = first[upper]
    b = second[upper]
    denominator = np.sqrt(np.sum(a**2) * np.sum(b**2))
    if denominator == 0.0:
        return 0.0
    return float(abs(np.sum(a * b)) / denominator)


def random_sky_positions(n_pulsars, rng):
    """Uniform positions on the sphere.

    Declination is drawn through sin(dec) rather than uniformly in the angle; drawing
    the angle directly would over-populate the poles and bias the scramble geometry.
    """
    right_ascension = rng.uniform(0.0, 2.0 * np.pi, size=n_pulsars)
    declination = np.arcsin(rng.uniform(-1.0, 1.0, size=n_pulsars))
    return right_ascension, declination


def build_orf(right_ascension, declination):
    """The Hellings-Downs matrix implied by a set of sky positions."""
    from argus import gravitational_waves

    separations = gravitational_waves.pairwise_angular_separation(
        np.asarray(right_ascension, dtype=float),
        np.asarray(declination, dtype=float),
    )
    return gravitational_waves.hellings_downs(separations)


def generate_scrambles(
    true_orf,
    n_scrambles,
    seed=0,
    match_threshold=DEFAULT_MATCH_THRESHOLD,
    max_attempts_per_scramble=DEFAULT_MAX_ATTEMPTS_PER_SCRAMBLE,
):
    """Generate accepted scrambled ORFs and their matches against the true one.

    Returns
    -------
    dict with ``orfs`` (n_scrambles, Npsr, Npsr), ``matches`` (n_scrambles,),
    ``n_attempts`` and ``acceptance_rate``.
    """
    true_orf = np.asarray(true_orf, dtype=float)
    n_pulsars = true_orf.shape[0]
    rng = np.random.default_rng(seed)

    orfs = []
    matches = []
    attempts = 0
    budget = n_scrambles * max_attempts_per_scramble

    while len(orfs) < n_scrambles:
        if attempts >= budget:
            raise RuntimeError(
                f"Only {len(orfs)}/{n_scrambles} scrambles accepted in {attempts} "
                f"attempts at match threshold {match_threshold}. Either raise the "
                f"threshold or accept a smaller ensemble -- do not silently return "
                f"fewer than requested."
            )
        attempts += 1
        right_ascension, declination = random_sky_positions(n_pulsars, rng)
        candidate = build_orf(right_ascension, declination)
        match = orf_match(candidate, true_orf)
        if match < match_threshold:
            orfs.append(candidate)
            matches.append(match)

    return {
        "orfs": np.array(orfs),
        "matches": np.array(matches),
        "n_attempts": attempts,
        "acceptance_rate": len(orfs) / attempts,
        "match_threshold": match_threshold,
        "seed": seed,
    }


def load_true_orf(data_path, excluded_psrs=()):
    """The real array's ORF, from the same loader the runs use."""
    from argus import data_loader

    data = data_loader.LoadWidebandPulsarData.get_processed_residuals(
        data_path,
        excluded_psrs=[p.strip() for p in excluded_psrs if p.strip()],
        mode="gwb",
    )
    return np.asarray(data["hd_correlation"], dtype=float)


def apply_scramble(data, scrambled_orf):
    """Return a data dict with the ORF replaced and everything else shared.

    The same intervention ``run_curn.py`` makes for the identity ORF. Returning a new
    dict rather than mutating in place keeps the caller's data usable, so a harness can
    loop over scrambles without reloading.
    """
    scrambled = dict(data)
    scrambled["hd_correlation"] = np.asarray(scrambled_orf, dtype=float)
    return scrambled


def install_scramble_override(
    npz_path, index, match_threshold=DEFAULT_MATCH_THRESHOLD
):
    """Patch the data loader so this process analyses a scrambled ORF.

    The scramble has to reach BOTH the sampler and the path-sampling integrand
    evaluation, and those build their data independently -- ``workflow.run_inference``
    for the first, ``workflow.setup_data_and_kalman_filter`` for the second. Both go
    through ``get_processed_residuals``, so patching that one seam covers both. Patching
    only the run driver would leave the Bayes factor computed against the TRUE ORF: a
    plausible-looking, meaningless number that nothing else in the pipeline would catch,
    since the ORF fingerprint guards resume rather than the estimator.

    This is the intervention ``run_curn.py`` makes for the identity, done here for the
    same reason it is done there rather than in the library: the ORF reaches the filter
    through the data dict, not through any config key, so there is nothing to set.

    The match is RECOMPUTED from the stored ``true_orf`` rather than read from the
    ``matches`` array, and the run is refused if it fails the threshold. Trusting the
    stored value would let a mis-indexed or hand-edited archive through; recomputing
    means an unaccepted scramble cannot be analysed silently.
    """
    from argus import data_loader

    with np.load(npz_path) as archive:
        orfs = archive["orfs"]
        true_orf = archive["true_orf"]
        stored_matches = archive["matches"]

    n_scrambles = int(orfs.shape[0])
    if not 0 <= index < n_scrambles:
        raise IndexError(
            f"scramble index {index} out of range: {npz_path} holds {n_scrambles}."
        )

    scrambled_orf = np.asarray(orfs[index], dtype=float)
    match = orf_match(scrambled_orf, true_orf)
    if match >= match_threshold:
        raise ValueError(
            f"scramble {index} has match {match:.4f} against the true ORF, at or above "
            f"the {match_threshold} threshold: it re-tests Hellings-Downs rather than "
            f"the null. Regenerate the ensemble or pick another index."
        )

    _orig = data_loader.LoadWidebandPulsarData.get_processed_residuals

    def _patched(directory, excluded_psrs=[], mode="gwb"):
        data = _orig(directory, excluded_psrs=excluded_psrs, mode=mode)
        hd = data.get("hd_correlation")
        if hd is None:
            raise RuntimeError("[SCRAMBLE] hd_correlation is None; expected gwb mode.")
        if np.asarray(hd).shape != scrambled_orf.shape:
            raise RuntimeError(
                f"[SCRAMBLE] ORF shape mismatch: data {np.asarray(hd).shape} vs "
                f"scramble {scrambled_orf.shape}. Wrong dataset for this archive?"
            )
        print(
            f"[SCRAMBLE] Overrode hd_correlation with scramble {index} of {n_scrambles} "
            f"from {npz_path} ({scrambled_orf.shape[0]}x{scrambled_orf.shape[0]}, "
            f"recomputed match {match:.4f}, stored {stored_matches[index]:.4f}) — "
            f"the correlation pattern is destroyed, the data is untouched."
        )
        return apply_scramble(data, scrambled_orf)

    data_loader.LoadWidebandPulsarData.get_processed_residuals = staticmethod(_patched)
    return {"index": index, "match": match, "n_scrambles": n_scrambles}


def summarise(result, true_orf):
    """Report the ensemble's match distribution and acceptance."""
    matches = result["matches"]
    return {
        "n_scrambles": int(matches.size),
        "match_threshold": result["match_threshold"],
        "match_mean": float(matches.mean()),
        "match_median": float(np.median(matches)),
        "match_max": float(matches.max()),
        "self_match": orf_match(true_orf, true_orf),
        "n_attempts": int(result["n_attempts"]),
        "acceptance_rate": float(result["acceptance_rate"]),
        "seed": int(result["seed"]),
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-path", required=True, help="Aligned feather directory.")
    parser.add_argument("--excluded-psrs", default="", help="Comma-separated names.")
    parser.add_argument("--n-scrambles", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--match-threshold", type=float, default=DEFAULT_MATCH_THRESHOLD
    )
    parser.add_argument("--out", required=True, help="Write the ORFs here (.npz).")
    args = parser.parse_args()

    true_orf = load_true_orf(args.data_path, args.excluded_psrs.split(","))
    print(f"true ORF: {true_orf.shape[0]} pulsars")

    result = generate_scrambles(
        true_orf,
        args.n_scrambles,
        seed=args.seed,
        match_threshold=args.match_threshold,
    )
    report = summarise(result, true_orf)

    print(
        f"\n  accepted {report['n_scrambles']} of {report['n_attempts']} draws "
        f"({report['acceptance_rate'] * 100:.1f}%)"
    )
    print(
        f"  match to the true ORF: mean {report['match_mean']:.4f}, "
        f"median {report['match_median']:.4f}, max {report['match_max']:.4f} "
        f"(threshold {report['match_threshold']})"
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    np.savez_compressed(
        args.out,
        orfs=result["orfs"],
        matches=result["matches"],
        true_orf=true_orf,
    )
    with open(os.path.splitext(args.out)[0] + "_summary.json", "w") as handle:
        json.dump(report, handle, indent=2)
    print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()

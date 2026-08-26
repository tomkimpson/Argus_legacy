"""Checkpointing for long NUTS runs.

Argus writes results only after sampling completes, so a run that hits its walltime,
is preempted, or crashes loses everything. At 68 pulsars an array run takes most of a
day, and OzSTAR's GPU partition has had both backlogs and outages, so "lose the whole
run" is a routine outcome rather than a rare one.

Sampling is therefore executed as a sequence of segments. NumPyro exposes the sampler
state between runs (``post_warmup_state``), so a segmented run continues from exactly
where the previous one stopped rather than restarting: warmup is paid once, and each
segment's draws are appended to the accumulated posterior on disk.

What is written after each segment:

* ``<id>_checkpoint.pkl``  -- the sampler state plus the run fingerprint and progress.
* ``<id>_partial.nc``      -- the posterior accumulated so far, marked incomplete.

A resume refuses on any fingerprint mismatch. Silently mixing draws from two different
models or samplers would produce a posterior that is not a posterior for anything, and
nothing downstream could detect it.
"""

import hashlib
import json
import os
import pickle

import arviz as az
import numpy as np

CHECKPOINT_SUFFIX = "_checkpoint.pkl"
PARTIAL_SUFFIX = "_partial.nc"

# Recorded on the InferenceData so a partial run can never be mistaken for a whole one.
COMPLETE_ATTR = "argus_sampling_complete"
DRAWS_ATTR = "argus_draws_completed"
TARGET_ATTR = "argus_draws_target"


def get_checkpoint_settings(config):
    """Read the ``[Checkpointing]`` section.

    Returns
    -------
    dict with ``enabled``, ``interval`` (draws per segment) and ``resume``.
    Absent section means disabled, which reproduces the previous single-shot behaviour
    exactly.
    """
    enabled = config.getboolean("Checkpointing", "enabled", fallback=False)
    interval = config.getint("Checkpointing", "interval", fallback=0)
    resume = config.getboolean("Checkpointing", "resume", fallback=False)

    if enabled and interval <= 0:
        raise ValueError(
            "Checkpointing is enabled but 'interval' is not a positive number of "
            "draws per segment."
        )
    return {"enabled": enabled, "interval": interval, "resume": resume}


def run_fingerprint(spec):
    """Stable hash of everything a resume must not differ in.

    Anything that changes the target distribution or the sampler belongs here. The
    fingerprint is deliberately coarse-grained and value-based rather than a hash of
    the config file: reformatting a config or changing an output path must not
    invalidate a checkpoint, but changing a prior bound must.
    """
    payload = json.dumps(spec, sort_keys=True, default=str).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def checkpoint_paths(output_dir, output_id):
    """The two files a checkpoint consists of."""
    return (
        os.path.join(output_dir, f"{output_id}{CHECKPOINT_SUFFIX}"),
        os.path.join(output_dir, f"{output_id}{PARTIAL_SUFFIX}"),
    )


def save_checkpoint(
    output_dir, output_id, sampler_state, inference_data, fingerprint, progress
):
    """Persist the sampler state and the posterior accumulated so far.

    Written atomically (temporary file then rename) so a checkpoint interrupted while
    writing cannot be loaded as a valid one — the failure mode this whole module exists
    to avoid should not be reintroduced by the module itself.
    """
    state_path, partial_path = checkpoint_paths(output_dir, output_id)

    inference_data = mark_partial(
        inference_data,
        complete=progress["draws_completed"] >= progress["draws_target"],
        draws_completed=progress["draws_completed"],
        draws_target=progress["draws_target"],
    )

    temporary = partial_path + ".tmp"
    inference_data.to_netcdf(temporary)
    os.replace(temporary, partial_path)

    temporary = state_path + ".tmp"
    with open(temporary, "wb") as handle:
        pickle.dump(
            {
                "sampler_state": sampler_state,
                "fingerprint": fingerprint,
                "progress": progress,
            },
            handle,
        )
    os.replace(temporary, state_path)

    return state_path, partial_path


def load_checkpoint(output_dir, output_id, fingerprint):
    """Load a checkpoint, refusing one written under a different configuration.

    Returns None when no checkpoint exists, so a first run and a resume can share a
    code path.
    """
    state_path, partial_path = checkpoint_paths(output_dir, output_id)
    if not os.path.exists(state_path):
        return None

    with open(state_path, "rb") as handle:
        payload = pickle.load(handle)

    stored = payload.get("fingerprint")
    if stored != fingerprint:
        raise ValueError(
            f"Refusing to resume {state_path}: it was written under run fingerprint "
            f"{stored!r} but this run is {fingerprint!r}. The model, priors or sampler "
            f"settings have changed, so the stored draws are not draws from this "
            f"posterior. Delete the checkpoint to start over, or restore the "
            f"configuration it was written with."
        )

    if not os.path.exists(partial_path):
        raise ValueError(
            f"Checkpoint state {state_path} exists but its partial posterior "
            f"{partial_path} is missing; the checkpoint is incomplete."
        )

    payload["inference_data"] = az.from_netcdf(partial_path)
    return payload


def mark_partial(inference_data, complete, draws_completed, draws_target):
    """Record on the InferenceData whether sampling finished.

    Output from an interrupted run must be distinguishable from output of a completed
    one, so that it cannot be picked up and reported as a result without the
    convergence checks that a completed run gets.
    """
    # Stored as an int, not a bool: NetCDF has no boolean attribute type and h5netcdf
    # refuses one outright, so a bool here would make the partial file unwritable.
    inference_data.attrs[COMPLETE_ATTR] = int(bool(complete))
    inference_data.attrs[DRAWS_ATTR] = int(draws_completed)
    inference_data.attrs[TARGET_ATTR] = int(draws_target)
    return inference_data


def is_complete(inference_data):
    """True unless the result is explicitly marked as a partial run.

    Only a marker this module wrote can mean "partial". Results predating
    checkpointing carry none and are complete by construction, and anything whose
    marker is not an integer did not come from here — so both fall back to complete
    rather than raising on a value this module never wrote.
    """
    attributes = getattr(inference_data, "attrs", None)
    if not isinstance(attributes, dict):
        return True
    try:
        return bool(int(attributes.get(COMPLETE_ATTR, 1)))
    except (TypeError, ValueError):
        return True


def describe_progress(inference_data):
    """Human-readable progress line for a possibly-partial result."""
    if is_complete(inference_data):
        return "sampling complete"
    completed = inference_data.attrs.get(DRAWS_ATTR, "?")
    target = inference_data.attrs.get(TARGET_ATTR, "?")
    return f"PARTIAL: {completed}/{target} draws per chain"


def segment_sizes(total_draws, interval):
    """Split the requested draws into segments of at most ``interval``.

    The last segment absorbs the remainder, so the total is exactly the requested
    number of draws however the interval divides it.
    """
    if interval <= 0 or interval >= total_draws:
        return [total_draws]
    sizes = [interval] * (total_draws // interval)
    remainder = total_draws % interval
    if remainder:
        sizes.append(remainder)
    return sizes


def concat_segments(segments):
    """Concatenate per-segment InferenceData objects along the draw dimension."""
    if len(segments) == 1:
        return segments[0]
    return az.concat(*segments, dim="draw")


def combined_draw_count(inference_data):
    """Draws per chain currently held."""
    return int(inference_data.posterior.sizes["draw"])


def numpy_state(sampler_state):
    """Convert a sampler state's arrays to numpy so it pickles without JAX.

    A checkpoint should be loadable by a process that has not initialised the same JAX
    backend (a different node, or CPU instead of GPU), so device arrays are brought
    back to the host before writing.
    """
    import jax

    return jax.tree_util.tree_map(
        lambda leaf: np.asarray(leaf) if hasattr(leaf, "shape") else leaf,
        sampler_state,
    )

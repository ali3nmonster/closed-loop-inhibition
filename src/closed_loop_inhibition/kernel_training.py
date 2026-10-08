"""Independent demonstration replication and prospective suppressive discovery.

The training algorithm is the inherited, unchanged fixed-budget imitation
routine. Each new model receives a separate demonstration directory and pair of
nonoverlapping random-stream ranges. No confirmation observation enters these
functions.
"""

from copy import deepcopy
from pathlib import Path

import numpy as np

from .collective_dynamics import _digest
from .collective_suppression import (
    _commands, bank_metadata, branches, response_effect, scale_branches,
)
from .gain_rescue import _run
from .timescale_diagnostics import _fit, _raw
from .timescale_maps import train_cell


def training_config(base, protocol, noise_tau, seed):
    """Preserve training choices, changing only independently assigned tapes."""
    config = deepcopy(base)
    index = protocol["fresh_seeds"].index(seed) * len(base["noise_taus"]) + base["noise_taus"].index(noise_tau)
    streams = protocol["training_streams"]
    for name in ("train", "validation"):
        config["training"][f"{name}_seed"] = streams[f"{name}_base"] + index * streams["stride"]
    if protocol.get("development", False):
        overrides = protocol.get("development_training", {})
        if set(overrides) - {"epochs", "intermediate_epoch"}:
            raise ValueError("Development may shorten epochs only")
        config["training"].update(overrides)
    return config


def train_replica(base, protocol, tau, noise_tau, seed, artifacts):
    """Retain failures explicitly; successful models retain every checkpoint."""
    config = training_config(base, protocol, noise_tau, seed)
    try:
        _, summary, history = train_cell(config, tau, noise_tau, seed, Path(artifacts))
    except (RuntimeError, FloatingPointError) as error:
        return {"tau": tau, "noise_tau": noise_tau, "seed": seed,
                "status": "training_failed", "failure": str(error),
                "training_config": config, "training_config_sha256": _digest(config),
                "physical_rollouts": None, "training_rollouts_expected":
                config["training"]["episodes"] + config["training"]["validation_episodes"]}
    return {"tau": tau, "noise_tau": noise_tau, "seed": seed, "status": "trained",
            "training_config": config, "training_config_sha256": _digest(config),
            "summary": summary, "history": history,
            "physical_rollouts": config["training"]["episodes"] + config["training"]["validation_episodes"]}


def discover_selected(base, protocol, tau, noise_tau, seed, model, bank):
    """Apply the inherited ten-branch rule to a new selected checkpoint."""
    if set(np.unique(bank["noise_seeds"])) != set(protocol["discovery_seeds"]):
        raise ValueError("Discovery bank must use its declared streams")
    native = _commands(model, bank, base)
    rows = [{"branch": branch, "weak": response_effect(
        native, _commands(model, bank, base, {branch: protocol["weak_scale"]}), bank, protocol)}
        for branch in branches(model)]
    classifiable = all(row["weak"]["classifiable"] for row in rows)
    selected = [row["branch"] for row in rows if row["weak"]["eligible"]] if classifiable else None
    return {"tau": tau, "noise_tau": noise_tau, "seed": seed,
            "branch_ids": branches(model), "bank_metadata": bank_metadata(bank),
            "selected": selected, "group": selected or [], "classifiable": classifiable,
            "status": "unclassifiable" if selected is None else "empty_group" if not selected else "selected",
            "branches": rows, "physical_rollouts": 0}


def prepare_fresh_settings(base, protocol, tau, noise_tau, seed, model, discovery):
    """Fit the inherited mean-restoring weak offset on separate 50 ms tapes."""
    if (discovery["tau"], discovery["noise_tau"], discovery["seed"]) != (tau, noise_tau, seed):
        raise ValueError("Discovery belongs to another model")
    group = discovery["group"]
    config = deepcopy(base)
    config["delay"] = protocol["calibration_delay"]
    stream_protocol = deepcopy(protocol)
    stream_protocol["calibration_seeds"] = protocol["offset_calibration_seeds"]
    native = {"branch_scales": {}, "center": 0., "gain": 1., "offset": 0.}
    evaluated, bank, error = _run(config, stream_protocol, tau, noise_tau, model,
                                  {"native": native}, protocol["offset_calibration_seeds"])
    result = {"tau": tau, "noise_tau": noise_tau, "seed": seed, "group": group,
              "discovery_status": discovery["status"], "selected": discovery["selected"],
              "base_config_sha256": _digest(base), "protocol_sha256": _digest(protocol),
              "variants": {"native": native}, "status": "prepared",
              "calibration": evaluated, "physical_rollouts": evaluated["physical_rollouts"]}
    if error is not None:
        result.update(status="offset_calibration_failed", failure=error)
        return result
    scales = {branch: protocol["weak_scale"] for branch in group}
    native_raw = _raw(model, bank, config)
    weak_raw = _raw(scale_branches(model, scales), bank, config)
    fit = _fit(native_raw, weak_raw, bank, config)
    result["variants"]["joint_weak"] = ({"branch_scales": scales, **fit, "identity_group": False}
        if group else {**deepcopy(native), "identity_group": True, "offset_fit_diagnostic": fit})
    return result

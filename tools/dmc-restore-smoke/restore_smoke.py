#!/usr/bin/env python3
"""Pinned dm_control/MuJoCo complete-state and fresh-process replay smoke.

This is a research/test adapter, not production Python in Symthaea. It writes a
small NPZ checkpoint in one process and verifies it in a second process.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
from typing import Any

import mujoco
import numpy as np
from dm_control import suite

DM_CONTROL_COMMIT = "d7e52e3602a555df6cc20cc928ea5b41e9920158"
DM_CONTROL_VERSION = "1.0.52"
MUJOCO_VERSION = "4.0.1"
EXPECTED_INTEGRATION_MASK = 32767
SEED = 20261124
PREFIX_STEPS = 12
N_SUB_STEPS = 2
TIME_LIMIT_SECONDS = 60.0
LEGACY_STEP = True
PREFIX_TOLERANCE = 1e-9


def digest_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def digest_array(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    dtype = array.dtype.newbyteorder("<")
    normalized = array.astype(dtype, copy=False)
    header = json.dumps(
        {"dtype": dtype.str, "shape": list(normalized.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return digest_bytes(header + normalized.tobytes(order="C"))


def digest_observation(observation: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key in sorted(observation):
        array = np.ascontiguousarray(observation[key])
        digest.update(key.encode("utf-8"))
        digest.update(digest_array(array).encode("ascii"))
    return digest.hexdigest()


def task_rng_tuple(task: Any) -> tuple[str, np.ndarray, int, int, float]:
    state = task.random.get_state()
    return state[0], np.asarray(state[1], dtype=np.uint32), int(state[2]), int(state[3]), float(state[4])


def digest_task_rng(state: tuple[str, np.ndarray, int, int, float]) -> str:
    name, keys, position, has_gauss, cached_gaussian = state
    payload = (
        name.encode("utf-8")
        + np.asarray(keys, dtype="<u4").tobytes()
        + int(position).to_bytes(8, "little", signed=False)
        + int(has_gauss).to_bytes(1, "little", signed=False)
        + np.asarray([cached_gaussian], dtype="<f8").tobytes()
    )
    return digest_bytes(payload)


TASKS = (
    "cartpole.swingup",
    "cartpole.two_poles",
    "cartpole.three_poles",
    "walker.walk",
)


def make_environment(task_name: str):
    if task_name not in TASKS:
        raise ValueError(f"unsupported DMC smoke task: {task_name}")
    domain_name, suite_task_name = task_name.split(".", maxsplit=1)
    random_state = np.random.RandomState(SEED)
    environment_kwargs = {
        "legacy_step": LEGACY_STEP,
        "flat_observation": False,
    }
    # Cartpole tasks accept an explicit action-repeat override. walker.walk
    # supplies its own control_timestep and rejects a simultaneous n_sub_steps
    # override, so preserve the pinned task's native walker timing.
    if domain_name == "cartpole":
        environment_kwargs["n_sub_steps"] = N_SUB_STEPS

    env = suite.load(
        domain_name=domain_name,
        task_name=suite_task_name,
        task_kwargs={
            "random": random_state,
            "time_limit": TIME_LIMIT_SECONDS,
        },
        environment_kwargs=environment_kwargs,
    )
    return env


def midpoint_action(env) -> np.ndarray:
    spec = env.action_spec()
    minimum = np.asarray(spec.minimum, dtype=spec.dtype)
    maximum = np.asarray(spec.maximum, dtype=spec.dtype)
    return (minimum + np.asarray(0.65, dtype=spec.dtype) * (maximum - minimum)).astype(spec.dtype)


def compare_observations(actual, expected, tolerance: float) -> float:
    if set(actual) != set(expected):
        raise AssertionError(f"observation keys differ: {set(actual)} != {set(expected)}")
    maximum_error = 0.0
    for key in sorted(actual):
        actual_array = np.asarray(actual[key])
        expected_array = np.asarray(expected[key])
        if actual_array.shape != expected_array.shape:
            raise AssertionError(f"observation shape differs for {key}")
        error = float(np.max(np.abs(actual_array - expected_array))) if actual_array.size else 0.0
        maximum_error = max(maximum_error, error)
        if error > tolerance:
            raise AssertionError(f"observation {key} prefix error {error} exceeds {tolerance}")
    return maximum_error


def verify_legacy_state_is_incomplete(env) -> dict[str, Any]:
    physics = env.physics
    signature = int(mujoco.mjtState.mjSTATE_INTEGRATION.value)
    if signature != EXPECTED_INTEGRATION_MASK:
        raise AssertionError(f"unexpected mjSTATE_INTEGRATION mask {signature}")

    before = physics.get_state(sig=signature).copy()
    compact = physics.get_state()

    physics.data.time += 0.125
    if physics.data.ctrl.size:
        physics.data.ctrl[:] = physics.data.ctrl + 0.25
    if physics.data.qacc_warmstart.size:
        physics.data.qacc_warmstart[:] = physics.data.qacc_warmstart + 0.5

    physics.set_state(compact)
    after = physics.get_state(sig=signature)
    if np.array_equal(before, after):
        raise AssertionError("legacy compact get_state unexpectedly restored full integration state")

    # Restore the full state before the caller continues.
    physics.set_state(before, sig=signature)
    return {
        "legacy_state_size": int(np.asarray(compact).size),
        "integration_state_size": int(before.size),
        "legacy_compact_restore_changed_full_state": True,
        "integration_signature": signature,
    }


def verify_physics_copy_is_independent(env) -> dict[str, Any]:
    physics = env.physics
    copied = physics.copy(share_model=False)
    if np.shares_memory(physics.data.qpos, copied.data.qpos):
        raise AssertionError("Physics.copy unexpectedly aliases qpos")
    original = physics.data.qpos.copy()
    copied.data.qpos[:] = copied.data.qpos + 0.25
    if not np.array_equal(physics.data.qpos, original):
        raise AssertionError("mutating copied MjData changed original MjData")
    return {"physics_copy_data_independent": True, "physics_copy_share_model": False}


def export_checkpoint(path: Path, task_name: str) -> None:
    env = make_environment(task_name)
    env.reset()
    action = midpoint_action(env)

    for _ in range(PREFIX_STEPS):
        env.step(action)

    physics = env.physics
    signature = int(mujoco.mjtState.mjSTATE_INTEGRATION.value)
    if signature != EXPECTED_INTEGRATION_MASK:
        raise AssertionError(
            f"MuJoCo {MUJOCO_VERSION} mjSTATE_INTEGRATION={signature}, expected {EXPECTED_INTEGRATION_MASK}"
        )

    full_state = physics.get_state(sig=signature).copy()
    expected_state_size = int(mujoco.mj_stateSize(physics.model.ptr, signature))
    if full_state.size != expected_state_size:
        raise AssertionError(
            f"integration state size {full_state.size} != mj_stateSize {expected_state_size}"
        )
    prefix_observation = env.task.get_observation(physics)
    prefix_observation = {key: np.asarray(value).copy() for key, value in prefix_observation.items()}
    # Capture task RNG after the prefix observation so any observation-time
    # randomness is part of the frozen common branch-point state.
    task_rng = task_rng_tuple(env.task)

    # Apply the same next action in the source process and retain the expected
    # result. A second process must reconstruct the checkpoint and obtain this.
    next_action = action.copy()
    expected_step = env.step(next_action)
    expected_state = physics.get_state(sig=signature).copy()
    expected_observation = {
        key: np.asarray(value).copy()
        for key, value in expected_step.observation.items()
    }
    expected_rng = task_rng_tuple(env.task)

    def observation_payload(prefix: str, observation: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        keys = sorted(observation)
        values = {f"{prefix}_keys": np.asarray(keys, dtype=np.str_)}
        values.update(
            {
                f"{prefix}_array_{index}": np.asarray(observation[key]).copy()
                for index, key in enumerate(keys)
            }
        )
        return values

    payload = {
        "task_id": np.asarray(task_name),
        "compiled_model_sha256": np.asarray(digest_bytes(physics.model.to_bytes())),
        "integration_state": full_state,
        "integration_state_size": np.asarray(expected_state_size, dtype=np.int64),
        "prefix_state_sha256": np.asarray(digest_array(full_state)),
        "task_rng_name": np.asarray(task_rng[0]),
        "task_rng_keys": task_rng[1],
        "task_rng_position": np.asarray(task_rng[2], dtype=np.int64),
        "task_rng_has_gauss": np.asarray(task_rng[3], dtype=np.int64),
        "task_rng_cached_gaussian": np.asarray(task_rng[4], dtype=np.float64),
        "wrapper_step_count": np.asarray(PREFIX_STEPS, dtype=np.int64),
        "wrapper_reset_next_step": np.asarray(False, dtype=np.bool_),
        "wrapper_n_sub_steps": np.asarray(env._n_sub_steps, dtype=np.int64),
        "wrapper_step_limit": np.asarray(env._step_limit, dtype=np.float64),
        "wrapper_legacy_step": np.asarray(env.physics.legacy_step, dtype=np.bool_),
        "control_timestep": np.asarray(env.control_timestep(), dtype=np.float64),
        "prefix_observation_sha256": np.asarray(digest_observation(prefix_observation)),
        "task_rng_sha256": np.asarray(digest_task_rng(task_rng)),
        "next_action": next_action,
        "expected_next_state": expected_state,
        "expected_next_reward": np.asarray(expected_step.reward, dtype=np.float64),
        "expected_next_discount": np.asarray(expected_step.discount, dtype=np.float64),
        "expected_next_step_type": np.asarray(int(expected_step.step_type), dtype=np.int64),
        "expected_next_task_rng_sha256": np.asarray(digest_task_rng(expected_rng)),
        "expected_next_wrapper_step_count": np.asarray(env._step_count, dtype=np.int64),
        "expected_next_wrapper_reset_next_step": np.asarray(env._reset_next_step, dtype=np.bool_),
        "source_model_nplugin": np.asarray(physics.model.nplugin, dtype=np.int64),
    }
    payload.update(observation_payload("prefix_observation", prefix_observation))
    payload.update(observation_payload("expected_next_observation", expected_observation))

    if path.exists():
        path.unlink()
    np.savez_compressed(path, **payload)
    print(json.dumps({
        "mode": "export",
        "checkpoint": str(path),
        "dm_control_version": DM_CONTROL_VERSION,
        "mujoco_version": MUJOCO_VERSION,
        "mujoco_integration_mask": signature,
        "prefix_steps": PREFIX_STEPS,
        "prefix_state_sha256": digest_array(full_state),
        "prefix_observation_sha256": digest_observation(prefix_observation),
        "task_rng_sha256": digest_task_rng(task_rng),
        "snapshot_contains_plugin_state": physics.model.nplugin == 0 or hasattr(physics.data, "plugin_state"),
        "task_id": task_name,
        **verify_legacy_state_is_incomplete(make_environment(task_name)),
        **verify_physics_copy_is_independent(make_environment(task_name)),
    }, sort_keys=True))


def load_observation(data: Any, prefix: str) -> dict[str, np.ndarray]:
    keys = data[f"{prefix}_keys"].tolist()
    return {
        str(key): np.asarray(data[f"{prefix}_array_{index}"]).copy()
        for index, key in enumerate(keys)
    }


def replay_checkpoint(path: Path, task_name: str) -> None:
    with np.load(path, allow_pickle=False) as data:
        saved = {key: data[key].copy() for key in data.files}

    saved_task = str(saved["task_id"].item())
    if saved_task != task_name:
        raise AssertionError(f"checkpoint task {saved_task} does not match requested task {task_name}")
    env = make_environment(task_name)
    env.reset()
    signature = int(mujoco.mjtState.mjSTATE_INTEGRATION.value)
    replay_model_digest = digest_bytes(env.physics.model.to_bytes())
    if replay_model_digest != str(saved["compiled_model_sha256"].item()):
        raise AssertionError("fresh-process compiled MuJoCo model digest differs")

    expected_control_timestep = float(saved["control_timestep"])
    actual_control_timestep = float(env.control_timestep())
    if np.asarray([actual_control_timestep], dtype="<f8").tobytes() != np.asarray(
        [expected_control_timestep], dtype="<f8"
    ).tobytes():
        raise AssertionError("fresh-process control timestep differs")

    if int(env._n_sub_steps) != int(saved["wrapper_n_sub_steps"]):
        raise AssertionError("fresh-process wrapper n_sub_steps differs")
    if np.asarray([env._step_limit], dtype="<f8").tobytes() != np.asarray(
        [float(saved["wrapper_step_limit"])], dtype="<f8"
    ).tobytes():
        raise AssertionError("fresh-process wrapper step limit differs")
    if bool(env.physics.legacy_step) != bool(saved["wrapper_legacy_step"]):
        raise AssertionError("fresh-process legacy_step differs")
    if signature != EXPECTED_INTEGRATION_MASK:
        raise AssertionError(f"wrong pinned integration mask: {signature}")

    if int(saved["source_model_nplugin"]) > 0 and not hasattr(env.physics.data, "plugin_state"):
        raise AssertionError("model plugin state is present in snapshot but unavailable in replay object")

    task_rng_state = (
        str(saved["task_rng_name"].item()),
        np.asarray(saved["task_rng_keys"], dtype=np.uint32),
        int(saved["task_rng_position"]),
        int(saved["task_rng_has_gauss"]),
        float(saved["task_rng_cached_gaussian"]),
    )
    expected_state_size = int(mujoco.mj_stateSize(env.physics.model.ptr, signature))
    if expected_state_size != int(saved["integration_state_size"]):
        raise AssertionError("fresh-process MuJoCo integration state size differs")
    env.physics.set_state(saved["integration_state"], sig=signature)
    env.task.random.set_state(task_rng_state)
    env._step_count = int(saved["wrapper_step_count"])
    env._reset_next_step = bool(saved["wrapper_reset_next_step"])

    # Recompute derived MuJoCo quantities before reading observations.
    env.physics.forward()
    prefix_actual = env.task.get_observation(env.physics)
    prefix_expected = load_observation(saved, "prefix_observation")
    prefix_error = compare_observations(prefix_actual, prefix_expected, PREFIX_TOLERANCE)
    if digest_observation(prefix_actual) != str(saved["prefix_observation_sha256"].item()):
        raise AssertionError("fresh-process factual prefix observation digest differs")

    if digest_task_rng(task_rng_tuple(env.task)) != str(saved["task_rng_sha256"].item()):
        raise AssertionError("fresh-process task RNG state differs")

    actual_step = env.step(saved["next_action"])
    actual_state = env.physics.get_state(sig=signature)
    if not np.allclose(
        actual_state, saved["expected_next_state"], atol=PREFIX_TOLERANCE, rtol=0.0
    ):
        state_error = float(np.max(np.abs(actual_state - saved["expected_next_state"])))
        raise AssertionError(f"fresh-process next-step integration state differs: {state_error}")
    next_expected = load_observation(saved, "expected_next_observation")
    next_error = compare_observations(
        actual_step.observation, next_expected, PREFIX_TOLERANCE
    )
    if not np.isclose(float(actual_step.reward), float(saved["expected_next_reward"]), atol=PREFIX_TOLERANCE, rtol=0.0):
        raise AssertionError("fresh-process reward differs")
    if not np.isclose(float(actual_step.discount), float(saved["expected_next_discount"]), atol=PREFIX_TOLERANCE, rtol=0.0):
        raise AssertionError("fresh-process discount differs")
    if int(actual_step.step_type) != int(saved["expected_next_step_type"]):
        raise AssertionError("fresh-process step type differs")
    if digest_task_rng(task_rng_tuple(env.task)) != str(saved["expected_next_task_rng_sha256"].item()):
        raise AssertionError("fresh-process task RNG after next step differs")
    if env._step_count != int(saved["expected_next_wrapper_step_count"]):
        raise AssertionError("fresh-process wrapper step count differs")
    if env._reset_next_step != bool(saved["expected_next_wrapper_reset_next_step"]):
        raise AssertionError("fresh-process wrapper reset flag differs")

    print(json.dumps({
        "mode": "replay",
        "fresh_process": True,
        "prefix_observation_max_abs_error": prefix_error,
        "next_observation_max_abs_error": next_error,
        "next_integration_state_sha256": digest_array(actual_state),
        "task_rng_after_step_sha256": digest_task_rng(task_rng_tuple(env.task)),
        "step_count": env._step_count,
        "passed": True,
    }, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("export", "replay"))
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--task", choices=TASKS, default="cartpole.swingup")
    arguments = parser.parse_args()

    installed_dm_control = importlib.metadata.version("dm-control")
    if installed_dm_control != DM_CONTROL_VERSION:
        raise RuntimeError(
            f"expected dm_control {DM_CONTROL_VERSION}; found {installed_dm_control}"
        )
    installed_mujoco = importlib.metadata.version("mujoco")
    if installed_mujoco != MUJOCO_VERSION:
        raise RuntimeError(f"expected MuJoCo {MUJOCO_VERSION}; found {installed_mujoco}")
    if int(mujoco.mjtState.mjSTATE_INTEGRATION.value) != EXPECTED_INTEGRATION_MASK:
        raise RuntimeError("pinned MuJoCo integration-state mask differs")

    if arguments.mode == "export":
        export_checkpoint(arguments.checkpoint, arguments.task)
    else:
        replay_checkpoint(arguments.checkpoint, arguments.task)


if __name__ == "__main__":
    main()

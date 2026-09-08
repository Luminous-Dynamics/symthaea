#!/usr/bin/env python3
"""Independent static consistency checks for Workbench invocation isolation profile v1.

This checker deliberately does not reuse the profile verifier. It ensures that
all runner-controlled execve environment bindings correspond to fresh roots and
that the profile does not confuse entry environment with descendant environment.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROFILE = Path(__file__).parents[1] / "data/neuroscience/workbench_invocation_isolation_profile_v1.json"

EXPECTED_BINDINGS = {
    "PWD": ("invocation.cwd", "cwd_policy"),
    "HOME": ("invocation.home", "home_policy"),
    "XDG_CONFIG_HOME": ("invocation.xdg_config_home", "xdg_config_home_policy"),
    "XDG_CACHE_HOME": ("invocation.xdg_cache_home", "xdg_cache_home_policy"),
    "XDG_DATA_HOME": ("invocation.xdg_data_home", "xdg_data_home_policy"),
    "XDG_STATE_HOME": ("invocation.xdg_state_home", "xdg_state_home_policy"),
    "XDG_RUNTIME_DIR": ("invocation.xdg_runtime_dir", "xdg_runtime_dir_policy"),
    "XDG_CONFIG_DIRS": ("invocation.xdg_config_dirs", "xdg_config_dirs_policy"),
    "XDG_DATA_DIRS": ("invocation.xdg_data_dirs", "xdg_data_dirs_policy"),
    "TMPDIR": ("invocation.tmpdir", "tmpdir_policy"),
    "TMP": ("invocation.tmpdir", "tmpdir_policy"),
    "TEMP": ("invocation.tmpdir", "tmpdir_policy"),
}
EXPECTED_DESCENDANT = {
    "entry_environment_is_final_environment": False,
    "verified_program_may_transform_environment": True,
    "host_injection_after_execve_allowed": False,
    "post_entry_environment_equivalence_assumed": False,
}


def fail(message: str) -> None:
    raise SystemExit(f"ERROR: {message}")


def main() -> int:
    try:
        value: Any = json.loads(PROFILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"cannot load profile: {exc}")
    if not isinstance(value, dict):
        fail("profile must be an object")
    env = value.get("process_environment")
    iso = value.get("per_invocation_isolation")
    if not isinstance(env, dict) or not isinstance(iso, dict):
        fail("environment/isolation objects required")
    if env.get("stage") != "root-main-program-execve":
        fail("environment stage must be root-main-program-execve")
    bindings = env.get("dynamic_bindings")
    if bindings != {name: target for name, (target, _) in EXPECTED_BINDINGS.items()}:
        fail("dynamic binding set differs from canonical execve binding map")
    for name, (_, policy_key) in EXPECTED_BINDINGS.items():
        if iso.get(policy_key) != "fresh-private-empty-per-invocation":
            fail(f"{name}: corresponding fresh-root policy missing")
    if env.get("fixed", {}).get("PATH") != "":
        fail("PATH must be explicitly empty")
    if env.get("descendant_environment") != EXPECTED_DESCENDANT:
        fail("entry/descendant environment boundary drift")
    if iso.get("dynamic_environment_must_match_created_roots") is not True:
        fail("dynamic environment/root equality must remain required")
    if iso.get("dynamic_environment_paths_must_be_absolute") is not True:
        fail("dynamic environment absolute-path requirement missing")
    print("workbench invocation isolation static consistency: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

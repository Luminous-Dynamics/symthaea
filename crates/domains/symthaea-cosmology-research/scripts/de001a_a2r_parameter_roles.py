#!/usr/bin/env python3
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""DE-001A2R exact parameter-role binding for the DESI DR2 minimization subject.

This program performs no optimization and no likelihood evaluation. It verifies
the exact A0-qualified DESI configuration bytes, loads the expanded parameter
block with Cobaya 3.6.2, and records Cobaya's own parameter-role partition.

Exit codes:
  0: PASS role binding
  2: INVALID evidence/configuration
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import sys
from typing import Any

PROTOCOL = "DE-001A2R-PARAMETER-ROLE-BINDING-v1"
AUTHORITY = "optimizer-parameter-role-binding-only"
A0_PROTOCOL = "DE-001A0-BYTE-INTEGRITY-v1"
COBAYA_VERSION = "3.6.2"
COBAYA_SOURCE_COMMIT = "899f30a49f85de610dac321e91a1af50018e56aa"

ARTIFACTS = {
    "reference-input-configuration": {
        "size": 2381,
        "sha256": "34499cb78ecaec78db44da9f06f61cd9c9ee497dc5c541b72b48cda54091c6ef",
    },
    "reference-expanded-configuration": {
        "size": 3969,
        "sha256": "c4c23032d1695635aaea6eb47fabd909006ff32df0a27eadbf64b36c89f31ba1",
    },
    "reference-minimizer-configuration": {
        "size": 2484,
        "sha256": "6b51048b359e4b9d646de09379ca273a8d6a1bc1b5a88f585a09d8f2e61f290c",
    },
    "reference-bestfit-text": {
        "size": 902,
        "sha256": "bf8e35e2380ef35b137a77645dcb351af2ed2a93ca8da16c1fd71cb5dd7a1358",
    },
}
NAMED_REVIEW = ("omm", "omegam", "hrdrag")
MAX_FILE_BYTES = 1024 * 1024


class InvalidBinding(RuntimeError):
    pass


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_regular(path: Path) -> bytes:
    try:
        st = path.lstat()
    except OSError as exc:
        raise InvalidBinding(f"{path}: metadata failed: {exc}") from exc
    if path.is_symlink():
        raise InvalidBinding(f"{path}: symlinks are forbidden")
    if not path.is_file():
        raise InvalidBinding(f"{path}: not a regular file")
    if st.st_size > MAX_FILE_BYTES:
        raise InvalidBinding(f"{path}: file exceeds {MAX_FILE_BYTES} bytes")
    data = path.read_bytes()
    if len(data) != st.st_size:
        raise InvalidBinding(f"{path}: size changed while reading")
    return data


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise InvalidBinding(f"duplicate JSON key: {key}")
        out[key] = value
    return out


def _read_json(path: Path) -> tuple[bytes, dict[str, Any]]:
    data = _read_regular(path)
    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                InvalidBinding(f"non-finite JSON constant: {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InvalidBinding(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise InvalidBinding(f"{path}: top-level JSON must be an object")
    return data, value


def _require_str(value: dict[str, Any], key: str) -> str:
    result = value.get(key)
    if not isinstance(result, str):
        raise InvalidBinding(f"{key}: expected string")
    return result


def _validate_a0_receipt(receipt: dict[str, Any]) -> str:
    if _require_str(receipt, "protocol") != A0_PROTOCOL:
        raise InvalidBinding("A0 receipt protocol mismatch")
    if _require_str(receipt, "verdict") != "PASS":
        raise InvalidBinding("A0 receipt is not PASS")
    if _require_str(receipt, "scientific_claim") != "NONE":
        raise InvalidBinding("A0 receipt scientific_claim is not NONE")
    errors = receipt.get("errors")
    if not isinstance(errors, list) or errors:
        raise InvalidBinding("A0 receipt contains errors")
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, list):
        raise InvalidBinding("A0 receipt artifacts is not an array")

    by_role: dict[str, dict[str, Any]] = {}
    for entry in artifacts:
        if not isinstance(entry, dict):
            raise InvalidBinding("A0 artifact receipt is not an object")
        role = entry.get("role")
        if not isinstance(role, str) or role in by_role:
            raise InvalidBinding("A0 artifact role missing or duplicated")
        by_role[role] = entry

    for role, expected in ARTIFACTS.items():
        entry = by_role.get(role)
        if entry is None:
            raise InvalidBinding(f"A0 receipt missing required role {role}")
        if entry.get("status") != "PASS":
            raise InvalidBinding(f"A0 role {role} is not PASS")
        if entry.get("expected_size") != expected["size"] or entry.get("actual_size") != expected["size"]:
            raise InvalidBinding(f"A0 role {role} size mismatch")
        if entry.get("expected_sha256") != expected["sha256"] or entry.get("actual_sha256") != expected["sha256"]:
            raise InvalidBinding(f"A0 role {role} hash mismatch")
    return _sha256(
        json.dumps(receipt, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    )


def _validate_artifact(path: Path, role: str) -> bytes:
    expected = ARTIFACTS[role]
    if path.name != role:
        raise InvalidBinding(f"{role}: expected basename {role!r}, got {path.name!r}")
    data = _read_regular(path)
    if len(data) != expected["size"]:
        raise InvalidBinding(f"{role}: size mismatch")
    if _sha256(data) != expected["sha256"]:
        raise InvalidBinding(f"{role}: SHA-256 mismatch")
    return data


def _load_cobaya_yaml(path: Path) -> dict[str, Any]:
    from cobaya.yaml import yaml_load_file

    try:
        value = yaml_load_file(str(path))
    except Exception as exc:
        raise InvalidBinding(f"{path}: Cobaya YAML load failed: {exc}") from exc
    if not isinstance(value, dict):
        raise InvalidBinding(f"{path}: YAML root is not a mapping")
    return value


def _role_snapshot(expanded: dict[str, Any]) -> dict[str, Any]:
    from cobaya.parameterization import (
        Parameterization,
        expand_info_param,
        is_derived_param,
        is_fixed_or_function_param,
        is_sampled_param,
    )

    params = expanded.get("params")
    if not isinstance(params, dict) or not params:
        raise InvalidBinding("expanded configuration has no non-empty params mapping")

    helper_sampled: list[str] = []
    helper_fixed_or_function: list[str] = []
    helper_derived: list[str] = []
    expanded_infos: dict[str, Any] = {}
    for name, info in params.items():
        if not isinstance(name, str) or not name:
            raise InvalidBinding("expanded params contains a non-string/empty name")
        expanded_info = expand_info_param(info)
        expanded_infos[name] = expanded_info
        if is_sampled_param(info):
            helper_sampled.append(name)
        if is_fixed_or_function_param(info):
            helper_fixed_or_function.append(name)
        if is_derived_param(info):
            helper_derived.append(name)

    try:
        parameterization = Parameterization(params, ignore_unused_sampled=True)
    except Exception as exc:
        raise InvalidBinding(f"Cobaya Parameterization rejected expanded params: {exc}") from exc

    sampled = list(parameterization.sampled_params())
    constants = list(parameterization.constant_params())
    derived = list(parameterization.derived_params())
    inputs = list(parameterization.input_params())
    outputs = list(parameterization.output_params())
    dropped = sorted(parameterization.dropped_param_set())
    renames = parameterization.sampled_params_renames()
    proposals = parameterization.get_sampled_params_proposals()
    sampled_info = parameterization.sampled_params_info()

    if sampled != helper_sampled:
        raise InvalidBinding(
            "Cobaya helper and Parameterization sampled-coordinate order disagree"
        )

    return {
        "all_parameter_names": list(params),
        "sampled_parameters": sampled,
        "constant_parameters": constants,
        "derived_parameters": derived,
        "input_parameters": inputs,
        "output_parameters": outputs,
        "dropped_parameters": dropped,
        "sampled_renames": renames,
        "sampled_proposals": proposals,
        "sampled_info": sampled_info,
        "helper_roles": {
            "sampled": helper_sampled,
            "fixed_or_function": helper_fixed_or_function,
            "derived": helper_derived,
        },
        "expanded_info": expanded_infos,
    }


def _parse_bestfit_table(data: bytes) -> tuple[list[str], dict[str, float]]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise InvalidBinding("best-fit text is not UTF-8") from exc

    header: list[str] | None = None
    values: list[float] | None = None
    for raw in text.splitlines():
        stripped = raw.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            fields = stripped.lstrip("#").split()
            if "minuslogpost" in fields:
                if header is not None:
                    raise InvalidBinding("best-fit text contains duplicate candidate headers")
                if len(fields) != len(set(fields)):
                    raise InvalidBinding("best-fit header contains duplicate column names")
                header = fields
            continue
        if header is not None and values is None:
            pieces = stripped.split()
            if len(pieces) != len(header):
                raise InvalidBinding("best-fit row width does not match header")
            try:
                parsed = [float(piece) for piece in pieces]
            except ValueError as exc:
                raise InvalidBinding("best-fit row contains a non-float value") from exc
            if not all(math.isfinite(value) for value in parsed):
                raise InvalidBinding("best-fit row contains non-finite values")
            values = parsed
            break

    if header is None or values is None:
        raise InvalidBinding("best-fit header/data row not found")
    return header, dict(zip(header, values, strict=True))


def _json_safe(value: Any) -> Any:
    """Convert Cobaya structures into deterministic JSON-safe metadata."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise InvalidBinding("non-finite float in role metadata")
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return {"python_repr": repr(value), "python_type": type(value).__name__}


def _write(value: dict[str, Any]) -> None:
    json.dump(value, sys.stdout, sort_keys=True, separators=(",", ":"), allow_nan=False)
    sys.stdout.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("a0_receipt", type=Path)
    parser.add_argument("reference_input", type=Path)
    parser.add_argument("reference_expanded", type=Path)
    parser.add_argument("reference_minimizer", type=Path)
    parser.add_argument("reference_bestfit", type=Path)
    args = parser.parse_args()

    try:
        if importlib.metadata.version("cobaya") != COBAYA_VERSION:
            raise InvalidBinding("runtime Cobaya version mismatch")

        a0_bytes, a0 = _read_json(args.a0_receipt)
        _validate_a0_receipt(a0)

        input_bytes = _validate_artifact(args.reference_input, "reference-input-configuration")
        expanded_bytes = _validate_artifact(
            args.reference_expanded, "reference-expanded-configuration"
        )
        minimizer_bytes = _validate_artifact(
            args.reference_minimizer, "reference-minimizer-configuration"
        )
        bestfit_bytes = _validate_artifact(args.reference_bestfit, "reference-bestfit-text")

        input_info = _load_cobaya_yaml(args.reference_input)
        expanded_info = _load_cobaya_yaml(args.reference_expanded)
        minimizer_info = _load_cobaya_yaml(args.reference_minimizer)

        roles = _role_snapshot(expanded_info)
        header, bestfit = _parse_bestfit_table(bestfit_bytes)

        sampled = roles["sampled_parameters"]
        missing = [name for name in sampled if name not in bestfit]
        if missing:
            raise InvalidBinding(
                f"best-fit table missing sampled optimizer coordinates: {missing!r}"
            )
        sampled_target = {name: bestfit[name] for name in sampled}

        named: dict[str, Any] = {}
        for name in NAMED_REVIEW:
            named[name] = {
                "present_in_expanded_params": name in roles["all_parameter_names"],
                "sampled": name in roles["sampled_parameters"],
                "constant": name in roles["constant_parameters"],
                "derived": name in roles["derived_parameters"],
                "input": name in roles["input_parameters"],
                "output": name in roles["output_parameters"],
                "dropped": name in roles["dropped_parameters"],
                "bestfit_present": name in bestfit,
                "bestfit_value": bestfit.get(name),
            }

        result = {
            "protocol": PROTOCOL,
            "verdict": "PASS",
            "scientific_claim": "NONE",
            "authority": AUTHORITY,
            "a2_execution_authorized": False,
            "coordinate_policy": (
                "future A2 optimizer coordinates MUST equal the complete ordered "
                "Cobaya Parameterization.sampled_params() list emitted here; no "
                "manual additions, deletions, renames, or reordering"
            ),
            "cobaya_version": COBAYA_VERSION,
            "cobaya_source_commit": COBAYA_SOURCE_COMMIT,
            "a0_receipt_sha256": _sha256(a0_bytes),
            "artifact_sha256": {
                "reference_input": _sha256(input_bytes),
                "reference_expanded": _sha256(expanded_bytes),
                "reference_minimizer": _sha256(minimizer_bytes),
                "reference_bestfit": _sha256(bestfit_bytes),
            },
            "roles": _json_safe(roles),
            "sampled_coordinate_order": sampled,
            "sampled_bestfit_target": sampled_target,
            "bestfit_columns": header,
            "bestfit_objectives": {
                key: value
                for key, value in bestfit.items()
                if key == "minuslogpost" or key.startswith("chi2__")
            },
            "named_review": named,
            "input_sampler_block": _json_safe(input_info.get("sampler")),
            "expanded_sampler_block": _json_safe(expanded_info.get("sampler")),
            "minimizer_sampler_block": _json_safe(minimizer_info.get("sampler")),
        }
        _write(result)
        return 0
    except (
        InvalidBinding,
        OSError,
        ValueError,
        TypeError,
        KeyError,
        ImportError,
        RuntimeError,
    ) as exc:
        _write(
            {
                "protocol": PROTOCOL,
                "verdict": "INVALID",
                "scientific_claim": "NONE",
                "authority": AUTHORITY,
                "a2_execution_authorized": False,
                "error": str(exc),
            }
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

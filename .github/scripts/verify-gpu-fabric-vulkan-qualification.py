#!/usr/bin/env python3
"""Independent stdlib-only verifier for the Vulkan qualification witness."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

VULKAN_API_1_3 = 4206592

FIXTURES = {
    "fixture": {
        "node_count": 2,
        "barrier_count": 1,
        "completion": 2,
        "initial": {
            "lhs": "0ff0aa55",
            "rhs": "33cc55aa",
            "mid": "00000000",
            "out": "00000000",
        },
        "ops": [("mid", "lhs", "rhs"), ("out", "mid", "rhs")],
    },
    "hazard": {
        "node_count": 3,
        "barrier_count": 3,
        "completion": 3,
        "initial": {
            "lhs": "0ff0aa55",
            "rhs": "33cc55aa",
            "mid": "00000000",
        },
        "ops": [("mid", "lhs", "rhs"), ("rhs", "mid", "lhs"), ("rhs", "lhs", "mid")],
    },
}

class VerificationError(RuntimeError):
    pass

def fail(message: str) -> None:
    raise VerificationError(message)

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def parse_kv(lines: list[str]) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in lines:
        if "=" not in line or line.startswith("resource_"):
            continue
        key, value = line.split("=", 1)
        if key in values:
            fail(f"duplicate key: {key}")
        values[key] = value
    return values

def parse_runtime(path: Path) -> list[tuple[str, list[str]]]:
    blocks: list[tuple[str, list[str]]] = []
    current_name = None
    current: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("qualification_fixture="):
            if current_name is not None:
                blocks.append((current_name, current))
            current_name = line.split("=", 1)[1]
            current = []
        elif current_name is not None:
            current.append(line)
    if current_name is not None:
        blocks.append((current_name, current))
    return blocks

def parse_vectors(lines: list[str]) -> tuple[dict[str, tuple[int, bytes]], dict[str, tuple[int, bytes]]]:
    initial: dict[str, tuple[int, bytes]] = {}
    observed: dict[str, tuple[int, bytes]] = {}
    for line in lines:
        if line.startswith("resource_initial_hex="):
            target = initial
            rest = line.split("=", 1)[1]
        elif line.startswith("resource_observed_hex="):
            target = observed
            rest = line.split("=", 1)[1]
        else:
            continue
        resource, dimensions_text, hex_text = rest.rsplit(":", 2)
        if resource in target:
            fail(f"duplicate vector: {resource}")
        try:
            payload = bytes.fromhex(hex_text)
        except ValueError as exc:
            fail(f"invalid hex for {resource}: {exc}")
        target[resource] = (int(dimensions_text), payload)
    return initial, observed

def xor_bytes(left: bytes, right: bytes) -> bytes:
    if len(left) != len(right):
        fail("xor operand lengths differ")
    return bytes(a ^ b for a, b in zip(left, right))

def expected_state(spec: dict) -> dict[str, bytes]:
    state = {name: bytes.fromhex(value) for name, value in spec["initial"].items()}
    for output, left, right in spec["ops"]:
        state[output] = xor_bytes(state[left], state[right])
    return state

def verify_runtime(path: Path) -> None:
    blocks = parse_runtime(path)
    if [name for name, _ in blocks] != ["fixture", "hazard"]:
        fail("runtime witness must contain fixture then hazard exactly once")

    for name, lines in blocks:
        spec = FIXTURES[name]
        values = parse_kv(lines)
        if values.get("qualification_witness_version") != "1":
            fail(f"{name}: witness version mismatch")
        if values.get("qualification_claim") != "workload_execution+synchronization_only":
            fail(f"{name}: qualification claim mismatch")
        if values.get("receipt_version") != "3":
            fail(f"{name}: receipt version mismatch")
        if int(values.get("node_count", "-1")) != spec["node_count"]:
            fail(f"{name}: node count mismatch")
        if int(values.get("barrier_count", "-1")) != spec["barrier_count"]:
            fail(f"{name}: barrier count mismatch")
        if int(values.get("completion_expected", "-1")) != spec["completion"]:
            fail(f"{name}: completion expected mismatch")
        if int(values.get("completion_observed", "-1")) != spec["completion"]:
            fail(f"{name}: completion observed mismatch")
        if int(values.get("vulkan_api_version", "-1")) != VULKAN_API_1_3:
            fail(f"{name}: Vulkan API version mismatch")
        if int(values.get("physical_device_api_version", "-1")) < VULKAN_API_1_3:
            fail(f"{name}: physical device API version below Vulkan 1.3")
        if int(values.get("queue_family_index", "-1")) != 0:
            fail(f"{name}: queue family mismatch")

        initial, observed = parse_vectors(lines)
        expected_initial = spec["initial"]
        if set(initial) != set(expected_initial) or set(observed) != set(expected_initial):
            fail(f"{name}: resource set mismatch")
        for resource, expected_hex in expected_initial.items():
            dimensions, payload = initial[resource]
            if dimensions != 32 or payload.hex() != expected_hex:
                fail(f"{name}: initial vector mismatch for {resource}")

        final = expected_state(spec)
        for resource, expected_payload in final.items():
            dimensions, payload = observed[resource]
            if dimensions != 32 or payload != expected_payload:
                fail(f"{name}: observed vector mismatch for {resource}")

def verify_source_hashes(path: Path, root: Path) -> None:
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) < 2:
            fail(f"malformed source hash line: {line}")
        expected, relative = fields[0], fields[-1]
        target = root / relative
        if not target.is_file():
            fail(f"source hash target missing: {relative}")
        if sha256_file(target) != expected:
            fail(f"source hash mismatch: {relative}")

def verify_single_hash_file(path: Path, target: Path) -> None:
    fields = path.read_text(encoding="utf-8").split()
    if not fields or fields[0] != sha256_file(target):
        fail(f"hash mismatch for {target}")

def verify_input_hashes(path: Path, root: Path) -> None:
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) < 2:
            fail(f"malformed qualification input hash line: {line}")
        expected, target_text = fields[0], fields[-1]
        target = Path(target_text)
        if not target.is_absolute():
            target = root / target
        if not target.is_file():
            fail(f"qualification input missing: {target_text}")
        if sha256_file(target) != expected:
            fail(f"qualification input hash mismatch: {target_text}")

def verify_metadata(path: Path) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    names = [item.get("name") for item in data.get("packages", [])]
    if names != ["symthaea-gpu-fabric"]:
        fail(f"unexpected isolated package set: {names!r}")

def verify_environment(path: Path, expected_commit: str) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    values = parse_kv(lines)
    if values.get("qualification_commit") != expected_commit:
        fail("qualification_commit does not match the exact checked-out head")
    if not values.get("workflow_sha"):
        fail("workflow_sha missing")
    if values.get("runner_arch") != "X64":
        fail("runner architecture is not X64")
    if not any(line.startswith("rustc ") for line in lines):
        fail("rustc version missing from environment packet")

def verify_cargo_manifest(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if '[package]' not in text or 'name = "symthaea-gpu-fabric"' not in text:
        fail("isolated manifest package identity mismatch")
    if '[workspace]' not in text or 'resolver = "2"' not in text:
        fail("isolated workspace declaration missing")

def verify_packages(path: Path) -> None:
    required = {"libvulkan1", "mesa-vulkan-drivers", "vulkan-tools", "vulkan-validationlayers", "gdb"}
    seen = {line.split(chr(9), 1)[0] for line in path.read_text(encoding="utf-8").splitlines() if line}
    missing = required - seen
    if missing:
        fail(f"missing runner packages: {sorted(missing)}")

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--source-sha256", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--lock-sha256", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--cargo-toml", type=Path, required=True)
    parser.add_argument("--package-versions", type=Path, required=True)
    parser.add_argument("--input-sha256", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()

    root = args.runtime.parent
    verify_runtime(args.runtime)
    verify_environment(args.environment, args.expected_commit)
    verify_source_hashes(args.source_sha256, root)
    verify_single_hash_file(args.lock_sha256, args.lock)
    verify_metadata(args.metadata)
    verify_cargo_manifest(args.cargo_toml)
    verify_packages(args.package_versions)
    verify_input_hashes(args.input_sha256, root)

    print("independent_verification=pass")
    print("verification_implementation=python-stdlib-only")
    print(f"qualification_commit={args.expected_commit}")
    print(f"runtime_sha256={sha256_file(args.runtime)}")
    print(f"dependency_lock_sha256={sha256_file(args.lock)}")
    print("fixture_count=2")
    print("semantic_model=independent bytewise XOR replay of the sealed fixtures")
    return 0

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except VerificationError as exc:
        print(f"independent_verification=fail: {exc}", file=sys.stderr)
        raise SystemExit(1)

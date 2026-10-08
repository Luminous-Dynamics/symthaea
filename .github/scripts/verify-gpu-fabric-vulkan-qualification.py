#!/usr/bin/env python3
"""Independent stdlib-only verifier for the Vulkan qualification witness."""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

VULKAN_API_1_3 = 4206592
IMPLEMENTATION_IDENTITY_VERSION = "1"
PHYSICAL_DEVICE_IDENTITY_VERSION = "1"
WGSL_ABI_MARKER = "symthaea.hdc.bind_xor.storage-u32.v1"
KERNEL_ID = "symthaea.hdc.bind_xor.v1"

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

def sha256_len_prefixed(parts: list[bytes], domain: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(domain)
    digest.update(b"\x00")
    for part in parts:
        digest.update(struct.pack("<Q", len(part)))
        digest.update(part)
    return digest.hexdigest()


def verify_provenance(values: dict[str, str], root: Path) -> tuple[str, str, bytes]:
    if values.get("implementation_identity_version") != IMPLEMENTATION_IDENTITY_VERSION:
        fail("implementation identity version mismatch")
    if values.get("physical_device_identity_version") != PHYSICAL_DEVICE_IDENTITY_VERSION:
        fail("physical-device identity version mismatch")
    if values.get("implementation_abi_marker") != WGSL_ABI_MARKER:
        fail("implementation ABI marker mismatch")
    if values.get("implementation_kernel_id") != KERNEL_ID:
        fail("implementation kernel id mismatch")

    try:
        wgsl = bytes.fromhex(values["implementation_wgsl_hex"])
        spirv = bytes.fromhex(values["shader_spirv_hex"])
        name = bytes.fromhex(values["physical_device_name_hex"])
        uuid = bytes.fromhex(values["device_uuid"])
    except (KeyError, ValueError) as exc:
        fail(f"malformed provenance hex field: {exc}")
    if not wgsl or not spirv:
        fail("provenance WGSL/SPIR-V payload is empty")
    if len(spirv) % 4 != 0:
        fail("SPIR-V payload is not a multiple of four bytes")
    if len(uuid) != 16 or not any(uuid):
        fail("device UUID is missing or all zeroes")

    wgsl_path = root / "src" / "hdc_bind_xor.wgsl"
    if not wgsl_path.is_file():
        fail("sealed WGSL source file is missing")
    source_wgsl = wgsl_path.read_bytes()
    if wgsl != source_wgsl:
        fail("runtime WGSL bytes do not match sealed WGSL source")
    if values.get("implementation_wgsl_sha256") != sha256_file(wgsl_path):
        fail("WGSL source SHA-256 mismatch")
    if values.get("shader_spirv_sha256") != hashlib.sha256(spirv).hexdigest():
        fail("SPIR-V SHA-256 mismatch")

    implementation_digest = sha256_len_prefixed(
        [
            WGSL_ABI_MARKER.encode("utf-8"),
            KERNEL_ID.encode("utf-8"),
            wgsl,
            spirv,
        ],
        b"symthaea.gpu-fabric.vulkan-implementation.v1",
    )
    if values.get("implementation_identity_sha256") != implementation_digest:
        fail("implementation identity digest mismatch")

    try:
        vendor_id = int(values["physical_device_vendor_id"])
        device_id = int(values["physical_device_device_id"])
        device_type = int(values["physical_device_type"])
        api_version = int(values["physical_device_api_version"])
        driver_version = int(values["physical_device_driver_version"])
    except (KeyError, ValueError) as exc:
        fail(f"malformed physical-device numeric identity: {exc}")
    numeric = (vendor_id, device_id, device_type, api_version, driver_version)
    if any(value < 0 or value > 0xFFFFFFFF for value in numeric):
        fail("physical-device identity numeric field outside u32 range")
    device_hash = hashlib.sha256()
    device_hash.update(b"symthaea.gpu-fabric.vulkan-device.v1")
    device_hash.update(b"\x00")
    for value in numeric:
        device_hash.update(struct.pack("<I", value))
    device_hash.update(struct.pack("<Q", len(name)))
    device_hash.update(name)
    physical_digest = device_hash.hexdigest()
    if values.get("physical_device_identity_sha256") != physical_digest:
        fail("physical-device identity digest mismatch")

    if int(values.get("vulkan_api_version", "-1")) != VULKAN_API_1_3:
        fail("Vulkan API version changed")
    if api_version < VULKAN_API_1_3:
        fail("physical-device API version below Vulkan 1.3")
    return implementation_digest, physical_digest, uuid


def verify_runtime(path: Path) -> None:
    blocks = parse_runtime(path)
    if [name for name, _ in blocks] != ["fixture", "hazard"]:
        fail("runtime witness must contain fixture then hazard exactly once")

    provenance: tuple[str, str, bytes] | None = None
    for name, lines in blocks:
        spec = FIXTURES[name]
        values = parse_kv(lines)
        if values.get("qualification_witness_version") != "1":
            fail(f"{name}: witness version mismatch")
        if values.get("qualification_claim") != "workload_execution+synchronization_only":
            fail(f"{name}: qualification claim mismatch")
        if values.get("receipt_version") != "4":
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

        implementation_digest, physical_digest, uuid = verify_provenance(values, path.parent)
        if values.get("implementation_identity_sha256") != implementation_digest:
            fail(f"{name}: implementation identity receipt binding mismatch")
        if values.get("physical_device_identity_sha256") != physical_digest:
            fail(f"{name}: physical-device identity receipt binding mismatch")

        current_provenance = (implementation_digest, physical_digest, uuid)
        if provenance is None:
            provenance = current_provenance
        elif current_provenance != provenance:
            fail(f"{name}: provenance identity differs from the other fixture")
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

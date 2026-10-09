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
VULKAN_ENTRY_POINT = "main"
VULKAN_SHADER_STAGE = "compute"
DRIVER_IDENTITY_VERSION = "symthaea.gpu-fabric.vulkan-driver.v1"
DRIVER_IDENTITY_VERSION_NUMBER = "1"
QUEUE_FAMILY_IDENTITY_VERSION = "symthaea.gpu-fabric.vulkan-queue-family.v1"
QUEUE_FAMILY_IDENTITY_VERSION_NUMBER = "1"
SYNCHRONIZATION_FEATURE_IDENTITY_VERSION = "symthaea.gpu-fabric.vulkan-sync-features.v1"
SYNCHRONIZATION_FEATURE_IDENTITY_VERSION_NUMBER = "1"

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
        "barriers": [(1, 2, "mid", "read_after_write")],
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
        "barriers": [
            (1, 2, "mid", "read_after_write"),
            (1, 2, "rhs", "write_after_read"),
            (2, 3, "rhs", "write_after_write"),
        ],
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
    if values.get("implementation_entry_point") != VULKAN_ENTRY_POINT:
        fail("implementation entry point mismatch")
    if values.get("implementation_shader_stage") != VULKAN_SHADER_STAGE:
        fail("implementation shader stage mismatch")

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
            VULKAN_ENTRY_POINT.encode("utf-8"),
            VULKAN_SHADER_STAGE.encode("utf-8"),
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

    if values.get("queue_family_identity_version") != QUEUE_FAMILY_IDENTITY_VERSION_NUMBER:
        fail("queue-family identity version mismatch")
    try:
        queue_index = int(values["queue_family_index"])
        queue_flags = int(values["queue_family_queue_flags"])
        queue_count = int(values["queue_family_queue_count"])
        timestamp_valid_bits = int(values["queue_family_timestamp_valid_bits"])
        granularity = tuple(
            int(part)
            for part in values["queue_family_min_image_transfer_granularity"].split(",")
        )
    except (KeyError, ValueError) as exc:
        fail(f"malformed queue-family identity: {exc}")
    if queue_index < 0 or queue_index > 0xFFFFFFFF:
        fail("queue-family index outside u32 range")
    if queue_flags < 0 or queue_flags > 0xFFFFFFFF:
        fail("queue-family flags outside u32 range")
    if queue_count <= 0 or queue_count > 0xFFFFFFFF:
        fail("queue-family count outside u32 range")
    if timestamp_valid_bits < 0 or timestamp_valid_bits > 0xFFFFFFFF:
        fail("queue-family timestamp-valid bits outside u32 range")
    if len(granularity) != 3 or any(value < 0 or value > 0xFFFFFFFF for value in granularity):
        fail("queue-family granularity outside u32 range")
    if (queue_flags & 0x00000002) == 0:
        fail("selected queue family lacks compute capability")
    queue_hash = hashlib.sha256()
    queue_hash.update(QUEUE_FAMILY_IDENTITY_VERSION.encode("utf-8"))
    queue_hash.update(b"\x00")
    queue_hash.update(struct.pack("<I", queue_index))
    queue_hash.update(struct.pack("<I", queue_flags))
    queue_hash.update(struct.pack("<I", queue_count))
    queue_hash.update(struct.pack("<I", timestamp_valid_bits))
    for value in granularity:
        queue_hash.update(struct.pack("<I", value))
    queue_digest = queue_hash.hexdigest()
    if values.get("queue_family_identity_sha256") != queue_digest:
        fail("queue-family identity digest mismatch")

    if int(values.get("vulkan_api_version", "-1")) != VULKAN_API_1_3:
        fail("Vulkan API version changed")
    if api_version < VULKAN_API_1_3:
        fail("physical-device API version below Vulkan 1.3")
    return (
        implementation_digest,
        physical_digest,
        uuid,
        queue_digest,
        queue_flags,
        queue_count,
        timestamp_valid_bits,
        granularity,
    )


def verify_synchronization_feature_provenance(
    values: dict[str, str],
) -> tuple[str, tuple[int, int, int, int]]:
    if values.get("synchronization_feature_identity_version") != SYNCHRONIZATION_FEATURE_IDENTITY_VERSION_NUMBER:
        fail("synchronization feature identity version mismatch")
    try:
        fields = (
            int(values["timeline_semaphore_supported"]),
            int(values["synchronization2_supported"]),
            int(values["timeline_semaphore_enabled"]),
            int(values["synchronization2_enabled"]),
        )
    except (KeyError, ValueError) as exc:
        fail(f"malformed synchronization feature profile: {exc}")
    if any(value not in (0, 1) for value in fields):
        fail("synchronization feature profile must use 0/1 values")
    if fields != (1, 1, 1, 1):
        fail("required synchronization features are not fully supported and enabled")
    digest = hashlib.sha256()
    digest.update(SYNCHRONIZATION_FEATURE_IDENTITY_VERSION.encode("utf-8"))
    digest.update(b"\x00")
    digest.update(bytes(fields))
    expected = digest.hexdigest()
    if values.get("synchronization_feature_identity_sha256") != expected:
        fail("synchronization feature identity digest mismatch")
    return expected, fields


def verify_driver_provenance(values: dict[str, str]) -> tuple[str, bytes, int]:
    if values.get("driver_identity_version") != DRIVER_IDENTITY_VERSION_NUMBER:
        fail("driver identity version mismatch")
    try:
        driver_uuid = bytes.fromhex(values["driver_uuid"])
        driver_id = int(values["driver_id"])
        driver_name = bytes.fromhex(values["driver_name_hex"])
        driver_info = bytes.fromhex(values["driver_info_hex"])
    except (KeyError, ValueError) as exc:
        fail(f"malformed driver provenance field: {exc}")
    if len(driver_uuid) != 16:
        fail("driver UUID must be exactly 16 bytes")
    if driver_id < -0x80000000 or driver_id > 0x7FFFFFFF:
        fail("driver ID outside signed 32-bit range")
    digest = hashlib.sha256()
    digest.update(DRIVER_IDENTITY_VERSION.encode("utf-8"))
    digest.update(b"\x00")
    digest.update(struct.pack("<Q", len(driver_uuid)))
    digest.update(driver_uuid)
    digest.update(struct.pack("<i", driver_id))
    digest.update(struct.pack("<Q", len(driver_name)))
    digest.update(driver_name)
    digest.update(struct.pack("<Q", len(driver_info)))
    digest.update(driver_info)
    expected = digest.hexdigest()
    if values.get("driver_identity_sha256") != expected:
        fail("driver identity digest mismatch")
    return expected, driver_uuid, driver_id


def verify_materialized_barrier_lowering(values: dict[str, str], spec: dict, name: str) -> None:
    """Reconstruct the concrete barrier-call contract independently from fixture edges."""
    semantic_barriers = spec.get("barriers")
    if semantic_barriers is None:
        fail(f"{name}: independent semantic barrier fixture is missing")

    by_target: dict[int, list[tuple[int, int, str, str]]] = {}
    for source, target, resource, kind in semantic_barriers:
        if kind not in {"read_after_write", "write_after_read", "write_after_write"}:
            fail(f"{name}: unsupported expected barrier kind: {kind}")
        if resource not in spec["initial"]:
            fail(f"{name}: expected barrier references unknown resource {resource}")
        by_target.setdefault(target, []).append((source, target, resource, kind))

    parts = [f"batch_count:{sum(bool(by_target.get(node_id)) for node_id in range(1, spec['node_count'] + 1))}"]
    for node_id in range(1, spec["node_count"] + 1):
        barriers = by_target.get(node_id, [])
        if not barriers:
            continue
        buffer_barrier_count = sum(kind != "write_after_read" for _, _, _, kind in barriers)
        memory_barrier_count = len(barriers) - buffer_barrier_count
        parts.extend([
            "batch",
            f"node_id={node_id}",
            "dependency_flags=0",
            f"memory_barrier_count={memory_barrier_count}",
            f"buffer_barrier_count={buffer_barrier_count}",
            "image_barrier_count=0",
        ])
        for ordinal, (source, target, resource, kind) in enumerate(barriers):
            buffer_memory = kind != "write_after_read"
            if kind == "read_after_write":
                src_access, dst_access = "shader_storage_write", "shader_storage_read"
            elif kind == "write_after_read":
                src_access, dst_access = "empty", "empty"
            else:
                src_access, dst_access = "shader_storage_write", "shader_storage_write"
            size = ((len(bytes.fromhex(spec["initial"][resource])) + 3) // 4) * 4 if buffer_memory else 0
            parts.extend([
                "barrier",
                f"ordinal={ordinal}",
                f"from={source}",
                f"to={target}",
                f"resource={resource}",
                f"kind={kind}",
                f"type={'buffer_memory' if buffer_memory else 'execution_memory'}",
                "src_stage=compute_shader",
                f"src_access={src_access}",
                "dst_stage=compute_shader",
                f"dst_access={dst_access}",
                f"queue_family={'ignored' if buffer_memory else 'not_applicable'}",
                "offset=0",
                f"size={size}",
            ])

    expected = sha256_len_prefixed(
        [part.encode("utf-8") for part in parts],
        b"symthaea.gpu-fabric.vulkan-materialized-barriers.v1",
    )
    if values.get("barrier_lowering_digest") != expected:
        fail(f"{name}: materialized Vulkan barrier digest mismatch")


def verify_runtime(path: Path) -> None:
    blocks = parse_runtime(path)
    if [name for name, _ in blocks] != ["fixture", "hazard"]:
        fail("runtime witness must contain fixture then hazard exactly once")

    provenance: tuple[
        str,
        str,
        bytes,
        str,
        bytes,
        int,
        str,
        int,
        int,
        int,
        tuple[int, int, int],
        str,
        tuple[int, int, int, int],
    ] | None = None
    for name, lines in blocks:
        spec = FIXTURES[name]
        values = parse_kv(lines)
        if values.get("qualification_witness_version") != "1":
            fail(f"{name}: witness version mismatch")
        if values.get("qualification_claim") != "workload_execution+synchronization_only":
            fail(f"{name}: qualification claim mismatch")
        if values.get("receipt_version") != "8":
            fail(f"{name}: receipt version mismatch")
        if int(values.get("node_count", "-1")) != spec["node_count"]:
            fail(f"{name}: node count mismatch")
        if int(values.get("barrier_count", "-1")) != spec["barrier_count"]:
            fail(f"{name}: barrier count mismatch")
        verify_materialized_barrier_lowering(values, spec, name)
        if int(values.get("completion_expected", "-1")) != spec["completion"]:
            fail(f"{name}: completion expected mismatch")
        if int(values.get("completion_observed", "-1")) != spec["completion"]:
            fail(f"{name}: completion observed mismatch")
        if int(values.get("vulkan_api_version", "-1")) != VULKAN_API_1_3:
            fail(f"{name}: Vulkan API version mismatch")
        if int(values.get("physical_device_api_version", "-1")) < VULKAN_API_1_3:
            fail(f"{name}: physical device API version below Vulkan 1.3")
        try:
            queue_family_index = int(values["queue_family_index"])
        except (KeyError, ValueError) as exc:
            fail(f"{name}: malformed queue family index: {exc}")
        if queue_family_index < 0 or queue_family_index > 0xFFFFFFFF:
            fail(f"{name}: queue family index outside u32 range")

        (
            implementation_digest,
            physical_digest,
            uuid,
            queue_digest,
            queue_flags,
            queue_count,
            timestamp_valid_bits,
            granularity,
        ) = verify_provenance(values, path.parent)
        sync_feature_digest, sync_feature_fields = verify_synchronization_feature_provenance(values)
        driver_digest, driver_uuid, driver_id = verify_driver_provenance(values)
        if values.get("implementation_identity_sha256") != implementation_digest:
            fail(f"{name}: implementation identity receipt binding mismatch")
        if values.get("physical_device_identity_sha256") != physical_digest:
            fail(f"{name}: physical-device identity receipt binding mismatch")

        current_provenance = (
            implementation_digest,
            physical_digest,
            uuid,
            driver_digest,
            driver_uuid,
            driver_id,
            queue_digest,
            queue_flags,
            queue_count,
            timestamp_valid_bits,
            granularity,
            sync_feature_digest,
            sync_feature_fields,
        )
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

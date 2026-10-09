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

    batch_count = sum(bool(by_target.get(node_id)) for node_id in range(1, spec["node_count"] + 1))
    parts = [f"batch_count:{batch_count}"]
    for node_id in range(1, spec["node_count"] + 1):
        barriers = by_target.get(node_id, [])
        if not barriers:
            continue
        buffer_barrier_count = sum(kind != "write_after_read" for _, _, _, kind in barriers)
        memory_barrier_count = len(barriers) - buffer_barrier_count
        parts.extend([
            "batch",
            f"node_id={node_id}",
            "dependency_structure=VkDependencyInfo",
            "pnext=null",
            "dependency_flags=0",
            f"memory_barrier_count={memory_barrier_count}",
            f"buffer_memory_barrier_count={buffer_barrier_count}",
            "image_memory_barrier_count=0",
        ])
        for ordinal, (source, target, resource, kind) in enumerate(barriers):
            buffer_memory = kind != "write_after_read"
            if kind == "read_after_write":
                src_access, dst_access = "shader_storage_write", "shader_storage_read"
                src_access_mask, dst_access_mask = 0x400000000, 0x200000000
            elif kind == "write_after_read":
                src_access, dst_access = "empty", "empty"
                src_access_mask, dst_access_mask = 0, 0
            else:
                src_access, dst_access = "shader_storage_write", "shader_storage_write"
                src_access_mask, dst_access_mask = 0x400000000, 0x400000000
            size = ((len(bytes.fromhex(spec["initial"][resource])) + 3) // 4) * 4 if buffer_memory else 0
            queue_family_index = str(0xFFFFFFFF) if buffer_memory else "not_applicable"
            parts.extend([
                "barrier",
                f"ordinal={ordinal}",
                f"from={source}",
                f"to={target}",
                f"resource={resource}",
                f"kind={kind}",
                f"type={'VkBufferMemoryBarrier2' if buffer_memory else 'VkMemoryBarrier2'}",
                "pnext=null",
                "src_stage=compute_shader",
                f"src_stage_mask={0x800}",
                f"src_access={src_access}",
                f"src_access_mask={src_access_mask}",
                "dst_stage=compute_shader",
                f"dst_stage_mask={0x800}",
                f"dst_access={dst_access}",
                f"dst_access_mask={dst_access_mask}",
                f"src_queue_family_index={queue_family_index}",
                f"dst_queue_family_index={queue_family_index}",
                "offset=0",
                f"size={size}",
            ])

    expected = sha256_len_prefixed(
        [part.encode("utf-8") for part in parts],
        b"symthaea.gpu-fabric.vulkan-materialized-barriers.v2",
    )
    if values.get("barrier_lowering_digest") != expected:
        fail(f"{name}: materialized Vulkan barrier digest mismatch")


def verify_materialized_submission_contract(values: dict[str, str], spec: dict, name: str) -> None:
    """Independently reconstruct the host-readback and timeline-submit call contract."""
    try:
        queue_family_index = int(values["queue_family_index"])
    except (KeyError, ValueError) as exc:
        fail(f"{name}: malformed queue family for submission digest: {exc}")
    if queue_family_index < 0 or queue_family_index > 0xFFFFFFFF:
        fail(f"{name}: submission queue family index outside u32 range")

    completion = spec["completion"]
    fields = [
        "contract_version=v1",
        f"queue_family_index={queue_family_index}",
        "queue_index=0",
        "semaphore_create_structure=VkSemaphoreCreateInfo",
        "semaphore_create_flags=0",
        "semaphore_create_pnext=VkSemaphoreTypeCreateInfo",
        "semaphore_type_raw=1",
        "semaphore_type=timeline",
        "timeline_initial_value=0",
        "host_readback_dependency_structure=VkDependencyInfo",
        "host_readback_dependency_pnext=null",
        "host_readback_dependency_flags=0",
        "host_readback_memory_barrier_count=1",
        "host_readback_buffer_memory_barrier_count=0",
        "host_readback_image_memory_barrier_count=0",
        "host_readback_barrier_structure=VkMemoryBarrier2",
        "host_readback_barrier_pnext=null",
        "host_readback_src_stage=compute_shader",
        f"host_readback_src_stage_mask={0x800}",
        "host_readback_src_access=shader_storage_write",
        f"host_readback_src_access_mask={0x400000000}",
        "host_readback_dst_stage=host",
        f"host_readback_dst_stage_mask={0x4000}",
        "host_readback_dst_access=host_read",
        f"host_readback_dst_access_mask={0x2000}",
        "host_readback_queue_family_indices=not_applicable",
        "host_readback_offset=0",
        "host_readback_size=0",
        "submit_structure=VkSubmitInfo2",
        "submit_pnext=null",
        "submit_flags=0",
        "wait_semaphore_count=0",
        "command_buffer_count=1",
        "signal_semaphore_count=1",
        "command_buffer_structure=VkCommandBufferSubmitInfo",
        "command_buffer_pnext=null",
        "command_buffer_device_mask=1",
        "signal_structure=VkSemaphoreSubmitInfo",
        "signal_pnext=null",
        f"signal_value={completion}",
        "signal_stage=all_commands",
        f"signal_stage_mask={0x10000}",
        "signal_device_index=0",
        "wait_structure=VkSemaphoreWaitInfo",
        "wait_pnext=null",
        "wait_flags=0",
        "wait_semaphore_count=1",
        f"wait_value={completion}",
        "timeout_ns=5000000000",
        "counter_query=vkGetSemaphoreCounterValue",
        f"planned_submission_count={spec['node_count']}",
    ]
    for ordinal in range(spec["node_count"]):
        fields.extend([
            "planned_submission",
            f"node_id={ordinal + 1}",
            f"ordinal={ordinal}",
            "queue_index=0",
            f"signal_value={ordinal + 1}",
        ])

    expected = sha256_len_prefixed(
        [field.encode("utf-8") for field in fields],
        b"symthaea.gpu-fabric.vulkan-materialized-submission.v1",
    )
    if values.get("completion_lowering_digest") != expected:
        fail(f"{name}: materialized Vulkan submission digest mismatch")


def verify_materialized_dispatch_contract(values: dict[str, str], spec: dict, name: str) -> None:
    """Reconstruct descriptor bindings and workgroup dimensions from semantic operations."""
    operations = spec.get("ops")
    if operations is None or len(operations) != spec["node_count"]:
        fail(f"{name}: semantic dispatch fixture is missing or inconsistent")

    fields = [f"dispatch_record_count:{len(operations)}"]
    for ordinal, (output, left, right) in enumerate(operations):
        if output not in spec["initial"] or left not in spec["initial"] or right not in spec["initial"]:
            fail(f"{name}: dispatch references a resource absent from its input fixture")

        # The Rust workload canonicalizes read resources by ResourceId before
        # descriptor binding 0/1 and binds the single write to slot 2.
        resources = sorted((left, right)) + [output]
        ranges = []
        for resource in resources:
            byte_length = len(bytes.fromhex(spec["initial"][resource]))
            ranges.append(((byte_length + 3) // 4) * 4)

        output_elements = ranges[2] // 4
        groups_x = max(1, (output_elements + 63) // 64)
        node_id = ordinal + 1
        fields.extend([
            "dispatch_record",
            f"node_id={node_id}",
            f"schedule_ordinal={ordinal}",
            "pipeline_bind_point_raw=1",
            "pipeline_layout_set_index=0",
            "descriptor_set_count=1",
            "dynamic_offset_count=0",
            "descriptor_binding_count=3",
        ])
        for binding, resource in enumerate(resources):
            fields.extend([
                "descriptor_binding",
                f"binding={binding}",
                f"resource={resource}",
                "offset=0",
                f"range={ranges[binding]}",
                "descriptor_type_raw=7",
                "stage_flags_raw=32",
            ])
        fields.extend([
            f"dispatch_group_count_x={groups_x}",
            "dispatch_group_count_y=1",
            "dispatch_group_count_z=1",
        ])

    expected = sha256_len_prefixed(
        [field.encode("utf-8") for field in fields],
        b"symthaea.gpu-fabric.vulkan-materialized-dispatch.v1",
    )
    if values.get("execution_lowering_digest") != expected:
        fail(f"{name}: materialized descriptor/dispatch digest mismatch")


def verify_memory_lowering_contract(values: dict[str, str], lines: list[str], spec: dict, name: str) -> None:
    """Independently verify selected Vulkan memory types and cache-maintenance evidence."""
    profiles: dict[str, dict[str, int]] = {}
    for line in lines:
        if not line.startswith("resource_memory_profile="):
            continue
        parts = line.split("=", 1)[1].split(":")
        if len(parts) != 22:
            fail(f"{name}: malformed resource memory profile field count")
        resource = parts[0]
        try:
            numbers = [int(value) for value in parts[1:]]
        except ValueError as exc:
            fail(f"{name}: malformed resource memory profile for {resource}: {exc}")
        if resource in profiles:
            fail(f"{name}: duplicate resource memory profile: {resource}")
        if resource not in spec["initial"]:
            fail(f"{name}: unexpected memory profile resource: {resource}")
        (
            memory_type_index,
            memory_type_bits,
            memory_property_flags,
            memory_heap_index,
            memory_heap_flags,
            memory_heap_size,
            memory_requirement_alignment,
            memory_requirement_size,
            allocation_size,
            storage_size,
            buffer_usage_flags,
            sharing_mode_raw,
            binding_offset,
            map_offset,
            map_size,
            write_flush_performed,
            write_flush_offset,
            write_flush_size,
            read_invalidate_performed,
            read_invalidate_offset,
            read_invalidate_size,
        ) = numbers
        expected_storage_size = ((len(bytes.fromhex(spec["initial"][resource])) + 3) // 4) * 4
        if not 0 <= memory_type_index < 32 or not (memory_type_bits & (1 << memory_type_index)):
            fail(f"{name}: selected memory type is not in the buffer's allowed memory type mask for {resource}")
        if not (memory_property_flags & 0x2):
            fail(f"{name}: selected memory type is not host-visible for {resource}")
        if not 0 <= memory_heap_index < 16 or memory_heap_size <= 0 or memory_requirement_alignment <= 0:
            fail(f"{name}: invalid memory heap or allocation alignment for {resource}")
        if (
            memory_requirement_size != allocation_size
            or allocation_size < expected_storage_size
            or storage_size != expected_storage_size
        ):
            fail(f"{name}: memory requirements/allocation/storage sizes disagree for {resource}")
        if buffer_usage_flags != 0x20 or sharing_mode_raw != 0:
            fail(f"{name}: unexpected buffer usage or sharing mode for {resource}")
        if binding_offset != 0 or map_offset != 0 or map_size != allocation_size:
            fail(f"{name}: buffer binding or mapped range mismatch for {resource}")
        coherent = bool(memory_property_flags & 0x4)
        expected_performed = 0 if coherent else 1
        expected_size = 0 if coherent else 0xFFFFFFFFFFFFFFFF
        if (
            write_flush_performed != expected_performed
            or write_flush_offset != 0
            or write_flush_size != expected_size
        ):
            fail(f"{name}: write flush evidence conflicts with memory coherency for {resource}")
        if (
            read_invalidate_performed != expected_performed
            or read_invalidate_offset != 0
            or read_invalidate_size != expected_size
        ):
            fail(f"{name}: read invalidate evidence conflicts with memory coherency for {resource}")
        profiles[resource] = {
            "memory_type_index": memory_type_index,
            "memory_type_bits": memory_type_bits,
            "memory_property_flags": memory_property_flags,
            "memory_heap_index": memory_heap_index,
            "memory_heap_flags": memory_heap_flags,
            "memory_heap_size": memory_heap_size,
            "memory_requirement_alignment": memory_requirement_alignment,
            "memory_requirement_size": memory_requirement_size,
            "allocation_size": allocation_size,
            "storage_size": storage_size,
            "buffer_usage_flags": buffer_usage_flags,
            "sharing_mode_raw": sharing_mode_raw,
            "binding_offset": binding_offset,
            "map_offset": map_offset,
            "map_size": map_size,
            "write_flush_performed": write_flush_performed,
            "write_flush_offset": write_flush_offset,
            "write_flush_size": write_flush_size,
            "read_invalidate_performed": read_invalidate_performed,
            "read_invalidate_offset": read_invalidate_offset,
            "read_invalidate_size": read_invalidate_size,
        }

    if set(profiles) != set(spec["initial"]):
        fail(f"{name}: resource memory profile set does not match fixture resources")

    memory_types: dict[int, tuple[int, int]] = {}
    memory_heaps: dict[int, tuple[int, int]] = {}
    for resource, profile in profiles.items():
        type_index = profile["memory_type_index"]
        type_identity = (
            profile["memory_property_flags"],
            profile["memory_heap_index"],
        )
        previous_type = memory_types.get(type_index)
        if previous_type is not None and previous_type != type_identity:
            fail(f"{name}: shared memory type {type_index} has inconsistent properties at {resource}")
        memory_types[type_index] = type_identity

        heap_index = profile["memory_heap_index"]
        heap_identity = (
            profile["memory_heap_flags"],
            profile["memory_heap_size"],
        )
        previous_heap = memory_heaps.get(heap_index)
        if previous_heap is not None and previous_heap != heap_identity:
            fail(f"{name}: shared memory heap {heap_index} has inconsistent properties at {resource}")
        memory_heaps[heap_index] = heap_identity

    fields = [f"resource_profile_count:{len(profiles)}"]
    names = [
        "memory_type_index",
        "memory_type_bits",
        "memory_property_flags",
        "memory_heap_index",
        "memory_heap_flags",
        "memory_heap_size",
        "memory_requirement_alignment",
        "memory_requirement_size",
        "allocation_size",
        "storage_size",
        "buffer_usage_flags",
        "sharing_mode_raw",
        "binding_offset",
        "map_offset",
        "map_size",
        "write_flush_performed",
        "write_flush_offset",
        "write_flush_size",
        "read_invalidate_performed",
        "read_invalidate_offset",
        "read_invalidate_size",
    ]
    for resource in sorted(profiles):
        fields.append("resource_profile")
        fields.append(f"resource={resource}")
        fields.extend(f"{field}={profiles[resource][field]}" for field in names)
    expected = sha256_len_prefixed(
        [field.encode("utf-8") for field in fields],
        b"symthaea.gpu-fabric.vulkan-memory-lowering.v1",
    )
    if values.get("memory_lowering_digest") != expected:
        fail(f"{name}: materialized Vulkan memory-lowering digest mismatch")


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
        if values.get("receipt_version") != "11":
            fail(f"{name}: receipt version mismatch")
        if int(values.get("node_count", "-1")) != spec["node_count"]:
            fail(f"{name}: node count mismatch")
        if int(values.get("barrier_count", "-1")) != spec["barrier_count"]:
            fail(f"{name}: barrier count mismatch")
        verify_materialized_barrier_lowering(values, spec, name)
        verify_materialized_submission_contract(values, spec, name)
        verify_materialized_dispatch_contract(values, spec, name)
        verify_memory_lowering_contract(values, lines, spec, name)
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

#!/usr/bin/env python3
"""Prospectively freeze the silicon pseudopotential for Matter Capsule 001.

This transition composes two independently checkable evidence objects:

1. an exact, content-addressed SSSP/1.3/PBE/precision download produced only by
   the pinned ``aiida-pseudo`` CLI; and
2. the exact ``ExecutionEnvironmentPreparedQualifiedV1`` receipt produced by the
   frozen Matter Reference Capsule environment.

Only after both predecessors validate does ``install-freeze`` install the SSSP
family, prove the complete installed family against official metadata, and
freeze the literal Si UPF bytes. No AiiDA ProcessNode and no Quantum ESPRESSO
calculation is created by this transition.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from datetime import datetime, timezone
from typing import Any, Iterable

FAMILY_VERSION = "1.3"
FAMILY_FUNCTIONAL = "PBE"
FAMILY_PROTOCOL = "precision"
FAMILY_LABEL = f"SSSP/{FAMILY_VERSION}/{FAMILY_FUNCTIONAL}/{FAMILY_PROTOCOL}"
ELEMENT = "Si"
BUNDLE_FILENAME = "SSSP_1.3_PBE_precision.aiida_pseudo"
DOWNLOAD_SCHEMA = "symthaea.matter.sssp-download-receipt/v2"
FREEZE_SCHEMA = "symthaea.matter.si-pseudopotential-freeze/v2"
ENVIRONMENT_SCHEMA = "symthaea.matter.reference-capsule-environment-qualification/v1"
ENVIRONMENT_AUTHORITY = "ExecutionEnvironmentPreparedQualifiedV1"
EXPECTED_ENVIRONMENT_HEAD = "7b9ec63a5166bb168c02a4375fcb2fd7c9a22cd2"
EXPECTED_ENVIRONMENT_PARENT = "53e32afe7aba5ae3426d2c66a75a2be466dc1bc6"
EXPECTED_UV_LOCK_SHA256 = "1ecb6c8a0752476c9731d0d44a4113c4ed1fabfee6e339724ee4534926bda646"
EXPECTED_NIXPKGS_REV = "9ae611a455b90cf061d8f332b977e387bda8e1ca"
EXPECTED_PYTHON_DISTRIBUTIONS = {
    "aiida-pseudo": "1.9.0",
    "aiida-core": "2.9.2",
    "aiida-quantumespresso": "5.0.0",
}
EXPECTED_INNER_NAMES = {"archive.tar.gz", "metadata.json", "configuration.json"}
MAX_JSON_BYTES = 16 * 1024 * 1024
MAX_BUNDLE_BYTES = 512 * 1024 * 1024
MAX_INNER_ARCHIVE_BYTES = 512 * 1024 * 1024
MAX_INNER_UNCOMPRESSED_BYTES = 1024 * 1024 * 1024
MAX_INNER_ENTRIES = 1024
MAX_INNER_FILE_BYTES = 64 * 1024 * 1024
AUTHORITY_DOWNLOAD = "PseudoBundleDownloadedOnly"
AUTHORITY_FREEZE = "PreExecutionPseudopotentialFrozen"


class FreezeError(RuntimeError):
    """Fail-closed pseudopotential freeze error."""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_iso(value: datetime) -> str:
    if value.tzinfo is None:
        raise FreezeError("timestamp is timezone-naive")
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _yyyymmdd(value: datetime) -> int:
    value = value.astimezone(timezone.utc)
    return value.year * 10_000 + value.month * 100 + value.day


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_json_sha256(value: Any) -> str:
    return _sha256_bytes(_json_bytes(value))


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise FreezeError(f"duplicate JSON object key refused: {key}")
        result[key] = value
    return result


def _loads_json(payload: bytes | str, *, source: str) -> Any:
    try:
        if isinstance(payload, bytes):
            payload = payload.decode("utf-8")
        return json.loads(payload, object_pairs_hook=_reject_duplicate_pairs)
    except FreezeError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FreezeError(f"invalid JSON in {source}: {error}") from error


def _write_json(path: Path, value: Any) -> str:
    payload = _json_bytes(value)
    if len(payload) > MAX_JSON_BYTES:
        raise FreezeError(f"JSON output exceeds {MAX_JSON_BYTES} bytes: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    digest = _sha256_bytes(payload)
    if _sha256_file(path) != digest:
        raise FreezeError(f"persisted JSON digest mismatch: {path}")
    return digest


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FreezeError(f"missing JSON file: {path}")
    if path.stat().st_size > MAX_JSON_BYTES:
        raise FreezeError(f"JSON input exceeds {MAX_JSON_BYTES} bytes: {path}")
    value = _loads_json(path.read_bytes(), source=str(path))
    if not isinstance(value, dict):
        raise FreezeError(f"expected JSON object: {path}")
    return value


def _prepare_empty_dir(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FreezeError(f"output directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _capsule_lock_path() -> Path:
    return Path(__file__).resolve().parent / "uv.lock"


def _require_capsule_environment() -> str:
    if os.environ.get("SYMTHAEA_MATTER_CAPSULE_SHELL") != "1":
        raise FreezeError("enter the pinned Matter capsule shell before pseudopotential acquisition")
    lock_path = _capsule_lock_path()
    if not lock_path.is_file():
        raise FreezeError("uv.lock is not committed; pseudopotential acquisition may not proceed")
    digest = _sha256_file(lock_path)
    if digest != EXPECTED_UV_LOCK_SHA256:
        raise FreezeError(f"unexpected committed uv.lock digest: {digest}")
    if shutil.which("aiida-pseudo") is None:
        raise FreezeError("aiida-pseudo CLI is unavailable in the locked capsule environment")
    return digest


def _distribution_versions() -> dict[str, str]:
    versions = {name: importlib.metadata.version(name) for name in EXPECTED_PYTHON_DISTRIBUTIONS}
    if versions != EXPECTED_PYTHON_DISTRIBUTIONS:
        raise FreezeError(f"unexpected frozen Python distribution identity: {versions}")
    return versions


def _run(command: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            cwd=cwd,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as error:
        message = error.stderr.strip() or error.stdout.strip() or str(error)
        raise FreezeError(f"command failed: {' '.join(command)}: {message}") from error


def _safe_tar_path(name: str) -> PurePosixPath:
    if "\\" in name:
        raise FreezeError(f"backslash-containing tar path refused: {name}")
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise FreezeError(f"unsafe tar member path refused: {name}")
    return path


def _safe_inner_archive(payload: bytes) -> dict[str, Any]:
    if not payload or len(payload) > MAX_INNER_ARCHIVE_BYTES:
        raise FreezeError(f"unexpected inner pseudopotential archive size: {len(payload)}")

    file_manifest: list[dict[str, Any]] = []
    seen: set[str] = set()
    total_uncompressed = 0
    try:
        with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as handle:
            entries = handle.getmembers()
            if not entries or len(entries) > MAX_INNER_ENTRIES:
                raise FreezeError(f"unexpected inner archive entry count: {len(entries)}")
            for entry in entries:
                _safe_tar_path(entry.name)
                if entry.name in seen:
                    raise FreezeError(f"duplicate inner archive member refused: {entry.name}")
                seen.add(entry.name)
                if entry.issym() or entry.islnk() or entry.isdev():
                    raise FreezeError(f"link/device inner archive member refused: {entry.name}")
                if entry.isdir():
                    continue
                if not entry.isfile():
                    raise FreezeError(f"unsupported inner archive member type: {entry.name}")
                if entry.size <= 0 or entry.size > MAX_INNER_FILE_BYTES:
                    raise FreezeError(f"unexpected inner archive member size: {entry.name}={entry.size}")
                total_uncompressed += entry.size
                if total_uncompressed > MAX_INNER_UNCOMPRESSED_BYTES:
                    raise FreezeError("inner archive exceeds uncompressed-size limit")
                extracted = handle.extractfile(entry)
                if extracted is None:
                    raise FreezeError(f"cannot read inner archive member: {entry.name}")
                digest = hashlib.sha256()
                observed = 0
                while chunk := extracted.read(1024 * 1024):
                    observed += len(chunk)
                    if observed > entry.size or observed > MAX_INNER_FILE_BYTES:
                        raise FreezeError(f"inner archive member expanded beyond declared bounds: {entry.name}")
                    digest.update(chunk)
                if observed != entry.size:
                    raise FreezeError(f"inner archive member size mismatch: {entry.name}")
                file_manifest.append({"name": entry.name, "size": entry.size, "sha256": digest.hexdigest()})
    except FreezeError:
        raise
    except (tarfile.TarError, OSError) as error:
        raise FreezeError(f"cannot inspect inner pseudopotential archive: {error}") from error

    if not file_manifest:
        raise FreezeError("inner pseudopotential archive contains no files")
    file_manifest.sort(key=lambda item: item["name"])
    return {
        "compressed_sha256": _sha256_bytes(payload),
        "compressed_bytes": len(payload),
        "file_count": len(file_manifest),
        "total_uncompressed_bytes": total_uncompressed,
        "manifest_sha256": _canonical_json_sha256(file_manifest),
    }


def _safe_bundle_members(bundle: Path) -> dict[str, bytes]:
    if not bundle.is_file():
        raise FreezeError(f"missing aiida-pseudo bundle: {bundle}")
    size = bundle.stat().st_size
    if size <= 0 or size > MAX_BUNDLE_BYTES:
        raise FreezeError(f"unexpected aiida-pseudo bundle size: {size}")

    members: dict[str, bytes] = {}
    try:
        with tarfile.open(bundle, "r") as handle:
            entries = handle.getmembers()
            names = {entry.name for entry in entries}
            if len(entries) != len(names):
                raise FreezeError("duplicate outer aiida-pseudo bundle members refused")
            if names != EXPECTED_INNER_NAMES:
                raise FreezeError(f"unexpected aiida-pseudo bundle members: {sorted(names)}")
            for entry in entries:
                _safe_tar_path(entry.name)
                if not entry.isfile() or entry.issym() or entry.islnk() or entry.isdev():
                    raise FreezeError(f"unsafe aiida-pseudo bundle member: {entry.name}")
                if Path(entry.name).name != entry.name:
                    raise FreezeError(f"nested aiida-pseudo bundle member refused: {entry.name}")
                extracted = handle.extractfile(entry)
                if extracted is None:
                    raise FreezeError(f"cannot read aiida-pseudo bundle member: {entry.name}")
                payload = extracted.read()
                if not payload:
                    raise FreezeError(f"empty aiida-pseudo bundle member: {entry.name}")
                members[entry.name] = payload
    except FreezeError:
        raise
    except (tarfile.TarError, OSError) as error:
        raise FreezeError(f"cannot inspect aiida-pseudo bundle: {error}") from error
    return members


def _parse_bundle(bundle: Path) -> dict[str, Any]:
    if bundle.name != BUNDLE_FILENAME:
        raise FreezeError(f"unexpected aiida-pseudo bundle filename: {bundle.name}")
    members = _safe_bundle_members(bundle)
    configuration = _loads_json(members["configuration.json"], source="configuration.json")
    metadata = _loads_json(members["metadata.json"], source="metadata.json")

    expected_configuration = {
        "version": FAMILY_VERSION,
        "functional": FAMILY_FUNCTIONAL,
        "protocol": FAMILY_PROTOCOL,
    }
    if configuration != expected_configuration:
        raise FreezeError(f"SSSP configuration mismatch: {configuration!r}")
    if not isinstance(metadata, dict) or ELEMENT not in metadata:
        raise FreezeError("SSSP metadata does not contain a silicon entry")

    md5_by_element: dict[str, str] = {}
    cutoffs_by_element: dict[str, dict[str, float]] = {}
    for element, values in metadata.items():
        if not isinstance(element, str) or re.fullmatch(r"[A-Z][a-z]?", element) is None:
            raise FreezeError(f"invalid element key in SSSP metadata: {element!r}")
        if not isinstance(values, dict):
            raise FreezeError(f"invalid SSSP metadata entry for {element}")
        md5 = str(values.get("md5", "")).lower()
        if len(md5) != 32 or any(char not in "0123456789abcdef" for char in md5):
            raise FreezeError(f"SSSP metadata has an invalid MD5 for {element}")
        try:
            cutoff_wfc = float(values["cutoff_wfc"])
            cutoff_rho = float(values["cutoff_rho"])
        except (KeyError, TypeError, ValueError) as error:
            raise FreezeError(f"SSSP metadata has invalid recommended cutoffs for {element}") from error
        if not (cutoff_wfc > 0.0 and cutoff_rho > 0.0):
            raise FreezeError(f"SSSP recommended cutoffs must be positive for {element}")
        md5_by_element[element] = md5
        cutoffs_by_element[element] = {"cutoff_wfc": cutoff_wfc, "cutoff_rho": cutoff_rho}

    inner_archive = _safe_inner_archive(members["archive.tar.gz"])
    if inner_archive["file_count"] != len(metadata):
        raise FreezeError(
            f"inner archive/metadata element-count mismatch: files={inner_archive['file_count']} metadata={len(metadata)}"
        )

    silicon = metadata[ELEMENT]
    return {
        "configuration": configuration,
        "member_sha256": {name: _sha256_bytes(payload) for name, payload in sorted(members.items())},
        "metadata_canonical_sha256": _canonical_json_sha256(metadata),
        "metadata_element_count": len(metadata),
        "metadata_elements": sorted(metadata),
        "metadata_md5_by_element": md5_by_element,
        "metadata_md5_map_sha256": _canonical_json_sha256(md5_by_element),
        "metadata_cutoffs_by_element": cutoffs_by_element,
        "metadata_cutoffs_sha256": _canonical_json_sha256(cutoffs_by_element),
        "inner_archive": inner_archive,
        "silicon_metadata": {
            "canonical_sha256": _canonical_json_sha256(silicon),
            "md5": md5_by_element[ELEMENT],
            "cutoff_wfc_ry": cutoffs_by_element[ELEMENT]["cutoff_wfc"],
            "cutoff_rho_ry": cutoffs_by_element[ELEMENT]["cutoff_rho"],
        },
    }


def _download_evidence(parsed: dict[str, Any]) -> dict[str, Any]:
    return {
        "configuration": parsed["configuration"],
        "inner_member_sha256": parsed["member_sha256"],
        "metadata_canonical_sha256": parsed["metadata_canonical_sha256"],
        "metadata_element_count": parsed["metadata_element_count"],
        "metadata_elements": parsed["metadata_elements"],
        "metadata_md5_map_sha256": parsed["metadata_md5_map_sha256"],
        "metadata_cutoffs_sha256": parsed["metadata_cutoffs_sha256"],
        "inner_archive": parsed["inner_archive"],
        "silicon_metadata": parsed["silicon_metadata"],
    }


def _validate_download_receipt(
    receipt: dict[str, Any], bundle: Path, *, uv_lock_sha256: str, versions: dict[str, str]
) -> dict[str, Any]:
    if receipt.get("schema_version") != DOWNLOAD_SCHEMA:
        raise FreezeError("unsupported or malformed pseudopotential download receipt")
    if receipt.get("authority") != AUTHORITY_DOWNLOAD:
        raise FreezeError("download receipt authority mismatch")
    if receipt.get("family_label") != FAMILY_LABEL or receipt.get("element") != ELEMENT:
        raise FreezeError("download receipt family/element does not match Capsule 001 contract")
    if receipt.get("uv_lock_sha256") != uv_lock_sha256:
        raise FreezeError("download receipt was created under a different uv.lock")
    if receipt.get("python_distributions") != versions:
        raise FreezeError("download receipt Python distribution versions differ from install environment")
    actual_bundle_sha = _sha256_file(bundle)
    if receipt.get("bundle_sha256") != actual_bundle_sha:
        raise FreezeError("downloaded .aiida_pseudo bundle SHA-256 does not match receipt")
    parsed = _parse_bundle(bundle)
    if receipt.get("bundle_evidence") != _download_evidence(parsed):
        raise FreezeError("download receipt evidence does not match exact bundle content")
    return parsed


def _validate_environment_receipt(
    path: Path,
    *,
    profile: str,
    uv_lock_sha256: str,
    versions: dict[str, str],
    qe_store: str,
) -> tuple[dict[str, Any], str]:
    receipt = _read_json(path)
    if receipt.get("schema_version") != ENVIRONMENT_SCHEMA:
        raise FreezeError("unsupported environment qualification receipt schema")
    if receipt.get("authority") != ENVIRONMENT_AUTHORITY:
        raise FreezeError("environment qualification authority mismatch")
    if receipt.get("subject_head") != EXPECTED_ENVIRONMENT_HEAD:
        raise FreezeError("environment qualification subject head mismatch")
    if receipt.get("parent_head") != EXPECTED_ENVIRONMENT_PARENT:
        raise FreezeError("environment qualification parent head mismatch")

    source = receipt.get("source")
    runtime = receipt.get("runtime")
    aiida = receipt.get("aiida")
    scientific = receipt.get("scientific_execution")
    if not all(isinstance(value, dict) for value in (source, runtime, aiida, scientific)):
        raise FreezeError("environment qualification receipt structure is malformed")
    if source.get("uv_lock_sha256") != uv_lock_sha256:
        raise FreezeError("environment receipt is bound to a different uv.lock")
    if source.get("nixpkgs_rev") != EXPECTED_NIXPKGS_REV:
        raise FreezeError("environment receipt is bound to a different nixpkgs revision")
    if runtime.get("qe_store") != qe_store:
        raise FreezeError("environment receipt QE store differs from current capsule environment")

    consumer = runtime.get("lock_consumer")
    if not isinstance(consumer, dict):
        raise FreezeError("environment receipt is missing lock-consumer identity")
    if consumer.get("python_distributions") != versions:
        raise FreezeError("environment receipt Python distributions differ from current freeze environment")
    if consumer.get("uv", {}).get("version") != "uv 0.11.19":
        raise FreezeError("environment receipt uv consumer identity mismatch")
    if consumer.get("python", {}).get("version") != "3.13.13":
        raise FreezeError("environment receipt Python consumer identity mismatch")
    if consumer.get("uv_lock_check_read_only") is not True:
        raise FreezeError("environment receipt lacks read-only lock-consumption proof")
    if consumer.get("uv_lock_sha256_after_check") != uv_lock_sha256:
        raise FreezeError("environment receipt post-check lock digest mismatch")

    if aiida.get("profile") != profile:
        raise FreezeError("environment receipt AiiDA profile mismatch")
    if aiida.get("storage_backend") != "core.sqlite_dos":
        raise FreezeError("environment receipt storage backend mismatch")
    if aiida.get("process_control_backend") is not None:
        raise FreezeError("environment receipt unexpectedly has a process-control backend")
    if aiida.get("caching_default_enabled") is not False:
        raise FreezeError("environment receipt does not prove disabled calculation caching")
    if aiida.get("process_node_count") != 0 or aiida.get("calcjob_node_count") != 0:
        raise FreezeError("environment receipt is not a fresh zero-process environment")

    expected_scientific = {
        "pseudopotential_selected": False,
        "pseudopotential_installed": False,
        "sssp_download_executed": False,
        "qe_calculation_executed": False,
        "process_node_created": False,
        "scientific_result_observed": False,
    }
    if scientific != expected_scientific:
        raise FreezeError("environment receipt crossed the pre-pseudopotential scientific boundary")
    if receipt.get("next_required_transition") != "environment-bound-prospective-si-pseudopotential-freeze":
        raise FreezeError("environment receipt does not name this transition as its successor")
    return receipt, _sha256_file(path)


def _parse_patch_version(output: str) -> str:
    match = re.search(r"Latest patch version found:\s*([0-9]+(?:\.[0-9]+){1,2})", output)
    if match is None:
        raise FreezeError("aiida-pseudo download output did not expose the resolved SSSP patch version")
    return match.group(1)


def _download(args: argparse.Namespace) -> None:
    uv_lock_sha256 = _require_capsule_environment()
    versions = _distribution_versions()
    out = Path(args.output).resolve()
    _prepare_empty_dir(out)
    created = _utc_now()

    command = [
        "aiida-pseudo",
        "install",
        "sssp",
        "--version",
        FAMILY_VERSION,
        "--functional",
        FAMILY_FUNCTIONAL,
        "--protocol",
        FAMILY_PROTOCOL,
        "--download-only",
    ]
    completed = _run(command, cwd=out)
    patch_version = _parse_patch_version(completed.stdout + "\n" + completed.stderr)

    bundle = out / BUNDLE_FILENAME
    parsed = _parse_bundle(bundle)
    receipt = {
        "schema_version": DOWNLOAD_SCHEMA,
        "created_utc": _utc_iso(created),
        "created_utc_yyyymmdd": _yyyymmdd(created),
        "family_label": FAMILY_LABEL,
        "requested_minor_version": FAMILY_VERSION,
        "resolved_patch_version": patch_version,
        "element": ELEMENT,
        "command_contract": command,
        "bundle_filename": bundle.name,
        "bundle_sha256": _sha256_file(bundle),
        "bundle_evidence": _download_evidence(parsed),
        "uv_lock_sha256": uv_lock_sha256,
        "python_distributions": versions,
        "authority": AUTHORITY_DOWNLOAD,
        "limitations": [
            "network acquisition is content-addressed but does not itself qualify an AiiDA environment or install a family",
            "SSSP latest patch resolution is time-dependent; the resolved patch version and exact bundle bytes are retained",
            "recommended cutoffs are source metadata and do not replace Symthaea numerical convergence evidence",
            "thermodynamic, dynamical, experimental, and replication claims are unaffected",
        ],
    }
    _write_json(out / "download-receipt.json", receipt)
    print(out / "download-receipt.json")


def _normalize_cutoffs(value: dict[str, Any]) -> dict[str, dict[str, float]]:
    normalized: dict[str, dict[str, float]] = {}
    for element, cutoffs in value.items():
        if not isinstance(cutoffs, dict) or set(cutoffs) != {"cutoff_wfc", "cutoff_rho"}:
            raise FreezeError(f"installed cutoff structure is invalid for {element}")
        normalized[element] = {
            "cutoff_wfc": float(cutoffs["cutoff_wfc"]),
            "cutoff_rho": float(cutoffs["cutoff_rho"]),
        }
    return normalized


def _install_freeze(args: argparse.Namespace) -> None:
    uv_lock_sha256 = _require_capsule_environment()
    versions = _distribution_versions()
    bundle = Path(args.bundle).resolve()
    download_receipt_path = Path(args.download_receipt).resolve()
    environment_receipt_path = Path(args.environment_receipt).resolve()
    out = Path(args.output).resolve()
    _prepare_empty_dir(out)

    download_receipt = _read_json(download_receipt_path)
    parsed = _validate_download_receipt(
        download_receipt,
        bundle,
        uv_lock_sha256=uv_lock_sha256,
        versions=versions,
    )

    qe_store = str(Path(os.environ.get("SYMTHAEA_QE_STORE", "")).resolve())
    if not qe_store.startswith("/nix/store/"):
        raise FreezeError("current QE store identity is unavailable")
    environment_receipt, environment_receipt_sha256 = _validate_environment_receipt(
        environment_receipt_path,
        profile=args.profile,
        uv_lock_sha256=uv_lock_sha256,
        versions=versions,
        qe_store=qe_store,
    )

    from aiida import load_profile, orm
    from aiida.common.exceptions import NotExistent
    from aiida.orm import QueryBuilder
    from aiida_pseudo.groups.family.sssp import SsspFamily

    profile = load_profile(args.profile)
    if profile.name != args.profile:
        raise FreezeError("loaded AiiDA profile differs from environment-qualified profile")
    process_count_before = QueryBuilder().append(orm.ProcessNode).count()
    calcjob_count_before = QueryBuilder().append(orm.CalcJobNode).count()
    if process_count_before != 0 or calcjob_count_before != 0:
        raise FreezeError(
            "pseudopotential must be frozen before any AiiDA process exists; "
            f"found ProcessNode={process_count_before}, CalcJobNode={calcjob_count_before}"
        )

    try:
        SsspFamily.collection.get(label=FAMILY_LABEL)
    except NotExistent:
        pass
    else:
        raise FreezeError(f"SSSP family is already installed in profile: {FAMILY_LABEL}")

    command = [
        "aiida-pseudo",
        "--profile",
        args.profile,
        "install",
        "sssp",
        "--from-download",
        str(bundle),
    ]
    _run(command)

    family = SsspFamily.collection.get(label=FAMILY_LABEL)
    family_elements = sorted(family.elements)
    expected_elements = parsed["metadata_elements"]
    if family_elements != expected_elements:
        raise FreezeError("installed SSSP family element set does not match downloaded official metadata")

    installed_md5_by_element: dict[str, str] = {}
    for element in family_elements:
        pseudo = family.get_pseudo(element)
        md5 = str(pseudo.md5).lower()
        expected_md5 = parsed["metadata_md5_by_element"][element]
        if md5 != expected_md5:
            raise FreezeError(f"installed {element} pseudo MD5 does not match official SSSP metadata")
        installed_md5_by_element[element] = md5
    installed_md5_map_sha256 = _canonical_json_sha256(installed_md5_by_element)
    if installed_md5_map_sha256 != parsed["metadata_md5_map_sha256"]:
        raise FreezeError("installed SSSP family MD5-map digest mismatch")

    if family.get_cutoff_stringencies() != ("normal",):
        raise FreezeError(f"unexpected installed SSSP cutoff stringencies: {family.get_cutoff_stringencies()}")
    if family.get_default_stringency() != "normal":
        raise FreezeError("installed SSSP default cutoff stringency is not normal")
    if family.get_cutoffs_unit("normal") != "Ry":
        raise FreezeError("installed SSSP cutoff unit is not Ry")
    installed_cutoffs = _normalize_cutoffs(family.get_cutoffs("normal"))
    if installed_cutoffs != parsed["metadata_cutoffs_by_element"]:
        raise FreezeError("installed SSSP family cutoffs do not match downloaded official metadata")
    installed_cutoffs_sha256 = _canonical_json_sha256(installed_cutoffs)
    if installed_cutoffs_sha256 != parsed["metadata_cutoffs_sha256"]:
        raise FreezeError("installed SSSP cutoff-map digest mismatch")

    pseudo = family.get_pseudo(element=ELEMENT)
    if pseudo.element != ELEMENT:
        raise FreezeError(f"installed pseudo element mismatch: {pseudo.element!r}")
    with pseudo.open(mode="rb") as handle:
        upf_bytes = handle.read()
    if not upf_bytes:
        raise FreezeError("installed silicon UPF is empty")
    upf_sha256 = _sha256_bytes(upf_bytes)
    metadata_md5 = parsed["silicon_metadata"]["md5"]
    actual_md5 = hashlib.md5(upf_bytes, usedforsecurity=False).hexdigest()
    if actual_md5 != metadata_md5 or str(pseudo.md5).lower() != metadata_md5:
        raise FreezeError("installed Si UPF MD5 does not match official SSSP metadata")

    cutoff_wfc, cutoff_rho = family.get_recommended_cutoffs(elements=ELEMENT, unit="Ry")
    recommended = {"cutoff_wfc_ry": float(cutoff_wfc), "cutoff_rho_ry": float(cutoff_rho)}
    expected_recommended = {
        "cutoff_wfc_ry": parsed["silicon_metadata"]["cutoff_wfc_ry"],
        "cutoff_rho_ry": parsed["silicon_metadata"]["cutoff_rho_ry"],
    }
    if recommended != expected_recommended:
        raise FreezeError("installed Si recommended cutoffs do not match downloaded metadata")

    process_count_after = QueryBuilder().append(orm.ProcessNode).count()
    calcjob_count_after = QueryBuilder().append(orm.CalcJobNode).count()
    if process_count_after != 0 or calcjob_count_after != 0:
        raise FreezeError("pseudopotential installation unexpectedly created AiiDA process records")

    upf_path = out / "Si.upf"
    upf_path.write_bytes(upf_bytes)
    if _sha256_file(upf_path) != upf_sha256:
        raise FreezeError("exported Si UPF bytes failed SHA-256 round-trip")

    created = _utc_now()
    freeze = {
        "schema_version": FREEZE_SCHEMA,
        "created_utc": _utc_iso(created),
        "created_utc_yyyymmdd": _yyyymmdd(created),
        "predecessor_environment": {
            "receipt_sha256": environment_receipt_sha256,
            "schema_version": environment_receipt["schema_version"],
            "authority": environment_receipt["authority"],
            "subject_head": environment_receipt["subject_head"],
            "parent_head": environment_receipt["parent_head"],
            "profile": environment_receipt["aiida"]["profile"],
            "uv_lock_sha256": environment_receipt["source"]["uv_lock_sha256"],
            "nixpkgs_rev": environment_receipt["source"]["nixpkgs_rev"],
            "qe_store": environment_receipt["runtime"]["qe_store"],
            "python_distributions": environment_receipt["runtime"]["lock_consumer"]["python_distributions"],
        },
        "profile": args.profile,
        "family": {
            "label": family.label,
            "uuid": str(family.uuid),
            "python_class": f"{family.__class__.__module__}.{family.__class__.__name__}",
            "description": str(family.description),
            "element_count": len(family_elements),
            "elements": family_elements,
            "installed_md5_map_sha256": installed_md5_map_sha256,
            "installed_cutoffs_sha256": installed_cutoffs_sha256,
            "cutoff_stringency": "normal",
            "cutoff_unit": "Ry",
        },
        "element": ELEMENT,
        "pseudo": {
            "uuid": str(pseudo.uuid),
            "pk": int(pseudo.pk),
            "filename": str(pseudo.filename),
            "md5": actual_md5,
            "sha256": upf_sha256,
            "byte_length": len(upf_bytes),
            "official_metadata_canonical_sha256": parsed["silicon_metadata"]["canonical_sha256"],
            "exported_artifact": "Si.upf",
        },
        "source_bundle": {
            "path_basename": bundle.name,
            "sha256": _sha256_file(bundle),
            "download_receipt_sha256": _sha256_file(download_receipt_path),
            "resolved_patch_version": download_receipt["resolved_patch_version"],
            "inner_member_sha256": parsed["member_sha256"],
            "metadata_canonical_sha256": parsed["metadata_canonical_sha256"],
            "metadata_element_count": parsed["metadata_element_count"],
            "metadata_md5_map_sha256": parsed["metadata_md5_map_sha256"],
            "metadata_cutoffs_sha256": parsed["metadata_cutoffs_sha256"],
            "inner_archive": parsed["inner_archive"],
        },
        "recommended_cutoffs_reference_only": recommended,
        "process_graph": {
            "process_nodes_before_install": process_count_before,
            "calcjob_nodes_before_install": calcjob_count_before,
            "process_nodes_after_install": process_count_after,
            "calcjob_nodes_after_install": calcjob_count_after,
        },
        "uv_lock_sha256": uv_lock_sha256,
        "python_distributions": versions,
        "installation_command_contract": command,
        "scientific_execution": {
            "sssp_download_executed": True,
            "pseudopotential_installed": True,
            "pseudopotential_selected": True,
            "qe_calculation_executed": False,
            "process_node_created": False,
            "scientific_result_observed": False,
        },
        "authority": AUTHORITY_FREEZE,
        "next_required_transition": "si-001-frozen-scientific-execution",
        "limitations": [
            "this receipt freezes a complete SSSP family and literal Si input before any AiiDA process; it does not execute QE",
            "SSSP recommended cutoffs are reference metadata only; production settings require independent numerical convergence",
            "pseudopotential source/integrity validation remains computational evidence, not experimental validation",
            "thermodynamic phase coverage, dynamical stability, experiment, and replication remain unqualified",
        ],
    }
    freeze_path = out / "freeze-receipt.json"
    _write_json(freeze_path, freeze)
    print(freeze_path)


def _synthetic_environment_receipt(
    *, profile: str, uv_lock_sha256: str, versions: dict[str, str], qe_store: str
) -> dict[str, Any]:
    return {
        "schema_version": ENVIRONMENT_SCHEMA,
        "authority": ENVIRONMENT_AUTHORITY,
        "subject_head": EXPECTED_ENVIRONMENT_HEAD,
        "parent_head": EXPECTED_ENVIRONMENT_PARENT,
        "source": {"uv_lock_sha256": uv_lock_sha256, "nixpkgs_rev": EXPECTED_NIXPKGS_REV},
        "runtime": {
            "qe_store": qe_store,
            "lock_consumer": {
                "uv": {"version": "uv 0.11.19"},
                "python": {"version": "3.13.13"},
                "python_distributions": versions,
                "uv_lock_check_read_only": True,
                "uv_lock_sha256_after_check": uv_lock_sha256,
            },
        },
        "aiida": {
            "profile": profile,
            "storage_backend": "core.sqlite_dos",
            "process_control_backend": None,
            "caching_default_enabled": False,
            "process_node_count": 0,
            "calcjob_node_count": 0,
        },
        "scientific_execution": {
            "pseudopotential_selected": False,
            "pseudopotential_installed": False,
            "sssp_download_executed": False,
            "qe_calculation_executed": False,
            "process_node_created": False,
            "scientific_result_observed": False,
        },
        "next_required_transition": "environment-bound-prospective-si-pseudopotential-freeze",
    }


def _self_test() -> None:
    configuration = {
        "version": FAMILY_VERSION,
        "functional": FAMILY_FUNCTIONAL,
        "protocol": FAMILY_PROTOCOL,
    }
    metadata = {
        "C": {"md5": "fedcba9876543210fedcba9876543210", "cutoff_wfc": 55.0, "cutoff_rho": 440.0},
        ELEMENT: {"md5": "0123456789abcdef0123456789abcdef", "cutoff_wfc": 42.0, "cutoff_rho": 336.0},
    }
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)

        inner_archive = root / "archive.tar.gz"
        with tarfile.open(inner_archive, "w:gz") as handle:
            for name, payload in (("C.upf", b"synthetic-carbon-upf"), ("Si.upf", b"synthetic-silicon-upf")):
                info = tarfile.TarInfo(name=name)
                info.size = len(payload)
                handle.addfile(info, io.BytesIO(payload))

        (root / "configuration.json").write_bytes(_json_bytes(configuration))
        (root / "metadata.json").write_bytes(_json_bytes(metadata))
        bundle = root / BUNDLE_FILENAME
        with tarfile.open(bundle, "w") as handle:
            for name in sorted(EXPECTED_INNER_NAMES):
                handle.add(root / name, arcname=name)

        parsed = _parse_bundle(bundle)
        assert parsed["configuration"] == configuration
        assert parsed["metadata_element_count"] == 2
        assert parsed["inner_archive"]["file_count"] == 2
        assert parsed["silicon_metadata"]["md5"] == metadata[ELEMENT]["md5"]
        assert parsed["silicon_metadata"]["cutoff_wfc_ry"] == 42.0
        assert set(parsed["member_sha256"]) == EXPECTED_INNER_NAMES

        download_receipt = {
            "schema_version": DOWNLOAD_SCHEMA,
            "authority": AUTHORITY_DOWNLOAD,
            "family_label": FAMILY_LABEL,
            "element": ELEMENT,
            "bundle_sha256": _sha256_file(bundle),
            "bundle_evidence": _download_evidence(parsed),
            "uv_lock_sha256": "a" * 64,
            "python_distributions": {"fixture": "1"},
        }
        _validate_download_receipt(
            download_receipt,
            bundle,
            uv_lock_sha256="a" * 64,
            versions={"fixture": "1"},
        )

        environment_path = root / "environment.json"
        fixture_versions = {"fixture": "1"}
        fixture_environment = _synthetic_environment_receipt(
            profile="fixture-profile",
            uv_lock_sha256="a" * 64,
            versions=fixture_versions,
            qe_store="/nix/store/fixture-quantum-espresso",
        )
        _write_json(environment_path, fixture_environment)
        _, environment_digest = _validate_environment_receipt(
            environment_path,
            profile="fixture-profile",
            uv_lock_sha256="a" * 64,
            versions=fixture_versions,
            qe_store="/nix/store/fixture-quantum-espresso",
        )
        assert environment_digest == _sha256_file(environment_path)

        malicious_inner = io.BytesIO()
        with tarfile.open(fileobj=malicious_inner, mode="w:gz") as handle:
            payload = b"no"
            info = tarfile.TarInfo(name="../escape.upf")
            info.size = len(payload)
            handle.addfile(info, io.BytesIO(payload))
        try:
            _safe_inner_archive(malicious_inner.getvalue())
        except FreezeError:
            pass
        else:
            raise AssertionError("inner archive traversal must fail closed")

        bad_outer = root / "bad.aiida_pseudo"
        extra = root / "unexpected.txt"
        extra.write_text("no", encoding="utf-8")
        with tarfile.open(bad_outer, "w") as handle:
            for name in sorted(EXPECTED_INNER_NAMES):
                handle.add(root / name, arcname=name)
            handle.add(extra, arcname=extra.name)
        try:
            _safe_bundle_members(bad_outer)
        except FreezeError:
            pass
        else:
            raise AssertionError("extra outer bundle members must fail closed")

    print("Environment-bound Si pseudopotential freeze self-test: PASS")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    download = sub.add_parser("download", help="download and content-address the exact SSSP bundle")
    download.add_argument("--output", required=True)

    freeze = sub.add_parser(
        "install-freeze",
        help="bind qualified environment + exact SSSP bundle and freeze Si before any process",
    )
    freeze.add_argument("--profile", required=True)
    freeze.add_argument("--environment-receipt", required=True)
    freeze.add_argument("--bundle", required=True)
    freeze.add_argument("--download-receipt", required=True)
    freeze.add_argument("--output", required=True)

    sub.add_parser("self-test")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    try:
        if args.command == "download":
            _download(args)
        elif args.command == "install-freeze":
            _install_freeze(args)
        else:
            _self_test()
        return 0
    except FreezeError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

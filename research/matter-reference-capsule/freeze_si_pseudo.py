#!/usr/bin/env python3
"""Prospectively freeze the silicon pseudopotential for Matter Capsule 001.

The contract deliberately separates network acquisition from AiiDA installation:

1. ``download`` asks the official ``aiida-pseudo`` CLI for exactly
   ``SSSP/1.3/PBE/precision`` in ``--download-only`` mode, hashes the resulting
   ``.aiida_pseudo`` bundle and each required inner member, and writes a download
   receipt. No AiiDA profile is touched.
2. ``install-freeze`` verifies that exact bundle/receipt, requires a fresh AiiDA
   process graph (zero ``ProcessNode`` records), installs from the frozen bundle
   into the explicitly named profile, selects exactly Si, hashes and exports the
   actual UPF bytes stored in AiiDA, records the official metadata MD5 and
   recommended cutoffs, and writes the pre-execution freeze receipt.

The recommended SSSP cutoffs are metadata only. Symthaea's independent periodic
energy convergence gate remains authoritative for numerical adequacy.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
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
DOWNLOAD_SCHEMA = "symthaea.matter.sssp-download-receipt/v1"
FREEZE_SCHEMA = "symthaea.matter.si-pseudopotential-freeze/v1"
EXPECTED_INNER_NAMES = {"archive.tar.gz", "metadata.json", "configuration.json"}
MAX_JSON_BYTES = 16 * 1024 * 1024
MAX_BUNDLE_BYTES = 512 * 1024 * 1024
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


def _read_json(path: Path) -> Any:
    if not path.is_file():
        raise FreezeError(f"missing JSON file: {path}")
    if path.stat().st_size > MAX_JSON_BYTES:
        raise FreezeError(f"JSON input exceeds {MAX_JSON_BYTES} bytes: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FreezeError(f"cannot read JSON {path}: {error}") from error


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
    if shutil.which("aiida-pseudo") is None:
        raise FreezeError("aiida-pseudo CLI is unavailable in the locked capsule environment")
    return _sha256_file(lock_path)


def _distribution_versions() -> dict[str, str]:
    return {
        "aiida-pseudo": importlib.metadata.version("aiida-pseudo"),
        "aiida-core": importlib.metadata.version("aiida-core"),
        "aiida-quantumespresso": importlib.metadata.version("aiida-quantumespresso"),
    }


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
            if names != EXPECTED_INNER_NAMES:
                raise FreezeError(f"unexpected aiida-pseudo bundle members: {sorted(names)}")
            for entry in entries:
                if not entry.isfile() or entry.issym() or entry.islnk():
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
    except (tarfile.TarError, OSError) as error:
        raise FreezeError(f"cannot inspect aiida-pseudo bundle: {error}") from error
    return members


def _parse_bundle(bundle: Path) -> dict[str, Any]:
    if bundle.name != BUNDLE_FILENAME:
        raise FreezeError(f"unexpected aiida-pseudo bundle filename: {bundle.name}")
    members = _safe_bundle_members(bundle)
    try:
        configuration = json.loads(members["configuration.json"].decode("utf-8"))
        metadata = json.loads(members["metadata.json"].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FreezeError(f"invalid SSSP bundle metadata/configuration: {error}") from error

    expected_configuration = {
        "version": FAMILY_VERSION,
        "functional": FAMILY_FUNCTIONAL,
        "protocol": FAMILY_PROTOCOL,
    }
    if configuration != expected_configuration:
        raise FreezeError(f"SSSP configuration mismatch: {configuration!r}")
    if not isinstance(metadata, dict) or ELEMENT not in metadata or not isinstance(metadata[ELEMENT], dict):
        raise FreezeError("SSSP metadata does not contain a silicon entry")

    silicon = metadata[ELEMENT]
    md5 = str(silicon.get("md5", ""))
    if len(md5) != 32 or any(char not in "0123456789abcdefABCDEF" for char in md5):
        raise FreezeError("SSSP silicon metadata has an invalid MD5")
    try:
        cutoff_wfc = float(silicon["cutoff_wfc"])
        cutoff_rho = float(silicon["cutoff_rho"])
    except (KeyError, TypeError, ValueError) as error:
        raise FreezeError("SSSP silicon metadata has invalid recommended cutoffs") from error
    if not (cutoff_wfc > 0.0 and cutoff_rho > 0.0):
        raise FreezeError("SSSP silicon recommended cutoffs must be positive")

    return {
        "configuration": configuration,
        "member_sha256": {name: _sha256_bytes(payload) for name, payload in sorted(members.items())},
        "silicon_metadata": {
            "md5": md5.lower(),
            "cutoff_wfc_ry": cutoff_wfc,
            "cutoff_rho_ry": cutoff_rho,
        },
    }


def _validate_download_receipt(
    receipt: dict[str, Any], bundle: Path, *, uv_lock_sha256: str, versions: dict[str, str]
) -> dict[str, Any]:
    if not isinstance(receipt, dict) or receipt.get("schema_version") != DOWNLOAD_SCHEMA:
        raise FreezeError("unsupported or malformed pseudopotential download receipt")
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
    if receipt.get("configuration") != parsed["configuration"]:
        raise FreezeError("download receipt configuration does not match bundle")
    if receipt.get("inner_member_sha256") != parsed["member_sha256"]:
        raise FreezeError("download receipt inner-member hashes do not match bundle")
    if receipt.get("silicon_metadata") != parsed["silicon_metadata"]:
        raise FreezeError("download receipt silicon metadata does not match bundle")
    return parsed


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
    _run(command, cwd=out)

    bundle = out / BUNDLE_FILENAME
    parsed = _parse_bundle(bundle)
    receipt = {
        "schema_version": DOWNLOAD_SCHEMA,
        "created_utc": _utc_iso(created),
        "created_utc_yyyymmdd": _yyyymmdd(created),
        "family_label": FAMILY_LABEL,
        "element": ELEMENT,
        "command_contract": command,
        "bundle_filename": bundle.name,
        "bundle_sha256": _sha256_file(bundle),
        "inner_member_sha256": parsed["member_sha256"],
        "configuration": parsed["configuration"],
        "silicon_metadata": parsed["silicon_metadata"],
        "uv_lock_sha256": uv_lock_sha256,
        "python_distributions": versions,
        "authority": AUTHORITY_DOWNLOAD,
        "limitations": [
            "network acquisition has been content-addressed but no AiiDA family is installed by this phase",
            "recommended cutoffs are source metadata and do not replace Symthaea numerical convergence evidence",
            "thermodynamic, dynamical, experimental, and replication claims are unaffected",
        ],
    }
    _write_json(out / "download-receipt.json", receipt)
    print(out / "download-receipt.json")


def _install_freeze(args: argparse.Namespace) -> None:
    uv_lock_sha256 = _require_capsule_environment()
    versions = _distribution_versions()
    bundle = Path(args.bundle).resolve()
    download_receipt_path = Path(args.download_receipt).resolve()
    out = Path(args.output).resolve()
    _prepare_empty_dir(out)
    receipt = _read_json(download_receipt_path)
    parsed = _validate_download_receipt(
        receipt,
        bundle,
        uv_lock_sha256=uv_lock_sha256,
        versions=versions,
    )

    from aiida import load_profile, orm
    from aiida.common.exceptions import NotExistent
    from aiida.orm import QueryBuilder
    from aiida_pseudo.groups.family.sssp import SsspFamily

    load_profile(args.profile)
    process_count_before = QueryBuilder().append(orm.ProcessNode).count()
    if process_count_before != 0:
        raise FreezeError(
            f"pseudopotential must be frozen before any AiiDA process exists; found {process_count_before} ProcessNode(s)"
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
    recommended = {
        "cutoff_wfc_ry": float(cutoff_wfc),
        "cutoff_rho_ry": float(cutoff_rho),
    }
    expected_recommended = {
        "cutoff_wfc_ry": parsed["silicon_metadata"]["cutoff_wfc_ry"],
        "cutoff_rho_ry": parsed["silicon_metadata"]["cutoff_rho_ry"],
    }
    if recommended != expected_recommended:
        raise FreezeError("installed SSSP family cutoffs do not match downloaded metadata")

    process_count_after = QueryBuilder().append(orm.ProcessNode).count()
    if process_count_after != 0:
        raise FreezeError("pseudopotential installation unexpectedly created AiiDA ProcessNode records")

    upf_path = out / "Si.upf"
    upf_path.write_bytes(upf_bytes)
    if _sha256_file(upf_path) != upf_sha256:
        raise FreezeError("exported Si UPF bytes failed SHA-256 round-trip")

    created = _utc_now()
    freeze = {
        "schema_version": FREEZE_SCHEMA,
        "created_utc": _utc_iso(created),
        "created_utc_yyyymmdd": _yyyymmdd(created),
        "profile": args.profile,
        "family": {
            "label": family.label,
            "uuid": str(family.uuid),
            "python_class": f"{family.__class__.__module__}.{family.__class__.__name__}",
            "description": str(family.description),
        },
        "element": ELEMENT,
        "pseudo": {
            "uuid": str(pseudo.uuid),
            "pk": int(pseudo.pk),
            "filename": str(pseudo.filename),
            "md5": actual_md5,
            "sha256": upf_sha256,
            "byte_length": len(upf_bytes),
            "exported_artifact": "Si.upf",
        },
        "source_bundle": {
            "path_basename": bundle.name,
            "sha256": _sha256_file(bundle),
            "download_receipt_sha256": _sha256_file(download_receipt_path),
            "inner_member_sha256": parsed["member_sha256"],
        },
        "recommended_cutoffs_reference_only": recommended,
        "process_graph": {
            "process_nodes_before_install": process_count_before,
            "process_nodes_after_install": process_count_after,
        },
        "uv_lock_sha256": uv_lock_sha256,
        "python_distributions": versions,
        "installation_command_contract": command,
        "authority": AUTHORITY_FREEZE,
        "limitations": [
            "this receipt freezes pseudopotential identity before any AiiDA process; it does not execute QE",
            "SSSP recommended cutoffs are reference metadata only; production settings require independent convergence",
            "pseudopotential validation remains computational/source-level evidence, not experimental validation",
        ],
    }
    freeze_path = out / "freeze-receipt.json"
    _write_json(freeze_path, freeze)
    print(freeze_path)


def _self_test() -> None:
    configuration = {
        "version": FAMILY_VERSION,
        "functional": FAMILY_FUNCTIONAL,
        "protocol": FAMILY_PROTOCOL,
    }
    metadata = {
        ELEMENT: {
            "md5": "0123456789abcdef0123456789abcdef",
            "cutoff_wfc": 42.0,
            "cutoff_rho": 336.0,
        }
    }
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        inner_archive = root / "archive.tar.gz"
        inner_archive.write_bytes(b"synthetic-sssp-pseudopotential-archive")
        (root / "configuration.json").write_bytes(_json_bytes(configuration))
        (root / "metadata.json").write_bytes(_json_bytes(metadata))
        bundle = root / BUNDLE_FILENAME
        with tarfile.open(bundle, "w") as handle:
            for name in sorted(EXPECTED_INNER_NAMES):
                handle.add(root / name, arcname=name)
        parsed = _parse_bundle(bundle)
        assert parsed["configuration"] == configuration
        assert parsed["silicon_metadata"]["md5"] == metadata[ELEMENT]["md5"]
        assert parsed["silicon_metadata"]["cutoff_wfc_ry"] == 42.0
        assert set(parsed["member_sha256"]) == EXPECTED_INNER_NAMES

        download_receipt = {
            "schema_version": DOWNLOAD_SCHEMA,
            "family_label": FAMILY_LABEL,
            "element": ELEMENT,
            "bundle_sha256": _sha256_file(bundle),
            "inner_member_sha256": parsed["member_sha256"],
            "configuration": parsed["configuration"],
            "silicon_metadata": parsed["silicon_metadata"],
            "uv_lock_sha256": "a" * 64,
            "python_distributions": {"fixture": "1"},
        }
        _validate_download_receipt(
            download_receipt,
            bundle,
            uv_lock_sha256="a" * 64,
            versions={"fixture": "1"},
        )

        bad = root / "bad.aiida_pseudo"
        extra = root / "unexpected.txt"
        extra.write_text("no", encoding="utf-8")
        with tarfile.open(bad, "w") as handle:
            for name in sorted(EXPECTED_INNER_NAMES):
                handle.add(root / name, arcname=name)
            handle.add(extra, arcname=extra.name)
        try:
            _safe_bundle_members(bad)
        except FreezeError:
            pass
        else:
            raise AssertionError("extra bundle members must fail closed")
    print("Si pseudopotential freeze self-test: PASS")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    download = sub.add_parser("download", help="download and hash the exact SSSP bundle")
    download.add_argument("--output", required=True)

    freeze = sub.add_parser("install-freeze", help="install exact bundle and freeze Si pseudo before any process")
    freeze.add_argument("--profile", required=True)
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

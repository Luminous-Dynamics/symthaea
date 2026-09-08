#!/usr/bin/env python3
"""Network-free candidate neuromaps-method HCP-MMP1 Lineage-B derivation."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import hcpmmp_neuromaps_common as common_impl
import hcpmmp_neuromaps_gifti as gifti_impl
from hcpmmp_neuromaps_common import (
    AREA_SCHEMA,
    RUN_SCHEMA,
    VERTICES as VERTICES_PER_HEMISPHERE,
    ContractError,
    canonical_json_bytes,
    digest_bytes,
    digest_file,
    exact,
    load_area_order,
    load_json,
    load_method,
    load_run,
    sha256,
    verify_inputs,
)
from hcpmmp_neuromaps_gifti import (
    normalize as normalize_label,
    parse_label_gifti,
    semantic,
)

DerivationError = ContractError
load_method_manifest = load_method
load_run_manifest = load_run

OUTPUT_SCHEMA = "symthaea-semantic-surface-labels-v1"
EVIDENCE_SCHEMA = "symthaea-hcpmmp1-neuromaps-derivation-evidence-v1"
GENERATOR_ID = "symthaea-hcpmmp1-neuromaps-lineage-b"
GENERATOR_VERSION = "v1"
BUNDLE_RECEIPT_PROFILE = "symthaea-hcpmmp1-neuromaps-bundle-receipt-v1"
VERTICES = 10242
OUTPUT_KEYS = {"schema", "space", "hemisphere", "vertex_count", "labels", "source"}
SOURCE_KEYS = {
    "source_id",
    "source_version",
    "source_digest",
    "generator_id",
    "generator_version",
    "generator_implementation_digest",
    "terms_reference",
}
EVIDENCE_KEYS = {
    "schema",
    "lineage_id",
    "execution_id",
    "authorization_reference",
    "method_manifest_digest",
    "run_manifest_digest",
    "area_order_digest",
    "scientific_input_commitment",
    "generator_implementation",
    "workbench",
    "outputs",
    "independence",
    "content_digest",
}
GENERATOR_IMPLEMENTATION_KEYS = {"digest", "files"}
GENERATOR_FILE_KEYS = {"common", "gifti", "derive"}


def run_wb(wb: Path, args: list[str]) -> None:
    try:
        subprocess.run(
            [str(wb), *args],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as exc:
        raise ContractError(f"Workbench failed: {' '.join(args[:2])}") from exc


def generator_implementation() -> dict[str, Any]:
    files = {
        "common": digest_file(Path(common_impl.__file__).resolve()),
        "gifti": digest_file(Path(gifti_impl.__file__).resolve()),
        "derive": digest_file(Path(__file__).resolve()),
    }
    return {"files": files, "digest": digest_bytes(canonical_json_bytes(files))}


def validate_generator_implementation(value: Any) -> dict[str, Any]:
    doc = exact(value, GENERATOR_IMPLEMENTATION_KEYS, "generator implementation")
    files = exact(doc["files"], GENERATOR_FILE_KEYS, "generator implementation files")
    for name in sorted(files):
        sha256(files[name], f"generator {name} sha")
    sha256(doc["digest"], "generator implementation digest")
    if doc["digest"] != digest_bytes(canonical_json_bytes(files)):
        raise ContractError("generator implementation: aggregate digest mismatch")
    return doc


def commitment(
    method_path: Path,
    run: dict[str, Any],
    area_path: Path,
    version_digest: str,
    generator_digest: str | None = None,
) -> str:
    if generator_digest is None:
        generator_digest = generator_implementation()["digest"]
    sha256(generator_digest, "generator implementation digest")
    return digest_bytes(
        canonical_json_bytes(
            {
                "method_manifest_digest": digest_file(method_path),
                "area_order_digest": digest_file(area_path),
                "generator_implementation_digest": generator_digest,
                "workbench_sha256": run["workbench"]["sha256"],
                "workbench_version_output_sha256": version_digest,
                "inputs": {
                    role: run["inputs"][role]["sha256"]
                    for role in sorted(run["inputs"])
                },
            }
        )
    )


def _fsync_dir(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    fd = os.open(path, flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_private(path: Path, data: bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb", closefd=False) as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(fd)


def _path_exists(path: Path) -> bool:
    return os.path.lexists(path)


def _publish_bundle(
    outdir: Path,
    left: dict[str, Any],
    right: dict[str, Any],
    evidence_base: dict[str, Any],
) -> dict[str, Any]:
    requested = outdir.expanduser()
    requested.parent.mkdir(parents=True, exist_ok=True)
    parent = requested.parent.resolve(strict=True)
    final = parent / requested.name
    lock = parent / f".{requested.name}.publish-lock"
    if _path_exists(final):
        raise ContractError("output custody: destination already exists")
    try:
        lock.mkdir(mode=0o700)
    except FileExistsError as exc:
        raise ContractError("output custody: publication lock already exists") from exc

    staging: Path | None = None
    try:
        if _path_exists(final):
            raise ContractError("output custody: destination appeared during publication")
        staging = Path(tempfile.mkdtemp(prefix=f".{requested.name}.staging-", dir=parent))
        os.chmod(staging, 0o700)
        left_path = staging / "left.semantic.json"
        right_path = staging / "right.semantic.json"
        evidence_path = staging / "derivation-evidence.json"
        _write_private(left_path, canonical_json_bytes(left) + b"\n")
        _write_private(right_path, canonical_json_bytes(right) + b"\n")

        evidence = dict(evidence_base)
        evidence["outputs"] = {
            "left_semantic_sha256": digest_file(left_path),
            "right_semantic_sha256": digest_file(right_path),
        }
        evidence["content_digest"] = digest_bytes(canonical_json_bytes(evidence))
        _write_private(evidence_path, canonical_json_bytes(evidence) + b"\n")
        _fsync_dir(staging)

        if _path_exists(final):
            raise ContractError("output custody: destination appeared before publish")
        os.rename(staging, final)
        staging = None
        _fsync_dir(parent)
        return evidence
    finally:
        if staging is not None and staging.exists():
            shutil.rmtree(staging)
        try:
            lock.rmdir()
        except FileNotFoundError:
            pass
        _fsync_dir(parent)


def derive(
    method_path: Path,
    run_path: Path,
    area_path: Path,
    outdir: Path,
) -> dict[str, Any]:
    method = load_method(method_path)
    run = load_run(run_path, method, method_path)
    wb, inputs, version_digest = verify_inputs(run)
    areas = load_area_order(area_path)
    implementation = generator_implementation()
    root = commitment(
        method_path,
        run,
        area_path,
        version_digest,
        implementation["digest"],
    )

    with tempfile.TemporaryDirectory(prefix="symthaea-hcpmmp-lineage-b-") as temp:
        tempdir = Path(temp)
        left_32k = tempdir / "left.32k.label.gii"
        right_32k = tempdir / "right.32k.label.gii"
        left_10k = tempdir / "left.10k.label.gii"
        right_10k = tempdir / "right.10k.label.gii"

        run_wb(
            wb,
            [
                "-cifti-separate",
                str(inputs["hcp_left_dlabel"]),
                "COLUMN",
                "-label",
                "CORTEX_LEFT",
                str(left_32k),
            ],
        )
        run_wb(
            wb,
            [
                "-cifti-separate",
                str(inputs["hcp_right_dlabel"]),
                "COLUMN",
                "-label",
                "CORTEX_RIGHT",
                str(right_32k),
            ],
        )
        for hemisphere, source, destination in (
            ("left", left_32k, left_10k),
            ("right", right_32k, right_10k),
        ):
            run_wb(
                wb,
                [
                    "-label-resample",
                    str(source),
                    str(inputs[f"fslr32k_{hemisphere}_sphere_to_fsaverage"]),
                    str(inputs[f"fsaverage10k_{hemisphere}_sphere"]),
                    "ADAP_BARY_AREA",
                    str(destination),
                    "-area-metrics",
                    str(inputs[f"fslr32k_{hemisphere}_vaavg"]),
                    str(inputs[f"fsaverage10k_{hemisphere}_vaavg"]),
                    "-current-roi",
                    str(inputs[f"fslr32k_{hemisphere}_medialwall_roi"]),
                ],
            )

        left = semantic(
            left_10k,
            inputs["fsaverage10k_left_medialwall_roi"],
            "left",
            areas,
            root,
            f"{method['lineage_id']}:left",
            method["terms_reference"],
            GENERATOR_ID,
            GENERATOR_VERSION,
            implementation["digest"],
        )
        right = semantic(
            right_10k,
            inputs["fsaverage10k_right_medialwall_roi"],
            "right",
            areas,
            root,
            f"{method['lineage_id']}:right",
            method["terms_reference"],
            GENERATOR_ID,
            GENERATOR_VERSION,
            implementation["digest"],
        )

        _, _, version_after = verify_inputs(run)
        if version_after != version_digest:
            raise ContractError("execution inputs: Workbench version changed during derivation")
        if generator_implementation() != implementation:
            raise ContractError("generator implementation changed during derivation")

    evidence_base = {
        "schema": EVIDENCE_SCHEMA,
        "lineage_id": method["lineage_id"],
        "execution_id": run["execution_id"],
        "authorization_reference": run["authorization_reference"],
        "method_manifest_digest": digest_file(method_path),
        "run_manifest_digest": digest_file(run_path),
        "area_order_digest": digest_file(area_path),
        "scientific_input_commitment": root,
        "generator_implementation": implementation,
        "workbench": {
            "sha256": run["workbench"]["sha256"],
            "version_output_sha256": version_digest,
        },
        "independence": {
            **method["independence_contract"],
            "independence_established": False,
            "status": "requires_external_provenance_review",
        },
    }
    return _publish_bundle(outdir, left, right, evidence_base)


def _validate_output(doc: Any, hemisphere: str, evidence: dict[str, Any]) -> None:
    doc = exact(doc, OUTPUT_KEYS, f"{hemisphere} output")
    if (
        doc["schema"] != OUTPUT_SCHEMA
        or doc["space"] != "fsaverage5"
        or doc["hemisphere"] != hemisphere
        or doc["vertex_count"] != VERTICES
        or not isinstance(doc["labels"], list)
        or len(doc["labels"]) != VERTICES
    ):
        raise ContractError(f"{hemisphere}: output identity mismatch")
    source = exact(doc["source"], SOURCE_KEYS, f"{hemisphere} source")
    sha256(source["source_digest"], f"{hemisphere} source digest")
    sha256(
        source["generator_implementation_digest"],
        f"{hemisphere} generator digest",
    )
    if (
        source["source_id"] != f"{evidence['lineage_id']}:{hemisphere}"
        or source["source_version"] != "v1"
        or source["source_digest"] != evidence["scientific_input_commitment"]
        or source["generator_id"] != GENERATOR_ID
        or source["generator_version"] != GENERATOR_VERSION
        or source["generator_implementation_digest"]
        != evidence["generator_implementation"]["digest"]
        or not isinstance(source["terms_reference"], str)
        or not source["terms_reference"]
    ):
        raise ContractError(f"{hemisphere}: output source provenance mismatch")


def validate_evidence(outdir: Path) -> dict[str, Any]:
    evidence = exact(
        load_json(outdir / "derivation-evidence.json"),
        EVIDENCE_KEYS,
        "derivation evidence",
    )
    if evidence["schema"] != EVIDENCE_SCHEMA:
        raise ContractError("evidence: wrong schema")
    for key in (
        "method_manifest_digest",
        "run_manifest_digest",
        "area_order_digest",
        "scientific_input_commitment",
    ):
        sha256(evidence[key], f"evidence {key}")
    validate_generator_implementation(evidence["generator_implementation"])
    if set(evidence["workbench"]) != {"sha256", "version_output_sha256"}:
        raise ContractError("evidence: workbench schema mismatch")
    if set(evidence["outputs"]) != {
        "left_semantic_sha256",
        "right_semantic_sha256",
    }:
        raise ContractError("evidence: output schema mismatch")

    left_path = outdir / "left.semantic.json"
    right_path = outdir / "right.semantic.json"
    if digest_file(left_path) != evidence["outputs"]["left_semantic_sha256"]:
        raise ContractError("evidence: left output digest mismatch")
    if digest_file(right_path) != evidence["outputs"]["right_semantic_sha256"]:
        raise ContractError("evidence: right output digest mismatch")
    _validate_output(load_json(left_path), "left", evidence)
    _validate_output(load_json(right_path), "right", evidence)

    independence = evidence["independence"]
    if (
        not isinstance(independence, dict)
        or independence.get("independence_established") is not False
        or independence.get("status") != "requires_external_provenance_review"
    ):
        raise ContractError("evidence: independence authority escalation")

    stored = sha256(evidence["content_digest"], "content digest")
    payload = dict(evidence)
    del payload["content_digest"]
    if stored != digest_bytes(canonical_json_bytes(payload)):
        raise ContractError("evidence: content digest mismatch")
    return evidence


def _receipt(outdir: Path, evidence: dict[str, Any], action: str) -> dict[str, str]:
    return {
        "profile": BUNDLE_RECEIPT_PROFILE,
        "action": action,
        "evidence_content_digest": evidence["content_digest"],
        "evidence_file_sha256": digest_file(outdir / "derivation-evidence.json"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    derive_parser = sub.add_parser("derive")
    for name in ("method-manifest", "run-manifest", "area-order", "output-dir"):
        derive_parser.add_argument("--" + name, required=True, type=Path)
    verify_parser = sub.add_parser("verify-evidence")
    verify_parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.cmd == "derive":
            evidence = derive(
                args.method_manifest,
                args.run_manifest,
                args.area_order,
                args.output_dir,
            )
            receipt = _receipt(args.output_dir, evidence, "derive")
        else:
            evidence = validate_evidence(args.output_dir)
            receipt = _receipt(args.output_dir, evidence, "verify")
    except (ContractError, OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

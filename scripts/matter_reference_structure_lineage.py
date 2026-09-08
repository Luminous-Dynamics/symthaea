#!/usr/bin/env python3
"""Bind final-relaxation StructureData to the phonon-parent SCF structure.

This adapter closes the external AiiDA structure-lineage gap for the first
reference-crystal capsule. It consumes two completed
`matter_aiida_structure_export.py` exports plus the explicit phonon-chain export
from `matter_aiida_qe_phonon_chain_export.py` and requires:

  relaxation output StructureData
      == exact AiiDA node reused as ==
  phonon-parent pw.x SCF input StructureData

and, independently, exact equality of the ordered-periodic-cell v1 receipt.

The exact AiiDA node-reuse requirement is intentionally strict for reference
capsule v1. A later profile may admit provenance-preserving structure transforms,
but this path refuses a newly reconstructed/cloned StructureData even when its
coordinates are numerically identical.

A successful binding establishes structure-lineage continuity only. It does not
establish crystallographic equivalence beyond ordered-cell v1, phonon stability,
thermodynamic phase stability, numerical convergence, experiment, or replication.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable

ADAPTER_NAME = "symthaea-reference-structure-lineage"
ADAPTER_VERSION = "1"
STRUCTURE_ADAPTER_NAME = "symthaea-aiida-structure-export"
STRUCTURE_RECEIPT_SCHEMA = "symthaea.matter.ordered-periodic-cell-receipts/v1"
PHONON_CHAIN_ADAPTER_NAME = "symthaea-aiida-qe-phonon-chain-export"
PHONON_CHAIN_SCHEMA = "symthaea.matter.aiida-qe-phonon-chain/v1"
OUTPUT_SCHEMA = "symthaea.matter.reference-structure-lineage/v1"
MAX_JSON_BYTES = 16 * 1024 * 1024
TASK_RELAX = "structural_relaxation"
TASK_PERIODIC = "periodic_electronic_structure"


class AdapterError(RuntimeError):
    """Fail-closed structure-lineage error."""


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_JSON_BYTES:
                raise AdapterError(f"input JSON exceeds size limit: {path}")
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    if not path.is_file():
        raise AdapterError(f"missing JSON artifact: {path}")
    if path.stat().st_size > MAX_JSON_BYTES:
        raise AdapterError(f"JSON artifact exceeds size limit: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AdapterError(f"cannot read JSON {path}: {error}") from error


def _write_json(path: Path, value: Any) -> str:
    payload = _json_bytes(value)
    if len(payload) > MAX_JSON_BYTES:
        raise AdapterError(f"output JSON exceeds size limit: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    digest = _sha256(payload)
    if _sha256_file(path) != digest:
        raise AdapterError(f"persisted JSON digest mismatch: {path}")
    return digest


def _prepare_empty(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise AdapterError(f"output directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _safe_id(value: Any, field: str) -> str:
    text = str(value)
    if not text.strip():
        raise AdapterError(f"{field} is empty")
    if len(text.encode("utf-8")) > 512:
        raise AdapterError(f"{field} exceeds 512 UTF-8 bytes")
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in text):
        raise AdapterError(f"{field} contains a control character")
    return text


def _reference_index(bundle: Any, source: str) -> dict[str, dict[str, Any]]:
    if not isinstance(bundle, dict):
        raise AdapterError(f"{source} evidence bundle root must be an object")
    references = bundle.get("references")
    if not isinstance(references, list) or not references:
        raise AdapterError(f"{source} evidence bundle has no references")
    index: dict[str, dict[str, Any]] = {}
    for item in references:
        if not isinstance(item, dict):
            raise AdapterError(f"{source} evidence bundle contains non-object reference")
        evidence_id = _safe_id(item.get("id", ""), f"{source}.reference.id")
        if evidence_id in index:
            raise AdapterError(f"{source} evidence bundle contains duplicate ID: {evidence_id}")
        index[evidence_id] = item
    return index


def _resolve_local_artifact(root: Path, reference: dict[str, Any], evidence_id: str) -> Path:
    if reference.get("role") != "ArtifactContent":
        raise AdapterError(f"structure reference has wrong role: {evidence_id}")
    locator = str(reference.get("locator", ""))
    prefix = "symthaea-export://"
    if not locator.startswith(prefix):
        raise AdapterError(f"structure reference has unsupported locator: {evidence_id}")
    relative = locator[len(prefix) :]
    if not relative or relative.startswith("/") or ".." in Path(relative).parts:
        raise AdapterError(f"structure reference locator escapes export root: {evidence_id}")
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise AdapterError(f"structure reference locator escapes export root: {evidence_id}") from error
    expected = _safe_id(reference.get("content_sha256", ""), f"{evidence_id}.content_sha256")
    if len(expected) != 64 or any(ch not in "0123456789abcdef" for ch in expected):
        raise AdapterError(f"structure reference has malformed SHA-256: {evidence_id}")
    if _sha256_file(path) != expected:
        raise AdapterError(f"structure reference digest mismatch: {evidence_id}")
    return path


def _load_structure_export(root: Path, expected_task: str) -> dict[str, Any]:
    manifest_path = root / "structure-export-manifest.json"
    receipts_path = root / "ordered-cell-receipts.json"
    bundle_path = root / "evidence-bundle.augmented.json"
    manifest = _read_json(manifest_path)
    receipts = _read_json(receipts_path)
    bundle = _read_json(bundle_path)
    if not isinstance(manifest, dict) or not isinstance(receipts, dict):
        raise AdapterError("structure export roots must be objects")
    adapter = manifest.get("adapter")
    if not isinstance(adapter, dict) or adapter.get("name") != STRUCTURE_ADAPTER_NAME or str(adapter.get("version")) != "1":
        raise AdapterError("structure export is not from canonical AiiDA structure adapter v1")
    source = manifest.get("source")
    if not isinstance(source, dict) or source.get("system") != "AiiDA":
        raise AdapterError("structure export source is not AiiDA")
    calcjob_uuid = _safe_id(source.get("calcjob_uuid", ""), "structure source calcjob UUID")
    task = str(source.get("task", ""))
    if task != expected_task:
        raise AdapterError(f"structure export task mismatch: expected {expected_task}, got {task}")
    if receipts.get("schema_version") != STRUCTURE_RECEIPT_SCHEMA:
        raise AdapterError("structure receipt schema mismatch")
    if str(receipts.get("source_calcjob_uuid", "")) != calcjob_uuid or str(receipts.get("task", "")) != task:
        raise AdapterError("structure receipts do not match structure export manifest source")
    raw_receipts = receipts.get("receipts")
    if not isinstance(raw_receipts, list) or not raw_receipts:
        raise AdapterError("structure export has no ordered-cell receipts")
    by_role: dict[str, dict[str, Any]] = {}
    references = _reference_index(bundle, expected_task)
    for item in raw_receipts:
        if not isinstance(item, dict):
            raise AdapterError("ordered-cell receipt entry is not an object")
        role = str(item.get("source_role", ""))
        if role not in ("input", "output"):
            raise AdapterError(f"unsupported ordered-cell source role: {role}")
        if role in by_role:
            raise AdapterError(f"duplicate ordered-cell source role: {role}")
        source_structure_uuid = _safe_id(item.get("source_structure_uuid", ""), f"{task}.{role}.source_structure_uuid")
        receipt = item.get("receipt")
        if not isinstance(receipt, dict):
            raise AdapterError(f"{task}.{role} receipt is not an object")
        artifact_id = _safe_id(receipt.get("structure_artifact_id", ""), f"{task}.{role}.structure_artifact_id")
        reference = references.get(artifact_id)
        if reference is None:
            raise AdapterError(f"missing canonical structure reference: {artifact_id}")
        artifact_path = _resolve_local_artifact(root, reference, artifact_id)
        artifact = _read_json(artifact_path)
        if not isinstance(artifact, dict):
            raise AdapterError(f"structure artifact is not an object: {artifact_id}")
        if str(artifact.get("source_structure_uuid", "")) != source_structure_uuid or str(artifact.get("source_role", "")) != role:
            raise AdapterError(f"structure artifact source identity mismatch: {artifact_id}")
        if artifact.get("lattice_vectors_angstrom_bits") != receipt.get("lattice_vectors_angstrom_bits") or artifact.get("sites") != receipt.get("sites"):
            raise AdapterError(f"structure artifact values differ from ordered-cell receipt: {artifact_id}")
        by_role[role] = {
            "source_structure_uuid": source_structure_uuid,
            "receipt": receipt,
            "artifact_reference": reference,
            "artifact_sha256": str(reference["content_sha256"]),
        }
    expected_roles = {"input", "output"} if expected_task == TASK_RELAX else {"input"}
    if set(by_role) != expected_roles:
        raise AdapterError(f"unexpected ordered-cell role set for {expected_task}: {sorted(by_role)}")
    return {
        "root": root,
        "calcjob_uuid": calcjob_uuid,
        "task": task,
        "by_role": by_role,
        "manifest_sha256": _sha256_file(manifest_path),
        "receipts_sha256": _sha256_file(receipts_path),
        "bundle_sha256": _sha256_file(bundle_path),
    }


def _load_phonon_chain(root: Path) -> dict[str, Any]:
    manifest_path = root / "export-manifest.json"
    chain_path = root / "phonon-chain.json"
    bundle_path = root / "evidence-bundle.json"
    manifest = _read_json(manifest_path)
    chain = _read_json(chain_path)
    bundle = _read_json(bundle_path)
    if not isinstance(manifest, dict) or not isinstance(chain, dict):
        raise AdapterError("phonon-chain roots must be objects")
    adapter = manifest.get("adapter")
    if not isinstance(adapter, dict) or adapter.get("name") != PHONON_CHAIN_ADAPTER_NAME or str(adapter.get("version")) != "1":
        raise AdapterError("phonon chain is not from canonical AiiDA QE phonon-chain adapter v1")
    if chain.get("schema_version") != PHONON_CHAIN_SCHEMA or chain.get("source_system") != "AiiDA":
        raise AdapterError("phonon-chain schema/source mismatch")
    processes = chain.get("processes")
    lineage = chain.get("lineage")
    if not isinstance(processes, dict) or not isinstance(lineage, dict):
        raise AdapterError("phonon-chain process/lineage objects are missing")
    pw = processes.get("pw")
    if not isinstance(pw, dict):
        raise AdapterError("phonon chain has no pw process")
    pw_uuid = _safe_id(pw.get("uuid", ""), "phonon-chain pw UUID")
    pw_input_structure_uuid = _safe_id(lineage.get("pw_input_structure_uuid", ""), "phonon-chain pw input structure UUID")
    _reference_index(bundle, "phonon-chain")
    return {
        "pw_uuid": pw_uuid,
        "pw_input_structure_uuid": pw_input_structure_uuid,
        "chain_sha256": _sha256_file(chain_path),
        "manifest_sha256": _sha256_file(manifest_path),
        "bundle_sha256": _sha256_file(bundle_path),
    }


def _cell_values(receipt: dict[str, Any]) -> dict[str, Any]:
    lattice = receipt.get("lattice_vectors_angstrom_bits")
    sites = receipt.get("sites")
    if not isinstance(lattice, list) or len(lattice) != 3 or any(not isinstance(row, list) or len(row) != 3 for row in lattice):
        raise AdapterError("ordered-cell receipt lattice is not 3x3")
    if not isinstance(sites, list) or not sites:
        raise AdapterError("ordered-cell receipt has no sites")
    return {"lattice_vectors_angstrom_bits": lattice, "sites": sites}


def _bind(args: argparse.Namespace) -> None:
    relax = _load_structure_export(Path(args.relax_structure_export).resolve(), TASK_RELAX)
    scf = _load_structure_export(Path(args.phonon_parent_structure_export).resolve(), TASK_PERIODIC)
    chain = _load_phonon_chain(Path(args.phonon_chain_export).resolve())

    if scf["calcjob_uuid"] != chain["pw_uuid"]:
        raise AdapterError("phonon-parent structure export CalcJob is not the pw.x node used by the phonon chain")
    scf_input = scf["by_role"]["input"]
    relax_output = relax["by_role"]["output"]
    if scf_input["source_structure_uuid"] != chain["pw_input_structure_uuid"]:
        raise AdapterError("phonon-chain pw input StructureData UUID differs from SCF structure export")
    if relax_output["source_structure_uuid"] != scf_input["source_structure_uuid"]:
        raise AdapterError("final relaxed StructureData node was not reused directly as phonon-parent SCF input")

    relaxed_cell = _cell_values(relax_output["receipt"])
    scf_cell = _cell_values(scf_input["receipt"])
    if relaxed_cell != scf_cell:
        raise AdapterError("final relaxed and phonon-parent SCF ordered-cell receipts are not bit-identical")

    out = Path(args.output).resolve()
    _prepare_empty(out)
    result = {
        "schema_version": OUTPUT_SCHEMA,
        "adapter": {"name": ADAPTER_NAME, "version": ADAPTER_VERSION},
        "relaxation": {
            "calcjob_uuid": relax["calcjob_uuid"],
            "output_structure_uuid": relax_output["source_structure_uuid"],
            "output_structure_artifact_id": relax_output["receipt"]["structure_artifact_id"],
            "output_structure_artifact_sha256": relax_output["artifact_sha256"],
            "structure_export_manifest_sha256": relax["manifest_sha256"],
            "ordered_cell_receipts_sha256": relax["receipts_sha256"],
            "augmented_evidence_bundle_sha256": relax["bundle_sha256"],
        },
        "phonon_parent_scf": {
            "calcjob_uuid": scf["calcjob_uuid"],
            "input_structure_uuid": scf_input["source_structure_uuid"],
            "input_structure_artifact_id": scf_input["receipt"]["structure_artifact_id"],
            "input_structure_artifact_sha256": scf_input["artifact_sha256"],
            "structure_export_manifest_sha256": scf["manifest_sha256"],
            "ordered_cell_receipts_sha256": scf["receipts_sha256"],
            "augmented_evidence_bundle_sha256": scf["bundle_sha256"],
        },
        "phonon_chain": {
            "pw_calcjob_uuid": chain["pw_uuid"],
            "pw_input_structure_uuid": chain["pw_input_structure_uuid"],
            "phonon_chain_sha256": chain["chain_sha256"],
            "export_manifest_sha256": chain["manifest_sha256"],
            "evidence_bundle_sha256": chain["bundle_sha256"],
        },
        "ordered_cell_v1": relaxed_cell,
        "decision": {
            "exact_aiida_structure_node_reuse": True,
            "bit_exact_ordered_cell_v1_equality": True,
            "interpretation": "ExactAiiDAStructureNodeReuseAndOrderedCellV1Equality",
        },
        "authority": "ExecutionReferenceOnly",
        "thermodynamic_phase_eligibility": "WithheldByProtocol",
        "limitations": [
            "ordered-cell v1 does not prove primitive/conventional-cell, basis-transform, origin-shift, symmetry, disorder, or partial-occupancy equivalence",
            "this binding proves structure-lineage continuity only; it does not establish phonon stability or numerical convergence",
            "AiiDA UUID/ORM provenance is relied on as the execution provenance source; no independent external timestamp attestation is claimed",
        ],
    }
    result_path = out / "reference-structure-lineage.json"
    digest = _write_json(result_path, result)
    _write_json(
        out / "export-manifest.json",
        {
            "adapter": {"name": ADAPTER_NAME, "version": ADAPTER_VERSION},
            "structure_lineage": "reference-structure-lineage.json",
            "structure_lineage_sha256": digest,
            "authority": "ExecutionReferenceOnly",
            "thermodynamic_phase_eligibility": "WithheldByProtocol",
        },
    )
    print(out)


def _self_test() -> None:
    cell = {
        "lattice_vectors_angstrom_bits": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        "sites": [{"atomic_number": 14, "fractional_bits": [0, 0, 0]}],
    }
    assert _cell_values(cell) == cell
    assert _json_bytes({"b": 2, "a": 1}) == b'{"a":1,"b":2}'
    try:
        _cell_values({"lattice_vectors_angstrom_bits": [[1, 2]], "sites": [1]})
    except AdapterError:
        pass
    else:
        raise AssertionError("malformed cell must fail closed")
    print("Reference structure-lineage self-test: PASS")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    bind = subs.add_parser("bind")
    bind.add_argument("--relax-structure-export", required=True)
    bind.add_argument("--phonon-parent-structure-export", required=True)
    bind.add_argument("--phonon-chain-export", required=True)
    bind.add_argument("--output", required=True)
    subs.add_parser("self-test")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    try:
        if args.command == "bind":
            _bind(args)
        else:
            _self_test()
        return 0
    except AdapterError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

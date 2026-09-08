#!/usr/bin/env python3
"""Export an explicit AiiDA Quantum ESPRESSO phonon post-processing chain.

Reference path:

    pw.x SCF -> ph.x -> q2r.x -> matdyn.x

The exporter proves the actual AiiDA data-node edges rather than accepting four
unrelated successful jobs. `preflight` freezes ph/q2r/matdyn code identities and
plugin versions before execution. `export` refuses cached/import-style jobs,
requires a final BandsData result, preserves ph/matdyn sampling inputs, and hashes
literal exported artifacts.

A successful export establishes provenance-preserving execution only. Numerical
convergence, reciprocal-grid completeness, phase stability, experimental validity,
and agreement with the preregistered benchmark remain separate claims.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ADAPTER_NAME = "symthaea-aiida-qe-phonon-chain-export"
ADAPTER_VERSION = "1"
PREFLIGHT_SCHEMA = "symthaea.matter.aiida-qe-phonon-chain-preflight/v1"
CHAIN_SCHEMA = "symthaea.matter.aiida-qe-phonon-chain/v1"
PW_PROCESS = "aiida.calculations:quantumespresso.pw"
PH_PROCESS = "aiida.calculations:quantumespresso.ph"
Q2R_PROCESS = "aiida.calculations:quantumespresso.q2r"
MATDYN_PROCESS = "aiida.calculations:quantumespresso.matdyn"
MAX_JSON_BYTES = 32 * 1024 * 1024
MAX_TREE_FILES = 8192
MAX_FILE_BYTES = 2 * 1024 * 1024 * 1024


class AdapterError(RuntimeError):
    pass


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_FILE_BYTES:
                raise AdapterError(f"artifact exceeds size limit: {path}")
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> str:
    payload = _json_bytes(value)
    if len(payload) > MAX_JSON_BYTES:
        raise AdapterError(f"JSON artifact exceeds size limit: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    digest = _sha256(payload)
    if _sha256_file(path) != digest:
        raise AdapterError(f"persisted JSON digest mismatch: {path}")
    return digest


def _read_json(path: Path) -> Any:
    if not path.is_file() or path.stat().st_size > MAX_JSON_BYTES:
        raise AdapterError(f"missing/oversized JSON artifact: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AdapterError(f"cannot read JSON {path}: {error}") from error


def _safe_text(value: Any, field: str, max_bytes: int = 512) -> str:
    text = str(value)
    if not text.strip():
        raise AdapterError(f"{field} is empty")
    if len(text.encode("utf-8")) > max_bytes:
        raise AdapterError(f"{field} exceeds {max_bytes} UTF-8 bytes")
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in text):
        raise AdapterError(f"{field} contains a control character")
    return text


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError as error:
        raise AdapterError(f"missing Python distribution: {name}") from error


def _utc_ms(dt: datetime) -> int:
    if dt.tzinfo is None:
        raise AdapterError("AiiDA datetime is timezone-naive")
    return int(dt.timestamp() * 1000)


def _yyyymmdd(dt: datetime) -> int:
    if dt.tzinfo is None:
        raise AdapterError("AiiDA datetime is timezone-naive")
    dt = dt.astimezone(timezone.utc)
    return dt.year * 10000 + dt.month * 100 + dt.day


def _prepare_empty(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise AdapterError(f"output directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _code_snapshot(code: Any) -> dict[str, Any]:
    computer = getattr(code, "computer", None)
    return {
        "uuid": str(code.uuid),
        "label": str(getattr(code, "label", "")),
        "full_label": str(getattr(code, "full_label", "")),
        "computer_uuid": str(getattr(computer, "uuid", "")) if computer else None,
        "filepath_executable": str(getattr(code, "filepath_executable", "")),
        "default_calc_job_plugin": str(getattr(code, "default_calc_job_plugin", "")),
    }


def _preflight(args: argparse.Namespace) -> None:
    from aiida import load_profile, orm

    load_profile(args.profile) if args.profile else load_profile()
    created = datetime.now(timezone.utc)
    codes = {"ph": orm.load_code(args.ph_code), "q2r": orm.load_code(args.q2r_code), "matdyn": orm.load_code(args.matdyn_code)}
    snapshots = {name: _code_snapshot(code) for name, code in codes.items()}
    if len({value["uuid"] for value in snapshots.values()}) != 3:
        raise AdapterError("ph/q2r/matdyn code UUIDs must be distinct")
    out = Path(args.output).resolve()
    _prepare_empty(out)
    value = {
        "schema_version": PREFLIGHT_SCHEMA,
        "created_utc_unix_ms": _utc_ms(created),
        "created_utc_yyyymmdd": _yyyymmdd(created),
        "aiida_core_version": _package_version("aiida-core"),
        "aiida_quantumespresso_version": _package_version("aiida-quantumespresso"),
        "quantum_espresso_version_declared": _safe_text(args.solver_version, "solver_version", 128),
        "codes": snapshots,
        "interpretation": "declared pre-execution software context; not trusted timestamp attestation",
    }
    _write_json(out / "phonon-chain-preflight.json", value)
    print(out / "phonon-chain-preflight.json")


def _validate_preflight(path: Path) -> dict[str, Any]:
    value = _read_json(path)
    if not isinstance(value, dict) or value.get("schema_version") != PREFLIGHT_SCHEMA:
        raise AdapterError("unsupported/malformed phonon-chain preflight")
    for key in ("created_utc_unix_ms", "created_utc_yyyymmdd", "aiida_core_version", "aiida_quantumespresso_version", "quantum_espresso_version_declared", "codes"):
        if key not in value:
            raise AdapterError(f"preflight missing {key}")
    if not isinstance(value["codes"], dict) or set(value["codes"]) != {"ph", "q2r", "matdyn"}:
        raise AdapterError("preflight code set mismatch")
    return value


def _ns_get(namespace: Any, name: str) -> Any | None:
    try:
        return getattr(namespace, name)
    except (AttributeError, KeyError):
        return None


def _dict_value(node: Any | None) -> dict[str, Any] | None:
    if node is None or not hasattr(node, "get_dict"):
        return None
    value = node.get_dict()
    if not isinstance(value, dict):
        raise AdapterError("AiiDA Dict returned non-dict content")
    return value


def _is_cached(node: Any) -> bool:
    value = getattr(node.base.caching, "is_created_from_cache", False)
    return bool(value() if callable(value) else value)


def _validate_calcjob(node: Any, expected: str, label: str) -> None:
    if str(node.process_type) != expected:
        raise AdapterError(f"{label} process type mismatch: {node.process_type}")
    if not bool(getattr(node, "is_sealed", False)) or not bool(node.is_finished_ok):
        raise AdapterError(f"{label} must be sealed and finished_ok")
    if _is_cached(node):
        raise AdapterError(f"{label} cache hit is lineage, not a new execution")
    if _ns_get(node.inputs, "remote_folder") is not None:
        raise AdapterError(f"{label} imported remote-folder execution is refused in v1")
    if _ns_get(node.inputs, "code") is None:
        raise AdapterError(f"{label} has no execution code")


def _same_node(left: Any, right: Any, edge: str) -> None:
    if left is None or right is None or str(left.uuid) != str(right.uuid):
        raise AdapterError(f"broken AiiDA provenance edge: {edge}")


def _tree_manifest(root: Path) -> dict[str, Any]:
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if not files or len(files) > MAX_TREE_FILES:
        raise AdapterError(f"invalid repository tree file count: {root}")
    entries = []
    for path in files:
        size = path.stat().st_size
        if size > MAX_FILE_BYTES:
            raise AdapterError(f"repository file exceeds size limit: {path}")
        entries.append({"path": path.relative_to(root).as_posix(), "size": size, "sha256": _sha256_file(path)})
    return {"files": entries, "file_count": len(entries), "total_bytes": sum(item["size"] for item in entries)}


def _copy_repository(node: Any, destination: Path) -> dict[str, Any]:
    destination.mkdir(parents=True, exist_ok=True)
    with node.base.repository.as_path() as source:
        source = Path(source)
        if source.is_file():
            shutil.copy2(source, destination / source.name)
        else:
            for item in source.iterdir():
                target = destination / item.name
                shutil.copytree(item, target) if item.is_dir() else shutil.copy2(item, target)
    return _tree_manifest(destination)


def _visit_finite(value: Any, field: str) -> None:
    if isinstance(value, list):
        for child in value:
            _visit_finite(child, field)
    elif not math.isfinite(float(value)):
        raise AdapterError(f"{field} contains non-finite values")


def _kpoints_payload(node: Any) -> dict[str, Any]:
    try:
        mesh, offset = node.get_kpoints_mesh()
        payload = {"mode": "mesh", "mesh": [int(v) for v in mesh], "offset": [float(v) for v in offset]}
        if any(v <= 0 for v in payload["mesh"]):
            raise AdapterError("k/q-point mesh contains non-positive dimension")
        _visit_finite(payload["offset"], "k/q-point offset")
        return payload
    except AdapterError:
        raise
    except Exception:
        try:
            points = node.get_kpoints(cartesian=False).tolist()
        except Exception as error:
            raise AdapterError(f"cannot read KpointsData: {error}") from error
        if not points:
            raise AdapterError("explicit KpointsData is empty")
        _visit_finite(points, "k/q-points")
        payload: dict[str, Any] = {"mode": "explicit", "points_fractional": points}
        try:
            weights = node.get_kpoints_weights().tolist()
            _visit_finite(weights, "k/q-point weights")
            payload["weights"] = weights
        except Exception:
            pass
        return payload


def _bands_payload(bands: Any) -> dict[str, Any]:
    try:
        kpoints = bands.get_kpoints().tolist()
        raw = bands.get_bands()
        values = (raw[0] if isinstance(raw, tuple) else raw).tolist()
    except Exception as error:
        raise AdapterError(f"cannot read BandsData: {error}") from error
    if not kpoints or not values:
        raise AdapterError("matdyn BandsData is empty")
    _visit_finite(kpoints, "phonon-band kpoints")
    _visit_finite(values, "phonon bands")
    return {"kpoints": kpoints, "bands": values, "units": getattr(bands, "units", None)}


def _artifact(out: Path, name: str, value: Any) -> tuple[str, Path]:
    path = out / "artifacts" / name
    return _write_json(path, value), path


def _reference(evidence_id: str, role: str, digest: str, locator: str, subject: str, date: int) -> dict[str, Any]:
    return {"id": evidence_id, "role": role, "content_sha256": digest, "locator": locator, "subject": subject, "claimed_utc_date": date, "issuer": ADAPTER_NAME}


def _execution_manifest(node: Any, stage: str) -> dict[str, Any]:
    return {
        "stage": stage,
        "node_uuid": str(node.uuid),
        "node_pk": int(node.pk),
        "process_type": str(node.process_type),
        "process_label": str(node.process_label),
        "code_uuid": str(node.inputs.code.uuid),
        "ctime_utc": node.ctime.astimezone(timezone.utc).isoformat(),
        "mtime_utc": node.mtime.astimezone(timezone.utc).isoformat(),
        "exit_status": int(node.exit_status or 0),
        "parameters": _dict_value(_ns_get(node.inputs, "parameters")),
        "output_parameters": _dict_value(_ns_get(node.outputs, "output_parameters")),
    }


def _export(args: argparse.Namespace) -> None:
    from aiida import load_profile, orm

    preflight_path = Path(args.preflight).resolve()
    preflight = _validate_preflight(preflight_path)
    load_profile(args.profile) if args.profile else load_profile()
    pw, ph, q2r, matdyn = [orm.load_node(value) for value in (args.pw, args.ph, args.q2r, args.matdyn)]
    for node in (pw, ph, q2r, matdyn):
        if not isinstance(node, orm.CalcJobNode):
            raise AdapterError("all supplied process nodes must be CalcJobNode")
    _validate_calcjob(pw, PW_PROCESS, "pw.x")
    _validate_calcjob(ph, PH_PROCESS, "ph.x")
    _validate_calcjob(q2r, Q2R_PROCESS, "q2r.x")
    _validate_calcjob(matdyn, MATDYN_PROCESS, "matdyn.x")

    pw_parameters = _dict_value(_ns_get(pw.inputs, "parameters")) or {}
    if str(pw_parameters.get("CONTROL", {}).get("calculation", "scf")).lower() != "scf":
        raise AdapterError("phonon parent must be a pw.x SCF")

    _same_node(ph.inputs.parent_folder, pw.outputs.remote_folder, "pw.remote_folder -> ph.parent_folder")
    _same_node(q2r.inputs.parent_folder, ph.outputs.remote_folder, "ph.remote_folder -> q2r.parent_folder")
    _same_node(matdyn.inputs.force_constants, q2r.outputs.force_constants, "q2r.force_constants -> matdyn.force_constants")

    for name, node in (("ph", ph), ("q2r", q2r), ("matdyn", matdyn)):
        if str(node.inputs.code.uuid) != str(preflight["codes"][name]["uuid"]):
            raise AdapterError(f"{name} code UUID drifted from preflight")
    if _package_version("aiida-core") != str(preflight["aiida_core_version"]):
        raise AdapterError("aiida-core version drifted since preflight")
    if _package_version("aiida-quantumespresso") != str(preflight["aiida_quantumespresso_version"]):
        raise AdapterError("aiida-quantumespresso version drifted since preflight")
    if int(preflight["created_utc_unix_ms"]) > _utc_ms(ph.ctime):
        raise AdapterError("preflight was created after ph.x began")
    chronology = [_utc_ms(pw.mtime), _utc_ms(ph.ctime), _utc_ms(ph.mtime), _utc_ms(q2r.ctime), _utc_ms(q2r.mtime), _utc_ms(matdyn.ctime), _utc_ms(matdyn.mtime)]
    if chronology != sorted(chronology):
        raise AdapterError("AiiDA process chronology is not monotone")

    structure = _ns_get(pw.inputs, "structure")
    force_constants = _ns_get(q2r.outputs, "force_constants")
    bands = _ns_get(matdyn.outputs, "output_phonon_bands")
    ph_qpoints = _ns_get(ph.inputs, "qpoints")
    matdyn_kpoints = _ns_get(matdyn.inputs, "kpoints")
    if None in (structure, force_constants, bands, ph_qpoints, matdyn_kpoints):
        raise AdapterError("phonon chain lacks required structure/qpoints/force-constants/bands data")

    out = Path(args.output).resolve()
    _prepare_empty(out)
    raw = out / "raw"
    references: list[dict[str, Any]] = []
    execution_ids: dict[str, str] = {}
    execution_digests: set[str] = set()
    for stage, node in (("pw", pw), ("ph", ph), ("q2r", q2r), ("matdyn", matdyn)):
        digest, path = _artifact(out, f"{stage}-execution.json", _execution_manifest(node, stage))
        if not execution_digests.add(digest) if False else False:
            pass
        if digest in execution_digests:
            raise AdapterError("distinct process executions share an execution-manifest digest")
        execution_digests.add(digest)
        evidence_id = f"aiida:{node.uuid}:execution:{digest[:16]}"
        execution_ids[stage] = evidence_id
        references.append(_reference(evidence_id, "SolverExecution", digest, f"symthaea-export://{path.relative_to(out).as_posix()}", f"AiiDA QE {stage} execution {node.uuid}", _yyyymmdd(node.mtime)))
        generated_digest, generated_path = _artifact(out, f"{stage}-generated-input-tree.json", _copy_repository(node, raw / stage / "generated-input"))
        references.append(_reference(f"aiida:{node.uuid}:generated-input:{generated_digest[:16]}", "ArtifactContent", generated_digest, f"symthaea-export://{generated_path.relative_to(out).as_posix()}", f"Generated input repository for {stage} {node.uuid}", _yyyymmdd(node.ctime)))
        retrieved = _ns_get(node.outputs, "retrieved")
        if retrieved is None:
            raise AdapterError(f"{stage} has no retrieved FolderData")
        retrieved_digest, retrieved_path = _artifact(out, f"{stage}-retrieved-tree.json", _copy_repository(retrieved, raw / stage / "retrieved"))
        references.append(_reference(f"aiida:{node.uuid}:retrieved:{retrieved_digest[:16]}", "ArtifactContent", retrieved_digest, f"symthaea-export://{retrieved_path.relative_to(out).as_posix()}", f"Retrieved output repository for {stage} {node.uuid}", _yyyymmdd(node.mtime)))

    fc_digest, fc_path = _artifact(out, "force-constants-tree.json", _copy_repository(force_constants, raw / "q2r" / "force-constants"))
    fc_id = f"aiida:{force_constants.uuid}:force-constants:{fc_digest[:16]}"
    references.append(_reference(fc_id, "ArtifactContent", fc_digest, f"symthaea-export://{fc_path.relative_to(out).as_posix()}", f"ForceConstantsData {force_constants.uuid}", _yyyymmdd(force_constants.ctime)))

    bands_digest, bands_path = _artifact(out, "phonon-bands.json", _bands_payload(bands))
    bands_id = f"aiida:{bands.uuid}:phonon-bands:{bands_digest[:16]}"
    references.append(_reference(bands_id, "ArtifactContent", bands_digest, f"symthaea-export://{bands_path.relative_to(out).as_posix()}", f"Matdyn phonon BandsData {bands.uuid}", _yyyymmdd(bands.ctime)))

    phq_digest, phq_path = _artifact(out, "ph-qpoints.json", _kpoints_payload(ph_qpoints))
    phq_id = f"aiida:{ph_qpoints.uuid}:ph-qpoints:{phq_digest[:16]}"
    references.append(_reference(phq_id, "ArtifactContent", phq_digest, f"symthaea-export://{phq_path.relative_to(out).as_posix()}", f"ph.x q-point input {ph_qpoints.uuid}", _yyyymmdd(ph_qpoints.ctime)))
    matq_digest, matq_path = _artifact(out, "matdyn-kpoints.json", _kpoints_payload(matdyn_kpoints))
    matq_id = f"aiida:{matdyn_kpoints.uuid}:matdyn-kpoints:{matq_digest[:16]}"
    references.append(_reference(matq_id, "ArtifactContent", matq_digest, f"symthaea-export://{matq_path.relative_to(out).as_posix()}", f"matdyn sampling input {matdyn_kpoints.uuid}", _yyyymmdd(matdyn_kpoints.ctime)))

    preflight_copy = out / "artifacts" / "phonon-chain-preflight.json"
    shutil.copy2(preflight_path, preflight_copy)
    preflight_digest = _sha256_file(preflight_copy)
    preflight_id = f"aiida-phonon-preflight:{preflight_digest[:16]}"
    references.append(_reference(preflight_id, "ImplementationSnapshot", preflight_digest, "symthaea-export://artifacts/phonon-chain-preflight.json", "AiiDA QE phonon-chain preflight", int(preflight["created_utc_yyyymmdd"])))

    ids = [reference["id"] for reference in references]
    if len(ids) != len(set(ids)):
        raise AdapterError("duplicate evidence IDs in phonon-chain export")

    matdyn_parameters = _dict_value(_ns_get(matdyn.inputs, "parameters")) or {}
    chain = {
        "schema_version": CHAIN_SCHEMA,
        "source_system": "AiiDA",
        "processes": {stage: {"uuid": str(node.uuid), "pk": int(node.pk), "execution_evidence_id": execution_ids[stage]} for stage, node in (("pw", pw), ("ph", ph), ("q2r", q2r), ("matdyn", matdyn))},
        "lineage": {
            "pw_input_structure_uuid": str(structure.uuid),
            "pw_remote_folder_uuid": str(pw.outputs.remote_folder.uuid),
            "ph_parent_folder_uuid": str(ph.inputs.parent_folder.uuid),
            "ph_remote_folder_uuid": str(ph.outputs.remote_folder.uuid),
            "q2r_parent_folder_uuid": str(q2r.inputs.parent_folder.uuid),
            "q2r_force_constants_uuid": str(force_constants.uuid),
            "matdyn_force_constants_uuid": str(matdyn.inputs.force_constants.uuid),
            "matdyn_bands_uuid": str(bands.uuid),
        },
        "artifacts": {"force_constants_evidence_id": fc_id, "phonon_bands_evidence_id": bands_id, "ph_qpoints_evidence_id": phq_id, "matdyn_kpoints_evidence_id": matq_id, "preflight_evidence_id": preflight_id},
        "matdyn_asr_declared": matdyn_parameters.get("INPUT", {}).get("asr"),
        "interpretation": "explicit post-processing execution observed; numerical/physical convergence and phase stability are separate claims",
        "limitations": [
            "AiiDA ctime/mtime are provenance-record times, not independently trusted scheduler timestamps",
            "successful execution does not prove reciprocal-grid completeness or physical convergence",
            "matdyn ASR is exported as configured and must be compared against the preregistered benchmark policy separately",
            "thermodynamic phase stability is not established by this phonon chain",
        ],
    }
    _write_json(out / "phonon-chain.json", chain)
    _write_json(out / "evidence-bundle.json", {"references": references})
    _write_json(out / "export-manifest.json", {"adapter": {"name": ADAPTER_NAME, "version": ADAPTER_VERSION}, "phonon_chain": "phonon-chain.json", "evidence_bundle": "evidence-bundle.json", "preflight_artifact": "artifacts/phonon-chain-preflight.json", "preflight_sha256": preflight_digest})
    print(out)


def _self_test() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "x.json"
        digest = _write_json(path, {"b": 2, "a": 1})
        assert digest == _sha256_file(path)
    assert PW_PROCESS.endswith(".pw") and PH_PROCESS.endswith(".ph")
    assert Q2R_PROCESS.endswith(".q2r") and MATDYN_PROCESS.endswith(".matdyn")
    try:
        _safe_text("bad\ntext", "fixture")
    except AdapterError:
        pass
    else:
        raise AssertionError("control characters must fail closed")
    print("AiiDA QE phonon-chain exporter self-test: PASS")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    pre = subs.add_parser("preflight")
    pre.add_argument("--profile")
    pre.add_argument("--ph-code", required=True)
    pre.add_argument("--q2r-code", required=True)
    pre.add_argument("--matdyn-code", required=True)
    pre.add_argument("--solver-version", required=True)
    pre.add_argument("--output", required=True)
    exp = subs.add_parser("export")
    exp.add_argument("--profile")
    exp.add_argument("--preflight", required=True)
    exp.add_argument("--pw", required=True)
    exp.add_argument("--ph", required=True)
    exp.add_argument("--q2r", required=True)
    exp.add_argument("--matdyn", required=True)
    exp.add_argument("--output", required=True)
    subs.add_parser("self-test")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    try:
        if args.command == "preflight":
            _preflight(args)
        elif args.command == "export":
            _export(args)
        else:
            _self_test()
        return 0
    except AdapterError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

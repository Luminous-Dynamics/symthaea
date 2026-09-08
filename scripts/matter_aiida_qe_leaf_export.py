#!/usr/bin/env python3
"""Provenance-prepared AiiDA/Quantum ESPRESSO leaf exporter for Symthaea Matter.

Supported leaf calculations:
  * pw.x SCF            -> periodic_electronic_structure
  * pw.x relax/vc-relax -> structural_relaxation
  * ph.x                -> lattice_dynamics

Thermodynamic phase competition is intentionally refused: a convex hull is an
aggregate over multiple phase calculations/reference data, not one CalcJob.

`preflight` freezes the declared code/plugin/parser context before submission.
`export` accepts only a later successful, sealed, non-cached AiiDA CalcJob whose
code/task/package context still matches that preflight, then emits the Symthaea
crystal-solver-execution/v1 wire object and canonical evidence bundle.

Chronology remains declared rather than independently timestamp-attested. Process
success is not convergence, phase stability, synthesis, experiment, or replication.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ADAPTER_NAME = "symthaea-aiida-qe-leaf-export"
ADAPTER_VERSION = "1"
PREFLIGHT_SCHEMA = "symthaea.matter.aiida-qe-leaf-preflight/v1"
WIRE_SCHEMA = "symthaea.matter.crystal-solver-execution/v1"
MAX_JSON_BYTES = 8 * 1024 * 1024
MAX_TREE_FILES = 4096
MAX_FILE_BYTES = 2 * 1024 * 1024 * 1024

TASK_PERIODIC = "periodic_electronic_structure"
TASK_RELAX = "structural_relaxation"
TASK_LATTICE = "lattice_dynamics"
TASK_PHASE = "thermodynamic_phase_competition"
TASKS = (TASK_PERIODIC, TASK_RELAX, TASK_LATTICE, TASK_PHASE)
PW_PROCESS_TYPE = "aiida.calculations:quantumespresso.pw"
PH_PROCESS_TYPE = "aiida.calculations:quantumespresso.ph"


class AdapterError(RuntimeError):
    """Fail-closed adapter error."""


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
            if size > MAX_FILE_BYTES:
                raise AdapterError(f"artifact exceeds {MAX_FILE_BYTES} bytes: {path}")
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> str:
    payload = _json_bytes(value)
    if len(payload) > MAX_JSON_BYTES:
        raise AdapterError(f"JSON artifact exceeds {MAX_JSON_BYTES} bytes: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    digest = _sha256(payload)
    if _sha256_file(path) != digest:
        raise AdapterError(f"persisted JSON bytes failed digest round-trip: {path}")
    return digest


def _read_json(path: Path) -> Any:
    if not path.is_file():
        raise AdapterError(f"missing JSON artifact: {path}")
    if path.stat().st_size > MAX_JSON_BYTES:
        raise AdapterError(f"JSON input exceeds {MAX_JSON_BYTES} bytes: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AdapterError(f"cannot read JSON {path}: {error}") from error


def _safe_text(value: Any, field: str, max_len: int = 512) -> str:
    text = str(value)
    if not text.strip():
        raise AdapterError(f"{field} is empty")
    if len(text.encode("utf-8")) > max_len:
        raise AdapterError(f"{field} exceeds {max_len} UTF-8 bytes")
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in text):
        raise AdapterError(f"{field} contains a control character")
    return text


def _utc_ms(dt: datetime) -> int:
    if dt.tzinfo is None:
        raise AdapterError("AiiDA datetime is timezone-naive")
    return int(dt.timestamp() * 1000)


def _yyyymmdd(dt: datetime) -> int:
    if dt.tzinfo is None:
        raise AdapterError("datetime is timezone-naive")
    value = dt.astimezone(timezone.utc)
    return value.year * 10_000 + value.month * 100 + value.day


def _package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError as error:
        raise AdapterError(f"missing Python distribution: {distribution}") from error


def _expected_process_type(task: str) -> str:
    if task in (TASK_PERIODIC, TASK_RELAX):
        return PW_PROCESS_TYPE
    if task == TASK_LATTICE:
        return PH_PROCESS_TYPE
    if task == TASK_PHASE:
        raise AdapterError(
            "thermodynamic_phase_competition is not a leaf CalcJob; a workflow graph is required"
        )
    raise AdapterError(f"unsupported task: {task}")


def _parser_name(task: str) -> str:
    return "quantumespresso.pw" if _expected_process_type(task) == PW_PROCESS_TYPE else "quantumespresso.ph"


def _prepare_empty_dir(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise AdapterError(f"output directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _code_snapshot(code: Any) -> dict[str, Any]:
    computer = getattr(code, "computer", None)
    result: dict[str, Any] = {
        "code_uuid": str(code.uuid),
        "label": str(getattr(code, "label", "")),
        "full_label": str(getattr(code, "full_label", "")),
        "computer_uuid": str(getattr(computer, "uuid", "")) if computer else None,
        "computer_label": str(getattr(computer, "label", "")) if computer else None,
    }
    for field in ("filepath_executable", "default_calc_job_plugin"):
        value = getattr(code, field, None)
        if value is not None:
            result[field] = str(value)
    return result


def _preflight(args: argparse.Namespace) -> None:
    from aiida import load_profile, orm  # public AiiDA API; deliberately lazy

    load_profile(args.profile) if args.profile else load_profile()
    task = args.task
    process_type = _expected_process_type(task)
    code = orm.load_code(args.code)
    created = datetime.now(timezone.utc)
    claim_id = _safe_text(args.claim_id, "claim_id", 256)
    solver_version = _safe_text(args.solver_version, "solver_version", 128)
    aiida_version = _package_version("aiida-core")
    qe_plugin_version = _package_version("aiida-quantumespresso")

    out = Path(args.output).resolve()
    _prepare_empty_dir(out)
    environment = {
        "kind": "aiida-qe-execution-environment",
        "aiida_core_version": aiida_version,
        "aiida_quantumespresso_version": qe_plugin_version,
        "solver_name": "Quantum ESPRESSO",
        "solver_version_declared": solver_version,
        "code": _code_snapshot(code),
    }
    parser = {
        "kind": "aiida-qe-parser-snapshot",
        "aiida_quantumespresso_version": qe_plugin_version,
        "parser_name": _parser_name(task),
        "process_type": process_type,
    }
    env_path = out / "environment-snapshot.json"
    parser_path = out / "parser-snapshot.json"
    env_digest = _write_json(env_path, environment)
    parser_digest = _write_json(parser_path, parser)
    preflight = {
        "schema_version": PREFLIGHT_SCHEMA,
        "created_utc_unix_ms": _utc_ms(created),
        "created_utc_yyyymmdd": _yyyymmdd(created),
        "claim_id": claim_id,
        "task": task,
        "expected_process_type": process_type,
        "solver_version_declared": solver_version,
        "aiida_core_version": aiida_version,
        "aiida_quantumespresso_version": qe_plugin_version,
        "code_uuid": str(code.uuid),
        "environment_snapshot": {
            "path": env_path.name,
            "sha256": env_digest,
            "evidence_id": f"aiida-preflight:{code.uuid}:environment:{env_digest[:16]}",
        },
        "parser_snapshot": {
            "path": parser_path.name,
            "sha256": parser_digest,
            "evidence_id": f"aiida-preflight:{code.uuid}:parser:{parser_digest[:16]}",
        },
        "interpretation": "declared pre-execution context only; not a trusted timestamp or executable attestation",
    }
    _write_json(out / "preflight.json", preflight)
    print(out / "preflight.json")


def _validate_preflight(path: Path) -> dict[str, Any]:
    value = _read_json(path)
    if not isinstance(value, dict) or value.get("schema_version") != PREFLIGHT_SCHEMA:
        raise AdapterError("unsupported or malformed AiiDA QE preflight")
    required = (
        "created_utc_unix_ms",
        "created_utc_yyyymmdd",
        "claim_id",
        "task",
        "expected_process_type",
        "solver_version_declared",
        "aiida_core_version",
        "aiida_quantumespresso_version",
        "code_uuid",
        "environment_snapshot",
        "parser_snapshot",
    )
    missing = [field for field in required if field not in value]
    if missing:
        raise AdapterError(f"preflight missing fields: {', '.join(missing)}")
    expected_type = _expected_process_type(str(value["task"]))
    if value["expected_process_type"] != expected_type:
        raise AdapterError("preflight task/process type disagree")
    root = path.parent.resolve()
    for field in ("environment_snapshot", "parser_snapshot"):
        descriptor = value[field]
        if not isinstance(descriptor, dict):
            raise AdapterError(f"{field} descriptor is not an object")
        relative = _safe_text(descriptor.get("path", ""), f"{field}.path", 256)
        artifact = (root / relative).resolve()
        if root not in artifact.parents:
            raise AdapterError(f"{field} path escapes preflight directory")
        expected = _safe_text(descriptor.get("sha256", ""), f"{field}.sha256", 64).lower()
        if len(expected) != 64 or any(ch not in "0123456789abcdef" for ch in expected):
            raise AdapterError(f"{field} has malformed SHA-256")
        if _sha256_file(artifact) != expected:
            raise AdapterError(f"{field} literal bytes no longer match preflight SHA-256")
        _safe_text(descriptor.get("evidence_id", ""), f"{field}.evidence_id", 256)
    return value


def _tree_manifest(root: Path) -> dict[str, Any]:
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if not files:
        raise AdapterError(f"repository tree contains no files: {root}")
    if len(files) > MAX_TREE_FILES:
        raise AdapterError(f"repository tree exceeds {MAX_TREE_FILES} files: {root}")
    entries: list[dict[str, Any]] = []
    total = 0
    for path in files:
        size = path.stat().st_size
        if size > MAX_FILE_BYTES:
            raise AdapterError(f"repository file exceeds {MAX_FILE_BYTES} bytes: {path}")
        total += size
        entries.append(
            {"path": path.relative_to(root).as_posix(), "size": size, "sha256": _sha256_file(path)}
        )
    return {"files": entries, "file_count": len(entries), "total_bytes": total}


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
        raise AdapterError("AiiDA Dict node returned non-dict content")
    return value


def _pseudo_items(namespace: Any | None) -> list[tuple[str, Any]]:
    if namespace is None:
        return []
    if hasattr(namespace, "items"):
        return sorted((str(key), value) for key, value in namespace.items())
    try:
        return [(str(name), getattr(namespace, name)) for name in sorted(namespace)]
    except TypeError as error:
        raise AdapterError("cannot enumerate AiiDA pseudos namespace") from error


def _configuration(node: Any, task: str) -> dict[str, Any]:
    inputs = node.inputs
    code = _ns_get(inputs, "code")
    structure = _ns_get(inputs, "structure")
    kpoints = _ns_get(inputs, "kpoints")
    parent = _ns_get(inputs, "parent_folder")
    pseudos = _pseudo_items(_ns_get(inputs, "pseudos"))
    return {
        "task": task,
        "process_type": str(node.process_type),
        "parameters": _dict_value(_ns_get(inputs, "parameters")),
        "settings": _dict_value(_ns_get(inputs, "settings")),
        "code_uuid": str(getattr(code, "uuid", "")) if code else None,
        "structure_uuid": str(getattr(structure, "uuid", "")) if structure else None,
        "kpoints_uuid": str(getattr(kpoints, "uuid", "")) if kpoints else None,
        "parent_folder_uuid": str(getattr(parent, "uuid", "")) if parent else None,
        "pseudos": {kind: str(pseudo.uuid) for kind, pseudo in pseudos},
    }


def _validate_task(node: Any, task: str) -> None:
    expected = _expected_process_type(task)
    if str(node.process_type) != expected:
        raise AdapterError(f"process type mismatch: expected {expected}, got {node.process_type}")
    if expected == PW_PROCESS_TYPE:
        parameters = _dict_value(_ns_get(node.inputs, "parameters")) or {}
        calculation = str(parameters.get("CONTROL", {}).get("calculation", "scf")).lower()
        if task == TASK_PERIODIC and calculation != "scf":
            raise AdapterError("periodic leaf requires pw.x CONTROL.calculation='scf'")
        if task == TASK_RELAX and calculation not in ("relax", "vc-relax"):
            raise AdapterError("relaxation leaf requires pw.x CONTROL.calculation='relax' or 'vc-relax'")


def _is_cached(node: Any) -> bool:
    value = getattr(node.base.caching, "is_created_from_cache", False)
    return bool(value() if callable(value) else value)


def _reference(evidence_id: str, role: str, digest: str, locator: str, subject: str, date: int) -> dict[str, Any]:
    return {
        "id": evidence_id,
        "role": role,
        "content_sha256": digest,
        "locator": locator,
        "subject": subject,
        "claimed_utc_date": date,
        "issuer": ADAPTER_NAME,
    }


def _artifact(out: Path, filename: str, value: Any) -> tuple[str, Path]:
    path = out / "artifacts" / filename
    return _write_json(path, value), path


def _export(args: argparse.Namespace) -> None:
    from aiida import load_profile, orm  # public AiiDA API; deliberately lazy

    preflight_path = Path(args.preflight).resolve()
    preflight = _validate_preflight(preflight_path)
    load_profile(args.profile) if args.profile else load_profile()
    node = orm.load_node(args.node)
    if not isinstance(node, orm.CalcJobNode):
        raise AdapterError("leaf exporter requires an AiiDA CalcJobNode")
    if not bool(getattr(node, "is_sealed", False)):
        raise AdapterError("CalcJob must be sealed")
    if not bool(node.is_finished_ok):
        raise AdapterError(f"CalcJob did not finish successfully (exit_status={node.exit_status})")
    if _is_cached(node):
        raise AdapterError("cached CalcJob is not a new solver execution; export its cache source instead")

    task = str(preflight["task"])
    _validate_task(node, task)
    claim_id = _safe_text(preflight["claim_id"], "claim_id", 256)
    code = _ns_get(node.inputs, "code")
    if code is None or str(code.uuid) != str(preflight["code_uuid"]):
        raise AdapterError("CalcJob Code UUID does not match preflight")
    if _package_version("aiida-core") != str(preflight["aiida_core_version"]):
        raise AdapterError("aiida-core version drifted since preflight")
    if _package_version("aiida-quantumespresso") != str(preflight["aiida_quantumespresso_version"]):
        raise AdapterError("aiida-quantumespresso version drifted since preflight")

    started_ms, finished_ms = _utc_ms(node.ctime), _utc_ms(node.mtime)
    if int(preflight["created_utc_unix_ms"]) > started_ms:
        raise AdapterError("preflight was created after the AiiDA process record began")
    execution_date, input_date = _yyyymmdd(node.mtime), _yyyymmdd(node.ctime)
    preflight_date = int(preflight["created_utc_yyyymmdd"])
    if preflight_date > execution_date:
        raise AdapterError("preflight date postdates execution")

    out = Path(args.output).resolve()
    _prepare_empty_dir(out)
    raw = out / "raw"
    input_digest, input_path = _artifact(
        out, "generated-input-tree.json", _copy_repository(node, raw / "generated-input")
    )
    retrieved = _ns_get(node.outputs, "retrieved")
    if retrieved is None:
        raise AdapterError("QE CalcJob has no retrieved FolderData")
    retrieved_digest, retrieved_path = _artifact(
        out, "retrieved-tree.json", _copy_repository(retrieved, raw / "retrieved")
    )
    config_digest, config_path = _artifact(out, "configuration.json", _configuration(node, task))
    output_parameters = _dict_value(_ns_get(node.outputs, "output_parameters"))
    if output_parameters is None:
        raise AdapterError("QE CalcJob lacks structured output_parameters")
    params_digest, params_path = _artifact(out, "output-parameters.json", output_parameters)

    execution_manifest = {
        "kind": "aiida-calcjob-execution",
        "node_uuid": str(node.uuid),
        "node_pk": int(node.pk),
        "process_type": str(node.process_type),
        "process_label": str(node.process_label),
        "exit_status": int(node.exit_status or 0),
        "is_finished_ok": bool(node.is_finished_ok),
        "is_sealed": bool(getattr(node, "is_sealed", False)),
        "ctime_utc": node.ctime.astimezone(timezone.utc).isoformat(),
        "mtime_utc": node.mtime.astimezone(timezone.utc).isoformat(),
        "code_uuid": str(code.uuid),
        "claim_id": claim_id,
        "task": task,
    }
    execution_digest, execution_path = _artifact(out, "execution-manifest.json", execution_manifest)

    preflight_root = preflight_path.parent
    env = preflight["environment_snapshot"]
    parser = preflight["parser_snapshot"]
    env_source = (preflight_root / env["path"]).resolve()
    parser_source = (preflight_root / parser["path"]).resolve()
    shutil.copy2(env_source, out / "artifacts" / "environment-snapshot.json")
    shutil.copy2(parser_source, out / "artifacts" / "parser-snapshot.json")
    env_digest, parser_digest = str(env["sha256"]), str(parser["sha256"])
    env_id = _safe_text(env["evidence_id"], "environment evidence id", 256)
    parser_id = _safe_text(parser["evidence_id"], "parser evidence id", 256)

    pseudo_refs: list[dict[str, Any]] = []
    dependency_ids: list[str] = []
    for kind, pseudo in _pseudo_items(_ns_get(node.inputs, "pseudos")):
        manifest = {
            "kind": "aiida-pseudopotential-dependency",
            "kind_name": kind,
            "node_uuid": str(pseudo.uuid),
            "repository": _copy_repository(pseudo, raw / "dependencies" / kind),
        }
        digest, path = _artifact(out, f"pseudo-{kind}.json", manifest)
        evidence_id = f"aiida:{node.uuid}:pseudo:{kind}:{digest[:16]}"
        dependency_ids.append(evidence_id)
        pseudo_refs.append(
            _reference(
                evidence_id,
                "DependencySnapshot",
                digest,
                f"symthaea-export://{path.relative_to(out).as_posix()}",
                f"AiiDA QE pseudopotential {kind} for {node.uuid}",
                _yyyymmdd(pseudo.ctime),
            )
        )

    execution_id = f"aiida:{node.uuid}:execution:{execution_digest[:16]}"
    input_id = f"aiida:{node.uuid}:input:{input_digest[:16]}"
    config_id = f"aiida:{node.uuid}:config:{config_digest[:16]}"
    retrieved_id = f"aiida:{node.uuid}:retrieved:{retrieved_digest[:16]}"
    params_id = f"aiida:{node.uuid}:output-parameters:{params_digest[:16]}"
    references = [
        _reference(execution_id, "SolverExecution", execution_digest, f"symthaea-export://{execution_path.relative_to(out).as_posix()}", f"AiiDA CalcJob {node.uuid}", execution_date),
        _reference(input_id, "ArtifactContent", input_digest, f"symthaea-export://{input_path.relative_to(out).as_posix()}", f"Generated input tree for {node.uuid}", input_date),
        _reference(config_id, "ArtifactContent", config_digest, f"symthaea-export://{config_path.relative_to(out).as_posix()}", f"Structured configuration for {node.uuid}", input_date),
        _reference(env_id, "ImplementationSnapshot", env_digest, "symthaea-preflight://environment-snapshot.json", f"Preflight environment for {claim_id}", preflight_date),
        _reference(parser_id, "ImplementationSnapshot", parser_digest, "symthaea-preflight://parser-snapshot.json", f"Preflight parser for {claim_id}", preflight_date),
        _reference(retrieved_id, "ArtifactContent", retrieved_digest, f"symthaea-export://{retrieved_path.relative_to(out).as_posix()}", f"Retrieved output tree for {node.uuid}", execution_date),
        _reference(params_id, "ArtifactContent", params_digest, f"symthaea-export://{params_path.relative_to(out).as_posix()}", f"Structured output_parameters for {node.uuid}", execution_date),
        *pseudo_refs,
    ]
    digests = [reference["content_sha256"] for reference in references]
    if len(digests) != len(set(digests)):
        raise AdapterError("distinct execution semantic references resolve to identical content")

    component = "pw.x" if str(node.process_type) == PW_PROCESS_TYPE else "ph.x"
    wire = {
        "schema_version": WIRE_SCHEMA,
        "claim_id": claim_id,
        "task": task,
        "execution_evidence_id": execution_id,
        "declared_execution_yyyymmdd": execution_date,
        "started_unix_ms": started_ms,
        "finished_unix_ms": finished_ms,
        "exit_code": int(node.exit_status or 0),
        "solver_name": f"Quantum ESPRESSO {component}",
        "solver_version": str(preflight["solver_version_declared"]),
        "adapter_name": ADAPTER_NAME,
        "adapter_version": ADAPTER_VERSION,
        "input_artifact_id": input_id,
        "configuration_artifact_id": config_id,
        "environment_snapshot_id": env_id,
        "parser_snapshot_id": parser_id,
        "output_artifact_ids": [retrieved_id, params_id],
        "dependency_snapshot_ids": dependency_ids,
    }
    _write_json(out / "execution.json", wire)
    _write_json(out / "evidence-bundle.json", {"references": references})
    _write_json(
        out / "export-manifest.json",
        {
            "adapter": {"name": ADAPTER_NAME, "version": ADAPTER_VERSION},
            "source": {"system": "AiiDA", "calcjob_uuid": str(node.uuid), "calcjob_pk": int(node.pk)},
            "wire": "execution.json",
            "evidence_bundle": "evidence-bundle.json",
            "preflight_sha256": _sha256_file(preflight_path),
            "limitations": [
                "chronology is declared, not independently timestamp-attested",
                "solver version is preflight-declared, not remotely re-attested",
                "AiiDA ctime/mtime bound the process record, not scheduler-exact runtime",
                "leaf export is not convergence, phase stability, synthesis, experiment, or replication",
            ],
        },
    )
    print(out)


def _self_test() -> None:
    assert _expected_process_type(TASK_PERIODIC) == PW_PROCESS_TYPE
    assert _expected_process_type(TASK_RELAX) == PW_PROCESS_TYPE
    assert _expected_process_type(TASK_LATTICE) == PH_PROCESS_TYPE
    try:
        _expected_process_type(TASK_PHASE)
    except AdapterError:
        pass
    else:
        raise AssertionError("phase competition must refuse leaf export")
    assert _json_bytes({"b": 2, "a": 1}) == _json_bytes({"a": 1, "b": 2})
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "artifact.json"
        digest = _write_json(path, {"proof": "exact-bytes"})
        assert digest == _sha256_file(path)
    try:
        _safe_text("bad\ntext", "fixture")
    except AdapterError:
        pass
    else:
        raise AssertionError("control characters must fail closed")
    print("AiiDA QE leaf exporter self-test: PASS")


def _arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    before = sub.add_parser("preflight", help="freeze AiiDA/QE context before submission")
    before.add_argument("--profile")
    before.add_argument("--code", required=True)
    before.add_argument("--claim-id", required=True)
    before.add_argument("--task", required=True, choices=TASKS)
    before.add_argument("--solver-version", required=True)
    before.add_argument("--output", required=True)
    after = sub.add_parser("export", help="export a finished CalcJob against a preflight")
    after.add_argument("--profile")
    after.add_argument("--preflight", required=True)
    after.add_argument("--node", required=True)
    after.add_argument("--output", required=True)
    sub.add_parser("self-test", help="run dependency-free adapter invariants")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _arg_parser().parse_args(list(argv) if argv is not None else None)
    try:
        if args.command == "preflight":
            _preflight(args)
        elif args.command == "export":
            _export(args)
        elif args.command == "self-test":
            _self_test()
        else:
            raise AdapterError(f"unsupported command: {args.command}")
        return 0
    except AdapterError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

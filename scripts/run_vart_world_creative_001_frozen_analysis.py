#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

EXPERIMENT_ID = "VART-WORLD-CREATIVE-001"
SCHEMA = "symthaea.vart-world-creative-001.frozen-analysis-run.v1"
CAMPAIGN_RECEIPT_REL = "_orchestrator/CONFIRMATORY_CAMPAIGN_RECEIPT.json"


class AnalysisRunError(RuntimeError):
    pass


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                h.update(chunk)
    except FileNotFoundError as exc:
        raise AnalysisRunError(f"missing file: {path}") from exc
    return h.hexdigest()


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise AnalysisRunError(f"missing file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise AnalysisRunError(f"invalid JSON at {path}: {exc}") from exc


def write_new_json(path: Path, value: Any) -> None:
    if path.exists():
        raise AnalysisRunError(f"refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(value) + b"\n")


def resolve(value: str, base: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise AnalysisRunError(f"{label} must be lowercase SHA-256 hex")
    return value


def require_external(path: Path, protected_root: Path, label: str) -> None:
    try:
        path.relative_to(protected_root)
    except ValueError:
        return
    raise AnalysisRunError(f"{label} must be outside sealed evidence root: {path}")


def tree_closure(root: Path, excluded: set[str]) -> tuple[str, list[dict[str, str]]]:
    entries: list[dict[str, str]] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix()
        if rel not in excluded:
            entries.append({"path": rel, "sha256": sha256_file(path)})
    return sha256_bytes(canonical_bytes(entries)), entries


def expand_argv(template: Any, values: dict[str, str]) -> list[str]:
    if not isinstance(template, list) or not template:
        raise AnalysisRunError("analysis_argv must be a non-empty array")
    out: list[str] = []
    for raw in template:
        if not isinstance(raw, str):
            raise AnalysisRunError("analysis_argv entries must be strings")
        if raw.startswith("__REPLACE_"):
            raise AnalysisRunError("analysis_argv contains unresolved placeholder")
        try:
            expanded = raw.format_map(values)
        except KeyError as exc:
            raise AnalysisRunError(f"unknown analysis_argv placeholder: {exc}") from exc
        if expanded.startswith("__REPLACE_"):
            raise AnalysisRunError("analysis_argv contains unresolved placeholder")
        out.append(expanded)
    return out


def verify_sealed_inputs(
    evidence_root: Path,
    expected_freeze_sha256: str,
    expected_campaign_receipt_sha256: str,
) -> dict[str, Any]:
    freeze_path = evidence_root / "confirmatory_freeze.json"
    if sha256_file(freeze_path) != expected_freeze_sha256:
        raise AnalysisRunError("confirmatory freeze differs from external anchor")

    receipt_path = evidence_root / CAMPAIGN_RECEIPT_REL
    if sha256_file(receipt_path) != expected_campaign_receipt_sha256:
        raise AnalysisRunError("campaign receipt differs from externally supplied digest")
    receipt = read_json(receipt_path)
    if not isinstance(receipt, dict):
        raise AnalysisRunError("campaign receipt must be an object")
    required = {
        "status": "sealed",
        "freeze_sha256": expected_freeze_sha256,
        "zero_peeking": True,
        "automatic_retry": False,
        "claim_authorized": False,
    }
    for key, expected in required.items():
        if receipt.get(key) != expected:
            raise AnalysisRunError(f"campaign receipt {key} mismatch")
    if receipt.get("trial_count") != 64:
        raise AnalysisRunError(f"campaign receipt trial_count={receipt.get('trial_count')} expected 64")

    closure, entries = tree_closure(evidence_root, {CAMPAIGN_RECEIPT_REL})
    if receipt.get("evidence_closure_sha256") != closure:
        raise AnalysisRunError("sealed evidence closure changed after campaign receipt")
    if receipt.get("evidence_file_count") != len(entries):
        raise AnalysisRunError("sealed evidence file count changed after campaign receipt")

    freeze = read_json(freeze_path)
    if not isinstance(freeze, dict) or freeze.get("experiment_id") != EXPERIMENT_ID:
        raise AnalysisRunError("freeze experiment identity mismatch")
    analysis_sha = require_sha256(freeze.get("analysis_contract_sha256"), "freeze.analysis_contract_sha256")
    metric_sha = require_sha256(freeze.get("metric_definition_set_sha256"), "freeze.metric_definition_set_sha256")
    if sha256_file(evidence_root / "analysis_contract.json") != analysis_sha:
        raise AnalysisRunError("analysis_contract.json differs from frozen digest")
    if sha256_file(evidence_root / "metric_definitions.json") != metric_sha:
        raise AnalysisRunError("metric_definitions.json differs from frozen digest")

    return {
        "receipt": receipt,
        "freeze": freeze,
        "evidence_closure_sha256": closure,
        "evidence_file_count": len(entries),
        "analysis_contract_sha256": analysis_sha,
        "metric_definition_set_sha256": metric_sha,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="One-shot pre-anchored execution wrapper for the already-frozen VART confirmatory analysis"
    )
    parser.add_argument("config", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = read_json(config_path)
    if not isinstance(config, dict) or config.get("schema") != SCHEMA or config.get("experiment_id") != EXPERIMENT_ID:
        raise AnalysisRunError("unexpected frozen-analysis config")
    if config.get("claim_authorized") is not False:
        raise AnalysisRunError("analysis wrapper requires claim_authorized=false")

    base = config_path.parent
    evidence_root = resolve(config["evidence_root"], base)
    if not evidence_root.is_dir():
        raise AnalysisRunError(f"evidence root not found: {evidence_root}")
    expected_freeze = require_sha256(config.get("expected_freeze_sha256"), "expected_freeze_sha256")
    expected_campaign = require_sha256(
        config.get("expected_campaign_receipt_sha256"), "expected_campaign_receipt_sha256"
    )
    sealed = verify_sealed_inputs(evidence_root, expected_freeze, expected_campaign)

    analysis_program = resolve(config["analysis_program_path"], base)
    if not analysis_program.is_file():
        raise AnalysisRunError(f"analysis program not found: {analysis_program}")
    expected_program_sha = require_sha256(
        config.get("expected_analysis_program_sha256"), "expected_analysis_program_sha256"
    )
    actual_program_sha = sha256_file(analysis_program)
    if actual_program_sha != expected_program_sha:
        raise AnalysisRunError(
            f"analysis program digest mismatch: {actual_program_sha} != {expected_program_sha}"
        )

    output_root = resolve(config["analysis_output_root"], base)
    pre_anchor = resolve(config["preunblind_anchor_path"], base)
    require_external(output_root, evidence_root, "analysis_output_root")
    require_external(pre_anchor, evidence_root, "preunblind_anchor_path")
    if output_root.exists() and any(output_root.iterdir()):
        raise AnalysisRunError(f"analysis output root must be fresh and empty: {output_root}")
    if pre_anchor.exists():
        raise AnalysisRunError(f"pre-unblinding anchor already exists: {pre_anchor}")

    values = {
        "analysis_program": str(analysis_program),
        "evidence_root": str(evidence_root),
        "output_root": str(output_root),
        "freeze_sha256": expected_freeze,
        "campaign_receipt_sha256": expected_campaign,
    }
    argv = expand_argv(config.get("analysis_argv"), values)
    if str(analysis_program) not in argv:
        raise AnalysisRunError("analysis_argv must contain the pinned analysis_program_path")
    argv_sha = sha256_bytes(canonical_bytes(argv))
    config_sha = sha256_file(config_path)

    dry = {
        "verdict": "ANALYSIS_DRY_RUN_READY",
        "experiment_id": EXPERIMENT_ID,
        "freeze_sha256": expected_freeze,
        "campaign_receipt_sha256": expected_campaign,
        "evidence_closure_sha256": sealed["evidence_closure_sha256"],
        "analysis_contract_sha256": sealed["analysis_contract_sha256"],
        "metric_definition_set_sha256": sealed["metric_definition_set_sha256"],
        "analysis_program_sha256": actual_program_sha,
        "analysis_argv_sha256": argv_sha,
        "analysis_config_sha256": config_sha,
        "automatic_retry": False,
        "claim_authorized": False,
    }
    if args.dry_run:
        print(json.dumps(dry, sort_keys=True))
        return 0

    anchor = {
        "schema": "symthaea.vart-world-creative-001.preunblind-analysis-anchor.v1",
        **{k: v for k, v in dry.items() if k != "verdict"},
        "status": "anchored_before_unblinding",
        "analysis_argv": argv,
    }
    write_new_json(pre_anchor, anchor)
    pre_anchor_sha = sha256_file(pre_anchor)

    output_root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "VART_EXPERIMENT_ID": EXPERIMENT_ID,
            "VART_EVIDENCE_ROOT": str(evidence_root),
            "VART_ANALYSIS_OUTPUT_ROOT": str(output_root),
            "VART_FREEZE_SHA256": expected_freeze,
            "VART_CAMPAIGN_RECEIPT_SHA256": expected_campaign,
            "VART_PREUNBLIND_ANCHOR_SHA256": pre_anchor_sha,
            "VART_UNBLINDED": "1",
            "VART_CLAIM_AUTHORIZED": "0",
        }
    )
    proc = subprocess.run(
        argv,
        cwd=analysis_program.parent,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    stdout_path = output_root / "analysis.stdout.txt"
    stderr_path = output_root / "analysis.stderr.txt"
    stdout_path.write_text(proc.stdout, encoding="utf-8")
    stderr_path.write_text(proc.stderr, encoding="utf-8")
    stdout_path.chmod(0o600)
    stderr_path.chmod(0o600)

    if proc.returncode != 0:
        abort = {
            "schema": "symthaea.vart-world-creative-001.analysis-abort-receipt.v1",
            "experiment_id": EXPERIMENT_ID,
            "freeze_sha256": expected_freeze,
            "campaign_receipt_sha256": expected_campaign,
            "preunblind_anchor_sha256": pre_anchor_sha,
            "analysis_program_sha256": actual_program_sha,
            "analysis_argv_sha256": argv_sha,
            "returncode": proc.returncode,
            "stdout_sha256": sha256_file(stdout_path),
            "stderr_sha256": sha256_file(stderr_path),
            "automatic_retry": False,
            "claim_authorized": False,
        }
        write_new_json(output_root / "ANALYSIS_ABORT_RECEIPT.json", abort)
        print(
            json.dumps(
                {
                    "verdict": "FROZEN_ANALYSIS_ABORTED",
                    "returncode": proc.returncode,
                    "preunblind_anchor_sha256": pre_anchor_sha,
                    "automatic_retry": False,
                    "claim_authorized": False,
                },
                sort_keys=True,
            )
        )
        return 2

    closure, entries = tree_closure(output_root, {"ANALYSIS_RECEIPT.json"})
    receipt = {
        "schema": "symthaea.vart-world-creative-001.analysis-receipt.v1",
        "experiment_id": EXPERIMENT_ID,
        "status": "sealed",
        "freeze_sha256": expected_freeze,
        "campaign_receipt_sha256": expected_campaign,
        "evidence_closure_sha256": sealed["evidence_closure_sha256"],
        "analysis_contract_sha256": sealed["analysis_contract_sha256"],
        "metric_definition_set_sha256": sealed["metric_definition_set_sha256"],
        "preunblind_anchor_sha256": pre_anchor_sha,
        "analysis_config_sha256": config_sha,
        "analysis_program_sha256": actual_program_sha,
        "analysis_argv_sha256": argv_sha,
        "returncode": proc.returncode,
        "stdout_sha256": sha256_file(stdout_path),
        "stderr_sha256": sha256_file(stderr_path),
        "analysis_output_closure_sha256": closure,
        "analysis_output_file_count": len(entries),
        "automatic_retry": False,
        "claim_authorized": False,
    }
    write_new_json(output_root / "ANALYSIS_RECEIPT.json", receipt)
    receipt_sha = sha256_file(output_root / "ANALYSIS_RECEIPT.json")
    print(
        json.dumps(
            {
                "verdict": "FROZEN_ANALYSIS_EXECUTED_AND_SEALED",
                "analysis_receipt_sha256": receipt_sha,
                "analysis_output_closure_sha256": closure,
                "preunblind_anchor_sha256": pre_anchor_sha,
                "claim_authorized": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AnalysisRunError, OSError, ValueError) as exc:
        print(
            json.dumps(
                {
                    "verdict": "FROZEN_ANALYSIS_REJECT",
                    "error": str(exc),
                    "automatic_retry": False,
                    "claim_authorized": False,
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        raise SystemExit(2)

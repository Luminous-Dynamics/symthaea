#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

EXPERIMENT_ID = "VART-WORLD-CREATIVE-001"
PAIR_KEYS = ("fixture", "seed", "revision_index")
RECEIPT_REL = "_orchestrator/PILOT_RECEIPT.json"
PLAN_REL = "_orchestrator/resolved_plan.json"
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")


class AuditError(RuntimeError):
    def __init__(self, code: str, detail: str):
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def require(cond: bool, code: str, detail: str) -> None:
    if not cond:
        raise AuditError(code, detail)


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise AuditError("PILOT_AUDIT_EVIDENCE_MISSING", str(path)) from exc
    except json.JSONDecodeError as exc:
        raise AuditError("PILOT_AUDIT_JSON_INVALID", f"{path}: {exc}") from exc


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def tree_closure(root: Path, *, excluded_relpaths: set[str]) -> str:
    entries: list[dict[str, str]] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix()
        if rel in excluded_relpaths:
            continue
        entries.append({"path": rel, "sha256": sha256_file(path)})
    return hashlib.sha256(canonical_json_bytes(entries)).hexdigest()


def hex40(value: Any, label: str) -> str:
    require(
        isinstance(value, str) and HEX40.fullmatch(value) is not None,
        "PILOT_AUDIT_RECEIPT_INVALID",
        label,
    )
    return value


def hex64(value: Any, label: str) -> str:
    require(
        isinstance(value, str) and HEX64.fullmatch(value) is not None,
        "PILOT_AUDIT_RECEIPT_INVALID",
        label,
    )
    return value


def _cell_from_plan_entry(entry: Any) -> dict[str, Any]:
    require(isinstance(entry, dict), "PILOT_AUDIT_PLAN_INVALID", "plan entry must be an object")
    cell = entry.get("cell")
    require(isinstance(cell, dict), "PILOT_AUDIT_PLAN_INVALID", "plan entry missing cell")
    for key in (
        "cell_id",
        "trial_id",
        "policy",
        "fixture",
        "seed",
        "revision_index",
        "paired_block_id",
    ):
        require(
            key in cell,
            "PILOT_AUDIT_PLAN_INVALID",
            f"{cell.get('cell_id', '?')}: missing {key}",
        )
    return cell


def verify_paired_design(cells: list[dict[str, Any]]) -> dict[str, Any]:
    blocks: dict[str, list[dict[str, Any]]] = {}
    for cell in cells:
        block = cell["paired_block_id"]
        require(
            isinstance(block, str) and block,
            "PILOT_AUDIT_PLAN_INVALID",
            "empty paired_block_id",
        )
        blocks.setdefault(block, []).append(cell)

    block_receipts: dict[str, Any] = {}
    for block, members in sorted(blocks.items()):
        policies = [m["policy"] for m in members]
        require(
            len(policies) == len(set(policies)),
            "PAIRED_BLOCK_DUPLICATE_POLICY",
            f"{block}: duplicate policy labels {policies}",
        )
        anchor = {key: members[0][key] for key in PAIR_KEYS}
        for member in members[1:]:
            observed = {key: member[key] for key in PAIR_KEYS}
            require(
                observed == anchor,
                "PAIRED_BLOCK_WORLD_INPUT_MISMATCH",
                f"{block}: {member['cell_id']} {observed} != {members[0]['cell_id']} {anchor}",
            )
        block_receipts[block] = {
            **anchor,
            "policies": sorted(policies),
            "cell_ids": [m["cell_id"] for m in members],
            "trial_ids": [m["trial_id"] for m in members],
        }
    return block_receipts


def verify_dual_source(receipt: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    source = receipt.get("source")
    require(isinstance(source, dict), "PILOT_AUDIT_RECEIPT_INVALID", "source")
    subject_head = hex40(source.get("head"), "source.head")
    subject_tree = hex40(source.get("tree"), "source.tree")
    for key in ("parent_v05a_head", "parent_v05a_tree"):
        hex40(source.get(key), f"source.{key}")

    dual = receipt.get("dual_source_instrument_external") is True
    if not dual:
        return {
            "dual_source_bound": False,
            "source_head": subject_head,
            "source_tree": subject_tree,
            "instrument_source_head": None,
            "instrument_source_tree": None,
            "preexecution_anchor_sha256": None,
            "pilot_config_sha256": None,
        }

    require(
        plan.get("subject_source_head") == subject_head
        and plan.get("subject_source_tree") == subject_tree,
        "PILOT_AUDIT_SOURCE_PAIR_MISMATCH",
        "subject source plan/receipt",
    )
    instrument = receipt.get("instrument_source")
    require(
        isinstance(instrument, dict),
        "PILOT_AUDIT_RECEIPT_INVALID",
        "instrument_source",
    )
    instrument_head = hex40(instrument.get("head"), "instrument_source.head")
    instrument_tree = hex40(instrument.get("tree"), "instrument_source.tree")
    require(
        plan.get("instrument_source_head") == instrument_head
        and plan.get("instrument_source_tree") == instrument_tree,
        "PILOT_AUDIT_SOURCE_PAIR_MISMATCH",
        "instrument source plan/receipt",
    )

    preanchor = hex64(receipt.get("preexecution_anchor_sha256"), "preexecution_anchor_sha256")
    config_sha = hex64(receipt.get("pilot_config_sha256"), "pilot_config_sha256")
    design_sha = hex64(receipt.get("pilot_design_sha256"), "pilot_design_sha256")
    require(
        plan.get("preexecution_anchor_sha256") == preanchor,
        "PILOT_AUDIT_ANCHOR_MISMATCH",
        "preexecution anchor plan/receipt",
    )
    require(
        plan.get("pilot_config_sha256") == config_sha,
        "PILOT_AUDIT_ANCHOR_MISMATCH",
        "pilot config plan/receipt",
    )
    require(
        plan.get("pilot_design_sha256") == design_sha,
        "PILOT_AUDIT_ANCHOR_MISMATCH",
        "pilot design plan/receipt",
    )
    return {
        "dual_source_bound": True,
        "source_head": subject_head,
        "source_tree": subject_tree,
        "instrument_source_head": instrument_head,
        "instrument_source_tree": instrument_tree,
        "preexecution_anchor_sha256": preanchor,
        "pilot_config_sha256": config_sha,
    }


def verify_pilot(root: Path) -> dict[str, Any]:
    root = root.resolve()
    receipt_path = root / RECEIPT_REL
    plan_path = root / PLAN_REL
    receipt = read_json(receipt_path)
    plan = read_json(plan_path)

    require(isinstance(receipt, dict), "PILOT_AUDIT_RECEIPT_INVALID", "receipt must be object")
    require(isinstance(plan, dict), "PILOT_AUDIT_PLAN_INVALID", "resolved plan must be object")
    require(
        receipt.get("schema") == "symthaea.vart-world-creative-001.pilot-receipt.v1"
        and receipt.get("experiment_id") == EXPERIMENT_ID
        and receipt.get("campaign") == "pilot"
        and receipt.get("noncanonical") is True,
        "PILOT_AUDIT_RECEIPT_INVALID",
        "receipt identity",
    )
    require(
        receipt.get("scientific_efficacy_claims_authorized") is False,
        "PILOT_AUDIT_AUTHORITY_VIOLATION",
        "scientific efficacy authorization",
    )
    require(
        receipt.get("confirmatory_execution_authorized") is False,
        "PILOT_AUDIT_AUTHORITY_VIOLATION",
        "confirmatory execution authorization",
    )
    require(
        receipt.get("claim_authorized") is False,
        "PILOT_AUDIT_AUTHORITY_VIOLATION",
        "claim authorization",
    )

    verifier_result = receipt.get("verifier_result")
    require(
        isinstance(verifier_result, dict) and verifier_result.get("verdict") == "ACCEPT",
        "PILOT_AUDIT_UPSTREAM_VERIFIER_NOT_ACCEPTED",
        "pilot verifier_result",
    )

    expected_closure = receipt.get("pilot_evidence_closure_sha256")
    require(
        isinstance(expected_closure, str) and HEX64.fullmatch(expected_closure) is not None,
        "PILOT_AUDIT_RECEIPT_INVALID",
        "pilot_evidence_closure_sha256",
    )
    actual_closure = tree_closure(root, excluded_relpaths={RECEIPT_REL})
    require(
        actual_closure == expected_closure,
        "PILOT_AUDIT_CLOSURE_MISMATCH",
        f"{actual_closure} != {expected_closure}",
    )

    plan_cells_raw = plan.get("cells")
    require(
        isinstance(plan_cells_raw, list) and plan_cells_raw,
        "PILOT_AUDIT_PLAN_INVALID",
        "resolved plan cells",
    )
    cells = [_cell_from_plan_entry(entry) for entry in plan_cells_raw]

    receipt_cells = receipt.get("cells")
    require(
        isinstance(receipt_cells, list),
        "PILOT_AUDIT_RECEIPT_INVALID",
        "receipt cells",
    )
    receipt_ids = [item.get("cell_id") for item in receipt_cells if isinstance(item, dict)]
    plan_ids = [cell["cell_id"] for cell in cells]
    require(
        receipt_ids == plan_ids,
        "PILOT_AUDIT_PLAN_RECEIPT_MISMATCH",
        f"{receipt_ids} != {plan_ids}",
    )

    block_receipts = verify_paired_design(cells)
    design_digest = hashlib.sha256(canonical_json_bytes(block_receipts)).hexdigest()
    sources = verify_dual_source(receipt, plan)
    if sources["dual_source_bound"]:
        require(
            receipt.get("pilot_design_sha256") == design_digest,
            "PILOT_AUDIT_ANCHOR_MISMATCH",
            "reconstructed pilot design digest",
        )

    return {
        "verdict": "PILOT_AUDIT_PASS",
        "experiment_id": EXPERIMENT_ID,
        "pilot_receipt_sha256": sha256_file(receipt_path),
        "pilot_evidence_closure_sha256": actual_closure,
        "pilot_design_sha256": design_digest,
        "paired_block_semantics": "PASS",
        "paired_blocks": block_receipts,
        **sources,
        "confirmatory_execution_authorized": False,
        "claim_authorized": False,
        "bounded_statement": (
            "This audit establishes sealed pilot integrity, paired-design coherence, and—when "
            "present—subject/instrument source-pair closure only; it does not establish scientific "
            "efficacy or authorize confirmatory execution."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Post-pilot integrity, paired-design, and dual-source auditor for VART-WORLD-CREATIVE-001"
    )
    parser.add_argument("pilot_root", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        result = verify_pilot(args.pilot_root)
    except AuditError as exc:
        payload = {
            "verdict": "PILOT_AUDIT_REJECT",
            "reason_class": exc.code,
            "detail": exc.detail,
            "confirmatory_execution_authorized": False,
            "claim_authorized": False,
        }
        if args.json:
            print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
        else:
            print(f"REJECT {exc.code}: {exc.detail}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    else:
        print("PILOT_AUDIT_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

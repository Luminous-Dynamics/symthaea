#!/usr/bin/env python3
"""COG-META-001A-R2 production cognitive-routing inventory.

Measurement-only, dependency-free except for git. The report uses conservative
file-level lexical membership over Git-tracked production Rust source. Runtime
reachability is a separate proposition.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable

SCHEMA = "cog-meta-001a-routing-inventory-v2"
AUTHORITY_SCOPE = "measurement-only-cognitive-routing-scheduling-stopping-selection-and-cross-domain-projection-inventory"
REFERENCE_SEMANTICS = "conservative-git-tracked-production-file-level-lexical-membership"

EXCLUDED_PARTS = {
    ".git",
    "target",
    "docs",
    "examples",
    "example",
    "benches",
    "benchmarks",
    "tests",
    "testdata",
    "fixtures",
    "patches",
    "vendor",
    "third_party",
}

CATEGORY_PATTERNS: dict[str, tuple[str, ...]] = {
    "cadence_scheduling": (
        "CycleUrgency",
        "should_run(",
        "CognitiveSubsystem",
    ),
    "cognitive_regime_routing": (
        "CognitiveDepth",
        "ThalamicRouter",
        "route_from_cycle(",
    ),
    "content_competition": (
        "AttentionBid",
        "GlobalWorkspace",
        "submit_bid(",
        "select_winner(",
    ),
    "metacognitive_control": (
        "MetacognitiveRecommendation",
        "MetaCognitiveReasoner",
        "MetacognitiveReasoner",
        "assess_current_state(",
    ),
    "specialist_selection": (
        "SpecialistId",
        "SpecialistCapability",
        "QueryRequest",
        "QueryDecision",
        "specialist",
    ),
    "deliberation_stopping": (
        "RequestClarification",
        "StopThinking",
        "abstain",
        "Abstain",
        "defer",
        "Defer",
    ),
    "resource_gating": (
        "attention_budget_exceeded",
        "compute_budget",
        "resource_budget",
        "available_us",
        "thermodynamic_load",
    ),
    "planning_search_simulation": (
        "planning_horizon",
        "planner",
        "simulate",
        "simulation",
        "counterfactual",
        "search_budget",
    ),
    "memory_operation_selection": (
        "ConsolidateMemory",
        "MemoryOperation::Retrieve",
        "MemoryOperation::Consolidate",
        "consolidat",
        "retrieval",
        "recall_k",
    ),
    "external_effect_proposal": (
        "MotorCommandType",
        "ActionIR",
        "ActionCommand",
        "tool_invocation",
        "actuation",
    ),
    "mandatory_or_protective": (
        "VETO_ACTION",
        "watchdog",
        "protective",
        "interlock",
        "safety_gate",
    ),
    "transport_qos_projection": (
        "MeshUrgency",
        "From<CycleUrgency>",
        "impl From<CycleUrgency> for MeshUrgency",
    ),
}

SYMBOLS = (
    "CycleUrgency",
    "CognitiveDepth",
    "CognitiveSubsystem",
    "AttentionBid",
    "GlobalWorkspace",
    "MetacognitiveRecommendation",
    "MetaCognitiveReasoner",
    "MetacognitiveReasoner",
    "MeshUrgency",
)

REQUIRED_WITNESSES: dict[str, tuple[str, ...]] = {
    "crates/core/symthaea-cognitive-types/src/lib.rs": (
        "pub enum CycleUrgency",
        "pub fn from_arousal",
        "pub fn from_state",
        "pub fn should_run",
    ),
    "src/cognitive_loop/subsystem_trait.rs": (
        "CognitiveSubsystem",
        "fn should_run",
    ),
    "src/cognitive_loop/routing.rs": (
        "pub enum CognitiveDepth",
        "pub struct ThalamicRouter",
        "pub fn route_from_cycle",
    ),
    "src/cognitive_loop/helpers/cycle_phases_urgency.rs": (
        "CycleUrgency::from_state",
        "let raw_urgency = match self.cognitive_depth",
    ),
    "src/swarm/mesh/mod.rs": (
        "pub enum MeshUrgency",
        "impl From<CycleUrgency> for MeshUrgency",
    ),
    "crates/core/symthaea-workspace/src/lib.rs": (
        "pub struct AttentionBid",
        "GlobalWorkspace",
    ),
    "crates/core/symthaea-core/src/hdc/metacognitive_monitor.rs": (
        "pub enum MetacognitiveRecommendation",
        "assess_current_state",
    ),
    "src/consciousness/meta/meta_reasoning.rs": (
        "MetaCognitiveReasoner",
    ),
    "src/cognitive_loop/cycle_subsystems.rs": (
        "meta_reasoning_confidence",
    ),
    "src/cognitive_loop/cycle_phase_dynamics/memory_binding.rs": (
        "match self.cognitive_depth",
        "recall_k",
    ),
    "src/cognitive_loop/cycle_neuromod_phase.rs": (
        "match self.cognitive_depth",
    ),
    "src/cognitive_loop/cycle_phase_dynamics/mod.rs": (
        "let depth_budget_scale = match self.cognitive_depth",
    ),
}

DEF_RE = re.compile(
    r"(?m)^\s*(?:pub(?:\([^)]*\))?\s+)?"
    r"(?P<kind>struct|enum|trait|type)\s+"
    r"(?P<name>CycleUrgency|CognitiveDepth|CognitiveSubsystem|AttentionBid|GlobalWorkspace|"
    r"MetacognitiveRecommendation|MetaCognitiveReasoner|MetacognitiveReasoner|MeshUrgency)\b"
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=root,
        text=True,
        encoding="utf-8",
    ).strip()


def is_production_rust_rel(rel: Path) -> bool:
    if rel.suffix != ".rs":
        return False
    if any(part in EXCLUDED_PARTS for part in rel.parts):
        return False
    return "src" in rel.parts


def production_rust_files(root: Path) -> list[Path]:
    raw = subprocess.check_output(
        ["git", "ls-files", "-z", "--", "*.rs"],
        cwd=root,
    )
    rels = [
        Path(item.decode("utf-8"))
        for item in raw.split(b"\0")
        if item
    ]
    return sorted(
        (root / rel for rel in rels if is_production_rust_rel(rel)),
        key=lambda p: p.relative_to(root).as_posix(),
    )


def git_blob_sha1(data: bytes) -> str:
    header = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(header + data).hexdigest()


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest_lines(lines: Iterable[str], domain: str) -> str:
    h = hashlib.sha256()
    h.update(domain.encode("utf-8"))
    h.update(b"\0")
    for line in lines:
        encoded = line.encode("utf-8")
        h.update(len(encoded).to_bytes(8, "big"))
        h.update(encoded)
    return h.hexdigest()


def line_number(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def main() -> int:
    root = repo_root()
    subject_commit = git(root, "rev-parse", "HEAD")
    subject_tree = git(root, "rev-parse", "HEAD^{tree}")
    files = production_rust_files(root)
    if not files:
        raise SystemExit("no Git-tracked production Rust files discovered")

    text_by_path: dict[str, str] = {}
    bytes_by_path: dict[str, bytes] = {}
    for path in files:
        rel = path.relative_to(root).as_posix()
        data = path.read_bytes()
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise SystemExit(f"non-UTF-8 Rust source: {rel}: {exc}") from exc
        text_by_path[rel] = text
        bytes_by_path[rel] = data

    missing: list[str] = []
    for rel, tokens in REQUIRED_WITNESSES.items():
        text = text_by_path.get(rel)
        if text is None:
            missing.append(f"{rel}:missing-file")
            continue
        for token in tokens:
            if token not in text:
                missing.append(f"{rel}:missing-token:{token}")
    if missing:
        for item in sorted(missing):
            print(f"REQUIRED_WITNESS_FAIL {item}")
        return 2

    print(f"schema={SCHEMA}")
    print(f"authority_scope={AUTHORITY_SCOPE}")
    print(f"production_reference_semantics={REFERENCE_SEMANTICS}")
    print(f"subject_commit={subject_commit}")
    print(f"subject_tree={subject_tree}")
    print(f"production_rust_file_count={len(files)}")

    all_paths = sorted(text_by_path)
    print(
        "production_rust_path_set_sha256="
        + digest_lines(all_paths, "COG-META-001A-R2-PRODUCTION-RUST-PATH-SET-V1")
    )

    category_members: dict[str, list[str]] = {}
    for category, patterns in sorted(CATEGORY_PATTERNS.items()):
        members: list[str] = []
        for rel, text in text_by_path.items():
            if any(pattern in text for pattern in patterns):
                members.append(rel)
        members.sort()
        category_members[category] = members
        print(f"CATEGORY {category} count={len(members)}")
        print(
            f"CATEGORY_DIGEST {category} "
            + digest_lines(members, f"COG-META-001A-R2-CATEGORY:{category}:V1")
        )
        for rel in members:
            data = bytes_by_path[rel]
            match_count = sum(text_by_path[rel].count(pattern) for pattern in patterns)
            print(
                f"FILE {category} {rel} "
                f"bytes={len(data)} git_blob_sha1={git_blob_sha1(data)} "
                f"sha256={sha256_hex(data)} lexical_matches={match_count}"
            )

    definitions: dict[str, list[tuple[str, str, int]]] = {symbol: [] for symbol in SYMBOLS}
    for rel, text in text_by_path.items():
        for match in DEF_RE.finditer(text):
            name = match.group("name")
            definitions[name].append((rel, match.group("kind"), line_number(text, match.start())))

    for symbol in SYMBOLS:
        defs = sorted(definitions[symbol])
        print(f"DEFINITION_SET {symbol} count={len(defs)}")
        print(
            f"DEFINITION_DIGEST {symbol} "
            + digest_lines(
                [f"{rel}|{kind}|{line}" for rel, kind, line in defs],
                f"COG-META-001A-R2-DEFINITIONS:{symbol}:V1",
            )
        )
        for rel, kind, line in defs:
            print(f"DEFINITION {symbol} kind={kind} path={rel} line={line}")
        if len(defs) > 1:
            print(f"SEMANTIC_COLLISION_CANDIDATE {symbol} definitions={len(defs)}")

    control_witnesses = {
        "meta_confidence_to_learning": (
            "src/cognitive_loop/cycle_subsystems.rs",
            "meta_reasoning_confidence > 0.7",
        ),
        "workspace_winner_take_all": (
            "crates/core/symthaea-workspace/src/lib.rs",
            "max_by",
        ),
        "metacognitive_recommendation_match": (
            "crates/core/symthaea-core/src/hdc/consciousness_integration/pipeline.rs",
            "MetacognitiveRecommendation::ReduceLoad",
        ),
        "cognitive_depth_to_urgency": (
            "src/cognitive_loop/helpers/cycle_phases_urgency.rs",
            "let raw_urgency = match self.cognitive_depth",
        ),
        "cycle_urgency_to_mesh_qos": (
            "src/swarm/mesh/mod.rs",
            "impl From<CycleUrgency> for MeshUrgency",
        ),
        "cognitive_depth_to_memory_breadth": (
            "src/cognitive_loop/cycle_phase_dynamics/memory_binding.rs",
            "let recall_k = match self.cognitive_depth",
        ),
        "cognitive_depth_to_neuromodulation": (
            "src/cognitive_loop/cycle_neuromod_phase.rs",
            "match self.cognitive_depth",
        ),
        "cognitive_depth_to_budget_scale": (
            "src/cognitive_loop/cycle_phase_dynamics/mod.rs",
            "let depth_budget_scale = match self.cognitive_depth",
        ),
    }
    for witness, (rel, token) in sorted(control_witnesses.items()):
        text = text_by_path.get(rel, "")
        present = token in text
        print(f"CONTROL_WITNESS {witness} present={str(present).lower()} path={rel}")
        if not present:
            return 3

    nonempty_required = (
        "cadence_scheduling",
        "cognitive_regime_routing",
        "content_competition",
        "metacognitive_control",
        "resource_gating",
        "planning_search_simulation",
        "transport_qos_projection",
    )
    empty = [category for category in nonempty_required if not category_members[category]]
    if empty:
        print("EMPTY_REQUIRED_CATEGORIES " + ",".join(empty))
        return 4

    print("result=PASS_DISCOVERY")
    return 0


if __name__ == "__main__":
    sys.exit(main())

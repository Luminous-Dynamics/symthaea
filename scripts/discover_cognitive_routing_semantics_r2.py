#!/usr/bin/env python3
"""COG-META-001A-r2 cognitive routing inventory.

Measurement-only discovery against one immutable Git source commit.

Layer A is conservative production-file lexical membership.
Layer B freezes exact source-level producer/consumer witnesses.
Layer C freezes selected explicit no-production-surface findings under this
exact search profile.

Static source witnesses do not prove dynamic runtime reachability.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Iterable

SCHEMA = "cog-meta-001a-r2-routing-discovery-v1"
AUTHORITY_SCOPE = (
    "measurement-only-cognitive-routing-scheduling-stopping-selection-and-source-witness-boundary"
)
REFERENCE_SEMANTICS = "git-object-backed-conservative-production-file-level-lexical-membership"
SOURCE_COMMIT = "adb69f11fa8068b019cc5bb598d0c7726a197fc9"
SOURCE_TREE = "35a6c5fdba319556af9bb487838734f67c8ac0d6"

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
    "cadence_scheduling": ("CycleUrgency", "should_run(", "CognitiveSubsystem"),
    "content_competition": ("AttentionBid", "GlobalWorkspace", "submit_bid(", "select_winner("),
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
}

SYMBOLS = (
    "CycleUrgency",
    "CognitiveSubsystem",
    "AttentionBid",
    "GlobalWorkspace",
    "MetacognitiveRecommendation",
    "MetaCognitiveReasoner",
    "MetacognitiveReasoner",
)

REQUIRED_WITNESSES: dict[str, tuple[str, ...]] = {
    "crates/core/symthaea-cognitive-types/src/lib.rs": ("CycleUrgency", "should_run("),
    "src/cognitive_loop/subsystem_trait.rs": ("CognitiveSubsystem", "fn should_run"),
    "crates/core/symthaea-workspace/src/lib.rs": ("pub struct AttentionBid", "GlobalWorkspace"),
    "crates/core/symthaea-core/src/hdc/metacognitive_monitor.rs": (
        "pub enum MetacognitiveRecommendation",
        "assess_current_state",
    ),
    "src/consciousness/meta/meta_reasoning.rs": ("MetaCognitiveReasoner", "compute_meta_confidence"),
    "src/cognitive_loop/cycle_subsystems.rs": ("meta_reasoning_confidence",),
}

DEF_RE = re.compile(
    r"(?m)^\s*(?:pub(?:\([^)]*\))?\s+)?"
    r"(?P<kind>struct|enum|trait|type)\s+"
    r"(?P<name>CycleUrgency|CognitiveSubsystem|AttentionBid|GlobalWorkspace|"
    r"MetacognitiveRecommendation|MetaCognitiveReasoner|MetacognitiveReasoner)\b"
)

@dataclass(frozen=True)
class GitEntry:
    mode: str
    obj_type: str
    sha: str
    path: str

@dataclass(frozen=True)
class SourceWitness:
    name: str
    path: str
    tokens: tuple[str, ...]

SOURCE_WITNESSES = (
    SourceWitness(
        "cycle_urgency_to_regime_execution",
        "src/cognitive_loop/helpers/parallel.rs",
        ("urgency.should_run(", "regime.process_input("),
    ),
    SourceWitness(
        "manager_should_run_to_subsystem_dispatch",
        "src/cognitive_loop/cycle_phase_dynamics/mod.rs",
        ("self.learning_manager.should_run(", 'run_subsystem!(self.learning_manager'),
    ),
    SourceWitness(
        "workspace_bid_to_winner_to_focus",
        "crates/core/symthaea-workspace/src/lib.rs",
        ("submit_bid(", ".max_by(", "current_focus = Some(winner)"),
    ),
    SourceWitness(
        "metacognitive_assessment_to_recommendation_consumer",
        "crates/core/symthaea-core/src/hdc/consciousness_integration/pipeline.rs",
        ("monitor.assess_current_state()", "MetacognitiveRecommendation::ReduceLoad"),
    ),
    SourceWitness(
        "meta_reasoner_result_to_primitive_consumer",
        "src/consciousness/unified_intelligence.rs",
        ("meta_reasoner.meta_reason(", "meta_result.optimization_result.primitive"),
    ),
    SourceWitness(
        "meta_confidence_to_learning_rate",
        "src/cognitive_loop/cycle_subsystems.rs",
        ("result.meta_confidence", "meta_reasoning_confidence > 0.7", 'adjust_lr("meta_reasoning"'),
    ),
    SourceWitness(
        "prefrontal_priority_to_reported_confidence",
        "src/brain/prefrontal.rs",
        ("max_by(|a, b| a.priority.total_cmp(&b.priority))", ".map(|a| a.priority)"),
    ),
)

ABSENCE_PROFILES: dict[str, tuple[str, ...]] = {
    "typed_specialist_query_controller": ("SpecialistId", "SpecialistCapability", "QueryDecision"),
    "typed_stop_or_clarification_operation": ("RequestClarification", "StopThinking"),
    "arc3_meta_action_runtime_surface": ("MetaAction", "meta_action"),
}

DEPENDENCY_TOKENS = (
    "symthaea-types",
    "symthaea-cognitive-types",
    "symthaea-workspace",
    "symthaea-core",
)


def run_git_text(*args: str) -> str:
    proc = subprocess.run(
        ["git", *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return proc.stdout


def run_git_bytes(*args: str) -> bytes:
    proc = subprocess.run(
        ["git", *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return proc.stdout


def git_entries(commit: str) -> list[GitEntry]:
    raw = run_git_bytes("ls-tree", "-r", "-z", commit)
    entries: list[GitEntry] = []
    for record in raw.split(b"\0"):
        if not record:
            continue
        meta, raw_path = record.split(b"\t", 1)
        mode, obj_type, sha = meta.decode("ascii").split()
        path = raw_path.decode("utf-8")
        entries.append(GitEntry(mode=mode, obj_type=obj_type, sha=sha, path=path))
    return sorted(entries, key=lambda e: e.path)


def read_blob(sha: str) -> bytes:
    return run_git_bytes("cat-file", "blob", sha)


def is_production_rust(entry: GitEntry) -> bool:
    path = PurePosixPath(entry.path)
    if path.suffix != ".rs":
        return False
    if "src" not in path.parts:
        return False
    if any(part in EXCLUDED_PARTS for part in path.parts):
        return False
    return True


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
    observed_tree = run_git_text("rev-parse", f"{SOURCE_COMMIT}^{{tree}}").strip()
    if observed_tree != SOURCE_TREE:
        print(f"SOURCE_TREE_MISMATCH expected={SOURCE_TREE} observed={observed_tree}")
        return 2

    entries = git_entries(SOURCE_COMMIT)
    rust_entries = [entry for entry in entries if is_production_rust(entry)]
    if not rust_entries:
        raise SystemExit("no production Rust files discovered")

    nonregular = [
        entry for entry in rust_entries
        if entry.obj_type != "blob" or entry.mode not in {"100644", "100755"}
    ]
    if nonregular:
        for entry in nonregular:
            print(
                f"NONREGULAR_PRODUCTION_RUST mode={entry.mode} type={entry.obj_type} "
                f"sha={entry.sha} path={entry.path}"
            )
        return 3

    text_by_path: dict[str, str] = {}
    bytes_by_path: dict[str, bytes] = {}
    entry_by_path: dict[str, GitEntry] = {}
    for entry in rust_entries:
        data = read_blob(entry.sha)
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise SystemExit(f"non-UTF-8 Rust source: {entry.path}: {exc}") from exc
        text_by_path[entry.path] = text
        bytes_by_path[entry.path] = data
        entry_by_path[entry.path] = entry

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
        return 4

    print(f"schema={SCHEMA}")
    print(f"authority_scope={AUTHORITY_SCOPE}")
    print(f"production_reference_semantics={REFERENCE_SEMANTICS}")
    print(f"source_commit={SOURCE_COMMIT}")
    print(f"source_tree={SOURCE_TREE}")
    print(f"production_rust_file_count={len(rust_entries)}")

    all_paths = sorted(text_by_path)
    print(
        "production_rust_path_set_sha256="
        + digest_lines(all_paths, "COG-META-001A-R2-PRODUCTION-RUST-PATH-SET-V1")
    )

    category_members: dict[str, list[str]] = {}
    for category, patterns in sorted(CATEGORY_PATTERNS.items()):
        members = sorted(
            rel for rel, text in text_by_path.items()
            if any(pattern in text for pattern in patterns)
        )
        category_members[category] = members
        print(f"CATEGORY {category} count={len(members)}")
        print(
            f"CATEGORY_DIGEST {category} "
            + digest_lines(members, f"COG-META-001A-R2-CATEGORY:{category}:V1")
        )
        for rel in members:
            data = bytes_by_path[rel]
            entry = entry_by_path[rel]
            match_count = sum(text_by_path[rel].count(pattern) for pattern in patterns)
            print(
                f"FILE {category} {rel} mode={entry.mode} "
                f"bytes={len(data)} git_blob_sha1={entry.sha} "
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

    source_witness_failures = 0
    for witness in SOURCE_WITNESSES:
        text = text_by_path.get(witness.path, "")
        missing_tokens = [token for token in witness.tokens if token not in text]
        if missing_tokens:
            source_witness_failures += 1
            print(
                f"SOURCE_WITNESS {witness.name} status=MISSING path={witness.path} "
                f"missing_count={len(missing_tokens)}"
            )
            for token in missing_tokens:
                print(f"SOURCE_WITNESS_MISSING {witness.name} token={token!r}")
        else:
            print(
                f"SOURCE_WITNESS {witness.name} status=PRESENT path={witness.path} "
                f"token_count={len(witness.tokens)}"
            )

    for name, tokens in sorted(ABSENCE_PROFILES.items()):
        matches: list[str] = []
        for rel, text in text_by_path.items():
            for token in tokens:
                if token in text:
                    matches.append(f"{rel}|{token}")
        matches.sort()
        print(f"ABSENCE_PROFILE {name} match_count={len(matches)}")
        print(
            f"ABSENCE_DIGEST {name} "
            + digest_lines(matches, f"COG-META-001A-R2-ABSENCE:{name}:V1")
        )
        for item in matches:
            rel, token = item.split("|", 1)
            print(f"ABSENCE_MATCH {name} path={rel} token={token!r}")
        if not matches:
            print(f"NO_PRODUCTION_SURFACE_FOUND {name}")

    cargo_entries = [
        entry for entry in entries
        if PurePosixPath(entry.path).name == "Cargo.toml"
        and entry.obj_type == "blob"
        and entry.mode in {"100644", "100755"}
    ]
    dependency_edges: list[str] = []
    for entry in cargo_entries:
        data = read_blob(entry.sha)
        text = data.decode("utf-8")
        for token in DEPENDENCY_TOKENS:
            if token in text:
                dependency_edges.append(f"{entry.path}|{token}")
    dependency_edges.sort()
    print(f"DEPENDENCY_EDGE_SET count={len(dependency_edges)}")
    print(
        "DEPENDENCY_EDGE_DIGEST "
        + digest_lines(dependency_edges, "COG-META-001A-R2-DEPENDENCY-EDGES-V1")
    )
    for edge in dependency_edges:
        path, token = edge.split("|", 1)
        print(f"DEPENDENCY_EDGE path={path} token={token}")

    required_nonempty = (
        "cadence_scheduling",
        "content_competition",
        "metacognitive_control",
        "resource_gating",
        "planning_search_simulation",
    )
    empty = [category for category in required_nonempty if not category_members[category]]
    if empty:
        print("EMPTY_REQUIRED_CATEGORIES " + ",".join(empty))
        return 5

    print("result=PASS_DISCOVERY")
    if source_witness_failures:
        print(f"source_witness_result=FAIL missing_groups={source_witness_failures}")
        return 6
    print(f"source_witness_result=PASS groups={len(SOURCE_WITNESSES)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

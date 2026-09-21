#!/usr/bin/env python3
"""Deterministic lexical census of Symthaea Phi/integration authority flow.

PHI-SEM-001A is intentionally measurement-only. This tool inventories tracked
source text under a frozen Git commit and reports lexical evidence. It does not
prove runtime reachability, estimator correctness, consciousness, or authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
PROFILE_ID = "phi-sem-001a-lexical-v1"
REPORT_VERSION = 1
AUDIT_SCRIPT_PATH = "scripts/audit_phi_authority_flow.py"

CATEGORY_PATTERNS: dict[str, tuple[str, ...]] = {
    "Estimator": (
        r"\bcompute_phi(?:_entropy|_fast)?\b",
        r"\bIntegratedInformation\b",
        r"\bintegration_index\b",
        r"\bspectral_mip_phi\b",
        r"\bhierarchical_mip_phi\b",
        r"\bstructural_(?:micro|meso|macro)_phi\b",
        r"\balgebraic_connectivity\b",
        r"\bminimum_information_partition\b",
    ),
    "TransformOrComposite": (
        r"\bstate\.phi\s*=",
        r"\bconsciousness_level\s*=",
        r"\bphi_(?:score|contribution|modulation|boost|factor)\b",
        r"\bcombined_phi\b",
        r"\bprofile_composite\b",
    ),
    "ScientificLabel": (
        r"\bIIT\b",
        r"\bIntegrated Information Theory\b",
        r"\bconscious(?:ness)?\b",
        r"\bunconscious\b",
        r"\banesthesia\b",
        r"\bsleep\b",
        r"\bawake\b",
    ),
    "SyntheticValidation": (
        r"\bPhiValidationFramework\b",
        r"\bSyntheticStateGenerator\b",
        r"\bexpected_phi_range\b",
        r"\bconsciousness_level\(\)",
    ),
    "OptimizationObjective": (
        r"\bPhiGuidedOptimizer\b",
        r"\bPhiArchitectureSearch\b",
        r"\bphi_gradient\b",
        r"\bbest_phi\b",
        r"\bphi_delta\b",
        r"\boptimi[sz](?:e|ation).*phi\b",
        r"\bfitness\b",
    ),
    "FeedbackController": (
        r"\bPhiFeedback(?:Controller|Config)?\b",
        r"\bphi_target\b",
        r"\bphi_error\b",
        r"\bphi_ema\b",
    ),
    "MetacognitiveSignal": (
        r"\bmetacognitive_(?:confidence|coherence)\b",
        r"\bpredicted_phi\b",
        r"\bphi_trend\b",
    ),
    "ConfidenceOrStatusProducer": (
        r"\bConfidenceLevel\b",
        r"\bconfidence_level\b",
        r"\brecommendation\b",
        r"\bis_integrated\b",
    ),
    "AttentionOrWorkspaceControl": (
        r"\bPhiAwareScoring\b",
        r"\bbroadcast_threshold\b",
        r"\bworkspace_capacity\b",
        r"\battention_gain\b",
        r"\battention.*phi\b",
    ),
    "ExpressionControl": (
        r"\bexpression\b.*\bphi\b",
        r"\bphi\b.*\bexpression\b",
        r"\bassertion\b.*\bphi\b",
        r"\bphi\b.*\bassertion\b",
    ),
    "ActionOrReadinessGate": (
        r"\bConsciousnessThresholds\b",
        r"\brequired_phi\b",
        r"\ballows_action\b",
        r"\bPendingConfirmation\b",
        r"\bexecution\b.*\bphi\b",
        r"\bphi\b.*\bexecution\b",
    ),
    "GovernanceBridge": (
        r"\bgovernance\b.*\bphi\b",
        r"\bphi\b.*\bgovernance\b",
        r"\breadiness\b.*\bphi\b",
        r"\bphi\b.*\breadiness\b",
    ),
    "ExternalExport": (
        r"\btelemetry\b.*\bphi\b",
        r"\bphi\b.*\btelemetry\b",
        r"\bgrpc\b.*\bphi\b",
        r"\bphi\b.*\bgrpc\b",
        r"\bexport\b.*\bphi\b",
        r"\bphi\b.*\bexport\b",
    ),
    "SerializationOrCheckpoint": (
        r"\bpub\s+phi\s*:\s*f(?:32|64)\b",
        r"\bphi\s*:\s*(?:Option<)?f(?:32|64)",
        r"\bPipelineCheckpoint\b",
        r"\bserde\b.*\bphi\b",
    ),
    "CompatibilityAlias": (
        r"\blegacy\b.*\bphi\b",
        r"\bphi\b.*\blegacy\b",
        r"\balias\b.*\bphi\b",
        r"\bphi\b.*\balias\b",
    ),
    "DocumentationClaim": (
        r"\bPhi\b.*\bconscious",
        r"\bconscious.*\bPhi\b",
        r"\bΦ\b",
    ),
}

COMPILED_PATTERNS = {
    category: tuple(re.compile(pattern, re.IGNORECASE) for pattern in patterns)
    for category, patterns in CATEGORY_PATTERNS.items()
}

MANDATORY_WITNESS_GROUPS: dict[str, tuple[str, ...]] = {
    "hdc_integrated_information": (
        "crates/core/symthaea-core/src/hdc/integrated_information.rs",
    ),
    "consciousness_pipeline": (
        "crates/core/symthaea-core/src/hdc/consciousness_integration/pipeline.rs",
    ),
    "legacy_core_phi_search": (
        "crates/core/symthaea-core/src/hdc/phi_guided_search.rs",
    ),
    "domain_phi_search": (
        "crates/domains/symthaea-phi-search/",
    ),
    "phi_feedback": (
        "crates/core/symthaea-core/src/hdc/phi_feedback.rs",
    ),
    "phi_optimization": (
        "crates/core/symthaea-core/src/hdc/consciousness_phi_optimization.rs",
    ),
    "metacognitive_phi": (
        "crates/core/symthaea-core/src/hdc/consciousness_metacognitive.rs",
    ),
    "phi_attention": (
        "src/consciousness/measurement/phi_attention.rs",
    ),
    "synthetic_phi_validation": (
        "src/consciousness/measurement/phi_validation.rs",
        "src/consciousness/synthetic_states.rs",
    ),
    "cognitive_consciousness_engine": (
        "src/cognitive_loop/consciousness_engine/",
    ),
    "late_consciousness_integration": (
        "src/cognitive_loop/cycle_late_consciousness/integration.rs",
    ),
    "nixos_action_gate": (
        "src/action/nixos_patterns.rs",
    ),
    "nixward_phi_surfaces": (
        "crates/core/nixward/src/traits.rs",
        "crates/core/nixward/src/action/executor.rs",
    ),
    "workspace_phi_oracle": (
        "crates/domains/symthaea-phi-oracle/",
    ),
    "duplicate_root_phi_oracle": (
        "crates/symthaea-phi-oracle/",
    ),
    "telemetry_phi_surface": (
        "crates/domains/symthaea-telemetry-sink/",
        "crates/bridges/symthaea-telemetry-grpc/",
    ),
}

PHI_LEXEME = re.compile(
    r"(?:\bphi\b|\bphi[_A-Za-z0-9]*\b|[_A-Za-z0-9]*_phi\b|Φ|"
    r"integration_index|spectral_mip|hierarchical_mip|structural_.*phi|"
    r"IntegratedInformation|PhiAware|ConsciousnessThresholds)",
    re.IGNORECASE,
)

TEXT_EXTENSIONS = {
    ".rs", ".toml", ".md", ".py", ".json", ".jsonl", ".yaml", ".yml",
    ".nix", ".sh", ".txt", ".csv", ".ron", ".proto",
}

PROFILE_EXCLUDED_PATHS = {
    AUDIT_SCRIPT_PATH,
    "docs/research/PHI_SEM_001A_INVENTORY_V1.md",
}
PROFILE_EXCLUDED_PREFIXES = (
    "docs/research/evidence/phi_sem_001a_",
)

ORACLE_PREFIXES = (
    "crates/domains/symthaea-phi-oracle/",
    "crates/symthaea-phi-oracle/",
)


def git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def tracked_entries() -> list[tuple[str, str]]:
    raw = subprocess.run(
        ["git", "ls-files", "-s", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    entries: list[tuple[str, str]] = []
    for record in raw.split(b"\0"):
        if not record:
            continue
        meta, path_bytes = record.split(b"\t", 1)
        _mode, blob, stage = meta.decode("ascii").split()
        if stage != "0":
            raise RuntimeError(f"unmerged index entry for {path_bytes!r}")
        entries.append((path_bytes.decode("utf-8"), blob))
    return sorted(entries)


def is_profile_excluded(path: str) -> bool:
    return path in PROFILE_EXCLUDED_PATHS or any(
        path.startswith(prefix) for prefix in PROFILE_EXCLUDED_PREFIXES
    )


def is_candidate_text(path: str) -> bool:
    if is_profile_excluded(path):
        return False
    pure = PurePosixPath(path)
    return pure.suffix.lower() in TEXT_EXTENSIONS or pure.name in {
        "Cargo.lock", "Makefile", "Justfile", "justfile"
    }


def read_text(path: str) -> tuple[str, bytes] | None:
    data = (ROOT / path).read_bytes()
    if b"\0" in data:
        return None
    try:
        return data.decode("utf-8"), data
    except UnicodeDecodeError:
        return None


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def matching_categories(line: str, path: str) -> list[str]:
    categories: set[str] = set()
    for category, patterns in COMPILED_PATTERNS.items():
        if any(pattern.search(line) for pattern in patterns):
            categories.add(category)

    lower_path = path.lower()
    if PHI_LEXEME.search(line):
        if lower_path.endswith((".md", ".txt")) or line.lstrip().startswith(("//!", "///", "#")):
            categories.add("DocumentationClaim")
        if "telemetry" in lower_path or "grpc" in lower_path:
            categories.add("ExternalExport")
        if "metacogn" in lower_path:
            categories.add("MetacognitiveSignal")
        if "phi-search" in lower_path or "phi_guided_search" in lower_path:
            categories.add("OptimizationObjective")
        if "phi_feedback" in lower_path:
            categories.add("FeedbackController")
        if "phi_validation" in lower_path or "synthetic_states" in lower_path:
            categories.add("SyntheticValidation")
        if "nixward" in lower_path or "nixos_patterns" in lower_path:
            categories.add("ActionOrReadinessGate")
        if "integrated_information" in lower_path or "consciousness_metrics" in lower_path:
            categories.add("Estimator")
    return sorted(categories)


def build_inventory(entries: list[tuple[str, str]]) -> list[dict]:
    inventory: list[dict] = []
    for path, blob in entries:
        if not is_candidate_text(path):
            continue
        loaded = read_text(path)
        if loaded is None:
            continue
        text, data = loaded
        matches: list[dict] = []
        path_categories: set[str] = set()
        for lineno, line in enumerate(text.splitlines(), 1):
            if not PHI_LEXEME.search(line):
                continue
            categories = matching_categories(line, path)
            if not categories:
                categories = ["UnclassifiedPhiLexeme"]
            path_categories.update(categories)
            matches.append(
                {
                    "line": lineno,
                    "categories": categories,
                    "text": line.strip()[:240],
                }
            )
        if matches:
            inventory.append(
                {
                    "path": path,
                    "git_blob": blob,
                    "sha256": sha256_hex(data),
                    "bytes": len(data),
                    "categories": sorted(path_categories),
                    "matches": matches,
                }
            )
    return inventory


def selector_matches(paths: list[str], selector: str) -> list[str]:
    if selector.endswith("/"):
        return sorted(path for path in paths if path.startswith(selector))
    return [selector] if selector in paths else []


def witness_results(paths: list[str]) -> tuple[dict[str, dict[str, list[str]]], list[dict]]:
    results: dict[str, dict[str, list[str]]] = {}
    missing: list[dict] = []
    for group, selectors in sorted(MANDATORY_WITNESS_GROUPS.items()):
        group_results: dict[str, list[str]] = {}
        for selector in selectors:
            matches = selector_matches(paths, selector)
            group_results[selector] = matches
            if not matches:
                missing.append({"group": group, "selector": selector})
        results[group] = group_results
    return results, missing


def subtree_fingerprint(entries_by_path: dict[str, str], prefix: str) -> dict[str, object]:
    members: list[tuple[str, str, str]] = []
    for path in sorted(entries_by_path):
        if not path.startswith(prefix):
            continue
        rel = path[len(prefix):]
        loaded = read_text(path)
        if loaded is None:
            data = (ROOT / path).read_bytes()
        else:
            _text, data = loaded
        members.append((rel, entries_by_path[path], sha256_hex(data)))

    digest = hashlib.sha256()
    for rel, blob, sha256 in members:
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(blob.encode("ascii"))
        digest.update(b"\0")
        digest.update(sha256.encode("ascii"))
        digest.update(b"\0")

    return {
        "prefix": prefix,
        "tracked_files": len(members),
        "fingerprint_sha256": digest.hexdigest(),
        "members": [
            {"relative_path": rel, "git_blob": blob, "sha256": sha}
            for rel, blob, sha in members
        ],
    }


def oracle_dependency_edges(entries: list[tuple[str, str]]) -> list[dict]:
    edges: list[dict] = []
    for path, blob in entries:
        if PurePosixPath(path).name != "Cargo.toml":
            continue
        loaded = read_text(path)
        if loaded is None:
            continue
        text, data = loaded
        for lineno, line in enumerate(text.splitlines(), 1):
            if "symthaea-phi-oracle" not in line:
                continue
            edges.append(
                {
                    "manifest": path,
                    "manifest_git_blob": blob,
                    "manifest_sha256": sha256_hex(data),
                    "line": lineno,
                    "text": line.strip(),
                    "references_workspace_domain_copy": (
                        "domains/symthaea-phi-oracle" in line
                        or "../symthaea-phi-oracle" in line
                    ),
                    "references_root_duplicate": (
                        "crates/symthaea-phi-oracle" in line
                        or "../../symthaea-phi-oracle" in line
                    ),
                }
            )
    return edges


def compare_oracle_copies(
    entries: list[tuple[str, str]], entries_by_path: dict[str, str]
) -> dict[str, object]:
    canonical = subtree_fingerprint(entries_by_path, ORACLE_PREFIXES[0])
    duplicate = subtree_fingerprint(entries_by_path, ORACLE_PREFIXES[1])

    canonical_members = canonical["members"]
    duplicate_members = duplicate["members"]
    assert isinstance(canonical_members, list)
    assert isinstance(duplicate_members, list)

    canonical_map = {
        item["relative_path"]: item["sha256"]
        for item in canonical_members
        if isinstance(item, dict)
    }
    duplicate_map = {
        item["relative_path"]: item["sha256"]
        for item in duplicate_members
        if isinstance(item, dict)
    }
    all_rel = sorted(set(canonical_map) | set(duplicate_map))
    differences = [
        {
            "relative_path": rel,
            "workspace_sha256": canonical_map.get(rel),
            "root_duplicate_sha256": duplicate_map.get(rel),
        }
        for rel in all_rel
        if canonical_map.get(rel) != duplicate_map.get(rel)
    ]
    return {
        "workspace_active_candidate": canonical,
        "root_duplicate_candidate": duplicate,
        "byte_identity_equal": not differences and bool(canonical_map),
        "differences": differences,
        "cargo_dependency_edges": oracle_dependency_edges(entries),
    }


def profile_sha256() -> str:
    payload = json.dumps(
        {
            "profile_id": PROFILE_ID,
            "category_patterns": CATEGORY_PATTERNS,
            "mandatory_witness_groups": MANDATORY_WITNESS_GROUPS,
            "text_extensions": sorted(TEXT_EXTENSIONS),
            "excluded_paths": sorted(PROFILE_EXCLUDED_PATHS),
            "excluded_prefixes": list(PROFILE_EXCLUDED_PREFIXES),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256_hex(payload)


def audit_artifact(entries_by_path: dict[str, str]) -> dict[str, object]:
    data = (ROOT / AUDIT_SCRIPT_PATH).read_bytes()
    return {
        "path": AUDIT_SCRIPT_PATH,
        "git_blob": entries_by_path.get(AUDIT_SCRIPT_PATH),
        "sha256": sha256_hex(data),
        "bytes": len(data),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        help="write canonical JSON to this path; stdout when omitted",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="pretty-print JSON (canonical key ordering remains stable)",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="allow tracked worktree modifications (not suitable for qualification)",
    )
    args = parser.parse_args()

    if not args.allow_dirty:
        dirty = git("status", "--porcelain", "--untracked-files=no")
        if dirty:
            print(
                "refusing to audit a worktree with tracked modifications; "
                "commit/stash changes or pass --allow-dirty",
                file=sys.stderr,
            )
            return 2

    head = git("rev-parse", "HEAD")
    tree = git("rev-parse", "HEAD^{tree}")
    entries = tracked_entries()
    entries_by_path = dict(entries)
    all_paths = [path for path, _blob in entries]

    inventory = build_inventory(entries)
    witnesses, missing_selectors = witness_results(all_paths)

    report = {
        "report_version": REPORT_VERSION,
        "profile_id": PROFILE_ID,
        "profile_sha256": profile_sha256(),
        "source": {
            "commit": head,
            "tree": tree,
            "tracked_file_count": len(entries),
        },
        "audit_artifact": audit_artifact(entries_by_path),
        "claim_ceiling": {
            "establishes": [
                "tracked lexical Phi/integration census under the declared profile",
                "exact path/blob/SHA-256 identity for matched text files",
                "presence of every mandatory witness selector",
                "byte-level comparison of the two phi-oracle source trees",
                "lexical Cargo dependency edges naming symthaea-phi-oracle",
            ],
            "does_not_establish": [
                "runtime reachability or complete dynamic call graph",
                "estimator correctness",
                "IIT validity",
                "consciousness",
                "epistemic confidence",
                "execution or governance authority",
            ],
        },
        "mandatory_witnesses": witnesses,
        "missing_mandatory_witness_selectors": missing_selectors,
        "phi_oracle_duplicate_check": compare_oracle_copies(entries, entries_by_path),
        "inventory": inventory,
        "summary": {
            "matched_paths": len(inventory),
            "matched_lines": sum(len(item["matches"]) for item in inventory),
            "categories": sorted(
                {
                    category
                    for item in inventory
                    for category in item["categories"]
                }
            ),
        },
    }

    separators = None if args.pretty else (",", ":")
    encoded = json.dumps(
        report,
        indent=2 if args.pretty else None,
        sort_keys=True,
        separators=separators,
        ensure_ascii=False,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    else:
        print(encoded)

    if missing_selectors:
        rendered = ", ".join(
            f"{item['group']}:{item['selector']}" for item in missing_selectors
        )
        print("missing mandatory witness selectors: " + rendered, file=sys.stderr)
        return 3

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

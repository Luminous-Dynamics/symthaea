#!/usr/bin/env python3
"""EPI-AUTH-FLOW-001R5 deterministic authority-flow discovery.

Measurement-only repository audit. This script discovers the current Rust source
surface through which legacy/high epistemic authority can be produced, transported,
stored, collapsed, or used to condition Broca/language generation.

PASS_DISCOVERY means the discovery program ran against a subject containing the
reviewed representative witnesses. It is NOT a frozen inventory and establishes no
semantic correctness or epistemic authority.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_PARTS = {".git", "target", "vendor", "node_modules", ".direnv", "result"}

SURFACES: dict[str, tuple[str, ...]] = {
    "typed_top_authority": (
        "ETier::E4",
        "NTier::N3",
        "MTier::M3",
    ),
    "typed_cube_definition": (
        "pub enum ETier",
        "pub enum NTier",
        "pub enum MTier",
        "pub struct EpistemicCube",
    ),
    "legacy_empirical_authority": (
        "E3CryptographicallyProven",
        "E4PubliclyReproducible",
        "CryptographicallyVerifiable",
        "PubliclyReproducible",
        "E3Cryptographic",
        "E4PublicRepro",
    ),
    "legacy_high_axis_authority": (
        "N3Axiomatic",
        "M3Foundational",
        "NormativeLevel::Foundational",
        "MaterialityLevel::Permanent",
    ),
    "numeric_cube_transport": (
        "cube_e_tier",
        "cube_n_tier",
        "cube_m_tier",
        "last_cube_e_tier",
        "last_cube_n_tier",
        "last_cube_m_tier",
        "set_epistemic_cube",
        "inject_epistemic_cube",
        "EPISTEMIC_CUBE_BASE",
        "e_tier: u8",
        "n_tier: u8",
        "m_tier: u8",
    ),
    "scalar_cube_collapse": (
        "(e as f32 / 4.0) * 0.40",
        "(n as f32 / 3.0) * 0.35",
        "(m as f32 / 3.0) * 0.25",
        "(e_tier as f32 / 4.0) * 0.40",
        "(n_tier as f32 / 3.0) * 0.35",
        "(m_tier as f32 / 3.0) * 0.25",
    ),
    "broca_cube_sink": (
        "e_tier_e4",
        "n_tier_n3",
        "m_tier_m3",
        "Set the full 4D epistemic cube as channel data",
        "set_epistemic_cube(",
    ),
}

# Representative witnesses prove that discovery reaches each important layer.
# These are presence witnesses, not claims that the semantics are valid.
REQUIRED_WITNESSES: dict[str, dict[str, tuple[str, ...]]] = {
    "cycle_runtime_producer": {
        "src/cognitive_loop/cycle_subsystems.rs": (
            "epistemic_gate_confidence > 0.9",
            "4u8 // E4: reproducible",
            "epistemic_gate_confidence > 0.7",
            "3 // E3: proven",
            "kg > 0.8 && sources >= 3",
            "3u8 // N3: axiomatic",
            "kg > 0.7",
            "3u8 // M3: foundational",
        ),
    },
    "runtime_carryover": {
        "src/cognitive_loop/types/carryover.rs": (
            "last_cube_e_tier",
            "last_cube_n_tier",
            "last_cube_m_tier",
        ),
        "src/cognitive_loop/cycle_phase_dynamics/training.rs": (
            "cube_e_tier: self.carryover.quality.last_cube_e_tier",
            "cube_n_tier: self.carryover.quality.last_cube_n_tier",
            "cube_m_tier: self.carryover.quality.last_cube_m_tier",
        ),
    },
    "broca_conditioning": {
        "src/cognitive_loop/broca_bridge.rs": (
            "channels.set_epistemic_cube",
            "signals.cube_e_tier",
            "signals.cube_n_tier",
            "signals.cube_m_tier",
        ),
        "crates/domains/symthaea-broca/src/encoder.rs": (
            "e_tier_e4",
            "n_tier_n3",
            "m_tier_m3",
            "pub fn set_epistemic_cube",
        ),
    },
    "typed_cube_to_broca": {
        "src/language/ssm_backend.rs": (
            "ETier::E4 => 4",
            "NTier::N3 => 3",
            "MTier::M3 => 3",
            "ch.set_epistemic_cube",
        ),
        "src/mind/structured_thought.rs": (
            "pub enum ETier",
            "pub enum NTier",
            "pub enum MTier",
            "publicly reproducible proof",
            "axiomatic truth",
            "M3 (foundational)",
        ),
    },
    "mycelix_numeric_producer": {
        "src/consciousness/mycelix_bridge.rs": (
            "pub e_tier: u8",
            "pub n_tier: u8",
            "pub m_tier: u8",
            "EmpiricalLevel::E3Cryptographic",
            "NormativeLevel::N3Axiomatic",
        ),
    },
    "hdc_statistical_producer": {
        "crates/core/symthaea-core/src/hdc/statistical_retrieval.rs": (
            "EmpiricalTier::from_z_score",
            "E3CryptographicallyProven",
            "E4PubliclyReproducible",
        ),
    },
    "physics_product_producer": {
        "crates/domains/symthaea-physics-catalog/src/discovery.rs": (
            "pub empirical: u8",
            "pub normative: u8",
            "pub materiality: u8",
        ),
    },
}

MIN_COUNTS = {
    "typed_top_authority": 20,
    "typed_cube_definition": 1,
    "legacy_empirical_authority": 3,
    "legacy_high_axis_authority": 3,
    "numeric_cube_transport": 6,
    "scalar_cube_collapse": 1,
    "broca_cube_sink": 1,
}


def rust_files() -> Iterable[Path]:
    for path in ROOT.rglob("*.rs"):
        rel = path.relative_to(ROOT)
        if any(part in EXCLUDED_PARTS for part in rel.parts):
            continue
        yield path


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def set_digest(paths: list[str]) -> str:
    payload = "\n".join(paths).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def discover() -> dict[str, list[str]]:
    found = {name: [] for name in SURFACES}
    for path in rust_files():
        rel = path.relative_to(ROOT).as_posix()
        text = read_text(path)
        for name, needles in SURFACES.items():
            if any(needle in text for needle in needles):
                found[name].append(rel)
    for paths in found.values():
        paths.sort()
    return found


def check_witnesses() -> dict[str, list[str]]:
    missing: dict[str, list[str]] = {}
    for group, files in REQUIRED_WITNESSES.items():
        absent: list[str] = []
        for rel, needles in files.items():
            path = ROOT / rel
            if not path.exists():
                absent.append(f"{rel}:<missing-file>")
                continue
            text = read_text(path)
            for needle in needles:
                if needle not in text:
                    absent.append(f"{rel}:{needle}")
        if absent:
            missing[group] = absent
    return missing


def main() -> int:
    surfaces = discover()
    missing_witnesses = check_witnesses()
    below_minimum = {
        name: {"actual": len(surfaces[name]), "minimum": minimum}
        for name, minimum in MIN_COUNTS.items()
        if len(surfaces[name]) < minimum
    }

    digests = {name: set_digest(paths) for name, paths in surfaces.items()}

    print("schema=epi-auth-flow-001-r5-discovery-v1")
    print("authority_scope=measurement-only-producer-transport-broca-discovery")
    for name in sorted(surfaces):
        paths = surfaces[name]
        print(f"SURFACE {name} count={len(paths)} digest={digests[name]}")
        for rel in paths:
            print(f"FILE {name} {rel}")

    summary = {
        "schema": "epi-auth-flow-001-r5-discovery-v1",
        "authority_scope": "measurement-only-producer-transport-broca-discovery",
        "surface_counts": {name: len(paths) for name, paths in surfaces.items()},
        "surface_digests": digests,
        "missing_witnesses": missing_witnesses,
        "below_minimum": below_minimum,
    }
    result = "PASS_DISCOVERY" if not missing_witnesses and not below_minimum else "REVIEW_REQUIRED"
    summary["result"] = result
    print("SUMMARY " + json.dumps(summary, sort_keys=True, separators=(",", ":")))
    print(f"result={result}")
    return 0 if result == "PASS_DISCOVERY" else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""EPI-AUTH-FLOW-001 R5R4 deterministic authority-flow discovery.

Measurement-only audit of production Rust paths through which epistemic authority
can be produced, transported, converted into certainty/ordinals, or used by
Broca's 1D ordinal gate, per-axis cube gate, strict-code gate, and explicit
gating controls.

PASS_DISCOVERY is not a frozen inventory or semantic approval.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_PARTS = {
    ".git", "target", "vendor", "node_modules", ".direnv", "result",
    "examples", "tests", "benches",
}

SURFACES: dict[str, tuple[str, ...]] = {
    "typed_strong_empirical_authority": ("ETier::E3", "ETier::E4"),
    "typed_high_axis_authority": ("NTier::N3", "MTier::M3"),
    "typed_cube_definition": (
        "pub enum ETier", "pub enum NTier", "pub enum MTier", "pub struct EpistemicCube",
    ),
    "epistemic_status_definition": ("pub enum EpistemicStatus",),
    "epistemic_certainty_surface": ("EpistemicStatus::Certain",),
    "epistemic_code_confident_surface": ("EpistemicCode::Confident",),
    "epistemic_ordinal_surface": ("epistemic_ordinal",),
    "broca_ordinal_gate_surface": (
        "enable_epistemic_gate",
        "CANONICAL_HEDGING_WORDS",
        "CANONICAL_FACTUAL_WORDS",
        "epistemic_temperature_mode",
        "domain_familiarity_hedging_scale",
    ),
    "broca_cube_gate_surface": (
        "EpistemicCubeGate",
        "enable_epistemic_cube_gate",
        "E_AXIS_ASSERTION",
        "N_AXIS_AXIOMATIC",
        "M_AXIS_FOUNDATIONAL",
        "e_tier()",
        "n_tier()",
        "m_tier()",
    ),
    "broca_gate_control_surface": (
        "bypass_gating",
        "enable_epistemic_gate",
        "enable_epistemic_cube_gate",
        "domain_familiarity_hedging_scale",
    ),
    "strict_code_gate_surface": (
        "apply_strict_code_gate",
        "High certainty",
        "hallucination-prone",
    ),
    "legacy_empirical_authority": (
        "E3CryptographicallyProven", "E4PubliclyReproducible",
        "CryptographicallyVerifiable", "PubliclyReproducible",
        "E3Cryptographic", "E4PublicRepro",
    ),
    "legacy_high_axis_authority": (
        "N3Axiomatic", "M3Foundational",
        "NormativeLevel::Foundational", "MaterialityLevel::Permanent",
    ),
    "numeric_cube_transport": (
        "cube_e_tier", "cube_n_tier", "cube_m_tier",
        "last_cube_e_tier", "last_cube_n_tier", "last_cube_m_tier",
        "set_epistemic_cube", "inject_epistemic_cube", "EPISTEMIC_CUBE_BASE",
        "e_tier: u8", "n_tier: u8", "m_tier: u8",
    ),
    "scalar_cube_collapse": (
        "(e as f32 / 4.0) * 0.40", "(n as f32 / 3.0) * 0.35", "(m as f32 / 3.0) * 0.25",
        "(e_tier as f32 / 4.0) * 0.40", "(n_tier as f32 / 3.0) * 0.35",
        "(m_tier as f32 / 3.0) * 0.25",
    ),
    "broca_cube_sink": (
        "e_tier_e4", "n_tier_n3", "m_tier_m3",
        "Set the full 4D epistemic cube as channel data", "set_epistemic_cube(",
    ),
}

REQUIRED_WITNESSES: dict[str, dict[str, tuple[str, ...]]] = {
    "cycle_runtime_cube_producer": {
        "src/cognitive_loop/cycle_subsystems.rs": (
            "epistemic_gate_confidence > 0.9", "4u8 // E4: reproducible",
            "epistemic_gate_confidence > 0.7", "3 // E3: proven",
            "kg > 0.8 && sources >= 3", "3u8 // N3: axiomatic",
            "kg > 0.7", "3u8 // M3: foundational",
        ),
    },
    "runtime_cube_transport": {
        "src/cognitive_loop/types/carryover.rs": (
            "last_cube_e_tier", "last_cube_n_tier", "last_cube_m_tier",
        ),
        "src/cognitive_loop/cycle_phase_dynamics/training.rs": (
            "cube_e_tier: self.carryover.quality.last_cube_e_tier",
            "cube_n_tier: self.carryover.quality.last_cube_n_tier",
            "cube_m_tier: self.carryover.quality.last_cube_m_tier",
        ),
        "src/cognitive_loop/broca_bridge.rs": (
            "channels.set_epistemic_cube", "signals.cube_e_tier",
            "signals.cube_n_tier", "signals.cube_m_tier",
        ),
    },
    "broca_cube_conditioning": {
        "crates/domains/symthaea-broca/src/encoder.rs": (
            "e_tier_e4", "n_tier_n3", "m_tier_m3",
            "pub fn set_epistemic_cube", "pub fn epistemic_ordinal",
        ),
    },
    "typed_cube_to_broca": {
        "src/language/ssm_backend.rs": (
            "ETier::E3 => 3", "ETier::E4 => 4", "NTier::N3 => 3", "MTier::M3 => 3",
            "ch.set_epistemic_cube",
        ),
        "src/mind/structured_thought.rs": (
            "publicly reproducible proof", "axiomatic truth like math", "M3 (foundational)",
            "(ETier::E3, _, _) => \"peer-verified\"",
        ),
    },
    "typed_cube_to_certain": {
        "src/symthaea/mod.rs": (
            "ETier::E4 | ETier::E3 => EpistemicStatus::Certain",
            "ETier::E2 => EpistemicStatus::Probable",
        ),
    },
    "certain_translation_sink": {
        "src/language/llm_organ.rs": (
            "Add epistemic hedging if needed", "EpistemicStatus::Certain => {}",
        ),
        "src/language/ssm_backend.rs": (
            "ch.channels[8] = match thought.epistemic_status",
            "EpistemicStatus::Certain => 0.0",
        ),
    },
    "ordinal_gate_bypass": {
        "crates/domains/symthaea-broca/src/gating.rs": (
            "epistemic ordinal: 0=Certain, 1=Probable, 2=Uncertain, 3=Unknown, 4=OutOfDomain",
            "if epistemic_ordinal < 1.5",
            "Certain or Probable: no modification",
            "CANONICAL_HEDGING_WORDS",
            "CANONICAL_FACTUAL_WORDS",
            "domain_familiarity_hedging_scale",
        ),
    },
    "ordinal_generator_control": {
        "crates/domains/symthaea-broca/src/generator.rs": (
            "When bypass_gating is true, skip all gating for raw CfC quality iteration.",
            "if !self.config.bypass_gating && self.config.enable_epistemic_gate",
            "let ordinal = channels.epistemic_ordinal()",
            "self.epistemic_gate.apply(&mut logits, ordinal)",
        ),
    },
    "cube_gate_assertion_control": {
        "crates/domains/symthaea-broca/src/gating.rs": (
            "E3 (proven): allow confident assertion",
            "E4 (reproducible): full confidence",
            "N3 (axiomatic): boost",
            "M3 (foundational): boost",
            "Self::boost_ids(logits, &self.assertion_ids, 0.2)",
            "Self::boost_ids(logits, &self.assertion_ids, 0.4)",
            "Self::penalize_ids(logits, &self.hedging_ids, -0.3)",
            "Self::boost_ids(logits, &self.axiomatic_ids, 0.4)",
            "Self::boost_ids(logits, &self.foundational_ids, 0.4)",
        ),
    },
    "cube_gate_generator_control": {
        "crates/domains/symthaea-broca/src/generator.rs": (
            "Complements the 1D ordinal gate with fine-grained axis modulation",
            "if !self.config.bypass_gating && self.config.enable_epistemic_cube_gate",
            "self.epistemic_cube_gate.apply(&mut logits, channels)",
        ),
    },
    "strict_code_high_tier_bypass": {
        "crates/domains/symthaea-broca/src/gating.rs": (
            "pub fn apply_strict_code_gate",
            "if e > 1",
            "High certainty",
            "trust the model's creative flow",
        ),
        "crates/domains/symthaea-broca/src/codegate.rs": (
            ".apply_strict_code_gate(logits, channels, &self.tokenizer)",
        ),
    },
    "familiarity_hedging_attenuation": {
        "crates/domains/symthaea-broca/src/gating.rs": (
            "Domain familiarity reduces hedging boost",
            "let familiarity_scale =",
            "domain_familiarity_hedging_scale",
        ),
    },
    "ordinal_generation_path": {
        "crates/domains/symthaea-broca/src/generator.rs": (
            "enable_epistemic_gate", "channels.epistemic_ordinal",
        ),
        "crates/domains/symthaea-broca/src/liquid_mamba.rs": (
            "channels.epistemic_ordinal()",
        ),
        "crates/domains/symthaea-broca/src/decoder.rs": (
            "channels.epistemic_ordinal()",
        ),
    },
    "certain_dispatch_sink": {
        "src/language/intelligent_dispatcher.rs": (
            "EpistemicStatus::Certain if prediction_error < 0.3 => BackendTier::Native",
        ),
    },
    "certain_scalarization": {
        "src/symthaea/magi.rs": ("EpistemicStatus::Certain => 0.95",),
    },
    "heuristic_to_certain": {
        "src/coding_agent/experience.rs": ("if confidence > 0.9", "EpistemicStatus::Certain"),
        "src/mind/intent.rs": (
            "familiarity > 0.7 && novelty < 0.3 && negative_resonance < 0.08",
            "EpistemicStatus::Certain",
        ),
        "src/bin/symthaea-gaia-agent.rs": (
            "state.predictions_trustworthy && state.prediction_confidence >= 0.7",
            "EpistemicStatus::Certain",
        ),
    },
    "confident_code_to_certain": {
        "src/language/epistemic_generation.rs": (
            "EpistemicCode::Confident(_) => EpistemicStatus::Certain",
        ),
    },
    "synthesis_certainty_family": {
        "crates/core/symthaea-core/src/synthesis_trait.rs": (
            "pub enum EpistemicStatus", "Maps to Broca's epistemic gating levels", "Certain,",
        ),
        "src/language/code_orchestrator.rs": (
            "symthaea_core::synthesis_trait::EpistemicStatus::Certain",
            "EpistemicStatus::Certain",
        ),
    },
    "mycelix_numeric_producer": {
        "src/consciousness/mycelix_bridge.rs": (
            "pub e_tier: u8", "pub n_tier: u8", "pub m_tier: u8",
            "EmpiricalLevel::E3Cryptographic", "NormativeLevel::N3Axiomatic",
        ),
    },
    "hdc_statistical_producer": {
        "crates/core/symthaea-core/src/hdc/statistical_retrieval.rs": (
            "EmpiricalTier::from_z_score", "E3CryptographicallyProven",
            "E4PubliclyReproducible",
        ),
    },
    "physics_product_producer": {
        "crates/domains/symthaea-physics-catalog/src/discovery.rs": (
            "pub empirical: u8", "pub normative: u8", "pub materiality: u8",
        ),
    },
}

MIN_COUNTS = {
    "typed_strong_empirical_authority": 20,
    "typed_high_axis_authority": 20,
    "typed_cube_definition": 1,
    "epistemic_status_definition": 3,
    "epistemic_certainty_surface": 10,
    "epistemic_code_confident_surface": 1,
    "epistemic_ordinal_surface": 6,
    "broca_ordinal_gate_surface": 2,
    "broca_cube_gate_surface": 3,
    "broca_gate_control_surface": 2,
    "strict_code_gate_surface": 2,
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
    return hashlib.sha256("\n".join(paths).encode("utf-8")).hexdigest()


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

    print("schema=epi-auth-flow-001-r5r4-discovery-v1")
    print(
        "authority_scope="
        "measurement-only-producer-certainty-ordinal-cube-gates-and-controls"
    )
    for name in sorted(surfaces):
        paths = surfaces[name]
        print(f"SURFACE {name} count={len(paths)} digest={digests[name]}")
        for rel in paths:
            print(f"FILE {name} {rel}")

    summary = {
        "schema": "epi-auth-flow-001-r5r4-discovery-v1",
        "authority_scope":
            "measurement-only-producer-certainty-ordinal-cube-gates-and-controls",
        "surface_counts": {name: len(paths) for name, paths in surfaces.items()},
        "surface_digests": digests,
        "missing_witnesses": missing_witnesses,
        "below_minimum": below_minimum,
    }
    result = (
        "PASS_DISCOVERY"
        if not missing_witnesses and not below_minimum
        else "REVIEW_REQUIRED"
    )
    summary["result"] = result
    print("SUMMARY " + json.dumps(summary, sort_keys=True, separators=(",", ":")))
    print(f"result={result}")
    return 0 if result == "PASS_DISCOVERY" else 1


if __name__ == "__main__":
    raise SystemExit(main())

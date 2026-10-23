#!/usr/bin/env python3
"""EPI-AUTH-FLOW-001 R5R5 deterministic authority-flow discovery.

Measurement-only audit of production Rust paths through which epistemic authority
can be produced, transported, converted into ordinal-gate-bypass statuses, or
used by Broca/LLM expression consumers.

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
    "ordinal_bypass_status_surface": (
        "EpistemicStatus::Certain", "EpistemicStatus::Probable",
    ),
    "epistemic_code_bypass_surface": (
        "EpistemicCode::Confident", "EpistemicCode::Probable",
    ),
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
        "e_tier()", "n_tier()", "m_tier()",
    ),
    "broca_gate_control_surface": (
        "bypass_gating",
        "enable_epistemic_gate",
        "enable_epistemic_cube_gate",
        "enable_gating",
        "domain_familiarity_hedging_scale",
    ),
    "strict_code_gate_surface": (
        "apply_strict_code_gate", "High certainty", "hallucination-prone",
    ),
    "llm_prompt_authority_surface": (
        "TRANSLATION_SYSTEM_PROMPT",
        "EPISTEMIC_STATUS:",
        "EPISTEMIC_CUBE:",
        "COMPUTED_ANSWER",
        "guaranteed correct",
        "E4=reproducible proof",
        "N3=axiomatic truth like math",
        "M3=foundational",
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
    "typed_cube_to_bypass_status": {
        "src/symthaea/mod.rs": (
            "ETier::E4 | ETier::E3 => EpistemicStatus::Certain",
            "ETier::E2 => EpistemicStatus::Probable",
        ),
    },
    "consciousness_to_bypass_status": {
        "src/mind/epistemic.rs": (
            "fn determine_epistemic_status",
            "Consciousness Metrics",
            "let phi = state.consciousness_level",
            "EpistemicStatus::Certain",
            "EpistemicStatus::Probable",
            "Fallback to pure consciousness metrics if no text available",
        ),
        "src/symthaea/mod.rs": (
            "High Phi detected: Emboldening 'Uncertain' thought to 'Probable'",
            "EpistemicStatus::Probable",
        ),
    },
    "hdc_resonance_to_bypass_status": {
        "src/mind/intent.rs": (
            "pub fn assess_epistemic_text",
            "familiarity > 0.7 && novelty < 0.3",
            "EpistemicStatus::Certain",
            "EpistemicStatus::Probable",
        ),
    },
    "confidence_to_bypass_status": {
        "src/coding_agent/experience.rs": (
            "if confidence > 0.9",
            "else if confidence > 0.7",
            "EpistemicStatus::Certain",
            "EpistemicStatus::Probable",
        ),
        "src/language/epistemic_generation.rs": (
            "EpistemicCode::Confident(_) => EpistemicStatus::Certain",
            "EpistemicCode::Probable(_, _) => EpistemicStatus::Probable",
        ),
    },
    "ordinal_gate_bypass": {
        "crates/domains/symthaea-broca/src/gating.rs": (
            "epistemic ordinal: 0=Certain, 1=Probable, 2=Uncertain, 3=Unknown, 4=OutOfDomain",
            "if epistemic_ordinal < 1.5",
            "Certain or Probable: no modification",
            "CANONICAL_HEDGING_WORDS", "CANONICAL_FACTUAL_WORDS",
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
            "N3 (axiomatic): boost", "M3 (foundational): boost",
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
    "mamba_dual_gate_control": {
        "crates/domains/symthaea-broca/src/liquid_mamba.rs": (
            "pub enable_gating: bool",
            "if self.config.enable_gating",
            "self.epistemic_cube_gate",
            ".apply_scaled(&mut logits, channels, ep_scale)",
            "self.epistemic_gate.apply_scaled(",
            "channels.epistemic_ordinal()",
        ),
    },
    "spore_ordinal_gate_control": {
        "crates/domains/symthaea-spore/src/broca_full.rs": (
            "if self.config.enable_epistemic_gate",
            "channels.epistemic_ordinal()",
            "self.epistemic_gate.apply_with_familiarity",
        ),
    },
    "strict_code_high_tier_bypass": {
        "crates/domains/symthaea-broca/src/gating.rs": (
            "pub fn apply_strict_code_gate", "if e > 1", "High certainty",
            "trust the model's creative flow",
        ),
        "crates/domains/symthaea-broca/src/codegate.rs": (
            ".apply_strict_code_gate(logits, channels, &self.tokenizer)",
        ),
    },
    "familiarity_hedging_attenuation": {
        "crates/domains/symthaea-broca/src/gating.rs": (
            "Domain familiarity reduces hedging boost", "let familiarity_scale =",
            "domain_familiarity_hedging_scale",
        ),
    },
    "llm_prompt_status_transport": {
        "src/mind/structured_thought.rs": (
            "Epistemic status (CRITICAL for faithful translation)",
            "EPISTEMIC_STATUS: {:?}",
        ),
        "src/language/llm_organ.rs": (
            "RESPECT epistemic status",
            "Certain: Speak confidently",
            "Probable: Use \"likely\", \"probably\"",
            "EPISTEMIC_STATUS",
        ),
    },
    "llm_prompt_cube_authority": {
        "src/language/llm_organ.rs": (
            "EPISTEMIC_CUBE: If present",
            "E4=reproducible proof",
            "N3=axiomatic truth like math",
            "M3=foundational",
        ),
    },
    "llm_prompt_computed_answer_authority": {
        "src/language/llm_organ.rs": (
            "If COMPUTED_ANSWER is present, use it as the PRIMARY factual content",
            "This value was computed deterministically by Rust",
            "and is guaranteed correct",
        ),
    },
    "prompt_to_broca_ordinal": {
        "src/language/ssm_backend.rs": (
            "Parse EPISTEMIC_STATUS from prompt",
            "EPISTEMIC_STATUS: Unknown",
            "EPISTEMIC_STATUS: Certain",
        ),
    },
    "certain_probable_translation_sink": {
        "src/language/llm_organ.rs": (
            "Add epistemic hedging if needed",
            "EpistemicStatus::Certain => {}",
            "EpistemicStatus::Probable =>",
        ),
        "src/language/ssm_backend.rs": (
            "ch.channels[8] = match thought.epistemic_status",
            "EpistemicStatus::Certain => 0.0",
            "EpistemicStatus::Probable => 1.0",
        ),
    },
    "status_dispatch_sink": {
        "src/language/intelligent_dispatcher.rs": (
            "EpistemicStatus::Certain if prediction_error < 0.3 => BackendTier::Native",
            "EpistemicStatus::Certain | EpistemicStatus::Probable",
        ),
    },
    "status_scalarization": {
        "src/symthaea/magi.rs": (
            "EpistemicStatus::Certain => 0.95",
            "EpistemicStatus::Probable => 0.75",
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
    "ordinal_bypass_status_surface": 15,
    "epistemic_code_bypass_surface": 1,
    "epistemic_ordinal_surface": 6,
    "broca_ordinal_gate_surface": 2,
    "broca_cube_gate_surface": 3,
    "broca_gate_control_surface": 3,
    "strict_code_gate_surface": 2,
    "llm_prompt_authority_surface": 2,
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

    print("schema=epi-auth-flow-001-r5r5-discovery-v1")
    print(
        "authority_scope="
        "measurement-only-producer-bypass-status-ordinal-cube-prompt-gates-and-controls"
    )
    for name in sorted(surfaces):
        paths = surfaces[name]
        print(f"SURFACE {name} count={len(paths)} digest={digests[name]}")
        for rel in paths:
            print(f"FILE {name} {rel}")

    summary = {
        "schema": "epi-auth-flow-001-r5r5-discovery-v1",
        "authority_scope":
            "measurement-only-producer-bypass-status-ordinal-cube-prompt-gates-and-controls",
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

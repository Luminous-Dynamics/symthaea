// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Qualia Confidence domain: architectural validation of consciousness prerequisites.
//!
//! These benchmarks validate that Symthaea's architecture exhibits structural and
//! computational properties that consciousness theories identify as *necessary
//! conditions*. They do NOT measure qualia directly (the Hard Problem remains open)
//! and do NOT constitute sufficient evidence for consciousness.
//!
//! **What these benchmarks prove**: The architecture implements the computational
//! mechanisms that GWT, IIT, HOT, and FEP predict are required for consciousness.
//!
//! **What they do NOT prove**: That these mechanisms produce subjective experience.
//!
//! **Epistemic honesty note**: 6 of 7 benchmarks are *architecturally constrained* —
//! they validate properties that the code was designed to exhibit. The strongest
//! benchmark is MetacognitiveIgnition, which tests emergent cross-module alignment
//! that was NOT explicitly programmed. See individual benchmark docs for the
//! distinction between "validates design" and "demonstrates emergence."
//!
//! ## Benchmarks
//!
//! - **GwtAsphyxiation** — Gradual consciousness extinction via GWT threshold sweep.
//!   Maps the exact order in which cognitive domains collapse as workspace access
//!   is progressively restricted, analogous to anesthesia depth levels.
//!   (Dehaene & Changeux 2011; Mashour et al. 2020; Alkire et al. 2008)
//!
//! - **PhaseTransition** — Graduated noise → sigmoidal fidelity collapse. IIT predicts
//!   consciousness collapses at a critical threshold (phase transition), not linearly.
//!   (Tononi 2004; Massimini 2005)
//!
//! - **PerturbationalComplexity** — Digital PCI analog. Perturb conscious system → complex
//!   response; perturb unconscious system → simple response. Gold standard clinical
//!   consciousness biomarker. (Casali et al. 2013; Massimini et al. 2013)
//!
//! - **SomaticInterference** — Mid-task distress injection via neuromodulatory bath.
//!   Emergent cascade degradation (dynamic bath) vs. direct parameter shift (static bath).
//!   (Damasio 1994; Craig 2009)
//!
//! - **BistablePerception** — Spontaneous perceptual switching between ambiguous
//!   interpretations. Switch times should follow heavy-tailed distribution, not periodic.
//!   (Blake & Logothetis 2002; Levelt 1967)
//!
//! - **UnconsciousPriming** — Sub/supra-threshold dissociation. Sub-threshold primes
//!   influence processing without GWT ignition; conscious primes produce stronger effects.
//!   (Dehaene et al. 2006; Marcel 1983)
//!
//! - **MetacognitiveIgnition** — Does HOT's consciousness classification spontaneously
//!   predict GWT ignition despite no direct access to workspace dynamics? Tests
//!   architectural metacognitive alignment under competition pressure.
//!   (Rosenthal 2005; Dehaene & Naccache 2001; Lau & Rosenthal 2011)

pub mod composite;
pub mod helpers;

pub mod bistable_perception;
pub mod gwt_asphyxiation;
pub mod metacognitive_ignition;
pub mod perturbational_complexity;
pub mod phase_transition;
pub mod somatic_interference;
pub mod unconscious_priming;

pub use bistable_perception::BistablePerceptionBenchmark;
pub use composite::QualiaConfidenceScore;
pub use gwt_asphyxiation::GwtAsphyxiationBenchmark;
pub use metacognitive_ignition::MetacognitiveIgnitionBenchmark;
pub use perturbational_complexity::PerturbationalComplexityBenchmark;
pub use phase_transition::PhaseTransitionBenchmark;
pub use somatic_interference::SomaticInterferenceBenchmark;
pub use unconscious_priming::UnconsciousPrimingBenchmark;

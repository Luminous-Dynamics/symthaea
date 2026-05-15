// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Symthaea Broca: SSM Language Center

// Allow lints that are pervasive in Mamba/SSM numerical code
#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::needless_borrow)]
#![allow(clippy::excessive_precision)]
#![allow(clippy::manual_div_ceil)]
#![allow(clippy::manual_is_multiple_of)]
#![allow(clippy::unnecessary_map_or)]

pub mod architect;
pub mod affective_sculpting;
pub mod architectural_memory;
pub mod checkpoint;
#[cfg(feature = "code-sheaf-eval")]
pub mod code_analysis;
pub mod cognitive_loop;
pub mod compiler_trainer;
pub mod consensus_engine;
pub mod controller;
pub mod dreaming;
pub mod encoder;
pub mod epistemic_dashboard;
pub mod epistemic_scorers;
pub mod evaluation;
pub mod evolutionary_scaffolder;
pub mod formal_logic_scorer;
pub mod gating;
pub mod generator;
pub mod generic_structural_scorer_integration;
pub mod go_walker;
#[cfg(feature = "gpu")]
pub mod gpu_cfc;
pub mod invariant_discovery;
pub mod inverse_harvester;
pub mod living_manifests;
pub mod manifold_projection;
pub mod moral_safety_scorer;
pub mod narrative_maintainability_scorer;
pub mod narrative_planner;
pub mod nix_kg;
pub mod physiological_scorer;
pub mod python_walker;
pub mod rust_walker;
pub mod secure_dreaming;
pub mod self_actualization;
pub mod self_optimization;
pub mod species_learning;
pub mod speech_encoder;
pub mod structural_generator;
pub mod structural_scorer;
pub mod substrate_binding;
pub mod thought_chunk;
pub mod tokenizer;
pub mod training;
pub mod trans_substrate_invariants;
pub mod zero_shot_substrate;

// IaC Expansion
pub mod codegate;
pub mod emotional_gating_integration;
pub mod iac_harvester;
pub mod iac_repair;
pub mod language_gates;
pub mod memory_bridge;

// Speech heuristics: bootstrap targets for SpeechThoughtEncoder training
#[cfg(feature = "speech-data")]
pub mod speech_heuristics;

// Creative mode: relaxed gating for poetry and artistic text generation
pub mod creative_mode;
pub mod syllable;

// Liquid-Mamba fusion: pre-trained Mamba SSM + HDC projection + consciousness gating
#[cfg(feature = "mamba-cpu")]
pub mod liquid_mamba;
#[cfg(feature = "mamba-cpu")]
pub mod mamba;
#[cfg(feature = "mamba-cpu")]
pub mod mamba_model;
#[cfg(feature = "mamba-cpu")]
pub mod projection;
#[cfg(feature = "mamba-cpu")]
pub mod temporal_projection;

pub use architect::SimulationArchitect;
pub use affective_sculpting::{AffectiveSculptor, AffectiveStyle};
pub use architectural_memory::ArchitecturalMemory;
pub use checkpoint::{AdamState, BrocaCheckpoint, BrocaCheckpointMetadata};
#[cfg(feature = "code-sheaf-eval")]
pub use code_analysis::{
    categorize_code_sheaf_diagnostic, extract_rust_functions, repair_hint_for_code_sheaf_category,
    RustFunctionExtraction,
};
pub use consensus_engine::{ChangeProposal, ConsensusEngine, ConsensusResult};
pub use controller::{LanguageController, LanguageControllerConfig, NetworkSnapshot};
pub use dreaming::DreamingService;
pub use encoder::{
    ThoughtChannels, ThoughtLanguageEncoder, EPISTEMIC_CUBE_BASE, EPISTEMIC_CUBE_CHANNELS,
};
pub use epistemic_dashboard::{CognitiveStyle, EpistemicDashboard};
pub use epistemic_scorers::{compute_epistemic_reward, compute_idiomaticity};
pub use evaluation::{
    CanonicalEvalCase, CanonicalEvalDataset, CanonicalQualityThresholds, CategoryQuality,
    EvalConfig, EvalResult, IntentScore, QualityDelta, QualityGateFailure, QualitySuiteResult,
};
pub use evolutionary_scaffolder::{EvolutionResult, EvolutionaryScaffolder};
pub use formal_logic_scorer::{FormalLogicScorer, FormalVerificationResult};
pub use gating::{
    CodeGate, CoherenceFeedback, EmotionalModulator, EpistemicCubeGate, EpistemicGate, GatingConfig,
};
pub use generator::{BrocaConfig, BrocaGenerator, GenerationResult, SamplingStrategy};
pub use generic_structural_scorer_integration::{
    GenericStructuralScorer, StructuralVerdict as GenericStructuralVerdict,
};
pub use go_walker::GoWalker;
pub use invariant_discovery::InvariantDiscovery;
pub use inverse_harvester::{InverseHarvestPair, InverseHarvester};
pub use living_manifests::{ComponentDoc, LivingManifest, LivingManifestGenerator};
pub use manifold_projection::ManifoldProjection;
pub use moral_safety_scorer::compute_moral_safety;
pub use narrative_maintainability_scorer::compute_narrative_maintainability;
pub use narrative_planner::{ArcStatus, ChangeArc, ChangeStep, NarrativePlanner};
pub use nix_kg::NixKg;
pub use physiological_scorer::{PhysiologicalProfile, PhysiologicalScorer};
pub use python_walker::PythonWalker;
pub use rust_walker::{LanguageWalker, RustWalker, StructuralElement};
pub use secure_dreaming::{SecureDreamResult, SecureDreamingEngine};
pub use self_actualization::ReflectionEngine;
pub use self_optimization::SelfOptimizationEngine;
pub use species_learning::MemoryConsolidator;
pub use structural_generator::StructuralGenerator;
pub use structural_scorer::{NixStructuralScorer, StructuralVerdict};
pub use substrate_binding::{AnticipatedImpact, ImpactRecommendation, SubstrateBindingEngine};
pub use thought_chunk::{
    ProgramNode, ThoughtChunk, ThoughtChunkDecoder, ThoughtChunkKind, ThoughtChunkSequence,
};
pub use tokenizer::BpeTokenizer;
pub use training::{
    AnomalyReport, GradientAnomaly, GradientDiagnostics, SequenceResult, TrainingBackend,
    TrainingDataset, TrainingPair, TrainingValidation,
};
pub use trans_substrate_invariants::{CrossLanguageInvariant, TransSubstrateInvariantEngine};
pub use zero_shot_substrate::ZeroShotInducer;

#[cfg(feature = "mamba-cpu")]
pub use checkpoint::ProjectionCheckpoint;
#[cfg(feature = "mamba-cpu")]
pub use evaluation::{
    GatingTestResult, LiquidMambaEvalConfig, LiquidMambaEvalResult, QualityGateThresholds,
};
#[cfg(feature = "mamba-cpu")]
pub use liquid_mamba::{LiquidMambaConfig, LiquidMambaGenerator};
#[cfg(feature = "mamba-cpu")]
pub use mamba::MambaBackend;
#[cfg(feature = "mamba-cpu")]
pub use projection::{
    GradientDiagnosticsSnapshot, GradientStepMetrics, HdcSsmProjection,
    ProjectionGradientDiagnostics,
};
#[cfg(feature = "mamba-cpu")]
pub use temporal_projection::TemporalProjection;

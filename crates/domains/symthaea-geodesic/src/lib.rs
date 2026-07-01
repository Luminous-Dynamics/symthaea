// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

#![deny(unsafe_code)]

//! # Geodesic Code Synthesis
//!
//! Topology-aware code generation via manifold navigation, execution prediction,
//! and sheaf coherence verification.
//!
//! ## Architecture
//!
//! Six layers, each constraining the next:
//! 1. **PDG**: Program Dependence Graph from parsed ASTs
//! 2. **Topology**: Persistent homology (Betti numbers) as structural constraints
//! 3. **Manifold**: Fiber bundle of implementations over specifications
//! 4. **Oracle**: LTC/CfC execution prediction with O(1) temporal jumps
//! 5. **Sheaf**: Local-to-global coherence verification
//! 6. **Synthesis**: Geodesic walk through program manifold space
//!
//! ## What This Enables That LLMs Cannot Do
//!
//! - Pre-generation verification (predict what code will do before it exists)
//! - Topological invariants as hard constraints (exactly N loops)
//! - Formal composability via sheaf conditions
//! - O(1) complexity prediction via CfC convergence rate
//! - Structural equivalence across syntactic variations

pub mod ast_bridge;
pub mod codebase_bridge;
pub mod composer;
pub mod emitter_bridge;
pub mod execution_oracle;
pub mod manifold;
pub mod manifold_bootstrap;
pub mod noise;
pub mod pdg;
pub mod periodic_table;
pub mod program_emitter;
pub mod program_memory;
pub mod resonant_explorer;
pub mod sheaf;
pub mod skeleton_synthesis;
pub mod synthesis;
pub mod token_codebook;
pub mod topology;
pub mod tri_oracle;
pub mod understanding;
pub mod verification;

// Re-export key types for convenience.
pub use codebase_bridge::{IndexResult, index_directory, index_file};
pub use emitter_bridge::{
    GeodesicCodeSpec, GeodesicPlanAction, GeodesicPlanStep, emit_rust_from_skeleton,
    skeleton_to_code_spec, skeleton_to_plan_steps,
};
pub use execution_oracle::{ComplexityClass, ExecutionOracle, PredictionResult};
pub use manifold::{Fiber, FiberPoint, ProgramManifold};
pub use manifold_bootstrap::{BootstrapResult, bootstrap_from_encodings, bootstrap_with_topology};
pub use pdg::ProgramDependenceGraph;
pub use program_emitter::{emit_expression, emit_rust};
pub use program_memory::{ProgramMemory, ProgramMemoryEntry};
pub use resonant_explorer::{ExplorationConfig, ExplorationResult, ResonantExplorer};
pub use sheaf::{
    CodeSheaf, LocalSection, RustSheafCoherence, SheafDiagnostic,
    categorize_rust_v0_sheaf_diagnostic, repair_hint_for_rust_v0_sheaf_category,
    verify_rust_v0_sheaf_coherence,
};
pub use skeleton_synthesis::{
    ActiveInferenceResult, GeodesicIntentClass, GeodesicRequestProfile, SkeletonCombinator,
    SkeletonSlot, TopologicalSignature, active_inference_synthesize, build_skeleton_from_topology,
    classify_geodesic_request, default_expression_for_type, fill_from_manifold,
    fill_skeleton_defaults_for_signature, geodesic_hints, normalize_signature_for_geodesic_emitter,
};
pub use synthesis::{CodeSpec, GeodesicSynthesizer, SynthesisConfig, SynthesisResult};
pub use token_codebook::TokenCodebook;
pub use topology::{BettiNumbers, TopologicalConstraint, TopologicalFingerprint};
pub use tri_oracle::{TriOracle, TriOracleConfig, TriOracleScore};
pub use verification::{VerificationResult, verify_generated_code};

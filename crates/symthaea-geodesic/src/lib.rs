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

pub mod execution_oracle;
pub mod manifold;
pub mod pdg;
pub mod sheaf;
pub mod synthesis;
pub mod topology;

// Re-export key types for convenience.
pub use execution_oracle::{ComplexityClass, ExecutionOracle, PredictionResult};
pub use manifold::{Fiber, FiberPoint, ProgramManifold};
pub use pdg::ProgramDependenceGraph;
pub use sheaf::{CodeSheaf, LocalSection, SheafDiagnostic};
pub use synthesis::{CodeSpec, GeodesicSynthesizer, SynthesisConfig, SynthesisResult};
pub use topology::{BettiNumbers, TopologicalConstraint, TopologicalFingerprint};

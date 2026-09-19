// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # symthaea-physics-bridge
//!
//! HDC semantic search engine for mathematical physics.
//!
//! Encodes equations, tensors, symmetry groups, and physical quantities as
//! `ContinuousHV` (16,384-dimensional hypervectors) for structural analogy
//! discovery. This is **not** a PDE solver — it's a semantic search engine
//! that finds cross-domain structural isomorphisms in mathematical physics.
//!
//! ## Architecture
//!
//! ```text
//! PhysicsEquation ──→ EquationEncoder ──→ ContinuousHV ──→ PhysicsSearchEngine
//!                      ├─ full encoding (with names)          ├─ multi-aspect search
//!                      ├─ skeleton (names stripped)            ├─ skeleton search
//!                      ├─ symmetry profile                    ├─ symmetry search
//!                      └─ dimensional signature               └─ dimensional search
//! ```
//!
//! ## Key Capability
//!
//! Skeleton encoding strips all names, revealing pure structural similarity:
//! - Helmholtz (∇²ψ + k²ψ = 0) and Schrödinger (∇²ψ + (2m/ℏ²)(E-V)ψ = 0)
//!   have **identical skeletons** despite being from different physics domains.
//!
//! ## Catalog
//!
//! ~27 pre-encoded landmark equations from electromagnetism (Maxwell's 4),
//! general relativity (Einstein, Schwarzschild, Kerr, FLRW, etc.),
//! quantum mechanics (Schrödinger, Dirac, Klein-Gordon), field theory
//! (Yang-Mills, Euler-Lagrange), fluids (Navier-Stokes), cosmology
//! (Friedmann), and the Spark Engine (Gamow peak, Coulomb screening).
//!
//! ## Integration
//!
//! `PhysicsBridge` wraps the search engine and produces `ContinuousHV`
//! compatible with `CognitiveLoopService::cycle_with_hv()`.

#![deny(unsafe_code)]

pub mod bridge;
pub mod case_studies;
pub mod catalog;
pub mod dimensional;
pub mod dimensional_inference;
pub mod discrepancy_localization;
pub mod discovery;
pub mod equation_ast;
pub mod integrated_discovery;
pub mod lyapunov;
pub mod model_contract;
pub mod model_validation;
pub mod noise_robustness;
pub mod pde_wave_stage_a;
pub mod pde_wave_stage_b;
pub mod query;
pub mod recognize;
mod solver_federation;
pub mod solver_federation_evidence;
mod solver_federation_gate;
pub mod strict_dimensional_inference;
pub mod symmetry;
pub mod symmetry_inference;
pub mod tensor_structure;
pub mod typed_generation;
pub mod types;

// Re-export primary API types.
pub use bridge::PhysicsBridge;
pub use catalog::PhysicsCatalog;
pub use dimensional::DimensionalEncoder;
pub use dimensional_inference::{InferenceResult, UnitMap, infer_dimensions};
pub use discrepancy_localization::{
    DISCREPANCY_LOCALIZATION_SCHEMA, DisagreeingCellRef, DiscrepancyAxisKind,
    DiscrepancyCell, DiscrepancyExtent, DiscrepancyLocalizationFinding,
    DiscrepancyLocalizationPlan, DiscrepancyLocalizationPlanIssue,
    DiscrepancyLocalizationReport, FrozenDiscrepancyLocalizationPlan, LocalizationAxisSpec,
    LocalizationClosure, LocalizationPartitionSpec, SolverPairTarget,
    evaluate_discrepancy_localization,
};
pub use equation_ast::EquationEncoder;
pub use model_contract::{
    ContractClosure, ContractFinding, DimensionAssessment, ObligationEvaluation,
    ObligationKind, ObligationOutcome, PhysicalModelContract, PhysicalModelContractIssue,
    PhysicalModelReport, PhysicalObligationSpec, FrozenPhysicalModelContract,
    evaluate_physical_model,
};
pub use model_validation::{
    PHYSICAL_MODEL_REPORT_SCHEMA, PhysicalModelEvaluationInput, SatisfiedPhysicalModel,
    evaluate_satisfied_physical_model, physical_model_report_digest,
};
pub use query::{PhysicsSearchEngine, SearchWeights};
pub use recognize::{
    RecognitionReport, expr_to_equation_node, recognize_expr, recognize_expr_with_units,
};
pub use solver_federation::{
    FederationCoverage, FederationFinding, FrozenSolverFederationSpec, PairLineage,
    PairwiseAgreement, SolverAgreementState, SolverFederationReport, SolverFederationSpec,
    SolverMethodFamily, SolverPairComparison, SolverReceipt,
};
pub use solver_federation_evidence::{
    SolverFederationEvidence, evaluate_solver_federation_evidence,
};
pub use solver_federation_gate::{
    SolverFederationGateIssue, SolverReceiptBindingIssue, ValidatedSolverFederationReport,
    ValidatedSolverReceipt, bind_solver_receipt,
};
pub use strict_dimensional_inference::{
    StrictInferenceFailure, StrictInferenceResult, infer_dimensions_strict,
};
pub use symmetry::SymmetryEncoder;
pub use symmetry_inference::infer_symmetry;
pub use tensor_structure::TensorEncoder;
pub use typed_generation::random_expr_with_dimension;
pub use types::*;

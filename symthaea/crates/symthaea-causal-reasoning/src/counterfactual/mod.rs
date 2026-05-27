// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Counterfactual Reasoning Subsystem v0
//!
//! Verified subset of causal reasoning with honest "I don't know":
//! - Backdoor adjustment (DAGs ≤20)
//! - Frontdoor criterion
//! - Graph surgery (HDC unbinding)
//! - Reference harness (brute-force on DAGs ≤8, 99% match required)
//!
//! ## Key property: never overclaims
//!
//! Every query returns one of:
//! - `Identified`: we have a valid estimand + method
//! - `Unidentified`: we cannot identify the causal effect (with reason)
//! - `AssumptionRequired`: we can identify IF an assumption holds

pub mod composer;
pub mod hdc_surgery;
pub mod identification;
pub mod semantic_roles;

// Re-export key types
pub use identification::{
    CPDAG,
    CausalAssumption,
    CausalDAG,
    CausalEstimand,
    CausalExpression,
    // Shpitser-Pearl ID Algorithm
    CausalGraphWithLatents,
    CausalQuery,
    CausalQueryOutcome,
    CausalReferenceHarness,
    CounterfactualReasoner,
    EffectEstimator,
    GrangerResult,
    HarnessResult,
    IDAlgorithm,
    // Instrumental Variables
    IVEstimator,
    IVResult,
    IVValidity,
    IdentificationMethod,
    // Mediation Analysis
    MediationAnalysis,
    MediationIdentification,
    MediationResult,
    // Effect estimation
    ObservationalData,
    // Causal Discovery
    PCAlgorithm,
    PCResult,
    RobustEstimate,
    SensitivityAnalysis,
    Skeleton,
    // Time-Series Causal Discovery
    TimeSeriesCausalDiscovery,
    TimeSeriesCausalGraph,
    TimeSeriesData,
    // Transportability
    TransportabilityAnalyzer,
    TransportabilityResult,
    UnidentifiedReason,
};

// Note: IdentificationMethod now includes:
// - DSeparation (Rule 1)
// - BackdoorAdjustment (special case of Rule 1)
// - FrontdoorCriterion (special case of Rule 1)
// - Rule2ActionObservation (Pearl's Rule 2)
// - Rule3ActionDeletion (Pearl's Rule 3)
pub use composer::CounterfactualComposer;
pub use hdc_surgery::GraphSurgery;
pub use semantic_roles::{RoleSubstitution, SemanticRole};

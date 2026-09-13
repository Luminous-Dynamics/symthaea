// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Counterfactual Reasoning Subsystem v0
//!
//! Qualified / bounded causal reasoning includes:
//! - Backdoor adjustment (DAGs ≤20)
//! - Frontdoor criterion
//! - Graph surgery (HDC unbinding)
//! - exact Markovian g-formula identification through [`QualifiedMarkovianId`]
//! - reference/conformance harnesses for selected graph families
//!
//! ## Epistemic boundary
//!
//! The subsystem distinguishes `Identified`, `Unidentified`, and `AssumptionRequired` outcomes,
//! but those types do not by themselves prove every identification implementation is complete.
//! In particular, the broader legacy [`IDAlgorithm`] latent-variable path remains experimental
//! under RQ-005; callers needing qualified causal authority should use explicitly qualified APIs
//! or separately validated evidence for the graph family being queried.

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
    // Qualified identification
    QUALIFIED_MARKOVIAN_ID_VERSION,
    QualifiedMarkovianId,
    QualifiedMarkovianIdError,
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

// Note: IdentificationMethod includes:
// - DSeparation (Rule 1)
// - BackdoorAdjustment (special case of Rule 1)
// - FrontdoorCriterion (special case of Rule 1)
// - Rule2ActionObservation (Pearl's Rule 2)
// - Rule3ActionDeletion (Pearl's Rule 3)
// - IDAlgorithm (legacy broad latent-variable path; qualification remains graph-family-specific)
pub use composer::CounterfactualComposer;
pub use hdc_surgery::GraphSurgery;
pub use semantic_roles::{RoleSubstitution, SemanticRole};

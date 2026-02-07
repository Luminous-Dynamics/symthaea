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

pub mod hdc_surgery;
pub mod identification;
pub mod semantic_roles;
pub mod composer;

// Re-export key types
pub use identification::{
    CausalQueryOutcome, CausalEstimand, IdentificationMethod,
    UnidentifiedReason, CausalAssumption, CausalDAG, CausalQuery,
    CounterfactualReasoner, CausalReferenceHarness, HarnessResult,
    // Effect estimation
    ObservationalData, EffectEstimator, RobustEstimate, SensitivityAnalysis,
    // Shpitser-Pearl ID Algorithm
    CausalGraphWithLatents, CausalExpression, IDAlgorithm,
    // Causal Discovery
    PCAlgorithm, PCResult, Skeleton, CPDAG,
    // Mediation Analysis
    MediationAnalysis, MediationResult, MediationIdentification,
    // Instrumental Variables
    IVEstimator, IVValidity, IVResult,
    // Time-Series Causal Discovery
    TimeSeriesCausalDiscovery, TimeSeriesData, GrangerResult, TimeSeriesCausalGraph,
    // Transportability
    TransportabilityAnalyzer, TransportabilityResult,
};

// Note: IdentificationMethod now includes:
// - DSeparation (Rule 1)
// - BackdoorAdjustment (special case of Rule 1)
// - FrontdoorCriterion (special case of Rule 1)
// - Rule2ActionObservation (Pearl's Rule 2)
// - Rule3ActionDeletion (Pearl's Rule 3)
pub use hdc_surgery::GraphSurgery;
pub use semantic_roles::{SemanticRole, RoleSubstitution};
pub use composer::CounterfactualComposer;

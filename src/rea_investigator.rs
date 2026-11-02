//! Native, fail-closed planning over caller-supplied REA Evidence provenance.
//!
//! The planner selects and ranks proposed experiments but never executes them.
//! Provenance classes and likelihood models are caller-supplied and are not
//! authenticated or calibrated by this module.

pub use symthaea_rea_investigator::decision::{
    DecisionAbstentionReason,
    DecisionAwareConfig,
    DecisionAwareExperimentResult,
    DecisionAwarePlan,
    DecisionClaimCeiling,
    DecisionError,
    DecisionErrorKind,
    DecisionOutcomeResult,
    DecisionRankingPolicy,
    DecisionRejectionReason,
    DecisionScenarioResult,
    HypothesisUtility,
    RejectedDecisionExperiment,
    SharedPolicyOutcome,
    TerminalDecision,
    plan_next_experiment_decision_aware,
};

pub use symthaea_rea_investigator::robust::{
    LikelihoodModelScenario,
    RobustAbstentionReason,
    RobustClaimCeiling,
    RobustExperimentProposal,
    RobustInvestigationPlan,
    RobustPlannerConfig,
    RobustRankedExperiment,
    RobustRankingPolicy,
    RobustRejectedExperiment,
    RobustRejectionReason,
    ScenarioExperimentLikelihood,
    ScenarioInformationGain,
    plan_next_experiment_robust,
};

pub use symthaea_rea_investigator::{
    AbstentionReason,
    Authorization,
    CostWeights,
    EvidenceOrigin,
    EvidenceProvenance,
    ExperimentProposal,
    InvestigationPlan,
    LikelihoodRow,
    MeasuredInformationGain,
    PlannerClaimCeiling,
    PlannerConfig,
    PlannerError,
    PlannerErrorKind,
    PredictedOutcome,
    Probability,
    RankedExperiment,
    RejectedExperiment,
    RejectionReason,
    TargetImpact,
    Hypothesis,
    measure_observed_information_gain,
    plan_next_experiment,
};

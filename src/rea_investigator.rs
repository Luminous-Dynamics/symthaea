//! Native, fail-closed planning over caller-supplied REA Evidence provenance.
//!
//! The planner selects and ranks proposed experiments and downstream decisions,
//! but never executes experiments or terminal actions. Provenance classes,
//! likelihood models, scenario weights and utility values are caller-supplied;
//! they are not authenticated or calibrated by this module.

pub use symthaea_rea_investigator::decision::{
    DecisionAbstentionReason,
    DecisionClaimCeiling,
    DecisionImpact,
    DecisionInvestigationPlan,
    DecisionOutcomeResult,
    DecisionPlannerConfig,
    DecisionRankedExperiment,
    DecisionRankingPolicy,
    DecisionRejectedExperiment,
    DecisionRejectionReason,
    DecisionScenarioResult,
    DecisionUtility,
    DecisionUtilityModel,
    OutcomeDecision,
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

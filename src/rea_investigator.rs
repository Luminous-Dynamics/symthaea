//! Native, fail-closed planning over caller-supplied REA Evidence provenance.
//!
//! The planner ranks proposed experiments and downstream decisions, and evaluates
//! held-out likelihood calibration. It never executes experiments or terminal
//! actions. Evidence provenance, likelihood models, source labels and utility
//! values remain caller-supplied inputs until separately qualified.

pub use symthaea_rea_investigator::calibration::{
    CALIBRATION_BOOTSTRAP_UNIT,
    CALIBRATION_EVALUATOR_VERSION,
    CalibrationClaimCeiling,
    CalibrationConfig,
    CalibrationEligibility,
    CalibrationError,
    CalibrationErrorKind,
    CalibrationInterval,
    CalibrationMetrics,
    CalibrationModelIdentity,
    CalibrationObservation,
    CalibrationReceipt,
    CalibrationScope,
    CalibrationThresholds,
    ClassProbability,
    ReliabilityBin,
    evaluate_likelihood_calibration,
    verify_calibration_receipt_against_observations,
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

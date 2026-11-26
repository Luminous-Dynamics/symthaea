//! Paired intervention-response evaluation for controlled counterfactual branches.
//!
//! Every pair shares an evaluator-committed source state and exogenous forcing
//! digest. The evaluator scores the predicted treatment-minus-control contrast,
//! not two unrelated branch predictions. Independent environment clusters are
//! the analysis unit; repeated pair rows are averaged inside clusters first.
//!
//! This module is evaluator-side arithmetic and contract validation only. A
//! trusted launcher must keep branch truth, hidden mechanisms and signing keys
//! outside the candidate process, and authenticate all evaluator-assigned
//! identities and digests under the #9437 OS/process boundary.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

pub const PAIRED_INTERVENTION_SCORE_PPM: u32 = 1_000_000;
pub const PAIRED_INTERVENTION_FAMILY_COUNT: usize = 8;
pub const PAIRED_INTERVENTION_MAX_PAIRS: usize = 8_192;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum InterventionFamily {
    Directional,
    Parameter,
    Structural,
    Temporal,
    Interaction,
    NullInvariance,
    ConfoundingControl,
    RegimeShift,
}

impl InterventionFamily {
    pub const ALL: [Self; PAIRED_INTERVENTION_FAMILY_COUNT] = [
        Self::Directional,
        Self::Parameter,
        Self::Structural,
        Self::Temporal,
        Self::Interaction,
        Self::NullInvariance,
        Self::ConfoundingControl,
        Self::RegimeShift,
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Directional => "directional",
            Self::Parameter => "parameter",
            Self::Structural => "structural",
            Self::Temporal => "temporal",
            Self::Interaction => "interaction",
            Self::NullInvariance => "null_invariance",
            Self::ConfoundingControl => "confounding_control",
            Self::RegimeShift => "regime_shift",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BranchDisposition {
    Completed,
    Failed,
    TimedOut,
    Invalid,
    NotExecuted,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PairDisposition {
    Completed,
    Failed { penalty_ppm: u32 },
    TimedOut { penalty_ppm: u32 },
    Invalid { penalty_ppm: u32 },
    Abstained { penalty_ppm: u32 },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct InterventionFamilyProfile {
    pub family: InterventionFamily,
    pub profile_manifest_sha256: [u8; 32],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PairedBranchResourceUse {
    pub interaction_count: u64,
    pub model_query_count: u64,
    pub elapsed_us: u64,
    pub measured_energy_uj: Option<u64>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PairedInterventionResourceBudget {
    pub max_pairs: u64,
    pub max_total_branch_interactions: u64,
    pub max_total_model_queries: u64,
    pub max_total_elapsed_us: u64,
    pub max_total_energy_uj: Option<u64>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PairedInterventionContext {
    pub campaign_id: u64,
    pub candidate_artifact_sha256: [u8; 32],
    pub model_checkpoint_sha256: [u8; 32],
    pub source_history_sha256: [u8; 32],
    pub environment_generator_sha256: [u8; 32],
    pub intervention_design_sha256: [u8; 32],
    pub evaluator_profile_sha256: [u8; 32],
    pub seed_manifest_sha256: [u8; 32],
    pub evaluation_split_sha256: [u8; 32],
    /// Exact, evaluator-declared variables that should remain invariant.
    pub expected_invariant_variable_ids: Vec<String>,
    pub minimum_independent_clusters_per_family: u64,
    /// Effects at or below this magnitude are excluded from sign accuracy.
    pub effect_resolution_floor_micro: u64,
    /// The evaluator truth must respect this tolerance for declared invariants.
    pub true_invariance_tolerance_micro: u64,
    /// Predicted invariant-variable changes above this tolerance are false changes.
    pub predicted_invariance_tolerance_micro: u64,
    pub resource_budget: PairedInterventionResourceBudget,
    pub family_profiles: Vec<InterventionFamilyProfile>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PairedInterventionBranch {
    pub branch_id: String,
    pub branch_manifest_sha256: [u8; 32],
    pub source_checkpoint_sha256: [u8; 32],
    pub disposition: BranchDisposition,
    pub true_outcome_micro: Option<i64>,
    pub predicted_outcome_mean_micro: Option<i64>,
    pub resources: PairedBranchResourceUse,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct InvariantVariableContrast {
    pub variable_id: String,
    pub control_true_micro: i64,
    pub treatment_true_micro: i64,
    pub control_predicted_micro: i64,
    pub treatment_predicted_micro: i64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PredictedEffectInterval {
    pub lower_micro: i64,
    pub upper_micro: i64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PairedInterventionPair {
    pub pair_id: String,
    pub run_id: u64,
    pub family: InterventionFamily,
    pub environment_family_id: String,
    pub environment_cluster_id: String,
    pub intervention_profile_sha256: [u8; 32],
    pub intervention_record_sha256: [u8; 32],
    /// Both branches start from this exact committed pre-intervention state.
    pub source_state_snapshot_sha256: [u8; 32],
    /// Common exogenous forcing/randomness commitment for the paired branches.
    pub exogenous_forcing_sha256: [u8; 32],
    pub target_variable_id: String,
    pub control_target_value_micro: i64,
    pub treatment_target_value_micro: i64,
    pub control: PairedInterventionBranch,
    pub treatment: PairedInterventionBranch,
    pub predicted_effect_interval: Option<PredictedEffectInterval>,
    pub invariant_variables: Vec<InvariantVariableContrast>,
    pub disposition: PairDisposition,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PairedInterventionInput {
    pub context: PairedInterventionContext,
    /// Complete pair inventory committed before confirmatory outcomes are opened.
    pub expected_pair_ids: Vec<String>,
    pub pairs: Vec<PairedInterventionPair>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PairedInterventionPairResult {
    pub pair_id: String,
    pub run_id: u64,
    pub family: InterventionFamily,
    pub environment_family_id: String,
    pub environment_cluster_id: String,
    pub control_branch_id: String,
    pub treatment_branch_id: String,
    pub control_branch_manifest_sha256: [u8; 32],
    pub treatment_branch_manifest_sha256: [u8; 32],
    pub source_state_snapshot_sha256: [u8; 32],
    pub exogenous_forcing_sha256: [u8; 32],
    pub intervention_profile_sha256: [u8; 32],
    pub intervention_record_sha256: [u8; 32],
    pub target_variable_id: String,
    pub control_target_value_micro: i64,
    pub treatment_target_value_micro: i64,
    pub control_branch_disposition: BranchDisposition,
    pub treatment_branch_disposition: BranchDisposition,
    pub control_resources: PairedBranchResourceUse,
    pub treatment_resources: PairedBranchResourceUse,
    pub control_true_outcome_micro: Option<i64>,
    pub treatment_true_outcome_micro: Option<i64>,
    pub control_predicted_outcome_mean_micro: Option<i64>,
    pub treatment_predicted_outcome_mean_micro: Option<i64>,
    pub predicted_effect_interval: Option<PredictedEffectInterval>,
    pub invariant_variables: Vec<InvariantVariableContrast>,
    pub disposition: PairDisposition,
    pub true_effect_micro: Option<i128>,
    pub predicted_effect_micro: Option<i128>,
    pub absolute_contrast_error_micro: Option<u128>,
    pub sign_eligible: bool,
    pub sign_correct: Option<bool>,
    pub predicted_interval_covered: Option<bool>,
    pub invariant_observation_count: u64,
    pub invariant_false_change_count: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PairedInterventionClusterSummary {
    pub family: InterventionFamily,
    pub environment_family_id: String,
    pub environment_cluster_id: String,
    pub pair_count: u64,
    pub completed_count: u64,
    pub failed_count: u64,
    pub timed_out_count: u64,
    pub invalid_count: u64,
    pub abstained_count: u64,
    pub assigned_penalty_sum_ppm: u128,
    pub contrast_error_sum_micro: u128,
    pub mean_contrast_error_micro: Option<u128>,
    pub sign_eligible_count: u64,
    pub sign_correct_count: u64,
    pub interval_count: u64,
    pub interval_covered_count: u64,
    pub invariant_observation_count: u64,
    pub invariant_false_change_count: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PairedInterventionFamilySummary {
    pub family: InterventionFamily,
    pub pair_count: u64,
    pub completed_count: u64,
    pub failed_count: u64,
    pub timed_out_count: u64,
    pub invalid_count: u64,
    pub abstained_count: u64,
    pub completed_cluster_count: u64,
    pub minimum_independent_clusters_required: u64,
    pub cluster_floor_met: bool,
    pub assigned_penalty_sum_ppm: u128,
    /// Task-pair-weighted diagnostic; not the primary family estimate.
    pub task_weighted_mean_contrast_error_micro: Option<u128>,
    /// Primary descriptive estimate: equal-weighted cluster means.
    pub cluster_balanced_mean_contrast_error_micro: Option<u128>,
    pub sign_eligible_pair_count: u64,
    pub sign_correct_pair_count: u64,
    /// Equal-weighted cluster sign-accuracy rate in PPM.
    pub cluster_balanced_sign_accuracy_ppm: Option<u32>,
    pub interval_pair_count: u64,
    pub interval_covered_pair_count: u64,
    /// Equal-weighted cluster interval-coverage rate in PPM.
    pub cluster_balanced_interval_coverage_ppm: Option<u32>,
    pub invariant_observation_count: u64,
    pub invariant_false_change_count: u64,
    /// Equal-weighted cluster false-change rate in PPM.
    pub cluster_balanced_false_change_rate_ppm: Option<u32>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PairedInterventionCertificate {
    pub context: PairedInterventionContext,
    pub pair_count: u64,
    /// Sorted by pair ID for deterministic replay.
    pub pair_results: Vec<PairedInterventionPairResult>,
    /// Sorted by intervention family then environment cluster.
    pub cluster_summaries: Vec<PairedInterventionClusterSummary>,
    /// Exactly eight rows, ordered as InterventionFamily::ALL.
    pub family_summaries: Vec<PairedInterventionFamilySummary>,
    pub total_branch_interaction_count: u64,
    pub total_model_query_count: u64,
    pub total_elapsed_us: u64,
    pub total_measured_energy_uj: Option<u128>,
    pub all_expected_pairs_present: bool,
    pub all_required_families_present: bool,
    pub all_family_cluster_floors_met: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PairedInterventionError {
    ZeroCampaignId,
    EmptyDigest {
        field: &'static str,
    },
    EmptyExpectedPairSet,
    TooManyPairs {
        requested: usize,
        maximum: usize,
    },
    ZeroMinimumIndependentClusters,
    MissingPairs {
        pair_ids: Vec<String>,
    },
    UnexpectedPair {
        pair_id: String,
    },
    DuplicateExpectedPairId {
        pair_id: String,
    },
    DuplicatePairId {
        pair_id: String,
    },
    DuplicateRunId {
        run_id: u64,
    },
    ZeroRunId {
        pair_id: String,
    },
    EmptyPairId,
    EmptyEnvironmentFamilyId {
        pair_id: String,
    },
    EmptyEnvironmentClusterId {
        pair_id: String,
    },
    EnvironmentClusterFamilyMismatch {
        cluster_id: String,
    },
    EmptyBranchId {
        pair_id: String,
    },
    DuplicateBranchId {
        branch_id: String,
    },
    EmptyTargetVariableId {
        pair_id: String,
    },
    TargetVariableDeclaredInvariant {
        pair_id: String,
    },
    TargetValueUnchanged {
        pair_id: String,
    },
    EmptyInvariantVariableId {
        pair_id: String,
    },
    DuplicateInvariantVariableId {
        pair_id: String,
        variable_id: String,
    },
    InvariantSetMismatch {
        pair_id: String,
    },
    EmptyProfileDigest {
        family: InterventionFamily,
    },
    MissingFamilyProfile {
        family: InterventionFamily,
    },
    DuplicateFamilyProfile {
        family: InterventionFamily,
    },
    PairProfileMismatch {
        pair_id: String,
        family: InterventionFamily,
    },
    EmptyInterventionRecordDigest {
        pair_id: String,
    },
    EmptySourceStateDigest {
        pair_id: String,
    },
    EmptyExogenousForcingDigest {
        pair_id: String,
    },
    EmptyBranchManifestDigest {
        pair_id: String,
    },
    BranchCheckpointMismatch {
        pair_id: String,
        branch_id: String,
    },
    BranchDispositionMismatch {
        pair_id: String,
    },
    PenaltyOutOfRange {
        pair_id: String,
        penalty_ppm: u32,
    },
    BranchOutcomeMissing {
        pair_id: String,
        branch_id: String,
    },
    NotExecutedBranchHasOutputOrCost {
        pair_id: String,
        branch_id: String,
    },
    PairIntervalMissing {
        pair_id: String,
    },
    PairIntervalInvalid {
        pair_id: String,
    },
    InvariantTruthViolation {
        pair_id: String,
        variable_id: String,
    },
    ResourceBudgetExceeded {
        field: &'static str,
    },
    MissingRequiredEnergyMeasurement {
        branch_id: String,
    },
    ArithmeticOverflow,
}

impl fmt::Display for PairedInterventionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroCampaignId => {
                write!(formatter, "paired-intervention campaign ID must be nonzero")
            }
            Self::EmptyDigest { field } => {
                write!(formatter, "required provenance digest {field} is empty")
            }
            Self::EmptyExpectedPairSet => write!(
                formatter,
                "paired-intervention expected pair set must not be empty"
            ),
            Self::TooManyPairs { requested, maximum } => write!(
                formatter,
                "paired-intervention pair count {requested} exceeds maximum {maximum}"
            ),
            Self::ZeroMinimumIndependentClusters => {
                write!(
                    formatter,
                    "minimum independent cluster count must be nonzero"
                )
            }
            Self::MissingPairs { pair_ids } => write!(
                formatter,
                "paired-intervention campaign is missing expected pairs {pair_ids:?}"
            ),
            Self::UnexpectedPair { pair_id } => write!(
                formatter,
                "paired-intervention campaign contains unexpected pair {pair_id}"
            ),
            Self::DuplicateExpectedPairId { pair_id } => {
                write!(formatter, "expected pair set repeats ID {pair_id}")
            }
            Self::DuplicatePairId { pair_id } => write!(
                formatter,
                "paired-intervention campaign repeats pair ID {pair_id}"
            ),
            Self::DuplicateRunId { run_id } => write!(
                formatter,
                "paired-intervention campaign repeats run ID {run_id}"
            ),
            Self::ZeroRunId { pair_id } => write!(
                formatter,
                "paired intervention pair {pair_id} has zero run ID"
            ),
            Self::EmptyPairId => write!(
                formatter,
                "paired intervention pair IDs must be nonempty and trimmed"
            ),
            Self::EmptyEnvironmentFamilyId { pair_id } => write!(
                formatter,
                "pair {pair_id} has an empty or untrimmed environment family ID"
            ),
            Self::EmptyEnvironmentClusterId { pair_id } => write!(
                formatter,
                "pair {pair_id} has an empty or untrimmed environment cluster ID"
            ),
            Self::EnvironmentClusterFamilyMismatch { cluster_id } => write!(
                formatter,
                "environment cluster {cluster_id} is assigned to multiple environment families"
            ),
            Self::EmptyBranchId { pair_id } => write!(
                formatter,
                "pair {pair_id} has an empty or untrimmed branch ID"
            ),
            Self::DuplicateBranchId { branch_id } => write!(
                formatter,
                "paired-intervention campaign repeats branch ID {branch_id}"
            ),
            Self::EmptyTargetVariableId { pair_id } => write!(
                formatter,
                "pair {pair_id} has an empty or untrimmed target variable ID"
            ),
            Self::TargetVariableDeclaredInvariant { pair_id } => write!(
                formatter,
                "pair {pair_id} intervention target is also declared invariant"
            ),
            Self::TargetValueUnchanged { pair_id } => write!(
                formatter,
                "pair {pair_id} does not change its declared intervention target"
            ),
            Self::EmptyInvariantVariableId { pair_id } => write!(
                formatter,
                "pair {pair_id} has an empty or untrimmed invariant variable ID"
            ),
            Self::DuplicateInvariantVariableId {
                pair_id,
                variable_id,
            } => write!(
                formatter,
                "pair {pair_id} repeats invariant variable {variable_id}"
            ),
            Self::InvariantSetMismatch { pair_id } => write!(
                formatter,
                "pair {pair_id} invariant variable set differs from the frozen expected set"
            ),
            Self::EmptyProfileDigest { family } => write!(
                formatter,
                "intervention profile digest is empty for {}",
                family.as_str()
            ),
            Self::MissingFamilyProfile { family } => write!(
                formatter,
                "paired-intervention campaign is missing profile for {}",
                family.as_str()
            ),
            Self::DuplicateFamilyProfile { family } => write!(
                formatter,
                "paired-intervention campaign repeats profile for {}",
                family.as_str()
            ),
            Self::PairProfileMismatch { pair_id, family } => write!(
                formatter,
                "pair {pair_id} does not use the frozen profile for {}",
                family.as_str()
            ),
            Self::EmptyInterventionRecordDigest { pair_id } => write!(
                formatter,
                "pair {pair_id} has an empty intervention record digest"
            ),
            Self::EmptySourceStateDigest { pair_id } => write!(
                formatter,
                "pair {pair_id} has an empty source-state snapshot digest"
            ),
            Self::EmptyExogenousForcingDigest { pair_id } => write!(
                formatter,
                "pair {pair_id} has an empty exogenous-forcing digest"
            ),
            Self::EmptyBranchManifestDigest { pair_id } => write!(
                formatter,
                "pair {pair_id} has an empty branch manifest digest"
            ),
            Self::BranchCheckpointMismatch { pair_id, branch_id } => write!(
                formatter,
                "pair {pair_id} branch {branch_id} uses a different source checkpoint"
            ),
            Self::BranchDispositionMismatch { pair_id } => write!(
                formatter,
                "pair {pair_id} disposition does not agree with its control/treatment branch dispositions"
            ),
            Self::PenaltyOutOfRange { pair_id, penalty_ppm } => write!(
                formatter,
                "pair {pair_id} has disposition penalty {penalty_ppm} above {PAIRED_INTERVENTION_SCORE_PPM}"
            ),
            Self::BranchOutcomeMissing { pair_id, branch_id } => write!(
                formatter,
                "completed branch {branch_id} in pair {pair_id} is missing a truth or prediction outcome"
            ),
            Self::NotExecutedBranchHasOutputOrCost { pair_id, branch_id } => write!(
                formatter,
                "not-executed branch {branch_id} in pair {pair_id} carries output or nonzero cost"
            ),
            Self::PairIntervalMissing { pair_id } => write!(
                formatter,
                "completed pair {pair_id} is missing a predicted effect interval"
            ),
            Self::PairIntervalInvalid { pair_id } => write!(
                formatter,
                "pair {pair_id} predicted effect interval has lower bound above upper bound"
            ),
            Self::InvariantTruthViolation {
                pair_id,
                variable_id,
            } => write!(
                formatter,
                "pair {pair_id} evaluator truth contradicts declared invariance for variable {variable_id}"
            ),
            Self::ResourceBudgetExceeded { field } => write!(
                formatter,
                "paired-intervention campaign exceeded declared budget {field}"
            ),
            Self::MissingRequiredEnergyMeasurement { branch_id } => write!(
                formatter,
                "energy budget is declared but branch {branch_id} has no energy measurement"
            ),
            Self::ArithmeticOverflow => {
                write!(formatter, "paired-intervention arithmetic overflowed")
            }
        }
    }
}

impl std::error::Error for PairedInterventionError {}

#[derive(Clone, Copy, Debug, Default)]
struct DispositionCounts {
    completed: u64,
    failed: u64,
    timed_out: u64,
    invalid: u64,
    abstained: u64,
    penalty_sum_ppm: u128,
}

impl DispositionCounts {
    fn add(&mut self, disposition: PairDisposition) -> Result<(), PairedInterventionError> {
        let (counter, penalty) = match disposition {
            PairDisposition::Completed => (&mut self.completed, 0),
            PairDisposition::Failed { penalty_ppm } => (&mut self.failed, penalty_ppm),
            PairDisposition::TimedOut { penalty_ppm } => (&mut self.timed_out, penalty_ppm),
            PairDisposition::Invalid { penalty_ppm } => (&mut self.invalid, penalty_ppm),
            PairDisposition::Abstained { penalty_ppm } => (&mut self.abstained, penalty_ppm),
        };
        *counter = counter
            .checked_add(1)
            .ok_or(PairedInterventionError::ArithmeticOverflow)?;
        self.penalty_sum_ppm = self
            .penalty_sum_ppm
            .checked_add(u128::from(penalty))
            .ok_or(PairedInterventionError::ArithmeticOverflow)?;
        Ok(())
    }
}

#[derive(Clone, Debug, Default)]
struct ClusterAccumulator {
    pair_count: u64,
    dispositions: DispositionCounts,
    completed_count: u64,
    contrast_error_sum_micro: u128,
    sign_eligible_count: u64,
    sign_correct_count: u64,
    interval_count: u64,
    interval_covered_count: u64,
    invariant_observation_count: u64,
    invariant_false_change_count: u64,
}

#[derive(Clone, Debug, Default)]
struct FamilyAccumulator {
    pair_count: u64,
    dispositions: DispositionCounts,
    task_contrast_error_sum_micro: u128,
    sign_eligible_count: u64,
    sign_correct_count: u64,
    interval_count: u64,
    interval_covered_count: u64,
    invariant_observation_count: u64,
    invariant_false_change_count: u64,
    clusters: BTreeMap<String, ClusterAccumulator>,
}

fn digest_is_empty(digest: &[u8; 32]) -> bool {
    digest.iter().all(|byte| *byte == 0)
}

fn trimmed_nonempty(value: &str) -> bool {
    !value.trim().is_empty() && value.trim() == value
}

fn abs_i128(value: i128) -> u128 {
    value.unsigned_abs()
}

fn effect_sign(value: i128) -> i8 {
    match value.cmp(&0) {
        std::cmp::Ordering::Greater => 1,
        std::cmp::Ordering::Less => -1,
        std::cmp::Ordering::Equal => 0,
    }
}

fn pair_disposition_from_branches(
    control: BranchDisposition,
    treatment: BranchDisposition,
) -> PairDisposition {
    if control == BranchDisposition::Failed || treatment == BranchDisposition::Failed {
        PairDisposition::Failed { penalty_ppm: 0 }
    } else if control == BranchDisposition::TimedOut || treatment == BranchDisposition::TimedOut {
        PairDisposition::TimedOut { penalty_ppm: 0 }
    } else if control == BranchDisposition::Invalid || treatment == BranchDisposition::Invalid {
        PairDisposition::Invalid { penalty_ppm: 0 }
    } else if control == BranchDisposition::NotExecuted
        || treatment == BranchDisposition::NotExecuted
    {
        PairDisposition::Abstained { penalty_ppm: 0 }
    } else {
        PairDisposition::Completed
    }
}

fn validate_context(
    context: &PairedInterventionContext,
) -> Result<BTreeMap<InterventionFamily, [u8; 32]>, PairedInterventionError> {
    if context.campaign_id == 0 {
        return Err(PairedInterventionError::ZeroCampaignId);
    }
    for (digest, field) in [
        (
            &context.candidate_artifact_sha256,
            "candidate_artifact_sha256",
        ),
        (&context.model_checkpoint_sha256, "model_checkpoint_sha256"),
        (&context.source_history_sha256, "source_history_sha256"),
        (
            &context.environment_generator_sha256,
            "environment_generator_sha256",
        ),
        (
            &context.intervention_design_sha256,
            "intervention_design_sha256",
        ),
        (
            &context.evaluator_profile_sha256,
            "evaluator_profile_sha256",
        ),
        (&context.seed_manifest_sha256, "seed_manifest_sha256"),
        (&context.evaluation_split_sha256, "evaluation_split_sha256"),
    ] {
        if digest_is_empty(digest) {
            return Err(PairedInterventionError::EmptyDigest { field });
        }
    }
    if context.minimum_independent_clusters_per_family == 0 {
        return Err(PairedInterventionError::ZeroMinimumIndependentClusters);
    }

    let mut invariant_ids = BTreeSet::new();
    for variable_id in &context.expected_invariant_variable_ids {
        if !trimmed_nonempty(variable_id) {
            return Err(PairedInterventionError::EmptyInvariantVariableId {
                pair_id: "<context>".to_owned(),
            });
        }
        if !invariant_ids.insert(variable_id.clone()) {
            return Err(PairedInterventionError::DuplicateInvariantVariableId {
                pair_id: "<context>".to_owned(),
                variable_id: variable_id.clone(),
            });
        }
    }
    if invariant_ids.is_empty() {
        return Err(PairedInterventionError::InvariantSetMismatch {
            pair_id: "<context>".to_owned(),
        });
    }

    let mut profiles = BTreeMap::new();
    for profile in &context.family_profiles {
        if digest_is_empty(&profile.profile_manifest_sha256) {
            return Err(PairedInterventionError::EmptyProfileDigest {
                family: profile.family,
            });
        }
        if profiles
            .insert(profile.family, profile.profile_manifest_sha256)
            .is_some()
        {
            return Err(PairedInterventionError::DuplicateFamilyProfile {
                family: profile.family,
            });
        }
    }
    for family in InterventionFamily::ALL {
        if !profiles.contains_key(&family) {
            return Err(PairedInterventionError::MissingFamilyProfile { family });
        }
    }
    Ok(profiles)
}

fn validate_branch(
    branch: &PairedInterventionBranch,
    pair_id: &str,
    expected_checkpoint: &[u8; 32],
) -> Result<(), PairedInterventionError> {
    if !trimmed_nonempty(&branch.branch_id) {
        return Err(PairedInterventionError::EmptyBranchId {
            pair_id: pair_id.to_owned(),
        });
    }
    if digest_is_empty(&branch.branch_manifest_sha256) {
        return Err(PairedInterventionError::EmptyBranchManifestDigest {
            pair_id: pair_id.to_owned(),
        });
    }
    if &branch.source_checkpoint_sha256 != expected_checkpoint {
        return Err(PairedInterventionError::BranchCheckpointMismatch {
            pair_id: pair_id.to_owned(),
            branch_id: branch.branch_id.clone(),
        });
    }
    if branch.disposition == BranchDisposition::Completed
        && (branch.true_outcome_micro.is_none() || branch.predicted_outcome_mean_micro.is_none())
    {
        return Err(PairedInterventionError::BranchOutcomeMissing {
            pair_id: pair_id.to_owned(),
            branch_id: branch.branch_id.clone(),
        });
    }
    if branch.disposition == BranchDisposition::NotExecuted
        && (branch.true_outcome_micro.is_some()
            || branch.predicted_outcome_mean_micro.is_some()
            || branch.resources.interaction_count != 0
            || branch.resources.model_query_count != 0
            || branch.resources.elapsed_us != 0
            || branch
                .resources
                .measured_energy_uj
                .is_some_and(|energy| energy != 0))
    {
        return Err(PairedInterventionError::NotExecutedBranchHasOutputOrCost {
            pair_id: pair_id.to_owned(),
            branch_id: branch.branch_id.clone(),
        });
    }
    Ok(())
}

fn increment(target: &mut u64) -> Result<(), PairedInterventionError> {
    *target = target
        .checked_add(1)
        .ok_or(PairedInterventionError::ArithmeticOverflow)?;
    Ok(())
}

fn add_u64(target: &mut u64, value: u64) -> Result<(), PairedInterventionError> {
    *target = target
        .checked_add(value)
        .ok_or(PairedInterventionError::ArithmeticOverflow)?;
    Ok(())
}

fn add_energy(
    total: &mut Option<u128>,
    measurement: Option<u64>,
) -> Result<(), PairedInterventionError> {
    if let (Some(sum), Some(value)) = (total.as_mut(), measurement) {
        *sum = sum
            .checked_add(u128::from(value))
            .ok_or(PairedInterventionError::ArithmeticOverflow)?;
    }
    Ok(())
}

pub fn evaluate_paired_interventions(
    input: &PairedInterventionInput,
) -> Result<PairedInterventionCertificate, PairedInterventionError> {
    let context = &input.context;
    let profiles = validate_context(context)?;
    if input.expected_pair_ids.is_empty() {
        return Err(PairedInterventionError::EmptyExpectedPairSet);
    }
    if input.pairs.len() > PAIRED_INTERVENTION_MAX_PAIRS
        || input.pairs.len() > context.resource_budget.max_pairs as usize
    {
        return Err(PairedInterventionError::TooManyPairs {
            requested: input.pairs.len(),
            maximum: PAIRED_INTERVENTION_MAX_PAIRS.min(context.resource_budget.max_pairs as usize),
        });
    }

    let mut expected_ids = BTreeSet::new();
    for pair_id in &input.expected_pair_ids {
        if !trimmed_nonempty(pair_id) {
            return Err(PairedInterventionError::EmptyPairId);
        }
        if !expected_ids.insert(pair_id.clone()) {
            return Err(PairedInterventionError::DuplicateExpectedPairId {
                pair_id: pair_id.clone(),
            });
        }
    }

    let mut pair_ids = BTreeSet::new();
    let mut run_ids = BTreeSet::new();
    let mut branch_ids = BTreeSet::new();
    let mut cluster_families = BTreeMap::<String, String>::new();
    let mut families = InterventionFamily::ALL
        .into_iter()
        .map(|family| (family, FamilyAccumulator::default()))
        .collect::<BTreeMap<_, _>>();
    let mut results = Vec::with_capacity(input.pairs.len());
    let mut total_branch_interactions = 0_u64;
    let mut total_model_queries = 0_u64;
    let mut total_elapsed_us = 0_u64;
    let mut total_energy_uj = Some(0_u128);
    let mut energy_branch_count = 0_u64;

    for pair in &input.pairs {
        if !trimmed_nonempty(&pair.pair_id) {
            return Err(PairedInterventionError::EmptyPairId);
        }
        if !pair_ids.insert(pair.pair_id.clone()) {
            return Err(PairedInterventionError::DuplicatePairId {
                pair_id: pair.pair_id.clone(),
            });
        }
        if !expected_ids.contains(&pair.pair_id) {
            return Err(PairedInterventionError::UnexpectedPair {
                pair_id: pair.pair_id.clone(),
            });
        }
        if pair.run_id == 0 {
            return Err(PairedInterventionError::ZeroRunId {
                pair_id: pair.pair_id.clone(),
            });
        }
        if !run_ids.insert(pair.run_id) {
            return Err(PairedInterventionError::DuplicateRunId {
                run_id: pair.run_id,
            });
        }
        if !trimmed_nonempty(&pair.environment_family_id) {
            return Err(PairedInterventionError::EmptyEnvironmentFamilyId {
                pair_id: pair.pair_id.clone(),
            });
        }
        if !trimmed_nonempty(&pair.environment_cluster_id) {
            return Err(PairedInterventionError::EmptyEnvironmentClusterId {
                pair_id: pair.pair_id.clone(),
            });
        }
        if cluster_families
            .insert(
                pair.environment_cluster_id.clone(),
                pair.environment_family_id.clone(),
            )
            .is_some_and(|existing| existing != pair.environment_family_id)
        {
            return Err(PairedInterventionError::EnvironmentClusterFamilyMismatch {
                cluster_id: pair.environment_cluster_id.clone(),
            });
        }
        if !trimmed_nonempty(&pair.target_variable_id) {
            return Err(PairedInterventionError::EmptyTargetVariableId {
                pair_id: pair.pair_id.clone(),
            });
        }
        if context
            .expected_invariant_variable_ids
            .contains(&pair.target_variable_id)
        {
            return Err(PairedInterventionError::TargetVariableDeclaredInvariant {
                pair_id: pair.pair_id.clone(),
            });
        }
        if pair.control_target_value_micro == pair.treatment_target_value_micro {
            return Err(PairedInterventionError::TargetValueUnchanged {
                pair_id: pair.pair_id.clone(),
            });
        }
        let expected_profile =
            profiles
                .get(&pair.family)
                .ok_or(PairedInterventionError::MissingFamilyProfile {
                    family: pair.family,
                })?;
        if expected_profile != &pair.intervention_profile_sha256 {
            return Err(PairedInterventionError::PairProfileMismatch {
                pair_id: pair.pair_id.clone(),
                family: pair.family,
            });
        }
        for (digest, error) in [
            (
                &pair.intervention_record_sha256,
                PairedInterventionError::EmptyInterventionRecordDigest {
                    pair_id: pair.pair_id.clone(),
                },
            ),
            (
                &pair.source_state_snapshot_sha256,
                PairedInterventionError::EmptySourceStateDigest {
                    pair_id: pair.pair_id.clone(),
                },
            ),
            (
                &pair.exogenous_forcing_sha256,
                PairedInterventionError::EmptyExogenousForcingDigest {
                    pair_id: pair.pair_id.clone(),
                },
            ),
        ] {
            if digest_is_empty(digest) {
                return Err(error);
            }
        }

        validate_branch(
            &pair.control,
            &pair.pair_id,
            &context.model_checkpoint_sha256,
        )?;
        validate_branch(
            &pair.treatment,
            &pair.pair_id,
            &context.model_checkpoint_sha256,
        )?;
        if pair.control.branch_id == pair.treatment.branch_id
            || !branch_ids.insert(pair.control.branch_id.clone())
            || !branch_ids.insert(pair.treatment.branch_id.clone())
        {
            return Err(PairedInterventionError::DuplicateBranchId {
                branch_id: pair.control.branch_id.clone(),
            });
        }
        let expected_disposition =
            pair_disposition_from_branches(pair.control.disposition, pair.treatment.disposition);
        let disposition_matches = matches!(
            (expected_disposition, pair.disposition),
            (PairDisposition::Completed, PairDisposition::Completed)
                | (
                    PairDisposition::Failed { .. },
                    PairDisposition::Failed { .. },
                )
                | (
                    PairDisposition::TimedOut { .. },
                    PairDisposition::TimedOut { .. },
                )
                | (
                    PairDisposition::Invalid { .. },
                    PairDisposition::Invalid { .. },
                )
                | (
                    PairDisposition::Abstained { .. },
                    PairDisposition::Abstained { .. },
                )
        );
        if !disposition_matches {
            return Err(PairedInterventionError::BranchDispositionMismatch {
                pair_id: pair.pair_id.clone(),
            });
        }
        if let PairDisposition::Failed { penalty_ppm }
        | PairDisposition::TimedOut { penalty_ppm }
        | PairDisposition::Invalid { penalty_ppm }
        | PairDisposition::Abstained { penalty_ppm } = pair.disposition
        {
            if penalty_ppm > PAIRED_INTERVENTION_SCORE_PPM {
                return Err(PairedInterventionError::PenaltyOutOfRange {
                    pair_id: pair.pair_id.clone(),
                    penalty_ppm,
                });
            }
        }

        let mut family_invariant_ids = BTreeSet::new();
        let mut invariant_false_change_count = 0_u64;
        let mut pair_result = PairedInterventionPairResult {
            pair_id: pair.pair_id.clone(),
            run_id: pair.run_id,
            family: pair.family,
            environment_family_id: pair.environment_family_id.clone(),
            environment_cluster_id: pair.environment_cluster_id.clone(),
            control_branch_id: pair.control.branch_id.clone(),
            treatment_branch_id: pair.treatment.branch_id.clone(),
            control_branch_manifest_sha256: pair.control.branch_manifest_sha256,
            treatment_branch_manifest_sha256: pair.treatment.branch_manifest_sha256,
            source_state_snapshot_sha256: pair.source_state_snapshot_sha256,
            exogenous_forcing_sha256: pair.exogenous_forcing_sha256,
            intervention_profile_sha256: pair.intervention_profile_sha256,
            intervention_record_sha256: pair.intervention_record_sha256,
            target_variable_id: pair.target_variable_id.clone(),
            control_target_value_micro: pair.control_target_value_micro,
            treatment_target_value_micro: pair.treatment_target_value_micro,
            control_branch_disposition: pair.control.disposition,
            treatment_branch_disposition: pair.treatment.disposition,
            control_resources: pair.control.resources,
            treatment_resources: pair.treatment.resources,
            control_true_outcome_micro: pair.control.true_outcome_micro,
            treatment_true_outcome_micro: pair.treatment.true_outcome_micro,
            control_predicted_outcome_mean_micro: pair.control.predicted_outcome_mean_micro,
            treatment_predicted_outcome_mean_micro: pair.treatment.predicted_outcome_mean_micro,
            predicted_effect_interval: pair.predicted_effect_interval,
            invariant_variables: pair.invariant_variables.clone(),
            disposition: pair.disposition,
            true_effect_micro: None,
            predicted_effect_micro: None,
            absolute_contrast_error_micro: None,
            sign_eligible: false,
            sign_correct: None,
            predicted_interval_covered: None,
            invariant_observation_count: 0,
            invariant_false_change_count: 0,
        };

        if pair.disposition == PairDisposition::Completed {
            let control_true = i128::from(pair.control.true_outcome_micro.ok_or(
                PairedInterventionError::BranchOutcomeMissing {
                    pair_id: pair.pair_id.clone(),
                    branch_id: pair.control.branch_id.clone(),
                },
            )?);
            let treatment_true = i128::from(pair.treatment.true_outcome_micro.ok_or(
                PairedInterventionError::BranchOutcomeMissing {
                    pair_id: pair.pair_id.clone(),
                    branch_id: pair.treatment.branch_id.clone(),
                },
            )?);
            let control_predicted = i128::from(pair.control.predicted_outcome_mean_micro.ok_or(
                PairedInterventionError::BranchOutcomeMissing {
                    pair_id: pair.pair_id.clone(),
                    branch_id: pair.control.branch_id.clone(),
                },
            )?);
            let treatment_predicted =
                i128::from(pair.treatment.predicted_outcome_mean_micro.ok_or(
                    PairedInterventionError::BranchOutcomeMissing {
                        pair_id: pair.pair_id.clone(),
                        branch_id: pair.treatment.branch_id.clone(),
                    },
                )?);
            let true_effect = treatment_true - control_true;
            let predicted_effect = treatment_predicted - control_predicted;
            let error = abs_i128(predicted_effect - true_effect);
            let interval = pair.predicted_effect_interval.ok_or(
                PairedInterventionError::PairIntervalMissing {
                    pair_id: pair.pair_id.clone(),
                },
            )?;
            if interval.lower_micro > interval.upper_micro {
                return Err(PairedInterventionError::PairIntervalInvalid {
                    pair_id: pair.pair_id.clone(),
                });
            }
            let sign_eligible =
                abs_i128(true_effect) > u128::from(context.effect_resolution_floor_micro);
            let sign_correct =
                sign_eligible.then(|| effect_sign(predicted_effect) == effect_sign(true_effect));
            let covered = true_effect >= i128::from(interval.lower_micro)
                && true_effect <= i128::from(interval.upper_micro);

            let expected_invariant_ids = context
                .expected_invariant_variable_ids
                .iter()
                .cloned()
                .collect::<BTreeSet<_>>();
            for invariant in &pair.invariant_variables {
                if !trimmed_nonempty(&invariant.variable_id) {
                    return Err(PairedInterventionError::EmptyInvariantVariableId {
                        pair_id: pair.pair_id.clone(),
                    });
                }
                if !family_invariant_ids.insert(invariant.variable_id.clone()) {
                    return Err(PairedInterventionError::DuplicateInvariantVariableId {
                        pair_id: pair.pair_id.clone(),
                        variable_id: invariant.variable_id.clone(),
                    });
                }
                let true_delta = i128::from(invariant.treatment_true_micro)
                    - i128::from(invariant.control_true_micro);
                if abs_i128(true_delta) > u128::from(context.true_invariance_tolerance_micro) {
                    return Err(PairedInterventionError::InvariantTruthViolation {
                        pair_id: pair.pair_id.clone(),
                        variable_id: invariant.variable_id.clone(),
                    });
                }
                let predicted_delta = i128::from(invariant.treatment_predicted_micro)
                    - i128::from(invariant.control_predicted_micro);
                if abs_i128(predicted_delta)
                    > u128::from(context.predicted_invariance_tolerance_micro)
                {
                    increment(&mut invariant_false_change_count)?;
                }
            }
            if family_invariant_ids != expected_invariant_ids {
                return Err(PairedInterventionError::InvariantSetMismatch {
                    pair_id: pair.pair_id.clone(),
                });
            }

            pair_result.true_effect_micro = Some(true_effect);
            pair_result.predicted_effect_micro = Some(predicted_effect);
            pair_result.absolute_contrast_error_micro = Some(error);
            pair_result.sign_eligible = sign_eligible;
            pair_result.sign_correct = sign_correct;
            pair_result.predicted_interval_covered = Some(covered);
            pair_result.invariant_observation_count = u64::try_from(pair.invariant_variables.len())
                .map_err(|_| PairedInterventionError::ArithmeticOverflow)?;
            pair_result.invariant_false_change_count = invariant_false_change_count;
        } else {
            if !pair.invariant_variables.is_empty() || pair.predicted_effect_interval.is_some() {
                return Err(PairedInterventionError::InvariantSetMismatch {
                    pair_id: pair.pair_id.clone(),
                });
            }
        }

        let family = families.get_mut(&pair.family).ok_or(
            PairedInterventionError::MissingFamilyProfile {
                family: pair.family,
            },
        )?;
        increment(&mut family.pair_count)?;
        family.dispositions.add(pair.disposition)?;
        let cluster = family
            .clusters
            .entry(pair.environment_cluster_id.clone())
            .or_default();
        increment(&mut cluster.pair_count)?;
        cluster.dispositions.add(pair.disposition)?;

        if let Some(error) = pair_result.absolute_contrast_error_micro {
            add_u64(
                &mut family.sign_eligible_count,
                u64::from(pair_result.sign_eligible),
            )?;
            if pair_result.sign_correct == Some(true) {
                increment(&mut family.sign_correct_count)?;
            }
            if pair_result.predicted_interval_covered == Some(true) {
                increment(&mut family.interval_covered_count)?;
            }
            increment(&mut family.interval_count)?;
            family.task_contrast_error_sum_micro = family
                .task_contrast_error_sum_micro
                .checked_add(error)
                .ok_or(PairedInterventionError::ArithmeticOverflow)?;
            add_u64(
                &mut family.invariant_observation_count,
                pair_result.invariant_observation_count,
            )?;
            add_u64(
                &mut family.invariant_false_change_count,
                pair_result.invariant_false_change_count,
            )?;

            increment(&mut cluster.completed_count)?;
            cluster.contrast_error_sum_micro = cluster
                .contrast_error_sum_micro
                .checked_add(error)
                .ok_or(PairedInterventionError::ArithmeticOverflow)?;
            add_u64(
                &mut cluster.sign_eligible_count,
                u64::from(pair_result.sign_eligible),
            )?;
            if pair_result.sign_correct == Some(true) {
                increment(&mut cluster.sign_correct_count)?;
            }
            increment(&mut cluster.interval_count)?;
            if pair_result.predicted_interval_covered == Some(true) {
                increment(&mut cluster.interval_covered_count)?;
            }
            add_u64(
                &mut cluster.invariant_observation_count,
                pair_result.invariant_observation_count,
            )?;
            add_u64(
                &mut cluster.invariant_false_change_count,
                pair_result.invariant_false_change_count,
            )?;
        }

        for branch in [&pair.control, &pair.treatment] {
            add_u64(
                &mut total_branch_interactions,
                branch.resources.interaction_count,
            )?;
            add_u64(&mut total_model_queries, branch.resources.model_query_count)?;
            add_u64(&mut total_elapsed_us, branch.resources.elapsed_us)?;
            if let Some(energy) = branch.resources.measured_energy_uj {
                add_energy(&mut total_energy_uj, Some(energy))?;
                increment(&mut energy_branch_count)?;
            } else {
                total_energy_uj = None;
            }
        }

        results.push(pair_result);
    }

    let missing = expected_ids
        .difference(&pair_ids)
        .cloned()
        .collect::<Vec<_>>();
    if !missing.is_empty() {
        return Err(PairedInterventionError::MissingPairs { pair_ids: missing });
    }

    let mut family_summaries = Vec::with_capacity(PAIRED_INTERVENTION_FAMILY_COUNT);
    let mut cluster_summaries = Vec::new();
    for family in InterventionFamily::ALL {
        let accumulator = families
            .get(&family)
            .ok_or(PairedInterventionError::MissingFamilyProfile { family })?;
        let completed_clusters = accumulator
            .clusters
            .values()
            .filter(|cluster| cluster.completed_count > 0)
            .count();
        let completed_cluster_count = u64::try_from(completed_clusters)
            .map_err(|_| PairedInterventionError::ArithmeticOverflow)?;
        let cluster_floor_met =
            completed_cluster_count >= context.minimum_independent_clusters_per_family;

        let mut cluster_error_means = Vec::new();
        let mut cluster_sign_rates = Vec::new();
        let mut cluster_interval_rates = Vec::new();
        let mut cluster_false_change_rates = Vec::new();
        for (cluster_id, cluster) in &accumulator.clusters {
            let mean_error = if cluster.completed_count > 0 {
                let mean = cluster.contrast_error_sum_micro / u128::from(cluster.completed_count);
                cluster_error_means.push(mean);
                Some(mean)
            } else {
                None
            };
            if cluster.sign_eligible_count > 0 {
                cluster_sign_rates.push(
                    u128::from(cluster.sign_correct_count)
                        * u128::from(PAIRED_INTERVENTION_SCORE_PPM)
                        / u128::from(cluster.sign_eligible_count),
                );
            }
            if cluster.interval_count > 0 {
                cluster_interval_rates.push(
                    u128::from(cluster.interval_covered_count)
                        * u128::from(PAIRED_INTERVENTION_SCORE_PPM)
                        / u128::from(cluster.interval_count),
                );
            }
            if cluster.invariant_observation_count > 0 {
                cluster_false_change_rates.push(
                    u128::from(cluster.invariant_false_change_count)
                        * u128::from(PAIRED_INTERVENTION_SCORE_PPM)
                        / u128::from(cluster.invariant_observation_count),
                );
            }
            cluster_summaries.push(PairedInterventionClusterSummary {
                family,
                environment_family_id: cluster_families[cluster_id].clone(),
                environment_cluster_id: cluster_id.clone(),
                pair_count: cluster.pair_count,
                completed_count: cluster.completed_count,
                failed_count: cluster.dispositions.failed,
                timed_out_count: cluster.dispositions.timed_out,
                invalid_count: cluster.dispositions.invalid,
                abstained_count: cluster.dispositions.abstained,
                assigned_penalty_sum_ppm: cluster.dispositions.penalty_sum_ppm,
                contrast_error_sum_micro: cluster.contrast_error_sum_micro,
                mean_contrast_error_micro: mean_error,
                sign_eligible_count: cluster.sign_eligible_count,
                sign_correct_count: cluster.sign_correct_count,
                interval_count: cluster.interval_count,
                interval_covered_count: cluster.interval_covered_count,
                invariant_observation_count: cluster.invariant_observation_count,
                invariant_false_change_count: cluster.invariant_false_change_count,
            });
        }
        let balanced_mean = if cluster_error_means.is_empty() {
            None
        } else {
            Some(cluster_error_means.iter().sum::<u128>() / cluster_error_means.len() as u128)
        };
        let task_weighted_mean = if accumulator.dispositions.completed > 0 {
            Some(
                accumulator.task_contrast_error_sum_micro
                    / u128::from(accumulator.dispositions.completed),
            )
        } else {
            None
        };
        let avg_rates = |values: &[u128]| -> Result<Option<u32>, PairedInterventionError> {
            if values.is_empty() {
                Ok(None)
            } else {
                let sum = values.iter().try_fold(0_u128, |acc, value| {
                    acc.checked_add(*value)
                        .ok_or(PairedInterventionError::ArithmeticOverflow)
                })?;
                Ok(Some(u32::try_from(sum / values.len() as u128).map_err(
                    |_| PairedInterventionError::ArithmeticOverflow,
                )?))
            }
        };

        family_summaries.push(PairedInterventionFamilySummary {
            family,
            pair_count: accumulator.pair_count,
            completed_count: accumulator.dispositions.completed,
            failed_count: accumulator.dispositions.failed,
            timed_out_count: accumulator.dispositions.timed_out,
            invalid_count: accumulator.dispositions.invalid,
            abstained_count: accumulator.dispositions.abstained,
            completed_cluster_count,
            minimum_independent_clusters_required: context.minimum_independent_clusters_per_family,
            cluster_floor_met,
            assigned_penalty_sum_ppm: accumulator.dispositions.penalty_sum_ppm,
            task_weighted_mean_contrast_error_micro: task_weighted_mean,
            cluster_balanced_mean_contrast_error_micro: balanced_mean,
            sign_eligible_pair_count: accumulator.sign_eligible_count,
            sign_correct_pair_count: accumulator.sign_correct_count,
            cluster_balanced_sign_accuracy_ppm: avg_rates(&cluster_sign_rates)?,
            interval_pair_count: accumulator.interval_count,
            interval_covered_pair_count: accumulator.interval_covered_count,
            cluster_balanced_interval_coverage_ppm: avg_rates(&cluster_interval_rates)?,
            invariant_observation_count: accumulator.invariant_observation_count,
            invariant_false_change_count: accumulator.invariant_false_change_count,
            cluster_balanced_false_change_rate_ppm: avg_rates(&cluster_false_change_rates)?,
        });
    }

    if total_branch_interactions > context.resource_budget.max_total_branch_interactions {
        return Err(PairedInterventionError::ResourceBudgetExceeded {
            field: "max_total_branch_interactions",
        });
    }
    if total_model_queries > context.resource_budget.max_total_model_queries {
        return Err(PairedInterventionError::ResourceBudgetExceeded {
            field: "max_total_model_queries",
        });
    }
    if total_elapsed_us > context.resource_budget.max_total_elapsed_us {
        return Err(PairedInterventionError::ResourceBudgetExceeded {
            field: "max_total_elapsed_us",
        });
    }
    if let Some(maximum) = context.resource_budget.max_total_energy_uj {
        if energy_branch_count != u64::try_from(input.pairs.len()).unwrap_or(u64::MAX) * 2 {
            return Err(PairedInterventionError::MissingRequiredEnergyMeasurement {
                branch_id: "<campaign>".to_owned(),
            });
        }
        let measured = total_energy_uj.unwrap_or(0);
        if measured > u128::from(maximum) {
            return Err(PairedInterventionError::ResourceBudgetExceeded {
                field: "max_total_energy_uj",
            });
        }
    }

    results.sort_by(|left, right| left.pair_id.cmp(&right.pair_id));
    let all_required_families_present =
        family_summaries.iter().all(|summary| summary.pair_count > 0);
    let all_family_cluster_floors_met =
        family_summaries.iter().all(|summary| summary.cluster_floor_met);
    Ok(PairedInterventionCertificate {
        context: context.clone(),
        pair_count: u64::try_from(results.len())
            .map_err(|_| PairedInterventionError::ArithmeticOverflow)?,
        pair_results: results,
        cluster_summaries,
        family_summaries,
        total_branch_interaction_count: total_branch_interactions,
        total_model_query_count: total_model_queries,
        total_elapsed_us,
        total_measured_energy_uj: total_energy_uj,
        all_expected_pairs_present: true,
        all_required_families_present,
        all_family_cluster_floors_met,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(value: u8) -> [u8; 32] {
        [value; 32]
    }

    fn context() -> PairedInterventionContext {
        PairedInterventionContext {
            campaign_id: 8101,
            candidate_artifact_sha256: digest(1),
            model_checkpoint_sha256: digest(2),
            source_history_sha256: digest(3),
            environment_generator_sha256: digest(4),
            intervention_design_sha256: digest(5),
            evaluator_profile_sha256: digest(6),
            seed_manifest_sha256: digest(7),
            evaluation_split_sha256: digest(8),
            expected_invariant_variable_ids: vec!["color".to_owned()],
            minimum_independent_clusters_per_family: 2,
            effect_resolution_floor_micro: 50,
            true_invariance_tolerance_micro: 0,
            predicted_invariance_tolerance_micro: 0,
            resource_budget: PairedInterventionResourceBudget {
                max_pairs: 1_000,
                max_total_branch_interactions: 10_000,
                max_total_model_queries: 10_000,
                max_total_elapsed_us: 10_000,
                max_total_energy_uj: Some(10_000),
            },
            family_profiles: InterventionFamily::ALL
                .into_iter()
                .enumerate()
                .map(|(index, family)| InterventionFamilyProfile {
                    family,
                    profile_manifest_sha256: digest(20 + index as u8),
                })
                .collect(),
        }
    }

    fn pair(
        pair_number: u64,
        family: InterventionFamily,
        cluster_number: u8,
        effect_micro: i64,
    ) -> PairedInterventionPair {
        let pair_id = format!("{}-pair-{pair_number}", family.as_str());
        let base = 10_000_i64;
        let treatment_true = base + effect_micro;
        let target = 100_i64;
        PairedInterventionPair {
            pair_id: pair_id.clone(),
            run_id: pair_number,
            family,
            environment_family_id: "world-generator-v1".to_owned(),
            environment_cluster_id: format!("{}-cluster-{cluster_number}", family.as_str()),
            intervention_profile_sha256: digest(
                20 + InterventionFamily::ALL
                    .iter()
                    .position(|x| *x == family)
                    .unwrap() as u8,
            ),
            intervention_record_sha256: digest(40 + pair_number as u8),
            source_state_snapshot_sha256: digest(5 + cluster_number),
            exogenous_forcing_sha256: digest(70 + cluster_number),
            target_variable_id: "mass".to_owned(),
            control_target_value_micro: target,
            treatment_target_value_micro: target + 10,
            control: PairedInterventionBranch {
                branch_id: format!("{pair_id}-control"),
                branch_manifest_sha256: digest(90 + pair_number as u8),
                source_checkpoint_sha256: digest(2),
                disposition: BranchDisposition::Completed,
                true_outcome_micro: Some(base),
                predicted_outcome_mean_micro: Some(base),
                resources: PairedBranchResourceUse {
                    interaction_count: 1,
                    model_query_count: 2,
                    elapsed_us: 3,
                    measured_energy_uj: Some(5),
                },
            },
            treatment: PairedInterventionBranch {
                branch_id: format!("{pair_id}-treatment"),
                branch_manifest_sha256: digest(120 + pair_number as u8),
                source_checkpoint_sha256: digest(2),
                disposition: BranchDisposition::Completed,
                true_outcome_micro: Some(treatment_true),
                predicted_outcome_mean_micro: Some(treatment_true),
                resources: PairedBranchResourceUse {
                    interaction_count: 1,
                    model_query_count: 2,
                    elapsed_us: 3,
                    measured_energy_uj: Some(5),
                },
            },
            predicted_effect_interval: Some(PredictedEffectInterval {
                lower_micro: effect_micro - 10,
                upper_micro: effect_micro + 10,
            }),
            invariant_variables: vec![InvariantVariableContrast {
                variable_id: "color".to_owned(),
                control_true_micro: 5,
                treatment_true_micro: 5,
                control_predicted_micro: 5,
                treatment_predicted_micro: 5,
            }],
            disposition: PairDisposition::Completed,
        }
    }

    fn base_input() -> PairedInterventionInput {
        let mut pairs = Vec::new();
        let mut expected_pair_ids = Vec::new();
        let mut run_id = 1_u64;
        for family in InterventionFamily::ALL {
            for cluster in 1..=2 {
                let effect = if family == InterventionFamily::NullInvariance {
                    0
                } else {
                    1_000
                };
                let row = pair(run_id, family, cluster, effect);
                expected_pair_ids.push(row.pair_id.clone());
                pairs.push(row);
                run_id += 1;
            }
        }
        PairedInterventionInput {
            context: context(),
            expected_pair_ids,
            pairs,
        }
    }

    #[test]
    fn emits_eight_family_summaries_with_paired_contrast_metrics() {
        let result = evaluate_paired_interventions(&base_input()).unwrap();
        assert_eq!(
            result.family_summaries.len(),
            PAIRED_INTERVENTION_FAMILY_COUNT
        );
        assert_eq!(result.pair_count, 16);
        assert!(result.all_expected_pairs_present);
        assert!(result.all_required_families_present);
        for summary in &result.family_summaries {
            assert_eq!(summary.pair_count, 2);
            assert_eq!(summary.completed_count, 2);
            assert_eq!(summary.completed_cluster_count, 2);
            assert_eq!(summary.cluster_balanced_mean_contrast_error_micro, Some(0));
            assert_eq!(
                summary.cluster_balanced_interval_coverage_ppm,
                Some(1_000_000)
            );
            assert_eq!(summary.cluster_balanced_false_change_rate_ppm, Some(0));
        }
        let null = result
            .family_summaries
            .iter()
            .find(|row| row.family == InterventionFamily::NullInvariance)
            .unwrap();
        assert_eq!(null.sign_eligible_pair_count, 0);
        assert_eq!(null.cluster_balanced_sign_accuracy_ppm, None);
        assert_eq!(result.total_branch_interaction_count, 32);
        assert_eq!(result.total_model_query_count, 64);
    }

    #[test]
    fn paired_contrast_exposes_prediction_error_hidden_by_plausible_branches() {
        let mut input = base_input();
        let row = input
            .pairs
            .iter_mut()
            .find(|row| row.family == InterventionFamily::Directional)
            .unwrap();
        row.treatment.predicted_outcome_mean_micro = Some(10_500);
        let result = evaluate_paired_interventions(&input).unwrap();
        let directional = result
            .family_summaries
            .iter()
            .find(|row| row.family == InterventionFamily::Directional)
            .unwrap();
        assert_eq!(
            directional.task_weighted_mean_contrast_error_micro,
            Some(250)
        );
        assert_eq!(
            directional.cluster_balanced_mean_contrast_error_micro,
            Some(250)
        );
    }

    #[test]
    fn sign_accuracy_uses_effect_resolution_floor() {
        let mut input = base_input();
        for row in &mut input.pairs {
            if row.family == InterventionFamily::Directional {
                row.treatment.predicted_outcome_mean_micro = Some(9_990);
            }
        }
        let result = evaluate_paired_interventions(&input).unwrap();
        let directional = result
            .family_summaries
            .iter()
            .find(|row| row.family == InterventionFamily::Directional)
            .unwrap();
        assert_eq!(directional.sign_eligible_pair_count, 2);
        assert_eq!(directional.sign_correct_pair_count, 0);
        assert_eq!(directional.cluster_balanced_sign_accuracy_ppm, Some(0));
    }

    #[test]
    fn null_intervention_is_not_counted_as_sign_correctness() {
        let result = evaluate_paired_interventions(&base_input()).unwrap();
        let null = result
            .family_summaries
            .iter()
            .find(|row| row.family == InterventionFamily::NullInvariance)
            .unwrap();
        assert_eq!(null.sign_eligible_pair_count, 0);
        assert_eq!(null.cluster_balanced_sign_accuracy_ppm, None);
    }

    #[test]
    fn invariant_variable_false_change_is_reported() {
        let mut input = base_input();
        let row = input
            .pairs
            .iter_mut()
            .find(|row| row.family == InterventionFamily::Temporal)
            .unwrap();
        row.invariant_variables[0].treatment_predicted_micro = 50;
        let result = evaluate_paired_interventions(&input).unwrap();
        let temporal = result
            .family_summaries
            .iter()
            .find(|row| row.family == InterventionFamily::Temporal)
            .unwrap();
        assert_eq!(temporal.invariant_observation_count, 2);
        assert_eq!(temporal.invariant_false_change_count, 1);
        assert_eq!(
            temporal.cluster_balanced_false_change_rate_ppm,
            Some(500_000)
        );
    }

    #[test]
    fn evaluator_truth_must_respect_declared_invariance() {
        let mut input = base_input();
        input.pairs[0].invariant_variables[0].treatment_true_micro = 100;
        assert!(matches!(
            evaluate_paired_interventions(&input),
            Err(PairedInterventionError::InvariantTruthViolation { .. })
        ));
    }

    #[test]
    fn missing_expected_pair_is_rejected() {
        let mut input = base_input();
        input.pairs.pop();
        assert!(matches!(
            evaluate_paired_interventions(&input),
            Err(PairedInterventionError::MissingPairs { .. })
        ));
    }

    #[test]
    fn duplicate_pair_id_is_rejected() {
        let mut input = base_input();
        input.pairs[1].pair_id = input.pairs[0].pair_id.clone();
        assert!(matches!(
            evaluate_paired_interventions(&input),
            Err(PairedInterventionError::DuplicatePairId { .. })
        ));
    }

    #[test]
    fn source_checkpoint_mutation_is_rejected() {
        let mut input = base_input();
        input.pairs[0].treatment.source_checkpoint_sha256 = digest(99);
        assert!(matches!(
            evaluate_paired_interventions(&input),
            Err(PairedInterventionError::BranchCheckpointMismatch { .. })
        ));
    }

    #[test]
    fn branch_disposition_must_match_pair_disposition() {
        let mut input = base_input();
        input.pairs[0].treatment.disposition = BranchDisposition::Failed;
        assert!(matches!(
            evaluate_paired_interventions(&input),
            Err(PairedInterventionError::BranchDispositionMismatch { .. })
        ));
    }

    #[test]
    fn failed_pair_is_retained_with_penalty_and_in_original_denominator() {
        let mut input = base_input();
        let row = input
            .pairs
            .iter_mut()
            .find(|row| {
                row.family == InterventionFamily::Directional
                    && row.environment_cluster_id.ends_with("cluster-1")
            })
            .unwrap();
        row.treatment.disposition = BranchDisposition::Failed;
        row.disposition = PairDisposition::Failed { penalty_ppm: 0 };
        row.predicted_effect_interval = None;
        row.invariant_variables.clear();
        let result = evaluate_paired_interventions(&input).unwrap();
        let directional = result
            .family_summaries
            .iter()
            .find(|row| row.family == InterventionFamily::Directional)
            .unwrap();
        assert_eq!(directional.pair_count, 2);
        assert_eq!(directional.failed_count, 1);
        assert_eq!(directional.completed_count, 1);
        assert_eq!(directional.completed_cluster_count, 1);
        assert!(!directional.cluster_floor_met);
        assert!(!result.all_family_cluster_floors_met);
        assert_eq!(directional.assigned_penalty_sum_ppm, 0);
    }

    #[test]
    fn abstention_before_execution_is_retained_and_costs_zero() {
        let mut input = base_input();
        let row = input
            .pairs
            .iter_mut()
            .find(|row| {
                row.family == InterventionFamily::Parameter
                    && row.environment_cluster_id.ends_with("cluster-1")
            })
            .unwrap();
        row.control.disposition = BranchDisposition::NotExecuted;
        row.control.true_outcome_micro = None;
        row.control.predicted_outcome_mean_micro = None;
        row.control.resources = PairedBranchResourceUse {
            interaction_count: 0,
            model_query_count: 0,
            elapsed_us: 0,
            measured_energy_uj: Some(0),
        };
        row.treatment.disposition = BranchDisposition::NotExecuted;
        row.treatment.true_outcome_micro = None;
        row.treatment.predicted_outcome_mean_micro = None;
        row.treatment.resources = row.control.resources;
        row.disposition = PairDisposition::Abstained {
            penalty_ppm: 100_000,
        };
        row.predicted_effect_interval = None;
        row.invariant_variables.clear();
        let result = evaluate_paired_interventions(&input).unwrap();
        let parameter = result
            .family_summaries
            .iter()
            .find(|row| row.family == InterventionFamily::Parameter)
            .unwrap();
        assert_eq!(parameter.pair_count, 2);
        assert_eq!(parameter.abstained_count, 1);
        assert_eq!(parameter.assigned_penalty_sum_ppm, 100_000);
    }

    #[test]
    fn family_with_only_failed_pairs_is_preserved_as_underpowered() {
        let mut input = base_input();
        for row in &mut input.pairs {
            if row.family == InterventionFamily::RegimeShift {
                row.control.disposition = BranchDisposition::Failed;
                row.treatment.disposition = BranchDisposition::Failed;
                row.disposition = PairDisposition::Failed { penalty_ppm: 0 };
                row.predicted_effect_interval = None;
                row.invariant_variables.clear();
            }
        }
        let result = evaluate_paired_interventions(&input).unwrap();
        let family = result
            .family_summaries
            .iter()
            .find(|row| row.family == InterventionFamily::RegimeShift)
            .unwrap();
        assert_eq!(family.pair_count, 2);
        assert_eq!(family.failed_count, 2);
        assert_eq!(family.completed_cluster_count, 0);
        assert!(!family.cluster_floor_met);
        assert_eq!(family.cluster_balanced_mean_contrast_error_micro, None);
        assert_eq!(family.task_weighted_mean_contrast_error_micro, None);
        assert!(!result.all_family_cluster_floors_met);
    }

    #[test]
    fn environment_cluster_cannot_claim_two_families() {
        let mut input = base_input();
        input.pairs[1].environment_cluster_id = input.pairs[0].environment_cluster_id.clone();
        input.pairs[1].environment_family_id = "other-family".to_owned();
        assert!(matches!(
            evaluate_paired_interventions(&input),
            Err(PairedInterventionError::EnvironmentClusterFamilyMismatch { .. })
        ));
    }

    #[test]
    fn paired_interval_must_be_ordered_and_cover_true_effect() {
        let mut input = base_input();
        input.pairs[0].predicted_effect_interval = Some(PredictedEffectInterval {
            lower_micro: 2_000,
            upper_micro: 1_000,
        });
        assert!(matches!(
            evaluate_paired_interventions(&input),
            Err(PairedInterventionError::PairIntervalInvalid { .. })
        ));
        let mut input = base_input();
        input.pairs[0].predicted_effect_interval = Some(PredictedEffectInterval {
            lower_micro: 1_100,
            upper_micro: 1_200,
        });
        let result = evaluate_paired_interventions(&input).unwrap();
        assert!(
            !result
                .pair_results
                .iter()
                .find(|row| row.pair_id == input.pairs[0].pair_id)
                .unwrap()
                .predicted_interval_covered
                .unwrap()
        );
    }

    #[test]
    fn invariant_variable_set_is_frozen_and_complete() {
        let mut input = base_input();
        input.pairs[0].invariant_variables.clear();
        assert!(matches!(
            evaluate_paired_interventions(&input),
            Err(PairedInterventionError::InvariantSetMismatch { .. })
        ));
        let mut input = base_input();
        input
            .context
            .expected_invariant_variable_ids
            .push("color".to_owned());
        assert!(matches!(
            evaluate_paired_interventions(&input),
            Err(PairedInterventionError::DuplicateInvariantVariableId { .. })
        ));
    }

    #[test]
    fn repeated_pairs_inside_one_cluster_do_not_increase_cluster_count() {
        let mut input = base_input();
        let mut repeated = input.pairs[0].clone();
        repeated.pair_id = "directional-repeat".to_owned();
        repeated.run_id = 1000;
        repeated.control.branch_id = "directional-repeat-control".to_owned();
        repeated.treatment.branch_id = "directional-repeat-treatment".to_owned();
        input.expected_pair_ids.push(repeated.pair_id.clone());
        input.pairs.push(repeated);
        let result = evaluate_paired_interventions(&input).unwrap();
        let directional = result
            .family_summaries
            .iter()
            .find(|row| row.family == InterventionFamily::Directional)
            .unwrap();
        assert_eq!(directional.pair_count, 3);
        assert_eq!(directional.completed_cluster_count, 2);
    }

    #[test]
    fn energy_budget_requires_both_branch_measurements() {
        let mut input = base_input();
        input.pairs[0].control.resources.measured_energy_uj = None;
        assert!(matches!(
            evaluate_paired_interventions(&input),
            Err(PairedInterventionError::MissingRequiredEnergyMeasurement { .. })
        ));
    }

    #[test]
    fn intervention_target_cannot_be_in_the_invariant_variable_set() {
        let mut input = base_input();
        input.context.expected_invariant_variable_ids = vec!["mass".to_owned()];
        assert!(matches!(
            evaluate_paired_interventions(&input),
            Err(PairedInterventionError::TargetVariableDeclaredInvariant { .. })
        ));
    }

    #[test]
    fn branch_ids_must_be_unique_and_pair_must_change_one_target_value() {
        let mut input = base_input();
        input.pairs[0].treatment.branch_id = input.pairs[0].control.branch_id.clone();
        assert!(matches!(
            evaluate_paired_interventions(&input),
            Err(PairedInterventionError::DuplicateBranchId { .. })
        ));
        let mut input = base_input();
        input.pairs[0].treatment_target_value_micro = input.pairs[0].control_target_value_micro;
        assert!(matches!(
            evaluate_paired_interventions(&input),
            Err(PairedInterventionError::TargetValueUnchanged { .. })
        ));
    }

    #[test]
    fn resource_budget_is_enforced_after_both_branches_are_charged() {
        let mut input = base_input();
        input.context.resource_budget.max_total_branch_interactions = 31;
        assert!(matches!(
            evaluate_paired_interventions(&input),
            Err(PairedInterventionError::ResourceBudgetExceeded {
                field: "max_total_branch_interactions"
            })
        ));
    }
}

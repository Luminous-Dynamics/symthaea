//! Deterministic familiar-source to hidden-source robotics transfer-gap reference.
//!
//! This crate validates frozen source-group assignments and matched, paired
//! cluster summaries. It reports descriptive transfer gaps only. It does not
//! ingest data, establish truthful labels, prove independence, or perform
//! statistical inference.

#![forbid(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};

/// Fixed-point rate scale.
pub const PPM_SCALE: u32 = 1_000_000;

/// A source dimension along which transfer may fail.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SourceDimension {
    /// Collection facility, laboratory or deployment site.
    CollectionSite,
    /// Physical robot article or serial unit.
    RobotArticle,
    /// Kinematic, actuator or embodiment family.
    RobotMorphology,
    /// Operator, teleoperator or demonstration collector.
    Operator,
    /// Sensor and calibration stack.
    SensorStack,
    /// Simulator, renderer, physics backend or asset family.
    SimulatorFamily,
    /// Upstream dataset, capture pipeline or annotation source.
    DatasetSource,
    /// Task, object, fixture or workcell family.
    TaskObjectFixture,
    /// Payload or load regime.
    PayloadRegime,
    /// Environment or operating-domain regime.
    EnvironmentRegime,
}

/// A complete source group is assigned to one partition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SourcePartition {
    /// Source groups represented in policy/model fitting data.
    Training,
    /// Source groups used for calibration or model selection.
    Calibration,
    /// Whole source groups withheld for confirmatory transfer evaluation.
    HiddenConfirmatory,
}

/// Required source coverage or explicit non-applicability.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DimensionApplicability {
    /// Required minimums for both familiar-reference and hidden-transfer regimes.
    Required {
        /// Minimum distinct source groups in the familiar-reference regime.
        minimum_familiar_groups: u16,
        /// Minimum distinct source groups in hidden confirmatory transfer.
        minimum_hidden_groups: u16,
        /// Minimum independent clusters in the familiar-reference regime.
        minimum_familiar_clusters: u16,
        /// Minimum independent clusters in hidden confirmatory transfer.
        minimum_hidden_clusters: u16,
    },
    /// This source dimension does not apply to the declared benchmark.
    NotApplicable {
        /// Explicit rationale; an empty value is invalid.
        reason: String,
    },
}

/// Policy for one source dimension.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DimensionPolicy {
    /// Source dimension.
    pub dimension: SourceDimension,
    /// Applicability and frozen minimums.
    pub applicability: DimensionApplicability,
}

/// Frozen whole-source-group assignment receipt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceGroupAssignment {
    /// Source dimension.
    pub dimension: SourceDimension,
    /// Stable privacy-preserving source-group identity.
    pub group_digest: String,
    /// Exactly one source partition owns the complete group.
    pub partition: SourcePartition,
    /// Receipt from the upstream dataset/split evidence owner.
    pub assignment_receipt_digest: String,
}

/// Evidence quality for a caller-supplied receipt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvidenceDisposition {
    /// The responsible evidence owner has qualified the receipt for this scope.
    Qualified,
    /// Evidence is missing, incomplete or unresolved.
    Unknown,
    /// Evidence is contradictory or explicitly rejected.
    Rejected,
}

/// Whether the upstream split owner confirmed the evaluation manifests are disjoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SplitSeparationDisposition {
    /// No overlap was found under the frozen manifest/case-ID comparison.
    QualifiedNoOverlap,
    /// Training or calibration data overlap an evaluation regime.
    OverlapDetected,
    /// The split owner could not establish complete separation.
    Unknown,
}

/// Frozen dataset, evaluation and split identities.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransferScope {
    /// Benchmark identifier.
    pub benchmark_id: String,
    /// Exact benchmark version.
    pub benchmark_version: String,
    /// Training-data manifest digest.
    pub training_manifest_digest: String,
    /// Calibration/model-selection manifest digest.
    pub calibration_manifest_digest: String,
    /// Fresh familiar-source evaluation manifest digest.
    pub familiar_reference_manifest_digest: String,
    /// Whole-source-group hidden confirmatory evaluation manifest digest.
    pub hidden_confirmatory_manifest_digest: String,
    /// Source/grouping policy digest.
    pub grouping_policy_digest: String,
    /// Frozen source-group split policy digest.
    pub split_policy_digest: String,
    /// Receipt attesting the four manifests are separated as declared.
    pub split_separation_receipt_digest: String,
    /// Upstream split owner disposition for case-level separation.
    pub split_separation_disposition: SplitSeparationDisposition,
    /// Task/environment/work mix manifest digest.
    pub task_manifest_digest: String,
    /// Physical environment and fixture digest.
    pub environment_manifest_digest: String,
    /// Robot article and configuration digest.
    pub robot_configuration_digest: String,
    /// Safety specification and outcome evaluator digest.
    pub evaluator_digest: String,
    /// Resource and staffing boundary digest.
    pub resource_budget_digest: String,
}

/// One comparison arm; all operating scope fields must match.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArmManifest {
    /// Stable arm identity.
    pub arm_id: String,
    /// Model artifact digest.
    pub model_artifact_digest: String,
    /// Policy artifact/configuration digest.
    pub policy_digest: String,
    /// Runtime/harness digest.
    pub harness_digest: String,
    /// Task/work mix digest.
    pub task_manifest_digest: String,
    /// Environment/fixture digest.
    pub environment_manifest_digest: String,
    /// Robot article/configuration digest.
    pub robot_configuration_digest: String,
    /// Safety/evaluation digest.
    pub evaluator_digest: String,
    /// Resource/staffing boundary digest.
    pub resource_budget_digest: String,
}

/// Two separately reported evaluation regimes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum EvaluationRegime {
    /// Fresh episodes from source groups represented in training data.
    FamiliarSourceReference,
    /// Episodes from source groups withheld as complete groups.
    HiddenSourceTransfer,
}

/// Mutually exclusive disposition of one scheduled episode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EpisodeOutcome {
    /// Task succeeded and every applicable safety specification passed.
    SafeSuccess,
    /// Task succeeded but at least one applicable safety specification failed.
    SuccessfulButUnsafe,
    /// Task failed and no applicable safety specification was violated.
    SafeFailure,
    /// Task failed and at least one applicable safety specification was violated.
    UnsafeFailure,
    /// Outcome is missing, censored or unresolved.
    Unknown,
}

/// Complete terminal outcome tally for one arm and independent cluster.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OutcomeTally {
    /// Number of episodes scheduled before execution.
    pub scheduled_episodes: u32,
    /// Task success with all applicable safety specifications satisfied.
    pub safe_successes: u32,
    /// Task success with at least one safety specification violated.
    pub successful_but_unsafe: u32,
    /// Task failure with no applicable safety specification violated.
    pub safe_failures: u32,
    /// Task failure with at least one safety specification violated.
    pub unsafe_failures: u32,
    /// Unresolved/censored episode outcomes.
    pub unknown: u32,
}

impl OutcomeTally {
    fn validate(&self, arm_id: &str, cluster_id: &str) -> Result<(), TransferGapError> {
        if self.scheduled_episodes == 0 {
            return Err(TransferGapError::InvalidOutcomeTally {
                arm_id: arm_id.to_owned(),
                cluster_id: cluster_id.to_owned(),
            });
        }
        let classified = [
            self.safe_successes,
            self.successful_but_unsafe,
            self.safe_failures,
            self.unsafe_failures,
            self.unknown,
        ]
        .into_iter()
        .try_fold(0_u32, |sum, value| sum.checked_add(value))
        .ok_or_else(|| TransferGapError::InvalidOutcomeTally {
            arm_id: arm_id.to_owned(),
            cluster_id: cluster_id.to_owned(),
        })?;
        if classified != self.scheduled_episodes {
            return Err(TransferGapError::InvalidOutcomeTally {
                arm_id: arm_id.to_owned(),
                cluster_id: cluster_id.to_owned(),
            });
        }
        Ok(())
    }

    fn rate_ppm(&self, axis: OutcomeAxis) -> Result<u32, TransferGapError> {
        let successes = match axis {
            OutcomeAxis::TaskSuccess => self
                .safe_successes
                .checked_add(self.successful_but_unsafe)
                .ok_or(TransferGapError::ArithmeticOverflow)?,
            OutcomeAxis::SafeSuccess => self.safe_successes,
            OutcomeAxis::SuccessfulButUnsafe => self.successful_but_unsafe,
        };
        let numerator = u64::from(successes)
            .checked_mul(u64::from(PPM_SCALE))
            .ok_or(TransferGapError::ArithmeticOverflow)?;
        let rate = numerator / u64::from(self.scheduled_episodes);
        u32::try_from(rate).map_err(|_| TransferGapError::ArithmeticOverflow)
    }
}

/// One matched, source-group-bound independent cluster in one regime.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClusterEvaluation {
    /// Source dimension measured by this record.
    pub dimension: SourceDimension,
    /// Familiar-source or hidden-source regime.
    pub regime: EvaluationRegime,
    /// Complete source-group identity for this dimension.
    pub source_group_digest: String,
    /// Independent experimental cluster identity.
    pub cluster_id: String,
    /// Exact evaluation manifest for this regime.
    pub evaluation_manifest_digest: String,
    /// Paired case/scenario/seed manifest shared by both arms.
    pub paired_case_manifest_digest: String,
    /// Receipt supporting construction of this independent cluster.
    pub independence_receipt_digest: String,
    /// Independence evidence disposition from the experiment owner.
    pub independence_disposition: EvidenceDisposition,
    /// Baseline outcome tally.
    pub baseline: OutcomeTally,
    /// Candidate outcome tally.
    pub candidate: OutcomeTally,
}

/// Complete input to the descriptive transfer-gap audit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransferGapInput {
    /// Frozen data, evaluator and operating scope.
    pub scope: TransferScope,
    /// Required source-dimension policies.
    pub dimensions: Vec<DimensionPolicy>,
    /// Frozen whole-source-group assignments.
    pub assignments: Vec<SourceGroupAssignment>,
    /// Matched baseline arm.
    pub baseline_manifest: ArmManifest,
    /// Matched candidate arm.
    pub candidate_manifest: ArmManifest,
    /// Paired independent-cluster outcomes across both regimes.
    pub evaluations: Vec<ClusterEvaluation>,
}

/// The outcome axes are intentionally not collapsed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum OutcomeAxis {
    /// Native task success, regardless of safety outcome.
    TaskSuccess,
    /// Task success with every applicable safety specification satisfied.
    SafeSuccess,
    /// Task successes that violated at least one applicable safety specification.
    SuccessfulButUnsafe,
}

/// Descriptive cluster-level result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClusterTransferReport {
    /// Source dimension.
    pub dimension: SourceDimension,
    /// Evaluation regime.
    pub regime: EvaluationRegime,
    /// Source group identity.
    pub source_group_digest: String,
    /// Independent cluster identity.
    pub cluster_id: String,
    /// Episodes scheduled in the cluster.
    pub scheduled_episodes: u32,
    /// Unknown outcomes retained in the original denominator.
    pub baseline_unknown: u32,
    /// Unknown outcomes retained in the original denominator.
    pub candidate_unknown: u32,
    /// Baseline task/safety vector.
    pub baseline_task_success_ppm: u32,
    /// Candidate task/safety vector.
    pub candidate_task_success_ppm: u32,
    /// Baseline safe-success rate.
    pub baseline_safe_success_ppm: u32,
    /// Candidate safe-success rate.
    pub candidate_safe_success_ppm: u32,
    /// Baseline successful-but-unsafe rate.
    pub baseline_successful_but_unsafe_ppm: u32,
    /// Candidate successful-but-unsafe rate.
    pub candidate_successful_but_unsafe_ppm: u32,
    /// Cluster independence state.
    pub independence_disposition: EvidenceDisposition,
    /// Whether this cluster has complete terminal outcomes and independence evidence.
    pub evidence_complete: bool,
}

/// Equal-weighted cluster-rate summary for one arm/regime/outcome axis.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegimeRateSummary {
    /// Number of independent clusters.
    pub cluster_count: u16,
    /// Number of distinct source groups.
    pub source_group_count: u16,
    /// Total scheduled episodes; shown for coverage, not as the statistical sample size.
    pub episode_count: u64,
    /// Equal-weighted mean of per-cluster rates.
    pub mean_cluster_rate_ppm: Option<u32>,
    /// Minimum per-cluster rate.
    pub minimum_cluster_rate_ppm: Option<u32>,
    /// Maximum per-cluster rate.
    pub maximum_cluster_rate_ppm: Option<u32>,
    /// Whether any cluster has Unknown/censored outcomes or independence evidence.
    pub has_unknown_evidence: bool,
}

/// Familiar-to-hidden descriptive gap for one outcome axis and one dimension.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AxisTransferGapReport {
    /// Outcome axis; task success, safe-success and unsafe-success are separate.
    pub axis: OutcomeAxis,
    /// Baseline familiar-source rate.
    pub baseline_familiar: RegimeRateSummary,
    /// Baseline hidden-source rate.
    pub baseline_hidden: RegimeRateSummary,
    /// Baseline transfer gap: hidden minus familiar.
    pub baseline_transfer_gap_ppm: Option<i64>,
    /// Candidate familiar-source rate.
    pub candidate_familiar: RegimeRateSummary,
    /// Candidate hidden-source rate.
    pub candidate_hidden: RegimeRateSummary,
    /// Candidate transfer gap: hidden minus familiar.
    pub candidate_transfer_gap_ppm: Option<i64>,
    /// Difference-in-differences: candidate gap minus baseline gap.
    ///
    /// Positive means the candidate has a less negative observed transfer
    /// gap than the matched baseline for this frozen source scope.
    pub difference_in_differences_ppm: Option<i64>,
}

/// Completeness for one source dimension.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DimensionTransferDisposition {
    /// Both regimes meet all frozen minimums and all outcomes are resolved.
    Complete,
    /// One regime lacks coverage or at least one outcome/independence is Unknown.
    Inconclusive,
    /// The dimension is explicitly outside benchmark applicability.
    NotApplicable,
}

/// Transfer-gap report for one source dimension.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DimensionTransferReport {
    /// Source dimension.
    pub dimension: SourceDimension,
    /// Number of whole groups frozen in the training partition.
    pub training_group_count: u16,
    /// Number of whole groups frozen in hidden confirmatory partition.
    pub hidden_group_assignment_count: u16,
    /// Dimension disposition.
    pub disposition: DimensionTransferDisposition,
    /// Reports for task success, safe-success and successful-but-unsafe.
    pub axes: Vec<AxisTransferGapReport>,
}

/// Overall transfer-gap disposition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransferGapDisposition {
    /// All Required dimensions are complete and all outcomes resolved.
    Complete,
    /// Missing regime coverage, Unknown outcomes or Unknown independence.
    Inconclusive,
}

/// Deterministic familiar-to-hidden transfer-gap report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransferGapReport {
    /// Benchmark identity.
    pub benchmark_id: String,
    /// Dataset split-separation receipt digest.
    pub split_separation_receipt_digest: String,
    /// Split owner disposition; Unknown makes the report inconclusive.
    pub split_separation_disposition: SplitSeparationDisposition,
    /// Baseline arm identity.
    pub baseline_arm_id: String,
    /// Candidate arm identity.
    pub candidate_arm_id: String,
    /// Overall disposition.
    pub disposition: TransferGapDisposition,
    /// Per-cluster rates and source group identities.
    pub clusters: Vec<ClusterTransferReport>,
    /// Per-dimension transfer gaps.
    pub dimensions: Vec<DimensionTransferReport>,
    /// This evaluator has no actuation authority.
    pub actuation_authorized: bool,
}

/// Structural, scope or evidence-contract failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransferGapError {
    /// Required scope identity is empty.
    EmptyIdentity(&'static str),
    /// Duplicate dimension policy.
    DuplicateDimensionPolicy(SourceDimension),
    /// Required dimension has zero cluster/group minimum.
    InvalidDimensionMinimum(SourceDimension),
    /// NotApplicable dimension has no rationale.
    MissingNotApplicableReason(SourceDimension),
    /// Assignment refers to an unconfigured dimension.
    AssignmentForUnknownDimension(SourceDimension),
    /// Assignment is present for a NotApplicable dimension.
    AssignmentForNotApplicableDimension(SourceDimension),
    /// Duplicate source group assignment.
    DuplicateSourceGroupAssignment {
        dimension: SourceDimension,
        group_digest: String,
    },
    /// Missing source-group assignment receipt.
    MissingSourceGroupReceipt(SourceDimension, String),
    /// One assignment receipt was reused for different source-group identities.
    SourceGroupReceiptReused(String),
    /// Required familiar/training or hidden partition is empty.
    EmptyRequiredGroupPartition(SourceDimension, SourcePartition),
    /// Frozen minimum source-group count is not met.
    InsufficientAssignedGroups {
        dimension: SourceDimension,
        partition: SourcePartition,
        minimum: u16,
        observed: usize,
    },
    /// Candidate and baseline arm scope does not match.
    ArmScopeMismatch { arm_id: String, field: &'static str },
    /// Candidate and baseline arm IDs are identical.
    DuplicateArmId(String),
    /// No paired cluster evaluations supplied.
    NoEvaluations,
    /// Cluster evaluation refers to an unconfigured dimension.
    EvaluationForUnknownDimension(SourceDimension),
    /// Evaluation exists for a NotApplicable dimension.
    EvaluationForNotApplicableDimension(SourceDimension),
    /// Empty cluster/group/manifest identity or receipt.
    MissingEvaluationIdentity(String),
    /// Cluster source group is not in the correct source partition.
    EvaluationGroupPartitionMismatch {
        dimension: SourceDimension,
        regime: EvaluationRegime,
        group_digest: String,
    },
    /// Regime evaluation manifest does not match the frozen scope.
    EvaluationManifestMismatch(String),
    /// A familiar reference manifest aliases training or calibration data.
    ReferenceManifestAliasesFitData,
    /// The split owner explicitly found training/calibration overlap with evaluation data.
    EvaluationDataOverlap,
    /// A cluster identifier is reused across familiar and hidden regimes.
    ClusterReusedAcrossRegimes(String),
    /// Cluster identity is reused with inconsistent independence receipts.
    ClusterReceiptMismatch(String),
    /// One independence receipt is reused for different cluster identities.
    IndependenceReceiptReused(String),
    /// Duplicate cluster record for one dimension/regime.
    DuplicateClusterEvaluation {
        dimension: SourceDimension,
        regime: EvaluationRegime,
        cluster_id: String,
    },
    /// Duplicate paired case manifest within one dimension/regime.
    DuplicatePairedCaseManifest {
        dimension: SourceDimension,
        regime: EvaluationRegime,
        manifest_digest: String,
    },
    /// Independence evidence is explicitly rejected.
    RejectedIndependenceEvidence(String),
    /// Candidate and baseline did not receive the same scheduled episode count.
    PairingMismatch(String),
    /// Outcome categories do not exactly cover the scheduled denominator.
    InvalidOutcomeTally { arm_id: String, cluster_id: String },
    /// Arithmetic overflow in an integer rate or aggregate.
    ArithmeticOverflow,
}

fn require_nonempty(name: &'static str, value: &str) -> Result<(), TransferGapError> {
    if value.trim().is_empty() {
        return Err(TransferGapError::EmptyIdentity(name));
    }
    Ok(())
}

fn validate_scope(scope: &TransferScope) -> Result<(), TransferGapError> {
    for (name, value) in [
        ("benchmark_id", scope.benchmark_id.as_str()),
        ("benchmark_version", scope.benchmark_version.as_str()),
        (
            "training_manifest_digest",
            scope.training_manifest_digest.as_str(),
        ),
        (
            "calibration_manifest_digest",
            scope.calibration_manifest_digest.as_str(),
        ),
        (
            "familiar_reference_manifest_digest",
            scope.familiar_reference_manifest_digest.as_str(),
        ),
        (
            "hidden_confirmatory_manifest_digest",
            scope.hidden_confirmatory_manifest_digest.as_str(),
        ),
        (
            "grouping_policy_digest",
            scope.grouping_policy_digest.as_str(),
        ),
        ("split_policy_digest", scope.split_policy_digest.as_str()),
        (
            "split_separation_receipt_digest",
            scope.split_separation_receipt_digest.as_str(),
        ),
        ("task_manifest_digest", scope.task_manifest_digest.as_str()),
        (
            "environment_manifest_digest",
            scope.environment_manifest_digest.as_str(),
        ),
        (
            "robot_configuration_digest",
            scope.robot_configuration_digest.as_str(),
        ),
        ("evaluator_digest", scope.evaluator_digest.as_str()),
        (
            "resource_budget_digest",
            scope.resource_budget_digest.as_str(),
        ),
    ] {
        require_nonempty(name, value)?;
    }
    if scope.split_separation_disposition == SplitSeparationDisposition::OverlapDetected {
        return Err(TransferGapError::EvaluationDataOverlap);
    }
    if scope.training_manifest_digest == scope.calibration_manifest_digest
        || scope.familiar_reference_manifest_digest == scope.training_manifest_digest
        || scope.familiar_reference_manifest_digest == scope.calibration_manifest_digest
        || scope.hidden_confirmatory_manifest_digest == scope.training_manifest_digest
        || scope.hidden_confirmatory_manifest_digest == scope.calibration_manifest_digest
        || scope.familiar_reference_manifest_digest == scope.hidden_confirmatory_manifest_digest
    {
        return Err(TransferGapError::ReferenceManifestAliasesFitData);
    }
    Ok(())
}

fn validate_arm(scope: &TransferScope, arm: &ArmManifest) -> Result<(), TransferGapError> {
    for (name, value) in [
        ("arm_id", arm.arm_id.as_str()),
        ("model_artifact_digest", arm.model_artifact_digest.as_str()),
        ("policy_digest", arm.policy_digest.as_str()),
        ("harness_digest", arm.harness_digest.as_str()),
    ] {
        require_nonempty(name, value)?;
    }
    for (field, observed, expected) in [
        (
            "task_manifest_digest",
            arm.task_manifest_digest.as_str(),
            scope.task_manifest_digest.as_str(),
        ),
        (
            "environment_manifest_digest",
            arm.environment_manifest_digest.as_str(),
            scope.environment_manifest_digest.as_str(),
        ),
        (
            "robot_configuration_digest",
            arm.robot_configuration_digest.as_str(),
            scope.robot_configuration_digest.as_str(),
        ),
        (
            "evaluator_digest",
            arm.evaluator_digest.as_str(),
            scope.evaluator_digest.as_str(),
        ),
        (
            "resource_budget_digest",
            arm.resource_budget_digest.as_str(),
            scope.resource_budget_digest.as_str(),
        ),
    ] {
        if observed != expected {
            return Err(TransferGapError::ArmScopeMismatch {
                arm_id: arm.arm_id.clone(),
                field,
            });
        }
    }
    Ok(())
}

#[derive(Debug, Clone)]
struct ValidatedPolicy {
    applicability: DimensionApplicability,
    assignments: BTreeMap<String, SourcePartition>,
    partition_counts: BTreeMap<SourcePartition, usize>,
}

fn validate_policies(
    policies: &[DimensionPolicy],
    assignments: &[SourceGroupAssignment],
) -> Result<BTreeMap<SourceDimension, ValidatedPolicy>, TransferGapError> {
    let mut validated = BTreeMap::new();
    let mut assignment_receipts: BTreeMap<String, (SourceDimension, String)> = BTreeMap::new();
    for policy in policies {
        if validated
            .insert(
                policy.dimension,
                ValidatedPolicy {
                    applicability: policy.applicability.clone(),
                    assignments: BTreeMap::new(),
                    partition_counts: BTreeMap::new(),
                },
            )
            .is_some()
        {
            return Err(TransferGapError::DuplicateDimensionPolicy(policy.dimension));
        }
        match &policy.applicability {
            DimensionApplicability::Required {
                minimum_familiar_groups,
                minimum_hidden_groups,
                minimum_familiar_clusters,
                minimum_hidden_clusters,
            } if *minimum_familiar_groups < 2
                || *minimum_hidden_groups < 2
                || *minimum_familiar_clusters < 2
                || *minimum_hidden_clusters < 2 =>
            {
                return Err(TransferGapError::InvalidDimensionMinimum(policy.dimension));
            }
            DimensionApplicability::NotApplicable { reason } if reason.trim().is_empty() => {
                return Err(TransferGapError::MissingNotApplicableReason(
                    policy.dimension,
                ));
            }
            _ => {}
        }
    }

    if validated.is_empty() {
        return Err(TransferGapError::NoEvaluations);
    }

    for assignment in assignments {
        let policy = validated.get_mut(&assignment.dimension).ok_or(
            TransferGapError::AssignmentForUnknownDimension(assignment.dimension),
        )?;
        if matches!(
            &policy.applicability,
            DimensionApplicability::NotApplicable { .. }
        ) {
            return Err(TransferGapError::AssignmentForNotApplicableDimension(
                assignment.dimension,
            ));
        }
        require_nonempty("group_digest", &assignment.group_digest)?;
        if assignment.assignment_receipt_digest.trim().is_empty() {
            return Err(TransferGapError::MissingSourceGroupReceipt(
                assignment.dimension,
                assignment.group_digest.clone(),
            ));
        }
        if policy
            .assignments
            .insert(assignment.group_digest.clone(), assignment.partition)
            .is_some()
        {
            return Err(TransferGapError::DuplicateSourceGroupAssignment {
                dimension: assignment.dimension,
                group_digest: assignment.group_digest.clone(),
            });
        }
        match assignment_receipts.get(&assignment.assignment_receipt_digest) {
            Some(existing)
                if existing != &(assignment.dimension, assignment.group_digest.clone()) =>
            {
                return Err(TransferGapError::SourceGroupReceiptReused(
                    assignment.assignment_receipt_digest.clone(),
                ));
            }
            Some(_) => {
                return Err(TransferGapError::SourceGroupReceiptReused(
                    assignment.assignment_receipt_digest.clone(),
                ));
            }
            None => {
                assignment_receipts.insert(
                    assignment.assignment_receipt_digest.clone(),
                    (assignment.dimension, assignment.group_digest.clone()),
                );
            }
        }
        let count = policy
            .partition_counts
            .entry(assignment.partition)
            .or_insert(0_usize);
        *count = count
            .checked_add(1)
            .ok_or(TransferGapError::ArithmeticOverflow)?;
    }

    for (dimension, policy) in &validated {
        if let DimensionApplicability::Required {
            minimum_familiar_groups,
            minimum_hidden_groups,
            ..
        } = &policy.applicability
        {
            for partition in [
                SourcePartition::Training,
                SourcePartition::Calibration,
                SourcePartition::HiddenConfirmatory,
            ] {
                if policy
                    .partition_counts
                    .get(&partition)
                    .copied()
                    .unwrap_or(0)
                    == 0
                {
                    return Err(TransferGapError::EmptyRequiredGroupPartition(
                        *dimension, partition,
                    ));
                }
            }
            let training = policy
                .partition_counts
                .get(&SourcePartition::Training)
                .copied()
                .unwrap_or(0);
            if training < usize::from(*minimum_familiar_groups) {
                return Err(TransferGapError::InsufficientAssignedGroups {
                    dimension: *dimension,
                    partition: SourcePartition::Training,
                    minimum: *minimum_familiar_groups,
                    observed: training,
                });
            }
            let hidden = policy
                .partition_counts
                .get(&SourcePartition::HiddenConfirmatory)
                .copied()
                .unwrap_or(0);
            if hidden < usize::from(*minimum_hidden_groups) {
                return Err(TransferGapError::InsufficientAssignedGroups {
                    dimension: *dimension,
                    partition: SourcePartition::HiddenConfirmatory,
                    minimum: *minimum_hidden_groups,
                    observed: hidden,
                });
            }
        }
    }
    Ok(validated)
}

#[derive(Debug, Clone)]
struct ClusterGroup {
    regime: EvaluationRegime,
    group_digest: String,
    independence_receipt_digest: String,
    independence_disposition: EvidenceDisposition,
}

fn validate_evaluations(
    scope: &TransferScope,
    policies: &BTreeMap<SourceDimension, ValidatedPolicy>,
    baseline_arm_id: &str,
    candidate_arm_id: &str,
    evaluations: &[ClusterEvaluation],
) -> Result<
    BTreeMap<(SourceDimension, EvaluationRegime), Vec<ClusterTransferReport>>,
    TransferGapError,
> {
    if evaluations.is_empty() {
        return Err(TransferGapError::NoEvaluations);
    }
    let mut seen_clusters = BTreeSet::new();
    let mut seen_case_manifests = BTreeSet::new();
    let mut cluster_identity: BTreeMap<(SourceDimension, String), ClusterGroup> = BTreeMap::new();
    let mut receipt_clusters: BTreeMap<String, String> = BTreeMap::new();
    let mut reports: BTreeMap<(SourceDimension, EvaluationRegime), Vec<ClusterTransferReport>> =
        BTreeMap::new();

    for evaluation in evaluations {
        let policy = policies.get(&evaluation.dimension).ok_or(
            TransferGapError::EvaluationForUnknownDimension(evaluation.dimension),
        )?;
        if matches!(
            &policy.applicability,
            DimensionApplicability::NotApplicable { .. }
        ) {
            return Err(TransferGapError::EvaluationForNotApplicableDimension(
                evaluation.dimension,
            ));
        }

        require_nonempty("source_group_digest", &evaluation.source_group_digest).map_err(|_| {
            TransferGapError::MissingEvaluationIdentity(evaluation.cluster_id.clone())
        })?;
        require_nonempty("cluster_id", &evaluation.cluster_id).map_err(|_| {
            TransferGapError::MissingEvaluationIdentity(evaluation.cluster_id.clone())
        })?;
        require_nonempty(
            "evaluation_manifest_digest",
            &evaluation.evaluation_manifest_digest,
        )
        .map_err(|_| TransferGapError::MissingEvaluationIdentity(evaluation.cluster_id.clone()))?;
        require_nonempty(
            "paired_case_manifest_digest",
            &evaluation.paired_case_manifest_digest,
        )
        .map_err(|_| TransferGapError::MissingEvaluationIdentity(evaluation.cluster_id.clone()))?;
        require_nonempty(
            "independence_receipt_digest",
            &evaluation.independence_receipt_digest,
        )
        .map_err(|_| TransferGapError::MissingEvaluationIdentity(evaluation.cluster_id.clone()))?;

        let expected_manifest = match evaluation.regime {
            EvaluationRegime::FamiliarSourceReference => &scope.familiar_reference_manifest_digest,
            EvaluationRegime::HiddenSourceTransfer => &scope.hidden_confirmatory_manifest_digest,
        };
        if &evaluation.evaluation_manifest_digest != expected_manifest {
            return Err(TransferGapError::EvaluationManifestMismatch(
                evaluation.cluster_id.clone(),
            ));
        }

        let expected_partition = match evaluation.regime {
            EvaluationRegime::FamiliarSourceReference => SourcePartition::Training,
            EvaluationRegime::HiddenSourceTransfer => SourcePartition::HiddenConfirmatory,
        };
        if policy.assignments.get(&evaluation.source_group_digest) != Some(&expected_partition) {
            return Err(TransferGapError::EvaluationGroupPartitionMismatch {
                dimension: evaluation.dimension,
                regime: evaluation.regime,
                group_digest: evaluation.source_group_digest.clone(),
            });
        }

        if evaluation.independence_disposition == EvidenceDisposition::Rejected {
            return Err(TransferGapError::RejectedIndependenceEvidence(
                evaluation.cluster_id.clone(),
            ));
        }

        let cluster_key = (
            evaluation.dimension,
            evaluation.regime,
            evaluation.cluster_id.clone(),
        );
        if !seen_clusters.insert(cluster_key.clone()) {
            return Err(TransferGapError::DuplicateClusterEvaluation {
                dimension: evaluation.dimension,
                regime: evaluation.regime,
                cluster_id: evaluation.cluster_id.clone(),
            });
        }
        let case_key = (
            evaluation.dimension,
            evaluation.regime,
            evaluation.paired_case_manifest_digest.clone(),
        );
        if !seen_case_manifests.insert(case_key) {
            return Err(TransferGapError::DuplicatePairedCaseManifest {
                dimension: evaluation.dimension,
                regime: evaluation.regime,
                manifest_digest: evaluation.paired_case_manifest_digest.clone(),
            });
        }

        let cluster_identity_key = (evaluation.dimension, evaluation.cluster_id.clone());
        match cluster_identity.get(&cluster_identity_key) {
            Some(existing)
                if existing.regime != evaluation.regime
                    || existing.group_digest != evaluation.source_group_digest
                    || existing.independence_receipt_digest
                        != evaluation.independence_receipt_digest
                    || existing.independence_disposition != evaluation.independence_disposition =>
            {
                return Err(TransferGapError::ClusterReceiptMismatch(
                    evaluation.cluster_id.clone(),
                ));
            }
            Some(_) => {}
            None => {
                cluster_identity.insert(
                    cluster_identity_key,
                    ClusterGroup {
                        regime: evaluation.regime,
                        group_digest: evaluation.source_group_digest.clone(),
                        independence_receipt_digest: evaluation.independence_receipt_digest.clone(),
                        independence_disposition: evaluation.independence_disposition,
                    },
                );
            }
        }

        match receipt_clusters.get(&evaluation.independence_receipt_digest) {
            Some(existing_cluster) if existing_cluster != &evaluation.cluster_id => {
                return Err(TransferGapError::IndependenceReceiptReused(
                    evaluation.independence_receipt_digest.clone(),
                ));
            }
            Some(_) => {}
            None => {
                receipt_clusters.insert(
                    evaluation.independence_receipt_digest.clone(),
                    evaluation.cluster_id.clone(),
                );
            }
        }

        evaluation
            .baseline
            .validate(baseline_arm_id, &evaluation.cluster_id)?;
        evaluation
            .candidate
            .validate(candidate_arm_id, &evaluation.cluster_id)?;
        if evaluation.baseline.scheduled_episodes != evaluation.candidate.scheduled_episodes {
            return Err(TransferGapError::PairingMismatch(
                evaluation.cluster_id.clone(),
            ));
        }

        let unknown = evaluation.baseline.unknown > 0
            || evaluation.candidate.unknown > 0
            || evaluation.independence_disposition == EvidenceDisposition::Unknown;
        let report = ClusterTransferReport {
            dimension: evaluation.dimension,
            regime: evaluation.regime,
            source_group_digest: evaluation.source_group_digest.clone(),
            cluster_id: evaluation.cluster_id.clone(),
            scheduled_episodes: evaluation.baseline.scheduled_episodes,
            baseline_unknown: evaluation.baseline.unknown,
            candidate_unknown: evaluation.candidate.unknown,
            baseline_task_success_ppm: evaluation.baseline.rate_ppm(OutcomeAxis::TaskSuccess)?,
            candidate_task_success_ppm: evaluation.candidate.rate_ppm(OutcomeAxis::TaskSuccess)?,
            baseline_safe_success_ppm: evaluation.baseline.rate_ppm(OutcomeAxis::SafeSuccess)?,
            candidate_safe_success_ppm: evaluation.candidate.rate_ppm(OutcomeAxis::SafeSuccess)?,
            baseline_successful_but_unsafe_ppm: evaluation
                .baseline
                .rate_ppm(OutcomeAxis::SuccessfulButUnsafe)?,
            candidate_successful_but_unsafe_ppm: evaluation
                .candidate
                .rate_ppm(OutcomeAxis::SuccessfulButUnsafe)?,
            independence_disposition: evaluation.independence_disposition,
            evidence_complete: !unknown,
        };
        reports
            .entry((evaluation.dimension, evaluation.regime))
            .or_default()
            .push(report);
    }

    for values in reports.values_mut() {
        values.sort_by(|a, b| a.cluster_id.cmp(&b.cluster_id));
    }
    Ok(reports)
}

fn axis_rate(report: &ClusterTransferReport, axis: OutcomeAxis, candidate: bool) -> u32 {
    match (axis, candidate) {
        (OutcomeAxis::TaskSuccess, false) => report.baseline_task_success_ppm,
        (OutcomeAxis::TaskSuccess, true) => report.candidate_task_success_ppm,
        (OutcomeAxis::SafeSuccess, false) => report.baseline_safe_success_ppm,
        (OutcomeAxis::SafeSuccess, true) => report.candidate_safe_success_ppm,
        (OutcomeAxis::SuccessfulButUnsafe, false) => report.baseline_successful_but_unsafe_ppm,
        (OutcomeAxis::SuccessfulButUnsafe, true) => report.candidate_successful_but_unsafe_ppm,
    }
}

fn summarize_regime(
    reports: &[ClusterTransferReport],
    axis: OutcomeAxis,
    candidate: bool,
) -> Result<RegimeRateSummary, TransferGapError> {
    let mut rates = Vec::with_capacity(reports.len());
    let mut groups = BTreeSet::new();
    let mut episodes = 0_u64;
    let mut unknown = false;
    for report in reports {
        rates.push(axis_rate(report, axis, candidate));
        groups.insert(report.source_group_digest.as_str());
        episodes = episodes
            .checked_add(u64::from(report.scheduled_episodes))
            .ok_or(TransferGapError::ArithmeticOverflow)?;
        unknown |= !report.evidence_complete;
    }
    rates.sort_unstable();
    let mean = if rates.is_empty() {
        None
    } else {
        let sum = rates
            .iter()
            .try_fold(0_u128, |acc, rate| acc.checked_add(u128::from(*rate)));
        let total = sum.ok_or(TransferGapError::ArithmeticOverflow)?;
        Some(
            u32::try_from(
                total
                    / u128::try_from(rates.len())
                        .map_err(|_| TransferGapError::ArithmeticOverflow)?,
            )
            .map_err(|_| TransferGapError::ArithmeticOverflow)?,
        )
    };
    Ok(RegimeRateSummary {
        cluster_count: u16::try_from(reports.len())
            .map_err(|_| TransferGapError::ArithmeticOverflow)?,
        source_group_count: u16::try_from(groups.len())
            .map_err(|_| TransferGapError::ArithmeticOverflow)?,
        episode_count: episodes,
        mean_cluster_rate_ppm: mean,
        minimum_cluster_rate_ppm: rates.first().copied(),
        maximum_cluster_rate_ppm: rates.last().copied(),
        has_unknown_evidence: unknown,
    })
}

fn subtract_rates(hidden: Option<u32>, familiar: Option<u32>) -> Option<i64> {
    hidden
        .zip(familiar)
        .map(|(hidden, familiar)| i64::from(hidden) - i64::from(familiar))
}

/// Validate and summarize familiar-source versus hidden-source transfer.
///
/// The mean gives each independent cluster equal weight. It does not use
/// episode count as the statistical sample size and emits no inferential result.
pub fn audit_transfer_gap(input: &TransferGapInput) -> Result<TransferGapReport, TransferGapError> {
    validate_scope(&input.scope)?;
    validate_arm(&input.scope, &input.baseline_manifest)?;
    validate_arm(&input.scope, &input.candidate_manifest)?;
    if input.baseline_manifest.arm_id == input.candidate_manifest.arm_id {
        return Err(TransferGapError::DuplicateArmId(
            input.baseline_manifest.arm_id.clone(),
        ));
    }

    let policies = validate_policies(&input.dimensions, &input.assignments)?;
    let split_evidence_unknown =
        input.scope.split_separation_disposition == SplitSeparationDisposition::Unknown;
    let grouped = validate_evaluations(
        &input.scope,
        &policies,
        &input.baseline_manifest.arm_id,
        &input.candidate_manifest.arm_id,
        &input.evaluations,
    )?;

    let mut dimension_reports = Vec::with_capacity(policies.len());
    let mut overall_inconclusive = false;

    for (dimension, policy) in &policies {
        match &policy.applicability {
            DimensionApplicability::NotApplicable { .. } => {
                dimension_reports.push(DimensionTransferReport {
                    dimension: *dimension,
                    training_group_count: 0,
                    hidden_group_assignment_count: 0,
                    disposition: DimensionTransferDisposition::NotApplicable,
                    axes: Vec::new(),
                });
            }
            DimensionApplicability::Required {
                minimum_familiar_clusters,
                minimum_hidden_clusters,
                ..
            } => {
                let familiar = grouped
                    .get(&(*dimension, EvaluationRegime::FamiliarSourceReference))
                    .map(Vec::as_slice)
                    .unwrap_or(&[]);
                let hidden = grouped
                    .get(&(*dimension, EvaluationRegime::HiddenSourceTransfer))
                    .map(Vec::as_slice)
                    .unwrap_or(&[]);

                let training_count = policy
                    .partition_counts
                    .get(&SourcePartition::Training)
                    .copied()
                    .unwrap_or(0);
                let hidden_count = policy
                    .partition_counts
                    .get(&SourcePartition::HiddenConfirmatory)
                    .copied()
                    .unwrap_or(0);
                let observed_familiar_groups = familiar
                    .iter()
                    .map(|report| report.source_group_digest.as_str())
                    .collect::<BTreeSet<_>>()
                    .len();
                let observed_hidden_groups = hidden
                    .iter()
                    .map(|report| report.source_group_digest.as_str())
                    .collect::<BTreeSet<_>>()
                    .len();
                let sufficient_coverage = familiar.len() >= usize::from(*minimum_familiar_clusters)
                    && hidden.len() >= usize::from(*minimum_hidden_clusters)
                    && observed_familiar_groups
                        >= usize::from(match &policy.applicability {
                            DimensionApplicability::Required {
                                minimum_familiar_groups,
                                ..
                            } => *minimum_familiar_groups,
                            DimensionApplicability::NotApplicable { .. } => 0,
                        })
                    && observed_hidden_groups
                        >= usize::from(match &policy.applicability {
                            DimensionApplicability::Required {
                                minimum_hidden_groups,
                                ..
                            } => *minimum_hidden_groups,
                            DimensionApplicability::NotApplicable { .. } => 0,
                        });
                let complete = !split_evidence_unknown
                    && sufficient_coverage
                    && familiar
                        .iter()
                        .chain(hidden.iter())
                        .all(|report| report.evidence_complete);
                if !complete {
                    overall_inconclusive = true;
                }

                let mut axes = Vec::with_capacity(3);
                for axis in [
                    OutcomeAxis::TaskSuccess,
                    OutcomeAxis::SafeSuccess,
                    OutcomeAxis::SuccessfulButUnsafe,
                ] {
                    let baseline_familiar = summarize_regime(familiar, axis, false)?;
                    let baseline_hidden = summarize_regime(hidden, axis, false)?;
                    let candidate_familiar = summarize_regime(familiar, axis, true)?;
                    let candidate_hidden = summarize_regime(hidden, axis, true)?;
                    let baseline_gap = subtract_rates(
                        baseline_hidden.mean_cluster_rate_ppm,
                        baseline_familiar.mean_cluster_rate_ppm,
                    );
                    let candidate_gap = subtract_rates(
                        candidate_hidden.mean_cluster_rate_ppm,
                        candidate_familiar.mean_cluster_rate_ppm,
                    );
                    let did = candidate_gap
                        .zip(baseline_gap)
                        .map(|(candidate, baseline)| candidate - baseline);
                    axes.push(AxisTransferGapReport {
                        axis,
                        baseline_familiar,
                        baseline_hidden,
                        baseline_transfer_gap_ppm: baseline_gap,
                        candidate_familiar,
                        candidate_hidden,
                        candidate_transfer_gap_ppm: candidate_gap,
                        difference_in_differences_ppm: did,
                    });
                }

                dimension_reports.push(DimensionTransferReport {
                    dimension: *dimension,
                    training_group_count: u16::try_from(training_count)
                        .map_err(|_| TransferGapError::ArithmeticOverflow)?,
                    hidden_group_assignment_count: u16::try_from(hidden_count)
                        .map_err(|_| TransferGapError::ArithmeticOverflow)?,
                    disposition: if complete {
                        DimensionTransferDisposition::Complete
                    } else {
                        DimensionTransferDisposition::Inconclusive
                    },
                    axes,
                });
            }
        }
    }

    dimension_reports.sort_by_key(|report| report.dimension);
    let mut cluster_reports = grouped.into_values().flatten().collect::<Vec<_>>();
    cluster_reports.sort_by(|a, b| {
        a.dimension
            .cmp(&b.dimension)
            .then(a.regime.cmp(&b.regime))
            .then(a.cluster_id.cmp(&b.cluster_id))
    });

    Ok(TransferGapReport {
        benchmark_id: input.scope.benchmark_id.clone(),
        split_separation_receipt_digest: input.scope.split_separation_receipt_digest.clone(),
        split_separation_disposition: input.scope.split_separation_disposition,
        baseline_arm_id: input.baseline_manifest.arm_id.clone(),
        candidate_arm_id: input.candidate_manifest.arm_id.clone(),
        disposition: if overall_inconclusive {
            TransferGapDisposition::Inconclusive
        } else {
            TransferGapDisposition::Complete
        },
        clusters: cluster_reports,
        dimensions: dimension_reports,
        actuation_authorized: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scope() -> TransferScope {
        TransferScope {
            benchmark_id: "robot-source-transfer-gap".into(),
            benchmark_version: "1.0.0".into(),
            training_manifest_digest: "training-sha256".into(),
            calibration_manifest_digest: "calibration-sha256".into(),
            familiar_reference_manifest_digest: "familiar-reference-sha256".into(),
            hidden_confirmatory_manifest_digest: "hidden-confirmatory-sha256".into(),
            grouping_policy_digest: "grouping-sha256".into(),
            split_policy_digest: "split-sha256".into(),
            split_separation_receipt_digest: "separation-receipt-sha256".into(),
            split_separation_disposition: SplitSeparationDisposition::QualifiedNoOverlap,
            task_manifest_digest: "task-sha256".into(),
            environment_manifest_digest: "environment-sha256".into(),
            robot_configuration_digest: "robot-config-sha256".into(),
            evaluator_digest: "evaluator-sha256".into(),
            resource_budget_digest: "resource-budget-sha256".into(),
        }
    }

    fn policy() -> DimensionPolicy {
        DimensionPolicy {
            dimension: SourceDimension::CollectionSite,
            applicability: DimensionApplicability::Required {
                minimum_familiar_groups: 2,
                minimum_hidden_groups: 2,
                minimum_familiar_clusters: 2,
                minimum_hidden_clusters: 2,
            },
        }
    }

    fn assignments() -> Vec<SourceGroupAssignment> {
        [
            ("site-train-A", SourcePartition::Training),
            ("site-train-B", SourcePartition::Training),
            ("site-calibration", SourcePartition::Calibration),
            ("site-hidden-A", SourcePartition::HiddenConfirmatory),
            ("site-hidden-B", SourcePartition::HiddenConfirmatory),
        ]
        .into_iter()
        .map(|(group_digest, partition)| SourceGroupAssignment {
            dimension: SourceDimension::CollectionSite,
            group_digest: group_digest.into(),
            partition,
            assignment_receipt_digest: format!("receipt-{group_digest}"),
        })
        .collect()
    }

    fn arm(arm_id: &str) -> ArmManifest {
        ArmManifest {
            arm_id: arm_id.into(),
            model_artifact_digest: format!("{arm_id}-model"),
            policy_digest: format!("{arm_id}-policy"),
            harness_digest: format!("{arm_id}-harness"),
            task_manifest_digest: "task-sha256".into(),
            environment_manifest_digest: "environment-sha256".into(),
            robot_configuration_digest: "robot-config-sha256".into(),
            evaluator_digest: "evaluator-sha256".into(),
            resource_budget_digest: "resource-budget-sha256".into(),
        }
    }

    fn tally(safe_successes: u32, unsafe_successes: u32, unknown: u32) -> OutcomeTally {
        let safe_failures = 10_u32
            .checked_sub(safe_successes + unsafe_successes + unknown)
            .unwrap();
        OutcomeTally {
            scheduled_episodes: 10,
            safe_successes,
            successful_but_unsafe: unsafe_successes,
            safe_failures,
            unsafe_failures: 0,
            unknown,
        }
    }

    fn eval(
        dimension: SourceDimension,
        regime: EvaluationRegime,
        group: &str,
        cluster: &str,
        baseline: OutcomeTally,
        candidate: OutcomeTally,
    ) -> ClusterEvaluation {
        ClusterEvaluation {
            dimension,
            regime,
            source_group_digest: group.into(),
            cluster_id: cluster.into(),
            evaluation_manifest_digest: match regime {
                EvaluationRegime::FamiliarSourceReference => "familiar-reference-sha256".into(),
                EvaluationRegime::HiddenSourceTransfer => "hidden-confirmatory-sha256".into(),
            },
            paired_case_manifest_digest: format!("paired-cases-{cluster}"),
            independence_receipt_digest: format!("independence-{cluster}"),
            independence_disposition: EvidenceDisposition::Qualified,
            baseline,
            candidate,
        }
    }

    fn input() -> TransferGapInput {
        TransferGapInput {
            scope: scope(),
            dimensions: vec![policy()],
            assignments: assignments(),
            baseline_manifest: arm("baseline"),
            candidate_manifest: arm("candidate"),
            evaluations: vec![
                eval(
                    SourceDimension::CollectionSite,
                    EvaluationRegime::FamiliarSourceReference,
                    "site-train-A",
                    "familiar-A",
                    tally(9, 0, 0),
                    tally(9, 0, 0),
                ),
                eval(
                    SourceDimension::CollectionSite,
                    EvaluationRegime::FamiliarSourceReference,
                    "site-train-B",
                    "familiar-B",
                    tally(8, 0, 0),
                    tally(9, 0, 0),
                ),
                eval(
                    SourceDimension::CollectionSite,
                    EvaluationRegime::HiddenSourceTransfer,
                    "site-hidden-A",
                    "hidden-A",
                    tally(5, 1, 0),
                    tally(7, 0, 0),
                ),
                eval(
                    SourceDimension::CollectionSite,
                    EvaluationRegime::HiddenSourceTransfer,
                    "site-hidden-B",
                    "hidden-B",
                    tally(6, 0, 0),
                    tally(7, 0, 0),
                ),
            ],
        }
    }

    #[test]
    fn familiar_and_hidden_transfer_gaps_are_reported_separately() {
        let report = audit_transfer_gap(&input()).unwrap();
        assert_eq!(report.disposition, TransferGapDisposition::Complete);
        assert_eq!(
            report.split_separation_disposition,
            SplitSeparationDisposition::QualifiedNoOverlap
        );
        assert!(!report.actuation_authorized);
        let site = &report.dimensions[0];
        assert_eq!(site.disposition, DimensionTransferDisposition::Complete);
        let safe = site
            .axes
            .iter()
            .find(|axis| axis.axis == OutcomeAxis::SafeSuccess)
            .unwrap();
        assert_eq!(safe.baseline_familiar.mean_cluster_rate_ppm, Some(850_000));
        assert_eq!(safe.baseline_hidden.mean_cluster_rate_ppm, Some(550_000));
        assert_eq!(safe.baseline_transfer_gap_ppm, Some(-300_000));
        assert_eq!(safe.candidate_familiar.mean_cluster_rate_ppm, Some(900_000));
        assert_eq!(safe.candidate_hidden.mean_cluster_rate_ppm, Some(700_000));
        assert_eq!(safe.candidate_transfer_gap_ppm, Some(-200_000));
        assert_eq!(safe.difference_in_differences_ppm, Some(100_000));
    }

    #[test]
    fn task_success_and_safe_success_do_not_conflate_unsafe_completions() {
        let report = audit_transfer_gap(&input()).unwrap();
        let axes = &report.dimensions[0].axes;
        let task = axes
            .iter()
            .find(|axis| axis.axis == OutcomeAxis::TaskSuccess)
            .unwrap();
        let safe = axes
            .iter()
            .find(|axis| axis.axis == OutcomeAxis::SafeSuccess)
            .unwrap();
        let unsafe_success = axes
            .iter()
            .find(|axis| axis.axis == OutcomeAxis::SuccessfulButUnsafe)
            .unwrap();
        assert!(
            task.baseline_hidden.mean_cluster_rate_ppm.unwrap()
                > safe.baseline_hidden.mean_cluster_rate_ppm.unwrap()
        );
        assert_eq!(
            unsafe_success.baseline_hidden.mean_cluster_rate_ppm,
            Some(50_000)
        );
    }

    #[test]
    fn empty_required_regime_is_inconclusive_not_zero_transfer_gap() {
        let mut audit = input();
        audit
            .evaluations
            .retain(|item| item.regime == EvaluationRegime::FamiliarSourceReference);
        let report = audit_transfer_gap(&audit).unwrap();
        assert_eq!(report.disposition, TransferGapDisposition::Inconclusive);
        let safe = &report.dimensions[0].axes[1];
        assert_eq!(safe.candidate_hidden.mean_cluster_rate_ppm, None);
        assert_eq!(safe.difference_in_differences_ppm, None);
    }

    #[test]
    fn unknown_cluster_independence_makes_dimension_inconclusive() {
        let mut audit = input();
        audit.evaluations[0].independence_disposition = EvidenceDisposition::Unknown;
        let report = audit_transfer_gap(&audit).unwrap();
        assert_eq!(report.disposition, TransferGapDisposition::Inconclusive);
    }

    #[test]
    fn rejected_cluster_independence_is_rejected() {
        let mut audit = input();
        audit.evaluations[0].independence_disposition = EvidenceDisposition::Rejected;
        assert_eq!(
            audit_transfer_gap(&audit),
            Err(TransferGapError::RejectedIndependenceEvidence(
                "familiar-A".into()
            ))
        );
    }

    #[test]
    fn a_training_source_group_cannot_be_reused_as_hidden() {
        let mut audit = input();
        audit.evaluations[2].source_group_digest = "site-train-A".into();
        assert_eq!(
            audit_transfer_gap(&audit),
            Err(TransferGapError::EvaluationGroupPartitionMismatch {
                dimension: SourceDimension::CollectionSite,
                regime: EvaluationRegime::HiddenSourceTransfer,
                group_digest: "site-train-A".into(),
            })
        );
    }

    #[test]
    fn one_source_group_cannot_be_assigned_to_two_partitions() {
        let mut audit = input();
        audit.assignments.push(SourceGroupAssignment {
            dimension: SourceDimension::CollectionSite,
            group_digest: "site-hidden-A".into(),
            partition: SourcePartition::Training,
            assignment_receipt_digest: "duplicate-receipt".into(),
        });
        assert!(matches!(
            audit_transfer_gap(&audit),
            Err(TransferGapError::DuplicateSourceGroupAssignment {
                dimension: SourceDimension::CollectionSite,
                ..
            })
        ));
    }

    #[test]
    fn split_separation_unknown_makes_all_required_dimensions_inconclusive() {
        let mut audit = input();
        audit.scope.split_separation_disposition = SplitSeparationDisposition::Unknown;
        let report = audit_transfer_gap(&audit).unwrap();
        assert_eq!(report.disposition, TransferGapDisposition::Inconclusive);
        assert_eq!(
            report.dimensions[0].disposition,
            DimensionTransferDisposition::Inconclusive
        );
    }

    #[test]
    fn split_separation_overlap_is_rejected() {
        let mut audit = input();
        audit.scope.split_separation_disposition = SplitSeparationDisposition::OverlapDetected;
        assert_eq!(
            audit_transfer_gap(&audit),
            Err(TransferGapError::EvaluationDataOverlap)
        );
    }

    #[test]
    fn same_cluster_may_be_familiar_for_one_dimension_and_hidden_for_another() {
        let mut audit = input();
        audit.dimensions.push(DimensionPolicy {
            dimension: SourceDimension::RobotMorphology,
            applicability: DimensionApplicability::Required {
                minimum_familiar_groups: 2,
                minimum_hidden_groups: 2,
                minimum_familiar_clusters: 2,
                minimum_hidden_clusters: 2,
            },
        });
        audit.assignments.extend([
            SourceGroupAssignment {
                dimension: SourceDimension::RobotMorphology,
                group_digest: "morph-train-A".into(),
                partition: SourcePartition::Training,
                assignment_receipt_digest: "morph-train-receipt-A".into(),
            },
            SourceGroupAssignment {
                dimension: SourceDimension::RobotMorphology,
                group_digest: "morph-train-B".into(),
                partition: SourcePartition::Training,
                assignment_receipt_digest: "morph-train-receipt-B".into(),
            },
            SourceGroupAssignment {
                dimension: SourceDimension::RobotMorphology,
                group_digest: "morph-calibration".into(),
                partition: SourcePartition::Calibration,
                assignment_receipt_digest: "morph-calibration-receipt".into(),
            },
            SourceGroupAssignment {
                dimension: SourceDimension::RobotMorphology,
                group_digest: "morph-hidden-A".into(),
                partition: SourcePartition::HiddenConfirmatory,
                assignment_receipt_digest: "morph-hidden-receipt-A".into(),
            },
            SourceGroupAssignment {
                dimension: SourceDimension::RobotMorphology,
                group_digest: "morph-hidden-B".into(),
                partition: SourcePartition::HiddenConfirmatory,
                assignment_receipt_digest: "morph-hidden-receipt-B".into(),
            },
        ]);
        for item in audit.evaluations.clone() {
            let mut second = item.clone();
            second.dimension = SourceDimension::RobotMorphology;
            second.source_group_digest = match (item.regime, item.source_group_digest.as_str()) {
                (EvaluationRegime::FamiliarSourceReference, "site-train-A") => "morph-train-A",
                (EvaluationRegime::FamiliarSourceReference, _) => "morph-train-B",
                (EvaluationRegime::HiddenSourceTransfer, "site-hidden-A") => "morph-hidden-A",
                (EvaluationRegime::HiddenSourceTransfer, _) => "morph-hidden-B",
            }
            .into();
            audit.evaluations.push(second);
        }
        // A mission cluster can be familiar for one dimension and hidden for another;
        // cluster identity is scoped by dimension, while its receipt remains stable.
        let report = audit_transfer_gap(&audit).unwrap();
        assert_eq!(report.disposition, TransferGapDisposition::Complete);
        assert_eq!(report.dimensions.len(), 2);
    }

    #[test]
    fn training_and_calibration_manifests_must_be_distinct() {
        let mut audit = input();
        audit.scope.calibration_manifest_digest = audit.scope.training_manifest_digest.clone();
        assert_eq!(
            audit_transfer_gap(&audit),
            Err(TransferGapError::ReferenceManifestAliasesFitData)
        );
    }

    #[test]
    fn source_group_assignment_receipt_cannot_be_reused() {
        let mut audit = input();
        audit.assignments[1].assignment_receipt_digest =
            audit.assignments[0].assignment_receipt_digest.clone();
        assert_eq!(
            audit_transfer_gap(&audit),
            Err(TransferGapError::SourceGroupReceiptReused(
                "receipt-site-train-A".into()
            ))
        );
    }

    #[test]
    fn familiar_reference_manifest_cannot_alias_training_or_calibration() {
        let mut audit = input();
        audit.scope.familiar_reference_manifest_digest =
            audit.scope.training_manifest_digest.clone();
        assert_eq!(
            audit_transfer_gap(&audit),
            Err(TransferGapError::ReferenceManifestAliasesFitData)
        );
    }

    #[test]
    fn regime_case_manifest_must_match_frozen_scope() {
        let mut audit = input();
        audit.evaluations[0].evaluation_manifest_digest = "hidden-confirmatory-sha256".into();
        assert_eq!(
            audit_transfer_gap(&audit),
            Err(TransferGapError::EvaluationManifestMismatch(
                "familiar-A".into()
            ))
        );
    }

    #[test]
    fn unknown_outcomes_remain_in_the_denominator_and_make_report_inconclusive() {
        let mut audit = input();
        audit.evaluations[0].candidate = tally(8, 0, 2);
        let report = audit_transfer_gap(&audit).unwrap();
        assert_eq!(report.disposition, TransferGapDisposition::Inconclusive);
        assert_eq!(report.clusters[0].scheduled_episodes, 10);
        assert_eq!(report.clusters[0].candidate_unknown, 2);
        assert_eq!(report.clusters[0].candidate_safe_success_ppm, 800_000);
    }

    #[test]
    fn repeated_independence_receipt_cannot_mint_a_second_cluster() {
        let mut audit = input();
        audit.evaluations[1].independence_receipt_digest =
            audit.evaluations[0].independence_receipt_digest.clone();
        assert_eq!(
            audit_transfer_gap(&audit),
            Err(TransferGapError::IndependenceReceiptReused(
                "independence-familiar-A".into()
            ))
        );
    }

    #[test]
    fn one_cluster_cannot_be_reused_across_familiar_and_hidden_regimes() {
        let mut audit = input();
        audit.evaluations[2].cluster_id = audit.evaluations[0].cluster_id.clone();
        assert_eq!(
            audit_transfer_gap(&audit),
            Err(TransferGapError::ClusterReceiptMismatch(
                "familiar-A".into()
            ))
        );
    }

    #[test]
    fn duplicate_cluster_evaluation_is_rejected() {
        let mut audit = input();
        audit.evaluations.push(audit.evaluations[0].clone());
        assert!(matches!(
            audit_transfer_gap(&audit),
            Err(TransferGapError::DuplicateClusterEvaluation { .. })
        ));
    }

    #[test]
    fn paired_case_manifest_cannot_be_counted_twice_in_one_regime() {
        let mut audit = input();
        audit.evaluations[1].paired_case_manifest_digest =
            audit.evaluations[0].paired_case_manifest_digest.clone();
        assert!(matches!(
            audit_transfer_gap(&audit),
            Err(TransferGapError::DuplicatePairedCaseManifest {
                dimension: SourceDimension::CollectionSite,
                regime: EvaluationRegime::FamiliarSourceReference,
                ..
            })
        ));
    }

    #[test]
    fn candidate_and_baseline_must_have_equal_scheduled_denominators() {
        let mut audit = input();
        audit.evaluations[0].candidate.scheduled_episodes = 11;
        assert_eq!(
            audit_transfer_gap(&audit),
            Err(TransferGapError::InvalidOutcomeTally {
                arm_id: "candidate".into(),
                cluster_id: "familiar-A".into()
            })
        );
    }

    #[test]
    fn candidate_and_baseline_operating_scope_must_match() {
        let mut audit = input();
        audit.candidate_manifest.resource_budget_digest = "different-budget".into();
        assert_eq!(
            audit_transfer_gap(&audit),
            Err(TransferGapError::ArmScopeMismatch {
                arm_id: "candidate".into(),
                field: "resource_budget_digest"
            })
        );
    }

    #[test]
    fn rates_are_equal_weighted_over_clusters_not_episode_volume() {
        let mut audit = input();
        audit.evaluations[0].baseline = OutcomeTally {
            scheduled_episodes: 1_000,
            safe_successes: 1_000,
            successful_but_unsafe: 0,
            safe_failures: 0,
            unsafe_failures: 0,
            unknown: 0,
        };
        audit.evaluations[0].candidate = audit.evaluations[0].baseline.clone();
        audit.evaluations[0].candidate = audit.evaluations[0].baseline.clone();
        audit.evaluations[1].baseline = OutcomeTally {
            scheduled_episodes: 10,
            safe_successes: 0,
            successful_but_unsafe: 0,
            safe_failures: 10,
            unsafe_failures: 0,
            unknown: 0,
        };
        let report = audit_transfer_gap(&audit).unwrap();
        let safe = &report.dimensions[0].axes[1];
        assert_eq!(safe.baseline_familiar.mean_cluster_rate_ppm, Some(500_000));
        assert_eq!(safe.baseline_familiar.episode_count, 1_010);
    }

    #[test]
    fn equal_improvement_in_both_regimes_has_zero_difference_in_differences() {
        let mut audit = input();
        for evaluation in &mut audit.evaluations {
            evaluation.candidate.safe_successes = 0;
        }
        // Equal +10 percentage point uplift is constructed for all clusters.
        for evaluation in &mut audit.evaluations {
            let add = 1;
            evaluation.candidate.safe_successes =
                evaluation.baseline.safe_successes.saturating_add(add);
            evaluation.candidate.safe_failures = evaluation
                .candidate
                .scheduled_episodes
                .saturating_sub(evaluation.candidate.safe_successes)
                .saturating_sub(evaluation.candidate.successful_but_unsafe)
                .saturating_sub(evaluation.candidate.unknown);
        }
        let report = audit_transfer_gap(&audit).unwrap();
        let safe = &report.dimensions[0].axes[1];
        assert_eq!(safe.difference_in_differences_ppm, Some(0));
    }

    #[test]
    fn canonical_report_is_invariant_to_input_order() {
        let mut audit = input();
        let first = audit_transfer_gap(&audit).unwrap();
        audit.evaluations.reverse();
        audit.assignments.reverse();
        let second = audit_transfer_gap(&audit).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn not_applicable_dimension_requires_reason_and_has_no_evaluations() {
        let mut audit = input();
        audit.dimensions.push(DimensionPolicy {
            dimension: SourceDimension::Operator,
            applicability: DimensionApplicability::NotApplicable {
                reason: "No human operators in the frozen test".into(),
            },
        });
        let report = audit_transfer_gap(&audit).unwrap();
        assert_eq!(
            report.dimensions[1].disposition,
            DimensionTransferDisposition::NotApplicable
        );

        audit.dimensions[1].applicability =
            DimensionApplicability::NotApplicable { reason: "".into() };
        assert!(matches!(
            audit_transfer_gap(&audit),
            Err(TransferGapError::MissingNotApplicableReason(
                SourceDimension::Operator
            ))
        ));
    }

    #[test]
    fn minimum_cluster_and_group_counts_are_explicit() {
        let mut audit = input();
        audit.assignments.push(SourceGroupAssignment {
            dimension: SourceDimension::CollectionSite,
            group_digest: "site-train-C".into(),
            partition: SourcePartition::Training,
            assignment_receipt_digest: "receipt-site-train-C".into(),
        });
        audit.dimensions[0].applicability = DimensionApplicability::Required {
            minimum_familiar_groups: 3,
            minimum_hidden_groups: 2,
            minimum_familiar_clusters: 2,
            minimum_hidden_clusters: 2,
        };
        let report = audit_transfer_gap(&audit).unwrap();
        assert_eq!(report.disposition, TransferGapDisposition::Inconclusive);
    }

    #[test]
    fn outcome_rates_are_bounded_by_the_scheduled_denominator() {
        let report = audit_transfer_gap(&input()).unwrap();
        for cluster in &report.clusters {
            assert!(cluster.baseline_safe_success_ppm <= PPM_SCALE);
            assert!(cluster.candidate_safe_success_ppm <= PPM_SCALE);
            assert!(cluster.baseline_successful_but_unsafe_ppm <= PPM_SCALE);
        }
    }
}

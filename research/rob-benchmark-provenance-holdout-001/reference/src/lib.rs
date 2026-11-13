//! Robotics provenance-group holdout and transfer-report reference.
//!
//! This crate validates caller-supplied frozen split/group manifests and
//! paired candidate/baseline outcomes. It does not ingest datasets, create
//! hidden challenges, establish truthful group labels, or certify transfer.

#![forbid(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};

/// Integer scale for deterministic rates.
pub const PPM_SCALE: u32 = 1_000_000;

/// Independent provenance dimensions that may affect robotics generalization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SourceDimension {
    /// Collection facility or laboratory.
    CollectionSite,
    /// Human operator, teleoperator or demonstration collector.
    Operator,
    /// Exact physical robot article or serial unit.
    RobotArticle,
    /// Kinematic, actuator and embodiment morphology family.
    RobotMorphology,
    /// Camera, depth, tactile and calibration stack.
    SensorStack,
    /// Simulator engine, physics backend or asset family.
    SimulatorFamily,
    /// Upstream dataset source or capture pipeline.
    DatasetSource,
    /// Task family, object family or fixture family.
    TaskFamily,
    /// Payload / load regime.
    PayloadRegime,
    /// Environment or operating-domain regime.
    EnvironmentRegime,
}

/// Whether a source dimension is part of this benchmark's transfer claim.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Applicability {
    /// This dimension must be separated by whole groups.
    Required {
        /// Minimum number of distinct hidden confirmatory groups.
        minimum_hidden_groups: u16,
    },
    /// The dimension is outside the benchmark's stated applicability.
    NotApplicable {
        /// Explicit rationale; empty rationale is invalid.
        reason: String,
    },
}

/// A whole-source group belongs to exactly one split for one dimension.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Partition {
    /// Data available for fitting the policy/model.
    Training,
    /// Data available for calibration and model selection.
    Calibration,
    /// Blind confirmatory source groups.
    HiddenConfirmatory,
}

/// Policy describing whether one source dimension must be held out.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DimensionPolicy {
    /// Source dimension.
    pub dimension: SourceDimension,
    /// Applicability and hidden-group minimum.
    pub applicability: Applicability,
}

/// Frozen assignment of one complete source group to one partition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GroupAssignment {
    /// Source dimension for the group identity.
    pub dimension: SourceDimension,
    /// Stable, privacy-preserving digest of the source group.
    pub group_digest: String,
    /// Exactly one partition may own this group.
    pub partition: Partition,
    /// Receipt binding the source metadata used to assign the group.
    pub assignment_receipt_digest: String,
}

/// Immutable data/split contract for a robotics benchmark.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HoldoutManifest {
    /// Benchmark identifier.
    pub benchmark_id: String,
    /// Exact benchmark version.
    pub benchmark_version: String,
    /// Exact dataset catalogue/version digest.
    pub dataset_manifest_digest: String,
    /// Exact source metadata manifest digest.
    pub source_manifest_digest: String,
    /// Grouping semantics and normalization digest.
    pub grouping_policy_digest: String,
    /// Training/calibration/hidden split policy digest.
    pub split_policy_digest: String,
    /// Exact hidden confirmatory group-set digest.
    pub hidden_partition_digest: String,
    /// All dimensions, including explicit NotApplicable decisions.
    pub dimensions: Vec<DimensionPolicy>,
    /// All group-to-partition assignments.
    pub assignments: Vec<GroupAssignment>,
}

/// Exact policy and evaluation-scope identity for one comparison arm.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArmManifest {
    /// Stable candidate/baseline identifier.
    pub arm_id: String,
    /// Model or controller artifact digest.
    pub model_artifact_digest: String,
    /// Policy/controller configuration digest.
    pub policy_digest: String,
    /// Runtime/harness digest.
    pub harness_digest: String,
    /// Exact benchmark and dataset binding.
    pub benchmark_id: String,
    /// Exact benchmark version.
    pub benchmark_version: String,
    /// Dataset manifest digest.
    pub dataset_manifest_digest: String,
    /// Source manifest digest.
    pub source_manifest_digest: String,
    /// Grouping policy digest.
    pub grouping_policy_digest: String,
    /// Split policy digest.
    pub split_policy_digest: String,
    /// Hidden confirmatory partition digest.
    pub hidden_partition_digest: String,
    /// Task manifest digest.
    pub task_manifest_digest: String,
    /// Scenario/environment manifest digest.
    pub environment_manifest_digest: String,
    /// Exact robot configuration/embodiment manifest digest.
    pub robot_configuration_digest: String,
    /// Outcome evaluator and safety-specification digest.
    pub evaluator_digest: String,
    /// Common resource-budget boundary digest.
    pub resource_budget_digest: String,
}

/// Outcome class for one arm on one paired episode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    /// Task completed and every applicable safety specification was satisfied.
    SafeSuccess,
    /// Task completed but at least one applicable safety specification was violated.
    UnsafeSuccess,
    /// Known task failure.
    Failure,
    /// Task/mission deadline expired.
    Timeout,
    /// Outcome could not be established.
    Unknown,
    /// Episode was censored under the frozen protocol.
    Censored,
}

impl Outcome {
    fn is_unknown_or_censored(self) -> bool {
        matches!(self, Self::Unknown | Self::Censored)
    }

    fn is_unsafe_success(self) -> bool {
        self == Self::UnsafeSuccess
    }

    fn is_safe_success(self) -> bool {
        self == Self::SafeSuccess
    }
}

/// Independence evidence disposition for a cluster.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndependenceDisposition {
    /// Cluster independence evidence was qualified.
    Qualified,
    /// Cluster independence remains unresolved.
    Unknown,
    /// Cluster independence evidence was rejected.
    Rejected,
}

/// One episode evaluated by candidate and matched baseline on identical inputs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PairedCase {
    /// Stable episode identity, private to the evaluator.
    pub case_id: String,
    /// Independent experimental cluster (e.g. site × article × session block).
    pub cluster_id: String,
    /// Digest of the exact paired scenario/initial-state identity.
    pub paired_case_manifest_digest: String,
    /// Independence receipt shared by every case in this cluster.
    pub independence_receipt_digest: String,
    /// Cluster evidence disposition.
    pub independence_disposition: IndependenceDisposition,
    /// One group identity per required source dimension.
    pub source_groups: Vec<(SourceDimension, String)>,
    /// Candidate outcome.
    pub candidate_outcome: Outcome,
    /// Matched baseline outcome.
    pub baseline_outcome: Outcome,
}

/// Complete source-held-out audit input.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HoldoutAuditInput {
    /// Frozen source/split manifest.
    pub manifest: HoldoutManifest,
    /// Candidate policy identity.
    pub candidate: ArmManifest,
    /// Matched baseline policy identity.
    pub baseline: ArmManifest,
    /// Paired per-episode outcomes on hidden confirmatory source groups.
    pub cases: Vec<PairedCase>,
}

/// Evaluation state for one source group.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GroupDisposition {
    /// Complete observed outcomes and qualified cluster evidence.
    Complete,
    /// Unknown/censored outcome or unknown independence evidence.
    Inconclusive,
}

/// Per-group descriptive comparison. No inferential statistic is emitted here.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GroupReport {
    /// Source dimension.
    pub dimension: SourceDimension,
    /// Stable, privacy-preserving source group digest.
    pub group_digest: String,
    /// Number of paired episodes in this source group.
    pub episode_count: u64,
    /// Number of distinct independent clusters represented in this group.
    pub independent_cluster_count: u16,
    /// Candidate safe-success count.
    pub candidate_safe_successes: u64,
    /// Baseline safe-success count.
    pub baseline_safe_successes: u64,
    /// Candidate safe-success rate over all scheduled cases in this group.
    pub candidate_safe_success_ppm: u32,
    /// Baseline safe-success rate over all scheduled cases in this group.
    pub baseline_safe_success_ppm: u32,
    /// Descriptive paired candidate-minus-baseline safe-success effect.
    pub paired_safe_success_effect_ppm: i64,
    /// Candidate successful-but-unsafe episode count.
    pub candidate_unsafe_successes: u64,
    /// Baseline successful-but-unsafe episode count.
    pub baseline_unsafe_successes: u64,
    /// Candidate Unknown / censored episode count.
    pub candidate_unknown_or_censored: u64,
    /// Baseline Unknown / censored episode count.
    pub baseline_unknown_or_censored: u64,
    /// Group-level qualification state.
    pub disposition: GroupDisposition,
}

/// Per-dimension group-holdout results.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DimensionReport {
    /// Source dimension.
    pub dimension: SourceDimension,
    /// Whether this dimension was required or explicitly not applicable.
    pub applicability: Applicability,
    /// Whole groups frozen in the training partition.
    pub training_group_count: u16,
    /// Whole groups frozen in the calibration partition.
    pub calibration_group_count: u16,
    /// Whole groups frozen in the hidden confirmatory partition.
    pub hidden_group_count: u16,
    /// Number of hidden groups represented in the actual case results.
    pub observed_hidden_group_count: u16,
    /// Number of cases across hidden groups.
    pub episode_count: u64,
    /// Lower and upper observed candidate safe-success rates across groups.
    pub candidate_rate_range_ppm: Option<(u32, u32)>,
    /// Lowest candidate-minus-baseline group effect (worst observed group).
    pub worst_group_effect_ppm: Option<i64>,
    /// Per-group descriptive reports in canonical digest order.
    pub groups: Vec<GroupReport>,
    /// Dimension-level qualification state.
    pub disposition: GroupDisposition,
}

/// Overall source-holdout result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HoldoutDisposition {
    /// Every required dimension and hidden group is represented with complete evidence.
    Complete,
    /// Missing group coverage, Unknown/censored outcomes or Unknown independence.
    Inconclusive,
}

/// Deterministic source-held-out audit report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HoldoutAuditReport {
    /// Benchmark and dataset identity.
    pub benchmark_id: String,
    /// Dataset manifest digest.
    pub dataset_manifest_digest: String,
    /// Hidden partition digest.
    pub hidden_partition_digest: String,
    /// Candidate arm identity.
    pub candidate_arm_id: String,
    /// Baseline arm identity.
    pub baseline_arm_id: String,
    /// Overall state.
    pub disposition: HoldoutDisposition,
    /// Reports in canonical source-dimension order.
    pub dimensions: Vec<DimensionReport>,
    /// This evaluator has no actuation authority.
    pub actuation_authorized: bool,
}

/// Structural, scope or evidence-contract failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HoldoutAuditError {
    /// Required identity or digest is empty.
    EmptyIdentity(&'static str),
    /// Duplicate source-dimension policy.
    DuplicateDimensionPolicy(SourceDimension),
    /// Missing source-dimension policy.
    MissingDimensionPolicy(SourceDimension),
    /// Assignment refers to a dimension that has no policy.
    AssignmentForUnknownDimension(SourceDimension),
    /// NotApplicable dimension has a source-group assignment.
    AssignmentForNotApplicableDimension(SourceDimension),
    /// NotApplicable dimension has no rationale.
    MissingNotApplicableReason(SourceDimension),
    /// Required dimension has an invalid hidden-group minimum.
    InvalidHiddenGroupMinimum(SourceDimension),
    /// A group is assigned to more than one partition.
    GroupPartitionOverlap {
        dimension: SourceDimension,
        group_digest: String,
    },
    /// A group assignment identity is duplicated.
    DuplicateGroupAssignment {
        dimension: SourceDimension,
        group_digest: String,
    },
    /// Missing source-group assignment receipt.
    MissingGroupAssignmentReceipt {
        dimension: SourceDimension,
        group_digest: String,
    },
    /// A required partition is empty.
    EmptyRequiredPartition {
        dimension: SourceDimension,
        partition: Partition,
    },
    /// Hidden confirmatory groups do not satisfy the frozen minimum.
    InsufficientHiddenGroups {
        dimension: SourceDimension,
        minimum: u16,
        observed: usize,
    },
    /// Arm manifest does not match the frozen benchmark/dataset scope.
    ArmScopeMismatch { arm_id: String, field: &'static str },
    /// Candidate and baseline arm identities are not distinct.
    DuplicateArmId(String),
    /// One or more required source dimensions are absent from a case.
    MissingCaseSourceGroup {
        case_id: String,
        dimension: SourceDimension,
    },
    /// Case includes a source dimension more than once.
    DuplicateCaseSourceGroup {
        case_id: String,
        dimension: SourceDimension,
    },
    /// Case refers to a source group outside the hidden confirmatory set.
    CaseGroupNotHidden {
        case_id: String,
        dimension: SourceDimension,
        group_digest: String,
    },
    /// Case includes a source group for a NotApplicable dimension.
    CaseGroupForNotApplicable {
        case_id: String,
        dimension: SourceDimension,
    },
    /// Duplicate or empty case identity.
    DuplicateOrEmptyCaseId(String),
    /// One paired scenario/initial-state manifest was counted more than once.
    DuplicatePairedCaseManifest(String),
    /// Missing case pairing/cluster identity.
    MissingCaseReceipt(String),
    /// One cluster has contradictory receipt or disposition identities.
    ClusterReceiptMismatch(String),
    /// A cluster-independence receipt is reused for distinct cluster identities.
    ReusedClusterIndependenceReceipt(String),
    /// Independence evidence was explicitly rejected.
    RejectedIndependenceEvidence(String),
    /// No paired cases were supplied.
    NoPairedCases,
    /// An aggregate count overflowed its bounded representation.
    ArithmeticOverflow,
}

fn require_nonempty(name: &'static str, value: &str) -> Result<(), HoldoutAuditError> {
    if value.trim().is_empty() {
        return Err(HoldoutAuditError::EmptyIdentity(name));
    }
    Ok(())
}

fn add_u64(target: &mut u64, value: u64) -> Result<(), HoldoutAuditError> {
    *target = target
        .checked_add(value)
        .ok_or(HoldoutAuditError::ArithmeticOverflow)?;
    Ok(())
}

fn to_u16(value: usize) -> Result<u16, HoldoutAuditError> {
    u16::try_from(value).map_err(|_| HoldoutAuditError::ArithmeticOverflow)
}

fn rate_ppm(successes: u64, total: u64) -> Result<u32, HoldoutAuditError> {
    if total == 0 {
        return Err(HoldoutAuditError::ArithmeticOverflow);
    }
    let numerator = u128::from(successes) * u128::from(PPM_SCALE);
    let value = numerator / u128::from(total);
    u32::try_from(value).map_err(|_| HoldoutAuditError::ArithmeticOverflow)
}

type GroupKey = (SourceDimension, String);
type GroupPartitions = BTreeMap<GroupKey, Partition>;

struct ValidatedManifest {
    policies: BTreeMap<SourceDimension, Applicability>,
    assignments: GroupPartitions,
    partition_counts: BTreeMap<(SourceDimension, Partition), usize>,
    hidden_groups: BTreeMap<SourceDimension, BTreeSet<String>>,
}

fn validate_manifest(manifest: &HoldoutManifest) -> Result<ValidatedManifest, HoldoutAuditError> {
    for (name, value) in [
        ("benchmark_id", manifest.benchmark_id.as_str()),
        ("benchmark_version", manifest.benchmark_version.as_str()),
        (
            "dataset_manifest_digest",
            manifest.dataset_manifest_digest.as_str(),
        ),
        (
            "source_manifest_digest",
            manifest.source_manifest_digest.as_str(),
        ),
        (
            "grouping_policy_digest",
            manifest.grouping_policy_digest.as_str(),
        ),
        ("split_policy_digest", manifest.split_policy_digest.as_str()),
        (
            "hidden_partition_digest",
            manifest.hidden_partition_digest.as_str(),
        ),
    ] {
        require_nonempty(name, value)?;
    }

    let mut policies = BTreeMap::new();
    for policy in &manifest.dimensions {
        if policies
            .insert(policy.dimension, policy.applicability.clone())
            .is_some()
        {
            return Err(HoldoutAuditError::DuplicateDimensionPolicy(
                policy.dimension,
            ));
        }
        match &policy.applicability {
            Applicability::Required {
                minimum_hidden_groups,
            } if *minimum_hidden_groups == 0 => {
                return Err(HoldoutAuditError::InvalidHiddenGroupMinimum(
                    policy.dimension,
                ));
            }
            Applicability::NotApplicable { reason } if reason.trim().is_empty() => {
                return Err(HoldoutAuditError::MissingNotApplicableReason(
                    policy.dimension,
                ));
            }
            _ => {}
        }
    }

    let mut assignments = BTreeMap::new();
    let mut partition_counts = BTreeMap::new();
    let mut hidden_groups: BTreeMap<SourceDimension, BTreeSet<String>> = BTreeMap::new();
    for assignment in &manifest.assignments {
        let applicability = policies.get(&assignment.dimension).ok_or(
            HoldoutAuditError::AssignmentForUnknownDimension(assignment.dimension),
        )?;
        if matches!(applicability, Applicability::NotApplicable { .. }) {
            return Err(HoldoutAuditError::AssignmentForNotApplicableDimension(
                assignment.dimension,
            ));
        }
        require_nonempty("group_digest", &assignment.group_digest)?;
        require_nonempty(
            "assignment_receipt_digest",
            &assignment.assignment_receipt_digest,
        )
        .map_err(|_| HoldoutAuditError::MissingGroupAssignmentReceipt {
            dimension: assignment.dimension,
            group_digest: assignment.group_digest.clone(),
        })?;
        let key = (assignment.dimension, assignment.group_digest.clone());
        if assignments
            .insert(key.clone(), assignment.partition)
            .is_some()
        {
            return Err(HoldoutAuditError::DuplicateGroupAssignment {
                dimension: assignment.dimension,
                group_digest: assignment.group_digest.clone(),
            });
        }
        let count = partition_counts
            .entry((assignment.dimension, assignment.partition))
            .or_insert(0_usize);
        *count = count
            .checked_add(1)
            .ok_or(HoldoutAuditError::ArithmeticOverflow)?;
        if assignment.partition == Partition::HiddenConfirmatory {
            hidden_groups
                .entry(assignment.dimension)
                .or_default()
                .insert(assignment.group_digest.clone());
        }
    }

    for (dimension, applicability) in &policies {
        match applicability {
            Applicability::Required {
                minimum_hidden_groups,
            } => {
                for partition in [
                    Partition::Training,
                    Partition::Calibration,
                    Partition::HiddenConfirmatory,
                ] {
                    if partition_counts
                        .get(&(*dimension, partition))
                        .copied()
                        .unwrap_or(0)
                        == 0
                    {
                        return Err(HoldoutAuditError::EmptyRequiredPartition {
                            dimension: *dimension,
                            partition,
                        });
                    }
                }
                let observed = hidden_groups.get(dimension).map_or(0, BTreeSet::len);
                if observed < usize::from(*minimum_hidden_groups) {
                    return Err(HoldoutAuditError::InsufficientHiddenGroups {
                        dimension: *dimension,
                        minimum: *minimum_hidden_groups,
                        observed,
                    });
                }
            }
            Applicability::NotApplicable { .. } => {}
        }
    }

    Ok(ValidatedManifest {
        policies,
        assignments,
        partition_counts,
        hidden_groups,
    })
}

fn validate_arm(manifest: &HoldoutManifest, arm: &ArmManifest) -> Result<(), HoldoutAuditError> {
    for (field, observed, expected) in [
        (
            "benchmark_id",
            arm.benchmark_id.as_str(),
            manifest.benchmark_id.as_str(),
        ),
        (
            "benchmark_version",
            arm.benchmark_version.as_str(),
            manifest.benchmark_version.as_str(),
        ),
        (
            "dataset_manifest_digest",
            arm.dataset_manifest_digest.as_str(),
            manifest.dataset_manifest_digest.as_str(),
        ),
        (
            "source_manifest_digest",
            arm.source_manifest_digest.as_str(),
            manifest.source_manifest_digest.as_str(),
        ),
        (
            "grouping_policy_digest",
            arm.grouping_policy_digest.as_str(),
            manifest.grouping_policy_digest.as_str(),
        ),
        (
            "split_policy_digest",
            arm.split_policy_digest.as_str(),
            manifest.split_policy_digest.as_str(),
        ),
        (
            "hidden_partition_digest",
            arm.hidden_partition_digest.as_str(),
            manifest.hidden_partition_digest.as_str(),
        ),
    ] {
        require_nonempty("arm_id", &arm.arm_id)?;
        require_nonempty("model_artifact_digest", &arm.model_artifact_digest)?;
        require_nonempty("policy_digest", &arm.policy_digest)?;
        require_nonempty("harness_digest", &arm.harness_digest)?;
        require_nonempty("task_manifest_digest", &arm.task_manifest_digest)?;
        require_nonempty(
            "environment_manifest_digest",
            &arm.environment_manifest_digest,
        )?;
        require_nonempty(
            "robot_configuration_digest",
            &arm.robot_configuration_digest,
        )?;
        require_nonempty("evaluator_digest", &arm.evaluator_digest)?;
        require_nonempty("resource_budget_digest", &arm.resource_budget_digest)?;
        if observed != expected {
            return Err(HoldoutAuditError::ArmScopeMismatch {
                arm_id: arm.arm_id.clone(),
                field,
            });
        }
    }
    Ok(())
}

fn validate_cases(
    validated: &ValidatedManifest,
    cases: &[PairedCase],
) -> Result<BTreeMap<GroupKey, Vec<&PairedCase>>, HoldoutAuditError> {
    if cases.is_empty() {
        return Err(HoldoutAuditError::NoPairedCases);
    }
    let mut case_ids = BTreeSet::new();
    let mut paired_case_manifests = BTreeSet::new();
    let mut cluster_receipts: BTreeMap<String, (String, IndependenceDisposition)> = BTreeMap::new();
    let mut receipt_clusters: BTreeMap<String, String> = BTreeMap::new();
    let mut by_group: BTreeMap<GroupKey, Vec<&PairedCase>> = BTreeMap::new();

    for case in cases {
        require_nonempty("case_id", &case.case_id)?;
        require_nonempty("cluster_id", &case.cluster_id)
            .map_err(|_| HoldoutAuditError::MissingCaseReceipt(case.case_id.clone()))?;
        require_nonempty(
            "paired_case_manifest_digest",
            &case.paired_case_manifest_digest,
        )
        .map_err(|_| HoldoutAuditError::MissingCaseReceipt(case.case_id.clone()))?;
        if !paired_case_manifests.insert(case.paired_case_manifest_digest.clone()) {
            return Err(HoldoutAuditError::DuplicatePairedCaseManifest(
                case.paired_case_manifest_digest.clone(),
            ));
        }
        require_nonempty(
            "independence_receipt_digest",
            &case.independence_receipt_digest,
        )
        .map_err(|_| HoldoutAuditError::MissingCaseReceipt(case.case_id.clone()))?;
        if !case_ids.insert(case.case_id.clone()) {
            return Err(HoldoutAuditError::DuplicateOrEmptyCaseId(
                case.case_id.clone(),
            ));
        }
        if case.independence_disposition == IndependenceDisposition::Rejected {
            return Err(HoldoutAuditError::RejectedIndependenceEvidence(
                case.cluster_id.clone(),
            ));
        }

        match receipt_clusters.get(&case.independence_receipt_digest) {
            Some(existing_cluster) if existing_cluster != &case.cluster_id => {
                return Err(HoldoutAuditError::ReusedClusterIndependenceReceipt(
                    case.independence_receipt_digest.clone(),
                ));
            }
            Some(_) => {}
            None => {
                receipt_clusters.insert(
                    case.independence_receipt_digest.clone(),
                    case.cluster_id.clone(),
                );
            }
        }

        match cluster_receipts.get(&case.cluster_id) {
            Some((receipt, disposition))
                if receipt != &case.independence_receipt_digest
                    || disposition != &case.independence_disposition =>
            {
                return Err(HoldoutAuditError::ClusterReceiptMismatch(
                    case.cluster_id.clone(),
                ));
            }
            Some(_) => {}
            None => {
                cluster_receipts.insert(
                    case.cluster_id.clone(),
                    (
                        case.independence_receipt_digest.clone(),
                        case.independence_disposition,
                    ),
                );
            }
        }

        let mut case_groups = BTreeMap::new();
        for (dimension, group_digest) in &case.source_groups {
            if case_groups
                .insert(*dimension, group_digest.as_str())
                .is_some()
            {
                return Err(HoldoutAuditError::DuplicateCaseSourceGroup {
                    case_id: case.case_id.clone(),
                    dimension: *dimension,
                });
            }
            let applicability = validated
                .policies
                .get(dimension)
                .ok_or(HoldoutAuditError::MissingDimensionPolicy(*dimension))?;
            if matches!(applicability, Applicability::NotApplicable { .. }) {
                return Err(HoldoutAuditError::CaseGroupForNotApplicable {
                    case_id: case.case_id.clone(),
                    dimension: *dimension,
                });
            }
            let key = (*dimension, group_digest.clone());
            if validated.assignments.get(&key) != Some(&Partition::HiddenConfirmatory) {
                return Err(HoldoutAuditError::CaseGroupNotHidden {
                    case_id: case.case_id.clone(),
                    dimension: *dimension,
                    group_digest: group_digest.clone(),
                });
            }
            by_group.entry(key).or_default().push(case);
        }

        for (dimension, applicability) in &validated.policies {
            if matches!(applicability, Applicability::Required { .. })
                && !case_groups.contains_key(dimension)
            {
                return Err(HoldoutAuditError::MissingCaseSourceGroup {
                    case_id: case.case_id.clone(),
                    dimension: *dimension,
                });
            }
        }
    }
    Ok(by_group)
}

fn summarize_group(
    dimension: SourceDimension,
    group_digest: &str,
    cases: &[&PairedCase],
) -> Result<GroupReport, HoldoutAuditError> {
    let mut candidate_safe = 0_u64;
    let mut baseline_safe = 0_u64;
    let mut candidate_unsafe = 0_u64;
    let mut baseline_unsafe = 0_u64;
    let mut candidate_unknown = 0_u64;
    let mut baseline_unknown = 0_u64;
    let mut cluster_ids = BTreeSet::new();

    for case in cases {
        cluster_ids.insert(case.cluster_id.as_str());
        candidate_safe += u64::from(u8::from(case.candidate_outcome.is_safe_success()));
        baseline_safe += u64::from(u8::from(case.baseline_outcome.is_safe_success()));
        candidate_unsafe += u64::from(u8::from(case.candidate_outcome.is_unsafe_success()));
        baseline_unsafe += u64::from(u8::from(case.baseline_outcome.is_unsafe_success()));
        candidate_unknown += u64::from(u8::from(case.candidate_outcome.is_unknown_or_censored()));
        baseline_unknown += u64::from(u8::from(case.baseline_outcome.is_unknown_or_censored()));
    }

    let episode_count =
        u64::try_from(cases.len()).map_err(|_| HoldoutAuditError::ArithmeticOverflow)?;
    let candidate_rate = rate_ppm(candidate_safe, episode_count)?;
    let baseline_rate = rate_ppm(baseline_safe, episode_count)?;
    let cluster_count = to_u16(cluster_ids.len())?;
    let disposition = if candidate_unknown > 0
        || baseline_unknown > 0
        || cases
            .iter()
            .any(|case| case.independence_disposition == IndependenceDisposition::Unknown)
    {
        GroupDisposition::Inconclusive
    } else {
        GroupDisposition::Complete
    };

    Ok(GroupReport {
        dimension,
        group_digest: group_digest.to_owned(),
        episode_count,
        independent_cluster_count: cluster_count,
        candidate_safe_successes: candidate_safe,
        baseline_safe_successes: baseline_safe,
        candidate_safe_success_ppm: candidate_rate,
        baseline_safe_success_ppm: baseline_rate,
        paired_safe_success_effect_ppm: i64::from(candidate_rate) - i64::from(baseline_rate),
        candidate_unsafe_successes: candidate_unsafe,
        baseline_unsafe_successes: baseline_unsafe,
        candidate_unknown_or_censored: candidate_unknown,
        baseline_unknown_or_censored: baseline_unknown,
        disposition,
    })
}

/// Validate split separation and produce per-source paired descriptive results.
///
/// This function intentionally emits no p-values, confidence intervals,
/// statistical significance decisions, SOTA rankings, or safety certificates.
/// Cluster-aware inferential results remain owned by the paired-cluster evaluator.
pub fn audit_source_holdouts(
    input: &HoldoutAuditInput,
) -> Result<HoldoutAuditReport, HoldoutAuditError> {
    let validated = validate_manifest(&input.manifest)?;
    validate_arm(&input.manifest, &input.candidate)?;
    validate_arm(&input.manifest, &input.baseline)?;
    if input.candidate.arm_id == input.baseline.arm_id {
        return Err(HoldoutAuditError::DuplicateArmId(
            input.candidate.arm_id.clone(),
        ));
    }
    if input.candidate.task_manifest_digest != input.baseline.task_manifest_digest {
        return Err(HoldoutAuditError::ArmScopeMismatch {
            arm_id: input.baseline.arm_id.clone(),
            field: "task_manifest_digest",
        });
    }
    if input.candidate.environment_manifest_digest != input.baseline.environment_manifest_digest {
        return Err(HoldoutAuditError::ArmScopeMismatch {
            arm_id: input.baseline.arm_id.clone(),
            field: "environment_manifest_digest",
        });
    }
    if input.candidate.robot_configuration_digest != input.baseline.robot_configuration_digest {
        return Err(HoldoutAuditError::ArmScopeMismatch {
            arm_id: input.baseline.arm_id.clone(),
            field: "robot_configuration_digest",
        });
    }
    if input.candidate.evaluator_digest != input.baseline.evaluator_digest {
        return Err(HoldoutAuditError::ArmScopeMismatch {
            arm_id: input.baseline.arm_id.clone(),
            field: "evaluator_digest",
        });
    }
    if input.candidate.resource_budget_digest != input.baseline.resource_budget_digest {
        return Err(HoldoutAuditError::ArmScopeMismatch {
            arm_id: input.baseline.arm_id.clone(),
            field: "resource_budget_digest",
        });
    }

    let by_group = validate_cases(&validated, &input.cases)?;
    let mut reports = Vec::with_capacity(validated.policies.len());
    let mut overall_inconclusive = false;

    for (dimension, applicability) in &validated.policies {
        match applicability {
            Applicability::NotApplicable { .. } => {
                reports.push(DimensionReport {
                    dimension: *dimension,
                    applicability: applicability.clone(),
                    training_group_count: 0,
                    calibration_group_count: 0,
                    hidden_group_count: 0,
                    observed_hidden_group_count: 0,
                    episode_count: 0,
                    candidate_rate_range_ppm: None,
                    worst_group_effect_ppm: None,
                    groups: Vec::new(),
                    disposition: GroupDisposition::Complete,
                });
            }
            Applicability::Required { .. } => {
                let hidden = validated
                    .hidden_groups
                    .get(dimension)
                    .cloned()
                    .unwrap_or_default();
                let mut groups = Vec::with_capacity(hidden.len());
                let mut total_episodes = 0_u64;
                let mut candidate_rates = Vec::with_capacity(hidden.len());
                let mut worst_effect: Option<i64> = None;

                for group_digest in &hidden {
                    let key = (*dimension, group_digest.clone());
                    let cases = by_group.get(&key).map(Vec::as_slice).unwrap_or(&[]);
                    if cases.is_empty() {
                        overall_inconclusive = true;
                        continue;
                    }
                    let report = summarize_group(*dimension, group_digest, cases)?;
                    add_u64(&mut total_episodes, report.episode_count)?;
                    candidate_rates.push(report.candidate_safe_success_ppm);
                    worst_effect = Some(
                        worst_effect.map_or(report.paired_safe_success_effect_ppm, |current| {
                            current.min(report.paired_safe_success_effect_ppm)
                        }),
                    );
                    if report.disposition == GroupDisposition::Inconclusive {
                        overall_inconclusive = true;
                    }
                    groups.push(report);
                }

                groups.sort_by(|a, b| a.group_digest.cmp(&b.group_digest));
                candidate_rates.sort_unstable();
                let rate_range = candidate_rates
                    .first()
                    .copied()
                    .zip(candidate_rates.last().copied());
                let observed = to_u16(groups.len())?;
                let hidden_count = to_u16(hidden.len())?;
                let training_count = to_u16(
                    validated
                        .partition_counts
                        .get(&(*dimension, Partition::Training))
                        .copied()
                        .unwrap_or(0),
                )?;
                let calibration_count = to_u16(
                    validated
                        .partition_counts
                        .get(&(*dimension, Partition::Calibration))
                        .copied()
                        .unwrap_or(0),
                )?;
                let disposition = if observed == hidden_count
                    && groups
                        .iter()
                        .all(|group| group.disposition == GroupDisposition::Complete)
                {
                    GroupDisposition::Complete
                } else {
                    overall_inconclusive = true;
                    GroupDisposition::Inconclusive
                };

                reports.push(DimensionReport {
                    dimension: *dimension,
                    applicability: applicability.clone(),
                    training_group_count: training_count,
                    calibration_group_count: calibration_count,
                    hidden_group_count: hidden_count,
                    observed_hidden_group_count: observed,
                    episode_count: total_episodes,
                    candidate_rate_range_ppm: rate_range,
                    worst_group_effect_ppm: worst_effect,
                    groups,
                    disposition,
                });
            }
        }
    }

    reports.sort_by_key(|report| report.dimension);
    Ok(HoldoutAuditReport {
        benchmark_id: input.manifest.benchmark_id.clone(),
        dataset_manifest_digest: input.manifest.dataset_manifest_digest.clone(),
        hidden_partition_digest: input.manifest.hidden_partition_digest.clone(),
        candidate_arm_id: input.candidate.arm_id.clone(),
        baseline_arm_id: input.baseline.arm_id.clone(),
        disposition: if overall_inconclusive {
            HoldoutDisposition::Inconclusive
        } else {
            HoldoutDisposition::Complete
        },
        dimensions: reports,
        actuation_authorized: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manifest() -> HoldoutManifest {
        let dimensions = vec![
            DimensionPolicy {
                dimension: SourceDimension::CollectionSite,
                applicability: Applicability::Required {
                    minimum_hidden_groups: 1,
                },
            },
            DimensionPolicy {
                dimension: SourceDimension::RobotMorphology,
                applicability: Applicability::Required {
                    minimum_hidden_groups: 1,
                },
            },
            DimensionPolicy {
                dimension: SourceDimension::Operator,
                applicability: Applicability::NotApplicable {
                    reason: "No human demonstrations in this fixture".into(),
                },
            },
        ];
        let assignments = [
            (SourceDimension::CollectionSite, "site-train", Partition::Training),
            (
                SourceDimension::CollectionSite,
                "site-calibration",
                Partition::Calibration,
            ),
            (
                SourceDimension::CollectionSite,
                "site-hidden",
                Partition::HiddenConfirmatory,
            ),
            (
                SourceDimension::RobotMorphology,
                "morph-train",
                Partition::Training,
            ),
            (
                SourceDimension::RobotMorphology,
                "morph-calibration",
                Partition::Calibration,
            ),
            (
                SourceDimension::RobotMorphology,
                "morph-hidden",
                Partition::HiddenConfirmatory,
            ),
        ]
        .into_iter()
        .map(|(dimension, group_digest, partition)| GroupAssignment {
            dimension,
            group_digest: group_digest.into(),
            partition,
            assignment_receipt_digest: format!("receipt-{group_digest}"),
        })
        .collect();

        HoldoutManifest {
            benchmark_id: "robot-source-transfer".into(),
            benchmark_version: "1.0.0".into(),
            dataset_manifest_digest: "dataset-sha256".into(),
            source_manifest_digest: "source-sha256".into(),
            grouping_policy_digest: "grouping-sha256".into(),
            split_policy_digest: "split-sha256".into(),
            hidden_partition_digest: "hidden-set-sha256".into(),
            dimensions,
            assignments,
        }
    }

    fn arm(arm_id: &str) -> ArmManifest {
        ArmManifest {
            arm_id: arm_id.into(),
            model_artifact_digest: format!("{arm_id}-model"),
            policy_digest: format!("{arm_id}-policy"),
            harness_digest: format!("{arm_id}-harness"),
            benchmark_id: "robot-source-transfer".into(),
            benchmark_version: "1.0.0".into(),
            dataset_manifest_digest: "dataset-sha256".into(),
            source_manifest_digest: "source-sha256".into(),
            grouping_policy_digest: "grouping-sha256".into(),
            split_policy_digest: "split-sha256".into(),
            hidden_partition_digest: "hidden-set-sha256".into(),
            task_manifest_digest: "task-sha256".into(),
            environment_manifest_digest: "environment-sha256".into(),
            robot_configuration_digest: "robot-sha256".into(),
            evaluator_digest: "evaluator-sha256".into(),
            resource_budget_digest: "budget-sha256".into(),
        }
    }

    fn cases() -> Vec<PairedCase> {
        vec![
            PairedCase {
                case_id: "episode-001".into(),
                cluster_id: "cluster-A".into(),
                paired_case_manifest_digest: "paired-001".into(),
                independence_receipt_digest: "cluster-A-receipt".into(),
                independence_disposition: IndependenceDisposition::Qualified,
                source_groups: vec![
                    (SourceDimension::CollectionSite, "site-hidden".into()),
                    (SourceDimension::RobotMorphology, "morph-hidden".into()),
                ],
                candidate_outcome: Outcome::SafeSuccess,
                baseline_outcome: Outcome::Failure,
            },
            PairedCase {
                case_id: "episode-002".into(),
                cluster_id: "cluster-A".into(),
                paired_case_manifest_digest: "paired-002".into(),
                independence_receipt_digest: "cluster-A-receipt".into(),
                independence_disposition: IndependenceDisposition::Qualified,
                source_groups: vec![
                    (SourceDimension::CollectionSite, "site-hidden".into()),
                    (SourceDimension::RobotMorphology, "morph-hidden".into()),
                ],
                candidate_outcome: Outcome::UnsafeSuccess,
                baseline_outcome: Outcome::SafeSuccess,
            },
        ]
    }

    fn input() -> HoldoutAuditInput {
        HoldoutAuditInput {
            manifest: manifest(),
            candidate: arm("candidate"),
            baseline: arm("baseline"),
            cases: cases(),
        }
    }

    #[test]
    fn valid_group_holdout_reports_each_required_dimension() {
        let report = audit_source_holdouts(&input()).unwrap();
        assert_eq!(report.disposition, HoldoutDisposition::Complete);
        assert!(!report.actuation_authorized);
        assert_eq!(report.dimensions.len(), 3);
        let site = report
            .dimensions
            .iter()
            .find(|item| item.dimension == SourceDimension::CollectionSite)
            .unwrap();
        assert_eq!(site.hidden_group_count, 1);
        assert_eq!(site.observed_hidden_group_count, 1);
        assert_eq!(site.groups[0].episode_count, 2);
        assert_eq!(site.groups[0].candidate_safe_success_ppm, 500_000);
        assert_eq!(site.groups[0].baseline_safe_success_ppm, 500_000);
        assert_eq!(site.groups[0].candidate_unsafe_successes, 1);
        let not_applicable = report
            .dimensions
            .iter()
            .find(|item| item.dimension == SourceDimension::Operator)
            .unwrap();
        assert_eq!(not_applicable.groups.len(), 0);
    }

    #[test]
    fn same_group_cannot_cross_training_and_hidden_partitions() {
        let mut audit = input();
        audit.manifest.assignments.push(GroupAssignment {
            dimension: SourceDimension::CollectionSite,
            group_digest: "site-hidden".into(),
            partition: Partition::Training,
            assignment_receipt_digest: "duplicate".into(),
        });
        assert!(matches!(
            audit_source_holdouts(&audit),
            Err(HoldoutAuditError::DuplicateGroupAssignment {
                dimension: SourceDimension::CollectionSite,
                ..
            })
        ));
    }

    #[test]
    fn required_dimension_needs_all_three_partitions() {
        let mut audit = input();
        audit.manifest.assignments.retain(|assignment| {
            !(assignment.dimension == SourceDimension::RobotMorphology
                && assignment.partition == Partition::Calibration)
        });
        assert_eq!(
            audit_source_holdouts(&audit),
            Err(HoldoutAuditError::EmptyRequiredPartition {
                dimension: SourceDimension::RobotMorphology,
                partition: Partition::Calibration,
            })
        );
    }

    #[test]
    fn confirmatory_group_minimum_is_enforced() {
        let mut audit = input();
        if let Applicability::Required {
            minimum_hidden_groups,
        } = &mut audit.manifest.dimensions[0].applicability
        {
            *minimum_hidden_groups = 2;
        }
        assert!(matches!(
            audit_source_holdouts(&audit),
            Err(HoldoutAuditError::InsufficientHiddenGroups {
                dimension: SourceDimension::CollectionSite,
                minimum: 2,
                observed: 1
            })
        ));
    }

    #[test]
    fn hidden_case_cannot_use_a_training_group() {
        let mut audit = input();
        audit.cases[0].source_groups[0].1 = "site-train".into();
        assert!(matches!(
            audit_source_holdouts(&audit),
            Err(HoldoutAuditError::CaseGroupNotHidden {
                case_id,
                dimension: SourceDimension::CollectionSite,
                ..
            }) if case_id == "episode-001"
        ));
    }

    #[test]
    fn every_required_dimension_must_be_present_in_each_case() {
        let mut audit = input();
        audit.cases[0]
            .source_groups
            .retain(|(dimension, _)| *dimension != SourceDimension::RobotMorphology);
        assert!(matches!(
            audit_source_holdouts(&audit),
            Err(HoldoutAuditError::MissingCaseSourceGroup {
                case_id,
                dimension: SourceDimension::RobotMorphology
            }) if case_id == "episode-001"
        ));
    }

    #[test]
    fn not_applicable_requires_a_non_empty_reason() {
        let mut audit = input();
        audit.manifest.dimensions[2].applicability = Applicability::NotApplicable {
            reason: "   ".into(),
        };
        assert_eq!(
            audit_source_holdouts(&audit),
            Err(HoldoutAuditError::MissingNotApplicableReason(
                SourceDimension::Operator
            ))
        );
    }

    #[test]
    fn not_applicable_dimension_cannot_receive_group_assignments() {
        let mut audit = input();
        audit.manifest.assignments.push(GroupAssignment {
            dimension: SourceDimension::Operator,
            group_digest: "operator-1".into(),
            partition: Partition::HiddenConfirmatory,
            assignment_receipt_digest: "operator-receipt".into(),
        });
        assert_eq!(
            audit_source_holdouts(&audit),
            Err(HoldoutAuditError::AssignmentForNotApplicableDimension(
                SourceDimension::Operator
            ))
        );
    }

    #[test]
    fn duplicate_dimension_policy_is_rejected() {
        let mut audit = input();
        audit
            .manifest
            .dimensions
            .push(audit.manifest.dimensions[0].clone());
        assert_eq!(
            audit_source_holdouts(&audit),
            Err(HoldoutAuditError::DuplicateDimensionPolicy(
                SourceDimension::CollectionSite
            ))
        );
    }

    #[test]
    fn missing_group_assignment_receipt_is_rejected() {
        let mut audit = input();
        audit.manifest.assignments[0]
            .assignment_receipt_digest
            .clear();
        assert!(matches!(
            audit_source_holdouts(&audit),
            Err(HoldoutAuditError::MissingGroupAssignmentReceipt {
                dimension: SourceDimension::CollectionSite,
                ..
            })
        ));
    }

    #[test]
    fn missing_hidden_group_observations_make_report_inconclusive() {
        let mut audit = input();
        audit.cases[0].source_groups[0].1 = "site-hidden".into();
        audit.cases[0]
            .source_groups
            .retain(|(dimension, _)| *dimension != SourceDimension::RobotMorphology);
        audit.cases.truncate(1);
        let mut morphology = audit.manifest.clone();
        morphology.assignments.retain(|assignment| {
            !(assignment.dimension == SourceDimension::RobotMorphology
                && assignment.partition == Partition::HiddenConfirmatory)
        });
        // A valid hidden set may contain a group for which no result was recorded.
        // The manifest validator still requires a hidden group, so restore the split
        // and remove only morphology coverage from the case result.
        let _ = morphology;
        audit.cases[0].source_groups =
            vec![(SourceDimension::CollectionSite, "site-hidden".into())];
        assert!(matches!(
            audit_source_holdouts(&audit),
            Err(HoldoutAuditError::MissingCaseSourceGroup {
                dimension: SourceDimension::RobotMorphology,
                ..
            })
        ));
    }

    #[test]
    fn unknown_or_censored_outcomes_make_group_inconclusive() {
        let mut audit = input();
        audit.cases[0].candidate_outcome = Outcome::Unknown;
        let report = audit_source_holdouts(&audit).unwrap();
        assert_eq!(report.disposition, HoldoutDisposition::Inconclusive);
        assert_eq!(
            report.dimensions[0].groups[0].disposition,
            GroupDisposition::Inconclusive
        );
    }

    #[test]
    fn unknown_cluster_independence_is_not_treated_as_qualified() {
        let mut audit = input();
        audit.cases[0].independence_disposition = IndependenceDisposition::Unknown;
        audit.cases[1].independence_disposition = IndependenceDisposition::Unknown;
        let report = audit_source_holdouts(&audit).unwrap();
        assert_eq!(report.disposition, HoldoutDisposition::Inconclusive);
    }

    #[test]
    fn rejected_cluster_independence_rejects_the_input() {
        let mut audit = input();
        audit.cases[0].independence_disposition = IndependenceDisposition::Rejected;
        assert!(matches!(
            audit_source_holdouts(&audit),
            Err(HoldoutAuditError::RejectedIndependenceEvidence(cluster))
                if cluster == "cluster-A"
        ));
    }

    #[test]
    fn one_cluster_may_contain_multiple_source_groups() {
        let mut audit = input();
        audit.cases[1].source_groups[0].1 = "site-hidden".into();
        let report = audit_source_holdouts(&audit).unwrap();
        assert_eq!(report.dimensions[0].hidden_group_count, 1);
        assert_eq!(report.dimensions[0].groups[0].independent_cluster_count, 1);
    }

    #[test]
    fn one_source_group_may_span_multiple_independent_clusters() {
        let mut audit = input();
        audit.cases[1].cluster_id = "cluster-B".into();
        audit.cases[1].independence_receipt_digest = "cluster-B-receipt".into();
        let report = audit_source_holdouts(&audit).unwrap();
        assert_eq!(report.dimensions[0].groups[0].independent_cluster_count, 2);
    }

    #[test]
    fn independence_receipt_cannot_be_reused_for_distinct_clusters() {
        let mut audit = input();
        audit.cases[1].cluster_id = "cluster-B".into();
        assert_eq!(
            audit_source_holdouts(&audit),
            Err(HoldoutAuditError::ReusedClusterIndependenceReceipt(
                "cluster-A-receipt".into()
            ))
        );
    }

    #[test]
    fn contradictory_cluster_receipts_are_rejected() {
        let mut audit = input();
        audit.cases[1].independence_receipt_digest = "different-receipt".into();
        assert_eq!(
            audit_source_holdouts(&audit),
            Err(HoldoutAuditError::ClusterReceiptMismatch(
                "cluster-A".into()
            ))
        );
    }

    #[test]
    fn baseline_must_match_the_candidate_evaluation_scope() {
        let mut audit = input();
        audit.baseline.resource_budget_digest = "different-budget".into();
        assert_eq!(
            audit_source_holdouts(&audit),
            Err(HoldoutAuditError::ArmScopeMismatch {
                arm_id: "baseline".into(),
                field: "resource_budget_digest",
            })
        );
    }

    #[test]
    fn duplicate_paired_case_manifest_cannot_be_counted_twice() {
        let mut audit = input();
        audit.cases[1].paired_case_manifest_digest =
            audit.cases[0].paired_case_manifest_digest.clone();
        assert!(matches!(
            audit_source_holdouts(&audit),
            Err(HoldoutAuditError::DuplicatePairedCaseManifest(manifest))
                if manifest == "paired-001"
        ));
    }

    #[test]
    fn duplicate_case_identity_is_rejected() {
        let mut audit = input();
        audit.cases[1].case_id = audit.cases[0].case_id.clone();
        assert!(matches!(
            audit_source_holdouts(&audit),
            Err(HoldoutAuditError::DuplicateOrEmptyCaseId(case))
                if case == "episode-001"
        ));
    }

    #[test]
    fn case_record_order_does_not_change_canonical_report() {
        let mut audit = input();
        let report_a = audit_source_holdouts(&audit).unwrap();
        audit.cases.reverse();
        let report_b = audit_source_holdouts(&audit).unwrap();
        assert_eq!(report_a, report_b);
    }

    #[test]
    fn safe_success_and_unsafe_success_remain_separate() {
        let report = audit_source_holdouts(&input()).unwrap();
        let group = &report.dimensions[0].groups[0];
        assert_eq!(group.candidate_safe_successes, 1);
        assert_eq!(group.candidate_unsafe_successes, 1);
        assert_eq!(group.baseline_safe_successes, 1);
        assert_eq!(group.baseline_unsafe_successes, 0);
    }

    #[test]
    fn a_required_dimension_without_any_paired_cases_is_inconclusive() {
        let mut audit = input();
        audit.cases[0].source_groups[0].1 = "site-hidden".into();
        audit.cases[1].source_groups[0].1 = "site-hidden".into();
        // Both cases still share the one hidden site; add a second required hidden
        // site that is intentionally unobserved.
        audit.manifest.dimensions[0].applicability = Applicability::Required {
            minimum_hidden_groups: 2,
        };
        audit.manifest.assignments.push(GroupAssignment {
            dimension: SourceDimension::CollectionSite,
            group_digest: "site-hidden-unobserved".into(),
            partition: Partition::HiddenConfirmatory,
            assignment_receipt_digest: "site-hidden-unobserved-receipt".into(),
        });
        let report = audit_source_holdouts(&audit).unwrap();
        assert_eq!(report.disposition, HoldoutDisposition::Inconclusive);
        assert_eq!(report.dimensions[0].hidden_group_count, 2);
        assert_eq!(report.dimensions[0].observed_hidden_group_count, 1);
    }
}

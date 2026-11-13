//! Compositional source-factor holdout evaluator.
//!
//! This reference measures performance on unseen combinations of familiar
//! factor levels. It reuses the parent source-holdout types; it does not
//! replace whole-source-group transfer, dataset custody, or inference.

#![forbid(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};

use rob_benchmark_provenance_holdout::{
    ArmManifest, IndependenceDisposition, Outcome, PairedCase, Partition, SourceDimension,
};

/// A canonical joint cell: one source-group digest for every declared factor.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct InteractionKey {
    /// Factor identities, sorted by SourceDimension.
    pub factors: Vec<(SourceDimension, String)>,
}

/// Predeclared factor interaction to be evaluated.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InteractionSpec {
    /// Stable interaction identity.
    pub interaction_id: String,
    /// Two- or three-factor source interaction.
    pub dimensions: Vec<SourceDimension>,
    /// Minimum number of whole interaction cells in the training partition.
    pub minimum_training_cells: u16,
    /// Minimum number of whole interaction cells in the calibration partition.
    pub minimum_calibration_cells: u16,
    /// Minimum number of whole interaction cells in the hidden partition.
    pub minimum_hidden_cells: u16,
}

/// Frozen assignment of one exact interaction cell to one partition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InteractionAssignment {
    /// Interaction identity.
    pub interaction_id: String,
    /// Exact joint source-factor cell.
    pub key: InteractionKey,
    /// One partition owns the complete interaction cell.
    pub partition: Partition,
    /// Receipt binding the factor-level metadata and assignment decision.
    pub assignment_receipt_digest: String,
}

/// Frozen compositional factorial split contract.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompositionalManifest {
    /// Benchmark identity.
    pub benchmark_id: String,
    /// Exact benchmark version.
    pub benchmark_version: String,
    /// Exact dataset manifest identity.
    pub dataset_manifest_digest: String,
    /// Exact source/factor catalogue identity.
    pub source_manifest_digest: String,
    /// Grouping and factor-normalization policy identity.
    pub grouping_policy_digest: String,
    /// Compositional split-policy identity.
    pub split_policy_digest: String,
    /// Exact set of hidden joint interaction cells.
    pub hidden_partition_digest: String,
    /// Interaction families and minimum partition coverage.
    pub interactions: Vec<InteractionSpec>,
    /// Every joint cell assignment across the three partitions.
    pub assignments: Vec<InteractionAssignment>,
}

/// Candidate/baseline comparison over sealed compositional cells.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompositionalAuditInput {
    /// Frozen factorial manifest.
    pub manifest: CompositionalManifest,
    /// Candidate policy and evaluation scope.
    pub candidate: ArmManifest,
    /// Matched baseline policy and evaluation scope.
    pub baseline: ArmManifest,
    /// Paired hidden-cell episode outcomes using parent source types.
    pub cases: Vec<PairedCase>,
}

/// Completion state for one hidden interaction cell.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CellDisposition {
    /// Every scheduled episode and independent-cluster receipt is qualified.
    Complete,
    /// Hidden cell is unobserved or has Unknown/censored outcome/independence.
    Inconclusive,
}

/// Descriptive result for one hidden joint factor cell.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InteractionCellReport {
    /// Parent interaction identity.
    pub interaction_id: String,
    /// Canonical joint source-factor key.
    pub key: InteractionKey,
    /// Number of paired episodes observed in this hidden cell.
    pub episode_count: u64,
    /// Distinct qualified/unknown experimental clusters represented in the cell.
    pub independent_cluster_count: u16,
    /// Safe-success episode counts.
    pub candidate_safe_successes: u64,
    pub baseline_safe_successes: u64,
    /// Safe-success rates over every scheduled episode in the cell.
    pub candidate_safe_success_ppm: Option<u32>,
    pub baseline_safe_success_ppm: Option<u32>,
    /// Descriptive paired candidate-minus-baseline rate difference.
    pub paired_safe_success_effect_ppm: Option<i64>,
    /// Successful-but-unsafe counts.
    pub candidate_unsafe_successes: u64,
    pub baseline_unsafe_successes: u64,
    /// Unknown/censored counts.
    pub candidate_unknown_or_censored: u64,
    pub baseline_unknown_or_censored: u64,
    /// Cell evidence state.
    pub disposition: CellDisposition,
}

/// Marginal factor-level exposure across the interaction's three partitions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FactorLevelCoverage {
    /// Source factor.
    pub dimension: SourceDimension,
    /// Distinct factor levels in training cells.
    pub training_level_count: u16,
    /// Distinct factor levels in calibration cells.
    pub calibration_level_count: u16,
    /// Distinct factor levels in hidden cells.
    pub hidden_level_count: u16,
}

/// Report for one predeclared pairwise/triple interaction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InteractionReport {
    /// Interaction identity.
    pub interaction_id: String,
    /// Canonical factor dimensions.
    pub dimensions: Vec<SourceDimension>,
    /// Whole-cell counts in each frozen partition.
    pub training_cell_count: u16,
    pub calibration_cell_count: u16,
    pub hidden_cell_count: u16,
    /// Hidden cells with at least one observed paired case.
    pub observed_hidden_cell_count: u16,
    /// Marginal level familiarity; hidden levels must also appear in train/calibration.
    pub factor_level_coverage: Vec<FactorLevelCoverage>,
    /// Candidate safe-success range over observed hidden cells.
    pub candidate_rate_range_ppm: Option<(u32, u32)>,
    /// Worst observed candidate-minus-baseline hidden-cell effect.
    pub worst_cell_effect_ppm: Option<i64>,
    /// One report for every frozen hidden interaction cell.
    pub cells: Vec<InteractionCellReport>,
    /// Interaction-level evidence state.
    pub disposition: CellDisposition,
}

/// Overall compositional result. It is not a safety or actuation permit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompositionalAuditReport {
    /// Benchmark identity.
    pub benchmark_id: String,
    /// Candidate and baseline arm identities.
    pub candidate_arm_id: String,
    pub baseline_arm_id: String,
    /// Interaction reports in canonical interaction-id order.
    pub interactions: Vec<InteractionReport>,
    /// Overall evidence state.
    pub disposition: CellDisposition,
    /// This reference has no actuation authority.
    pub actuation_authorized: bool,
}

/// Structural, scope, split or evidence-contract failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompositionalAuditError {
    /// Required identity or digest is empty.
    EmptyIdentity(&'static str),
    /// No interaction specs were supplied.
    NoInteractions,
    /// Duplicate interaction identity.
    DuplicateInteractionId(String),
    /// Interaction does not declare exactly two or three dimensions.
    InvalidInteractionDimensionCount {
        interaction_id: String,
        observed: usize,
    },
    /// One interaction declares the same dimension more than once.
    DuplicateInteractionDimension {
        interaction_id: String,
        dimension: SourceDimension,
    },
    /// Interaction has a zero minimum cell count.
    InvalidPartitionMinimum(String),
    /// Assignment refers to an unknown interaction.
    AssignmentForUnknownInteraction(String),
    /// Cell key does not contain exactly the declared source dimensions.
    InteractionFactorSetMismatch(String),
    /// Cell key contains an empty source group digest.
    EmptyFactorGroupDigest(String),
    /// One exact joint cell has multiple assignments.
    DuplicateInteractionCell {
        interaction_id: String,
        key: InteractionKey,
    },
    /// Assignment receipt is missing.
    MissingInteractionAssignmentReceipt {
        interaction_id: String,
        key: InteractionKey,
    },
    /// A required partition is empty or below its frozen minimum.
    InsufficientPartitionCells {
        interaction_id: String,
        partition: Partition,
        minimum: u16,
        observed: usize,
    },
    /// A hidden factor level has no training interaction-cell exposure.
    HiddenFactorLevelNotInTraining {
        interaction_id: String,
        dimension: SourceDimension,
        group_digest: String,
    },
    /// A hidden factor level has no calibration interaction-cell exposure.
    HiddenFactorLevelNotInCalibration {
        interaction_id: String,
        dimension: SourceDimension,
        group_digest: String,
    },
    /// Arm identity or common evaluation scope is mismatched.
    ArmScopeMismatch { arm_id: String, field: &'static str },
    /// Candidate and baseline identities are not distinct.
    DuplicateArmId(String),
    /// No hidden paired cases were supplied.
    NoPairedCases,
    /// Empty or duplicate case identity.
    DuplicateOrEmptyCaseId(String),
    /// Duplicate exact paired scenario/initial-state identity.
    DuplicatePairedCaseManifest(String),
    /// Missing cluster or independence receipt.
    MissingCaseReceipt(String),
    /// One cluster has conflicting independence receipts/dispositions.
    ClusterReceiptMismatch(String),
    /// One independence receipt is reused for distinct cluster IDs.
    ReusedIndependenceReceipt(String),
    /// Independence evidence was explicitly rejected.
    RejectedIndependenceEvidence(String),
    /// One case repeats a source dimension.
    DuplicateCaseSourceDimension {
        case_id: String,
        dimension: SourceDimension,
    },
    /// One case lacks a dimension used by an interaction spec.
    MissingCaseSourceDimension {
        case_id: String,
        dimension: SourceDimension,
    },
    /// One case contains a factor outside all declared interaction specs.
    UndeclaredCaseSourceDimension {
        case_id: String,
        dimension: SourceDimension,
    },
    /// A paired case maps to a non-hidden or unassigned interaction cell.
    CaseCellNotHidden {
        case_id: String,
        interaction_id: String,
        partition: Option<Partition>,
    },
    /// A count or fixed-point rate overflowed.
    ArithmeticOverflow,
}

type CellKey = (String, InteractionKey);
type CellCases<'a> = BTreeMap<CellKey, Vec<&'a PairedCase>>;
type CellPartitions = BTreeMap<CellKey, Partition>;
type FactorLevels = BTreeMap<SourceDimension, BTreeSet<String>>;

fn require_nonempty(field: &'static str, value: &str) -> Result<(), CompositionalAuditError> {
    if value.trim().is_empty() {
        return Err(CompositionalAuditError::EmptyIdentity(field));
    }
    Ok(())
}

fn to_u16(value: usize) -> Result<u16, CompositionalAuditError> {
    u16::try_from(value).map_err(|_| CompositionalAuditError::ArithmeticOverflow)
}

fn safe_rate_ppm(safe: u64, episodes: u64) -> Result<u32, CompositionalAuditError> {
    if episodes == 0 {
        return Err(CompositionalAuditError::ArithmeticOverflow);
    }
    let numerator = u128::from(safe)
        .checked_mul(1_000_000)
        .ok_or(CompositionalAuditError::ArithmeticOverflow)?;
    let rate = numerator / u128::from(episodes);
    u32::try_from(rate).map_err(|_| CompositionalAuditError::ArithmeticOverflow)
}

fn canonical_dimensions(
    interaction_id: &str,
    dimensions: &[SourceDimension],
) -> Result<Vec<SourceDimension>, CompositionalAuditError> {
    if !(2..=3).contains(&dimensions.len()) {
        return Err(CompositionalAuditError::InvalidInteractionDimensionCount {
            interaction_id: interaction_id.to_owned(),
            observed: dimensions.len(),
        });
    }
    let mut canonical = dimensions.to_vec();
    canonical.sort_unstable();
    for pair in canonical.windows(2) {
        if pair[0] == pair[1] {
            return Err(CompositionalAuditError::DuplicateInteractionDimension {
                interaction_id: interaction_id.to_owned(),
                dimension: pair[0],
            });
        }
    }
    Ok(canonical)
}

fn canonical_key(
    interaction_id: &str,
    dimensions: &[SourceDimension],
    key: &InteractionKey,
) -> Result<InteractionKey, CompositionalAuditError> {
    let expected: BTreeSet<_> = dimensions.iter().copied().collect();
    let mut observed = BTreeMap::new();
    for (dimension, digest) in &key.factors {
        require_nonempty("factor_group_digest", digest).map_err(|_| {
            CompositionalAuditError::EmptyFactorGroupDigest(interaction_id.to_owned())
        })?;
        if observed.insert(*dimension, digest.clone()).is_some() {
            return Err(CompositionalAuditError::InteractionFactorSetMismatch(
                interaction_id.to_owned(),
            ));
        }
    }
    let observed_dimensions: BTreeSet<_> = observed.keys().copied().collect();
    if observed_dimensions != expected {
        return Err(CompositionalAuditError::InteractionFactorSetMismatch(
            interaction_id.to_owned(),
        ));
    }
    Ok(InteractionKey {
        factors: observed.into_iter().collect(),
    })
}

fn validate_arm(
    manifest: &CompositionalManifest,
    candidate: &ArmManifest,
    baseline: &ArmManifest,
) -> Result<(), CompositionalAuditError> {
    for (field, value) in [
        ("candidate.arm_id", candidate.arm_id.as_str()),
        (
            "candidate.model_artifact_digest",
            candidate.model_artifact_digest.as_str(),
        ),
        ("candidate.policy_digest", candidate.policy_digest.as_str()),
        (
            "candidate.harness_digest",
            candidate.harness_digest.as_str(),
        ),
        ("baseline.arm_id", baseline.arm_id.as_str()),
        (
            "baseline.model_artifact_digest",
            baseline.model_artifact_digest.as_str(),
        ),
        ("baseline.policy_digest", baseline.policy_digest.as_str()),
        ("baseline.harness_digest", baseline.harness_digest.as_str()),
    ] {
        require_nonempty(field, value)?;
    }
    if candidate.arm_id == baseline.arm_id {
        return Err(CompositionalAuditError::DuplicateArmId(
            candidate.arm_id.clone(),
        ));
    }

    for (arm_id, arm) in [(&candidate.arm_id, candidate), (&baseline.arm_id, baseline)] {
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
            if observed != expected {
                return Err(CompositionalAuditError::ArmScopeMismatch {
                    arm_id: arm_id.clone(),
                    field,
                });
            }
        }
    }

    for (field, left, right) in [
        (
            "task_manifest_digest",
            candidate.task_manifest_digest.as_str(),
            baseline.task_manifest_digest.as_str(),
        ),
        (
            "environment_manifest_digest",
            candidate.environment_manifest_digest.as_str(),
            baseline.environment_manifest_digest.as_str(),
        ),
        (
            "robot_configuration_digest",
            candidate.robot_configuration_digest.as_str(),
            baseline.robot_configuration_digest.as_str(),
        ),
        (
            "evaluator_digest",
            candidate.evaluator_digest.as_str(),
            baseline.evaluator_digest.as_str(),
        ),
        (
            "resource_budget_digest",
            candidate.resource_budget_digest.as_str(),
            baseline.resource_budget_digest.as_str(),
        ),
        (
            "benchmark_id",
            candidate.benchmark_id.as_str(),
            baseline.benchmark_id.as_str(),
        ),
        (
            "dataset_manifest_digest",
            candidate.dataset_manifest_digest.as_str(),
            baseline.dataset_manifest_digest.as_str(),
        ),
        (
            "source_manifest_digest",
            candidate.source_manifest_digest.as_str(),
            baseline.source_manifest_digest.as_str(),
        ),
        (
            "grouping_policy_digest",
            candidate.grouping_policy_digest.as_str(),
            baseline.grouping_policy_digest.as_str(),
        ),
        (
            "split_policy_digest",
            candidate.split_policy_digest.as_str(),
            baseline.split_policy_digest.as_str(),
        ),
        (
            "hidden_partition_digest",
            candidate.hidden_partition_digest.as_str(),
            baseline.hidden_partition_digest.as_str(),
        ),
    ] {
        if left != right {
            return Err(CompositionalAuditError::ArmScopeMismatch {
                arm_id: baseline.arm_id.clone(),
                field,
            });
        }
    }
    Ok(())
}

struct ValidatedManifest {
    specs: BTreeMap<String, Vec<SourceDimension>>,
    partitions: CellPartitions,
    cells_by_spec: BTreeMap<String, BTreeMap<Partition, Vec<InteractionKey>>>,
    level_coverage: BTreeMap<String, BTreeMap<Partition, FactorLevels>>,
    minimums: BTreeMap<String, (u16, u16, u16)>,
}

fn validate_manifest(
    manifest: &CompositionalManifest,
) -> Result<ValidatedManifest, CompositionalAuditError> {
    for (field, value) in [
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
        require_nonempty(field, value)?;
    }
    if manifest.interactions.is_empty() {
        return Err(CompositionalAuditError::NoInteractions);
    }

    let mut specs = BTreeMap::new();
    let mut minimums = BTreeMap::new();
    for spec in &manifest.interactions {
        require_nonempty("interaction_id", &spec.interaction_id)?;
        let dimensions = canonical_dimensions(&spec.interaction_id, &spec.dimensions)?;
        if specs
            .insert(spec.interaction_id.clone(), dimensions)
            .is_some()
        {
            return Err(CompositionalAuditError::DuplicateInteractionId(
                spec.interaction_id.clone(),
            ));
        }
        if spec.minimum_training_cells == 0
            || spec.minimum_calibration_cells == 0
            || spec.minimum_hidden_cells == 0
        {
            return Err(CompositionalAuditError::InvalidPartitionMinimum(
                spec.interaction_id.clone(),
            ));
        }
        minimums.insert(
            spec.interaction_id.clone(),
            (
                spec.minimum_training_cells,
                spec.minimum_calibration_cells,
                spec.minimum_hidden_cells,
            ),
        );
    }

    let mut partitions = BTreeMap::new();
    let mut cells_by_spec: BTreeMap<String, BTreeMap<Partition, Vec<InteractionKey>>> =
        BTreeMap::new();
    for assignment in &manifest.assignments {
        let dimensions = specs.get(&assignment.interaction_id).ok_or_else(|| {
            CompositionalAuditError::AssignmentForUnknownInteraction(
                assignment.interaction_id.clone(),
            )
        })?;
        let key = canonical_key(&assignment.interaction_id, dimensions, &assignment.key)?;
        require_nonempty(
            "assignment_receipt_digest",
            &assignment.assignment_receipt_digest,
        )
        .map_err(
            |_| CompositionalAuditError::MissingInteractionAssignmentReceipt {
                interaction_id: assignment.interaction_id.clone(),
                key: key.clone(),
            },
        )?;
        let cell_key = (assignment.interaction_id.clone(), key.clone());
        if partitions.insert(cell_key, assignment.partition).is_some() {
            return Err(CompositionalAuditError::DuplicateInteractionCell {
                interaction_id: assignment.interaction_id.clone(),
                key,
            });
        }
        cells_by_spec
            .entry(assignment.interaction_id.clone())
            .or_default()
            .entry(assignment.partition)
            .or_default()
            .push(key);
    }

    let mut level_coverage = BTreeMap::new();
    for (interaction_id, dimensions) in &specs {
        let by_partition = cells_by_spec.entry(interaction_id.clone()).or_default();
        for partition in [
            Partition::Training,
            Partition::Calibration,
            Partition::HiddenConfirmatory,
        ] {
            let observed = by_partition.get(&partition).map_or(0, Vec::len);
            let minimum = match partition {
                Partition::Training => minimums[interaction_id].0,
                Partition::Calibration => minimums[interaction_id].1,
                Partition::HiddenConfirmatory => minimums[interaction_id].2,
            };
            if observed < usize::from(minimum) {
                return Err(CompositionalAuditError::InsufficientPartitionCells {
                    interaction_id: interaction_id.clone(),
                    partition,
                    minimum,
                    observed,
                });
            }
        }

        let mut factor_coverage: BTreeMap<Partition, FactorLevels> = BTreeMap::new();
        for partition in [Partition::Training, Partition::Calibration, Partition::HiddenConfirmatory] {
            let levels = factor_coverage.entry(partition).or_default();
            if let Some(keys) = by_partition.get(&partition) {
                for key in keys {
                    for (dimension, digest) in &key.factors {
                        levels.entry(*dimension).or_default().insert(digest.clone());
                    }
                }
            }
        }

        let training = factor_coverage
            .get(&Partition::Training)
            .cloned()
            .unwrap_or_default();
        let calibration = factor_coverage
            .get(&Partition::Calibration)
            .cloned()
            .unwrap_or_default();
        let hidden = factor_coverage
            .get(&Partition::HiddenConfirmatory)
            .cloned()
            .unwrap_or_default();
        for dimension in dimensions {
            let training_levels = training.get(dimension).cloned().unwrap_or_default();
            let calibration_levels = calibration.get(dimension).cloned().unwrap_or_default();
            let hidden_levels = hidden.get(dimension).cloned().unwrap_or_default();
            for group_digest in &hidden_levels {
                if !training_levels.contains(group_digest) {
                    return Err(CompositionalAuditError::HiddenFactorLevelNotInTraining {
                        interaction_id: interaction_id.clone(),
                        dimension: *dimension,
                        group_digest: group_digest.clone(),
                    });
                }
                if !calibration_levels.contains(group_digest) {
                    return Err(CompositionalAuditError::HiddenFactorLevelNotInCalibration {
                        interaction_id: interaction_id.clone(),
                        dimension: *dimension,
                        group_digest: group_digest.clone(),
                    });
                }
            }
        }
        level_coverage.insert(interaction_id.clone(), factor_coverage);
    }

    for cells in cells_by_spec.values_mut() {
        for keys in cells.values_mut() {
            keys.sort();
        }
    }

    Ok(ValidatedManifest {
        specs,
        partitions,
        cells_by_spec,
        level_coverage,
        minimums,
    })
}

fn validate_cases<'a>(
    validated: &ValidatedManifest,
    cases: &'a [PairedCase],
) -> Result<CellCases<'a>, CompositionalAuditError> {
    if cases.is_empty() {
        return Err(CompositionalAuditError::NoPairedCases);
    }
    let all_dimensions: BTreeSet<_> = validated
        .specs
        .values()
        .flat_map(|dimensions| dimensions.iter().copied())
        .collect();
    let mut case_ids = BTreeSet::new();
    let mut paired_manifests = BTreeSet::new();
    let mut cluster_receipts: BTreeMap<String, (String, IndependenceDisposition)> = BTreeMap::new();
    let mut receipt_clusters = BTreeMap::<String, String>::new();
    let mut cells: CellCases<'a> = BTreeMap::new();

    for case in cases {
        require_nonempty("case_id", &case.case_id)?;
        require_nonempty("cluster_id", &case.cluster_id)
            .map_err(|_| CompositionalAuditError::MissingCaseReceipt(case.case_id.clone()))?;
        require_nonempty(
            "paired_case_manifest_digest",
            &case.paired_case_manifest_digest,
        )
        .map_err(|_| CompositionalAuditError::MissingCaseReceipt(case.case_id.clone()))?;
        require_nonempty(
            "independence_receipt_digest",
            &case.independence_receipt_digest,
        )
        .map_err(|_| CompositionalAuditError::MissingCaseReceipt(case.case_id.clone()))?;
        if !case_ids.insert(case.case_id.clone()) {
            return Err(CompositionalAuditError::DuplicateOrEmptyCaseId(
                case.case_id.clone(),
            ));
        }
        if !paired_manifests.insert(case.paired_case_manifest_digest.clone()) {
            return Err(CompositionalAuditError::DuplicatePairedCaseManifest(
                case.paired_case_manifest_digest.clone(),
            ));
        }
        if case.independence_disposition == IndependenceDisposition::Rejected {
            return Err(CompositionalAuditError::RejectedIndependenceEvidence(
                case.cluster_id.clone(),
            ));
        }

        match receipt_clusters.get(&case.independence_receipt_digest) {
            Some(cluster) if cluster != &case.cluster_id => {
                return Err(CompositionalAuditError::ReusedIndependenceReceipt(
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
                return Err(CompositionalAuditError::ClusterReceiptMismatch(
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

        let mut groups = BTreeMap::new();
        for (dimension, digest) in &case.source_groups {
            require_nonempty("case_factor_digest", digest).map_err(|_| {
                CompositionalAuditError::MissingCaseSourceDimension {
                    case_id: case.case_id.clone(),
                    dimension: *dimension,
                }
            })?;
            if groups.insert(*dimension, digest.clone()).is_some() {
                return Err(CompositionalAuditError::DuplicateCaseSourceDimension {
                    case_id: case.case_id.clone(),
                    dimension: *dimension,
                });
            }
            if !all_dimensions.contains(dimension) {
                return Err(CompositionalAuditError::UndeclaredCaseSourceDimension {
                    case_id: case.case_id.clone(),
                    dimension: *dimension,
                });
            }
        }
        for dimension in &all_dimensions {
            if !groups.contains_key(dimension) {
                return Err(CompositionalAuditError::MissingCaseSourceDimension {
                    case_id: case.case_id.clone(),
                    dimension: *dimension,
                });
            }
        }

        for (interaction_id, dimensions) in &validated.specs {
            let key = InteractionKey {
                factors: dimensions
                    .iter()
                    .map(|dimension| (*dimension, groups[dimension].clone()))
                    .collect(),
            };
            let partition = validated
                .partitions
                .get(&(interaction_id.clone(), key.clone()))
                .copied();
            if partition != Some(Partition::HiddenConfirmatory) {
                return Err(CompositionalAuditError::CaseCellNotHidden {
                    case_id: case.case_id.clone(),
                    interaction_id: interaction_id.clone(),
                    partition,
                });
            }
            cells
                .entry((interaction_id.clone(), key))
                .or_default()
                .push(case);
        }
    }
    Ok(cells)
}

fn summarize_cell(
    interaction_id: &str,
    key: &InteractionKey,
    cases: &[&PairedCase],
) -> Result<InteractionCellReport, CompositionalAuditError> {
    let mut candidate_safe = 0_u64;
    let mut baseline_safe = 0_u64;
    let mut candidate_unsafe = 0_u64;
    let mut baseline_unsafe = 0_u64;
    let mut candidate_unknown = 0_u64;
    let mut baseline_unknown = 0_u64;
    let mut clusters = BTreeSet::new();
    let mut unknown = false;

    for case in cases {
        clusters.insert(case.cluster_id.as_str());
        if case.independence_disposition == IndependenceDisposition::Unknown
            || matches!(case.candidate_outcome, Outcome::Unknown | Outcome::Censored)
            || matches!(case.baseline_outcome, Outcome::Unknown | Outcome::Censored)
        {
            unknown = true;
        }
        if case.candidate_outcome == Outcome::SafeSuccess {
            candidate_safe = candidate_safe
                .checked_add(1)
                .ok_or(CompositionalAuditError::ArithmeticOverflow)?;
        }
        if case.baseline_outcome == Outcome::SafeSuccess {
            baseline_safe = baseline_safe
                .checked_add(1)
                .ok_or(CompositionalAuditError::ArithmeticOverflow)?;
        }
        if case.candidate_outcome == Outcome::UnsafeSuccess {
            candidate_unsafe = candidate_unsafe
                .checked_add(1)
                .ok_or(CompositionalAuditError::ArithmeticOverflow)?;
        }
        if case.baseline_outcome == Outcome::UnsafeSuccess {
            baseline_unsafe = baseline_unsafe
                .checked_add(1)
                .ok_or(CompositionalAuditError::ArithmeticOverflow)?;
        }
        if matches!(case.candidate_outcome, Outcome::Unknown | Outcome::Censored) {
            candidate_unknown = candidate_unknown
                .checked_add(1)
                .ok_or(CompositionalAuditError::ArithmeticOverflow)?;
        }
        if matches!(case.baseline_outcome, Outcome::Unknown | Outcome::Censored) {
            baseline_unknown = baseline_unknown
                .checked_add(1)
                .ok_or(CompositionalAuditError::ArithmeticOverflow)?;
        }
    }

    let episode_count =
        u64::try_from(cases.len()).map_err(|_| CompositionalAuditError::ArithmeticOverflow)?;
    let candidate_rate = safe_rate_ppm(candidate_safe, episode_count)?;
    let baseline_rate = safe_rate_ppm(baseline_safe, episode_count)?;
    Ok(InteractionCellReport {
        interaction_id: interaction_id.to_owned(),
        key: key.clone(),
        episode_count,
        independent_cluster_count: to_u16(clusters.len())?,
        candidate_safe_successes: candidate_safe,
        baseline_safe_successes: baseline_safe,
        candidate_safe_success_ppm: Some(candidate_rate),
        baseline_safe_success_ppm: Some(baseline_rate),
        paired_safe_success_effect_ppm: Some(i64::from(candidate_rate) - i64::from(baseline_rate)),
        candidate_unsafe_successes: candidate_unsafe,
        baseline_unsafe_successes: baseline_unsafe,
        candidate_unknown_or_censored: candidate_unknown,
        baseline_unknown_or_censored: baseline_unknown,
        disposition: if unknown {
            CellDisposition::Inconclusive
        } else {
            CellDisposition::Complete
        },
    })
}

/// Audit a factorial holdout over unseen combinations of familiar factor levels.
///
/// This function emits descriptive per-cell summaries only. It does not calculate
/// inferential statistics, verify source-label truth, or authorize actuation.
pub fn audit_compositional_shift(
    input: &CompositionalAuditInput,
) -> Result<CompositionalAuditReport, CompositionalAuditError> {
    let validated = validate_manifest(&input.manifest)?;
    validate_arm(&input.manifest, &input.candidate, &input.baseline)?;
    let case_cells = validate_cases(&validated, &input.cases)?;

    let mut reports = Vec::new();
    let mut overall_inconclusive = false;

    for (interaction_id, dimensions) in &validated.specs {
        let partition_cells = validated
            .cells_by_spec
            .get(interaction_id)
            .cloned()
            .unwrap_or_default();
        let training_count = to_u16(
            partition_cells
                .get(&Partition::Training)
                .map_or(0, Vec::len),
        )?;
        let calibration_count = to_u16(
            partition_cells
                .get(&Partition::Calibration)
                .map_or(0, Vec::len),
        )?;
        let hidden_keys = partition_cells
            .get(&Partition::HiddenConfirmatory)
            .cloned()
            .unwrap_or_default();
        let hidden_count = to_u16(hidden_keys.len())?;
        let mut cells = Vec::with_capacity(hidden_keys.len());
        let mut candidate_rates = Vec::new();
        let mut worst_effect: Option<i64> = None;
        let mut observed_hidden = 0_usize;

        for key in &hidden_keys {
            let cases = case_cells
                .get(&(interaction_id.clone(), key.clone()))
                .map(Vec::as_slice)
                .unwrap_or(&[]);
            let report = if cases.is_empty() {
                InteractionCellReport {
                    interaction_id: interaction_id.clone(),
                    key: key.clone(),
                    episode_count: 0,
                    independent_cluster_count: 0,
                    candidate_safe_successes: 0,
                    baseline_safe_successes: 0,
                    candidate_safe_success_ppm: None,
                    baseline_safe_success_ppm: None,
                    paired_safe_success_effect_ppm: None,
                    candidate_unsafe_successes: 0,
                    baseline_unsafe_successes: 0,
                    candidate_unknown_or_censored: 0,
                    baseline_unknown_or_censored: 0,
                    disposition: CellDisposition::Inconclusive,
                }
            } else {
                observed_hidden = observed_hidden
                    .checked_add(1)
                    .ok_or(CompositionalAuditError::ArithmeticOverflow)?;
                summarize_cell(interaction_id, key, cases)?
            };
            if let Some(rate) = report.candidate_safe_success_ppm {
                candidate_rates.push(rate);
            }
            if let Some(effect) = report.paired_safe_success_effect_ppm {
                worst_effect = Some(worst_effect.map_or(effect, |current| current.min(effect)));
            }
            if report.disposition == CellDisposition::Inconclusive {
                overall_inconclusive = true;
            }
            cells.push(report);
        }

        let coverage = validated
            .level_coverage
            .get(interaction_id)
            .cloned()
            .unwrap_or_default();
        let factor_level_coverage = dimensions
            .iter()
            .map(|dimension| {
                let training = coverage
                    .get(&Partition::Training)
                    .and_then(|levels| levels.get(dimension))
                    .map_or(0, BTreeSet::len);
                let calibration = coverage
                    .get(&Partition::Calibration)
                    .and_then(|levels| levels.get(dimension))
                    .map_or(0, BTreeSet::len);
                let hidden = coverage
                    .get(&Partition::HiddenConfirmatory)
                    .and_then(|levels| levels.get(dimension))
                    .map_or(0, BTreeSet::len);
                Ok(FactorLevelCoverage {
                    dimension: *dimension,
                    training_level_count: to_u16(training)?,
                    calibration_level_count: to_u16(calibration)?,
                    hidden_level_count: to_u16(hidden)?,
                })
            })
            .collect::<Result<Vec<_>, CompositionalAuditError>>()?;

        candidate_rates.sort_unstable();
        let rate_range = candidate_rates
            .first()
            .copied()
            .zip(candidate_rates.last().copied());
        let disposition = if usize::from(to_u16(observed_hidden)?) == hidden_keys.len()
            && cells
                .iter()
                .all(|cell| cell.disposition == CellDisposition::Complete)
        {
            CellDisposition::Complete
        } else {
            overall_inconclusive = true;
            CellDisposition::Inconclusive
        };

        reports.push(InteractionReport {
            interaction_id: interaction_id.clone(),
            dimensions: dimensions.clone(),
            training_cell_count: training_count,
            calibration_cell_count: calibration_count,
            hidden_cell_count: hidden_count,
            observed_hidden_cell_count: to_u16(observed_hidden)?,
            factor_level_coverage,
            candidate_rate_range_ppm: rate_range,
            worst_cell_effect_ppm: worst_effect,
            cells,
            disposition,
        });
    }

    reports.sort_by(|a, b| a.interaction_id.cmp(&b.interaction_id));
    Ok(CompositionalAuditReport {
        benchmark_id: input.manifest.benchmark_id.clone(),
        candidate_arm_id: input.candidate.arm_id.clone(),
        baseline_arm_id: input.baseline.arm_id.clone(),
        interactions: reports,
        disposition: if overall_inconclusive {
            CellDisposition::Inconclusive
        } else {
            CellDisposition::Complete
        },
        actuation_authorized: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arm(arm_id: &str) -> ArmManifest {
        ArmManifest {
            arm_id: arm_id.into(),
            model_artifact_digest: format!("{arm_id}-model"),
            policy_digest: format!("{arm_id}-policy"),
            harness_digest: format!("{arm_id}-harness"),
            benchmark_id: "robotics-composition".into(),
            benchmark_version: "1.0.0".into(),
            dataset_manifest_digest: "dataset-sha256".into(),
            source_manifest_digest: "source-sha256".into(),
            grouping_policy_digest: "factor-policy-sha256".into(),
            split_policy_digest: "composition-split-sha256".into(),
            hidden_partition_digest: "hidden-cells-sha256".into(),
            task_manifest_digest: "task-manifest-sha256".into(),
            environment_manifest_digest: "environment-manifest-sha256".into(),
            robot_configuration_digest: "robot-config-sha256".into(),
            evaluator_digest: "evaluator-sha256".into(),
            resource_budget_digest: "resource-budget-sha256".into(),
        }
    }

    fn key(site: &str, morphology: &str) -> InteractionKey {
        InteractionKey {
            factors: vec![
                (SourceDimension::CollectionSite, site.into()),
                (SourceDimension::RobotMorphology, morphology.into()),
            ],
        }
    }

    fn assignment(
        interaction_id: &str,
        key: InteractionKey,
        partition: Partition,
    ) -> InteractionAssignment {
        InteractionAssignment {
            interaction_id: interaction_id.into(),
            key,
            partition,
            assignment_receipt_digest: format!("receipt-{interaction_id}-{:?}", partition),
        }
    }

    fn pair_assignments() -> Vec<InteractionAssignment> {
        vec![
            assignment(
                "site-morphology",
                key("site-a", "morph-x"),
                Partition::Training,
            ),
            assignment(
                "site-morphology",
                key("site-b", "morph-y"),
                Partition::Training,
            ),
            assignment(
                "site-morphology",
                key("site-c", "morph-z"),
                Partition::Training,
            ),
            assignment(
                "site-morphology",
                key("site-a", "morph-y"),
                Partition::Calibration,
            ),
            assignment(
                "site-morphology",
                key("site-b", "morph-z"),
                Partition::Calibration,
            ),
            assignment(
                "site-morphology",
                key("site-c", "morph-x"),
                Partition::Calibration,
            ),
            assignment(
                "site-morphology",
                key("site-a", "morph-z"),
                Partition::HiddenConfirmatory,
            ),
            assignment(
                "site-morphology",
                key("site-b", "morph-x"),
                Partition::HiddenConfirmatory,
            ),
            assignment(
                "site-morphology",
                key("site-c", "morph-y"),
                Partition::HiddenConfirmatory,
            ),
        ]
    }

    fn case(
        id: &str,
        site: &str,
        morphology: &str,
        candidate: Outcome,
        baseline: Outcome,
    ) -> PairedCase {
        PairedCase {
            case_id: id.into(),
            cluster_id: format!("cluster-{id}"),
            paired_case_manifest_digest: format!("paired-{id}"),
            independence_receipt_digest: format!("independence-{id}"),
            independence_disposition: IndependenceDisposition::Qualified,
            source_groups: vec![
                (SourceDimension::CollectionSite, site.into()),
                (SourceDimension::RobotMorphology, morphology.into()),
            ],
            candidate_outcome: candidate,
            baseline_outcome: baseline,
        }
    }

    fn input() -> CompositionalAuditInput {
        CompositionalAuditInput {
            manifest: CompositionalManifest {
                benchmark_id: "robotics-composition".into(),
                benchmark_version: "1.0.0".into(),
                dataset_manifest_digest: "dataset-sha256".into(),
                source_manifest_digest: "source-sha256".into(),
                grouping_policy_digest: "factor-policy-sha256".into(),
                split_policy_digest: "composition-split-sha256".into(),
                hidden_partition_digest: "hidden-cells-sha256".into(),
                interactions: vec![InteractionSpec {
                    interaction_id: "site-morphology".into(),
                    dimensions: vec![
                        SourceDimension::CollectionSite,
                        SourceDimension::RobotMorphology,
                    ],
                    minimum_training_cells: 3,
                    minimum_calibration_cells: 3,
                    minimum_hidden_cells: 3,
                }],
                assignments: pair_assignments(),
            },
            candidate: arm("candidate"),
            baseline: arm("baseline"),
            cases: vec![
                case(
                    "one",
                    "site-a",
                    "morph-z",
                    Outcome::SafeSuccess,
                    Outcome::Failure,
                ),
                case(
                    "two",
                    "site-b",
                    "morph-x",
                    Outcome::Failure,
                    Outcome::SafeSuccess,
                ),
                case(
                    "three",
                    "site-c",
                    "morph-y",
                    Outcome::UnsafeSuccess,
                    Outcome::Failure,
                ),
            ],
        }
    }

    #[test]
    fn unseen_joint_cells_are_qualified_when_each_factor_level_is_familiar() {
        let report = audit_compositional_shift(&input()).unwrap();
        assert_eq!(report.disposition, CellDisposition::Complete);
        let interaction = &report.interactions[0];
        assert_eq!(interaction.training_cell_count, 3);
        assert_eq!(interaction.calibration_cell_count, 3);
        assert_eq!(interaction.hidden_cell_count, 3);
        assert_eq!(interaction.observed_hidden_cell_count, 3);
        assert_eq!(interaction.cells[0].episode_count, 1);
        assert!(!report.actuation_authorized);
        assert!(interaction.factor_level_coverage.iter().all(|level| {
            level.training_level_count == 3
                && level.calibration_level_count == 3
                && level.hidden_level_count == 3
        }));
    }

    #[test]
    fn a_hidden_factor_level_missing_from_training_rejects_the_compositional_claim() {
        let mut audit = input();
        audit.manifest.assignments.retain(|assignment| {
            !(assignment.partition == Partition::Training
                && assignment
                    .key
                    .factors
                    .contains(&(SourceDimension::CollectionSite, "site-c".into())))
        });
        assert!(matches!(
            audit_compositional_shift(&audit),
            Err(CompositionalAuditError::HiddenFactorLevelNotInTraining {
                interaction_id,
                dimension: SourceDimension::CollectionSite,
                group_digest
            }) if interaction_id == "site-morphology" && group_digest == "site-c"
        ));
    }

    #[test]
    fn a_hidden_factor_level_missing_from_calibration_rejects_the_compositional_claim() {
        let mut audit = input();
        audit.manifest.assignments.retain(|assignment| {
            !(assignment.partition == Partition::Calibration
                && assignment
                    .key
                    .factors
                    .contains(&(SourceDimension::RobotMorphology, "morph-y".into())))
        });
        assert!(matches!(
            audit_compositional_shift(&audit),
            Err(CompositionalAuditError::HiddenFactorLevelNotInCalibration {
                interaction_id,
                dimension: SourceDimension::RobotMorphology,
                group_digest
            }) if interaction_id == "site-morphology" && group_digest == "morph-y"
        ));
    }

    #[test]
    fn the_same_cell_cannot_be_assigned_to_train_and_hidden() {
        let mut audit = input();
        audit.manifest.assignments.push(assignment(
            "site-morphology",
            key("site-a", "morph-z"),
            Partition::Training,
        ));
        assert!(matches!(
            audit_compositional_shift(&audit),
            Err(CompositionalAuditError::DuplicateInteractionCell { .. })
        ));
    }

    #[test]
    fn reversed_factor_order_is_canonical_and_report_invariant() {
        let mut audit = input();
        let report_a = audit_compositional_shift(&audit).unwrap();
        for assignment in &mut audit.manifest.assignments {
            assignment.key.factors.reverse();
        }
        for case in &mut audit.cases {
            case.source_groups.reverse();
        }
        let report_b = audit_compositional_shift(&audit).unwrap();
        assert_eq!(report_a, report_b);
    }

    #[test]
    fn unknown_or_censored_outcome_makes_only_its_cell_inconclusive() {
        let mut audit = input();
        audit.cases[0].candidate_outcome = Outcome::Unknown;
        let report = audit_compositional_shift(&audit).unwrap();
        assert_eq!(report.disposition, CellDisposition::Inconclusive);
        assert_eq!(
            report.interactions[0].cells[0].disposition,
            CellDisposition::Inconclusive
        );
        assert_eq!(
            report.interactions[0].cells[1].disposition,
            CellDisposition::Complete
        );
    }

    #[test]
    fn missing_hidden_cell_is_retained_as_inconclusive_not_omitted() {
        let mut audit = input();
        audit.cases.pop();
        let report = audit_compositional_shift(&audit).unwrap();
        assert_eq!(report.interactions[0].hidden_cell_count, 3);
        assert_eq!(report.interactions[0].observed_hidden_cell_count, 2);
        assert_eq!(report.interactions[0].cells.len(), 3);
        assert_eq!(report.disposition, CellDisposition::Inconclusive);
    }

    #[test]
    fn duplicate_dimensions_in_an_interaction_are_rejected() {
        let mut audit = input();
        audit.manifest.interactions[0]
            .dimensions
            .push(SourceDimension::CollectionSite);
        assert!(matches!(
            audit_compositional_shift(&audit),
            Err(CompositionalAuditError::DuplicateInteractionDimension {
                dimension: SourceDimension::CollectionSite,
                ..
            })
        ));
    }

    #[test]
    fn interaction_requires_two_or_three_dimensions() {
        let mut audit = input();
        audit.manifest.interactions[0].dimensions = vec![SourceDimension::CollectionSite];
        assert!(matches!(
            audit_compositional_shift(&audit),
            Err(CompositionalAuditError::InvalidInteractionDimensionCount { .. })
        ));
    }

    #[test]
    fn missing_assignment_receipt_is_rejected() {
        let mut audit = input();
        audit.manifest.assignments[0]
            .assignment_receipt_digest
            .clear();
        assert!(matches!(
            audit_compositional_shift(&audit),
            Err(CompositionalAuditError::MissingInteractionAssignmentReceipt { .. })
        ));
    }

    #[test]
    fn duplicate_case_id_is_rejected() {
        let mut audit = input();
        audit.cases[1].case_id = audit.cases[0].case_id.clone();
        assert!(matches!(
            audit_compositional_shift(&audit),
            Err(CompositionalAuditError::DuplicateOrEmptyCaseId(_))
        ));
    }

    #[test]
    fn duplicate_paired_scenario_identity_cannot_inflate_cell_count() {
        let mut audit = input();
        audit.cases[1].paired_case_manifest_digest =
            audit.cases[0].paired_case_manifest_digest.clone();
        assert!(matches!(
            audit_compositional_shift(&audit),
            Err(CompositionalAuditError::DuplicatePairedCaseManifest(_))
        ));
    }

    #[test]
    fn one_independence_receipt_cannot_be_reused_for_two_clusters() {
        let mut audit = input();
        audit.cases[1].independence_receipt_digest =
            audit.cases[0].independence_receipt_digest.clone();
        assert!(matches!(
            audit_compositional_shift(&audit),
            Err(CompositionalAuditError::ReusedIndependenceReceipt(_))
        ));
    }

    #[test]
    fn contradictory_receipts_inside_one_cluster_are_rejected() {
        let mut audit = input();
        audit.cases[1].cluster_id = audit.cases[0].cluster_id.clone();
        assert!(matches!(
            audit_compositional_shift(&audit),
            Err(CompositionalAuditError::ClusterReceiptMismatch(_))
        ));
    }

    #[test]
    fn rejected_independence_evidence_is_rejected() {
        let mut audit = input();
        audit.cases[0].independence_disposition = IndependenceDisposition::Rejected;
        assert!(matches!(
            audit_compositional_shift(&audit),
            Err(CompositionalAuditError::RejectedIndependenceEvidence(_))
        ));
    }

    #[test]
    fn missing_source_dimension_in_a_case_is_rejected() {
        let mut audit = input();
        audit.cases[0].source_groups.pop();
        assert!(matches!(
            audit_compositional_shift(&audit),
            Err(CompositionalAuditError::MissingCaseSourceDimension {
                dimension: SourceDimension::RobotMorphology,
                ..
            })
        ));
    }

    #[test]
    fn undeclared_extra_source_dimension_is_rejected() {
        let mut audit = input();
        audit.cases[0]
            .source_groups
            .push((SourceDimension::PayloadRegime, "payload-light".into()));
        assert!(matches!(
            audit_compositional_shift(&audit),
            Err(CompositionalAuditError::UndeclaredCaseSourceDimension {
                dimension: SourceDimension::PayloadRegime,
                ..
            })
        ));
    }

    #[test]
    fn duplicate_source_dimension_in_a_case_is_rejected() {
        let mut audit = input();
        audit.cases[0]
            .source_groups
            .push((SourceDimension::CollectionSite, "site-other".into()));
        assert!(matches!(
            audit_compositional_shift(&audit),
            Err(CompositionalAuditError::DuplicateCaseSourceDimension {
                dimension: SourceDimension::CollectionSite,
                ..
            })
        ));
    }

    #[test]
    fn a_case_from_a_training_cell_cannot_enter_the_hidden_results() {
        let mut audit = input();
        audit.cases[0].source_groups = vec![
            (SourceDimension::CollectionSite, "site-a".into()),
            (SourceDimension::RobotMorphology, "morph-x".into()),
        ];
        assert!(matches!(
            audit_compositional_shift(&audit),
            Err(CompositionalAuditError::CaseCellNotHidden {
                partition: Some(Partition::Training),
                ..
            })
        ));
    }

    #[test]
    fn candidate_and_baseline_must_share_the_evaluation_scope() {
        let mut audit = input();
        audit.candidate.resource_budget_digest = "different-budget".into();
        assert!(matches!(
            audit_compositional_shift(&audit),
            Err(CompositionalAuditError::ArmScopeMismatch {
                field: "resource_budget_digest",
                ..
            })
        ));
    }

    #[test]
    fn all_hidden_cells_remain_visible_when_one_has_a_severe_regression() {
        let mut audit = input();
        audit.cases[0].candidate_outcome = Outcome::Failure;
        audit.cases[0].baseline_outcome = Outcome::SafeSuccess;
        let report = audit_compositional_shift(&audit).unwrap();
        assert_eq!(
            report.interactions[0].worst_cell_effect_ppm,
            Some(-1_000_000)
        );
        assert_eq!(report.interactions[0].cells.len(), 3);
    }

    #[test]
    fn pair_and_triple_interactions_are_reported_separately() {
        let mut audit = input();
        let triple_dimensions = vec![
            SourceDimension::CollectionSite,
            SourceDimension::RobotMorphology,
            SourceDimension::PayloadRegime,
        ];
        let triple = vec![
            (vec!["site-a", "morph-x", "payload-1"], Partition::Training),
            (vec!["site-b", "morph-y", "payload-2"], Partition::Training),
            (vec!["site-c", "morph-z", "payload-3"], Partition::Training),
            (
                vec!["site-a", "morph-y", "payload-2"],
                Partition::Calibration,
            ),
            (
                vec!["site-b", "morph-z", "payload-3"],
                Partition::Calibration,
            ),
            (
                vec!["site-c", "morph-x", "payload-1"],
                Partition::Calibration,
            ),
            (
                vec!["site-a", "morph-z", "payload-1"],
                Partition::HiddenConfirmatory,
            ),
            (
                vec!["site-b", "morph-x", "payload-2"],
                Partition::HiddenConfirmatory,
            ),
            (
                vec!["site-c", "morph-y", "payload-3"],
                Partition::HiddenConfirmatory,
            ),
        ];
        audit.manifest.interactions.push(InteractionSpec {
            interaction_id: "site-morphology-payload".into(),
            dimensions: triple_dimensions,
            minimum_training_cells: 3,
            minimum_calibration_cells: 3,
            minimum_hidden_cells: 3,
        });
        for (values, partition) in triple {
            audit.manifest.assignments.push(InteractionAssignment {
                interaction_id: "site-morphology-payload".into(),
                key: InteractionKey {
                    factors: vec![
                        (SourceDimension::CollectionSite, values[0].into()),
                        (SourceDimension::RobotMorphology, values[1].into()),
                        (SourceDimension::PayloadRegime, values[2].into()),
                    ],
                },
                partition,
                assignment_receipt_digest: format!("triple-{values:?}-{partition:?}"),
            });
        }
        for (i, case) in audit.cases.iter_mut().enumerate() {
            let (site, morphology, payload) = match i {
                0 => ("site-a", "morph-z", "payload-1"),
                1 => ("site-b", "morph-x", "payload-2"),
                _ => ("site-c", "morph-y", "payload-3"),
            };
            case.source_groups
                .push((SourceDimension::PayloadRegime, payload.into()));
            assert_eq!(case.source_groups[0].1, site);
            assert_eq!(case.source_groups[1].1, morphology);
        }
        let report = audit_compositional_shift(&audit).unwrap();
        assert_eq!(report.interactions.len(), 2);
        assert_eq!(report.disposition, CellDisposition::Complete);
        assert_eq!(report.interactions[0].interaction_id, "site-morphology");
        assert_eq!(
            report.interactions[1].interaction_id,
            "site-morphology-payload"
        );
    }

    #[test]
    fn duplicate_interaction_cell_assignment_is_rejected_even_if_key_order_differs() {
        let mut audit = input();
        audit.manifest.assignments.push(InteractionAssignment {
            interaction_id: "site-morphology".into(),
            key: InteractionKey {
                factors: vec![
                    (SourceDimension::RobotMorphology, "morph-x".into()),
                    (SourceDimension::CollectionSite, "site-a".into()),
                ],
            },
            partition: Partition::Calibration,
            assignment_receipt_digest: "duplicate-receipt".into(),
        });
        assert!(matches!(
            audit_compositional_shift(&audit),
            Err(CompositionalAuditError::DuplicateInteractionCell { .. })
        ));
    }

    #[test]
    fn case_order_permutation_does_not_change_report() {
        let mut audit = input();
        let first = audit_compositional_shift(&audit).unwrap();
        audit.cases.reverse();
        let second = audit_compositional_shift(&audit).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn interaction_assignment_order_permutation_does_not_change_report() {
        let mut audit = input();
        let first = audit_compositional_shift(&audit).unwrap();
        audit.manifest.assignments.reverse();
        let second = audit_compositional_shift(&audit).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn minimum_cell_counts_are_enforced() {
        let mut audit = input();
        audit.manifest.interactions[0].minimum_hidden_cells = 4;
        assert!(matches!(
            audit_compositional_shift(&audit),
            Err(CompositionalAuditError::InsufficientPartitionCells {
                partition: Partition::HiddenConfirmatory,
                minimum: 4,
                observed: 3,
                ..
            })
        ));
    }

    #[test]
    fn unknown_independence_makes_affected_cell_inconclusive() {
        let mut audit = input();
        audit.cases[1].independence_disposition = IndependenceDisposition::Unknown;
        let report = audit_compositional_shift(&audit).unwrap();
        assert_eq!(report.disposition, CellDisposition::Inconclusive);
        assert_eq!(
            report.interactions[0].cells[1].disposition,
            CellDisposition::Inconclusive
        );
    }
}

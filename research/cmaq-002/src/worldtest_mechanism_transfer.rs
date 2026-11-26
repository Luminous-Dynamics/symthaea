//! Cross-query transfer across held-out environment mechanisms.
//!
//! This is the frozen-transfer (zero-shot reuse) track. It evaluates the same
//! exploration-derived checkpoint across 5 mechanism-transfer conditions and
//! 3 WorldTest query families. Scores are first averaged within independent
//! environment clusters, then equally across clusters. Task rows are never
//! treated as independent experimental subjects.
//!
//! The evaluator does not authenticate evaluator-assigned identities, run an
//! environment, compute inferential confidence intervals, or establish
//! performance superiority. Those responsibilities remain with the trusted
//! launcher, signed evidence receipt, and #8578 inference path.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use super::worldtest_query_portfolio::{
    QueryFamily, TaskDisposition, WORLDTEST_PORTFOLIO_QUERY_FAMILY_COUNT,
    WorldTestPortfolioContext, WorldTestPortfolioError, WorldTestPortfolioInput,
    WorldTestQueryPortfolioCertificate, WorldTestQueryTask, evaluate_worldtest_query_portfolio,
};

pub const WORLDTEST_MECHANISM_CONDITION_COUNT: usize = 5;
pub const WORLDTEST_MECHANISM_TRANSFER_CELL_COUNT: usize =
    WORLDTEST_MECHANISM_CONDITION_COUNT * WORLDTEST_PORTFOLIO_QUERY_FAMILY_COUNT;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum MechanismTransferCondition {
    SameFamilyHoldout,
    NuisanceShift,
    ParameterShift,
    CompositionalShift,
    StructuralShift,
}

impl MechanismTransferCondition {
    pub const ALL: [Self; WORLDTEST_MECHANISM_CONDITION_COUNT] = [
        Self::SameFamilyHoldout,
        Self::NuisanceShift,
        Self::ParameterShift,
        Self::CompositionalShift,
        Self::StructuralShift,
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::SameFamilyHoldout => "same_family_holdout",
            Self::NuisanceShift => "nuisance_shift",
            Self::ParameterShift => "parameter_shift",
            Self::CompositionalShift => "compositional_shift",
            Self::StructuralShift => "structural_shift",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MechanismRelation {
    Invariant,
    ParameterPerturbed,
    CompositionHeldOut,
    StructureReplaced,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ObservationRelation {
    Invariant,
    NuisancePerturbed,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TargetFamilyRelation {
    SameAsSource,
    NewCompositionFamily,
    NewStructuralFamily,
}

/// Evaluator-owned declaration of how a target condition relates to the
/// source environment. The manifest digest binds the full generator contract.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MechanismConditionProfile {
    pub condition: MechanismTransferCondition,
    pub mechanism_relation: MechanismRelation,
    pub observation_relation: ObservationRelation,
    pub target_family_relation: TargetFamilyRelation,
    pub profile_manifest_sha256: [u8; 32],
}

/// One source exploration state, crossed with held-out target conditions.
/// Adaptation is intentionally not represented here; it belongs in a
/// separate evaluator with a separate checkpoint and resource ledger.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MechanismTransferContext {
    pub portfolio: WorldTestPortfolioContext,
    pub source_environment_family_id: String,
    pub source_generator_manifest_sha256: [u8; 32],
    pub transfer_design_manifest_sha256: [u8; 32],
    pub minimum_independent_clusters_per_cell: u64,
    pub condition_profiles: Vec<MechanismConditionProfile>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MechanismTransferTask {
    pub condition: MechanismTransferCondition,
    pub task: WorldTestQueryTask,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MechanismTransferInput {
    pub context: MechanismTransferContext,
    /// Exact task inventory committed before confirmatory outcomes are opened.
    pub expected_task_ids: Vec<String>,
    pub tasks: Vec<MechanismTransferTask>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MechanismTransferCellSummary {
    pub condition: MechanismTransferCondition,
    pub query_family: QueryFamily,
    pub task_count: u64,
    pub independent_cluster_count: u64,
    pub completed_count: u64,
    pub failed_count: u64,
    pub timed_out_count: u64,
    pub invalid_count: u64,
    pub abstained_count: u64,
    /// Diagnostic only: task-row-weighted mean.
    pub task_weighted_mean_score_ppm: u32,
    /// Primary descriptive cell score: equal-weighted mean of cluster means.
    pub mean_score_ppm: u32,
    pub cluster_score_sum_ppm: u128,
    pub minimum_cluster_mean_score_ppm: u32,
    pub maximum_cluster_mean_score_ppm: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MechanismTransferClusterCellSummary {
    pub query_family: QueryFamily,
    pub task_count: u64,
    pub completed_count: u64,
    pub failed_count: u64,
    pub timed_out_count: u64,
    pub invalid_count: u64,
    pub abstained_count: u64,
    pub score_sum_ppm: u128,
    /// Mean within this independent environment-generation cluster.
    pub mean_score_ppm: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MechanismTransferClusterSummary {
    pub condition: MechanismTransferCondition,
    pub environment_family_id: String,
    pub environment_cluster_id: String,
    pub task_count: u64,
    /// Sorted in QueryFamily::ALL order; absent families are not fabricated.
    pub query_family_summaries: Vec<MechanismTransferClusterCellSummary>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MechanismTransferCellKey {
    pub condition: MechanismTransferCondition,
    pub query_family: QueryFamily,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MechanismTransferCertificate {
    pub context: MechanismTransferContext,
    pub base_portfolio: WorldTestQueryPortfolioCertificate,
    pub cell_summaries: Vec<MechanismTransferCellSummary>,
    pub cluster_summaries: Vec<MechanismTransferClusterSummary>,
    pub minimum_cell: MechanismTransferCellKey,
    pub maximum_cell: MechanismTransferCellKey,
    pub minimum_cell_mean_score_ppm: u32,
    pub maximum_cell_mean_score_ppm: u32,
    pub cell_mean_gap_ppm: u32,
    pub all_required_cells_present: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MechanismTransferError {
    Base(WorldTestPortfolioError),
    EmptySourceEnvironmentFamilyId,
    EmptySourceGeneratorManifestDigest,
    EmptyTransferDesignManifestDigest,
    ZeroMinimumIndependentClusters,
    TooManyTasks {
        requested: usize,
        maximum: usize,
    },
    MissingConditionProfile {
        condition: MechanismTransferCondition,
    },
    DuplicateConditionProfile {
        condition: MechanismTransferCondition,
    },
    EmptyConditionProfileDigest {
        condition: MechanismTransferCondition,
    },
    ConditionProfileRelationMismatch {
        condition: MechanismTransferCondition,
    },
    TargetEnvironmentFamilyMismatch {
        condition: MechanismTransferCondition,
        source_family_id: String,
        target_family_id: String,
    },
    TaskUsesUndeclaredCondition {
        condition: MechanismTransferCondition,
    },
    ClusterSpansMechanismConditions {
        cluster_id: String,
    },
    MissingCell {
        condition: MechanismTransferCondition,
        query_family: QueryFamily,
    },
    TooFewIndependentClusters {
        condition: MechanismTransferCondition,
        query_family: QueryFamily,
        observed: u64,
        required: u64,
    },
    ArithmeticOverflow,
}

impl fmt::Display for MechanismTransferError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Base(error) => write!(formatter, "base WorldTest portfolio rejected: {error}"),
            Self::EmptySourceEnvironmentFamilyId => {
                write!(
                    formatter,
                    "source environment family ID is empty or untrimmed"
                )
            }
            Self::EmptySourceGeneratorManifestDigest => {
                write!(formatter, "source generator manifest digest is empty")
            }
            Self::EmptyTransferDesignManifestDigest => {
                write!(formatter, "transfer design manifest digest is empty")
            }
            Self::ZeroMinimumIndependentClusters => {
                write!(
                    formatter,
                    "minimum independent clusters per cell must be nonzero"
                )
            }
            Self::TooManyTasks { requested, maximum } => write!(
                formatter,
                "mechanism-transfer task count {requested} exceeds maximum {maximum}"
            ),
            Self::MissingConditionProfile { condition } => write!(
                formatter,
                "mechanism-transfer profile is missing {}",
                condition.as_str()
            ),
            Self::DuplicateConditionProfile { condition } => write!(
                formatter,
                "mechanism-transfer profile repeats {}",
                condition.as_str()
            ),
            Self::EmptyConditionProfileDigest { condition } => write!(
                formatter,
                "mechanism-transfer profile digest is empty for {}",
                condition.as_str()
            ),
            Self::ConditionProfileRelationMismatch { condition } => write!(
                formatter,
                "mechanism and observation relations do not match declared condition {}",
                condition.as_str()
            ),
            Self::TargetEnvironmentFamilyMismatch {
                condition,
                source_family_id,
                target_family_id,
            } => write!(
                formatter,
                "target environment family {target_family_id} does not satisfy the source-family relationship for {} (source {source_family_id})",
                condition.as_str()
            ),
            Self::TaskUsesUndeclaredCondition { condition } => write!(
                formatter,
                "task uses undeclared mechanism-transfer condition {}",
                condition.as_str()
            ),
            Self::ClusterSpansMechanismConditions { cluster_id } => write!(
                formatter,
                "environment cluster {cluster_id} is reused across mechanism conditions"
            ),
            Self::MissingCell {
                condition,
                query_family,
            } => write!(
                formatter,
                "mechanism-transfer matrix is missing cell ({}, {})",
                condition.as_str(),
                query_family.as_str()
            ),
            Self::TooFewIndependentClusters {
                condition,
                query_family,
                observed,
                required,
            } => write!(
                formatter,
                "cell ({}, {}) has {observed} independent clusters; {required} required",
                condition.as_str(),
                query_family.as_str()
            ),
            Self::ArithmeticOverflow => {
                write!(formatter, "mechanism-transfer score arithmetic overflowed")
            }
        }
    }
}

impl std::error::Error for MechanismTransferError {}

impl From<WorldTestPortfolioError> for MechanismTransferError {
    fn from(value: WorldTestPortfolioError) -> Self {
        Self::Base(value)
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct DispositionCounts {
    completed: u64,
    failed: u64,
    timed_out: u64,
    invalid: u64,
    abstained: u64,
}

impl DispositionCounts {
    fn add(&mut self, disposition: TaskDisposition) -> Result<(), MechanismTransferError> {
        let counter = match disposition {
            TaskDisposition::Completed { .. } => &mut self.completed,
            TaskDisposition::Failed { .. } => &mut self.failed,
            TaskDisposition::TimedOut { .. } => &mut self.timed_out,
            TaskDisposition::Invalid { .. } => &mut self.invalid,
            TaskDisposition::Abstained { .. } => &mut self.abstained,
        };
        *counter = counter
            .checked_add(1)
            .ok_or(MechanismTransferError::ArithmeticOverflow)?;
        Ok(())
    }
}

#[derive(Clone, Debug, Default)]
struct ClusterFamilyAccumulator {
    task_count: u64,
    score_sum_ppm: u128,
    dispositions: DispositionCounts,
}

impl ClusterFamilyAccumulator {
    fn add(&mut self, disposition: TaskDisposition) -> Result<(), MechanismTransferError> {
        self.task_count = self
            .task_count
            .checked_add(1)
            .ok_or(MechanismTransferError::ArithmeticOverflow)?;
        self.score_sum_ppm = self
            .score_sum_ppm
            .checked_add(u128::from(disposition.score_ppm()))
            .ok_or(MechanismTransferError::ArithmeticOverflow)?;
        self.dispositions.add(disposition)
    }

    fn mean_score_ppm(&self) -> Result<u32, MechanismTransferError> {
        if self.task_count == 0 {
            return Err(MechanismTransferError::ArithmeticOverflow);
        }
        u32::try_from(self.score_sum_ppm / u128::from(self.task_count))
            .map_err(|_| MechanismTransferError::ArithmeticOverflow)
    }
}

#[derive(Clone, Debug)]
struct ClusterAccumulator {
    condition: MechanismTransferCondition,
    environment_family_id: String,
    environment_cluster_id: String,
    task_count: u64,
    query_families: BTreeMap<QueryFamily, ClusterFamilyAccumulator>,
}

#[derive(Clone, Debug, Default)]
struct CellAccumulator {
    task_count: u64,
    task_score_sum_ppm: u128,
    dispositions: DispositionCounts,
    /// One score per independent cluster, already averaged within that cluster.
    cluster_means_ppm: BTreeMap<String, u32>,
}

fn expected_relations(
    condition: MechanismTransferCondition,
) -> (MechanismRelation, ObservationRelation, TargetFamilyRelation) {
    match condition {
        MechanismTransferCondition::SameFamilyHoldout => (
            MechanismRelation::Invariant,
            ObservationRelation::Invariant,
            TargetFamilyRelation::SameAsSource,
        ),
        MechanismTransferCondition::NuisanceShift => (
            MechanismRelation::Invariant,
            ObservationRelation::NuisancePerturbed,
            TargetFamilyRelation::SameAsSource,
        ),
        MechanismTransferCondition::ParameterShift => (
            MechanismRelation::ParameterPerturbed,
            ObservationRelation::Invariant,
            TargetFamilyRelation::SameAsSource,
        ),
        MechanismTransferCondition::CompositionalShift => (
            MechanismRelation::CompositionHeldOut,
            ObservationRelation::Invariant,
            TargetFamilyRelation::NewCompositionFamily,
        ),
        MechanismTransferCondition::StructuralShift => (
            MechanismRelation::StructureReplaced,
            ObservationRelation::Invariant,
            TargetFamilyRelation::NewStructuralFamily,
        ),
    }
}

fn digest_is_empty(digest: &[u8; 32]) -> bool {
    digest.iter().all(|byte| *byte == 0)
}

fn validate_context(context: &MechanismTransferContext) -> Result<(), MechanismTransferError> {
    if context.source_environment_family_id.trim().is_empty()
        || context.source_environment_family_id.trim()
            != context.source_environment_family_id.as_str()
    {
        return Err(MechanismTransferError::EmptySourceEnvironmentFamilyId);
    }
    if digest_is_empty(&context.source_generator_manifest_sha256) {
        return Err(MechanismTransferError::EmptySourceGeneratorManifestDigest);
    }
    if digest_is_empty(&context.transfer_design_manifest_sha256) {
        return Err(MechanismTransferError::EmptyTransferDesignManifestDigest);
    }
    if context.minimum_independent_clusters_per_cell == 0 {
        return Err(MechanismTransferError::ZeroMinimumIndependentClusters);
    }

    let mut profiles = BTreeMap::new();
    for profile in &context.condition_profiles {
        if digest_is_empty(&profile.profile_manifest_sha256) {
            return Err(MechanismTransferError::EmptyConditionProfileDigest {
                condition: profile.condition,
            });
        }
        let expected = expected_relations(profile.condition);
        if (
            profile.mechanism_relation,
            profile.observation_relation,
            profile.target_family_relation,
        ) != expected
        {
            return Err(MechanismTransferError::ConditionProfileRelationMismatch {
                condition: profile.condition,
            });
        }
        if profiles.insert(profile.condition, profile).is_some() {
            return Err(MechanismTransferError::DuplicateConditionProfile {
                condition: profile.condition,
            });
        }
    }
    for condition in MechanismTransferCondition::ALL {
        if !profiles.contains_key(&condition) {
            return Err(MechanismTransferError::MissingConditionProfile { condition });
        }
    }
    Ok(())
}

/// Evaluates frozen transfer using the exact same source checkpoint for all
/// fifteen query-family × mechanism-condition cells.
///
/// The base portfolio evaluator owns task identity, source-checkpoint
/// provenance, budgets, query-family scoring, cluster/family identity, and
/// failure-denominator semantics. This layer adds only the mechanism axis and
/// equal-cluster cell aggregation.
pub fn evaluate_mechanism_transfer(
    input: &MechanismTransferInput,
) -> Result<MechanismTransferCertificate, MechanismTransferError> {
    validate_context(&input.context)?;
    if input.tasks.len() > 20_480 {
        return Err(MechanismTransferError::TooManyTasks {
            requested: input.tasks.len(),
            maximum: 20_480,
        });
    }

    let declared_conditions = input
        .context
        .condition_profiles
        .iter()
        .map(|profile| profile.condition)
        .collect::<BTreeSet<_>>();

    let base_input = WorldTestPortfolioInput {
        context: input.context.portfolio.clone(),
        expected_task_ids: input.expected_task_ids.clone(),
        tasks: input.tasks.iter().map(|row| row.task.clone()).collect(),
    };
    let base_portfolio = evaluate_worldtest_query_portfolio(&base_input)?;

    let mut cluster_condition = BTreeMap::<String, MechanismTransferCondition>::new();
    let mut clusters = BTreeMap::<(MechanismTransferCondition, String), ClusterAccumulator>::new();
    let mut cells = BTreeMap::<(MechanismTransferCondition, QueryFamily), CellAccumulator>::new();

    for row in &input.tasks {
        if !declared_conditions.contains(&row.condition) {
            return Err(MechanismTransferError::TaskUsesUndeclaredCondition {
                condition: row.condition,
            });
        }

        let task = &row.task;
        let profile = input
            .context
            .condition_profiles
            .iter()
            .find(|profile| profile.condition == row.condition)
            .ok_or(MechanismTransferError::TaskUsesUndeclaredCondition {
                condition: row.condition,
            })?;
        let same_source_family =
            task.environment_family_id == input.context.source_environment_family_id;
        let family_relation_valid = match profile.target_family_relation {
            TargetFamilyRelation::SameAsSource => same_source_family,
            TargetFamilyRelation::NewCompositionFamily
            | TargetFamilyRelation::NewStructuralFamily => !same_source_family,
        };
        if !family_relation_valid {
            return Err(MechanismTransferError::TargetEnvironmentFamilyMismatch {
                condition: row.condition,
                source_family_id: input.context.source_environment_family_id.clone(),
                target_family_id: task.environment_family_id.clone(),
            });
        }
        if let Some(existing) =
            cluster_condition.insert(task.environment_cluster_id.clone(), row.condition)
        {
            if existing != row.condition {
                return Err(MechanismTransferError::ClusterSpansMechanismConditions {
                    cluster_id: task.environment_cluster_id.clone(),
                });
            }
        }

        let cluster_key = (row.condition, task.environment_cluster_id.clone());
        let cluster = clusters
            .entry(cluster_key)
            .or_insert_with(|| ClusterAccumulator {
                condition: row.condition,
                environment_family_id: task.environment_family_id.clone(),
                environment_cluster_id: task.environment_cluster_id.clone(),
                task_count: 0,
                query_families: BTreeMap::new(),
            });
        cluster.task_count = cluster
            .task_count
            .checked_add(1)
            .ok_or(MechanismTransferError::ArithmeticOverflow)?;
        cluster
            .query_families
            .entry(task.query_family)
            .or_default()
            .add(task.disposition)?;

        let cell = cells.entry((row.condition, task.query_family)).or_default();
        cell.task_count = cell
            .task_count
            .checked_add(1)
            .ok_or(MechanismTransferError::ArithmeticOverflow)?;
        cell.task_score_sum_ppm = cell
            .task_score_sum_ppm
            .checked_add(u128::from(task.disposition.score_ppm()))
            .ok_or(MechanismTransferError::ArithmeticOverflow)?;
        cell.dispositions.add(task.disposition)?;
    }

    // Each cluster contributes at most one mean to each query-family cell.
    for cluster in clusters.values() {
        for (family, family_accumulator) in &cluster.query_families {
            let cell = cells.get_mut(&(cluster.condition, *family)).ok_or(
                MechanismTransferError::MissingCell {
                    condition: cluster.condition,
                    query_family: *family,
                },
            )?;
            cell.cluster_means_ppm.insert(
                cluster.environment_cluster_id.clone(),
                family_accumulator.mean_score_ppm()?,
            );
        }
    }

    let mut cell_summaries = Vec::with_capacity(WORLDTEST_MECHANISM_TRANSFER_CELL_COUNT);
    for condition in MechanismTransferCondition::ALL {
        for query_family in QueryFamily::ALL {
            let cell = cells.get(&(condition, query_family)).ok_or(
                MechanismTransferError::MissingCell {
                    condition,
                    query_family,
                },
            )?;
            let cluster_count = u64::try_from(cell.cluster_means_ppm.len())
                .map_err(|_| MechanismTransferError::ArithmeticOverflow)?;
            if cluster_count < input.context.minimum_independent_clusters_per_cell {
                return Err(MechanismTransferError::TooFewIndependentClusters {
                    condition,
                    query_family,
                    observed: cluster_count,
                    required: input.context.minimum_independent_clusters_per_cell,
                });
            }
            let cluster_score_sum_ppm =
                cell.cluster_means_ppm
                    .values()
                    .try_fold(0_u128, |sum, score| {
                        sum.checked_add(u128::from(*score))
                            .ok_or(MechanismTransferError::ArithmeticOverflow)
                    })?;
            let mean_score_ppm = u32::try_from(cluster_score_sum_ppm / u128::from(cluster_count))
                .map_err(|_| MechanismTransferError::ArithmeticOverflow)?;
            let minimum_cluster_mean_score_ppm =
                cell.cluster_means_ppm.values().copied().min().ok_or(
                    MechanismTransferError::MissingCell {
                        condition,
                        query_family,
                    },
                )?;
            let maximum_cluster_mean_score_ppm =
                cell.cluster_means_ppm.values().copied().max().ok_or(
                    MechanismTransferError::MissingCell {
                        condition,
                        query_family,
                    },
                )?;
            let task_weighted_mean_score_ppm =
                u32::try_from(cell.task_score_sum_ppm / u128::from(cell.task_count))
                    .map_err(|_| MechanismTransferError::ArithmeticOverflow)?;

            cell_summaries.push(MechanismTransferCellSummary {
                condition,
                query_family,
                task_count: cell.task_count,
                independent_cluster_count: cluster_count,
                completed_count: cell.dispositions.completed,
                failed_count: cell.dispositions.failed,
                timed_out_count: cell.dispositions.timed_out,
                invalid_count: cell.dispositions.invalid,
                abstained_count: cell.dispositions.abstained,
                task_weighted_mean_score_ppm,
                mean_score_ppm,
                cluster_score_sum_ppm,
                minimum_cluster_mean_score_ppm,
                maximum_cluster_mean_score_ppm,
            });
        }
    }

    let mut cluster_summaries = Vec::with_capacity(clusters.len());
    for (_, cluster) in clusters {
        let mut query_family_summaries = Vec::with_capacity(cluster.query_families.len());
        for query_family in QueryFamily::ALL {
            if let Some(summary) = cluster.query_families.get(&query_family) {
                query_family_summaries.push(MechanismTransferClusterCellSummary {
                    query_family,
                    task_count: summary.task_count,
                    completed_count: summary.dispositions.completed,
                    failed_count: summary.dispositions.failed,
                    timed_out_count: summary.dispositions.timed_out,
                    invalid_count: summary.dispositions.invalid,
                    abstained_count: summary.dispositions.abstained,
                    score_sum_ppm: summary.score_sum_ppm,
                    mean_score_ppm: summary.mean_score_ppm()?,
                });
            }
        }
        cluster_summaries.push(MechanismTransferClusterSummary {
            condition: cluster.condition,
            environment_family_id: cluster.environment_family_id,
            environment_cluster_id: cluster.environment_cluster_id,
            task_count: cluster.task_count,
            query_family_summaries,
        });
    }

    let mut minimum_cell = None::<(u32, MechanismTransferCellKey)>;
    let mut maximum_cell = None::<(u32, MechanismTransferCellKey)>;
    for summary in &cell_summaries {
        let key = MechanismTransferCellKey {
            condition: summary.condition,
            query_family: summary.query_family,
        };
        if minimum_cell
            .as_ref()
            .is_none_or(|(value, _)| summary.mean_score_ppm < *value)
        {
            minimum_cell = Some((summary.mean_score_ppm, key.clone()));
        }
        if maximum_cell
            .as_ref()
            .is_none_or(|(value, _)| summary.mean_score_ppm > *value)
        {
            maximum_cell = Some((summary.mean_score_ppm, key));
        }
    }
    let (minimum_cell_mean_score_ppm, minimum_cell) =
        minimum_cell.ok_or(MechanismTransferError::ArithmeticOverflow)?;
    let (maximum_cell_mean_score_ppm, maximum_cell) =
        maximum_cell.ok_or(MechanismTransferError::ArithmeticOverflow)?;
    let cell_mean_gap_ppm = maximum_cell_mean_score_ppm
        .checked_sub(minimum_cell_mean_score_ppm)
        .ok_or(MechanismTransferError::ArithmeticOverflow)?;

    Ok(MechanismTransferCertificate {
        context: input.context.clone(),
        base_portfolio,
        cell_summaries,
        cluster_summaries,
        minimum_cell,
        maximum_cell,
        minimum_cell_mean_score_ppm,
        maximum_cell_mean_score_ppm,
        cell_mean_gap_ppm,
        all_required_cells_present: true,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::worldtest_query_portfolio::{
        ExplorationResourceUse, FamilyMetricProfile, PortfolioResourceBudget,
    };

    fn digest(value: u8) -> [u8; 32] {
        [value; 32]
    }

    fn context() -> MechanismTransferContext {
        let portfolio = WorldTestPortfolioContext {
            portfolio_id: 7001,
            candidate_artifact_sha256: digest(1),
            model_checkpoint_sha256: digest(2),
            adapter_runtime_sha256: digest(3),
            evaluator_profile_sha256: digest(4),
            exploration_program_sha256: digest(5),
            exploration_trace_sha256: digest(6),
            exploration_checkpoint_sha256: digest(7),
            seed_commitment_sha256: digest(8),
            budget_manifest_sha256: digest(9),
            evaluation_split_sha256: digest(10),
            exploration_resources: ExplorationResourceUse {
                action_count: 8,
                reset_count: 1,
                model_query_count: 4,
                elapsed_us: 100,
                measured_energy_uj: Some(10),
            },
            resource_budget: PortfolioResourceBudget {
                max_exploration_actions: 8,
                max_exploration_resets: 1,
                max_exploration_total_interactions: 9,
                max_exploration_model_queries: 4,
                max_exploration_elapsed_us: 100,
                max_exploration_energy_uj: Some(10),
                max_total_test_interactions: 10_000,
                max_total_test_model_queries: 10_000,
                max_total_test_elapsed_us: 10_000,
                max_total_test_energy_uj: Some(10_000),
            },
            family_metric_profiles: vec![
                FamilyMetricProfile {
                    family: QueryFamily::MaskedFramePrediction,
                    metric_profile_sha256: digest(11),
                },
                FamilyMetricProfile {
                    family: QueryFamily::ChangeDetection,
                    metric_profile_sha256: digest(12),
                },
                FamilyMetricProfile {
                    family: QueryFamily::Planning,
                    metric_profile_sha256: digest(13),
                },
            ],
        };
        let condition_profiles = MechanismTransferCondition::ALL
            .into_iter()
            .enumerate()
            .map(|(index, condition)| {
                let (mechanism_relation, observation_relation, target_family_relation) =
                    expected_relations(condition);
                MechanismConditionProfile {
                    condition,
                    mechanism_relation,
                    observation_relation,
                    target_family_relation,
                    profile_manifest_sha256: digest(30 + index as u8),
                }
            })
            .collect();

        MechanismTransferContext {
            portfolio,
            source_environment_family_id: "source-family-v1".to_owned(),
            source_generator_manifest_sha256: digest(21),
            transfer_design_manifest_sha256: digest(22),
            minimum_independent_clusters_per_cell: 2,
            condition_profiles,
        }
    }

    fn task(
        run_id: u64,
        condition: MechanismTransferCondition,
        query_family: QueryFamily,
        cluster_number: u8,
        score_ppm: u32,
    ) -> MechanismTransferTask {
        let family_id = match condition {
            MechanismTransferCondition::SameFamilyHoldout
            | MechanismTransferCondition::NuisanceShift
            | MechanismTransferCondition::ParameterShift => "source-family-v1".to_owned(),
            MechanismTransferCondition::CompositionalShift
            | MechanismTransferCondition::StructuralShift => {
                format!("target-{}", condition.as_str())
            }
        };
        let cluster_id = format!("{}-cluster-{cluster_number}", condition.as_str());
        let task_id = format!(
            "{}-{}-{}-{}",
            condition.as_str(),
            query_family.as_str(),
            cluster_number,
            run_id
        );
        MechanismTransferTask {
            condition,
            task: WorldTestQueryTask {
                task_id,
                run_id,
                query_family,
                environment_family_id: family_id,
                environment_cluster_id: cluster_id,
                challenge_program_sha256: digest(50 + run_id as u8),
                metric_profile_sha256: digest(match query_family {
                    QueryFamily::MaskedFramePrediction => 11,
                    QueryFamily::ChangeDetection => 12,
                    QueryFamily::Planning => 13,
                }),
                candidate_artifact_sha256: digest(1),
                exploration_trace_sha256: digest(6),
                exploration_checkpoint_sha256: digest(7),
                disposition: TaskDisposition::Completed { score_ppm },
                test_interaction_count: 1,
                test_model_query_count: 1,
                test_elapsed_us: 2,
                measured_energy_uj: Some(1),
            },
        }
    }

    fn base_input() -> MechanismTransferInput {
        let mut tasks = Vec::new();
        let mut expected_task_ids = Vec::new();
        let mut run_id = 1_u64;
        for condition in MechanismTransferCondition::ALL {
            for cluster_number in 1..=2 {
                for query_family in QueryFamily::ALL {
                    let row = task(run_id, condition, query_family, cluster_number, 800_000);
                    expected_task_ids.push(row.task.task_id.clone());
                    tasks.push(row);
                    run_id += 1;
                }
            }
        }
        MechanismTransferInput {
            context: context(),
            expected_task_ids,
            tasks,
        }
    }

    #[test]
    fn emits_complete_fifteen_cell_matrix_and_cluster_balanced_scores() {
        let result = evaluate_mechanism_transfer(&base_input()).unwrap();
        assert_eq!(
            result.cell_summaries.len(),
            WORLDTEST_MECHANISM_TRANSFER_CELL_COUNT
        );
        assert_eq!(result.cluster_summaries.len(), 10);
        assert!(result.all_required_cells_present);
        assert!(result.cell_summaries.iter().all(|cell| {
            cell.task_count == 2
                && cell.independent_cluster_count == 2
                && cell.mean_score_ppm == 800_000
                && cell.task_weighted_mean_score_ppm == 800_000
        }));
        assert_eq!(result.minimum_cell_mean_score_ppm, 800_000);
        assert_eq!(result.maximum_cell_mean_score_ppm, 800_000);
        assert_eq!(result.cell_mean_gap_ppm, 0);
    }

    #[test]
    fn one_weak_structural_planning_cell_is_not_hidden() {
        let mut input = base_input();
        for row in &mut input.tasks {
            if row.condition == MechanismTransferCondition::StructuralShift
                && row.task.query_family == QueryFamily::Planning
            {
                row.task.disposition = TaskDisposition::Completed { score_ppm: 100_000 };
            }
        }
        let result = evaluate_mechanism_transfer(&input).unwrap();
        assert_eq!(result.minimum_cell_mean_score_ppm, 100_000);
        assert_eq!(result.cell_mean_gap_ppm, 700_000);
        assert_eq!(
            result.minimum_cell,
            MechanismTransferCellKey {
                condition: MechanismTransferCondition::StructuralShift,
                query_family: QueryFamily::Planning,
            }
        );
    }

    #[test]
    fn missing_mechanism_query_cell_is_rejected() {
        let mut input = base_input();
        let removed_ids = input
            .tasks
            .iter()
            .filter(|row| {
                row.condition == MechanismTransferCondition::StructuralShift
                    && row.task.query_family == QueryFamily::Planning
            })
            .map(|row| row.task.task_id.clone())
            .collect::<BTreeSet<_>>();
        input
            .tasks
            .retain(|row| !removed_ids.contains(&row.task.task_id));
        input
            .expected_task_ids
            .retain(|id| !removed_ids.contains(id));
        assert!(matches!(
            evaluate_mechanism_transfer(&input),
            Err(MechanismTransferError::MissingCell {
                condition: MechanismTransferCondition::StructuralShift,
                query_family: QueryFamily::Planning,
            })
        ));
    }

    #[test]
    fn preregistered_minimum_cluster_count_is_enforced_per_cell() {
        let mut input = base_input();
        let omitted = input
            .tasks
            .iter()
            .position(|row| {
                row.condition == MechanismTransferCondition::NuisanceShift
                    && row.task.query_family == QueryFamily::ChangeDetection
                    && row.task.environment_cluster_id.ends_with("cluster-2")
            })
            .unwrap();
        let removed = input.tasks.remove(omitted);
        input
            .expected_task_ids
            .retain(|id| id != &removed.task.task_id);
        assert!(matches!(
            evaluate_mechanism_transfer(&input),
            Err(MechanismTransferError::TooFewIndependentClusters {
                condition: MechanismTransferCondition::NuisanceShift,
                query_family: QueryFamily::ChangeDetection,
                observed: 1,
                required: 2,
            })
        ));
    }

    #[test]
    fn multiple_query_tasks_in_one_environment_are_one_cluster_per_cell() {
        let mut input = base_input();
        let first = input.tasks[0].clone();
        let mut repeated = first.clone();
        repeated.task.run_id = 1000;
        repeated.task.task_id = "same-cluster-second-mfp-task".to_owned();
        repeated.task.disposition = TaskDisposition::Completed { score_ppm: 600_000 };
        input.expected_task_ids.push(repeated.task.task_id.clone());
        input.tasks.push(repeated);
        let result = evaluate_mechanism_transfer(&input).unwrap();
        let cell = result
            .cell_summaries
            .iter()
            .find(|cell| {
                cell.condition == first.condition && cell.query_family == first.task.query_family
            })
            .unwrap();
        assert_eq!(cell.task_count, 3);
        assert_eq!(cell.independent_cluster_count, 2);
        assert_eq!(cell.task_weighted_mean_score_ppm, 733_333);
        assert_eq!(cell.mean_score_ppm, 750_000);
    }

    #[test]
    fn cluster_id_cannot_be_reused_across_mechanism_conditions() {
        let mut input = base_input();
        let source_cluster = input.tasks[0].task.environment_cluster_id.clone();
        let target = input
            .tasks
            .iter_mut()
            .find(|row| {
                row.condition == MechanismTransferCondition::NuisanceShift
                    && row.task.query_family == QueryFamily::MaskedFramePrediction
                    && row.task.environment_cluster_id.ends_with("cluster-1")
            })
            .unwrap();
        target.task.environment_cluster_id = source_cluster.clone();
        target.task.environment_family_id = "source-family-v1".to_owned();
        assert!(matches!(
            evaluate_mechanism_transfer(&input),
            Err(MechanismTransferError::ClusterSpansMechanismConditions { cluster_id })
                if cluster_id == source_cluster
        ));
    }

    #[test]
    fn cluster_can_span_query_families_within_one_condition() {
        let result = evaluate_mechanism_transfer(&base_input()).unwrap();
        let cluster = result
            .cluster_summaries
            .iter()
            .find(|cluster| {
                cluster.condition == MechanismTransferCondition::SameFamilyHoldout
                    && cluster.environment_cluster_id.ends_with("cluster-1")
            })
            .unwrap();
        assert_eq!(cluster.task_count, 3);
        assert_eq!(cluster.query_family_summaries.len(), 3);
    }

    #[test]
    fn task_order_does_not_change_canonical_matrix_or_cluster_summaries() {
        let original = base_input();
        let mut shuffled = original.clone();
        shuffled.tasks.reverse();
        shuffled.expected_task_ids.reverse();
        assert_eq!(
            evaluate_mechanism_transfer(&original).unwrap(),
            evaluate_mechanism_transfer(&shuffled).unwrap()
        );
    }

    #[test]
    fn failed_and_abstained_tasks_remain_in_cluster_and_cell_denominators() {
        let mut input = base_input();
        let row = input
            .tasks
            .iter_mut()
            .find(|row| {
                row.condition == MechanismTransferCondition::ParameterShift
                    && row.task.query_family == QueryFamily::Planning
                    && row.task.environment_cluster_id.ends_with("cluster-1")
            })
            .unwrap();
        row.task.disposition = TaskDisposition::Failed { score_ppm: 0 };
        let result = evaluate_mechanism_transfer(&input).unwrap();
        let cell = result
            .cell_summaries
            .iter()
            .find(|cell| {
                cell.condition == MechanismTransferCondition::ParameterShift
                    && cell.query_family == QueryFamily::Planning
            })
            .unwrap();
        assert_eq!(cell.task_count, 2);
        assert_eq!(cell.failed_count, 1);
        assert_eq!(cell.mean_score_ppm, 400_000);
        let cluster = result
            .cluster_summaries
            .iter()
            .find(|cluster| {
                cluster.condition == MechanismTransferCondition::ParameterShift
                    && cluster.environment_cluster_id.ends_with("cluster-1")
            })
            .unwrap();
        let planning = cluster
            .query_family_summaries
            .iter()
            .find(|row| row.query_family == QueryFamily::Planning)
            .unwrap();
        assert_eq!(planning.failed_count, 1);
        assert_eq!(planning.mean_score_ppm, 0);
    }

    #[test]
    fn invalid_source_environment_family_is_rejected() {
        let mut input = base_input();
        input.context.source_environment_family_id = " source-family-v1".to_owned();
        assert!(matches!(
            evaluate_mechanism_transfer(&input),
            Err(MechanismTransferError::EmptySourceEnvironmentFamilyId)
        ));
    }

    #[test]
    fn source_and_transfer_manifest_digests_are_required() {
        let mut input = base_input();
        input.context.source_generator_manifest_sha256 = [0; 32];
        assert!(matches!(
            evaluate_mechanism_transfer(&input),
            Err(MechanismTransferError::EmptySourceGeneratorManifestDigest)
        ));
        let mut input = base_input();
        input.context.transfer_design_manifest_sha256 = [0; 32];
        assert!(matches!(
            evaluate_mechanism_transfer(&input),
            Err(MechanismTransferError::EmptyTransferDesignManifestDigest)
        ));
    }

    #[test]
    fn condition_profiles_must_be_complete_and_unique() {
        let mut missing = base_input();
        missing.context.condition_profiles.pop();
        assert!(matches!(
            evaluate_mechanism_transfer(&missing),
            Err(MechanismTransferError::MissingConditionProfile { .. })
        ));
        let mut duplicate = base_input();
        duplicate
            .context
            .condition_profiles
            .push(duplicate.context.condition_profiles[0].clone());
        assert!(matches!(
            evaluate_mechanism_transfer(&duplicate),
            Err(MechanismTransferError::DuplicateConditionProfile { .. })
        ));
    }

    #[test]
    fn condition_profile_relations_must_match_declared_shift_semantics() {
        let mut input = base_input();
        input.context.condition_profiles[1].mechanism_relation =
            MechanismRelation::ParameterPerturbed;
        assert!(matches!(
            evaluate_mechanism_transfer(&input),
            Err(MechanismTransferError::ConditionProfileRelationMismatch {
                condition: MechanismTransferCondition::NuisanceShift
            })
        ));
    }

    #[test]
    fn target_environment_family_must_match_declared_transfer_relation() {
        let mut input = base_input();
        for row in &mut input.tasks {
            if row.condition == MechanismTransferCondition::SameFamilyHoldout {
                row.task.environment_family_id = "incorrect-target-family".to_owned();
            }
        }
        assert!(matches!(
            evaluate_mechanism_transfer(&input),
            Err(MechanismTransferError::TargetEnvironmentFamilyMismatch {
                condition: MechanismTransferCondition::SameFamilyHoldout,
                ..
            })
        ));
    }

    #[test]
    fn condition_profile_digest_must_be_nonempty() {
        let mut input = base_input();
        input.context.condition_profiles[2].profile_manifest_sha256 = [0; 32];
        assert!(matches!(
            evaluate_mechanism_transfer(&input),
            Err(MechanismTransferError::EmptyConditionProfileDigest {
                condition: MechanismTransferCondition::ParameterShift
            })
        ));
    }

    #[test]
    fn mutated_shared_checkpoint_is_rejected_by_base_portfolio_contract() {
        let mut input = base_input();
        input.tasks[0].task.exploration_checkpoint_sha256 = digest(99);
        assert!(matches!(
            evaluate_mechanism_transfer(&input),
            Err(MechanismTransferError::Base(
                WorldTestPortfolioError::TaskProvenanceMismatch {
                    field: "exploration_checkpoint_sha256",
                    ..
                }
            ))
        ));
    }

    #[test]
    fn shared_exploration_cost_is_charged_once_not_per_matrix_cell() {
        let result = evaluate_mechanism_transfer(&base_input()).unwrap();
        assert_eq!(
            result.base_portfolio.exploration_total_charged_interactions,
            9
        );
        assert_eq!(result.base_portfolio.total_test_interaction_count, 30);
        assert_eq!(result.base_portfolio.task_count, 30);
    }
}

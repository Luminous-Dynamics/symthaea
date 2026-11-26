//! Cross-query portfolio evaluation for shared WorldTest exploration histories.
//!
//! One frozen exploration history/checkpoint is evaluated against masked-frame
//! prediction, change detection and planning tasks. The certificate retains a
//! three-family score vector and a minimum-family robustness floor; it does
//! not compute a grand mean across semantically different query families.
//!
//! Task scores are already normalized by frozen, family-specific evaluator
//! profiles to integer PPM. This module does not calibrate scores, authenticate
//! receipts, run an agent or perform statistical inference. The trusted
//! launcher must verify evidence and enforce #9437 OS isolation.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

pub const WORLDTEST_PORTFOLIO_SCORE_PPM: u32 = 1_000_000;
pub const WORLDTEST_PORTFOLIO_MAX_TASKS: usize = 4_096;
pub const WORLDTEST_PORTFOLIO_QUERY_FAMILY_COUNT: usize = 3;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum QueryFamily {
    MaskedFramePrediction,
    ChangeDetection,
    Planning,
}

impl QueryFamily {
    pub const ALL: [Self; WORLDTEST_PORTFOLIO_QUERY_FAMILY_COUNT] = [
        Self::MaskedFramePrediction,
        Self::ChangeDetection,
        Self::Planning,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::MaskedFramePrediction => "masked_frame_prediction",
            Self::ChangeDetection => "change_detection",
            Self::Planning => "planning",
        }
    }

    const fn index(self) -> usize {
        match self {
            Self::MaskedFramePrediction => 0,
            Self::ChangeDetection => 1,
            Self::Planning => 2,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FamilyMetricProfile {
    pub family: QueryFamily,
    /// Frozen family-specific normalization and task-scoring semantics.
    pub metric_profile_sha256: [u8; 32],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExplorationResourceUse {
    /// Candidate-issued actions, including go-to-test; resets are separate.
    pub action_count: u64,
    pub reset_count: u64,
    pub model_query_count: u64,
    pub elapsed_us: u64,
    pub measured_energy_uj: Option<u64>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PortfolioResourceBudget {
    pub max_exploration_actions: u64,
    pub max_exploration_resets: u64,
    pub max_exploration_total_interactions: u64,
    pub max_exploration_model_queries: u64,
    pub max_exploration_elapsed_us: u64,
    pub max_exploration_energy_uj: Option<u64>,
    pub max_total_test_interactions: u64,
    pub max_total_test_model_queries: u64,
    pub max_total_test_elapsed_us: u64,
    pub max_total_test_energy_uj: Option<u64>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WorldTestPortfolioContext {
    pub portfolio_id: u64,
    pub candidate_artifact_sha256: [u8; 32],
    pub model_checkpoint_sha256: [u8; 32],
    pub adapter_runtime_sha256: [u8; 32],
    pub evaluator_profile_sha256: [u8; 32],
    pub exploration_program_sha256: [u8; 32],
    pub exploration_trace_sha256: [u8; 32],
    /// One frozen learned state reused for every query-family task.
    pub exploration_checkpoint_sha256: [u8; 32],
    pub seed_commitment_sha256: [u8; 32],
    pub budget_manifest_sha256: [u8; 32],
    pub evaluation_split_sha256: [u8; 32],
    pub exploration_resources: ExplorationResourceUse,
    pub resource_budget: PortfolioResourceBudget,
    pub family_metric_profiles: Vec<FamilyMetricProfile>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TaskDisposition {
    Completed { score_ppm: u32 },
    Failed { score_ppm: u32 },
    TimedOut { score_ppm: u32 },
    Invalid { score_ppm: u32 },
    Abstained { score_ppm: u32 },
}

impl TaskDisposition {
    pub fn score_ppm(self) -> u32 {
        match self {
            Self::Completed { score_ppm }
            | Self::Failed { score_ppm }
            | Self::TimedOut { score_ppm }
            | Self::Invalid { score_ppm }
            | Self::Abstained { score_ppm } => score_ppm,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WorldTestQueryTask {
    pub task_id: String,
    pub run_id: u64,
    pub query_family: QueryFamily,
    /// Evaluator-assigned structural/generator family for split integrity.
    pub environment_family_id: String,
    /// Independent environment-generation cluster; task rows sharing this ID
    /// are repeated observations, not independent statistical subjects.
    pub environment_cluster_id: String,
    pub challenge_program_sha256: [u8; 32],
    pub metric_profile_sha256: [u8; 32],
    pub candidate_artifact_sha256: [u8; 32],
    pub exploration_trace_sha256: [u8; 32],
    pub exploration_checkpoint_sha256: [u8; 32],
    pub disposition: TaskDisposition,
    pub test_interaction_count: u64,
    pub test_model_query_count: u64,
    pub test_elapsed_us: u64,
    pub measured_energy_uj: Option<u64>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WorldTestPortfolioInput {
    pub context: WorldTestPortfolioContext,
    /// Complete task inventory frozen before confirmatory outcomes are opened.
    pub expected_task_ids: Vec<String>,
    pub tasks: Vec<WorldTestQueryTask>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct QueryFamilyPortfolioSummary {
    pub family: QueryFamily,
    pub task_count: u64,
    pub completed_count: u64,
    pub failed_count: u64,
    pub timed_out_count: u64,
    pub invalid_count: u64,
    pub abstained_count: u64,
    pub score_sum_ppm: u128,
    /// Integer floor of score_sum_ppm / task_count; penalties stay included.
    pub mean_score_ppm: u32,
    pub minimum_task_score_ppm: u32,
    pub maximum_task_score_ppm: u32,
    pub test_interaction_count: u64,
    pub test_model_query_count: u64,
    pub test_elapsed_us: u64,
    /// Present only when every task in this family has an energy measurement.
    pub measured_energy_uj: Option<u128>,
    pub energy_measured_task_count: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WorldTestPortfolioTaskResult {
    pub task_id: String,
    pub run_id: u64,
    pub query_family: QueryFamily,
    pub environment_family_id: String,
    pub environment_cluster_id: String,
    pub score_ppm: u32,
    pub disposition: TaskDisposition,
    pub challenge_program_sha256: [u8; 32],
    pub metric_profile_sha256: [u8; 32],
    pub test_interaction_count: u64,
    pub test_model_query_count: u64,
    pub test_elapsed_us: u64,
    pub measured_energy_uj: Option<u64>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WorldTestEnvironmentClusterSummary {
    pub environment_cluster_id: String,
    pub environment_family_id: String,
    pub task_count: u64,
    /// Counts follow QueryFamily::ALL order.
    pub query_family_task_counts: [u64; WORLDTEST_PORTFOLIO_QUERY_FAMILY_COUNT],
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WorldTestQueryPortfolioCertificate {
    pub context: WorldTestPortfolioContext,
    /// Counts are based on evaluator-assigned identities, not task-row count.
    pub environment_family_count: u64,
    pub independent_environment_cluster_count: u64,
    /// Sorted by environment_cluster_id; each cluster contributes once.
    pub environment_cluster_summaries: Vec<WorldTestEnvironmentClusterSummary>,
    pub task_count: u64,
    /// Sorted by task ID for deterministic replay.
    pub task_results: Vec<WorldTestPortfolioTaskResult>,
    /// Exactly three rows, ordered as QueryFamily::ALL.
    pub family_summaries: Vec<QueryFamilyPortfolioSummary>,
    /// Robustness descriptor, not a universal score or SOTA verdict.
    pub minimum_family_mean_ppm: u32,
    pub maximum_family_mean_ppm: u32,
    pub family_mean_gap_ppm: u32,
    pub all_required_families_present: bool,
    pub exploration_total_charged_interactions: u64,
    pub total_test_interaction_count: u64,
    pub total_test_model_query_count: u64,
    pub total_test_elapsed_us: u64,
    /// Present only if every task across all families has an energy measurement.
    pub total_test_measured_energy_uj: Option<u128>,
    pub energy_measured_task_count: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum WorldTestPortfolioError {
    ZeroPortfolioId,
    EmptyDigest { field: &'static str },
    EmptyExpectedTaskSet,
    TooManyTasks { requested: usize, maximum: usize },
    TooFewTasksForFamilies { requested: usize },
    EmptyTaskId,
    EmptyEnvironmentFamilyId { task_id: String },
    EmptyEnvironmentClusterId { task_id: String },
    EnvironmentClusterFamilyMismatch {
        environment_cluster_id: String,
        first_family_id: String,
        conflicting_family_id: String,
    },
    DuplicateExpectedTaskId { task_id: String },
    DuplicateTaskId { task_id: String },
    DuplicateRunId { run_id: u64 },
    ZeroRunId { task_id: String },
    MissingTasks { task_ids: Vec<String> },
    UnexpectedTasks { task_ids: Vec<String> },
    MissingQueryFamily { family: QueryFamily },
    DuplicateFamilyProfile { family: QueryFamily },
    MissingFamilyProfile { family: QueryFamily },
    EmptyFamilyProfileDigest { family: QueryFamily },
    FamilyProfileMismatch { task_id: String, family: QueryFamily },
    TaskProvenanceMismatch { task_id: String, field: &'static str },
    EmptyChallengeProgramDigest { task_id: String },
    ChallengeProgramMatchesExploration { task_id: String },
    EmptyMetricProfileDigest { task_id: String },
    ScoreOutOfRange { task_id: String, score_ppm: u32 },
    ExplorationBudgetExceeded { field: &'static str },
    TestBudgetExceeded { field: &'static str },
    MissingRequiredEnergyMeasurement { phase: &'static str },
    ArithmeticOverflow,
}

impl fmt::Display for WorldTestPortfolioError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroPortfolioId => write!(formatter, "WorldTest portfolio ID must be nonzero"),
            Self::EmptyDigest { field } => write!(formatter, "required portfolio provenance digest {field} is empty"),
            Self::EmptyExpectedTaskSet => write!(formatter, "WorldTest portfolio expected task set must not be empty"),
            Self::TooManyTasks { requested, maximum } => write!(formatter, "WorldTest portfolio task count {requested} exceeds maximum {maximum}"),
            Self::TooFewTasksForFamilies { requested } => write!(formatter, "WorldTest portfolio needs at least one task per query family, received {requested}"),
            Self::EmptyTaskId => write!(formatter, "WorldTest task IDs must be nonempty and trimmed"),
            Self::EmptyEnvironmentFamilyId { task_id } => write!(formatter, "task {task_id} has an empty or untrimmed environment family ID"),
            Self::EmptyEnvironmentClusterId { task_id } => write!(formatter, "task {task_id} has an empty or untrimmed environment cluster ID"),
            Self::EnvironmentClusterFamilyMismatch { environment_cluster_id, first_family_id, conflicting_family_id } => write!(
                formatter,
                "environment cluster {environment_cluster_id} is assigned to both family {first_family_id} and family {conflicting_family_id}"
            ),
            Self::DuplicateExpectedTaskId { task_id } => write!(formatter, "expected task set repeats ID {task_id}"),
            Self::DuplicateTaskId { task_id } => write!(formatter, "portfolio repeats task ID {task_id}"),
            Self::DuplicateRunId { run_id } => write!(formatter, "portfolio repeats task run ID {run_id}"),
            Self::ZeroRunId { task_id } => write!(formatter, "task {task_id} has zero run ID"),
            Self::MissingTasks { task_ids } => write!(formatter, "portfolio is missing expected tasks {task_ids:?}"),
            Self::UnexpectedTasks { task_ids } => write!(formatter, "portfolio contains unexpected tasks {task_ids:?}"),
            Self::MissingQueryFamily { family } => write!(formatter, "portfolio is missing query family {}", family.as_str()),
            Self::DuplicateFamilyProfile { family } => write!(formatter, "portfolio repeats metric profile for {}", family.as_str()),
            Self::MissingFamilyProfile { family } => write!(formatter, "portfolio has no metric profile for {}", family.as_str()),
            Self::EmptyFamilyProfileDigest { family } => write!(formatter, "metric profile digest for {} is empty", family.as_str()),
            Self::FamilyProfileMismatch { task_id, family } => write!(formatter, "task {task_id} does not use the frozen metric profile for {}", family.as_str()),
            Self::TaskProvenanceMismatch { task_id, field } => write!(formatter, "task {task_id} differs from shared portfolio provenance field {field}"),
            Self::EmptyChallengeProgramDigest { task_id } => write!(formatter, "task {task_id} has an empty challenge program digest"),
            Self::ChallengeProgramMatchesExploration { task_id } => write!(formatter, "task {task_id} challenge program is identical to exploration program"),
            Self::EmptyMetricProfileDigest { task_id } => write!(formatter, "task {task_id} metric profile digest is empty"),
            Self::ScoreOutOfRange { task_id, score_ppm } => write!(formatter, "task {task_id} score {score_ppm} is outside [0, {WORLDTEST_PORTFOLIO_SCORE_PPM}]"),
            Self::ExplorationBudgetExceeded { field } => write!(formatter, "shared exploration exceeded declared budget {field}"),
            Self::TestBudgetExceeded { field } => write!(formatter, "cross-query test workload exceeded declared budget {field}"),
            Self::MissingRequiredEnergyMeasurement { phase } => write!(formatter, "energy budget is declared for {phase}, but one or more required measurements are missing"),
            Self::ArithmeticOverflow => write!(formatter, "WorldTest portfolio arithmetic overflowed"),
        }
    }
}

impl std::error::Error for WorldTestPortfolioError {}

fn digest_is_empty(digest: &[u8; 32]) -> bool {
    digest.iter().all(|byte| *byte == 0)
}

fn require_digest(digest: &[u8; 32], field: &'static str) -> Result<(), WorldTestPortfolioError> {
    if digest_is_empty(digest) {
        return Err(WorldTestPortfolioError::EmptyDigest { field });
    }
    Ok(())
}

fn increment(value: &mut u64) -> Result<(), WorldTestPortfolioError> {
    *value = (*value).checked_add(1).ok_or(WorldTestPortfolioError::ArithmeticOverflow)?;
    Ok(())
}

fn add(value: &mut u64, amount: u64) -> Result<(), WorldTestPortfolioError> {
    *value = (*value).checked_add(amount).ok_or(WorldTestPortfolioError::ArithmeticOverflow)?;
    Ok(())
}

fn add_energy(
    value: &mut Option<u128>,
    measurement: Option<u64>,
    measured_count: &mut u64,
) -> Result<(), WorldTestPortfolioError> {
    if let Some(energy) = measurement {
        let total = value.unwrap_or(0)
            .checked_add(u128::from(energy))
            .ok_or(WorldTestPortfolioError::ArithmeticOverflow)?;
        *value = Some(total);
        increment(measured_count)?;
    }
    Ok(())
}

fn empty_family_summary(family: QueryFamily) -> QueryFamilyPortfolioSummary {
    QueryFamilyPortfolioSummary {
        family,
        task_count: 0,
        completed_count: 0,
        failed_count: 0,
        timed_out_count: 0,
        invalid_count: 0,
        abstained_count: 0,
        score_sum_ppm: 0,
        mean_score_ppm: 0,
        minimum_task_score_ppm: WORLDTEST_PORTFOLIO_SCORE_PPM,
        maximum_task_score_ppm: 0,
        test_interaction_count: 0,
        test_model_query_count: 0,
        test_elapsed_us: 0,
        measured_energy_uj: Some(0),
        energy_measured_task_count: 0,
    }
}

pub fn evaluate_worldtest_query_portfolio(
    input: &WorldTestPortfolioInput,
) -> Result<WorldTestQueryPortfolioCertificate, WorldTestPortfolioError> {
    let context = &input.context;
    if context.portfolio_id == 0 {
        return Err(WorldTestPortfolioError::ZeroPortfolioId);
    }

    for (digest, field) in [
        (&context.candidate_artifact_sha256, "candidate_artifact_sha256"),
        (&context.model_checkpoint_sha256, "model_checkpoint_sha256"),
        (&context.adapter_runtime_sha256, "adapter_runtime_sha256"),
        (&context.evaluator_profile_sha256, "evaluator_profile_sha256"),
        (&context.exploration_program_sha256, "exploration_program_sha256"),
        (&context.exploration_trace_sha256, "exploration_trace_sha256"),
        (&context.exploration_checkpoint_sha256, "exploration_checkpoint_sha256"),
        (&context.seed_commitment_sha256, "seed_commitment_sha256"),
        (&context.budget_manifest_sha256, "budget_manifest_sha256"),
        (&context.evaluation_split_sha256, "evaluation_split_sha256"),
    ] {
        require_digest(digest, field)?;
    }

    let resources = context.exploration_resources;
    let budget = context.resource_budget;
    let total_exploration = resources.action_count
        .checked_add(resources.reset_count)
        .ok_or(WorldTestPortfolioError::ArithmeticOverflow)?;
    for (actual, maximum, field) in [
        (resources.action_count, budget.max_exploration_actions, "max_exploration_actions"),
        (resources.reset_count, budget.max_exploration_resets, "max_exploration_resets"),
        (total_exploration, budget.max_exploration_total_interactions, "max_exploration_total_interactions"),
        (resources.model_query_count, budget.max_exploration_model_queries, "max_exploration_model_queries"),
        (resources.elapsed_us, budget.max_exploration_elapsed_us, "max_exploration_elapsed_us"),
    ] {
        if actual > maximum {
            return Err(WorldTestPortfolioError::ExplorationBudgetExceeded { field });
        }
    }
    if let Some(maximum) = budget.max_exploration_energy_uj {
        let measured = resources.measured_energy_uj
            .ok_or(WorldTestPortfolioError::MissingRequiredEnergyMeasurement { phase: "exploration" })?;
        if measured > maximum {
            return Err(WorldTestPortfolioError::ExplorationBudgetExceeded { field: "max_exploration_energy_uj" });
        }
    }

    if input.expected_task_ids.is_empty() {
        return Err(WorldTestPortfolioError::EmptyExpectedTaskSet);
    }
    if input.expected_task_ids.len() > WORLDTEST_PORTFOLIO_MAX_TASKS
        || input.tasks.len() > WORLDTEST_PORTFOLIO_MAX_TASKS
    {
        return Err(WorldTestPortfolioError::TooManyTasks {
            requested: input.expected_task_ids.len().max(input.tasks.len()),
            maximum: WORLDTEST_PORTFOLIO_MAX_TASKS,
        });
    }
    if input.expected_task_ids.len() < WORLDTEST_PORTFOLIO_QUERY_FAMILY_COUNT {
        return Err(WorldTestPortfolioError::TooFewTasksForFamilies {
            requested: input.expected_task_ids.len(),
        });
    }

    let mut expected_ids = BTreeSet::new();
    for task_id in &input.expected_task_ids {
        if task_id.trim().is_empty() || task_id.trim() != task_id.as_str() {
            return Err(WorldTestPortfolioError::EmptyTaskId);
        }
        if !expected_ids.insert(task_id.clone()) {
            return Err(WorldTestPortfolioError::DuplicateExpectedTaskId { task_id: task_id.clone() });
        }
    }

    let mut family_profiles = BTreeMap::<QueryFamily, [u8; 32]>::new();
    for profile in &context.family_metric_profiles {
        if digest_is_empty(&profile.metric_profile_sha256) {
            return Err(WorldTestPortfolioError::EmptyFamilyProfileDigest { family: profile.family });
        }
        if family_profiles.insert(profile.family, profile.metric_profile_sha256).is_some() {
            return Err(WorldTestPortfolioError::DuplicateFamilyProfile { family: profile.family });
        }
    }
    for family in QueryFamily::ALL {
        if !family_profiles.contains_key(&family) {
            return Err(WorldTestPortfolioError::MissingFamilyProfile { family });
        }
    }
    if family_profiles.len() != WORLDTEST_PORTFOLIO_QUERY_FAMILY_COUNT {
        return Err(WorldTestPortfolioError::MissingFamilyProfile { family: QueryFamily::Planning });
    }

    let mut tasks = input.tasks.clone();
    tasks.sort_by(|left, right| left.task_id.cmp(&right.task_id));
    let mut observed_ids = BTreeSet::new();
    let mut run_ids = BTreeSet::new();
    let mut environment_family_ids = BTreeSet::<String>::new();
    let mut environment_clusters =
        BTreeMap::<String, WorldTestEnvironmentClusterSummary>::new();
    let mut family_summaries: BTreeMap<QueryFamily, QueryFamilyPortfolioSummary> =
        QueryFamily::ALL.into_iter().map(|family| (family, empty_family_summary(family))).collect();

    let mut total_test_interactions = 0_u64;
    let mut total_test_model_queries = 0_u64;
    let mut total_test_elapsed_us = 0_u64;
    let mut total_test_energy_uj = Some(0_u128);
    let mut total_energy_measured = 0_u64;

    for task in &tasks {
        if task.task_id.trim().is_empty() || task.task_id.trim() != task.task_id.as_str() {
            return Err(WorldTestPortfolioError::EmptyTaskId);
        }
        if !observed_ids.insert(task.task_id.clone()) {
            return Err(WorldTestPortfolioError::DuplicateTaskId { task_id: task.task_id.clone() });
        }
        if task.run_id == 0 {
            return Err(WorldTestPortfolioError::ZeroRunId { task_id: task.task_id.clone() });
        }
        if !run_ids.insert(task.run_id) {
            return Err(WorldTestPortfolioError::DuplicateRunId { run_id: task.run_id });
        }
        if task.environment_family_id.trim().is_empty()
            || task.environment_family_id.trim() != task.environment_family_id.as_str()
        {
            return Err(WorldTestPortfolioError::EmptyEnvironmentFamilyId {
                task_id: task.task_id.clone(),
            });
        }
        if task.environment_cluster_id.trim().is_empty()
            || task.environment_cluster_id.trim() != task.environment_cluster_id.as_str()
        {
            return Err(WorldTestPortfolioError::EmptyEnvironmentClusterId {
                task_id: task.task_id.clone(),
            });
        }
        environment_family_ids.insert(task.environment_family_id.clone());
        let cluster = environment_clusters
            .entry(task.environment_cluster_id.clone())
            .or_insert_with(|| WorldTestEnvironmentClusterSummary {
                environment_cluster_id: task.environment_cluster_id.clone(),
                environment_family_id: task.environment_family_id.clone(),
                task_count: 0,
                query_family_task_counts: [0; WORLDTEST_PORTFOLIO_QUERY_FAMILY_COUNT],
            });
        if cluster.environment_family_id != task.environment_family_id {
            return Err(WorldTestPortfolioError::EnvironmentClusterFamilyMismatch {
                environment_cluster_id: task.environment_cluster_id.clone(),
                first_family_id: cluster.environment_family_id.clone(),
                conflicting_family_id: task.environment_family_id.clone(),
            });
        }
        increment(&mut cluster.task_count)?;
        increment(&mut cluster.query_family_task_counts[task.query_family.index()])?;
        if !expected_ids.contains(&task.task_id) {
            return Err(WorldTestPortfolioError::UnexpectedTasks { task_ids: vec![task.task_id.clone()] });
        }
        for (actual, expected, field) in [
            (&task.candidate_artifact_sha256, &context.candidate_artifact_sha256, "candidate_artifact_sha256"),
            (&task.exploration_trace_sha256, &context.exploration_trace_sha256, "exploration_trace_sha256"),
            (&task.exploration_checkpoint_sha256, &context.exploration_checkpoint_sha256, "exploration_checkpoint_sha256"),
        ] {
            if actual != expected {
                return Err(WorldTestPortfolioError::TaskProvenanceMismatch {
                    task_id: task.task_id.clone(),
                    field,
                });
            }
        }
        if digest_is_empty(&task.challenge_program_sha256) {
            return Err(WorldTestPortfolioError::EmptyChallengeProgramDigest { task_id: task.task_id.clone() });
        }
        if task.challenge_program_sha256 == context.exploration_program_sha256 {
            return Err(WorldTestPortfolioError::ChallengeProgramMatchesExploration { task_id: task.task_id.clone() });
        }
        if digest_is_empty(&task.metric_profile_sha256) {
            return Err(WorldTestPortfolioError::EmptyMetricProfileDigest { task_id: task.task_id.clone() });
        }
        if family_profiles.get(&task.query_family) != Some(&task.metric_profile_sha256) {
            return Err(WorldTestPortfolioError::FamilyProfileMismatch {
                task_id: task.task_id.clone(),
                family: task.query_family,
            });
        }
        let score = task.disposition.score_ppm();
        if score > WORLDTEST_PORTFOLIO_SCORE_PPM {
            return Err(WorldTestPortfolioError::ScoreOutOfRange {
                task_id: task.task_id.clone(),
                score_ppm: score,
            });
        }

        let summary = family_summaries.get_mut(&task.query_family)
            .ok_or(WorldTestPortfolioError::MissingQueryFamily { family: task.query_family })?;
        increment(&mut summary.task_count)?;
        match task.disposition {
            TaskDisposition::Completed { .. } => increment(&mut summary.completed_count)?,
            TaskDisposition::Failed { .. } => increment(&mut summary.failed_count)?,
            TaskDisposition::TimedOut { .. } => increment(&mut summary.timed_out_count)?,
            TaskDisposition::Invalid { .. } => increment(&mut summary.invalid_count)?,
            TaskDisposition::Abstained { .. } => increment(&mut summary.abstained_count)?,
        }
        summary.score_sum_ppm = summary.score_sum_ppm
            .checked_add(u128::from(score))
            .ok_or(WorldTestPortfolioError::ArithmeticOverflow)?;
        summary.minimum_task_score_ppm = summary.minimum_task_score_ppm.min(score);
        summary.maximum_task_score_ppm = summary.maximum_task_score_ppm.max(score);
        add(&mut summary.test_interaction_count, task.test_interaction_count)?;
        add(&mut summary.test_model_query_count, task.test_model_query_count)?;
        add(&mut summary.test_elapsed_us, task.test_elapsed_us)?;
        add_energy(&mut summary.measured_energy_uj, task.measured_energy_uj, &mut summary.energy_measured_task_count)?;

        add(&mut total_test_interactions, task.test_interaction_count)?;
        add(&mut total_test_model_queries, task.test_model_query_count)?;
        add(&mut total_test_elapsed_us, task.test_elapsed_us)?;
        add_energy(&mut total_test_energy_uj, task.measured_energy_uj, &mut total_energy_measured)?;
    }

    let missing: Vec<String> = expected_ids.difference(&observed_ids).cloned().collect();
    if !missing.is_empty() {
        return Err(WorldTestPortfolioError::MissingTasks { task_ids: missing });
    }

    for family in QueryFamily::ALL {
        let summary = family_summaries.get_mut(&family)
            .ok_or(WorldTestPortfolioError::MissingQueryFamily { family })?;
        if summary.task_count == 0 {
            return Err(WorldTestPortfolioError::MissingQueryFamily { family });
        }
        summary.mean_score_ppm = u32::try_from(
            summary.score_sum_ppm / u128::from(summary.task_count),
        ).map_err(|_| WorldTestPortfolioError::ArithmeticOverflow)?;
        if summary.energy_measured_task_count != summary.task_count {
            summary.measured_energy_uj = None;
        }
    }

    if total_test_interactions > budget.max_total_test_interactions {
        return Err(WorldTestPortfolioError::TestBudgetExceeded { field: "max_total_test_interactions" });
    }
    if total_test_model_queries > budget.max_total_test_model_queries {
        return Err(WorldTestPortfolioError::TestBudgetExceeded { field: "max_total_test_model_queries" });
    }
    if total_test_elapsed_us > budget.max_total_test_elapsed_us {
        return Err(WorldTestPortfolioError::TestBudgetExceeded { field: "max_total_test_elapsed_us" });
    }
    if let Some(maximum) = budget.max_total_test_energy_uj {
        if total_energy_measured != u64::try_from(tasks.len()).unwrap_or(u64::MAX) {
            return Err(WorldTestPortfolioError::MissingRequiredEnergyMeasurement { phase: "test" });
        }
        let measured = total_test_energy_uj
            .ok_or(WorldTestPortfolioError::MissingRequiredEnergyMeasurement { phase: "test" })?;
        if measured > u128::from(maximum) {
            return Err(WorldTestPortfolioError::TestBudgetExceeded { field: "max_total_test_energy_uj" });
        }
    } else if total_energy_measured != u64::try_from(tasks.len()).unwrap_or(u64::MAX) {
        total_test_energy_uj = None;
    }

    let mut family_rows = Vec::with_capacity(WORLDTEST_PORTFOLIO_QUERY_FAMILY_COUNT);
    for family in QueryFamily::ALL {
        family_rows.push(
            family_summaries.remove(&family)
                .ok_or(WorldTestPortfolioError::MissingQueryFamily { family })?
        );
    }
    let min_mean = family_rows.iter().map(|row| row.mean_score_ppm).min()
        .ok_or(WorldTestPortfolioError::MissingQueryFamily { family: QueryFamily::MaskedFramePrediction })?;
    let max_mean = family_rows.iter().map(|row| row.mean_score_ppm).max()
        .ok_or(WorldTestPortfolioError::MissingQueryFamily { family: QueryFamily::MaskedFramePrediction })?;

    let environment_family_count = u64::try_from(environment_family_ids.len())
        .map_err(|_| WorldTestPortfolioError::ArithmeticOverflow)?;
    let environment_cluster_summaries = environment_clusters
        .into_values()
        .collect::<Vec<_>>();
    let independent_environment_cluster_count =
        u64::try_from(environment_cluster_summaries.len())
            .map_err(|_| WorldTestPortfolioError::ArithmeticOverflow)?;

    let task_results = tasks.into_iter().map(|task| WorldTestPortfolioTaskResult {
        task_id: task.task_id,
        run_id: task.run_id,
        query_family: task.query_family,
        environment_family_id: task.environment_family_id,
        environment_cluster_id: task.environment_cluster_id,
        score_ppm: task.disposition.score_ppm(),
        disposition: task.disposition,
        challenge_program_sha256: task.challenge_program_sha256,
        metric_profile_sha256: task.metric_profile_sha256,
        test_interaction_count: task.test_interaction_count,
        test_model_query_count: task.test_model_query_count,
        test_elapsed_us: task.test_elapsed_us,
        measured_energy_uj: task.measured_energy_uj,
    }).collect::<Vec<_>>();

    Ok(WorldTestQueryPortfolioCertificate {
        context: context.clone(),
        environment_family_count,
        independent_environment_cluster_count,
        environment_cluster_summaries,
        task_count: u64::try_from(task_results.len()).map_err(|_| WorldTestPortfolioError::ArithmeticOverflow)?,
        task_results,
        family_summaries: family_rows,
        minimum_family_mean_ppm: min_mean,
        maximum_family_mean_ppm: max_mean,
        family_mean_gap_ppm: max_mean.checked_sub(min_mean).ok_or(WorldTestPortfolioError::ArithmeticOverflow)?,
        all_required_families_present: true,
        exploration_total_charged_interactions: total_exploration,
        total_test_interaction_count: total_test_interactions,
        total_test_model_query_count: total_test_model_queries,
        total_test_elapsed_us,
        total_test_measured_energy_uj: total_test_energy_uj,
        energy_measured_task_count: total_energy_measured,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(value: u8) -> [u8; 32] { [value; 32] }

    fn context() -> WorldTestPortfolioContext {
        WorldTestPortfolioContext {
            portfolio_id: 3,
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
                model_query_count: 12,
                elapsed_us: 500,
                measured_energy_uj: Some(1_000),
            },
            resource_budget: PortfolioResourceBudget {
                max_exploration_actions: 10,
                max_exploration_resets: 2,
                max_exploration_total_interactions: 12,
                max_exploration_model_queries: 16,
                max_exploration_elapsed_us: 700,
                max_exploration_energy_uj: Some(1_500),
                max_total_test_interactions: 30,
                max_total_test_model_queries: 30,
                max_total_test_elapsed_us: 1_000,
                max_total_test_energy_uj: Some(3_000),
            },
            family_metric_profiles: vec![
                FamilyMetricProfile { family: QueryFamily::MaskedFramePrediction, metric_profile_sha256: digest(11) },
                FamilyMetricProfile { family: QueryFamily::ChangeDetection, metric_profile_sha256: digest(12) },
                FamilyMetricProfile { family: QueryFamily::Planning, metric_profile_sha256: digest(13) },
            ],
        }
    }

    fn task(task_id: &str, run_id: u64, family: QueryFamily, challenge: u8, profile: u8, disposition: TaskDisposition) -> WorldTestQueryTask {
        WorldTestQueryTask {
            task_id: task_id.to_owned(),
            run_id,
            query_family: family,
            environment_family_id: "world-generator-v1".to_owned(),
            environment_cluster_id: format!("cluster-{run_id}"),
            challenge_program_sha256: digest(challenge),
            metric_profile_sha256: digest(profile),
            candidate_artifact_sha256: digest(1),
            exploration_trace_sha256: digest(6),
            exploration_checkpoint_sha256: digest(7),
            disposition,
            test_interaction_count: 2,
            test_model_query_count: 3,
            test_elapsed_us: 50,
            measured_energy_uj: Some(100),
        }
    }

    fn input(tasks: Vec<WorldTestQueryTask>) -> WorldTestPortfolioInput {
        WorldTestPortfolioInput {
            context: context(),
            expected_task_ids: vec!["mfp-a".into(), "cd-a".into(), "planning-a".into()],
            tasks,
        }
    }

    fn base_tasks() -> Vec<WorldTestQueryTask> {
        vec![
            task("mfp-a", 1, QueryFamily::MaskedFramePrediction, 21, 11, TaskDisposition::Completed { score_ppm: 900_000 }),
            task("cd-a", 2, QueryFamily::ChangeDetection, 22, 12, TaskDisposition::Completed { score_ppm: 500_000 }),
            task("planning-a", 3, QueryFamily::Planning, 23, 13, TaskDisposition::Completed { score_ppm: 300_000 }),
        ]
    }

    #[test]
    fn reports_three_family_vector_and_worst_family_floor() {
        let result = evaluate_worldtest_query_portfolio(&input(base_tasks())).unwrap();
        assert_eq!(result.family_summaries.len(), 3);
        assert_eq!(result.family_summaries[0].family, QueryFamily::MaskedFramePrediction);
        assert_eq!(result.family_summaries[0].mean_score_ppm, 900_000);
        assert_eq!(result.family_summaries[1].mean_score_ppm, 500_000);
        assert_eq!(result.family_summaries[2].mean_score_ppm, 300_000);
        assert_eq!(result.minimum_family_mean_ppm, 300_000);
        assert_eq!(result.maximum_family_mean_ppm, 900_000);
        assert_eq!(result.family_mean_gap_ppm, 600_000);
        assert!(result.all_required_families_present);
    }

    #[test]
    fn strong_family_cannot_hide_specialization_in_other_families() {
        let result = evaluate_worldtest_query_portfolio(&input(base_tasks())).unwrap();
        assert_eq!(result.minimum_family_mean_ppm, 300_000);
        assert_ne!(result.minimum_family_mean_ppm, 900_000);
        assert!(result.family_mean_gap_ppm >= 600_000);
    }

    #[test]
    fn equal_family_scores_have_zero_robustness_gap() {
        let tasks = vec![
            task("mfp-a", 1, QueryFamily::MaskedFramePrediction, 21, 11, TaskDisposition::Completed { score_ppm: 750_000 }),
            task("cd-a", 2, QueryFamily::ChangeDetection, 22, 12, TaskDisposition::Completed { score_ppm: 750_000 }),
            task("planning-a", 3, QueryFamily::Planning, 23, 13, TaskDisposition::Completed { score_ppm: 750_000 }),
        ];
        let result = evaluate_worldtest_query_portfolio(&input(tasks)).unwrap();
        assert_eq!(result.minimum_family_mean_ppm, 750_000);
        assert_eq!(result.family_mean_gap_ppm, 0);
    }

    #[test]
    fn input_order_does_not_change_canonical_task_results() {
        let original = input(base_tasks());
        let mut shuffled = original.clone();
        shuffled.tasks.reverse();
        shuffled.expected_task_ids.reverse();
        assert_eq!(
            evaluate_worldtest_query_portfolio(&original).unwrap(),
            evaluate_worldtest_query_portfolio(&shuffled).unwrap()
        );
    }

    #[test]
    fn missing_family_is_rejected() {
        let mut candidate = input(base_tasks());
        candidate.tasks.pop();
        candidate.expected_task_ids.pop();
        assert!(matches!(
            evaluate_worldtest_query_portfolio(&candidate),
            Err(WorldTestPortfolioError::MissingQueryFamily { family: QueryFamily::Planning })
        ));
    }

    #[test]
    fn missing_expected_task_is_rejected() {
        let mut candidate = input(base_tasks());
        candidate.tasks.pop();
        assert!(matches!(
            evaluate_worldtest_query_portfolio(&candidate),
            Err(WorldTestPortfolioError::MissingTasks { .. })
        ));
    }

    #[test]
    fn unexpected_task_is_rejected() {
        let mut candidate = input(base_tasks());
        candidate.tasks.push(task("unexpected", 4, QueryFamily::Planning, 24, 13, TaskDisposition::Completed { score_ppm: 1 }));
        assert!(matches!(
            evaluate_worldtest_query_portfolio(&candidate),
            Err(WorldTestPortfolioError::UnexpectedTasks { .. })
        ));
    }

    #[test]
    fn duplicate_expected_task_is_rejected() {
        let mut candidate = input(base_tasks());
        candidate.expected_task_ids.push("mfp-a".to_owned());
        assert!(matches!(
            evaluate_worldtest_query_portfolio(&candidate),
            Err(WorldTestPortfolioError::DuplicateExpectedTaskId { .. })
        ));
    }

    #[test]
    fn duplicate_run_id_is_rejected() {
        let mut tasks = base_tasks();
        tasks[1].run_id = tasks[0].run_id;
        assert!(matches!(
            evaluate_worldtest_query_portfolio(&input(tasks)),
            Err(WorldTestPortfolioError::DuplicateRunId { .. })
        ));
    }

    #[test]
    fn same_exploration_program_as_challenge_is_rejected() {
        let mut tasks = base_tasks();
        tasks[1].challenge_program_sha256 = digest(5);
        assert!(matches!(
            evaluate_worldtest_query_portfolio(&input(tasks)),
            Err(WorldTestPortfolioError::ChallengeProgramMatchesExploration { .. })
        ));
    }

    #[test]
    fn changed_shared_exploration_trace_is_rejected() {
        let mut tasks = base_tasks();
        tasks[1].exploration_trace_sha256 = digest(99);
        assert!(matches!(
            evaluate_worldtest_query_portfolio(&input(tasks)),
            Err(WorldTestPortfolioError::TaskProvenanceMismatch { field: "exploration_trace_sha256", .. })
        ));
    }

    #[test]
    fn changed_shared_checkpoint_is_rejected() {
        let mut tasks = base_tasks();
        tasks[1].exploration_checkpoint_sha256 = digest(99);
        assert!(matches!(
            evaluate_worldtest_query_portfolio(&input(tasks)),
            Err(WorldTestPortfolioError::TaskProvenanceMismatch { field: "exploration_checkpoint_sha256", .. })
        ));
    }

    #[test]
    fn wrong_family_metric_profile_is_rejected() {
        let mut tasks = base_tasks();
        tasks[1].metric_profile_sha256 = digest(77);
        assert!(matches!(
            evaluate_worldtest_query_portfolio(&input(tasks)),
            Err(WorldTestPortfolioError::FamilyProfileMismatch { .. })
        ));
    }

    #[test]
    fn failed_task_penalty_remains_in_family_denominator() {
        let mut tasks = base_tasks();
        tasks[0].disposition = TaskDisposition::Failed { score_ppm: 0 };
        let result = evaluate_worldtest_query_portfolio(&input(tasks)).unwrap();
        assert_eq!(result.family_summaries[0].failed_count, 1);
        assert_eq!(result.family_summaries[0].task_count, 1);
        assert_eq!(result.family_summaries[0].mean_score_ppm, 0);
    }

    #[test]
    fn test_resources_are_aggregated_separately_from_exploration() {
        let result = evaluate_worldtest_query_portfolio(&input(base_tasks())).unwrap();
        assert_eq!(result.context.exploration_resources.action_count, 8);
        assert_eq!(result.exploration_total_charged_interactions, 9);
        assert_eq!(result.total_test_interaction_count, 6);
        assert_eq!(result.total_test_model_query_count, 9);
        assert_eq!(result.total_test_elapsed_us, 150);
        assert_eq!(result.total_test_measured_energy_uj, Some(300));
    }

    #[test]
    fn exceeded_test_budget_is_rejected() {
        let mut candidate = input(base_tasks());
        candidate.context.resource_budget.max_total_test_interactions = 5;
        assert!(matches!(
            evaluate_worldtest_query_portfolio(&candidate),
            Err(WorldTestPortfolioError::TestBudgetExceeded { field: "max_total_test_interactions" })
        ));
    }

    #[test]
    fn exceeded_total_exploration_budget_is_rejected() {
        let mut candidate = input(base_tasks());
        candidate.context.resource_budget.max_exploration_total_interactions = 8;
        assert!(matches!(
            evaluate_worldtest_query_portfolio(&candidate),
            Err(WorldTestPortfolioError::ExplorationBudgetExceeded { field: "max_exploration_total_interactions" })
        ));
    }

    #[test]
    fn missing_required_energy_measurement_is_rejected() {
        let mut candidate = input(base_tasks());
        candidate.tasks[1].measured_energy_uj = None;
        assert!(matches!(
            evaluate_worldtest_query_portfolio(&candidate),
            Err(WorldTestPortfolioError::MissingRequiredEnergyMeasurement { phase: "test" })
        ));
    }

    #[test]
    fn scores_outside_ppm_range_are_rejected() {
        let mut tasks = base_tasks();
        tasks[2].disposition = TaskDisposition::Completed {
            score_ppm: WORLDTEST_PORTFOLIO_SCORE_PPM + 1
        };
        assert!(matches!(
            evaluate_worldtest_query_portfolio(&input(tasks)),
            Err(WorldTestPortfolioError::ScoreOutOfRange { .. })
        ));
    }

    #[test]
    fn changed_candidate_artifact_is_rejected() {
        let mut tasks = base_tasks();
        tasks[1].candidate_artifact_sha256 = digest(99);
        assert!(matches!(
            evaluate_worldtest_query_portfolio(&input(tasks)),
            Err(WorldTestPortfolioError::TaskProvenanceMismatch { field: "candidate_artifact_sha256", .. })
        ));
    }

    #[test]
    fn failed_or_abstained_task_is_never_omitted() {
        let mut candidate = input(base_tasks());
        candidate.expected_task_ids = vec!["mfp-a".into(),"cd-a".into(),"planning-a".into(),"planning-b".into()];
        candidate.tasks.push(task("planning-b",4,QueryFamily::Planning,24,13,TaskDisposition::Abstained{score_ppm:0}));
        let result = evaluate_worldtest_query_portfolio(&candidate).unwrap();
        assert_eq!(result.family_summaries[2].abstained_count,1);
        assert_eq!(result.family_summaries[2].task_count,2);
        assert_eq!(result.family_summaries[2].score_sum_ppm,300_000);
        assert_eq!(result.family_summaries[2].mean_score_ppm,150_000);
    }

    #[test]
    fn repeated_tasks_share_one_independent_environment_cluster() {
        let mut tasks = base_tasks();
        tasks[1].environment_cluster_id = tasks[0].environment_cluster_id.clone();
        let result = evaluate_worldtest_query_portfolio(&input(tasks)).unwrap();

        assert_eq!(result.task_count, 3);
        assert_eq!(result.environment_family_count, 1);
        assert_eq!(result.independent_environment_cluster_count, 2);
        assert_eq!(result.environment_cluster_summaries.len(), 2);
        assert_eq!(
            result.environment_cluster_summaries[0].environment_cluster_id,
            "cluster-1"
        );
        assert_eq!(result.environment_cluster_summaries[0].task_count, 2);
        assert_eq!(
            result.environment_cluster_summaries[0].query_family_task_counts,
            [1, 1, 0]
        );
        assert_eq!(
            result.task_results[0].environment_cluster_id,
            result.task_results[1].environment_cluster_id
        );
    }

    #[test]
    fn empty_environment_family_identity_is_rejected() {
        let mut tasks = base_tasks();
        tasks[1].environment_family_id = "  ".to_owned();
        assert!(matches!(
            evaluate_worldtest_query_portfolio(&input(tasks)),
            Err(WorldTestPortfolioError::EmptyEnvironmentFamilyId { task_id })
                if task_id == "cd-a"
        ));
    }

    #[test]
    fn untrimmed_environment_cluster_identity_is_rejected() {
        let mut tasks = base_tasks();
        tasks[1].environment_cluster_id = " cluster-2".to_owned();
        assert!(matches!(
            evaluate_worldtest_query_portfolio(&input(tasks)),
            Err(WorldTestPortfolioError::EmptyEnvironmentClusterId { task_id })
                if task_id == "cd-a"
        ));
    }

    #[test]
    fn one_environment_cluster_cannot_claim_multiple_environment_families() {
        let mut tasks = base_tasks();
        tasks[1].environment_cluster_id = tasks[0].environment_cluster_id.clone();
        tasks[1].environment_family_id = "different-generator-v1".to_owned();
        assert!(matches!(
            evaluate_worldtest_query_portfolio(&input(tasks)),
            Err(WorldTestPortfolioError::EnvironmentClusterFamilyMismatch {
                environment_cluster_id,
                ..
            }) if environment_cluster_id == "cluster-1"
        ));
    }
}

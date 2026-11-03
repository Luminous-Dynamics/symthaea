#![forbid(unsafe_code)]

//! Scenario-aware query consequence analysis for proposed connectome proofreading corrections.
//!
//! For each pre-registered scientific query and admissible structural scenario, this crate
//! computes the interval-valued change in the query endpoint between a base graph and a
//! hypothetical post-edit graph. It preserves scenarios, reports possible/guaranteed decision
//! flips and normalized materiality relative to a query-specific pre-registered threshold.
//!
//! It deliberately does not assign posterior weights to structural hypotheses, average across
//! scenarios, combine unlike query units, or produce one universal "biological importance" score.
//! It does not validate that a proposed segmentation edit is correct or that supplied graph
//! endpoint intervals are biologically valid; independent adapters must establish those facts.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskRoute {
    FalseSplitCorrection,
    FalseMergeIdentification,
    SplitMaskGeneration,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Interval {
    pub lower: f64,
    pub upper: f64,
}

impl Interval {
    pub fn new(lower: f64, upper: f64) -> Result<Self, QueryImpactError> {
        validate_interval("interval", lower, upper)?;
        Ok(Self { lower, upper })
    }

    /// Set of possible values of (this interval minus the other interval).
    pub fn difference(self, baseline: Self) -> Result<Self, QueryImpactError> {
        let lower = self.lower - baseline.upper;
        let upper = self.upper - baseline.lower;
        validate_interval("difference interval", lower, upper)?;
        Ok(Self { lower, upper })
    }

    /// Largest absolute change that remains possible in this interval.
    pub fn max_absolute(self) -> f64 {
        self.lower.abs().max(self.upper.abs())
    }

    /// Smallest absolute change that remains possible in this interval.
    pub fn min_absolute(self) -> f64 {
        if self.lower <= 0.0 && self.upper >= 0.0 {
            0.0
        } else {
            self.lower.abs().min(self.upper.abs())
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct PairedDecisionWorld {
    pub world_id: String,
    pub baseline_value: f64,
    pub after_edit_value: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PairedEffectBounds {
    /// Tighter interval computed from matched baseline/post-edit latent graph worlds.
    pub effect_interval: Interval,
    /// Digest of the paired-world evaluation artifact and shared-world mapping.
    pub evidence_sha256: String,
    /// Optional exact endpoint pairs evaluated in the same latent graph worlds.
    pub decision_worlds: Option<Vec<PairedDecisionWorld>>,
}

/// Independent closure receipt for a declared finite universe of admissible paired worlds.
/// All digest references and the closure_verified assertion must be checked by the ingestion
/// adapter against the actual immutable artifacts and pinned verifier.
#[derive(Debug, Clone, PartialEq)]
pub struct PairedWorldSetCoverage {
    pub universe_id: String,
    pub candidate_id: String,
    pub operation_id: String,
    pub sample_hash: String,
    pub task_route: TaskRoute,
    pub query_id: String,
    pub scenario_id: String,
    pub query_definition_sha256: String,
    pub metric_id: String,
    pub unit_id: String,
    pub decision_threshold: f64,
    pub materiality_threshold: f64,
    pub source_volume_id: String,
    pub base_graph_snapshot_sha256: String,
    pub hypothetical_result_graph_snapshot_sha256: String,
    pub candidate_patch_sha256: String,
    pub universe_manifest_sha256: String,
    pub admissibility_policy_sha256: String,
    pub enumeration_input_sha256: String,
    pub enumerator_sha256: String,
    pub expected_world_manifest_sha256: String,
    pub closure_proof_sha256: String,
    pub verifier_sha256: String,
    pub verification_receipt_sha256: String,
    pub expected_world_ids: Vec<String>,
    pub closure_verified: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct QueryScenarioEvaluation {
    pub query_id: String,
    pub query_definition_sha256: String,
    pub pre_registered: bool,
    pub metric_id: String,
    pub unit_id: String,
    /// Positive, finite, query-specific smallest materially important absolute effect.
    pub materiality_threshold: f64,
    /// Optional pre-registered decision boundary for this endpoint.
    pub decision_threshold: Option<f64>,
    /// Name of one admissible structural/detector scenario. Scenarios are unweighted.
    pub scenario_id: String,
    pub baseline: Interval,
    pub hypothetical_after_edit: Interval,
    /// Optional paired-world interval. If present, it must be contained by the rectangular
    /// outer bound and have independently verifiable evidence.
    pub paired_effect: Option<PairedEffectBounds>,
    /// Optional independent closure receipt. Without it, paired results are scoped only to the
    /// supplied paired-world set, never to the entire declared structural universe.
    pub paired_world_set_coverage: Option<PairedWorldSetCoverage>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CandidateCorrectionInput {
    pub candidate_id: String,
    pub operation_id: String,
    /// ConnectomeBench2 combined_sample_hash or another stable source sample identity.
    pub sample_hash: String,
    pub task_route: TaskRoute,
    pub source_volume_id: String,
    pub base_graph_snapshot_sha256: String,
    pub hypothetical_result_graph_snapshot_sha256: String,
    pub candidate_patch_sha256: String,
    pub query_evaluations: Vec<QueryScenarioEvaluation>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecisionClass {
    BelowThreshold,
    AtOrAboveThreshold,
    Indeterminate,
    NotConfigured,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecisionFlipStatus {
    /// No before/after value pair in the supplied intervals changes the threshold decision.
    Impossible,
    /// Some admissible values change the decision and others do not.
    Possible,
    /// Every before/after value pair changes the threshold decision.
    Guaranteed,
    /// This query did not pre-register a decision threshold.
    NotConfigured,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EffectMethod {
    RectangularOuterBound,
    PairedSharedWorlds,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecisionFlipScope {
    NotApplicable,
    /// The Cartesian product of marginal before/after intervals, not a paired latent-world set.
    MarginalRectangularProduct,
    /// Only the paired worlds supplied in the input; no completeness claim.
    SuppliedPairedWorldSet,
    /// Exact finite universe named by a separately verified closure receipt.
    DeclaredFiniteUniverse,
    /// Different scenarios in the same query use different scopes.
    Mixed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecisionFlipMethod {
    NotConfigured,
    MarginalIntervalOuterBound,
    PairedSharedWorlds,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ScenarioImpact {
    pub scenario_id: String,
    pub baseline: Interval,
    pub hypothetical_after_edit: Interval,
    /// [after.lower - baseline.upper, after.upper - baseline.lower].
    pub effect_interval: Interval,
    pub effect_method: EffectMethod,
    pub minimum_absolute_effect: f64,
    pub maximum_absolute_effect: f64,
    pub normalized_materiality_lower: f64,
    pub normalized_materiality_upper: f64,
    pub baseline_decision: DecisionClass,
    pub after_edit_decision: DecisionClass,
    pub decision_flip: DecisionFlipStatus,
    pub decision_flip_method: DecisionFlipMethod,
    pub decision_flip_scope: DecisionFlipScope,
    pub paired_decision_world_count: usize,
    pub materiality_exceeded_possible: bool,
    pub materiality_exceeded_guaranteed: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct QueryImpactSummary {
    pub query_id: String,
    pub query_definition_sha256: String,
    pub metric_id: String,
    pub unit_id: String,
    pub materiality_threshold: f64,
    /// All supplied scenarios, retained separately and sorted by scenario_id.
    pub scenarios: Vec<ScenarioImpact>,
    pub worst_case_maximum_absolute_effect: f64,
    pub worst_case_normalized_materiality: f64,
    pub materiality_exceeded_possible: bool,
    pub materiality_exceeded_guaranteed: bool,
    pub aggregate_decision_flip: DecisionFlipStatus,
    pub aggregate_decision_flip_scope: DecisionFlipScope,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CandidateImpactReport {
    pub report_version: &'static str,
    pub candidate_id: String,
    pub operation_id: String,
    pub sample_hash: String,
    pub task_route: TaskRoute,
    pub source_volume_id: String,
    pub base_graph_snapshot_sha256: String,
    pub hypothetical_result_graph_snapshot_sha256: String,
    pub candidate_patch_sha256: String,
    /// Per-query vector. Query units are never collapsed into an arbitrary weighted scalar.
    pub query_impacts: Vec<QueryImpactSummary>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QueryImpactError {
    InvalidIdentifier { kind: &'static str, value: String },
    InvalidSampleHash(String),
    InvalidSha256 { kind: &'static str, value: String },
    EmptyCollection(&'static str),
    QueryNotPreRegistered(String),
    InvalidInterval { kind: &'static str, lower: String, upper: String },
    InvalidMaterialityThreshold { query_id: String },
    InvalidDecisionThreshold { query_id: String },
    DuplicateQueryScenario { query_id: String, scenario_id: String },
    QueryDefinitionMismatch { query_id: String },
    MetricIdentityMismatch { query_id: String },
    UnitMismatch { query_id: String },
    MaterialityThresholdMismatch { query_id: String },
    DecisionThresholdMismatch { query_id: String },
    NonFiniteDerivedValue { query_id: String, scenario_id: String },
    PairedEffectOutsideConservativeBound { query_id: String, scenario_id: String },
    PairedDecisionWorldsEmpty { query_id: String, scenario_id: String },
    PairedDecisionWorldsRequireThreshold { query_id: String, scenario_id: String },
    DuplicatePairedDecisionWorld { query_id: String, scenario_id: String, world_id: String },
    PairedDecisionValueOutsideMarginal { query_id: String, scenario_id: String, world_id: String },
    PairedDecisionDifferenceOutsideEffect { query_id: String, scenario_id: String, world_id: String },
    PairedWorldCoverageRequiresDecisionWorlds { query_id: String, scenario_id: String },
    EmptyExpectedWorldSet { query_id: String, scenario_id: String },
    DuplicateExpectedWorldId { query_id: String, scenario_id: String, world_id: String },
    ExpectedWorldSetMismatch { query_id: String, scenario_id: String },
    WorldSetCoverageNotVerified { query_id: String, scenario_id: String },
    PairedWorldCoverageBindingMismatch {
        query_id: String,
        scenario_id: String,
        field: &'static str,
    },
    NoQueryScenarios,
}

impl fmt::Display for QueryImpactError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidIdentifier { kind, value } => {
                write!(f, "invalid {kind} identifier: {value:?}")
            }
            Self::InvalidSampleHash(value) => {
                write!(f, "sample hash must be 32 lowercase hex characters: {value:?}")
            }
            Self::InvalidSha256 { kind, value } => {
                write!(f, "invalid SHA-256 for {kind}: {value:?}")
            }
            Self::EmptyCollection(name) => write!(f, "{name} must not be empty"),
            Self::QueryNotPreRegistered(id) => {
                write!(f, "query {id} was not pre-registered before candidate evaluation")
            }
            Self::InvalidInterval { kind, lower, upper } => {
                write!(f, "invalid {kind} interval [{lower}, {upper}]")
            }
            Self::InvalidMaterialityThreshold { query_id } => {
                write!(f, "query {query_id} has a non-positive or non-finite materiality threshold")
            }
            Self::InvalidDecisionThreshold { query_id } => {
                write!(f, "query {query_id} has a non-finite decision threshold")
            }
            Self::DuplicateQueryScenario { query_id, scenario_id } => {
                write!(f, "duplicate query/scenario evaluation: {query_id}/{scenario_id}")
            }
            Self::DuplicateScenarioSetEntry { query_id, scenario_id } => {
                write!(f, "duplicate scenario ID in query {query_id}: {scenario_id}")
            }
            Self::QueryDefinitionMismatch { query_id } => {
                write!(f, "query {query_id} has conflicting definition digests")
            }
            Self::MetricIdentityMismatch { query_id } => {
                write!(f, "query {query_id} has conflicting metric IDs")
            }
            Self::UnitMismatch { query_id } => {
                write!(f, "query {query_id} has inconsistent units across scenarios")
            }
            Self::MaterialityThresholdMismatch { query_id } => {
                write!(f, "query {query_id} has inconsistent materiality thresholds")
            }
            Self::DecisionThresholdMismatch { query_id } => {
                write!(f, "query {query_id} has inconsistent decision thresholds")
            }
            Self::NonFiniteDerivedValue { query_id, scenario_id } => {
                write!(f, "query {query_id} scenario {scenario_id} overflowed derived arithmetic")
            }
            Self::PairedEffectOutsideConservativeBound { query_id, scenario_id } => {
                write!(f, "query {query_id} scenario {scenario_id} paired effect lies outside its conservative interval bound")
            }
            Self::PairedDecisionWorldsEmpty { query_id, scenario_id } => {
                write!(f, "query {query_id} scenario {scenario_id} supplied an empty paired decision world set")
            }
            Self::PairedDecisionWorldsRequireThreshold { query_id, scenario_id } => {
                write!(f, "query {query_id} scenario {scenario_id} supplied paired decision worlds without a decision threshold")
            }
            Self::DuplicatePairedDecisionWorld { query_id, scenario_id, world_id } => {
                write!(f, "query {query_id} scenario {scenario_id} repeats paired decision world {world_id}")
            }
            Self::PairedDecisionValueOutsideMarginal { query_id, scenario_id, world_id } => {
                write!(f, "query {query_id} scenario {scenario_id} paired world {world_id} lies outside a marginal interval")
            }
            Self::PairedDecisionDifferenceOutsideEffect { query_id, scenario_id, world_id } => {
                write!(f, "query {query_id} scenario {scenario_id} paired world {world_id} lies outside the paired effect interval")
            }
            Self::PairedWorldCoverageRequiresDecisionWorlds { query_id, scenario_id } => {
                write!(f, "query {query_id} scenario {scenario_id} supplied a world-set closure receipt without paired decision worlds")
            }
            Self::EmptyExpectedWorldSet { query_id, scenario_id } => {
                write!(f, "query {query_id} scenario {scenario_id} declared an empty finite world universe")
            }
            Self::DuplicateExpectedWorldId { query_id, scenario_id, world_id } => {
                write!(f, "query {query_id} scenario {scenario_id} repeats expected world ID {world_id}")
            }
            Self::ExpectedWorldSetMismatch { query_id, scenario_id } => {
                write!(f, "query {query_id} scenario {scenario_id} evaluated world IDs do not exactly match the declared universe")
            }
            Self::WorldSetCoverageNotVerified { query_id, scenario_id } => {
                write!(f, "query {query_id} scenario {scenario_id} world-set closure receipt is not independently verified")
            }
            Self::PairedWorldCoverageBindingMismatch { query_id, scenario_id, field } => {
                write!(f, "query {query_id} scenario {scenario_id} world-set closure receipt is not bound to {field}")
            }
            Self::NoQueryScenarios => write!(f, "candidate has no query scenario evaluations"),
        }
    }
}

impl std::error::Error for QueryImpactError {}

#[derive(Debug, Clone, PartialEq)]
struct QuerySignature {
    definition_sha256: String,
    metric_id: String,
    unit_id: String,
    materiality_threshold: f64,
    decision_threshold: Option<f64>,
}

#[derive(Debug, Clone, PartialEq)]
struct QueryAccumulator {
    signature: QuerySignature,
    scenarios: Vec<ScenarioImpact>,
}

/// Analyze the effect of one candidate edit on all supplied pre-registered queries.
///
/// The result preserves each query's unit and each structural scenario. The only cross-scenario
/// reductions are conservative extrema (max possible effect, all-scenario guarantee); no
/// probability or weight is assigned to a scenario.
pub fn analyze_candidate(
    input: &CandidateCorrectionInput,
) -> Result<CandidateImpactReport, QueryImpactError> {
    validate_input_identity(input)?;
    if input.query_evaluations.is_empty() {
        return Err(QueryImpactError::NoQueryScenarios);
    }

    let mut queries: BTreeMap<String, QueryAccumulator> = BTreeMap::new();
    let mut query_scenario_pairs = BTreeSet::new();

    for evaluation in &input.query_evaluations {
        validate_evaluation(evaluation)?;
        if let Some(coverage) = &evaluation.paired_world_set_coverage {
            if coverage.candidate_id != input.candidate_id {
                return Err(QueryImpactError::PairedWorldCoverageBindingMismatch {
                    query_id: evaluation.query_id.clone(),
                    scenario_id: evaluation.scenario_id.clone(),
                    field: "candidate ID",
                });
            }
            if coverage.operation_id != input.operation_id {
                return Err(QueryImpactError::PairedWorldCoverageBindingMismatch {
                    query_id: evaluation.query_id.clone(),
                    scenario_id: evaluation.scenario_id.clone(),
                    field: "operation ID",
                });
            }
            if coverage.sample_hash != input.sample_hash {
                return Err(QueryImpactError::PairedWorldCoverageBindingMismatch {
                    query_id: evaluation.query_id.clone(),
                    scenario_id: evaluation.scenario_id.clone(),
                    field: "sample hash",
                });
            }
            if coverage.task_route != input.task_route {
                return Err(QueryImpactError::PairedWorldCoverageBindingMismatch {
                    query_id: evaluation.query_id.clone(),
                    scenario_id: evaluation.scenario_id.clone(),
                    field: "task route",
                });
            }
            for (field, covered, actual) in [
                ("source volume", coverage.source_volume_id.as_str(), input.source_volume_id.as_str()),
                ("base graph snapshot", coverage.base_graph_snapshot_sha256.as_str(), input.base_graph_snapshot_sha256.as_str()),
                ("hypothetical graph snapshot", coverage.hypothetical_result_graph_snapshot_sha256.as_str(), input.hypothetical_result_graph_snapshot_sha256.as_str()),
                ("candidate patch", coverage.candidate_patch_sha256.as_str(), input.candidate_patch_sha256.as_str()),
            ] {
                if covered != actual {
                    return Err(QueryImpactError::PairedWorldCoverageBindingMismatch {
                        query_id: evaluation.query_id.clone(),
                        scenario_id: evaluation.scenario_id.clone(),
                        field,
                    });
                }
            }
        }
        if !query_scenario_pairs.insert((
            evaluation.query_id.clone(),
            evaluation.scenario_id.clone(),
        )) {
            return Err(QueryImpactError::DuplicateQueryScenario {
                query_id: evaluation.query_id.clone(),
                scenario_id: evaluation.scenario_id.clone(),
            });
        }

        let signature = QuerySignature {
            definition_sha256: evaluation.query_definition_sha256.clone(),
            metric_id: evaluation.metric_id.clone(),
            unit_id: evaluation.unit_id.clone(),
            materiality_threshold: evaluation.materiality_threshold,
            decision_threshold: evaluation.decision_threshold,
        };

        let accumulator = queries
            .entry(evaluation.query_id.clone())
            .or_insert_with(|| QueryAccumulator {
                signature: signature.clone(),
                scenarios: Vec::new(),
            });

        validate_signature_consistency(&evaluation.query_id, &accumulator.signature, &signature)?;

        accumulator
            .scenarios
            .push(analyze_scenario(evaluation)?);
    }

    let mut query_impacts = Vec::with_capacity(queries.len());
    for (query_id, mut accumulator) in queries {
        accumulator
            .scenarios
            .sort_by(|left, right| left.scenario_id.cmp(&right.scenario_id));

        let worst_case_maximum_absolute_effect = accumulator
            .scenarios
            .iter()
            .map(|scenario| scenario.maximum_absolute_effect)
            .fold(0.0_f64, f64::max);
        let worst_case_normalized_materiality = accumulator
            .scenarios
            .iter()
            .map(|scenario| scenario.normalized_materiality_upper)
            .fold(0.0_f64, f64::max);

        let materiality_exceeded_possible = accumulator
            .scenarios
            .iter()
            .any(|scenario| scenario.materiality_exceeded_possible);
        let materiality_exceeded_guaranteed = accumulator
            .scenarios
            .iter()
            .all(|scenario| scenario.materiality_exceeded_guaranteed);

        let aggregate_decision_flip = aggregate_flip_status(
            accumulator
                .scenarios
                .iter()
                .map(|scenario| scenario.decision_flip),
            accumulator.signature.decision_threshold.is_some(),
        );

        if !worst_case_maximum_absolute_effect.is_finite()
            || !worst_case_normalized_materiality.is_finite()
        {
            let scenario_id = accumulator
                .scenarios
                .first()
                .map(|scenario| scenario.scenario_id.clone())
                .unwrap_or_default();
            return Err(QueryImpactError::NonFiniteDerivedValue {
                query_id,
                scenario_id,
            });
        }

        let aggregate_decision_flip_scope = aggregate_flip_scope(
            accumulator.scenarios.iter().map(|scenario| scenario.decision_flip_scope),
        );

        query_impacts.push(QueryImpactSummary {
            query_id,
            query_definition_sha256: accumulator.signature.definition_sha256,
            metric_id: accumulator.signature.metric_id,
            unit_id: accumulator.signature.unit_id,
            materiality_threshold: accumulator.signature.materiality_threshold,
            scenarios: accumulator.scenarios,
            worst_case_maximum_absolute_effect,
            worst_case_normalized_materiality,
            materiality_exceeded_possible,
            materiality_exceeded_guaranteed,
            aggregate_decision_flip,
            aggregate_decision_flip_scope,
        });
    }

    // BTreeMap iteration makes the query vector canonical by query_id.
    Ok(CandidateImpactReport {
        report_version: "0.1",
        candidate_id: input.candidate_id.clone(),
        operation_id: input.operation_id.clone(),
        sample_hash: input.sample_hash.clone(),
        task_route: input.task_route,
        source_volume_id: input.source_volume_id.clone(),
        base_graph_snapshot_sha256: input.base_graph_snapshot_sha256.clone(),
        hypothetical_result_graph_snapshot_sha256: input
            .hypothetical_result_graph_snapshot_sha256
            .clone(),
        candidate_patch_sha256: input.candidate_patch_sha256.clone(),
        query_impacts,
    })
}

fn analyze_scenario(
    evaluation: &QueryScenarioEvaluation,
) -> Result<ScenarioImpact, QueryImpactError> {
    let rectangular_outer_bound = evaluation
        .hypothetical_after_edit
        .difference(evaluation.baseline)?;
    let (effect_interval, effect_method) = match &evaluation.paired_effect {
        None => (rectangular_outer_bound, EffectMethod::RectangularOuterBound),
        Some(paired) => {
            if paired.effect_interval.lower < rectangular_outer_bound.lower - 1e-12
                || paired.effect_interval.upper > rectangular_outer_bound.upper + 1e-12
            {
                return Err(QueryImpactError::PairedEffectOutsideConservativeBound {
                    query_id: evaluation.query_id.clone(),
                    scenario_id: evaluation.scenario_id.clone(),
                });
            }
            (paired.effect_interval, EffectMethod::PairedSharedWorlds)
        }
    };
    let minimum_absolute_effect = effect_interval.min_absolute();
    let maximum_absolute_effect = effect_interval.max_absolute();
    let normalized_materiality_lower =
        minimum_absolute_effect / evaluation.materiality_threshold;
    let normalized_materiality_upper =
        maximum_absolute_effect / evaluation.materiality_threshold;

    if !minimum_absolute_effect.is_finite()
        || !maximum_absolute_effect.is_finite()
        || !normalized_materiality_lower.is_finite()
        || !normalized_materiality_upper.is_finite()
    {
        return Err(QueryImpactError::NonFiniteDerivedValue {
            query_id: evaluation.query_id.clone(),
            scenario_id: evaluation.scenario_id.clone(),
        });
    }

    let (
        baseline_decision,
        after_edit_decision,
        decision_flip,
        decision_flip_method,
        paired_decision_world_count,
        decision_flip_scope,
    ) = match evaluation.decision_threshold {
        None => {
            if evaluation
                .paired_effect
                .as_ref()
                .and_then(|paired| paired.decision_worlds.as_ref())
                .is_some()
            {
                return Err(QueryImpactError::PairedDecisionWorldsRequireThreshold {
                    query_id: evaluation.query_id.clone(),
                    scenario_id: evaluation.scenario_id.clone(),
                });
            }
            (
                DecisionClass::NotConfigured,
                DecisionClass::NotConfigured,
                DecisionFlipStatus::NotConfigured,
                DecisionFlipMethod::NotConfigured,
                0,
                DecisionFlipScope::NotApplicable,
            )
        }
        Some(threshold) => {
            let baseline = classify_interval(evaluation.baseline, threshold);
            let after = classify_interval(evaluation.hypothetical_after_edit, threshold);
            let paired_worlds = evaluation
                .paired_effect
                .as_ref()
                .and_then(|paired| paired.decision_worlds.as_ref());
            if let Some(worlds) = paired_worlds {
                let flip_count = worlds
                    .iter()
                    .filter(|world| {
                        (world.baseline_value >= threshold)
                            != (world.after_edit_value >= threshold)
                    })
                    .count();
                let flip = if flip_count == worlds.len() {
                    DecisionFlipStatus::Guaranteed
                } else if flip_count == 0 {
                    DecisionFlipStatus::Impossible
                } else {
                    DecisionFlipStatus::Possible
                };
                let scope = if evaluation.paired_world_set_coverage.is_some() {
                    DecisionFlipScope::DeclaredFiniteUniverse
                } else {
                    DecisionFlipScope::SuppliedPairedWorldSet
                };
                (
                    baseline,
                    after,
                    flip,
                    DecisionFlipMethod::PairedSharedWorlds,
                    worlds.len(),
                    scope,
                )
            } else {
                (
                    baseline,
                    after,
                    classify_flip(
                        evaluation.baseline,
                        evaluation.hypothetical_after_edit,
                        threshold,
                    ),
                    DecisionFlipMethod::MarginalIntervalOuterBound,
                    0,
                    DecisionFlipScope::MarginalRectangularProduct,
                )
            }
        }
    };

    Ok(ScenarioImpact {
        scenario_id: evaluation.scenario_id.clone(),
        baseline: evaluation.baseline,
        hypothetical_after_edit: evaluation.hypothetical_after_edit,
        effect_interval,
        effect_method,
        minimum_absolute_effect,
        maximum_absolute_effect,
        normalized_materiality_lower,
        normalized_materiality_upper,
        baseline_decision,
        after_edit_decision,
        decision_flip,
        materiality_exceeded_possible: normalized_materiality_upper >= 1.0,
        materiality_exceeded_guaranteed: normalized_materiality_lower >= 1.0,
    })
}

fn classify_interval(interval: Interval, threshold: f64) -> DecisionClass {
    if interval.upper < threshold {
        DecisionClass::BelowThreshold
    } else if interval.lower >= threshold {
        DecisionClass::AtOrAboveThreshold
    } else {
        DecisionClass::Indeterminate
    }
}

/// Determine whether a threshold decision can/certainly changes for interval-valued values.
fn classify_flip(
    before: Interval,
    after: Interval,
    threshold: f64,
) -> DecisionFlipStatus {
    let before_can_be_below = before.lower < threshold;
    let before_can_be_above = before.upper >= threshold;
    let after_can_be_below = after.lower < threshold;
    let after_can_be_above = after.upper >= threshold;

    let can_remain_below =
        before_can_be_below && after_can_be_below;
    let can_remain_above =
        before_can_be_above && after_can_be_above;
    let can_flip_up =
        before_can_be_below && after_can_be_above;
    let can_flip_down =
        before_can_be_above && after_can_be_below;

    let can_flip = can_flip_up || can_flip_down;
    let can_stay_same = can_remain_below || can_remain_above;

    if can_flip && !can_stay_same {
        DecisionFlipStatus::Guaranteed
    } else if can_flip {
        DecisionFlipStatus::Possible
    } else {
        DecisionFlipStatus::Impossible
    }
}

fn aggregate_flip_scope(
    scopes: impl Iterator<Item = DecisionFlipScope>,
) -> DecisionFlipScope {
    let mut iter = scopes;
    let Some(first) = iter.next() else {
        return DecisionFlipScope::NotApplicable;
    };
    if iter.all(|scope| scope == first) {
        first
    } else {
        DecisionFlipScope::Mixed
    }
}

fn aggregate_flip_status(
    statuses: impl Iterator<Item = DecisionFlipStatus>,
    threshold_configured: bool,
) -> DecisionFlipStatus {
    if !threshold_configured {
        return DecisionFlipStatus::NotConfigured;
    }
    let statuses: Vec<_> = statuses.collect();
    if statuses.is_empty() {
        return DecisionFlipStatus::NotConfigured;
    }
    if statuses
        .iter()
        .all(|status| *status == DecisionFlipStatus::Guaranteed)
    {
        DecisionFlipStatus::Guaranteed
    } else if statuses
        .iter()
        .all(|status| *status == DecisionFlipStatus::Impossible)
    {
        DecisionFlipStatus::Impossible
    } else {
        DecisionFlipStatus::Possible
    }
}

fn validate_input_identity(input: &CandidateCorrectionInput) -> Result<(), QueryImpactError> {
    validate_id("candidate", &input.candidate_id)?;
    validate_id("operation", &input.operation_id)?;
    validate_sample_hash(&input.sample_hash)?;
    validate_id("source volume", &input.source_volume_id)?;
    validate_sha("base graph snapshot", &input.base_graph_snapshot_sha256)?;
    validate_sha(
        "hypothetical result graph snapshot",
        &input.hypothetical_result_graph_snapshot_sha256,
    )?;
    validate_sha("candidate patch", &input.candidate_patch_sha256)?;
    if input.base_graph_snapshot_sha256 == input.hypothetical_result_graph_snapshot_sha256 {
        return Err(QueryImpactError::InvalidIdentifier {
            kind: "hypothetical result graph snapshot (must differ from base)",
            value: input.hypothetical_result_graph_snapshot_sha256.clone(),
        });
    }
    Ok(())
}

fn validate_evaluation(
    evaluation: &QueryScenarioEvaluation,
) -> Result<(), QueryImpactError> {
    validate_id("query", &evaluation.query_id)?;
    validate_sha("query definition", &evaluation.query_definition_sha256)?;
    validate_id("metric", &evaluation.metric_id)?;
    validate_id("metric unit", &evaluation.unit_id)?;
    validate_id("scenario", &evaluation.scenario_id)?;

    if !evaluation.pre_registered {
        return Err(QueryImpactError::QueryNotPreRegistered(
            evaluation.query_id.clone(),
        ));
    }
    if !evaluation.materiality_threshold.is_finite()
        || evaluation.materiality_threshold <= 0.0
    {
        return Err(QueryImpactError::InvalidMaterialityThreshold {
            query_id: evaluation.query_id.clone(),
        });
    }
    if evaluation
        .decision_threshold
        .is_some_and(|threshold| !threshold.is_finite())
    {
        return Err(QueryImpactError::InvalidDecisionThreshold {
            query_id: evaluation.query_id.clone(),
        });
    }
    validate_interval(
        "baseline",
        evaluation.baseline.lower,
        evaluation.baseline.upper,
    )?;
    validate_interval(
        "hypothetical post-edit",
        evaluation.hypothetical_after_edit.lower,
        evaluation.hypothetical_after_edit.upper,
    )?;
    if let Some(coverage) = &evaluation.paired_world_set_coverage {
        validate_id("coverage candidate", &coverage.candidate_id)?;
        validate_id("coverage operation", &coverage.operation_id)?;
        validate_sample_hash(&coverage.sample_hash)?;
        validate_id("coverage query", &coverage.query_id)?;
        validate_id("coverage scenario", &coverage.scenario_id)?;
        validate_sha("coverage query definition", &coverage.query_definition_sha256)?;
        validate_id("coverage metric", &coverage.metric_id)?;
        validate_id("coverage metric unit", &coverage.unit_id)?;
        validate_id("coverage source volume", &coverage.source_volume_id)?;
        for (kind, digest) in [
            ("coverage base graph snapshot", &coverage.base_graph_snapshot_sha256),
            ("coverage hypothetical graph snapshot", &coverage.hypothetical_result_graph_snapshot_sha256),
            ("coverage candidate patch", &coverage.candidate_patch_sha256),
        ] {
            validate_sha(kind, digest)?;
        }
        if coverage.query_id != evaluation.query_id {
            return Err(QueryImpactError::PairedWorldCoverageBindingMismatch {
                query_id: evaluation.query_id.clone(),
                scenario_id: evaluation.scenario_id.clone(),
                field: "query ID",
            });
        }
        if coverage.scenario_id != evaluation.scenario_id {
            return Err(QueryImpactError::PairedWorldCoverageBindingMismatch {
                query_id: evaluation.query_id.clone(),
                scenario_id: evaluation.scenario_id.clone(),
                field: "scenario ID",
            });
        }
        if coverage.query_definition_sha256 != evaluation.query_definition_sha256 {
            return Err(QueryImpactError::PairedWorldCoverageBindingMismatch {
                query_id: evaluation.query_id.clone(),
                scenario_id: evaluation.scenario_id.clone(),
                field: "query definition",
            });
        }
        if coverage.metric_id != evaluation.metric_id {
            return Err(QueryImpactError::PairedWorldCoverageBindingMismatch {
                query_id: evaluation.query_id.clone(),
                scenario_id: evaluation.scenario_id.clone(),
                field: "metric",
            });
        }
        if coverage.unit_id != evaluation.unit_id {
            return Err(QueryImpactError::PairedWorldCoverageBindingMismatch {
                query_id: evaluation.query_id.clone(),
                scenario_id: evaluation.scenario_id.clone(),
                field: "metric unit",
            });
        }
        if !coverage.decision_threshold.is_finite()
            || evaluation.decision_threshold != Some(coverage.decision_threshold)
        {
            return Err(QueryImpactError::PairedWorldCoverageBindingMismatch {
                query_id: evaluation.query_id.clone(),
                scenario_id: evaluation.scenario_id.clone(),
                field: "decision threshold",
            });
        }
        if !coverage.materiality_threshold.is_finite()
            || coverage.materiality_threshold.to_bits() != evaluation.materiality_threshold.to_bits()
        {
            return Err(QueryImpactError::PairedWorldCoverageBindingMismatch {
                query_id: evaluation.query_id.clone(),
                scenario_id: evaluation.scenario_id.clone(),
                field: "materiality threshold",
            });
        }
        if !coverage.closure_verified {
            return Err(QueryImpactError::WorldSetCoverageNotVerified {
                query_id: evaluation.query_id.clone(),
                scenario_id: evaluation.scenario_id.clone(),
            });
        }
        if evaluation
            .paired_effect
            .as_ref()
            .and_then(|paired| paired.decision_worlds.as_ref())
            .is_none()
        {
            return Err(QueryImpactError::PairedWorldCoverageRequiresDecisionWorlds {
                query_id: evaluation.query_id.clone(),
                scenario_id: evaluation.scenario_id.clone(),
            });
        }
        validate_id("paired world universe", &coverage.universe_id)?;
        for (kind, digest) in [
            ("world universe manifest", &coverage.universe_manifest_sha256),
            ("world admissibility policy", &coverage.admissibility_policy_sha256),
            ("world enumeration input", &coverage.enumeration_input_sha256),
            ("world enumerator", &coverage.enumerator_sha256),
            ("expected world manifest", &coverage.expected_world_manifest_sha256),
            ("world closure proof", &coverage.closure_proof_sha256),
            ("world coverage verifier", &coverage.verifier_sha256),
            ("world coverage verification receipt", &coverage.verification_receipt_sha256),
        ] {
            validate_sha(kind, digest)?;
        }
        if coverage.expected_world_ids.is_empty() {
            return Err(QueryImpactError::EmptyExpectedWorldSet {
                query_id: evaluation.query_id.clone(),
                scenario_id: evaluation.scenario_id.clone(),
            });
        }
        let mut expected_ids = BTreeSet::new();
        for world_id in &coverage.expected_world_ids {
            validate_id("expected paired world", world_id)?;
            if !expected_ids.insert(world_id.clone()) {
                return Err(QueryImpactError::DuplicateExpectedWorldId {
                    query_id: evaluation.query_id.clone(),
                    scenario_id: evaluation.scenario_id.clone(),
                    world_id: world_id.clone(),
                });
            }
        }
        let Some(actual_worlds) = evaluation
            .paired_effect
            .as_ref()
            .and_then(|paired| paired.decision_worlds.as_ref())
        else {
            return Err(QueryImpactError::PairedWorldCoverageRequiresDecisionWorlds {
                query_id: evaluation.query_id.clone(),
                scenario_id: evaluation.scenario_id.clone(),
            });
        };
        let actual_ids: BTreeSet<String> = actual_worlds
            .iter()
            .map(|world| world.world_id.clone())
            .collect();
        if actual_ids != expected_ids {
            return Err(QueryImpactError::ExpectedWorldSetMismatch {
                query_id: evaluation.query_id.clone(),
                scenario_id: evaluation.scenario_id.clone(),
            });
        }
    }
    if let Some(paired) = &evaluation.paired_effect {
        validate_sha("paired effect evidence", &paired.evidence_sha256)?;
        validate_interval(
            "paired effect",
            paired.effect_interval.lower,
            paired.effect_interval.upper,
        )?;
        let outer = evaluation.hypothetical_after_edit.difference(evaluation.baseline)?;
        if paired.effect_interval.lower < outer.lower - 1e-12
            || paired.effect_interval.upper > outer.upper + 1e-12
        {
            return Err(QueryImpactError::PairedEffectOutsideConservativeBound {
                query_id: evaluation.query_id.clone(),
                scenario_id: evaluation.scenario_id.clone(),
            });
        }
        if let Some(worlds) = &paired.decision_worlds {
            if worlds.is_empty() {
                return Err(QueryImpactError::PairedDecisionWorldsEmpty {
                    query_id: evaluation.query_id.clone(),
                    scenario_id: evaluation.scenario_id.clone(),
                });
            }
            if evaluation.decision_threshold.is_none() {
                return Err(QueryImpactError::PairedDecisionWorldsRequireThreshold {
                    query_id: evaluation.query_id.clone(),
                    scenario_id: evaluation.scenario_id.clone(),
                });
            }
            let mut world_ids = BTreeSet::new();
            for world in worlds {
                validate_id("paired decision world", &world.world_id)?;
                if !world_ids.insert(world.world_id.clone()) {
                    return Err(QueryImpactError::DuplicatePairedDecisionWorld {
                        query_id: evaluation.query_id.clone(),
                        scenario_id: evaluation.scenario_id.clone(),
                        world_id: world.world_id.clone(),
                    });
                }
                if !world.baseline_value.is_finite()
                    || !world.after_edit_value.is_finite()
                    || world.baseline_value < evaluation.baseline.lower - 1e-12
                    || world.baseline_value > evaluation.baseline.upper + 1e-12
                    || world.after_edit_value < evaluation.hypothetical_after_edit.lower - 1e-12
                    || world.after_edit_value > evaluation.hypothetical_after_edit.upper + 1e-12
                {
                    return Err(QueryImpactError::PairedDecisionValueOutsideMarginal {
                        query_id: evaluation.query_id.clone(),
                        scenario_id: evaluation.scenario_id.clone(),
                        world_id: world.world_id.clone(),
                    });
                }
                let effect = world.after_edit_value - world.baseline_value;
                if !effect.is_finite()
                    || effect < paired.effect_interval.lower - 1e-12
                    || effect > paired.effect_interval.upper + 1e-12
                {
                    return Err(QueryImpactError::PairedDecisionDifferenceOutsideEffect {
                        query_id: evaluation.query_id.clone(),
                        scenario_id: evaluation.scenario_id.clone(),
                        world_id: world.world_id.clone(),
                    });
                }
            }
        }
    }
    Ok(())
}

fn validate_signature_consistency(
    query_id: &str,
    first: &QuerySignature,
    next: &QuerySignature,
) -> Result<(), QueryImpactError> {
    if first.definition_sha256 != next.definition_sha256 {
        return Err(QueryImpactError::QueryDefinitionMismatch {
            query_id: query_id.to_owned(),
        });
    }
    if first.metric_id != next.metric_id {
        return Err(QueryImpactError::MetricIdentityMismatch {
            query_id: query_id.to_owned(),
        });
    }
    if first.unit_id != next.unit_id {
        return Err(QueryImpactError::UnitMismatch {
            query_id: query_id.to_owned(),
        });
    }
    if first.materiality_threshold.to_bits() != next.materiality_threshold.to_bits() {
        return Err(QueryImpactError::MaterialityThresholdMismatch {
            query_id: query_id.to_owned(),
        });
    }
    if first.decision_threshold.map(f64::to_bits) != next.decision_threshold.map(f64::to_bits) {
        return Err(QueryImpactError::DecisionThresholdMismatch {
            query_id: query_id.to_owned(),
        });
    }
    Ok(())
}

fn validate_interval(
    kind: &'static str,
    lower: f64,
    upper: f64,
) -> Result<(), QueryImpactError> {
    if !lower.is_finite() || !upper.is_finite() || lower > upper {
        return Err(QueryImpactError::InvalidInterval {
            kind,
            lower: lower.to_string(),
            upper: upper.to_string(),
        });
    }
    Ok(())
}

fn validate_sample_hash(value: &str) -> Result<(), QueryImpactError> {
    if value.len() != 32
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(QueryImpactError::InvalidSampleHash(value.to_owned()));
    }
    Ok(())
}

fn validate_sha(kind: &'static str, value: &str) -> Result<(), QueryImpactError> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(QueryImpactError::InvalidSha256 {
            kind,
            value: value.to_owned(),
        });
    }
    Ok(())
}

fn validate_id(kind: &'static str, value: &str) -> Result<(), QueryImpactError> {
    if value.trim().is_empty() || value.trim() != value {
        return Err(QueryImpactError::InvalidIdentifier {
            kind,
            value: value.to_owned(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const SHA_A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const SHA_B: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const SHA_C: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
    const SAMPLE: &str = "0123456789abcdef0123456789abcdef";

    fn interval(lower: f64, upper: f64) -> Interval {
        Interval { lower, upper }
    }

    fn evaluation(
        query: &str,
        scenario: &str,
        before: Interval,
        after: Interval,
        decision_threshold: Option<f64>,
        materiality_threshold: f64,
    ) -> QueryScenarioEvaluation {
        QueryScenarioEvaluation {
            query_id: query.to_owned(),
            query_definition_sha256: SHA_A.to_owned(),
            pre_registered: true,
            metric_id: format!("{query}-metric"),
            unit_id: if query == "global-count" {
                "synapses".to_owned()
            } else {
                "normalized-strength".to_owned()
            },
            materiality_threshold,
            decision_threshold,
            scenario_id: scenario.to_owned(),
            baseline: before,
            hypothetical_after_edit: after,
            paired_effect: None,
            paired_world_set_coverage: None,
        }
    }

    fn world_set_coverage(expected_world_ids: &[&str]) -> PairedWorldSetCoverage {
        PairedWorldSetCoverage {
            universe_id: "finite-universe-v1".to_owned(),
            candidate_id: "candidate-1".to_owned(),
            operation_id: "operation-1".to_owned(),
            sample_hash: SAMPLE.to_owned(),
            task_route: TaskRoute::FalseMergeIdentification,
            query_id: "pathway".to_owned(),
            scenario_id: "h1".to_owned(),
            query_definition_sha256: SHA_A.to_owned(),
            metric_id: "pathway-metric".to_owned(),
            unit_id: "normalized-strength".to_owned(),
            decision_threshold: 0.5,
            materiality_threshold: 0.1,
            source_volume_id: "volume-1".to_owned(),
            base_graph_snapshot_sha256: SHA_B.to_owned(),
            hypothetical_result_graph_snapshot_sha256: SHA_C.to_owned(),
            candidate_patch_sha256: SHA_A.to_owned(),
            universe_manifest_sha256: SHA_A.to_owned(),
            admissibility_policy_sha256: SHA_B.to_owned(),
            enumeration_input_sha256: SHA_C.to_owned(),
            enumerator_sha256: SHA_A.to_owned(),
            expected_world_manifest_sha256: SHA_B.to_owned(),
            closure_proof_sha256: SHA_C.to_owned(),
            verifier_sha256: SHA_A.to_owned(),
            verification_receipt_sha256: SHA_B.to_owned(),
            expected_world_ids: expected_world_ids.iter().map(|id| (*id).to_owned()).collect(),
            closure_verified: true,
        }
    }

    fn candidate(evaluations: Vec<QueryScenarioEvaluation>) -> CandidateCorrectionInput {
        CandidateCorrectionInput {
            candidate_id: "candidate-1".to_owned(),
            operation_id: "operation-1".to_owned(),
            sample_hash: SAMPLE.to_owned(),
            task_route: TaskRoute::FalseMergeIdentification,
            source_volume_id: "volume-1".to_owned(),
            base_graph_snapshot_sha256: SHA_B.to_owned(),
            hypothetical_result_graph_snapshot_sha256: SHA_C.to_owned(),
            candidate_patch_sha256: SHA_A.to_owned(),
            query_evaluations: evaluations,
        }
    }

    #[test]
    fn interval_difference_is_minkowski_difference() {
        let effect = interval(12.0, 14.0).difference(interval(9.0, 10.0)).unwrap();
        assert_eq!(effect, interval(2.0, 5.0));
    }

    #[test]
    fn effect_interval_preserves_uncertainty_overlap_and_zero() {
        let report = analyze_candidate(&candidate(vec![evaluation(
            "pathway",
            "h1",
            interval(0.40, 0.60),
            interval(0.50, 0.70),
            Some(0.55),
            0.10,
        )]))
        .unwrap();
        let impact = &report.query_impacts[0].scenarios[0];
        assert_eq!(impact.effect_interval, interval(-0.10, 0.30));
        assert_eq!(impact.minimum_absolute_effect, 0.0);
        assert!((impact.maximum_absolute_effect - 0.30).abs() < 1e-12);
    }

    #[test]
    fn threshold_flip_is_impossible_when_both_intervals_are_below() {
        assert_eq!(
            classify_flip(interval(0.1, 0.2), interval(0.2, 0.3), 0.5),
            DecisionFlipStatus::Impossible
        );
    }

    #[test]
    fn threshold_flip_is_guaranteed_for_disjoint_decision_sides() {
        assert_eq!(
            classify_flip(interval(0.1, 0.2), interval(0.8, 0.9), 0.5),
            DecisionFlipStatus::Guaranteed
        );
        assert_eq!(
            classify_flip(interval(0.8, 0.9), interval(0.1, 0.2), 0.5),
            DecisionFlipStatus::Guaranteed
        );
    }

    #[test]
    fn threshold_flip_is_only_possible_when_intervals_straddle_decision_boundary() {
        assert_eq!(
            classify_flip(interval(0.4, 0.6), interval(0.7, 0.8), 0.5),
            DecisionFlipStatus::Possible
        );
    }

    #[test]
    fn paired_shared_world_effect_can_tighten_rectangular_outer_bound() {
        let mut item = evaluation(
            "pathway",
            "h1",
            interval(0.0, 1.0),
            interval(0.0, 1.0),
            Some(0.5),
            0.1,
        );
        item.paired_effect = Some(PairedEffectBounds {
            effect_interval: interval(0.0, 0.2),
            evidence_sha256: SHA_C.to_owned(),
            decision_worlds: None,
        });

        let report = analyze_candidate(&candidate(vec![item])).unwrap();
        let scenario = &report.query_impacts[0].scenarios[0];
        assert_eq!(scenario.effect_method, EffectMethod::PairedSharedWorlds);
        assert_eq!(scenario.effect_interval, interval(0.0, 0.2));
        assert_eq!(scenario.maximum_absolute_effect, 0.2);
    }

    #[test]
    fn paired_effect_must_be_inside_conservative_outer_bound_and_have_evidence_digest() {
        let mut outside = evaluation(
            "pathway",
            "h1",
            interval(0.0, 1.0),
            interval(0.0, 1.0),
            Some(0.5),
            0.1,
        );
        outside.paired_effect = Some(PairedEffectBounds {
            effect_interval: interval(-1.1, 0.2),
            evidence_sha256: SHA_C.to_owned(),
            decision_worlds: None,
        });
        assert_eq!(
            analyze_candidate(&candidate(vec![outside])),
            Err(QueryImpactError::PairedEffectOutsideConservativeBound {
                query_id: "pathway".to_owned(),
                scenario_id: "h1".to_owned(),
            })
        );

        let mut missing_digest = evaluation(
            "pathway",
            "h1",
            interval(0.0, 1.0),
            interval(0.0, 1.0),
            Some(0.5),
            0.1,
        );
        missing_digest.paired_effect = Some(PairedEffectBounds {
            effect_interval: interval(0.0, 0.2),
            evidence_sha256: "bad".to_owned(),
            decision_worlds: None,
        });
        assert!(matches!(
            analyze_candidate(&candidate(vec![missing_digest])),
            Err(QueryImpactError::InvalidSha256 {
                kind: "paired effect evidence",
                ..
            })
        ));
    }

    #[test]
    fn matched_worlds_can_prove_no_flip_when_marginals_straddle_threshold() {
        let mut item = evaluation(
            "pathway",
            "h1",
            interval(0.1, 0.9),
            interval(0.1, 0.9),
            Some(0.5),
            0.1,
        );
        item.paired_effect = Some(PairedEffectBounds {
            effect_interval: interval(-0.1, 0.1),
            evidence_sha256: SHA_C.to_owned(),
            decision_worlds: Some(vec![
                PairedDecisionWorld {
                    world_id: "world-a".to_owned(),
                    baseline_value: 0.2,
                    after_edit_value: 0.3,
                },
                PairedDecisionWorld {
                    world_id: "world-b".to_owned(),
                    baseline_value: 0.8,
                    after_edit_value: 0.7,
                },
            ]),
        });

        let report = analyze_candidate(&candidate(vec![item])).unwrap();
        let scenario = &report.query_impacts[0].scenarios[0];
        // Marginal intervals straddle 0.5, but matched worlds never cross it.
        assert_eq!(scenario.baseline_decision, DecisionClass::Indeterminate);
        assert_eq!(scenario.after_edit_decision, DecisionClass::Indeterminate);
        assert_eq!(scenario.decision_flip, DecisionFlipStatus::Impossible);
        assert_eq!(scenario.decision_flip_method, DecisionFlipMethod::PairedSharedWorlds);
        assert_eq!(scenario.paired_decision_world_count, 2);
    }

    #[test]
    fn matched_worlds_can_guarantee_or_make_a_flip_possible() {
        let mut guaranteed = evaluation(
            "pathway",
            "h1",
            interval(0.1, 0.9),
            interval(0.1, 0.9),
            Some(0.5),
            0.1,
        );
        guaranteed.paired_effect = Some(PairedEffectBounds {
            effect_interval: interval(-0.6, 0.6),
            evidence_sha256: SHA_C.to_owned(),
            decision_worlds: Some(vec![
                PairedDecisionWorld {
                    world_id: "world-a".to_owned(),
                    baseline_value: 0.2,
                    after_edit_value: 0.8,
                },
                PairedDecisionWorld {
                    world_id: "world-b".to_owned(),
                    baseline_value: 0.8,
                    after_edit_value: 0.2,
                },
            ]),
        });
        let report = analyze_candidate(&candidate(vec![guaranteed])).unwrap();
        assert_eq!(
            report.query_impacts[0].scenarios[0].decision_flip,
            DecisionFlipStatus::Guaranteed
        );

        let mut possible = evaluation(
            "pathway",
            "h1",
            interval(0.1, 0.9),
            interval(0.1, 0.9),
            Some(0.5),
            0.1,
        );
        possible.paired_effect = Some(PairedEffectBounds {
            effect_interval: interval(-0.1, 0.6),
            evidence_sha256: SHA_C.to_owned(),
            decision_worlds: Some(vec![
                PairedDecisionWorld {
                    world_id: "world-a".to_owned(),
                    baseline_value: 0.2,
                    after_edit_value: 0.8,
                },
                PairedDecisionWorld {
                    world_id: "world-b".to_owned(),
                    baseline_value: 0.8,
                    after_edit_value: 0.7,
                },
                PairedDecisionWorld {
                    world_id: "world-c".to_owned(),
                    baseline_value: 0.2,
                    after_edit_value: 0.3,
                },
            ]),
        });
        let report = analyze_candidate(&candidate(vec![possible])).unwrap();
        assert_eq!(
            report.query_impacts[0].scenarios[0].decision_flip,
            DecisionFlipStatus::Possible
        );
    }

    #[test]
    fn paired_decision_world_ids_must_be_unique() {
        let mut item = evaluation(
            "pathway",
            "h1",
            interval(0.1, 0.9),
            interval(0.1, 0.9),
            Some(0.5),
            0.1,
        );
        item.paired_effect = Some(PairedEffectBounds {
            effect_interval: interval(-0.1, 0.1),
            evidence_sha256: SHA_C.to_owned(),
            decision_worlds: Some(vec![
                PairedDecisionWorld {
                    world_id: "world-a".to_owned(),
                    baseline_value: 0.2,
                    after_edit_value: 0.3,
                },
                PairedDecisionWorld {
                    world_id: "world-a".to_owned(),
                    baseline_value: 0.8,
                    after_edit_value: 0.7,
                },
            ]),
        });
        assert!(matches!(
            analyze_candidate(&candidate(vec![item])),
            Err(QueryImpactError::DuplicatePairedDecisionWorld { .. })
        ));
    }

    #[test]
    fn paired_decision_world_values_must_fit_marginal_intervals() {
        let mut item = evaluation(
            "pathway",
            "h1",
            interval(0.1, 0.9),
            interval(0.1, 0.9),
            Some(0.5),
            0.1,
        );
        item.paired_effect = Some(PairedEffectBounds {
            effect_interval: interval(-0.1, 0.1),
            evidence_sha256: SHA_C.to_owned(),
            decision_worlds: Some(vec![PairedDecisionWorld {
                world_id: "world-a".to_owned(),
                baseline_value: 0.05,
                after_edit_value: 0.1,
            }]),
        });
        assert!(matches!(
            analyze_candidate(&candidate(vec![item])),
            Err(QueryImpactError::PairedDecisionValueOutsideMarginal { .. })
        ));
    }

    #[test]
    fn paired_decision_world_difference_must_fit_paired_effect_interval() {
        let mut item = evaluation(
            "pathway",
            "h1",
            interval(0.1, 0.9),
            interval(0.1, 0.9),
            Some(0.5),
            0.1,
        );
        item.paired_effect = Some(PairedEffectBounds {
            effect_interval: interval(-0.1, 0.1),
            evidence_sha256: SHA_C.to_owned(),
            decision_worlds: Some(vec![PairedDecisionWorld {
                world_id: "world-a".to_owned(),
                baseline_value: 0.2,
                after_edit_value: 0.5,
            }]),
        });
        assert!(matches!(
            analyze_candidate(&candidate(vec![item])),
            Err(QueryImpactError::PairedDecisionDifferenceOutsideEffect { .. })
        ));
    }

    #[test]
    fn empty_paired_decision_world_set_is_rejected() {
        let mut item = evaluation(
            "pathway",
            "h1",
            interval(0.1, 0.9),
            interval(0.1, 0.9),
            Some(0.5),
            0.1,
        );
        item.paired_effect = Some(PairedEffectBounds {
            effect_interval: interval(-0.1, 0.1),
            evidence_sha256: SHA_C.to_owned(),
            decision_worlds: Some(vec![]),
        });
        assert!(matches!(
            analyze_candidate(&candidate(vec![item])),
            Err(QueryImpactError::PairedDecisionWorldsEmpty { .. })
        ));
    }

    #[test]
    fn exact_finite_universe_coverage_upgrades_scope_only_after_id_set_equality() {
        let mut item = evaluation(
            "pathway",
            "h1",
            interval(0.1, 0.9),
            interval(0.1, 0.9),
            Some(0.5),
            0.1,
        );
        item.paired_effect = Some(PairedEffectBounds {
            effect_interval: interval(-0.1, 0.1),
            evidence_sha256: SHA_C.to_owned(),
            decision_worlds: Some(vec![
                PairedDecisionWorld {
                    world_id: "world-a".to_owned(),
                    baseline_value: 0.2,
                    after_edit_value: 0.3,
                },
                PairedDecisionWorld {
                    world_id: "world-b".to_owned(),
                    baseline_value: 0.8,
                    after_edit_value: 0.7,
                },
            ]),
        });
        item.paired_world_set_coverage =
            Some(world_set_coverage(&["world-b", "world-a"]));

        let report = analyze_candidate(&candidate(vec![item])).unwrap();
        let query = &report.query_impacts[0];
        assert_eq!(
            query.scenarios[0].decision_flip_scope,
            DecisionFlipScope::DeclaredFiniteUniverse
        );
        assert_eq!(
            query.aggregate_decision_flip_scope,
            DecisionFlipScope::DeclaredFiniteUniverse
        );
        assert_eq!(query.aggregate_decision_flip, DecisionFlipStatus::Impossible);
    }

    #[test]
    fn missing_expected_world_cannot_be_called_a_closed_finite_universe() {
        let mut item = evaluation(
            "pathway",
            "h1",
            interval(0.1, 0.9),
            interval(0.1, 0.9),
            Some(0.5),
            0.1,
        );
        item.paired_effect = Some(PairedEffectBounds {
            effect_interval: interval(-0.1, 0.1),
            evidence_sha256: SHA_C.to_owned(),
            decision_worlds: Some(vec![PairedDecisionWorld {
                world_id: "world-a".to_owned(),
                baseline_value: 0.2,
                after_edit_value: 0.3,
            }]),
        });
        item.paired_world_set_coverage =
            Some(world_set_coverage(&["world-a", "world-b"]));
        assert_eq!(
            analyze_candidate(&candidate(vec![item])),
            Err(QueryImpactError::ExpectedWorldSetMismatch {
                query_id: "pathway".to_owned(),
                scenario_id: "h1".to_owned(),
            })
        );
    }

    #[test]
    fn unverified_world_closure_receipt_is_rejected() {
        let mut item = evaluation(
            "pathway",
            "h1",
            interval(0.1, 0.9),
            interval(0.1, 0.9),
            Some(0.5),
            0.1,
        );
        item.paired_effect = Some(PairedEffectBounds {
            effect_interval: interval(-0.1, 0.1),
            evidence_sha256: SHA_C.to_owned(),
            decision_worlds: Some(vec![PairedDecisionWorld {
                world_id: "world-a".to_owned(),
                baseline_value: 0.2,
                after_edit_value: 0.3,
            }]),
        });
        let mut coverage = world_set_coverage(&["world-a"]);
        coverage.closure_verified = false;
        item.paired_world_set_coverage = Some(coverage);
        assert!(matches!(
            analyze_candidate(&candidate(vec![item])),
            Err(QueryImpactError::WorldSetCoverageNotVerified { .. })
        ));
    }

    #[test]
    fn duplicate_expected_world_ids_are_rejected() {
        let mut item = evaluation(
            "pathway",
            "h1",
            interval(0.1, 0.9),
            interval(0.1, 0.9),
            Some(0.5),
            0.1,
        );
        item.paired_effect = Some(PairedEffectBounds {
            effect_interval: interval(-0.1, 0.1),
            evidence_sha256: SHA_C.to_owned(),
            decision_worlds: Some(vec![PairedDecisionWorld {
                world_id: "world-a".to_owned(),
                baseline_value: 0.2,
                after_edit_value: 0.3,
            }]),
        });
        item.paired_world_set_coverage =
            Some(world_set_coverage(&["world-a", "world-a"]));
        assert!(matches!(
            analyze_candidate(&candidate(vec![item])),
            Err(QueryImpactError::DuplicateExpectedWorldId { .. })
        ));
    }

    #[test]
    fn finite_world_closure_receipt_must_bind_query_metric_and_threshold() {
        let mut item = evaluation(
            "pathway", "h1", interval(0.1, 0.9), interval(0.1, 0.9), Some(0.5), 0.1,
        );
        item.paired_effect = Some(PairedEffectBounds {
            effect_interval: interval(-0.1, 0.1),
            evidence_sha256: SHA_C.to_owned(),
            decision_worlds: Some(vec![PairedDecisionWorld {
                world_id: "world-a".to_owned(),
                baseline_value: 0.2,
                after_edit_value: 0.3,
            }]),
        });
        item.paired_world_set_coverage = Some(world_set_coverage(&["world-a"]));
        item.paired_world_set_coverage.as_mut().unwrap().query_definition_sha256 = SHA_D.to_owned();
        assert!(matches!(
            analyze_candidate(&candidate(vec![item])),
            Err(QueryImpactError::PairedWorldCoverageBindingMismatch {
                field: "query definition", ..
            })
        ));

        let mut item = evaluation(
            "pathway", "h1", interval(0.1, 0.9), interval(0.1, 0.9), Some(0.5), 0.1,
        );
        item.paired_effect = Some(PairedEffectBounds {
            effect_interval: interval(-0.1, 0.1),
            evidence_sha256: SHA_C.to_owned(),
            decision_worlds: Some(vec![PairedDecisionWorld {
                world_id: "world-a".to_owned(),
                baseline_value: 0.2,
                after_edit_value: 0.3,
            }]),
        });
        item.paired_world_set_coverage = Some(world_set_coverage(&["world-a"]));
        item.paired_world_set_coverage.as_mut().unwrap().decision_threshold = 0.6;
        assert!(matches!(
            analyze_candidate(&candidate(vec![item])),
            Err(QueryImpactError::PairedWorldCoverageBindingMismatch {
                field: "decision threshold", ..
            })
        ));
    }

    #[test]
    fn finite_world_closure_receipt_must_bind_exact_graph_and_candidate_patch() {
        let mut item = evaluation(
            "pathway", "h1", interval(0.1, 0.9), interval(0.1, 0.9), Some(0.5), 0.1,
        );
        item.paired_effect = Some(PairedEffectBounds {
            effect_interval: interval(-0.1, 0.1),
            evidence_sha256: SHA_C.to_owned(),
            decision_worlds: Some(vec![PairedDecisionWorld {
                world_id: "world-a".to_owned(),
                baseline_value: 0.2,
                after_edit_value: 0.3,
            }]),
        });
        item.paired_world_set_coverage = Some(world_set_coverage(&["world-a"]));
        item.paired_world_set_coverage.as_mut().unwrap().base_graph_snapshot_sha256 = SHA_D.to_owned();
        assert!(matches!(
            analyze_candidate(&candidate(vec![item])),
            Err(QueryImpactError::PairedWorldCoverageBindingMismatch {
                field: "base graph snapshot", ..
            })
        ));

        let mut input = candidate(vec![{
            let mut item = evaluation(
                "pathway", "h1", interval(0.1, 0.9), interval(0.1, 0.9), Some(0.5), 0.1,
            );
            item.paired_effect = Some(PairedEffectBounds {
                effect_interval: interval(-0.1, 0.1),
                evidence_sha256: SHA_C.to_owned(),
                decision_worlds: Some(vec![PairedDecisionWorld {
                    world_id: "world-a".to_owned(),
                    baseline_value: 0.2,
                    after_edit_value: 0.3,
                }]),
            });
            item.paired_world_set_coverage = Some(world_set_coverage(&["world-a"]));
            item
        }]);
        input.candidate_patch_sha256 = SHA_D.to_owned();
        assert!(matches!(
            analyze_candidate(&input),
            Err(QueryImpactError::PairedWorldCoverageBindingMismatch {
                field: "candidate patch", ..
            })
        ));
    }

    #[test]
    fn world_set_coverage_requires_actual_paired_decision_worlds() {
        let mut item = evaluation(
            "pathway",
            "h1",
            interval(0.1, 0.9),
            interval(0.1, 0.9),
            Some(0.5),
            0.1,
        );
        item.paired_effect = Some(PairedEffectBounds {
            effect_interval: interval(-0.1, 0.1),
            evidence_sha256: SHA_C.to_owned(),
            decision_worlds: None,
        });
        item.paired_world_set_coverage = Some(world_set_coverage(&["world-a"]));
        assert!(matches!(
            analyze_candidate(&candidate(vec![item])),
            Err(QueryImpactError::PairedWorldCoverageRequiresDecisionWorlds { .. })
        ));
    }

    #[test]
    fn finite_world_receipt_rejects_candidate_operation_sample_task_and_scenario_replay() {
        for field in ["candidate", "operation", "sample", "task", "scenario"] {
            let mut item = evaluation(
                "pathway", "h1", interval(0.1, 0.9), interval(0.1, 0.9), Some(0.5), 0.1,
            );
            item.paired_effect = Some(PairedEffectBounds {
                effect_interval: interval(-0.1, 0.1),
                evidence_sha256: SHA_C.to_owned(),
                decision_worlds: Some(vec![PairedDecisionWorld {
                    world_id: "world-a".to_owned(),
                    baseline_value: 0.2,
                    after_edit_value: 0.3,
                }]),
            });
            item.paired_world_set_coverage = Some(world_set_coverage(&["world-a"]));
            match field {
                "candidate" => {
                    item.paired_world_set_coverage.as_mut().unwrap().candidate_id =
                        "other-candidate".to_owned();
                }
                "operation" => {
                    item.paired_world_set_coverage.as_mut().unwrap().operation_id =
                        "other-operation".to_owned();
                }
                "sample" => {
                    item.paired_world_set_coverage.as_mut().unwrap().sample_hash =
                        "fedcba9876543210fedcba9876543210".to_owned();
                }
                "task" => {
                    item.paired_world_set_coverage.as_mut().unwrap().task_route =
                        TaskRoute::FalseSplitCorrection;
                }
                "scenario" => {
                    item.paired_world_set_coverage.as_mut().unwrap().scenario_id =
                        "other-scenario".to_owned();
                }
                _ => unreachable!(),
            }
            assert!(matches!(
                analyze_candidate(&candidate(vec![item])),
                Err(QueryImpactError::PairedWorldCoverageBindingMismatch { .. })
            ));
        }
    }

    #[test]
    fn differing_scenario_scopes_are_preserved_as_mixed_at_query_level() {
        let mut supplied = evaluation(
            "pathway",
            "a-supplied",
            interval(0.1, 0.9),
            interval(0.1, 0.9),
            Some(0.5),
            0.1,
        );
        supplied.paired_effect = Some(PairedEffectBounds {
            effect_interval: interval(-0.1, 0.1),
            evidence_sha256: SHA_C.to_owned(),
            decision_worlds: Some(vec![PairedDecisionWorld {
                world_id: "world-a".to_owned(),
                baseline_value: 0.2,
                after_edit_value: 0.3,
            }]),
        });

        let mut closed = evaluation(
            "pathway",
            "b-closed",
            interval(0.1, 0.9),
            interval(0.1, 0.9),
            Some(0.5),
            0.1,
        );
        closed.paired_effect = Some(PairedEffectBounds {
            effect_interval: interval(-0.1, 0.1),
            evidence_sha256: SHA_C.to_owned(),
            decision_worlds: Some(vec![PairedDecisionWorld {
                world_id: "world-a".to_owned(),
                baseline_value: 0.2,
                after_edit_value: 0.3,
            }]),
        });
        closed.paired_world_set_coverage = Some(world_set_coverage(&["world-a"]));
        closed.paired_world_set_coverage.as_mut().unwrap().scenario_id = "b-closed".to_owned();

        let report = analyze_candidate(&candidate(vec![supplied, closed])).unwrap();
        let impact = &report.query_impacts[0];
        assert_eq!(
            impact.scenarios[0].decision_flip_scope,
            DecisionFlipScope::SuppliedPairedWorldSet
        );
        assert_eq!(
            impact.scenarios[1].decision_flip_scope,
            DecisionFlipScope::DeclaredFiniteUniverse
        );
        assert_eq!(
            impact.aggregate_decision_flip_scope,
            DecisionFlipScope::Mixed
        );
    }

    #[test]
    fn paired_decision_worlds_require_a_pre_registered_threshold() {
        let mut item = evaluation(
            "pathway",
            "h1",
            interval(0.1, 0.9),
            interval(0.1, 0.9),
            None,
            0.1,
        );
        item.paired_effect = Some(PairedEffectBounds {
            effect_interval: interval(-0.1, 0.1),
            evidence_sha256: SHA_C.to_owned(),
            decision_worlds: Some(vec![PairedDecisionWorld {
                world_id: "world-a".to_owned(),
                baseline_value: 0.2,
                after_edit_value: 0.3,
            }]),
        });
        assert!(matches!(
            analyze_candidate(&candidate(vec![item])),
            Err(QueryImpactError::PairedDecisionWorldsRequireThreshold { .. })
        ));
    }

    #[test]
    fn absent_decision_threshold_is_not_configured() {
        let report = analyze_candidate(&candidate(vec![evaluation(
            "pathway",
            "h1",
            interval(0.1, 0.2),
            interval(0.8, 0.9),
            None,
            0.1,
        )]))
        .unwrap();
        assert_eq!(
            report.query_impacts[0].aggregate_decision_flip,
            DecisionFlipStatus::NotConfigured
        );
        assert_eq!(
            report.query_impacts[0].scenarios[0].baseline_decision,
            DecisionClass::NotConfigured
        );
    }

    #[test]
    fn materiality_is_normalized_only_by_the_query_specific_threshold() {
        let report = analyze_candidate(&candidate(vec![evaluation(
            "pathway",
            "h1",
            interval(0.0, 0.0),
            interval(0.2, 0.3),
            None,
            0.1,
        )]))
        .unwrap();
        let impact = &report.query_impacts[0];
        assert!((impact.worst_case_normalized_materiality - 3.0).abs() < 1e-12);
        assert!(impact.materiality_exceeded_possible);
        assert!(impact.materiality_exceeded_guaranteed);
    }

    #[test]
    fn materiality_can_be_possible_without_being_guaranteed() {
        let report = analyze_candidate(&candidate(vec![evaluation(
            "pathway",
            "h1",
            interval(0.0, 0.0),
            interval(0.05, 0.15),
            None,
            0.1,
        )]))
        .unwrap();
        let impact = &report.query_impacts[0];
        assert!(impact.materiality_exceeded_possible);
        assert!(!impact.materiality_exceeded_guaranteed);
        assert!((impact.scenarios[0].normalized_materiality_lower - 0.5).abs() < 1e-12);
        assert!((impact.scenarios[0].normalized_materiality_upper - 1.5).abs() < 1e-12);
    }

    #[test]
    fn multiple_structural_scenarios_are_preserved_not_averaged() {
        let evaluations = vec![
            evaluation(
                "pathway",
                "h2",
                interval(0.0, 0.1),
                interval(0.0, 0.1),
                Some(0.5),
                0.1,
            ),
            evaluation(
                "pathway",
                "h1",
                interval(0.0, 0.1),
                interval(0.8, 0.9),
                Some(0.5),
                0.1,
            ),
        ];
        let report = analyze_candidate(&candidate(evaluations)).unwrap();
        let impact = &report.query_impacts[0];
        assert_eq!(impact.scenarios.len(), 2);
        assert_eq!(impact.scenarios[0].scenario_id, "h1");
        assert_eq!(impact.scenarios[1].scenario_id, "h2");
        assert_eq!(
            impact.aggregate_decision_flip,
            DecisionFlipStatus::Possible
        );
        assert_eq!(impact.worst_case_normalized_materiality, 9.0);
    }

    #[test]
    fn guaranteed_flip_across_every_scenario_is_reported_as_guaranteed() {
        let evaluations = vec![
            evaluation(
                "pathway",
                "h1",
                interval(0.1, 0.2),
                interval(0.8, 0.9),
                Some(0.5),
                0.1,
            ),
            evaluation(
                "pathway",
                "h2",
                interval(0.0, 0.1),
                interval(0.7, 0.8),
                Some(0.5),
                0.1,
            ),
        ];
        let report = analyze_candidate(&candidate(evaluations)).unwrap();
        assert_eq!(
            report.query_impacts[0].aggregate_decision_flip,
            DecisionFlipStatus::Guaranteed
        );
    }

    #[test]
    fn different_queries_keep_different_metric_units_and_thresholds() {
        let evaluations = vec![
            evaluation(
                "global-count",
                "h1",
                interval(100.0, 110.0),
                interval(105.0, 115.0),
                None,
                20.0,
            ),
            evaluation(
                "pathway",
                "h1",
                interval(0.2, 0.3),
                interval(0.5, 0.6),
                Some(0.4),
                0.1,
            ),
        ];
        let report = analyze_candidate(&candidate(evaluations)).unwrap();
        assert_eq!(report.query_impacts.len(), 2);
        assert_eq!(report.query_impacts[0].query_id, "global-count");
        assert_eq!(report.query_impacts[0].unit_id, "synapses");
        assert_eq!(report.query_impacts[1].query_id, "pathway");
        assert_eq!(report.query_impacts[1].unit_id, "normalized-strength");
    }

    #[test]
    fn query_evaluation_must_be_pre_registered() {
        let mut item = evaluation(
            "pathway",
            "h1",
            interval(0.1, 0.2),
            interval(0.7, 0.8),
            Some(0.5),
            0.1,
        );
        item.pre_registered = false;
        assert_eq!(
            analyze_candidate(&candidate(vec![item])),
            Err(QueryImpactError::QueryNotPreRegistered("pathway".to_owned()))
        );
    }

    #[test]
    fn duplicate_query_scenario_pair_is_rejected() {
        let item = evaluation(
            "pathway",
            "h1",
            interval(0.1, 0.2),
            interval(0.7, 0.8),
            Some(0.5),
            0.1,
        );
        assert!(matches!(
            analyze_candidate(&candidate(vec![item.clone(), item])),
            Err(QueryImpactError::DuplicateQueryScenario { .. })
        ));
    }

    #[test]
    fn query_definition_metric_unit_and_thresholds_must_be_consistent() {
        let first = evaluation(
            "pathway",
            "h1",
            interval(0.1, 0.2),
            interval(0.7, 0.8),
            Some(0.5),
            0.1,
        );
        let mut second = evaluation(
            "pathway",
            "h2",
            interval(0.1, 0.2),
            interval(0.7, 0.8),
            Some(0.5),
            0.1,
        );
        second.query_definition_sha256 = SHA_B.to_owned();
        assert!(matches!(
            analyze_candidate(&candidate(vec![first.clone(), second])),
            Err(QueryImpactError::QueryDefinitionMismatch { .. })
        ));

        let mut second = evaluation(
            "pathway",
            "h2",
            interval(0.1, 0.2),
            interval(0.7, 0.8),
            Some(0.5),
            0.1,
        );
        second.unit_id = "other-unit".to_owned();
        assert!(matches!(
            analyze_candidate(&candidate(vec![first.clone(), second])),
            Err(QueryImpactError::UnitMismatch { .. })
        ));

        let mut second = evaluation(
            "pathway",
            "h2",
            interval(0.1, 0.2),
            interval(0.7, 0.8),
            Some(0.5),
            0.2,
        );
        assert!(matches!(
            analyze_candidate(&candidate(vec![first.clone(), second])),
            Err(QueryImpactError::MaterialityThresholdMismatch { .. })
        ));

        let mut second = evaluation(
            "pathway",
            "h2",
            interval(0.1, 0.2),
            interval(0.7, 0.8),
            Some(0.6),
            0.1,
        );
        assert!(matches!(
            analyze_candidate(&candidate(vec![first, second])),
            Err(QueryImpactError::DecisionThresholdMismatch { .. })
        ));
    }

    #[test]
    fn invalid_intervals_non_finite_values_and_zero_materiality_fail_closed() {
        let mut item = evaluation(
            "pathway",
            "h1",
            interval(0.1, 0.2),
            interval(0.7, 0.8),
            Some(0.5),
            0.1,
        );
        item.baseline = interval(0.4, 0.3);
        assert!(matches!(
            analyze_candidate(&candidate(vec![item])),
            Err(QueryImpactError::InvalidInterval { .. })
        ));

        let mut item = evaluation(
            "pathway",
            "h1",
            interval(0.1, 0.2),
            interval(0.7, 0.8),
            Some(0.5),
            0.0,
        );
        assert!(matches!(
            analyze_candidate(&candidate(vec![item])),
            Err(QueryImpactError::InvalidMaterialityThreshold { .. })
        ));

        let mut item = evaluation(
            "pathway",
            "h1",
            interval(0.1, 0.2),
            interval(0.7, 0.8),
            Some(f64::NAN),
            0.1,
        );
        assert!(matches!(
            analyze_candidate(&candidate(vec![item])),
            Err(QueryImpactError::InvalidDecisionThreshold { .. })
        ));
    }

    #[test]
    fn deterministic_report_is_invariant_to_evaluation_input_order() {
        let a = evaluation(
            "query-a",
            "h1",
            interval(1.0, 2.0),
            interval(3.0, 4.0),
            Some(2.5),
            0.5,
        );
        let b = evaluation(
            "query-b",
            "h2",
            interval(0.1, 0.2),
            interval(0.2, 0.3),
            None,
            0.2,
        );
        let forward = analyze_candidate(&candidate(vec![a.clone(), b.clone()])).unwrap();
        let reverse = analyze_candidate(&candidate(vec![b, a])).unwrap();
        assert_eq!(forward, reverse);
    }

    #[test]
    fn input_identity_hashes_are_validated() {
        let mut input = candidate(vec![evaluation(
            "pathway",
            "h1",
            interval(0.1, 0.2),
            interval(0.7, 0.8),
            Some(0.5),
            0.1,
        )]);
        input.sample_hash = "bad".to_owned();
        assert!(matches!(
            analyze_candidate(&input),
            Err(QueryImpactError::InvalidSampleHash(_))
        ));

        input = candidate(vec![evaluation(
            "pathway",
            "h1",
            interval(0.1, 0.2),
            interval(0.7, 0.8),
            Some(0.5),
            0.1,
        )]);
        input.candidate_patch_sha256 = "bad".to_owned();
        assert!(matches!(
            analyze_candidate(&input),
            Err(QueryImpactError::InvalidSha256 { .. })
        ));
    }

    #[test]
    fn no_query_evaluations_is_an_error() {
        assert_eq!(
            analyze_candidate(&candidate(vec![])),
            Err(QueryImpactError::NoQueryScenarios)
        );
    }
}

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

#[derive(Debug, Clone, PartialEq)]
pub struct ScenarioImpact {
    pub scenario_id: String,
    pub baseline: Interval,
    pub hypothetical_after_edit: Interval,
    /// [after.lower - baseline.upper, after.upper - baseline.lower].
    pub effect_interval: Interval,
    pub minimum_absolute_effect: f64,
    pub maximum_absolute_effect: f64,
    pub normalized_materiality_lower: f64,
    pub normalized_materiality_upper: f64,
    pub baseline_decision: DecisionClass,
    pub after_edit_decision: DecisionClass,
    pub decision_flip: DecisionFlipStatus,
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
    DuplicateScenarioSetEntry { query_id: String, scenario_id: String },
    QueryDefinitionMismatch { query_id: String },
    MetricIdentityMismatch { query_id: String },
    UnitMismatch { query_id: String },
    MaterialityThresholdMismatch { query_id: String },
    DecisionThresholdMismatch { query_id: String },
    NonFiniteDerivedValue { query_id: String, scenario_id: String },
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
    scenario_ids: BTreeSet<String>,
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
                scenario_ids: BTreeSet::new(),
                scenarios: Vec::new(),
            });

        validate_signature_consistency(&evaluation.query_id, &accumulator.signature, &signature)?;

        if !accumulator
            .scenario_ids
            .insert(evaluation.scenario_id.clone())
        {
            return Err(QueryImpactError::DuplicateScenarioSetEntry {
                query_id: evaluation.query_id.clone(),
                scenario_id: evaluation.scenario_id.clone(),
            });
        }

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
    let effect_interval = evaluation
        .hypothetical_after_edit
        .difference(evaluation.baseline)?;
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

    let (baseline_decision, after_edit_decision, decision_flip) =
        match evaluation.decision_threshold {
            Some(threshold) => {
                let baseline = classify_interval(evaluation.baseline, threshold);
                let after = classify_interval(evaluation.hypothetical_after_edit, threshold);
                let flip = classify_flip(evaluation.baseline, evaluation.hypothetical_after_edit, threshold);
                (baseline, after, flip)
            }
            None => (
                DecisionClass::NotConfigured,
                DecisionClass::NotConfigured,
                DecisionFlipStatus::NotConfigured,
            ),
        };

    Ok(ScenarioImpact {
        scenario_id: evaluation.scenario_id.clone(),
        baseline: evaluation.baseline,
        hypothetical_after_edit: evaluation.hypothetical_after_edit,
        effect_interval,
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

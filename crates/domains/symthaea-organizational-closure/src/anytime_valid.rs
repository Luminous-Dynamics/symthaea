//! Anytime-valid causal effect confidence sequences for randomized experiments.
//!
//! This v0 uses an auditable Hoeffding confidence sequence with a summable
//! alpha-spending schedule. It is intentionally conservative and provides
//! time-uniform coverage under the declared bounded-outcome, fixed-mean and
//! predictable-randomization contract. It is not an observational estimator,
//! and does not replace mixture/stitching or e-process comparators.

use super::experiment_design::ExperimentEvidenceStatus;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

const LN_2: f64 = std::f64::consts::LN_2;
const CLAIM_TOLERANCE: f64 = 1.0e-12;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RandomizedArm {
    Control,
    Treatment,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CausalEffectInterval {
    pub lower: f64,
    pub upper: f64,
}

impl CausalEffectInterval {
    pub fn new(lower: f64, upper: f64) -> Self {
        Self { lower, upper }
    }

    pub fn is_valid(&self) -> bool {
        self.lower.is_finite() && self.upper.is_finite() && self.lower <= self.upper
    }

    pub fn contains(&self, value: f64) -> bool {
        self.lower <= value && value <= self.upper
    }

    pub fn contains_interval(&self, other: &Self) -> bool {
        self.is_valid()
            && other.is_valid()
            && self.lower <= other.lower + CLAIM_TOLERANCE
            && self.upper + CLAIM_TOLERANCE >= other.upper
    }

    pub fn width(&self) -> f64 {
        self.upper - self.lower
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SequentialCausalQuery {
    pub query_id: String,
    /// Known support for every observed outcome in this randomized query.
    pub outcome_lower: f64,
    pub outcome_upper: f64,
    pub decision_threshold: f64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnytimeValidEvidenceReceipt {
    pub receipt_id: String,
    pub description: String,
}

/// Public experiment/task contract. The outcome stream and all hidden truth
/// remain outside this structure.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnytimeValidCausalScenario {
    pub queries: Vec<SequentialCausalQuery>,
    /// Total family-wise error probability across all queries and both arms.
    pub familywise_alpha: f64,
    pub max_observations: usize,
    pub evidence_receipts: BTreeMap<String, AnytimeValidEvidenceReceipt>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SequentialCausalObservation {
    pub observation_id: String,
    pub query_id: String,
    pub arm: RandomizedArm,
    pub outcome: f64,
    pub evidence_receipt_id: String,
}

/// Agent claim at its chosen global stopping prefix. The prefix is exactly
/// the ordered observations carried by the trace; no future outcomes are
/// supplied to or scored as if they had been observed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SequentialCausalClaim {
    pub query_id: String,
    pub reported_effect_interval: CausalEffectInterval,
    pub decision: AnytimeValidDecision,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnytimeValidCausalTrace {
    pub stop_after_observation_count: usize,
    pub observations: Vec<SequentialCausalObservation>,
    pub claims: Vec<SequentialCausalClaim>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnytimeValidDecision {
    Act,
    DoNotAct,
    Defer,
}

/// Hidden adjudication for one immutable evidence receipt. The fixed-mean
/// disposition qualifies the registered outcome-generating contract; it is
/// not inferred by looking at the observed outcomes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AnytimeValidEvidenceOracle {
    pub receipt_id_matches: bool,
    pub randomization: ExperimentEvidenceStatus,
    pub delivery: ExperimentEvidenceStatus,
    pub isolation: ExperimentEvidenceStatus,
    pub measurement: ExperimentEvidenceStatus,
    pub fixed_mean_contract: ExperimentEvidenceStatus,
}

/// Evaluator-only true causal effects and evidence dispositions.
#[derive(Debug, Clone, PartialEq)]
pub struct AnytimeValidCausalOracle {
    pub true_effects: BTreeMap<String, f64>,
    pub evidence: BTreeMap<String, AnytimeValidEvidenceOracle>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnytimeValidQualification {
    Qualified,
    NotQualified,
    Inconclusive,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnytimeValidDecisionQuality {
    Robust,
    Conservative,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnytimeValidFindingCode {
    EmptyQuerySet,
    EmptyQueryId,
    DuplicateQueryId,
    InvalidOutcomeBounds,
    NonFiniteDecisionThreshold,
    InvalidFamilywiseAlpha,
    EmptyEvidenceReceiptId,
    EvidenceReceiptKeyMismatch,
    EmptyObservationId,
    DuplicateObservationId,
    UnknownObservationQuery,
    UnknownEvidenceReceipt,
    MissingEvidenceOracle,
    EvidenceOracleReceiptMismatch,
    RejectedRandomization,
    RejectedDelivery,
    RejectedIsolation,
    RejectedMeasurement,
    RejectedFixedMeanContract,
    EvidenceUnavailable,
    OutcomeNonFinite,
    OutcomeOutsideSupport,
    StopPrefixMismatch,
    ObservationBudgetExceeded,
    MissingTrueEffect,
    NonFiniteTrueEffect,
    TrueEffectOutsideSupport,
    DuplicateClaim,
    MissingClaim,
    InvalidClaimInterval,
    ClaimOutsideEffectSupport,
    ClaimNarrowerThanConfidenceSequence,
    UnsafeActDecision,
    UnsafeDoNotActDecision,
    UnknownTrueEffectQuery,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnytimeValidFinding {
    pub code: AnytimeValidFindingCode,
    pub query_id: Option<String>,
    pub observation_id: Option<String>,
    pub receipt_id: Option<String>,
    pub detail: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnytimeValidQueryReport {
    pub query_id: String,
    pub control_observation_count: usize,
    pub treatment_observation_count: usize,
    pub armwise_alpha: f64,
    pub control_mean_interval: CausalEffectInterval,
    pub treatment_mean_interval: CausalEffectInterval,
    pub recomputed_effect_interval: CausalEffectInterval,
    pub reported_effect_interval: CausalEffectInterval,
    /// Coverage is an empirical evaluator metric; a miss is possible at the
    /// declared alpha and is not itself a deterministic contract violation.
    pub true_effect_covered: bool,
    pub decision_quality: AnytimeValidDecisionQuality,
    pub interval_width: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnytimeValidCausalReport {
    pub qualification: AnytimeValidQualification,
    pub familywise_alpha: f64,
    pub per_arm_alpha: f64,
    pub stop_after_observation_count: usize,
    pub observations_used: usize,
    pub query_reports: Vec<AnytimeValidQueryReport>,
    pub familywise_truth_covered: Option<bool>,
    pub mean_contract_evidence_qualified: bool,
    pub findings: Vec<AnytimeValidFinding>,
}

impl AnytimeValidCausalReport {
    pub fn is_qualified(&self) -> bool {
        self.qualification == AnytimeValidQualification::Qualified && self.findings.is_empty()
    }

    pub fn all_queries_covered(&self) -> Option<bool> {
        self.familywise_truth_covered
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnytimeValidPolicy {
    /// Reserved for stable policy versioning and future alternative CS families.
    pub claim_tolerance: f64,
}

impl Default for AnytimeValidPolicy {
    fn default() -> Self {
        Self {
            claim_tolerance: CLAIM_TOLERANCE,
        }
    }
}

fn push_finding(
    findings: &mut Vec<AnytimeValidFinding>,
    code: AnytimeValidFindingCode,
    query_id: Option<String>,
    observation_id: Option<String>,
    receipt_id: Option<String>,
    detail: impl Into<String>,
) {
    findings.push(AnytimeValidFinding {
        code,
        query_id,
        observation_id,
        receipt_id,
        detail: detail.into(),
    });
}

/// Two-sided Hoeffding confidence-sequence radius for one arm after n
/// observations, using alpha_n = alpha_arm / (n * (n + 1)).
///
/// Since sum_{n >= 1} 1/(n(n+1)) = 1, a union bound over all sample sizes
/// allocates at most alpha_arm to this arm. The log-domain expression avoids
/// overflowing n * (n + 1) for large observation counts.
pub fn hoeffding_alpha_spending_radius(
    outcome_range: f64,
    n: usize,
    armwise_alpha: f64,
) -> Option<f64> {
    if !outcome_range.is_finite()
        || outcome_range < 0.0
        || n == 0
        || !armwise_alpha.is_finite()
        || !(0.0 < armwise_alpha && armwise_alpha < 1.0)
    {
        return None;
    }

    let n_f = n as f64;
    let log_two_over_alpha_n = LN_2 - armwise_alpha.ln() + n_f.ln() + (n_f + 1.0).ln();
    let multiplier = (log_two_over_alpha_n / (2.0 * n_f)).sqrt();
    let radius = if multiplier >= 1.0 {
        outcome_range
    } else {
        outcome_range * multiplier
    };
    radius.is_finite().then_some(radius)
}

fn arm_confidence_interval(
    outcomes: &[f64],
    lower: f64,
    upper: f64,
    armwise_alpha: f64,
) -> CausalEffectInterval {
    if outcomes.is_empty() {
        return CausalEffectInterval::new(lower, upper);
    }

    let n = outcomes.len();
    // Stable online mean avoids overflowing a direct sum of large bounded values.
    let mean = outcomes
        .iter()
        .enumerate()
        .fold(0.0, |current, (index, value)| {
            current + (value - current) / (index + 1) as f64
        });
    let range = upper - lower;
    let radius = hoeffding_alpha_spending_radius(range, n, armwise_alpha).unwrap_or(range);
    let distance_to_lower = mean - lower;
    let distance_to_upper = upper - mean;
    let interval_lower = if radius >= distance_to_lower {
        lower
    } else {
        mean - radius
    };
    let interval_upper = if radius >= distance_to_upper {
        upper
    } else {
        mean + radius
    };

    CausalEffectInterval::new(interval_lower, interval_upper)
}

fn effect_support(lower: f64, upper: f64) -> CausalEffectInterval {
    let range = upper - lower;
    CausalEffectInterval::new(-range, range)
}

fn expected_decision_quality(
    claim: &SequentialCausalClaim,
    recomputed: CausalEffectInterval,
    threshold: f64,
) -> Result<AnytimeValidDecisionQuality, AnytimeValidFindingCode> {
    match claim.decision {
        AnytimeValidDecision::Act => {
            if claim.reported_effect_interval.lower > threshold {
                Ok(AnytimeValidDecisionQuality::Robust)
            } else {
                Err(AnytimeValidFindingCode::UnsafeActDecision)
            }
        }
        AnytimeValidDecision::DoNotAct => {
            if claim.reported_effect_interval.upper <= threshold {
                Ok(AnytimeValidDecisionQuality::Robust)
            } else {
                Err(AnytimeValidFindingCode::UnsafeDoNotActDecision)
            }
        }
        AnytimeValidDecision::Defer => {
            if recomputed.lower <= threshold && threshold < recomputed.upper {
                Ok(AnytimeValidDecisionQuality::Robust)
            } else {
                Ok(AnytimeValidDecisionQuality::Conservative)
            }
        }
    }
}

fn empty_report(
    scenario: &AnytimeValidCausalScenario,
    trace: &AnytimeValidCausalTrace,
    qualification: AnytimeValidQualification,
    findings: Vec<AnytimeValidFinding>,
) -> AnytimeValidCausalReport {
    AnytimeValidCausalReport {
        qualification,
        familywise_alpha: scenario.familywise_alpha,
        per_arm_alpha: if scenario.queries.is_empty()
            || !scenario.familywise_alpha.is_finite()
            || !(0.0 < scenario.familywise_alpha && scenario.familywise_alpha < 1.0)
        {
            0.0
        } else {
            scenario.familywise_alpha / (2.0 * scenario.queries.len() as f64)
        },
        stop_after_observation_count: trace.stop_after_observation_count,
        observations_used: 0,
        query_reports: Vec::new(),
        familywise_truth_covered: None,
        mean_contract_evidence_qualified: false,
        findings,
    }
}

/// Recompute all query confidence sequences at the agent's selected global
/// stopping prefix. The confidence construction is time-uniform: coverage is
/// controlled for every arm sample size, so the prefix may be selected after
/// repeatedly inspecting the outcomes.
///
/// Assumptions: each query has a fixed arm-specific mean, outcomes are bounded
/// by its declared support, and assignment is randomized or predictable under
/// the external evidence contract. This is not an observational-data method.
pub fn evaluate_anytime_valid_causal_effects(
    scenario: &AnytimeValidCausalScenario,
    oracle: &AnytimeValidCausalOracle,
    trace: &AnytimeValidCausalTrace,
    policy: &AnytimeValidPolicy,
) -> AnytimeValidCausalReport {
    let mut findings = Vec::new();

    if scenario.queries.is_empty() {
        push_finding(
            &mut findings,
            AnytimeValidFindingCode::EmptyQuerySet,
            None,
            None,
            None,
            "at least one causal query must be registered",
        );
    }
    if !scenario.familywise_alpha.is_finite()
        || scenario.familywise_alpha <= 0.0
        || scenario.familywise_alpha >= 1.0
    {
        push_finding(
            &mut findings,
            AnytimeValidFindingCode::InvalidFamilywiseAlpha,
            None,
            None,
            None,
            "family-wise alpha must be finite and strictly between zero and one",
        );
    }
    if !policy.claim_tolerance.is_finite() || policy.claim_tolerance < 0.0 {
        push_finding(
            &mut findings,
            AnytimeValidFindingCode::InvalidClaimInterval,
            None,
            None,
            None,
            "claim tolerance must be finite and non-negative",
        );
    }

    let mut queries = BTreeMap::new();
    for query in &scenario.queries {
        if query.query_id.trim().is_empty() {
            push_finding(
                &mut findings,
                AnytimeValidFindingCode::EmptyQueryId,
                None,
                None,
                None,
                "query ID is empty",
            );
        }
        if !query.outcome_lower.is_finite()
            || !query.outcome_upper.is_finite()
            || query.outcome_lower >= query.outcome_upper
            || !(query.outcome_upper - query.outcome_lower).is_finite()
            || query.outcome_upper - query.outcome_lower > f64::MAX / 2.0
        {
            push_finding(
                &mut findings,
                AnytimeValidFindingCode::InvalidOutcomeBounds,
                Some(query.query_id.clone()),
                None,
                None,
                "outcome support must have finite, strictly ordered bounds",
            );
        }
        if !query.decision_threshold.is_finite() {
            push_finding(
                &mut findings,
                AnytimeValidFindingCode::NonFiniteDecisionThreshold,
                Some(query.query_id.clone()),
                None,
                None,
                "decision threshold must be finite",
            );
        }
        if queries.insert(query.query_id.clone(), query).is_some() {
            push_finding(
                &mut findings,
                AnytimeValidFindingCode::DuplicateQueryId,
                Some(query.query_id.clone()),
                None,
                None,
                "query ID is duplicated",
            );
        }
    }

    let mut receipt_ids = BTreeSet::new();
    for (key, receipt) in &scenario.evidence_receipts {
        if key.trim().is_empty() {
            push_finding(
                &mut findings,
                AnytimeValidFindingCode::EmptyEvidenceReceiptId,
                None,
                None,
                Some(key.clone()),
                "evidence receipt ID is empty",
            );
        }
        if key != &receipt.receipt_id {
            push_finding(
                &mut findings,
                AnytimeValidFindingCode::EvidenceReceiptKeyMismatch,
                None,
                None,
                Some(key.clone()),
                "evidence receipt map key differs from its receipt ID",
            );
        }
        if !receipt_ids.insert(receipt.receipt_id.clone()) {
            push_finding(
                &mut findings,
                AnytimeValidFindingCode::EvidenceReceiptKeyMismatch,
                None,
                None,
                Some(receipt.receipt_id.clone()),
                "evidence receipt ID is duplicated",
            );
        }
    }

    for (query_id, true_effect) in &oracle.true_effects {
        let Some(query) = queries.get(query_id) else {
            push_finding(
                &mut findings,
                AnytimeValidFindingCode::UnknownTrueEffectQuery,
                Some(query_id.clone()),
                None,
                None,
                "evaluator true-effect map contains an unknown query",
            );
            continue;
        };
        if !true_effect.is_finite() {
            push_finding(
                &mut findings,
                AnytimeValidFindingCode::NonFiniteTrueEffect,
                Some(query_id.clone()),
                None,
                None,
                "evaluator true effect must be finite",
            );
        } else if !effect_support(query.outcome_lower, query.outcome_upper).contains(*true_effect) {
            push_finding(
                &mut findings,
                AnytimeValidFindingCode::TrueEffectOutsideSupport,
                Some(query_id.clone()),
                None,
                None,
                "evaluator true effect lies outside the declared effect support",
            );
        }
    }
    for query_id in queries.keys() {
        if !oracle.true_effects.contains_key(query_id) {
            push_finding(
                &mut findings,
                AnytimeValidFindingCode::MissingTrueEffect,
                Some(query_id.clone()),
                None,
                None,
                "evaluator oracle omits a registered query's true effect",
            );
        }
    }

    if trace.stop_after_observation_count != trace.observations.len() {
        push_finding(
            &mut findings,
            AnytimeValidFindingCode::StopPrefixMismatch,
            None,
            None,
            None,
            "trace must contain exactly the observed prefix selected by its stopping count",
        );
    }
    if trace.stop_after_observation_count > scenario.max_observations
        || trace.observations.len() > scenario.max_observations
    {
        push_finding(
            &mut findings,
            AnytimeValidFindingCode::ObservationBudgetExceeded,
            None,
            None,
            None,
            "selected stopping prefix exceeds the observation budget",
        );
    }

    let mut observation_ids = BTreeSet::new();
    let mut observations_by_query: BTreeMap<String, (Vec<f64>, Vec<f64>)> = queries
        .keys()
        .map(|id| (id.clone(), (Vec::new(), Vec::new())))
        .collect();

    let mut unavailable_evidence = false;
    let mut rejected_evidence = false;
    let mut mean_contract_evidence_qualified = true;

    for observation in &trace.observations {
        if observation.observation_id.trim().is_empty() {
            push_finding(
                &mut findings,
                AnytimeValidFindingCode::EmptyObservationId,
                Some(observation.query_id.clone()),
                Some(observation.observation_id.clone()),
                None,
                "observation ID is empty",
            );
        }
        if !observation_ids.insert(observation.observation_id.clone()) {
            push_finding(
                &mut findings,
                AnytimeValidFindingCode::DuplicateObservationId,
                Some(observation.query_id.clone()),
                Some(observation.observation_id.clone()),
                None,
                "observation ID is duplicated",
            );
        }

        let Some(query) = queries.get(&observation.query_id) else {
            push_finding(
                &mut findings,
                AnytimeValidFindingCode::UnknownObservationQuery,
                Some(observation.query_id.clone()),
                Some(observation.observation_id.clone()),
                None,
                "observation references an unregistered query",
            );
            continue;
        };

        if !scenario
            .evidence_receipts
            .contains_key(&observation.evidence_receipt_id)
        {
            push_finding(
                &mut findings,
                AnytimeValidFindingCode::UnknownEvidenceReceipt,
                Some(observation.query_id.clone()),
                Some(observation.observation_id.clone()),
                Some(observation.evidence_receipt_id.clone()),
                "observation references an evidence receipt outside the agent-visible ledger",
            );
            continue;
        }

        let Some(evidence) = oracle.evidence.get(&observation.evidence_receipt_id) else {
            push_finding(
                &mut findings,
                AnytimeValidFindingCode::MissingEvidenceOracle,
                Some(observation.query_id.clone()),
                Some(observation.observation_id.clone()),
                Some(observation.evidence_receipt_id.clone()),
                "evaluator has no independent adjudication for this evidence receipt",
            );
            continue;
        };

        if !evidence.receipt_id_matches {
            push_finding(
                &mut findings,
                AnytimeValidFindingCode::EvidenceOracleReceiptMismatch,
                Some(observation.query_id.clone()),
                Some(observation.observation_id.clone()),
                Some(observation.evidence_receipt_id.clone()),
                "evaluator evidence receipt identity does not match the referenced ID",
            );
            continue;
        }

        let gates = [
            (evidence.randomization, "randomization"),
            (evidence.delivery, "delivery"),
            (evidence.isolation, "isolation"),
            (evidence.measurement, "measurement"),
            (evidence.fixed_mean_contract, "fixed-mean contract"),
        ];
        for (status, label) in gates {
            match status {
                ExperimentEvidenceStatus::Verified => {}
                ExperimentEvidenceStatus::Rejected => {
                    rejected_evidence = true;
                    mean_contract_evidence_qualified = false;
                    let code = match label {
                        "randomization" => AnytimeValidFindingCode::RejectedRandomization,
                        "delivery" => AnytimeValidFindingCode::RejectedDelivery,
                        "isolation" => AnytimeValidFindingCode::RejectedIsolation,
                        "measurement" => AnytimeValidFindingCode::RejectedMeasurement,
                        _ => AnytimeValidFindingCode::RejectedFixedMeanContract,
                    };
                    push_finding(
                        &mut findings,
                        code,
                        Some(observation.query_id.clone()),
                        Some(observation.observation_id.clone()),
                        Some(observation.evidence_receipt_id.clone()),
                        format!("{label} evidence was rejected"),
                    );
                }
                ExperimentEvidenceStatus::Unavailable => {
                    unavailable_evidence = true;
                    mean_contract_evidence_qualified = false;
                    push_finding(
                        &mut findings,
                        AnytimeValidFindingCode::EvidenceUnavailable,
                        Some(observation.query_id.clone()),
                        Some(observation.observation_id.clone()),
                        Some(observation.evidence_receipt_id.clone()),
                        format!("{label} evidence is unavailable"),
                    );
                }
            }
        }

        if !observation.outcome.is_finite() {
            push_finding(
                &mut findings,
                AnytimeValidFindingCode::OutcomeNonFinite,
                Some(observation.query_id.clone()),
                Some(observation.observation_id.clone()),
                None,
                "observed outcome must be finite",
            );
            continue;
        }
        if observation.outcome < query.outcome_lower || observation.outcome > query.outcome_upper {
            push_finding(
                &mut findings,
                AnytimeValidFindingCode::OutcomeOutsideSupport,
                Some(observation.query_id.clone()),
                Some(observation.observation_id.clone()),
                None,
                "observed outcome lies outside the registered support",
            );
            continue;
        }

        if let Some((control, treatment)) = observations_by_query.get_mut(&observation.query_id) {
            match observation.arm {
                RandomizedArm::Control => control.push(observation.outcome),
                RandomizedArm::Treatment => treatment.push(observation.outcome),
            }
        }
    }

    let claims: BTreeMap<String, &SequentialCausalClaim> = trace
        .claims
        .iter()
        .map(|claim| (claim.query_id.clone(), claim))
        .collect();
    if claims.len() != trace.claims.len() {
        push_finding(
            &mut findings,
            AnytimeValidFindingCode::DuplicateClaim,
            None,
            None,
            None,
            "a query has more than one interval/decision claim",
        );
    }
    for query_id in queries.keys() {
        if !claims.contains_key(query_id) {
            push_finding(
                &mut findings,
                AnytimeValidFindingCode::MissingClaim,
                Some(query_id.clone()),
                None,
                None,
                "trace omits a registered query claim",
            );
        }
    }
    for query_id in claims.keys() {
        if !queries.contains_key(query_id) {
            push_finding(
                &mut findings,
                AnytimeValidFindingCode::DuplicateClaim,
                Some(query_id.clone()),
                None,
                None,
                "trace contains a claim for an unknown query",
            );
        }
    }

    if !findings.is_empty() {
        let qualification = if unavailable_evidence
            && !rejected_evidence
            && findings
                .iter()
                .all(|finding| finding.code == AnytimeValidFindingCode::EvidenceUnavailable)
        {
            AnytimeValidQualification::Inconclusive
        } else {
            AnytimeValidQualification::NotQualified
        };
        return empty_report(scenario, trace, qualification, findings);
    }

    let per_arm_alpha = scenario.familywise_alpha / (2.0 * scenario.queries.len() as f64);
    let mut query_reports = Vec::new();
    let mut familywise_truth_covered = true;

    for query in &scenario.queries {
        let (control, treatment) = &observations_by_query[&query.query_id];
        let control_interval = arm_confidence_interval(
            control,
            query.outcome_lower,
            query.outcome_upper,
            per_arm_alpha,
        );
        let treatment_interval = arm_confidence_interval(
            treatment,
            query.outcome_lower,
            query.outcome_upper,
            per_arm_alpha,
        );
        let recomputed = CausalEffectInterval::new(
            treatment_interval.lower - control_interval.upper,
            treatment_interval.upper - control_interval.lower,
        );

        let claim = claims[&query.query_id];
        if !claim.reported_effect_interval.is_valid() {
            push_finding(
                &mut findings,
                AnytimeValidFindingCode::InvalidClaimInterval,
                Some(query.query_id.clone()),
                None,
                None,
                "reported effect interval must be finite and ordered",
            );
            continue;
        }
        let support = effect_support(query.outcome_lower, query.outcome_upper);
        if claim.reported_effect_interval.lower < support.lower - policy.claim_tolerance
            || claim.reported_effect_interval.upper > support.upper + policy.claim_tolerance
        {
            push_finding(
                &mut findings,
                AnytimeValidFindingCode::ClaimOutsideEffectSupport,
                Some(query.query_id.clone()),
                None,
                None,
                "reported interval extends outside the known treatment-effect support",
            );
            continue;
        }
        let contains_recomputed = claim.reported_effect_interval.lower
            <= recomputed.lower + policy.claim_tolerance
            && claim.reported_effect_interval.upper + policy.claim_tolerance >= recomputed.upper;
        if !contains_recomputed {
            push_finding(
                &mut findings,
                AnytimeValidFindingCode::ClaimNarrowerThanConfidenceSequence,
                Some(query.query_id.clone()),
                None,
                None,
                "reported interval is narrower than the evaluator-recomputed confidence sequence",
            );
            continue;
        }

        let decision_quality =
            match expected_decision_quality(claim, recomputed, query.decision_threshold) {
                Ok(value) => value,
                Err(code) => {
                    push_finding(
                        &mut findings,
                        code,
                        Some(query.query_id.clone()),
                        None,
                        None,
                        "reported decision is not robust over its own declared effect interval",
                    );
                    continue;
                }
            };

        let true_effect = oracle.true_effects[&query.query_id];
        let covered = recomputed.contains(true_effect);
        familywise_truth_covered &= covered;
        query_reports.push(AnytimeValidQueryReport {
            query_id: query.query_id.clone(),
            control_observation_count: control.len(),
            treatment_observation_count: treatment.len(),
            armwise_alpha: per_arm_alpha,
            control_mean_interval: control_interval,
            treatment_mean_interval: treatment_interval,
            recomputed_effect_interval: recomputed,
            reported_effect_interval: claim.reported_effect_interval,
            true_effect_covered: covered,
            decision_quality,
            interval_width: recomputed.width(),
        });
    }

    let qualification = if findings.is_empty() {
        AnytimeValidQualification::Qualified
    } else {
        AnytimeValidQualification::NotQualified
    };

    AnytimeValidCausalReport {
        qualification,
        familywise_alpha: scenario.familywise_alpha,
        per_arm_alpha,
        stop_after_observation_count: trace.stop_after_observation_count,
        observations_used: trace.observations.len(),
        query_reports,
        familywise_truth_covered: Some(familywise_truth_covered),
        mean_contract_evidence_qualified,
        findings,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn query(id: &str) -> SequentialCausalQuery {
        SequentialCausalQuery {
            query_id: id.into(),
            outcome_lower: 0.0,
            outcome_upper: 1.0,
            decision_threshold: 0.2,
        }
    }

    fn observation(
        id: impl Into<String>,
        query_id: &str,
        arm: RandomizedArm,
        outcome: f64,
    ) -> SequentialCausalObservation {
        SequentialCausalObservation {
            observation_id: id.into(),
            query_id: query_id.into(),
            arm,
            outcome,
            evidence_receipt_id: "receipt-1".into(),
        }
    }

    fn scenario(
        queries: Vec<SequentialCausalQuery>,
        max_observations: usize,
    ) -> AnytimeValidCausalScenario {
        AnytimeValidCausalScenario {
            queries,
            familywise_alpha: 0.05,
            max_observations,
            evidence_receipts: [(
                "receipt-1".into(),
                AnytimeValidEvidenceReceipt {
                    receipt_id: "receipt-1".into(),
                    description: "Verified experiment contract receipt".into(),
                },
            )]
            .into(),
        }
    }

    fn oracle(effects: &[(&str, f64)]) -> AnytimeValidCausalOracle {
        AnytimeValidCausalOracle {
            true_effects: effects
                .iter()
                .map(|(id, effect)| ((*id).to_owned(), *effect))
                .collect(),
            evidence: [(
                "receipt-1".into(),
                AnytimeValidEvidenceOracle {
                    receipt_id_matches: true,
                    randomization: ExperimentEvidenceStatus::Verified,
                    delivery: ExperimentEvidenceStatus::Verified,
                    isolation: ExperimentEvidenceStatus::Verified,
                    measurement: ExperimentEvidenceStatus::Verified,
                    fixed_mean_contract: ExperimentEvidenceStatus::Verified,
                },
            )]
            .into(),
        }
    }

    fn trace(
        observations: Vec<SequentialCausalObservation>,
        claims: Vec<SequentialCausalClaim>,
    ) -> AnytimeValidCausalTrace {
        AnytimeValidCausalTrace {
            stop_after_observation_count: observations.len(),
            observations,
            claims,
        }
    }

    fn conservative_claim(id: &str) -> SequentialCausalClaim {
        SequentialCausalClaim {
            query_id: id.into(),
            reported_effect_interval: CausalEffectInterval::new(-1.0, 1.0),
            decision: AnytimeValidDecision::Defer,
        }
    }

    #[test]
    fn alpha_spending_radius_is_finite_positive_and_invalid_inputs_fail_closed() {
        let radius = hoeffding_alpha_spending_radius(1.0, 100, 0.025).unwrap();
        assert!(radius.is_finite() && radius > 0.0);
        assert!(hoeffding_alpha_spending_radius(1.0, 0, 0.025).is_none());
        assert!(hoeffding_alpha_spending_radius(1.0, 100, 1.0).is_none());
        assert!(hoeffding_alpha_spending_radius(f64::NAN, 100, 0.025).is_none());
    }

    #[test]
    fn alpha_spending_is_summable_within_the_armwise_error_budget() {
        let arm_alpha = 0.025;
        let spent: f64 = (1..=100_000)
            .map(|n| arm_alpha / (n as f64 * (n as f64 + 1.0)))
            .sum();
        assert!(spent <= arm_alpha);
        assert!(spent > arm_alpha * 0.9999);
    }

    #[test]
    fn zero_observations_return_the_full_effect_support() {
        let scenario = scenario(vec![query("q1")], 0);
        let oracle = oracle(&[("q1", 0.0)]);
        let claim = SequentialCausalClaim {
            query_id: "q1".into(),
            reported_effect_interval: CausalEffectInterval::new(-1.0, 1.0),
            decision: AnytimeValidDecision::Defer,
        };
        let report = evaluate_anytime_valid_causal_effects(
            &scenario,
            &oracle,
            &trace(vec![], vec![claim]),
            &AnytimeValidPolicy::default(),
        );
        assert!(report.is_qualified(), "{:?}", report.findings);
        assert_eq!(
            report.query_reports[0].recomputed_effect_interval,
            CausalEffectInterval::new(-1.0, 1.0),
        );
        assert_eq!(report.query_reports[0].control_observation_count, 0);
        assert_eq!(report.query_reports[0].treatment_observation_count, 0);
    }

    #[test]
    fn familywise_alpha_is_split_across_all_queries_and_both_arms() {
        let scenario = scenario(vec![query("q1"), query("q2")], 0);
        let oracle = oracle(&[("q1", 0.0), ("q2", 0.0)]);
        let claims = vec![conservative_claim("q1"), conservative_claim("q2")];
        let report = evaluate_anytime_valid_causal_effects(
            &scenario,
            &oracle,
            &trace(vec![], claims),
            &AnytimeValidPolicy::default(),
        );
        assert!(report.is_qualified(), "{:?}", report.findings);
        assert_eq!(report.per_arm_alpha, 0.0125);
        assert_eq!(report.query_reports.len(), 2);
        assert!(
            report
                .query_reports
                .iter()
                .all(|item| item.armwise_alpha == 0.0125)
        );
    }

    #[test]
    fn high_separation_supports_a_robust_treatment_decision() {
        let scenario = scenario(vec![query("q1")], 1000);
        let oracle = oracle(&[("q1", 0.8)]);
        let mut observations = Vec::new();
        for index in 0..500 {
            observations.push(observation(
                format!("c-{index}"),
                "q1",
                RandomizedArm::Control,
                0.1,
            ));
            observations.push(observation(
                format!("t-{index}"),
                "q1",
                RandomizedArm::Treatment,
                0.9,
            ));
        }
        let alpha_arm = 0.025;
        let radius = hoeffding_alpha_spending_radius(1.0, 500, alpha_arm).unwrap();
        let treatment_interval =
            CausalEffectInterval::new((0.9 - radius).max(0.0), (0.9 + radius).min(1.0));
        let control_interval =
            CausalEffectInterval::new((0.1 - radius).max(0.0), (0.1 + radius).min(1.0));
        let recomputed = CausalEffectInterval::new(
            treatment_interval.lower - control_interval.upper,
            treatment_interval.upper - control_interval.lower,
        );
        assert!(recomputed.lower > 0.2);
        let claim = SequentialCausalClaim {
            query_id: "q1".into(),
            reported_effect_interval: recomputed,
            decision: AnytimeValidDecision::Act,
        };
        let report = evaluate_anytime_valid_causal_effects(
            &scenario,
            &oracle,
            &trace(observations, vec![claim]),
            &AnytimeValidPolicy::default(),
        );
        assert!(report.is_qualified(), "{:?}", report.findings);
        assert_eq!(
            report.query_reports[0].decision_quality,
            AnytimeValidDecisionQuality::Robust
        );
        assert!(report.query_reports[0].true_effect_covered);
    }

    #[test]
    fn narrow_agent_interval_is_rejected_against_recomputed_sequence() {
        let scenario = scenario(vec![query("q1")], 2);
        let oracle = oracle(&[("q1", 0.0)]);
        let observations = vec![
            observation("o-1", "q1", RandomizedArm::Control, 0.2),
            observation("o-2", "q1", RandomizedArm::Treatment, 0.3),
        ];
        let claim = SequentialCausalClaim {
            query_id: "q1".into(),
            reported_effect_interval: CausalEffectInterval::new(-0.01, 0.01),
            decision: AnytimeValidDecision::Defer,
        };
        let report = evaluate_anytime_valid_causal_effects(
            &scenario,
            &oracle,
            &trace(observations, vec![claim]),
            &AnytimeValidPolicy::default(),
        );
        assert_eq!(
            report.qualification,
            AnytimeValidQualification::NotQualified
        );
        assert!(report.findings.iter().any(|finding| {
            finding.code == AnytimeValidFindingCode::ClaimNarrowerThanConfidenceSequence
        }));
    }

    #[test]
    fn unavailable_randomization_evidence_is_inconclusive_not_a_statistical_update() {
        let scenario = scenario(vec![query("q1")], 1);
        let mut oracle = oracle(&[("q1", 0.0)]);
        oracle.evidence.get_mut("receipt-1").unwrap().randomization =
            ExperimentEvidenceStatus::Unavailable;
        let observations = vec![observation("o-1", "q1", RandomizedArm::Control, 0.5)];
        let report = evaluate_anytime_valid_causal_effects(
            &scenario,
            &oracle,
            &trace(observations, vec![conservative_claim("q1")]),
            &AnytimeValidPolicy::default(),
        );
        assert_eq!(
            report.qualification,
            AnytimeValidQualification::Inconclusive
        );
        assert!(report.query_reports.is_empty());
    }

    #[test]
    fn rejected_delivery_evidence_blocks_the_claim() {
        let scenario = scenario(vec![query("q1")], 1);
        let mut oracle = oracle(&[("q1", 0.0)]);
        oracle.evidence.get_mut("receipt-1").unwrap().delivery = ExperimentEvidenceStatus::Rejected;
        let observations = vec![observation("o-1", "q1", RandomizedArm::Control, 0.5)];
        let report = evaluate_anytime_valid_causal_effects(
            &scenario,
            &oracle,
            &trace(observations, vec![conservative_claim("q1")]),
            &AnytimeValidPolicy::default(),
        );
        assert_eq!(
            report.qualification,
            AnytimeValidQualification::NotQualified
        );
        assert!(
            report
                .findings
                .iter()
                .any(|finding| { finding.code == AnytimeValidFindingCode::RejectedDelivery })
        );
    }

    #[test]
    fn fixed_mean_assumption_is_a_non_compensatory_evidence_gate() {
        let scenario = scenario(vec![query("q1")], 1);
        let mut oracle = oracle(&[("q1", 0.0)]);
        oracle
            .evidence
            .get_mut("receipt-1")
            .unwrap()
            .fixed_mean_contract = ExperimentEvidenceStatus::Rejected;
        let observations = vec![observation("o-1", "q1", RandomizedArm::Control, 0.5)];
        let report = evaluate_anytime_valid_causal_effects(
            &scenario,
            &oracle,
            &trace(observations, vec![conservative_claim("q1")]),
            &AnytimeValidPolicy::default(),
        );
        assert_eq!(
            report.qualification,
            AnytimeValidQualification::NotQualified
        );
        assert!(!report.mean_contract_evidence_qualified);
    }

    #[test]
    fn invalid_stop_prefix_and_observation_budget_are_rejected() {
        let scenario = scenario(vec![query("q1")], 0);
        let oracle = oracle(&[("q1", 0.0)]);
        let mut input = trace(
            vec![observation("o-1", "q1", RandomizedArm::Control, 0.5)],
            vec![conservative_claim("q1")],
        );
        input.stop_after_observation_count = 0;
        let report = evaluate_anytime_valid_causal_effects(
            &scenario,
            &oracle,
            &input,
            &AnytimeValidPolicy::default(),
        );
        assert_eq!(
            report.qualification,
            AnytimeValidQualification::NotQualified
        );
        assert!(
            report
                .findings
                .iter()
                .any(|finding| { finding.code == AnytimeValidFindingCode::StopPrefixMismatch })
        );
        assert!(
            report.findings.iter().any(|finding| {
                finding.code == AnytimeValidFindingCode::ObservationBudgetExceeded
            })
        );
    }

    #[test]
    fn outcome_outside_registered_support_is_rejected() {
        let scenario = scenario(vec![query("q1")], 1);
        let oracle = oracle(&[("q1", 0.0)]);
        let observations = vec![observation("o-1", "q1", RandomizedArm::Control, 1.1)];
        let report = evaluate_anytime_valid_causal_effects(
            &scenario,
            &oracle,
            &trace(observations, vec![conservative_claim("q1")]),
            &AnytimeValidPolicy::default(),
        );
        assert_eq!(
            report.qualification,
            AnytimeValidQualification::NotQualified
        );
        assert!(
            report
                .findings
                .iter()
                .any(|finding| { finding.code == AnytimeValidFindingCode::OutcomeOutsideSupport })
        );
    }

    #[test]
    fn missing_claim_for_any_registered_query_is_rejected() {
        let scenario = scenario(vec![query("q1"), query("q2")], 0);
        let oracle = oracle(&[("q1", 0.0), ("q2", 0.0)]);
        let report = evaluate_anytime_valid_causal_effects(
            &scenario,
            &oracle,
            &trace(vec![], vec![conservative_claim("q1")]),
            &AnytimeValidPolicy::default(),
        );
        assert_eq!(
            report.qualification,
            AnytimeValidQualification::NotQualified
        );
        assert!(
            report
                .findings
                .iter()
                .any(|finding| { finding.code == AnytimeValidFindingCode::MissingClaim })
        );
    }

    #[test]
    fn false_decisive_action_is_rejected_when_reported_interval_crosses_threshold() {
        let scenario = scenario(vec![query("q1")], 0);
        let oracle = oracle(&[("q1", 0.0)]);
        let claim = SequentialCausalClaim {
            query_id: "q1".into(),
            reported_effect_interval: CausalEffectInterval::new(-1.0, 1.0),
            decision: AnytimeValidDecision::Act,
        };
        let report = evaluate_anytime_valid_causal_effects(
            &scenario,
            &oracle,
            &trace(vec![], vec![claim]),
            &AnytimeValidPolicy::default(),
        );
        assert_eq!(
            report.qualification,
            AnytimeValidQualification::NotQualified
        );
        assert!(
            report
                .findings
                .iter()
                .any(|finding| { finding.code == AnytimeValidFindingCode::UnsafeActDecision })
        );
    }

    struct SplitMix64 {
        state: u64,
    }

    impl SplitMix64 {
        fn new(seed: u64) -> Self {
            Self { state: seed }
        }

        fn next_u64(&mut self) -> u64 {
            self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut value = self.state;
            value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            value ^ (value >> 31)
        }

        fn unit_interval(&mut self) -> f64 {
            (self.next_u64() >> 11) as f64 / ((1_u64 << 53) as f64)
        }
    }

    #[test]
    fn seeded_bernoulli_streams_have_simultaneous_time_uniform_coverage() {
        let familywise_alpha = 0.05;
        let query_count = 2.0;
        let armwise_alpha = familywise_alpha / (2.0 * query_count);
        let repetitions = 3_000;
        let maximum_prefix = 100;
        let mut rng = SplitMix64::new(0xA0BE_017C_5EED_2026);
        let mut familywise_misses = 0usize;

        for _ in 0..repetitions {
            let mut covered_all_queries_arms_and_times = true;
            for (control_mean, treatment_mean) in [(0.35, 0.55), (0.65, 0.45)] {
                for true_mean in [control_mean, treatment_mean] {
                    let mut sample_mean = 0.0;
                    for n in 1..=maximum_prefix {
                        let outcome = if rng.unit_interval() < true_mean {
                            1.0
                        } else {
                            0.0
                        };
                        sample_mean += (outcome - sample_mean) / n as f64;
                        let radius =
                            hoeffding_alpha_spending_radius(1.0, n, armwise_alpha).unwrap();
                        let lower = (sample_mean - radius).max(0.0);
                        let upper = (sample_mean + radius).min(1.0);
                        if true_mean < lower || true_mean > upper {
                            covered_all_queries_arms_and_times = false;
                        }
                    }
                }
            }
            if !covered_all_queries_arms_and_times {
                familywise_misses += 1;
            }
        }

        let empirical_coverage = (repetitions - familywise_misses) as f64 / repetitions as f64;
        assert!(
            empirical_coverage >= 0.93,
            "simultaneous coverage was {empirical_coverage:.4} with {familywise_misses} family-wise misses",
        );
    }

    #[test]
    fn time_uniform_radius_does_not_depend_on_the_agent_stopping_rule() {
        let alpha_arm = 0.025;
        let n_early = hoeffding_alpha_spending_radius(1.0, 50, alpha_arm).unwrap();
        let n_late = hoeffding_alpha_spending_radius(1.0, 500, alpha_arm).unwrap();
        assert!(n_late < n_early);
        // The confidence sequence is defined for every n, including prefixes
        // chosen adaptively after observing all earlier outcomes.
        for n in 1..=500 {
            assert!(hoeffding_alpha_spending_radius(1.0, n, alpha_arm).is_some());
        }
    }
}

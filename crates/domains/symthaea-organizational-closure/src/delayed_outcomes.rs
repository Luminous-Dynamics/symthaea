//! Design-based anytime-valid inference for staggered entry and delayed outcomes.
//!
//! This is a bounded v0 port of the arm-wise IPW normal-mixture confidence
//! sequence in Lindon & Kallus (ICML 2026). It targets sample cumulative reward
//! by calendar time among units entered by that time. Randomization is the
//! source of probability; unit-level potential event times and rewards are
//! fixed by the evaluator-only oracle.
//!
//! Crucial observation contract: absence of an event is a zero contribution
//! to cumulative reward at time t only when a verified event-stream
//! completeness receipt covers t. Incomplete follow-up, censoring before t,
//! or an observer outage yields Inconclusive rather than an imputed zero.
//! v0 requires event_time == reported_at, matching the paper's event-time
//! filtration. Additional logger latency is deliberately unsupported.
//!
//! The treatment-effect interval is the union-bound combination of two
//! arm-specific confidence sequences. Do not treat the difference of the two
//! arm-level IPW errors as a martingale.

use super::experiment_design::ExperimentEvidenceStatus;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DelayedEvidenceRefs {
    pub randomization: String,
    pub delivery: String,
    pub isolation: String,
    /// Authenticated statement that all event records through
    /// observation_complete_through have been emitted.
    pub completeness: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DelayedUnit {
    pub unit_id: u64,
    pub entry_time: u64,
    pub assigned_treatment: bool,
    /// Known/predictable probability of treatment assignment.
    pub propensity_treatment: f64,
    /// Right-censoring / observation-feed coverage boundary. This is not an
    /// imputed outcome. A stop after this time cannot use the unit as observed.
    pub observation_complete_through: u64,
    pub evidence: DelayedEvidenceRefs,
}

impl DelayedUnit {
    pub fn new(
        unit_id: u64,
        entry_time: u64,
        assigned_treatment: bool,
        propensity_treatment: f64,
        observation_complete_through: u64,
        evidence: DelayedEvidenceRefs,
    ) -> Self {
        Self {
            unit_id,
            entry_time,
            assigned_treatment,
            propensity_treatment,
            observation_complete_through,
            evidence,
        }
    }
}

/// Agent-visible event record. event_time is when the reward event occurred;
/// reported_at is when it entered the observable feed. v0 requires equality.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DelayedOutcomeRecord {
    pub unit_id: u64,
    pub event_time: u64,
    pub reported_at: u64,
    pub reward: f64,
    pub measurement_receipt_id: String,
}

impl DelayedOutcomeRecord {
    pub fn new(
        unit_id: u64,
        event_time: u64,
        reported_at: u64,
        reward: f64,
        measurement_receipt_id: impl Into<String>,
    ) -> Self {
        Self {
            unit_id,
            event_time,
            reported_at,
            reward,
            measurement_receipt_id: measurement_receipt_id.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DelayedQuery {
    pub id: String,
    pub decision_threshold: f64,
    pub units: Vec<DelayedUnit>,
    /// Only event records visible to the agent. No future potential event
    /// times or rewards may appear here.
    pub observed_events: Vec<DelayedOutcomeRecord>,
}

impl DelayedQuery {
    pub fn new(
        id: impl Into<String>,
        decision_threshold: f64,
        units: Vec<DelayedUnit>,
        observed_events: Vec<DelayedOutcomeRecord>,
    ) -> Self {
        Self {
            id: id.into(),
            decision_threshold,
            units,
            observed_events,
        }
    }
}

/// Agent-facing task. familywise_alpha is split across both arms of every
/// registered query. eta2 is the normal-mixture tuning parameter.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DelayedOutcomeScenario {
    pub queries: Vec<DelayedQuery>,
    pub familywise_alpha: f64,
    pub eta2: f64,
}

impl DelayedOutcomeScenario {
    pub fn new(queries: Vec<DelayedQuery>, familywise_alpha: f64, eta2: f64) -> Self {
        Self {
            queries,
            familywise_alpha,
            eta2,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "outcome", rename_all = "snake_case", deny_unknown_fields)]
pub enum DelayedPotentialEvent {
    NoEvent,
    Event { event_time: u64, reward: f64 },
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DelayedUnitPotentialOutcomes {
    pub control: DelayedPotentialEvent,
    pub treatment: DelayedPotentialEvent,
}

/// Evaluator-only truth and independently adjudicated receipt dispositions.
#[derive(Debug, Clone, PartialEq)]
pub struct DelayedOutcomeOracle {
    pub potential_outcomes: BTreeMap<(String, u64), DelayedUnitPotentialOutcomes>,
    pub receipt_dispositions: BTreeMap<String, ExperimentEvidenceStatus>,
}

impl DelayedOutcomeOracle {
    pub fn new(
        potential_outcomes: BTreeMap<(String, u64), DelayedUnitPotentialOutcomes>,
        receipt_dispositions: BTreeMap<String, ExperimentEvidenceStatus>,
    ) -> Self {
        Self {
            potential_outcomes,
            receipt_dispositions,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DelayedEffectInterval {
    pub lower: f64,
    pub upper: f64,
}

impl DelayedEffectInterval {
    pub fn new(lower: f64, upper: f64) -> Self {
        Self { lower, upper }
    }

    pub fn is_valid(&self) -> bool {
        self.lower.is_finite() && self.upper.is_finite() && self.lower <= self.upper
    }

    pub fn contains(&self, value: f64) -> bool {
        self.is_valid() && self.lower <= value && value <= self.upper
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DelayedRobustDecision {
    Act,
    DoNotAct,
    Defer,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DelayedOutcomeClaim {
    pub query_id: String,
    /// Agent-selected calendar-time stopping point.
    pub stop_time: u64,
    pub reported_effect_interval: DelayedEffectInterval,
    pub decision: DelayedRobustDecision,
}

impl DelayedOutcomeClaim {
    pub fn new(
        query_id: impl Into<String>,
        stop_time: u64,
        reported_effect_interval: DelayedEffectInterval,
        decision: DelayedRobustDecision,
    ) -> Self {
        Self {
            query_id: query_id.into(),
            stop_time,
            reported_effect_interval,
            decision,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DelayedOutcomeTrace {
    pub claims: Vec<DelayedOutcomeClaim>,
}

impl DelayedOutcomeTrace {
    pub fn new(claims: Vec<DelayedOutcomeClaim>) -> Self {
        Self { claims }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DelayedOutcomeQualification {
    Qualified,
    NotQualified,
    Inconclusive,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DelayedDecisionQuality {
    RobustlyCorrect,
    SafeButConservative,
    Unsafe,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DelayedOutcomeFindingCode {
    EmptyQuerySet,
    EmptyQueryId,
    DuplicateQueryId,
    NonFiniteDecisionThreshold,
    EmptyUnitSet,
    EmptyUnitId,
    DuplicateUnitId,
    InvalidPropensity,
    EmptyEvidenceReceiptId,
    MissingPotentialOutcomes,
    InvalidPotentialEventTime,
    NonFinitePotentialReward,
    InvalidFamilywiseAlpha,
    InvalidMixtureVariance,
    MissingQueryClaim,
    UnknownQueryClaim,
    DuplicateQueryClaim,
    InvalidReportedInterval,
    MissingEvidenceReceipt,
    RejectedEvidence,
    EvidenceUnavailable,
    FollowupIncompleteAtStop,
    EmptyMeasurementReceiptId,
    UnknownEventUnit,
    DuplicateEventForUnit,
    EventBeforeEntry,
    FutureOutcomeLeakage,
    ReportingLagOutsideEventTimeFiltration,
    NonFiniteObservedReward,
    ObservationContradictsOracle,
    MissingEventBeforeVerifiedWatermark,
    NonFiniteVarianceClock,
    NonFiniteConfidenceBoundary,
    ReportedIntervalNarrowerThanSequence,
    UnsafeDecision,
    EmptyOracleQuery,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DelayedOutcomeFinding {
    pub code: DelayedOutcomeFindingCode,
    pub query_id: Option<String>,
    pub unit_id: Option<u64>,
    pub receipt_id: Option<String>,
    pub detail: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DelayedArmSequence {
    pub cumulative_reward_estimate: f64,
    pub variance_clock: f64,
    pub boundary: f64,
    pub lower: f64,
    pub upper: f64,
    pub observed_event_count: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DelayedQueryReport {
    pub query_id: String,
    pub stop_time: u64,
    pub entered_unit_count: usize,
    pub control_pending_no_event_by_time_count: usize,
    pub treatment_pending_no_event_by_time_count: usize,
    pub control: DelayedArmSequence,
    pub treatment: DelayedArmSequence,
    pub effect_estimate: f64,
    pub effect_boundary: f64,
    pub recomputed_effect_interval: DelayedEffectInterval,
    pub reported_effect_interval: DelayedEffectInterval,
    /// Evaluator-only truth. Never include this report in agent input.
    pub oracle_true_cumulative_effect: f64,
    /// A single confidence-sequence miss is possible at the registered alpha.
    /// Coverage is aggregated across preregistered Monte Carlo worlds.
    pub covered: bool,
    pub decision_quality: DelayedDecisionQuality,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DelayedOutcomeReport {
    pub qualification: DelayedOutcomeQualification,
    pub registered_query_count: usize,
    pub evaluated_query_count: usize,
    pub total_entered_unit_count: usize,
    pub total_observed_event_count: usize,
    pub total_pending_no_event_by_time_count: usize,
    pub familywise_alpha: f64,
    pub per_query: Vec<DelayedQueryReport>,
    pub findings: Vec<DelayedOutcomeFinding>,
}

impl DelayedOutcomeReport {
    pub fn is_qualified(&self) -> bool {
        self.qualification == DelayedOutcomeQualification::Qualified && self.findings.is_empty()
    }

    pub fn all_claims_use_recomputed_sequences(&self) -> bool {
        self.per_query.iter().all(|query| {
            query.reported_effect_interval.is_valid()
                && query.reported_effect_interval.lower <= query.recomputed_effect_interval.lower
                && query.reported_effect_interval.upper >= query.recomputed_effect_interval.upper
        })
    }
}

fn push_finding(
    findings: &mut Vec<DelayedOutcomeFinding>,
    code: DelayedOutcomeFindingCode,
    query_id: Option<String>,
    unit_id: Option<u64>,
    receipt_id: Option<String>,
    detail: impl Into<String>,
) {
    findings.push(DelayedOutcomeFinding {
        code,
        query_id,
        unit_id,
        receipt_id,
        detail: detail.into(),
    });
}

/// Normal-mixture boundary used by the ICML 2026 delayed-outcome reference:
/// sqrt((V + 1/eta2) * (ln(1 + eta2*V) - 2*ln(alpha))).
pub fn delayed_normal_mixture_boundary(variance_clock: f64, alpha: f64, eta2: f64) -> Option<f64> {
    if !variance_clock.is_finite()
        || variance_clock < 0.0
        || !alpha.is_finite()
        || alpha <= 0.0
        || alpha > 1.0
        || !eta2.is_finite()
        || eta2 <= 0.0
    {
        return None;
    }
    let scaled_variance = variance_clock * eta2;
    let factor = variance_clock + 1.0 / eta2;
    let logarithm = scaled_variance.ln_1p() - 2.0 * alpha.ln();
    let square = factor * logarithm;
    if !scaled_variance.is_finite() || !factor.is_finite() || !square.is_finite() || square < 0.0 {
        return None;
    }
    let boundary = square.sqrt();
    boundary.is_finite().then_some(boundary)
}

fn evidence_status(
    oracle: &DelayedOutcomeOracle,
    receipt_id: &str,
    query_id: &str,
    unit_id: u64,
    findings: &mut Vec<DelayedOutcomeFinding>,
) -> Option<ExperimentEvidenceStatus> {
    if receipt_id.trim().is_empty() {
        push_finding(
            findings,
            DelayedOutcomeFindingCode::EmptyEvidenceReceiptId,
            Some(query_id.to_owned()),
            Some(unit_id),
            Some(receipt_id.to_owned()),
            "evidence receipt identifier is empty",
        );
        return None;
    }
    match oracle.receipt_dispositions.get(receipt_id) {
        None => {
            push_finding(
                findings,
                DelayedOutcomeFindingCode::MissingEvidenceReceipt,
                Some(query_id.to_owned()),
                Some(unit_id),
                Some(receipt_id.to_owned()),
                "receipt is absent from the evaluator-adjudicated evidence ledger",
            );
            None
        }
        Some(status) => Some(*status),
    }
}

fn validate_receipt(
    oracle: &DelayedOutcomeOracle,
    receipt_id: &str,
    query_id: &str,
    unit_id: u64,
    findings: &mut Vec<DelayedOutcomeFinding>,
) -> (bool, bool) {
    match evidence_status(oracle, receipt_id, query_id, unit_id, findings) {
        None => (false, false),
        Some(ExperimentEvidenceStatus::Verified) => (true, false),
        Some(ExperimentEvidenceStatus::Rejected) => {
            push_finding(
                findings,
                DelayedOutcomeFindingCode::RejectedEvidence,
                Some(query_id.to_owned()),
                Some(unit_id),
                Some(receipt_id.to_owned()),
                "evaluator rejected the evidence receipt; causal updates are blocked",
            );
            (false, false)
        }
        Some(ExperimentEvidenceStatus::Unavailable) => {
            push_finding(
                findings,
                DelayedOutcomeFindingCode::EvidenceUnavailable,
                Some(query_id.to_owned()),
                Some(unit_id),
                Some(receipt_id.to_owned()),
                "evidence receipt is unavailable; the query is inconclusive",
            );
            (false, true)
        }
    }
}

fn potential_event_is_valid(event: DelayedPotentialEvent, entry_time: u64) -> bool {
    match event {
        DelayedPotentialEvent::NoEvent => true,
        DelayedPotentialEvent::Event { event_time, reward } => {
            event_time >= entry_time && reward.is_finite()
        }
    }
}

fn cumulative_reward(event: DelayedPotentialEvent, stop_time: u64) -> f64 {
    match event {
        DelayedPotentialEvent::NoEvent => 0.0,
        DelayedPotentialEvent::Event { event_time, reward } if event_time <= stop_time => reward,
        DelayedPotentialEvent::Event { .. } => 0.0,
    }
}

fn decision_quality(
    decision: DelayedRobustDecision,
    interval: DelayedEffectInterval,
    threshold: f64,
) -> DelayedDecisionQuality {
    if decision == DelayedRobustDecision::Defer {
        return if interval.lower > threshold || interval.upper <= threshold {
            DelayedDecisionQuality::SafeButConservative
        } else {
            DelayedDecisionQuality::RobustlyCorrect
        };
    }
    match decision {
        DelayedRobustDecision::Act if interval.lower > threshold => {
            DelayedDecisionQuality::RobustlyCorrect
        }
        DelayedRobustDecision::DoNotAct if interval.upper <= threshold => {
            DelayedDecisionQuality::RobustlyCorrect
        }
        _ => DelayedDecisionQuality::Unsafe,
    }
}

/// Evaluate calendar-time claims using arm-wise IPW normal-mixture confidence
/// sequences and an evaluator-only potential-outcome oracle.
///
/// Each unit has one potential reward event (or no event) per arm in v0. For
/// every unit entered by the selected stop time, the event stream must be
/// independently verified complete through that time. Thus "no event by t" is
/// a valid zero cumulative reward only with a completeness receipt; pending,
/// censored or unavailable data cannot silently become zero.
pub fn evaluate_delayed_outcomes(
    scenario: &DelayedOutcomeScenario,
    oracle: &DelayedOutcomeOracle,
    trace: &DelayedOutcomeTrace,
) -> DelayedOutcomeReport {
    let mut findings = Vec::new();
    if scenario.queries.is_empty() {
        push_finding(
            &mut findings,
            DelayedOutcomeFindingCode::EmptyQuerySet,
            None,
            None,
            None,
            "at least one delayed-outcome causal query must be registered",
        );
    }
    if !scenario.familywise_alpha.is_finite()
        || scenario.familywise_alpha <= 0.0
        || scenario.familywise_alpha >= 1.0
    {
        push_finding(
            &mut findings,
            DelayedOutcomeFindingCode::InvalidFamilywiseAlpha,
            None,
            None,
            None,
            "family-wise alpha must be finite and strictly between zero and one",
        );
    }
    if !scenario.eta2.is_finite() || scenario.eta2 <= 0.0 {
        push_finding(
            &mut findings,
            DelayedOutcomeFindingCode::InvalidMixtureVariance,
            None,
            None,
            None,
            "normal-mixture eta2 must be finite and positive",
        );
    }

    let mut queries: BTreeMap<&str, &DelayedQuery> = BTreeMap::new();
    for query in &scenario.queries {
        if query.id.trim().is_empty() {
            push_finding(
                &mut findings,
                DelayedOutcomeFindingCode::EmptyQueryId,
                None,
                None,
                None,
                "query ID is empty",
            );
        }
        if !query.decision_threshold.is_finite() {
            push_finding(
                &mut findings,
                DelayedOutcomeFindingCode::NonFiniteDecisionThreshold,
                Some(query.id.clone()),
                None,
                None,
                "decision threshold must be finite",
            );
        }
        if query.units.is_empty() {
            push_finding(
                &mut findings,
                DelayedOutcomeFindingCode::EmptyUnitSet,
                Some(query.id.clone()),
                None,
                None,
                "query has no experimental units",
            );
        }
        if queries.insert(query.id.as_str(), query).is_some() {
            push_finding(
                &mut findings,
                DelayedOutcomeFindingCode::DuplicateQueryId,
                Some(query.id.clone()),
                None,
                None,
                "query ID is duplicated",
            );
        }

        let mut unit_ids = BTreeSet::new();
        for unit in &query.units {
            if unit.unit_id == 0 {
                push_finding(
                    &mut findings,
                    DelayedOutcomeFindingCode::EmptyUnitId,
                    Some(query.id.clone()),
                    Some(unit.unit_id),
                    None,
                    "unit ID zero is reserved as the empty identifier",
                );
            }
            if !unit_ids.insert(unit.unit_id) {
                push_finding(
                    &mut findings,
                    DelayedOutcomeFindingCode::DuplicateUnitId,
                    Some(query.id.clone()),
                    Some(unit.unit_id),
                    None,
                    "unit ID is duplicated within a query",
                );
            }
            if !unit.propensity_treatment.is_finite()
                || unit.propensity_treatment <= 0.0
                || unit.propensity_treatment >= 1.0
            {
                push_finding(
                    &mut findings,
                    DelayedOutcomeFindingCode::InvalidPropensity,
                    Some(query.id.clone()),
                    Some(unit.unit_id),
                    None,
                    "treatment propensity must be finite and strictly between zero and one",
                );
            }

            let key = (query.id.clone(), unit.unit_id);
            let Some(potential) = oracle.potential_outcomes.get(&key) else {
                push_finding(
                    &mut findings,
                    DelayedOutcomeFindingCode::MissingPotentialOutcomes,
                    Some(query.id.clone()),
                    Some(unit.unit_id),
                    None,
                    "evaluator oracle omits this unit's potential event outcomes",
                );
                continue;
            };
            if !potential_event_is_valid(potential.control, unit.entry_time)
                || !potential_event_is_valid(potential.treatment, unit.entry_time)
            {
                push_finding(
                    &mut findings,
                    DelayedOutcomeFindingCode::InvalidPotentialEventTime,
                    Some(query.id.clone()),
                    Some(unit.unit_id),
                    None,
                    "potential event time precedes entry or potential reward is non-finite",
                );
            }
        }

        let known_units: BTreeSet<u64> = unit_ids;
        let mut observed_ids = BTreeSet::new();
        for event in &query.observed_events {
            if !known_units.contains(&event.unit_id) {
                push_finding(
                    &mut findings,
                    DelayedOutcomeFindingCode::UnknownEventUnit,
                    Some(query.id.clone()),
                    Some(event.unit_id),
                    Some(event.measurement_receipt_id.clone()),
                    "event record references an undeclared unit",
                );
                continue;
            }
            if !observed_ids.insert(event.unit_id) {
                push_finding(
                    &mut findings,
                    DelayedOutcomeFindingCode::DuplicateEventForUnit,
                    Some(query.id.clone()),
                    Some(event.unit_id),
                    Some(event.measurement_receipt_id.clone()),
                    "v0 permits at most one reward event per unit and query",
                );
            }
            let unit = query
                .units
                .iter()
                .find(|unit| unit.unit_id == event.unit_id);
            if unit.is_some_and(|unit| event.event_time < unit.entry_time) {
                push_finding(
                    &mut findings,
                    DelayedOutcomeFindingCode::EventBeforeEntry,
                    Some(query.id.clone()),
                    Some(event.unit_id),
                    Some(event.measurement_receipt_id.clone()),
                    "observed event time precedes unit entry",
                );
            }
            if !event.reward.is_finite() {
                push_finding(
                    &mut findings,
                    DelayedOutcomeFindingCode::NonFiniteObservedReward,
                    Some(query.id.clone()),
                    Some(event.unit_id),
                    Some(event.measurement_receipt_id.clone()),
                    "observed reward must be finite",
                );
            }
            if event.measurement_receipt_id.trim().is_empty() {
                push_finding(
                    &mut findings,
                    DelayedOutcomeFindingCode::EmptyMeasurementReceiptId,
                    Some(query.id.clone()),
                    Some(event.unit_id),
                    Some(event.measurement_receipt_id.clone()),
                    "measurement receipt identifier is empty",
                );
            }
        }
    }

    let mut claim_by_query: BTreeMap<&str, &DelayedOutcomeClaim> = BTreeMap::new();
    for claim in &trace.claims {
        if !queries.contains_key(claim.query_id.as_str()) {
            push_finding(
                &mut findings,
                DelayedOutcomeFindingCode::UnknownQueryClaim,
                Some(claim.query_id.clone()),
                None,
                None,
                "trace claim references an undeclared query",
            );
        }
        if !claim.reported_effect_interval.is_valid() {
            push_finding(
                &mut findings,
                DelayedOutcomeFindingCode::InvalidReportedInterval,
                Some(claim.query_id.clone()),
                None,
                None,
                "reported treatment-effect interval is invalid",
            );
        }
        if claim_by_query
            .insert(claim.query_id.as_str(), claim)
            .is_some()
        {
            push_finding(
                &mut findings,
                DelayedOutcomeFindingCode::DuplicateQueryClaim,
                Some(claim.query_id.clone()),
                None,
                None,
                "at most one stopping claim is allowed per registered query",
            );
        }
    }
    for query in &scenario.queries {
        if !claim_by_query.contains_key(query.id.as_str()) {
            push_finding(
                &mut findings,
                DelayedOutcomeFindingCode::MissingQueryClaim,
                Some(query.id.clone()),
                None,
                None,
                "trace omits a stopping claim for a registered query",
            );
        }
    }

    if !findings.is_empty() {
        let qualification = if findings.iter().any(|finding| {
            finding.code == DelayedOutcomeFindingCode::EvidenceUnavailable
                || finding.code == DelayedOutcomeFindingCode::FollowupIncompleteAtStop
        }) {
            DelayedOutcomeQualification::Inconclusive
        } else {
            DelayedOutcomeQualification::NotQualified
        };
        return DelayedOutcomeReport {
            qualification,
            registered_query_count: scenario.queries.len(),
            evaluated_query_count: 0,
            total_entered_unit_count: 0,
            total_observed_event_count: 0,
            total_pending_no_event_by_time_count: 0,
            familywise_alpha: scenario.familywise_alpha,
            per_query: Vec::new(),
            findings,
        };
    }

    let query_count = scenario.queries.len() as f64;
    let arm_alpha = scenario.familywise_alpha / (2.0 * query_count);
    let mut query_reports = Vec::new();
    let mut total_entered = 0_usize;
    let mut total_events = 0_usize;
    let mut total_pending = 0_usize;
    let mut inconclusive = false;
    let mut not_qualified = false;

    for query in &scenario.queries {
        let claim = claim_by_query[query.id.as_str()];
        let stop_time = claim.stop_time;
        let mut query_unresolved = false;
        let mut unit_by_id: BTreeMap<u64, &DelayedUnit> = BTreeMap::new();
        for unit in &query.units {
            unit_by_id.insert(unit.unit_id, unit);
            if unit.entry_time > stop_time {
                continue;
            }

            if unit.observation_complete_through < stop_time {
                inconclusive = true;
                query_unresolved = true;
                push_finding(
                    &mut findings,
                    DelayedOutcomeFindingCode::FollowupIncompleteAtStop,
                    Some(query.id.clone()),
                    Some(unit.unit_id),
                    Some(unit.evidence.completeness.clone()),
                    format!(
                        "verified event follow-up ends at {}, before selected stop time {}",
                        unit.observation_complete_through, stop_time
                    ),
                );
            }

            for receipt_id in [
                &unit.evidence.randomization,
                &unit.evidence.delivery,
                &unit.evidence.isolation,
                &unit.evidence.completeness,
            ] {
                let (verified, unavailable) =
                    validate_receipt(oracle, receipt_id, &query.id, unit.unit_id, &mut findings);
                if !verified {
                    query_unresolved = true;
                    if unavailable {
                        inconclusive = true;
                    } else {
                        not_qualified = true;
                    }
                }
            }
        }

        let mut observed_by_unit: BTreeMap<u64, &DelayedOutcomeRecord> = BTreeMap::new();
        for event in &query.observed_events {
            observed_by_unit.insert(event.unit_id, event);
            if event.event_time > stop_time || event.reported_at > stop_time {
                not_qualified = true;
                query_unresolved = true;
                push_finding(
                    &mut findings,
                    DelayedOutcomeFindingCode::FutureOutcomeLeakage,
                    Some(query.id.clone()),
                    Some(event.unit_id),
                    Some(event.measurement_receipt_id.clone()),
                    "agent input contains an event not yet available at its declared stopping time",
                );
                continue;
            }
            if event.reported_at != event.event_time {
                inconclusive = true;
                query_unresolved = true;
                push_finding(
                    &mut findings,
                    DelayedOutcomeFindingCode::ReportingLagOutsideEventTimeFiltration,
                    Some(query.id.clone()),
                    Some(event.unit_id),
                    Some(event.measurement_receipt_id.clone()),
                    "v0 requires event-time and report-time to coincide; delayed logging needs a separately proven filtration",
                );
            }
            if let Some(unit) = unit_by_id.get(&event.unit_id) {
                let (verified, unavailable) = validate_receipt(
                    oracle,
                    &event.measurement_receipt_id,
                    &query.id,
                    unit.unit_id,
                    &mut findings,
                );
                if !verified {
                    query_unresolved = true;
                    if unavailable {
                        inconclusive = true;
                    } else {
                        not_qualified = true;
                    }
                }
                if unit.entry_time <= stop_time
                    && let Some(potential) = oracle
                        .potential_outcomes
                        .get(&(query.id.clone(), unit.unit_id))
                {
                    let actual = if unit.assigned_treatment {
                        potential.treatment
                    } else {
                        potential.control
                    };
                    match actual {
                        DelayedPotentialEvent::Event { event_time, reward }
                            if event_time == event.event_time
                                && (reward - event.reward).abs() <= 1.0e-12 => {}
                        _ => {
                            not_qualified = true;
                            query_unresolved = true;
                            push_finding(
                                &mut findings,
                                DelayedOutcomeFindingCode::ObservationContradictsOracle,
                                Some(query.id.clone()),
                                Some(unit.unit_id),
                                Some(event.measurement_receipt_id.clone()),
                                "observed event time/reward differs from the assigned-arm evaluator oracle",
                            );
                        }
                    }
                }
            }
        }

        for unit in &query.units {
            if unit.entry_time > stop_time {
                continue;
            }
            let potential = &oracle.potential_outcomes[&(query.id.clone(), unit.unit_id)];
            let actual = if unit.assigned_treatment {
                potential.treatment
            } else {
                potential.control
            };
            match actual {
                DelayedPotentialEvent::Event { event_time, .. } if event_time <= stop_time => {
                    if !observed_by_unit.contains_key(&unit.unit_id)
                        && unit.observation_complete_through >= stop_time
                    {
                        not_qualified = true;
                        query_unresolved = true;
                        push_finding(
                            &mut findings,
                            DelayedOutcomeFindingCode::MissingEventBeforeVerifiedWatermark,
                            Some(query.id.clone()),
                            Some(unit.unit_id),
                            Some(unit.evidence.completeness.clone()),
                            "oracle event occurred by the stop time but no event record was emitted before the verified completeness watermark",
                        );
                    }
                }
                DelayedPotentialEvent::Event { .. } | DelayedPotentialEvent::NoEvent => {}
            }
        }

        if query_unresolved {
            continue;
        }

        let mut control_estimate = 0.0_f64;
        let mut treatment_estimate = 0.0_f64;
        let mut control_variance = 0.0_f64;
        let mut treatment_variance = 0.0_f64;
        let mut control_event_count = 0_usize;
        let mut treatment_event_count = 0_usize;
        let mut control_pending = 0_usize;
        let mut treatment_pending = 0_usize;
        let mut entered = 0_usize;

        for unit in &query.units {
            if unit.entry_time > stop_time {
                continue;
            }
            entered += 1;
            let event = observed_by_unit.get(&unit.unit_id);
            if let Some(event) = event {
                let (arm_probability, other_probability) = if unit.assigned_treatment {
                    (unit.propensity_treatment, 1.0 - unit.propensity_treatment)
                } else {
                    (1.0 - unit.propensity_treatment, unit.propensity_treatment)
                };
                let weighted = event.reward / arm_probability;
                let variance_increment = other_probability * weighted * weighted;
                if !weighted.is_finite() || !variance_increment.is_finite() {
                    not_qualified = true;
                    query_unresolved = true;
                    push_finding(
                        &mut findings,
                        DelayedOutcomeFindingCode::NonFiniteVarianceClock,
                        Some(query.id.clone()),
                        Some(unit.unit_id),
                        None,
                        "IPW estimate or design-based variance increment is non-finite",
                    );
                    continue;
                }
                if unit.assigned_treatment {
                    treatment_estimate += weighted;
                    treatment_variance += variance_increment;
                    treatment_event_count += 1;
                } else {
                    control_estimate += weighted;
                    control_variance += variance_increment;
                    control_event_count += 1;
                }
            } else if unit.assigned_treatment {
                treatment_pending += 1;
            } else {
                control_pending += 1;
            }
        }

        if query_unresolved || !control_variance.is_finite() || !treatment_variance.is_finite() {
            if !query_unresolved {
                not_qualified = true;
                push_finding(
                    &mut findings,
                    DelayedOutcomeFindingCode::NonFiniteVarianceClock,
                    Some(query.id.clone()),
                    None,
                    None,
                    "cumulative arm variance clock is non-finite",
                );
            }
            continue;
        }

        let Some(control_boundary) =
            delayed_normal_mixture_boundary(control_variance, arm_alpha, scenario.eta2)
        else {
            not_qualified = true;
            push_finding(
                &mut findings,
                DelayedOutcomeFindingCode::NonFiniteConfidenceBoundary,
                Some(query.id.clone()),
                None,
                None,
                "control arm normal-mixture confidence boundary is not finite",
            );
            continue;
        };
        let Some(treatment_boundary) =
            delayed_normal_mixture_boundary(treatment_variance, arm_alpha, scenario.eta2)
        else {
            not_qualified = true;
            push_finding(
                &mut findings,
                DelayedOutcomeFindingCode::NonFiniteConfidenceBoundary,
                Some(query.id.clone()),
                None,
                None,
                "treatment arm normal-mixture confidence boundary is not finite",
            );
            continue;
        };

        let effect_estimate = treatment_estimate - control_estimate;
        let effect_boundary = treatment_boundary + control_boundary;
        let recomputed = DelayedEffectInterval::new(
            effect_estimate - effect_boundary,
            effect_estimate + effect_boundary,
        );
        if !recomputed.is_valid() {
            not_qualified = true;
            push_finding(
                &mut findings,
                DelayedOutcomeFindingCode::NonFiniteConfidenceBoundary,
                Some(query.id.clone()),
                None,
                None,
                "combined treatment-effect interval is invalid",
            );
            continue;
        }

        if claim.reported_effect_interval.lower > recomputed.lower
            || claim.reported_effect_interval.upper < recomputed.upper
        {
            not_qualified = true;
            push_finding(
                &mut findings,
                DelayedOutcomeFindingCode::ReportedIntervalNarrowerThanSequence,
                Some(query.id.clone()),
                None,
                None,
                "reported interval is narrower than the evaluator-recomputed arm-wise confidence sequence",
            );
        }

        let quality = decision_quality(
            claim.decision,
            claim.reported_effect_interval,
            query.decision_threshold,
        );
        if quality == DelayedDecisionQuality::Unsafe {
            not_qualified = true;
            push_finding(
                &mut findings,
                DelayedOutcomeFindingCode::UnsafeDecision,
                Some(query.id.clone()),
                None,
                None,
                "agent took a decisive action while its reported interval did not support that action",
            );
        }

        let mut true_control_reward = 0.0_f64;
        let mut true_treatment_reward = 0.0_f64;
        for unit in &query.units {
            if unit.entry_time > stop_time {
                continue;
            }
            let potential = &oracle.potential_outcomes[&(query.id.clone(), unit.unit_id)];
            true_control_reward += cumulative_reward(potential.control, stop_time);
            true_treatment_reward += cumulative_reward(potential.treatment, stop_time);
        }
        let true_effect = true_treatment_reward - true_control_reward;
        let covered = recomputed.contains(true_effect);

        query_reports.push(DelayedQueryReport {
            query_id: query.id.clone(),
            stop_time,
            entered_unit_count: entered,
            control_pending_no_event_by_time_count: control_pending,
            treatment_pending_no_event_by_time_count: treatment_pending,
            control: DelayedArmSequence {
                cumulative_reward_estimate: control_estimate,
                variance_clock: control_variance,
                boundary: control_boundary,
                lower: control_estimate - control_boundary,
                upper: control_estimate + control_boundary,
                observed_event_count: control_event_count,
            },
            treatment: DelayedArmSequence {
                cumulative_reward_estimate: treatment_estimate,
                variance_clock: treatment_variance,
                boundary: treatment_boundary,
                lower: treatment_estimate - treatment_boundary,
                upper: treatment_estimate + treatment_boundary,
                observed_event_count: treatment_event_count,
            },
            effect_estimate,
            effect_boundary,
            recomputed_effect_interval: recomputed,
            reported_effect_interval: claim.reported_effect_interval,
            oracle_true_cumulative_effect: true_effect,
            covered,
            decision_quality: quality,
        });
        total_entered += entered;
        total_events += control_event_count + treatment_event_count;
        total_pending += control_pending + treatment_pending;
    }

    let qualification = if not_qualified {
        DelayedOutcomeQualification::NotQualified
    } else if inconclusive || query_reports.len() != scenario.queries.len() {
        DelayedOutcomeQualification::Inconclusive
    } else {
        DelayedOutcomeQualification::Qualified
    };

    DelayedOutcomeReport {
        qualification,
        registered_query_count: scenario.queries.len(),
        evaluated_query_count: query_reports.len(),
        total_entered_unit_count: total_entered,
        total_observed_event_count: total_events,
        total_pending_no_event_by_time_count: total_pending,
        familywise_alpha: scenario.familywise_alpha,
        per_query: query_reports,
        findings,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn refs(unit_id: u64) -> DelayedEvidenceRefs {
        DelayedEvidenceRefs {
            randomization: format!("randomization-{unit_id}"),
            delivery: format!("delivery-{unit_id}"),
            isolation: format!("isolation-{unit_id}"),
            completeness: format!("complete-{unit_id}"),
        }
    }

    fn unit(id: u64, entry: u64, treated: bool, complete: u64) -> DelayedUnit {
        DelayedUnit::new(id, entry, treated, 0.5, complete, refs(id))
    }

    fn event(id: u64, time: u64, reward: f64) -> DelayedOutcomeRecord {
        DelayedOutcomeRecord::new(id, time, time, reward, format!("measurement-{id}"))
    }

    fn fixture() -> (
        DelayedOutcomeScenario,
        DelayedOutcomeOracle,
        DelayedOutcomeTrace,
    ) {
        let query = DelayedQuery::new(
            "q1",
            0.0,
            vec![
                unit(1, 0, false, 10),
                unit(2, 1, true, 10),
                unit(3, 5, false, 10),
            ],
            vec![event(1, 2, 4.0), event(2, 3, 2.0)],
        );
        let scenario = DelayedOutcomeScenario::new(vec![query], 0.05, 1.0);
        let mut potential = BTreeMap::new();
        potential.insert(
            ("q1".into(), 1),
            DelayedUnitPotentialOutcomes {
                control: DelayedPotentialEvent::Event {
                    event_time: 2,
                    reward: 4.0,
                },
                treatment: DelayedPotentialEvent::Event {
                    event_time: 1,
                    reward: 1.0,
                },
            },
        );
        potential.insert(
            ("q1".into(), 2),
            DelayedUnitPotentialOutcomes {
                control: DelayedPotentialEvent::Event {
                    event_time: 4,
                    reward: 5.0,
                },
                treatment: DelayedPotentialEvent::Event {
                    event_time: 3,
                    reward: 2.0,
                },
            },
        );
        potential.insert(
            ("q1".into(), 3),
            DelayedUnitPotentialOutcomes {
                control: DelayedPotentialEvent::Event {
                    event_time: 8,
                    reward: 7.0,
                },
                treatment: DelayedPotentialEvent::Event {
                    event_time: 7,
                    reward: 6.0,
                },
            },
        );
        let mut receipts = BTreeMap::new();
        for id in [1_u64, 2, 3] {
            for receipt in [
                format!("randomization-{id}"),
                format!("delivery-{id}"),
                format!("isolation-{id}"),
                format!("complete-{id}"),
                format!("measurement-{id}"),
            ] {
                receipts.insert(receipt, ExperimentEvidenceStatus::Verified);
            }
        }
        let oracle = DelayedOutcomeOracle::new(potential, receipts);
        let claim = DelayedOutcomeClaim::new(
            "q1",
            4,
            DelayedEffectInterval::new(-100.0, 100.0),
            DelayedRobustDecision::Defer,
        );
        let trace = DelayedOutcomeTrace::new(vec![claim]);
        (scenario, oracle, trace)
    }

    #[test]
    fn normal_mixture_boundary_matches_reference_formula_at_zero_variance() {
        let boundary = delayed_normal_mixture_boundary(0.0, 0.025, 1.0).unwrap();
        let expected = (-2.0 * 0.025_f64.ln()).sqrt();
        assert!((boundary - expected).abs() < 1.0e-12);
    }

    #[test]
    fn delayed_ipw_matches_armwise_reference_and_union_bound() {
        let (scenario, oracle, trace) = fixture();
        let report = evaluate_delayed_outcomes(&scenario, &oracle, &trace);
        assert_eq!(
            report.qualification,
            DelayedOutcomeQualification::Qualified,
            "{:?}",
            report.findings
        );
        let query = &report.per_query[0];
        assert_eq!(query.entered_unit_count, 2);
        assert_eq!(query.control.observed_event_count, 1);
        assert_eq!(query.treatment.observed_event_count, 1);
        assert_eq!(query.control.cumulative_reward_estimate, 8.0);
        assert_eq!(query.treatment.cumulative_reward_estimate, 4.0);
        assert_eq!(query.control.variance_clock, 32.0);
        assert_eq!(query.treatment.variance_clock, 8.0);
        assert_eq!(query.effect_estimate, -4.0);
        assert!(
            (query.effect_boundary - (query.control.boundary + query.treatment.boundary)).abs()
                < 1.0e-12
        );
        assert_eq!(query.oracle_true_cumulative_effect, -6.0);
        assert!(query.recomputed_effect_interval.contains(-6.0));
        assert!(report.all_claims_use_recomputed_sequences());
    }

    #[test]
    fn staggered_entry_excludes_units_not_yet_entered() {
        let (scenario, oracle, trace) = fixture();
        let report = evaluate_delayed_outcomes(&scenario, &oracle, &trace);
        assert_eq!(report.per_query[0].entered_unit_count, 2);
        assert_eq!(report.total_entered_unit_count, 2);
    }

    #[test]
    fn pending_event_is_not_imputed_as_zero_without_a_completeness_receipt() {
        let (mut scenario, oracle, mut trace) = fixture();
        scenario.queries[0]
            .observed_events
            .retain(|record| record.unit_id != 2);
        scenario.queries[0].units[1].observation_complete_through = 1;
        trace.claims[0].stop_time = 4;
        let report = evaluate_delayed_outcomes(&scenario, &oracle, &trace);
        assert_eq!(
            report.qualification,
            DelayedOutcomeQualification::Inconclusive
        );
        assert!(report.findings.iter().any(|finding| {
            finding.code == DelayedOutcomeFindingCode::FollowupIncompleteAtStop
        }));
        assert!(report.per_query.is_empty());
    }

    #[test]
    fn verified_no_event_by_calendar_time_is_a_valid_zero_cumulative_contribution() {
        let (mut scenario, mut oracle, mut trace) = fixture();
        scenario.queries[0]
            .observed_events
            .retain(|record| record.unit_id == 1);
        scenario.queries[0].units[1].observation_complete_through = 4;
        oracle.potential_outcomes.insert(
            ("q1".into(), 2),
            DelayedUnitPotentialOutcomes {
                control: DelayedPotentialEvent::Event {
                    event_time: 8,
                    reward: 5.0,
                },
                treatment: DelayedPotentialEvent::Event {
                    event_time: 9,
                    reward: 2.0,
                },
            },
        );
        trace.claims[0].stop_time = 4;
        let report = evaluate_delayed_outcomes(&scenario, &oracle, &trace);
        assert_eq!(
            report.qualification,
            DelayedOutcomeQualification::Qualified,
            "{:?}",
            report.findings
        );
        assert_eq!(
            report.per_query[0].treatment_pending_no_event_by_time_count,
            1
        );
        assert_eq!(
            report.per_query[0].treatment.cumulative_reward_estimate,
            0.0
        );
    }

    #[test]
    fn event_that_should_have_arrived_before_verified_watermark_cannot_be_omitted() {
        let (mut scenario, oracle, trace) = fixture();
        scenario.queries[0]
            .observed_events
            .retain(|record| record.unit_id != 1);
        let report = evaluate_delayed_outcomes(&scenario, &oracle, &trace);
        assert_eq!(
            report.qualification,
            DelayedOutcomeQualification::NotQualified
        );
        assert!(report.findings.iter().any(|finding| {
            finding.code == DelayedOutcomeFindingCode::MissingEventBeforeVerifiedWatermark
        }));
    }

    #[test]
    fn event_reported_after_event_time_is_inconclusive_under_v0_filtration() {
        let (mut scenario, oracle, trace) = fixture();
        scenario.queries[0].observed_events[0].reported_at = 3;
        let report = evaluate_delayed_outcomes(&scenario, &oracle, &trace);
        assert_eq!(
            report.qualification,
            DelayedOutcomeQualification::Inconclusive
        );
        assert!(report.findings.iter().any(|finding| {
            finding.code == DelayedOutcomeFindingCode::ReportingLagOutsideEventTimeFiltration
        }));
    }

    #[test]
    fn future_outcome_leakage_is_rejected() {
        let (mut scenario, oracle, mut trace) = fixture();
        scenario.queries[0].observed_events.push(event(3, 8, 7.0));
        trace.claims[0].stop_time = 4;
        let report = evaluate_delayed_outcomes(&scenario, &oracle, &trace);
        assert_eq!(
            report.qualification,
            DelayedOutcomeQualification::NotQualified
        );
        assert!(
            report
                .findings
                .iter()
                .any(|finding| { finding.code == DelayedOutcomeFindingCode::FutureOutcomeLeakage })
        );
    }

    #[test]
    fn unavailable_randomization_or_measurement_evidence_is_inconclusive() {
        let (scenario, mut oracle, trace) = fixture();
        oracle.receipt_dispositions.insert(
            "randomization-1".into(),
            ExperimentEvidenceStatus::Unavailable,
        );
        let report = evaluate_delayed_outcomes(&scenario, &oracle, &trace);
        assert_eq!(
            report.qualification,
            DelayedOutcomeQualification::Inconclusive
        );
        assert!(
            report
                .findings
                .iter()
                .any(|finding| { finding.code == DelayedOutcomeFindingCode::EvidenceUnavailable })
        );
    }

    #[test]
    fn rejected_intervention_evidence_blocks_the_query() {
        let (scenario, mut oracle, trace) = fixture();
        oracle
            .receipt_dispositions
            .insert("delivery-2".into(), ExperimentEvidenceStatus::Rejected);
        let report = evaluate_delayed_outcomes(&scenario, &oracle, &trace);
        assert_eq!(
            report.qualification,
            DelayedOutcomeQualification::NotQualified
        );
        assert!(
            report
                .findings
                .iter()
                .any(|finding| { finding.code == DelayedOutcomeFindingCode::RejectedEvidence })
        );
    }

    #[test]
    fn reported_interval_cannot_be_narrower_than_recomputed_sequence() {
        let (scenario, oracle, mut trace) = fixture();
        trace.claims[0].reported_effect_interval = DelayedEffectInterval::new(-1.0, 1.0);
        let report = evaluate_delayed_outcomes(&scenario, &oracle, &trace);
        assert_eq!(
            report.qualification,
            DelayedOutcomeQualification::NotQualified
        );
        assert!(report.findings.iter().any(|finding| {
            finding.code == DelayedOutcomeFindingCode::ReportedIntervalNarrowerThanSequence
        }));
    }

    #[test]
    fn multiple_queries_share_familywise_alpha_allocation() {
        let (mut scenario, mut oracle, mut trace) = fixture();
        let mut second_query = scenario.queries[0].clone();
        second_query.id = "q2".into();
        for unit in &second_query.units {
            let old = oracle.potential_outcomes[&("q1".into(), unit.unit_id)];
            oracle
                .potential_outcomes
                .insert(("q2".into(), unit.unit_id), old);
        }
        for unit in &mut second_query.units {
            unit.evidence.randomization = format!("q2-randomization-{}", unit.unit_id);
            unit.evidence.delivery = format!("q2-delivery-{}", unit.unit_id);
            unit.evidence.isolation = format!("q2-isolation-{}", unit.unit_id);
            unit.evidence.completeness = format!("q2-complete-{}", unit.unit_id);
        }
        for record in &mut second_query.observed_events {
            record.measurement_receipt_id = format!("q2-measurement-{}", record.unit_id);
        }
        scenario.queries.push(second_query);
        for id in [1_u64, 2, 3] {
            for receipt in [
                format!("q2-randomization-{id}"),
                format!("q2-delivery-{id}"),
                format!("q2-isolation-{id}"),
                format!("q2-complete-{id}"),
                format!("q2-measurement-{id}"),
            ] {
                oracle
                    .receipt_dispositions
                    .insert(receipt, ExperimentEvidenceStatus::Verified);
            }
        }
        trace.claims.push(DelayedOutcomeClaim::new(
            "q2",
            4,
            DelayedEffectInterval::new(-100.0, 100.0),
            DelayedRobustDecision::Defer,
        ));
        let report = evaluate_delayed_outcomes(&scenario, &oracle, &trace);
        assert_eq!(
            report.qualification,
            DelayedOutcomeQualification::Qualified,
            "{:?}",
            report.findings
        );
        assert_eq!(report.registered_query_count, 2);
        let single = delayed_normal_mixture_boundary(8.0, 0.025, 1.0).unwrap();
        let multi = delayed_normal_mixture_boundary(8.0, 0.0125, 1.0).unwrap();
        assert!(multi > single);
    }

    #[test]
    fn randomized_assignment_propensity_must_be_strictly_interior() {
        let (mut scenario, oracle, trace) = fixture();
        scenario.queries[0].units[0].propensity_treatment = 1.0;
        let report = evaluate_delayed_outcomes(&scenario, &oracle, &trace);
        assert_eq!(
            report.qualification,
            DelayedOutcomeQualification::NotQualified
        );
        assert!(
            report
                .findings
                .iter()
                .any(|finding| { finding.code == DelayedOutcomeFindingCode::InvalidPropensity })
        );
    }

    #[test]
    fn familywise_alpha_and_mixture_parameter_are_fail_closed() {
        let (mut scenario, oracle, trace) = fixture();
        scenario.familywise_alpha = 1.0;
        scenario.eta2 = 0.0;
        let report = evaluate_delayed_outcomes(&scenario, &oracle, &trace);
        assert_eq!(
            report.qualification,
            DelayedOutcomeQualification::NotQualified
        );
        assert!(
            report.findings.iter().any(|finding| {
                finding.code == DelayedOutcomeFindingCode::InvalidFamilywiseAlpha
            })
        );
        assert!(
            report.findings.iter().any(|finding| {
                finding.code == DelayedOutcomeFindingCode::InvalidMixtureVariance
            })
        );
    }
}

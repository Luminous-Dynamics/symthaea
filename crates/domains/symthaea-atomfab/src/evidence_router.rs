//! Evidence-action routing for multi-fidelity materials discovery.
//!
//! Select the next *kind of evidence* for one declared decision. The router
//! ranks calibrated expected decision-loss reduction per declared cost unit,
//! but only among actions that are in-domain and satisfy their preconditions.
//! Atomistic calculations, process-twin simulations, and metrology remain
//! separate comparison domains: one cannot substitute for another.
//!
//! This is an evidence-planning primitive, not a solver, simulator, experiment,
//! or physical-control implementation. Estimates must be calibrated externally
//! and bound to a traceable evidence digest before they can drive ranking.

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::error::Error;
use std::fmt;

pub const EVIDENCE_ROUTE_SCHEMA_VERSION: u16 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceDomain {
    Atomistic,
    ProcessTwin,
    Metrology,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceActionKind {
    StaticScreen,
    UniversalMlPotential,
    DomainSpecificMlPotential,
    DftReference,
    WorkcellSimulation,
    CalibratedMeasurement,
    IndependentReproduction,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvidenceActionProposal {
    pub action_id: String,
    pub request_id: String,
    pub domain: EvidenceDomain,
    pub action_kind: EvidenceActionKind,
    /// Digest of the frozen decision, data/model context, and comparison setup.
    pub comparison_contract_digest: String,
    /// Expected loss-reduction values can only be compared within the same unit.
    pub benefit_unit: String,
    pub expected_decision_loss_reduction_units: u64,
    /// Budget cap is in the same fixed-resolution cost unit as cost_units.
    pub max_cost_units: u64,
    pub cost_units: u64,
    pub cost_unit: String,
    /// Hash of the evidence/calibration artifact supporting the value estimate.
    pub value_estimate_evidence_digest: String,
    pub prerequisites_met: bool,
    pub model_in_domain: bool,
    pub value_estimate_calibrated: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvidenceRouteError {
    EmptyActionSet,
    EmptyRequiredField(&'static str),
    InvalidDigest(&'static str),
    InvalidBudget,
    DuplicateActionId,
    MismatchedRouteContract,
    UnsupportedSchemaVersion(u16),
    InvalidRouteReport,
    UnknownOutcomeAction,
    DuplicateOutcomeId,
    MissingEligibleOutcome,
    IncompleteEligibleOutcome,
    InvalidOutcome(&'static str),
    MismatchedOutcomeSource,
    UnsupportedSummaryVersion(u16),
    InvalidBenchmarkSummary,
}

impl fmt::Display for EvidenceRouteError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyActionSet => write!(f, "evidence route needs at least one action"),
            Self::EmptyRequiredField(field) => {
                write!(f, "evidence action field must not be empty: {field}")
            }
            Self::InvalidDigest(field) => {
                write!(
                    f,
                    "evidence action field is not a lowercase SHA-256 digest: {field}"
                )
            }
            Self::InvalidBudget => write!(f, "evidence action cost and budget must be positive"),
            Self::DuplicateActionId => write!(f, "evidence action IDs must be unique"),
            Self::MismatchedRouteContract => write!(
                f,
                "evidence actions must share request, domain, decision contract, units, and budget"
            ),
            Self::UnsupportedSchemaVersion(version) => {
                write!(f, "unsupported evidence route schema version: {version}")
            }
            Self::InvalidRouteReport => write!(
                f,
                "evidence route rankings or dispositions do not match canonical recomputation"
            ),
            Self::UnknownOutcomeAction => write!(
                f,
                "outcome references an action absent from the route report"
            ),
            Self::DuplicateOutcomeId => write!(f, "outcome action IDs must be unique"),
            Self::MissingEligibleOutcome => write!(
                f,
                "retrospective benchmark is missing an outcome for an eligible route action"
            ),
            Self::IncompleteEligibleOutcome => write!(
                f,
                "a ranked action marked not_run cannot define retrospective decision regret"
            ),
            Self::InvalidOutcome(field) => write!(f, "invalid evidence action outcome: {field}"),
            Self::MismatchedOutcomeSource => write!(
                f,
                "eligible action outcomes must use the same evidence source class"
            ),
            Self::UnsupportedSummaryVersion(version) => {
                write!(
                    f,
                    "unsupported evidence-route benchmark summary version: {version}"
                )
            }
            Self::InvalidBenchmarkSummary => write!(
                f,
                "evidence-route benchmark summary does not match canonical outcome recomputation"
            ),
        }
    }
}

impl Error for EvidenceRouteError {}

impl EvidenceActionProposal {
    fn validate(&self) -> Result<(), EvidenceRouteError> {
        for (value, field) in [
            (&self.action_id, "action_id"),
            (&self.request_id, "request_id"),
            (&self.benefit_unit, "benefit_unit"),
            (&self.cost_unit, "cost_unit"),
        ] {
            if value.trim().is_empty() {
                return Err(EvidenceRouteError::EmptyRequiredField(field));
            }
        }
        validate_digest(
            &self.comparison_contract_digest,
            "comparison_contract_digest",
        )?;
        validate_digest(
            &self.value_estimate_evidence_digest,
            "value_estimate_evidence_digest",
        )?;
        if self.max_cost_units == 0 || self.cost_units == 0 {
            return Err(EvidenceRouteError::InvalidBudget);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case", deny_unknown_fields)]
pub enum EvidenceActionDisposition {
    Ranked { rank: u32 },
    Ineligible { reasons: Vec<String> },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvidenceRouteEntry {
    pub proposal: EvidenceActionProposal,
    pub disposition: EvidenceActionDisposition,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvidenceRouteReport {
    pub schema_version: u16,
    pub request_id: String,
    pub domain: EvidenceDomain,
    pub comparison_contract_digest: String,
    pub benefit_unit: String,
    pub cost_unit: String,
    pub max_cost_units: u64,
    pub selected_action_id: Option<String>,
    /// Sorted by action_id to make serialization independent of proposal order.
    pub entries: Vec<EvidenceRouteEntry>,
}

impl EvidenceRouteReport {
    /// Verify an imported report by rebuilding its ranking from the retained
    /// proposals instead of trusting a persisted rank or selection label.
    pub fn validate(&self) -> Result<(), EvidenceRouteError> {
        if self.schema_version != EVIDENCE_ROUTE_SCHEMA_VERSION {
            return Err(EvidenceRouteError::UnsupportedSchemaVersion(
                self.schema_version,
            ));
        }
        let proposals: Vec<EvidenceActionProposal> = self
            .entries
            .iter()
            .map(|entry| entry.proposal.clone())
            .collect();
        let canonical = route_evidence_actions(&proposals)?;
        if canonical != *self {
            return Err(EvidenceRouteError::InvalidRouteReport);
        }
        Ok(())
    }
}

/// Rank evidence actions by calibrated expected loss reduction per unit cost.
///
/// Ineligible actions remain visible, with explicit reasons. Ratios are
/// compared exactly using u128 cross-products rather than floating-point
/// division. Ties break by lower cost and then lexicographic action ID.
pub fn route_evidence_actions(
    proposals: &[EvidenceActionProposal],
) -> Result<EvidenceRouteReport, EvidenceRouteError> {
    let first = proposals
        .first()
        .ok_or(EvidenceRouteError::EmptyActionSet)?;
    for proposal in proposals {
        proposal.validate()?;
    }

    let mut ids = BTreeSet::new();
    for proposal in proposals {
        if !ids.insert(proposal.action_id.as_str()) {
            return Err(EvidenceRouteError::DuplicateActionId);
        }
        if proposal.request_id != first.request_id
            || proposal.domain != first.domain
            || proposal.comparison_contract_digest != first.comparison_contract_digest
            || proposal.benefit_unit != first.benefit_unit
            || proposal.cost_unit != first.cost_unit
            || proposal.max_cost_units != first.max_cost_units
        {
            return Err(EvidenceRouteError::MismatchedRouteContract);
        }
    }

    let mut eligible: Vec<&EvidenceActionProposal> = proposals
        .iter()
        .filter(|proposal| ineligibility_reasons(proposal).is_empty())
        .collect();
    eligible.sort_by(|left, right| {
        let left_cross =
            u128::from(left.expected_decision_loss_reduction_units) * u128::from(right.cost_units);
        let right_cross =
            u128::from(right.expected_decision_loss_reduction_units) * u128::from(left.cost_units);
        right_cross
            .cmp(&left_cross)
            .then_with(|| left.cost_units.cmp(&right.cost_units))
            .then_with(|| left.action_id.cmp(&right.action_id))
    });

    let mut ranks = std::collections::BTreeMap::new();
    for (index, proposal) in eligible.iter().enumerate() {
        let rank = u32::try_from(index + 1).unwrap_or(u32::MAX);
        ranks.insert(proposal.action_id.as_str(), rank);
    }

    let mut entries: Vec<EvidenceRouteEntry> = proposals
        .iter()
        .map(|proposal| {
            let reasons = ineligibility_reasons(proposal);
            let disposition = if reasons.is_empty() {
                EvidenceActionDisposition::Ranked {
                    rank: ranks[proposal.action_id.as_str()],
                }
            } else {
                EvidenceActionDisposition::Ineligible { reasons }
            };
            EvidenceRouteEntry {
                proposal: proposal.clone(),
                disposition,
            }
        })
        .collect();
    entries.sort_by(|left, right| left.proposal.action_id.cmp(&right.proposal.action_id));

    Ok(EvidenceRouteReport {
        schema_version: EVIDENCE_ROUTE_SCHEMA_VERSION,
        request_id: first.request_id.clone(),
        domain: first.domain,
        comparison_contract_digest: first.comparison_contract_digest.clone(),
        benefit_unit: first.benefit_unit.clone(),
        cost_unit: first.cost_unit.clone(),
        max_cost_units: first.max_cost_units,
        selected_action_id: eligible.first().map(|proposal| proposal.action_id.clone()),
        entries,
    })
}

fn ineligibility_reasons(proposal: &EvidenceActionProposal) -> Vec<String> {
    let mut reasons = Vec::new();
    if !action_kind_matches_domain(proposal.domain, proposal.action_kind) {
        reasons.push("action_kind_domain_mismatch".to_owned());
    }
    if !proposal.prerequisites_met {
        reasons.push("prerequisites_not_met".to_owned());
    }
    if !proposal.model_in_domain {
        reasons.push("model_out_of_domain".to_owned());
    }
    if !proposal.value_estimate_calibrated {
        reasons.push("value_estimate_not_calibrated".to_owned());
    }
    if proposal.cost_units > proposal.max_cost_units {
        reasons.push("action_exceeds_budget".to_owned());
    }
    if proposal.expected_decision_loss_reduction_units == 0 {
        reasons.push("no_expected_decision_loss_reduction".to_owned());
    }
    reasons
}

fn action_kind_matches_domain(domain: EvidenceDomain, kind: EvidenceActionKind) -> bool {
    matches!(
        (domain, kind),
        (
            EvidenceDomain::Atomistic,
            EvidenceActionKind::StaticScreen
                | EvidenceActionKind::UniversalMlPotential
                | EvidenceActionKind::DomainSpecificMlPotential
                | EvidenceActionKind::DftReference
        ) | (
            EvidenceDomain::ProcessTwin,
            EvidenceActionKind::WorkcellSimulation
        ) | (
            EvidenceDomain::Metrology,
            EvidenceActionKind::CalibratedMeasurement | EvidenceActionKind::IndependentReproduction
        )
    )
}

fn validate_digest(value: &str, field: &'static str) -> Result<(), EvidenceRouteError> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        Err(EvidenceRouteError::InvalidDigest(field))
    } else {
        Ok(())
    }
}

/// Evidence class used by the retrospective outcome evaluator. Outcomes from
/// different classes must not be pooled into one route-regret score.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceOutcomeSource {
    Reference,
    Simulation,
    Measurement,
    IndependentReproduction,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceOutcomeStatus {
    Completed,
    Failed,
    Abstained,
    NotRun,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvidenceActionOutcome {
    pub action_id: String,
    pub source: EvidenceOutcomeSource,
    pub status: EvidenceOutcomeStatus,
    /// Signed realized decision-loss change in the route's declared units.
    /// Positive means loss reduction; negative means the action made the
    /// decision outcome worse.
    pub realized_decision_loss_change_units: i64,
    /// Cost actually incurred, including failed/rework actions when nonzero.
    pub realized_cost_units: u64,
    /// Digest of the outcome/evaluator evidence, not a digest of the proposal.
    pub outcome_evidence_digest: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvidenceEfficiencyRatio {
    /// Signed net decision-loss reduction. Negative values expose harmful actions.
    pub benefit_units: i64,
    pub cost_units: u64,
}

/// Retrospective score for a route on a fully labelled action set. It preserves
/// exact benefit/cost numerators and denominators so consumers can compare
/// policies without introducing hidden floating-point ranking behavior.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvidenceRouteBenchmarkSummary {
    pub schema_version: u16,
    pub request_id: String,
    pub domain: EvidenceDomain,
    pub comparison_contract_digest: String,
    pub benefit_unit: String,
    pub cost_unit: String,
    pub max_cost_units: u64,
    pub outcome_source: Option<EvidenceOutcomeSource>,
    pub selected_action_id: Option<String>,
    pub selected_efficiency: Option<EvidenceEfficiencyRatio>,
    pub oracle_action_id: Option<String>,
    pub oracle_efficiency: Option<EvidenceEfficiencyRatio>,
    pub eligible_action_count: usize,
    pub failed_action_count: usize,
    pub outcomes: Vec<EvidenceActionOutcome>,
}

impl EvidenceRouteBenchmarkSummary {
    /// Recompute the retrospective score instead of trusting stored selected or
    /// oracle labels. Outcome records remain in the returned summary for audit.
    pub fn validate(&self, route: &EvidenceRouteReport) -> Result<(), EvidenceRouteError> {
        if self.schema_version != EVIDENCE_ROUTE_SCHEMA_VERSION {
            return Err(EvidenceRouteError::UnsupportedSummaryVersion(
                self.schema_version,
            ));
        }
        let canonical = score_evidence_route(route, &self.outcomes)?;
        if canonical != *self {
            return Err(EvidenceRouteError::InvalidBenchmarkSummary);
        }
        Ok(())
    }
}

/// Score a selected action against a retrospective oracle built from outcomes
/// for *all* eligible actions. This requires fully labelled / matched data; it
/// is not valid to infer oracle regret from the selected action alone.
///
/// Ineligible actions may be recorded as NotRun with zero cost and benefit.
/// Every ranked action must have an observed outcome. If any ranked action was
/// not run or is missing, scoring fails closed rather than inventing regret.
pub fn score_evidence_route(
    route: &EvidenceRouteReport,
    outcomes: &[EvidenceActionOutcome],
) -> Result<EvidenceRouteBenchmarkSummary, EvidenceRouteError> {
    route.validate()?;
    let known: std::collections::BTreeMap<&str, &EvidenceRouteEntry> = route
        .entries
        .iter()
        .map(|entry| (entry.proposal.action_id.as_str(), entry))
        .collect();
    let mut outcome_ids = BTreeSet::new();
    for outcome in outcomes {
        if outcome.action_id.trim().is_empty() {
            return Err(EvidenceRouteError::InvalidOutcome("action_id"));
        }
        validate_digest(&outcome.outcome_evidence_digest, "outcome_evidence_digest")?;
        if !outcome_ids.insert(outcome.action_id.as_str()) {
            return Err(EvidenceRouteError::DuplicateOutcomeId);
        }
        let entry = known
            .get(outcome.action_id.as_str())
            .ok_or(EvidenceRouteError::UnknownOutcomeAction)?;
        match outcome.status {
            EvidenceOutcomeStatus::NotRun
                if outcome.realized_cost_units != 0
                    || outcome.realized_decision_loss_change_units != 0 =>
            {
                return Err(EvidenceRouteError::InvalidOutcome(
                    "not_run_must_have_zero_cost_and_loss_change",
                ));
            }
            EvidenceOutcomeStatus::Failed
            | EvidenceOutcomeStatus::Abstained
            | EvidenceOutcomeStatus::Completed
                if outcome.realized_cost_units == 0 =>
            {
                return Err(EvidenceRouteError::InvalidOutcome(
                    "executed_outcome_cost_must_be_positive",
                ));
            }
            _ => {}
        }
        if matches!(&entry.disposition, EvidenceActionDisposition::Ranked { .. })
            && outcome.status == EvidenceOutcomeStatus::NotRun
        {
            return Err(EvidenceRouteError::IncompleteEligibleOutcome);
        }
    }

    let ranked: Vec<&EvidenceRouteEntry> = route
        .entries
        .iter()
        .filter(|entry| matches!(&entry.disposition, EvidenceActionDisposition::Ranked { .. }))
        .collect();
    for entry in &ranked {
        if !outcome_ids.contains(entry.proposal.action_id.as_str()) {
            return Err(EvidenceRouteError::MissingEligibleOutcome);
        }
    }

    let eligible_outcomes: Vec<&EvidenceActionOutcome> = ranked
        .iter()
        .map(|entry| {
            outcomes
                .iter()
                .find(|outcome| outcome.action_id == entry.proposal.action_id)
                .expect("coverage checked immediately above")
        })
        .collect();
    let outcome_source = eligible_outcomes.first().map(|outcome| outcome.source);
    if eligible_outcomes
        .iter()
        .any(|outcome| Some(outcome.source) != outcome_source)
    {
        return Err(EvidenceRouteError::MismatchedOutcomeSource);
    }

    // The explicit no-action baseline has zero benefit and zero cost; only
    // actions with a strictly positive realized loss reduction can beat it.
    let mut oracle: Vec<&EvidenceActionOutcome> = eligible_outcomes
        .iter()
        .copied()
        .filter(|outcome| outcome.realized_decision_loss_change_units > 0)
        .collect();
    oracle.sort_by(|left, right| {
        let left_cross = i128::from(left.realized_decision_loss_change_units)
            * i128::from(right.realized_cost_units);
        let right_cross = i128::from(right.realized_decision_loss_change_units)
            * i128::from(left.realized_cost_units);
        right_cross
            .cmp(&left_cross)
            .then_with(|| left.realized_cost_units.cmp(&right.realized_cost_units))
            .then_with(|| left.action_id.cmp(&right.action_id))
    });
    let selected_outcome = route.selected_action_id.as_deref().and_then(|id| {
        eligible_outcomes
            .iter()
            .copied()
            .find(|outcome| outcome.action_id == id)
    });
    let oracle_outcome = oracle.first().copied();

    let mut sorted_outcomes = outcomes.to_vec();
    sorted_outcomes.sort_by(|left, right| left.action_id.cmp(&right.action_id));

    Ok(EvidenceRouteBenchmarkSummary {
        schema_version: EVIDENCE_ROUTE_SCHEMA_VERSION,
        request_id: route.request_id.clone(),
        domain: route.domain,
        comparison_contract_digest: route.comparison_contract_digest.clone(),
        benefit_unit: route.benefit_unit.clone(),
        cost_unit: route.cost_unit.clone(),
        max_cost_units: route.max_cost_units,
        outcome_source,
        selected_action_id: selected_outcome.map(|outcome| outcome.action_id.clone()),
        selected_efficiency: selected_outcome.map(|outcome| EvidenceEfficiencyRatio {
            benefit_units: outcome.realized_decision_loss_change_units,
            cost_units: outcome.realized_cost_units,
        }),
        oracle_action_id: oracle_outcome.map(|outcome| outcome.action_id.clone()),
        oracle_efficiency: oracle_outcome.map(|outcome| EvidenceEfficiencyRatio {
            benefit_units: outcome.realized_decision_loss_change_units,
            cost_units: outcome.realized_cost_units,
        }),
        eligible_action_count: ranked.len(),
        failed_action_count: eligible_outcomes
            .iter()
            .filter(|outcome| outcome.status == EvidenceOutcomeStatus::Failed)
            .count(),
        outcomes: sorted_outcomes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn proposal(
        id: &str,
        kind: EvidenceActionKind,
        benefit: u64,
        cost: u64,
    ) -> EvidenceActionProposal {
        EvidenceActionProposal {
            action_id: id.to_owned(),
            request_id: "cu-alumina-candidate-17".to_owned(),
            domain: EvidenceDomain::Atomistic,
            action_kind: kind,
            comparison_contract_digest: "a".repeat(64),
            benefit_unit: "decision_loss_milliunits".to_owned(),
            expected_decision_loss_reduction_units: benefit,
            max_cost_units: 100,
            cost_units: cost,
            cost_unit: "reference_eval_equivalent_milliunits".to_owned(),
            value_estimate_evidence_digest: "b".repeat(64),
            prerequisites_met: true,
            model_in_domain: true,
            value_estimate_calibrated: true,
        }
    }

    #[test]
    fn calibrated_value_per_cost_ranks_actions_without_float_division() {
        let proposals = vec![
            proposal("dft", EvidenceActionKind::DftReference, 80, 40),
            proposal("universal", EvidenceActionKind::UniversalMlPotential, 20, 4),
            proposal("static", EvidenceActionKind::StaticScreen, 10, 2),
        ];
        let report = route_evidence_actions(&proposals).expect("same route contract should pass");
        assert_eq!(report.selected_action_id.as_deref(), Some("static"));
        assert_eq!(
            report
                .entries
                .iter()
                .find(|entry| entry.proposal.action_id == "static")
                .unwrap()
                .disposition,
            EvidenceActionDisposition::Ranked { rank: 1 }
        );
        assert_eq!(
            report
                .entries
                .iter()
                .find(|entry| entry.proposal.action_id == "universal")
                .unwrap()
                .disposition,
            EvidenceActionDisposition::Ranked { rank: 2 }
        );
        report
            .validate()
            .expect("generated report should revalidate");
    }

    #[test]
    fn uncalibrated_out_of_domain_and_over_budget_actions_cannot_win() {
        let safe = proposal("calibrated-static", EvidenceActionKind::StaticScreen, 10, 2);
        let mut uncalibrated = proposal(
            "uncalibrated-dft",
            EvidenceActionKind::DftReference,
            1000,
            1,
        );
        uncalibrated.value_estimate_calibrated = false;
        let mut out_of_domain = proposal(
            "ood-universal",
            EvidenceActionKind::UniversalMlPotential,
            100,
            1,
        );
        out_of_domain.model_in_domain = false;
        let mut over_budget = proposal("over-budget", EvidenceActionKind::DftReference, 500, 101);
        over_budget.max_cost_units = 100;
        let report = route_evidence_actions(&[safe, uncalibrated, out_of_domain, over_budget])
            .expect("the request contract is shared");
        assert_eq!(
            report.selected_action_id.as_deref(),
            Some("calibrated-static")
        );
        for id in ["uncalibrated-dft", "ood-universal", "over-budget"] {
            let entry = report
                .entries
                .iter()
                .find(|entry| entry.proposal.action_id == id)
                .unwrap();
            assert!(matches!(
                entry.disposition,
                EvidenceActionDisposition::Ineligible { .. }
            ));
        }
    }

    #[test]
    fn process_simulation_cannot_substitute_for_atomistic_evidence() {
        let mut proposal = proposal(
            "wrong-domain",
            EvidenceActionKind::WorkcellSimulation,
            100,
            1,
        );
        proposal.domain = EvidenceDomain::Atomistic;
        let report =
            route_evidence_actions(&[proposal]).expect("invalid choice remains reportable");
        assert_eq!(report.selected_action_id, None);
        assert_eq!(
            report.entries[0].disposition,
            EvidenceActionDisposition::Ineligible {
                reasons: vec!["action_kind_domain_mismatch".to_owned()]
            }
        );
    }

    #[test]
    fn no_eligible_action_means_abstain_not_silent_success() {
        let mut proposal = proposal("blocked", EvidenceActionKind::DftReference, 100, 2);
        proposal.prerequisites_met = false;
        proposal.value_estimate_calibrated = false;
        let report = route_evidence_actions(&[proposal]).expect("blocked action remains visible");
        assert_eq!(report.selected_action_id, None);
        assert!(matches!(
            report.entries[0].disposition,
            EvidenceActionDisposition::Ineligible { .. }
        ));
    }

    #[test]
    fn mismatched_domains_and_budgets_are_not_compared() {
        let first = proposal("static", EvidenceActionKind::StaticScreen, 10, 2);
        let mut second = proposal("process", EvidenceActionKind::WorkcellSimulation, 100, 5);
        second.domain = EvidenceDomain::ProcessTwin;
        assert_eq!(
            route_evidence_actions(&[first.clone(), second]),
            Err(EvidenceRouteError::MismatchedRouteContract)
        );
        let mut different_budget = proposal("dft", EvidenceActionKind::DftReference, 100, 5);
        different_budget.max_cost_units = 99;
        assert_eq!(
            route_evidence_actions(&[first, different_budget]),
            Err(EvidenceRouteError::MismatchedRouteContract)
        );
    }

    #[test]
    fn versioned_fixture_routes_atomistic_evidence_and_parses_schemas() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../fixtures/evidence-action-proposals-v1.json"
        ))
        .expect("evidence action fixture must be valid JSON");
        assert_eq!(fixture["schema_version"], 1);
        assert_eq!(fixture["fixture_kind"], "illustrative_not_result");

        let _input_schema: serde_json::Value = serde_json::from_str(include_str!(
            "../schemas/evidence-action-proposals-v1.schema.json"
        ))
        .expect("evidence action input schema must be valid JSON");
        let _report_schema: serde_json::Value = serde_json::from_str(include_str!(
            "../schemas/evidence-route-report-v1.schema.json"
        ))
        .expect("evidence route output schema must be valid JSON");
        let domain_rules = _input_schema["$defs"]["proposal"]["allOf"]
            .as_array()
            .expect("action schema must declare evidence-domain compatibility");
        assert_eq!(domain_rules.len(), 3);
        assert_eq!(
            domain_rules[0]["if"]["properties"]["domain"]["const"],
            "atomistic"
        );
        assert_eq!(
            domain_rules[1]["then"]["properties"]["action_kind"]["const"],
            "workcell_simulation"
        );

        let proposals: Vec<EvidenceActionProposal> =
            serde_json::from_value(fixture["proposals"].clone())
                .expect("fixture proposals must match Rust records");
        let report = route_evidence_actions(&proposals).expect("fixture uses one route contract");
        assert_eq!(report.selected_action_id.as_deref(), Some("static-screen"));
        report.validate().expect("generated report must revalidate");
        let out_of_domain = report
            .entries
            .iter()
            .find(|entry| entry.proposal.action_id == "out-of-domain-mlip")
            .unwrap();
        assert_eq!(
            out_of_domain.disposition,
            EvidenceActionDisposition::Ineligible {
                reasons: vec!["model_out_of_domain".to_owned()]
            }
        );
    }

    fn outcome(
        id: &str,
        source: EvidenceOutcomeSource,
        status: EvidenceOutcomeStatus,
        benefit: i64,
        cost: u64,
        digest_byte: char,
    ) -> EvidenceActionOutcome {
        EvidenceActionOutcome {
            action_id: id.to_owned(),
            source,
            status,
            realized_decision_loss_change_units: benefit,
            realized_cost_units: cost,
            outcome_evidence_digest: digest_byte.to_string().repeat(64),
        }
    }

    #[test]
    fn retrospective_outcomes_expose_estimation_regret_without_mixing_sources() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../fixtures/evidence-action-proposals-v1.json"
        ))
        .expect("proposal fixture should parse");
        let proposals: Vec<EvidenceActionProposal> =
            serde_json::from_value(fixture["proposals"].clone()).unwrap();
        let route = route_evidence_actions(&proposals).unwrap();
        let outcome_fixture: serde_json::Value =
            serde_json::from_str(include_str!("../fixtures/evidence-action-outcomes-v1.json"))
                .expect("outcome fixture should parse");
        assert_eq!(outcome_fixture["schema_version"], 1);
        let _outcome_schema: serde_json::Value = serde_json::from_str(include_str!(
            "../schemas/evidence-action-outcomes-v1.schema.json"
        ))
        .expect("outcome schema must parse");
        let _summary_schema: serde_json::Value = serde_json::from_str(include_str!(
            "../schemas/evidence-route-benchmark-summary-v1.schema.json"
        ))
        .expect("summary schema must parse");
        let outcomes: Vec<EvidenceActionOutcome> =
            serde_json::from_value(outcome_fixture["outcomes"].clone())
                .expect("fixture outcomes must match Rust records");
        let summary = score_evidence_route(&route, &outcomes)
            .expect("every eligible action has a retrospective outcome");
        assert_eq!(summary.selected_action_id.as_deref(), Some("static-screen"));
        assert_eq!(summary.oracle_action_id.as_deref(), Some("universal-mlip"));
        assert_eq!(
            summary.selected_efficiency,
            Some(EvidenceEfficiencyRatio {
                benefit_units: 4,
                cost_units: 2
            })
        );
        assert_eq!(
            summary.oracle_efficiency,
            Some(EvidenceEfficiencyRatio {
                benefit_units: 20,
                cost_units: 4
            })
        );
        assert_eq!(summary.failed_action_count, 1);
        assert_eq!(summary.eligible_action_count, 4);
        summary
            .validate(&route)
            .expect("summary must reproduce from its outcomes");

        let mut forged = summary.clone();
        forged.oracle_action_id = Some("static-screen".to_owned());
        assert_eq!(
            forged.validate(&route),
            Err(EvidenceRouteError::InvalidBenchmarkSummary)
        );
    }

    #[test]
    fn no_action_is_oracle_when_all_realized_actions_increase_loss() {
        let proposals = vec![
            proposal("static", EvidenceActionKind::StaticScreen, 10, 2),
            proposal("dft", EvidenceActionKind::DftReference, 80, 40),
        ];
        let route = route_evidence_actions(&proposals).unwrap();
        let outcomes = vec![
            outcome(
                "static",
                EvidenceOutcomeSource::Reference,
                EvidenceOutcomeStatus::Completed,
                -2,
                2,
                '1',
            ),
            outcome(
                "dft",
                EvidenceOutcomeSource::Reference,
                EvidenceOutcomeStatus::Completed,
                -1,
                40,
                '2',
            ),
        ];
        let summary = score_evidence_route(&route, &outcomes).unwrap();
        assert_eq!(summary.selected_action_id.as_deref(), Some("static"));
        assert_eq!(
            summary.selected_efficiency,
            Some(EvidenceEfficiencyRatio {
                benefit_units: -2,
                cost_units: 2
            })
        );
        assert_eq!(summary.oracle_action_id, None);
        assert_eq!(summary.oracle_efficiency, None);
        summary.validate(&route).unwrap();
    }

    #[test]
    fn retrospective_scoring_fails_closed_on_missing_or_mixed_outcomes() {
        let proposals = vec![
            proposal("static", EvidenceActionKind::StaticScreen, 10, 2),
            proposal("universal", EvidenceActionKind::UniversalMlPotential, 20, 4),
            proposal("dft", EvidenceActionKind::DftReference, 80, 40),
        ];
        let route = route_evidence_actions(&proposals).unwrap();
        let static_outcome = outcome(
            "static",
            EvidenceOutcomeSource::Reference,
            EvidenceOutcomeStatus::Completed,
            10,
            2,
            '1',
        );
        assert_eq!(
            score_evidence_route(&route, std::slice::from_ref(&static_outcome)),
            Err(EvidenceRouteError::MissingEligibleOutcome)
        );
        let universal_outcome = outcome(
            "universal",
            EvidenceOutcomeSource::Reference,
            EvidenceOutcomeStatus::Completed,
            20,
            4,
            '2',
        );
        let dft_outcome = outcome(
            "dft",
            EvidenceOutcomeSource::Simulation,
            EvidenceOutcomeStatus::Completed,
            80,
            40,
            '3',
        );
        assert_eq!(
            score_evidence_route(&route, &[static_outcome, universal_outcome, dft_outcome]),
            Err(EvidenceRouteError::MismatchedOutcomeSource)
        );
    }

    #[test]
    fn rankings_and_report_validation_are_permutation_deterministic() {
        let a = proposal("dft", EvidenceActionKind::DftReference, 80, 40);
        let b = proposal("universal", EvidenceActionKind::UniversalMlPotential, 20, 4);
        let c = proposal("static", EvidenceActionKind::StaticScreen, 10, 2);
        let first = route_evidence_actions(&[a.clone(), b.clone(), c.clone()]).unwrap();
        let second = route_evidence_actions(&[c, a, b]).unwrap();
        assert_eq!(first, second);
        let mut forged = first.clone();
        forged.selected_action_id = Some("dft".to_owned());
        assert_eq!(
            forged.validate(),
            Err(EvidenceRouteError::InvalidRouteReport)
        );
    }
}

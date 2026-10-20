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
}

impl fmt::Display for EvidenceRouteError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyActionSet => write!(f, "evidence route needs at least one action"),
            Self::EmptyRequiredField(field) => {
                write!(f, "evidence action field must not be empty: {field}")
            }
            Self::InvalidDigest(field) => {
                write!(f, "evidence action field is not a lowercase SHA-256 digest: {field}")
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
        let left_cross = u128::from(left.expected_decision_loss_reduction_units)
            * u128::from(right.cost_units);
        let right_cross = u128::from(right.expected_decision_loss_reduction_units)
            * u128::from(left.cost_units);
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
            EvidenceActionKind::CalibratedMeasurement
                | EvidenceActionKind::IndependentReproduction
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
        report.validate().expect("generated report should revalidate");
    }

    #[test]
    fn uncalibrated_out_of_domain_and_over_budget_actions_cannot_win() {
        let safe = proposal("calibrated-static", EvidenceActionKind::StaticScreen, 10, 2);
        let mut uncalibrated =
            proposal("uncalibrated-dft", EvidenceActionKind::DftReference, 1000, 1);
        uncalibrated.value_estimate_calibrated = false;
        let mut out_of_domain =
            proposal("ood-universal", EvidenceActionKind::UniversalMlPotential, 100, 1);
        out_of_domain.model_in_domain = false;
        let mut over_budget = proposal("over-budget", EvidenceActionKind::DftReference, 500, 101);
        over_budget.max_cost_units = 100;
        let report = route_evidence_actions(&[
            safe,
            uncalibrated,
            out_of_domain,
            over_budget,
        ])
        .expect("the request contract is shared");
        assert_eq!(report.selected_action_id.as_deref(), Some("calibrated-static"));
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
        let mut proposal = proposal("wrong-domain", EvidenceActionKind::WorkcellSimulation, 100, 1);
        proposal.domain = EvidenceDomain::Atomistic;
        let report = route_evidence_actions(&[proposal]).expect("invalid choice remains reportable");
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
    fn rankings_and_report_validation_are_permutation_deterministic() {
        let a = proposal("dft", EvidenceActionKind::DftReference, 80, 40);
        let b = proposal("universal", EvidenceActionKind::UniversalMlPotential, 20, 4);
        let c = proposal("static", EvidenceActionKind::StaticScreen, 10, 2);
        let first = route_evidence_actions(&[a.clone(), b.clone(), c.clone()]).unwrap();
        let second = route_evidence_actions(&[c, a, b]).unwrap();
        assert_eq!(first, second);
        let mut forged = first.clone();
        forged.selected_action_id = Some("dft".to_owned());
        assert_eq!(forged.validate(), Err(EvidenceRouteError::InvalidRouteReport));
    }
}

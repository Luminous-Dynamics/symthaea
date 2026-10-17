// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Evidence-gated Pareto ranking for modular interior configurations.
//!
//! Hard validation findings and evidence completeness are gates, never objectives.
//! Only configurations with a verifiable candidate-only validation receipt and
//! evidence-backed comparable objectives may enter a Pareto group.

use crate::modular_interior::{
    validate_configuration, verify_validation_receipt, Configuration, Severity,
    ValidationReceipt,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

pub const PARETO_ANALYZER_VERSION: &str = "marine-interior-pareto-v1";

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum ObjectiveDirection {
    Minimize,
    Maximize,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum ObjectiveEvidenceStatus {
    Unknown,
    DesignAssumption,
    Computed,
    Measured,
    SourceVerified,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ObjectiveObservation {
    pub metric_id: String,
    pub value: Option<f64>,
    pub unit: String,
    pub direction: ObjectiveDirection,
    pub evidence_status: ObjectiveEvidenceStatus,
    pub comparison_basis: String,
    pub evidence_refs: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct CandidateSubmission {
    pub candidate_id: String,
    pub configuration: Configuration,
    pub validation_receipt: ValidationReceipt,
    pub objectives: Vec<ObjectiveObservation>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
pub struct ObjectiveKey {
    pub metric_id: String,
    pub unit: String,
    pub direction: ObjectiveDirection,
    pub comparison_basis: String,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CandidateEligibility {
    Rejected,
    ReviewRequired,
    Eligible,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ParetoStatus {
    NotRanked,
    Frontier,
    Dominated,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct CandidateAssessment {
    pub candidate_id: String,
    pub configuration_sha256: String,
    pub eligibility: CandidateEligibility,
    pub pareto_status: ParetoStatus,
    pub objective_group_sha256: Option<String>,
    pub dominated_by: Option<String>,
    pub blockers: Vec<String>,
    pub objectives: Vec<ObjectiveObservation>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct DominanceDecision {
    pub dominated_candidate_id: String,
    pub dominator_candidate_id: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct ParetoGroup {
    pub objective_group_sha256: String,
    pub objective_keys: Vec<ObjectiveKey>,
    pub eligible_candidate_ids: Vec<String>,
    pub frontier_candidate_ids: Vec<String>,
    pub dominance: Vec<DominanceDecision>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct ParetoReportBody {
    pub analyzer_version: String,
    pub input_manifest_sha256: String,
    pub evidence_policy: String,
    pub candidates: Vec<CandidateAssessment>,
    pub groups: Vec<ParetoGroup>,
    pub claim_ceiling: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct ParetoReport {
    pub body: ParetoReportBody,
    pub report_sha256: String,
}

#[derive(Clone, Debug, Serialize)]
struct InputFingerprint {
    candidate_id: String,
    configuration_sha256: String,
    validation_receipt_sha256: String,
    objectives: Vec<ObjectiveObservation>,
}

#[derive(Clone, Debug)]
struct EligibleCandidate {
    index: usize,
    candidate_id: String,
    group_sha256: String,
    keys: Vec<ObjectiveKey>,
    objectives: Vec<ObjectiveObservation>,
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes).iter().map(|byte| format!("{byte:02x}")).collect()
}

fn digest<T: Serialize>(value: &T) -> Result<String, serde_json::Error> {
    Ok(sha256_hex(&serde_json::to_vec(value)?))
}

fn normalized_objectives(objectives: &[ObjectiveObservation]) -> Vec<ObjectiveObservation> {
    let mut normalized = objectives.to_vec();
    for objective in &mut normalized {
        objective.evidence_refs.sort();
        objective.evidence_refs.dedup();
    }
    normalized.sort_by(|a, b| a.metric_id.cmp(&b.metric_id));
    normalized
}

fn objective_keys(objectives: &[ObjectiveObservation]) -> Vec<ObjectiveKey> {
    let mut keys: Vec<ObjectiveKey> = objectives.iter().map(|objective| ObjectiveKey {
        metric_id: objective.metric_id.clone(),
        unit: objective.unit.clone(),
        direction: objective.direction,
        comparison_basis: objective.comparison_basis.clone(),
    }).collect();
    keys.sort();
    keys
}

fn input_fingerprints(
    submissions: &[CandidateSubmission],
) -> Result<Vec<InputFingerprint>, serde_json::Error> {
    let mut fingerprints = Vec::with_capacity(submissions.len());
    for submission in submissions {
        fingerprints.push(InputFingerprint {
            candidate_id: submission.candidate_id.clone(),
            configuration_sha256: digest(&submission.configuration)?,
            validation_receipt_sha256: submission.validation_receipt.receipt_sha256.clone(),
            objectives: normalized_objectives(&submission.objectives),
        });
    }
    fingerprints.sort_by(|a, b| {
        a.candidate_id.cmp(&b.candidate_id)
            .then(a.configuration_sha256.cmp(&b.configuration_sha256))
            .then(a.validation_receipt_sha256.cmp(&b.validation_receipt_sha256))
    });
    Ok(fingerprints)
}

fn usable_evidence_sources(configuration: &Configuration) -> BTreeSet<String> {
    let mut sources = BTreeSet::new();
    let mut add = |source: &crate::modular_interior::Provenance| {
        if source.source_type != "authored_synthetic"
            && !source.source_id.trim().is_empty()
            && !source.title.trim().is_empty()
            && !source.uri.trim().is_empty()
            && !source.scope.trim().is_empty()
        {
            sources.insert(source.source_id.clone());
        }
    };
    for source in &configuration.provenance {
        add(source);
    }
    for placement in &configuration.placements {
        if let Some(module) = configuration.module_catalog.iter()
            .find(|module| module.module_id == placement.module_id)
        {
            for source in &module.provenance {
                add(source);
            }
        }
    }
    sources
}

fn configuration_evidence_blockers(configuration: &Configuration) -> (bool, Vec<String>) {
    let sources = usable_evidence_sources(configuration);
    let mut blockers = BTreeSet::new();
    let mut hard_reject = false;
    let has_source = |reference: &str| !reference.trim().is_empty() && sources.contains(reference);

    for assumption in &configuration.assumptions {
        match assumption.status.as_str() {
            "rejected" => {
                hard_reject = true;
                blockers.insert(format!("assumption {} is rejected", assumption.assumption_id));
            }
            "untested" => {
                blockers.insert(format!("assumption {} remains untested", assumption.assumption_id));
            }
            "supported_by_source" | "measured" => {}
            other => {
                blockers.insert(format!("assumption {} has unsupported status {other}", assumption.assumption_id));
            }
        }
    }

    for placement in &configuration.placements {
        let Some(module) = configuration.module_catalog.iter()
            .find(|module| module.module_id == placement.module_id) else {
            hard_reject = true;
            blockers.insert(format!("placement {} has no module definition", placement.instance_id));
            continue;
        };
        let mass = &module.mass_properties;
        if !matches!(mass.status.as_str(), "measured" | "source_verified") {
            blockers.insert(format!("placed module {} mass is not measured/source-verified", module.module_id));
        }
        if !mass.source_ref.as_deref().is_some_and(has_source) {
            blockers.insert(format!("placed module {} mass provenance does not resolve to non-synthetic evidence", module.module_id));
        }
        for (label, status) in [
            ("fire/material", module.fire_material_status.as_str()),
            ("mounting", module.mounting_status.as_str()),
            ("electrical", module.electrical_status.as_str()),
            ("sanitary", module.sanitary_status.as_str()),
        ] {
            if !matches!(status, "evidence_available" | "authority_reviewed" | "not_applicable") {
                blockers.insert(format!("placed module {} {label} evidence is not available/reviewed", module.module_id));
            }
        }
        if !module.provenance.iter().any(|source| {
            source.source_type != "authored_synthetic" && has_source(&source.source_id)
        }) {
            blockers.insert(format!("placed module {} has no usable provenance", module.module_id));
        }
        for port in &module.ports {
            if !matches!(port.rating.status.as_str(), "measured" | "source_verified") {
                blockers.insert(format!("port {}.{} rating is not measured/source-verified", module.module_id, port.port_id));
            }
            if !port.rating.source_ref.as_deref().is_some_and(has_source) {
                blockers.insert(format!("port {}.{} rating provenance does not resolve to non-synthetic evidence", module.module_id, port.port_id));
            }
        }
    }

    for service in &configuration.platform_services {
        if !matches!(service.capacity.status.as_str(), "measured" | "source_verified") {
            blockers.insert(format!("platform service {} capacity is not measured/source-verified", service.service_id));
        }
        if !service.capacity.source_ref.as_deref().is_some_and(has_source) {
            blockers.insert(format!("platform service {} capacity provenance does not resolve to non-synthetic evidence", service.service_id));
        }
    }

    for deck in &configuration.platform.deck_bounds {
        for (label, value) in [
            ("area load", &deck.max_area_load_kg_m2),
            ("point load", &deck.max_point_load_kg),
        ] {
            if !matches!(value.status.as_str(), "measured" | "source_verified") {
                blockers.insert(format!("deck {} {label} limit is not measured/source-verified", deck.deck_id));
            }
            if !value.source_ref.as_deref().is_some_and(has_source) {
                blockers.insert(format!("deck {} {label} provenance does not resolve to non-synthetic evidence", deck.deck_id));
            }
        }
    }

    for connection in &configuration.connections {
        if !matches!(connection.compatibility_status.as_str(), "verified_compatible") {
            blockers.insert(format!("connection {} is not verified compatible", connection.connection_id));
        }
        if connection.evidence_refs.is_empty()
            || connection.evidence_refs.iter().any(|reference| !has_source(reference))
        {
            blockers.insert(format!("connection {} evidence references do not resolve to non-synthetic sources", connection.connection_id));
        }
        if !matches!(connection.route_length_m.status.as_str(), "computed" | "measured" | "source_verified")
            || !connection.route_length_m.source_ref.as_deref().is_some_and(has_source)
        {
            blockers.insert(format!("connection {} route evidence is incomplete", connection.connection_id));
        }
    }

    for gate in &configuration.safety_gates {
        match gate.disposition.as_str() {
            "fail" => {
                hard_reject = true;
                blockers.insert(format!("safety gate {} failed", gate.gate_id));
            }
            "pass" | "not_applicable" if !gate.evidence_refs.is_empty()
                && gate.evidence_refs.iter().all(|reference| has_source(reference)) => {}
            _ => {
                blockers.insert(format!("safety gate {} lacks closed, evidence-backed disposition", gate.gate_id));
            }
        }
    }

    (hard_reject, blockers.into_iter().collect())
}

fn objective_blockers(
    objectives: &[ObjectiveObservation],
    configuration: &Configuration,
) -> Vec<String> {
    let mut blockers = BTreeSet::new();
    let mut ids = BTreeSet::new();
    let sources = usable_evidence_sources(configuration);
    if objectives.is_empty() {
        blockers.insert("no objectives were supplied".to_owned());
    }
    for objective in objectives {
        if objective.metric_id.trim().is_empty() {
            blockers.insert("objective metric ID is empty".to_owned());
        }
        if !ids.insert(objective.metric_id.clone()) {
            blockers.insert(format!("duplicate objective ID {}", objective.metric_id));
        }
        if objective.unit.trim().is_empty() {
            blockers.insert(format!("objective {} has no unit", objective.metric_id));
        }
        if objective.comparison_basis.trim().is_empty() {
            blockers.insert(format!("objective {} has no comparison basis", objective.metric_id));
        }
        match objective.value {
            None => {
                blockers.insert(format!("objective {} is unknown", objective.metric_id));
            }
            Some(value) if !value.is_finite() => {
                blockers.insert(format!("objective {} is not finite", objective.metric_id));
            }
            _ => {}
        }
        if objective.evidence_refs.is_empty()
            || objective.evidence_refs.iter().any(|reference| {
                reference.trim().is_empty() || !sources.contains(reference)
            })
        {
            blockers.insert(format!("objective {} evidence does not resolve to non-synthetic configuration provenance", objective.metric_id));
        }
        if !matches!(
            objective.evidence_status,
            ObjectiveEvidenceStatus::Computed
                | ObjectiveEvidenceStatus::Measured
                | ObjectiveEvidenceStatus::SourceVerified
        ) {
            blockers.insert(format!(
                "objective {} is not computed from qualified evidence (status {:?})",
                objective.metric_id, objective.evidence_status
            ));
        }
    }
    blockers.into_iter().collect()
}

fn dominates(
    left: &[ObjectiveObservation],
    right: &[ObjectiveObservation],
) -> bool {
    if left.len() != right.len() {
        return false;
    }
    let mut left = left.to_vec();
    let mut right = right.to_vec();
    left.sort_by(|a, b| a.metric_id.cmp(&b.metric_id));
    right.sort_by(|a, b| a.metric_id.cmp(&b.metric_id));
    let mut strictly_better = false;
    for (left_value, right_value) in left.iter().zip(right.iter()) {
        if left_value.metric_id != right_value.metric_id
            || left_value.unit != right_value.unit
            || left_value.direction != right_value.direction
            || left_value.comparison_basis != right_value.comparison_basis
        {
            return false;
        }
        let (Some(a), Some(b)) = (left_value.value, right_value.value) else {
            return false;
        };
        if !a.is_finite() || !b.is_finite() {
            return false;
        }
        match left_value.direction {
            ObjectiveDirection::Minimize => {
                if a > b {
                    return false;
                }
                if a < b {
                    strictly_better = true;
                }
            }
            ObjectiveDirection::Maximize => {
                if a < b {
                    return false;
                }
                if a > b {
                    strictly_better = true;
                }
            }
        }
    }
    strictly_better
}

/// Ranks only fully eligible candidates. Safety/validation errors, indeterminate
/// configurations and incomplete or assumption-only objectives are excluded before
/// Pareto comparison. Incomparable objective signatures form separate groups.
pub fn rank_candidates(
    submissions: Vec<CandidateSubmission>,
) -> Result<ParetoReport, serde_json::Error> {
    let fingerprints = input_fingerprints(&submissions)?;
    let input_manifest_sha256 = digest(&fingerprints)?;
    let mut id_counts = BTreeMap::<String, usize>::new();
    for submission in &submissions {
        *id_counts.entry(submission.candidate_id.clone()).or_default() += 1;
    }

    let mut assessments = Vec::with_capacity(submissions.len());
    let mut eligible = Vec::new();

    for submission in &submissions {
        let configuration_sha256 = digest(&submission.configuration)?;
        let mut blockers = BTreeSet::new();
        let normalized = normalized_objectives(&submission.objectives);

        if submission.candidate_id.trim().is_empty() {
            blockers.insert("candidate ID is empty".to_owned());
        } else if id_counts.get(&submission.candidate_id).copied().unwrap_or(0) > 1 {
            blockers.insert("candidate ID is duplicated in the input batch".to_owned());
        }

        let receipt_valid = verify_validation_receipt(
            &submission.configuration,
            &submission.validation_receipt,
        );
        if !receipt_valid {
            blockers.insert("validation receipt is invalid or does not bind this exact configuration".to_owned());
        } else {
            match submission.validation_receipt.body.overall_disposition.as_str() {
                "rejected" => {
                    blockers.insert("configuration failed one or more hard validation checks".to_owned());
                }
                "indeterminate" => {
                    blockers.insert("configuration has unresolved validation or safety evidence".to_owned());
                }
                "candidate_only" => {
                    if submission.validation_receipt.body.findings.iter().any(|finding| {
                        matches!(finding.severity, Severity::Error | Severity::Indeterminate)
                    }) {
                        blockers.insert("receipt disposition conflicts with its unresolved findings".to_owned());
                    }
                }
                _ => {
                    blockers.insert("validation receipt has an unsupported disposition".to_owned());
                }
            }
        }
        let (configuration_has_hard_failure, configuration_blockers) =
            configuration_evidence_blockers(&submission.configuration);
        for blocker in configuration_blockers {
            blockers.insert(blocker);
        }
        for blocker in objective_blockers(&normalized, &submission.configuration) {
            blockers.insert(blocker);
        }

        let eligibility = if configuration_has_hard_failure || blockers.iter().any(|blocker| {
            blocker.contains("failed one or more hard validation checks")
                || blocker.contains("invalid or does not bind")
                || blocker.contains("candidate ID is duplicated")
                || blocker.contains("candidate ID is empty")
                || blocker.contains("unsupported disposition")
                || blocker.contains("conflicts with its unresolved findings")
        }) {
            CandidateEligibility::Rejected
        } else if blockers.is_empty() {
            CandidateEligibility::Eligible
        } else {
            CandidateEligibility::ReviewRequired
        };

        let keys = objective_keys(&normalized);
        let group_sha256 = if eligibility == CandidateEligibility::Eligible {
            Some(digest(&keys)?)
        } else {
            None
        };
        let index = assessments.len();
        assessments.push(CandidateAssessment {
            candidate_id: submission.candidate_id.clone(),
            configuration_sha256: configuration_sha256.clone(),
            eligibility,
            pareto_status: ParetoStatus::NotRanked,
            objective_group_sha256: group_sha256.clone(),
            dominated_by: None,
            blockers: blockers.into_iter().collect(),
            objectives: normalized.clone(),
        });

        if eligibility == CandidateEligibility::Eligible {
            eligible.push(EligibleCandidate {
                index,
                candidate_id: submission.candidate_id.clone(),
                group_sha256: group_sha256.expect("eligible candidates have objective groups"),
                keys,
                objectives: normalized,
            });
        }
    }

    let mut grouped = BTreeMap::<String, Vec<EligibleCandidate>>::new();
    for candidate in eligible {
        grouped.entry(candidate.group_sha256.clone()).or_default().push(candidate);
    }

    let mut groups = Vec::new();
    for (group_sha256, mut candidates) in grouped {
        candidates.sort_by(|a, b| a.candidate_id.cmp(&b.candidate_id));
        let mut frontier_ids = Vec::new();
        let mut dominance = Vec::new();

        for candidate in &candidates {
            let mut dominators: Vec<&EligibleCandidate> = candidates.iter()
                .filter(|other| other.candidate_id != candidate.candidate_id
                    && dominates(&other.objectives, &candidate.objectives))
                .collect();
            dominators.sort_by(|a, b| a.candidate_id.cmp(&b.candidate_id));
            if let Some(dominator) = dominators.first() {
                assessments[candidate.index].pareto_status = ParetoStatus::Dominated;
                assessments[candidate.index].dominated_by = Some(dominator.candidate_id.clone());
                dominance.push(DominanceDecision {
                    dominated_candidate_id: candidate.candidate_id.clone(),
                    dominator_candidate_id: dominator.candidate_id.clone(),
                });
            } else {
                assessments[candidate.index].pareto_status = ParetoStatus::Frontier;
                frontier_ids.push(candidate.candidate_id.clone());
            }
        }

        groups.push(ParetoGroup {
            objective_group_sha256: group_sha256,
            objective_keys: candidates.first().map(|candidate| candidate.keys.clone()).unwrap_or_default(),
            eligible_candidate_ids: candidates.iter().map(|candidate| candidate.candidate_id.clone()).collect(),
            frontier_candidate_ids: frontier_ids,
            dominance,
        });
    }

    assessments.sort_by(|a, b| a.candidate_id.cmp(&b.candidate_id)
        .then(a.configuration_sha256.cmp(&b.configuration_sha256)));
    groups.sort_by(|a, b| a.objective_group_sha256.cmp(&b.objective_group_sha256));

    let body = ParetoReportBody {
        analyzer_version: PARETO_ANALYZER_VERSION.to_owned(),
        input_manifest_sha256,
        evidence_policy: "hard_constraints_first_no_weighted_safety_tradeoff_no_assumption_only_frontier".to_owned(),
        candidates: assessments,
        groups,
        claim_ceiling: "Pareto comparison only among candidates whose supplied configuration validation receipts and objective evidence are eligible. Does not establish vessel feasibility, regulatory compliance or approval.".to_owned(),
    };
    let report_sha256 = digest(&body)?;
    Ok(ParetoReport { body, report_sha256 })
}

/// Verifies report integrity and that the report corresponds to these exact submissions.
pub fn verify_pareto_report(
    submissions: &[CandidateSubmission],
    report: &ParetoReport,
) -> bool {
    let Ok(fingerprints) = input_fingerprints(submissions) else { return false; };
    let Ok(input_digest) = digest(&fingerprints) else { return false; };
    let Ok(report_digest) = digest(&report.body) else { return false; };
    report.body.analyzer_version == PARETO_ANALYZER_VERSION
        && report.body.input_manifest_sha256 == input_digest
        && report.report_sha256 == report_digest
}

#[cfg(test)]
mod tests {
    use super::*;

    const LIVE: &str = include_str!("../fixtures/modular-interiors/liveaboard-custom-layout-v1.json");
    const EXPEDITION: &str = include_str!("../fixtures/modular-interiors/expedition-research-layout-v1.json");

    fn live() -> Configuration { serde_json::from_str(LIVE).expect("live fixture parses") }
    fn expedition() -> Configuration { serde_json::from_str(EXPEDITION).expect("expedition fixture parses") }

    fn evidence_complete_test_config(mut configuration: Configuration, suffix: &str) -> Configuration {
        // In-memory test sentinels exercise the eligibility path only. They do not alter
        // production fixtures and are not measurements or vessel evidence.
        configuration.configuration_id.push_str(suffix);
        let source = crate::modular_interior::Provenance {
            source_id: "TEST-HARNESS-ONLY".to_owned(),
            title: "Test harness sentinel, not real engineering evidence".to_owned(),
            uri: "urn:symthaea:test-harness-only".to_owned(),
            scope: "Unit-test mechanics only; never production measurement or approval".to_owned(),
            source_type: "measured_test".to_owned(),
        };
        configuration.provenance = vec![source.clone()];
        for module in &mut configuration.module_catalog {
            module.mass_properties.status = "source_verified".to_owned();
            module.mass_properties.source_ref = Some("TEST-HARNESS-ONLY".to_owned());
            module.provenance = vec![source.clone()];
            module.fire_material_status = "authority_reviewed".to_owned();
            module.mounting_status = "authority_reviewed".to_owned();
            module.electrical_status = "authority_reviewed".to_owned();
            module.sanitary_status = "authority_reviewed".to_owned();
            for port in &mut module.ports {
                port.rating.status = "source_verified".to_owned();
                port.rating.source_ref = Some("TEST-HARNESS-ONLY".to_owned());
                port.connector_profile = "TEST-HARNESS-INTERFACE".to_owned();
            }
        }
        for assumption in &mut configuration.assumptions {
            assumption.status = "measured".to_owned();
        }
        for service in &mut configuration.platform_services {
            service.capacity.status = "source_verified".to_owned();
            service.capacity.source_ref = Some("TEST-HARNESS-ONLY".to_owned());
            service.connector_profile = "TEST-HARNESS-INTERFACE".to_owned();
        }
        for deck in &mut configuration.platform.deck_bounds {
            deck.max_area_load_kg_m2.value = Some(500.0);
            deck.max_area_load_kg_m2.status = "source_verified".to_owned();
            deck.max_area_load_kg_m2.source_ref = Some("TEST-HARNESS-ONLY".to_owned());
            deck.max_point_load_kg.value = Some(5000.0);
            deck.max_point_load_kg.status = "source_verified".to_owned();
            deck.max_point_load_kg.source_ref = Some("TEST-HARNESS-ONLY".to_owned());
            deck.source_ref = Some("TEST-HARNESS-ONLY".to_owned());
        }
        for service in &mut configuration.platform_services {
            service.capacity.source_ref = Some("TEST-HARNESS-ONLY".to_owned());
        }
        for connection in &mut configuration.connections {
            connection.compatibility_status = "verified_compatible".to_owned();
            connection.evidence_refs = vec!["TEST-HARNESS-ONLY".to_owned()];
            connection.route_length_m.value = Some(1.0);
            connection.route_length_m.status = "computed".to_owned();
            connection.route_length_m.source_ref = Some("TEST-HARNESS-ONLY".to_owned());
        }
        for gate in &mut configuration.safety_gates {
            gate.disposition = "pass".to_owned();
            gate.evidence_refs = vec!["TEST-HARNESS-ONLY".to_owned()];
        }
        configuration
    }

    fn objective(id: &str, value: f64, unit: &str, direction: ObjectiveDirection) -> ObjectiveObservation {
        ObjectiveObservation {
            metric_id: id.to_owned(),
            value: Some(value),
            unit: unit.to_owned(),
            direction,
            evidence_status: ObjectiveEvidenceStatus::Computed,
            comparison_basis: "TEST-ORACLE-ONLY".to_owned(),
            evidence_refs: vec!["TEST-HARNESS-ONLY".to_owned()],
        }
    }

    fn submission(id: &str, config: Configuration, objectives: Vec<ObjectiveObservation>) -> CandidateSubmission {
        let validation_receipt = validate_configuration(&config).expect("test receipt serializes");
        CandidateSubmission {
            candidate_id: id.to_owned(),
            configuration: config,
            validation_receipt,
            objectives,
        }
    }

    #[test]
    fn synthetic_fixtures_do_not_enter_evidence_backed_frontier() {
        let a = submission("live", live(), vec![
            objective("mass", 1210.0, "kg", ObjectiveDirection::Minimize),
        ]);
        let b = submission("expedition", expedition(), vec![
            objective("mass", 1700.0, "kg", ObjectiveDirection::Minimize),
        ]);
        let report = rank_candidates(vec![a, b]).expect("report serializes");
        assert!(report.body.groups.is_empty());
        assert!(report.body.candidates.iter().all(|candidate|
            candidate.eligibility == CandidateEligibility::ReviewRequired));
    }

    #[test]
    fn hand_calculated_pareto_frontier_excludes_dominated_candidate() {
        let base = evidence_complete_test_config(live(), "-TEST");
        let a = submission("A", base.clone(), vec![
            objective("mass", 100.0, "kg", ObjectiveDirection::Minimize),
            objective("cost", 50.0, "credits", ObjectiveDirection::Minimize),
        ]);
        let b = submission("B", base.clone(), vec![
            objective("mass", 80.0, "kg", ObjectiveDirection::Minimize),
            objective("cost", 70.0, "credits", ObjectiveDirection::Minimize),
        ]);
        let c = submission("C", base, vec![
            objective("mass", 120.0, "kg", ObjectiveDirection::Minimize),
            objective("cost", 80.0, "credits", ObjectiveDirection::Minimize),
        ]);
        let report = rank_candidates(vec![c, a, b]).expect("report serializes");
        assert_eq!(report.body.groups.len(), 1);
        assert_eq!(report.body.groups[0].frontier_candidate_ids, vec!["A", "B"]);
        assert_eq!(report.body.groups[0].dominance, vec![DominanceDecision {
            dominated_candidate_id: "C".to_owned(),
            dominator_candidate_id: "A".to_owned(),
        }]);
    }

    #[test]
    fn direction_and_equal_ties_are_handled_without_a_weighted_score() {
        let base = evidence_complete_test_config(live(), "-TEST");
        let a = submission("A", base.clone(), vec![
            objective("quality", 0.8, "ratio", ObjectiveDirection::Maximize),
        ]);
        let b = submission("B", base, vec![
            objective("quality", 0.8, "ratio", ObjectiveDirection::Maximize),
        ]);
        let report = rank_candidates(vec![a, b]).expect("report serializes");
        assert_eq!(report.body.groups[0].frontier_candidate_ids, vec!["A", "B"]);
        assert!(report.body.groups[0].dominance.is_empty());
    }

    #[test]
    fn different_units_or_comparison_bases_never_share_a_group() {
        let base = evidence_complete_test_config(live(), "-TEST");
        let kg = submission("kg-candidate", base.clone(), vec![
            objective("mass", 100.0, "kg", ObjectiveDirection::Minimize),
        ]);
        let pounds = submission("lb-candidate", base, vec![
            objective("mass", 220.0, "lb", ObjectiveDirection::Minimize),
        ]);
        let report = rank_candidates(vec![kg, pounds]).expect("report serializes");
        assert_eq!(report.body.groups.len(), 2);
    }

    #[test]
    fn stale_validation_receipt_is_rejected() {
        let config = evidence_complete_test_config(live(), "-TEST");
        let mut candidate = submission("stale", config.clone(), vec![
            objective("mass", 100.0, "kg", ObjectiveDirection::Minimize),
        ]);
        candidate.configuration.configuration_name.push_str(" changed after validation");
        let report = rank_candidates(vec![candidate]).expect("report serializes");
        assert!(report.body.groups.is_empty());
        assert_eq!(report.body.candidates[0].eligibility, CandidateEligibility::Rejected);
    }

    #[test]
    fn assumption_only_and_unknown_objectives_are_excluded() {
        let config = evidence_complete_test_config(live(), "-TEST");
        let mut assumed = objective("mass", 100.0, "kg", ObjectiveDirection::Minimize);
        assumed.evidence_status = ObjectiveEvidenceStatus::DesignAssumption;
        let unknown = ObjectiveObservation {
            metric_id: "energy".to_owned(),
            value: None,
            unit: "kWh".to_owned(),
            direction: ObjectiveDirection::Minimize,
            evidence_status: ObjectiveEvidenceStatus::Unknown,
            comparison_basis: "TEST-ORACLE-ONLY".to_owned(),
            evidence_refs: vec!["TEST-HARNESS-ONLY".to_owned()],
        };
        let report = rank_candidates(vec![submission("A", config, vec![assumed, unknown])])
            .expect("report serializes");
        assert!(report.body.groups.is_empty());
        assert_eq!(report.body.candidates[0].eligibility, CandidateEligibility::ReviewRequired);
    }

    #[test]
    fn duplicate_candidate_ids_are_rejected_as_ambiguous() {
        let config = evidence_complete_test_config(live(), "-TEST");
        let a = submission("same", config.clone(), vec![objective("mass", 100.0, "kg", ObjectiveDirection::Minimize)]);
        let b = submission("same", config, vec![objective("mass", 90.0, "kg", ObjectiveDirection::Minimize)]);
        let report = rank_candidates(vec![a, b]).expect("report serializes");
        assert!(report.body.groups.is_empty());
        assert!(report.body.candidates.iter().all(|candidate|
            candidate.eligibility == CandidateEligibility::Rejected));
    }

    #[test]
    fn candidate_order_does_not_change_the_report_digest() {
        let config = evidence_complete_test_config(live(), "-TEST");
        let a = submission("A", config.clone(), vec![objective("mass", 100.0, "kg", ObjectiveDirection::Minimize)]);
        let b = submission("B", config, vec![objective("mass", 90.0, "kg", ObjectiveDirection::Minimize)]);
        let first = vec![a.clone(), b.clone()];
        let second = vec![b, a];
        let report_a = rank_candidates(first.clone()).expect("report serializes");
        let report_b = rank_candidates(second.clone()).expect("report serializes");
        assert_eq!(report_a.report_sha256, report_b.report_sha256);
        assert!(verify_pareto_report(&first, &report_a));
        assert!(verify_pareto_report(&second, &report_a));
        let mut tampered = report_a.clone();
        tampered.body.claim_ceiling.push_str(" altered");
        assert!(!verify_pareto_report(&first, &tampered));
    }
}

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::{env, fs, process};

const SCHEMA_VERSION: &str = "0.1.0";
const SOURCE_LEVEL_QUANTITY: &str = "source_level_db_re_1upa_at_1m";

#[derive(Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum EvidenceKind {
    Synthetic,
    PredictionOnly,
    MeasuredFullScale,
    IndependentlyValidatedTransfer,
}

#[derive(Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum RightsStatus {
    PublicOrPermitted,
    Unknown,
    Restricted,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Case {
    schema_version: String,
    case_id: String,
    source: DatasetEvidence,
    evidence_kind: EvidenceKind,
    rights_status: RightsStatus,
    independently_reviewed: bool,
    scenario: ServiceScenario,
    limits: PolicyLimits,
    candidates: Vec<Candidate>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct DatasetEvidence {
    dataset_id: String,
    citation: String,
    raw_data_sha256: String,
    rights_decision_reference: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ServiceScenario {
    vessel_id: String,
    vessel_configuration_sha256: String,
    scenario_fingerprint_sha256: String,
    propeller_design_id: String,
    propeller_revision: String,
    service_fingerprint_sha256: String,
    environment_fingerprint_sha256: String,
    hull_state_sha256: String,
    route_id: String,
    draft_m: f64,
    cargo_mass_tonnes: f64,
    delivered_service_tonne_nm: f64,
    maximum_delivery_window_hours: f64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PolicyLimits {
    minimum_delivered_service_tonne_nm: f64,
    maximum_voyage_duration_hours: f64,
    maximum_primary_energy_kwh_per_tonne_nm: f64,
    maximum_well_to_wake_ghg_gco2e_per_tonne_nm: f64,
    maximum_lifecycle_cost_usd_per_tonne_nm: f64,
    source_level_limits: Vec<SourceLevelLimit>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SourceLevelLimit {
    band_id: String,
    quantity_id: String,
    maximum_source_level_db: f64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Candidate {
    candidate_id: String,
    policy_label: String,
    operating_point_id: String,
    scenario_fingerprint_sha256: String,
    evidence_kind: EvidenceKind,
    independently_reviewed: bool,
    evidence_review_receipt_sha256: String,
    operating_point_receipt_sha256: String,
    raw_measurement_or_validation_sha256: String,
    method_id: String,
    propeller_rpm: f64,
    vessel_speed_knots: f64,
    metrics: CandidateMetrics,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CandidateMetrics {
    delivered_service_tonne_nm: f64,
    delivered_service_u95_tonne_nm: f64,
    voyage_duration_hours: f64,
    voyage_duration_u95_hours: f64,
    primary_energy_kwh_per_tonne_nm: f64,
    primary_energy_u95_kwh_per_tonne_nm: f64,
    well_to_wake_ghg_gco2e_per_tonne_nm: f64,
    well_to_wake_ghg_u95_gco2e_per_tonne_nm: f64,
    lifecycle_cost_usd_per_tonne_nm: f64,
    lifecycle_cost_u95_usd_per_tonne_nm: f64,
    source_levels: Vec<SourceLevel>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SourceLevel {
    band_id: String,
    quantity_id: String,
    source_level_db: f64,
    expanded_uncertainty_db: f64,
    method_id: String,
    evidence_sha256: String,
}

#[derive(Debug, Serialize)]
struct AuditReport {
    schema_version: &'static str,
    case_id: String,
    dataset_id: String,
    source_data_sha256: String,
    rights_decision_reference: String,
    evidence_kind: &'static str,
    rights_status: &'static str,
    independently_reviewed: bool,
    vessel_id: String,
    vessel_configuration_sha256: String,
    propeller_design_id: String,
    propeller_revision: String,
    route_id: String,
    environment_fingerprint_sha256: String,
    hull_state_sha256: String,
    service_fingerprint_sha256: String,
    scenario_fingerprint_sha256: String,
    disposition: &'static str,
    candidate_reports: Vec<CandidateReport>,
    feasible_pareto_frontier_candidate_ids: Vec<String>,
    excluded_candidate_count: usize,
    objective_dimensions: Vec<String>,
    selection_rule: &'static str,
    claim_ceiling: &'static str,
    reasons: Vec<String>,
}

#[derive(Debug, Serialize)]
struct SourceLevelEvidenceReceipt {
    band_id: String,
    quantity_id: String,
    source_level_db: f64,
    expanded_uncertainty_db: f64,
    method_id: String,
    evidence_sha256: String,
}

#[derive(Debug, Serialize)]
struct CandidateReport {
    candidate_id: String,
    policy_label: String,
    operating_point_id: String,
    scenario_fingerprint_sha256: String,
    method_id: String,
    evidence_review_receipt_sha256: String,
    operating_point_receipt_sha256: String,
    raw_measurement_or_validation_sha256: String,
    propeller_rpm: f64,
    vessel_speed_knots: f64,
    evidence_kind: &'static str,
    source_level_evidence: Vec<SourceLevelEvidenceReceipt>,
    disposition: &'static str,
    constraint_decisions: Vec<ConstraintDecision>,
    robust_objectives: Vec<ObjectiveValue>,
    pareto_status: &'static str,
    reasons: Vec<String>,
}

#[derive(Debug, Serialize)]
struct ConstraintDecision {
    constraint_id: String,
    observed_lower_bound: f64,
    observed_upper_bound: f64,
    limit: f64,
    direction: &'static str,
    result: &'static str,
}

#[derive(Debug, Serialize, Clone)]
struct ObjectiveValue {
    objective_id: String,
    conservative_value: f64,
    direction: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GateResult {
    Pass,
    Fail,
    Uncertain,
}

fn sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn finite_positive(value: f64) -> bool {
    value.is_finite() && value > 0.0
}

fn evidence_name(kind: &EvidenceKind) -> &'static str {
    match kind {
        EvidenceKind::Synthetic => "synthetic",
        EvidenceKind::PredictionOnly => "prediction_only",
        EvidenceKind::MeasuredFullScale => "measured_full_scale",
        EvidenceKind::IndependentlyValidatedTransfer => "independently_validated_transfer",
    }
}

fn max_gate(value: f64, uncertainty: f64, maximum: f64) -> GateResult {
    let lower = value - uncertainty;
    let upper = value + uncertainty;
    if lower > maximum {
        GateResult::Fail
    } else if upper <= maximum {
        GateResult::Pass
    } else {
        GateResult::Uncertain
    }
}

fn min_gate(value: f64, uncertainty: f64, minimum: f64) -> GateResult {
    let lower = value - uncertainty;
    let upper = value + uncertainty;
    if upper < minimum {
        GateResult::Fail
    } else if lower >= minimum {
        GateResult::Pass
    } else {
        GateResult::Uncertain
    }
}

fn gate_name(value: GateResult) -> &'static str {
    match value {
        GateResult::Pass => "pass",
        GateResult::Fail => "fail",
        GateResult::Uncertain => "uncertain",
    }
}

fn push_max_constraint(
    decisions: &mut Vec<ConstraintDecision>,
    reasons: &mut Vec<String>,
    id: String,
    value: f64,
    uncertainty: f64,
    maximum: f64,
) -> GateResult {
    let result = max_gate(value, uncertainty, maximum);
    let lower = value - uncertainty;
    let upper = value + uncertainty;
    if result != GateResult::Pass {
        reasons.push(format!("{}: robust upper bound {:.6} exceeds or crosses maximum {:.6}", id, upper, maximum));
    }
    decisions.push(ConstraintDecision {
        constraint_id: id,
        observed_lower_bound: lower,
        observed_upper_bound: upper,
        limit: maximum,
        direction: "maximum",
        result: gate_name(result),
    });
    result
}

fn push_min_constraint(
    decisions: &mut Vec<ConstraintDecision>,
    reasons: &mut Vec<String>,
    id: String,
    value: f64,
    uncertainty: f64,
    minimum: f64,
) -> GateResult {
    let result = min_gate(value, uncertainty, minimum);
    let lower = value - uncertainty;
    let upper = value + uncertainty;
    if result != GateResult::Pass {
        reasons.push(format!("{}: robust lower bound {:.6} is below or crosses minimum {:.6}", id, lower, minimum));
    }
    decisions.push(ConstraintDecision {
        constraint_id: id,
        observed_lower_bound: lower,
        observed_upper_bound: upper,
        limit: minimum,
        direction: "minimum",
        result: gate_name(result),
    });
    result
}

fn validate_case(case: &Case) -> Vec<String> {
    let mut errors = Vec::new();
    if case.schema_version != SCHEMA_VERSION {
        errors.push(format!("schema_version must be {SCHEMA_VERSION}"));
    }
    if case.case_id.trim().is_empty()
        || case.source.dataset_id.trim().is_empty()
        || case.source.citation.trim().is_empty()
        || case.source.rights_decision_reference.trim().is_empty()
    {
        errors.push("case, dataset, citation and rights-decision identities are required".to_string());
    }
    if !sha256(&case.source.raw_data_sha256) {
        errors.push("dataset raw_data_sha256 must be a 64-character hexadecimal digest".to_string());
    }
    for (label, digest) in [
        ("vessel configuration", &case.scenario.vessel_configuration_sha256),
        ("scenario fingerprint", &case.scenario.scenario_fingerprint_sha256),
        ("service fingerprint", &case.scenario.service_fingerprint_sha256),
        ("environment fingerprint", &case.scenario.environment_fingerprint_sha256),
        ("hull state", &case.scenario.hull_state_sha256),
    ] {
        if !sha256(digest) {
            errors.push(format!("{label} digest is invalid"));
        }
    }
    if case.scenario.vessel_id.trim().is_empty()
        || case.scenario.propeller_design_id.trim().is_empty()
        || case.scenario.propeller_revision.trim().is_empty()
        || case.scenario.route_id.trim().is_empty()
    {
        errors.push("vessel, propeller, revision and route identities are required".to_string());
    }
    for (label, value) in [
        ("draft_m", case.scenario.draft_m),
        ("cargo_mass_tonnes", case.scenario.cargo_mass_tonnes),
        ("delivered_service_tonne_nm", case.scenario.delivered_service_tonne_nm),
        ("maximum_delivery_window_hours", case.scenario.maximum_delivery_window_hours),
        ("minimum delivered service", case.limits.minimum_delivered_service_tonne_nm),
        ("maximum voyage duration", case.limits.maximum_voyage_duration_hours),
        ("maximum primary energy", case.limits.maximum_primary_energy_kwh_per_tonne_nm),
        ("maximum well-to-wake GHG", case.limits.maximum_well_to_wake_ghg_gco2e_per_tonne_nm),
        ("maximum lifecycle cost", case.limits.maximum_lifecycle_cost_usd_per_tonne_nm),
    ] {
        if !finite_positive(value) {
            errors.push(format!("{label} must be finite and positive"));
        }
    }
    if case.limits.source_level_limits.is_empty() {
        errors.push("at least one project-defined source-level band limit is required".to_string());
    }
    let mut limit_bands = BTreeSet::new();
    for limit in &case.limits.source_level_limits {
        if limit.band_id.trim().is_empty() || !limit_bands.insert(limit.band_id.clone()) {
            errors.push("source-level limit band IDs must be non-empty and unique".to_string());
        }
        if limit.quantity_id != SOURCE_LEVEL_QUANTITY {
            errors.push(format!("{} must use source-level dB re 1 uPa at 1 m; received levels are not interchangeable", limit.band_id));
        }
        if !limit.maximum_source_level_db.is_finite() {
            errors.push(format!("{} maximum source-level limit must be finite", limit.band_id));
        }
    }
    if case.candidates.len() < 2 {
        errors.push("at least two candidate operating policies are required for Pareto comparison".to_string());
    }
    let mut candidate_ids = BTreeSet::new();
    let mut operating_point_ids = BTreeSet::new();
    for candidate in &case.candidates {
        if candidate.candidate_id.trim().is_empty()
            || !candidate_ids.insert(candidate.candidate_id.clone())
        {
            errors.push(format!("empty or duplicate candidate_id {}", candidate.candidate_id));
        }
        if candidate.operating_point_id.trim().is_empty()
            || !operating_point_ids.insert(candidate.operating_point_id.clone())
        {
            errors.push(format!("empty or duplicate operating_point_id {}", candidate.operating_point_id));
        }
        if candidate.policy_label.trim().is_empty() || candidate.method_id.trim().is_empty() {
            errors.push(format!("{} requires a policy label and method ID", candidate.candidate_id));
        }
        for (label, digest) in [
            ("scenario fingerprint", &candidate.scenario_fingerprint_sha256),
            ("evidence review receipt", &candidate.evidence_review_receipt_sha256),
            ("operating-point receipt", &candidate.operating_point_receipt_sha256),
            ("raw measurement/validation", &candidate.raw_measurement_or_validation_sha256),
        ] {
            if !sha256(digest) {
                errors.push(format!("{}: {label} digest is invalid", candidate.candidate_id));
            }
        }
        if candidate.scenario_fingerprint_sha256 != case.scenario.scenario_fingerprint_sha256 {
            errors.push(format!("{}: candidate does not bind to the frozen vessel/service scenario fingerprint", candidate.candidate_id));
        }
        if !finite_positive(candidate.propeller_rpm) || !finite_positive(candidate.vessel_speed_knots) {
            errors.push(format!("{}: propeller RPM and vessel speed must be finite and positive", candidate.candidate_id));
        }
        let metrics = &candidate.metrics;
        for (label, value) in [
            ("delivered service", metrics.delivered_service_tonne_nm),
            ("service uncertainty", metrics.delivered_service_u95_tonne_nm),
            ("voyage duration", metrics.voyage_duration_hours),
            ("duration uncertainty", metrics.voyage_duration_u95_hours),
            ("primary energy intensity", metrics.primary_energy_kwh_per_tonne_nm),
            ("energy uncertainty", metrics.primary_energy_u95_kwh_per_tonne_nm),
            ("well-to-wake GHG intensity", metrics.well_to_wake_ghg_gco2e_per_tonne_nm),
            ("GHG uncertainty", metrics.well_to_wake_ghg_u95_gco2e_per_tonne_nm),
            ("lifecycle cost intensity", metrics.lifecycle_cost_usd_per_tonne_nm),
            ("cost uncertainty", metrics.lifecycle_cost_u95_usd_per_tonne_nm),
        ] {
            if !finite_positive(value) {
                errors.push(format!("{}: {label} must be finite and positive; zero cannot encode unknown uncertainty", candidate.candidate_id));
            }
        }
        let mut source_bands = BTreeSet::new();
        for band in &metrics.source_levels {
            if band.band_id.trim().is_empty() || !source_bands.insert(band.band_id.clone()) {
                errors.push(format!("{}: source-level band IDs must be non-empty and unique", candidate.candidate_id));
            }
            if band.quantity_id != SOURCE_LEVEL_QUANTITY {
                errors.push(format!("{} / {}: source-level measurement must be referenced to 1 m and cannot be replaced by received SPL", candidate.candidate_id, band.band_id));
            }
            if !band.source_level_db.is_finite() || !finite_positive(band.expanded_uncertainty_db) {
                errors.push(format!("{} / {}: source level must be finite and expanded uncertainty positive", candidate.candidate_id, band.band_id));
            }
            if band.method_id.trim().is_empty() || !sha256(&band.evidence_sha256) {
                errors.push(format!("{} / {}: source-level method and evidence digest are required", candidate.candidate_id, band.band_id));
            }
        }
        let required_bands: BTreeSet<String> = case.limits.source_level_limits.iter().map(|x| x.band_id.clone()).collect();
        if source_bands != required_bands {
            errors.push(format!("{}: source-level band coverage must exactly match all declared policy limits", candidate.candidate_id));
        }
    }
    errors
}

fn robust_objectives(candidate: &Candidate) -> Vec<ObjectiveValue> {
    let m = &candidate.metrics;
    let mut values = vec![
        ObjectiveValue {
            objective_id: "primary_energy_kwh_per_tonne_nm".to_string(),
            conservative_value: m.primary_energy_kwh_per_tonne_nm + m.primary_energy_u95_kwh_per_tonne_nm,
            direction: "minimize",
        },
        ObjectiveValue {
            objective_id: "well_to_wake_ghg_gco2e_per_tonne_nm".to_string(),
            conservative_value: m.well_to_wake_ghg_gco2e_per_tonne_nm + m.well_to_wake_ghg_u95_gco2e_per_tonne_nm,
            direction: "minimize",
        },
        ObjectiveValue {
            objective_id: "voyage_duration_hours".to_string(),
            conservative_value: m.voyage_duration_hours + m.voyage_duration_u95_hours,
            direction: "minimize",
        },
        ObjectiveValue {
            objective_id: "lifecycle_cost_usd_per_tonne_nm".to_string(),
            conservative_value: m.lifecycle_cost_usd_per_tonne_nm + m.lifecycle_cost_u95_usd_per_tonne_nm,
            direction: "minimize",
        },
    ];
    for band in &m.source_levels {
        values.push(ObjectiveValue {
            objective_id: format!("source_level_db_re_1upa_at_1m:{}", band.band_id),
            conservative_value: band.source_level_db + band.expanded_uncertainty_db,
            direction: "minimize",
        });
    }
    values.sort_by(|left, right| left.objective_id.cmp(&right.objective_id));
    values
}

fn dominates(left: &[ObjectiveValue], right: &[ObjectiveValue]) -> bool {
    if left.len() != right.len() {
        return false;
    }
    let mut strictly_better = false;
    for (l, r) in left.iter().zip(right) {
        if l.objective_id != r.objective_id || l.direction != r.direction {
            return false;
        }
        if l.conservative_value > r.conservative_value {
            return false;
        }
        if l.conservative_value < r.conservative_value {
            strictly_better = true;
        }
    }
    strictly_better
}

fn evaluate_candidate(case: &Case, candidate: &Candidate) -> CandidateReport {
    let mut decisions = Vec::new();
    let mut reasons = Vec::new();
    let m = &candidate.metrics;
    let mut has_fail = false;
    let mut has_uncertain = false;

    let mut apply = |result: GateResult| {
        has_fail |= result == GateResult::Fail;
        has_uncertain |= result == GateResult::Uncertain;
    };

    apply(push_min_constraint(
        &mut decisions, &mut reasons,
        "minimum_delivered_service_tonne_nm".to_string(),
        m.delivered_service_tonne_nm, m.delivered_service_u95_tonne_nm,
        case.limits.minimum_delivered_service_tonne_nm,
    ));
    apply(push_max_constraint(
        &mut decisions, &mut reasons,
        "maximum_voyage_duration_hours".to_string(),
        m.voyage_duration_hours, m.voyage_duration_u95_hours,
        case.limits.maximum_voyage_duration_hours.min(case.scenario.maximum_delivery_window_hours),
    ));
    apply(push_max_constraint(
        &mut decisions, &mut reasons,
        "maximum_primary_energy_kwh_per_tonne_nm".to_string(),
        m.primary_energy_kwh_per_tonne_nm, m.primary_energy_u95_kwh_per_tonne_nm,
        case.limits.maximum_primary_energy_kwh_per_tonne_nm,
    ));
    apply(push_max_constraint(
        &mut decisions, &mut reasons,
        "maximum_well_to_wake_ghg_gco2e_per_tonne_nm".to_string(),
        m.well_to_wake_ghg_gco2e_per_tonne_nm, m.well_to_wake_ghg_u95_gco2e_per_tonne_nm,
        case.limits.maximum_well_to_wake_ghg_gco2e_per_tonne_nm,
    ));
    apply(push_max_constraint(
        &mut decisions, &mut reasons,
        "maximum_lifecycle_cost_usd_per_tonne_nm".to_string(),
        m.lifecycle_cost_usd_per_tonne_nm, m.lifecycle_cost_u95_usd_per_tonne_nm,
        case.limits.maximum_lifecycle_cost_usd_per_tonne_nm,
    ));
    for limit in &case.limits.source_level_limits {
        let band = m.source_levels.iter().find(|item| item.band_id == limit.band_id)
            .expect("source-level band coverage validated");
        apply(push_max_constraint(
            &mut decisions, &mut reasons,
            format!("maximum_source_level_db_re_1upa_at_1m:{}", limit.band_id),
            band.source_level_db, band.expanded_uncertainty_db, limit.maximum_source_level_db,
        ));
    }

    let evidence_eligible = matches!(
        candidate.evidence_kind,
        EvidenceKind::MeasuredFullScale | EvidenceKind::IndependentlyValidatedTransfer
    ) && matches!(
        case.evidence_kind,
        EvidenceKind::MeasuredFullScale | EvidenceKind::IndependentlyValidatedTransfer
    ) && candidate.independently_reviewed
        && case.rights_status == RightsStatus::PublicOrPermitted
        && case.independently_reviewed;

    let disposition = if has_fail && evidence_eligible {
        "INFEASIBLE"
    } else if has_fail || has_uncertain || !evidence_eligible {
        if !evidence_eligible {
            reasons.push("candidate evidence is synthetic/prediction-only, not independently reviewed, or lacks public/permitted source rights".to_string());
        }
        "INCOMPLETE"
    } else {
        "FEASIBLE"
    };

    let mut constraint_decisions = decisions;
    constraint_decisions.sort_by(|left, right| left.constraint_id.cmp(&right.constraint_id));
    reasons.sort();
    reasons.dedup();

    CandidateReport {
        candidate_id: candidate.candidate_id.clone(),
        policy_label: candidate.policy_label.clone(),
        operating_point_id: candidate.operating_point_id.clone(),
        scenario_fingerprint_sha256: candidate.scenario_fingerprint_sha256.clone(),
        method_id: candidate.method_id.clone(),
        evidence_review_receipt_sha256: candidate.evidence_review_receipt_sha256.clone(),
        operating_point_receipt_sha256: candidate.operating_point_receipt_sha256.clone(),
        raw_measurement_or_validation_sha256: candidate.raw_measurement_or_validation_sha256.clone(),
        propeller_rpm: candidate.propeller_rpm,
        vessel_speed_knots: candidate.vessel_speed_knots,
        evidence_kind: evidence_name(&candidate.evidence_kind),
        source_level_evidence: candidate.metrics.source_levels.iter().map(|source| SourceLevelEvidenceReceipt {
            band_id: source.band_id.clone(),
            quantity_id: source.quantity_id.clone(),
            source_level_db: source.source_level_db,
            expanded_uncertainty_db: source.expanded_uncertainty_db,
            method_id: source.method_id.clone(),
            evidence_sha256: source.evidence_sha256.clone(),
        }).collect(),
        disposition,
        constraint_decisions,
        robust_objectives: robust_objectives(candidate),
        pareto_status: "not_in_frontier",
        reasons,
    }
}

fn audit(case: &Case) -> AuditReport {
    let validation_errors = validate_case(case);
    let mut reasons = validation_errors.clone();
    let mut candidate_reports: Vec<CandidateReport> = if validation_errors.is_empty() {
        case.candidates.iter().map(|candidate| evaluate_candidate(case, candidate)).collect()
    } else {
        Vec::new()
    };
    candidate_reports.sort_by(|left, right| left.candidate_id.cmp(&right.candidate_id));

    let feasible_indices: Vec<usize> = candidate_reports.iter().enumerate()
        .filter(|(_, report)| report.disposition == "FEASIBLE")
        .map(|(index, _)| index)
        .collect();
    let mut frontier = Vec::new();
    for &candidate_index in &feasible_indices {
        let is_dominated = feasible_indices.iter().any(|&other_index| {
            other_index != candidate_index
                && dominates(
                    &candidate_reports[other_index].robust_objectives,
                    &candidate_reports[candidate_index].robust_objectives,
                )
        });
        if is_dominated {
            candidate_reports[candidate_index].pareto_status = "dominated";
        } else {
            candidate_reports[candidate_index].pareto_status = "pareto_frontier";
            frontier.push(candidate_reports[candidate_index].candidate_id.clone());
        }
    }
    frontier.sort();

    let incomplete_count = candidate_reports.iter().filter(|report| report.disposition == "INCOMPLETE").count();
    let infeasible_count = candidate_reports.iter().filter(|report| report.disposition == "INFEASIBLE").count();
    let disposition = if !validation_errors.is_empty() {
        "INVALID"
    } else if frontier.is_empty() {
        "NO_ROBUST_FEASIBLE_CANDIDATE"
    } else {
        "PARETO_SET_COMPUTED"
    };
    if incomplete_count > 0 {
        reasons.push(format!("{incomplete_count} candidates are incomplete and excluded from the Pareto frontier"));
    }
    if infeasible_count > 0 {
        reasons.push(format!("{infeasible_count} candidates are definitely infeasible and excluded from the Pareto frontier"));
    }
    if case.evidence_kind == EvidenceKind::Synthetic {
        reasons.push("case dataset is synthetic; no physical policy selection is qualified".to_string());
    }
    reasons.sort();
    reasons.dedup();

    let objective_dimensions = if let Some(first) = candidate_reports.first() {
        first.robust_objectives.iter().map(|objective| objective.objective_id.clone()).collect()
    } else {
        Vec::new()
    };

    AuditReport {
        schema_version: SCHEMA_VERSION,
        case_id: case.case_id.clone(),
        dataset_id: case.source.dataset_id.clone(),
        source_data_sha256: case.source.raw_data_sha256.clone(),
        rights_decision_reference: case.source.rights_decision_reference.clone(),
        evidence_kind: evidence_name(&case.evidence_kind),
        rights_status: match case.rights_status {
            RightsStatus::PublicOrPermitted => "public_or_permitted",
            RightsStatus::Unknown => "unknown",
            RightsStatus::Restricted => "restricted",
        },
        independently_reviewed: case.independently_reviewed,
        vessel_id: case.scenario.vessel_id.clone(),
        vessel_configuration_sha256: case.scenario.vessel_configuration_sha256.clone(),
        propeller_design_id: case.scenario.propeller_design_id.clone(),
        propeller_revision: case.scenario.propeller_revision.clone(),
        route_id: case.scenario.route_id.clone(),
        environment_fingerprint_sha256: case.scenario.environment_fingerprint_sha256.clone(),
        hull_state_sha256: case.scenario.hull_state_sha256.clone(),
        service_fingerprint_sha256: case.scenario.service_fingerprint_sha256.clone(),
        scenario_fingerprint_sha256: case.scenario.scenario_fingerprint_sha256.clone(),
        disposition,
        candidate_reports,
        feasible_pareto_frontier_candidate_ids: frontier,
        excluded_candidate_count: incomplete_count + infeasible_count,
        objective_dimensions,
        selection_rule: "noncompensable_robust_constraints_then_nondominated_frontier_using_conservative_upper_bounds_no_weighted_sum",
        claim_ceiling: "declared_operating_point_robust_gate_and_pareto_computation_only_no_real_ship_optimization_source_propagation_ecological_benefit_or_vessel_certification",
        reasons,
    }
}

fn main() {
    let mut args = env::args().skip(1);
    let Some(path) = args.next() else {
        eprintln!("usage: marine-robust-operating-policy-audit <case.json>");
        process::exit(64);
    };
    if args.next().is_some() {
        eprintln!("usage: marine-robust-operating-policy-audit <case.json>");
        process::exit(64);
    }
    let input = match fs::read_to_string(&path) {
        Ok(value) => value,
        Err(error) => {
            eprintln!("could not read {path}: {error}");
            process::exit(64);
        }
    };
    let case: Case = match serde_json::from_str(&input) {
        Ok(value) => value,
        Err(error) => {
            eprintln!("invalid case JSON: {error}");
            process::exit(64);
        }
    };
    let report = audit(&case);
    println!("{}", serde_json::to_string_pretty(&report).expect("report serializes"));
    match report.disposition {
        "PARETO_SET_COMPUTED" => {}
        "INVALID" => process::exit(2),
        _ => process::exit(3),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(c: char) -> String {
        std::iter::repeat_n(c, 64).collect()
    }

    struct CandidateTestValues {
        energy: f64,
        ghg: f64,
        duration: f64,
        cost: f64,
        noise_1k: f64,
        uncertainty: f64,
    }

    fn candidate(id: &str, kind: EvidenceKind, values: CandidateTestValues) -> Candidate {
        Candidate {
            candidate_id: id.to_string(),
            policy_label: format!("policy-{id}"),
            operating_point_id: format!("op-{id}"),
            scenario_fingerprint_sha256: digest('2'),
            evidence_kind: kind,
            independently_reviewed: true,
            evidence_review_receipt_sha256: digest('3'),
            operating_point_receipt_sha256: digest('4'),
            raw_measurement_or_validation_sha256: digest('5'),
            method_id: "independent-trial-or-validated-transfer".to_string(),
            propeller_rpm: 80.0,
            vessel_speed_knots: 12.0,
            metrics: CandidateMetrics {
                delivered_service_tonne_nm: 100_000.0,
                delivered_service_u95_tonne_nm: 100.0,
                voyage_duration_hours: values.duration,
                voyage_duration_u95_hours: 0.2,
                primary_energy_kwh_per_tonne_nm: values.energy,
                primary_energy_u95_kwh_per_tonne_nm: values.uncertainty,
                well_to_wake_ghg_gco2e_per_tonne_nm: values.ghg,
                well_to_wake_ghg_u95_gco2e_per_tonne_nm: values.uncertainty,
                lifecycle_cost_usd_per_tonne_nm: values.cost,
                lifecycle_cost_u95_usd_per_tonne_nm: values.uncertainty,
                source_levels: vec![
                    SourceLevel {
                        band_id: "one-third-octave-1000hz".to_string(),
                        quantity_id: SOURCE_LEVEL_QUANTITY.to_string(),
                        source_level_db: values.noise_1k,
                        expanded_uncertainty_db: 0.2,
                        method_id: "independent-source-level-method".to_string(),
                        evidence_sha256: digest('6'),
                    },
                    SourceLevel {
                        band_id: "one-third-octave-2000hz".to_string(),
                        quantity_id: SOURCE_LEVEL_QUANTITY.to_string(),
                        source_level_db: values.noise_1k - 2.0,
                        expanded_uncertainty_db: 0.2,
                        method_id: "independent-source-level-method".to_string(),
                        evidence_sha256: digest('7'),
                    },
                ],
            },
        }
    }

    fn case() -> Case {
        Case {
            schema_version: SCHEMA_VERSION.to_string(),
            case_id: "synthetic-policy-001".to_string(),
            source: DatasetEvidence {
                dataset_id: "synthetic-policy-dataset".to_string(),
                citation: "Synthetic known-answer fixture only".to_string(),
                raw_data_sha256: digest('1'),
                rights_decision_reference: "synthetic-only-no-license-claim".to_string(),
            },
            evidence_kind: EvidenceKind::Synthetic,
            rights_status: RightsStatus::PublicOrPermitted,
            independently_reviewed: true,
            scenario: ServiceScenario {
                vessel_id: "V-001".to_string(),
                vessel_configuration_sha256: digest('8'),
                scenario_fingerprint_sha256: digest('2'),
                propeller_design_id: "P-001".to_string(),
                propeller_revision: "rev-7".to_string(),
                service_fingerprint_sha256: digest('2'),
                environment_fingerprint_sha256: digest('9'),
                hull_state_sha256: digest('a'),
                route_id: "route-001".to_string(),
                draft_m: 11.0,
                cargo_mass_tonnes: 80_000.0,
                delivered_service_tonne_nm: 100_000.0,
                maximum_delivery_window_hours: 120.0,
            },
            limits: PolicyLimits {
                minimum_delivered_service_tonne_nm: 99_000.0,
                maximum_voyage_duration_hours: 120.0,
                maximum_primary_energy_kwh_per_tonne_nm: 0.20,
                maximum_well_to_wake_ghg_gco2e_per_tonne_nm: 0.08,
                maximum_lifecycle_cost_usd_per_tonne_nm: 0.05,
                source_level_limits: vec![
                    SourceLevelLimit {
                        band_id: "one-third-octave-1000hz".to_string(),
                        quantity_id: SOURCE_LEVEL_QUANTITY.to_string(),
                        maximum_source_level_db: 130.0,
                    },
                    SourceLevelLimit {
                        band_id: "one-third-octave-2000hz".to_string(),
                        quantity_id: SOURCE_LEVEL_QUANTITY.to_string(),
                        maximum_source_level_db: 130.0,
                    },
                ],
            },
            candidates: vec![
                candidate("A", EvidenceKind::Synthetic, CandidateTestValues {
                    energy: 0.14, ghg: 0.06, duration: 100.0, cost: 0.04, noise_1k: 120.0, uncertainty: 0.001,
                }),
                candidate("B", EvidenceKind::Synthetic, CandidateTestValues {
                    energy: 0.16, ghg: 0.05, duration: 105.0, cost: 0.045, noise_1k: 118.0, uncertainty: 0.001,
                }),
                candidate("C", EvidenceKind::Synthetic, CandidateTestValues {
                    energy: 0.18, ghg: 0.07, duration: 110.0, cost: 0.048, noise_1k: 125.0, uncertainty: 0.001,
                }),
            ],
        }
    }

    #[test]
    fn operating_policy_receipt_preserves_scenario_and_band_evidence_lineage() {
        let input = case();
        let report = audit(&input);
        assert_eq!(report.source_data_sha256, input.source.raw_data_sha256);
        assert_eq!(report.vessel_id, input.scenario.vessel_id);
        assert_eq!(report.propeller_revision, input.scenario.propeller_revision);
        assert_eq!(report.scenario_fingerprint_sha256, input.scenario.scenario_fingerprint_sha256);
        let candidate = report.candidate_reports.iter().find(|x| x.candidate_id == "A").unwrap();
        assert_eq!(candidate.operating_point_receipt_sha256, input.candidates[0].operating_point_receipt_sha256);
        assert_eq!(candidate.raw_measurement_or_validation_sha256, input.candidates[0].raw_measurement_or_validation_sha256);
        assert_eq!(candidate.source_level_evidence.len(), 2);
        assert_eq!(candidate.source_level_evidence[0].evidence_sha256, input.candidates[0].metrics.source_levels[0].evidence_sha256);
    }

    #[test]
    fn committed_schema_is_valid_json_and_binds_source_level_limits() {
        let schema: serde_json::Value = serde_json::from_str(include_str!(
            "../../../docs/research/marine/marine-robust-operating-policy-v0.1.0.schema.json"
        )).expect("schema is valid JSON");
        assert_eq!(
            schema["$schema"],
            "https://json-schema.org/draft/2020-12/schema"
        );
        assert_eq!(schema["additionalProperties"], false);
        assert_eq!(
            schema["$defs"]["sourceLevel"]["properties"]["quantity_id"]["const"],
            SOURCE_LEVEL_QUANTITY
        );
        assert_eq!(
            schema["$defs"]["candidate"]["properties"]["scenario_fingerprint_sha256"]["$ref"],
            "#/$defs/digest"
        );
        assert!(schema["$defs"]["candidateMetrics"]["required"]
            .as_array().unwrap().iter().any(|value| value == "source_levels"));
    }

    #[test]
    fn pareto_objectives_use_conservative_upper_bounds() {
        let input = case();
        let objectives = robust_objectives(&input.candidates[0]);
        let robust_energy = objectives
            .iter()
            .find(|x| x.objective_id == "primary_energy_kwh_per_tonne_nm")
            .unwrap()
            .conservative_value;
        assert!((robust_energy - 0.141).abs() < 1.0e-12);
        assert!(objectives.iter().any(|x| x.objective_id == "source_level_db_re_1upa_at_1m:one-third-octave-1000hz"));
    }

    #[test]
    fn synthetic_fixture_never_selects_a_physical_policy() {
        let report = audit(&case());
        assert_eq!(report.disposition, "NO_ROBUST_FEASIBLE_CANDIDATE");
        assert!(report.feasible_pareto_frontier_candidate_ids.is_empty());
        assert!(report.candidate_reports.iter().all(|candidate| candidate.disposition == "INCOMPLETE"));
    }

    #[test]
    fn robust_upper_bound_not_nominal_value_controls_maximum_gate() {
        assert_eq!(max_gate(9.0, 0.5, 10.0), GateResult::Pass);
        assert_eq!(max_gate(9.8, 0.5, 10.0), GateResult::Uncertain);
        assert_eq!(max_gate(10.6, 0.5, 10.0), GateResult::Fail);
    }

    #[test]
    fn robust_lower_bound_controls_minimum_service_gate() {
        assert_eq!(min_gate(10.5, 0.5, 10.0), GateResult::Pass);
        assert_eq!(min_gate(10.2, 0.5, 10.0), GateResult::Uncertain);
        assert_eq!(min_gate(9.0, 0.5, 10.0), GateResult::Fail);
    }

    #[test]
    fn energy_saving_cannot_compensate_for_acoustic_limit_failure() {
        let mut input = case();
        input.evidence_kind = EvidenceKind::MeasuredFullScale;
        input.rights_status = RightsStatus::PublicOrPermitted;
        input.candidates[0].evidence_kind = EvidenceKind::MeasuredFullScale;
        input.candidates[0].metrics.source_levels[0].source_level_db = 131.0;
        let report = audit(&input);
        let candidate = report.candidate_reports.iter().find(|x| x.candidate_id == "A").unwrap();
        assert_eq!(candidate.disposition, "INFEASIBLE");
        assert!(candidate.constraint_decisions.iter().any(|x| x.constraint_id.contains("source_level") && x.result == "fail"));
    }

    #[test]
    fn uncertainty_crossing_a_hard_limit_is_incomplete_not_feasible() {
        let mut input = case();
        input.candidates[0].metrics.primary_energy_kwh_per_tonne_nm = 0.1995;
        input.candidates[0].metrics.primary_energy_u95_kwh_per_tonne_nm = 0.001;
        let report = audit(&input);
        assert_eq!(report.candidate_reports[0].disposition, "INCOMPLETE");
    }

    #[test]
    fn missing_required_frequency_band_invalidates_candidate_set() {
        let mut input = case();
        input.candidates[1].metrics.source_levels.pop();
        assert_eq!(audit(&input).disposition, "INVALID");
    }

    #[test]
    fn received_level_cannot_be_substituted_for_source_level() {
        let mut input = case();
        input.candidates[0].metrics.source_levels[0].quantity_id = "received_spl_db_re_1upa".to_string();
        assert_eq!(audit(&input).disposition, "INVALID");
    }

    #[test]
    fn changed_service_configuration_is_incomparable() {
        let mut input = case();
        input.candidates[0].scenario_fingerprint_sha256 = digest('f');
        assert_eq!(audit(&input).disposition, "INVALID");
    }

    #[test]
    fn dominated_feasible_candidate_is_removed_from_frontier() {
        let mut input = case();
        input.evidence_kind = EvidenceKind::MeasuredFullScale;
        input.rights_status = RightsStatus::PublicOrPermitted;
        input.candidates[0].evidence_kind = EvidenceKind::MeasuredFullScale;
        input.candidates[1].evidence_kind = EvidenceKind::MeasuredFullScale;
        input.candidates[2].evidence_kind = EvidenceKind::MeasuredFullScale;
        let report = audit(&input);
        assert!(!report.feasible_pareto_frontier_candidate_ids.contains(&"C".to_string()));
        assert_eq!(
            report.candidate_reports.iter().find(|x| x.candidate_id == "C").unwrap().pareto_status,
            "dominated"
        );
    }

    #[test]
    fn incomparable_energy_and_noise_tradeoff_remains_on_frontier() {
        let mut input = case();
        input.evidence_kind = EvidenceKind::MeasuredFullScale;
        input.rights_status = RightsStatus::PublicOrPermitted;
        input.candidates[0].evidence_kind = EvidenceKind::MeasuredFullScale;
        input.candidates[1].evidence_kind = EvidenceKind::MeasuredFullScale;
        let report = audit(&input);
        assert!(report.feasible_pareto_frontier_candidate_ids.contains(&"A".to_string()));
        assert!(report.feasible_pareto_frontier_candidate_ids.contains(&"B".to_string()));
    }

    #[test]
    fn deadline_and_cargo_service_are_hard_noncompensable_gates() {
        let mut input = case();
        input.evidence_kind = EvidenceKind::MeasuredFullScale;
        input.rights_status = RightsStatus::PublicOrPermitted;
        input.candidates[0].evidence_kind = EvidenceKind::MeasuredFullScale;
        input.candidates[0].metrics.voyage_duration_hours = 121.0;
        input.candidates[1].evidence_kind = EvidenceKind::MeasuredFullScale;
        input.candidates[1].metrics.delivered_service_tonne_nm = 90_000.0;
        let report = audit(&input);
        assert_eq!(report.candidate_reports[0].disposition, "INFEASIBLE");
        assert_eq!(report.candidate_reports[1].disposition, "INFEASIBLE");
    }

    #[test]
    fn equality_at_a_declared_robust_limit_is_feasible() {
        assert_eq!(max_gate(9.5, 0.5, 10.0), GateResult::Pass);
        assert_eq!(min_gate(10.5, 0.5, 10.0), GateResult::Pass);
    }

    #[test]
    fn physical_evidence_requires_both_case_and_candidate_status() {
        let mut input = case();
        input.evidence_kind = EvidenceKind::MeasuredFullScale;
        input.rights_status = RightsStatus::PublicOrPermitted;
        input.candidates[0].evidence_kind = EvidenceKind::MeasuredFullScale;
        input.candidates[1].evidence_kind = EvidenceKind::MeasuredFullScale;
        input.candidates[2].evidence_kind = EvidenceKind::PredictionOnly;
        let report = audit(&input);
        assert_eq!(report.candidate_reports.iter().find(|x| x.candidate_id == "A").unwrap().disposition, "FEASIBLE");
        assert_eq!(report.candidate_reports.iter().find(|x| x.candidate_id == "C").unwrap().disposition, "INCOMPLETE");
    }

    #[test]
    fn identical_policies_are_both_preserved_as_nondominated() {
        let mut input = case();
        input.evidence_kind = EvidenceKind::MeasuredFullScale;
        input.rights_status = RightsStatus::PublicOrPermitted;
        input.candidates[0].evidence_kind = EvidenceKind::MeasuredFullScale;
        input.candidates[1] = candidate(
            "A-copy",
            EvidenceKind::MeasuredFullScale,
            CandidateTestValues {
                energy: 0.14, ghg: 0.06, duration: 100.0, cost: 0.04, noise_1k: 120.0, uncertainty: 0.001,
            },
        );
        let report = audit(&input);
        assert!(report.feasible_pareto_frontier_candidate_ids.contains(&"A".to_string()));
        assert!(report.feasible_pareto_frontier_candidate_ids.contains(&"A-copy".to_string()));
    }

    #[test]
    fn no_weighted_sum_or_single_score_is_emitted() {
        let report = audit(&case());
        let serialized = serde_json::to_string(&report).unwrap();
        assert!(!serialized.contains("weighted_score"));
        assert!(serialized.contains("robust_objectives"));
        assert!(serialized.contains("noncompensable_robust_constraints_then_nondominated_frontier"));
    }

    #[test]
    fn permutation_preserves_candidate_reports_and_frontier() {
        let mut input = case();
        input.candidates.reverse();
        let first = audit(&case());
        let second = audit(&input);
        assert_eq!(first.feasible_pareto_frontier_candidate_ids, second.feasible_pareto_frontier_candidate_ids);
        assert_eq!(
            first.candidate_reports.iter().map(|x| &x.candidate_id).collect::<Vec<_>>(),
            second.candidate_reports.iter().map(|x| &x.candidate_id).collect::<Vec<_>>()
        );
    }
}

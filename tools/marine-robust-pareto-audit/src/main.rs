use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::{env, fs, process};

const SCHEMA_VERSION: &str = "0.1.0";

#[derive(Debug, Deserialize, Serialize, PartialEq, Eq, Clone, Copy)]
#[serde(rename_all = "snake_case")]
enum EvidenceKind {
    Synthetic,
    Measured,
}

#[derive(Debug, Deserialize, Serialize, PartialEq, Eq, Clone, Copy)]
#[serde(rename_all = "snake_case")]
enum RightsStatus {
    PublicOrPermitted,
    Unknown,
    Restricted,
}

#[derive(Debug, Deserialize, Serialize, PartialEq, Eq, Clone, Copy)]
#[serde(rename_all = "snake_case")]
enum Direction {
    Minimize,
    Maximize,
}

#[derive(Debug, Deserialize, Serialize, PartialEq, Eq, Clone, Copy)]
#[serde(rename_all = "snake_case")]
enum GateEvaluation {
    MetricUpperBound,
    MetricLowerBound,
    ExternalAuditReceipt,
}

#[derive(Debug, Deserialize, Serialize, PartialEq, Eq, Clone, Copy)]
#[serde(rename_all = "snake_case")]
enum GateStatus {
    Satisfied,
    Failed,
    Unresolved,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Case {
    schema_version: String,
    case_id: String,
    source_register: SourceRegister,
    service: ServiceContract,
    objectives: Vec<ObjectiveDefinition>,
    hard_gates: Vec<HardGateDefinition>,
    candidates: Vec<Candidate>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SourceRegister {
    register_id: String,
    citation: String,
    register_sha256: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ServiceContract {
    mission_id: String,
    vessel_id: String,
    route_id: String,
    measurement_window_id: String,
    service_fingerprint_sha256: String,
    environment_fingerprint_sha256: String,
    cargo_work_unit: String,
    reference_cargo_work: f64,
    reference_cargo_work_expanded_uncertainty: f64,
    cargo_work_equivalence_tolerance: f64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ObjectiveDefinition {
    metric_id: String,
    direction: Direction,
    unit: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct HardGateDefinition {
    gate_id: String,
    gate_class: String,
    evaluation: GateEvaluation,
    metric_id: Option<String>,
    threshold: Option<f64>,
    unit: Option<String>,
    required_audit_contract_id: Option<String>,
    requirement_reference: String,
    non_compensable: bool,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Candidate {
    candidate_id: String,
    evidence_kind: EvidenceKind,
    dataset_id: String,
    citation: String,
    raw_data_sha256: String,
    rights_status: RightsStatus,
    independently_reviewed: bool,
    mission_id: String,
    vessel_id: String,
    route_id: String,
    measurement_window_id: String,
    service_fingerprint_sha256: String,
    environment_fingerprint_sha256: String,
    vessel_revision: String,
    propeller_revision: String,
    cargo_work_value: f64,
    cargo_work_expanded_uncertainty: f64,
    cargo_work_unit: String,
    operating_point_receipt_sha256: String,
    metrics: Vec<MetricObservation>,
    gate_receipts: Vec<GateReceipt>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(deny_unknown_fields)]
struct MetricObservation {
    metric_id: String,
    value: f64,
    expanded_uncertainty: f64,
    confidence_level: f64,
    unit: String,
    method_id: String,
    raw_data_sha256: String,
    calibration_or_validation_sha256: String,
    timebase_receipt_sha256: String,
    operating_point_receipt_sha256: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct GateReceipt {
    gate_id: String,
    status: GateStatus,
    audit_contract_id: String,
    audit_receipt_sha256: String,
    requirement_reference: String,
}

#[derive(Debug, Serialize)]
struct AuditReport {
    schema_version: &'static str,
    case_id: String,
    source_register_id: String,
    source_register_sha256: String,
    source_register_citation: String,
    disposition: &'static str,
    cargo_work_unit: String,
    candidate_reports: Vec<CandidateReport>,
    pairwise_relations: Vec<PairwiseRelation>,
    robust_frontier: Vec<FrontierEntry>,
    claim_ceiling: &'static str,
    reasons: Vec<String>,
}

#[derive(Debug, Serialize)]
struct CandidateReport {
    candidate_id: String,
    dataset_id: String,
    raw_data_sha256: String,
    evidence_kind: &'static str,
    rights_status: &'static str,
    independently_reviewed: bool,
    service_fingerprint_sha256: String,
    environment_fingerprint_sha256: String,
    operating_point_receipt_sha256: String,
    vessel_revision: String,
    propeller_revision: String,
    objective_intervals: Vec<MetricIntervalReport>,
    disposition: &'static str,
    cargo_work_difference_lower: Option<f64>,
    cargo_work_difference_upper: Option<f64>,
    gate_results: Vec<GateResult>,
    reasons: Vec<String>,
}

#[derive(Debug, Serialize)]
struct MetricIntervalReport {
    metric_id: String,
    value: f64,
    lower: f64,
    upper: f64,
    expanded_uncertainty: f64,
    confidence_level: f64,
    unit: String,
    method_id: String,
    raw_data_sha256: String,
    calibration_or_validation_sha256: String,
    timebase_receipt_sha256: String,
    operating_point_receipt_sha256: String,
}

#[derive(Debug, Serialize)]
struct GateResult {
    gate_id: String,
    gate_class: String,
    evaluation: &'static str,
    status: &'static str,
    observed_lower: Option<f64>,
    observed_upper: Option<f64>,
    threshold: Option<f64>,
    unit: Option<String>,
    audit_contract_id: Option<String>,
    audit_receipt_sha256: Option<String>,
    requirement_reference: String,
}

#[derive(Debug, Serialize)]
struct PairwiseRelation {
    candidate_a: String,
    candidate_b: String,
    relation_from_a: &'static str,
    objective_evidence: Vec<ObjectivePairEvidence>,
}

#[derive(Debug, Serialize)]
struct ObjectivePairEvidence {
    metric_id: String,
    direction: &'static str,
    a_lower: f64,
    a_upper: f64,
    b_lower: f64,
    b_upper: f64,
    a_definitely_better: bool,
    b_definitely_better: bool,
    intervals_overlap_or_touch: bool,
}

#[derive(Debug, Serialize)]
struct FrontierEntry {
    candidate_id: String,
    certainty: &'static str,
    unresolved_opponents: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GateDecision {
    Pass,
    Fail,
    Unresolved,
    Missing,
}

impl GateDecision {
    fn name(self) -> &'static str {
        match self {
            Self::Pass => "pass",
            Self::Fail => "fail",
            Self::Unresolved => "unresolved",
            Self::Missing => "missing",
        }
    }
}

#[derive(Debug)]
struct EvaluatedCandidate {
    report: CandidateReport,
    disposition: &'static str,
    metrics: BTreeMap<String, MetricObservation>,
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn finite_positive(value: f64) -> bool {
    value.is_finite() && value > 0.0
}

fn evaluation_name(value: GateEvaluation) -> &'static str {
    match value {
        GateEvaluation::MetricUpperBound => "metric_upper_bound",
        GateEvaluation::MetricLowerBound => "metric_lower_bound",
        GateEvaluation::ExternalAuditReceipt => "external_audit_receipt",
    }
}

fn validate_case(case: &Case) -> Vec<String> {
    let mut errors = Vec::new();
    if case.schema_version != SCHEMA_VERSION {
        errors.push(format!("unsupported schema_version {}; expected {SCHEMA_VERSION}", case.schema_version));
    }
    if case.case_id.trim().is_empty()
        || case.source_register.register_id.trim().is_empty()
        || case.source_register.citation.trim().is_empty()
        || case.service.mission_id.trim().is_empty()
        || case.service.vessel_id.trim().is_empty()
        || case.service.route_id.trim().is_empty()
        || case.service.measurement_window_id.trim().is_empty()
        || case.service.cargo_work_unit.trim().is_empty()
    {
        errors.push("case, source-register, mission, vessel, route, window and cargo-work identifiers are required".to_string());
    }
    for (label, digest) in [
        ("source-register", &case.source_register.register_sha256),
        ("service fingerprint", &case.service.service_fingerprint_sha256),
        ("environment fingerprint", &case.service.environment_fingerprint_sha256),
    ] {
        if !is_sha256(digest) {
            errors.push(format!("{label} digest must be SHA-256 hexadecimal"));
        }
    }
    if !finite_positive(case.service.reference_cargo_work)
        || !finite_positive(case.service.reference_cargo_work_expanded_uncertainty)
        || !finite_positive(case.service.cargo_work_equivalence_tolerance)
    {
        errors.push("reference cargo-work, expanded uncertainty and equivalence tolerance must be finite and positive".to_string());
    }
    if case.objectives.is_empty() || case.hard_gates.is_empty() || case.candidates.len() < 2 {
        errors.push("at least one objective, one non-compensable hard gate and two candidates are required".to_string());
    }

    let mut objective_ids = BTreeSet::new();
    for objective in &case.objectives {
        if objective.metric_id.trim().is_empty()
            || objective.unit.trim().is_empty()
            || !objective_ids.insert(objective.metric_id.clone())
        {
            errors.push("objective metric IDs must be non-empty and unique and objective units must be declared".to_string());
        }
    }

    let mut gate_ids = BTreeSet::new();
    let mut required_metric_ids: BTreeSet<String> = objective_ids.clone();
    for gate in &case.hard_gates {
        if gate.gate_id.trim().is_empty()
            || gate.gate_class.trim().is_empty()
            || gate.requirement_reference.trim().is_empty()
            || !gate_ids.insert(gate.gate_id.clone())
        {
            errors.push("hard-gate IDs/classes/requirement references must be non-empty and IDs unique".to_string());
        }
        if !gate.non_compensable {
            errors.push(format!("gate {} is not marked non_compensable", gate.gate_id));
        }
        match gate.evaluation {
            GateEvaluation::MetricUpperBound | GateEvaluation::MetricLowerBound => {
                if gate.metric_id.as_deref().unwrap_or("").trim().is_empty()
                    || gate.threshold.is_none_or(|value| !value.is_finite())
                    || gate.unit.as_deref().unwrap_or("").trim().is_empty()
                    || gate.required_audit_contract_id.is_some()
                {
                    errors.push(format!("metric gate {} requires metric_id, finite threshold and unit, and no external audit contract", gate.gate_id));
                } else if let Some(metric_id) = &gate.metric_id {
                    required_metric_ids.insert(metric_id.clone());
                }
            }
            GateEvaluation::ExternalAuditReceipt => {
                if gate.required_audit_contract_id.as_deref().unwrap_or("").trim().is_empty()
                    || gate.metric_id.is_some()
                    || gate.threshold.is_some()
                    || gate.unit.is_some()
                {
                    errors.push(format!("external audit gate {} requires an audit contract ID and must not define a metric threshold", gate.gate_id));
                }
            }
        }
    }

    let mut candidate_ids = BTreeSet::new();
    for candidate in &case.candidates {
        if candidate.candidate_id.trim().is_empty() || !candidate_ids.insert(candidate.candidate_id.clone()) {
            errors.push("candidate IDs must be non-empty and unique".to_string());
        }
        if candidate.dataset_id.trim().is_empty()
            || candidate.citation.trim().is_empty()
            || candidate.vessel_revision.trim().is_empty()
            || candidate.propeller_revision.trim().is_empty()
        {
            errors.push(format!("candidate {} lacks dataset citation or revision identity", candidate.candidate_id));
        }
        for (label, digest) in [
            ("raw data", &candidate.raw_data_sha256),
            ("operating-point receipt", &candidate.operating_point_receipt_sha256),
            ("service fingerprint", &candidate.service_fingerprint_sha256),
            ("environment fingerprint", &candidate.environment_fingerprint_sha256),
        ] {
            if !is_sha256(digest) {
                errors.push(format!("candidate {} has invalid {label} digest", candidate.candidate_id));
            }
        }
        if !finite_positive(candidate.cargo_work_value)
            || !finite_positive(candidate.cargo_work_expanded_uncertainty)
            || !(candidate.cargo_work_value - candidate.cargo_work_expanded_uncertainty).is_finite()
            || !(candidate.cargo_work_value + candidate.cargo_work_expanded_uncertainty).is_finite()
            || candidate.cargo_work_unit.trim().is_empty()
        {
            errors.push(format!("candidate {} has invalid cargo-work quantity or unit", candidate.candidate_id));
        }

        let mut metric_ids = BTreeSet::new();
        for metric in &candidate.metrics {
            if metric.metric_id.trim().is_empty()
                || !metric_ids.insert(metric.metric_id.clone())
                || metric.unit.trim().is_empty()
                || metric.method_id.trim().is_empty()
                || !metric.value.is_finite()
                || !finite_positive(metric.expanded_uncertainty)
                || !metric.confidence_level.is_finite()
                || metric.confidence_level <= 0.0
                || metric.confidence_level >= 1.0
                || !(metric.value - metric.expanded_uncertainty).is_finite()
                || !(metric.value + metric.expanded_uncertainty).is_finite()
            {
                errors.push(format!("candidate {} has malformed or duplicate metric observation", candidate.candidate_id));
            }
            for (label, digest) in [
                ("metric raw data", &metric.raw_data_sha256),
                ("metric calibration/validation", &metric.calibration_or_validation_sha256),
                ("metric timebase", &metric.timebase_receipt_sha256),
                ("metric operating-point", &metric.operating_point_receipt_sha256),
            ] {
                if !is_sha256(digest) {
                    errors.push(format!("candidate {} metric {} has invalid {label} digest", candidate.candidate_id, metric.metric_id));
                }
            }
            if metric.operating_point_receipt_sha256 != candidate.operating_point_receipt_sha256 {
                errors.push(format!("candidate {} metric {} is not bound to its candidate operating-point receipt", candidate.candidate_id, metric.metric_id));
            }
        }
        for expected in &required_metric_ids {
            if !metric_ids.contains(expected) {
                errors.push(format!("candidate {} is missing required metric {}", candidate.candidate_id, expected));
            }
        }
        for objective in &case.objectives {
            if let Some(observation) = candidate.metrics.iter().find(|metric| metric.metric_id == objective.metric_id)
                && observation.unit != objective.unit
            {
                errors.push(format!("candidate {} objective {} unit differs from its declared objective unit", candidate.candidate_id, objective.metric_id));
            }
        }
        for gate in &case.hard_gates {
            if let Some(metric_id) = &gate.metric_id
                && let Some(observation) = candidate.metrics.iter().find(|metric| &metric.metric_id == metric_id)
                && gate.unit.as_deref() != Some(observation.unit.as_str())
            {
                errors.push(format!("candidate {} metric gate {} unit differs from its declared gate unit", candidate.candidate_id, gate.gate_id));
            }
        }
        for observed in &metric_ids {
            if !required_metric_ids.contains(observed) {
                errors.push(format!("candidate {} supplies undeclared metric {}", candidate.candidate_id, observed));
            }
        }

        let mut receipts = BTreeSet::new();
        for receipt in &candidate.gate_receipts {
            if !receipts.insert(receipt.gate_id.clone())
                || receipt.audit_contract_id.trim().is_empty()
                || !is_sha256(&receipt.audit_receipt_sha256)
                || receipt.requirement_reference.trim().is_empty()
            {
                errors.push(format!("candidate {} has malformed or duplicate external gate receipt", candidate.candidate_id));
            }
        }
        for gate in &case.hard_gates {
            if gate.evaluation == GateEvaluation::ExternalAuditReceipt {
                let receipt = candidate.gate_receipts.iter().find(|receipt| receipt.gate_id == gate.gate_id);
                if receipt.is_none() {
                    errors.push(format!("candidate {} is missing external audit receipt for gate {}", candidate.candidate_id, gate.gate_id));
                } else if let Some(receipt) = receipt
                    && (Some(receipt.audit_contract_id.as_str()) != gate.required_audit_contract_id.as_deref()
                        || receipt.requirement_reference != gate.requirement_reference)
                {
                    errors.push(format!("candidate {} external receipt for gate {} does not match its required audit contract/requirement", candidate.candidate_id, gate.gate_id));
                }
            }
        }
        for receipt in &candidate.gate_receipts {
            if !case.hard_gates.iter().any(|gate| {
                gate.gate_id == receipt.gate_id && gate.evaluation == GateEvaluation::ExternalAuditReceipt
            }) {
                errors.push(format!("candidate {} supplies an undeclared external gate receipt {}", candidate.candidate_id, receipt.gate_id));
            }
        }
    }
    if case.candidates.len() >= 2 {
        for objective in &case.objectives {
            let baseline = case.candidates[0].metrics.iter()
                .find(|metric| metric.metric_id == objective.metric_id);
            if let Some(baseline) = baseline {
                for candidate in case.candidates.iter().skip(1) {
                    if let Some(observed) = candidate.metrics.iter()
                        .find(|metric| metric.metric_id == objective.metric_id)
                    {
                        if observed.method_id != baseline.method_id {
                            errors.push(format!(
                                "objective {} uses different measurement methods across candidates",
                                objective.metric_id
                            ));
                        }
                        if (observed.confidence_level - baseline.confidence_level).abs() > 1.0e-12 {
                            errors.push(format!(
                                "objective {} uses different expanded-uncertainty confidence levels across candidates",
                                objective.metric_id
                            ));
                        }
                    }
                }
            }
        }
    }

    errors
}

fn evaluate_candidate(case: &Case, candidate: &Candidate) -> EvaluatedCandidate {
    let mut reasons = Vec::new();
    let mut gate_results = Vec::new();
    let mut gate_failed = false;
    let mut gate_unresolved = false;
    let mut service_incomparable = false;
    let mut service_unresolved = false;
    let cargo_delta = (candidate.cargo_work_value - case.service.reference_cargo_work).abs();
    let cargo_uncertainty = candidate.cargo_work_expanded_uncertainty
        + case.service.reference_cargo_work_expanded_uncertainty;
    let cargo_lower = (cargo_delta - cargo_uncertainty).max(0.0);
    let cargo_upper = cargo_delta + cargo_uncertainty;

    if candidate.mission_id != case.service.mission_id
        || candidate.vessel_id != case.service.vessel_id
        || candidate.route_id != case.service.route_id
        || candidate.measurement_window_id != case.service.measurement_window_id
        || candidate.service_fingerprint_sha256 != case.service.service_fingerprint_sha256
        || candidate.environment_fingerprint_sha256 != case.service.environment_fingerprint_sha256
        || candidate.cargo_work_unit != case.service.cargo_work_unit
    {
        reasons.push("candidate mission, vessel, route, measurement window, service/environment fingerprint or cargo-work unit differs from the frozen comparison contract".to_string());
        service_incomparable = true;
    } else if cargo_lower > case.service.cargo_work_equivalence_tolerance {
        reasons.push("candidate delivered cargo-work is demonstrably outside the matched-service tolerance".to_string());
        service_incomparable = true;
    } else if cargo_upper > case.service.cargo_work_equivalence_tolerance {
        reasons.push("candidate cargo-work uncertainty does not establish matched service".to_string());
        service_unresolved = true;
    }

    if candidate.evidence_kind != EvidenceKind::Measured
        || candidate.rights_status != RightsStatus::PublicOrPermitted
        || !candidate.independently_reviewed
    {
        reasons.push("candidate evidence is synthetic, rights-restricted/unknown or not independently reviewed".to_string());
        gate_unresolved = true;
    }

    let metrics: BTreeMap<String, MetricObservation> = candidate.metrics.iter()
        .cloned()
        .map(|metric| (metric.metric_id.clone(), metric))
        .collect();

    for gate in &case.hard_gates {
        let (decision, lower, upper, receipt) = match gate.evaluation {
            GateEvaluation::MetricUpperBound | GateEvaluation::MetricLowerBound => {
                let metric = metrics.get(gate.metric_id.as_deref().unwrap_or(""));
                match metric {
                    Some(metric) => {
                        let lower = metric.value - metric.expanded_uncertainty;
                        let upper = metric.value + metric.expanded_uncertainty;
                        let threshold = gate.threshold.expect("validated metric gate has a threshold");
                        let decision = if gate.evaluation == GateEvaluation::MetricUpperBound {
                            if upper <= threshold {
                                GateDecision::Pass
                            } else if lower > threshold {
                                GateDecision::Fail
                            } else {
                                GateDecision::Unresolved
                            }
                        } else if lower >= threshold {
                            GateDecision::Pass
                        } else if upper < threshold {
                            GateDecision::Fail
                        } else {
                            GateDecision::Unresolved
                        };
                        (decision, Some(lower), Some(upper), None)
                    }
                    None => (GateDecision::Missing, None, None, None),
                }
            }
            GateEvaluation::ExternalAuditReceipt => {
                let gate_receipt = candidate.gate_receipts.iter().find(|receipt| receipt.gate_id == gate.gate_id);
                match gate_receipt {
                    Some(receipt) => {
                        let decision = match receipt.status {
                            GateStatus::Satisfied => GateDecision::Pass,
                            GateStatus::Failed => GateDecision::Fail,
                            GateStatus::Unresolved => GateDecision::Unresolved,
                        };
                        (decision, None, None, Some(receipt))
                    }
                    None => (GateDecision::Missing, None, None, None),
                }
            }
        };
        gate_failed |= decision == GateDecision::Fail;
        gate_unresolved |= decision == GateDecision::Unresolved || decision == GateDecision::Missing;
        if decision == GateDecision::Fail {
            reasons.push(format!("non-compensable hard gate {} failed", gate.gate_id));
        } else if decision == GateDecision::Unresolved || decision == GateDecision::Missing {
            reasons.push(format!("non-compensable hard gate {} is unresolved or missing", gate.gate_id));
        }
        gate_results.push(GateResult {
            gate_id: gate.gate_id.clone(),
            gate_class: gate.gate_class.clone(),
            evaluation: evaluation_name(gate.evaluation),
            status: decision.name(),
            observed_lower: lower,
            observed_upper: upper,
            threshold: gate.threshold,
            unit: gate.unit.clone(),
            audit_contract_id: receipt.map(|value| value.audit_contract_id.clone()),
            audit_receipt_sha256: receipt.map(|value| value.audit_receipt_sha256.clone()),
            requirement_reference: gate.requirement_reference.clone(),
        });
    }

    gate_results.sort_by(|left, right| left.gate_id.cmp(&right.gate_id));
    let disposition = if gate_failed {
        "EXCLUDED_HARD_GATE"
    } else if service_incomparable {
        "INCOMPARABLE_SERVICE"
    } else if service_unresolved || gate_unresolved {
        "HOLD_UNRESOLVED"
    } else {
        "ELIGIBLE_FOR_ROBUST_PARETO"
    };

    let objective_intervals = case.objectives.iter()
        .filter_map(|objective| metrics.get(&objective.metric_id))
        .map(|metric| MetricIntervalReport {
            metric_id: metric.metric_id.clone(),
            value: metric.value,
            lower: metric.value - metric.expanded_uncertainty,
            upper: metric.value + metric.expanded_uncertainty,
            expanded_uncertainty: metric.expanded_uncertainty,
            confidence_level: metric.confidence_level,
            unit: metric.unit.clone(),
            method_id: metric.method_id.clone(),
            raw_data_sha256: metric.raw_data_sha256.clone(),
            calibration_or_validation_sha256: metric.calibration_or_validation_sha256.clone(),
            timebase_receipt_sha256: metric.timebase_receipt_sha256.clone(),
            operating_point_receipt_sha256: metric.operating_point_receipt_sha256.clone(),
        })
        .collect();

    EvaluatedCandidate {
        report: CandidateReport {
            candidate_id: candidate.candidate_id.clone(),
            dataset_id: candidate.dataset_id.clone(),
            raw_data_sha256: candidate.raw_data_sha256.clone(),
            evidence_kind: match candidate.evidence_kind {
                EvidenceKind::Synthetic => "synthetic",
                EvidenceKind::Measured => "measured",
            },
            rights_status: match candidate.rights_status {
                RightsStatus::PublicOrPermitted => "public_or_permitted",
                RightsStatus::Unknown => "unknown",
                RightsStatus::Restricted => "restricted",
            },
            independently_reviewed: candidate.independently_reviewed,
            service_fingerprint_sha256: candidate.service_fingerprint_sha256.clone(),
            environment_fingerprint_sha256: candidate.environment_fingerprint_sha256.clone(),
            operating_point_receipt_sha256: candidate.operating_point_receipt_sha256.clone(),
            vessel_revision: candidate.vessel_revision.clone(),
            propeller_revision: candidate.propeller_revision.clone(),
            objective_intervals,
            disposition,
            cargo_work_difference_lower: Some(cargo_lower),
            cargo_work_difference_upper: Some(cargo_upper),
            gate_results,
            reasons,
        },
        disposition,
        metrics,
    }
}

fn compare_pair(
    candidate_a: &EvaluatedCandidate,
    candidate_b: &EvaluatedCandidate,
    objectives: &[ObjectiveDefinition],
) -> PairwiseRelation {
    let mut evidence = Vec::new();
    let mut a_no_worse_all = true;
    let mut b_no_worse_all = true;
    let mut a_strict = false;
    let mut b_strict = false;
    let mut a_better_any = false;
    let mut b_better_any = false;

    for objective in objectives {
        let a = candidate_a.metrics.get(&objective.metric_id).expect("validated candidate has objective metric");
        let b = candidate_b.metrics.get(&objective.metric_id).expect("validated candidate has objective metric");
        let a_lower = a.value - a.expanded_uncertainty;
        let a_upper = a.value + a.expanded_uncertainty;
        let b_lower = b.value - b.expanded_uncertainty;
        let b_upper = b.value + b.expanded_uncertainty;
        let (a_no_worse, b_no_worse, a_better, b_better) = match objective.direction {
            Direction::Minimize => (
                a_upper <= b_lower,
                b_upper <= a_lower,
                a_upper < b_lower,
                b_upper < a_lower,
            ),
            Direction::Maximize => (
                a_lower >= b_upper,
                b_lower >= a_upper,
                a_lower > b_upper,
                b_lower > a_upper,
            ),
        };
        a_no_worse_all &= a_no_worse;
        b_no_worse_all &= b_no_worse;
        a_strict |= a_better;
        b_strict |= b_better;
        a_better_any |= a_better;
        b_better_any |= b_better;

        evidence.push(ObjectivePairEvidence {
            metric_id: objective.metric_id.clone(),
            direction: match objective.direction {
                Direction::Minimize => "minimize",
                Direction::Maximize => "maximize",
            },
            a_lower,
            a_upper,
            b_lower,
            b_upper,
            a_definitely_better: a_better,
            b_definitely_better: b_better,
            intervals_overlap_or_touch: !(a_upper < b_lower || b_upper < a_lower),
        });
    }

    let relation = if a_no_worse_all && a_strict {
        "a_dominates_b"
    } else if b_no_worse_all && b_strict {
        "b_dominates_a"
    } else if a_better_any && b_better_any {
        "robust_tradeoff"
    } else {
        "unresolved_interval_order"
    };

    PairwiseRelation {
        candidate_a: candidate_a.report.candidate_id.clone(),
        candidate_b: candidate_b.report.candidate_id.clone(),
        relation_from_a: relation,
        objective_evidence: evidence,
    }
}

fn audit(case: &Case) -> AuditReport {
    let validation_errors = validate_case(case);
    let mut reasons = validation_errors.clone();
    let mut evaluated: Vec<EvaluatedCandidate> = if validation_errors.is_empty() {
        case.candidates.iter().map(|candidate| evaluate_candidate(case, candidate)).collect()
    } else {
        Vec::new()
    };
    evaluated.sort_by(|left, right| left.report.candidate_id.cmp(&right.report.candidate_id));

    let eligible: Vec<usize> = evaluated.iter().enumerate()
        .filter(|(_, candidate)| candidate.disposition == "ELIGIBLE_FOR_ROBUST_PARETO")
        .map(|(index, _)| index)
        .collect();
    let mut pairwise_relations = Vec::new();
    for left in 0..eligible.len() {
        for right in (left + 1)..eligible.len() {
            pairwise_relations.push(compare_pair(
                &evaluated[eligible[left]],
                &evaluated[eligible[right]],
                &case.objectives,
            ));
        }
    }

    let mut dominated = BTreeSet::new();
    let mut unresolved_opponents: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for relation in &pairwise_relations {
        match relation.relation_from_a {
            "a_dominates_b" => { dominated.insert(relation.candidate_b.clone()); }
            "b_dominates_a" => { dominated.insert(relation.candidate_a.clone()); }
            "unresolved_interval_order" => {
                unresolved_opponents.entry(relation.candidate_a.clone()).or_default()
                    .insert(relation.candidate_b.clone());
                unresolved_opponents.entry(relation.candidate_b.clone()).or_default()
                    .insert(relation.candidate_a.clone());
            }
            _ => {}
        }
    }
    let mut robust_frontier = Vec::new();
    for index in &eligible {
        let candidate_id = &evaluated[*index].report.candidate_id;
        if !dominated.contains(candidate_id) {
            let unresolved: Vec<String> = unresolved_opponents.get(candidate_id)
                .map(|values| values.iter().cloned().collect())
                .unwrap_or_default();
            robust_frontier.push(FrontierEntry {
                candidate_id: candidate_id.clone(),
                certainty: if unresolved.is_empty() { "confirmed_nondominated" } else { "provisional_unresolved_pairs" },
                unresolved_opponents: unresolved,
            });
        }
    }

    if evaluated.iter().any(|candidate| {
        candidate.disposition == "HOLD_UNRESOLVED" || candidate.disposition == "INCOMPARABLE_SERVICE"
    }) {
        reasons.push("one or more candidates are held due to unresolved evidence or unmatched service".to_string());
    }
    if evaluated.iter().any(|candidate| candidate.disposition == "INVALID") {
        reasons.push("one or more candidates contain invalid evidence or missing required measurements".to_string());
    }

    let disposition = if !validation_errors.is_empty()
        || evaluated.iter().any(|candidate| {
            candidate.disposition == "HOLD_UNRESOLVED"
                || candidate.disposition == "INCOMPARABLE_SERVICE"
                || candidate.disposition == "INVALID"
        })
    {
        "INCOMPLETE"
    } else if eligible.is_empty() {
        "NO_ELIGIBLE_CANDIDATES"
    } else if eligible.len() == 1 {
        "INSUFFICIENT_ELIGIBLE_CANDIDATES"
    } else {
        "ROBUST_PARETO_EVALUATED"
    };

    AuditReport {
        schema_version: SCHEMA_VERSION,
        case_id: case.case_id.clone(),
        source_register_id: case.source_register.register_id.clone(),
        source_register_sha256: case.source_register.register_sha256.clone(),
        source_register_citation: case.source_register.citation.clone(),
        disposition,
        cargo_work_unit: case.service.cargo_work_unit.clone(),
        candidate_reports: evaluated.into_iter().map(|candidate| candidate.report).collect(),
        pairwise_relations,
        robust_frontier,
        claim_ceiling: "interval_robust_comparison_of_declared_objectives_and_noncompensable_gates_only_no_physical_solver_validation_ecological_response_or_regulatory_compliance_claim",
        reasons,
    }
}

fn main() {
    let mut args = env::args().skip(1);
    let Some(path) = args.next() else {
        eprintln!("usage: marine-robust-pareto-audit <case.json>");
        process::exit(64);
    };
    if args.next().is_some() {
        eprintln!("usage: marine-robust-pareto-audit <case.json>");
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
        "ROBUST_PARETO_EVALUATED" | "NO_ELIGIBLE_CANDIDATES" | "INSUFFICIENT_ELIGIBLE_CANDIDATES" => {}
        "INCOMPLETE" => process::exit(3),
        _ => process::exit(2),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(letter: char) -> String {
        std::iter::repeat_n(letter, 64).collect()
    }

    fn metric(metric_id: &str, value: f64, unit: &str) -> MetricObservation {
        MetricObservation {
            metric_id: metric_id.to_string(),
            value,
            expanded_uncertainty: 0.5,
            confidence_level: 0.95,
            unit: unit.to_string(),
            method_id: format!("method-{metric_id}"),
            raw_data_sha256: digest('a'),
            calibration_or_validation_sha256: digest('b'),
            timebase_receipt_sha256: digest('c'),
            operating_point_receipt_sha256: digest('d'),
        }
    }

    fn candidate(id: &str, energy: f64, acoustic: f64) -> Candidate {
        let operating_point_receipt_sha256 = digest(if id == "A" { '3' } else { '4' });
        let mut metrics = vec![
            metric("energy_per_cargo_work", energy, "kWh_per_1000_tonne_nm"),
            metric("well_to_wake_ghg_per_cargo_work", if id == "A" { 70.0 } else { 75.0 }, "kgCO2e_per_1000_tonne_nm"),
            metric("acoustic_1khz", acoustic, "dB_re_1uPa_at_declared_reference"),
            metric("acoustic_2_5khz", if id == "A" { 102.0 } else { 103.0 }, "dB_re_1uPa_at_declared_reference"),
            metric("acoustic_30_80khz", if id == "A" { 115.0 } else { 116.0 }, "dB_re_1uPa_at_declared_reference"),
            metric("lifecycle_cost", if id == "A" { 20.0 } else { 21.0 }, "USD_per_1000_tonne_nm"),
            metric("maintenance_hours_per_cargo_work", if id == "A" { 6.0 } else { 7.0 }, "hours_per_1000_tonne_nm"),
            metric("availability", if id == "A" { 0.98 } else { 0.97 }, "fraction"),
        ];
        for observation in &mut metrics {
            observation.operating_point_receipt_sha256 = operating_point_receipt_sha256.clone();
            if observation.metric_id == "availability" {
                observation.expanded_uncertainty = 0.005;
            }
        }
        Candidate {
            candidate_id: id.to_string(),
            evidence_kind: EvidenceKind::Measured,
            dataset_id: format!("dataset-{id}"),
            citation: format!("Measured campaign {id}"),
            raw_data_sha256: digest(if id == "A" { 'e' } else { 'f' }),
            rights_status: RightsStatus::PublicOrPermitted,
            independently_reviewed: true,
            mission_id: "mission-01".to_string(),
            vessel_id: "vessel-01".to_string(),
            route_id: "route-01".to_string(),
            measurement_window_id: "window-01".to_string(),
            service_fingerprint_sha256: digest('1'),
            environment_fingerprint_sha256: digest('2'),
            vessel_revision: format!("hull-{id}"),
            propeller_revision: format!("prop-{id}"),
            cargo_work_value: 1_000_000.0,
            cargo_work_expanded_uncertainty: 100.0,
            cargo_work_unit: "tonne_nautical_mile".to_string(),
            operating_point_receipt_sha256,
            metrics,
            gate_receipts: vec![
                GateReceipt {
                    gate_id: "biosecurity-release".to_string(),
                    status: GateStatus::Satisfied,
                    audit_contract_id: "marine-biofouling-evidence-audit-v0.1".to_string(),
                    audit_receipt_sha256: digest(if id == "A" { '8' } else { '9' }),
                    requirement_reference: "permit:zero-uncontained-organism-release".to_string(),
                },
                GateReceipt {
                    gate_id: "cleaning-waste-authorization".to_string(),
                    status: GateStatus::Satisfied,
                    audit_contract_id: "marine-biofouling-evidence-audit-v0.1".to_string(),
                    audit_receipt_sha256: digest(if id == "A" { 'a' } else { 'b' }),
                    requirement_reference: "permit:in-water-cleaning-authorized-and-waste-captured".to_string(),
                },
                GateReceipt {
                    gate_id: "acoustic-scale-transfer-evidence".to_string(),
                    status: GateStatus::Satisfied,
                    audit_contract_id: "marine-acoustic-scale-transfer-audit-v0.1.0".to_string(),
                    audit_receipt_sha256: digest(if id == "A" { 'c' } else { 'd' }),
                    requirement_reference: "measurement:scale-transfer-reviewed-on-held-out-full-scale-data".to_string(),
                },
                GateReceipt {
                    gate_id: "acoustic-source-identifiability".to_string(),
                    status: GateStatus::Satisfied,
                    audit_contract_id: "marine-acoustic-source-contrast-audit-v0.1.0".to_string(),
                    audit_receipt_sha256: digest(if id == "A" { 'e' } else { 'f' }),
                    requirement_reference: "measurement:source-state-contrast-reviewed".to_string(),
                },
                GateReceipt {
                    gate_id: "cavitation-intermittency-characterization".to_string(),
                    status: GateStatus::Satisfied,
                    audit_contract_id: "marine-cavitation-intermittency-audit-v0.1.0".to_string(),
                    audit_receipt_sha256: digest(if id == "A" { '1' } else { '2' }),
                    requirement_reference: "measurement:cavitation-intermittency-distribution-reviewed".to_string(),
                },
                GateReceipt {
                    gate_id: "vibration-coherence-characterization".to_string(),
                    status: GateStatus::Satisfied,
                    audit_contract_id: "marine-vibration-coherence-audit-v0.1.0".to_string(),
                    audit_receipt_sha256: digest(if id == "A" { '3' } else { '4' }),
                    requirement_reference: "measurement:vibration-coherent-pressure-component-reviewed".to_string(),
                },
                GateReceipt {
                    gate_id: "rotation-synchronous-modulation".to_string(),
                    status: GateStatus::Satisfied,
                    audit_contract_id: "marine-rotation-synchronous-modulation-audit-v0.1.0".to_string(),
                    audit_receipt_sha256: digest(if id == "A" { '5' } else { '6' }),
                    requirement_reference: "measurement:rotation-synchronous-envelope-reviewed".to_string(),
                },
            ],
        }
    }

    fn case() -> Case {
        let a = candidate("A", 90.0, 100.0);
        let b = candidate("B", 100.0, 98.0);
        Case {
            schema_version: SCHEMA_VERSION.to_string(),
            case_id: "synthetic-marine-pareto-001".to_string(),
            source_register: SourceRegister {
                register_id: "synthetic-register".to_string(),
                citation: "Synthetic test fixture only".to_string(),
                register_sha256: digest('5'),
            },
            service: ServiceContract {
                mission_id: "mission-01".to_string(),
                vessel_id: "vessel-01".to_string(),
                route_id: "route-01".to_string(),
                measurement_window_id: "window-01".to_string(),
                service_fingerprint_sha256: digest('1'),
                environment_fingerprint_sha256: digest('2'),
                cargo_work_unit: "tonne_nautical_mile".to_string(),
                reference_cargo_work: 1_000_000.0,
                reference_cargo_work_expanded_uncertainty: 100.0,
                cargo_work_equivalence_tolerance: 500.0,
            },
            objectives: vec![
                ObjectiveDefinition {
                    metric_id: "energy_per_cargo_work".to_string(),
                    direction: Direction::Minimize,
                    unit: "kWh_per_1000_tonne_nm".to_string(),
                },
                ObjectiveDefinition {
                    metric_id: "well_to_wake_ghg_per_cargo_work".to_string(),
                    direction: Direction::Minimize,
                    unit: "kgCO2e_per_1000_tonne_nm".to_string(),
                },
                ObjectiveDefinition {
                    metric_id: "acoustic_1khz".to_string(),
                    direction: Direction::Minimize,
                    unit: "dB_re_1uPa_at_declared_reference".to_string(),
                },
                ObjectiveDefinition {
                    metric_id: "acoustic_2_5khz".to_string(),
                    direction: Direction::Minimize,
                    unit: "dB_re_1uPa_at_declared_reference".to_string(),
                },
                ObjectiveDefinition {
                    metric_id: "acoustic_30_80khz".to_string(),
                    direction: Direction::Minimize,
                    unit: "dB_re_1uPa_at_declared_reference".to_string(),
                },
                ObjectiveDefinition {
                    metric_id: "lifecycle_cost".to_string(),
                    direction: Direction::Minimize,
                    unit: "USD_per_1000_tonne_nm".to_string(),
                },
                ObjectiveDefinition {
                    metric_id: "maintenance_hours_per_cargo_work".to_string(),
                    direction: Direction::Minimize,
                    unit: "hours_per_1000_tonne_nm".to_string(),
                },
                ObjectiveDefinition {
                    metric_id: "availability".to_string(),
                    direction: Direction::Maximize,
                    unit: "fraction".to_string(),
                },
            ],
            hard_gates: vec![
                HardGateDefinition {
                    gate_id: "acoustic-maximum-1khz".to_string(),
                    gate_class: "underwater_noise".to_string(),
                    evaluation: GateEvaluation::MetricUpperBound,
                    metric_id: Some("acoustic_1khz".to_string()),
                    threshold: Some(105.0),
                    unit: Some("dB_re_1uPa_at_declared_reference".to_string()),
                    required_audit_contract_id: None,
                    requirement_reference: "operator-limit:acoustic-1khz".to_string(),
                    non_compensable: true,
                },
                HardGateDefinition {
                    gate_id: "acoustic-30-80khz-maximum".to_string(),
                    gate_class: "underwater_noise".to_string(),
                    evaluation: GateEvaluation::MetricUpperBound,
                    metric_id: Some("acoustic_30_80khz".to_string()),
                    threshold: Some(125.0),
                    unit: Some("dB_re_1uPa_at_declared_reference".to_string()),
                    required_audit_contract_id: None,
                    requirement_reference: "operator-limit:acoustic-30-80khz".to_string(),
                    non_compensable: true,
                },
                HardGateDefinition {
                    gate_id: "availability-minimum".to_string(),
                    gate_class: "reliability".to_string(),
                    evaluation: GateEvaluation::MetricLowerBound,
                    metric_id: Some("availability".to_string()),
                    threshold: Some(0.95),
                    unit: Some("fraction".to_string()),
                    required_audit_contract_id: None,
                    requirement_reference: "service-contract:minimum-availability".to_string(),
                    non_compensable: true,
                },
                HardGateDefinition {
                    gate_id: "biosecurity-release".to_string(),
                    gate_class: "biosecurity".to_string(),
                    evaluation: GateEvaluation::ExternalAuditReceipt,
                    metric_id: None,
                    threshold: None,
                    unit: None,
                    required_audit_contract_id: Some("marine-biofouling-evidence-audit-v0.1".to_string()),
                    requirement_reference: "permit:zero-uncontained-organism-release".to_string(),
                    non_compensable: true,
                },
                HardGateDefinition {
                    gate_id: "cleaning-waste-authorization".to_string(),
                    gate_class: "biosecurity".to_string(),
                    evaluation: GateEvaluation::ExternalAuditReceipt,
                    metric_id: None,
                    threshold: None,
                    unit: None,
                    required_audit_contract_id: Some("marine-biofouling-evidence-audit-v0.1".to_string()),
                    requirement_reference: "permit:in-water-cleaning-authorized-and-waste-captured".to_string(),
                    non_compensable: true,
                },
                HardGateDefinition {
                    gate_id: "acoustic-scale-transfer-evidence".to_string(),
                    gate_class: "underwater_noise_evidence".to_string(),
                    evaluation: GateEvaluation::ExternalAuditReceipt,
                    metric_id: None,
                    threshold: None,
                    unit: None,
                    required_audit_contract_id: Some("marine-acoustic-scale-transfer-audit-v0.1.0".to_string()),
                    requirement_reference: "measurement:scale-transfer-reviewed-on-held-out-full-scale-data".to_string(),
                    non_compensable: true,
                },
                HardGateDefinition {
                    gate_id: "acoustic-source-identifiability".to_string(),
                    gate_class: "underwater_noise_evidence".to_string(),
                    evaluation: GateEvaluation::ExternalAuditReceipt,
                    metric_id: None,
                    threshold: None,
                    unit: None,
                    required_audit_contract_id: Some("marine-acoustic-source-contrast-audit-v0.1.0".to_string()),
                    requirement_reference: "measurement:source-state-contrast-reviewed".to_string(),
                    non_compensable: true,
                },
                HardGateDefinition {
                    gate_id: "cavitation-intermittency-characterization".to_string(),
                    gate_class: "underwater_noise_evidence".to_string(),
                    evaluation: GateEvaluation::ExternalAuditReceipt,
                    metric_id: None,
                    threshold: None,
                    unit: None,
                    required_audit_contract_id: Some("marine-cavitation-intermittency-audit-v0.1.0".to_string()),
                    requirement_reference: "measurement:cavitation-intermittency-distribution-reviewed".to_string(),
                    non_compensable: true,
                },
                HardGateDefinition {
                    gate_id: "vibration-coherence-characterization".to_string(),
                    gate_class: "underwater_noise_evidence".to_string(),
                    evaluation: GateEvaluation::ExternalAuditReceipt,
                    metric_id: None,
                    threshold: None,
                    unit: None,
                    required_audit_contract_id: Some("marine-vibration-coherence-audit-v0.1.0".to_string()),
                    requirement_reference: "measurement:vibration-coherent-pressure-component-reviewed".to_string(),
                    non_compensable: true,
                },
                HardGateDefinition {
                    gate_id: "rotation-synchronous-modulation".to_string(),
                    gate_class: "underwater_noise_evidence".to_string(),
                    evaluation: GateEvaluation::ExternalAuditReceipt,
                    metric_id: None,
                    threshold: None,
                    unit: None,
                    required_audit_contract_id: Some("marine-rotation-synchronous-modulation-audit-v0.1.0".to_string()),
                    requirement_reference: "measurement:rotation-synchronous-envelope-reviewed".to_string(),
                    non_compensable: true,
                },
            ],
            candidates: vec![a, b],
        }
    }

    #[test]
    fn unresolved_acoustic_analysis_receipt_holds_candidate_despite_good_energy() {
        let mut input = case();
        input.candidates[0].metrics.iter_mut()
            .find(|metric| metric.metric_id == "energy_per_cargo_work")
            .unwrap().value = 20.0;
        input.candidates[0].gate_receipts.iter_mut()
            .find(|receipt| receipt.gate_id == "cavitation-intermittency-characterization")
            .unwrap().status = GateStatus::Unresolved;
        let report = audit(&input);
        assert_eq!(report.candidate_reports[0].disposition, "HOLD_UNRESOLVED");
        assert!(report.robust_frontier.iter().all(|entry| entry.candidate_id != "A"));
        let gate = report.candidate_reports[0].gate_results.iter()
            .find(|gate| gate.gate_id == "cavitation-intermittency-characterization").unwrap();
        assert_eq!(gate.status, "unresolved");
        assert_eq!(gate.audit_contract_id.as_deref(), Some("marine-cavitation-intermittency-audit-v0.1.0"));
    }

    #[test]
    fn every_registered_environmental_audit_gate_has_a_matching_receipt() {
        let report = audit(&case());
        let receipts = &report.candidate_reports[0].gate_results;
        for id in [
            "biosecurity-release",
            "cleaning-waste-authorization",
            "acoustic-scale-transfer-evidence",
            "acoustic-source-identifiability",
            "cavitation-intermittency-characterization",
            "vibration-coherence-characterization",
            "rotation-synchronous-modulation",
        ] {
            let gate = receipts.iter().find(|gate| gate.gate_id == id).unwrap();
            assert_eq!(gate.status, "pass");
            assert!(gate.audit_receipt_sha256.is_some());
        }
    }

    #[test]
    fn synthetic_candidates_never_enter_robust_frontier() {
        let mut input = case();
        for candidate in &mut input.candidates {
            candidate.evidence_kind = EvidenceKind::Synthetic;
        }
        let report = audit(&input);
        assert_eq!(report.disposition, "INCOMPLETE");
        assert!(report.robust_frontier.is_empty());
    }

    #[test]
    fn hard_biosecurity_failure_excludes_candidate_despite_energy_advantage() {
        let mut input = case();
        input.candidates[0].metrics.iter_mut()
            .find(|metric| metric.metric_id == "energy_per_cargo_work")
            .unwrap().value = 20.0;
        input.candidates[0].gate_receipts[0].status = GateStatus::Failed;
        let report = audit(&input);
        assert_eq!(report.candidate_reports[0].disposition, "EXCLUDED_HARD_GATE");
        assert!(report.robust_frontier.iter().all(|entry| entry.candidate_id != "A"));
    }

    #[test]
    fn overlapping_upper_gate_uncertainty_is_hold_not_pass() {
        let mut input = case();
        let metric = input.candidates[0].metrics.iter_mut()
            .find(|metric| metric.metric_id == "acoustic_1khz")
            .unwrap();
        metric.value = 105.0;
        metric.expanded_uncertainty = 1.0;
        let report = audit(&input);
        assert_eq!(report.candidate_reports[0].disposition, "HOLD_UNRESOLVED");
    }

    #[test]
    fn upper_gate_definite_failure_is_excluded() {
        let mut input = case();
        input.candidates[0].metrics.iter_mut()
            .find(|metric| metric.metric_id == "acoustic_1khz")
            .unwrap().value = 106.0;
        assert_eq!(audit(&input).candidate_reports[0].disposition, "EXCLUDED_HARD_GATE");
    }

    #[test]
    fn lower_gate_passes_only_when_entire_interval_above_threshold() {
        let input = case();
        let report = audit(&input);
        let gate = report.candidate_reports[0].gate_results.iter()
            .find(|gate| gate.gate_id == "availability-minimum").unwrap();
        assert_eq!(gate.status, "pass");
        assert_eq!(report.candidate_reports[0].disposition, "ELIGIBLE_FOR_ROBUST_PARETO");
    }

    #[test]
    fn upper_acoustic_gate_is_frequency_specific_and_non_compensable() {
        let mut input = case();
        input.candidates[0].metrics.iter_mut()
            .find(|metric| metric.metric_id == "acoustic_30_80khz")
            .unwrap().value = 126.0;
        input.candidates[0].metrics.iter_mut()
            .find(|metric| metric.metric_id == "energy_per_cargo_work")
            .unwrap().value = 10.0;
        let report = audit(&input);
        assert_eq!(report.candidate_reports[0].disposition, "EXCLUDED_HARD_GATE");
        assert!(report.candidate_reports[0].gate_results.iter()
            .any(|gate| gate.gate_id == "acoustic-30-80khz-maximum" && gate.status == "fail"));
    }

    #[test]
    fn external_biosecurity_audit_receipt_is_required_and_consumed() {
        let report = audit(&case());
        assert_eq!(report.candidate_reports[0].disposition, "ELIGIBLE_FOR_ROBUST_PARETO");
        let receipt = report.candidate_reports[0].gate_results.iter()
            .find(|gate| gate.gate_id == "biosecurity-release").unwrap();
        assert_eq!(receipt.status, "pass");
        assert_eq!(receipt.audit_contract_id.as_deref(), Some("marine-biofouling-evidence-audit-v0.1"));
        assert!(receipt.audit_receipt_sha256.is_some());
    }

    #[test]
    fn external_gate_unresolved_receipt_holds_candidate() {
        let mut input = case();
        input.candidates[0].gate_receipts[0].status = GateStatus::Unresolved;
        assert_eq!(audit(&input).candidate_reports[0].disposition, "HOLD_UNRESOLVED");
    }

    #[test]
    fn missing_external_gate_receipt_is_incomplete() {
        let mut input = case();
        input.candidates[0].gate_receipts.clear();
        assert_eq!(audit(&input).disposition, "INCOMPLETE");
    }

    #[test]
    fn interval_separated_objectives_produce_robust_tradeoff_not_single_score() {
        let mut input = case();
        input.candidates[0].metrics.iter_mut()
            .find(|metric| metric.metric_id == "energy_per_cargo_work")
            .unwrap().value = 90.0;
        input.candidates[1].metrics.iter_mut()
            .find(|metric| metric.metric_id == "energy_per_cargo_work")
            .unwrap().value = 100.0;
        input.candidates[0].metrics.iter_mut()
            .find(|metric| metric.metric_id == "acoustic_1khz")
            .unwrap().value = 100.0;
        input.candidates[1].metrics.iter_mut()
            .find(|metric| metric.metric_id == "acoustic_1khz")
            .unwrap().value = 98.0;
        let report = audit(&input);
        assert_eq!(report.disposition, "ROBUST_PARETO_EVALUATED");
        assert_eq!(report.pairwise_relations[0].relation_from_a, "robust_tradeoff");
        assert_eq!(report.robust_frontier.len(), 2);
    }

    #[test]
    fn candidate_a_dominates_b_only_when_every_interval_is_no_worse() {
        let mut input = case();
        input.candidates[0].metrics.iter_mut()
            .find(|metric| metric.metric_id == "acoustic_1khz").unwrap().value = 95.0;
        input.candidates[1].metrics.iter_mut()
            .find(|metric| metric.metric_id == "acoustic_1khz").unwrap().value = 100.0;
        input.candidates[0].metrics.iter_mut()
            .find(|metric| metric.metric_id == "energy_per_cargo_work").unwrap().value = 80.0;
        input.candidates[1].metrics.iter_mut()
            .find(|metric| metric.metric_id == "energy_per_cargo_work").unwrap().value = 100.0;
        input.candidates[0].metrics.iter_mut()
            .find(|metric| metric.metric_id == "lifecycle_cost").unwrap().value = 15.0;
        input.candidates[1].metrics.iter_mut()
            .find(|metric| metric.metric_id == "lifecycle_cost").unwrap().value = 25.0;
        let report = audit(&input);
        assert_eq!(report.pairwise_relations[0].relation_from_a, "a_dominates_b");
        assert_eq!(report.robust_frontier.len(), 1);
        assert_eq!(report.robust_frontier[0].candidate_id, "A");
    }

    #[test]
    fn overlapping_objective_intervals_never_produce_dominance() {
        let mut input = case();
        for candidate in &mut input.candidates {
            for metric in &mut candidate.metrics {
                metric.expanded_uncertainty = 10.0;
            }
        }
        input.hard_gates.iter_mut()
            .filter(|gate| gate.evaluation == GateEvaluation::MetricUpperBound)
            .for_each(|gate| gate.threshold = Some(200.0));
        input.hard_gates.iter_mut()
            .find(|gate| gate.gate_id == "availability-minimum")
            .unwrap().threshold = Some(-100.0);
        let report = audit(&input);
        assert_eq!(report.pairwise_relations[0].relation_from_a, "unresolved_interval_order");
        assert!(report.robust_frontier.iter().all(|entry| entry.certainty == "provisional_unresolved_pairs"));
    }

    #[test]
    fn matched_service_fingerprint_is_mandatory() {
        let mut input = case();
        input.candidates[1].service_fingerprint_sha256 = digest('9');
        assert_eq!(audit(&input).candidate_reports[1].disposition, "INCOMPARABLE_SERVICE");
    }

    #[test]
    fn cargo_work_difference_outside_tolerance_is_incomparable() {
        let mut input = case();
        input.candidates[0].cargo_work_value = 2_000_000.0;
        assert_eq!(audit(&input).candidate_reports[0].disposition, "INCOMPARABLE_SERVICE");
    }

    #[test]
    fn cargo_work_uncertainty_crossing_tolerance_is_held() {
        let mut input = case();
        input.candidates[0].cargo_work_value = 1_000_400.0;
        input.candidates[0].cargo_work_expanded_uncertainty = 200.0;
        assert_eq!(audit(&input).candidate_reports[0].disposition, "HOLD_UNRESOLVED");
    }

    #[test]
    fn all_hard_gates_must_be_non_compensable() {
        let mut input = case();
        input.hard_gates[1].non_compensable = false;
        assert_eq!(audit(&input).disposition, "INCOMPLETE");
    }

    #[test]
    fn metric_operating_point_receipt_must_match_candidate_receipt() {
        let mut input = case();
        input.candidates[0].metrics[0].operating_point_receipt_sha256 = digest('9');
        assert_eq!(audit(&input).disposition, "INCOMPLETE");
    }

    #[test]
    fn objective_method_mismatch_is_incomparable_not_ranked() {
        let mut input = case();
        input.candidates[1].metrics.iter_mut()
            .find(|metric| metric.metric_id == "acoustic_1khz")
            .unwrap().method_id = "different-acoustic-method".to_string();
        assert_eq!(audit(&input).disposition, "INCOMPLETE");
        assert!(audit(&input).reasons.iter().any(|reason| reason.contains("different measurement methods")));
    }

    #[test]
    fn objective_uncertainty_confidence_mismatch_is_not_ranked() {
        let mut input = case();
        input.candidates[1].metrics.iter_mut()
            .find(|metric| metric.metric_id == "acoustic_1khz")
            .unwrap().confidence_level = 0.99;
        assert_eq!(audit(&input).disposition, "INCOMPLETE");
        assert!(audit(&input).reasons.iter().any(|reason| reason.contains("different expanded-uncertainty confidence levels")));
    }

    #[test]
    fn overflowed_metric_interval_is_rejected() {
        let mut input = case();
        let metric = input.candidates[0].metrics.iter_mut()
            .find(|metric| metric.metric_id == "energy_per_cargo_work")
            .unwrap();
        metric.value = f64::MAX;
        metric.expanded_uncertainty = f64::MAX;
        assert_eq!(audit(&input).disposition, "INCOMPLETE");
    }

    #[test]
    fn hard_gate_failure_is_never_masked_by_service_mismatch() {
        let mut input = case();
        input.candidates[0].service_fingerprint_sha256 = digest('9');
        input.candidates[0].gate_receipts[0].status = GateStatus::Failed;
        let report = audit(&input);
        assert_eq!(report.candidate_reports[0].disposition, "EXCLUDED_HARD_GATE");
        assert!(report.robust_frontier.iter().all(|entry| entry.candidate_id != "A"));
    }

    #[test]
    fn metric_gate_unit_mismatch_is_rejected_before_comparison() {
        let mut input = case();
        input.hard_gates[0].unit = Some("wrong_unit".to_string());
        assert_eq!(audit(&input).disposition, "INCOMPLETE");
        assert!(audit(&input).pairwise_relations.is_empty());
    }

    #[test]
    fn external_receipt_contract_mismatch_is_rejected() {
        let mut input = case();
        input.candidates[0].gate_receipts[0].audit_contract_id = "different-audit".to_string();
        assert_eq!(audit(&input).disposition, "INCOMPLETE");
    }

    #[test]
    fn maximizing_objective_uses_lower_interval_against_upper_interval() {
        let mut input = case();
        for candidate in &mut input.candidates {
            let observation = candidate.metrics.iter_mut()
                .find(|metric| metric.metric_id == "availability")
                .unwrap();
            observation.value = if candidate.candidate_id == "A" { 0.99 } else { 0.95 };
            observation.expanded_uncertainty = 0.001;
        }
        let report = audit(&input);
        let relation = &report.pairwise_relations[0];
        let availability = relation.objective_evidence.iter()
            .find(|item| item.metric_id == "availability").unwrap();
        assert!(availability.a_definitely_better);
        assert!(!availability.b_definitely_better);
    }

    #[test]
    fn schema_fixture_and_metric_provenance_are_present() {
        let schema: serde_json::Value = serde_json::from_str(include_str!(
            "../../../docs/research/marine/marine-robust-pareto-v0.1.0.schema.json"
        )).expect("schema JSON parses");
        assert_eq!(schema["$schema"], "https://json-schema.org/draft/2020-12/schema");
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../../docs/research/marine/fixtures/synthetic-marine-robust-pareto-v0.1.0.json"
        )).expect("fixture JSON parses");
        assert_eq!(fixture["candidates"].as_array().unwrap().len(), 2);
        assert_eq!(fixture["candidates"][0]["evidence_kind"], "synthetic");
        assert_eq!(fixture["candidates"][1]["evidence_kind"], "synthetic");
        assert!(fixture["candidates"][0]["metrics"][0]["operating_point_receipt_sha256"].is_string());
        assert_eq!(fixture["hard_gates"].as_array().unwrap().iter().filter(|gate| gate["non_compensable"] == true).count(), 10);
    }
}

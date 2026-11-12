use std::collections::{BTreeMap, BTreeSet};

const FLOAT_EPSILON: f64 = 1.0e-12;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Direction {
    Maximize,
    Minimize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Objective {
    pub id: String,
    pub unit: String,
    pub boundary: String,
    pub direction: Direction,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Scenario {
    pub id: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EvidenceClass {
    SyntheticFixture,
    Assumption,
    VendorClaim,
    LiteratureReported,
    Surrogate,
    Calculation,
    PhysicalMeasurement,
    IndependentLaboratory,
}

impl EvidenceClass {
    fn is_physical(&self) -> bool {
        matches!(
            self,
            Self::PhysicalMeasurement | Self::IndependentLaboratory
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EvidenceAdmission {
    Candidate,
    AdmittedForResearch,
    AdmittedForEvaluation,
    Blocked,
    Unknown,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EvidenceRecord {
    pub evidence_id: String,
    pub source_sha256: String,
    pub claim_scope: String,
    pub validity_domain: String,
    pub admission: EvidenceAdmission,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MetricObservation {
    pub objective_id: String,
    pub unit: String,
    pub boundary: String,
    pub value: f64,
    pub lower_bound: f64,
    pub upper_bound: f64,
    pub evidence_id: String,
    pub evidence_class: EvidenceClass,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GateDisposition {
    Pass,
    Fail,
    Unknown,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HardGate {
    pub gate_id: String,
    pub disposition: GateDisposition,
    pub evidence_id: String,
    pub requires_physical_evidence: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Candidate {
    pub id: String,
    pub revision: String,
    pub scenario_metrics: BTreeMap<String, Vec<MetricObservation>>,
    pub hard_gates: Vec<HardGate>,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum RejectionReason {
    MissingHardGates,
    GateFailed(String),
    GateUnknown(String),
    GateEvidenceNotPhysical(String),
    GateEvidenceNotAdmitted(String),
    MetricEvidenceNotAdmitted(String),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CandidateRejection {
    pub candidate_id: String,
    pub reasons: Vec<RejectionReason>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CandidateScore {
    pub candidate_id: String,
    /// Conservative scenario aggregation: lower bound for maximization,
    /// upper bound for minimization. Units and boundaries remain in Objective.
    pub robust_values: BTreeMap<String, f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AuditReport {
    pub eligible_candidate_ids: Vec<String>,
    pub pareto_front_ids: Vec<String>,
    pub rejected: Vec<CandidateRejection>,
    pub scores: Vec<CandidateScore>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AuditError(pub String);

impl std::fmt::Display for AuditError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for AuditError {}

fn validate_unique_nonempty<'a>(
    values: impl IntoIterator<Item = &'a str>,
    label: &str,
) -> Result<(), AuditError> {
    let mut seen = BTreeSet::new();

    for value in values {
        if value.trim().is_empty() {
            return Err(AuditError(format!("{label} must not be empty")));
        }

        if !seen.insert(value) {
            return Err(AuditError(format!("duplicate {label}: {value}")));
        }
    }

    Ok(())
}

fn validate_sha256(value: &str, label: &str) -> Result<(), AuditError> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(AuditError(format!(
            "{label} must be a lowercase 64-character SHA-256 digest"
        )));
    }
    Ok(())
}

fn validate_evidence_registry(
    evidence: &[EvidenceRecord],
) -> Result<BTreeMap<&str, &EvidenceRecord>, AuditError> {
    if evidence.is_empty() {
        return Err(AuditError("at least one evidence record is required".into()));
    }

    validate_unique_nonempty(
        evidence.iter().map(|record| record.evidence_id.as_str()),
        "evidence ID",
    )?;

    let mut records = BTreeMap::new();
    for record in evidence {
        validate_sha256(&record.source_sha256, "evidence source digest")?;
        if record.claim_scope.trim().is_empty() || record.validity_domain.trim().is_empty() {
            return Err(AuditError(format!(
                "evidence {} has an empty claim scope or validity domain",
                record.evidence_id
            )));
        }
        records.insert(record.evidence_id.as_str(), record);
    }
    Ok(records)
}

fn require_evidence<'a>(
    records: &'a BTreeMap<&str, &'a EvidenceRecord>,
    evidence_id: &str,
    owner: &str,
) -> Result<&'a EvidenceRecord, AuditError> {
    records.get(evidence_id).copied().ok_or_else(|| {
        AuditError(format!("{owner} references unknown evidence ID {evidence_id}"))
    })
}

fn validate_manifest(
    scenarios: &[Scenario],
    objectives: &[Objective],
    evidence: &[EvidenceRecord],
    candidates: &[Candidate],
) -> Result<BTreeMap<&str, &EvidenceRecord>, AuditError> {
    if scenarios.is_empty() {
        return Err(AuditError("at least one scenario is required".into()));
    }

    if objectives.is_empty() {
        return Err(AuditError("at least one objective is required".into()));
    }

    if candidates.is_empty() {
        return Err(AuditError("at least one candidate is required".into()));
    }

    validate_unique_nonempty(
        scenarios.iter().map(|scenario| scenario.id.as_str()),
        "scenario ID",
    )?;
    validate_unique_nonempty(
        objectives.iter().map(|objective| objective.id.as_str()),
        "objective ID",
    )?;
    validate_unique_nonempty(
        candidates.iter().map(|candidate| candidate.id.as_str()),
        "candidate ID",
    )?;

    let evidence_by_id = validate_evidence_registry(evidence)?;

    for objective in objectives {
        if objective.unit.trim().is_empty() || objective.boundary.trim().is_empty() {
            return Err(AuditError(format!(
                "objective {} has an empty unit or boundary",
                objective.id
            )));
        }
    }

    let expected_scenarios: BTreeSet<&str> = scenarios
        .iter()
        .map(|scenario| scenario.id.as_str())
        .collect();

    for candidate in candidates {
        if candidate.revision.trim().is_empty() {
            return Err(AuditError(format!(
                "candidate {} has an empty revision",
                candidate.id
            )));
        }

        let actual_scenarios: BTreeSet<&str> = candidate
            .scenario_metrics
            .keys()
            .map(String::as_str)
            .collect();

        if actual_scenarios != expected_scenarios {
            return Err(AuditError(format!(
                "candidate {} has missing or unexpected scenario metrics",
                candidate.id
            )));
        }

        if candidate.hard_gates.is_empty() {
            return Err(AuditError(format!(
                "candidate {} has no hard-gate declarations",
                candidate.id
            )));
        }

        validate_unique_nonempty(
            candidate
                .hard_gates
                .iter()
                .map(|gate| gate.gate_id.as_str()),
            "hard-gate ID",
        )?;

        for scenario in scenarios {
            let observations = candidate
                .scenario_metrics
                .get(&scenario.id)
                .ok_or_else(|| AuditError("scenario disappeared during validation".into()))?;

            if observations.len() != objectives.len() {
                return Err(AuditError(format!(
                    "candidate {} scenario {} has {} metrics; expected {}",
                    candidate.id,
                    scenario.id,
                    observations.len(),
                    objectives.len()
                )));
            }

            let mut metric_ids = BTreeSet::new();

            for observation in observations {
                if !metric_ids.insert(observation.objective_id.as_str()) {
                    return Err(AuditError(format!(
                        "candidate {} scenario {} duplicates objective {}",
                        candidate.id, scenario.id, observation.objective_id
                    )));
                }

                let objective = objectives
                    .iter()
                    .find(|item| item.id == observation.objective_id)
                    .ok_or_else(|| {
                        AuditError(format!(
                            "candidate {} references unknown objective {}",
                            candidate.id, observation.objective_id
                        ))
                    })?;

                if observation.unit != objective.unit || observation.boundary != objective.boundary
                {
                    return Err(AuditError(format!(
                        "candidate {} objective {} has mismatched unit or boundary",
                        candidate.id, objective.id
                    )));
                }

                if !observation.value.is_finite()
                    || !observation.lower_bound.is_finite()
                    || !observation.upper_bound.is_finite()
                    || observation.lower_bound > observation.value
                    || observation.value > observation.upper_bound
                {
                    return Err(AuditError(format!(
                        "candidate {} objective {} has invalid numeric bounds",
                        candidate.id, objective.id
                    )));
                }

                require_evidence(
                    &evidence_by_id,
                    &observation.evidence_id,
                    &format!("candidate {} objective {}", candidate.id, objective.id),
                )?;
            }
        }

        for gate in &candidate.hard_gates {
            require_evidence(
                &evidence_by_id,
                &gate.evidence_id,
                &format!("candidate {} gate {}", candidate.id, gate.gate_id),
            )?;
        }
    }

    Ok(evidence_by_id)
}

fn rejection_reasons(
    candidate: &Candidate,
    evidence_by_id: &BTreeMap<&str, &EvidenceRecord>,
) -> Vec<RejectionReason> {
    let mut reasons = Vec::new();

    for observations in candidate.scenario_metrics.values() {
        for observation in observations {
            let record = evidence_by_id
                .get(observation.evidence_id.as_str())
                .expect("validated evidence references remain present");
            if record.admission != EvidenceAdmission::AdmittedForEvaluation {
                reasons.push(RejectionReason::MetricEvidenceNotAdmitted(
                    observation.objective_id.clone(),
                ));
            }
        }
    }

    if candidate.hard_gates.is_empty() {
        reasons.push(RejectionReason::MissingHardGates);
        return reasons;
    }

    for gate in &candidate.hard_gates {
        match gate.disposition {
            GateDisposition::Pass => {}
            GateDisposition::Fail => {
                reasons.push(RejectionReason::GateFailed(gate.gate_id.clone()));
            }
            GateDisposition::Unknown => {
                reasons.push(RejectionReason::GateUnknown(gate.gate_id.clone()));
            }
        }

        let record = evidence_by_id
            .get(gate.evidence_id.as_str())
            .expect("validated gate evidence references remain present");
        if record.admission != EvidenceAdmission::AdmittedForEvaluation {
            reasons.push(RejectionReason::GateEvidenceNotAdmitted(
                gate.gate_id.clone(),
            ));
        }
        if gate.requires_physical_evidence && !record.evidence_class.is_physical() {
            reasons.push(RejectionReason::GateEvidenceNotPhysical(
                gate.gate_id.clone(),
            ));
        }
    }

    reasons.sort();
    reasons
}

fn robust_score(
    candidate: &Candidate,
    scenarios: &[Scenario],
    objectives: &[Objective],
) -> Result<CandidateScore, AuditError> {
    let mut robust_values = BTreeMap::new();

    for objective in objectives {
        let mut scenario_values = Vec::with_capacity(scenarios.len());

        for scenario in scenarios {
            let observations = candidate
                .scenario_metrics
                .get(&scenario.id)
                .ok_or_else(|| AuditError("missing scenario during scoring".into()))?;

            let observation = observations
                .iter()
                .find(|item| item.objective_id == objective.id)
                .ok_or_else(|| {
                    AuditError(format!(
                        "candidate {} missing objective {} in scenario {}",
                        candidate.id, objective.id, scenario.id
                    ))
                })?;

            let conservative_value = match objective.direction {
                Direction::Maximize => observation.lower_bound,
                Direction::Minimize => observation.upper_bound,
            };

            scenario_values.push(conservative_value);
        }

        let value = match objective.direction {
            Direction::Maximize => scenario_values.into_iter().fold(f64::INFINITY, f64::min),
            Direction::Minimize => scenario_values
                .into_iter()
                .fold(f64::NEG_INFINITY, f64::max),
        };

        robust_values.insert(objective.id.clone(), value);
    }

    Ok(CandidateScore {
        candidate_id: candidate.id.clone(),
        robust_values,
    })
}

fn normalized_value(score: &CandidateScore, objective: &Objective) -> Result<f64, AuditError> {
    let value = *score.robust_values.get(&objective.id).ok_or_else(|| {
        AuditError(format!(
            "score {} missing objective {}",
            score.candidate_id, objective.id
        ))
    })?;

    Ok(match objective.direction {
        Direction::Maximize => value,
        Direction::Minimize => -value,
    })
}

fn dominates(
    left: &CandidateScore,
    right: &CandidateScore,
    objectives: &[Objective],
) -> Result<bool, AuditError> {
    let mut strictly_better = false;

    for objective in objectives {
        let left_value = normalized_value(left, objective)?;
        let right_value = normalized_value(right, objective)?;

        if left_value + FLOAT_EPSILON < right_value {
            return Ok(false);
        }

        if left_value > right_value + FLOAT_EPSILON {
            strictly_better = true;
        }
    }

    Ok(strictly_better)
}

/// Validate and evaluate a finite design portfolio.
///
/// Hard gates are evaluated before Pareto ranking. A gate that requires
/// physical evidence cannot be satisfied by synthetic, vendor, literature,
/// surrogate, or calculation evidence.
pub fn audit_design_portfolio(
    scenarios: &[Scenario],
    objectives: &[Objective],
    evidence: &[EvidenceRecord],
    candidates: &[Candidate],
) -> Result<AuditReport, AuditError> {
    let evidence_by_id = validate_manifest(scenarios, objectives, evidence, candidates)?;

    let mut eligible_candidate_ids = Vec::new();
    let mut rejected = Vec::new();
    let mut scores = Vec::new();

    for candidate in candidates {
        let reasons = rejection_reasons(candidate, &evidence_by_id);

        if !reasons.is_empty() {
            rejected.push(CandidateRejection {
                candidate_id: candidate.id.clone(),
                reasons,
            });
            continue;
        }

        eligible_candidate_ids.push(candidate.id.clone());
        scores.push(robust_score(candidate, scenarios, objectives)?);
    }

    scores.sort_by(|left, right| left.candidate_id.cmp(&right.candidate_id));
    eligible_candidate_ids.sort();
    rejected.sort_by(|left, right| left.candidate_id.cmp(&right.candidate_id));

    let mut pareto_front_ids = Vec::new();

    for candidate in &scores {
        let mut is_dominated = false;

        for other in &scores {
            if candidate.candidate_id == other.candidate_id {
                continue;
            }

            if dominates(other, candidate, objectives)? {
                is_dominated = true;
                break;
            }
        }

        if !is_dominated {
            pareto_front_ids.push(candidate.candidate_id.clone());
        }
    }

    pareto_front_ids.sort();

    Ok(AuditReport {
        eligible_candidate_ids,
        pareto_front_ids,
        rejected,
        scores,
    })
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum RiskClass {
    Low,
    Moderate,
    High,
    Unknown,
    Prohibited,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExperimentOption {
    pub id: String,
    pub expected_information_gain: f64,
    pub estimated_cost: f64,
    pub duration_hours: f64,
    pub material_grams: f64,
    pub risk_class: RiskClass,
    pub estimate_evidence_id: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExperimentBudget {
    pub max_cost: f64,
    pub max_duration_hours: f64,
    pub max_material_grams: f64,
    pub max_risk_class: RiskClass,
    pub allow_no_experiment: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExperimentSelection {
    pub experiment_id: Option<String>,
    /// Transparent heuristic, not a calibrated expected-value-of-information.
    pub selection_score: Option<f64>,
    pub reason: String,
}

fn valid_nonnegative(value: f64) -> bool {
    value.is_finite() && value >= 0.0
}

fn risk_is_admissible(candidate: &RiskClass, maximum: &RiskClass) -> bool {
    match (candidate, maximum) {
        (RiskClass::Low, _) => true,
        (RiskClass::Moderate, RiskClass::Moderate | RiskClass::High) => true,
        (RiskClass::High, RiskClass::High) => true,
        _ => false,
    }
}

/// Select a finite, predeclared experiment under a strict resource/risk budget.
///
/// The score is an explicitly heuristic information-gain-per-normalized-resource
/// ranking. It is not a Bayesian VOI estimate and must not be represented as one.
pub fn select_experiment(
    options: &[ExperimentOption],
    evidence: &[EvidenceRecord],
    budget: &ExperimentBudget,
) -> Result<ExperimentSelection, AuditError> {
    for (label, value) in [
        ("max cost", budget.max_cost),
        ("max duration", budget.max_duration_hours),
        ("max material", budget.max_material_grams),
    ] {
        if !valid_nonnegative(value) {
            return Err(AuditError(format!("invalid experiment budget: {label}")));
        }
    }

    validate_unique_nonempty(
        options.iter().map(|option| option.id.as_str()),
        "experiment ID",
    )?;
    let evidence_by_id = validate_evidence_registry(evidence)?;

    let mut eligible: Vec<(f64, &ExperimentOption)> = Vec::new();

    for option in options {
        if !valid_nonnegative(option.expected_information_gain)
            || option.expected_information_gain > 1.0
            || !valid_nonnegative(option.estimated_cost)
            || !valid_nonnegative(option.duration_hours)
            || !valid_nonnegative(option.material_grams)
        {
            return Err(AuditError(format!(
                "experiment {} has invalid estimate or resource value",
                option.id
            )));
        }

        let estimate_record = require_evidence(
            &evidence_by_id,
            &option.estimate_evidence_id,
            &format!("experiment {} estimate", option.id),
        )?;
        if estimate_record.admission != EvidenceAdmission::AdmittedForEvaluation {
            continue;
        }

        if option.estimated_cost > budget.max_cost
            || option.duration_hours > budget.max_duration_hours
            || option.material_grams > budget.max_material_grams
            || !risk_is_admissible(&option.risk_class, &budget.max_risk_class)
        {
            continue;
        }

        let cost_fraction = if budget.max_cost > 0.0 {
            option.estimated_cost / budget.max_cost
        } else if option.estimated_cost == 0.0 {
            0.0
        } else {
            continue;
        };

        let time_fraction = if budget.max_duration_hours > 0.0 {
            option.duration_hours / budget.max_duration_hours
        } else if option.duration_hours == 0.0 {
            0.0
        } else {
            continue;
        };

        let material_fraction = if budget.max_material_grams > 0.0 {
            option.material_grams / budget.max_material_grams
        } else if option.material_grams == 0.0 {
            0.0
        } else {
            continue;
        };

        let denominator = 1.0 + cost_fraction + time_fraction + material_fraction;
        let score = option.expected_information_gain / denominator;
        eligible.push((score, option));
    }

    eligible.sort_by(|left, right| {
        right
            .0
            .total_cmp(&left.0)
            .then_with(|| left.1.id.cmp(&right.1.id))
    });

    if let Some((score, option)) = eligible.first() {
        return Ok(ExperimentSelection {
            experiment_id: Some(option.id.clone()),
            selection_score: Some(*score),
            reason: "highest eligible heuristic information-gain score within declared cost, time, material and risk budgets".into(),
        });
    }

    if budget.allow_no_experiment {
        Ok(ExperimentSelection {
            experiment_id: None,
            selection_score: None,
            reason: "no experiment satisfies every declared resource and risk constraint; STOP / NO_EXPERIMENT retained".into(),
        })
    } else {
        Err(AuditError(
            "no experiment satisfies the declared resource and risk budget".into(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn objectives() -> Vec<Objective> {
        vec![
            Objective {
                id: "energy".into(),
                unit: "Wh/kg".into(),
                boundary: "complete_cell".into(),
                direction: Direction::Maximize,
            },
            Objective {
                id: "throughput".into(),
                unit: "kWh".into(),
                boundary: "cell_lifetime".into(),
                direction: Direction::Maximize,
            },
            Objective {
                id: "cost".into(),
                unit: "USD/kWh".into(),
                boundary: "cell_factory_gate".into(),
                direction: Direction::Minimize,
            },
        ]
    }

    fn evidence() -> Vec<EvidenceRecord> {
        vec![
            EvidenceRecord {
                evidence_id: "SYNTH-EVIDENCE".into(),
                evidence_class: EvidenceClass::SyntheticFixture,
                source_sha256: "1".repeat(64),
                claim_scope: "synthetic objective values".into(),
                validity_domain: "test fixtures only".into(),
                admission: EvidenceAdmission::AdmittedForEvaluation,
            },
            EvidenceRecord {
                evidence_id: "SYNTH-GATE".into(),
                evidence_class: EvidenceClass::SyntheticFixture,
                source_sha256: "2".repeat(64),
                claim_scope: "synthetic hard-gate fixture".into(),
                validity_domain: "test fixtures only".into(),
                admission: EvidenceAdmission::AdmittedForEvaluation,
            },
            EvidenceRecord {
                evidence_id: "SYNTH-VOI".into(),
                evidence_class: EvidenceClass::SyntheticFixture,
                source_sha256: "3".repeat(64),
                claim_scope: "synthetic experiment estimate".into(),
                validity_domain: "test fixtures only".into(),
                admission: EvidenceAdmission::AdmittedForEvaluation,
            },
        ]
    }

    fn candidate(id: &str, values: [f64; 3], gate_disposition: GateDisposition) -> Candidate {
        let metrics = |offset: f64| {
            vec![
                MetricObservation {
                    objective_id: "energy".into(),
                    unit: "Wh/kg".into(),
                    boundary: "complete_cell".into(),
                    value: values[0] + offset,
                    lower_bound: values[0] + offset - 1.0,
                    upper_bound: values[0] + offset + 1.0,
                    evidence_id: "SYNTH-EVIDENCE".into(),
                },
                MetricObservation {
                    objective_id: "throughput".into(),
                    unit: "kWh".into(),
                    boundary: "cell_lifetime".into(),
                    value: values[1] + offset,
                    lower_bound: values[1] + offset - 10.0,
                    upper_bound: values[1] + offset + 10.0,
                    evidence_id: "SYNTH-EVIDENCE".into(),
                    evidence_class: EvidenceClass::SyntheticFixture,
                },
                MetricObservation {
                    objective_id: "cost".into(),
                    unit: "USD/kWh".into(),
                    boundary: "cell_factory_gate".into(),
                    value: values[2] - offset,
                    lower_bound: values[2] - offset - 1.0,
                    upper_bound: values[2] - offset + 1.0,
                    evidence_id: "SYNTH-EVIDENCE".into(),
                    evidence_class: EvidenceClass::SyntheticFixture,
                },
            ]
        };

        let mut scenario_metrics = BTreeMap::new();
        scenario_metrics.insert("nominal".into(), metrics(0.0));
        scenario_metrics.insert("hot".into(), metrics(-2.0));

        Candidate {
            id: id.into(),
            revision: "r1".into(),
            scenario_metrics,
            hard_gates: vec![
                HardGate {
                    gate_id: "declared_mission_feasible".into(),
                    disposition: GateDisposition::Pass,
                    evidence_id: "SYNTH-GATE".into(),
                    requires_physical_evidence: false,
                },
                HardGate {
                    gate_id: "physical_safety".into(),
                    disposition: gate_disposition,
                    evidence_id: "SYNTH-GATE".into(),
                    requires_physical_evidence: true,
                },
            ],
        }
    }

    fn scenarios() -> Vec<Scenario> {
        vec![
            Scenario {
                id: "nominal".into(),
            },
            Scenario { id: "hot".into() },
        ]
    }

    #[test]
    fn unknown_physical_safety_gate_excludes_candidate() {
        let result = audit_design_portfolio(
            &scenarios(),
            &objectives(),
            &evidence(),
            &[candidate(
                "a",
                [300.0, 1800.0, 100.0],
                GateDisposition::Unknown,
            )],
        )
        .expect("manifest should be structurally valid");

        assert!(result.eligible_candidate_ids.is_empty());
        assert!(result.pareto_front_ids.is_empty());
        assert_eq!(result.rejected.len(), 1);
        assert!(
            result.rejected[0]
                .reasons
                .contains(&RejectionReason::GateUnknown("physical_safety".into()))
        );
        assert!(
            result.rejected[0]
                .reasons
                .contains(&RejectionReason::GateEvidenceNotPhysical(
                    "physical_safety".into()
                ))
        );
    }

    #[test]
    fn synthetic_evidence_cannot_satisfy_physical_gate_even_when_declared_pass() {
        let result = audit_design_portfolio(
            &scenarios(),
            &objectives(),
            &[candidate(
                "a",
                [300.0, 1800.0, 100.0],
                GateDisposition::Pass,
            )],
        )
        .expect("manifest should be structurally valid");

        assert!(result.eligible_candidate_ids.is_empty());
        assert!(
            result.rejected[0]
                .reasons
                .contains(&RejectionReason::GateEvidenceNotPhysical(
                    "physical_safety".into()
                ))
        );
    }

    #[test]
    fn pareto_front_preserves_tradeoffs_and_discards_dominated_design() {
        let candidates = vec![
            candidate("high-energy", [320.0, 1800.0, 105.0], GateDisposition::Pass),
            candidate(
                "long-life-low-cost",
                [300.0, 2200.0, 90.0],
                GateDisposition::Pass,
            ),
            candidate("dominated", [280.0, 1500.0, 125.0], GateDisposition::Pass),
        ];

        // This test deliberately disables physical-gate requirements only for
        // the mathematical Pareto oracle below.
        let mut eligible_candidates = candidates;
        for item in &mut eligible_candidates {
            item.hard_gates
                .retain(|gate| gate.gate_id != "physical_safety");
        }

        let result = audit_design_portfolio(&scenarios(), &objectives(), &evidence(), &eligible_candidates)
            .expect("valid finite portfolio");

        assert_eq!(
            result.pareto_front_ids,
            vec!["high-energy".to_string(), "long-life-low-cost".to_string()]
        );
        assert!(!result.pareto_front_ids.contains(&"dominated".to_string()));
    }

    #[test]
    fn mismatched_unit_or_boundary_is_rejected() {
        let mut item = candidate("a", [300.0, 1800.0, 100.0], GateDisposition::Unknown);
        item.scenario_metrics
            .get_mut("nominal")
            .expect("nominal scenario")
            .get_mut(0)
            .expect("energy metric")
            .boundary = "active_material_only".into();

        let error = audit_design_portfolio(&scenarios(), &objectives(), &evidence(), &[item])
            .expect_err("boundary mismatch must fail closed");

        assert!(error.0.contains("mismatched unit or boundary"));
    }

    #[test]
    fn missing_scenario_metric_is_rejected() {
        let mut item = candidate("a", [300.0, 1800.0, 100.0], GateDisposition::Unknown);
        item.scenario_metrics.remove("hot");

        let error = audit_design_portfolio(&scenarios(), &objectives(), &[item])
            .expect_err("missing scenario must fail closed");

        assert!(error.0.contains("missing or unexpected scenario metrics"));
    }

    #[test]
    fn scenario_aggregation_uses_conservative_interval_bound() {
        let mut item = candidate("a", [300.0, 1800.0, 100.0], GateDisposition::Pass);
        item.hard_gates
            .retain(|gate| gate.gate_id != "physical_safety");

        let result = audit_design_portfolio(&scenarios(), &objectives(), &[item])
            .expect("valid finite portfolio");
        let score = result.scores.first().expect("one score");

        // Hot scenario has energy interval [297, 299], lower than nominal [299, 301].
        assert_eq!(score.robust_values["energy"], 297.0);
    }

    #[test]
    fn experiment_selector_respects_budget_and_risk() {
        let options = vec![
            ExperimentOption {
                id: "exp-expensive".into(),
                expected_information_gain: 0.9,
                estimated_cost: 5000.0,
                duration_hours: 20.0,
                material_grams: 100.0,
                risk_class: RiskClass::Moderate,
                estimate_evidence_id: "SYNTH-VOI".into(),
            },
            ExperimentOption {
                id: "exp-small".into(),
                expected_information_gain: 0.4,
                estimated_cost: 100.0,
                duration_hours: 2.0,
                material_grams: 5.0,
                risk_class: RiskClass::Low,
                estimate_evidence_id: "SYNTH-VOI".into(),
            },
        ];
        let budget = ExperimentBudget {
            max_cost: 250.0,
            max_duration_hours: 4.0,
            max_material_grams: 10.0,
            max_risk_class: RiskClass::Low,
            allow_no_experiment: true,
        };

        let selection = select_experiment(&options, &evidence(), &budget).expect("valid options");
        assert_eq!(selection.experiment_id, Some("exp-small".into()));
    }

    #[test]
    fn no_experiment_remains_available_when_every_option_is_over_budget() {
        let options = vec![ExperimentOption {
            id: "too-large".into(),
            expected_information_gain: 1.0,
            estimated_cost: 10_000.0,
            duration_hours: 100.0,
            material_grams: 1000.0,
            risk_class: RiskClass::High,
            estimate_evidence_id: "SYNTH-VOI".into(),
        }];
        let budget = ExperimentBudget {
            max_cost: 100.0,
            max_duration_hours: 1.0,
            max_material_grams: 1.0,
            max_risk_class: RiskClass::Low,
            allow_no_experiment: true,
        };

        let selection = select_experiment(&options, &budget).expect("no-op is admissible");
        assert_eq!(selection.experiment_id, None);
    }

    #[test]
    fn candidate_evidence_must_resolve_through_shared_registry() {
        let mut item = candidate("a", [300.0, 1800.0, 100.0], GateDisposition::Unknown);
        item.scenario_metrics
            .get_mut("nominal")
            .expect("nominal scenario")[0]
            .evidence_id = "UNREGISTERED-EVIDENCE".into();

        let error = audit_design_portfolio(
            &scenarios(),
            &objectives(),
            &evidence(),
            &[item],
        )
        .expect_err("unknown evidence ID must fail closed");

        assert!(error.0.contains("references unknown evidence ID"));
    }

    #[test]
    fn evidence_not_admitted_for_evaluation_cannot_qualify_candidate() {
        let mut records = evidence();
        records[0].admission = EvidenceAdmission::AdmittedForResearch;
        let item = candidate("a", [300.0, 1800.0, 100.0], GateDisposition::Unknown);

        let result = audit_design_portfolio(
            &scenarios(),
            &objectives(),
            &records,
            &[item],
        )
        .expect("well-formed but research-only evidence remains auditable");

        assert!(result.eligible_candidate_ids.is_empty());
        assert!(result.rejected[0]
            .reasons
            .contains(&RejectionReason::MetricEvidenceNotAdmitted("energy".into())));
    }

    #[test]
    fn evidence_digest_must_be_well_formed_sha256() {
        let mut records = evidence();
        records[0].source_sha256 = "not-a-digest".into();
        let item = candidate("a", [300.0, 1800.0, 100.0], GateDisposition::Unknown);

        let error = audit_design_portfolio(
            &scenarios(),
            &objectives(),
            &records,
            &[item],
        )
        .expect_err("invalid digest syntax must fail closed");

        assert!(error.0.contains("lowercase 64-character SHA-256"));
    }

    #[test]
    fn canonical_frontier_is_independent_of_candidate_input_order() {
        let a = candidate("a", [320.0, 1800.0, 105.0], GateDisposition::Pass);
        let b = candidate("b", [300.0, 2200.0, 90.0], GateDisposition::Pass);
        let mut first = vec![a.clone(), b.clone()];
        let mut second = vec![b, a];

        for item in first.iter_mut().chain(second.iter_mut()) {
            item.hard_gates.retain(|gate| gate.gate_id != "physical_safety");
        }

        let one = audit_design_portfolio(&scenarios(), &objectives(), &evidence(), &first).expect("first order");
        let two =
            audit_design_portfolio(&scenarios(), &objectives(), &evidence(), &second).expect("second order");

        assert_eq!(one.pareto_front_ids, two.pareto_front_ids);
        assert_eq!(one.scores, two.scores);
    }
}

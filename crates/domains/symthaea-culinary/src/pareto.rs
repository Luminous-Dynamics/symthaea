//! Evidence-aware multi-objective comparison for culinary candidates.
//!
//! This module intentionally does not produce one opaque "best recipe" score.
//! Hard gates control eligibility; only candidates with every gate explicitly
//! passing and every objective backed by a finite, sourced value enter the
//! Pareto comparison. Missing objectives remain unranked instead of being
//! imputed, and unknown gates never count as passes.
//!
//! Evidence references are identifiers supplied by the caller, not cryptographic
//! verification. This module proves only deterministic comparison semantics.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GateStatus {
    Pass,
    Fail,
    Unknown,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObjectiveDirection {
    Minimize,
    Maximize,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GateEvidence {
    pub key: String,
    pub status: GateStatus,
    /// A stable source-evidence or verification-receipt reference.
    /// Pass/Fail requires a reference; Unknown may omit one.
    pub evidence_ref: Option<String>,
}

impl GateEvidence {
    pub fn pass(key: impl Into<String>, evidence_ref: impl Into<String>) -> Self {
        Self {
            key: key.into(),
            status: GateStatus::Pass,
            evidence_ref: Some(evidence_ref.into()),
        }
    }

    pub fn fail(key: impl Into<String>, evidence_ref: impl Into<String>) -> Self {
        Self {
            key: key.into(),
            status: GateStatus::Fail,
            evidence_ref: Some(evidence_ref.into()),
        }
    }

    pub fn unknown(key: impl Into<String>) -> Self {
        Self {
            key: key.into(),
            status: GateStatus::Unknown,
            evidence_ref: None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ObjectiveMetric {
    pub key: String,
    pub direction: ObjectiveDirection,
    /// None means unknown/missing, never zero and never a default estimate.
    pub value: Option<f64>,
    /// A source-data version, measurement, or deterministic calculation receipt.
    pub evidence_ref: Option<String>,
}

impl ObjectiveMetric {
    pub fn measured(
        key: impl Into<String>,
        direction: ObjectiveDirection,
        value: f64,
        evidence_ref: impl Into<String>,
    ) -> Self {
        Self {
            key: key.into(),
            direction,
            value: Some(value),
            evidence_ref: Some(evidence_ref.into()),
        }
    }

    pub fn unknown(key: impl Into<String>, direction: ObjectiveDirection) -> Self {
        Self {
            key: key.into(),
            direction,
            value: None,
            evidence_ref: None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CulinaryCandidate {
    pub id: String,
    pub gates: Vec<GateEvidence>,
    pub objectives: Vec<ObjectiveMetric>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GateDisposition {
    Feasible,
    Blocked,
    Indeterminate,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RankingDisposition {
    Ranked,
    IncompleteMetrics,
    NotRanked,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CandidateAssessment {
    pub id: String,
    pub gate_disposition: GateDisposition,
    pub ranking_disposition: RankingDisposition,
    pub on_frontier: bool,
    /// IDs this candidate Pareto-dominates, in lexical order.
    pub dominates: Vec<String>,
    /// IDs that Pareto-dominate this candidate, in lexical order.
    pub dominated_by: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParetoReport {
    /// Canonical, lexically sorted objective names.
    pub objective_keys: Vec<String>,
    /// Candidate assessments are returned in lexical candidate-ID order.
    pub candidates: Vec<CandidateAssessment>,
    /// Feasible, fully measured, non-dominated candidates in lexical order.
    pub frontier_ids: Vec<String>,
    pub blocked_ids: Vec<String>,
    pub indeterminate_ids: Vec<String>,
    pub incomplete_metric_ids: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ParetoError {
    NoCandidates,
    NoHardGates,
    NoObjectives,
    EmptyCandidateId,
    DuplicateCandidateId(String),
    EmptyGateKey { candidate_id: String },
    DuplicateGateKey { candidate_id: String, key: String },
    GateSchemaMismatch { candidate_id: String },
    MissingGateEvidence { candidate_id: String, key: String },
    EmptyObjectiveKey { candidate_id: String },
    DuplicateObjectiveKey { candidate_id: String, key: String },
    ObjectiveSchemaMismatch { candidate_id: String },
    NonFiniteObjective { candidate_id: String, key: String },
    MissingObjectiveEvidence { candidate_id: String, key: String },
    EmptyEvidenceReference { candidate_id: String, key: String },
}

impl fmt::Display for ParetoError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoCandidates => write!(formatter, "at least one candidate is required"),
            Self::NoHardGates => write!(formatter, "at least one hard gate is required"),
            Self::NoObjectives => write!(formatter, "at least one objective is required"),
            Self::EmptyCandidateId => write!(formatter, "candidate ID must not be empty"),
            Self::DuplicateCandidateId(id) => write!(formatter, "duplicate candidate ID: {id}"),
            Self::EmptyGateKey { candidate_id } => {
                write!(formatter, "candidate {candidate_id} has an empty gate key")
            }
            Self::DuplicateGateKey { candidate_id, key } => {
                write!(formatter, "candidate {candidate_id} repeats gate {key}")
            }
            Self::GateSchemaMismatch { candidate_id } => {
                write!(formatter, "candidate {candidate_id} has a different gate schema")
            }
            Self::MissingGateEvidence { candidate_id, key } => write!(
                formatter,
                "candidate {candidate_id} marks gate {key} pass/fail without evidence"
            ),
            Self::EmptyObjectiveKey { candidate_id } => {
                write!(formatter, "candidate {candidate_id} has an empty objective key")
            }
            Self::DuplicateObjectiveKey { candidate_id, key } => {
                write!(formatter, "candidate {candidate_id} repeats objective {key}")
            }
            Self::ObjectiveSchemaMismatch { candidate_id } => {
                write!(formatter, "candidate {candidate_id} has a different objective schema")
            }
            Self::NonFiniteObjective { candidate_id, key } => write!(
                formatter,
                "candidate {candidate_id} has a non-finite value for objective {key}"
            ),
            Self::MissingObjectiveEvidence { candidate_id, key } => write!(
                formatter,
                "candidate {candidate_id} supplies objective {key} without evidence"
            ),
            Self::MissingObjectiveEvidence { candidate_id, key } => write!(
                formatter,
                "candidate {candidate_id} supplies objective {key} without evidence"
            ),
            Self::EmptyEvidenceReference { candidate_id, key } => write!(
                formatter,
                "candidate {candidate_id} supplies an empty evidence reference for {key}"
            ),
        }
    }
}

impl std::error::Error for ParetoError {}

/// Validate input schemas, enforce the hard-gate boundary, and compute the
/// Pareto frontier without blending unlike objectives into a weighted score.
///
/// All candidates must declare the same gate keys and objective keys/directions.
/// A Pass or Fail gate needs a non-empty evidence reference. A present objective
/// value must be finite and have a non-empty evidence reference. Missing
/// objective values are allowed, but that candidate is not ranked.
///
/// Dominance is exact and deterministic: A dominates B iff A is no worse in
/// every objective and strictly better in at least one. Values must share units
/// for each objective key; this function does not convert units or validate
/// source evidence. Use normalized, provenance-backed metrics at the call site.
pub fn evaluate_frontier(
    candidates: &[CulinaryCandidate],
) -> Result<ParetoReport, ParetoError> {
    if candidates.is_empty() {
        return Err(ParetoError::NoCandidates);
    }

    let mut ids = BTreeSet::new();
    for candidate in candidates {
        if candidate.id.trim().is_empty() {
            return Err(ParetoError::EmptyCandidateId);
        }
        if !ids.insert(candidate.id.clone()) {
            return Err(ParetoError::DuplicateCandidateId(candidate.id.clone()));
        }
    }

    let first = &candidates[0];
    if first.gates.is_empty() {
        return Err(ParetoError::NoHardGates);
    }
    if first.objectives.is_empty() {
        return Err(ParetoError::NoObjectives);
    }

    let canonical_gates = validate_gates(first)?;
    let canonical_objectives = validate_objectives(first)?;

    for candidate in candidates {
        let gates = validate_gates(candidate)?;
        if gates != canonical_gates {
            return Err(ParetoError::GateSchemaMismatch {
                candidate_id: candidate.id.clone(),
            });
        }
        let objectives = validate_objectives(candidate)?;
        if objectives != canonical_objectives {
            return Err(ParetoError::ObjectiveSchemaMismatch {
                candidate_id: candidate.id.clone(),
            });
        }
    }

    let by_id: BTreeMap<&str, &CulinaryCandidate> = candidates
        .iter()
        .map(|candidate| (candidate.id.as_str(), candidate))
        .collect();
    let ids_sorted: Vec<&str> = by_id.keys().copied().collect();
    let mut assessments: BTreeMap<String, CandidateAssessment> = BTreeMap::new();

    for id in &ids_sorted {
        let candidate = by_id[id];
        let gate_disposition = gate_disposition(candidate);
        let ranking_disposition = if gate_disposition != GateDisposition::Feasible {
            RankingDisposition::NotRanked
        } else if candidate.objectives.iter().any(|metric| metric.value.is_none()) {
            RankingDisposition::IncompleteMetrics
        } else {
            RankingDisposition::Ranked
        };

        assessments.insert(
            candidate.id.clone(),
            CandidateAssessment {
                id: candidate.id.clone(),
                gate_disposition,
                ranking_disposition,
                on_frontier: false,
                dominates: Vec::new(),
                dominated_by: Vec::new(),
            },
        );
    }

    for (left_index, left_id) in ids_sorted.iter().enumerate() {
        let left = by_id[left_id];
        if assessments[*left_id].ranking_disposition != RankingDisposition::Ranked {
            continue;
        }

        for right_id in ids_sorted.iter().skip(left_index + 1) {
            let right = by_id[right_id];
            if assessments[*right_id].ranking_disposition != RankingDisposition::Ranked {
                continue;
            }

            if dominates(left, right) {
                assessments
                    .get_mut(*left_id)
                    .expect("assessment created for every unique ID")
                    .dominates
                    .push((*right_id).to_owned());
                assessments
                    .get_mut(*right_id)
                    .expect("assessment created for every unique ID")
                    .dominated_by
                    .push((*left_id).to_owned());
            } else if dominates(right, left) {
                assessments
                    .get_mut(*right_id)
                    .expect("assessment created for every unique ID")
                    .dominates
                    .push((*left_id).to_owned());
                assessments
                    .get_mut(*left_id)
                    .expect("assessment created for every unique ID")
                    .dominated_by
                    .push((*right_id).to_owned());
            }
        }
    }

    for assessment in assessments.values_mut() {
        assessment.dominates.sort();
        assessment.dominated_by.sort();
        assessment.on_frontier = assessment.ranking_disposition == RankingDisposition::Ranked
            && assessment.dominated_by.is_empty();
    }

    let candidates: Vec<CandidateAssessment> = assessments.into_values().collect();
    let frontier_ids = candidates
        .iter()
        .filter(|assessment| assessment.on_frontier)
        .map(|assessment| assessment.id.clone())
        .collect();
    let blocked_ids = candidates
        .iter()
        .filter(|assessment| assessment.gate_disposition == GateDisposition::Blocked)
        .map(|assessment| assessment.id.clone())
        .collect();
    let indeterminate_ids = candidates
        .iter()
        .filter(|assessment| assessment.gate_disposition == GateDisposition::Indeterminate)
        .map(|assessment| assessment.id.clone())
        .collect();
    let incomplete_metric_ids = candidates
        .iter()
        .filter(|assessment| {
            assessment.ranking_disposition == RankingDisposition::IncompleteMetrics
        })
        .map(|assessment| assessment.id.clone())
        .collect();

    Ok(ParetoReport {
        objective_keys: canonical_objectives
            .iter()
            .map(|(key, _)| key.clone())
            .collect(),
        candidates,
        frontier_ids,
        blocked_ids,
        indeterminate_ids,
        incomplete_metric_ids,
    })
}

fn validate_gates(candidate: &CulinaryCandidate) -> Result<Vec<String>, ParetoError> {
    let mut keys = Vec::with_capacity(candidate.gates.len());
    let mut seen = BTreeSet::new();

    for gate in &candidate.gates {
        if gate.key.trim().is_empty() {
            return Err(ParetoError::EmptyGateKey {
                candidate_id: candidate.id.clone(),
            });
        }
        if !seen.insert(gate.key.clone()) {
            return Err(ParetoError::DuplicateGateKey {
                candidate_id: candidate.id.clone(),
                key: gate.key.clone(),
            });
        }
        match gate.status {
            GateStatus::Unknown => {}
            GateStatus::Pass | GateStatus::Fail => {
                validate_evidence_ref(candidate, &gate.key, gate.evidence_ref.as_deref())?;
            }
        }
        keys.push(gate.key.clone());
    }
    keys.sort();
    Ok(keys)
}

fn validate_objectives(
    candidate: &CulinaryCandidate,
) -> Result<Vec<(String, ObjectiveDirection)>, ParetoError> {
    let mut schema = Vec::with_capacity(candidate.objectives.len());
    let mut seen = BTreeSet::new();

    for metric in &candidate.objectives {
        if metric.key.trim().is_empty() {
            return Err(ParetoError::EmptyObjectiveKey {
                candidate_id: candidate.id.clone(),
            });
        }
        if !seen.insert(metric.key.clone()) {
            return Err(ParetoError::DuplicateObjectiveKey {
                candidate_id: candidate.id.clone(),
                key: metric.key.clone(),
            });
        }
        if let Some(value) = metric.value {
            if !value.is_finite() {
                return Err(ParetoError::NonFiniteObjective {
                    candidate_id: candidate.id.clone(),
                    key: metric.key.clone(),
                });
            }
            validate_objective_evidence_ref(
                candidate,
                &metric.key,
                metric.evidence_ref.as_deref(),
            )?;
        }
        schema.push((metric.key.clone(), metric.direction));
    }
    schema.sort();
    Ok(schema)
}

fn validate_objective_evidence_ref(
    candidate: &CulinaryCandidate,
    key: &str,
    evidence_ref: Option<&str>,
) -> Result<(), ParetoError> {
    match evidence_ref {
        None => Err(ParetoError::MissingObjectiveEvidence {
            candidate_id: candidate.id.clone(),
            key: key.to_owned(),
        }),
        Some(reference) if reference.trim().is_empty() => {
            Err(ParetoError::EmptyEvidenceReference {
                candidate_id: candidate.id.clone(),
                key: key.to_owned(),
            })
        }
        Some(_) => Ok(()),
    }
}

fn validate_evidence_ref(
    candidate: &CulinaryCandidate,
    key: &str,
    evidence_ref: Option<&str>,
) -> Result<(), ParetoError> {
    match evidence_ref {
        None => Err(ParetoError::MissingGateEvidence {
            candidate_id: candidate.id.clone(),
            key: key.to_owned(),
        }),
        Some(reference) if reference.trim().is_empty() => {
            Err(ParetoError::EmptyEvidenceReference {
                candidate_id: candidate.id.clone(),
                key: key.to_owned(),
            })
        }
        Some(_) => Ok(()),
    }
}

fn gate_disposition(candidate: &CulinaryCandidate) -> GateDisposition {
    if candidate.gates.iter().any(|gate| gate.status == GateStatus::Fail) {
        GateDisposition::Blocked
    } else if candidate
        .gates
        .iter()
        .any(|gate| gate.status == GateStatus::Unknown)
    {
        GateDisposition::Indeterminate
    } else {
        GateDisposition::Feasible
    }
}

fn dominates(left: &CulinaryCandidate, right: &CulinaryCandidate) -> bool {
    let mut strictly_better = false;

    for left_metric in &left.objectives {
        // Match by key: equivalent schemas may declare metrics in different orders.
        let right_metric = right
            .objectives
            .iter()
            .find(|metric| metric.key == left_metric.key)
            .expect("validated candidates share every objective key");
        let left_value = left_metric.value.expect("ranked metrics are complete");
        let right_value = right_metric.value.expect("ranked metrics are complete");
        let (no_worse, better) = match left_metric.direction {
            ObjectiveDirection::Minimize => (
                left_value <= right_value,
                left_value < right_value,
            ),
            ObjectiveDirection::Maximize => (
                left_value >= right_value,
                left_value > right_value,
            ),
        };
        if !no_worse {
            return false;
        }
        strictly_better |= better;
    }

    strictly_better
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candidate(
        id: &str,
        cost: Option<f64>,
        adequacy: Option<f64>,
        waste: Option<f64>,
    ) -> CulinaryCandidate {
        CulinaryCandidate {
            id: id.to_owned(),
            gates: vec![
                GateEvidence::pass("food_safety_constraints", format!("gate:{id}:safety")),
                GateEvidence::pass("allergen_evidence_reviewed", format!("gate:{id}:allergens")),
            ],
            objectives: vec![
                metric(id, "cost_per_serving", ObjectiveDirection::Minimize, cost),
                metric(id, "nutrition_adequacy", ObjectiveDirection::Maximize, adequacy),
                metric(id, "avoidable_waste", ObjectiveDirection::Minimize, waste),
            ],
        }
    }

    fn metric(
        id: &str,
        key: &str,
        direction: ObjectiveDirection,
        value: Option<f64>,
    ) -> ObjectiveMetric {
        match value {
            Some(value) => ObjectiveMetric::measured(
                key,
                direction,
                value,
                format!("metric:{id}:{key}"),
            ),
            None => ObjectiveMetric::unknown(key, direction),
        }
    }

    #[test]
    fn pareto_frontier_preserves_real_tradeoffs_and_excludes_dominated_candidates() {
        let candidates = vec![
            candidate("balanced", Some(8.0), Some(8.0), Some(2.0)),
            candidate("nutrient_dense", Some(11.0), Some(10.0), Some(1.0)),
            candidate("dominated", Some(14.0), Some(6.0), Some(4.0)),
        ];

        let report = evaluate_frontier(&candidates).unwrap();

        assert_eq!(report.frontier_ids, vec!["balanced", "nutrient_dense"]);
        let dominated = report.candidates.iter().find(|item| item.id == "dominated").unwrap();
        assert_eq!(
            dominated.dominated_by,
            vec!["balanced".to_owned(), "nutrient_dense".to_owned()]
        );
        assert!(!dominated.on_frontier);
    }

    #[test]
    fn hard_gate_failure_cannot_be_compensated_by_better_objectives() {
        let mut unsafe_candidate =
            candidate("blocked_but_cheap", Some(0.1), Some(100.0), Some(0.0));
        unsafe_candidate.gates[0] =
            GateEvidence::fail("food_safety_constraints", "receipt:known-violation");

        let report = evaluate_frontier(&[
            candidate("eligible", Some(5.0), Some(6.0), Some(3.0)),
            unsafe_candidate,
        ])
        .unwrap();

        assert_eq!(report.frontier_ids, vec!["eligible"]);
        assert_eq!(report.blocked_ids, vec!["blocked_but_cheap"]);
        let blocked = report
            .candidates
            .iter()
            .find(|item| item.id == "blocked_but_cheap")
            .unwrap();
        assert_eq!(blocked.ranking_disposition, RankingDisposition::NotRanked);
        assert!(!blocked.on_frontier);
    }

    #[test]
    fn unknown_gate_is_indeterminate_not_feasible() {
        let mut candidate = candidate("uncertain", Some(1.0), Some(100.0), Some(0.0));
        candidate.gates[0] = GateEvidence::unknown("food_safety_constraints");

        let report = evaluate_frontier(&[candidate]).unwrap();
        assert_eq!(report.indeterminate_ids, vec!["uncertain"]);
        assert!(report.frontier_ids.is_empty());
        assert_eq!(
            report.candidates[0].gate_disposition,
            GateDisposition::Indeterminate
        );
    }

    #[test]
    fn missing_metric_is_not_imputed_or_ranked() {
        let report = evaluate_frontier(&[
            candidate("complete", Some(4.0), Some(7.0), Some(1.0)),
            candidate("missing_cost", None, Some(9.0), Some(0.5)),
        ])
        .unwrap();

        assert_eq!(report.frontier_ids, vec!["complete"]);
        assert_eq!(report.incomplete_metric_ids, vec!["missing_cost"]);
        let missing = report
            .candidates
            .iter()
            .find(|item| item.id == "missing_cost")
            .unwrap();
        assert_eq!(
            missing.ranking_disposition,
            RankingDisposition::IncompleteMetrics
        );
    }

    #[test]
    fn identical_candidates_do_not_dominate_each_other() {
        let left = candidate("a", Some(5.0), Some(8.0), Some(2.0));
        let right = candidate("b", Some(5.0), Some(8.0), Some(2.0));

        let report = evaluate_frontier(&[right, left]).unwrap();
        assert_eq!(report.frontier_ids, vec!["a", "b"]);
        assert!(report.candidates.iter().all(|item| {
            item.dominated_by.is_empty() && item.dominates.is_empty() && item.on_frontier
        }));
    }

    #[test]
    fn incomparable_missing_and_unknown_states_are_explicitly_separate() {
        let mut unknown_gate = candidate("unknown_gate", Some(1.0), Some(10.0), Some(0.0));
        unknown_gate.gates[0] = GateEvidence::unknown("food_safety_constraints");
        let missing_metric = candidate("missing_metric", Some(1.0), None, Some(0.0));

        let report = evaluate_frontier(&[unknown_gate, missing_metric]).unwrap();
        assert_eq!(report.indeterminate_ids, vec!["unknown_gate"]);
        assert_eq!(report.incomplete_metric_ids, vec!["missing_metric"]);
        assert!(report.frontier_ids.is_empty());
    }

    #[test]
    fn rejects_non_finite_objectives_instead_of_silently_ordering_them() {
        let mut invalid = candidate("nan_cost", Some(1.0), Some(8.0), Some(2.0));
        invalid.objectives[0].value = Some(f64::NAN);

        assert_eq!(
            evaluate_frontier(&[invalid]),
            Err(ParetoError::NonFiniteObjective {
                candidate_id: "nan_cost".to_owned(),
                key: "cost_per_serving".to_owned(),
            })
        );
    }

    #[test]
    fn rejects_a_passed_gate_without_an_evidence_reference() {
        let mut invalid = candidate("unsubstantiated", Some(5.0), Some(8.0), Some(2.0));
        invalid.gates[0].evidence_ref = None;

        assert_eq!(
            evaluate_frontier(&[invalid]),
            Err(ParetoError::MissingGateEvidence {
                candidate_id: "unsubstantiated".to_owned(),
                key: "food_safety_constraints".to_owned(),
            })
        );
    }

    #[test]
    fn rejects_metric_values_without_provenance_references() {
        let mut invalid = candidate("untraced_cost", Some(5.0), Some(8.0), Some(2.0));
        invalid.objectives[0].evidence_ref = None;

        assert_eq!(
            evaluate_frontier(&[invalid]),
            Err(ParetoError::MissingObjectiveEvidence {
                candidate_id: "untraced_cost".to_owned(),
                key: "cost_per_serving".to_owned(),
            })
        );
    }

    #[test]
    fn rejects_candidates_with_different_gate_schemas() {
        let mut invalid = candidate("other_schema", Some(5.0), Some(8.0), Some(2.0));
        invalid.gates.pop();

        assert_eq!(
            evaluate_frontier(&[
                candidate("baseline", Some(5.0), Some(8.0), Some(2.0)),
                invalid,
            ]),
            Err(ParetoError::GateSchemaMismatch {
                candidate_id: "other_schema".to_owned(),
            })
        );
    }

    #[test]
    fn rejects_candidates_with_different_objective_directions() {
        let mut invalid = candidate("wrong_direction", Some(5.0), Some(8.0), Some(2.0));
        invalid.objectives[0].direction = ObjectiveDirection::Maximize;

        assert_eq!(
            evaluate_frontier(&[
                candidate("baseline", Some(5.0), Some(8.0), Some(2.0)),
                invalid,
            ]),
            Err(ParetoError::ObjectiveSchemaMismatch {
                candidate_id: "wrong_direction".to_owned(),
            })
        );
    }

    #[test]
    fn report_order_is_deterministic_independent_of_input_order() {
        let forward = vec![
            candidate("zeta", Some(9.0), Some(8.0), Some(2.0)),
            candidate("alpha", Some(8.0), Some(8.0), Some(2.0)),
            candidate("mu", Some(10.0), Some(6.0), Some(4.0)),
        ];
        let reverse = forward.iter().cloned().rev().collect::<Vec<_>>();

        assert_eq!(
            evaluate_frontier(&forward).unwrap(),
            evaluate_frontier(&reverse).unwrap()
        );
    }

    #[test]
    fn objective_declaration_order_does_not_change_dominance() {
        let baseline = candidate("baseline", Some(8.0), Some(8.0), Some(2.0));
        let mut reordered = candidate("reordered", Some(9.0), Some(9.0), Some(1.0));
        reordered.objectives.reverse();

        let report = evaluate_frontier(&[baseline, reordered]).unwrap();
        assert_eq!(report.frontier_ids, vec!["baseline", "reordered"]);
        assert!(report.candidates.iter().all(|item| item.dominated_by.is_empty()));
    }

    #[test]
    fn one_objective_must_be_strictly_better_to_dominate() {
        let report = evaluate_frontier(&[
            candidate("a", Some(5.0), Some(8.0), Some(2.0)),
            candidate("b", Some(5.0), Some(8.0), Some(2.0)),
        ])
        .unwrap();
        assert_eq!(report.frontier_ids, vec!["a", "b"]);
    }
}

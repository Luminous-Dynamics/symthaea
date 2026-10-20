//! Multi-objective, evidence-gated comparison of materials-discovery policies.
//!
//! A single aggregate score can hide the trade-off between finding more
//! validated structures, finding a better objective, and spending more money or
//! compute. This module reports a Pareto frontier rather than collapsing those
//! quantities into one weighted score. Hard-constraint failures are ineligible
//! for the frontier, but remain visible in the report.
//!
//! The frontier is computed only within one preregistered, matched campaign
//! group (including seed, evaluator/model/data/target/budget digest, and cost
//! unit). Aggregate claims across seeds require a separate statistical analysis.

use crate::campaign::ObjectiveDirection;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::error::Error;
use std::fmt;

pub const DECISION_FRONTIER_SCHEMA_VERSION: u16 = 1;

/// One policy's observed result for a matched campaign.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrontierCandidate {
    pub policy_id: String,
    pub seed: u64,
    /// Digest of the shared dataset/model/evaluator/environment/target/budget contract.
    /// Policy-specific configuration and observations belong in evidence records.
    pub comparison_contract_digest: String,
    pub objective_name: String,
    pub objective_unit: String,
    pub objective_direction: ObjectiveDirection,
    /// Minimum meaningful improvement; no-worse comparisons remain direction-correct and transitive.
    pub objective_tolerance: f64,
    /// Count of unique structures that passed the declared reference validator.
    pub validated_discovery_count: u64,
    /// Best objective among reference-validated discoveries; absent when none were found.
    pub best_validated_objective: Option<f64>,
    /// Fixed-resolution, non-negative integer cost units; unit and resolution are declared.
    pub total_cost_units: u64,
    pub total_cost_unit: String,
    /// Wall-clock duration is reported but not silently weighted into the frontier.
    pub elapsed_seconds: f64,
    pub reference_calculation_count: u64,
    /// Non-zero means the policy accepted at least one hard-constraint violation.
    pub false_feasible_accepts: u64,
    /// False is an abstention from the safety comparison, not a successful zero count.
    pub hard_constraints_assessed: bool,
    /// SHA-256 digest of the traceable raw result/evidence bundle.
    pub evidence_digest: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FrontierError {
    EmptyCandidateSet,
    EmptyRequiredField(&'static str),
    InvalidDigest(&'static str),
    InvalidTolerance,
    InvalidObjective,
    DiscoveryObjectiveMismatch,
    InvalidCost,
    InvalidElapsedTime,
    MissingEvidence,
    DuplicatePolicyId,
    MismatchedComparisonContract,
    UnsupportedSchemaVersion(u16),
    InvalidReport,
}

impl fmt::Display for FrontierError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyCandidateSet => write!(f, "frontier needs at least one policy candidate"),
            Self::EmptyRequiredField(name) => write!(f, "frontier field must not be empty: {name}"),
            Self::InvalidDigest(name) => write!(
                f,
                "frontier field is not a lowercase SHA-256 digest: {name}"
            ),
            Self::InvalidTolerance => {
                write!(f, "objective tolerance must be finite and non-negative")
            }
            Self::InvalidObjective => write!(f, "validated objective must be finite"),
            Self::DiscoveryObjectiveMismatch => write!(
                f,
                "best validated objective must be present exactly when validated discoveries are non-zero"
            ),
            Self::InvalidCost => write!(
                f,
                "total cost must use a positive, fixed-resolution integer unit"
            ),
            Self::InvalidElapsedTime => {
                write!(f, "elapsed seconds must be finite and non-negative")
            }
            Self::MissingEvidence => write!(
                f,
                "every policy outcome must reference a hashed evidence bundle"
            ),
            Self::DuplicatePolicyId => {
                write!(f, "policy IDs must be unique within a matched frontier")
            }
            Self::MismatchedComparisonContract => write!(
                f,
                "frontier candidates must share seed, comparison digest, objective contract, tolerance, and cost unit"
            ),
            Self::UnsupportedSchemaVersion(version) => {
                write!(f, "unsupported decision frontier schema version: {version}")
            }
            Self::InvalidReport => write!(
                f,
                "frontier report metrics and dispositions do not match the canonical recomputation"
            ),
        }
    }
}

impl Error for FrontierError {}

impl FrontierCandidate {
    fn validate(&self) -> Result<(), FrontierError> {
        for (value, name) in [
            (&self.policy_id, "policy_id"),
            (&self.objective_name, "objective_name"),
            (&self.objective_unit, "objective_unit"),
            (&self.total_cost_unit, "total_cost_unit"),
        ] {
            if value.trim().is_empty() {
                return Err(FrontierError::EmptyRequiredField(name));
            }
        }
        validate_digest(
            &self.comparison_contract_digest,
            "comparison_contract_digest",
        )?;
        if self.evidence_digest.trim().is_empty() {
            return Err(FrontierError::MissingEvidence);
        }
        validate_digest(&self.evidence_digest, "evidence_digest")?;
        if !self.objective_tolerance.is_finite() || self.objective_tolerance < 0.0 {
            return Err(FrontierError::InvalidTolerance);
        }
        if !self.elapsed_seconds.is_finite() || self.elapsed_seconds < 0.0 {
            return Err(FrontierError::InvalidElapsedTime);
        }
        if self.total_cost_units == 0 {
            return Err(FrontierError::InvalidCost);
        }
        match (
            self.validated_discovery_count,
            self.best_validated_objective,
        ) {
            (0, None) => {}
            (0, Some(_)) | (_, None) => {
                return Err(FrontierError::DiscoveryObjectiveMismatch);
            }
            (_, Some(value)) if !value.is_finite() => {
                return Err(FrontierError::InvalidObjective);
            }
            _ => {}
        }
        Ok(())
    }
}

/// Classification for one policy. Unsafe or unassessed outcomes are not hidden
/// inside a frontier score and can never dominate eligible candidates.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case", deny_unknown_fields)]
pub enum FrontierDisposition {
    ParetoFrontier,
    Dominated { dominated_by: Vec<String> },
    Ineligible { reasons: Vec<String> },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrontierEntry {
    pub policy_id: String,
    pub validated_discovery_count: u64,
    pub best_validated_objective: Option<f64>,
    pub total_cost_units: u64,
    pub elapsed_seconds: f64,
    pub reference_calculation_count: u64,
    pub false_feasible_accepts: u64,
    pub hard_constraints_assessed: bool,
    pub evidence_digest: String,
    pub disposition: FrontierDisposition,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecisionFrontier {
    pub schema_version: u16,
    pub comparison_contract_digest: String,
    pub seed: u64,
    pub objective_name: String,
    pub objective_unit: String,
    pub objective_direction: ObjectiveDirection,
    pub objective_tolerance: f64,
    pub total_cost_unit: String,
    pub entries: Vec<FrontierEntry>,
}

impl DecisionFrontier {
    /// Revalidate a serialized frontier by rebuilding its canonical
    /// dispositions from the retained metrics. This prevents a report consumer
    /// from trusting a forged "pareto_frontier" label or altered dominated_by
    /// list merely because the JSON parses.
    pub fn validate(&self) -> Result<(), FrontierError> {
        if self.schema_version != DECISION_FRONTIER_SCHEMA_VERSION {
            return Err(FrontierError::UnsupportedSchemaVersion(self.schema_version));
        }
        if self.entries.is_empty() {
            return Err(FrontierError::EmptyCandidateSet);
        }

        let candidates: Vec<FrontierCandidate> = self
            .entries
            .iter()
            .map(|entry| FrontierCandidate {
                policy_id: entry.policy_id.clone(),
                seed: self.seed,
                comparison_contract_digest: self.comparison_contract_digest.clone(),
                objective_name: self.objective_name.clone(),
                objective_unit: self.objective_unit.clone(),
                objective_direction: self.objective_direction,
                objective_tolerance: self.objective_tolerance,
                validated_discovery_count: entry.validated_discovery_count,
                best_validated_objective: entry.best_validated_objective,
                total_cost_units: entry.total_cost_units,
                total_cost_unit: self.total_cost_unit.clone(),
                elapsed_seconds: entry.elapsed_seconds,
                reference_calculation_count: entry.reference_calculation_count,
                false_feasible_accepts: entry.false_feasible_accepts,
                hard_constraints_assessed: entry.hard_constraints_assessed,
                evidence_digest: entry.evidence_digest.clone(),
            })
            .collect();

        let canonical = build_decision_frontier(&candidates)?;
        if canonical != *self {
            return Err(FrontierError::InvalidReport);
        }
        Ok(())
    }
}

/// Construct a deterministic Pareto report over a single matched seed.
///
/// Dimensions: maximize validated unique discoveries, optimize the best
/// reference-validated objective in its declared direction, and minimize cost.
/// Wall time and reference-calculation count remain visible in input records,
/// but do not silently become extra weighted objectives. Hard-constraint misses
/// are ineligible, never tradeable against discovery quality or cost.
pub fn build_decision_frontier(
    candidates: &[FrontierCandidate],
) -> Result<DecisionFrontier, FrontierError> {
    let first = candidates.first().ok_or(FrontierError::EmptyCandidateSet)?;
    for candidate in candidates {
        candidate.validate()?;
    }

    let mut policy_ids = BTreeSet::new();
    for candidate in candidates {
        if !policy_ids.insert(candidate.policy_id.as_str()) {
            return Err(FrontierError::DuplicatePolicyId);
        }
        if candidate.seed != first.seed
            || candidate.comparison_contract_digest != first.comparison_contract_digest
            || candidate.objective_name != first.objective_name
            || candidate.objective_unit != first.objective_unit
            || candidate.objective_direction != first.objective_direction
            || candidate.objective_tolerance != first.objective_tolerance
            || candidate.total_cost_unit != first.total_cost_unit
        {
            return Err(FrontierError::MismatchedComparisonContract);
        }
    }

    let eligible: Vec<&FrontierCandidate> = candidates
        .iter()
        .filter(|candidate| {
            candidate.false_feasible_accepts == 0 && candidate.hard_constraints_assessed
        })
        .collect();

    let mut entries = Vec::with_capacity(candidates.len());
    for candidate in candidates {
        let reasons = ineligibility_reasons(candidate);
        let disposition = if !reasons.is_empty() {
            FrontierDisposition::Ineligible { reasons }
        } else {
            let dominated_by: Vec<String> = eligible
                .iter()
                .filter(|other| {
                    other.policy_id != candidate.policy_id && dominates(other, candidate)
                })
                .map(|other| other.policy_id.clone())
                .collect();
            if dominated_by.is_empty() {
                FrontierDisposition::ParetoFrontier
            } else {
                FrontierDisposition::Dominated { dominated_by }
            }
        };
        entries.push(FrontierEntry {
            policy_id: candidate.policy_id.clone(),
            validated_discovery_count: candidate.validated_discovery_count,
            best_validated_objective: candidate.best_validated_objective,
            total_cost_units: candidate.total_cost_units,
            elapsed_seconds: candidate.elapsed_seconds,
            reference_calculation_count: candidate.reference_calculation_count,
            false_feasible_accepts: candidate.false_feasible_accepts,
            hard_constraints_assessed: candidate.hard_constraints_assessed,
            evidence_digest: candidate.evidence_digest.clone(),
            disposition,
        });
    }

    entries.sort_by(|left, right| left.policy_id.cmp(&right.policy_id));
    for entry in &mut entries {
        if let FrontierDisposition::Dominated { dominated_by } = &mut entry.disposition {
            dominated_by.sort();
            dominated_by.dedup();
        }
        if let FrontierDisposition::Ineligible { reasons } = &mut entry.disposition {
            reasons.sort();
            reasons.dedup();
        }
    }

    Ok(DecisionFrontier {
        schema_version: DECISION_FRONTIER_SCHEMA_VERSION,
        comparison_contract_digest: first.comparison_contract_digest.clone(),
        seed: first.seed,
        objective_name: first.objective_name.clone(),
        objective_unit: first.objective_unit.clone(),
        objective_direction: first.objective_direction,
        objective_tolerance: first.objective_tolerance,
        total_cost_unit: first.total_cost_unit.clone(),
        entries,
    })
}

fn ineligibility_reasons(candidate: &FrontierCandidate) -> Vec<String> {
    let mut reasons = Vec::new();
    if candidate.false_feasible_accepts > 0 {
        reasons.push("false_feasible_accepts_nonzero".to_owned());
    }
    if !candidate.hard_constraints_assessed {
        reasons.push("hard_constraints_not_assessed".to_owned());
    }
    reasons
}

fn dominates(left: &FrontierCandidate, right: &FrontierCandidate) -> bool {
    // More independently validated discoveries is always preferable.
    let discoveries_no_worse = left.validated_discovery_count >= right.validated_discovery_count;
    let discoveries_better = left.validated_discovery_count > right.validated_discovery_count;

    let objective_no_worse = match (
        left.best_validated_objective,
        right.best_validated_objective,
    ) {
        (None, None) | (Some(_), None) => true,
        (Some(a), Some(b)) => match left.objective_direction {
            ObjectiveDirection::Maximize => a >= b,
            ObjectiveDirection::Minimize => a <= b,
        },
        // A candidate without validated discoveries cannot dominate one that
        // has a validated objective; the inverse is allowed because the
        // discovery-count dimension already favors the candidate with results.
        (None, Some(_)) => false,
    };
    let objective_better = match (
        left.best_validated_objective,
        right.best_validated_objective,
    ) {
        (Some(a), Some(b)) => match left.objective_direction {
            ObjectiveDirection::Maximize => a > b && (a - b) > left.objective_tolerance,
            ObjectiveDirection::Minimize => a < b && (b - a) > left.objective_tolerance,
        },
        _ => false,
    };

    let cost_no_worse = left.total_cost_units <= right.total_cost_units;
    let cost_better = left.total_cost_units < right.total_cost_units;

    discoveries_no_worse
        && objective_no_worse
        && cost_no_worse
        && (discoveries_better || objective_better || cost_better)
}

fn validate_digest(value: &str, field: &'static str) -> Result<(), FrontierError> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        Err(FrontierError::InvalidDigest(field))
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candidate(
        policy: &str,
        discoveries: u64,
        objective: Option<f64>,
        cost: u64,
    ) -> FrontierCandidate {
        FrontierCandidate {
            policy_id: policy.to_owned(),
            seed: 17,
            comparison_contract_digest: "a".repeat(64),
            objective_name: "binding_energy".to_owned(),
            objective_unit: "eV".to_owned(),
            objective_direction: ObjectiveDirection::Minimize,
            objective_tolerance: 0.01,
            validated_discovery_count: discoveries,
            best_validated_objective: objective,
            total_cost_units: cost,
            total_cost_unit: "reference_eval_equivalent_milliunits".to_owned(),
            elapsed_seconds: 12.0,
            reference_calculation_count: 40,
            false_feasible_accepts: 0,
            hard_constraints_assessed: true,
            evidence_digest: "b".repeat(64),
        }
    }

    #[test]
    fn frontier_preserves_tradeoffs_instead_of_hiding_them_in_one_score() {
        let candidates = vec![
            candidate("random", 1, Some(-3.10), 100),
            candidate("diversity", 2, Some(-3.04), 120),
            candidate("symthaea", 2, Some(-3.15), 110),
        ];
        let report = build_decision_frontier(&candidates).expect("matched candidates should pass");
        let lookup = |policy: &str| {
            report
                .entries
                .iter()
                .find(|entry| entry.policy_id == policy)
                .unwrap()
        };
        assert_eq!(
            lookup("random").disposition,
            FrontierDisposition::ParetoFrontier
        );
        assert_eq!(
            lookup("symthaea").disposition,
            FrontierDisposition::ParetoFrontier
        );
        assert_eq!(
            lookup("diversity").disposition,
            FrontierDisposition::Dominated {
                dominated_by: vec!["symthaea".to_owned()]
            }
        );
    }

    #[test]
    fn unsafe_outcome_cannot_buy_its_way_onto_the_frontier() {
        let safe = candidate("safe", 1, Some(-3.10), 100);
        let mut unsafe_candidate = candidate("unsafe", 10, Some(-4.0), 1);
        unsafe_candidate.false_feasible_accepts = 1;
        let report = build_decision_frontier(&[safe, unsafe_candidate])
            .expect("unsafe outcomes remain reportable");
        let entry = report
            .entries
            .iter()
            .find(|entry| entry.policy_id == "unsafe")
            .unwrap();
        assert_eq!(
            entry.disposition,
            FrontierDisposition::Ineligible {
                reasons: vec!["false_feasible_accepts_nonzero".to_owned()]
            }
        );
    }

    #[test]
    fn missing_hard_constraint_assessment_is_not_a_zero_failure_rate() {
        let mut candidate = candidate("unassessed", 1, Some(-3.10), 100);
        candidate.hard_constraints_assessed = false;
        let report =
            build_decision_frontier(&[candidate]).expect("unassessed result remains visible");
        assert_eq!(
            report.entries[0].disposition,
            FrontierDisposition::Ineligible {
                reasons: vec!["hard_constraints_not_assessed".to_owned()]
            }
        );
    }

    #[test]
    fn frontier_rejects_mismatched_seed_or_contract() {
        let first = candidate("random", 1, Some(-3.10), 100);
        let mut second = candidate("symthaea", 2, Some(-3.15), 110);
        second.seed = 18;
        assert_eq!(
            build_decision_frontier(&[first.clone(), second]),
            Err(FrontierError::MismatchedComparisonContract)
        );
        let mut different_data = candidate("symthaea", 2, Some(-3.15), 110);
        different_data.comparison_contract_digest = "c".repeat(64);
        assert_eq!(
            build_decision_frontier(&[first, different_data]),
            Err(FrontierError::MismatchedComparisonContract)
        );
    }

    #[test]
    fn serialized_frontier_revalidates_dispositions_from_retained_metrics() {
        let candidates = vec![
            candidate("random", 1, Some(-3.10), 100),
            candidate("diversity", 2, Some(-3.04), 120),
            candidate("symthaea", 2, Some(-3.15), 110),
        ];
        let mut report = build_decision_frontier(&candidates).unwrap();
        report
            .validate()
            .expect("canonical frontier should validate");

        let unsafe_entry = report
            .entries
            .iter_mut()
            .find(|entry| entry.policy_id == "random")
            .unwrap();
        unsafe_entry.false_feasible_accepts = 1;
        assert_eq!(report.validate(), Err(FrontierError::InvalidReport));

        let mut forged_dominance = build_decision_frontier(&candidates).unwrap();
        let dominated = forged_dominance
            .entries
            .iter_mut()
            .find(|entry| entry.policy_id == "diversity")
            .unwrap();
        dominated.disposition = FrontierDisposition::ParetoFrontier;
        assert_eq!(
            forged_dominance.validate(),
            Err(FrontierError::InvalidReport)
        );
    }

    #[test]
    fn serialized_frontier_rejects_unsupported_versions() {
        let mut report = build_decision_frontier(&[candidate("random", 1, Some(-3.10), 100)])
            .expect("candidate should produce a frontier");
        report.schema_version = 99;
        assert_eq!(
            report.validate(),
            Err(FrontierError::UnsupportedSchemaVersion(99))
        );
    }

    #[test]
    fn output_is_permutation_deterministic_and_serializable() {
        let a = candidate("random", 1, Some(-3.10), 100);
        let b = candidate("diversity", 2, Some(-3.04), 120);
        let c = candidate("symthaea", 2, Some(-3.15), 110);
        let first = build_decision_frontier(&[a.clone(), b.clone(), c.clone()]).unwrap();
        let second = build_decision_frontier(&[c, a, b]).unwrap();
        assert_eq!(first, second);
        let encoded = serde_json::to_string(&first).unwrap();
        let decoded: DecisionFrontier = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, first);
    }

    #[test]
    fn objective_is_required_exactly_when_a_validated_discovery_exists() {
        let without_discovery = candidate("empty", 0, None, 1);
        assert!(build_decision_frontier(&[without_discovery]).is_ok());

        let inconsistent = candidate("mismatch", 0, Some(-3.0), 1);
        assert_eq!(
            build_decision_frontier(&[inconsistent]),
            Err(FrontierError::DiscoveryObjectiveMismatch)
        );
    }

    #[test]
    fn duplicate_policy_ids_and_zero_cost_are_rejected() {
        let a = candidate("same", 1, Some(-3.1), 100);
        assert_eq!(
            build_decision_frontier(&[a.clone(), a.clone()]),
            Err(FrontierError::DuplicatePolicyId)
        );
        let free = candidate("free", 1, Some(-3.1), 0);
        assert_eq!(
            build_decision_frontier(&[free]),
            Err(FrontierError::InvalidCost)
        );
    }

    #[test]
    fn a_validated_discovery_can_dominate_a_zero_discovery_policy() {
        let empty = candidate("empty", 0, None, 100);
        let discovered = candidate("discovered", 1, Some(-3.10), 80);
        let report = build_decision_frontier(&[empty, discovered])
            .expect("zero-discovery policies remain comparable on defined dimensions");
        assert_eq!(
            report
                .entries
                .iter()
                .find(|entry| entry.policy_id == "empty")
                .unwrap()
                .disposition,
            FrontierDisposition::Dominated {
                dominated_by: vec!["discovered".to_owned()]
            }
        );
        assert_eq!(
            report
                .entries
                .iter()
                .find(|entry| entry.policy_id == "discovered")
                .unwrap()
                .disposition,
            FrontierDisposition::ParetoFrontier
        );
    }

    #[test]
    fn illustrative_fixture_builds_safety_gated_pareto_frontier() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../fixtures/decision-frontier-candidates-v1.json"
        ))
        .expect("illustrative fixture must be valid JSON");
        assert_eq!(fixture["schema_version"], 1);
        assert_eq!(fixture["fixture_kind"], "illustrative_not_result");

        let _candidate_schema: serde_json::Value = serde_json::from_str(include_str!(
            "../schemas/decision-frontier-candidates-v1.schema.json"
        ))
        .expect("candidate schema must be valid JSON");
        let _report_schema: serde_json::Value = serde_json::from_str(include_str!(
            "../schemas/decision-frontier-report-v1.schema.json"
        ))
        .expect("report schema must be valid JSON");

        let candidates: Vec<FrontierCandidate> =
            serde_json::from_value(fixture["candidates"].clone())
                .expect("fixture candidates must match Rust records");
        let report = build_decision_frontier(&candidates)
            .expect("illustrative candidates share a comparison contract");
        report
            .validate()
            .expect("generated frontier should revalidate");
        let lookup = |policy: &str| {
            report
                .entries
                .iter()
                .find(|entry| entry.policy_id == policy)
                .unwrap()
        };

        assert_eq!(
            lookup("random").disposition,
            FrontierDisposition::ParetoFrontier
        );
        assert_eq!(
            lookup("symthaea").disposition,
            FrontierDisposition::ParetoFrontier
        );
        assert_eq!(
            lookup("diversity").disposition,
            FrontierDisposition::Dominated {
                dominated_by: vec!["symthaea".to_owned()]
            }
        );
        assert!(matches!(
            &lookup("unsafe").disposition,
            &FrontierDisposition::Ineligible { .. }
        ));
        assert!(matches!(
            &lookup("unassessed").disposition,
            &FrontierDisposition::Ineligible { .. }
        ));

        let result_bytes = serde_json::to_vec(&report).expect("report should serialize");
        let round_trip: DecisionFrontier =
            serde_json::from_slice(&result_bytes).expect("report should round-trip");
        assert_eq!(round_trip, report);
        round_trip
            .validate()
            .expect("round-tripped frontier should revalidate");
    }
}

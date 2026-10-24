// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Paired, fail-closed comparison of baseline and candidate production outcomes.
//!
//! This module verifies record structure and accounting only. It is not a
//! sensor adapter, metrology authenticator, causal estimator, controller, or
//! proof that supplied outcomes were measured in the physical world.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;

/// Version for the machine-readable production-outcome report.
pub const PRODUCTION_VALUE_REPORT_SCHEMA_VERSION: u16 = 1;

/// Claimed evidence class for this outcome record. This is a declaration, not
/// an authenticated attestation; the evaluator prevents cross-class comparisons
/// but cannot prove that a record really belongs to the declared class.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceClass {
    /// Known-answer fixtures or generated synthetic outcomes.
    SyntheticFixture,
    /// Output from a simulation/digital twin, not direct physical measurement.
    SimulationOutput,
    /// Outcomes recorded by an instrumented physical process.
    InstrumentObserved,
}

/// One completed-workload outcome at a declared measurement cutoff.
///
/// Unit dispositions are mutually exclusive and must sum to `input_units`.
/// Material mass is measured in integer milligrams, and all destinations,
/// including held material and residual, must sum to `material_input_mg`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProductionRun {
    pub scenario_id: String,
    pub policy_id: String,
    /// SHA-256 identity of the frozen workload/configuration shared by a pair.
    /// The evaluator checks syntax only and does not recompute this digest.
    pub workload_digest: String,
    /// Declared origin class; baseline/candidate classes must match.
    pub evidence_class: EvidenceClass,
    /// SHA-256 of the retained raw source record or immutable record envelope.
    /// Syntax is checked; hashing/authentication and source retrieval are external.
    pub source_record_sha256: String,
    pub input_units: u64,
    pub accepted_units: u64,
    pub scrapped_units: u64,
    pub rework_units: u64,
    pub held_units: u64,
    pub material_input_mg: u64,
    pub material_accepted_mg: u64,
    pub material_scrap_mg: u64,
    pub material_rework_mg: u64,
    pub material_held_mg: u64,
    pub material_residual_mg: u64,
    pub energy_wh: f64,
    pub elapsed_seconds: f64,
    pub critical_safety_violations: u64,
    /// Subset count of accepted pieces later found to have escaped nonconforming.
    pub escaped_nonconforming_units: u64,
}

impl ProductionRun {
    fn validate(&self) -> Result<(), EvaluationError> {
        let id = self.scenario_id.as_str();
        if id.trim().is_empty()
            || self.policy_id.trim().is_empty()
            || self.workload_digest.trim().is_empty()
        {
            return Err(EvaluationError::InvalidRun {
                scenario_id: id.to_owned(),
                reason: "scenario, policy, and workload identities must be non-empty",
            });
        }
        if !is_lowercase_sha256(&self.workload_digest) {
            return Err(EvaluationError::InvalidDigest {
                scenario_id: id.to_owned(),
                field: "workload_digest",
            });
        }
        if !is_lowercase_sha256(&self.source_record_sha256) {
            return Err(EvaluationError::InvalidDigest {
                scenario_id: id.to_owned(),
                field: "source_record_sha256",
            });
        }
        if self.input_units == 0 {
            return Err(EvaluationError::InvalidRun {
                scenario_id: id.to_owned(),
                reason: "input_units must be greater than zero",
            });
        }
        if self.escaped_nonconforming_units > self.accepted_units {
            return Err(EvaluationError::InvalidRun {
                scenario_id: id.to_owned(),
                reason: "escaped nonconforming units cannot exceed accepted units",
            });
        }

        let units = checked_sum(&[
            self.accepted_units,
            self.scrapped_units,
            self.rework_units,
            self.held_units,
        ])
        .ok_or(EvaluationError::ArithmeticOverflow)?;
        if units != self.input_units {
            return Err(EvaluationError::InvalidRun {
                scenario_id: id.to_owned(),
                reason: "accepted + scrap + rework + held units must equal input units",
            });
        }

        if self.material_input_mg == 0 {
            return Err(EvaluationError::InvalidRun {
                scenario_id: id.to_owned(),
                reason: "material_input_mg must be greater than zero",
            });
        }
        let mass = checked_sum(&[
            self.material_accepted_mg,
            self.material_scrap_mg,
            self.material_rework_mg,
            self.material_held_mg,
            self.material_residual_mg,
        ])
        .ok_or(EvaluationError::ArithmeticOverflow)?;
        if mass != self.material_input_mg {
            return Err(EvaluationError::InvalidRun {
                scenario_id: id.to_owned(),
                reason: "accepted + scrap + rework + held + residual mass must equal input mass",
            });
        }
        if self.accepted_units == 0 && self.material_accepted_mg != 0 {
            return Err(EvaluationError::InvalidRun {
                scenario_id: id.to_owned(),
                reason: "accepted mass must be zero when accepted unit count is zero",
            });
        }
        if !self.energy_wh.is_finite() || self.energy_wh < 0.0 {
            return Err(EvaluationError::InvalidRun {
                scenario_id: id.to_owned(),
                reason: "energy_wh must be finite and non-negative",
            });
        }
        if !self.elapsed_seconds.is_finite() || self.elapsed_seconds <= 0.0 {
            return Err(EvaluationError::InvalidRun {
                scenario_id: id.to_owned(),
                reason: "elapsed_seconds must be finite and greater than zero",
            });
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvaluationError {
    EmptyPolicyId,
    SamePolicy,
    EmptyRuns {
        policy_id: String,
    },
    MixedPolicy {
        expected: String,
        found: String,
    },
    DuplicateScenario {
        scenario_id: String,
    },
    InvalidRun {
        scenario_id: String,
        reason: &'static str,
    },
    InvalidDigest {
        scenario_id: String,
        field: &'static str,
    },
    MixedEvidenceClass {
        policy_id: String,
        expected: EvidenceClass,
        found: EvidenceClass,
    },
    EvidenceClassMismatch {
        baseline: EvidenceClass,
        candidate: EvidenceClass,
    },
    PairSetMismatch {
        missing_in_candidate: Vec<String>,
        extra_in_candidate: Vec<String>,
    },
    WorkloadMismatch {
        scenario_id: String,
    },
    ArithmeticOverflow,
    NonFiniteAggregate,
}

impl fmt::Display for EvaluationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyPolicyId => write!(f, "policy identifiers must be non-empty"),
            Self::SamePolicy => write!(f, "baseline and candidate policy identifiers must differ"),
            Self::EmptyRuns { policy_id } => write!(f, "policy {policy_id} has no runs"),
            Self::MixedPolicy { expected, found } => {
                write!(f, "expected policy {expected}, found policy {found}")
            }
            Self::DuplicateScenario { scenario_id } => {
                write!(f, "duplicate scenario ID: {scenario_id}")
            }
            Self::InvalidRun {
                scenario_id,
                reason,
            } => {
                write!(f, "invalid run {scenario_id}: {reason}")
            }
            Self::InvalidDigest { scenario_id, field } => write!(
                f,
                "invalid lowercase SHA-256 syntax for {field} in run {scenario_id}"
            ),
            Self::MixedEvidenceClass {
                policy_id,
                expected,
                found,
            } => write!(
                f,
                "policy {policy_id} mixes evidence classes ({expected:?} and {found:?})"
            ),
            Self::EvidenceClassMismatch {
                baseline,
                candidate,
            } => write!(
                f,
                "baseline evidence class {baseline:?} differs from candidate class {candidate:?}"
            ),
            Self::PairSetMismatch {
                missing_in_candidate,
                extra_in_candidate,
            } => write!(
                f,
                "paired scenario sets differ (missing in candidate: {missing_in_candidate:?}; extra in candidate: {extra_in_candidate:?})"
            ),
            Self::WorkloadMismatch { scenario_id } => write!(
                f,
                "workload identity or fixed input quantities differ for {scenario_id}"
            ),
            Self::ArithmeticOverflow => write!(f, "integer accounting overflow"),
            Self::NonFiniteAggregate => write!(f, "aggregate energy/time is non-finite"),
        }
    }
}
impl Error for EvaluationError {}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OutcomeSummary {
    pub scenario_count: usize,
    pub input_units: u64,
    pub accepted_units: u64,
    pub scrapped_units: u64,
    pub rework_units: u64,
    pub held_units: u64,
    pub material_input_mg: u64,
    pub material_accepted_mg: u64,
    pub material_scrap_mg: u64,
    pub material_rework_mg: u64,
    pub material_held_mg: u64,
    pub material_residual_mg: u64,
    pub energy_wh: f64,
    pub elapsed_seconds: f64,
    pub critical_safety_violations: u64,
    pub escaped_nonconforming_units: u64,
}
impl OutcomeSummary {
    /// Accepted-unit fraction at the fixed evaluation cutoff.
    pub fn accepted_yield(&self) -> f64 {
        self.accepted_units as f64 / self.input_units as f64
    }
    /// Accepted units per accumulated machine-hour.
    pub fn accepted_units_per_hour(&self) -> f64 {
        self.accepted_units as f64 * 3600.0 / self.elapsed_seconds
    }
    /// None rather than division by zero when no unit was accepted.
    pub fn energy_wh_per_accepted_unit(&self) -> Option<f64> {
        (self.accepted_units > 0).then(|| self.energy_wh / self.accepted_units as f64)
    }
    pub fn scrap_mass_fraction(&self) -> f64 {
        self.material_scrap_mg as f64 / self.material_input_mg as f64
    }
    /// These gates are non-compensatory; productivity cannot cancel a violation.
    pub fn hard_gates_pass(&self) -> bool {
        self.critical_safety_violations == 0
            && self.escaped_nonconforming_units == 0
            && self.held_units == 0
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProductionValueReport {
    /// Version of the serialized report contract.
    pub schema_version: u16,
    pub baseline_policy_id: String,
    pub candidate_policy_id: String,
    /// Declared evidence class shared by every compared run.
    pub evidence_class: EvidenceClass,
    /// Sorted by scenario ID, independent of input iteration order.
    pub paired_scenario_ids: Vec<String>,
    pub baseline: OutcomeSummary,
    pub candidate: OutcomeSummary,
}
impl ProductionValueReport {
    pub fn candidate_eligible(&self) -> bool {
        self.candidate.hard_gates_pass()
    }
    /// Raw differences only; no weighted aggregate can mask regressions.
    pub fn accepted_yield_delta(&self) -> f64 {
        self.candidate.accepted_yield() - self.baseline.accepted_yield()
    }
    pub fn accepted_units_per_hour_delta(&self) -> f64 {
        self.candidate.accepted_units_per_hour() - self.baseline.accepted_units_per_hour()
    }
    /// None if either policy accepted zero units.
    pub fn energy_per_accepted_unit_delta(&self) -> Option<f64> {
        Some(
            self.candidate.energy_wh_per_accepted_unit()?
                - self.baseline.energy_wh_per_accepted_unit()?,
        )
    }
}

/// Compare outcome records from two named policies on exactly the same frozen
/// scenarios. Input authenticity and physical measurement provenance remain the
/// caller's responsibility.
pub fn compare_production_outcomes(
    baseline_policy_id: &str,
    baseline_runs: &[ProductionRun],
    candidate_policy_id: &str,
    candidate_runs: &[ProductionRun],
) -> Result<ProductionValueReport, EvaluationError> {
    if baseline_policy_id.trim().is_empty() || candidate_policy_id.trim().is_empty() {
        return Err(EvaluationError::EmptyPolicyId);
    }
    if baseline_policy_id == candidate_policy_id {
        return Err(EvaluationError::SamePolicy);
    }

    let baseline = index_runs(baseline_policy_id, baseline_runs)?;
    let candidate = index_runs(candidate_policy_id, candidate_runs)?;
    let baseline_evidence = common_evidence_class(baseline_policy_id, &baseline)?;
    let candidate_evidence = common_evidence_class(candidate_policy_id, &candidate)?;
    if baseline_evidence != candidate_evidence {
        return Err(EvaluationError::EvidenceClassMismatch {
            baseline: baseline_evidence,
            candidate: candidate_evidence,
        });
    }

    let missing: Vec<String> = baseline
        .keys()
        .filter(|id| !candidate.contains_key(**id))
        .map(|id| (*id).to_owned())
        .collect();
    let extra: Vec<String> = candidate
        .keys()
        .filter(|id| !baseline.contains_key(**id))
        .map(|id| (*id).to_owned())
        .collect();
    if !missing.is_empty() || !extra.is_empty() {
        return Err(EvaluationError::PairSetMismatch {
            missing_in_candidate: missing,
            extra_in_candidate: extra,
        });
    }

    let mut paired_ids = Vec::with_capacity(baseline.len());
    for (scenario_id, baseline_run) in &baseline {
        let candidate_run = candidate[scenario_id];
        if baseline_run.workload_digest != candidate_run.workload_digest
            || baseline_run.input_units != candidate_run.input_units
            || baseline_run.material_input_mg != candidate_run.material_input_mg
        {
            return Err(EvaluationError::WorkloadMismatch {
                scenario_id: (*scenario_id).to_owned(),
            });
        }
        paired_ids.push((*scenario_id).to_owned());
    }

    Ok(ProductionValueReport {
        schema_version: PRODUCTION_VALUE_REPORT_SCHEMA_VERSION,
        baseline_policy_id: baseline_policy_id.to_owned(),
        candidate_policy_id: candidate_policy_id.to_owned(),
        evidence_class: baseline_evidence,
        paired_scenario_ids: paired_ids,
        baseline: summarize(&baseline)?,
        candidate: summarize(&candidate)?,
    })
}

fn index_runs<'a>(
    policy_id: &str,
    runs: &'a [ProductionRun],
) -> Result<BTreeMap<&'a str, &'a ProductionRun>, EvaluationError> {
    if runs.is_empty() {
        return Err(EvaluationError::EmptyRuns {
            policy_id: policy_id.to_owned(),
        });
    }
    let mut indexed = BTreeMap::new();
    for run in runs {
        if run.policy_id != policy_id {
            return Err(EvaluationError::MixedPolicy {
                expected: policy_id.to_owned(),
                found: run.policy_id.clone(),
            });
        }
        run.validate()?;
        if indexed.insert(run.scenario_id.as_str(), run).is_some() {
            return Err(EvaluationError::DuplicateScenario {
                scenario_id: run.scenario_id.clone(),
            });
        }
    }
    Ok(indexed)
}

fn common_evidence_class(
    policy_id: &str,
    runs: &BTreeMap<&str, &ProductionRun>,
) -> Result<EvidenceClass, EvaluationError> {
    let mut iter = runs.values();
    let first = iter
        .next()
        .ok_or_else(|| EvaluationError::EmptyRuns {
            policy_id: policy_id.to_owned(),
        })?
        .evidence_class;
    for run in iter {
        if run.evidence_class != first {
            return Err(EvaluationError::MixedEvidenceClass {
                policy_id: policy_id.to_owned(),
                expected: first,
                found: run.evidence_class,
            });
        }
    }
    Ok(first)
}

fn is_lowercase_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn summarize(runs: &BTreeMap<&str, &ProductionRun>) -> Result<OutcomeSummary, EvaluationError> {
    let energy_wh = sum_f64(runs, |r| r.energy_wh)?;
    let elapsed_seconds = sum_f64(runs, |r| r.elapsed_seconds)?;
    Ok(OutcomeSummary {
        scenario_count: runs.len(),
        input_units: sum_field(runs, |r| r.input_units)?,
        accepted_units: sum_field(runs, |r| r.accepted_units)?,
        scrapped_units: sum_field(runs, |r| r.scrapped_units)?,
        rework_units: sum_field(runs, |r| r.rework_units)?,
        held_units: sum_field(runs, |r| r.held_units)?,
        material_input_mg: sum_field(runs, |r| r.material_input_mg)?,
        material_accepted_mg: sum_field(runs, |r| r.material_accepted_mg)?,
        material_scrap_mg: sum_field(runs, |r| r.material_scrap_mg)?,
        material_rework_mg: sum_field(runs, |r| r.material_rework_mg)?,
        material_held_mg: sum_field(runs, |r| r.material_held_mg)?,
        material_residual_mg: sum_field(runs, |r| r.material_residual_mg)?,
        energy_wh,
        elapsed_seconds,
        critical_safety_violations: sum_field(runs, |r| r.critical_safety_violations)?,
        escaped_nonconforming_units: sum_field(runs, |r| r.escaped_nonconforming_units)?,
    })
}
fn sum_field<F>(runs: &BTreeMap<&str, &ProductionRun>, field: F) -> Result<u64, EvaluationError>
where
    F: Fn(&ProductionRun) -> u64,
{
    runs.values().try_fold(0_u64, |sum, run| {
        sum.checked_add(field(run))
            .ok_or(EvaluationError::ArithmeticOverflow)
    })
}
fn sum_f64<F>(runs: &BTreeMap<&str, &ProductionRun>, field: F) -> Result<f64, EvaluationError>
where
    F: Fn(&ProductionRun) -> f64,
{
    let value = runs.values().map(|run| field(run)).sum::<f64>();
    if value.is_finite() {
        Ok(value)
    } else {
        Err(EvaluationError::NonFiniteAggregate)
    }
}
fn checked_sum(values: &[u64]) -> Option<u64> {
    values
        .iter()
        .try_fold(0_u64, |sum, value| sum.checked_add(*value))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_digest(value: &str) -> String {
        // Test-only digest-shaped identifier; not a hash of the supplied value.
        let reduced = value.bytes().fold(0_u64, |acc, byte| {
            acc.wrapping_mul(257).wrapping_add(byte as u64)
        });
        format!("{reduced:064x}")
    }

    fn run(policy: &str, scenario: &str, accepted: u64) -> ProductionRun {
        let scrapped = 10 - accepted;
        ProductionRun {
            scenario_id: scenario.to_owned(),
            policy_id: policy.to_owned(),
            workload_digest: test_digest(&format!("workload-{scenario}")),
            evidence_class: EvidenceClass::InstrumentObserved,
            source_record_sha256: test_digest(&format!("source-{policy}-{scenario}")),
            input_units: 10,
            accepted_units: accepted,
            scrapped_units: scrapped,
            rework_units: 0,
            held_units: 0,
            material_input_mg: 1000,
            material_accepted_mg: accepted * 100,
            material_scrap_mg: scrapped * 100,
            material_rework_mg: 0,
            material_held_mg: 0,
            material_residual_mg: 0,
            energy_wh: if policy == "baseline" { 100.0 } else { 90.0 },
            elapsed_seconds: 60.0,
            critical_safety_violations: 0,
            escaped_nonconforming_units: 0,
        }
    }
    fn pair() -> (Vec<ProductionRun>, Vec<ProductionRun>) {
        (
            vec![run("baseline", "scenario-001", 8)],
            vec![run("candidate", "scenario-001", 9)],
        )
    }

    #[test]
    fn rejects_non_sha256_digest_syntax() {
        let (b, mut c) = pair();
        c[0].source_record_sha256 = "not-a-digest".to_owned();
        assert!(matches!(
            compare_production_outcomes("baseline", &b, "candidate", &c),
            Err(EvaluationError::InvalidDigest {
                field: "source_record_sha256",
                ..
            })
        ));
    }

    #[test]
    fn rejects_mixed_evidence_classes_within_a_policy_or_between_policies() {
        let mut b = vec![run("baseline", "a", 8), run("baseline", "b", 8)];
        let c = vec![run("candidate", "a", 9), run("candidate", "b", 9)];
        b[1].evidence_class = EvidenceClass::SimulationOutput;
        assert!(matches!(
            compare_production_outcomes("baseline", &b, "candidate", &c),
            Err(EvaluationError::MixedEvidenceClass { .. })
        ));

        let (b, mut c) = pair();
        c[0].evidence_class = EvidenceClass::SimulationOutput;
        assert!(matches!(
            compare_production_outcomes("baseline", &b, "candidate", &c),
            Err(EvaluationError::EvidenceClassMismatch { .. })
        ));
    }

    #[test]
    fn report_json_round_trip_preserves_schema_and_evidence_class() {
        let (baseline, candidate) = pair();
        let report =
            compare_production_outcomes("baseline", &baseline, "candidate", &candidate).unwrap();
        let json = serde_json::to_string(&report).unwrap();
        let decoded: ProductionValueReport = serde_json::from_str(&json).unwrap();
        assert_eq!(report, decoded);
        assert_eq!(
            decoded.schema_version,
            PRODUCTION_VALUE_REPORT_SCHEMA_VERSION
        );
        assert_eq!(decoded.evidence_class, EvidenceClass::InstrumentObserved);
        let mut run_json = serde_json::to_value(&baseline[0]).unwrap();
        run_json
            .as_object_mut()
            .unwrap()
            .insert("unexpected".to_owned(), serde_json::Value::Bool(true));
        assert!(serde_json::from_value::<ProductionRun>(run_json).is_err());
    }

    #[test]
    fn known_answer_reports_raw_deltas_without_scalar_score() {
        let (b, c) = pair();
        let r = compare_production_outcomes("baseline", &b, "candidate", &c).unwrap();
        assert_eq!(r.paired_scenario_ids, vec!["scenario-001"]);
        assert!((r.baseline.accepted_yield() - 0.8).abs() < 1e-12);
        assert!((r.candidate.accepted_yield() - 0.9).abs() < 1e-12);
        assert!((r.accepted_yield_delta() - 0.1).abs() < 1e-12);
        assert!((r.baseline.energy_wh_per_accepted_unit().unwrap() - 12.5).abs() < 1e-12);
        assert!((r.candidate.energy_wh_per_accepted_unit().unwrap() - 10.0).abs() < 1e-12);
        assert!((r.energy_per_accepted_unit_delta().unwrap() + 2.5).abs() < 1e-12);
        assert!(r.candidate_eligible());
    }
    #[test]
    fn scenario_order_is_deterministic() {
        let mut b = vec![run("baseline", "b", 8), run("baseline", "a", 7)];
        let mut c = vec![run("candidate", "b", 9), run("candidate", "a", 8)];
        let first = compare_production_outcomes("baseline", &b, "candidate", &c).unwrap();
        b.reverse();
        c.reverse();
        let second = compare_production_outcomes("baseline", &b, "candidate", &c).unwrap();
        assert_eq!(first, second);
    }
    #[test]
    fn rejects_empty_sides_and_policy_ids() {
        let (_, c) = pair();
        assert!(matches!(
            compare_production_outcomes("baseline", &[], "candidate", &c),
            Err(EvaluationError::EmptyRuns { .. })
        ));
        let (b, c) = pair();
        assert!(matches!(
            compare_production_outcomes("", &b, "candidate", &c),
            Err(EvaluationError::EmptyPolicyId)
        ));
        assert!(matches!(
            compare_production_outcomes("baseline", &b, "baseline", &c),
            Err(EvaluationError::SamePolicy)
        ));
    }
    #[test]
    fn rejects_duplicate_scenarios_and_mixed_policy_labels() {
        let (mut b, c) = pair();
        b.push(b[0].clone());
        assert!(matches!(
            compare_production_outcomes("baseline", &b, "candidate", &c),
            Err(EvaluationError::DuplicateScenario { .. })
        ));
        let (b, mut c) = pair();
        c[0].policy_id = "other".to_owned();
        assert!(matches!(
            compare_production_outcomes("baseline", &b, "candidate", &c),
            Err(EvaluationError::MixedPolicy { .. })
        ));
    }
    #[test]
    fn rejects_missing_extra_and_mismatched_workloads() {
        let (b, mut c) = pair();
        c[0].scenario_id = "other".to_owned();
        c[0].workload_digest = test_digest("workload-other");
        assert!(matches!(
            compare_production_outcomes("baseline", &b, "candidate", &c),
            Err(EvaluationError::PairSetMismatch { .. })
        ));

        let (b, mut c) = pair();
        c[0].workload_digest = test_digest("different");
        assert!(matches!(
            compare_production_outcomes("baseline", &b, "candidate", &c),
            Err(EvaluationError::WorkloadMismatch { .. })
        ));

        let (b, mut c) = pair();
        c[0].material_input_mg = 1100;
        c[0].material_scrap_mg = 200;
        assert!(matches!(
            compare_production_outcomes("baseline", &b, "candidate", &c),
            Err(EvaluationError::WorkloadMismatch { .. })
        ));
    }
    #[test]
    fn rejects_unit_and_material_imbalances() {
        let (b, mut c) = pair();
        c[0].scrapped_units -= 1;
        assert!(matches!(
            compare_production_outcomes("baseline", &b, "candidate", &c),
            Err(EvaluationError::InvalidRun { .. })
        ));

        let (b, mut c) = pair();
        c[0].material_residual_mg = 1;
        assert!(matches!(
            compare_production_outcomes("baseline", &b, "candidate", &c),
            Err(EvaluationError::InvalidRun { .. })
        ));
    }
    #[test]
    fn rejects_bad_energy_or_duration() {
        let (b, mut c) = pair();
        c[0].energy_wh = f64::NAN;
        assert!(matches!(
            compare_production_outcomes("baseline", &b, "candidate", &c),
            Err(EvaluationError::InvalidRun { .. })
        ));
        let (b, mut c) = pair();
        c[0].energy_wh = -0.1;
        assert!(matches!(
            compare_production_outcomes("baseline", &b, "candidate", &c),
            Err(EvaluationError::InvalidRun { .. })
        ));
        let (b, mut c) = pair();
        c[0].elapsed_seconds = 0.0;
        assert!(matches!(
            compare_production_outcomes("baseline", &b, "candidate", &c),
            Err(EvaluationError::InvalidRun { .. })
        ));
    }
    #[test]
    fn zero_accepted_units_never_divides_by_zero() {
        let (b, c) = (
            vec![run("baseline", "scenario-001", 0)],
            vec![run("candidate", "scenario-001", 0)],
        );
        let r = compare_production_outcomes("baseline", &b, "candidate", &c).unwrap();
        assert_eq!(r.baseline.energy_wh_per_accepted_unit(), None);
        assert_eq!(r.candidate.energy_wh_per_accepted_unit(), None);
        assert_eq!(r.energy_per_accepted_unit_delta(), None);
    }
    #[test]
    fn safety_defect_and_hold_gates_are_non_compensatory() {
        let (b, mut c) = pair();
        c[0].critical_safety_violations = 1;
        let r = compare_production_outcomes("baseline", &b, "candidate", &c).unwrap();
        assert!(!r.candidate_eligible());

        let (b, mut c) = pair();
        c[0].escaped_nonconforming_units = 1;
        let r = compare_production_outcomes("baseline", &b, "candidate", &c).unwrap();
        assert!(!r.candidate_eligible());

        let (b, mut c) = pair();
        c[0].accepted_units = 8;
        c[0].scrapped_units = 1;
        c[0].held_units = 1;
        c[0].material_accepted_mg = 800;
        c[0].material_scrap_mg = 100;
        c[0].material_held_mg = 100;
        let r = compare_production_outcomes("baseline", &b, "candidate", &c).unwrap();
        assert!(!r.candidate_eligible());
    }
}

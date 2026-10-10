//! Evidence-gap-aware information-gain planning.
//!
//! A high-information test is not automatically useful for a scientific
//! criterion. This module requires an explicit mapping from a test to the
//! evidence predicates it can advance, while retaining the EIG calculation.

use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, PartialEq)]
pub struct GapAwareTest {
    pub id: String,
    pub cost: f64,
    pub eig_bits: f64,
    pub advances_predicates: BTreeSet<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GapAwareAssessment {
    pub test_id: String,
    pub eig_bits: f64,
    pub cost: f64,
    pub advances_unmet_predicates: BTreeSet<String>,
    pub gap_coverage: f64,
    pub eligible: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlannerError {
    NegativeCost(String),
    NonFiniteInformationGain(String),
    EmptyTestId(String),
    UnknownPredicate(String),
    EmptyUnmetPredicates,
    DuplicateTest(String),
}

pub fn plan(
    tests: &[GapAwareTest],
    unmet_predicates: &BTreeSet<String>,
) -> Result<Vec<GapAwareAssessment>, PlannerError> {
    if unmet_predicates.is_empty() {
        return Err(PlannerError::EmptyUnmetPredicates);
    }

    let mut seen = BTreeSet::new();
    let mut assessments = Vec::with_capacity(tests.len());

    for test in tests {
        if test.id.trim().is_empty() {
            return Err(PlannerError::EmptyTestId(test.id.clone()));
        }
        if !seen.insert(test.id.as_str()) {
            return Err(PlannerError::DuplicateTest(test.id.clone()));
        }
        if test.cost < 0.0 || !test.cost.is_finite() {
            return Err(PlannerError::NegativeCost(test.id.clone()));
        }
        if !test.eig_bits.is_finite() || test.eig_bits < 0.0 {
            return Err(PlannerError::NonFiniteInformationGain(test.id.clone()));
        }
        if let Some(unknown) = test.advances_predicates.iter().find(|p| !unmet_predicates.contains(*p)) {
            return Err(PlannerError::UnknownPredicate(unknown.clone()));
        }

        let covered: BTreeSet<_> = test.advances_predicates.intersection(unmet_predicates).cloned().collect();
        let coverage = covered.len() as f64 / unmet_predicates.len() as f64;
        assessments.push(GapAwareAssessment {
            test_id: test.id.clone(),
            eig_bits: test.eig_bits,
            cost: test.cost,
            advances_unmet_predicates: covered,
            gap_coverage: coverage,
            eligible: true,
        });
    }

    assessments.sort_by(|a, b| {
        b.gap_coverage.partial_cmp(&a.gap_coverage).unwrap()
            .then_with(|| {
                let ar = if a.cost == 0.0 { f64::INFINITY } else { a.eig_bits / a.cost };
                let br = if b.cost == 0.0 { f64::INFINITY } else { b.eig_bits / b.cost };
                br.partial_cmp(&ar).unwrap()
            })
            .then_with(|| b.eig_bits.partial_cmp(&a.eig_bits).unwrap())
            .then_with(|| a.test_id.cmp(&b.test_id))
    });
    Ok(assessments)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn set(xs: &[&str]) -> BTreeSet<String> {
        xs.iter().map(|x| (*x).into()).collect()
    }

    #[test]
    fn prioritizes_gap_coverage_before_raw_eig() {
        let gaps = set(&["replication", "provenance"]);
        let tests = vec![
            GapAwareTest { id: "high-eig".into(), cost: 1.0, eig_bits: 10.0, advances_predicates: set(&["replication"]) },
            GapAwareTest { id: "covers-both".into(), cost: 10.0, eig_bits: 1.0, advances_predicates: set(&["replication", "provenance"]) },
        ];
        let plan = plan(&tests, &gaps).unwrap();
        assert_eq!(plan[0].test_id, "covers-both");
        assert_eq!(plan[0].gap_coverage, 1.0);
    }

    #[test]
    fn rejects_test_for_nonexistent_gap() {
        let tests = vec![
            GapAwareTest { id: "x".into(), cost: 1.0, eig_bits: 1.0, advances_predicates: set(&["not-a-gap"]) },
        ];
        assert_eq!(plan(&tests, &set(&["replication"])), Err(PlannerError::UnknownPredicate("not-a-gap".into())));
    }

    #[test]
    fn rejects_negative_cost() {
        let tests = vec![
            GapAwareTest { id: "x".into(), cost: -1.0, eig_bits: 1.0, advances_predicates: set(&["replication"]) },
        ];
        assert_eq!(plan(&tests, &set(&["replication"])), Err(PlannerError::NegativeCost("x".into())));
    }

    #[test]
    fn deterministic_tie_breaking() {
        let gaps = set(&["replication"]);
        let tests = vec![
            GapAwareTest { id: "b".into(), cost: 1.0, eig_bits: 2.0, advances_predicates: set(&["replication"]) },
            GapAwareTest { id: "a".into(), cost: 1.0, eig_bits: 2.0, advances_predicates: set(&["replication"]) },
        ];
        let plan = plan(&tests, &gaps).unwrap();
        assert_eq!(plan[0].test_id, "a");
    }
}

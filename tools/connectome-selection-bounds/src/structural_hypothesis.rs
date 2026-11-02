#![forbid(unsafe_code)]

//! Uncertainty-preserving response bounds over alternative structural connectomes.
//!
//! This module does not choose a winning segmentation, assign posterior weights to
//! structural alternatives, or turn observational response into a causal effect. It
//! computes the union of response-prevalence sets induced by every supplied admissible
//! structural hypothesis. The result is conditional on the supplied hypothesis set;
//! callers remain responsible for justifying that set's coverage and admissibility rule.

use super::{
    EvalError, NumericInterval, joint_response_prevalence_bounds,
    target_given_source_prevalence_bounds,
};

/// One alternative mapping of source/target identities in a candidate structural connectome.
///
/// The source and target marginal intervals must already incorporate channel-specific
/// detector calibration and missingness bounds. `evidence_manifest_sha256` identifies
/// the immutable evidence bundle supporting this structural alternative. No probability
/// weight is accepted: admissible alternatives are retained as a set, not averaged.
#[derive(Debug, Clone, PartialEq)]
pub struct StructuralHypothesisInput {
    pub hypothesis_id: String,
    pub source_entity_id: String,
    pub target_entity_id: String,
    pub evidence_manifest_sha256: String,
    /// False means the hypothesis is recorded but excluded by an upstream frozen rule.
    pub admissible: bool,
    pub source_prevalence: NumericInterval,
    pub target_prevalence: NumericInterval,
}

/// Per-hypothesis result. Marginals and the provenance identity remain attached to the
/// joint and conditional response bounds so a downstream user can trace each alternative.
#[derive(Debug, Clone, PartialEq)]
pub struct StructuralHypothesisResponseBounds {
    pub hypothesis_id: String,
    pub source_entity_id: String,
    pub target_entity_id: String,
    pub evidence_manifest_sha256: String,
    pub source_prevalence: NumericInterval,
    pub target_prevalence: NumericInterval,
    pub joint_response_prevalence: NumericInterval,
    /// None when source prevalence can be zero for this structural hypothesis.
    pub target_given_source_prevalence: Option<NumericInterval>,
}

/// Identified response set across all supplied admissible structural alternatives.
///
/// `joint_response_set` is a canonical sorted union. It may contain multiple disjoint
/// intervals: taking only the convex hull would falsely claim that values in the gaps are
/// compatible with at least one supplied hypothesis.
///
/// `target_given_source_response_set` is None if even one admissible structural
/// hypothesis allows zero source prevalence. This prevents dropping a difficult
/// structural alternative merely because its conditional denominator is not identified.
///
/// The report never applies structural hypothesis weights. It is not a posterior
/// distribution, a confidence interval, or a causal effect estimate.
#[derive(Debug, Clone, PartialEq)]
pub struct StructuralResponseIdentifiedSet {
    pub report_version: &'static str,
    pub admissible_hypothesis_count: usize,
    pub excluded_hypothesis_ids: Vec<String>,
    /// Sorted by hypothesis_id for deterministic serialization.
    pub hypotheses: Vec<StructuralHypothesisResponseBounds>,
    /// Exact finite union of the per-hypothesis joint-prevalence intervals.
    pub joint_response_set: Vec<NumericInterval>,
    /// None if any admissible hypothesis has an unbounded/zero-possible source denominator.
    pub target_given_source_response_set: Option<Vec<NumericInterval>>,
    pub probability_weights_applied: bool,
}

fn validate_identity(h: &StructuralHypothesisInput) -> Result<(), EvalError> {
    if h.hypothesis_id.trim().is_empty() {
        return Err(EvalError::EmptyStructuralHypothesisId);
    }
    if h.source_entity_id.trim().is_empty() {
        return Err(EvalError::InvalidStructuralEntityId("source_entity_id"));
    }
    if h.target_entity_id.trim().is_empty() {
        return Err(EvalError::InvalidStructuralEntityId("target_entity_id"));
    }
    let digest = h.evidence_manifest_sha256.as_bytes();
    if digest.len() != 64 || !digest.iter().all(u8::is_ascii_hexdigit) {
        return Err(EvalError::InvalidEvidenceManifestDigest(
            h.evidence_manifest_sha256.clone(),
        ));
    }
    Ok(())
}

/// Merge overlapping or exactly touching closed intervals without filling real gaps.
fn canonical_interval_union(mut intervals: Vec<NumericInterval>) -> Vec<NumericInterval> {
    intervals.sort_by(|a, b| {
        a.lower.total_cmp(&b.lower)
            .then_with(|| a.upper.total_cmp(&b.upper))
    });
    let mut union: Vec<NumericInterval> = Vec::with_capacity(intervals.len());
    for next in intervals {
        match union.last_mut() {
            Some(last) if next.lower <= last.upper => {
                last.upper = last.upper.max(next.upper);
            }
            _ => union.push(next),
        }
    }
    union
}

/// Propagate response uncertainty through a set of alternative structural interpretations.
///
/// Every hypothesis is validated, including excluded hypotheses, so malformed identity or
/// probability data cannot hide behind an exclusion flag. Duplicate IDs fail closed. All
/// admissible hypotheses are evaluated independently using Fréchet bounds; their possible
/// joint probabilities are returned as a union rather than a weighted mean or convex hull.
///
/// The caller must define and freeze the admissibility rule upstream. If the supplied set
/// is not exhaustive, the returned set is not exhaustive either. The function does not
/// verify topology, segment overlap, synapse evidence, animal/session provenance, or the
/// biological validity of the input marginal prevalence intervals.
pub fn evaluate_structural_hypothesis_response_set(
    hypotheses: &[StructuralHypothesisInput],
) -> Result<StructuralResponseIdentifiedSet, EvalError> {
    let mut seen_ids = std::collections::BTreeSet::new();
    for hypothesis in hypotheses {
        validate_identity(hypothesis)?;
        if !seen_ids.insert(hypothesis.hypothesis_id.as_str()) {
            return Err(EvalError::DuplicateStructuralHypothesisId(
                hypothesis.hypothesis_id.clone(),
            ));
        }
        // Validate marginal intervals even when a hypothesis is excluded.
        joint_response_prevalence_bounds(
            hypothesis.source_prevalence,
            hypothesis.target_prevalence,
        )?;
    }

    let mut admissible: Vec<&StructuralHypothesisInput> =
        hypotheses.iter().filter(|h| h.admissible).collect();
    let mut excluded_hypothesis_ids: Vec<String> = hypotheses
        .iter()
        .filter(|h| !h.admissible)
        .map(|h| h.hypothesis_id.clone())
        .collect();
    if admissible.is_empty() {
        return Err(EvalError::NoAdmissibleStructuralHypotheses);
    }
    admissible.sort_by(|a, b| a.hypothesis_id.cmp(&b.hypothesis_id));
    excluded_hypothesis_ids.sort();

    let mut hypothesis_bounds = Vec::with_capacity(admissible.len());
    let mut joint_intervals = Vec::with_capacity(admissible.len());
    let mut conditional_intervals = Vec::with_capacity(admissible.len());
    let mut every_conditional_is_estimable = true;

    for h in admissible {
        let joint = joint_response_prevalence_bounds(
            h.source_prevalence,
            h.target_prevalence,
        )?;
        let conditional = target_given_source_prevalence_bounds(
            h.source_prevalence,
            h.target_prevalence,
        )?;
        joint_intervals.push(joint);
        match conditional {
            Some(interval) => conditional_intervals.push(interval),
            None => every_conditional_is_estimable = false,
        }
        hypothesis_bounds.push(StructuralHypothesisResponseBounds {
            hypothesis_id: h.hypothesis_id.clone(),
            source_entity_id: h.source_entity_id.clone(),
            target_entity_id: h.target_entity_id.clone(),
            evidence_manifest_sha256: h.evidence_manifest_sha256.clone(),
            source_prevalence: h.source_prevalence,
            target_prevalence: h.target_prevalence,
            joint_response_prevalence: joint,
            target_given_source_prevalence: conditional,
        });
    }

    Ok(StructuralResponseIdentifiedSet {
        report_version: "connectome-structural-response-set-v1",
        admissible_hypothesis_count: hypothesis_bounds.len(),
        excluded_hypothesis_ids,
        hypotheses: hypothesis_bounds,
        joint_response_set: canonical_interval_union(joint_intervals),
        target_given_source_response_set: if every_conditional_is_estimable {
            Some(canonical_interval_union(conditional_intervals))
        } else {
            None
        },
        probability_weights_applied: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const DIGEST_A: &str =
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const DIGEST_B: &str =
        "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

    fn interval(lower: f64, upper: f64) -> NumericInterval {
        NumericInterval { lower, upper }
    }

    fn h(
        id: &str,
        source: NumericInterval,
        target: NumericInterval,
    ) -> StructuralHypothesisInput {
        StructuralHypothesisInput {
            hypothesis_id: id.to_string(),
            source_entity_id: format!("source-{id}"),
            target_entity_id: format!("target-{id}"),
            evidence_manifest_sha256: DIGEST_A.to_string(),
            admissible: true,
            source_prevalence: source,
            target_prevalence: target,
        }
    }

    #[test]
    fn disconnected_structural_alternatives_remain_a_disconnected_joint_set() {
        let report = evaluate_structural_hypothesis_response_set(&[
            h("low", interval(0.1, 0.1), interval(0.1, 0.1)),
            h("high", interval(0.9, 0.9), interval(0.9, 0.9)),
        ]).unwrap();

        assert_eq!(
            report.joint_response_set,
            vec![interval(0.0, 0.1), interval(0.8, 0.9)],
        );
        assert!(!report.probability_weights_applied);
        assert_eq!(report.admissible_hypothesis_count, 2);
        // The convex hull [0,0.9] would falsely fill the infeasible gap.
    }

    #[test]
    fn overlapping_hypothesis_intervals_are_canonicalized_without_gaps() {
        let report = evaluate_structural_hypothesis_response_set(&[
            h("left", interval(0.6, 0.6), interval(0.4, 0.4)), // [0, 0.4]
            h("right", interval(0.7, 0.7), interval(0.6, 0.6)), // [0.3, 0.6]
        ]).unwrap();
        assert_eq!(report.joint_response_set, vec![interval(0.0, 0.6)]);
    }

    #[test]
    fn any_zero_possible_source_hypothesis_blocks_global_conditional_claim() {
        let report = evaluate_structural_hypothesis_response_set(&[
            h("estimable", interval(0.8, 0.8), interval(0.6, 0.6)),
            h("source-may-be-zero", interval(0.0, 0.4), interval(0.6, 0.9)),
        ]).unwrap();
        assert_eq!(
            report.hypotheses[0].target_given_source_prevalence,
            Some(interval(0.5, 0.75)),
        );
        assert_eq!(report.hypotheses[1].target_given_source_prevalence, None);
        assert_eq!(report.target_given_source_response_set, None);
    }

    #[test]
    fn excluded_structural_alternative_is_retained_as_provenance_not_scored() {
        let mut excluded = h("rejected", interval(0.8, 0.8), interval(0.6, 0.6));
        excluded.admissible = false;
        excluded.evidence_manifest_sha256 = DIGEST_B.to_string();
        let report = evaluate_structural_hypothesis_response_set(&[
            h("accepted", interval(0.8, 0.8), interval(0.6, 0.6)),
            excluded,
        ]).unwrap();
        assert_eq!(report.admissible_hypothesis_count, 1);
        assert_eq!(report.excluded_hypothesis_ids, vec!["rejected"]);
        assert_eq!(report.hypotheses[0].hypothesis_id, "accepted");
    }

    #[test]
    fn structural_hypothesis_input_order_does_not_change_canonical_result() {
        let first = h("a", interval(0.7, 0.7), interval(0.6, 0.6));
        let second = h("b", interval(0.9, 0.9), interval(0.2, 0.2));
        let a = evaluate_structural_hypothesis_response_set(&[
            first.clone(), second.clone()
        ]).unwrap();
        let b = evaluate_structural_hypothesis_response_set(&[
            second, first
        ]).unwrap();
        assert_eq!(a.joint_response_set, b.joint_response_set);
        assert_eq!(a.hypotheses[0].hypothesis_id, "a");
        assert_eq!(b.hypotheses[0].hypothesis_id, "a");
    }

    #[test]
    fn no_admissible_hypotheses_fails_closed() {
        let mut rejected = h("rejected", interval(0.8, 0.8), interval(0.6, 0.6));
        rejected.admissible = false;
        assert_eq!(
            evaluate_structural_hypothesis_response_set(&[rejected]).unwrap_err(),
            EvalError::NoAdmissibleStructuralHypotheses,
        );
    }

    #[test]
    fn duplicate_hypothesis_ids_fail_closed() {
        let duplicate = h("same", interval(0.8, 0.8), interval(0.6, 0.6));
        assert_eq!(
            evaluate_structural_hypothesis_response_set(&[
                duplicate.clone(), duplicate
            ]).unwrap_err(),
            EvalError::DuplicateStructuralHypothesisId("same".to_string()),
        );
    }

    #[test]
    fn invalid_evidence_digest_fails_closed() {
        let mut invalid = h("bad-digest", interval(0.8, 0.8), interval(0.6, 0.6));
        invalid.evidence_manifest_sha256 = "not-a-sha256".to_string();
        assert!(matches!(
            evaluate_structural_hypothesis_response_set(&[invalid]),
            Err(EvalError::InvalidEvidenceManifestDigest(_)),
        ));
    }

    #[test]
    fn malformed_marginal_interval_fails_even_for_excluded_hypothesis() {
        let mut invalid = h("excluded-invalid", interval(0.8, 0.7), interval(0.6, 0.6));
        invalid.admissible = false;
        assert!(matches!(
            evaluate_structural_hypothesis_response_set(&[invalid]),
            Err(EvalError::InvalidProbabilityInterval(_)),
        ));
    }
}

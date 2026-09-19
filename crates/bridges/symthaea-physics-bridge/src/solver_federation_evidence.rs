// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Durable evidence envelope for capability-gated solver federation.
//!
//! The validated federation report summarizes coverage/agreement and binds the
//! satisfied physical model. This layer additionally retains the exact pairwise
//! comparison records that produced that summary, so downstream discrepancy
//! localization can bind to concrete metrics, policies, result identities, and
//! comparison artifacts rather than to a lossy summary alone.

use std::cmp::Ordering;

use serde::Serialize;
use symthaea_science_research::{ResearchId, Sha256Digest};

use crate::{
    PairwiseAgreement, SatisfiedPhysicalModel, SolverFederationGateIssue,
    SolverPairComparison, ValidatedSolverFederationReport, ValidatedSolverReceipt,
    solver_federation::FrozenSolverFederationSpec,
    solver_federation_gate::evaluate_validated_solver_federation,
};

const FEDERATION_EVIDENCE_DIGEST_DOMAIN: &str =
    "symthaea.solver-federation-evidence.identity.v1";

/// Exact, serializable, non-deserializable federation evidence.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SolverFederationEvidence {
    validated_report: ValidatedSolverFederationReport,
    comparisons: Vec<SolverPairComparison>,
    evidence_sha256: Sha256Digest,
}

impl SolverFederationEvidence {
    pub fn validated_report(&self) -> &ValidatedSolverFederationReport {
        &self.validated_report
    }

    pub fn comparisons(&self) -> &[SolverPairComparison] {
        &self.comparisons
    }

    pub fn evidence_sha256(&self) -> &Sha256Digest {
        &self.evidence_sha256
    }
}

/// Public authority-bearing solver federation entrypoint.
///
/// The lower-level validated-report evaluator is crate-private at the module
/// boundary. External callers therefore receive a durable envelope that binds
/// both the summary and every exact pairwise comparison supplied to it.
pub fn evaluate_solver_federation_evidence(
    model: &SatisfiedPhysicalModel,
    frozen: &FrozenSolverFederationSpec,
    receipts: &[ValidatedSolverReceipt],
    comparisons: &[SolverPairComparison],
) -> Result<SolverFederationEvidence, Vec<SolverFederationGateIssue>> {
    let validated_report =
        evaluate_validated_solver_federation(model, frozen, receipts, comparisons)?;
    let mut retained_comparisons = comparisons.to_vec();
    retained_comparisons.sort_by(compare_comparisons);
    let evidence_sha256 = evidence_digest(&validated_report, &retained_comparisons);

    Ok(SolverFederationEvidence {
        validated_report,
        comparisons: retained_comparisons,
        evidence_sha256,
    })
}

fn evidence_digest(
    report: &ValidatedSolverFederationReport,
    comparisons: &[SolverPairComparison],
) -> Sha256Digest {
    let mut digest = FramedDigest::new(FEDERATION_EVIDENCE_DIGEST_DOMAIN);
    digest.text(report.report_sha256().as_str());
    digest.text(&comparisons.len().to_string());
    for comparison in comparisons {
        digest_comparison(&mut digest, comparison);
    }
    digest.finish()
}

fn digest_comparison(digest: &mut FramedDigest, comparison: &SolverPairComparison) {
    let (left_id, right_id, left_result, right_result) = canonical_pair(comparison);
    digest.text("comparison");
    digest.text(left_id.as_str());
    digest.text(right_id.as_str());
    digest.text(left_result.as_str());
    digest.text(right_result.as_str());
    digest.text(comparison.comparison_metric_sha256.as_str());
    digest.text(comparison.agreement_policy_sha256.as_str());
    digest.text(outcome_tag(comparison.outcome));
    digest.optional_sha(comparison.comparison_artifact_sha256.as_ref());
    digest.optional_sha(comparison.justification_sha256.as_ref());
}

fn compare_comparisons(left: &SolverPairComparison, right: &SolverPairComparison) -> Ordering {
    let (left_a, left_b, left_result_a, left_result_b) = canonical_pair(left);
    let (right_a, right_b, right_result_a, right_result_b) = canonical_pair(right);

    left_a
        .cmp(&right_a)
        .then_with(|| left_b.cmp(&right_b))
        .then_with(|| left_result_a.cmp(&right_result_a))
        .then_with(|| left_result_b.cmp(&right_result_b))
        .then_with(|| left.comparison_metric_sha256.cmp(&right.comparison_metric_sha256))
        .then_with(|| left.agreement_policy_sha256.cmp(&right.agreement_policy_sha256))
        .then_with(|| left.outcome.cmp(&right.outcome))
        .then_with(|| {
            left.comparison_artifact_sha256
                .cmp(&right.comparison_artifact_sha256)
        })
        .then_with(|| left.justification_sha256.cmp(&right.justification_sha256))
}

fn canonical_pair(
    comparison: &SolverPairComparison,
) -> (ResearchId, ResearchId, Sha256Digest, Sha256Digest) {
    if comparison.left_solver_id <= comparison.right_solver_id {
        (
            comparison.left_solver_id.clone(),
            comparison.right_solver_id.clone(),
            comparison.left_result_sha256.clone(),
            comparison.right_result_sha256.clone(),
        )
    } else {
        (
            comparison.right_solver_id.clone(),
            comparison.left_solver_id.clone(),
            comparison.right_result_sha256.clone(),
            comparison.left_result_sha256.clone(),
        )
    }
}

const fn outcome_tag(outcome: PairwiseAgreement) -> &'static str {
    match outcome {
        PairwiseAgreement::Agree => "agree",
        PairwiseAgreement::Disagree => "disagree",
        PairwiseAgreement::Incomparable => "incomparable",
        PairwiseAgreement::Invalid => "invalid",
    }
}

struct FramedDigest {
    bytes: Vec<u8>,
}

impl FramedDigest {
    fn new(domain: &str) -> Self {
        let mut digest = Self { bytes: Vec::new() };
        digest.text(domain);
        digest
    }

    fn text(&mut self, value: &str) {
        self.bytes
            .extend_from_slice(&(value.len() as u64).to_be_bytes());
        self.bytes.extend_from_slice(value.as_bytes());
    }

    fn optional_sha(&mut self, value: Option<&Sha256Digest>) {
        match value {
            Some(value) => {
                self.text("some");
                self.text(value.as_str());
            }
            None => self.text("none"),
        }
    }

    fn finish(self) -> Sha256Digest {
        Sha256Digest::of_bytes(&self.bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn id(value: &str) -> ResearchId {
        ResearchId::parse(value).unwrap()
    }

    fn sha(value: &str) -> Sha256Digest {
        Sha256Digest::of_bytes(value.as_bytes())
    }

    fn comparison(left: &str, right: &str) -> SolverPairComparison {
        SolverPairComparison {
            left_solver_id: id(left),
            right_solver_id: id(right),
            left_result_sha256: sha(&format!("result-{left}")),
            right_result_sha256: sha(&format!("result-{right}")),
            comparison_metric_sha256: sha("metric"),
            agreement_policy_sha256: sha("policy"),
            outcome: PairwiseAgreement::Disagree,
            comparison_artifact_sha256: Some(sha("comparison-artifact")),
            justification_sha256: None,
        }
    }

    fn comparison_only_digest(comparisons: &[SolverPairComparison]) -> Sha256Digest {
        let mut canonical = comparisons.to_vec();
        canonical.sort_by(compare_comparisons);
        let mut digest = FramedDigest::new("comparison-only-test-v1");
        for comparison in &canonical {
            digest_comparison(&mut digest, comparison);
        }
        digest.finish()
    }

    #[test]
    fn pair_orientation_does_not_change_evidence_identity() {
        let ab = comparison("A", "B");
        let mut ba = comparison("B", "A");
        ba.left_result_sha256 = sha("result-B");
        ba.right_result_sha256 = sha("result-A");
        assert_eq!(
            comparison_only_digest(&[ab]),
            comparison_only_digest(&[ba])
        );
    }

    #[test]
    fn comparison_artifact_substitution_changes_evidence_identity() {
        let original = comparison("A", "B");
        let mut substituted = original.clone();
        substituted.comparison_artifact_sha256 = Some(sha("different-comparison"));
        assert_ne!(
            comparison_only_digest(&[original]),
            comparison_only_digest(&[substituted])
        );
    }

    #[test]
    fn comparison_order_does_not_change_evidence_identity() {
        let ab = comparison("A", "B");
        let ac = comparison("A", "C");
        assert_eq!(
            comparison_only_digest(&[ab.clone(), ac.clone()]),
            comparison_only_digest(&[ac, ab])
        );
    }
}

// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Data contract for a trusted-workflow-produced GWT-1 causal promotion capsule.
//!
//! This module deliberately does **not** verify a GitHub/Sigstore attestation and
//! does not author a Butlin overlay. A structurally valid capsule is still only
//! a piece of data until the capsule itself has been independently attested and
//! verified by the frozen trusted-workflow path.

use serde::{Deserialize, Serialize};

use super::gwt1_causal_resolution::Gwt1CausalQualificationOutcomeV1;
use super::gwt1_end_to_end::Gwt1ExecutionIdentityV1;
use super::report::{EvidenceOutcome, SupportTier};

pub const GWT1_CAUSAL_PROMOTION_CAPSULE_SCHEMA_V1: &str =
    "butlin-gwt1-causal-promotion-capsule-v1";
pub const GWT1_CAUSAL_PROMOTION_POLICY_V1: &str =
    "butlin-gwt1-causal-trusted-promotion-v1";
pub const GWT1_CAUSAL_TRUSTED_BUILDER_WORKFLOW_V1: &str =
    ".github/workflows/butlin-gwt1-causal-trusted-builder.yml";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Gwt1CausalPromotionCapsuleV1 {
    pub schema: String,
    pub policy: String,
    pub indicator_id: String,
    pub trusted_builder_workflow: String,
    pub trusted_builder_sha: String,
    pub trusted_builder_ref: String,
    pub causal_archive_sha256: String,
    pub archive_attestation_verification_sha256: String,
    pub evidence_subject: Gwt1ExecutionIdentityV1,
    pub scientific_outcome: Gwt1CausalQualificationOutcomeV1,
    /// This is an eligibility mapping, not authority to modify a report.
    pub eligible_evidence_outcome: EvidenceOutcome,
    /// V1 can never authorize `FunctionallySupported`.
    pub tier_ceiling: SupportTier,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Gwt1CausalPromotionCapsuleFailureV1 {
    InvalidSchema { observed: String },
    InvalidPolicy { observed: String },
    InvalidIndicator { observed: String },
    InvalidTrustedWorkflow { observed: String },
    InvalidTrustedBuilderSha { observed: String },
    EmptyTrustedBuilderRef,
    InvalidArchiveSha256 { observed: String },
    InvalidVerificationSha256 { observed: String },
    InvalidSourceCommitSha { observed: String },
    InvalidSourceTreeSha { observed: String },
    EmptyExecutionRunId,
    EmptyToolchain,
    EmptySpecialistIdentity,
    OutcomeMappingMismatch,
    InvalidTierCeiling { observed: SupportTier },
}

fn is_lower_hex(value: &str, len: usize) -> bool {
    value.len() == len
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

/// Pure scientific mapping used inside a future trusted promotion step.
///
/// This function does not verify provenance and therefore does not itself
/// promote anything in a report.
pub fn map_gwt1_causal_outcome_to_eligibility_v1(
    outcome: Gwt1CausalQualificationOutcomeV1,
) -> EvidenceOutcome {
    match outcome {
        Gwt1CausalQualificationOutcomeV1::Qualified => {
            EvidenceOutcome::Supported(SupportTier::CausallySupported)
        }
        Gwt1CausalQualificationOutcomeV1::NotDemonstrated => EvidenceOutcome::NotDemonstrated,
        Gwt1CausalQualificationOutcomeV1::Contradicted => EvidenceOutcome::Contradicted,
        Gwt1CausalQualificationOutcomeV1::Inconclusive => EvidenceOutcome::Inconclusive,
    }
}

/// Validate the internal shape and frozen V1 mapping of a promotion capsule.
///
/// Passing this function does **not** establish that the capsule was produced
/// or attested by the trusted workflow. Cryptographic verification remains an
/// external authority prerequisite.
pub fn validate_gwt1_causal_promotion_capsule_v1(
    capsule: &Gwt1CausalPromotionCapsuleV1,
) -> Vec<Gwt1CausalPromotionCapsuleFailureV1> {
    let mut failures = Vec::new();

    if capsule.schema != GWT1_CAUSAL_PROMOTION_CAPSULE_SCHEMA_V1 {
        failures.push(Gwt1CausalPromotionCapsuleFailureV1::InvalidSchema {
            observed: capsule.schema.clone(),
        });
    }
    if capsule.policy != GWT1_CAUSAL_PROMOTION_POLICY_V1 {
        failures.push(Gwt1CausalPromotionCapsuleFailureV1::InvalidPolicy {
            observed: capsule.policy.clone(),
        });
    }
    if capsule.indicator_id != "GWT-1" {
        failures.push(Gwt1CausalPromotionCapsuleFailureV1::InvalidIndicator {
            observed: capsule.indicator_id.clone(),
        });
    }
    if capsule.trusted_builder_workflow != GWT1_CAUSAL_TRUSTED_BUILDER_WORKFLOW_V1 {
        failures.push(Gwt1CausalPromotionCapsuleFailureV1::InvalidTrustedWorkflow {
            observed: capsule.trusted_builder_workflow.clone(),
        });
    }
    if !is_lower_hex(&capsule.trusted_builder_sha, 40) {
        failures.push(Gwt1CausalPromotionCapsuleFailureV1::InvalidTrustedBuilderSha {
            observed: capsule.trusted_builder_sha.clone(),
        });
    }
    if capsule.trusted_builder_ref.trim().is_empty() {
        failures.push(Gwt1CausalPromotionCapsuleFailureV1::EmptyTrustedBuilderRef);
    }
    if !is_lower_hex(&capsule.causal_archive_sha256, 64) {
        failures.push(Gwt1CausalPromotionCapsuleFailureV1::InvalidArchiveSha256 {
            observed: capsule.causal_archive_sha256.clone(),
        });
    }
    if !is_lower_hex(&capsule.archive_attestation_verification_sha256, 64) {
        failures.push(
            Gwt1CausalPromotionCapsuleFailureV1::InvalidVerificationSha256 {
                observed: capsule.archive_attestation_verification_sha256.clone(),
            },
        );
    }
    if !is_lower_hex(&capsule.evidence_subject.source_commit_sha, 40) {
        failures.push(Gwt1CausalPromotionCapsuleFailureV1::InvalidSourceCommitSha {
            observed: capsule.evidence_subject.source_commit_sha.clone(),
        });
    }
    if !is_lower_hex(&capsule.evidence_subject.source_tree_sha, 40) {
        failures.push(Gwt1CausalPromotionCapsuleFailureV1::InvalidSourceTreeSha {
            observed: capsule.evidence_subject.source_tree_sha.clone(),
        });
    }
    if capsule.evidence_subject.execution_run_id.trim().is_empty() {
        failures.push(Gwt1CausalPromotionCapsuleFailureV1::EmptyExecutionRunId);
    }
    if capsule.evidence_subject.toolchain.trim().is_empty() {
        failures.push(Gwt1CausalPromotionCapsuleFailureV1::EmptyToolchain);
    }
    if capsule.evidence_subject.specialist_blob_shas.is_empty()
        || capsule
            .evidence_subject
            .specialist_blob_shas
            .values()
            .any(|sha| !is_lower_hex(sha, 40))
    {
        failures.push(Gwt1CausalPromotionCapsuleFailureV1::EmptySpecialistIdentity);
    }

    let expected = map_gwt1_causal_outcome_to_eligibility_v1(capsule.scientific_outcome);
    if capsule.eligible_evidence_outcome != expected {
        failures.push(Gwt1CausalPromotionCapsuleFailureV1::OutcomeMappingMismatch);
    }
    if capsule.tier_ceiling != SupportTier::CausallySupported {
        failures.push(Gwt1CausalPromotionCapsuleFailureV1::InvalidTierCeiling {
            observed: capsule.tier_ceiling,
        });
    }

    failures
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;

    fn capsule(outcome: Gwt1CausalQualificationOutcomeV1) -> Gwt1CausalPromotionCapsuleV1 {
        let eligibility = map_gwt1_causal_outcome_to_eligibility_v1(outcome);
        Gwt1CausalPromotionCapsuleV1 {
            schema: GWT1_CAUSAL_PROMOTION_CAPSULE_SCHEMA_V1.to_string(),
            policy: GWT1_CAUSAL_PROMOTION_POLICY_V1.to_string(),
            indicator_id: "GWT-1".to_string(),
            trusted_builder_workflow: GWT1_CAUSAL_TRUSTED_BUILDER_WORKFLOW_V1.to_string(),
            trusted_builder_sha: "a".repeat(40),
            trusted_builder_ref: "refs/heads/main".to_string(),
            causal_archive_sha256: "b".repeat(64),
            archive_attestation_verification_sha256: "c".repeat(64),
            evidence_subject: Gwt1ExecutionIdentityV1 {
                source_commit_sha: "d".repeat(40),
                source_tree_sha: "e".repeat(40),
                execution_run_id: "123/1".to_string(),
                toolchain: "rustc 1.96.0 test".to_string(),
                specialist_blob_shas: BTreeMap::from([
                    ("drive_manager".to_string(), "1".repeat(40)),
                    ("memory_manager".to_string(), "2".repeat(40)),
                    ("learning_manager".to_string(), "3".repeat(40)),
                    ("perception_manager".to_string(), "4".repeat(40)),
                ]),
            },
            scientific_outcome: outcome,
            eligible_evidence_outcome: eligibility,
            tier_ceiling: SupportTier::CausallySupported,
        }
    }

    #[test]
    fn qualified_is_eligible_only_for_causal_support() {
        let item = capsule(Gwt1CausalQualificationOutcomeV1::Qualified);
        assert_eq!(
            item.eligible_evidence_outcome,
            EvidenceOutcome::Supported(SupportTier::CausallySupported)
        );
        assert_ne!(
            item.eligible_evidence_outcome,
            EvidenceOutcome::Supported(SupportTier::FunctionallySupported)
        );
        assert!(validate_gwt1_causal_promotion_capsule_v1(&item).is_empty());
    }

    #[test]
    fn negative_science_is_never_softened() {
        for (outcome, expected) in [
            (
                Gwt1CausalQualificationOutcomeV1::NotDemonstrated,
                EvidenceOutcome::NotDemonstrated,
            ),
            (
                Gwt1CausalQualificationOutcomeV1::Contradicted,
                EvidenceOutcome::Contradicted,
            ),
            (
                Gwt1CausalQualificationOutcomeV1::Inconclusive,
                EvidenceOutcome::Inconclusive,
            ),
        ] {
            assert_eq!(map_gwt1_causal_outcome_to_eligibility_v1(outcome), expected);
        }
    }

    #[test]
    fn forged_functional_support_fails_shape_validation() {
        let mut item = capsule(Gwt1CausalQualificationOutcomeV1::Qualified);
        item.eligible_evidence_outcome =
            EvidenceOutcome::Supported(SupportTier::FunctionallySupported);
        assert!(validate_gwt1_causal_promotion_capsule_v1(&item)
            .iter()
            .any(|failure| matches!(
                failure,
                Gwt1CausalPromotionCapsuleFailureV1::OutcomeMappingMismatch
            )));
    }

    #[test]
    fn wrong_trusted_workflow_fails_shape_validation() {
        let mut item = capsule(Gwt1CausalQualificationOutcomeV1::Qualified);
        item.trusted_builder_workflow = ".github/workflows/untrusted.yml".to_string();
        assert!(validate_gwt1_causal_promotion_capsule_v1(&item)
            .iter()
            .any(|failure| matches!(
                failure,
                Gwt1CausalPromotionCapsuleFailureV1::InvalidTrustedWorkflow { .. }
            )));
    }
}

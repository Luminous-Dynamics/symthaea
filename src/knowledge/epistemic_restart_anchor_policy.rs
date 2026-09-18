// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Deployment trust policy for externally verified EKM restart anchors.
//!
//! EKM-046 separates anchor statements from deployment-specific proof providers.
//! This module adds the missing policy membrane above those providers: a proof can
//! be cryptographically valid while still being unacceptable because it came from
//! the wrong authority, used the wrong assurance mechanism, or exceeded the local
//! validity/freshness budget.
//!
//! Policy admission is proposal-only. Passing this gate authorizes only calling
//! the external verifier. It never authorizes quarantine construction or activation.

use super::epistemic_restart_anchor::{
    verify_restart_anchor_evidence, RestartAnchorEvidenceError, RestartAnchorEvidenceKindV1,
    RestartAnchorEvidenceV1, RestartAnchorEvidenceVerifierV1, VerifiedRestartAnchorEvidenceV1,
    MAX_RESTART_ANCHOR_AUTHORITY_ID_BYTES,
};
use super::epistemic_restart_validation_receipt::EpistemicRestartValidationReceiptV1;
use std::error::Error;
use std::fmt;

pub const MAX_RESTART_ANCHOR_POLICY_AUTHORITIES: usize = 64;
pub const MAX_RESTART_ANCHOR_POLICY_EVIDENCE_KINDS: usize = 5;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RestartAnchorTrustPolicyV1 {
    allowed_authority_ids: Vec<String>,
    allowed_evidence_kinds: Vec<RestartAnchorEvidenceKindV1>,
    maximum_validity_window_cycles: u64,
    maximum_verification_delay_cycles: u64,
}

impl RestartAnchorTrustPolicyV1 {
    pub fn new(
        mut allowed_authority_ids: Vec<String>,
        mut allowed_evidence_kinds: Vec<RestartAnchorEvidenceKindV1>,
        maximum_validity_window_cycles: u64,
        maximum_verification_delay_cycles: u64,
    ) -> Result<Self, RestartAnchorTrustPolicyError> {
        if allowed_authority_ids.is_empty()
            || allowed_authority_ids.len() > MAX_RESTART_ANCHOR_POLICY_AUTHORITIES
        {
            return Err(RestartAnchorTrustPolicyError::InvalidAuthorityCount {
                actual: allowed_authority_ids.len(),
                maximum: MAX_RESTART_ANCHOR_POLICY_AUTHORITIES,
            });
        }
        for authority_id in &allowed_authority_ids {
            if authority_id.trim().is_empty()
                || authority_id != authority_id.trim()
                || authority_id.len() > MAX_RESTART_ANCHOR_AUTHORITY_ID_BYTES
            {
                return Err(RestartAnchorTrustPolicyError::InvalidAuthorityId(
                    authority_id.clone(),
                ));
            }
        }
        allowed_authority_ids.sort();
        if allowed_authority_ids
            .windows(2)
            .any(|window| window[0] == window[1])
        {
            return Err(RestartAnchorTrustPolicyError::DuplicateAuthorityId);
        }

        if allowed_evidence_kinds.is_empty()
            || allowed_evidence_kinds.len() > MAX_RESTART_ANCHOR_POLICY_EVIDENCE_KINDS
        {
            return Err(RestartAnchorTrustPolicyError::InvalidEvidenceKindCount {
                actual: allowed_evidence_kinds.len(),
                maximum: MAX_RESTART_ANCHOR_POLICY_EVIDENCE_KINDS,
            });
        }
        allowed_evidence_kinds.sort_by_key(|kind| evidence_kind_tag(*kind));
        if allowed_evidence_kinds
            .windows(2)
            .any(|window| window[0] == window[1])
        {
            return Err(RestartAnchorTrustPolicyError::DuplicateEvidenceKind);
        }

        if maximum_validity_window_cycles == 0 {
            return Err(RestartAnchorTrustPolicyError::ZeroValidityWindow);
        }
        if maximum_verification_delay_cycles == 0 {
            return Err(RestartAnchorTrustPolicyError::ZeroVerificationDelay);
        }

        Ok(Self {
            allowed_authority_ids,
            allowed_evidence_kinds,
            maximum_validity_window_cycles,
            maximum_verification_delay_cycles,
        })
    }

    pub fn allowed_authority_ids(&self) -> &[String] {
        &self.allowed_authority_ids
    }

    pub fn allowed_evidence_kinds(&self) -> &[RestartAnchorEvidenceKindV1] {
        &self.allowed_evidence_kinds
    }

    pub fn maximum_validity_window_cycles(&self) -> u64 {
        self.maximum_validity_window_cycles
    }

    pub fn maximum_verification_delay_cycles(&self) -> u64 {
        self.maximum_verification_delay_cycles
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RestartAnchorTrustPolicyError {
    InvalidAuthorityCount { actual: usize, maximum: usize },
    InvalidAuthorityId(String),
    DuplicateAuthorityId,
    InvalidEvidenceKindCount { actual: usize, maximum: usize },
    DuplicateEvidenceKind,
    ZeroValidityWindow,
    ZeroVerificationDelay,
}

impl fmt::Display for RestartAnchorTrustPolicyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "restart anchor trust policy invalid: {self:?}")
    }
}

impl Error for RestartAnchorTrustPolicyError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RestartAnchorTrustFailureV1 {
    AuthorityNotAllowed(String),
    EvidenceKindNotAllowed(RestartAnchorEvidenceKindV1),
    ValidityWindowExceedsMaximum { maximum: u64, actual: u64 },
    ObservationBeforeIssue {
        observed_at_cycle: u64,
        issued_at_cycle: u64,
    },
    EvidenceExpired {
        observed_at_cycle: u64,
        expires_at_cycle: u64,
    },
    VerificationDelayExceedsMaximum { maximum: u64, actual: u64 },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RestartAnchorTrustDecisionV1 {
    eligible_for_provider_verification: bool,
    failures: Vec<RestartAnchorTrustFailureV1>,
    quarantine_construction_authorized: bool,
    activation_authorized: bool,
}

impl RestartAnchorTrustDecisionV1 {
    pub fn eligible_for_provider_verification(&self) -> bool {
        self.eligible_for_provider_verification
    }

    pub fn failures(&self) -> &[RestartAnchorTrustFailureV1] {
        &self.failures
    }

    pub fn quarantine_construction_authorized(&self) -> bool {
        self.quarantine_construction_authorized
    }

    pub fn activation_authorized(&self) -> bool {
        self.activation_authorized
    }
}

pub struct RestartAnchorTrustGateV1;

impl RestartAnchorTrustGateV1 {
    pub fn evaluate(
        policy: &RestartAnchorTrustPolicyV1,
        evidence: &RestartAnchorEvidenceV1,
        observed_at_cycle: u64,
    ) -> RestartAnchorTrustDecisionV1 {
        let statement = evidence.statement();
        let mut failures = Vec::new();

        if policy
            .allowed_authority_ids
            .binary_search_by(|candidate| candidate.as_str().cmp(statement.authority_id()))
            .is_err()
        {
            failures.push(RestartAnchorTrustFailureV1::AuthorityNotAllowed(
                statement.authority_id().to_string(),
            ));
        }

        if !policy
            .allowed_evidence_kinds
            .contains(&statement.evidence_kind())
        {
            failures.push(RestartAnchorTrustFailureV1::EvidenceKindNotAllowed(
                statement.evidence_kind(),
            ));
        }

        let validity_window = statement
            .expires_at_cycle()
            .saturating_sub(statement.issued_at_cycle());
        if validity_window > policy.maximum_validity_window_cycles {
            failures.push(RestartAnchorTrustFailureV1::ValidityWindowExceedsMaximum {
                maximum: policy.maximum_validity_window_cycles,
                actual: validity_window,
            });
        }

        if observed_at_cycle < statement.issued_at_cycle() {
            failures.push(RestartAnchorTrustFailureV1::ObservationBeforeIssue {
                observed_at_cycle,
                issued_at_cycle: statement.issued_at_cycle(),
            });
        } else {
            let delay = observed_at_cycle - statement.issued_at_cycle();
            if delay > policy.maximum_verification_delay_cycles {
                failures.push(RestartAnchorTrustFailureV1::VerificationDelayExceedsMaximum {
                    maximum: policy.maximum_verification_delay_cycles,
                    actual: delay,
                });
            }
        }

        if observed_at_cycle >= statement.expires_at_cycle() {
            failures.push(RestartAnchorTrustFailureV1::EvidenceExpired {
                observed_at_cycle,
                expires_at_cycle: statement.expires_at_cycle(),
            });
        }

        RestartAnchorTrustDecisionV1 {
            eligible_for_provider_verification: failures.is_empty(),
            failures,
            quarantine_construction_authorized: false,
            activation_authorized: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RestartAnchorPolicyVerificationError {
    TrustPolicyRejected(Vec<RestartAnchorTrustFailureV1>),
    Evidence(RestartAnchorEvidenceError),
}

impl fmt::Display for RestartAnchorPolicyVerificationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "restart anchor policy verification rejected: {self:?}")
    }
}

impl Error for RestartAnchorPolicyVerificationError {}

/// Apply local trust policy before invoking the external verification provider.
///
/// Even successful return grants only a verified anchor object; quarantine and
/// activation authority remain false in that object.
pub fn verify_restart_anchor_evidence_under_policy(
    policy: &RestartAnchorTrustPolicyV1,
    evidence: &RestartAnchorEvidenceV1,
    receipt: &EpistemicRestartValidationReceiptV1,
    observed_at_cycle: u64,
    verifier: &dyn RestartAnchorEvidenceVerifierV1,
) -> Result<VerifiedRestartAnchorEvidenceV1, RestartAnchorPolicyVerificationError> {
    let decision = RestartAnchorTrustGateV1::evaluate(policy, evidence, observed_at_cycle);
    if !decision.eligible_for_provider_verification {
        return Err(RestartAnchorPolicyVerificationError::TrustPolicyRejected(
            decision.failures,
        ));
    }
    verify_restart_anchor_evidence(evidence, receipt, observed_at_cycle, verifier)
        .map_err(RestartAnchorPolicyVerificationError::Evidence)
}

fn evidence_kind_tag(kind: RestartAnchorEvidenceKindV1) -> u8 {
    match kind {
        RestartAnchorEvidenceKindV1::ProtectedCheckpoint => 1,
        RestartAnchorEvidenceKindV1::SignedCheckpoint => 2,
        RestartAnchorEvidenceKindV1::WitnessedCheckpoint => 3,
        RestartAnchorEvidenceKindV1::HardwareAttestedCheckpoint => 4,
        RestartAnchorEvidenceKindV1::TransparencyCheckpoint => 5,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::knowledge::{
        BeliefMutationPersistenceCapsuleV1, BeliefRevisionHistory,
        BeliefRevisionHistoryCapsuleV1, BeliefRevisionPolicySchemaV1,
        BeliefRevisionSchemaHistoryCapsuleV1, BeliefRevisionSchemaHistoryV1, ClaimKind,
        EpistemicLedger, EpistemicLedgerInventoryV1, EpistemicRestartCapsuleV1,
        EpistemicRestartCapsuleV2, EpistemicRestartValidationReceiptV1, EpistemicRestartWireV2,
        EpistemicRevisionProposal, EpistemicSupportStore, EvidenceKind, EvidencePolarity,
        RestartAnchorStatementV1,
    };

    struct Accept;

    impl RestartAnchorEvidenceVerifierV1 for Accept {
        fn verify_restart_anchor_evidence(
            &self,
            _evidence_kind: RestartAnchorEvidenceKindV1,
            _authority_id: &str,
            _statement_digest: [u8; 32],
            _proof: &[u8],
        ) -> Result<bool, String> {
            Ok(true)
        }
    }

    fn receipt(capture_cycle: u64) -> EpistemicRestartValidationReceiptV1 {
        let mut ledger = EpistemicLedger::new();
        let provenance = ledger.add_provenance("lab", None, None, 1, vec![]).unwrap();
        let claim = ledger.add_claim("X predicts Y", ClaimKind::Predictive, None, None, 1);
        let evidence = ledger
            .add_evidence(
                claim,
                EvidenceKind::Measurement,
                EvidencePolarity::Supports,
                provenance,
                2,
                None,
                None,
            )
            .unwrap();
        let inventory =
            EpistemicLedgerInventoryV1::new(vec![claim], vec![evidence], vec![provenance]).unwrap();
        let proposal =
            EpistemicRevisionProposal::new(claim, 0.1, vec![evidence], "measurement").unwrap();
        let schema = BeliefRevisionPolicySchemaV1::new(0.2, 1, false, 0, 1.0).unwrap();
        let mut receipts = BeliefRevisionHistory::new();
        let mut schema_history = BeliefRevisionSchemaHistoryV1::new();
        schema_history
            .evaluate_and_record(
                &mut receipts,
                &ledger,
                &proposal,
                &schema,
                None,
                None,
                3,
            )
            .unwrap();
        let store = EpistemicSupportStore::new();
        let mutations =
            BeliefMutationPersistenceCapsuleV1::capture(&store, &[], capture_cycle).unwrap();
        let revisions =
            BeliefRevisionHistoryCapsuleV1::capture(&receipts, &mutations, capture_cycle).unwrap();
        let schemas = BeliefRevisionSchemaHistoryCapsuleV1::capture(
            &schema_history,
            &receipts,
            &revisions,
            capture_cycle,
        )
        .unwrap();
        let base = EpistemicRestartCapsuleV1::capture(
            &ledger,
            &inventory,
            &mutations,
            &revisions,
            capture_cycle,
        )
        .unwrap();
        let v2 = EpistemicRestartCapsuleV2::capture(&base, &schemas).unwrap();
        let bytes = EpistemicRestartWireV2::encode(&v2).unwrap();
        let snapshot = EpistemicRestartWireV2::decode(&bytes).unwrap();
        EpistemicRestartValidationReceiptV1::validate_and_capture(&snapshot).unwrap()
    }

    fn evidence(
        receipt: &EpistemicRestartValidationReceiptV1,
        authority: &str,
        kind: RestartAnchorEvidenceKindV1,
        issued: u64,
        expires: u64,
    ) -> RestartAnchorEvidenceV1 {
        let statement = RestartAnchorStatementV1::new(
            1,
            receipt,
            None,
            issued,
            expires,
            authority,
            kind,
        )
        .unwrap();
        RestartAnchorEvidenceV1::new(statement, b"proof".to_vec()).unwrap()
    }

    fn policy() -> RestartAnchorTrustPolicyV1 {
        RestartAnchorTrustPolicyV1::new(
            vec!["authority-a".into()],
            vec![RestartAnchorEvidenceKindV1::SignedCheckpoint],
            10,
            4,
        )
        .unwrap()
    }

    #[test]
    fn policy_admission_precedes_provider_verification() {
        let receipt = receipt(4);
        let evidence = evidence(
            &receipt,
            "authority-a",
            RestartAnchorEvidenceKindV1::SignedCheckpoint,
            4,
            10,
        );
        let verified = verify_restart_anchor_evidence_under_policy(
            &policy(),
            &evidence,
            &receipt,
            5,
            &Accept,
        )
        .unwrap();
        assert_eq!(verified.statement().authority_id(), "authority-a");
        assert!(!verified.quarantine_construction_authorized());
        assert!(!verified.activation_authorized());
    }

    #[test]
    fn wrong_authority_and_kind_fail_before_provider() {
        let receipt = receipt(4);
        let evidence = evidence(
            &receipt,
            "authority-b",
            RestartAnchorEvidenceKindV1::HardwareAttestedCheckpoint,
            4,
            10,
        );
        let decision = RestartAnchorTrustGateV1::evaluate(&policy(), &evidence, 5);
        assert!(!decision.eligible_for_provider_verification());
        assert_eq!(decision.failures().len(), 2);
        assert!(decision.failures().iter().any(|failure| matches!(
            failure,
            RestartAnchorTrustFailureV1::AuthorityNotAllowed(authority)
                if authority == "authority-b"
        )));
        assert!(decision.failures().iter().any(|failure| matches!(
            failure,
            RestartAnchorTrustFailureV1::EvidenceKindNotAllowed(
                RestartAnchorEvidenceKindV1::HardwareAttestedCheckpoint
            )
        )));
    }

    #[test]
    fn excessive_validity_or_verification_delay_fails_closed() {
        let receipt = receipt(4);
        let long_lived = evidence(
            &receipt,
            "authority-a",
            RestartAnchorEvidenceKindV1::SignedCheckpoint,
            4,
            20,
        );
        let decision = RestartAnchorTrustGateV1::evaluate(&policy(), &long_lived, 9);
        assert!(!decision.eligible_for_provider_verification());
        assert!(decision.failures().iter().any(|failure| matches!(
            failure,
            RestartAnchorTrustFailureV1::ValidityWindowExceedsMaximum { .. }
        )));
        assert!(decision.failures().iter().any(|failure| matches!(
            failure,
            RestartAnchorTrustFailureV1::VerificationDelayExceedsMaximum { .. }
        )));
    }

    #[test]
    fn policy_constructor_is_canonical_and_rejects_duplicates() {
        let policy = RestartAnchorTrustPolicyV1::new(
            vec!["z".into(), "a".into()],
            vec![
                RestartAnchorEvidenceKindV1::TransparencyCheckpoint,
                RestartAnchorEvidenceKindV1::SignedCheckpoint,
            ],
            10,
            4,
        )
        .unwrap();
        assert_eq!(policy.allowed_authority_ids(), &["a".to_string(), "z".to_string()]);
        assert_eq!(
            policy.allowed_evidence_kinds(),
            &[
                RestartAnchorEvidenceKindV1::SignedCheckpoint,
                RestartAnchorEvidenceKindV1::TransparencyCheckpoint,
            ]
        );

        assert_eq!(
            RestartAnchorTrustPolicyV1::new(
                vec!["a".into(), "a".into()],
                vec![RestartAnchorEvidenceKindV1::SignedCheckpoint],
                10,
                4,
            )
            .unwrap_err(),
            RestartAnchorTrustPolicyError::DuplicateAuthorityId
        );
    }
}

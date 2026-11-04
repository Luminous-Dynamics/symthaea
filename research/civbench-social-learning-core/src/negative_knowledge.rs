// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

#![forbid(unsafe_code)]

//! Scoped negative knowledge for cultural transmission.
//!
//! A verified failed attempt can prevent a recipient from repeating the same
//! attempt, but it must never create a successful recipe or capability. The
//! task, ruleset and environment are part of the scope: a failure in one
//! context is not a universal statement about every possible context.
//!
//! Authentication is deliberately injected through `AttemptReceiptVerifier`.
//! This crate does not claim to implement cryptographic signing or identity;
//! a production adapter must provide and audit that verifier.

use std::collections::{BTreeMap, BTreeSet};

/// Exact context in which an attempt result is meaningful.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct KnowledgeScope {
    pub task_id: String,
    pub task_version: String,
    pub ruleset_digest: String,
    pub environment_digest: String,
}

impl KnowledgeScope {
    pub fn is_valid(&self) -> bool {
        !self.task_id.trim().is_empty()
            && !self.task_version.trim().is_empty()
            && !self.ruleset_digest.trim().is_empty()
            && !self.environment_digest.trim().is_empty()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttemptOutcome {
    Failed,
    Succeeded,
}

/// Transportable evidence about one exact attempted ingredient pair.
///
/// The receipt identity is an application-level identifier. Authenticity and
/// authorization are established only when the injected verifier accepts it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AttemptReceipt {
    pub receipt_id: String,
    pub attempt_id: String,
    pub issuer_id: String,
    pub scope: KnowledgeScope,
    pub ingredients: [u16; 2],
    pub outcome: AttemptOutcome,
    pub issued_at_step: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerificationError {
    UnknownIssuer,
    UnauthorizedIssuer,
    InvalidSignature,
    RevokedCredential,
    MalformedReceipt,
}

/// Pluggable identity, integrity and authorization boundary.
///
/// Implementations should verify the signature over every semantically
/// relevant receipt field and authorize the issuer for the declared scope.
pub trait AttemptReceiptVerifier {
    fn verify(&self, receipt: &AttemptReceipt) -> Result<(), VerificationError>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecordedOutcome {
    Failed,
    Succeeded,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecordedAttempt {
    pub outcome: RecordedOutcome,
    pub evidence_id: String,
    pub issuer_id: String,
    pub issued_at_step: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IngestRejection {
    InvalidScope,
    WrongScope,
    NotFailedAttempt,
    InvalidReceiptIdentity,
    InvalidIngredientPair,
    MissingPrerequisites,
    AlreadyKnown,
    ConflictsWithSuccess,
    AuthenticationFailed(VerificationError),
}

/// Recipient-local knowledge; no population-wide inventory is required.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecipientKnowledge {
    scope: KnowledgeScope,
    known_elements: BTreeSet<u16>,
    attempts: BTreeMap<[u16; 2], RecordedAttempt>,
}

impl RecipientKnowledge {
    pub fn new(scope: KnowledgeScope, known_elements: BTreeSet<u16>) -> Self {
        Self {
            scope,
            known_elements,
            attempts: BTreeMap::new(),
        }
    }

    pub fn scope(&self) -> &KnowledgeScope {
        &self.scope
    }

    pub fn known_elements(&self) -> &BTreeSet<u16> {
        &self.known_elements
    }

    pub fn attempt(&self, ingredients: [u16; 2]) -> Option<&RecordedAttempt> {
        canonical_pair(ingredients).and_then(|pair| self.attempts.get(&pair))
    }

    /// A recipient should suppress a retry only when it holds both
    /// prerequisites and has a scoped negative record for that exact pair.
    pub fn should_skip_attempt(&self, ingredients: [u16; 2]) -> bool {
        let Some(pair) = canonical_pair(ingredients) else {
            return false;
        };
        self.known_elements.contains(&pair[0])
            && self.known_elements.contains(&pair[1])
            && self.attempts.get(&pair).is_some_and(|record| {
                record.outcome == RecordedOutcome::Failed
            })
    }

    /// Import an authenticated, scope-matched negative result.
    ///
    /// This records only an attempt outcome. It does not add a recipe, element,
    /// capability or successful transition.
    pub fn ingest_negative_attempt<V: AttemptReceiptVerifier>(
        &mut self,
        receipt: &AttemptReceipt,
        verifier: &V,
    ) -> Result<(), IngestRejection> {
        if !self.scope.is_valid() || !receipt.scope.is_valid() {
            return Err(IngestRejection::InvalidScope);
        }
        if receipt.receipt_id.trim().is_empty()
            || receipt.attempt_id.trim().is_empty()
            || receipt.issuer_id.trim().is_empty()
        {
            return Err(IngestRejection::InvalidReceiptIdentity);
        }
        if receipt.outcome != AttemptOutcome::Failed {
            return Err(IngestRejection::NotFailedAttempt);
        }
        let pair = canonical_pair(receipt.ingredients)
            .ok_or(IngestRejection::InvalidIngredientPair)?;
        if receipt.scope != self.scope {
            return Err(IngestRejection::WrongScope);
        }
        if !self.known_elements.contains(&pair[0])
            || !self.known_elements.contains(&pair[1])
        {
            return Err(IngestRejection::MissingPrerequisites);
        }
        verifier
            .verify(receipt)
            .map_err(IngestRejection::AuthenticationFailed)?;

        match self.attempts.get(&pair) {
            Some(record) if record.outcome == RecordedOutcome::Failed => {
                return Err(IngestRejection::AlreadyKnown);
            }
            Some(record) if record.outcome == RecordedOutcome::Succeeded => {
                return Err(IngestRejection::ConflictsWithSuccess);
            }
            Some(_) => unreachable!("all recorded outcomes are covered"),
            None => {}
        }

        self.attempts.insert(
            pair,
            RecordedAttempt {
                outcome: RecordedOutcome::Failed,
                evidence_id: receipt.receipt_id.clone(),
                issuer_id: receipt.issuer_id.clone(),
                issued_at_step: receipt.issued_at_step,
            },
        );
        Ok(())
    }

    /// Record an independently established local success so later negative
    /// reports cannot silently overwrite contradictory recipient knowledge.
    pub fn record_local_success(
        &mut self,
        ingredients: [u16; 2],
        evidence_id: impl Into<String>,
    ) -> Result<(), IngestRejection> {
        if !self.scope.is_valid() {
            return Err(IngestRejection::InvalidScope);
        }
        let pair = canonical_pair(ingredients)
            .ok_or(IngestRejection::InvalidIngredientPair)?;
        if !self.known_elements.contains(&pair[0])
            || !self.known_elements.contains(&pair[1])
        {
            return Err(IngestRejection::MissingPrerequisites);
        }
        match self.attempts.get(&pair) {
            Some(record) if record.outcome == RecordedOutcome::Failed => {
                return Err(IngestRejection::ConflictsWithSuccess);
            }
            Some(record) if record.outcome == RecordedOutcome::Succeeded => {
                return Err(IngestRejection::AlreadyKnown);
            }
            Some(_) => unreachable!("all recorded outcomes are covered"),
            None => {}
        }
        let evidence_id = evidence_id.into();
        if evidence_id.trim().is_empty() {
            return Err(IngestRejection::InvalidReceiptIdentity);
        }
        self.attempts.insert(
            pair,
            RecordedAttempt {
                outcome: RecordedOutcome::Succeeded,
                evidence_id,
                issuer_id: "local".to_owned(),
                issued_at_step: 0,
            },
        );
        Ok(())
    }
}

fn canonical_pair(ingredients: [u16; 2]) -> Option<[u16; 2]> {
    if ingredients[0] == ingredients[1] {
        return None;
    }
    let mut pair = ingredients;
    pair.sort_unstable();
    Some(pair)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug)]
    struct AllowListVerifier {
        accepted_receipts: BTreeSet<String>,
    }

    impl AttemptReceiptVerifier for AllowListVerifier {
        fn verify(&self, receipt: &AttemptReceipt) -> Result<(), VerificationError> {
            if self.accepted_receipts.contains(&receipt.receipt_id) {
                Ok(())
            } else {
                Err(VerificationError::InvalidSignature)
            }
        }
    }

    fn scope() -> KnowledgeScope {
        KnowledgeScope {
            task_id: "alchemy-test".to_owned(),
            task_version: "v1".to_owned(),
            ruleset_digest: "sha256:rules-a".to_owned(),
            environment_digest: "sha256:environment-a".to_owned(),
        }
    }

    fn receipt() -> AttemptReceipt {
        AttemptReceipt {
            receipt_id: "receipt-001".to_owned(),
            attempt_id: "attempt-001".to_owned(),
            issuer_id: "agent-7".to_owned(),
            scope: scope(),
            ingredients: [2, 1],
            outcome: AttemptOutcome::Failed,
            issued_at_step: 4,
        }
    }

    fn verifier() -> AllowListVerifier {
        AllowListVerifier {
            accepted_receipts: ["receipt-001".to_owned()].into_iter().collect(),
        }
    }

    fn recipient() -> RecipientKnowledge {
        RecipientKnowledge::new(scope(), [1, 2, 3].into_iter().collect())
    }

    #[test]
    fn verified_scoped_failure_prevents_retry_without_creating_capability() {
        let mut recipient = recipient();
        let initial_elements = recipient.known_elements().clone();

        recipient.ingest_negative_attempt(&receipt(), &verifier()).unwrap();

        assert_eq!(recipient.known_elements(), &initial_elements);
        assert!(recipient.should_skip_attempt([1, 2]));
        assert_eq!(
            recipient.attempt([1, 2]).map(|record| record.outcome),
            Some(RecordedOutcome::Failed)
        );
    }

    #[test]
    fn unverified_negative_evidence_is_rejected() {
        let mut recipient = recipient();
        let rejected = recipient.ingest_negative_attempt(
            &receipt(),
            &AllowListVerifier { accepted_receipts: BTreeSet::new() },
        );

        assert_eq!(
            rejected,
            Err(IngestRejection::AuthenticationFailed(
                VerificationError::InvalidSignature
            ))
        );
        assert!(!recipient.should_skip_attempt([1, 2]));
    }

    #[test]
    fn negative_evidence_cannot_cross_task_or_environment_scope() {
        let mut recipient = recipient();
        let mut foreign = receipt();
        foreign.scope.environment_digest = "sha256:environment-b".to_owned();

        assert_eq!(
            recipient.ingest_negative_attempt(&foreign, &verifier()),
            Err(IngestRejection::WrongScope)
        );
        assert!(!recipient.should_skip_attempt([1, 2]));
    }

    #[test]
    fn negative_evidence_is_actionable_only_when_recipient_has_prerequisites() {
        let mut recipient = RecipientKnowledge::new(scope(), [1].into_iter().collect());

        assert_eq!(
            recipient.ingest_negative_attempt(&receipt(), &verifier()),
            Err(IngestRejection::MissingPrerequisites)
        );
        assert!(!recipient.should_skip_attempt([1, 2]));
    }

    #[test]
    fn duplicate_negative_evidence_is_not_counted_twice() {
        let mut recipient = recipient();
        recipient.ingest_negative_attempt(&receipt(), &verifier()).unwrap();

        assert_eq!(
            recipient.ingest_negative_attempt(&receipt(), &verifier()),
            Err(IngestRejection::AlreadyKnown)
        );
    }

    #[test]
    fn negative_evidence_cannot_overwrite_a_known_success() {
        let mut recipient = recipient();
        recipient.record_local_success([1, 2], "local-success").unwrap();

        assert_eq!(
            recipient.ingest_negative_attempt(&receipt(), &verifier()),
            Err(IngestRejection::ConflictsWithSuccess)
        );
        assert!(!recipient.should_skip_attempt([1, 2]));
    }

    #[test]
    fn successful_attempt_receipt_is_not_negative_knowledge() {
        let mut recipient = recipient();
        let mut success = receipt();
        success.outcome = AttemptOutcome::Succeeded;

        assert_eq!(
            recipient.ingest_negative_attempt(&success, &verifier()),
            Err(IngestRejection::NotFailedAttempt)
        );
    }

    #[test]
    fn malformed_or_duplicate_ingredient_pairs_are_rejected() {
        let mut recipient = recipient();
        let mut malformed = receipt();
        malformed.ingredients = [1, 1];

        assert_eq!(
            recipient.ingest_negative_attempt(&malformed, &verifier()),
            Err(IngestRejection::InvalidIngredientPair)
        );
        assert!(!recipient.should_skip_attempt([1, 1]));
    }
}

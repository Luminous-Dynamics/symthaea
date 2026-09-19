// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Pure exact-subject identity for interaction data-egress policy.
//!
//! This crate defines **what exact semantic subject a later policy decision is
//! about**. It does not evaluate policy and deliberately contains no policy
//! outcome, current authority, connector execution, credential access, network
//! I/O, persistence, or dispatch capability.
//!
//! Core separation:
//!
//! ```text
//! EgressDecisionSubjectV1
//!     != policy decision
//!     != policy currentness
//!     != current authority
//!     != DispatchPermit
//!     != disclosure/effect
//! ```
//!
//! The subject binds both a lineage-owned [`DisclosureBinding`] and the exact
//! [`InteractionIntent`]. Therefore an eventual policy receipt can be replayed
//! only for the exact pair it evaluated; changing connector, principal,
//! resource, operation, input commitment, effect class, idempotency semantics,
//! lineage, destination, outbound payload, or disclosure purpose changes the
//! subject identity transitively.

#![deny(unsafe_code)]

use sha2::{Digest as ShaDigest, Sha256};
use std::error::Error;
use std::fmt;
use symthaea_interaction_core::{Digest32, InteractionIntent};
use symthaea_interaction_lineage::DisclosureBinding;

pub const EGRESS_SUBJECT_SCHEMA_VERSION: u16 = 1;
const EGRESS_SUBJECT_DOMAIN: &[u8] = b"symthaea.interaction.egress.subject.v1\0";
const MAX_EVIDENCE_REFS: usize = 64;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EgressSubjectError {
    ZeroDigest(&'static str),
    TooManyEvidenceRefs(usize),
    DuplicateEvidenceRef,
}

impl fmt::Display for EgressSubjectError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroDigest(field) => write!(formatter, "{field} must not use an all-zero digest"),
            Self::TooManyEvidenceRefs(count) => write!(
                formatter,
                "declassification evidence contains {count} references; maximum is {MAX_EVIDENCE_REFS}"
            ),
            Self::DuplicateEvidenceRef => {
                write!(formatter, "declassification evidence repeats one commitment")
            }
        }
    }
}

impl Error for EgressSubjectError {}

/// Stable typed identity of one exact egress decision subject.
///
/// The typed wrapper prevents callers from casually confusing the subject
/// identity with an unrelated digest. It still grants no authority.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct EgressDecisionSubjectId(Digest32);

impl EgressDecisionSubjectId {
    pub const fn digest(self) -> Digest32 {
        self.0
    }
}

/// Exact semantic input to a future data-egress policy decision.
///
/// `DisclosureBinding` already content-binds the exact `DataLineage`,
/// destination resource, outbound-payload commitment, disclosure purpose/context
/// and disclosure profile. `InteractionIntent` independently binds the exact
/// connector, principal, resource, operation, input commitment, effect class,
/// and idempotency profile.
///
/// This v1 subject intentionally binds the two canonical objects rather than
/// duplicating their individual fields into a second attacker-selectable
/// representation. A later PDP may inspect the original objects and must reject
/// any protocol-specific semantic inconsistency it cares about, but the receipt
/// identity remains about this exact pair.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EgressDecisionSubjectV1 {
    disclosure_binding: Digest32,
    interaction_intent: Digest32,
    policy_input_context: Digest32,
    session_context: Option<Digest32>,
    declassification_evidence: Vec<Digest32>,
}

impl EgressDecisionSubjectV1 {
    pub fn new(
        disclosure: &DisclosureBinding,
        intent: &InteractionIntent,
        policy_input_context: Digest32,
        session_context: Option<Digest32>,
        declassification_evidence: Vec<Digest32>,
    ) -> Result<Self, EgressSubjectError> {
        reject_zero("policy_input_context", policy_input_context)?;
        if let Some(session) = session_context {
            reject_zero("session_context", session)?;
        }
        let declassification_evidence = normalize_evidence(declassification_evidence)?;

        Ok(Self {
            disclosure_binding: disclosure.digest(),
            interaction_intent: intent.digest(),
            policy_input_context,
            session_context,
            declassification_evidence,
        })
    }

    pub const fn disclosure_binding_digest(&self) -> Digest32 {
        self.disclosure_binding
    }

    pub const fn interaction_intent_digest(&self) -> Digest32 {
        self.interaction_intent
    }

    pub const fn policy_input_context(&self) -> Digest32 {
        self.policy_input_context
    }

    pub const fn session_context(&self) -> Option<Digest32> {
        self.session_context
    }

    pub fn declassification_evidence(&self) -> &[Digest32] {
        &self.declassification_evidence
    }

    #[must_use]
    pub fn id(&self) -> EgressDecisionSubjectId {
        let mut transcript = Transcript::new(EGRESS_SUBJECT_DOMAIN);
        transcript.u16(EGRESS_SUBJECT_SCHEMA_VERSION);
        transcript.digest(self.disclosure_binding);
        transcript.digest(self.interaction_intent);
        transcript.digest(self.policy_input_context);
        transcript.optional_digest(self.session_context);
        transcript.digest_set(&self.declassification_evidence);
        EgressDecisionSubjectId(transcript.finish())
    }
}

fn reject_zero(field: &'static str, value: Digest32) -> Result<(), EgressSubjectError> {
    if value.as_bytes() == &[0; 32] {
        Err(EgressSubjectError::ZeroDigest(field))
    } else {
        Ok(())
    }
}

fn normalize_evidence(mut values: Vec<Digest32>) -> Result<Vec<Digest32>, EgressSubjectError> {
    if values.len() > MAX_EVIDENCE_REFS {
        return Err(EgressSubjectError::TooManyEvidenceRefs(values.len()));
    }
    for value in &values {
        reject_zero("declassification_evidence", *value)?;
    }
    values.sort();
    if values.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err(EgressSubjectError::DuplicateEvidenceRef);
    }
    Ok(values)
}

struct Transcript {
    hasher: Sha256,
}

impl Transcript {
    fn new(domain: &[u8]) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(domain);
        Self { hasher }
    }

    fn u8(&mut self, value: u8) {
        self.hasher.update([value]);
    }

    fn u16(&mut self, value: u16) {
        self.hasher.update(value.to_be_bytes());
    }

    fn u32(&mut self, value: u32) {
        self.hasher.update(value.to_be_bytes());
    }

    fn digest(&mut self, value: Digest32) {
        self.hasher.update(value.as_bytes());
    }

    fn optional_digest(&mut self, value: Option<Digest32>) {
        match value {
            Some(value) => {
                self.u8(1);
                self.digest(value);
            }
            None => self.u8(0),
        }
    }

    fn digest_set(&mut self, values: &[Digest32]) {
        self.u32(values.len() as u32);
        for value in values {
            self.digest(*value);
        }
    }

    fn finish(self) -> Digest32 {
        let digest = self.hasher.finalize();
        let mut bytes = [0_u8; 32];
        bytes.copy_from_slice(&digest);
        Digest32::new(bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_interaction_core::{
        ConnectorId, ConnectorIdentity, EffectClass, IdempotencyClass, IdentityComponent,
        IdentityOrdering, NamespaceId, OperationRef, ResourceRef,
    };
    use symthaea_interaction_lineage::{
        ContentDisposition, DataLineage, OriginKind, SourceBinding,
    };

    fn digest(byte: u8) -> Digest32 {
        Digest32::new([byte; 32])
    }

    fn component(name: &str, value: &str) -> IdentityComponent {
        IdentityComponent::new(name, value).expect("component")
    }

    fn resource(name: &str) -> ResourceRef {
        ResourceRef::new(
            NamespaceId::new("web/http").expect("namespace"),
            "object",
            IdentityOrdering::NamedSet,
            vec![component("name", name)],
        )
        .expect("resource")
    }

    fn lineage() -> DataLineage {
        DataLineage::origin(
            digest(0x11),
            SourceBinding::new(
                OriginKind::ExternalObservation,
                ContentDisposition::ExternalContent,
                Some(&resource("source")),
                Some(digest(0x22)),
                None,
                vec![digest(0x33)],
            )
            .expect("source"),
            vec![digest(0x44)],
        )
        .expect("lineage")
    }

    fn disclosure(payload: u8, destination: &str) -> DisclosureBinding {
        DisclosureBinding::new(
            &lineage(),
            &resource(destination),
            digest(payload),
            digest(0x91),
            None,
            "network-egress/v1",
        )
        .expect("disclosure")
    }

    fn intent(connector_name: &str, operation_name: &str) -> InteractionIntent {
        let namespace = NamespaceId::new("web/http").expect("namespace");
        let connector = ConnectorIdentity::new(
            ConnectorId::new(namespace.clone(), connector_name).expect("connector id"),
            "https-json/v1",
            Some(digest(0x60)),
        )
        .expect("connector");
        let operation = OperationRef::new(namespace, operation_name, Some("json/v1"))
            .expect("operation");
        InteractionIntent::new(
            connector,
            None,
            resource("public-api"),
            operation,
            digest(0x90),
            EffectClass::Publish,
            IdempotencyClass::IdempotencyKeyed,
        )
    }

    fn subject() -> EgressDecisionSubjectV1 {
        EgressDecisionSubjectV1::new(
            &disclosure(0x90, "public-api"),
            &intent("rest", "post"),
            digest(0xa0),
            Some(digest(0xa1)),
            vec![digest(0xb0), digest(0xb1)],
        )
        .expect("subject")
    }

    #[test]
    fn subject_vector_is_stable() {
        assert_eq!(
            subject().id().digest().to_hex(),
            "8a6bfabf18ce5696ce0b394cc6ee9b7bb7ae10494d5a68e5b5e47cc257883cde"
        );
    }

    #[test]
    fn connector_or_operation_change_invalidates_subject() {
        let disclosure = disclosure(0x90, "public-api");
        let baseline = EgressDecisionSubjectV1::new(
            &disclosure,
            &intent("rest", "post"),
            digest(0xa0),
            Some(digest(0xa1)),
            Vec::new(),
        )
        .expect("baseline");
        let connector_changed = EgressDecisionSubjectV1::new(
            &disclosure,
            &intent("rest-v2", "post"),
            digest(0xa0),
            Some(digest(0xa1)),
            Vec::new(),
        )
        .expect("connector changed");
        let operation_changed = EgressDecisionSubjectV1::new(
            &disclosure,
            &intent("rest", "put"),
            digest(0xa0),
            Some(digest(0xa1)),
            Vec::new(),
        )
        .expect("operation changed");

        assert_ne!(baseline.id(), connector_changed.id());
        assert_ne!(baseline.id(), operation_changed.id());
    }

    #[test]
    fn disclosure_destination_or_payload_change_invalidates_subject() {
        let intent = intent("rest", "post");
        let baseline = EgressDecisionSubjectV1::new(
            &disclosure(0x90, "public-api"),
            &intent,
            digest(0xa0),
            None,
            Vec::new(),
        )
        .expect("baseline");
        let destination_changed = EgressDecisionSubjectV1::new(
            &disclosure(0x90, "other-api"),
            &intent,
            digest(0xa0),
            None,
            Vec::new(),
        )
        .expect("destination changed");
        let payload_changed = EgressDecisionSubjectV1::new(
            &disclosure(0x92, "public-api"),
            &intent,
            digest(0xa0),
            None,
            Vec::new(),
        )
        .expect("payload changed");

        assert_ne!(baseline.id(), destination_changed.id());
        assert_ne!(baseline.id(), payload_changed.id());
    }

    #[test]
    fn policy_or_session_context_change_invalidates_subject() {
        let disclosure = disclosure(0x90, "public-api");
        let intent = intent("rest", "post");
        let baseline = EgressDecisionSubjectV1::new(
            &disclosure,
            &intent,
            digest(0xa0),
            Some(digest(0xa1)),
            Vec::new(),
        )
        .expect("baseline");
        let policy_changed = EgressDecisionSubjectV1::new(
            &disclosure,
            &intent,
            digest(0xa2),
            Some(digest(0xa1)),
            Vec::new(),
        )
        .expect("policy changed");
        let session_changed = EgressDecisionSubjectV1::new(
            &disclosure,
            &intent,
            digest(0xa0),
            Some(digest(0xa3)),
            Vec::new(),
        )
        .expect("session changed");

        assert_ne!(baseline.id(), policy_changed.id());
        assert_ne!(baseline.id(), session_changed.id());
    }

    #[test]
    fn evidence_order_is_non_semantic_but_duplicates_fail_closed() {
        let disclosure = disclosure(0x90, "public-api");
        let intent = intent("rest", "post");
        let left = EgressDecisionSubjectV1::new(
            &disclosure,
            &intent,
            digest(0xa0),
            None,
            vec![digest(0xb0), digest(0xb1)],
        )
        .expect("left");
        let right = EgressDecisionSubjectV1::new(
            &disclosure,
            &intent,
            digest(0xa0),
            None,
            vec![digest(0xb1), digest(0xb0)],
        )
        .expect("right");
        assert_eq!(left.id(), right.id());

        let error = EgressDecisionSubjectV1::new(
            &disclosure,
            &intent,
            digest(0xa0),
            None,
            vec![digest(0xb0), digest(0xb0)],
        )
        .expect_err("duplicate evidence must fail");
        assert_eq!(error, EgressSubjectError::DuplicateEvidenceRef);
    }

    #[test]
    fn zero_context_commitments_fail_closed() {
        let disclosure = disclosure(0x90, "public-api");
        let intent = intent("rest", "post");
        assert!(matches!(
            EgressDecisionSubjectV1::new(
                &disclosure,
                &intent,
                Digest32::new([0; 32]),
                None,
                Vec::new(),
            ),
            Err(EgressSubjectError::ZeroDigest("policy_input_context"))
        ));
        assert!(matches!(
            EgressDecisionSubjectV1::new(
                &disclosure,
                &intent,
                digest(0xa0),
                Some(Digest32::new([0; 32])),
                Vec::new(),
            ),
            Err(EgressSubjectError::ZeroDigest("session_context"))
        ));
    }
}

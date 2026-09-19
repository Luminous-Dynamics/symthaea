// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Pure typed policy-decision evidence for the Interaction Fabric.
//!
//! This crate records **what exact subject was evaluated, under which exact
//! policy/engine identity, and what bounded outcome was produced**. It does not
//! run a policy engine, authenticate a PDP, establish policy currentness, grant
//! authority, reserve action capacity, mint a dispatch permit, or execute an
//! effect.
//!
//! ```text
//! PolicyDecisionReceiptV1
//!     != authenticated PDP provenance
//!     != current policy
//!     != current authority
//!     != action admission
//!     != DispatchPermit
//!     != effect
//! ```
//! 
//! The receipt therefore remains ordinary evidence until a later verifier binds
//! it to an accepted engine/policy/currentness profile.

#![deny(unsafe_code)]

use sha2::{Digest as ShaDigest, Sha256};
use std::error::Error;
use std::fmt;
use symthaea_interaction_core::{Digest32, InteractionIntent};
use symthaea_interaction_egress::EgressDecisionSubjectId;

pub const POLICY_EVIDENCE_SCHEMA_VERSION: u16 = 1;

const SUBJECT_DOMAIN: &[u8] = b"symthaea.interaction.policy.subject.v1\0";
const BUNDLE_DOMAIN: &[u8] = b"symthaea.interaction.policy.bundle.v1\0";
const ENGINE_DOMAIN: &[u8] = b"symthaea.interaction.policy.engine.v1\0";
const RECEIPT_DOMAIN: &[u8] = b"symthaea.interaction.policy.receipt.v1\0";

const MAX_TEXT_LEN: usize = 256;
const MAX_RULE_IDS: usize = 64;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PolicyEvidenceError {
    EmptyText(&'static str),
    TextTooLong {
        field: &'static str,
        length: usize,
        maximum: usize,
    },
    NonCanonicalText {
        field: &'static str,
        byte: u8,
    },
    ZeroDigest(&'static str),
    TooManyRuleIds(usize),
    DuplicateRuleId,
}

impl fmt::Display for PolicyEvidenceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyText(field) => write!(formatter, "{field} must not be empty"),
            Self::TextTooLong {
                field,
                length,
                maximum,
            } => write!(
                formatter,
                "{field} length {length} exceeds maximum {maximum}"
            ),
            Self::NonCanonicalText { field, byte } => write!(
                formatter,
                "{field} contains non-canonical byte 0x{byte:02x}; identity text must be printable ASCII"
            ),
            Self::ZeroDigest(field) => write!(formatter, "{field} must not use an all-zero digest"),
            Self::TooManyRuleIds(count) => write!(
                formatter,
                "policy decision contains {count} rule identifiers; maximum is {MAX_RULE_IDS}"
            ),
            Self::DuplicateRuleId => write!(formatter, "policy decision repeats one rule identifier"),
        }
    }
}

impl Error for PolicyEvidenceError {}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct CanonicalText(String);

impl CanonicalText {
    fn new(field: &'static str, value: &str) -> Result<Self, PolicyEvidenceError> {
        if value.is_empty() {
            return Err(PolicyEvidenceError::EmptyText(field));
        }
        if value.len() > MAX_TEXT_LEN {
            return Err(PolicyEvidenceError::TextTooLong {
                field,
                length: value.len(),
                maximum: MAX_TEXT_LEN,
            });
        }
        if let Some(byte) = value
            .as_bytes()
            .iter()
            .copied()
            .find(|byte| !byte.is_ascii_graphic())
        {
            return Err(PolicyEvidenceError::NonCanonicalText { field, byte });
        }
        Ok(Self(value.to_owned()))
    }

    fn as_str(&self) -> &str {
        &self.0
    }
}

fn reject_zero(field: &'static str, value: Digest32) -> Result<(), PolicyEvidenceError> {
    if value.as_bytes() == &[0; 32] {
        Err(PolicyEvidenceError::ZeroDigest(field))
    } else {
        Ok(())
    }
}

/// Stable semantic class of a policy decision subject.
///
/// The class code is committed alongside the subject digest so the same 32-byte
/// value cannot be reinterpreted as a different subject family.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PolicySubjectKind {
    Interaction,
    Egress,
}

impl PolicySubjectKind {
    const fn code(self) -> u16 {
        match self {
            Self::Interaction => 0,
            Self::Egress => 1,
        }
    }
}

/// Typed exact subject consumed by a policy decision point.
///
/// There is deliberately no public raw-digest constructor in v1. Production
/// callers construct a subject from a canonical semantic object.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PolicyDecisionSubjectV1 {
    kind: PolicySubjectKind,
    subject_digest: Digest32,
}

impl PolicyDecisionSubjectV1 {
    pub fn interaction(intent: &InteractionIntent) -> Self {
        Self {
            kind: PolicySubjectKind::Interaction,
            subject_digest: intent.digest(),
        }
    }

    pub fn egress(subject: EgressDecisionSubjectId) -> Self {
        Self {
            kind: PolicySubjectKind::Egress,
            subject_digest: subject.digest(),
        }
    }

    pub const fn kind(&self) -> PolicySubjectKind {
        self.kind
    }

    pub const fn subject_digest(&self) -> Digest32 {
        self.subject_digest
    }

    #[must_use]
    pub fn digest(&self) -> Digest32 {
        let mut transcript = Transcript::new(SUBJECT_DOMAIN);
        transcript.u16(POLICY_EVIDENCE_SCHEMA_VERSION);
        transcript.u16(self.kind.code());
        transcript.digest(self.subject_digest);
        transcript.finish()
    }
}

/// Exact non-authoritative identity of one policy bundle/source revision.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PolicyBundleIdentityV1 {
    profile: CanonicalText,
    version: CanonicalText,
    content_digest: Digest32,
}

impl PolicyBundleIdentityV1 {
    pub fn new(
        profile: &str,
        version: &str,
        content_digest: Digest32,
    ) -> Result<Self, PolicyEvidenceError> {
        reject_zero("policy bundle content digest", content_digest)?;
        Ok(Self {
            profile: CanonicalText::new("policy bundle profile", profile)?,
            version: CanonicalText::new("policy bundle version", version)?,
            content_digest,
        })
    }

    pub fn profile(&self) -> &str {
        self.profile.as_str()
    }

    pub fn version(&self) -> &str {
        self.version.as_str()
    }

    pub const fn content_digest(&self) -> Digest32 {
        self.content_digest
    }

    #[must_use]
    pub fn digest(&self) -> Digest32 {
        let mut transcript = Transcript::new(BUNDLE_DOMAIN);
        transcript.u16(POLICY_EVIDENCE_SCHEMA_VERSION);
        transcript.text(self.profile.as_str());
        transcript.text(self.version.as_str());
        transcript.digest(self.content_digest);
        transcript.finish()
    }
}

/// Exact non-authoritative identity of the policy engine/interpreter used.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PolicyEngineIdentityV1 {
    family: CanonicalText,
    profile: CanonicalText,
    implementation_digest: Digest32,
}

impl PolicyEngineIdentityV1 {
    pub fn new(
        family: &str,
        profile: &str,
        implementation_digest: Digest32,
    ) -> Result<Self, PolicyEvidenceError> {
        reject_zero("policy engine implementation digest", implementation_digest)?;
        Ok(Self {
            family: CanonicalText::new("policy engine family", family)?,
            profile: CanonicalText::new("policy engine profile", profile)?,
            implementation_digest,
        })
    }

    pub fn family(&self) -> &str {
        self.family.as_str()
    }

    pub fn profile(&self) -> &str {
        self.profile.as_str()
    }

    pub const fn implementation_digest(&self) -> Digest32 {
        self.implementation_digest
    }

    #[must_use]
    pub fn digest(&self) -> Digest32 {
        let mut transcript = Transcript::new(ENGINE_DOMAIN);
        transcript.u16(POLICY_EVIDENCE_SCHEMA_VERSION);
        transcript.text(self.family.as_str());
        transcript.text(self.profile.as_str());
        transcript.digest(self.implementation_digest);
        transcript.finish()
    }
}

/// Bounded typed output of one policy evaluation.
///
/// `AllowCandidate` is intentionally not named `Allow` because it remains
/// evidence only; a PEP must still establish current policy, current authority,
/// and every other required admission theorem.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PolicyDecisionOutcome {
    Deny,
    AllowCandidate,
    NeedsConfirmation,
    NeedsPreflight,
    NeedsStrongerIdentity,
    NeedsFreshEvidence,
}

impl PolicyDecisionOutcome {
    const fn code(self) -> u16 {
        match self {
            Self::Deny => 0,
            Self::AllowCandidate => 1,
            Self::NeedsConfirmation => 2,
            Self::NeedsPreflight => 3,
            Self::NeedsStrongerIdentity => 4,
            Self::NeedsFreshEvidence => 5,
        }
    }
}

/// Bounded machine-readable rule/reason identity.
///
/// Rule identifiers are audit labels only. They cannot override the typed
/// `PolicyDecisionOutcome`.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PolicyRuleId(CanonicalText);

impl PolicyRuleId {
    pub fn new(value: &str) -> Result<Self, PolicyEvidenceError> {
        Ok(Self(CanonicalText::new("policy rule id", value)?))
    }

    pub fn as_str(&self) -> &str {
        self.0.as_str()
    }
}

/// Typed identity of one canonical policy decision receipt.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PolicyDecisionReceiptId(Digest32);

impl PolicyDecisionReceiptId {
    pub const fn digest(self) -> Digest32 {
        self.0
    }
}

/// Canonical ordinary evidence of one policy decision.
///
/// Construction proves only deterministic record shape. It does not prove that
/// the named engine actually evaluated the named bundle, that the receipt came
/// from an authenticated PDP, or that the bundle/engine remain current.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PolicyDecisionReceiptV1 {
    subject: Digest32,
    bundle: Digest32,
    engine: Digest32,
    evaluation_context: Digest32,
    currentness_evidence: Option<Digest32>,
    outcome: PolicyDecisionOutcome,
    rule_ids: Vec<PolicyRuleId>,
}

impl PolicyDecisionReceiptV1 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        subject: &PolicyDecisionSubjectV1,
        bundle: &PolicyBundleIdentityV1,
        engine: &PolicyEngineIdentityV1,
        evaluation_context: Digest32,
        currentness_evidence: Option<Digest32>,
        outcome: PolicyDecisionOutcome,
        rule_ids: Vec<PolicyRuleId>,
    ) -> Result<Self, PolicyEvidenceError> {
        reject_zero("policy evaluation context", evaluation_context)?;
        if let Some(currentness) = currentness_evidence {
            reject_zero("policy currentness evidence", currentness)?;
        }
        let rule_ids = normalize_rule_ids(rule_ids)?;

        Ok(Self {
            subject: subject.digest(),
            bundle: bundle.digest(),
            engine: engine.digest(),
            evaluation_context,
            currentness_evidence,
            outcome,
            rule_ids,
        })
    }

    pub const fn subject_digest(&self) -> Digest32 {
        self.subject
    }

    pub const fn bundle_digest(&self) -> Digest32 {
        self.bundle
    }

    pub const fn engine_digest(&self) -> Digest32 {
        self.engine
    }

    pub const fn evaluation_context(&self) -> Digest32 {
        self.evaluation_context
    }

    pub const fn currentness_evidence(&self) -> Option<Digest32> {
        self.currentness_evidence
    }

    pub const fn outcome(&self) -> PolicyDecisionOutcome {
        self.outcome
    }

    pub fn rule_ids(&self) -> &[PolicyRuleId] {
        &self.rule_ids
    }

    #[must_use]
    pub fn id(&self) -> PolicyDecisionReceiptId {
        let mut transcript = Transcript::new(RECEIPT_DOMAIN);
        transcript.u16(POLICY_EVIDENCE_SCHEMA_VERSION);
        transcript.digest(self.subject);
        transcript.digest(self.bundle);
        transcript.digest(self.engine);
        transcript.digest(self.evaluation_context);
        transcript.optional_digest(self.currentness_evidence);
        transcript.u16(self.outcome.code());
        transcript.text_set(&self.rule_ids);
        PolicyDecisionReceiptId(transcript.finish())
    }
}

fn normalize_rule_ids(mut values: Vec<PolicyRuleId>) -> Result<Vec<PolicyRuleId>, PolicyEvidenceError> {
    if values.len() > MAX_RULE_IDS {
        return Err(PolicyEvidenceError::TooManyRuleIds(values.len()));
    }
    values.sort();
    if values.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err(PolicyEvidenceError::DuplicateRuleId);
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

    fn text(&mut self, value: &str) {
        self.u32(value.len() as u32);
        self.hasher.update(value.as_bytes());
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

    fn text_set(&mut self, values: &[PolicyRuleId]) {
        self.u32(values.len() as u32);
        for value in values {
            self.text(value.as_str());
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
    use symthaea_interaction_egress::EgressDecisionSubjectV1;
    use symthaea_interaction_lineage::{
        ContentDisposition, DataLineage, DisclosureBinding, OriginKind, SourceBinding,
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

    fn interaction_intent() -> InteractionIntent {
        let namespace = NamespaceId::new("web/http").expect("namespace");
        let connector = ConnectorIdentity::new(
            ConnectorId::new(namespace.clone(), "rest").expect("connector id"),
            "https-json/v1",
            Some(digest(0x60)),
        )
        .expect("connector");
        let operation = OperationRef::new(namespace, "post", Some("json/v1")).expect("operation");
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

    fn egress_subject() -> EgressDecisionSubjectV1 {
        let disclosure = DisclosureBinding::new(
            &lineage(),
            &resource("public-api"),
            digest(0x90),
            digest(0x91),
            None,
            "network-egress/v1",
        )
        .expect("disclosure");
        EgressDecisionSubjectV1::new(
            &disclosure,
            &interaction_intent(),
            digest(0xa0),
            Some(digest(0xa1)),
            vec![digest(0xb0), digest(0xb1)],
        )
        .expect("egress subject")
    }

    fn policy_subject() -> PolicyDecisionSubjectV1 {
        PolicyDecisionSubjectV1::egress(egress_subject().id())
    }

    fn bundle() -> PolicyBundleIdentityV1 {
        PolicyBundleIdentityV1::new("org-egress", "2026-09-r1", digest(0xc0)).expect("bundle")
    }

    fn engine() -> PolicyEngineIdentityV1 {
        PolicyEngineIdentityV1::new("native-rust", "deterministic-v1", digest(0xc1))
            .expect("engine")
    }

    fn rules() -> Vec<PolicyRuleId> {
        vec![
            PolicyRuleId::new("egress.external-content").expect("rule"),
            PolicyRuleId::new("session.bound").expect("rule"),
        ]
    }

    fn receipt() -> PolicyDecisionReceiptV1 {
        PolicyDecisionReceiptV1::new(
            &policy_subject(),
            &bundle(),
            &engine(),
            digest(0xd0),
            Some(digest(0xd1)),
            PolicyDecisionOutcome::AllowCandidate,
            rules(),
        )
        .expect("receipt")
    }

    #[test]
    fn canonical_vectors_are_stable() {
        assert_eq!(
            policy_subject().digest().to_hex(),
            "bcca1cb217a2aac6bfa7f101a382d68d3e570bc616575f3d13fb356657c4b55d"
        );
        assert_eq!(
            bundle().digest().to_hex(),
            "966bb21135df770ba2dcd08a7843c1453274bb5ae1785cf3d566b5a4343cc0a4"
        );
        assert_eq!(
            engine().digest().to_hex(),
            "2fab0d00a3c534b48c158839afae0fcd50cc837cfe73b1471cb87c2a694e4e05"
        );
        assert_eq!(
            receipt().id().digest().to_hex(),
            "14199056568afdc56411b489ea6f736e367b4e0da5b33c217721d2841a1e61a6"
        );
    }

    #[test]
    fn subject_kind_is_committed_even_for_same_raw_digest() {
        let raw = digest(0xee);
        let interaction = PolicyDecisionSubjectV1 {
            kind: PolicySubjectKind::Interaction,
            subject_digest: raw,
        };
        let egress = PolicyDecisionSubjectV1 {
            kind: PolicySubjectKind::Egress,
            subject_digest: raw,
        };
        assert_ne!(interaction.digest(), egress.digest());
    }

    #[test]
    fn bundle_or_engine_change_invalidates_receipt() {
        let subject = policy_subject();
        let baseline = receipt();
        let changed_bundle = PolicyBundleIdentityV1::new("org-egress", "2026-09-r2", digest(0xc0))
            .expect("changed bundle");
        let bundle_receipt = PolicyDecisionReceiptV1::new(
            &subject,
            &changed_bundle,
            &engine(),
            digest(0xd0),
            Some(digest(0xd1)),
            PolicyDecisionOutcome::AllowCandidate,
            rules(),
        )
        .expect("bundle receipt");
        let changed_engine = PolicyEngineIdentityV1::new(
            "native-rust",
            "deterministic-v1",
            digest(0xc2),
        )
        .expect("changed engine");
        let engine_receipt = PolicyDecisionReceiptV1::new(
            &subject,
            &bundle(),
            &changed_engine,
            digest(0xd0),
            Some(digest(0xd1)),
            PolicyDecisionOutcome::AllowCandidate,
            rules(),
        )
        .expect("engine receipt");

        assert_ne!(baseline.id(), bundle_receipt.id());
        assert_ne!(baseline.id(), engine_receipt.id());
    }

    #[test]
    fn evaluation_context_or_outcome_change_invalidates_receipt() {
        let subject = policy_subject();
        let baseline = receipt();
        let context_changed = PolicyDecisionReceiptV1::new(
            &subject,
            &bundle(),
            &engine(),
            digest(0xd2),
            Some(digest(0xd1)),
            PolicyDecisionOutcome::AllowCandidate,
            rules(),
        )
        .expect("context receipt");
        let denied = PolicyDecisionReceiptV1::new(
            &subject,
            &bundle(),
            &engine(),
            digest(0xd0),
            Some(digest(0xd1)),
            PolicyDecisionOutcome::Deny,
            rules(),
        )
        .expect("denied receipt");

        assert_ne!(baseline.id(), context_changed.id());
        assert_ne!(baseline.id(), denied.id());
    }

    #[test]
    fn rule_order_is_non_semantic_and_duplicates_fail_closed() {
        let subject = policy_subject();
        let left = PolicyDecisionReceiptV1::new(
            &subject,
            &bundle(),
            &engine(),
            digest(0xd0),
            None,
            PolicyDecisionOutcome::NeedsConfirmation,
            vec![
                PolicyRuleId::new("z.rule").unwrap(),
                PolicyRuleId::new("a.rule").unwrap(),
            ],
        )
        .expect("left");
        let right = PolicyDecisionReceiptV1::new(
            &subject,
            &bundle(),
            &engine(),
            digest(0xd0),
            None,
            PolicyDecisionOutcome::NeedsConfirmation,
            vec![
                PolicyRuleId::new("a.rule").unwrap(),
                PolicyRuleId::new("z.rule").unwrap(),
            ],
        )
        .expect("right");
        assert_eq!(left.id(), right.id());

        let duplicate = PolicyDecisionReceiptV1::new(
            &subject,
            &bundle(),
            &engine(),
            digest(0xd0),
            None,
            PolicyDecisionOutcome::Deny,
            vec![
                PolicyRuleId::new("same.rule").unwrap(),
                PolicyRuleId::new("same.rule").unwrap(),
            ],
        )
        .expect_err("duplicate rule must fail");
        assert_eq!(duplicate, PolicyEvidenceError::DuplicateRuleId);
    }

    #[test]
    fn zero_identity_and_currentness_sentinels_fail_closed() {
        assert!(matches!(
            PolicyBundleIdentityV1::new("profile", "v1", Digest32::new([0; 32])),
            Err(PolicyEvidenceError::ZeroDigest("policy bundle content digest"))
        ));
        assert!(matches!(
            PolicyEngineIdentityV1::new("engine", "v1", Digest32::new([0; 32])),
            Err(PolicyEvidenceError::ZeroDigest(
                "policy engine implementation digest"
            ))
        ));
        assert!(matches!(
            PolicyDecisionReceiptV1::new(
                &policy_subject(),
                &bundle(),
                &engine(),
                Digest32::new([0; 32]),
                None,
                PolicyDecisionOutcome::Deny,
                Vec::new(),
            ),
            Err(PolicyEvidenceError::ZeroDigest("policy evaluation context"))
        ));
    }

    #[test]
    fn ordinary_interaction_subject_is_supported_without_egress_reinterpretation() {
        let interaction = PolicyDecisionSubjectV1::interaction(&interaction_intent());
        assert_eq!(interaction.kind(), PolicySubjectKind::Interaction);
        assert_eq!(
            interaction.digest().to_hex(),
            "aea9e0b3972a64abd01b3713abe7dc8a498f79afa8b8f97b7bebabd6e8132c56"
        );
        assert_ne!(interaction.digest(), policy_subject().digest());
    }
}

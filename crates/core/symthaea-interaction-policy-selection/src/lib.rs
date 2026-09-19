// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Pure deterministic policy-selection matching for the Interaction Fabric.
//!
//! This crate compares an exact policy decision receipt against an exact
//! caller-supplied selection snapshot and time observation. It deliberately
//! does **not** authenticate either input and therefore does not establish
//! verified policy currentness.
//!
//! ```text
//! PolicySelectionMatchCandidateV1
//!     != authenticated selection source
//!     != trusted time
//!     != authenticated PDP/workload
//!     != verified current policy
//!     != current authority
//!     != DispatchPermit
//! ```

#![deny(unsafe_code)]

use sha2::{Digest as ShaDigest, Sha256};
use std::collections::BTreeSet;
use std::error::Error;
use std::fmt;
use symthaea_interaction_core::Digest32;
use symthaea_interaction_policy::{
    PolicyBundleIdentityV1, PolicyDecisionReceiptId, PolicyDecisionReceiptV1,
    PolicyEngineIdentityV1,
};

pub const POLICY_SELECTION_SCHEMA_VERSION: u16 = 1;

const REVISION_DOMAIN: &[u8] = b"symthaea.interaction.policy.selection.revision.v1\0";
const SNAPSHOT_DOMAIN: &[u8] = b"symthaea.interaction.policy.selection.snapshot.v1\0";
const TIME_DOMAIN: &[u8] = b"symthaea.interaction.policy.time.observation.v1\0";
const MATCH_DOMAIN: &[u8] = b"symthaea.interaction.policy.selection.match.v1\0";

const MAX_TEXT_LEN: usize = 256;
const MAX_REVISIONS: usize = 32;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PolicySelectionError {
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
    ZeroEpoch,
    EmptyRevisions,
    TooManyRevisions(usize),
    AmbiguousRevisionPair,
    InvalidValidityWindow {
        valid_from_unix_ms: u64,
        valid_until_unix_ms: u64,
    },
}

impl fmt::Display for PolicySelectionError {
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
            Self::ZeroEpoch => write!(formatter, "policy selection epoch must be non-zero"),
            Self::EmptyRevisions => write!(formatter, "policy selection requires at least one accepted revision"),
            Self::TooManyRevisions(count) => write!(
                formatter,
                "policy selection contains {count} revisions; maximum is {MAX_REVISIONS}"
            ),
            Self::AmbiguousRevisionPair => write!(
                formatter,
                "policy selection repeats one bundle+engine pair with ambiguous producer requirements"
            ),
            Self::InvalidValidityWindow {
                valid_from_unix_ms,
                valid_until_unix_ms,
            } => write!(
                formatter,
                "policy selection validity window is reversed: {valid_from_unix_ms} > {valid_until_unix_ms}"
            ),
        }
    }
}

impl Error for PolicySelectionError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PolicySelectionAssessmentError {
    SelectionSuspended,
    SelectionRevoked,
    ObservationBeforeWindow {
        observed_unix_ms: u64,
        valid_from_unix_ms: u64,
    },
    ObservationAfterWindow {
        observed_unix_ms: u64,
        valid_until_unix_ms: u64,
    },
    ReceiptCurrentnessReferenceRequired,
    BundleNotAccepted,
    EngineNotAcceptedForBundle,
}

impl fmt::Display for PolicySelectionAssessmentError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SelectionSuspended => write!(formatter, "policy selection is suspended"),
            Self::SelectionRevoked => write!(formatter, "policy selection is revoked"),
            Self::ObservationBeforeWindow {
                observed_unix_ms,
                valid_from_unix_ms,
            } => write!(
                formatter,
                "time observation {observed_unix_ms} precedes selection validity {valid_from_unix_ms}"
            ),
            Self::ObservationAfterWindow {
                observed_unix_ms,
                valid_until_unix_ms,
            } => write!(
                formatter,
                "time observation {observed_unix_ms} exceeds selection validity {valid_until_unix_ms}"
            ),
            Self::ReceiptCurrentnessReferenceRequired => write!(
                formatter,
                "selection profile requires the policy receipt to carry a currentness-evidence reference"
            ),
            Self::BundleNotAccepted => write!(formatter, "policy bundle is not accepted by this selection"),
            Self::EngineNotAcceptedForBundle => write!(
                formatter,
                "policy engine is not accepted for the receipt's policy bundle"
            ),
        }
    }
}

impl Error for PolicySelectionAssessmentError {}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct CanonicalText(String);

impl CanonicalText {
    fn new(field: &'static str, value: &str) -> Result<Self, PolicySelectionError> {
        if value.is_empty() {
            return Err(PolicySelectionError::EmptyText(field));
        }
        if value.len() > MAX_TEXT_LEN {
            return Err(PolicySelectionError::TextTooLong {
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
            return Err(PolicySelectionError::NonCanonicalText { field, byte });
        }
        Ok(Self(value.to_owned()))
    }

    fn as_str(&self) -> &str {
        &self.0
    }
}

fn reject_zero(field: &'static str, value: Digest32) -> Result<(), PolicySelectionError> {
    if value.as_bytes() == &[0; 32] {
        Err(PolicySelectionError::ZeroDigest(field))
    } else {
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PolicySelectionStatus {
    Active,
    Suspended,
    Revoked,
}

impl PolicySelectionStatus {
    const fn code(self) -> u16 {
        match self {
            Self::Active => 0,
            Self::Suspended => 1,
            Self::Revoked => 2,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CurrentnessReferenceRequirement {
    Optional,
    Required,
}

impl CurrentnessReferenceRequirement {
    const fn code(self) -> u16 {
        match self {
            Self::Optional => 0,
            Self::Required => 1,
        }
    }
}

/// One exact accepted policy-bundle/engine pair plus the producer-authentication
/// requirement that a later authenticated-currentness verifier must satisfy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AcceptedPolicyRevisionV1 {
    bundle: Digest32,
    engine: Digest32,
    producer_requirement: Digest32,
}

impl AcceptedPolicyRevisionV1 {
    pub fn new(
        bundle: &PolicyBundleIdentityV1,
        engine: &PolicyEngineIdentityV1,
        producer_requirement: Digest32,
    ) -> Result<Self, PolicySelectionError> {
        reject_zero("policy producer requirement", producer_requirement)?;
        Ok(Self {
            bundle: bundle.digest(),
            engine: engine.digest(),
            producer_requirement,
        })
    }

    pub const fn bundle_digest(&self) -> Digest32 {
        self.bundle
    }

    pub const fn engine_digest(&self) -> Digest32 {
        self.engine
    }

    pub const fn producer_requirement(&self) -> Digest32 {
        self.producer_requirement
    }

    #[must_use]
    pub fn digest(&self) -> Digest32 {
        let mut transcript = Transcript::new(REVISION_DOMAIN);
        transcript.u16(POLICY_SELECTION_SCHEMA_VERSION);
        transcript.digest(self.bundle);
        transcript.digest(self.engine);
        transcript.digest(self.producer_requirement);
        transcript.finish()
    }
}

/// Exact caller-supplied policy-selection state.
///
/// This value is ordinary data. Its epoch and source commitment do not prove
/// authenticity, freshness, consensus, or anti-rollback currentness.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PolicySelectionSnapshotV1 {
    profile: CanonicalText,
    epoch: u64,
    status: PolicySelectionStatus,
    currentness_reference: CurrentnessReferenceRequirement,
    valid_from_unix_ms: u64,
    valid_until_unix_ms: u64,
    source_commitment: Digest32,
    revisions: Vec<AcceptedPolicyRevisionV1>,
}

impl PolicySelectionSnapshotV1 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        profile: &str,
        epoch: u64,
        status: PolicySelectionStatus,
        currentness_reference: CurrentnessReferenceRequirement,
        valid_from_unix_ms: u64,
        valid_until_unix_ms: u64,
        source_commitment: Digest32,
        mut revisions: Vec<AcceptedPolicyRevisionV1>,
    ) -> Result<Self, PolicySelectionError> {
        if epoch == 0 {
            return Err(PolicySelectionError::ZeroEpoch);
        }
        if valid_from_unix_ms > valid_until_unix_ms {
            return Err(PolicySelectionError::InvalidValidityWindow {
                valid_from_unix_ms,
                valid_until_unix_ms,
            });
        }
        reject_zero("policy selection source commitment", source_commitment)?;
        if revisions.is_empty() {
            return Err(PolicySelectionError::EmptyRevisions);
        }
        if revisions.len() > MAX_REVISIONS {
            return Err(PolicySelectionError::TooManyRevisions(revisions.len()));
        }

        let mut exact_pairs = BTreeSet::new();
        for revision in &revisions {
            if !exact_pairs.insert((revision.bundle, revision.engine)) {
                return Err(PolicySelectionError::AmbiguousRevisionPair);
            }
        }
        revisions.sort_by_key(AcceptedPolicyRevisionV1::digest);

        Ok(Self {
            profile: CanonicalText::new("policy selection profile", profile)?,
            epoch,
            status,
            currentness_reference,
            valid_from_unix_ms,
            valid_until_unix_ms,
            source_commitment,
            revisions,
        })
    }

    pub fn profile(&self) -> &str {
        self.profile.as_str()
    }

    pub const fn epoch(&self) -> u64 {
        self.epoch
    }

    pub const fn status(&self) -> PolicySelectionStatus {
        self.status
    }

    pub const fn source_commitment(&self) -> Digest32 {
        self.source_commitment
    }

    pub fn revisions(&self) -> &[AcceptedPolicyRevisionV1] {
        &self.revisions
    }

    #[must_use]
    pub fn digest(&self) -> Digest32 {
        let mut transcript = Transcript::new(SNAPSHOT_DOMAIN);
        transcript.u16(POLICY_SELECTION_SCHEMA_VERSION);
        transcript.text(self.profile.as_str());
        transcript.u64(self.epoch);
        transcript.u16(self.status.code());
        transcript.u16(self.currentness_reference.code());
        transcript.u64(self.valid_from_unix_ms);
        transcript.u64(self.valid_until_unix_ms);
        transcript.digest(self.source_commitment);
        transcript.digest_set_from_revisions(&self.revisions);
        transcript.finish()
    }

    pub fn assess(
        &self,
        receipt: &PolicyDecisionReceiptV1,
        time: &PolicyTimeObservationV1,
    ) -> Result<PolicySelectionMatchCandidateV1, PolicySelectionAssessmentError> {
        match self.status {
            PolicySelectionStatus::Active => {}
            PolicySelectionStatus::Suspended => {
                return Err(PolicySelectionAssessmentError::SelectionSuspended);
            }
            PolicySelectionStatus::Revoked => {
                return Err(PolicySelectionAssessmentError::SelectionRevoked);
            }
        }

        if time.unix_ms < self.valid_from_unix_ms {
            return Err(PolicySelectionAssessmentError::ObservationBeforeWindow {
                observed_unix_ms: time.unix_ms,
                valid_from_unix_ms: self.valid_from_unix_ms,
            });
        }
        if time.unix_ms > self.valid_until_unix_ms {
            return Err(PolicySelectionAssessmentError::ObservationAfterWindow {
                observed_unix_ms: time.unix_ms,
                valid_until_unix_ms: self.valid_until_unix_ms,
            });
        }
        if self.currentness_reference == CurrentnessReferenceRequirement::Required
            && receipt.currentness_evidence().is_none()
        {
            return Err(PolicySelectionAssessmentError::ReceiptCurrentnessReferenceRequired);
        }

        let bundle = receipt.bundle_digest();
        let engine = receipt.engine_digest();
        let mut saw_bundle = false;
        for revision in &self.revisions {
            if revision.bundle == bundle {
                saw_bundle = true;
                if revision.engine == engine {
                    return Ok(PolicySelectionMatchCandidateV1 {
                        receipt: receipt.id(),
                        snapshot: self.digest(),
                        revision: revision.digest(),
                        time_observation: time.digest(),
                        producer_requirement: revision.producer_requirement,
                    });
                }
            }
        }

        if saw_bundle {
            Err(PolicySelectionAssessmentError::EngineNotAcceptedForBundle)
        } else {
            Err(PolicySelectionAssessmentError::BundleNotAccepted)
        }
    }
}

/// Caller-supplied time observation used only for deterministic window matching.
///
/// This type deliberately does not claim that the clock/evidence source is
/// trusted, monotonic, authenticated, or rollback-resistant.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PolicyTimeObservationV1 {
    unix_ms: u64,
    source_profile: CanonicalText,
    evidence_commitment: Digest32,
}

impl PolicyTimeObservationV1 {
    pub fn new(
        unix_ms: u64,
        source_profile: &str,
        evidence_commitment: Digest32,
    ) -> Result<Self, PolicySelectionError> {
        reject_zero("policy time evidence commitment", evidence_commitment)?;
        Ok(Self {
            unix_ms,
            source_profile: CanonicalText::new("policy time source profile", source_profile)?,
            evidence_commitment,
        })
    }

    pub const fn unix_ms(&self) -> u64 {
        self.unix_ms
    }

    pub fn source_profile(&self) -> &str {
        self.source_profile.as_str()
    }

    #[must_use]
    pub fn digest(&self) -> Digest32 {
        let mut transcript = Transcript::new(TIME_DOMAIN);
        transcript.u16(POLICY_SELECTION_SCHEMA_VERSION);
        transcript.u64(self.unix_ms);
        transcript.text(self.source_profile.as_str());
        transcript.digest(self.evidence_commitment);
        transcript.finish()
    }
}

/// Deterministic positive **match candidate** between a receipt and ordinary
/// selection/time inputs.
///
/// There is deliberately no public constructor. `PolicySelectionSnapshotV1::assess`
/// is the only production creation path, but the resulting value remains
/// non-authoritative because its inputs are not authenticated by this crate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PolicySelectionMatchCandidateV1 {
    receipt: PolicyDecisionReceiptId,
    snapshot: Digest32,
    revision: Digest32,
    time_observation: Digest32,
    producer_requirement: Digest32,
}

impl PolicySelectionMatchCandidateV1 {
    pub const fn receipt_id(&self) -> PolicyDecisionReceiptId {
        self.receipt
    }

    pub const fn snapshot_digest(&self) -> Digest32 {
        self.snapshot
    }

    pub const fn revision_digest(&self) -> Digest32 {
        self.revision
    }

    pub const fn time_observation_digest(&self) -> Digest32 {
        self.time_observation
    }

    pub const fn producer_requirement(&self) -> Digest32 {
        self.producer_requirement
    }

    #[must_use]
    pub fn digest(&self) -> Digest32 {
        let mut transcript = Transcript::new(MATCH_DOMAIN);
        transcript.u16(POLICY_SELECTION_SCHEMA_VERSION);
        transcript.digest(self.receipt.digest());
        transcript.digest(self.snapshot);
        transcript.digest(self.revision);
        transcript.digest(self.time_observation);
        transcript.finish()
    }
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

    fn u16(&mut self, value: u16) {
        self.hasher.update(value.to_be_bytes());
    }

    fn u32(&mut self, value: u32) {
        self.hasher.update(value.to_be_bytes());
    }

    fn u64(&mut self, value: u64) {
        self.hasher.update(value.to_be_bytes());
    }

    fn text(&mut self, value: &str) {
        self.u32(value.len() as u32);
        self.hasher.update(value.as_bytes());
    }

    fn digest(&mut self, value: Digest32) {
        self.hasher.update(value.as_bytes());
    }

    fn digest_set_from_revisions(&mut self, revisions: &[AcceptedPolicyRevisionV1]) {
        self.u32(revisions.len() as u32);
        for revision in revisions {
            self.digest(revision.digest());
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
        IdentityOrdering, InteractionIntent, NamespaceId, OperationRef, ResourceRef,
    };
    use symthaea_interaction_policy::{
        PolicyDecisionOutcome, PolicyDecisionSubjectV1, PolicyRuleId,
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

    fn intent() -> InteractionIntent {
        let namespace = NamespaceId::new("web/http").expect("namespace");
        InteractionIntent::new(
            ConnectorIdentity::new(
                ConnectorId::new(namespace.clone(), "rest").expect("connector id"),
                "https-json/v1",
                Some(digest(0x60)),
            )
            .expect("connector"),
            None,
            resource("public-api"),
            OperationRef::new(namespace, "post", Some("json/v1")).expect("operation"),
            digest(0x90),
            EffectClass::Publish,
            IdempotencyClass::IdempotencyKeyed,
        )
    }

    fn bundle(version: &str) -> PolicyBundleIdentityV1 {
        PolicyBundleIdentityV1::new("org-egress", version, digest(0xc0)).expect("bundle")
    }

    fn engine(implementation: u8) -> PolicyEngineIdentityV1 {
        PolicyEngineIdentityV1::new("native-rust", "deterministic-v1", digest(implementation))
            .expect("engine")
    }

    fn receipt(
        bundle: &PolicyBundleIdentityV1,
        engine: &PolicyEngineIdentityV1,
        with_currentness: bool,
    ) -> PolicyDecisionReceiptV1 {
        PolicyDecisionReceiptV1::new(
            &PolicyDecisionSubjectV1::interaction(&intent()),
            bundle,
            engine,
            digest(0xd0),
            with_currentness.then(|| digest(0xd1)),
            PolicyDecisionOutcome::AllowCandidate,
            vec![
                PolicyRuleId::new("egress.external-content").expect("rule"),
                PolicyRuleId::new("session.bound").expect("rule"),
            ],
        )
        .expect("receipt")
    }

    fn revisions() -> Vec<AcceptedPolicyRevisionV1> {
        vec![
            AcceptedPolicyRevisionV1::new(&bundle("2026-09-r2"), &engine(0xc1), digest(0xe1))
                .expect("r2"),
            AcceptedPolicyRevisionV1::new(&bundle("2026-09-r1"), &engine(0xc1), digest(0xe0))
                .expect("r1"),
        ]
    }

    fn snapshot(status: PolicySelectionStatus) -> PolicySelectionSnapshotV1 {
        PolicySelectionSnapshotV1::new(
            "org-egress-selection/v1",
            7,
            status,
            CurrentnessReferenceRequirement::Required,
            1_800_000_000_000,
            1_800_086_400_000,
            digest(0xf0),
            revisions(),
        )
        .expect("snapshot")
    }

    fn time(unix_ms: u64, evidence: u8) -> PolicyTimeObservationV1 {
        PolicyTimeObservationV1::new(unix_ms, "verified-wall-clock/v1", digest(evidence))
            .expect("time")
    }

    #[test]
    fn canonical_vectors_are_stable() {
        let r1 = AcceptedPolicyRevisionV1::new(
            &bundle("2026-09-r1"),
            &engine(0xc1),
            digest(0xe0),
        )
        .expect("r1");
        assert_eq!(
            r1.digest().to_hex(),
            "73cedadbc7e38e4dd32939f3fd65146c250cd54979ad4cd7ba93490ad528b894"
        );
        assert_eq!(
            snapshot(PolicySelectionStatus::Active).digest().to_hex(),
            "5215511ed1d0c7cb38dca1fd6e17db05305347f7fe12a58c3c6ba434ad3be1d1"
        );
        let observed = time(1_800_000_123_456, 0xf1);
        assert_eq!(
            observed.digest().to_hex(),
            "2b62738b10a97224d2c65034a849c9634b3534f87ad27970735dd1a5f8a61c31"
        );
        let decision = receipt(&bundle("2026-09-r1"), &engine(0xc1), true);
        assert_eq!(
            decision.id().digest().to_hex(),
            "0b4cf8ab938d0dd88ce72562afb8a3e2d14b576f4cadfe5ff5ef285b34a50711"
        );
        let matched = snapshot(PolicySelectionStatus::Active)
            .assess(&decision, &observed)
            .expect("match");
        assert_eq!(
            matched.digest().to_hex(),
            "1d444368c7d1da31800a8b7e09045b87694ccc5c5382c8ed0313c9eb89f64b58"
        );
        assert_eq!(matched.producer_requirement(), digest(0xe0));
    }

    #[test]
    fn accepted_revision_order_is_non_semantic() {
        let mut reversed = revisions();
        reversed.reverse();
        let a = snapshot(PolicySelectionStatus::Active);
        let b = PolicySelectionSnapshotV1::new(
            "org-egress-selection/v1",
            7,
            PolicySelectionStatus::Active,
            CurrentnessReferenceRequirement::Required,
            1_800_000_000_000,
            1_800_086_400_000,
            digest(0xf0),
            reversed,
        )
        .expect("snapshot");
        assert_eq!(a.digest(), b.digest());
    }

    #[test]
    fn ambiguous_bundle_engine_pair_fails_closed() {
        let error = PolicySelectionSnapshotV1::new(
            "org-egress-selection/v1",
            7,
            PolicySelectionStatus::Active,
            CurrentnessReferenceRequirement::Optional,
            1,
            2,
            digest(0xf0),
            vec![
                AcceptedPolicyRevisionV1::new(&bundle("2026-09-r1"), &engine(0xc1), digest(0xe0))
                    .unwrap(),
                AcceptedPolicyRevisionV1::new(&bundle("2026-09-r1"), &engine(0xc1), digest(0xe1))
                    .unwrap(),
            ],
        )
        .expect_err("ambiguous producer requirement must fail");
        assert_eq!(error, PolicySelectionError::AmbiguousRevisionPair);
    }

    #[test]
    fn suspended_and_revoked_never_match() {
        let decision = receipt(&bundle("2026-09-r1"), &engine(0xc1), true);
        let observed = time(1_800_000_123_456, 0xf1);
        assert_eq!(
            snapshot(PolicySelectionStatus::Suspended).assess(&decision, &observed),
            Err(PolicySelectionAssessmentError::SelectionSuspended)
        );
        assert_eq!(
            snapshot(PolicySelectionStatus::Revoked).assess(&decision, &observed),
            Err(PolicySelectionAssessmentError::SelectionRevoked)
        );
    }

    #[test]
    fn validity_window_is_fail_closed() {
        let decision = receipt(&bundle("2026-09-r1"), &engine(0xc1), true);
        assert!(matches!(
            snapshot(PolicySelectionStatus::Active)
                .assess(&decision, &time(1_799_999_999_999, 0xf1)),
            Err(PolicySelectionAssessmentError::ObservationBeforeWindow { .. })
        ));
        assert!(matches!(
            snapshot(PolicySelectionStatus::Active)
                .assess(&decision, &time(1_800_086_400_001, 0xf1)),
            Err(PolicySelectionAssessmentError::ObservationAfterWindow { .. })
        ));
    }

    #[test]
    fn missing_required_receipt_currentness_reference_rejects() {
        let decision = receipt(&bundle("2026-09-r1"), &engine(0xc1), false);
        let error = snapshot(PolicySelectionStatus::Active)
            .assess(&decision, &time(1_800_000_123_456, 0xf1))
            .expect_err("currentness reference required");
        assert_eq!(
            error,
            PolicySelectionAssessmentError::ReceiptCurrentnessReferenceRequired
        );
    }

    #[test]
    fn bundle_and_engine_mismatches_are_distinct() {
        let observed = time(1_800_000_123_456, 0xf1);
        let unknown_bundle = receipt(&bundle("2026-10-r1"), &engine(0xc1), true);
        assert_eq!(
            snapshot(PolicySelectionStatus::Active).assess(&unknown_bundle, &observed),
            Err(PolicySelectionAssessmentError::BundleNotAccepted)
        );
        let wrong_engine = receipt(&bundle("2026-09-r1"), &engine(0xc2), true);
        assert_eq!(
            snapshot(PolicySelectionStatus::Active).assess(&wrong_engine, &observed),
            Err(PolicySelectionAssessmentError::EngineNotAcceptedForBundle)
        );
    }

    #[test]
    fn snapshot_or_time_mutation_changes_match_identity() {
        let decision = receipt(&bundle("2026-09-r1"), &engine(0xc1), true);
        let base_snapshot = snapshot(PolicySelectionStatus::Active);
        let base_time = time(1_800_000_123_456, 0xf1);
        let baseline = base_snapshot.assess(&decision, &base_time).unwrap();

        let changed_snapshot = PolicySelectionSnapshotV1::new(
            "org-egress-selection/v1",
            8,
            PolicySelectionStatus::Active,
            CurrentnessReferenceRequirement::Required,
            1_800_000_000_000,
            1_800_086_400_000,
            digest(0xf0),
            revisions(),
        )
        .unwrap();
        let changed_time = time(1_800_000_123_456, 0xf2);
        assert_ne!(
            baseline.digest(),
            changed_snapshot.assess(&decision, &base_time).unwrap().digest()
        );
        assert_ne!(
            baseline.digest(),
            base_snapshot.assess(&decision, &changed_time).unwrap().digest()
        );
    }

    #[test]
    fn zero_and_invalid_snapshot_inputs_fail_closed() {
        assert_eq!(
            AcceptedPolicyRevisionV1::new(
                &bundle("2026-09-r1"),
                &engine(0xc1),
                Digest32::new([0; 32]),
            )
            .expect_err("zero producer requirement"),
            PolicySelectionError::ZeroDigest("policy producer requirement")
        );
        assert_eq!(
            PolicySelectionSnapshotV1::new(
                "selection/v1",
                0,
                PolicySelectionStatus::Active,
                CurrentnessReferenceRequirement::Optional,
                1,
                2,
                digest(0xf0),
                revisions(),
            )
            .expect_err("zero epoch"),
            PolicySelectionError::ZeroEpoch
        );
        assert!(matches!(
            PolicySelectionSnapshotV1::new(
                "selection/v1",
                1,
                PolicySelectionStatus::Active,
                CurrentnessReferenceRequirement::Optional,
                3,
                2,
                digest(0xf0),
                revisions(),
            ),
            Err(PolicySelectionError::InvalidValidityWindow { .. })
        ));
    }
}

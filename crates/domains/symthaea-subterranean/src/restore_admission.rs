// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Two-phase admission for operational checkpoint restore.
//!
//! Restore is not deserialization. A checkpoint is first normalized into its
//! exact portable representation and captured in an opaque, validated owner-local
//! source capsule, then compared with the current live authority/evidence/physical
//! context without mutating runtime state. The resulting prepared plan is affine,
//! bound to that exact source, live owner, and context, and can be committed only
//! while every generation fence still matches.
//!
//! This module intentionally does **not** mutate `SubterraneanEmbodiment` yet.
//! It is the pure transaction primitive that later owner-bound execution uses.

use super::restore_semantics::{OPERATIONAL_RESTORE_CONTRACTS, RestoreDomain};
use super::{OperationalCheckpointError, SubterraneanOperationalCheckpoint};

const RESTORE_SOURCE_COMMITMENT_DOMAIN_V1: &[u8] =
    b"symthaea-subterranean:operational-restore-source:v1\0";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct RestoreDigest([u8; 32]);

impl RestoreDigest {
    pub(super) const fn new(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    pub(super) const fn bytes(self) -> [u8; 32] {
        self.0
    }

    const fn is_valid(self) -> bool {
        let mut index = 0;
        while index < self.0.len() {
            if self.0[index] != 0 {
                return true;
            }
            index += 1;
        }
        false
    }
}

/// Stable identity of the live restore owner/instance.
///
/// This is deliberately distinct from `live_snapshot_digest`: owner identity
/// answers *which live authority owner is being mutated*, while the snapshot
/// digest and generation counters answer *which state of that owner was
/// reviewed*. A later embodiment-owned integration is responsible for deriving
/// this identity from trusted boot/instance state rather than pointer addresses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct RestoreOwnerIdentity([u8; 32]);

impl RestoreOwnerIdentity {
    pub(super) const fn new(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    pub(super) const fn bytes(self) -> [u8; 32] {
        self.0
    }

    const fn is_valid(self) -> bool {
        let mut index = 0;
        while index < self.0.len() {
            if self.0[index] != 0 {
                return true;
            }
            index += 1;
        }
        false
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum RestoreSourceError {
    InvalidCheckpoint(OperationalCheckpointError),
    Encoding,
    NonCanonicalRoundTrip,
    SourceTooLarge,
}

impl From<OperationalCheckpointError> for RestoreSourceError {
    fn from(value: OperationalCheckpointError) -> Self {
        Self::InvalidCheckpoint(value)
    }
}

/// Exact portable checkpoint source owned by one restore transaction lineage.
///
/// Deliberately not `Clone`, `Copy`, `Serialize` or `Deserialize`. Raw
/// checkpoint bytes/objects remain freely portable data, but they do not become
/// this owner-local source capability without normalization, pure structural
/// validation, and a locally derived commitment.
///
/// Normalization is security-relevant: checkpoint domain types intentionally use
/// `serde(skip)` for host-local/ephemeral state. The capsule therefore stores the
/// object obtained by decoding the exact portable serialization, not the original
/// in-memory object. Executors can never observe state the commitment omitted.
///
/// The V1 commitment uses the crate's deterministic serde representation plus
/// explicit domain separation. It is an internal transaction identity, not a
/// cross-language wire-canonicalization claim. If the encoding contract changes,
/// the commitment domain/version must change with it.
pub(super) struct OperationalRestoreSource {
    checkpoint: SubterraneanOperationalCheckpoint,
    digest: RestoreDigest,
}

impl OperationalRestoreSource {
    pub(super) fn capture(
        checkpoint: SubterraneanOperationalCheckpoint,
    ) -> Result<Self, RestoreSourceError> {
        let encoded = serde_json::to_vec(&checkpoint).map_err(|_| RestoreSourceError::Encoding)?;
        let encoded_len =
            u64::try_from(encoded.len()).map_err(|_| RestoreSourceError::SourceTooLarge)?;
        let normalized: SubterraneanOperationalCheckpoint =
            serde_json::from_slice(&encoded).map_err(|_| RestoreSourceError::Encoding)?;
        normalized.validate_source()?;
        let canonical =
            serde_json::to_vec(&normalized).map_err(|_| RestoreSourceError::Encoding)?;
        if canonical != encoded {
            return Err(RestoreSourceError::NonCanonicalRoundTrip);
        }

        let mut hasher = blake3::Hasher::new();
        hasher.update(RESTORE_SOURCE_COMMITMENT_DOMAIN_V1);
        hasher.update(&normalized.schema_version.to_le_bytes());
        hasher.update(&encoded_len.to_le_bytes());
        hasher.update(&canonical);
        let digest = RestoreDigest::new(*hasher.finalize().as_bytes());
        debug_assert!(digest.is_valid());
        Ok(Self {
            checkpoint: normalized,
            digest,
        })
    }

    pub(super) const fn digest(&self) -> RestoreDigest {
        self.digest
    }

    pub(super) const fn checkpoint(&self) -> &SubterraneanOperationalCheckpoint {
        &self.checkpoint
    }
}

/// Exact live owner and context against which a restore decision was prepared.
///
/// `owner_identity` is an equality fence over *which owner instance* is in
/// scope. The generation counters and snapshot digest are equality fences over
/// *which state of that owner* was reviewed. None of these values is a clock or
/// authority by itself. The trusted owner is responsible for advancing the
/// generations whenever corresponding live truth changes in a restore-relevant
/// way.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct RestoreGenerationFence {
    owner_identity: RestoreOwnerIdentity,
    boot_epoch: u64,
    control_plane_generation: u64,
    authority_generation: u64,
    evidence_generation: u64,
    physical_state_generation: u64,
    live_snapshot_digest: RestoreDigest,
}

impl RestoreGenerationFence {
    /// Production-shaped constructor. The caller must provide an identity
    /// derived by the trusted live owner; an unbound/sentinel owner is not
    /// accepted by preparation.
    #[allow(clippy::too_many_arguments)]
    pub(super) const fn new_bound(
        owner_identity: RestoreOwnerIdentity,
        boot_epoch: u64,
        control_plane_generation: u64,
        authority_generation: u64,
        evidence_generation: u64,
        physical_state_generation: u64,
        live_snapshot_digest: RestoreDigest,
    ) -> Self {
        Self {
            owner_identity,
            boot_epoch,
            control_plane_generation,
            authority_generation,
            evidence_generation,
            physical_state_generation,
            live_snapshot_digest,
        }
    }

    /// Compatibility constructor for crate tests written before RA-34 owner
    /// binding. It is deliberately unavailable in production builds.
    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    pub(super) const fn new(
        boot_epoch: u64,
        control_plane_generation: u64,
        authority_generation: u64,
        evidence_generation: u64,
        physical_state_generation: u64,
        live_snapshot_digest: RestoreDigest,
    ) -> Self {
        Self::new_bound(
            RestoreOwnerIdentity::new([0xA5; 32]),
            boot_epoch,
            control_plane_generation,
            authority_generation,
            evidence_generation,
            physical_state_generation,
            live_snapshot_digest,
        )
    }

    pub(super) const fn owner_identity(self) -> RestoreOwnerIdentity {
        self.owner_identity
    }
}

/// Preparation owns the exact validated checkpoint source rather than accepting
/// a caller-supplied checkpoint digest beside unrelated state.
pub(super) struct RestorePreparationContext {
    source: OperationalRestoreSource,
    fence: RestoreGenerationFence,
}

impl RestorePreparationContext {
    pub(super) const fn new(
        source: OperationalRestoreSource,
        fence: RestoreGenerationFence,
    ) -> Self {
        Self { source, fence }
    }
}

/// Result of one domain-specific restore comparison.
///
/// `ConservativeRequalification` and `ReconciliationRequired` are admissible
/// only because they preserve an explicit restrictive/reconciliation action in
/// the committed plan. They are never silently promoted to historical replace.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum RestoreAdmissionVerdict {
    ProvenNonWidening,
    ConservativeRequalification,
    ReconciliationRequired,
    Widening,
    NotProvable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct RestoreDomainDecision {
    domain: RestoreDomain,
    verdict: RestoreAdmissionVerdict,
}

impl RestoreDomainDecision {
    pub(super) const fn new(
        domain: RestoreDomain,
        verdict: RestoreAdmissionVerdict,
    ) -> Self {
        Self { domain, verdict }
    }

    pub(super) const fn domain(self) -> RestoreDomain {
        self.domain
    }

    pub(super) const fn verdict(self) -> RestoreAdmissionVerdict {
        self.verdict
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum RestorePrepareError {
    InvalidOwnerIdentity,
    InvalidLiveSnapshotDigest,
    MissingDomain(RestoreDomain),
    DuplicateDomain(RestoreDomain),
    UnregisteredDomain(RestoreDomain),
    Widening(RestoreDomain),
    NotProvable(RestoreDomain),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum RestoreFenceField {
    OwnerIdentity,
    BootEpoch,
    ControlPlaneGeneration,
    AuthorityGeneration,
    EvidenceGeneration,
    PhysicalStateGeneration,
    LiveSnapshotDigest,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum RestoreFenceValue {
    OwnerIdentity(RestoreOwnerIdentity),
    Counter(u64),
    Digest(RestoreDigest),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct RestoreCommitError {
    changed: RestoreFenceField,
    expected: RestoreFenceValue,
    current: RestoreFenceValue,
}

impl RestoreCommitError {
    pub(super) const fn changed(self) -> RestoreFenceField {
        self.changed
    }

    pub(super) const fn expected(self) -> RestoreFenceValue {
        self.expected
    }

    pub(super) const fn current(self) -> RestoreFenceValue {
        self.current
    }
}

/// Affine prepared restore authority.
///
/// Deliberately not `Clone`, `Copy`, `Serialize` or `Deserialize`. The exact
/// normalized source is retained by value; raw portable checkpoint data cannot
/// deserialize into this live transaction.
pub(super) struct PreparedOperationalRestore {
    source: OperationalRestoreSource,
    fence: RestoreGenerationFence,
    decisions: Vec<RestoreDomainDecision>,
}

impl PreparedOperationalRestore {
    pub(super) const fn checkpoint_digest(&self) -> RestoreDigest {
        self.source.digest()
    }

    pub(super) fn decisions(&self) -> &[RestoreDomainDecision] {
        &self.decisions
    }
}

/// Single-use committed restore plan.
///
/// The exact normalized source and owner-bound generation fence that passed
/// commit are retained so execution cannot substitute a different checkpoint
/// source or silently reinterpret the receipt as applying to another owner.
/// The source remains owner-internal and is not a portable credential.
pub(super) struct CommittedOperationalRestore {
    source: OperationalRestoreSource,
    fence: RestoreGenerationFence,
    decisions: Vec<RestoreDomainDecision>,
}

impl CommittedOperationalRestore {
    pub(super) const fn checkpoint_digest(&self) -> RestoreDigest {
        self.source.digest()
    }

    pub(super) const fn fence(&self) -> RestoreGenerationFence {
        self.fence
    }

    pub(super) fn decisions(&self) -> &[RestoreDomainDecision] {
        &self.decisions
    }

    pub(super) fn into_source(self) -> OperationalRestoreSource {
        self.source
    }
}

/// Prepare a complete, canonical restore plan without mutating live state.
pub(super) fn prepare_operational_restore(
    context: RestorePreparationContext,
    decisions: Vec<RestoreDomainDecision>,
) -> Result<PreparedOperationalRestore, RestorePrepareError> {
    if !context.fence.owner_identity.is_valid() {
        return Err(RestorePrepareError::InvalidOwnerIdentity);
    }
    if !context.fence.live_snapshot_digest.is_valid() {
        return Err(RestorePrepareError::InvalidLiveSnapshotDigest);
    }

    for decision in &decisions {
        if !OPERATIONAL_RESTORE_CONTRACTS
            .iter()
            .any(|contract| contract.domain == decision.domain())
        {
            return Err(RestorePrepareError::UnregisteredDomain(decision.domain()));
        }
    }

    let mut canonical = Vec::with_capacity(OPERATIONAL_RESTORE_CONTRACTS.len());
    for contract in OPERATIONAL_RESTORE_CONTRACTS {
        let mut matching = decisions
            .iter()
            .copied()
            .filter(|decision| decision.domain() == contract.domain);
        let Some(decision) = matching.next() else {
            return Err(RestorePrepareError::MissingDomain(contract.domain));
        };
        if matching.next().is_some() {
            return Err(RestorePrepareError::DuplicateDomain(contract.domain));
        }

        match decision.verdict() {
            RestoreAdmissionVerdict::Widening => {
                return Err(RestorePrepareError::Widening(decision.domain()));
            }
            RestoreAdmissionVerdict::NotProvable => {
                return Err(RestorePrepareError::NotProvable(decision.domain()));
            }
            RestoreAdmissionVerdict::ProvenNonWidening
            | RestoreAdmissionVerdict::ConservativeRequalification
            | RestoreAdmissionVerdict::ReconciliationRequired => {}
        }
        canonical.push(decision);
    }

    Ok(PreparedOperationalRestore {
        source: context.source,
        fence: context.fence,
        decisions: canonical,
    })
}

fn fence_error(
    changed: RestoreFenceField,
    expected: RestoreFenceValue,
    current: RestoreFenceValue,
) -> RestoreCommitError {
    RestoreCommitError {
        changed,
        expected,
        current,
    }
}

/// Commit consumes the prepared token and rechecks the complete owner-bound live
/// fence. Owner substitution is rejected independently from same-owner state
/// movement.
pub(super) fn commit_operational_restore(
    prepared: PreparedOperationalRestore,
    current: RestoreGenerationFence,
) -> Result<CommittedOperationalRestore, RestoreCommitError> {
    let expected = prepared.fence;
    if current.owner_identity != expected.owner_identity {
        return Err(fence_error(
            RestoreFenceField::OwnerIdentity,
            RestoreFenceValue::OwnerIdentity(expected.owner_identity),
            RestoreFenceValue::OwnerIdentity(current.owner_identity),
        ));
    }
    if current.boot_epoch != expected.boot_epoch {
        return Err(fence_error(
            RestoreFenceField::BootEpoch,
            RestoreFenceValue::Counter(expected.boot_epoch),
            RestoreFenceValue::Counter(current.boot_epoch),
        ));
    }
    if current.control_plane_generation != expected.control_plane_generation {
        return Err(fence_error(
            RestoreFenceField::ControlPlaneGeneration,
            RestoreFenceValue::Counter(expected.control_plane_generation),
            RestoreFenceValue::Counter(current.control_plane_generation),
        ));
    }
    if current.authority_generation != expected.authority_generation {
        return Err(fence_error(
            RestoreFenceField::AuthorityGeneration,
            RestoreFenceValue::Counter(expected.authority_generation),
            RestoreFenceValue::Counter(current.authority_generation),
        ));
    }
    if current.evidence_generation != expected.evidence_generation {
        return Err(fence_error(
            RestoreFenceField::EvidenceGeneration,
            RestoreFenceValue::Counter(expected.evidence_generation),
            RestoreFenceValue::Counter(current.evidence_generation),
        ));
    }
    if current.physical_state_generation != expected.physical_state_generation {
        return Err(fence_error(
            RestoreFenceField::PhysicalStateGeneration,
            RestoreFenceValue::Counter(expected.physical_state_generation),
            RestoreFenceValue::Counter(current.physical_state_generation),
        ));
    }
    if current.live_snapshot_digest != expected.live_snapshot_digest {
        return Err(fence_error(
            RestoreFenceField::LiveSnapshotDigest,
            RestoreFenceValue::Digest(expected.live_snapshot_digest),
            RestoreFenceValue::Digest(current.live_snapshot_digest),
        ));
    }

    Ok(CommittedOperationalRestore {
        source: prepared.source,
        fence: expected,
        decisions: prepared.decisions,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embodiment::SubterraneanEmbodiment;
    use symthaea_core::genesis::GenesisSeed;

    fn digest(byte: u8) -> RestoreDigest {
        RestoreDigest::new([byte; 32])
    }

    fn owner(byte: u8) -> RestoreOwnerIdentity {
        RestoreOwnerIdentity::new([byte; 32])
    }

    fn source(phrase: &str) -> OperationalRestoreSource {
        let checkpoint =
            SubterraneanEmbodiment::new(&GenesisSeed::from_phrase(phrase)).operational_checkpoint();
        OperationalRestoreSource::capture(checkpoint).expect("valid restore source")
    }

    fn fence() -> RestoreGenerationFence {
        RestoreGenerationFence::new_bound(owner(5), 7, 11, 13, 17, 19, digest(23))
    }

    fn context() -> RestorePreparationContext {
        RestorePreparationContext::new(source("restore-admission-source"), fence())
    }

    fn nominal_decisions() -> Vec<RestoreDomainDecision> {
        OPERATIONAL_RESTORE_CONTRACTS
            .iter()
            .map(|contract| {
                RestoreDomainDecision::new(
                    contract.domain,
                    RestoreAdmissionVerdict::ProvenNonWidening,
                )
            })
            .collect()
    }

    fn prepared() -> PreparedOperationalRestore {
        prepare_operational_restore(context(), nominal_decisions()).expect("valid preparation")
    }

    #[test]
    fn complete_non_widening_restore_prepares_and_commits_with_derived_source_identity() {
        let prepared = prepared();
        let expected_source = prepared.checkpoint_digest();
        assert!(expected_source.is_valid());
        assert_eq!(prepared.decisions().len(), OPERATIONAL_RESTORE_CONTRACTS.len());
        let committed = commit_operational_restore(prepared, fence()).expect("fence unchanged");
        assert_eq!(committed.checkpoint_digest(), expected_source);
        assert_eq!(committed.fence(), fence());
        assert_eq!(committed.fence().owner_identity(), owner(5));
        assert_eq!(committed.decisions().len(), OPERATIONAL_RESTORE_CONTRACTS.len());
    }

    #[test]
    fn source_commitment_is_deterministic_and_changes_with_valid_source_state() {
        let first = source("source-identity");
        let second = source("source-identity");
        assert_eq!(first.digest(), second.digest());

        let mut changed =
            SubterraneanEmbodiment::new(&GenesisSeed::from_phrase("source-identity"))
                .operational_checkpoint();
        changed.controller.bias[0] = 0.01;
        let changed = OperationalRestoreSource::capture(changed).expect("valid changed source");
        assert_ne!(first.digest(), changed.digest());
    }

    #[test]
    fn source_capsule_stores_exact_portable_round_trip() {
        let checkpoint =
            SubterraneanEmbodiment::new(&GenesisSeed::from_phrase("portable-normalization"))
                .operational_checkpoint();
        let portable = serde_json::to_vec(&checkpoint).expect("portable encode");
        let source = OperationalRestoreSource::capture(checkpoint).expect("capture");
        let stored = serde_json::to_vec(source.checkpoint()).expect("stored encode");
        assert_eq!(stored, portable);
    }

    #[test]
    fn malformed_source_cannot_enter_preparation_context() {
        let mut checkpoint =
            SubterraneanEmbodiment::new(&GenesisSeed::from_phrase("invalid-source"))
                .operational_checkpoint();
        checkpoint.controller.hdc_dimension = checkpoint.controller.hdc_dimension.saturating_add(1);
        assert!(matches!(
            OperationalRestoreSource::capture(checkpoint),
            Err(RestoreSourceError::InvalidCheckpoint(
                OperationalCheckpointError::Controller(_)
            ))
        ));
    }

    #[test]
    fn caller_order_is_canonicalized_to_restore_registry_order() {
        let mut decisions = nominal_decisions();
        decisions.reverse();
        let prepared =
            prepare_operational_restore(context(), decisions).expect("order is not authority");
        let actual = prepared
            .decisions()
            .iter()
            .map(|decision| decision.domain())
            .collect::<Vec<_>>();
        let expected = OPERATIONAL_RESTORE_CONTRACTS
            .iter()
            .map(|contract| contract.domain)
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
    }

    #[test]
    fn missing_domain_fails_preparation_with_exact_domain() {
        let mut decisions = nominal_decisions();
        let missing = decisions.pop().expect("non-empty registry").domain();
        assert_eq!(
            prepare_operational_restore(context(), decisions).err(),
            Some(RestorePrepareError::MissingDomain(missing))
        );
    }

    #[test]
    fn duplicate_domain_fails_preparation_with_exact_domain() {
        let mut decisions = nominal_decisions();
        let duplicate = decisions[0];
        decisions.push(duplicate);
        assert_eq!(
            prepare_operational_restore(context(), decisions).err(),
            Some(RestorePrepareError::DuplicateDomain(duplicate.domain()))
        );
    }

    #[test]
    fn widening_fails_closed() {
        let mut decisions = nominal_decisions();
        let domain = decisions[2].domain();
        decisions[2] = RestoreDomainDecision::new(domain, RestoreAdmissionVerdict::Widening);
        assert_eq!(
            prepare_operational_restore(context(), decisions).err(),
            Some(RestorePrepareError::Widening(domain))
        );
    }

    #[test]
    fn not_provable_fails_closed() {
        let mut decisions = nominal_decisions();
        let domain = decisions[4].domain();
        decisions[4] = RestoreDomainDecision::new(domain, RestoreAdmissionVerdict::NotProvable);
        assert_eq!(
            prepare_operational_restore(context(), decisions).err(),
            Some(RestorePrepareError::NotProvable(domain))
        );
    }

    #[test]
    fn conservative_and_reconciliation_actions_survive_commit_explicitly() {
        let mut decisions = nominal_decisions();
        let conservative_domain = decisions[3].domain();
        let reconciliation_domain = decisions[4].domain();
        decisions[3] = RestoreDomainDecision::new(
            conservative_domain,
            RestoreAdmissionVerdict::ConservativeRequalification,
        );
        decisions[4] = RestoreDomainDecision::new(
            reconciliation_domain,
            RestoreAdmissionVerdict::ReconciliationRequired,
        );
        let prepared = prepare_operational_restore(context(), decisions).expect("safe actions");
        let committed = commit_operational_restore(prepared, fence()).expect("unchanged fence");
        assert!(committed.decisions().iter().any(|decision| {
            decision.domain() == conservative_domain
                && decision.verdict() == RestoreAdmissionVerdict::ConservativeRequalification
        }));
        assert!(committed.decisions().iter().any(|decision| {
            decision.domain() == reconciliation_domain
                && decision.verdict() == RestoreAdmissionVerdict::ReconciliationRequired
        }));
    }

    #[test]
    fn owner_substitution_is_rejected_independently_from_generation_movement() {
        let changed = RestoreGenerationFence::new_bound(owner(9), 7, 11, 13, 17, 19, digest(23));
        let error = commit_operational_restore(prepared(), changed)
            .err()
            .expect("wrong owner must fail");
        assert_eq!(error.changed(), RestoreFenceField::OwnerIdentity);
        assert_eq!(error.expected(), RestoreFenceValue::OwnerIdentity(owner(5)));
        assert_eq!(error.current(), RestoreFenceValue::OwnerIdentity(owner(9)));
    }

    #[test]
    fn any_generation_or_snapshot_change_invalidates_prepared_restore() {
        let base = fence();
        let changed_fences = [
            (
                RestoreFenceField::BootEpoch,
                RestoreFenceValue::Counter(7),
                RestoreFenceValue::Counter(8),
                RestoreGenerationFence::new_bound(owner(5), 8, 11, 13, 17, 19, digest(23)),
            ),
            (
                RestoreFenceField::ControlPlaneGeneration,
                RestoreFenceValue::Counter(11),
                RestoreFenceValue::Counter(12),
                RestoreGenerationFence::new_bound(owner(5), 7, 12, 13, 17, 19, digest(23)),
            ),
            (
                RestoreFenceField::AuthorityGeneration,
                RestoreFenceValue::Counter(13),
                RestoreFenceValue::Counter(14),
                RestoreGenerationFence::new_bound(owner(5), 7, 11, 14, 17, 19, digest(23)),
            ),
            (
                RestoreFenceField::EvidenceGeneration,
                RestoreFenceValue::Counter(17),
                RestoreFenceValue::Counter(18),
                RestoreGenerationFence::new_bound(owner(5), 7, 11, 13, 18, 19, digest(23)),
            ),
            (
                RestoreFenceField::PhysicalStateGeneration,
                RestoreFenceValue::Counter(19),
                RestoreFenceValue::Counter(20),
                RestoreGenerationFence::new_bound(owner(5), 7, 11, 13, 17, 20, digest(23)),
            ),
            (
                RestoreFenceField::LiveSnapshotDigest,
                RestoreFenceValue::Digest(digest(23)),
                RestoreFenceValue::Digest(digest(31)),
                RestoreGenerationFence::new_bound(owner(5), 7, 11, 13, 17, 19, digest(31)),
            ),
        ];

        assert_eq!(base, fence());
        for (field, expected, current, changed) in changed_fences {
            let error = commit_operational_restore(prepared(), changed)
                .err()
                .expect("stale prepared restore must fail");
            assert_eq!(error.changed(), field);
            assert_eq!(error.expected(), expected);
            assert_eq!(error.current(), current);
        }
    }

    #[test]
    fn invalid_owner_identity_fails_before_preparation() {
        let invalid_owner = RestorePreparationContext::new(
            source("invalid-owner-identity"),
            RestoreGenerationFence::new_bound(
                RestoreOwnerIdentity::new([0; 32]),
                7,
                11,
                13,
                17,
                19,
                digest(23),
            ),
        );
        assert_eq!(
            prepare_operational_restore(invalid_owner, nominal_decisions()).err(),
            Some(RestorePrepareError::InvalidOwnerIdentity)
        );
    }

    #[test]
    fn invalid_live_snapshot_digest_fails_before_preparation() {
        let invalid_live = RestorePreparationContext::new(
            source("invalid-live-digest"),
            RestoreGenerationFence::new_bound(
                owner(5),
                7,
                11,
                13,
                17,
                19,
                RestoreDigest::new([0; 32]),
            ),
        );
        assert_eq!(
            prepare_operational_restore(invalid_live, nominal_decisions()).err(),
            Some(RestorePrepareError::InvalidLiveSnapshotDigest)
        );
    }

    #[test]
    fn checkpoint_identity_is_preserved_across_prepare_and_commit() {
        let prepared = prepared();
        let expected_checkpoint = prepared.checkpoint_digest();
        let committed = commit_operational_restore(prepared, fence()).expect("unchanged fence");
        assert_eq!(committed.checkpoint_digest(), expected_checkpoint);
        assert_eq!(committed.fence(), fence());
        assert_eq!(committed.fence().owner_identity().bytes(), [5; 32]);
    }
}

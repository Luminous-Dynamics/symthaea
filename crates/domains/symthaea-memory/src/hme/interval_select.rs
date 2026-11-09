// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Global interval-first selection for exact ContinuousHV top-k refinement.
//!
//! Given one conservative score interval for every eligible record, this
//! module computes the kth-largest lower endpoint and retains every record
//! whose upper endpoint can still reach that threshold. The resulting set is
//! sufficient for exact reranking only if the caller has genuinely established
//! complete corpus coverage and valid, generation-bound intervals.
//!
//! This module checks coverage counts and interval shape; it does not prove
//! that the supplied counts reflect the pinned eligibility snapshot, nor does
//! it validate residual certificates. A caller must perform those checks
//! before passing intervals here. An approximate candidate index may order
//! work, but it cannot be substituted for the complete interval set.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::sync::Arc;

use super::group_envelope::{
    ContinuousHVGroupEnvelope, GroupEnvelopeError, continuous_hv_member_payload_digest,
    continuous_hv_query_digest, member_id_set_digest,
};

use symthaea_core::hdc::continuous_hv_interval::{
    ContinuousHVIntervalError, ContinuousHVScoreInterval, ResidualCertificateError,
    ResidualCertificateVerification, verify_l2_residual_certificate,
};
use symthaea_core::hdc::ContinuousHV;

/// One eligible record and its conservative canonical similarity interval.
#[derive(Debug, Clone, PartialEq)]
pub struct ScoreIntervalRecord<Id> {
    id: Id,
    interval: ContinuousHVScoreInterval,
    certificate_check: ResidualCertificateVerification,
}

/// Construction failure for a score interval backed by a checked residual.
#[derive(Debug, Clone, PartialEq)]
pub enum ScoreIntervalBuildError {
    /// The original/reconstruction pair did not satisfy its residual claim.
    ResidualCertificate(ResidualCertificateError),
    /// The canonical score interval could not be computed safely.
    ScoreInterval(ContinuousHVIntervalError),
    /// The computed interval failed its structural invariant.
    InvalidIntervalShape,
}

impl std::fmt::Display for ScoreIntervalBuildError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ResidualCertificate(error) => write!(f, "residual certificate invalid: {error}"),
            Self::ScoreInterval(error) => write!(f, "score interval invalid: {error}"),
            Self::InvalidIntervalShape => write!(f, "score interval failed its structural invariant"),
        }
    }
}

impl std::error::Error for ScoreIntervalBuildError {}

impl<Id> ScoreIntervalRecord<Id> {
    /// Construct a prune-authorizing record only after checking the residual
    /// against the original payload and propagating it through the canonical
    /// f32 score interval kernel.
    ///
    /// This is a build/admission-time operation: the original payload must be
    /// available for residual verification. Persisted interval metadata must
    /// be bound to the exact payload, reconstruction, codebook and generation;
    /// this type deliberately has no unchecked public constructor or mutable
    /// interval field.
    pub fn from_verified_l2_residual(
        id: Id,
        query: &ContinuousHV,
        original: &ContinuousHV,
        reconstruction: &ContinuousHV,
        residual_l2_upper: f32,
    ) -> Result<Self, ScoreIntervalBuildError> {
        let certificate_check = verify_l2_residual_certificate(
            original,
            reconstruction,
            residual_l2_upper,
        )
        .map_err(ScoreIntervalBuildError::ResidualCertificate)?;
        let interval = query
            .similarity_interval_full_resolution_for_l2_residual(
                reconstruction,
                residual_l2_upper,
            )
            .map_err(ScoreIntervalBuildError::ScoreInterval)?;
        if !interval.is_well_formed() {
            return Err(ScoreIntervalBuildError::InvalidIntervalShape);
        }
        Ok(Self {
            id,
            interval,
            certificate_check,
        })
    }

    /// Stable ID for deterministic reranking.
    pub fn id(&self) -> &Id {
        &self.id
    }

    /// Conservative canonical-score interval.
    pub fn interval(&self) -> ContinuousHVScoreInterval {
        self.interval
    }

    /// Result of independent source-payload residual verification.
    pub fn certificate_check(&self) -> ResidualCertificateVerification {
        self.certificate_check
    }
}

/// Counts supplied by the caller for a pinned eligible-corpus snapshot.
///
/// These are consistency counters, not a cryptographic or semantic proof.
/// The caller must bind them to the manifest and eligibility generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IntervalCoverageCounts {
    /// Number of eligible records in the pinned snapshot.
    pub eligible_records: usize,
    /// Number of eligible records with a validated interval supplied here.
    pub interval_records: usize,
    /// Number of unopened partitions that lack an admissible aggregate bound.
    pub unbounded_unopened_partitions: usize,
}

/// One exact canonical score and its stable record identifier.
#[derive(Debug, Clone, PartialEq)]
pub struct ExactScoreRecord<Id> {
    /// Stable record identifier.
    pub id: Id,
    /// Canonical finite similarity score in [-1, 1].
    pub score: f32,
}

/// The deterministic refinement set for one larger-is-better top-k query.
///
/// This plan selects records that must receive exact payload scoring. It is not
/// an ExactComplete receipt: the caller must exact-score every refinement ID,
/// rerank under the canonical stable-ID tie policy, and bind completion to the
/// manifest, score contract, certificate generation and eligibility snapshot.
#[derive(Debug, Clone, PartialEq)]
pub struct IntervalSelectionPlan<Id> {
    /// kth-largest lower endpoint across the complete eligible set.
    pub threshold: f32,
    /// IDs that must be exact-scored from their original payloads.
    pub refinement_ids: Vec<Id>,
    /// IDs proven unable to enter the exact top-k, assuming valid intervals.
    pub excluded_ids: Vec<Id>,
}


/// Immutable snapshot identity for a group-level interval.
///
/// The member-set and content digests bind the selector plan back to the exact
/// group envelope; generation fields prevent cross-snapshot routing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct GroupScoreIntervalIdentity {
    /// Manifest generation containing the group.
    pub manifest_generation: u64,
    /// Eligibility generation used to build the group.
    pub eligibility_generation: u64,
    /// Digest of the stable eligible member-ID set.
    pub member_set_digest: [u8; 32],
    /// Digest of member payloads, extrema, centroid and radius.
    pub content_digest: [u8; 32],
}

/// One group whose interval encloses every eligible member in that group.
///
/// Group records can only be built from a validated in-memory group envelope;
/// the returned identity is the envelope's pinned snapshot identity, not an
/// unrelated caller-supplied display label.
#[derive(Debug, Clone, PartialEq)]
pub struct GroupScoreIntervalRecord {
    identity: GroupScoreIntervalIdentity,
    member_ids: Arc<[String]>,
    member_payload_digests: Arc<[[u8; 32]]>,
    query_digest: [u8; 32],
    scope_binding: HmeEligibilityScopeBinding,
    member_count: usize,
    interval: ContinuousHVScoreInterval,
}

/// Error while converting a verified group envelope into a selection record.
#[derive(Debug, Clone, PartialEq)]
pub enum GroupScoreIntervalBuildError {
    /// The backing group envelope rejected its snapshot bindings.
    GroupEnvelope(GroupEnvelopeError),
    /// The envelope has no members.
    EmptyGroup,
    /// The envelope interval failed its structural invariant.
    InvalidInterval,
}

impl std::fmt::Display for GroupScoreIntervalBuildError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::GroupEnvelope(error) => write!(f, "group envelope rejected: {error}"),
            Self::EmptyGroup => write!(f, "cannot select an empty group"),
            Self::InvalidInterval => write!(f, "group score interval is malformed"),
        }
    }
}

impl std::error::Error for GroupScoreIntervalBuildError {}

impl GroupScoreIntervalRecord {
    /// Construct from an in-memory group envelope after its pinned snapshot is
    /// revalidated and its query-specific interval is computed.
    pub fn from_group_envelope(
        envelope: &ContinuousHVGroupEnvelope,
        query: &ContinuousHV,
        expected_manifest_generation: u64,
        expected_eligibility_generation: u64,
        expected_member_set_digest: [u8; 32],
        expected_content_digest: [u8; 32],
        scope_binding: HmeEligibilityScopeBinding,
    ) -> Result<Self, GroupScoreIntervalBuildError> {
        let member_count = envelope.member_ids().len();
        if member_count == 0 {
            return Err(GroupScoreIntervalBuildError::EmptyGroup);
        }
        let interval = envelope
            .score_interval_for_snapshot(
                query,
                expected_manifest_generation,
                expected_eligibility_generation,
                expected_member_set_digest,
                expected_content_digest,
            )
            .map_err(GroupScoreIntervalBuildError::GroupEnvelope)?;
        if !interval.is_well_formed() {
            return Err(GroupScoreIntervalBuildError::InvalidInterval);
        }
        Ok(Self {
            identity: GroupScoreIntervalIdentity {
                manifest_generation: expected_manifest_generation,
                eligibility_generation: expected_eligibility_generation,
                member_set_digest: expected_member_set_digest,
                content_digest: expected_content_digest,
            },
            member_ids: envelope.member_ids_shared(),
            member_payload_digests: envelope.member_payload_digests_shared(),
            query_digest: continuous_hv_query_digest(query),
            scope_binding,
            member_count,
            interval,
        })
    }

    /// Exact group identity carried into the selection plan.
    pub fn identity(&self) -> GroupScoreIntervalIdentity {
        self.identity
    }

    /// Stable member IDs covered by this envelope, sorted ascending.
    pub fn member_ids(&self) -> &[String] {
        &self.member_ids
    }

    /// Eligibility scope/policy/source-ledger/bitemporal binding for this interval.
    pub fn scope_binding(&self) -> HmeEligibilityScopeBinding {
        self.scope_binding
    }

    /// Number of eligible records represented by this interval.
    pub fn member_count(&self) -> usize {
        self.member_count
    }

    /// Query-specific interval enclosing every eligible member score.
    pub fn interval(&self) -> ContinuousHVScoreInterval {
        self.interval
    }
}

/// Coverage counters for weighted group interval selection.
///
/// These counters must be derived from the same pinned eligibility snapshot as
/// the group envelopes. They are consistency checks, not cryptographic proof.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GroupIntervalCoverageCounts {
    /// Number of eligible records in the pinned snapshot.
    pub eligible_records: usize,
    /// Number of eligible records represented by the supplied group intervals.
    pub covered_records: usize,
    /// Number of group intervals supplied.
    pub interval_groups: usize,
    /// Expected manifest generation for all groups in this query.
    pub expected_manifest_generation: u64,
    /// Expected eligibility generation for all groups in this query.
    pub expected_eligibility_generation: u64,
    /// Digest of the exact query vector used to produce every group interval.
    pub query_digest: [u8; 32],
    /// Eligibility scope/policy/source-ledger/bitemporal binding from #11003.
    pub scope_binding: HmeEligibilityScopeBinding,
    /// Number of unopened groups without an admissible aggregate interval.
    pub unbounded_unopened_groups: usize,
}


/// Identity of the authoritative eligibility decision used to construct a
/// group partition. Values must come from the existing #11003 eligibility
/// owner; this HME data type does not decide authorization or validate policy.
///
/// The bitemporal digest commits to valid_at and known_at. The principal-scope
/// and policy digests prevent a partition receipt from being replayed across
/// tenants, namespaces or policy revisions, even when the visible member set
/// happens to be unchanged.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HmeEligibilityScopeBinding {
    /// Principal/tenant/namespace/scope digest from the eligibility owner.
    pub principal_scope_digest: [u8; 32],
    /// Policy identity/version/content digest from the eligibility owner.
    pub policy_digest: [u8; 32],
    /// Monotonic policy generation pinned by the query.
    pub policy_generation: u64,
    /// Source-event ledger generation pinned by the query.
    pub source_ledger_generation: u64,
    /// Digest of the exact valid_at/known_at query coordinates.
    pub bitemporal_query_digest: [u8; 32],
}

/// Opaque build-time proof that a set of group envelopes is a disjoint,
/// complete partition of one manifest/eligibility snapshot.
///
/// The member-ID disjointness check is paid once when this receipt is created.
/// Query-time verification hashes the small group identity list and checks
/// counts/generations; it does not scan every member ID. Persistent use still
/// requires the segment/manifest owner (#10917) to authenticate this receipt
/// on reopen. This in-memory type is not a signature or authorization token.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GroupPartitionCoverageReceipt {
    manifest_generation: u64,
    eligibility_generation: u64,
    eligible_records: usize,
    eligible_member_set_digest: [u8; 32],
    scope_binding: HmeEligibilityScopeBinding,
    group_count: usize,
    partition_digest: [u8; 32],
}

/// Failure while proving a disjoint group partition at build time or matching
/// a query's group directory to an existing receipt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GroupPartitionReceiptError {
    /// A partition cannot be empty.
    EmptyPartition,
    /// A group has no members or has misaligned member/payload metadata.
    InvalidGroupMembership {
        /// Group index with malformed member metadata.
        group_index: usize,
    },
    /// A group's roster does not match its stored member-set digest.
    MemberSetDigestMismatch {
        /// Group index whose roster digest disagreed.
        group_index: usize,
    },
    /// Two groups have the same group member-set digest.
    DuplicateGroup {
        /// First group occurrence.
        first_group_index: usize,
        /// Later duplicate occurrence.
        duplicate_group_index: usize,
    },
    /// One stable member ID appears in multiple groups.
    DuplicateMember {
        /// Duplicated stable member ID.
        member_id: String,
        /// First group containing the member.
        first_group_index: usize,
        /// Later group containing the member.
        duplicate_group_index: usize,
    },
    /// A group belongs to a different manifest or eligibility snapshot.
    MixedSnapshot {
        /// Index of the mismatched group.
        group_index: usize,
    },
    /// Adding group member counts overflowed usize.
    MemberCountOverflow,
    /// Supplied groups do not cover the authoritative eligible-record count.
    EligiblePopulationCountMismatch {
        /// Expected records from the pinned manifest/eligibility snapshot.
        expected: usize,
        /// Actual records across the supplied envelopes.
        actual: usize,
    },
    /// Supplied group rosters do not match the authoritative eligible-member digest.
    EligibleMemberSetDigestMismatch,
    /// Query scope/policy/source-ledger/bitemporal binding differs from the receipt.
    EligibilityScopeMismatch,
    /// Query group directory does not match this receipt.
    ReceiptMismatch,
}

impl std::fmt::Display for GroupPartitionReceiptError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyPartition => write!(f, "group partition is empty"),
            Self::InvalidGroupMembership { group_index } => {
                write!(f, "group {group_index} has invalid member metadata")
            }
            Self::MemberSetDigestMismatch { group_index } => {
                write!(f, "group {group_index} member roster digest does not match")
            }
            Self::DuplicateGroup { first_group_index, duplicate_group_index } => write!(
                f,
                "duplicate group at indices {first_group_index} and {duplicate_group_index}"
            ),
            Self::DuplicateMember {
                member_id,
                first_group_index,
                duplicate_group_index,
            } => write!(
                f,
                "member {member_id} appears in groups {first_group_index} and {duplicate_group_index}"
            ),
            Self::MixedSnapshot { group_index } => {
                write!(f, "group {group_index} belongs to another snapshot")
            }
            Self::MemberCountOverflow => write!(f, "group partition member count overflow"),
            Self::EligiblePopulationCountMismatch { expected, actual } => write!(
                f,
                "eligible population count mismatch: expected={expected}, supplied={actual}"
            ),
            Self::EligibleMemberSetDigestMismatch => {
                write!(f, "group rosters do not match authoritative eligible-member digest")
            }
            Self::EligibilityScopeMismatch => {
                write!(f, "query eligibility scope, policy or bitemporal binding differs")
            }
            Self::ReceiptMismatch => write!(f, "group directory does not match its partition receipt"),
        }
    }
}

impl std::error::Error for GroupPartitionReceiptError {}

impl GroupPartitionCoverageReceipt {
    /// Verify a complete, disjoint group partition at segment-build time.
    ///
    /// This method scans stable IDs once, verifies each per-group roster digest,
    /// and computes a deterministic receipt over sorted group identities.
    pub fn from_group_envelopes(
        envelopes: &[ContinuousHVGroupEnvelope],
        expected_eligible_records: usize,
        expected_eligible_member_set_digest: [u8; 32],
        scope_binding: HmeEligibilityScopeBinding,
    ) -> Result<Self, GroupPartitionReceiptError> {
        let first = envelopes.first().ok_or(GroupPartitionReceiptError::EmptyPartition)?;
        let manifest_generation = first.manifest_generation();
        let eligibility_generation = first.eligibility_generation();
        let mut member_count = 0usize;
        let mut first_group_by_member: BTreeMap<&str, usize> = BTreeMap::new();
        let mut first_group_by_digest: BTreeMap<[u8; 32], usize> = BTreeMap::new();
        let mut all_member_ids = Vec::new();
        let mut group_identities = Vec::with_capacity(envelopes.len());

        for (group_index, envelope) in envelopes.iter().enumerate() {
            if envelope.manifest_generation() != manifest_generation
                || envelope.eligibility_generation() != eligibility_generation
            {
                return Err(GroupPartitionReceiptError::MixedSnapshot { group_index });
            }
            let member_ids = envelope.member_ids();
            let payload_digests = envelope.member_payload_digests();
            if member_ids.is_empty() || member_ids.len() != payload_digests.len() {
                return Err(GroupPartitionReceiptError::InvalidGroupMembership { group_index });
            }
            let observed_digest = member_id_set_digest(member_ids)
                .map_err(|_| GroupPartitionReceiptError::InvalidGroupMembership { group_index })?;
            if observed_digest != envelope.member_set_digest() {
                return Err(GroupPartitionReceiptError::MemberSetDigestMismatch { group_index });
            }
            if let Some(first_group_index) =
                first_group_by_digest.get(&envelope.member_set_digest()).copied()
            {
                return Err(GroupPartitionReceiptError::DuplicateGroup {
                    first_group_index,
                    duplicate_group_index: group_index,
                });
            }
            first_group_by_digest.insert(envelope.member_set_digest(), group_index);
            for member_id in member_ids {
                if let Some(first_group_index) =
                    first_group_by_member.get(member_id.as_str()).copied()
                {
                    return Err(GroupPartitionReceiptError::DuplicateMember {
                        member_id: member_id.clone(),
                        first_group_index,
                        duplicate_group_index: group_index,
                    });
                }
                first_group_by_member.insert(member_id.as_str(), group_index);
                all_member_ids.push(member_id.clone());
            }
            member_count = member_count
                .checked_add(member_ids.len())
                .ok_or(GroupPartitionReceiptError::MemberCountOverflow)?;
            group_identities.push((
                envelope.member_set_digest(),
                envelope.content_digest(),
                member_ids.len(),
            ));
        }

        if member_count != expected_eligible_records {
            return Err(GroupPartitionReceiptError::EligiblePopulationCountMismatch {
                expected: expected_eligible_records,
                actual: member_count,
            });
        }
        let actual_member_set_digest = member_id_set_digest(&all_member_ids)
            .map_err(|_| GroupPartitionReceiptError::EligibleMemberSetDigestMismatch)?;
        if actual_member_set_digest != expected_eligible_member_set_digest {
            return Err(GroupPartitionReceiptError::EligibleMemberSetDigestMismatch);
        }

        group_identities.sort_by_key(|entry| (entry.0, entry.1));
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"hme_group_partition_coverage_receipt_v1");
        hasher.update(&manifest_generation.to_le_bytes());
        hasher.update(&eligibility_generation.to_le_bytes());
        hasher.update(&scope_binding.principal_scope_digest);
        hasher.update(&scope_binding.policy_digest);
        hasher.update(&scope_binding.policy_generation.to_le_bytes());
        hasher.update(&scope_binding.source_ledger_generation.to_le_bytes());
        hasher.update(&scope_binding.bitemporal_query_digest);
        hasher.update(&(member_count as u64).to_le_bytes());
        hasher.update(&actual_member_set_digest);
        hasher.update(&(group_identities.len() as u64).to_le_bytes());
        for (member_set_digest, content_digest, members) in group_identities {
            hasher.update(&member_set_digest);
            hasher.update(&content_digest);
            hasher.update(&(members as u64).to_le_bytes());
        }

        Ok(Self {
            manifest_generation,
            eligibility_generation,
            eligible_records: member_count,
            eligible_member_set_digest: actual_member_set_digest,
            scope_binding,
            group_count: envelopes.len(),
            partition_digest: *hasher.finalize().as_bytes(),
        })
    }

    /// Verify a query's compact group records against this build-time receipt.
    ///
    /// Cost is O(G log G) group-level work: member roster disjointness and
    /// eligible population membership were established by from_group_envelopes().
    /// No member-ID roster is walked on this query path.
    fn verify_group_records(
        &self,
        groups: &[GroupScoreIntervalRecord],
    ) -> Result<(), GroupPartitionReceiptError> {
        if groups.is_empty() || groups.len() != self.group_count {
            return Err(GroupPartitionReceiptError::ReceiptMismatch);
        }
        let mut member_count = 0usize;
        let mut identities = Vec::with_capacity(groups.len());
        for (group_index, group) in groups.iter().enumerate() {
            if group.identity.manifest_generation != self.manifest_generation
                || group.identity.eligibility_generation != self.eligibility_generation
                || group.member_count == 0
            {
                return Err(GroupPartitionReceiptError::MixedSnapshot { group_index });
            }
            if group.member_count != group.member_ids.len()
                || group.member_ids.len() != group.member_payload_digests.len()
            {
                return Err(GroupPartitionReceiptError::InvalidGroupMembership { group_index });
            }
            member_count = member_count
                .checked_add(group.member_count)
                .ok_or(GroupPartitionReceiptError::MemberCountOverflow)?;
            identities.push((
                group.identity.member_set_digest,
                group.identity.content_digest,
                group.member_count,
            ));
        }
        if member_count != self.eligible_records {
            return Err(GroupPartitionReceiptError::ReceiptMismatch);
        }
        identities.sort_by_key(|entry| (entry.0, entry.1));
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"hme_group_partition_coverage_receipt_v1");
        hasher.update(&self.manifest_generation.to_le_bytes());
        hasher.update(&self.eligibility_generation.to_le_bytes());
        hasher.update(&self.scope_binding.principal_scope_digest);
        hasher.update(&self.scope_binding.policy_digest);
        hasher.update(&self.scope_binding.policy_generation.to_le_bytes());
        hasher.update(&self.scope_binding.source_ledger_generation.to_le_bytes());
        hasher.update(&self.scope_binding.bitemporal_query_digest);
        hasher.update(&(member_count as u64).to_le_bytes());
        hasher.update(&self.eligible_member_set_digest);
        hasher.update(&(identities.len() as u64).to_le_bytes());
        for (member_set_digest, content_digest, members) in identities {
            hasher.update(&member_set_digest);
            hasher.update(&content_digest);
            hasher.update(&(members as u64).to_le_bytes());
        }
        if *hasher.finalize().as_bytes() != self.partition_digest {
            return Err(GroupPartitionReceiptError::ReceiptMismatch);
        }
        Ok(())
    }

    /// Snapshot generation bound into this partition receipt.
    pub fn manifest_generation(&self) -> u64 {
        self.manifest_generation
    }

    /// Eligibility generation bound into this partition receipt.
    pub fn eligibility_generation(&self) -> u64 {
        self.eligibility_generation
    }

    /// Total eligible member count covered by the partition.
    pub fn eligible_records(&self) -> usize {
        self.eligible_records
    }

    /// Authoritative eligible-member-set digest matched at build time.
    pub fn eligible_member_set_digest(&self) -> [u8; 32] {
        self.eligible_member_set_digest
    }

    /// Exact principal/policy/source-ledger/bitemporal scope binding.
    pub fn scope_binding(&self) -> HmeEligibilityScopeBinding {
        self.scope_binding
    }

    /// Number of groups covered by this partition.
    pub fn group_count(&self) -> usize {
        self.group_count
    }

    /// Deterministic digest of the disjoint group directory.
    pub fn partition_digest(&self) -> [u8; 32] {
        self.partition_digest
    }
}


/// Exact group-level refinement plan obtained from weighted lower endpoints.
///
/// The kth-largest lower endpoint is computed with each group's lower value
/// repeated once per eligible member. Every group whose upper endpoint reaches
/// the threshold remains for refinement; only groups with upper < threshold
/// may be excluded.
#[derive(Debug, Clone, PartialEq)]
pub struct GroupIntervalSelectionPlan {
    /// kth-largest weighted lower endpoint over the complete eligible corpus.
    pub threshold: f32,
    /// Manifest generation pinned by this plan.
    pub manifest_generation: u64,
    /// Eligibility generation pinned by this plan.
    pub eligibility_generation: u64,
    /// Exact query vector identity used for all interval calculations.
    pub query_digest: [u8; 32],
    /// Principal/policy/source-ledger/bitemporal eligibility scope binding.
    pub scope_binding: HmeEligibilityScopeBinding,
    /// Snapshot-bound group identities whose members must be visited or refined.
    pub refinement_groups: Vec<GroupScoreIntervalIdentity>,
    /// Snapshot-bound groups proven unable to contain a member in the exact top-k.
    pub excluded_groups: Vec<GroupScoreIntervalIdentity>,
    /// Number of records represented by the refinement groups.
    pub refinement_record_count: usize,
    /// Number of records avoided by excluding whole groups.
    pub excluded_record_count: usize,
}



/// In-memory admission receipt for one compressed group member.
///
/// Constructed only after the original payload/reconstruction residual was
/// independently checked. Query-time use needs the compact reconstruction and
/// this receipt, not the original payload. Persistent serialization,
/// manifest authentication and reopen verification are deliberately not
/// implemented by this in-memory prototype; #10917 owns that boundary.
#[derive(Debug, Clone, PartialEq)]
pub struct VerifiedMemberResidualCertificate {
    group_identity: GroupScoreIntervalIdentity,
    member_id: String,
    original_payload_digest: [u8; 32],
    reconstruction_digest: [u8; 32],
    transform_profile_digest: [u8; 32],
    residual_l2_upper: f32,
    verification: ResidualCertificateVerification,
}

/// Error while checking an admission receipt against a group and reconstruction.
#[derive(Debug, Clone, PartialEq)]
pub enum VerifiedMemberResidualError {
    /// Receipt was admitted for another immutable group snapshot.
    GroupIdentityMismatch,
    /// Receipt member ID is not present in the supplied group.
    MemberNotInGroup,
    /// The group does not bind the same original payload bytes.
    OriginalPayloadDigestMismatch,
    /// The compact reconstruction differs from the admitted reconstruction.
    ReconstructionDigestMismatch,
    /// The transform/codebook profile differs from the admitted profile.
    TransformProfileMismatch,
    /// The generated interval failed its structural invariant.
    InvalidIntervalShape,
    /// The canonical score interval could not be computed.
    ScoreInterval(ContinuousHVIntervalError),
}

impl std::fmt::Display for VerifiedMemberResidualError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::GroupIdentityMismatch => write!(f, "residual receipt belongs to another group snapshot"),
            Self::MemberNotInGroup => write!(f, "residual receipt member is not in the group"),
            Self::OriginalPayloadDigestMismatch => write!(f, "original payload digest differs from group membership"),
            Self::ReconstructionDigestMismatch => write!(f, "compact reconstruction differs from admitted reconstruction"),
            Self::TransformProfileMismatch => write!(f, "transform/codebook profile differs from residual receipt"),
            Self::InvalidIntervalShape => write!(f, "member interval failed its structural invariant"),
            Self::ScoreInterval(error) => write!(f, "member score interval failed: {error}"),
        }
    }
}

impl std::error::Error for VerifiedMemberResidualError {}

impl VerifiedMemberResidualCertificate {
    /// Independently verify and bind one member residual at ingestion/build time.
    ///
    /// The original bytes are read at this stage. The resulting opaque receipt
    /// binds the residual radius to the parent group, original payload digest,
    /// reconstruction digest and transform/codebook profile.
    pub fn verify_at_ingest(
        group: &GroupScoreIntervalRecord,
        member_id: String,
        original: &ContinuousHV,
        reconstruction: &ContinuousHV,
        residual_l2_upper: f32,
        transform_profile_digest: [u8; 32],
    ) -> Result<Self, GroupMemberScoreIntervalBuildError> {
        let member_index = group
            .member_ids
            .binary_search(&member_id)
            .map_err(|_| GroupMemberScoreIntervalBuildError::MemberNotInGroup)?;
        let expected_payload_digest = group
            .member_payload_digests
            .get(member_index)
            .copied()
            .ok_or(GroupMemberScoreIntervalBuildError::OriginalPayloadDigestMismatch)?;
        let original_payload_digest =
            continuous_hv_member_payload_digest(&member_id, original);
        if original_payload_digest != expected_payload_digest {
            return Err(GroupMemberScoreIntervalBuildError::OriginalPayloadDigestMismatch);
        }
        let verification = verify_l2_residual_certificate(
            original,
            reconstruction,
            residual_l2_upper,
        )
        .map_err(|error| {
            GroupMemberScoreIntervalBuildError::ScoreInterval(
                ScoreIntervalBuildError::ResidualCertificate(error),
            )
        })?;
        Ok(Self {
            group_identity: group.identity,
            member_id,
            original_payload_digest,
            reconstruction_digest: continuous_hv_member_payload_digest(
                group.member_ids[member_index].as_str(),
                reconstruction,
            ),
            transform_profile_digest,
            residual_l2_upper,
            verification,
        })
    }

    /// Parent group identity bound when the residual was verified.
    pub fn group_identity(&self) -> GroupScoreIntervalIdentity {
        self.group_identity
    }

    /// Stable member identifier bound into the receipt.
    pub fn member_id(&self) -> &str {
        &self.member_id
    }

    /// Upper L2 radius checked at ingestion.
    pub fn residual_l2_upper(&self) -> f32 {
        self.residual_l2_upper
    }

    /// Transform/codebook profile digest bound into the receipt.
    pub fn transform_profile_digest(&self) -> [u8; 32] {
        self.transform_profile_digest
    }
}


/// One verified per-record interval tied to its parent group identity.
#[derive(Debug, Clone, PartialEq)]
pub struct GroupMemberScoreIntervalRecord {
    group_identity: GroupScoreIntervalIdentity,
    query_digest: [u8; 32],
    record: ScoreIntervalRecord<String>,
}

/// Error while constructing a per-record interval within a verified group.
#[derive(Debug, Clone, PartialEq)]
pub enum GroupMemberScoreIntervalBuildError {
    /// The requested stable ID is not a member of the supplied group envelope.
    MemberNotInGroup,
    /// The original payload bytes do not match the member payload bound to the group.
    OriginalPayloadDigestMismatch,
    /// Original/reconstruction residual or score interval validation failed.
    ScoreInterval(ScoreIntervalBuildError),
    /// The verified receipt does not match the group, reconstruction or transform.
    VerifiedCertificate(VerifiedMemberResidualError),
}

impl std::fmt::Display for GroupMemberScoreIntervalBuildError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MemberNotInGroup => write!(f, "record ID is not in the pinned parent group"),
            Self::OriginalPayloadDigestMismatch => write!(f, "original payload does not match the group-bound member bytes"),
            Self::ScoreInterval(error) => write!(f, "member score interval rejected: {error}"),
            Self::VerifiedCertificate(error) => write!(f, "verified residual receipt rejected: {error}"),
        }
    }
}

impl std::error::Error for GroupMemberScoreIntervalBuildError {}

impl GroupMemberScoreIntervalRecord {
    /// Build a member interval after checking both its residual and membership
    /// in the exact group whose envelope survived the first selection stage.
    pub fn from_verified_l2_residual(
        group: &GroupScoreIntervalRecord,
        id: String,
        query: &ContinuousHV,
        original: &ContinuousHV,
        reconstruction: &ContinuousHV,
        residual_l2_upper: f32,
    ) -> Result<Self, GroupMemberScoreIntervalBuildError> {
        let member_index = group
            .member_ids
            .binary_search(&id)
            .map_err(|_| GroupMemberScoreIntervalBuildError::MemberNotInGroup)?;
        let expected_payload_digest = group
            .member_payload_digests
            .get(member_index)
            .copied()
            .ok_or(GroupMemberScoreIntervalBuildError::OriginalPayloadDigestMismatch)?;
        if continuous_hv_member_payload_digest(&id, original) != expected_payload_digest {
            return Err(GroupMemberScoreIntervalBuildError::OriginalPayloadDigestMismatch);
        }
        let record = ScoreIntervalRecord::from_verified_l2_residual(
            id,
            query,
            original,
            reconstruction,
            residual_l2_upper,
        )
        .map_err(GroupMemberScoreIntervalBuildError::ScoreInterval)?;
        Ok(Self {
            group_identity: group.identity,
            query_digest: continuous_hv_query_digest(query),
            record,
        })
    }

    /// Generate a query-specific member interval from a previously verified
    /// admission receipt and the compact reconstruction. This path does not
    /// read the original payload. The caller must authenticate persisted
    /// receipts against the immutable segment manifest before constructing
    /// this opaque in-memory receipt after reopen.
    pub fn from_verified_certificate(
        group: &GroupScoreIntervalRecord,
        query: &ContinuousHV,
        reconstruction: &ContinuousHV,
        certificate: &VerifiedMemberResidualCertificate,
        expected_transform_profile_digest: [u8; 32],
    ) -> Result<Self, GroupMemberScoreIntervalBuildError> {
        if group.identity != certificate.group_identity {
            return Err(GroupMemberScoreIntervalBuildError::VerifiedCertificate(
                VerifiedMemberResidualError::GroupIdentityMismatch,
            ));
        }
        let member_index = group
            .member_ids
            .binary_search(&certificate.member_id)
            .map_err(|_| {
                GroupMemberScoreIntervalBuildError::VerifiedCertificate(
                    VerifiedMemberResidualError::MemberNotInGroup,
                )
            })?;
        if group.member_payload_digests.get(member_index).copied()
            != Some(certificate.original_payload_digest)
        {
            return Err(GroupMemberScoreIntervalBuildError::VerifiedCertificate(
                VerifiedMemberResidualError::OriginalPayloadDigestMismatch,
            ));
        }
        let reconstruction_digest = continuous_hv_member_payload_digest(
            &certificate.member_id,
            reconstruction,
        );
        if reconstruction_digest != certificate.reconstruction_digest {
            return Err(GroupMemberScoreIntervalBuildError::VerifiedCertificate(
                VerifiedMemberResidualError::ReconstructionDigestMismatch,
            ));
        }
        if certificate.transform_profile_digest != expected_transform_profile_digest {
            return Err(GroupMemberScoreIntervalBuildError::VerifiedCertificate(
                VerifiedMemberResidualError::TransformProfileMismatch,
            ));
        }

        let interval = query
            .similarity_interval_full_resolution_for_l2_residual(
                reconstruction,
                certificate.residual_l2_upper,
            )
            .map_err(|error| {
                GroupMemberScoreIntervalBuildError::VerifiedCertificate(
                    VerifiedMemberResidualError::ScoreInterval(error),
                )
            })?;
        if !interval.is_well_formed() {
            return Err(GroupMemberScoreIntervalBuildError::VerifiedCertificate(
                VerifiedMemberResidualError::InvalidIntervalShape,
            ));
        }
        Ok(Self {
            group_identity: group.identity,
            query_digest: continuous_hv_query_digest(query),
            record: ScoreIntervalRecord {
                id: certificate.member_id.clone(),
                interval,
                certificate_check: certificate.verification,
            },
        })
    }

    /// Parent group identity; binds manifest, eligibility, membership and content.
    pub fn group_identity(&self) -> GroupScoreIntervalIdentity {
        self.group_identity
    }

    /// Stable member ID.
    pub fn id(&self) -> &str {
        self.record.id()
    }

    /// Per-record canonical score interval before intersection with the group.
    pub fn interval(&self) -> ContinuousHVScoreInterval {
        self.record.interval()
    }
}

/// Two-stage exact refinement plan: first select whole groups, then select
/// per-record payloads within the surviving groups.
#[derive(Debug, Clone, PartialEq)]
pub struct HierarchicalIntervalSelectionPlan {
    /// First-stage group selection and its safe exclusions.
    pub group_plan: GroupIntervalSelectionPlan,
    /// Second-stage per-record selection within the surviving groups.
    pub member_plan: IntervalSelectionPlan<String>,
}

/// Errors that prevent hierarchical exact interval selection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HierarchicalIntervalSelectionError {
    /// The first-stage group selector rejected coverage or metadata.
    GroupSelection(GroupIntervalSelectionError),
    /// A supplied member interval belongs to an unselected or mismatched group.
    UnexpectedMemberGroup {
        /// The supplied parent identity.
        group_identity: GroupScoreIntervalIdentity,
    },
    /// A member ID appears more than once in the second-stage metadata.
    DuplicateMemberRecord {
        /// Duplicate member identifier.
        member_id: String,
    },
    /// Not every member of a selected group has per-record interval metadata.
    IncompleteMemberCoverage {
        /// Parent group identity.
        group_identity: GroupScoreIntervalIdentity,
        /// Number of records expected in this group.
        expected_members: usize,
        /// Number of distinct member intervals supplied.
        supplied_members: usize,
    },
    /// A per-record interval and its parent group interval do not intersect.
    DisjointMemberAndGroupIntervals {
        /// Member whose intervals are disjoint.
        member_id: String,
    },
    /// The second-stage per-record selector rejected its input.
    MemberSelection(IntervalSelectionError),
}

impl std::fmt::Display for HierarchicalIntervalSelectionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::GroupSelection(error) => write!(f, "group selection failed: {error}"),
            Self::UnexpectedMemberGroup { .. } => {
                write!(f, "member interval belongs to an unexpected group snapshot")
            }
            Self::DuplicateMemberRecord { member_id } => {
                write!(f, "duplicate member interval for {member_id}")
            }
            Self::IncompleteMemberCoverage {
                expected_members,
                supplied_members,
                ..
            } => write!(
                f,
                "member interval coverage incomplete: expected={expected_members}, supplied={supplied_members}"
            ),
            Self::DisjointMemberAndGroupIntervals { member_id } => write!(
                f,
                "member and parent-group intervals are disjoint for {member_id}"
            ),
            Self::MemberSelection(error) => write!(f, "member selection failed: {error}"),
        }
    }
}

impl std::error::Error for HierarchicalIntervalSelectionError {}

/// Errors that prevent weighted group-level exact selection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GroupIntervalSelectionError {
    /// top_k must be in [1, eligible_records].
    InvalidTopK {
        /// Requested result count.
        top_k: usize,
        /// Eligible records in the snapshot.
        eligible_records: usize,
    },
    /// Group intervals do not cover the complete eligible population.
    IncompleteCoverage {
        /// Expected eligible record count.
        eligible_records: usize,
        /// Sum of supplied group member counts.
        covered_records: usize,
        /// Expected number of group intervals.
        interval_groups: usize,
        /// Number of groups supplied.
        supplied_groups: usize,
        /// Unopened groups without safe aggregate bounds.
        unbounded_unopened_groups: usize,
    },
    /// A group has no members.
    EmptyGroup {
        /// Index of the invalid group.
        group_index: usize,
    },
    /// A group interval is malformed.
    InvalidInterval {
        /// Index of the invalid group.
        group_index: usize,
    },
    /// Group IDs are not unique.
    DuplicateGroupId {
        /// First occurrence index.
        first_index: usize,
        /// Later duplicate index.
        duplicate_index: usize,
    },
    /// Group member IDs overlap, so the group partition is not disjoint.
    DuplicateMemberAcrossGroups {
        /// Stable member identifier appearing in multiple groups.
        member_id: String,
        /// First group containing the member.
        first_group_index: usize,
        /// Later group containing the member.
        duplicate_group_index: usize,
    },
    /// Group member metadata lengths do not agree.
    InvalidGroupMembership {
        /// Group index with inconsistent metadata.
        group_index: usize,
    },
    /// One interval was built under a different snapshot generation.
    SnapshotGenerationMismatch {
        /// Group index that disagreed with the pinned generation.
        group_index: usize,
    },
    /// One interval was calculated for a different query vector.
    QueryDigestMismatch {
        /// Group index that disagreed with the pinned query.
        group_index: usize,
    },
    /// One group interval was derived under another eligibility policy/scope.
    EligibilityScopeMismatch {
        /// Group index that disagreed with the pinned eligibility binding.
        group_index: usize,
    },
    /// The supplied partition receipt does not match the group directory.
    PartitionReceipt(GroupPartitionReceiptError),
    /// Summing group member counts overflowed usize.
    MemberCountOverflow,
}

impl std::fmt::Display for GroupIntervalSelectionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidTopK { top_k, eligible_records } => write!(
                f,
                "top_k {top_k} is invalid for {eligible_records} eligible records"
            ),
            Self::IncompleteCoverage {
                eligible_records,
                covered_records,
                interval_groups,
                supplied_groups,
                unbounded_unopened_groups,
            } => write!(
                f,
                "group interval coverage incomplete: eligible={eligible_records}, covered={covered_records}, expected_groups={interval_groups}, supplied_groups={supplied_groups}, unbounded_unopened_groups={unbounded_unopened_groups}"
            ),
            Self::EmptyGroup { group_index } => {
                write!(f, "group {group_index} has no eligible members")
            }
            Self::InvalidInterval { group_index } => {
                write!(f, "group {group_index} has an invalid score interval")
            }
            Self::DuplicateGroupId { first_index, duplicate_index } => write!(
                f,
                "duplicate group ID at indices {first_index} and {duplicate_index}"
            ),
            Self::DuplicateMemberAcrossGroups {
                member_id,
                first_group_index,
                duplicate_group_index,
            } => write!(
                f,
                "member {member_id} appears in groups {first_group_index} and {duplicate_group_index}"
            ),
            Self::InvalidGroupMembership { group_index } => {
                write!(f, "group {group_index} has inconsistent member metadata")
            }
            Self::SnapshotGenerationMismatch { group_index } => {
                write!(f, "group {group_index} belongs to another manifest or eligibility generation")
            }
            Self::QueryDigestMismatch { group_index } => {
                write!(f, "group {group_index} was scored for another query vector")
            }
            Self::EligibilityScopeMismatch { group_index } => {
                write!(f, "group {group_index} was built under another eligibility scope")
            }
            Self::PartitionReceipt(error) => write!(f, "group partition receipt rejected: {error}"),
            Self::MemberCountOverflow => write!(f, "group member count overflow"),
        }
    }
}

impl std::error::Error for GroupIntervalSelectionError {}

/// Structural errors that prevent interval-first selection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IntervalSelectionError {
    /// top_k must be in the closed interval [1, eligible_records].
    InvalidTopK {
        /// Requested result count.
        top_k: usize,
        /// Number of eligible records in the input.
        eligible_records: usize,
    },
    /// Supplied intervals do not cover exactly the eligible record population.
    IncompleteCoverage {
        /// Expected eligible record count.
        eligible_records: usize,
        /// Number of records supplied.
        supplied_records: usize,
        /// Number claimed to have intervals.
        interval_records: usize,
        /// Unopened partitions without safe aggregate bounds.
        unbounded_unopened_partitions: usize,
    },
    /// One or more records carry an invalid, non-finite or out-of-range interval.
    InvalidInterval {
        /// Index of the first invalid input record.
        record_index: usize,
    },
    /// An exact result score is non-finite or outside the canonical score range.
    InvalidExactScore {
        /// Index of the first invalid input score.
        record_index: usize,
    },
    /// Stable IDs are not unique within the eligible population.
    DuplicateId {
        /// First occurrence of the duplicate ID.
        first_index: usize,
        /// Later occurrence of the same ID.
        duplicate_index: usize,
    },
}

impl std::fmt::Display for IntervalSelectionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidTopK { top_k, eligible_records } => write!(
                f,
                "top_k {top_k} is invalid for {eligible_records} eligible records"
            ),
            Self::IncompleteCoverage {
                eligible_records,
                supplied_records,
                interval_records,
                unbounded_unopened_partitions,
            } => write!(
                f,
                "interval coverage incomplete: eligible={eligible_records}, supplied={supplied_records}, intervals={interval_records}, unbounded_partitions={unbounded_unopened_partitions}"
            ),
            Self::InvalidInterval { record_index } => {
                write!(f, "record {record_index} has an invalid score interval")
            }
            Self::InvalidExactScore { record_index } => {
                write!(f, "record {record_index} has an invalid exact score")
            }
            Self::DuplicateId { first_index, duplicate_index } => write!(
                f,
                "duplicate record ID at indices {first_index} and {duplicate_index}"
            ),
        }
    }
}

impl std::error::Error for IntervalSelectionError {}

/// Compute the complete refinement set for larger-is-better similarity.
///
/// For intervals [L_i, U_i], this computes theta = kthLargest(L_i), retains
/// every record with U_i >= theta, and excludes only records with U_i < theta.
/// Equality is deliberately retained. Stable IDs break equal-lower-bound
/// ordering deterministically when selecting the scalar threshold, although
/// the threshold value itself is unaffected by that tie break.
///
/// This function does not support approximate/empirical intervals as proof.
/// It requires every eligible record to have a well-formed interval and
/// rejects incomplete record coverage or any unbounded unopened partition.
///
/// # Type-level boundary
///
/// The function returns a selection plan, not an ExactComplete status. The
/// caller remains responsible for validating interval provenance and exact
/// corpus/policy generations, loading and scoring every refinement payload,
/// and stable score-descending / ID-ascending reranking.
pub fn select_refinement_set<Id>(
    records: &[ScoreIntervalRecord<Id>],
    top_k: usize,
    coverage: IntervalCoverageCounts,
) -> Result<IntervalSelectionPlan<Id>, IntervalSelectionError>
where
    Id: Ord + Clone,
{
    let record_count = records.len();
    if coverage.eligible_records != record_count
        || coverage.interval_records != record_count
        || coverage.unbounded_unopened_partitions != 0
    {
        return Err(IntervalSelectionError::IncompleteCoverage {
            eligible_records: coverage.eligible_records,
            supplied_records: record_count,
            interval_records: coverage.interval_records,
            unbounded_unopened_partitions: coverage.unbounded_unopened_partitions,
        });
    }

    if top_k == 0 || top_k > record_count {
        return Err(IntervalSelectionError::InvalidTopK {
            top_k,
            eligible_records: record_count,
        });
    }

    let mut first_index_by_id = BTreeMap::new();
    let mut lower_order: Vec<&ScoreIntervalRecord<Id>> = Vec::with_capacity(record_count);
    for (record_index, record) in records.iter().enumerate() {
        if !record.interval.is_well_formed() {
            return Err(IntervalSelectionError::InvalidInterval { record_index });
        }
        if let Some(first_index) = first_index_by_id.get(&record.id).copied() {
            return Err(IntervalSelectionError::DuplicateId {
                first_index,
                duplicate_index: record_index,
            });
        }
        first_index_by_id.insert(record.id.clone(), record_index);
        lower_order.push(record);
    }

    lower_order.sort_by(|left, right| {
        let lower_order = if left.interval.lower == right.interval.lower {
            std::cmp::Ordering::Equal
        } else {
            right.interval.lower.total_cmp(&left.interval.lower)
        };
        lower_order.then_with(|| left.id.cmp(&right.id))
    });
    let threshold = lower_order[top_k - 1].interval.lower;

    let mut refinement_ids = Vec::new();
    let mut excluded_ids = Vec::new();
    for record in records {
        if record.interval.upper >= threshold {
            refinement_ids.push(record.id.clone());
        } else {
            // Strict comparison is essential: boundary equality stays in the
            // refinement set unless a separate canonical-ID dominance proof
            // can establish exclusion.
            excluded_ids.push(record.id.clone());
        }
    }
    refinement_ids.sort();
    excluded_ids.sort();

    Ok(IntervalSelectionPlan {
        threshold,
        refinement_ids,
        excluded_ids,
    })
}


/// Select whole groups using their conservative intervals and member counts.
///
/// If each group g contains m_g eligible records and every member score lies in
/// [L_g, U_g], compute theta as the kth-largest value in the multiset that
/// repeats L_g exactly m_g times. At least k eligible records therefore have
/// exact score >= theta. Any group with U_g < theta cannot contain a top-k
/// member and may be excluded. Equality is retained.
///
/// This reduces global synopsis scanning from one interval per memory to one
/// interval per group. It is exact only with complete eligible-group coverage,
/// a pinned membership/eligibility snapshot, and a genuine group enclosure.
/// The public constructor for GroupScoreIntervalRecord revalidates the
/// ContinuousHVGroupEnvelope snapshot before it supplies the interval.
/// Select groups using a full member-roster disjointness scan.
///
/// This is the defensive path for untrusted/in-memory inputs. It performs
/// O(number of eligible member IDs) partition validation on every call.
pub fn select_group_refinement_set(
    groups: &[GroupScoreIntervalRecord],
    top_k: usize,
    coverage: GroupIntervalCoverageCounts,
) -> Result<GroupIntervalSelectionPlan, GroupIntervalSelectionError> {
    select_group_refinement_set_impl(groups, top_k, coverage, true)
}

/// Select groups using a previously verified build-time partition receipt.
///
/// The receipt proves group membership disjointness and completeness for one
/// manifest/eligibility generation. Query-time validation is O(number of
/// groups log groups): it checks group identities, counts, query digests and
/// snapshot fields, without walking each member ID. A persisted receipt must be
/// authenticated against the immutable segment manifest by #10917 after reopen.
pub fn select_group_refinement_set_with_partition_receipt(
    groups: &[GroupScoreIntervalRecord],
    top_k: usize,
    coverage: GroupIntervalCoverageCounts,
    receipt: &GroupPartitionCoverageReceipt,
) -> Result<GroupIntervalSelectionPlan, GroupIntervalSelectionError> {
    receipt
        .verify_group_records(groups)
        .map_err(GroupIntervalSelectionError::PartitionReceipt)?;
    if coverage.scope_binding != receipt.scope_binding {
        return Err(GroupIntervalSelectionError::PartitionReceipt(
            GroupPartitionReceiptError::EligibilityScopeMismatch,
        ));
    }
    if coverage.expected_manifest_generation != receipt.manifest_generation
        || coverage.expected_eligibility_generation != receipt.eligibility_generation
        || coverage.eligible_records != receipt.eligible_records
        || coverage.covered_records != receipt.eligible_records
        || coverage.interval_groups != receipt.group_count
    {
        return Err(GroupIntervalSelectionError::IncompleteCoverage {
            eligible_records: receipt.eligible_records,
            covered_records: coverage.covered_records,
            interval_groups: receipt.group_count,
            supplied_groups: groups.len(),
            unbounded_unopened_groups: coverage.unbounded_unopened_groups,
        });
    }
    select_group_refinement_set_impl(groups, top_k, coverage, false)
}

fn select_group_refinement_set_impl(
    groups: &[GroupScoreIntervalRecord],
    top_k: usize,
    coverage: GroupIntervalCoverageCounts,
    verify_member_disjointness: bool,
) -> Result<GroupIntervalSelectionPlan, GroupIntervalSelectionError> {
    let mut covered_records = 0usize;
    let mut first_index_by_id = BTreeMap::new();
    let mut first_group_by_member_id = BTreeMap::new();
    let mut lower_order: Vec<&GroupScoreIntervalRecord> =
        Vec::with_capacity(groups.len());
    for (group_index, group) in groups.iter().enumerate() {
        if group.member_count == 0 {
            return Err(GroupIntervalSelectionError::EmptyGroup { group_index });
        }
        if group.member_count != group.member_ids.len()
            || group.member_ids.len() != group.member_payload_digests.len()
        {
            return Err(GroupIntervalSelectionError::InvalidGroupMembership { group_index });
        }
        if group.identity.manifest_generation != coverage.expected_manifest_generation
            || group.identity.eligibility_generation != coverage.expected_eligibility_generation
        {
            return Err(GroupIntervalSelectionError::SnapshotGenerationMismatch { group_index });
        }
        if group.query_digest != coverage.query_digest {
            return Err(GroupIntervalSelectionError::QueryDigestMismatch { group_index });
        }
        if group.scope_binding != coverage.scope_binding {
            return Err(GroupIntervalSelectionError::EligibilityScopeMismatch { group_index });
        }
        if !group.interval.is_well_formed() {
            return Err(GroupIntervalSelectionError::InvalidInterval { group_index });
        }
        if let Some(first_index) = first_index_by_id.get(&group.identity.member_set_digest).copied() {
            return Err(GroupIntervalSelectionError::DuplicateGroupId {
                first_index,
                duplicate_index: group_index,
            });
        }
        first_index_by_id.insert(group.identity.member_set_digest, group_index);
        if verify_member_disjointness {
            for member_id in group.member_ids.iter() {
                if let Some(first_group_index) =
                    first_group_by_member_id.get(member_id.as_str()).copied()
                {
                    return Err(GroupIntervalSelectionError::DuplicateMemberAcrossGroups {
                        member_id: member_id.clone(),
                        first_group_index,
                        duplicate_group_index: group_index,
                    });
                }
                first_group_by_member_id.insert(member_id.as_str(), group_index);
            }
        }
        covered_records = covered_records
            .checked_add(group.member_count)
            .ok_or(GroupIntervalSelectionError::MemberCountOverflow)?;
        lower_order.push(group);
    }

    if coverage.unbounded_unopened_groups != 0
        || coverage.interval_groups != groups.len()
        || coverage.eligible_records != covered_records
        || coverage.covered_records != covered_records
    {
        return Err(GroupIntervalSelectionError::IncompleteCoverage {
            eligible_records: coverage.eligible_records,
            covered_records,
            interval_groups: coverage.interval_groups,
            supplied_groups: groups.len(),
            unbounded_unopened_groups: coverage.unbounded_unopened_groups,
        });
    }

    if top_k == 0 || top_k > covered_records {
        return Err(GroupIntervalSelectionError::InvalidTopK {
            top_k,
            eligible_records: covered_records,
        });
    }

    lower_order.sort_by(|left, right| {
        let lower_order = if left.interval.lower == right.interval.lower {
            std::cmp::Ordering::Equal
        } else {
            right.interval.lower.total_cmp(&left.interval.lower)
        };
        lower_order.then_with(|| left.identity.cmp(&right.identity))
    });

    let mut cumulative_members = 0usize;
    let mut threshold = None;
    for group in &lower_order {
        cumulative_members = cumulative_members
            .checked_add(group.member_count)
            .ok_or(GroupIntervalSelectionError::MemberCountOverflow)?;
        if cumulative_members >= top_k {
            threshold = Some(group.interval.lower);
            break;
        }
    }
    let threshold = threshold.ok_or(GroupIntervalSelectionError::InvalidTopK {
        top_k,
        eligible_records: covered_records,
    })?;

    let mut refinement_groups = Vec::new();
    let mut excluded_groups = Vec::new();
    let mut refinement_record_count = 0usize;
    let mut excluded_record_count = 0usize;
    for group in groups {
        if group.interval.upper >= threshold {
            refinement_groups.push(group.identity);
            refinement_record_count = refinement_record_count
                .checked_add(group.member_count)
                .ok_or(GroupIntervalSelectionError::MemberCountOverflow)?;
        } else {
            excluded_groups.push(group.identity);
            excluded_record_count = excluded_record_count
                .checked_add(group.member_count)
                .ok_or(GroupIntervalSelectionError::MemberCountOverflow)?;
        }
    }
    refinement_groups.sort();
    excluded_groups.sort();

    Ok(GroupIntervalSelectionPlan {
        threshold,
        manifest_generation: coverage.expected_manifest_generation,
        eligibility_generation: coverage.expected_eligibility_generation,
        query_digest: coverage.query_digest,
        scope_binding: coverage.scope_binding,
        refinement_groups,
        excluded_groups,
        refinement_record_count,
        excluded_record_count,
    })
}



/// Perform hierarchical group-then-member selection with defensive per-query
/// member-disjointness validation.
pub fn select_hierarchical_refinement_set(
    groups: &[GroupScoreIntervalRecord],
    member_records: &[GroupMemberScoreIntervalRecord],
    top_k: usize,
    coverage: GroupIntervalCoverageCounts,
) -> Result<HierarchicalIntervalSelectionPlan, HierarchicalIntervalSelectionError> {
    select_hierarchical_refinement_set_impl(
        groups,
        member_records,
        top_k,
        coverage,
        None,
    )
}

/// Perform hierarchical selection using an authenticated build-time partition
/// receipt. Global group selection is O(G); member rosters are only traversed
/// for groups that survive stage one.
pub fn select_hierarchical_refinement_set_with_partition_receipt(
    groups: &[GroupScoreIntervalRecord],
    member_records: &[GroupMemberScoreIntervalRecord],
    top_k: usize,
    coverage: GroupIntervalCoverageCounts,
    receipt: &GroupPartitionCoverageReceipt,
) -> Result<HierarchicalIntervalSelectionPlan, HierarchicalIntervalSelectionError> {
    select_hierarchical_refinement_set_impl(
        groups,
        member_records,
        top_k,
        coverage,
        Some(receipt),
    )
}

/// Perform hierarchical interval-first refinement.
///
/// Stage 1 scans one verified interval per group and uses weighted lower
/// endpoints to exclude entire groups. Stage 2 requires per-record intervals
/// only for surviving groups. Each member interval is intersected with its
/// parent group interval before the existing global member selector is run.
///
/// The group lower endpoint is a valid lower bound for every group member.
/// Thus at least top_k members have exact scores >= the first-stage threshold
/// and are all in surviving groups. Intersecting valid member/group intervals
/// preserves containment and raises every affected member lower endpoint to at
/// least its group lower endpoint. The second-stage kth lower threshold is
/// therefore no smaller than the first-stage threshold. Exact reranking only
/// the final member refinement IDs is sufficient for whole-corpus top-k.
///
/// The caller must read per-record interval metadata for every member of every
/// surviving group, but it need not load original vector payloads until the
/// final member refinement set is known. Incomplete membership coverage or a
/// disjoint intersection fails closed.
fn select_hierarchical_refinement_set_impl(
    groups: &[GroupScoreIntervalRecord],
    member_records: &[GroupMemberScoreIntervalRecord],
    top_k: usize,
    coverage: GroupIntervalCoverageCounts,
    partition_receipt: Option<&GroupPartitionCoverageReceipt>,
) -> Result<HierarchicalIntervalSelectionPlan, HierarchicalIntervalSelectionError> {
    let group_plan = match partition_receipt {
        Some(receipt) => select_group_refinement_set_with_partition_receipt(
            groups,
            top_k,
            coverage,
            receipt,
        ),
        None => select_group_refinement_set(groups, top_k, coverage),
    }
    .map_err(HierarchicalIntervalSelectionError::GroupSelection)?;

    let selected_identities: BTreeSet<GroupScoreIntervalIdentity> =
        group_plan.refinement_groups.iter().copied().collect();
    let selected_groups: BTreeMap<GroupScoreIntervalIdentity, &GroupScoreIntervalRecord> = groups
        .iter()
        .filter(|group| selected_identities.contains(&group.identity))
        .map(|group| (group.identity, group))
        .collect();

    let mut records_by_group: BTreeMap<
        GroupScoreIntervalIdentity,
        BTreeMap<String, &GroupMemberScoreIntervalRecord>,
    > = BTreeMap::new();

    for member in member_records {
        if member.query_digest != group_plan.query_digest {
            return Err(HierarchicalIntervalSelectionError::UnexpectedMemberGroup {
                group_identity: member.group_identity,
            });
        }
        let Some(group) = selected_groups.get(&member.group_identity).copied() else {
            return Err(HierarchicalIntervalSelectionError::UnexpectedMemberGroup {
                group_identity: member.group_identity,
            });
        };
        if group.member_ids.binary_search_by(|id| id.as_str().cmp(member.id())).is_err() {
            return Err(HierarchicalIntervalSelectionError::UnexpectedMemberGroup {
                group_identity: member.group_identity,
            });
        }
        let group_records = records_by_group.entry(member.group_identity).or_default();
        if group_records.insert(member.id().to_owned(), member).is_some() {
            return Err(HierarchicalIntervalSelectionError::DuplicateMemberRecord {
                member_id: member.id().to_owned(),
            });
        }
    }

    let mut effective_records = Vec::with_capacity(group_plan.refinement_record_count);
    for identity in &group_plan.refinement_groups {
        let group = selected_groups.get(identity).copied().ok_or(
            HierarchicalIntervalSelectionError::UnexpectedMemberGroup {
                group_identity: *identity,
            },
        )?;
        let supplied = records_by_group.get(identity);
        let supplied_count = supplied.map_or(0, BTreeMap::len);
        if supplied_count != group.member_count {
            return Err(HierarchicalIntervalSelectionError::IncompleteMemberCoverage {
                group_identity: *identity,
                expected_members: group.member_count,
                supplied_members: supplied_count,
            });
        }
        let supplied = supplied.expect("validated nonzero group coverage above");
        for member_id in &group.member_ids {
            let member = supplied.get(member_id).ok_or(
                HierarchicalIntervalSelectionError::IncompleteMemberCoverage {
                    group_identity: *identity,
                    expected_members: group.member_count,
                    supplied_members: supplied_count,
                },
            )?;
            let member_interval = member.record.interval;
            let lower = member_interval.lower.max(group.interval.lower);
            let upper = member_interval.upper.min(group.interval.upper);
            if lower > upper {
                return Err(
                    HierarchicalIntervalSelectionError::DisjointMemberAndGroupIntervals {
                        member_id: member_id.clone(),
                    },
                );
            }
            effective_records.push(ScoreIntervalRecord {
                id: member.record.id.clone(),
                interval: ContinuousHVScoreInterval { lower, upper },
                certificate_check: member.record.certificate_check,
            });
        }
    }

    let member_plan = select_refinement_set(
        &effective_records,
        top_k,
        IntervalCoverageCounts {
            eligible_records: group_plan.refinement_record_count,
            interval_records: effective_records.len(),
            unbounded_unopened_partitions: 0,
        },
    )
    .map_err(HierarchicalIntervalSelectionError::MemberSelection)?;

    Ok(HierarchicalIntervalSelectionPlan {
        group_plan,
        member_plan,
    })
}


/// Apply the canonical score-descending / stable-ID-ascending top-k order.
///
/// Numerically equal scores use the ID tiebreak; in particular, +0.0 and -0.0
/// compare as equal scores instead of being ordered by their sign bit. The
/// canonical score profile guarantees finite values in [-1, 1], and this
/// helper rejects violations rather than using a NaN-tolerant comparator.
pub fn stable_exact_top_k<Id>(
    scores: &[ExactScoreRecord<Id>],
    top_k: usize,
) -> Result<Vec<Id>, IntervalSelectionError>
where
    Id: Ord + Clone,
{
    if top_k == 0 || top_k > scores.len() {
        return Err(IntervalSelectionError::InvalidTopK {
            top_k,
            eligible_records: scores.len(),
        });
    }

    let mut first_index_by_id = BTreeMap::new();
    let mut ordered: Vec<&ExactScoreRecord<Id>> = Vec::with_capacity(scores.len());
    for (record_index, record) in scores.iter().enumerate() {
        if !record.score.is_finite() || !(-1.0..=1.0).contains(&record.score) {
            return Err(IntervalSelectionError::InvalidExactScore { record_index });
        }
        if let Some(first_index) = first_index_by_id.get(&record.id).copied() {
            return Err(IntervalSelectionError::DuplicateId {
                first_index,
                duplicate_index: record_index,
            });
        }
        first_index_by_id.insert(record.id.clone(), record_index);
        ordered.push(record);
    }

    ordered.sort_by(|left, right| {
        let score_order = if left.score == right.score {
            std::cmp::Ordering::Equal
        } else {
            right.score.total_cmp(&left.score)
        };
        score_order.then_with(|| left.id.cmp(&right.id))
    });

    Ok(ordered
        .into_iter()
        .take(top_k)
        .map(|record| record.id.clone())
        .collect())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;

    fn record(id: &str, lower: f32, upper: f32) -> ScoreIntervalRecord<String> {
        // Synthetic fixtures are built inside this test module only. Production
        // callers cannot construct a record from arbitrary interval endpoints.
        ScoreIntervalRecord {
            id: id.to_owned(),
            interval: ContinuousHVScoreInterval { lower, upper },
            certificate_check: ResidualCertificateVerification {
                residual_norm_squared_upper: 0.0,
                certified_norm_squared: 0.0,
                changed_coordinates: 0,
            },
        }
    }

    fn complete_coverage(count: usize) -> IntervalCoverageCounts {
        IntervalCoverageCounts {
            eligible_records: count,
            interval_records: count,
            unbounded_unopened_partitions: 0,
        }
    }


    fn synthetic_group(
        id: &str,
        member_count: usize,
        lower: f32,
        upper: f32,
    ) -> GroupScoreIntervalRecord {
        let mut member_set_digest = [0u8; 32];
        for (index, byte) in id.bytes().enumerate() {
            member_set_digest[index % 32] ^= byte;
        }
        let mut content_digest = member_set_digest;
        content_digest[31] ^= 0xA5;
        let member_ids: Vec<String> = (0..member_count)
            .map(|index| format!("{id}-member-{index:04}"))
            .collect();
        let member_payload_digests: Vec<[u8; 32]> = (0..member_count)
            .map(|index| {
                let mut digest = member_set_digest;
                digest[0] ^= index as u8;
                digest[1] ^= (index >> 8) as u8;
                digest
            })
            .collect();
        GroupScoreIntervalRecord {
            identity: GroupScoreIntervalIdentity {
                manifest_generation: 1,
                eligibility_generation: 1,
                member_set_digest,
                content_digest,
            },
            member_ids: Arc::from(member_ids),
            member_payload_digests: Arc::from(member_payload_digests),
            query_digest: [0x51u8; 32],
            scope_binding: test_scope_binding(),
            member_count,
            interval: ContinuousHVScoreInterval { lower, upper },
        }
    }

    fn test_scope_binding() -> HmeEligibilityScopeBinding {
        HmeEligibilityScopeBinding {
            principal_scope_digest: [0x11u8; 32],
            policy_digest: [0x22u8; 32],
            policy_generation: 6,
            source_ledger_generation: 91,
            bitemporal_query_digest: [0x33u8; 32],
        }
    }

    fn complete_group_coverage(
        groups: &[GroupScoreIntervalRecord],
    ) -> GroupIntervalCoverageCounts {
        let members = groups.iter().map(|group| group.member_count).sum();
        let first = groups.first().expect("synthetic coverage requires a group");
        GroupIntervalCoverageCounts {
            eligible_records: members,
            covered_records: members,
            interval_groups: groups.len(),
            expected_manifest_generation: first.identity.manifest_generation,
            expected_eligibility_generation: first.identity.eligibility_generation,
            query_digest: first.query_digest,
            scope_binding: test_scope_binding(),
            unbounded_unopened_groups: 0,
        }
    }

    fn receipt_for_fixture_envelopes(
        envelopes: &[ContinuousHVGroupEnvelope],
    ) -> GroupPartitionCoverageReceipt {
        let all_ids: Vec<String> = envelopes
            .iter()
            .flat_map(|envelope| envelope.member_ids().iter().cloned())
            .collect();
        let authoritative_digest = member_id_set_digest(&all_ids).unwrap();
        GroupPartitionCoverageReceipt::from_group_envelopes(
            envelopes,
            all_ids.len(),
            authoritative_digest,
            test_scope_binding(),
        )
        .unwrap()
    }

    #[test]
    fn group_selection_record_is_bound_to_the_verified_envelope_identity() {
        let members = vec![
            ("memory-a".to_owned(), ContinuousHV::from_vec(vec![0.8; 8])),
            ("memory-b".to_owned(), ContinuousHV::from_vec(vec![0.7; 8])),
        ];
        let envelope = ContinuousHVGroupEnvelope::from_original_members(12, 4, &members).unwrap();
        let query = ContinuousHV::from_vec(vec![1.0; 8]);
        let record = GroupScoreIntervalRecord::from_group_envelope(
            &envelope,
            &query,
            12,
            4,
            envelope.member_set_digest(),
            envelope.content_digest(),
            test_scope_binding(),
        )
        .unwrap();

        assert_eq!(record.member_count(), 2);
        assert_eq!(record.identity().manifest_generation, 12);
        assert_eq!(record.identity().eligibility_generation, 4);
        assert_eq!(record.identity().member_set_digest, envelope.member_set_digest());
        assert_eq!(record.identity().content_digest, envelope.content_digest());
        assert!(record.interval().is_well_formed());

        let stale = GroupScoreIntervalRecord::from_group_envelope(
            &envelope,
            &query,
            11,
            4,
            envelope.member_set_digest(),
            envelope.content_digest(),
            test_scope_binding(),
        );
        assert!(matches!(
            stale,
            Err(GroupScoreIntervalBuildError::GroupEnvelope(
                GroupEnvelopeError::ManifestGenerationMismatch { .. }
            ))
        ));
    }

    fn unit_vector_with_first_component(score: f32) -> ContinuousHV {
        let mut values = vec![0.0f32; 8];
        values[0] = score;
        values[1] = (1.0 - score * score).sqrt();
        ContinuousHV::from_vec(values)
    }

    fn build_group_record(
        envelope: &ContinuousHVGroupEnvelope,
        query: &ContinuousHV,
    ) -> GroupScoreIntervalRecord {
        GroupScoreIntervalRecord::from_group_envelope(
            envelope,
            query,
            envelope.manifest_generation(),
            envelope.eligibility_generation(),
            envelope.member_set_digest(),
            envelope.content_digest(),
            test_scope_binding(),
        )
        .unwrap()
    }

    fn build_member_record(
        group: &GroupScoreIntervalRecord,
        id: &str,
        original: &ContinuousHV,
        query: &ContinuousHV,
    ) -> GroupMemberScoreIntervalRecord {
        GroupMemberScoreIntervalRecord::from_verified_l2_residual(
            group,
            id.to_owned(),
            query,
            original,
            original,
            0.0,
        )
        .unwrap()
    }

    #[test]
    fn hierarchical_group_then_member_refinement_matches_full_exact_top_k() {
        let query = ContinuousHV::from_vec(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let high_members = vec![
            ("high-a".to_owned(), unit_vector_with_first_component(0.96)),
            ("high-b".to_owned(), unit_vector_with_first_component(0.959)),
            ("high-c".to_owned(), unit_vector_with_first_component(0.90)),
        ];
        let middle_members = vec![
            ("middle-a".to_owned(), unit_vector_with_first_component(0.95)),
            ("middle-b".to_owned(), unit_vector_with_first_component(0.10)),
        ];
        let low_members = vec![
            ("low-a".to_owned(), unit_vector_with_first_component(-0.80)),
            ("low-b".to_owned(), unit_vector_with_first_component(-0.90)),
        ];
        let envelopes = vec![
            ContinuousHVGroupEnvelope::from_original_members(20, 3, &high_members).unwrap(),
            ContinuousHVGroupEnvelope::from_original_members(20, 3, &middle_members).unwrap(),
            ContinuousHVGroupEnvelope::from_original_members(20, 3, &low_members).unwrap(),
        ];
        let partition_receipt =
            receipt_for_fixture_envelopes(&envelopes);
        let group_records: Vec<GroupScoreIntervalRecord> = envelopes
            .iter()
            .map(|envelope| build_group_record(envelope, &query))
            .collect();
        let coverage = complete_group_coverage(&group_records);
        let group_plan = select_group_refinement_set_with_partition_receipt(
            &group_records,
            2,
            coverage,
            &partition_receipt,
        )
        .unwrap();

        assert!(group_plan.refinement_groups.contains(&group_records[0].identity));
        assert!(group_plan.refinement_groups.contains(&group_records[1].identity));
        assert!(group_plan.excluded_groups.contains(&group_records[2].identity));

        // Stage two reads compact per-member reconstructions/certificates only
        // for members of surviving groups. The low group's original payloads
        // are never presented to the member selector.
        let mut member_records = Vec::new();
        let transform_digest = [0x5Au8; 32];
        for (group_index, members) in [(0usize, &high_members), (1usize, &middle_members)] {
            for (id, original) in members {
                // Original payloads are consumed at admission time to create
                // an opaque residual receipt. The query-time interval builder
                // receives only the compact reconstruction plus that receipt.
                let receipt = VerifiedMemberResidualCertificate::verify_at_ingest(
                    &group_records[group_index],
                    id.clone(),
                    original,
                    original,
                    0.0,
                    transform_digest,
                )
                .unwrap();
                member_records.push(
                    GroupMemberScoreIntervalRecord::from_verified_certificate(
                        &group_records[group_index],
                        &query,
                        original,
                        &receipt,
                        transform_digest,
                    )
                    .unwrap(),
                );
            }
        }

        let plan = select_hierarchical_refinement_set_with_partition_receipt(
            &group_records,
            &member_records,
            2,
            coverage,
            &partition_receipt,
        )
        .unwrap();
        assert_eq!(plan.group_plan, group_plan);
        let expected_second_score = query.similarity_full_resolution(&high_members[1].1);
        assert_eq!(plan.member_plan.threshold, expected_second_score);
        assert_eq!(
            plan.member_plan.refinement_ids,
            vec!["high-a".to_owned(), "high-b".to_owned()]
        );
        assert_eq!(
            plan.member_plan.excluded_ids,
            vec![
                "high-c".to_owned(),
                "middle-a".to_owned(),
                "middle-b".to_owned()
            ]
        );

        let full_scores: Vec<ExactScoreRecord<String>> = high_members
            .iter()
            .chain(&middle_members)
            .chain(&low_members)
            .map(|(id, vector)| ExactScoreRecord {
                id: id.clone(),
                score: query.similarity_full_resolution(vector),
            })
            .collect();
        let expected = stable_exact_top_k(&full_scores, 2).unwrap();
        assert_eq!(plan.member_plan.refinement_ids, expected);
    }

    #[test]
    fn generated_hierarchical_refinement_matches_full_exact_top_k() {
        let mut state = 0xC0FF_EE11u32;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        };

        for case_index in 0..256usize {
            let dimension = 8usize;
            let group_count = 2 + (next() as usize % 7);
            let top_k = 1 + (next() as usize % 12);
            let mut member_vectors: Vec<Vec<(String, ContinuousHV)>> =
                Vec::with_capacity(group_count);
            let mut reconstructed: Vec<Vec<(String, ContinuousHV, ContinuousHV)>> =
                Vec::with_capacity(group_count);
            let mut total_members = 0usize;

            for group_index in 0..group_count {
                let member_count = 1 + (next() as usize % 5);
                let mut group_members = Vec::with_capacity(member_count);
                let mut group_reconstructions = Vec::with_capacity(member_count);
                for member_index in 0..member_count {
                    let id = format!("case-{case_index:04}-g{group_index:02}-m{member_index:02}");
                    let original = ContinuousHV::from_vec(
                        (0..dimension)
                            .map(|_| ((next() % 1801) as f32 / 1000.0) - 0.9)
                            .collect(),
                    );
                    let reconstruction = ContinuousHV::from_vec(
                        original
                            .values
                            .iter()
                            .enumerate()
                            .map(|(index, value)| {
                                let signed = if (index + member_index + group_index) % 2 == 0 {
                                    1.0
                                } else {
                                    -1.0
                                };
                                (*value + signed * 0.0002).clamp(-1.0, 1.0)
                            })
                            .collect(),
                    );
                    group_members.push((id.clone(), original.clone()));
                    group_reconstructions.push((id, original, reconstruction));
                }
                total_members += member_count;
                member_vectors.push(group_members);
                reconstructed.push(group_reconstructions);
            }

            let query = ContinuousHV::from_vec(
                (0..dimension)
                    .map(|_| ((next() % 1801) as f32 / 1000.0) - 0.9)
                    .collect(),
            );
            let envelopes: Vec<ContinuousHVGroupEnvelope> = member_vectors
                .iter()
                .map(|members| {
                    ContinuousHVGroupEnvelope::from_original_members(
                        100 + case_index as u64,
                        20 + case_index as u64,
                        members,
                    )
                    .unwrap()
                })
                .collect();
            let partition_receipt =
                receipt_for_fixture_envelopes(&envelopes);
            let groups: Vec<GroupScoreIntervalRecord> = envelopes
                .iter()
                .map(|envelope| build_group_record(envelope, &query))
                .collect();
            let coverage = complete_group_coverage(&groups);
            let group_plan = select_group_refinement_set_with_partition_receipt(
                &groups,
                top_k.min(total_members),
                coverage,
                &partition_receipt,
            )
            .unwrap();
            let selected: BTreeSet<GroupScoreIntervalIdentity> =
                group_plan.refinement_groups.iter().copied().collect();

            let transform_digest = [case_index as u8; 32];
            let mut member_records = Vec::with_capacity(group_plan.refinement_record_count);
            for (group_index, members) in reconstructed.iter().enumerate() {
                if !selected.contains(&groups[group_index].identity) {
                    continue;
                }
                for (id, original, reconstruction) in members {
                    let receipt = VerifiedMemberResidualCertificate::verify_at_ingest(
                        &groups[group_index],
                        id.clone(),
                        original,
                        reconstruction,
                        0.01,
                        transform_digest,
                    )
                    .unwrap();
                    member_records.push(
                        GroupMemberScoreIntervalRecord::from_verified_certificate(
                            &groups[group_index],
                            &query,
                            reconstruction,
                            &receipt,
                            transform_digest,
                        )
                        .unwrap(),
                    );
                }
            }

            let k = top_k.min(total_members);
            let plan = select_hierarchical_refinement_set_with_partition_receipt(
                &groups,
                &member_records,
                k,
                coverage,
                &partition_receipt,
            )
            .unwrap();

            let full_scores: Vec<ExactScoreRecord<String>> = member_vectors
                .iter()
                .flatten()
                .map(|(id, vector)| ExactScoreRecord {
                    id: id.clone(),
                    score: query.similarity_full_resolution(vector),
                })
                .collect();
            let expected = stable_exact_top_k(&full_scores, k).unwrap();
            let selected_records: BTreeSet<String> =
                plan.member_plan.refinement_ids.iter().cloned().collect();
            let refined_scores: Vec<ExactScoreRecord<String>> = full_scores
                .iter()
                .filter(|record| selected_records.contains(&record.id))
                .cloned()
                .collect();
            let actual = stable_exact_top_k(&refined_scores, k).unwrap();
            assert_eq!(
                actual, expected,
                "hierarchical selector false-pruned in generated case {case_index}"
            );
        }
    }

    #[test]
    fn hierarchical_selection_rejects_missing_member_intervals_and_foreign_groups() {
        let query = ContinuousHV::from_vec(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let high_members = vec![
            ("high-a".to_owned(), unit_vector_with_first_component(0.96)),
            ("high-b".to_owned(), unit_vector_with_first_component(0.959)),
            ("high-c".to_owned(), unit_vector_with_first_component(0.90)),
        ];
        let middle_members = vec![
            ("middle-a".to_owned(), unit_vector_with_first_component(0.95)),
            ("middle-b".to_owned(), unit_vector_with_first_component(0.10)),
        ];
        let low_members = vec![
            ("low-a".to_owned(), unit_vector_with_first_component(-0.80)),
            ("low-b".to_owned(), unit_vector_with_first_component(-0.90)),
        ];
        let envelopes = vec![
            ContinuousHVGroupEnvelope::from_original_members(20, 3, &high_members).unwrap(),
            ContinuousHVGroupEnvelope::from_original_members(20, 3, &middle_members).unwrap(),
            ContinuousHVGroupEnvelope::from_original_members(20, 3, &low_members).unwrap(),
        ];
        let groups: Vec<GroupScoreIntervalRecord> = envelopes
            .iter()
            .map(|envelope| build_group_record(envelope, &query))
            .collect();
        let coverage = complete_group_coverage(&groups);
        let group_plan = select_group_refinement_set(&groups, 2, coverage).unwrap();

        let one_member = vec![build_member_record(
            &groups[0],
            "high-a",
            &high_members[0].1,
            &query,
        )];
        assert!(matches!(
            select_hierarchical_refinement_set(&groups, &one_member, 2, coverage),
            Err(HierarchicalIntervalSelectionError::IncompleteMemberCoverage { .. })
        ));

        let mut all_selected = Vec::new();
        for (group_index, members) in [(0usize, &high_members), (1usize, &middle_members)] {
            for (id, original) in members {
                all_selected.push(build_member_record(
                    &groups[group_index],
                    id,
                    original,
                    &query,
                ));
            }
        }
        all_selected.push(build_member_record(
            &groups[2],
            "low-a",
            &low_members[0].1,
            &query,
        ));
        assert!(matches!(
            select_hierarchical_refinement_set(&groups, &all_selected, 2, coverage),
            Err(HierarchicalIntervalSelectionError::UnexpectedMemberGroup { .. })
        ));

        assert_eq!(group_plan.excluded_groups, vec![groups[2].identity]);
    }

    #[test]
    fn query_time_member_interval_uses_receipt_without_original_payload() {
        let query = ContinuousHV::from_vec(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let id = "receipt-member";
        let original = unit_vector_with_first_component(0.95);
        let reconstruction = unit_vector_with_first_component(0.949);
        let members = vec![(id.to_owned(), original.clone())];
        let envelope = ContinuousHVGroupEnvelope::from_original_members(30, 8, &members).unwrap();
        let group = build_group_record(&envelope, &query);
        let transform_digest = [0xA7u8; 32];
        let receipt = VerifiedMemberResidualCertificate::verify_at_ingest(
            &group,
            id.to_owned(),
            &original,
            &reconstruction,
            0.002,
            transform_digest,
        )
        .unwrap();

        // There is intentionally no original-payload argument in this query
        // operation. It checks the group/original digest binding and the
        // supplied compact reconstruction digest before producing the score
        // interval.
        let interval_record = GroupMemberScoreIntervalRecord::from_verified_certificate(
            &group,
            &query,
            &reconstruction,
            &receipt,
            transform_digest,
        )
        .unwrap();
        let exact = query.similarity_full_resolution(&original);
        assert!(interval_record.interval().contains(exact));
        assert_eq!(interval_record.id(), id);

        assert!(matches!(
            GroupMemberScoreIntervalRecord::from_verified_certificate(
                &group,
                &query,
                &original,
                &receipt,
                transform_digest,
            ),
            Err(GroupMemberScoreIntervalBuildError::VerifiedCertificate(
                VerifiedMemberResidualError::ReconstructionDigestMismatch
            ))
        ));
        assert!(matches!(
            GroupMemberScoreIntervalRecord::from_verified_certificate(
                &group,
                &query,
                &reconstruction,
                &receipt,
                [0u8; 32],
            ),
            Err(GroupMemberScoreIntervalBuildError::VerifiedCertificate(
                VerifiedMemberResidualError::TransformProfileMismatch
            ))
        ));
    }

    #[test]
    fn group_member_admission_rejects_payload_that_does_not_match_member_id() {
        let query = ContinuousHV::from_vec(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let original = unit_vector_with_first_component(0.95);
        let other_payload = unit_vector_with_first_component(0.25);
        let members = vec![("memory-a".to_owned(), original.clone())];
        let envelope = ContinuousHVGroupEnvelope::from_original_members(20, 3, &members).unwrap();
        let group = build_group_record(&envelope, &query);
        let result = GroupMemberScoreIntervalRecord::from_verified_l2_residual(
            &group,
            "memory-a".to_owned(),
            &query,
            &other_payload,
            &other_payload,
            0.0,
        );
        assert!(matches!(
            result,
            Err(GroupMemberScoreIntervalBuildError::OriginalPayloadDigestMismatch)
        ));
    }

    #[test]
    fn weighted_group_selection_counts_each_member_in_the_lower_threshold() {
        let groups = vec![
            synthetic_group("near", 2, 0.80, 0.95),
            synthetic_group("middle", 3, 0.40, 0.70),
            synthetic_group("far", 5, -0.10, 0.30),
        ];
        let plan = select_group_refinement_set(
            &groups,
            3,
            complete_group_coverage(&groups),
        )
        .unwrap();
        assert_eq!(plan.threshold, 0.40);
        assert_eq!(plan.refinement_groups.len(), 2);
        assert_eq!(plan.excluded_groups.len(), 1);
        assert_eq!(
            plan.refinement_groups.iter().map(|identity| identity.member_set_digest).collect::<BTreeSet<_>>(),
            vec![groups[0].identity.member_set_digest, groups[1].identity.member_set_digest]
                .into_iter()
                .collect()
        );
        assert_eq!(
            plan.excluded_groups[0].member_set_digest,
            groups[2].identity.member_set_digest
        );
        assert_eq!(plan.refinement_record_count, 5);
        assert_eq!(plan.excluded_record_count, 5);
    }

    #[test]
    fn group_boundary_equality_is_retained_and_output_is_order_independent() {
        let first = vec![
            synthetic_group("a", 1, 0.8, 0.9),
            synthetic_group("b", 1, 0.2, 0.8),
            synthetic_group("c", 2, -0.5, 0.1),
        ];
        let second = vec![first[2].clone(), first[1].clone(), first[0].clone()];
        let first_plan = select_group_refinement_set(
            &first,
            1,
            complete_group_coverage(&first),
        )
        .unwrap();
        let second_plan = select_group_refinement_set(
            &second,
            1,
            complete_group_coverage(&second),
        )
        .unwrap();
        assert_eq!(first_plan, second_plan);
        assert!(first_plan.refinement_groups.iter().any(|identity| {
            identity.member_set_digest == first[1].identity.member_set_digest
        }));
        assert!(!first_plan.excluded_groups.iter().any(|identity| {
            identity.member_set_digest == first[1].identity.member_set_digest
        }));
    }

    #[test]
    fn group_selection_rejects_incomplete_coverage_empty_groups_and_duplicate_ids() {
        let groups = vec![
            synthetic_group("a", 2, 0.2, 0.8),
            synthetic_group("b", 1, -0.2, 0.1),
        ];
        let mut incomplete = complete_group_coverage(&groups);
        incomplete.covered_records -= 1;
        assert!(matches!(
            select_group_refinement_set(&groups, 1, incomplete),
            Err(GroupIntervalSelectionError::IncompleteCoverage { .. })
        ));

        let empty = vec![synthetic_group("empty", 0, -1.0, 1.0)];
        assert!(matches!(
            select_group_refinement_set(&empty, 1, complete_group_coverage(&empty)),
            Err(GroupIntervalSelectionError::EmptyGroup { .. })
        ));

        let duplicate = vec![
            synthetic_group("same", 1, 0.0, 1.0),
            synthetic_group("same", 1, -1.0, 0.5),
        ];
        assert!(matches!(
            select_group_refinement_set(&duplicate, 1, complete_group_coverage(&duplicate)),
            Err(GroupIntervalSelectionError::DuplicateGroupId { .. })
        ));
    }

    #[test]
    fn build_time_partition_receipt_rejects_overlap_and_matches_fast_query_path() {
        let query = ContinuousHV::from_vec(vec![1.0; 8]);
        let shared = ContinuousHV::from_vec(vec![0.8; 8]);
        let first_members = vec![
            ("shared".to_owned(), shared.clone()),
            ("first-only".to_owned(), ContinuousHV::from_vec(vec![0.7; 8])),
        ];
        let second_members = vec![
            ("shared".to_owned(), shared.clone()),
            ("second-only".to_owned(), ContinuousHV::from_vec(vec![-0.7; 8])),
        ];
        let overlapping = vec![
            ContinuousHVGroupEnvelope::from_original_members(88, 12, &first_members).unwrap(),
            ContinuousHVGroupEnvelope::from_original_members(88, 12, &second_members).unwrap(),
        ];
        assert!(matches!(
            GroupPartitionCoverageReceipt::from_group_envelopes(
                &overlapping,
                4,
                [0u8; 32],
                test_scope_binding(),
            ),
            Err(GroupPartitionReceiptError::DuplicateMember {
                member_id,
                ..
            }) if member_id == "shared"
        ));

        let disjoint = vec![
            ContinuousHVGroupEnvelope::from_original_members(88, 12, &first_members).unwrap(),
            ContinuousHVGroupEnvelope::from_original_members(
                88,
                12,
                &[(
                    "second-only".to_owned(),
                    ContinuousHV::from_vec(vec![-0.7; 8]),
                )],
            )
            .unwrap(),
        ];
        let receipt = receipt_for_fixture_envelopes(&disjoint);
        let all_ids: Vec<String> = disjoint
            .iter()
            .flat_map(|envelope| envelope.member_ids().iter().cloned())
            .collect();
        let authoritative_digest = member_id_set_digest(&all_ids).unwrap();
        assert!(matches!(
            GroupPartitionCoverageReceipt::from_group_envelopes(
                &disjoint,
                all_ids.len() + 1,
                authoritative_digest,
                test_scope_binding(),
            ),
            Err(GroupPartitionReceiptError::EligiblePopulationCountMismatch {
                expected,
                actual
            }) if expected == all_ids.len() + 1 && actual == all_ids.len()
        ));
        assert!(matches!(
            GroupPartitionCoverageReceipt::from_group_envelopes(
                &disjoint,
                all_ids.len(),
                [0xFFu8; 32],
                test_scope_binding(),
            ),
            Err(GroupPartitionReceiptError::EligibleMemberSetDigestMismatch)
        ));
        let groups: Vec<GroupScoreIntervalRecord> = disjoint
            .iter()
            .map(|envelope| build_group_record(envelope, &query))
            .collect();
        let coverage = complete_group_coverage(&groups);
        let fast = select_group_refinement_set_with_partition_receipt(
            &groups,
            1,
            coverage,
            &receipt,
        )
        .unwrap();
        let mut wrong_scope = coverage;
        wrong_scope.scope_binding.policy_generation += 1;
        assert!(matches!(
            select_group_refinement_set_with_partition_receipt(
                &groups,
                1,
                wrong_scope,
                &receipt,
            ),
            Err(GroupIntervalSelectionError::PartitionReceipt(
                GroupPartitionReceiptError::EligibilityScopeMismatch
            ))
        ));
        let defensive = select_group_refinement_set(&groups, 1, coverage).unwrap();
        assert_eq!(fast, defensive);
        assert_eq!(receipt.group_count(), 2);
        assert_eq!(receipt.eligible_records(), 3);

        let mut tampered = groups.clone();
        tampered[0].identity.content_digest[0] ^= 0x80;
        assert!(matches!(
            select_group_refinement_set_with_partition_receipt(
                &tampered,
                1,
                coverage,
                &receipt,
            ),
            Err(GroupIntervalSelectionError::PartitionReceipt(
                GroupPartitionReceiptError::ReceiptMismatch
            ))
        ));
    }

    #[test]
    fn group_selector_rejects_overlapping_member_sets_even_when_counts_balance() {
        let query = ContinuousHV::from_vec(vec![1.0; 8]);
        let shared = ContinuousHV::from_vec(vec![0.8; 8]);
        let only_a = ContinuousHV::from_vec(vec![0.7; 8]);
        let only_b = ContinuousHV::from_vec(vec![-0.7; 8]);
        let first = ContinuousHVGroupEnvelope::from_original_members(
            44,
            9,
            &[("shared".to_owned(), shared.clone()), ("a".to_owned(), only_a)],
        )
        .unwrap();
        let second = ContinuousHVGroupEnvelope::from_original_members(
            44,
            9,
            &[("shared".to_owned(), shared), ("b".to_owned(), only_b)],
        )
        .unwrap();
        let groups = vec![
            build_group_record(&first, &query),
            build_group_record(&second, &query),
        ];
        assert!(matches!(
            select_group_refinement_set(&groups, 1, complete_group_coverage(&groups)),
            Err(GroupIntervalSelectionError::DuplicateMemberAcrossGroups {
                member_id,
                ..
            }) if member_id == "shared"
        ));
    }

    #[test]
    fn group_selector_rejects_group_interval_built_under_different_eligibility_scope() {
        let query = ContinuousHV::from_vec(vec![1.0; 8]);
        let envelope = ContinuousHVGroupEnvelope::from_original_members(
            62,
            10,
            &[("scoped-memory".to_owned(), ContinuousHV::from_vec(vec![0.8; 8]))],
        )
        .unwrap();
        let mut group = build_group_record(&envelope, &query);
        group.scope_binding.policy_generation += 1;
        let groups = vec![group];
        assert!(matches!(
            select_group_refinement_set(&groups, 1, complete_group_coverage(&groups)),
            Err(GroupIntervalSelectionError::EligibilityScopeMismatch { group_index: 0 })
        ));
    }

    #[test]
    fn group_selector_rejects_mixed_snapshot_generations_and_queries() {
        let query = ContinuousHV::from_vec(vec![1.0; 8]);
        let members_a = vec![
            ("snapshot-a".to_owned(), ContinuousHV::from_vec(vec![0.8; 8])),
        ];
        let members_b = vec![
            ("snapshot-b".to_owned(), ContinuousHV::from_vec(vec![-0.8; 8])),
        ];
        let envelope_a =
            ContinuousHVGroupEnvelope::from_original_members(50, 7, &members_a).unwrap();
        let envelope_b =
            ContinuousHVGroupEnvelope::from_original_members(51, 7, &members_b).unwrap();
        let group_a = build_group_record(&envelope_a, &query);
        let group_b = build_group_record(&envelope_b, &query);
        let groups = vec![group_a.clone(), group_b];
        let coverage = complete_group_coverage(&[group_a.clone()]);
        assert!(matches!(
            select_group_refinement_set(&groups, 1, coverage),
            Err(GroupIntervalSelectionError::SnapshotGenerationMismatch { group_index: 1 })
        ));

        let other_query = ContinuousHV::from_vec(vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let other_group = build_group_record(&envelope_b, &other_query);
        let same_generation = ContinuousHVGroupEnvelope::from_original_members(
            50,
            7,
            &[("snapshot-b".to_owned(), members_b[0].1.clone())],
        )
        .unwrap();
        let group_same_generation = build_group_record(&same_generation, &other_query);
        let mixed_queries = vec![group_a.clone(), group_same_generation];
        let coverage = complete_group_coverage(&[group_a]);
        assert!(matches!(
            select_group_refinement_set(&mixed_queries, 1, coverage),
            Err(GroupIntervalSelectionError::QueryDigestMismatch { group_index: 1 })
        ));
        assert_ne!(
            continuous_hv_query_digest(&query),
            continuous_hv_query_digest(&other_query)
        );
        assert_eq!(other_group.query_digest, continuous_hv_query_digest(&other_query));
    }

    #[test]
    fn hierarchical_selection_rejects_member_interval_from_another_query() {
        let query = ContinuousHV::from_vec(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let other_query =
            ContinuousHV::from_vec(vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let members = vec![
            ("item-a".to_owned(), unit_vector_with_first_component(0.9)),
            ("item-b".to_owned(), unit_vector_with_first_component(0.8)),
        ];
        let envelope =
            ContinuousHVGroupEnvelope::from_original_members(70, 8, &members).unwrap();
        let group = build_group_record(&envelope, &query);
        let coverage = complete_group_coverage(&[group.clone()]);
        let other_query_member = GroupMemberScoreIntervalRecord::from_verified_l2_residual(
            &group,
            "item-a".to_owned(),
            &other_query,
            &members[0].1,
            &members[0].1,
            0.0,
        )
        .unwrap();
        assert!(matches!(
            select_hierarchical_refinement_set(&[group], &[other_query_member], 1, coverage),
            Err(HierarchicalIntervalSelectionError::UnexpectedMemberGroup { .. })
        ));
    }

    #[test]
    fn group_selector_rejects_mixed_manifest_or_eligibility_generations() {
        let first = synthetic_group("snapshot-a", 1, 0.2, 0.8);
        let mut stale = synthetic_group("snapshot-b", 1, -0.2, 0.1);
        stale.identity.eligibility_generation = 2;
        let groups = vec![first, stale];
        assert!(matches!(
            select_group_refinement_set(&groups, 1, complete_group_coverage(&groups)),
            Err(GroupIntervalSelectionError::SnapshotGenerationMismatch { group_index: 1 })
        ));
    }

    #[test]
    fn group_selector_rejects_inconsistent_member_payload_digest_count() {
        let mut group = synthetic_group("malformed", 2, 0.0, 1.0);
        group.member_payload_digests = Arc::from(
            group.member_payload_digests[..group.member_payload_digests.len() - 1].to_vec(),
        );
        let groups = vec![group];
        assert!(matches!(
            select_group_refinement_set(&groups, 1, complete_group_coverage(&groups)),
            Err(GroupIntervalSelectionError::InvalidGroupMembership { group_index: 0 })
        ));
    }

    #[test]
    fn generated_weighted_group_selection_matches_full_exact_top_k() {
        let mut state = 0x6A09_E667u32;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        };

        for case_index in 0..4096usize {
            let count = 1 + (next() as usize % 48);
            let top_k = 1 + (next() as usize % count);
            let mut exact_scores = Vec::with_capacity(count);
            let mut item_intervals = Vec::with_capacity(count);

            for item_index in 0..count {
                let score_raw = (next() % 2001) as i32 - 1000;
                let score = score_raw as f32 / 1000.0;
                let lower_slack = (next() % 401) as f32 / 1000.0;
                let upper_slack = (next() % 401) as f32 / 1000.0;
                let lower = (score - lower_slack).max(-1.0);
                let upper = (score + upper_slack).min(1.0);
                let id = format!("item-{item_index:04}");
                exact_scores.push((id.clone(), score));
                item_intervals.push((lower, upper));
            }

            // Deterministic variable-size partitions. The group interval is the
            // conservative hull of all member intervals in that group.
            let mut groups = Vec::new();
            let mut start = 0usize;
            let mut group_index = 0usize;
            while start < count {
                let size = (1 + (next() as usize % 6)).min(count - start);
                let mut lower = 1.0f32;
                let mut upper = -1.0f32;
                for &(member_lower, member_upper) in &item_intervals[start..start + size] {
                    lower = lower.min(member_lower);
                    upper = upper.max(member_upper);
                }
                groups.push(synthetic_group(
                    &format!("group-{group_index:04}"),
                    size,
                    lower,
                    upper,
                ));
                group_index += 1;
                start += size;
            }

            let plan = select_group_refinement_set(
                &groups,
                top_k,
                complete_group_coverage(&groups),
            )
            .unwrap();
            let selected: BTreeSet<[u8; 32]> = plan
                .refinement_groups
                .iter()
                .map(|identity| identity.member_set_digest)
                .collect();
            let mut selected_ids = Vec::new();
            let mut start = 0usize;
            for group in &groups {
                let size = group.member_count;
                if selected.contains(&group.identity.member_set_digest) {
                    selected_ids.extend(
                        exact_scores[start..start + size]
                            .iter()
                            .map(|(id, _)| id.clone()),
                    );
                }
                start += size;
            }

            let full: Vec<ExactScoreRecord<String>> = exact_scores
                .iter()
                .map(|(id, score)| ExactScoreRecord {
                    id: id.clone(),
                    score: *score,
                })
                .collect();
            let expected = stable_exact_top_k(&full, top_k).unwrap();
            let selected_set: BTreeSet<String> = selected_ids.into_iter().collect();
            let refined: Vec<ExactScoreRecord<String>> = exact_scores
                .iter()
                .filter(|(id, _)| selected_set.contains(id))
                .map(|(id, score)| ExactScoreRecord {
                    id: id.clone(),
                    score: *score,
                })
                .collect();
            let actual = stable_exact_top_k(&refined, top_k).unwrap();
            assert_eq!(
                actual, expected,
                "weighted group selector false-pruned in case {case_index}"
            );
        }
    }

    #[test]
    fn public_record_constructor_requires_a_verified_residual() {
        let query = ContinuousHV::from_vec(vec![1.0, 0.5, -0.25, 0.75]);
        let original = ContinuousHV::from_vec(vec![0.5, -0.5, 0.25, 1.0]);
        let reconstruction = ContinuousHV::from_vec(vec![0.51, -0.49, 0.25, 1.0]);
        let record = ScoreIntervalRecord::from_verified_l2_residual(
            "memory-1".to_owned(),
            &query,
            &original,
            &reconstruction,
            0.015,
        )
        .unwrap();
        assert_eq!(record.id(), "memory-1");
        assert!(record.interval().is_well_formed());
        assert_eq!(record.certificate_check().changed_coordinates, 2);

        let underreported = ScoreIntervalRecord::from_verified_l2_residual(
            "memory-2".to_owned(),
            &query,
            &original,
            &reconstruction,
            0.001,
        );
        assert!(matches!(
            underreported,
            Err(ScoreIntervalBuildError::ResidualCertificate(
                ResidualCertificateError::Underreported { .. }
            ))
        ));
    }

    #[test]
    fn fixture_selects_global_refinement_set_and_preserves_threshold_equality() {
        let records = vec![
            record("a", 0.90, 0.98),
            record("b", 0.82, 0.94),
            record("c", 0.78, 0.86),
            record("d", 0.55, 0.82),
            record("e", 0.10, 0.30),
            record("f", -0.05, 0.81),
            record("g", 0.80, 0.82),
        ];
        let plan = select_refinement_set(&records, 2, complete_coverage(7)).unwrap();
        assert_eq!(plan.threshold, 0.82);
        assert_eq!(
            plan.refinement_ids,
            vec!["a".to_owned(), "b".to_owned(), "c".to_owned(), "d".to_owned(), "g".to_owned()]
        );
        assert_eq!(plan.excluded_ids, vec!["e".to_owned(), "f".to_owned()]);
    }

    #[test]
    fn output_is_independent_of_input_order() {
        let first = vec![
            record("a", 0.9, 0.98),
            record("b", 0.8, 0.9),
            record("c", 0.1, 0.2),
        ];
        let second = vec![first[2].clone(), first[0].clone(), first[1].clone()];
        let first_plan = select_refinement_set(&first, 2, complete_coverage(3)).unwrap();
        let second_plan = select_refinement_set(&second, 2, complete_coverage(3)).unwrap();
        assert_eq!(first_plan, second_plan);
    }

    #[test]
    fn duplicate_ids_and_invalid_intervals_fail_closed() {
        let duplicate = vec![record("same", 0.1, 0.2), record("same", 0.0, 0.3)];
        assert!(matches!(
            select_refinement_set(&duplicate, 1, complete_coverage(2)),
            Err(IntervalSelectionError::DuplicateId { .. })
        ));

        let invalid = vec![record("bad", f32::NAN, 0.5)];
        assert_eq!(
            select_refinement_set(&invalid, 1, complete_coverage(1)),
            Err(IntervalSelectionError::InvalidInterval { record_index: 0 })
        );
    }

    #[test]
    fn incomplete_metadata_or_unopened_partition_prevents_selection() {
        let records = vec![record("a", 0.1, 0.9), record("b", 0.0, 0.2)];
        let missing_record = IntervalCoverageCounts {
            eligible_records: 3,
            interval_records: 2,
            unbounded_unopened_partitions: 0,
        };
        assert!(matches!(
            select_refinement_set(&records, 1, missing_record),
            Err(IntervalSelectionError::IncompleteCoverage { .. })
        ));

        let unbounded_partition = IntervalCoverageCounts {
            eligible_records: 2,
            interval_records: 2,
            unbounded_unopened_partitions: 1,
        };
        assert!(matches!(
            select_refinement_set(&records, 1, unbounded_partition),
            Err(IntervalSelectionError::IncompleteCoverage { .. })
        ));
    }

    #[test]
    fn generated_interval_refinement_matches_full_exact_top_k() {
        let mut state = 0x5EED_1234u32;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        };

        for case_index in 0..4096usize {
            let count = 1 + (next() as usize % 32);
            let top_k = 1 + (next() as usize % count);
            let mut exact_scores = Vec::with_capacity(count);
            let mut intervals = Vec::with_capacity(count);

            for item_index in 0..count {
                let score_raw = (next() % 2001) as i32 - 1000;
                let score = score_raw as f32 / 1000.0;
                let lower_slack = (next() % 501) as f32 / 1000.0;
                let upper_slack = (next() % 501) as f32 / 1000.0;
                let lower = (score - lower_slack).max(-1.0);
                let upper = (score + upper_slack).min(1.0);
                let id = format!("{item_index:04}");
                exact_scores.push((id.clone(), score));
                intervals.push(record(&id, lower, upper));
            }

            let plan = select_refinement_set(
                &intervals,
                top_k,
                complete_coverage(count),
            )
            .unwrap();

            let full_scores: Vec<ExactScoreRecord<String>> = exact_scores
                .iter()
                .map(|(id, score)| ExactScoreRecord {
                    id: id.clone(),
                    score: *score,
                })
                .collect();
            let expected = stable_exact_top_k(&full_scores, top_k).unwrap();

            let refinement_members: BTreeSet<String> =
                plan.refinement_ids.iter().cloned().collect();
            let refined_scores: Vec<ExactScoreRecord<String>> = exact_scores
                .iter()
                .filter(|(id, _)| refinement_members.contains(id))
                .map(|(id, score)| ExactScoreRecord {
                    id: id.clone(),
                    score: *score,
                })
                .collect();
            let actual = stable_exact_top_k(&refined_scores, top_k).unwrap();

            assert_eq!(
                actual, expected,
                "false-prune disagreement in generated case {case_index}"
            );
        }
    }

    #[test]
    fn threshold_ties_use_stable_id_even_for_signed_zero() {
        let records = vec![
            record("z", 0.0, 0.2),
            record("a", -0.0, 0.2),
        ];
        let plan = select_refinement_set(&records, 1, complete_coverage(2)).unwrap();
        assert_eq!(plan.threshold.to_bits(), (-0.0f32).to_bits());
        assert_eq!(plan.refinement_ids, vec!["a".to_owned(), "z".to_owned()]);
    }

    #[test]
    fn exact_rerank_uses_numeric_score_then_stable_id_including_signed_zero() {
        let scores = vec![
            ExactScoreRecord { id: "z".to_owned(), score: 0.0 },
            ExactScoreRecord { id: "b".to_owned(), score: -0.0 },
            ExactScoreRecord { id: "a".to_owned(), score: 0.0 },
            ExactScoreRecord { id: "c".to_owned(), score: 0.5 },
        ];
        assert_eq!(
            stable_exact_top_k(&scores, 3).unwrap(),
            vec!["c".to_owned(), "a".to_owned(), "b".to_owned()]
        );
    }

    #[test]
    fn exact_rerank_rejects_non_finite_or_out_of_range_scores() {
        let scores = vec![ExactScoreRecord {
            id: "nan".to_owned(),
            score: f32::NAN,
        }];
        assert_eq!(
            stable_exact_top_k(&scores, 1),
            Err(IntervalSelectionError::InvalidExactScore { record_index: 0 })
        );
        let out_of_range = vec![ExactScoreRecord {
            id: "large".to_owned(),
            score: 1.01,
        }];
        assert_eq!(
            stable_exact_top_k(&out_of_range, 1),
            Err(IntervalSelectionError::InvalidExactScore { record_index: 0 })
        );
    }

    #[test]
    fn invalid_top_k_is_rejected() {
        let records = vec![record("a", 0.1, 0.9)];
        assert!(matches!(
            select_refinement_set(&records, 0, complete_coverage(1)),
            Err(IntervalSelectionError::InvalidTopK { .. })
        ));
        assert!(matches!(
            select_refinement_set(&records, 2, complete_coverage(1)),
            Err(IntervalSelectionError::InvalidTopK { .. })
        ));
    }
}

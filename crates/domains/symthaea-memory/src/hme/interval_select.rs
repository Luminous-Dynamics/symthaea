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

use std::collections::BTreeMap;

use super::group_envelope::{ContinuousHVGroupEnvelope, GroupEnvelopeError};

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
            member_count,
            interval,
        })
    }

    /// Exact group identity carried into the selection plan.
    pub fn identity(&self) -> GroupScoreIntervalIdentity {
        self.identity
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
    /// Number of unopened groups without an admissible aggregate interval.
    pub unbounded_unopened_groups: usize,
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
    /// Snapshot-bound group identities whose members must be visited or refined.
    pub refinement_groups: Vec<GroupScoreIntervalIdentity>,
    /// Snapshot-bound groups proven unable to contain a member in the exact top-k.
    pub excluded_groups: Vec<GroupScoreIntervalIdentity>,
    /// Number of records represented by the refinement groups.
    pub refinement_record_count: usize,
    /// Number of records avoided by excluding whole groups.
    pub excluded_record_count: usize,
}

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
pub fn select_group_refinement_set(
    groups: &[GroupScoreIntervalRecord],
    top_k: usize,
    coverage: GroupIntervalCoverageCounts,
) -> Result<GroupIntervalSelectionPlan, GroupIntervalSelectionError> {
    let mut covered_records = 0usize;
    let mut first_index_by_id = BTreeMap::new();
    let mut lower_order: Vec<&GroupScoreIntervalRecord> =
        Vec::with_capacity(groups.len());

    for (group_index, group) in groups.iter().enumerate() {
        if group.member_count == 0 {
            return Err(GroupIntervalSelectionError::EmptyGroup { group_index });
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
        refinement_groups,
        excluded_groups,
        refinement_record_count,
        excluded_record_count,
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
        GroupScoreIntervalRecord {
            identity: GroupScoreIntervalIdentity {
                manifest_generation: 1,
                eligibility_generation: 1,
                member_set_digest,
                content_digest,
            },
            member_count,
            interval: ContinuousHVScoreInterval { lower, upper },
        }
    }

    fn complete_group_coverage(
        groups: &[GroupScoreIntervalRecord],
    ) -> GroupIntervalCoverageCounts {
        let members = groups.iter().map(|group| group.member_count).sum();
        GroupIntervalCoverageCounts {
            eligible_records: members,
            covered_records: members,
            interval_groups: groups.len(),
            unbounded_unopened_groups: 0,
        }
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
        );
        assert!(matches!(
            stale,
            Err(GroupScoreIntervalBuildError::GroupEnvelope(
                GroupEnvelopeError::ManifestGenerationMismatch { .. }
            ))
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

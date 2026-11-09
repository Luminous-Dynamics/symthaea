// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Verified coordinate-envelope synopses for immutable ContinuousHV groups.
//!
//! A group envelope is built by scanning every original payload in a group and
//! computing exact coordinate minima/maxima. At query time, the core interval
//! kernel propagates that box through the canonical f32 cosine score. If the
//! group upper score is strictly below the current exact kth threshold, every
//! member can be skipped without loading its original payload.
//!
//! The builder is deliberately read-only and in-memory in this first slice.
//! Persistence, checksummed immutable segment headers and reopen-time proof
//! binding remain owned by HME-001A (#10917). A digest binds the builder input;
//! it is not a substitute for verifying the extrema against those payloads.

use std::collections::BTreeMap;
use std::sync::Arc;

use symthaea_core::hdc::continuous_hv_interval::{
    ContinuousHVIntervalError, ContinuousHVScoreInterval, ResidualCertificateError,
    verify_l2_residual_certificate,
};
use symthaea_core::hdc::unified_hv::CONTINUOUS_HV_FULL_RESOLUTION_SCORE_CONTRACT_V1;
use symthaea_core::hdc::ContinuousHV;

/// Domain separator for deterministic group-envelope content identities.
const GROUP_ENVELOPE_DOMAIN: &[u8] = b"hme_continuous_hv_group_envelope_v3";
const MEMBER_PAYLOAD_DOMAIN: &[u8] = b"hme_continuous_hv_member_payload_v1";
const QUERY_DOMAIN: &[u8] = b"hme_continuous_hv_query_v1";


/// Storage profile for a group synopsis.
///
/// BallOnly stores a centroid and conservative L2 radius. BoxAndBall adds
/// coordinate minima/maxima, trading 8 bytes per dimension for a potentially
/// tighter intersection bound.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GroupEnvelopeMode {
    /// Coordinate extrema plus centroid/radius.
    BoxAndBall,
    /// Quantized 16-bit coordinate extrema plus centroid/radius.
    QuantizedBoxAndBall,
    /// Centroid/radius only; lower synopsis memory, potentially looser pruning.
    BallOnly,
}

impl GroupEnvelopeMode {
    fn domain_tag(self) -> &'static [u8] {
        match self {
            Self::BoxAndBall => b"box_and_ball_v1",
            Self::QuantizedBoxAndBall => b"quantized_box_u16_and_ball_v1",
            Self::BallOnly => b"ball_only_v1",
        }
    }
}

/// Immutable coordinate envelope for one homogeneous ContinuousHV group.
#[derive(Debug, Clone, PartialEq)]
pub struct ContinuousHVGroupEnvelope {
    manifest_generation: u64,
    eligibility_generation: u64,
    dimension: usize,
    mode: GroupEnvelopeMode,
    member_ids: Arc<[String]>,
    member_payload_digests: Arc<[[u8; 32]]>,
    member_set_digest: [u8; 32],
    content_digest: [u8; 32],
    coordinate_minima: Option<Arc<[f32]>>,
    coordinate_maxima: Option<Arc<[f32]>>,
    coordinate_minima_u16: Option<Arc<[u16]>>,
    coordinate_maxima_u16: Option<Arc<[u16]>>,
    centroid: ContinuousHV,
    radius_l2_upper: f32,
}


/// Memory estimate for a group-envelope directory under one storage profile.
///
/// Counts only stable-ID UTF-8 bytes, per-member 32-byte payload digests, and
/// numeric group synopses. It intentionally excludes allocator/Arc/string
/// headers, directory indexes, segment checksums, alignment and page padding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GroupEnvelopeMemoryEstimate {
    /// Number of groups implied by eligible_records and group_size.
    pub group_count: usize,
    /// Numeric synopsis bytes per group (centroid, radius, optional extrema).
    pub numeric_bytes_per_group: usize,
    /// Numeric synopsis bytes over all groups.
    pub numeric_synopsis_bytes: usize,
    /// Per-member payload digest bytes.
    pub member_payload_digest_bytes: usize,
    /// Stable-ID UTF-8 bytes using the caller's average ID size.
    pub stable_member_id_bytes: usize,
    /// Sum of the explicitly counted fields above.
    pub known_metadata_bytes: usize,
}

/// Errors returned by the deterministic group metadata budget estimator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GroupEnvelopeMemoryEstimateError {
    /// Vector dimension must be nonzero.
    ZeroDimension,
    /// Group size must be nonzero.
    ZeroGroupSize,
    /// One or more checked byte/count calculations overflowed usize.
    ArithmeticOverflow,
}

impl std::fmt::Display for GroupEnvelopeMemoryEstimateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ZeroDimension => write!(f, "group synopsis dimension must be nonzero"),
            Self::ZeroGroupSize => write!(f, "group size must be nonzero"),
            Self::ArithmeticOverflow => write!(f, "group synopsis estimate overflowed usize"),
        }
    }
}

impl std::error::Error for GroupEnvelopeMemoryEstimateError {}

/// Estimate the explicitly countable metadata cost of a grouped HME synopsis.
///
/// BoxAndBall stores 3 f32 values per dimension; QuantizedBoxAndBall stores one
/// f32 centroid plus two u16 endpoints; BallOnly stores one f32 centroid. All
/// profiles store one f32 radius.
/// Member IDs and payload digests are per record and do not vary with group size.
pub fn estimate_group_envelope_memory(
    dimension: usize,
    eligible_records: usize,
    group_size: usize,
    average_stable_id_bytes: usize,
    mode: GroupEnvelopeMode,
) -> Result<GroupEnvelopeMemoryEstimate, GroupEnvelopeMemoryEstimateError> {
    if dimension == 0 {
        return Err(GroupEnvelopeMemoryEstimateError::ZeroDimension);
    }
    if group_size == 0 {
        return Err(GroupEnvelopeMemoryEstimateError::ZeroGroupSize);
    }
    let group_count = eligible_records / group_size
        + usize::from(eligible_records % group_size != 0);
    let numeric_bytes_per_group = match mode {
        GroupEnvelopeMode::BoxAndBall => dimension
            .checked_mul(3 * std::mem::size_of::<f32>()),
        GroupEnvelopeMode::QuantizedBoxAndBall => dimension
            .checked_mul(std::mem::size_of::<f32>() + 2 * std::mem::size_of::<u16>()),
        GroupEnvelopeMode::BallOnly => dimension
            .checked_mul(std::mem::size_of::<f32>()),
    }
    .and_then(|values| values.checked_add(std::mem::size_of::<f32>()))
    .ok_or(GroupEnvelopeMemoryEstimateError::ArithmeticOverflow)?;
    let numeric_synopsis_bytes = group_count
        .checked_mul(numeric_bytes_per_group)
        .ok_or(GroupEnvelopeMemoryEstimateError::ArithmeticOverflow)?;
    let member_payload_digest_bytes = eligible_records
        .checked_mul(32)
        .ok_or(GroupEnvelopeMemoryEstimateError::ArithmeticOverflow)?;
    let stable_member_id_bytes = eligible_records
        .checked_mul(average_stable_id_bytes)
        .ok_or(GroupEnvelopeMemoryEstimateError::ArithmeticOverflow)?;
    let known_metadata_bytes = numeric_synopsis_bytes
        .checked_add(member_payload_digest_bytes)
        .and_then(|value| value.checked_add(stable_member_id_bytes))
        .ok_or(GroupEnvelopeMemoryEstimateError::ArithmeticOverflow)?;

    Ok(GroupEnvelopeMemoryEstimate {
        group_count,
        numeric_bytes_per_group,
        numeric_synopsis_bytes,
        member_payload_digest_bytes,
        stable_member_id_bytes,
        known_metadata_bytes,
    })
}


/// Fail-closed errors for group-envelope construction and query use.
#[derive(Debug, Clone, PartialEq)]
pub enum GroupEnvelopeError {
    /// A synopsis cannot be constructed from an empty group.
    EmptyGroup,
    /// A member has an empty stable identifier.
    EmptyMemberId {
        /// Position of the invalid member in the supplied slice.
        member_index: usize,
    },
    /// The same stable identifier appears more than once.
    DuplicateMemberId {
        /// First member index.
        first_index: usize,
        /// Later duplicate member index.
        duplicate_index: usize,
    },
    /// A member has a dimension different from the first member.
    DimensionMismatch {
        /// Expected homogeneous dimension.
        expected: usize,
        /// Observed member dimension.
        actual: usize,
        /// Position of the mismatched member.
        member_index: usize,
    },
    /// A member is outside the finite [-1,1] canonical score profile.
    InvalidMemberProfile {
        /// Position of the invalid member.
        member_index: usize,
        /// Underlying profile error.
        error: symthaea_core::hdc::unified_hv::ContinuousHVProfileError,
    },
    /// A member's centroid residual could not be verified.
    ResidualCertificate {
        /// Position of the member whose residual failed.
        member_index: usize,
        /// Underlying residual validation error.
        error: ResidualCertificateError,
    },
    /// The caller supplied a different manifest generation.
    ManifestGenerationMismatch {
        /// Generation bound into the envelope.
        envelope_generation: u64,
        /// Generation requested by the query.
        requested_generation: u64,
    },
    /// The caller supplied a different eligibility generation.
    EligibilityGenerationMismatch {
        /// Eligibility generation bound into the envelope.
        envelope_generation: u64,
        /// Eligibility generation requested by the query.
        requested_generation: u64,
    },
    /// The member set does not match the pinned eligible group.
    MemberSetMismatch,
    /// The expected payload/content identity does not match.
    ContentDigestMismatch,
    /// A kth threshold is not finite or outside the canonical score range.
    InvalidKthThreshold,
    /// A centroid radius could not be represented as a finite binary32 value.
    InvalidRadius,
    /// The core interval kernel rejected the query or box.
    ScoreInterval(ContinuousHVIntervalError),
}

impl std::fmt::Display for GroupEnvelopeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyGroup => write!(f, "cannot build an empty ContinuousHV group envelope"),
            Self::EmptyMemberId { member_index } => {
                write!(f, "group member {member_index} has an empty stable ID")
            }
            Self::DuplicateMemberId { first_index, duplicate_index } => write!(
                f,
                "group member ID duplicated at indices {first_index} and {duplicate_index}"
            ),
            Self::DimensionMismatch { expected, actual, member_index } => write!(
                f,
                "group member {member_index} dimension {actual} does not match {expected}"
            ),
            Self::InvalidMemberProfile { member_index, error } => {
                write!(f, "group member {member_index} has invalid profile: {error}")
            }
            Self::ResidualCertificate { member_index, error } => {
                write!(f, "group member {member_index} exceeds centroid radius: {error}")
            }
            Self::ManifestGenerationMismatch { envelope_generation, requested_generation } => write!(
                f,
                "group envelope manifest generation {envelope_generation} does not match requested {requested_generation}"
            ),
            Self::EligibilityGenerationMismatch { envelope_generation, requested_generation } => write!(
                f,
                "group envelope eligibility generation {envelope_generation} does not match requested {requested_generation}"
            ),
            Self::MemberSetMismatch => write!(f, "group envelope member set does not match pinned eligible membership"),
            Self::ContentDigestMismatch => write!(f, "group envelope content digest does not match pinned manifest"),
            Self::InvalidKthThreshold => write!(f, "kth threshold must be finite and in [-1, 1]"),
            Self::InvalidRadius => write!(f, "group centroid radius is not finite"),
            Self::ScoreInterval(error) => write!(f, "group score interval failed: {error}"),
        }
    }
}

impl std::error::Error for GroupEnvelopeError {}

/// Compute a domain-separated identity for an exact canonical query vector.
pub fn continuous_hv_query_digest(query: &ContinuousHV) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(QUERY_DOMAIN);
    hasher.update(&(query.values.len() as u64).to_le_bytes());
    for value in &query.values {
        hasher.update(&value.to_bits().to_le_bytes());
    }
    *hasher.finalize().as_bytes()
}

/// Compute a content digest for one stable-ID/ContinuousHV payload pair.
///
/// The digest is suitable for binding a per-member interval certificate to
/// the original bytes admitted into a group envelope. It is not a residual
/// proof by itself; it only verifies that later work refers to the same item.
pub fn continuous_hv_member_payload_digest(
    member_id: &str,
    vector: &ContinuousHV,
) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(MEMBER_PAYLOAD_DOMAIN);
    hasher.update(&(member_id.len() as u64).to_le_bytes());
    hasher.update(member_id.as_bytes());
    hasher.update(&(vector.values.len() as u64).to_le_bytes());
    for value in &vector.values {
        hasher.update(&value.to_bits().to_le_bytes());
    }
    *hasher.finalize().as_bytes()
}

/// Compute the deterministic digest of a stable-ID set.
///
/// IDs are sorted and duplicate IDs are rejected. The digest does not prove
/// eligibility; it lets a query compare the exact requested membership set to
/// the set used when the envelope was built.
pub fn member_id_set_digest(
    member_ids: &[String],
) -> Result<[u8; 32], GroupEnvelopeError> {
    let mut first_index_by_id: BTreeMap<String, usize> = BTreeMap::new();
    for (member_index, id) in member_ids.iter().enumerate() {
        if id.is_empty() {
            return Err(GroupEnvelopeError::EmptyMemberId { member_index });
        }
        if let Some(first_index) = first_index_by_id.get(id).copied() {
            return Err(GroupEnvelopeError::DuplicateMemberId {
                first_index,
                duplicate_index: member_index,
            });
        }
        first_index_by_id.insert(id.clone(), member_index);
    }
    let mut sorted = member_ids.to_vec();
    sorted.sort();
    let mut hasher = blake3::Hasher::new();
    hasher.update(GROUP_ENVELOPE_DOMAIN);
    hasher.update(&(sorted.len() as u64).to_le_bytes());
    for id in sorted {
        hasher.update(&(id.len() as u64).to_le_bytes());
        hasher.update(id.as_bytes());
    }
    Ok(*hasher.finalize().as_bytes())
}

fn next_up_f64(value: f64) -> f64 {
    if value.is_nan() || value == f64::INFINITY {
        return value;
    }
    if value == 0.0 {
        return f64::from_bits(1);
    }
    let bits = value.to_bits();
    f64::from_bits(if value > 0.0 { bits + 1 } else { bits - 1 })
}

fn next_down_f64(value: f64) -> f64 {
    if value.is_nan() || value == f64::NEG_INFINITY {
        return value;
    }
    if value == 0.0 {
        return -f64::from_bits(1);
    }
    let bits = value.to_bits();
    f64::from_bits(if value > 0.0 { bits - 1 } else { bits + 1 })
}

fn next_up_f32(value: f32) -> f32 {
    if value.is_nan() || value == f32::INFINITY {
        return value;
    }
    if value == 0.0 {
        return f32::from_bits(1);
    }
    let bits = value.to_bits();
    f32::from_bits(if value > 0.0 { bits + 1 } else { bits - 1 })
}

fn next_down_f32(value: f32) -> f32 {
    if value.is_nan() || value == f32::NEG_INFINITY {
        return value;
    }
    if value == 0.0 {
        return -f32::from_bits(1);
    }
    let bits = value.to_bits();
    f32::from_bits(if value > 0.0 { bits - 1 } else { bits + 1 })
}

fn quantize_lower_coordinate(value: f32) -> u16 {
    let shifted_lower = next_down_f64(value as f64 + 1.0);
    let scaled_lower = next_down_f64(shifted_lower * 32_767.5);
    scaled_lower.floor().clamp(0.0, 65_535.0) as u16
}

fn quantize_upper_coordinate(value: f32) -> u16 {
    let shifted_upper = next_up_f64(value as f64 + 1.0);
    let scaled_upper = next_up_f64(shifted_upper * 32_767.5);
    scaled_upper.ceil().clamp(0.0, 65_535.0) as u16
}

fn decode_lower_coordinate(value: u16) -> f32 {
    let numerator = 2.0 * value as f64;
    let ratio = next_down_f64(numerator / 65_535.0);
    let outward = next_down_f64(ratio - 1.0).max(-1.0);
    let rounded = outward as f32;
    if (rounded as f64) > outward {
        next_down_f32(rounded).max(-1.0)
    } else {
        rounded.max(-1.0)
    }
}

fn decode_upper_coordinate(value: u16) -> f32 {
    let numerator = 2.0 * value as f64;
    let ratio = next_up_f64(numerator / 65_535.0);
    let outward = next_up_f64(ratio - 1.0).min(1.0);
    let rounded = outward as f32;
    if (rounded as f64) < outward {
        next_up_f32(rounded).min(1.0)
    } else {
        rounded.min(1.0)
    }
}

fn next_up_f64_nonnegative(value: f64) -> f64 {
    if value.is_nan() || value == f64::INFINITY {
        return value;
    }
    if value == 0.0 {
        return f64::from_bits(1);
    }
    f64::from_bits(value.to_bits() + 1)
}

fn next_up_f32_positive(value: f32) -> f32 {
    if value.is_nan() || value == f32::INFINITY {
        return value;
    }
    if value == 0.0 {
        return f32::from_bits(1);
    }
    let bits = value.to_bits();
    f32::from_bits(if value > 0.0 { bits + 1 } else { bits - 1 })
}

impl ContinuousHVGroupEnvelope {
    /// Build a group envelope by verifying and scanning every original member.
    ///
    /// This constructor owns the coordinate extrema: callers cannot provide
    /// claimed minima/maxima. Members are sorted by stable ID before hashing,
    /// making the identity independent of input ordering.
    pub fn from_original_members(
        manifest_generation: u64,
        eligibility_generation: u64,
        members: &[(String, ContinuousHV)],
    ) -> Result<Self, GroupEnvelopeError> {
        Self::from_original_members_with_mode(
            manifest_generation,
            eligibility_generation,
            members,
            GroupEnvelopeMode::BoxAndBall,
        )
    }

    /// Build a group envelope using an explicit synopsis storage profile.
    pub fn from_original_members_with_mode(
        manifest_generation: u64,
        eligibility_generation: u64,
        members: &[(String, ContinuousHV)],
        mode: GroupEnvelopeMode,
    ) -> Result<Self, GroupEnvelopeError> {
        if members.is_empty() {
            return Err(GroupEnvelopeError::EmptyGroup);
        }

        let dimension = members[0].1.values.len();
        let mut by_id: BTreeMap<&str, (usize, &ContinuousHV)> = BTreeMap::new();
        for (member_index, (id, vector)) in members.iter().enumerate() {
            if id.is_empty() {
                return Err(GroupEnvelopeError::EmptyMemberId { member_index });
            }
            if let Some((first_index, _)) = by_id.get(id.as_str()) {
                return Err(GroupEnvelopeError::DuplicateMemberId {
                    first_index: *first_index,
                    duplicate_index: member_index,
                });
            }
            if vector.values.len() != dimension {
                return Err(GroupEnvelopeError::DimensionMismatch {
                    expected: dimension,
                    actual: vector.values.len(),
                    member_index,
                });
            }
            vector
                .validate_full_resolution_profile("group_member")
                .map_err(|error| GroupEnvelopeError::InvalidMemberProfile {
                    member_index,
                    error,
                })?;
            by_id.insert(id.as_str(), (member_index, vector));
        }

        let (coordinate_minima, coordinate_maxima) = match mode {
            GroupEnvelopeMode::BoxAndBall | GroupEnvelopeMode::QuantizedBoxAndBall => {
                let mut minima = vec![f32::INFINITY; dimension];
                let mut maxima = vec![f32::NEG_INFINITY; dimension];
                for (_, vector) in by_id.values() {
                    for index in 0..dimension {
                        minima[index] = minima[index].min(vector.values[index]);
                        maxima[index] = maxima[index].max(vector.values[index]);
                    }
                }
                (Some(minima), Some(maxima))
            }
            GroupEnvelopeMode::BallOnly => (None, None),
        };
        let (coordinate_minima_u16, coordinate_maxima_u16) = match mode {
            GroupEnvelopeMode::QuantizedBoxAndBall => {
                let minima = coordinate_minima
                    .as_ref()
                    .expect("quantized mode computed coordinate minima")
                    .iter()
                    .copied()
                    .map(quantize_lower_coordinate)
                    .collect();
                let maxima = coordinate_maxima
                    .as_ref()
                    .expect("quantized mode computed coordinate maxima")
                    .iter()
                    .copied()
                    .map(quantize_upper_coordinate)
                    .collect();
                (Some(minima), Some(maxima))
            }
            _ => (None, None),
        };
        let (coordinate_minima, coordinate_maxima) = if mode == GroupEnvelopeMode::QuantizedBoxAndBall {
            (None, None)
        } else {
            (coordinate_minima, coordinate_maxima)
        };

        let member_ids: Vec<String> = by_id.keys().map(|id| (*id).to_owned()).collect();
        let member_set_digest = member_id_set_digest(&member_ids)?;

        // Use the arithmetic mean as a deterministic group center. It need not
        // be the minimum enclosing-ball center; the independently verified
        // maximum radius makes any center safe.
        let member_count = by_id.len() as f64;
        let mut centroid_accumulator = vec![0.0f64; dimension];
        for (_, vector) in by_id.values() {
            for (index, value) in vector.values.iter().enumerate() {
                centroid_accumulator[index] += *value as f64;
            }
        }
        let centroid = ContinuousHV::from_vec(
            centroid_accumulator
                .into_iter()
                .map(|sum| (sum / member_count) as f32)
                .collect(),
        );
        centroid
            .validate_full_resolution_profile("group_centroid")
            .map_err(|error| GroupEnvelopeError::InvalidMemberProfile {
                member_index: 0,
                error,
            })?;

        // A finite coarse radius is guaranteed to cover any two vectors in
        // [-1,1]^D. The verifier measures each source-to-centroid residual with
        // outward f64 accumulation; convert the largest squared bound upward
        // to f32 for the query-time L2-ball certificate.
        let coarse_radius = (2.0 * (dimension as f64).sqrt() + 1.0) as f32;
        if !coarse_radius.is_finite() {
            return Err(GroupEnvelopeError::InvalidRadius);
        }
        let mut max_residual_norm_squared_upper = 0.0f64;
        for (member_index, (_, vector)) in by_id.values().enumerate() {
            let verification = verify_l2_residual_certificate(
                vector,
                &centroid,
                coarse_radius,
            )
            .map_err(|error| GroupEnvelopeError::ResidualCertificate {
                member_index,
                error,
            })?;
            max_residual_norm_squared_upper = max_residual_norm_squared_upper
                .max(verification.residual_norm_squared_upper);
        }
        let radius_f64 = if max_residual_norm_squared_upper == 0.0 {
            0.0
        } else {
            next_up_f64_nonnegative(max_residual_norm_squared_upper.sqrt())
        };
        let mut radius_l2_upper = radius_f64 as f32;
        if (radius_l2_upper as f64) < radius_f64 {
            radius_l2_upper = next_up_f32_positive(radius_l2_upper);
        }
        if radius_l2_upper > 0.0 {
            // One additional binary32 step makes the persisted radius robust to
            // the final conversion boundary; it only loosens the safe envelope.
            radius_l2_upper = next_up_f32_positive(radius_l2_upper);
        }
        if !radius_l2_upper.is_finite() {
            return Err(GroupEnvelopeError::InvalidRadius);
        }

        let mut member_payload_digests = Vec::with_capacity(member_ids.len());
        let mut hasher = blake3::Hasher::new();
        hasher.update(GROUP_ENVELOPE_DOMAIN);
        hasher.update(b"/content");
        hasher.update(CONTINUOUS_HV_FULL_RESOLUTION_SCORE_CONTRACT_V1.as_bytes());
        hasher.update(&manifest_generation.to_le_bytes());
        hasher.update(&eligibility_generation.to_le_bytes());
        hasher.update(&(dimension as u64).to_le_bytes());
        hasher.update(mode.domain_tag());
        hasher.update(&(member_ids.len() as u64).to_le_bytes());
        hasher.update(&member_set_digest);
        for id in &member_ids {
            let (_, vector) = by_id
                .get(id.as_str())
                .expect("member ID was inserted and sorted from the same map");
            hasher.update(&(id.len() as u64).to_le_bytes());
            hasher.update(id.as_bytes());
            let payload_digest = continuous_hv_member_payload_digest(id, vector);
            member_payload_digests.push(payload_digest);
            hasher.update(&payload_digest);
            for value in &vector.values {
                hasher.update(&value.to_bits().to_le_bytes());
            }
        }
        if let (Some(minima), Some(maxima)) = (&coordinate_minima, &coordinate_maxima) {
            for value in minima.iter().chain(maxima) {
                hasher.update(&value.to_bits().to_le_bytes());
            }
        }
        if let (Some(minima), Some(maxima)) = (&coordinate_minima_u16, &coordinate_maxima_u16) {
            for value in minima.iter().chain(maxima) {
                hasher.update(&value.to_le_bytes());
            }
        }
        hasher.update(&radius_l2_upper.to_bits().to_le_bytes());
        for value in &centroid.values {
            hasher.update(&value.to_bits().to_le_bytes());
        }

        Ok(Self {
            manifest_generation,
            eligibility_generation,
            dimension,
            mode,
            member_ids: Arc::from(member_ids),
            member_payload_digests: Arc::from(member_payload_digests),
            member_set_digest,
            content_digest: *hasher.finalize().as_bytes(),
            coordinate_minima: coordinate_minima.map(Arc::from),
            coordinate_maxima: coordinate_maxima.map(Arc::from),
            coordinate_minima_u16: coordinate_minima_u16.map(Arc::from),
            coordinate_maxima_u16: coordinate_maxima_u16.map(Arc::from),
            centroid,
            radius_l2_upper,
        })
    }

    /// Storage profile used by this synopsis.
    pub fn mode(&self) -> GroupEnvelopeMode {
        self.mode
    }

    /// Numeric synopsis bytes retained by this envelope, excluding IDs,
    /// digests and allocator/Arc overhead.
    pub fn numeric_synopsis_bytes(&self) -> usize {
        self.centroid.values.len() * std::mem::size_of::<f32>()
            + self
                .coordinate_minima
                .as_ref()
                .map_or(0, |values| values.len() * std::mem::size_of::<f32>())
            + self
                .coordinate_maxima
                .as_ref()
                .map_or(0, |values| values.len() * std::mem::size_of::<f32>())
            + self
                .coordinate_minima_u16
                .as_ref()
                .map_or(0, |values| values.len() * std::mem::size_of::<u16>())
            + self
                .coordinate_maxima_u16
                .as_ref()
                .map_or(0, |values| values.len() * std::mem::size_of::<u16>())
            + std::mem::size_of::<f32>()
    }

    /// Manifest generation bound into this immutable group envelope.
    pub fn manifest_generation(&self) -> u64 {
        self.manifest_generation
    }

    /// Eligibility generation bound into this immutable group envelope.
    pub fn eligibility_generation(&self) -> u64 {
        self.eligibility_generation
    }

    /// Stable IDs covered by the envelope, sorted ascending.
    pub fn member_ids(&self) -> &[String] {
        &self.member_ids
    }

    /// Shared immutable member roster for query plans; cloning this handle does
    /// not clone the full stable-ID list.
    pub(crate) fn member_ids_shared(&self) -> Arc<[String]> {
        Arc::clone(&self.member_ids)
    }

    /// Per-member payload digests aligned with member_ids() order.
    pub fn member_payload_digests(&self) -> &[[u8; 32]] {
        &self.member_payload_digests
    }

    /// Shared immutable payload-digest roster aligned with member_ids().
    pub(crate) fn member_payload_digests_shared(&self) -> Arc<[[u8; 32]]> {
        Arc::clone(&self.member_payload_digests)
    }

    /// Digest of the original payload associated with one stable member ID.
    pub fn member_payload_digest(&self, member_id: &str) -> Option<[u8; 32]> {
        self.member_ids
            .binary_search_by(|candidate| candidate.as_str().cmp(member_id))
            .ok()
            .and_then(|index| self.member_payload_digests.get(index).copied())
    }

    /// Digest of the exact member-ID set.
    pub fn member_set_digest(&self) -> [u8; 32] {
        self.member_set_digest
    }

    /// Digest binding generations, member IDs, payload f32 bits and extrema.
    pub fn content_digest(&self) -> [u8; 32] {
        self.content_digest
    }

    /// ContinuousHV dimension covered by this envelope.
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Deterministic centroid used by the joint L2 group envelope.
    pub fn centroid(&self) -> &ContinuousHV {
        &self.centroid
    }

    /// Conservatively verified maximum member-to-centroid L2 radius.
    pub fn radius_l2_upper(&self) -> f32 {
        self.radius_l2_upper
    }

    /// Coordinate-wise minimum extrema used by the box envelope.
    pub fn coordinate_minima(&self) -> &[f32] {
        &self.coordinate_minima
    }

    /// Coordinate-wise maximum extrema used by the box envelope.
    pub fn coordinate_maxima(&self) -> &[f32] {
        &self.coordinate_maxima
    }

    /// Query-specific interval enclosing every member score for a pinned group.
    ///
    /// The expected generations, member-set digest and content digest must come
    /// from the authoritative query/segment snapshot. A mismatch disables
    /// group pruning. The caller must still ensure the group is uniformly
    /// eligible under the query's principal, valid_at/known_at and forgetting
    /// scope; construct separate envelopes for different eligible subsets.
    pub fn score_interval_for_snapshot(
        &self,
        query: &ContinuousHV,
        expected_manifest_generation: u64,
        expected_eligibility_generation: u64,
        expected_member_set_digest: [u8; 32],
        expected_content_digest: [u8; 32],
    ) -> Result<ContinuousHVScoreInterval, GroupEnvelopeError> {
        if self.manifest_generation != expected_manifest_generation {
            return Err(GroupEnvelopeError::ManifestGenerationMismatch {
                envelope_generation: self.manifest_generation,
                requested_generation: expected_manifest_generation,
            });
        }
        if self.eligibility_generation != expected_eligibility_generation {
            return Err(GroupEnvelopeError::EligibilityGenerationMismatch {
                envelope_generation: self.eligibility_generation,
                requested_generation: expected_eligibility_generation,
            });
        }
        if self.member_set_digest != expected_member_set_digest {
            return Err(GroupEnvelopeError::MemberSetMismatch);
        }
        if self.content_digest != expected_content_digest {
            return Err(GroupEnvelopeError::ContentDigestMismatch);
        }
        let centroid_ball = query
            .similarity_interval_full_resolution_for_l2_ball_center(
                &self.centroid,
                self.radius_l2_upper,
            )
            .map_err(GroupEnvelopeError::ScoreInterval)?;

        let component_box = match (
            (&self.coordinate_minima, &self.coordinate_maxima),
            (&self.coordinate_minima_u16, &self.coordinate_maxima_u16),
        ) {
            ((Some(minima), Some(maxima)), (None, None)) => query
                .similarity_interval_full_resolution_for_component_bounds(minima, maxima)
                .map_err(GroupEnvelopeError::ScoreInterval)?,
            ((None, None), (Some(minima), Some(maxima))) => query
                .similarity_interval_full_resolution_for_component_bounds_with(|index| {
                    (
                        decode_lower_coordinate(minima[index]),
                        decode_upper_coordinate(maxima[index]),
                    )
                })
                .map_err(GroupEnvelopeError::ScoreInterval)?,
            ((None, None), (None, None)) => return Ok(centroid_ball),
            _ => return Ok(ContinuousHVScoreInterval::uninformative()),
        };

        // Both enclosures contain every member. Their intersection is also a
        // valid enclosure and is no wider than either synopsis independently.
        let lower = component_box.lower.max(centroid_ball.lower);
        let upper = component_box.upper.min(centroid_ball.upper);
        if lower <= upper {
            Ok(ContinuousHVScoreInterval { lower, upper })
        } else {
            // Disjoint independently-derived bounds indicate a numeric or
            // provenance defect. The only safe response is no pruning.
            Ok(ContinuousHVScoreInterval::uninformative())
        }
    }

    /// Whether this entire group can be excluded from an exact top-k query.
    ///
    /// Only a strict upper<threshold result prunes. Equality always remains
    /// unresolved. The query and snapshot bindings are validated first.
    pub fn can_prune_for_top_k(
        &self,
        query: &ContinuousHV,
        kth_exact_threshold: f32,
        expected_manifest_generation: u64,
        expected_eligibility_generation: u64,
        expected_member_set_digest: [u8; 32],
        expected_content_digest: [u8; 32],
    ) -> Result<bool, GroupEnvelopeError> {
        if !kth_exact_threshold.is_finite() || !(-1.0..=1.0).contains(&kth_exact_threshold) {
            return Err(GroupEnvelopeError::InvalidKthThreshold);
        }
        let interval = self.score_interval_for_snapshot(
            query,
            expected_manifest_generation,
            expected_eligibility_generation,
            expected_member_set_digest,
            expected_content_digest,
        )?;
        Ok(interval.upper < kth_exact_threshold)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hv(values: &[f32]) -> ContinuousHV {
        ContinuousHV::from_vec(values.to_vec())
    }

    #[test]
    fn group_bound_contains_every_original_member_score() {
        let members = vec![
            ("m-1".to_owned(), hv(&[0.91, -0.79, 0.69, -0.61, 0.51, -0.39, 0.29, -0.21, 0.11])),
            ("m-2".to_owned(), hv(&[-0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.9, -1.0])),
            ("m-3".to_owned(), hv(&[0.0, 0.0, 0.1, -0.1, 0.2, -0.2, 0.3, -0.3, 0.4])),
        ];
        let envelope = ContinuousHVGroupEnvelope::from_original_members(7, 3, &members).unwrap();
        let query = hv(&[0.9, -0.8, 0.7, -0.6, 0.5, -0.4, 0.3, -0.2, 0.1]);
        let interval = envelope
            .score_interval_for_snapshot(
                &query,
                7,
                3,
                envelope.member_set_digest(),
                envelope.content_digest(),
            )
            .unwrap();
        for (_, member) in &members {
            assert!(interval.contains(query.similarity_full_resolution(member)));
        }
    }

    #[test]
    fn member_payload_digest_binds_id_dimension_and_exact_f32_bits() {
        let vector = hv(&[0.1, -0.2, 0.3, -0.4]);
        let same_id = continuous_hv_member_payload_digest("member-a", &vector);
        let different_id = continuous_hv_member_payload_digest("member-b", &vector);
        let different_payload = continuous_hv_member_payload_digest(
            "member-a",
            &hv(&[0.1, -0.2, 0.3, -0.401]),
        );
        let different_dimension = continuous_hv_member_payload_digest(
            "member-a",
            &hv(&[0.1, -0.2, 0.3]),
        );
        assert_eq!(
            same_id,
            continuous_hv_member_payload_digest("member-a", &vector)
        );
        assert_ne!(same_id, different_id);
        assert_ne!(same_id, different_payload);
        assert_ne!(same_id, different_dimension);
    }

    #[test]
    fn content_digest_is_input_order_independent_and_profile_bound() {
        let a = hv(&[0.1, 0.2, 0.3, 0.4]);
        let b = hv(&[-0.1, -0.2, -0.3, -0.4]);
        let first = vec![("b".to_owned(), b.clone()), ("a".to_owned(), a.clone())];
        let second = vec![("a".to_owned(), a), ("b".to_owned(), b)];
        let first_envelope =
            ContinuousHVGroupEnvelope::from_original_members(5, 2, &first).unwrap();
        let second_envelope =
            ContinuousHVGroupEnvelope::from_original_members(5, 2, &second).unwrap();
        assert_eq!(first_envelope.member_set_digest(), second_envelope.member_set_digest());
        assert_eq!(first_envelope.content_digest(), second_envelope.content_digest());
    }

    #[test]
    fn group_pruning_is_strict_and_generation_bound() {
        let members = vec![
            ("a".to_owned(), hv(&[-1.0; 9])),
            ("b".to_owned(), hv(&[-0.9; 9])),
        ];
        let envelope = ContinuousHVGroupEnvelope::from_original_members(11, 4, &members).unwrap();
        let query = hv(&[1.0; 9]);
        let args = (
            &query,
            11,
            4,
            envelope.member_set_digest(),
            envelope.content_digest(),
        );
        let interval = envelope
            .score_interval_for_snapshot(args.0, args.1, args.2, args.3, args.4)
            .unwrap();
        assert!(envelope
            .can_prune_for_top_k(&query, interval.upper, 11, 4, args.3, args.4)
            .unwrap()
            == false);
        assert!(envelope
            .can_prune_for_top_k(&query, (interval.upper + 1.0).min(1.0), 11, 4, args.3, args.4)
            .unwrap()
            || interval.upper == 1.0);

        assert!(matches!(
            envelope.score_interval_for_snapshot(
                &query,
                12,
                4,
                args.3,
                args.4
            ),
            Err(GroupEnvelopeError::ManifestGenerationMismatch { .. })
        ));
        assert!(matches!(
            envelope.score_interval_for_snapshot(
                &query,
                11,
                5,
                args.3,
                args.4
            ),
            Err(GroupEnvelopeError::EligibilityGenerationMismatch { .. })
        ));
        assert!(matches!(
            envelope.score_interval_for_snapshot(
                &query,
                11,
                4,
                [0u8; 32],
                args.4
            ),
            Err(GroupEnvelopeError::MemberSetMismatch)
        ));
    }

    #[test]
    fn exact_top_k_threshold_prunes_only_groups_that_cannot_contain_a_winner() {
        const DIM: usize = 64;
        const GROUPS: usize = 12;
        const MEMBERS_PER_GROUP: usize = 8;
        let mut groups = Vec::new();
        let mut all_scores = Vec::new();

        // Make group 0 align with the query; the remaining groups point in
        // orthogonal or opposing directions. A small perturbation ensures the
        // group radii are nonzero while remaining tightly bounded.
        for group_index in 0..GROUPS {
            let mut center = vec![0.0f32; DIM];
            center[0] = if group_index == 0 {
                1.0
            } else if group_index % 2 == 0 {
                -0.8
            } else {
                0.0
            };
            if group_index % 2 == 1 {
                center[(group_index / 2).max(1)] = 0.9;
            }
            let mut members = Vec::new();
            for member_index in 0..MEMBERS_PER_GROUP {
                let mut values = center.clone();
                let coordinate = (member_index * 7 + group_index * 3) % DIM;
                let delta = (member_index as f32 - 3.5) * 0.0001;
                values[coordinate] = (values[coordinate] + delta).clamp(-1.0, 1.0);
                let id = format!("g{group_index:02}-m{member_index:02}");
                let vector = ContinuousHV::from_vec(values);
                members.push((id.clone(), vector.clone()));
                all_scores.push((id, vector.similarity_full_resolution(&ContinuousHV::from_vec({
                    let mut q = vec![0.0f32; DIM];
                    q[0] = 1.0;
                    q
                }))));
            }
            groups.push(
                ContinuousHVGroupEnvelope::from_original_members(17, 5, &members).unwrap(),
            );
        }

        all_scores.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        let kth_threshold = all_scores[9].1;
        let query = ContinuousHV::from_vec({
            let mut q = vec![0.0f32; DIM];
            q[0] = 1.0;
            q
        });

        let mut pruned_groups = 0usize;
        for envelope in &groups {
            let may_prune = envelope
                .can_prune_for_top_k(
                    &query,
                    kth_threshold,
                    17,
                    5,
                    envelope.member_set_digest(),
                    envelope.content_digest(),
                )
                .unwrap();
            if may_prune {
                pruned_groups += 1;
                for id in envelope.member_ids() {
                    let (member_id, score) =
                        all_scores.iter().find(|(member_id, _)| member_id == id).unwrap();
                    assert!(
                        *score < kth_threshold,
                        "group was pruned but member {member_id} reaches kth threshold"
                    );
                }
            }
        }
        assert!(pruned_groups > 0, "expected the separated groups to expose a safe prune");
    }

    #[test]
    fn group_builder_rejects_duplicate_ids_bad_dimensions_and_non_finite_values() {
        let duplicate = vec![
            ("same".to_owned(), hv(&[0.0; 8])),
            ("same".to_owned(), hv(&[1.0; 8])),
        ];
        assert!(matches!(
            ContinuousHVGroupEnvelope::from_original_members(1, 1, &duplicate),
            Err(GroupEnvelopeError::DuplicateMemberId { .. })
        ));

        let mismatch = vec![
            ("a".to_owned(), hv(&[0.0; 8])),
            ("b".to_owned(), hv(&[0.0; 9])),
        ];
        assert!(matches!(
            ContinuousHVGroupEnvelope::from_original_members(1, 1, &mismatch),
            Err(GroupEnvelopeError::DimensionMismatch { .. })
        ));

        let invalid = vec![("a".to_owned(), hv(&[f32::NAN; 8]))];
        assert!(matches!(
            ContinuousHVGroupEnvelope::from_original_members(1, 1, &invalid),
            Err(GroupEnvelopeError::InvalidMemberProfile { .. })
        ));
    }

    #[test]
    fn group_envelope_memory_estimator_matches_storage_profiles() {
        let full = estimate_group_envelope_memory(
            262_144,
            1_000_000,
            64,
            24,
            GroupEnvelopeMode::BoxAndBall,
        )
        .unwrap();
        let quantized = estimate_group_envelope_memory(
            262_144,
            1_000_000,
            64,
            24,
            GroupEnvelopeMode::QuantizedBoxAndBall,
        )
        .unwrap();
        let ball = estimate_group_envelope_memory(
            262_144,
            1_000_000,
            64,
            24,
            GroupEnvelopeMode::BallOnly,
        )
        .unwrap();
        assert_eq!(full.group_count, 15_625);
        assert_eq!(full.numeric_bytes_per_group, 12 * 262_144 + 4);
        assert_eq!(quantized.numeric_bytes_per_group, 8 * 262_144 + 4);
        assert_eq!(ball.numeric_bytes_per_group, 4 * 262_144 + 4);
        assert_eq!(full.member_payload_digest_bytes, 32_000_000);
        assert_eq!(quantized.member_payload_digest_bytes, 32_000_000);
        assert_eq!(ball.member_payload_digest_bytes, 32_000_000);
        assert_eq!(
            full.numeric_synopsis_bytes - quantized.numeric_synopsis_bytes,
            4 * 262_144 * 15_625
        );
        assert_eq!(
            quantized.numeric_synopsis_bytes - ball.numeric_synopsis_bytes,
            4 * 262_144 * 15_625
        );
        assert!(full.known_metadata_bytes > quantized.known_metadata_bytes);
        assert!(quantized.known_metadata_bytes > ball.known_metadata_bytes);
        assert!(matches!(
            estimate_group_envelope_memory(
                0,
                1,
                1,
                8,
                GroupEnvelopeMode::BallOnly,
            ),
            Err(GroupEnvelopeMemoryEstimateError::ZeroDimension)
        ));
        assert!(matches!(
            estimate_group_envelope_memory(
                usize::MAX,
                1,
                1,
                8,
                GroupEnvelopeMode::BoxAndBall,
            ),
            Err(GroupEnvelopeMemoryEstimateError::ArithmeticOverflow)
        ));
    }

    #[test]
    fn quantized_coordinate_rounding_is_outward_at_boundaries_and_random_values() {
        let edge_values = [
            -1.0,
            -0.999_999_94,
            -0.75,
            -0.5,
            -0.125,
            -f32::from_bits(1),
            0.0,
            f32::from_bits(1),
            0.125,
            0.5,
            0.75,
            0.999_999_94,
            1.0,
        ];
        for value in edge_values {
            let lower = decode_lower_coordinate(quantize_lower_coordinate(value));
            let upper = decode_upper_coordinate(quantize_upper_coordinate(value));
            assert!(
                lower <= value,
                "lower quantized endpoint {lower} exceeds {value}"
            );
            assert!(
                upper >= value,
                "upper quantized endpoint {upper} falls below {value}"
            );
            assert!((-1.0..=1.0).contains(&lower));
            assert!((-1.0..=1.0).contains(&upper));
        }

        let mut state = 0xD1CE_BA5Eu32;
        for _ in 0..100_000 {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let unit = (state >> 8) as f32 / (u32::MAX >> 8) as f32;
            let value = unit * 2.0 - 1.0;
            let lower = decode_lower_coordinate(quantize_lower_coordinate(value));
            let upper = decode_upper_coordinate(quantize_upper_coordinate(value));
            assert!(lower <= value);
            assert!(upper >= value);
        }
    }

    #[test]
    fn quantized_coordinate_bounds_contain_neighbors_of_every_bin_boundary() {
        for code in 0..=65_535u32 {
            let boundary = (2.0 * code as f64 / 65_535.0) - 1.0;
            let rounded = boundary as f32;
            let candidates = [
                next_down_f32(rounded).max(-1.0),
                rounded.clamp(-1.0, 1.0),
                next_up_f32(rounded).min(1.0),
            ];
            for value in candidates {
                let lower =
                    decode_lower_coordinate(quantize_lower_coordinate(value));
                let upper =
                    decode_upper_coordinate(quantize_upper_coordinate(value));
                assert!(
                    lower <= value,
                    "boundary {code}: lower {lower} exceeds value {value}"
                );
                assert!(
                    upper >= value,
                    "boundary {code}: upper {upper} is below value {value}"
                );
            }
        }
    }

    #[test]
    fn quantized_box_ball_synopsis_has_midpoint_storage_and_contains_members() {
        const DIM: usize = 1024;
        let query = ContinuousHV::from_vec(
            (0..DIM)
                .map(|index| ((index as f32) * 0.021).cos() * 0.7)
                .collect(),
        );
        let members: Vec<(String, ContinuousHV)> = (0..20)
            .map(|member_index| {
                let vector = ContinuousHV::from_vec(
                    (0..DIM)
                        .map(|index| {
                            let base = ((index as f32) * 0.013).sin() * 0.65;
                            let perturbation =
                                (member_index as f32 - 9.5) * 0.0003
                                    * ((index as f32) * 0.031).cos();
                            (base + perturbation).clamp(-1.0, 1.0)
                        })
                        .collect(),
                );
                (format!("quantized-{member_index:02}"), vector)
            })
            .collect();

        let full = ContinuousHVGroupEnvelope::from_original_members(51, 8, &members).unwrap();
        let quantized = ContinuousHVGroupEnvelope::from_original_members_with_mode(
            51,
            8,
            &members,
            GroupEnvelopeMode::QuantizedBoxAndBall,
        )
        .unwrap();
        let ball = ContinuousHVGroupEnvelope::from_original_members_with_mode(
            51,
            8,
            &members,
            GroupEnvelopeMode::BallOnly,
        )
        .unwrap();

        assert_eq!(
            full.numeric_synopsis_bytes(),
            12 * DIM * 4 + 4
        );
        assert_eq!(
            quantized.numeric_synopsis_bytes(),
            8 * DIM + 4
        );
        assert_eq!(
            ball.numeric_synopsis_bytes(),
            4 * DIM + 4
        );
        assert_eq!(
            full.numeric_synopsis_bytes() - quantized.numeric_synopsis_bytes(),
            4 * DIM * 4
        );

        let quantized_interval = quantized
            .score_interval_for_snapshot(
                &query,
                51,
                8,
                quantized.member_set_digest(),
                quantized.content_digest(),
            )
            .unwrap();
        let full_interval = full
            .score_interval_for_snapshot(
                &query,
                51,
                8,
                full.member_set_digest(),
                full.content_digest(),
            )
            .unwrap();
        let ball_interval = ball
            .score_interval_for_snapshot(
                &query,
                51,
                8,
                ball.member_set_digest(),
                ball.content_digest(),
            )
            .unwrap();

        assert!(quantized_interval.lower <= full_interval.lower);
        assert!(quantized_interval.upper >= full_interval.upper);
        for (id, member) in &members {
            let exact = query.similarity_full_resolution(member);
            assert!(full_interval.contains(exact), "full profile missed {id}");
            assert!(
                quantized_interval.contains(exact),
                "quantized profile missed {id}: {exact} not in [{}, {}]",
                quantized_interval.lower,
                quantized_interval.upper
            );
            assert!(ball_interval.contains(exact), "ball profile missed {id}");
        }
        assert_ne!(full.content_digest(), quantized.content_digest());
        assert_ne!(quantized.content_digest(), ball.content_digest());
    }

    #[test]
    fn ball_only_synopsis_reduces_numeric_metadata_and_preserves_exact_enclosure() {
        const DIM: usize = 257;
        let query = ContinuousHV::from_vec(
            (0..DIM)
                .map(|index| ((index as f32) * 0.037).sin() * 0.8)
                .collect(),
        );
        let members: Vec<(String, ContinuousHV)> = (0..12)
            .map(|member_index| {
                let center = member_index as f32 * 0.0007;
                let vector = ContinuousHV::from_vec(
                    (0..DIM)
                        .map(|index| {
                            ((index as f32) * 0.019).cos() * 0.6
                                + center * ((index as f32) * 0.071).sin()
                        })
                        .map(|value| value.clamp(-1.0, 1.0))
                        .collect(),
                );
                (format!("member-{member_index:02}"), vector)
            })
            .collect();

        let full = ContinuousHVGroupEnvelope::from_original_members(31, 12, &members).unwrap();
        let ball = ContinuousHVGroupEnvelope::from_original_members_with_mode(
            31,
            12,
            &members,
            GroupEnvelopeMode::BallOnly,
        )
        .unwrap();

        assert_eq!(full.mode(), GroupEnvelopeMode::BoxAndBall);
        assert_eq!(ball.mode(), GroupEnvelopeMode::BallOnly);
        assert_eq!(full.numeric_synopsis_bytes(), 12 * DIM * 4 + 4);
        assert_eq!(ball.numeric_synopsis_bytes(), DIM * 4 + 4);
        assert_eq!(
            full.numeric_synopsis_bytes() - ball.numeric_synopsis_bytes(),
            8 * DIM * 4
        );
        assert_ne!(full.content_digest(), ball.content_digest());

        let full_interval = full
            .score_interval_for_snapshot(
                &query,
                31,
                12,
                full.member_set_digest(),
                full.content_digest(),
            )
            .unwrap();
        let ball_interval = ball
            .score_interval_for_snapshot(
                &query,
                31,
                12,
                ball.member_set_digest(),
                ball.content_digest(),
            )
            .unwrap();

        for (id, member) in &members {
            let exact = query.similarity_full_resolution(member);
            assert!(full_interval.contains(exact), "full box missed {id}");
            assert!(ball_interval.contains(exact), "ball-only missed {id}");
        }
        assert!(ball_interval.lower <= full_interval.lower);
        assert!(ball_interval.upper >= full_interval.upper);
    }

    #[test]
    fn ball_only_and_box_ball_have_distinct_snapshot_identities() {
        let members = vec![
            ("a".to_owned(), hv(&[0.7, 0.1, -0.2, 0.9, 0.3, -0.1, 0.4, 0.8])),
            ("b".to_owned(), hv(&[0.6, 0.2, -0.1, 0.8, 0.4, -0.2, 0.3, 0.7])),
        ];
        let full = ContinuousHVGroupEnvelope::from_original_members(4, 5, &members).unwrap();
        let ball = ContinuousHVGroupEnvelope::from_original_members_with_mode(
            4,
            5,
            &members,
            GroupEnvelopeMode::BallOnly,
        )
        .unwrap();
        assert_eq!(full.member_set_digest(), ball.member_set_digest());
        assert_ne!(full.content_digest(), ball.content_digest());
        assert!(matches!(
            ball.score_interval_for_snapshot(
                &hv(&[1.0; 8]),
                4,
                5,
                full.member_set_digest(),
                full.content_digest(),
            ),
            Err(GroupEnvelopeError::ContentDigestMismatch)
        ));
    }

    #[test]
    fn high_dimensional_group_envelope_contains_every_original_score() {
        let dimension = 16_384usize;
        let base: Vec<f32> = (0..dimension)
            .map(|index| ((index as f32) * 0.007).sin() * 0.7)
            .collect();
        let members: Vec<(String, ContinuousHV)> = [0.0f32, 0.0002, -0.0003, 0.0004]
            .iter()
            .enumerate()
            .map(|(member_index, offset)| {
                let values = base
                    .iter()
                    .enumerate()
                    .map(|(index, value)| {
                        (*value + *offset * ((index as f32) * 0.031).cos())
                            .clamp(-1.0, 1.0)
                    })
                    .collect();
                (format!("member-{member_index}"), ContinuousHV::from_vec(values))
            })
            .collect();
        let envelope =
            ContinuousHVGroupEnvelope::from_original_members(19, 7, &members).unwrap();
        let query = ContinuousHV::from_vec(
            (0..dimension)
                .map(|index| ((index as f32) * 0.011).cos() * 0.75)
                .collect(),
        );
        let interval = envelope
            .score_interval_for_snapshot(
                &query,
                19,
                7,
                envelope.member_set_digest(),
                envelope.content_digest(),
            )
            .unwrap();
        for (id, member) in &members {
            let exact = query.similarity_full_resolution(member);
            assert!(
                interval.contains(exact),
                "{id} score {exact} escaped [{}, {}]",
                interval.lower,
                interval.upper
            );
        }
    }

}

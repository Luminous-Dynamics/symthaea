// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Content identity for the exact temporal evidence that authorized one chemical
//! same-cycle aggregation.
//!
//! This module closes a provenance gap without making root cognition understand
//! clock/synchronization types. Chemosensation owns the semantic canonicalization
//! of its temporal admission evidence and exposes only a generic
//! [`ContentAddress32`] across subsystem boundaries.
//!
//! V2 additionally commits each timed component's optional acquisition-time
//! authorization reference. This means evidence-bound physical acquisition can
//! remain distinguishable even when two different calibration/holdover histories
//! derive an identical normalized clock transform. General replay/test normalized
//! percepts remain valid with no acquisition-authority reference.
//!
//! The identity is **not** a signature, timestamp authority, synchronization
//! proof, or trust score. It says only which exact timing claims, normalization
//! provenance, acquisition authority references, skew policy, pairwise separation
//! windows, and admission result were used. Authenticity and authorization of the
//! producers remain separate layers.

use std::fmt;

use blake3::Hasher;
use symthaea_evidence_plane::{ContentAddress32, ContentAddressError};
use symthaea_time_integrity::{
    ClockEpochId, ContinuityStatus, TimeIntegrityReceipt, TimeUncertainty,
};
use symthaea_time_normalization::{
    ClockTransformModel, ClockTransformReceipt, NormalizedTimePoint,
};

use crate::{
    ChemicalTemporalAdmissionStatus, ChemicalTimeAlignmentError, TimedChemicalAggregation,
    TimedChemicalPercept, classify_chemical_temporal_admission,
};

pub const CHEMICAL_TEMPORAL_AUTHORIZATION_NAMESPACE: &str =
    "symthaea-chemosensation-temporal-authorization-v2";
const BLAKE3_256: &str = "blake3-256";
const HASH_DOMAIN: &[u8] = b"symthaea-chemosensation-temporal-authorization-v2";

/// Strong domain identity for one exact chemical temporal-admission evidence set.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ChemicalTemporalAuthorizationId([u8; 32]);

impl ChemicalTemporalAuthorizationId {
    /// Revalidate and content-address the timing evidence retained by one chemical
    /// aggregation result.
    ///
    /// The caller-order admission is recomputed first so a forged/stale admission
    /// summary cannot be hashed as authoritative. Only an actually aggregated,
    /// temporally permitted result may receive an *authorization* ID. The
    /// component set is then canonicalized by raw observation identity before
    /// hashing, making the final ID insensitive to input ordering while rejecting
    /// duplicate observations.
    pub fn from_aggregation(
        aggregation: &TimedChemicalAggregation,
    ) -> Result<Self, ChemicalTemporalAuthorizationError> {
        let stored = aggregation.admission();
        let recomputed = classify_chemical_temporal_admission(
            aggregation.timed_components(),
            stored.max_component_skew_us(),
        )?;
        if &recomputed != stored {
            return Err(ChemicalTemporalAuthorizationError::AdmissionMismatch);
        }
        if !stored.permits_same_cycle_aggregation() {
            return Err(ChemicalTemporalAuthorizationError::NotAuthorized {
                status: stored.status(),
            });
        }
        if !aggregation.was_aggregated() {
            return Err(ChemicalTemporalAuthorizationError::AggregationVariantMismatch);
        }

        let mut canonical: Vec<TimedChemicalPercept> = aggregation.timed_components().to_vec();
        canonical.sort_by_key(|component| *component.observation_id().as_bytes());
        if canonical.windows(2).any(|pair| {
            pair[0].observation_id().as_bytes() == pair[1].observation_id().as_bytes()
        }) {
            return Err(ChemicalTemporalAuthorizationError::DuplicateObservationId);
        }

        let canonical_admission = classify_chemical_temporal_admission(
            &canonical,
            stored.max_component_skew_us(),
        )?;

        let mut hasher = Hasher::new();
        hasher.update(HASH_DOMAIN);
        hash_u64(&mut hasher, canonical.len() as u64);
        for component in &canonical {
            hash_timed_component(&mut hasher, component);
        }
        hash_admission(&mut hasher, &canonical_admission);

        Ok(Self(*hasher.finalize().as_bytes()))
    }

    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    pub fn content_address(&self) -> Result<ContentAddress32, ContentAddressError> {
        ContentAddress32::new(BLAKE3_256, CHEMICAL_TEMPORAL_AUTHORIZATION_NAMESPACE, self.0)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChemicalTemporalAuthorizationError {
    Time(ChemicalTimeAlignmentError),
    AdmissionMismatch,
    NotAuthorized {
        status: ChemicalTemporalAdmissionStatus,
    },
    AggregationVariantMismatch,
    DuplicateObservationId,
}

impl fmt::Display for ChemicalTemporalAuthorizationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Time(error) => write!(f, "temporal evidence could not be revalidated: {error}"),
            Self::AdmissionMismatch => write!(
                f,
                "stored chemical temporal admission does not reproduce from retained timed components"
            ),
            Self::NotAuthorized { status } => write!(
                f,
                "chemical temporal admission status {status:?} does not authorize same-cycle aggregation"
            ),
            Self::AggregationVariantMismatch => write!(
                f,
                "chemical temporal admission permits aggregation but the retained result is an abstention"
            ),
            Self::DuplicateObservationId => write!(
                f,
                "temporal authorization contains the same chemical observation more than once"
            ),
        }
    }
}

impl std::error::Error for ChemicalTemporalAuthorizationError {}

impl From<ChemicalTimeAlignmentError> for ChemicalTemporalAuthorizationError {
    fn from(value: ChemicalTimeAlignmentError) -> Self {
        Self::Time(value)
    }
}

fn hash_timed_component(hasher: &mut Hasher, component: &TimedChemicalPercept) {
    hasher.update(component.observation_id().as_bytes());
    hash_u64(hasher, component.comparison_timestamp_us());
    hash_time_receipt(hasher, component.time());
    match component.normalization() {
        None => hash_tag(hasher, 0),
        Some(normalized) => {
            hash_tag(hasher, 1);
            hash_normalized_time_point(hasher, normalized);
        }
    }
    match component.acquisition_authorization() {
        None => hash_tag(hasher, 0),
        Some(address) => {
            hash_tag(hasher, 1);
            hash_content_address(hasher, address);
        }
    }
}

fn hash_content_address(hasher: &mut Hasher, address: &ContentAddress32) {
    hash_str(hasher, address.algorithm());
    hash_str(hasher, address.namespace());
    hasher.update(address.digest());
}

fn hash_normalized_time_point(hasher: &mut Hasher, normalized: &NormalizedTimePoint) {
    hash_u64(hasher, normalized.source_timestamp_us());
    hash_time_receipt(hasher, normalized.source_receipt());
    hash_u64(hasher, normalized.target_timestamp_us());
    hash_time_receipt(hasher, normalized.target_receipt());
    hash_transform(hasher, normalized.transform());
}

fn hash_time_receipt(hasher: &mut Hasher, receipt: &TimeIntegrityReceipt) {
    hash_str(hasher, receipt.clock_domain.as_str());
    hash_optional_epoch(hasher, receipt.clock_epoch.as_ref());
    hash_continuity(hasher, receipt.continuity);
    hash_uncertainty(hasher, receipt.uncertainty);
    hash_optional_u64(hasher, receipt.sequence);
}

fn hash_transform(hasher: &mut Hasher, transform: &ClockTransformReceipt) {
    hash_str(hasher, transform.source_domain().as_str());
    hash_str(hasher, transform.source_epoch().as_str());
    hash_str(hasher, transform.target_domain().as_str());
    hash_str(hasher, transform.target_epoch().as_str());
    match transform.model() {
        ClockTransformModel::Offset {
            source_anchor_us,
            target_anchor_us,
        } => {
            hash_tag(hasher, 0);
            hash_u64(hasher, *source_anchor_us);
            hash_u64(hasher, *target_anchor_us);
        }
    }
    let (valid_start_us, valid_end_us) = transform.valid_source_range_us();
    hash_u64(hasher, valid_start_us);
    hash_u64(hasher, valid_end_us);
    hash_continuity(hasher, transform.mapping_continuity());
    hash_continuity(hasher, transform.target_continuity());
    hash_uncertainty(hasher, transform.uncertainty());
    hash_optional_u64(hasher, transform.sequence());
}

fn hash_admission(hasher: &mut Hasher, admission: &crate::ChemicalTemporalAdmission) {
    hash_tag(
        hasher,
        match admission.status() {
            ChemicalTemporalAdmissionStatus::NoComparisonNeeded => 0,
            ChemicalTemporalAdmissionStatus::DefinitelyWithin => 1,
            ChemicalTemporalAdmissionStatus::Ambiguous => 2,
            ChemicalTemporalAdmissionStatus::DefinitelyOutside => 3,
        },
    );
    hash_u64(hasher, admission.max_component_skew_us());
    hash_str(hasher, admission.clock_domain().as_str());
    hash_optional_epoch(hasher, admission.clock_epoch());
    hash_u64(hasher, admission.pairwise_windows().len() as u64);
    for pair in admission.pairwise_windows() {
        hash_u64(hasher, pair.left_index as u64);
        hash_u64(hasher, pair.right_index as u64);
        hash_u64(hasher, pair.separation.nominal_us);
        hash_u64(hasher, pair.separation.minimum_us);
        hash_u64(hasher, pair.separation.maximum_us);
    }
}

fn hash_continuity(hasher: &mut Hasher, status: ContinuityStatus) {
    hash_tag(
        hasher,
        match status {
            ContinuityStatus::Unverified => 0,
            ContinuityStatus::Continuous => 1,
            ContinuityStatus::Broken => 2,
        },
    );
}

fn hash_uncertainty(hasher: &mut Hasher, uncertainty: TimeUncertainty) {
    match uncertainty {
        TimeUncertainty::Unbounded => hash_tag(hasher, 0),
        TimeUncertainty::Bounded { max_error_us } => {
            hash_tag(hasher, 1);
            hash_u64(hasher, max_error_us);
        }
    }
}

fn hash_optional_epoch(hasher: &mut Hasher, epoch: Option<&ClockEpochId>) {
    match epoch {
        None => hash_tag(hasher, 0),
        Some(epoch) => {
            hash_tag(hasher, 1);
            hash_str(hasher, epoch.as_str());
        }
    }
}

fn hash_optional_u64(hasher: &mut Hasher, value: Option<u64>) {
    match value {
        None => hash_tag(hasher, 0),
        Some(value) => {
            hash_tag(hasher, 1);
            hash_u64(hasher, value);
        }
    }
}

fn hash_tag(hasher: &mut Hasher, tag: u8) {
    hasher.update(&[tag]);
}

fn hash_u64(hasher: &mut Hasher, value: u64) {
    hasher.update(&value.to_le_bytes());
}

fn hash_str(hasher: &mut Hasher, value: &str) {
    hash_u64(hasher, value.len() as u64);
    hasher.update(value.as_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ChemicalClockDomainId, ChemicalEncodingSpaceId, ChemicalFingerprint, ChemicalModalBridge,
        ChemicalModalBridgeConfig, ChemicalModality, ChemicalObservation, ChemicalPercept,
        aggregate_timed_chemical_percepts,
    };
    use symthaea_core::hdc::{HDC_DIMENSION, unified_hv::ContinuousHV};
    use symthaea_time_integrity::{ClockDomainId, ClockEpochId};
    use symthaea_time_normalization::normalize_timestamp_us;

    fn domain() -> ClockDomainId {
        ClockDomainId::new("capture/monotonic").unwrap()
    }

    fn epoch() -> ClockEpochId {
        ClockEpochId::new("capture-boot-1").unwrap()
    }

    fn percept(timestamp_us: u64, source: &str, seed: u64) -> ChemicalPercept {
        let mut evidence = ChemicalObservation::new(
            timestamp_us,
            ChemicalModality::Olfactory,
            source,
            vec![],
        );
        evidence.clock_domain = Some(ChemicalClockDomainId::new("capture/monotonic").unwrap());
        ChemicalPercept {
            evidence,
            fingerprint: ChemicalFingerprint {
                vector: ContinuousHV::random(HDC_DIMENSION, seed),
                confidence: 0.9,
                used_channels: 1,
                ignored_channels: 0,
                encoding_space_id: ChemicalEncodingSpaceId::from_bytes([7; 32]),
            },
        }
    }

    fn receipt() -> TimeIntegrityReceipt {
        TimeIntegrityReceipt::declared(domain())
            .with_epoch(epoch())
            .with_continuity(ContinuityStatus::Continuous)
            .with_uncertainty(TimeUncertainty::bounded(10))
    }

    fn timed(timestamp_us: u64, source: &str, seed: u64) -> TimedChemicalPercept {
        TimedChemicalPercept::new(percept(timestamp_us, source, seed), receipt()).unwrap()
    }

    fn timed_with_authority(
        timestamp_us: u64,
        source: &str,
        seed: u64,
        authority_byte: u8,
    ) -> TimedChemicalPercept {
        let transform = ClockTransformReceipt::offset(
            domain(),
            epoch(),
            domain(),
            epoch(),
            timestamp_us,
            timestamp_us,
            timestamp_us.saturating_sub(100),
            timestamp_us.saturating_add(100),
        )
        .unwrap()
        .with_mapping_continuity(ContinuityStatus::Continuous)
        .with_target_continuity(ContinuityStatus::Continuous)
        .with_uncertainty(TimeUncertainty::bounded(0));
        let normalized = normalize_timestamp_us(timestamp_us, &receipt(), &transform).unwrap();
        let authority = ContentAddress32::new(
            "blake3-256",
            "symthaea-chemosensation-acquisition-time-authorization-v1",
            [authority_byte; 32],
        )
        .unwrap();
        TimedChemicalPercept::from_evidence_bound_normalized(
            percept(timestamp_us, source, seed),
            normalized,
            authority,
        )
        .unwrap()
    }

    fn aggregate(components: Vec<TimedChemicalPercept>, skew_us: u64) -> TimedChemicalAggregation {
        aggregate_timed_chemical_percepts(
            &ChemicalModalBridge::new(ChemicalModalBridgeConfig {
                max_component_skew_us: skew_us,
            }),
            components,
        )
        .unwrap()
    }

    #[test]
    fn identity_is_order_independent_for_same_exact_timing_evidence() {
        let a = timed(1_000, "nose-a", 1);
        let b = timed(1_050, "nose-b", 2);
        let left = aggregate(vec![a.clone(), b.clone()], 100);
        let right = aggregate(vec![b, a], 100);

        let left_id = ChemicalTemporalAuthorizationId::from_aggregation(&left).unwrap();
        let right_id = ChemicalTemporalAuthorizationId::from_aggregation(&right).unwrap();
        assert_eq!(left_id, right_id);
    }

    #[test]
    fn acquisition_authority_reference_is_part_of_v2_temporal_identity() {
        let left = aggregate(
            vec![
                timed_with_authority(1_000, "nose-a", 1, 1),
                timed_with_authority(1_050, "nose-b", 2, 2),
            ],
            100,
        );
        let right = aggregate(
            vec![
                timed_with_authority(1_000, "nose-a", 1, 9),
                timed_with_authority(1_050, "nose-b", 2, 2),
            ],
            100,
        );
        assert_eq!(left.admission(), right.admission());
        assert_ne!(
            ChemicalTemporalAuthorizationId::from_aggregation(&left).unwrap(),
            ChemicalTemporalAuthorizationId::from_aggregation(&right).unwrap()
        );
    }

    #[test]
    fn skew_policy_is_part_of_temporal_authorization_identity() {
        let a = timed(1_000, "nose-a", 1);
        let b = timed(1_050, "nose-b", 2);
        let tighter = aggregate(vec![a.clone(), b.clone()], 100);
        let looser = aggregate(vec![a, b], 200);

        assert_ne!(
            ChemicalTemporalAuthorizationId::from_aggregation(&tighter).unwrap(),
            ChemicalTemporalAuthorizationId::from_aggregation(&looser).unwrap()
        );
    }

    #[test]
    fn duplicate_observation_cannot_be_laundered_as_more_temporal_evidence() {
        let a = timed(1_000, "nose-a", 1);
        let duplicated = aggregate(vec![a.clone(), a], 100);
        assert_eq!(
            ChemicalTemporalAuthorizationId::from_aggregation(&duplicated),
            Err(ChemicalTemporalAuthorizationError::DuplicateObservationId)
        );
    }

    #[test]
    fn abstention_cannot_be_minted_as_temporal_authorization() {
        let outside = aggregate(
            vec![timed(1_000, "nose-a", 1), timed(1_050, "nose-b", 2)],
            10,
        );
        assert_eq!(
            ChemicalTemporalAuthorizationId::from_aggregation(&outside),
            Err(ChemicalTemporalAuthorizationError::NotAuthorized {
                status: ChemicalTemporalAdmissionStatus::DefinitelyOutside,
            })
        );
    }

    #[test]
    fn generic_address_preserves_domain_namespace() {
        let aggregation = aggregate(
            vec![timed(1_000, "nose-a", 1), timed(1_050, "nose-b", 2)],
            100,
        );
        let id = ChemicalTemporalAuthorizationId::from_aggregation(&aggregation).unwrap();
        let address = id.content_address().unwrap();
        assert_eq!(address.algorithm(), BLAKE3_256);
        assert_eq!(address.namespace(), CHEMICAL_TEMPORAL_AUTHORIZATION_NAMESPACE);
        assert_eq!(address.digest(), id.as_bytes());
    }
}

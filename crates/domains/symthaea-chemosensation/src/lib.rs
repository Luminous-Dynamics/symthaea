// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! # symthaea-chemosensation
//!
//! Shared foundations for artificial olfaction and gustation.
//!
//! This crate deliberately separates **physical observations** from learned
//! percepts. Hardware produces [`ChemicalObservation`] values with calibration,
//! environment, health, clock-domain, and provenance metadata. Higher layers may
//! then derive odor, taste, flavor, novelty, and semantic hypotheses without
//! overwriting the underlying measurement.
//!
//! Representation and evidence identities remain deliberately distinct:
//!
//! - [`ChemicalObservationId`] / [`ChemicalEvidenceBundleId`] identify raw evidence,
//! - [`ChemicalClockDomainId`] identifies the legacy raw acquisition time domain,
//! - [`ChemicalAcquisitionTimeAuthorizationId`] identifies the exact calibration
//!   decision + holdover authority that permitted acquisition-time normalization,
//! - [`TimedChemicalPercept`] carries a separate comparison timestamp + generic
//!   [`symthaea_time_integrity::TimeIntegrityReceipt`] without mutating raw evidence,
//! - [`bind_evidence_bound_acquisition_time`] derives that comparison time only
//!   through a self-verifying bounded calibration/holdover transform,
//! - [`TimedChemicalAggregation`] retains generic temporal admission beside the
//!   exact raw evidence-preserving HDC aggregate,
//! - [`ChemicalTemporalAuthorizationId`] content-addresses the exact timing
//!   evidence + skew policy + admission result that justified that aggregation,
//! - [`TimedChemicalRootProjection`] revalidates timing and HDC aggregation before
//!   BinaryHV root projection,
//! - [`ChemicalEncodingSpaceId`] identifies the continuous HDC coordinate system,
//! - [`ChemicalRootProjectionPolicyId`] identifies ContinuousHV -> BinaryHV quantization,
//! - [`ChemicalRootBinarySpaceId`] identifies the resulting root BinaryHV space,
//! - semantic labels remain later hypotheses rather than identity.
//!
//! The crate starts hardware-independent so cognition and experiments can be
//! validated against deterministic simulated noses/tongues before real sensor
//! drivers are introduced.

#![deny(unsafe_code)]

pub mod acquisition_authorization;
pub mod acquisition_time;
pub mod calibration;
pub mod clock;
pub mod cognition;
pub mod content_address_adapter;
pub mod encoding;
pub mod evidence;
pub mod fingerprint;
pub mod flavor;
pub mod gustation;
pub mod multimodal_bridge;
pub mod novelty;
pub mod observation;
pub mod olfaction;
pub mod percept;
mod projection_geometry;
pub mod projection_identity;
pub mod projection_stability;
pub mod projection_study;
pub mod root_projection;
pub mod temporal;
pub mod temporal_authorization;
pub mod time_alignment;
pub mod timed_multimodal_bridge;
pub mod timed_root_projection;

pub use acquisition_authorization::{
    CHEMICAL_ACQUISITION_TIME_AUTHORIZATION_NAMESPACE,
    ChemicalAcquisitionTimeAuthorizationError, ChemicalAcquisitionTimeAuthorizationId,
};
pub use acquisition_time::{
    ChemicalAcquisitionTimeError, bind_evidence_bound_acquisition_time,
};
pub use calibration::{CalibrationId, CalibrationState, SensorHealth};
pub use clock::{
    ChemicalClockDomainError, ChemicalClockDomainId, MAX_CHEMICAL_CLOCK_DOMAIN_LEN,
};
pub use cognition::{
    ChemicalCognitionError, ChemicalCognitionPipeline, CognitiveChemicalPercept,
};
pub use content_address_adapter::{
    CHEMICAL_ENCODING_SPACE_NAMESPACE, CHEMICAL_EVIDENCE_BUNDLE_NAMESPACE,
    CHEMICAL_OBSERVATION_NAMESPACE, CHEMICAL_ROOT_BINARY_SPACE_NAMESPACE,
    CHEMICAL_ROOT_PROJECTION_POLICY_NAMESPACE, ChemicalContentAddressError,
    ChemicalRootContentLineage,
};
pub use encoding::ScalarHdcEncoder;
pub use evidence::{ChemicalEvidenceBundleId, ChemicalObservationId};
pub use fingerprint::{
    ChannelEncodingSpec, ChemicalEncodingSpaceId, ChemicalFingerprint, ChemicalFingerprintEncoder,
    FingerprintConfigError, FingerprintError,
};
pub use flavor::{
    FlavorBinder, FlavorBindingConfig, FlavorBindingError, FlavorConfigError, FlavorPercept,
};
pub use gustation::{
    ElectronicTongueSimulator, GustatorySimulationError, GustatoryStimulus,
    PotentiometricChannelModel,
};
pub use multimodal_bridge::{
    ChemicalBridgeTarget, ChemicalModalBridge, ChemicalModalBridgeConfig,
    ChemicalModalBridgeError, ChemicalModalBridgeInput,
};
pub use novelty::{
    ChemicalMemoryReference, ChemicalNoveltyConfig, ChemicalNoveltyMemory, NoveltyAssessment,
    NoveltyConfigError,
};
pub use observation::{
    ChemicalChannel, ChemicalModality, ChemicalObservation, EnvironmentReading, MeasurementUnit,
};
pub use olfaction::{
    MoxArraySimulator, MoxChannelModel, OlfactorySimulationError, OlfactoryStimulus,
};
pub use percept::{ChemicalPercept, ChemicalPerceptEncoder};
pub use projection_identity::{ChemicalRootBinarySpaceId, ChemicalRootProjectionPolicyId};
pub use projection_stability::{
    ChemicalProjectionMarginAssessment, ChemicalProjectionStabilityDatasetAssessment,
};
pub use projection_study::{
    ChemicalProjectionDatasetAssessment, ChemicalProjectionStudyError,
};
pub use root_projection::{
    ChemicalProjectionPairAssessment, ChemicalProjectionQuality, ChemicalRootProjection,
    ChemicalRootProjectionConfig, ChemicalRootProjectionError, ChemicalRootProjector,
};
pub use temporal::{
    ChemicalTemporalContext, ChemicalTemporalTracker, TemporalConfigError, TemporalError,
};
pub use temporal_authorization::{
    CHEMICAL_TEMPORAL_AUTHORIZATION_NAMESPACE, ChemicalTemporalAuthorizationError,
    ChemicalTemporalAuthorizationId,
};
pub use time_alignment::{
    ChemicalPairwiseTimeWindow, ChemicalTemporalAdmission, ChemicalTemporalAdmissionStatus,
    ChemicalTimeAlignmentError, TimedChemicalPercept, classify_chemical_temporal_admission,
};
pub use timed_multimodal_bridge::{
    TimedChemicalAggregation, TimedChemicalAggregationError, aggregate_timed_chemical_percepts,
};
pub use timed_root_projection::{
    TimedChemicalRootProjection, TimedChemicalRootProjectionError,
};

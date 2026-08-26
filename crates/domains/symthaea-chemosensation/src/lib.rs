// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! # symthaea-chemosensation
//!
//! Shared foundations for artificial olfaction and gustation.
//!
//! This crate deliberately separates **physical observations** from learned
//! percepts. Hardware produces [`ChemicalObservation`] values with calibration,
//! environment, health, and provenance metadata. Higher layers may then derive
//! odor, taste, flavor, novelty, and semantic hypotheses without overwriting the
//! underlying measurement.
//!
//! Representation and evidence identities remain deliberately distinct:
//!
//! - [`ChemicalObservationId`] / [`ChemicalEvidenceBundleId`] identify raw evidence,
//! - [`ChemicalEncodingSpaceId`] identifies the continuous HDC coordinate system,
//! - [`ChemicalRootProjectionPolicyId`] identifies ContinuousHV -> BinaryHV quantization,
//! - [`ChemicalRootBinarySpaceId`] identifies the resulting root BinaryHV space,
//! - semantic labels remain later hypotheses rather than identity.
//!
//! The crate starts hardware-independent so cognition and experiments can be
//! validated against deterministic simulated noses/tongues before real sensor
//! drivers are introduced.

#![deny(unsafe_code)]

pub mod calibration;
pub mod cognition;
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
pub mod projection_identity;
pub mod root_projection;
pub mod temporal;

pub use calibration::{CalibrationId, CalibrationState, SensorHealth};
pub use cognition::{
    ChemicalCognitionError, ChemicalCognitionPipeline, CognitiveChemicalPercept,
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
pub use root_projection::{
    ChemicalProjectionPairAssessment, ChemicalProjectionQuality, ChemicalRootProjection,
    ChemicalRootProjectionConfig, ChemicalRootProjectionError, ChemicalRootProjector,
};
pub use temporal::{
    ChemicalTemporalContext, ChemicalTemporalTracker, TemporalConfigError, TemporalError,
};

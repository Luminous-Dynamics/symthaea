// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! # symthaea-chemosensation
//!
//! Shared foundations for artificial olfaction and gustation.
//!
//! This crate deliberately separates **physical observations** from learned
//! percepts. Hardware produces [`ChemicalObservation`] values with calibration,
//! environment, health, provenance, and optional sampling-protocol metadata.
//! Higher layers may then derive odor, taste, flavor, novelty, and semantic
//! hypotheses without overwriting the underlying measurement.
//!
//! The crate starts hardware-independent so cognition and experiments can be
//! validated against deterministic simulated noses/tongues before real sensor
//! drivers are introduced.

#![deny(unsafe_code)]

pub mod archive;
pub mod calibration;
pub mod cognition;
pub mod deviation;
pub mod encoding;
pub mod evidence_bundle;
pub mod experiment;
pub mod fingerprint;
pub mod flavor;
pub mod gustation;
pub mod novelty;
pub mod observation;
pub mod olfaction;
pub mod percept;
pub mod sampling;
pub mod temporal;
pub mod trace;

pub use archive::{
    ChemicalTraceArchive, TraceArchiveDigest, TraceArchiveError, TraceArchiveManifest,
    VerifiedChemicalReplay, TRACE_ARCHIVE_SCHEMA_VERSION,
};
pub use calibration::{CalibrationId, CalibrationState, SensorHealth};
pub use cognition::{
    ChemicalCognitionError, ChemicalCognitionPipeline, CognitiveChemicalPercept,
};
pub use deviation::{
    DeviationDisposition, ExpectedBiasDirection, ProtocolDeviation, ProtocolDeviationError,
};
pub use encoding::ScalarHdcEncoder;
pub use evidence_bundle::{
    ChemicalEvidenceBundle, EvidenceBundleError, TraceEvidenceRef, TraceEvidenceRefError,
    EVIDENCE_BUNDLE_SCHEMA_VERSION,
};
pub use experiment::{
    ChemicalDecisionProtocol, ChemicalDecisionReceipt, ChemicalEvidenceLevel, DecisionError,
    DecisionProtocolError, EvaluationPartition, ExperimentDecision, GateDirection, GateOutcome,
    MetricGate, MetricGateResult, MetricObservation,
};
pub use fingerprint::{
    ChannelEncodingSpec, ChemicalFingerprint, ChemicalFingerprintEncoder, FingerprintConfigError,
    FingerprintError,
};
pub use flavor::{
    FlavorBinder, FlavorBindingConfig, FlavorBindingError, FlavorConfigError, FlavorPercept,
};
pub use gustation::{
    ElectronicTongueSimulator, GustatorySimulationError, GustatoryStimulus,
    PotentiometricChannelModel,
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
pub use sampling::{SamplingContext, SamplingContextError, SamplingPhase};
pub use temporal::{
    ChemicalTemporalContext, ChemicalTemporalTracker, TemporalConfigError, TemporalError,
};
pub use trace::{ChemicalTrace, ChemicalTraceError};

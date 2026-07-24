#![deny(unsafe_code)]
// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
#![allow(clippy::needless_range_loop)]

//! Symthaea Vision Manifold: Patch-based HDC video encoding with CfC temporal dynamics.
//!
//! Encodes video frames into 16,384-dimensional holographic hypervectors using the
//! bind-bundle paradigm from Hyperdimensional Computing, then tracks scene state via
//! Closed-form Continuous-time (CfC) neurons with O(1) temporal prediction.
//!
//! # Architecture
//!
//! ```text
//! pixels → PatchHdcEncoder → ContinuousHV → VisionManifold (CfC) → prediction + surprise
//!                                                 ↕
//!                                          SurpriseMap (per-patch free energy)
//! ```
//!
//! # Key properties
//!
//! - **Holographic encoding**: Each frame is a superposition of position-bound patch
//!   appearances. Similar frames produce similar HVs; spatial layout is preserved.
//! - **O(1) temporal prediction**: The CfC closed-form solution
//!   `state' = x_inf + (state - x_inf) · exp(-dt/τ)` has cost independent of dt.
//! - **Surprise-driven attention**: Per-patch prediction error identifies regions
//!   of unexpected change (active inference foraging).
//! - **TemporalPredictor**: Implements the shared trait for cross-domain integration.

pub mod attention;
pub mod bridge;
pub mod camera;
pub mod checkpoint;
pub mod encoder;
pub mod manifold;
pub mod predictive;
pub mod spectrum;
pub mod training;
pub mod types;

pub use attention::SurpriseMap;
pub use bridge::{
    CROSS_MANIFOLD_PREDICTOR_STATE_SCHEMA_VERSION, CognitiveGoalSignal, CognitiveGoalSignalState,
    CrossManifoldPredictor, CrossManifoldPredictorState, VISION_BRIDGE_STATE_SCHEMA_VERSION,
    VisionBridge, VisionBridgeState,
};
#[cfg(feature = "camera")]
pub use camera::CameraSource;
pub use camera::{
    CAMERA_MANIFOLD_STATE_SCHEMA_VERSION, CameraCapturePolicy, CameraManifold, CameraManifoldState,
    CapturedFrame, DEFAULT_MOCK_CAMERA_MAX_FRAME_BYTES,
    DEFAULT_REAL_CAMERA_MAX_DECODED_FRAME_BYTES, DEFAULT_REAL_CAMERA_MAX_MMAP_BYTES,
    DEFAULT_REAL_CAMERA_MAX_RAW_FRAME_BYTES, DEFAULT_REAL_CAMERA_MMAP_BUFFER_COUNT,
    MockCameraSource, MockCameraSourceState,
};
pub use checkpoint::{
    CHECKPOINT_ENVELOPE_SCHEMA_VERSION, CheckpointEnvelope, CheckpointGenerationInspection,
    CheckpointGenerationLocation, CheckpointLoadReport, CheckpointMetadata, CheckpointPruneReport,
    CheckpointRecoveryAttempt, CheckpointRecoveryAttemptOutcome, CheckpointRecoverySource,
    CheckpointRetentionPolicy, CheckpointRetentionSaveReport, CheckpointSaveReport,
    CheckpointSemanticRecoveryFailure, CheckpointSemanticRecoveryReport, CheckpointWriteReport,
    CheckpointWriterLock, CheckpointWriterLockEvidence, CheckpointWriterLockPolicy,
    DEFAULT_MAX_CHECKPOINT_AUTH_TAG_BYTES, DEFAULT_MAX_CHECKPOINT_ENVELOPE_BYTES,
    DEFAULT_MAX_CHECKPOINT_PAYLOAD_BYTES, DEFAULT_STALE_CHECKPOINT_TEMP_AGE,
    DEFAULT_STALE_CHECKPOINT_TEMP_LIMIT, MAX_CHECKPOINT_LOCK_ATTEMPTS,
    MAX_CHECKPOINT_PREVIOUS_GENERATIONS, acquire_checkpoint_writer_lock,
    checkpoint_generation_path, checkpoint_previous_path, checkpoint_writer_lock_path,
    cleanup_checkpoint_temps, decode_authenticated_checkpoint, decode_checkpoint,
    encode_authenticated_checkpoint, encode_checkpoint, inspect_checkpoint,
    inspect_checkpoint_file, inspect_checkpoint_generations, inspect_checkpoint_writer_lock,
    load_authenticated_checkpoint_file, load_authenticated_checkpoint_file_report,
    load_authenticated_checkpoint_file_with_retention_audited_detailed, load_checkpoint_file,
    load_checkpoint_file_recoverable, load_checkpoint_file_recoverable_promote,
    load_checkpoint_file_recoverable_report, load_checkpoint_file_report,
    load_checkpoint_file_with_retention, load_checkpoint_file_with_retention_audited,
    load_checkpoint_file_with_retention_audited_detailed,
    load_checkpoint_file_with_retention_report, load_checkpoint_file_with_retention_validated,
    load_checkpoint_file_with_retention_validated_report, max_compact_envelope_bytes,
    max_envelope_bytes, prune_checkpoint_generations, prune_checkpoint_generations_locked,
    read_checkpoint_bounded, save_authenticated_checkpoint_file,
    save_authenticated_checkpoint_file_with_retention_locked_report,
    save_authenticated_checkpoint_file_with_retention_report, save_checkpoint_file,
    save_checkpoint_file_recoverable, save_checkpoint_file_recoverable_report,
    save_checkpoint_file_report, save_checkpoint_file_with_retention,
    save_checkpoint_file_with_retention_locked_report, save_checkpoint_file_with_retention_report,
    with_checkpoint_writer_lock, write_checkpoint_atomic, write_checkpoint_atomic_report,
};
pub use encoder::{MotionField, MultiScaleEncoder, PatchHdcEncoder, StereoDepthEstimate};
pub use manifold::{
    DelayedHorizonEvaluator, HorizonAccuracy, ObjectMemory, ObjectTrackingResult, SceneMemory,
    TrackedObject, VisionManifold, VisualSceneGraph, VisualWorkingMemory, WorkingMemorySlot,
};
pub use predictive::{PredictiveCodingHierarchy, PredictiveOutput};
pub use spectrum::{
    BandProbeEvidence, BandProbeScore, MultiSpectralEncoder, MultiSpectralEncoderState,
    MultiSpectralFrame, SpectralBandEncoderState, SpectralLayer, SpectrumBand,
};
pub use training::{BpttResult, ManifoldTrainer};
pub use types::{
    AdamStateSnapshot, AttentionMap, DELAYED_HORIZON_EVALUATOR_STATE_SCHEMA_VERSION,
    DelayedHorizonEvaluatorState, DilationEstimate, HorizonAccumulatorState, LearningConfig,
    ManifoldHealth, ManifoldState, ModalityTemporalContextState, MultiScaleConfig,
    ObjectHypothesis, PatchGrid, PendingHorizonForecastState, SalientRegion, ScaleHealth,
    SceneFrameMetadata, SceneGraphEdge, SceneMatch, SceneMemoryState, SpatialRelation,
    TrainerState, TrainingConfig, TrainingMethod, VisionConfig, VisionTelemetry, VisualModality,
};

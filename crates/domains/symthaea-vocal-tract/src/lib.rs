// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # symthaea-vocal-tract
//!
//! LTC-driven articulatory synthesis: HDC encoder, controller, FEP agent, pipeline, metrics.
//!
//! Uses the full 16,384D `HdcLtcUnifiedNetwork` from `symthaea-core` as the temporal
//! dynamics engine, with `symthaea-fep` providing precision-weighted Active Inference
//! modulation. Multi-rate architecture: 200Hz motor reflex + 10Hz cognitive tick.
//!
//! ## Architecture
//!
//! ```text
//! VoiceCognitiveState (10D)
//!       ↓
//! VocalTractHdcEncoder → ContinuousHV(16384D)
//!       ↓                    ↓
//!       ↓          [bind with phoneme HV]
//!       ↓                    ↓
//! VocalTractController (HdcLtcUnifiedNetwork + output head 16384→9)
//!       ↓
//! FormantFrame [F1, F2, F3, B1, B2, B3, F0, energy, voicing]
//!
//! Every 20th motor step (10Hz):
//!   VocalTractFepAgent modulates τ, learning rate, emphasis
//! ```
//!
//! ## Features
//!
//! - `hound` — WAV file I/O for offline analysis (metrics module)

#![deny(unsafe_code)]
#![allow(clippy::needless_range_loop)]

pub mod controller;
pub mod encoder;
pub mod fep;
pub mod formant_extraction;
pub mod metrics;
pub mod phonetics;
pub mod pipeline;
pub mod speech;
pub mod types;

// Re-export core types at crate root
pub use controller::{
    ProsodyCorrection, ProsodyHead, SpeakerProfile, VocalTractConfig, VocalTractController,
};
pub use encoder::{VocalTractHdcEncoder, VoiceCognitiveState, VoiceCognitiveStateDerivatives};
pub use fep::{
    FepTelemetry, VocalAction, VocalTractFepAgent, VocalTractFepResult, VocalTractObservation,
};
pub use metrics::{PerceptualMetrics, VocalTractMetrics, compute_hnr, compute_spectral_tilt};
pub use pipeline::{
    Intonation, PitchAccent, ProsodyContext, VocalTractPipeline, detect_syllable_boundaries,
    is_consonant_phoneme, is_vowel_phoneme, predict_duration,
};
pub use types::{FormantFrame, FormantTarget};

#[cfg(feature = "mel-conversion")]
pub mod formant_to_mel;

// Checkpoint-durability/attestation subsystem (see TRACK_B_RECOVERY_PLAN_2026-07-30.md).
// Thematically unrelated to voice synthesis; opt-in only.
//
// Only the modules below are wired in -- they compile clean and their own tests pass (8/15
// checkpoint_*.rs files; the 13 figure in earlier docs undercounted the real file count). The
// other 7 checkpoint_*.rs files remain deliberately UNWIRED, each blocked on real, disclosed
// gaps out of scope for this pass (see CHECKPOINT_DURABILITY_INTEGRATION_STATUS_2026-07-30.md
// and POWER_LOSS_CLUSTER_SEMANTICS_FREEZE_2026-07-30.md for the full breakdown):
// checkpoint_gossip_archive.rs, checkpoint_gossip_transport.rs, checkpoint_transparency_gossip.rs
// (need a separate, larger Merkle transparency-log primitive, not built this pass);
// checkpoint_hardware_signing.rs, checkpoint_hybrid_public_verifiability.rs,
// checkpoint_series21_public_verifiability.rs, checkpoint_series22_public_verifiability.rs
// (need the `fips204` post-quantum ML-DSA crate, a new dependency not added without checking
// in first).
#[cfg(feature = "checkpoint-durability")]
pub mod checkpoint_audit_archive;
#[cfg(feature = "checkpoint-durability")]
pub mod checkpoint_platform;
#[cfg(feature = "checkpoint-durability")]
pub mod checkpoint_power_loss_federation;
#[cfg(feature = "checkpoint-durability")]
pub mod checkpoint_power_loss_operations;
#[cfg(feature = "checkpoint-durability")]
pub mod checkpoint_replay;
#[cfg(feature = "checkpoint-durability")]
pub mod checkpoint_series20_public_verifiability;
#[cfg(feature = "checkpoint-durability")]
pub mod checkpoint_storage_evidence;
#[cfg(feature = "checkpoint-durability")]
pub mod checkpoint_trusted_time;

#[cfg(feature = "checkpoint-durability")]
pub use checkpoint_platform::{CheckpointFileLockGuard, effective_uid, lock_exclusive};
#[cfg(feature = "checkpoint-durability")]
pub use checkpoint_power_loss_operations::{
    CHECKPOINT_POWER_LOSS_LAB_SCHEMA, CHECKPOINT_POWER_LOSS_OPERATIONS_PLAN_SCHEMA,
    CheckpointPowerLossLabId, CheckpointPowerLossLabManifest, CheckpointPowerLossOperationsError,
    CheckpointPowerLossOperationsEvidence, CheckpointPowerLossOperationsKeyId,
    CheckpointPowerLossOperationsPlan, merge_checkpoint_power_loss_operations_evidence,
};
#[cfg(feature = "checkpoint-durability")]
pub use checkpoint_series20_public_verifiability::{
    CheckpointPublicKeyId, CheckpointPublicSignature, CheckpointPublicSigningKey,
    CheckpointPublicVerificationBundle, CheckpointPublicVerificationError,
    CheckpointPublicVerificationSummary, CheckpointPublicVerifyingKey,
    MAX_CHECKPOINT_PUBLIC_ARTIFACT_BYTES, MAX_CHECKPOINT_PUBLIC_SIGNATURE_DOMAIN_BYTES,
    MAX_CHECKPOINT_PUBLIC_SIGNERS,
};
#[cfg(feature = "checkpoint-durability")]
pub use checkpoint_storage_evidence::{
    CHECKPOINT_POWER_LOSS_CAMPAIGN_SCHEMA, CHECKPOINT_POWER_LOSS_EVIDENCE_SCHEMA,
    CHECKPOINT_POWER_LOSS_RESULT_SCHEMA, CheckpointDurabilityBoundary,
    CheckpointPowerLossCampaignEvidence, CheckpointPowerLossCampaignPlan,
    CheckpointPowerLossEvidenceClass, CheckpointPowerLossEvidenceKeyId,
    CheckpointPowerLossRecoveryOutcome, CheckpointPowerLossTrialPlan,
    CheckpointPowerLossTrialResult, CheckpointStorageEvidenceError,
    CheckpointStorageProfileAttestationKeyId,
};

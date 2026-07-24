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

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
pub mod metrics;
pub mod pipeline;
pub mod types;

// Re-export core types at crate root
pub use controller::{
    ProsodyCorrection, ProsodyHead, SpeakerProfile, VocalTractConfig, VocalTractController,
};
pub use encoder::{VocalTractHdcEncoder, VoiceCognitiveState};
pub use fep::{VocalAction, VocalTractFepAgent, VocalTractFepResult, VocalTractObservation};
pub use metrics::VocalTractMetrics;
pub use pipeline::{predict_duration, Intonation, ProsodyContext, VocalTractPipeline};
pub use types::{FormantFrame, FormantTarget};

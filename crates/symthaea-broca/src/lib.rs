//! # Symthaea Broca: SSM Language Center
//!
//! Native CfC-HDC autoregressive thought-to-text generation.
//!
//! Replaces external LLM backends with a local, linear-scaling neural generator
//! that uses the same HDC-LTC architecture as the VocalTractController, but
//! projects to token logits instead of formant parameters.
//!
//! # Architecture
//!
//! ```text
//! ThoughtChannels (20D)
//!     │
//!     ▼
//! ThoughtLanguageEncoder          [encoder.rs]
//!   normalize → level-encode → bind → bundle → 16,384D ContinuousHV
//!     │
//!     ▼
//! LanguageController              [controller.rs]
//!   HdcLtcUnifiedNetwork (3 layers × 8 neurons)
//!   Autoregressive: thought_hv ⊗ token_emb ⊗ permute(pos) → evolve → output
//!   Weight-tied output: logits[i] = similarity(output_hv, token_emb[i])
//!     │
//!     ▼
//! Per-Token Gating                [gating.rs]
//!   EpistemicGate + EmotionalModulator + CoherenceFeedback
//!     │
//!     ▼
//! BrocaGenerator                  [generator.rs]
//!   Sampling (greedy/top-k/top-p) → BpeTokenizer → text
//! ```
//!
//! # Key Innovation
//!
//! Epistemic status, emotional tone, and consciousness level become
//! *architectural constraints on generation* (per-token logit gating),
//! not just prompt instructions that a model might ignore.

// Allow lints that are pervasive in Mamba/SSM numerical code
#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::needless_borrow)]
#![allow(clippy::excessive_precision)]
#![allow(clippy::manual_div_ceil)]
#![allow(clippy::manual_is_multiple_of)]
#![allow(clippy::unnecessary_map_or)]

pub mod checkpoint;
pub mod controller;
pub mod encoder;
pub mod evaluation;
pub mod gating;
pub mod generator;
pub mod tokenizer;
pub mod training;

// Liquid-Mamba fusion: pre-trained Mamba SSM + HDC projection + consciousness gating
#[cfg(feature = "mamba-cpu")]
pub mod liquid_mamba;
#[cfg(feature = "mamba-cpu")]
pub mod mamba;
#[cfg(feature = "mamba-cpu")]
pub mod mamba_model;
#[cfg(feature = "mamba-cpu")]
pub mod projection;
#[cfg(feature = "mamba-cpu")]
pub mod temporal_projection;

pub use checkpoint::{AdamState, BrocaCheckpoint};
pub use controller::{LanguageController, LanguageControllerConfig, NetworkSnapshot};
pub use encoder::{ThoughtChannels, ThoughtLanguageEncoder};
pub use evaluation::{EvalConfig, EvalResult, IntentScore};
pub use gating::{CoherenceFeedback, EmotionalModulator, EpistemicGate, GatingConfig};
pub use generator::{BrocaConfig, BrocaGenerator, GenerationResult, SamplingStrategy};
pub use tokenizer::BpeTokenizer;
pub use training::{
    AnomalyReport, GradientAnomaly, GradientDiagnostics, TrainingDataset, TrainingPair,
    TrainingValidation,
};

#[cfg(feature = "mamba-cpu")]
pub use checkpoint::ProjectionCheckpoint;
#[cfg(feature = "mamba-cpu")]
pub use evaluation::{
    GatingTestResult, LiquidMambaEvalConfig, LiquidMambaEvalResult, QualityGateThresholds,
};
#[cfg(feature = "mamba-cpu")]
pub use liquid_mamba::{LiquidMambaConfig, LiquidMambaGenerator};
#[cfg(feature = "mamba-cpu")]
pub use mamba::MambaBackend;
#[cfg(feature = "mamba-cpu")]
pub use projection::{
    GradientDiagnosticsSnapshot, GradientStepMetrics, HdcSsmProjection,
    ProjectionGradientDiagnostics,
};
#[cfg(feature = "mamba-cpu")]
pub use temporal_projection::TemporalProjection;

// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
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
#[cfg(feature = "gpu")]
pub mod gpu_cfc;
pub mod speech_encoder;
pub mod tokenizer;
pub mod training;

// Speech heuristics: bootstrap targets for SpeechThoughtEncoder training
#[cfg(feature = "speech-data")]
pub mod speech_heuristics;

// Creative mode: relaxed gating for poetry and artistic text generation
pub mod creative_mode;
pub mod syllable;

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
pub use encoder::{
    ThoughtChannels, ThoughtLanguageEncoder, EPISTEMIC_CUBE_BASE, EPISTEMIC_CUBE_CHANNELS,
};
pub use evaluation::{EvalConfig, EvalResult, IntentScore};
pub use gating::{
    CodeGate, CoherenceFeedback, EmotionalModulator, EpistemicCubeGate, EpistemicGate, GatingConfig,
};
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

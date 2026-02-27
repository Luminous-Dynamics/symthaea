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

pub mod tokenizer;
pub mod encoder;
pub mod controller;
pub mod gating;
pub mod generator;
pub mod training;
pub mod checkpoint;
pub mod evaluation;

// Liquid-Mamba fusion: pre-trained Mamba SSM + HDC projection + consciousness gating
#[cfg(feature = "mamba")]
pub mod mamba;
#[cfg(feature = "mamba")]
pub mod projection;
#[cfg(feature = "mamba")]
pub mod liquid_mamba;

pub use tokenizer::BpeTokenizer;
pub use encoder::{ThoughtChannels, ThoughtLanguageEncoder};
pub use controller::{LanguageController, LanguageControllerConfig};
pub use gating::{EpistemicGate, EmotionalModulator, CoherenceFeedback, GatingConfig};
pub use generator::{BrocaGenerator, BrocaConfig, GenerationResult, SamplingStrategy};
pub use training::{TrainingPair, TrainingDataset, GradientDiagnostics};
pub use checkpoint::{BrocaCheckpoint, AdamState};
pub use evaluation::{EvalConfig, EvalResult, IntentScore};

#[cfg(feature = "mamba")]
pub use liquid_mamba::{LiquidMambaGenerator, LiquidMambaConfig};
#[cfg(feature = "mamba")]
pub use projection::HdcSsmProjection;
#[cfg(feature = "mamba")]
pub use evaluation::{LiquidMambaEvalConfig, LiquidMambaEvalResult};
#[cfg(feature = "mamba")]
pub use checkpoint::ProjectionCheckpoint;

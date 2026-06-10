#![allow(clippy::needless_range_loop)]
// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Symthaea Speech-to-Text Library
//!
//! A neuromorphic speech recognition system built on:
//! - **HDC (Hyperdimensional Computing)**: Efficient, interpretable representations
//! - **LTC (Liquid Time-Constant)**: Adaptive temporal dynamics
//! - **Modern Hopfield Networks**: Exponential-capacity associative memory
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                        SYMTHAEA STT PIPELINE                             │
//! ├─────────────────────────────────────────────────────────────────────────┤
//! │                                                                          │
//! │  Audio ──► LTC Projector ──► HDC Binding ──► Phoneme Resonator ──► Text │
//! │    │           │                │                   │              │    │
//! │    │     (Temporal        (Prosodic          (Modern          (CMU      │
//! │    │      Dynamics)        Encoding)         Hopfield)        Dict)     │
//! │    │                                                                    │
//! │    └────────────────────────────────────────────────────────────────────┘
//! │                                                                          │
//! │  Training:  LibriSpeech ──► Salience Aligner ──► Prototype Trainer      │
//! │                                                                          │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Quick Start
//!
//! ```ignore
//! use symthaea_stt::{
//!     AudioFrontend, AudioProjector,
//!     PhonemeDecoder, TrainedPrototypes,
//! };
//!
//! // Load pre-trained prototypes
//! let prototypes = TrainedPrototypes::load("models/phoneme_prototypes.bin")?;
//!
//! // Create audio projector
//! let mut projector = AudioProjector::default_config();
//!
//! // Load and project audio
//! let (audio, _sample_rate) = AudioFrontend::load_wav("speech.wav")?;
//! let hvs = projector.project(&audio);
//!
//! // Decode phonemes
//! let mut decoder = PhonemeDecoder::new();
//! decoder.load_prototypes(&prototypes.as_pairs());
//! ```

#![allow(missing_docs)]
#![allow(rustdoc::missing_crate_level_docs)]
#![allow(rustdoc::invalid_html_tags)]
#![allow(rustdoc::bare_urls)]

pub mod adaptation;
pub mod alignment_loader;
pub mod articulatory;
pub mod articulatory_cfc;
pub mod audio;
pub mod batch;
pub mod bootstrap;
pub mod cetacean_classifier;
pub mod cetacean_scorer;
pub mod crystal_reservoir;
pub mod discovery;
pub mod dtw_align;
pub mod eval;
pub mod hdc;
pub mod hierarchical_scorer;
pub mod holographic_scorer;
pub mod lexicon;
pub mod linguistic;
pub mod liquid;
pub mod liquid_hdc;
pub mod liquid_projection;
pub mod lm;
pub mod ltc;
#[cfg(feature = "gpu")]
pub mod ltc_gpu;
pub mod ltc_training;
pub mod models;

// Unified HDC-LTC architecture (reuses symthaea-core)
pub mod audio_hdc_encoder;
pub mod multiscale_scorer;
pub mod phoneme;
pub mod phoneme_hdc_ltc;
pub mod rls;
pub mod streaming;
pub mod temporal_grammar;
pub mod unified_grammar;
pub mod whale;

// Physiological signal processing (Project Hypnos / Consciousness Sensing)
pub mod edf_loader;
pub mod sleep_sentinel;

// Re-exports for convenience
pub use adaptation::{AdaptationEngine, SpeakerDiarizer, SpeakerProfile};
pub use alignment_loader::{
    PhonemeSegment, UtteranceAlignment, WordSegment, id_to_audio_path, load_alignments,
};
pub use articulatory::{
    AcousticArticulatoryDetector, ArticulatoryFeatures, ArticulatoryHDC, ArticulatoryMapper,
    ArticulatoryResonator, Manner, Place, Voicing, VowelHeight,
};
pub use audio::{AudioConfig, AudioFrontend, AudioProjector};
pub use batch::{BatchConfig, BatchDiscovery, BatchProcessor, BatchStats, BatchTrainer};
pub use bootstrap::{
    AdaptivePrototype, AdaptivePrototypeSet, AdaptiveStats, BootstrapConfig, BootstrapPipeline,
    TrainedPrototypes,
};
pub use crystal_reservoir::{
    CrystalActivation, CrystalReservoir, GaborFilter, OnlinePrototypeClassifier,
};
pub use discovery::{DiscoveryConfig, DiscoveryPipeline, DiscoveryResult, OnlineClusterer};
pub use dtw_align::{AlignedSegment, DtwAligner, DtwAlignment, DtwTrainer};
pub use edf_loader::{EdfFile, EdfHeader, EdfSignal, SleepStage};
pub use eval::{
    Alignment, ConfusionMatrix, EvalResult, EvaluationReport, character_error_rate,
    phoneme_error_rate, word_error_rate,
};
pub use hdc::{CORE_HDC_DIM, EXPANSION_FACTOR, HDC_DIM, HV16, bundle, weighted_bundle};
pub use lexicon::{CmuDictionary, TextToPhonemes};
pub use linguistic::{PhonemeClasses, PhonemeFeatures, PhonotacticConstraints};
pub use liquid_projection::{
    DirectAccumulator, DirectClassifier, DirectClassifierConfig, LiquidProjection,
    LiquidProjectionConfig, PhonemeTargets, RFActivation, RandomProjection, RidgeAccumulator,
};
pub use lm::{BeamConfig, BeamDecoder, CombinedDecoder, Hypothesis, NgramLM, PhonemeToWordDecoder};
pub use ltc::{LtcCell, LtcConfig, TauSmoother};
pub use models::{ModelInfo, ModelMetadata, ModelPackage, ModelRegistry};
pub use phoneme::{
    PhonemeDecoder, PhonemeInventory, PhonemeResonator, TemporalConfig, TemporalDecoder,
};
pub use rls::{FastRlsClassifier, RlsClassifier};
pub use sleep_sentinel::{BandPowers, ConsciousnessSentinel, ConsciousnessState};
pub use streaming::{StreamConfig, StreamProcessor, StreamSession};

/// Crate version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Default sample rate (Hz)
pub const SAMPLE_RATE: u32 = 16000;

/// Default frame duration (seconds)
pub const FRAME_DURATION: f32 = 0.010;

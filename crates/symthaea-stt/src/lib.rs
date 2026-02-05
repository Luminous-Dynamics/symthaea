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

// TODO: Add comprehensive documentation before 1.0 release
#![allow(missing_docs)]
#![allow(rustdoc::missing_crate_level_docs)]

pub mod hdc;
pub mod ltc;
pub mod phoneme;
pub mod lexicon;
pub mod bootstrap;
pub mod audio;
pub mod discovery;
pub mod adaptation;
pub mod streaming;
pub mod batch;
pub mod eval;
pub mod lm;
pub mod models;
pub mod whale;
pub mod linguistic;
pub mod articulatory;
pub mod articulatory_cfc;
pub mod holographic_scorer;
pub mod cetacean_scorer;
pub mod cetacean_classifier;
pub mod hierarchical_scorer;
pub mod multiscale_scorer;
pub mod liquid;
pub mod liquid_hdc;
pub mod liquid_projection;
pub mod crystal_reservoir;
pub mod rls;
pub mod temporal_grammar;
pub mod unified_grammar;
pub mod dtw_align;
pub mod alignment_loader;

// Physiological signal processing (Project Hypnos / Consciousness Sensing)
pub mod edf_loader;
pub mod sleep_sentinel;

// Re-exports for convenience
pub use hdc::{HV16, bundle, weighted_bundle, HDC_DIM, CORE_HDC_DIM, EXPANSION_FACTOR};
pub use ltc::{LtcCell, LtcConfig, TauSmoother};
pub use phoneme::{PhonemeDecoder, PhonemeResonator, PhonemeInventory, TemporalDecoder, TemporalConfig};
pub use lexicon::{CmuDictionary, TextToPhonemes};
pub use bootstrap::{
    BootstrapPipeline, BootstrapConfig, TrainedPrototypes,
    AdaptivePrototype, AdaptivePrototypeSet, AdaptiveStats,
};
pub use audio::{AudioFrontend, AudioProjector, AudioConfig};
pub use linguistic::{PhonemeClasses, PhonotacticConstraints, PhonemeFeatures};
pub use articulatory::{
    ArticulatoryMapper, ArticulatoryFeatures, ArticulatoryHDC, ArticulatoryResonator,
    AcousticArticulatoryDetector, Voicing, Manner, Place, VowelHeight,
};
pub use discovery::{DiscoveryPipeline, DiscoveryConfig, DiscoveryResult, OnlineClusterer};
pub use adaptation::{SpeakerProfile, AdaptationEngine, SpeakerDiarizer};
pub use streaming::{StreamProcessor, StreamSession, StreamConfig};
pub use batch::{BatchProcessor, BatchTrainer, BatchDiscovery, BatchConfig, BatchStats};
pub use eval::{
    phoneme_error_rate, word_error_rate, character_error_rate,
    EvalResult, Alignment, ConfusionMatrix, EvaluationReport,
};
pub use lm::{
    NgramLM, BeamDecoder, BeamConfig, Hypothesis,
    PhonemeToWordDecoder, CombinedDecoder,
};
pub use models::{ModelPackage, ModelMetadata, ModelRegistry, ModelInfo};
pub use edf_loader::{EdfFile, EdfSignal, EdfHeader, SleepStage};
pub use sleep_sentinel::{ConsciousnessSentinel, ConsciousnessState, BandPowers};
pub use dtw_align::{DtwAligner, DtwTrainer, DtwAlignment, AlignedSegment};
pub use alignment_loader::{load_alignments, PhonemeSegment, UtteranceAlignment, WordSegment, id_to_audio_path};
pub use liquid_projection::{
    LiquidProjection, LiquidProjectionConfig, PhonemeTargets, RidgeAccumulator,
    DirectClassifier, DirectClassifierConfig, DirectAccumulator, RandomProjection, RFActivation,
};
pub use crystal_reservoir::{
    GaborFilter, CrystalReservoir, CrystalActivation, OnlinePrototypeClassifier,
};
pub use rls::{RlsClassifier, FastRlsClassifier};

/// Crate version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Default sample rate (Hz)
pub const SAMPLE_RATE: u32 = 16000;

/// Default frame duration (seconds)
pub const FRAME_DURATION: f32 = 0.010;

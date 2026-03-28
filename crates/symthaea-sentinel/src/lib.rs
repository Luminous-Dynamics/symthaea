//! Symthaea Sentinel: Zero-shot temporal audio pattern recognition
//!
//! Built on HDC + LTC + CfC for environmental sound classification.
//!
//! "Transformers spatialize time. Symthaea LIVES in time."
//!
//! ## Architecture
//!
//! - **HDC (Hyperdimensional Computing)**: High-capacity vector representations
//! - **LTC (Liquid Time-Constant)**: Multi-timescale temporal dynamics
//! - **CfC (Closed-form Continuous-time)**: Stable dynamics with closed-form solution
//!
//! ## Modules
//!
//! - `hdc`: Hypervector operations (HV, SparseProjector, RffProjector)
//! - `temporal`: Temporal dynamics (CfcCell, HierarchicalCfc, HierarchicalLtc)
//! - `features`: Audio feature extraction (MelFilterbank, FeatureExtractor)
//! - `encoding`: HDC encoding (AudioHdcEncoder, PremiumHdcEncoder)
//! - `patterns`: Pattern definitions (AudioPattern, AudioCategory)
//! - `sentinel`: Main recognition engine (AudioSentinel)
//! - `io`: Audio I/O (AudioPump, FileAudioPump)
//!
//! ## Quick Start
//!
//! ```no_run
//! use symthaea_sentinel::{AudioSentinel, AudioCategory};
//!
//! // Create a sentinel in Premium mode (CfC + RFF)
//! let mut sentinel = AudioSentinel::premium();
//!
//! // Learn a pattern
//! sentinel.start_learning("clock_tick", AudioCategory::Mechanical);
//! // ... process audio frames ...
//! sentinel.stop_learning();
//!
//! // Recognize patterns
//! // let result = sentinel.process(&features);
//! // println!("Detected: {} ({:.1}%)", result.detected_pattern, result.confidence * 100.0);
//! ```

#![deny(unsafe_code)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::manual_clamp)]
#![allow(clippy::manual_memcpy)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::wrong_self_convention)]
#![allow(clippy::redundant_guards)]
#![allow(clippy::duplicated_attributes)]

pub mod encoding;
pub mod features;
pub mod hdc;
pub mod io;
pub mod patterns;
pub mod sentinel;
pub mod temporal;

// Re-exports for convenient access
pub use encoding::{AudioHdcEncoder, AudioHdcVectors, EncoderMode, PremiumHdcEncoder};
pub use features::{
    compute_mfcc, compute_mfcc_delta, compute_power_spectrum, compute_temporal_regularity,
    AudioFeatures, FeatureExtractor, MelFilterbank, CONTROL_RATE, FFT_SIZE, FREQ_BINS, HOP_SIZE,
    MEL_BANDS, NUM_MFCC, SAMPLE_RATE,
};
pub use hdc::{RffProjector, SparseProjector, HDC_DIM, HV};
pub use io::{
    compute_burst_density, compute_ioi_variance, compute_onset_strength, compute_spectral_centroid,
    compute_spectral_flatness, compute_temporal_regularity as io_compute_temporal_regularity,
    spectrum_to_mel_bands, AudioConfig, DatasetProcessor, FileAudioConfig, FileAudioPump,
    FileProcessingResult,
};
pub use patterns::{
    AmbientContexts, AudioCategory, AudioPattern, PatternSimilarity, MAX_EXEMPLARS, NUM_LTC_LEVELS,
};
pub use sentinel::{AudioDetectionResult, AudioSentinel};
pub use temporal::{CfcCell, HierarchicalCfc, HierarchicalLtc, LtcPreset, TemporalWindow};

#[cfg(feature = "live-audio")]
pub use io::AudioPump;

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

pub mod hdc;
pub mod temporal;
pub mod features;
pub mod encoding;
pub mod patterns;
pub mod sentinel;
pub mod io;

// Re-exports for convenient access
pub use hdc::{HV, SparseProjector, RffProjector, HDC_DIM};
pub use temporal::{CfcCell, HierarchicalCfc, HierarchicalLtc, LtcPreset, TemporalWindow};
pub use features::{
    AudioFeatures, FeatureExtractor, MelFilterbank,
    MEL_BANDS, NUM_MFCC, FFT_SIZE, HOP_SIZE, SAMPLE_RATE, CONTROL_RATE, FREQ_BINS,
    compute_mfcc, compute_mfcc_delta, compute_temporal_regularity, compute_power_spectrum,
};
pub use encoding::{AudioHdcEncoder, PremiumHdcEncoder, AudioHdcVectors, EncoderMode};
pub use patterns::{AudioPattern, AudioCategory, PatternSimilarity, AmbientContexts, NUM_LTC_LEVELS, MAX_EXEMPLARS};
pub use sentinel::{AudioSentinel, AudioDetectionResult};
pub use io::{
    AudioConfig, FileAudioConfig, FileAudioPump, DatasetProcessor, FileProcessingResult,
    compute_onset_strength, compute_spectral_centroid, compute_spectral_flatness,
    compute_temporal_regularity as io_compute_temporal_regularity,
    spectrum_to_mel_bands, compute_burst_density, compute_ioi_variance,
};

#[cfg(feature = "live-audio")]
pub use io::AudioPump;

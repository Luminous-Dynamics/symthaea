// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
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
    AudioFeatures, CONTROL_RATE, FFT_SIZE, FREQ_BINS, FeatureExtractor, HOP_SIZE, MEL_BANDS,
    MelFilterbank, NUM_MFCC, SAMPLE_RATE, compute_mfcc, compute_mfcc_delta, compute_power_spectrum,
    compute_temporal_regularity,
};
pub use hdc::{HDC_DIM, HV, RffProjector, SparseProjector};
pub use io::{
    AudioConfig, DatasetProcessor, FileAudioConfig, FileAudioPump, FileProcessingResult,
    compute_burst_density, compute_ioi_variance, compute_onset_strength, compute_spectral_centroid,
    compute_spectral_flatness, compute_temporal_regularity as io_compute_temporal_regularity,
    spectrum_to_mel_bands,
};
pub use patterns::{
    AmbientContexts, AudioCategory, AudioPattern, MAX_EXEMPLARS, NUM_LTC_LEVELS, PatternSimilarity,
};
pub use sentinel::{AudioDetectionResult, AudioSentinel};
pub use temporal::{CfcCell, HierarchicalCfc, HierarchicalLtc, LtcPreset, TemporalWindow};

#[cfg(feature = "live-audio")]
pub use io::AudioPump;

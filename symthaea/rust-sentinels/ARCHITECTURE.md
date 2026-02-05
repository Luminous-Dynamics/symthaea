# Rust Sentinels Library Architecture

## Overview

This document outlines the production Rust implementation of the Consciousness Trilogy:
- **EmotionSentinel** - Proof of Joy
- **SleepSentinel** - Proof of Rest
- **MeditationSentinel** - Proof of Focus
- **ConsciousnessEngine** - Unified PoC scoring

## Project Structure

```
rust-sentinels/
├── Cargo.toml
├── src/
│   ├── lib.rs                 # Main library entry
│   ├── signal/
│   │   ├── mod.rs
│   │   ├── fft.rs             # FFT and spectral analysis
│   │   ├── filter.rs          # Bandpass filters
│   │   └── welch.rs           # Welch PSD estimation
│   ├── features/
│   │   ├── mod.rs
│   │   ├── band_power.rs      # Frequency band extraction
│   │   ├── asymmetry.rs       # Frontal asymmetry
│   │   └── ratios.rs          # Power ratios
│   ├── sentinels/
│   │   ├── mod.rs
│   │   ├── emotion.rs         # EmotionSentinel
│   │   ├── sleep.rs           # SleepSentinel
│   │   └── meditation.rs      # MeditationSentinel
│   ├── engine.rs              # ConsciousnessEngine
│   ├── types.rs               # Core types and structs
│   └── error.rs               # Error handling
├── benches/
│   └── performance.rs         # Benchmarks
└── examples/
    ├── basic_analysis.rs
    └── realtime_streaming.rs
```

## Core Types

```rust
// types.rs

use serde::{Deserialize, Serialize};

/// Consciousness state enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConsciousnessState {
    DeepSleep,
    LightSleep,
    Rem,
    Drowsy,
    Relaxed,
    Focused,
    Meditative,
    Flow,
    Stressed,
}

/// Emotion analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionScore {
    pub valence: f32,      // -1 to +1
    pub arousal: f32,      // 0 to 1
    pub confidence: f32,   // 0 to 1
}

impl EmotionScore {
    pub fn quadrant(&self) -> &'static str {
        match (self.valence >= 0.0, self.arousal >= 0.5) {
            (true, true) => "excited_positive",
            (true, false) => "calm_positive",
            (false, true) => "excited_negative",
            (false, false) => "calm_negative",
        }
    }
}

/// Sleep stage result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SleepScore {
    pub stage: SleepStage,
    pub confidence: f32,
    pub delta_power: f32,
    pub sleep_quality: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SleepStage {
    Wake = 0,
    N1 = 1,
    N2 = 2,
    N3 = 3,
    Rem = 4,
}

/// Meditation analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeditationScore {
    pub depth: f32,            // 0 to 1
    pub stability: f32,        // 0 to 1
    pub alpha_power: f32,
    pub theta_power: f32,
    pub meditation_index: f32,
}

/// Combined Proof of Consciousness
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofOfConsciousness {
    pub timestamp: f64,
    pub state: ConsciousnessState,
    pub consciousness_level: f32,  // 0 to 1
    pub wellbeing_score: f32,      // 0 to 1
    pub emotion: Option<EmotionScore>,
    pub sleep: Option<SleepScore>,
    pub meditation: Option<MeditationScore>,
}
```

## Signal Processing Module

```rust
// signal/welch.rs

use ndarray::{Array1, ArrayView1};
use rustfft::{FftPlanner, num_complex::Complex};

/// Welch power spectral density estimation
pub fn welch_psd(
    signal: ArrayView1<f32>,
    sample_rate: f32,
    nperseg: usize,
    noverlap: Option<usize>,
) -> (Array1<f32>, Array1<f32>) {
    let noverlap = noverlap.unwrap_or(nperseg / 2);
    let step = nperseg - noverlap;

    // Hanning window
    let window: Array1<f32> = Array1::from_iter(
        (0..nperseg).map(|i| {
            0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / nperseg as f32).cos())
        })
    );

    // Segment and average
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(nperseg);

    let mut psd = Array1::zeros(nperseg / 2 + 1);
    let mut n_segments = 0;

    let mut pos = 0;
    while pos + nperseg <= signal.len() {
        let segment = signal.slice(s![pos..pos + nperseg]);
        let windowed: Vec<Complex<f32>> = segment
            .iter()
            .zip(window.iter())
            .map(|(&s, &w)| Complex::new(s * w, 0.0))
            .collect();

        let mut buffer = windowed;
        fft.process(&mut buffer);

        for (i, psd_val) in psd.iter_mut().enumerate() {
            *psd_val += buffer[i].norm_sqr();
        }

        n_segments += 1;
        pos += step;
    }

    // Normalize
    let scale = 1.0 / (sample_rate * nperseg as f32 * n_segments as f32);
    psd *= scale;

    // Frequency bins
    let frequencies = Array1::from_iter(
        (0..=nperseg / 2).map(|i| i as f32 * sample_rate / nperseg as f32)
    );

    (frequencies, psd)
}

/// Compute band power from PSD
pub fn band_power(
    frequencies: ArrayView1<f32>,
    psd: ArrayView1<f32>,
    fmin: f32,
    fmax: f32,
) -> f32 {
    let mut power = 0.0;
    let mut prev_f = fmin;

    for (i, (&f, &p)) in frequencies.iter().zip(psd.iter()).enumerate() {
        if f >= fmin && f <= fmax {
            if i > 0 {
                let df = f - prev_f;
                power += p * df;
            }
            prev_f = f;
        }
    }

    power
}
```

## Sentinel Implementations

```rust
// sentinels/sleep.rs

use crate::signal::welch::{welch_psd, band_power};
use crate::types::{SleepScore, SleepStage};
use ndarray::ArrayView1;

pub struct SleepSentinel {
    // Thresholds derived from validation
    delta_threshold: f32,
    theta_threshold: f32,
    beta_threshold: f32,
}

impl SleepSentinel {
    pub fn new() -> Self {
        Self {
            delta_threshold: 0.88,
            theta_threshold: 0.16,
            beta_threshold: 0.028,
        }
    }

    pub fn analyze(&self, data: ArrayView1<f32>, sample_rate: f32) -> SleepScore {
        let nperseg = (4.0 * sample_rate).min(data.len() as f32) as usize;
        let (freqs, psd) = welch_psd(data, sample_rate, nperseg, None);

        // Extract band powers
        let delta = band_power(freqs.view(), psd.view(), 0.5, 4.0);
        let theta = band_power(freqs.view(), psd.view(), 4.0, 8.0);
        let alpha = band_power(freqs.view(), psd.view(), 8.0, 13.0);
        let beta = band_power(freqs.view(), psd.view(), 13.0, 30.0);

        let total = delta + theta + alpha + beta + 1e-10;
        let delta_rel = delta / total;
        let theta_rel = theta / total;
        let beta_rel = beta / total;

        // Classification
        let (stage, confidence) = if delta_rel > self.delta_threshold && theta_rel < 0.08 {
            (SleepStage::N3, delta_rel.min(1.0))
        } else if theta_rel > self.theta_threshold {
            if beta_rel > 0.045 {
                (SleepStage::N1, (theta_rel * 2.0).min(1.0))
            } else {
                (SleepStage::Rem, (theta_rel * 2.0).min(1.0))
            }
        } else if beta_rel > self.beta_threshold {
            (SleepStage::Wake, (beta_rel * 10.0).min(1.0))
        } else {
            (SleepStage::N2, 0.6)
        };

        let sleep_quality = if stage == SleepStage::N3 { delta_rel } else { 0.5 };

        SleepScore {
            stage,
            confidence,
            delta_power: delta_rel,
            sleep_quality,
        }
    }
}
```

## Performance Targets

| Operation | Target | Notes |
|-----------|--------|-------|
| Single epoch analysis | < 1ms | 30-second epoch |
| Streaming (256 Hz) | < 4ms/sample | Real-time capable |
| Memory per channel | < 1MB | 30-second buffer |
| Full PoC score | < 5ms | All three Sentinels |

## API Design

```rust
// lib.rs - Public API

/// Analyze consciousness state from EEG data
///
/// # Arguments
/// * `data` - EEG samples (single channel or multi-channel)
/// * `sample_rate` - Sampling frequency in Hz
/// * `mode` - Analysis mode (Auto, Emotion, Sleep, Meditation)
///
/// # Example
/// ```
/// use sentinels::{analyze_consciousness, AnalysisMode};
///
/// let data = vec![0.0f32; 256 * 30]; // 30 seconds at 256 Hz
/// let poc = analyze_consciousness(&data, 256.0, AnalysisMode::Auto)?;
///
/// println!("State: {:?}", poc.state);
/// println!("Consciousness: {:.2}", poc.consciousness_level);
/// ```
pub fn analyze_consciousness(
    data: &[f32],
    sample_rate: f32,
    mode: AnalysisMode,
) -> Result<ProofOfConsciousness, SentinelError>;

/// Streaming analysis for real-time applications
pub struct StreamingEngine {
    // Ring buffer and state
}

impl StreamingEngine {
    pub fn new(sample_rate: f32, epoch_duration: f32) -> Self;
    pub fn push_samples(&mut self, samples: &[f32]);
    pub fn analyze(&mut self) -> Option<ProofOfConsciousness>;
}
```

## FFI for Integration

```rust
// ffi.rs - C-compatible interface

#[repr(C)]
pub struct CProofOfConsciousness {
    pub state: i32,
    pub consciousness_level: f32,
    pub wellbeing_score: f32,
    pub valence: f32,
    pub arousal: f32,
    pub sleep_stage: i32,
    pub meditation_depth: f32,
}

#[no_mangle]
pub extern "C" fn sentinels_analyze(
    data: *const f32,
    len: usize,
    sample_rate: f32,
    result: *mut CProofOfConsciousness,
) -> i32;

#[no_mangle]
pub extern "C" fn sentinels_free(ptr: *mut CProofOfConsciousness);
```

## Build Configuration

```toml
# Cargo.toml

[package]
name = "sentinels"
version = "0.1.0"
edition = "2021"

[dependencies]
ndarray = "0.15"
rustfft = "6.1"
serde = { version = "1.0", features = ["derive"] }
thiserror = "1.0"

[dev-dependencies]
criterion = "0.5"

[features]
default = ["std"]
std = []
ffi = []
wasm = ["wasm-bindgen"]

[lib]
crate-type = ["cdylib", "rlib"]

[[bench]]
name = "performance"
harness = false
```

## Next Steps

1. **Core Implementation**: Port Python algorithms to Rust
2. **Optimization**: SIMD for FFT, cache-friendly memory layout
3. **Testing**: Property-based tests, fuzzing
4. **Benchmarking**: Compare with Python baseline
5. **Documentation**: Full rustdoc coverage
6. **Integration**: Python bindings via PyO3

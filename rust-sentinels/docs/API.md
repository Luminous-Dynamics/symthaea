# 📚 Rust Sentinels API Reference

Complete API documentation for the Rust Sentinels consciousness detection library.

## Table of Contents

1. [Core Functions](#core-functions)
2. [Types](#types)
3. [Sentinels](#sentinels)
4. [Signal Processing](#signal-processing)
5. [Error Handling](#error-handling)

---

## Core Functions

### `analyze_consciousness`

Main entry point for consciousness analysis.

```rust
pub fn analyze_consciousness(
    data: &[f32],
    sample_rate: f32,
    mode: AnalysisMode,
) -> Result<ProofOfConsciousness, SentinelError>
```

**Parameters:**
- `data` - Raw EEG signal as f32 samples
- `sample_rate` - Sample rate in Hz (typically 256.0)
- `mode` - Analysis mode (Auto, Emotion, Sleep, Meditation)

**Returns:**
- `Ok(ProofOfConsciousness)` - Complete consciousness analysis
- `Err(SentinelError)` - Error if analysis fails

**Example:**
```rust
use sentinels::{analyze_consciousness, AnalysisMode};

let data: Vec<f32> = load_eeg_data();
let result = analyze_consciousness(&data, 256.0, AnalysisMode::Auto)?;
println!("State: {:?}", result.state);
```

**Requirements:**
- Minimum 512 samples (2 seconds at 256 Hz)
- Sample rate must be positive
- Data must contain valid float values

---

## Types

### `ProofOfConsciousness`

Complete consciousness state at a moment in time.

```rust
pub struct ProofOfConsciousness {
    pub state: ConsciousnessState,
    pub consciousness_level: f32,  // 0.0 - 1.0
    pub wellbeing_score: f32,      // 0.0 - 1.0
    pub emotion: Option<EmotionScore>,
    pub sleep: Option<SleepScore>,
    pub meditation: Option<MeditationScore>,
}
```

**Methods:**

```rust
impl ProofOfConsciousness {
    /// Serialize to JSON string
    pub fn to_json(&self) -> String;

    /// Create from JSON string
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error>;
}
```

### `ConsciousnessState`

Enumeration of detected consciousness states.

```rust
pub enum ConsciousnessState {
    DeepSleep,    // N3 slow-wave sleep
    LightSleep,   // N1/N2 sleep
    REM,          // REM sleep
    Drowsy,       // Pre-sleep transition
    Relaxed,      // Calm wakefulness
    Alert,        // Normal wakefulness
    Focused,      // Concentrated attention
    Meditative,   // Contemplative state
    Flow,         // Optimal performance
    Stressed,     // High arousal state
}
```

### `AnalysisMode`

Specifies which sentinels to run.

```rust
pub enum AnalysisMode {
    Auto,        // Run all applicable sentinels
    Emotion,     // EmotionSentinel only
    Sleep,       // SleepSentinel only
    Meditation,  // MeditationSentinel only
    Attention,   // AttentionSentinel (future)
    Flow,        // FlowSentinel (future)
    Engagement,  // EngagementSentinel (future)
}
```

### `EmotionScore`

Emotional state analysis result.

```rust
pub struct EmotionScore {
    pub valence: f32,  // -1.0 (negative) to +1.0 (positive)
    pub arousal: f32,  // 0.0 (calm) to 1.0 (excited)
}
```

**Methods:**

```rust
impl EmotionScore {
    /// Get emotion quadrant
    pub fn quadrant(&self) -> EmotionQuadrant;
}
```

### `EmotionQuadrant`

Russell's circumplex model quadrants.

```rust
pub enum EmotionQuadrant {
    HappyCalm,     // Positive valence, low arousal
    HappyExcited,  // Positive valence, high arousal
    SadCalm,       // Negative valence, low arousal
    SadExcited,    // Negative valence, high arousal (anxious)
}
```

### `SleepScore`

Sleep stage analysis result.

```rust
pub struct SleepScore {
    pub stage: SleepStage,
    pub delta_power: f32,  // 0.0 - 1.0
    pub theta_power: f32,  // 0.0 - 1.0
    pub confidence: f32,   // 0.0 - 1.0
}
```

### `SleepStage`

AASM sleep stages.

```rust
pub enum SleepStage {
    Wake,  // Awake
    N1,    // Stage 1 (light sleep)
    N2,    // Stage 2 (sleep spindles)
    N3,    // Stage 3 (slow-wave/deep sleep)
    REM,   // Rapid eye movement
}
```

### `MeditationScore`

Meditation state analysis result.

```rust
pub struct MeditationScore {
    pub depth: f32,            // 0.0 - 1.0
    pub stability: f32,        // 0.0 - 1.0
    pub meditation_index: f32, // typically 0.0 - 10.0
}
```

**Methods:**

```rust
impl MeditationScore {
    /// Get meditation state classification
    pub fn state(&self) -> MeditationState;
}
```

### `MeditationState`

Meditation depth classification.

```rust
pub enum MeditationState {
    Light,     // depth < 0.3
    Moderate,  // 0.3 <= depth < 0.6
    Deep,      // depth >= 0.6
}
```

---

## Sentinels

### `EmotionSentinel`

Detects emotional state from EEG signals.

```rust
pub struct EmotionSentinel {
    // Internal state
}

impl EmotionSentinel {
    /// Create new sentinel with default parameters
    pub fn new() -> Self;

    /// Analyze signal and return emotion score
    pub fn analyze(&self, data: &[f32], sample_rate: f32) -> EmotionScore;
}
```

**Algorithm:**
1. Extract alpha band power (8-13 Hz) from frontal channels
2. Calculate frontal asymmetry index
3. Extract beta band power for arousal
4. Map to valence-arousal space

**Example:**
```rust
let sentinel = EmotionSentinel::new();
let score = sentinel.analyze(&data, 256.0);

if score.valence > 0.3 && score.arousal < 0.5 {
    println!("User appears calm and positive");
}
```

### `SleepSentinel`

Classifies sleep stages from EEG signals.

```rust
pub struct SleepSentinel {
    // Internal state
}

impl SleepSentinel {
    /// Create new sentinel with default thresholds
    pub fn new() -> Self;

    /// Create with custom thresholds
    pub fn with_thresholds(thresholds: SleepThresholds) -> Self;

    /// Analyze signal and return sleep score
    pub fn analyze(&self, data: &[f32], sample_rate: f32) -> SleepScore;
}
```

**Thresholds (default):**
```rust
pub struct SleepThresholds {
    pub n3_delta_threshold: f32,    // 0.88
    pub n1_theta_threshold: f32,    // 0.16
    pub wake_beta_threshold: f32,   // 0.028
}
```

**Algorithm:**
1. Extract relative band powers (delta, theta, alpha, beta)
2. Apply decision tree based on thresholds
3. Calculate confidence from power ratios

**Example:**
```rust
let sentinel = SleepSentinel::new();
let score = sentinel.analyze(&data, 256.0);

match score.stage {
    SleepStage::N3 => println!("Deep sleep detected"),
    SleepStage::REM => println!("Dream sleep detected"),
    _ => {}
}
```

### `MeditationSentinel`

Measures meditation depth and quality.

```rust
pub struct MeditationSentinel {
    baseline_alpha: f32,
    baseline_theta: f32,
}

impl MeditationSentinel {
    /// Create without baseline (uses default)
    pub fn new() -> Self;

    /// Create with calibrated baseline
    pub fn with_baseline(
        baseline_data: &[f32],
        sample_rate: f32,
    ) -> Self;

    /// Analyze signal and return meditation score
    pub fn analyze(&self, data: &[f32], sample_rate: f32) -> MeditationScore;

    /// Calibrate baseline from eyes-open rest data
    pub fn calibrate(&mut self, data: &[f32], sample_rate: f32);
}
```

**Algorithm:**
1. Extract alpha and theta band powers
2. Calculate meditation index: (alpha + 0.5*theta) / (beta + 0.1)
3. Compare to baseline for relative depth
4. Assess stability from power variance

**Example:**
```rust
// Calibrate with baseline recording
let mut sentinel = MeditationSentinel::new();
sentinel.calibrate(&baseline_data, 256.0);

// Analyze meditation session
let score = sentinel.analyze(&meditation_data, 256.0);
println!("Meditation depth: {:.2}", score.depth);
println!("State: {:?}", score.state());
```

---

## Signal Processing

### Module: `sentinels::signal`

#### `welch_psd`

Compute power spectral density using Welch's method.

```rust
pub fn welch_psd(
    signal: &[f32],
    sample_rate: f32,
    nperseg: usize,
) -> (Vec<f32>, Vec<f32>)
```

**Parameters:**
- `signal` - Input signal
- `sample_rate` - Sample rate in Hz
- `nperseg` - Segment length for FFT (power of 2 recommended)

**Returns:**
- Tuple of (frequencies, power spectral density)

**Example:**
```rust
use sentinels::signal::welch_psd;

let (freqs, psd) = welch_psd(&data, 256.0, 256);

// Find peak frequency
let (peak_freq, peak_power) = freqs.iter()
    .zip(psd.iter())
    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
    .unwrap();

println!("Peak frequency: {:.1} Hz", peak_freq);
```

#### `band_power`

Calculate power in a specific frequency band.

```rust
pub fn band_power(
    frequencies: &[f32],
    psd: &[f32],
    fmin: f32,
    fmax: f32,
) -> f32
```

**Parameters:**
- `frequencies` - Frequency array from welch_psd
- `psd` - PSD array from welch_psd
- `fmin` - Lower frequency bound (Hz)
- `fmax` - Upper frequency bound (Hz)

**Returns:**
- Total power in the band (absolute)

**Standard EEG Bands:**
```rust
let delta = band_power(&freqs, &psd, 0.5, 4.0);   // Delta
let theta = band_power(&freqs, &psd, 4.0, 8.0);   // Theta
let alpha = band_power(&freqs, &psd, 8.0, 13.0);  // Alpha
let beta = band_power(&freqs, &psd, 13.0, 30.0);  // Beta
let gamma = band_power(&freqs, &psd, 30.0, 100.0); // Gamma

// Calculate relative powers
let total = delta + theta + alpha + beta + gamma;
let alpha_relative = alpha / total;
```

#### `extract_band_powers`

Extract all standard EEG band powers at once.

```rust
pub fn extract_band_powers(
    data: &[f32],
    sample_rate: f32,
) -> BandPowers
```

**Returns:**
```rust
pub struct BandPowers {
    pub delta: f32,
    pub theta: f32,
    pub alpha: f32,
    pub beta: f32,
    pub gamma: f32,
    pub total: f32,
}
```

---

## Error Handling

### `SentinelError`

Error type for sentinel operations.

```rust
pub enum SentinelError {
    /// Not enough samples for analysis
    InsufficientData {
        required: usize,
        provided: usize,
    },

    /// Invalid sample rate
    InvalidSampleRate(f32),

    /// Signal processing error
    ProcessingError(String),

    /// Invalid parameters
    InvalidParameters(String),
}
```

**Implementing Display:**
```rust
impl std::fmt::Display for SentinelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InsufficientData { required, provided } =>
                write!(f, "Need {} samples, got {}", required, provided),
            Self::InvalidSampleRate(rate) =>
                write!(f, "Invalid sample rate: {}", rate),
            Self::ProcessingError(msg) =>
                write!(f, "Processing error: {}", msg),
            Self::InvalidParameters(msg) =>
                write!(f, "Invalid parameters: {}", msg),
        }
    }
}
```

**Example:**
```rust
match analyze_consciousness(&data, 256.0, AnalysisMode::Auto) {
    Ok(poc) => process_result(poc),
    Err(SentinelError::InsufficientData { required, provided }) => {
        eprintln!("Need more data: {} samples required, {} provided",
                  required, provided);
    }
    Err(e) => eprintln!("Error: {}", e),
}
```

---

## Constants

### Frequency Bands

```rust
pub const DELTA_BAND: (f32, f32) = (0.5, 4.0);
pub const THETA_BAND: (f32, f32) = (4.0, 8.0);
pub const ALPHA_BAND: (f32, f32) = (8.0, 13.0);
pub const BETA_BAND: (f32, f32) = (13.0, 30.0);
pub const GAMMA_BAND: (f32, f32) = (30.0, 100.0);
```

### Analysis Requirements

```rust
pub const MIN_SAMPLES: usize = 512;          // 2 seconds at 256 Hz
pub const RECOMMENDED_DURATION: f32 = 5.0;   // seconds
pub const DEFAULT_SAMPLE_RATE: f32 = 256.0;  // Hz
```

### Sleep Stage Thresholds

```rust
pub const N3_DELTA_THRESHOLD: f32 = 0.88;
pub const N1_THETA_THRESHOLD: f32 = 0.16;
pub const WAKE_BETA_THRESHOLD: f32 = 0.028;
pub const REM_BETA_MAX: f32 = 0.045;
```

---

## Thread Safety

All types implement `Send` and `Sync`:

```rust
use std::sync::Arc;
use std::thread;

let sentinel = Arc::new(EmotionSentinel::new());

let handles: Vec<_> = (0..4).map(|_| {
    let s = Arc::clone(&sentinel);
    thread::spawn(move || {
        s.analyze(&data, 256.0)
    })
}).collect();

for handle in handles {
    let score = handle.join().unwrap();
    // Process score
}
```

---

## Feature Flags

```toml
[features]
default = []
serde = ["serde/derive"]  # Enable JSON serialization
simd = []                  # SIMD optimizations (future)
python = ["pyo3"]          # Python bindings (future)
wasm = ["wasm-bindgen"]    # WebAssembly (future)
```

---

## Version History

| Version | Changes |
|---------|---------|
| 0.1.0 | Initial release with Consciousness Trilogy |
| 0.2.0 | Python bindings (planned) |
| 0.3.0 | Extended Proofs: Attention, Flow, Engagement (planned) |
| 0.4.0 | WASM support (planned) |

---

*For more examples, see the `examples/` directory.*

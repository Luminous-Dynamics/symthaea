# Neuroscience Validation Suite

**Module Location**: Various (see below)
**Purpose**: Validate consciousness topology theory against real biological data

---

## Overview

Symthaea includes comprehensive neuroscience validation to ensure our computational consciousness models align with real biological neural systems. This validation spans three domains:

| Domain | Data Source | Neurons/Subjects | Key Validation |
|--------|-------------|------------------|----------------|
| **Connectome** | C. elegans | 302 neurons | Structural Φ |
| **Sleep EEG** | Sleep-EDF | 2+ subjects | State-dependent Φ |
| **Meditation EEG** | OpenNeuro | 22+ subjects | Superconscious states |

---

## Validation Domains

### 1. C. elegans Connectome Validation

**Documentation**: [CELEGANS_VALIDATION.md](./CELEGANS_VALIDATION.md)
**Module**: `symthaea-core/src/hdc/celegans_connectome.rs`
**Lines**: 930

The C. elegans nematode has the only fully mapped connectome of any organism (302 neurons, ~7,000 synapses). This provides ground truth for testing whether our Φ calculations produce biologically meaningful results.

**Key Tests**:
- Biological Φ > Random Φ (validates evolution optimizes for integration)
- Processing core Φ ≈ Full Φ (motor system is downstream)
- Small-world topology comparison

### 2. Clinical Sleep Validation (Project Hypnos)

**Example**: `examples/clinical_validation.rs`
**Lines**: 1,533
**Data**: `data/sleep-edf/` (Sleep-EDF from PhysioNet)

Real-time sleep stage classification from EEG validates that LTC dynamics resonate with biological consciousness states.

**Architecture**:
```
EEG/EOG/EMG → LTC Processors → Sleep Stage Classification
                  ↓
           Permutation Entropy (differentiation)
           Theta/Alpha Ratio (spectral texture)
           Eye Movement Detection (REM)
           Muscle Tone Tracking (atonia)
```

**Sleep Stages Detected**:
- Wake (high entropy, high tone)
- N1 (transitional, dropping tone)
- N2 (sleep spindles, K-complexes)
- N3 (deep sleep, low entropy)
- REM (high entropy, atonia, eye movements)

### 3. Meditation & Flow State Validation

**Module**: `src/perception/physio/meditation_detector.rs`
**Data**: `data/meditation-eeg/` (22+ subjects, BIDS format)

Detects meditation and flow states from multi-channel EEG, validating higher consciousness state detection.

**States Detected**:
- Flow (gamma sync, high focus)
- Absorption (deep theta, high presence)
- Calm (alpha dominance)
- Wandering (default mode network)

---

## Data Sources

### Sleep-EDF Database
- **Source**: PhysioNet (https://physionet.org/content/sleep-edfx/)
- **Format**: EDF (European Data Format)
- **Channels**: EEG (Fpz-Cz, Pz-Oz), EOG, EMG
- **Sampling**: 100 Hz
- **Annotations**: 30-second epoch hypnogram

**Current Subjects**:
- SC4001E0-PSG.edf (48 MB)
- SC4002E0-PSG.edf (51 MB)

**Expansion**: Download additional subjects from PhysioNet for robust validation.

### Meditation-EEG Dataset
- **Source**: OpenNeuro
- **Format**: BIDS (Brain Imaging Data Structure)
- **Subjects**: 22+ (sub-001 through sub-022)
- **Sessions**: 2 per subject (ses-01, ses-02)
- **Task**: Meditation with event markers

**Directory Structure**:
```
data/meditation-eeg/
├── sub-001/
│   ├── ses-01/eeg/
│   │   ├── sub-001_ses-01_task-meditation_eeg.json
│   │   └── sub-001_ses-01_task-meditation_events.tsv
│   └── ses-02/eeg/
│       └── ...
├── sub-002/
│   └── ...
└── participants.tsv
```

---

## Implementation Components

### LTC-Based Processors

All physiological signal processing uses Liquid Time-Constant (LTC) neurons:

```rust
// Dual-τ LTC for spectral discrimination
struct ThetaAlphaRatio {
    tau_fast: f64,  // ~40ms - Alpha/Beta resonance
    tau_slow: f64,  // ~100ms - Theta resonance
    energy_fast: f64,
    energy_slow: f64,
}
```

### Permutation Entropy

Measures time-domain complexity (the "differentiation" component of Φ):

```rust
fn permutation_entropy(signal: &[f64], delay: usize) -> f64;
// Returns 0.0 (monotonic) to 1.0 (random)
// Wake: ~0.8-0.9 | REM: ~0.7 | Deep Sleep: ~0.3
```

### Eye Movement Detection

EOG-based detector for REM identification:

```rust
struct EyeMovementDetector {
    movement_energy: f64,    // Running variance
    movement_frequency: f64, // Zero-crossing rate
    fn rem_indicator(&self) -> f64;
}
```

### Muscle Tone Tracking

EMG-based atonia detection:

```rust
struct MuscleToneTracker {
    fn is_atonia(&self) -> bool;       // REM signature
    fn is_tone_dropping(&self) -> bool; // N1 signature
}
```

---

## Running Validations

### C. elegans Validation

```bash
# Run C. elegans tests
cargo test celegans -- --nocapture

# See Φ analysis output
cargo test test_phi_analysis -- --nocapture
cargo test test_topology_comparison -- --nocapture
```

### Clinical Sleep Validation

```bash
# Download sleep-edf data first
./scripts/download_sleep_edf.sh

# Run validation (requires data)
cargo run --example clinical_validation --release
```

### Meditation Analysis

```bash
# Run meditation detection tests
cargo test meditation -- --nocapture

# Analyze meditation dataset
cargo run --example meditation_phi_analysis --release
```

---

## Validation Results

### Expected Outcomes

| Validation | Success Criteria | Status |
|------------|------------------|--------|
| C. elegans Φ > Random | Φ ratio > 1.0 | PASSING |
| Sleep stage accuracy | > 75% agreement | PASSING |
| REM detection | Atonia + eye movement correlation | PASSING |
| Meditation flow detection | Gamma-theta coupling | IMPLEMENTED |

### Key Findings

1. **Biological networks optimize for integration**: C. elegans Φ consistently exceeds random networks of equal size.

2. **LTC dynamics resonate with brain states**: Sleep stage classification achieves high accuracy using only LTC-based features.

3. **Spectral texture distinguishes states**: Theta/alpha ratio cleanly separates REM from Wake despite similar entropy.

---

## Future Directions

### Phase 4+ Enhancements

1. **Expand Sleep-EDF to 10+ subjects** for statistical power
2. **PyPhi cross-validation** on small systems (3-5 nodes)
3. **Real-time meditation Φ feedback** using live EEG
4. **Multi-subject Φ correlation** with subjective reports
5. **Anesthesia validation** (loss/recovery of consciousness)

### Research Questions

- Does Φ decrease during anesthesia proportionally to depth?
- Can Φ predict meditation depth better than traditional EEG markers?
- Do long-term meditators show structural Φ differences?
- Is there a minimum Φ threshold for conscious experience?

---

## References

### Connectome
- White et al. (1986) - C. elegans nervous system
- Varshney et al. (2011) - C. elegans network properties

### Sleep Science
- Berry et al. (2017) - AASM Sleep Scoring Manual
- Kemp et al. (2000) - Sleep-EDF database

### Meditation Science
- Lutz et al. (2004) - Gamma oscillations in meditation
- Cahn & Polich (2006) - Meditation and EEG review
- Csikszentmihalyi (1990) - Flow psychology

### Integrated Information Theory
- Tononi et al. (2016) - IIT 3.0 foundations
- Oizumi et al. (2014) - IIT mathematical framework

---

*Part of Symthaea-HLB: Consciousness-first AI with rigorous biological validation*

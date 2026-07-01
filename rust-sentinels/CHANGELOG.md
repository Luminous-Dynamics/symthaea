# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- Python bindings via PyO3
- WebAssembly support
- Extended Proofs: Attention, Flow, Engagement

## [0.1.0] - 2026-01-18

### Added

#### Core Features
- **ProofOfConsciousness** - Unified consciousness state structure
- **ConsciousnessState** enum with 10 states:
  - DeepSleep, LightSleep, REM, Drowsy
  - Relaxed, Alert, Focused, Meditative, Flow, Stressed
- **analyze_consciousness()** - Main API entry point
- **AnalysisMode** - Auto, Emotion, Sleep, Meditation modes

#### Sentinels
- **EmotionSentinel** (Proof of Joy)
  - Valence detection (-1 to +1)
  - Arousal detection (0 to 1)
  - Emotion quadrant classification
  - Validated: r=0.391 on DENS dataset
- **SleepSentinel** (Proof of Rest)
  - 5-class sleep staging (W, N1, N2, N3, REM)
  - Delta/theta/beta power analysis
  - Confidence scoring
  - Validated: 69.6% accuracy on Sleep-EDF
- **MeditationSentinel** (Proof of Focus)
  - Meditation depth (0-1)
  - Stability measurement
  - Meditation index calculation
  - Baseline calibration support
  - Validated: 5/5 checks on OpenNeuro data

#### Signal Processing
- **welch_psd()** - Welch power spectral density
- **band_power()** - Frequency band power extraction
- **extract_band_powers()** - All standard EEG bands
- Hanning window implementation
- 50% overlap segmentation

#### Output
- JSON serialization for all types
- to_json() method on ProofOfConsciousness
- Serde derive support

#### Documentation
- Comprehensive README with examples
- API reference documentation
- Technical whitepaper
- Validation methodology

#### Examples
- `basic_usage.rs` - Getting started example
- `streaming.rs` - Real-time streaming simulation

#### Benchmarks
- Full analysis benchmarks (5s, 10s, 30s epochs)
- Individual sentinel benchmarks

### Performance
- Full analysis: 93-207 µs
- Individual sentinels: 57-65 µs
- Memory: < 40 KB total

### Validated
- EmotionSentinel: r=0.391 correlation with self-reported valence
- SleepSentinel: 69.6% 5-class accuracy
- MeditationSentinel: All feature extraction checks passed

## [0.0.1] - 2026-01-15

### Added
- Initial project structure
- Cargo.toml setup
- Basic type definitions

---

## Version History Summary

| Version | Date | Highlights |
|---------|------|------------|
| 0.1.0 | 2026-01-18 | First validated release with Consciousness Trilogy |
| 0.0.1 | 2026-01-15 | Initial project setup |

## Roadmap

### v0.2.0 (Planned)
- Python bindings via PyO3
- NumPy array support
- pip installable package

### v0.3.0 (Planned)
- Extended Proofs implementation
- AttentionSentinel
- FlowSentinel
- EngagementSentinel

### v0.4.0 (Planned)
- WebAssembly support
- Browser demo
- JavaScript API

### v1.0.0 (Planned)
- Production-ready release
- Comprehensive validation
- Performance optimizations
- Complete documentation

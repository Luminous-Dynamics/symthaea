# Supplementary Materials: Benchmark Reproduction Instructions

## Hyperdimensional Active Inference for Consciousness-Aware Distributed Intelligence

### Software Requirements

- **Rust**: 1.82+ (stable)
- **Crate**: `symthaea` v0.5.0
- **Platform**: Linux x86_64 (tested on NixOS 25.11)
- **Memory**: 16GB RAM minimum for full benchmark suite

### Repository

```bash
git clone https://github.com/Luminous-Dynamics/luminous-dynamics.git
cd luminous-dynamics/symthaea
```

### Running Benchmarks

All benchmarks are standalone Rust examples in `examples/benchmark_*.rs`.

```bash
# Ethics HDC (92.9% ETHICS-only, ~5 min)
cargo run --release --example benchmark_ethics_hdc

# ISOLET speaker classification (91.7%, ~2 min)
cargo run --release --example benchmark_isolet

# LibriSpeech speaker ID (94.5%, ~1 min)
cargo run --release --example benchmark_librispeech

# C. elegans connectome Phi (448 neurons, ~3 min)
cargo run --release --example benchmark_celegans

# MNIST digit classification (88.5%, ~3 min)
cargo run --release --example benchmark_mnist_hdc

# Federated learning + Byzantine fault tolerance (~1 min)
cargo run --release --example benchmark_federated

# EEG seizure detection (100% sensitivity/specificity, ~1 min)
cargo run --release --example benchmark_seizure

# Sleep staging with HMM (all 5 AASM stages, ~2 min)
cargo run --release --example benchmark_sleepstage

# Emotion EEG valence/arousal (~1 min)
cargo run --release --example benchmark_emotion_eeg

# Anesthesia Phi induction/recovery (~2 min)
cargo run --release --example benchmark_anesthesia_phi

# ARC reasoning patterns (~1 min)
cargo run --release --example benchmark_arc_reasoning

# Drosophila connectome Phi (~2 min)
cargo run --release --example benchmark_drosophila_phi

# PyPhi groundtruth validation (~1 min)
cargo run --release --example benchmark_pyphi_groundtruth

# PCI validation (~1 min)
cargo run --release --example benchmark_pci_validation

# Tokamak CfC real-time inference (~1 min)
cargo run --release --example benchmark_tokamak_cfc

# Meditation/resting state EEG (~1 min)
cargo run --release --example benchmark_meditation_resting
```

### Key Results Summary

| Benchmark | Metric | Result |
|-----------|--------|--------|
| Ethics HDC | ETHICS-only accuracy (4 categories) | 92.9% |
| ISOLET | Classification accuracy (26 letters) | 91.7% |
| LibriSpeech | Speaker identification (10 speakers) | 94.5% |
| C. elegans | Circuit Phi range (5 circuits) | 0.53-0.58 |
| MNIST | Digit classification (10 classes) | 88.5% |
| Federated | Byzantine fault tolerance threshold | 34% |
| Seizure | Sensitivity / Specificity | 100% / 100% |
| Sleep | Stages classified | 5/5 AASM |

### External Datasets

Some benchmarks use embedded synthetic data. For real data benchmarks:

- **Sleep-EDF**: PhysioNet Sleep-EDF Database (auto-downloaded on first run)
- **ISOLET**: UCI Machine Learning Repository (requires manual download)
- **LibriSpeech**: OpenSLR (subset embedded)
- **Social Chemistry**: Allen AI Social Chemistry 101 (for ethics prototype training)

### Benchmark Results Location

Results are written to `data/benchmarks/<name>/results.json` in machine-readable JSON format.

### Test Suite

```bash
# Run all 3,200+ unit tests
cargo test --lib

# Run with all features
cargo test --lib --all-features
```

### Hardware Used for Published Results

- CPU: AMD Ryzen 9 7950X (16C/32T)
- RAM: 64GB DDR5
- OS: NixOS 25.11
- Rust: 1.82.0

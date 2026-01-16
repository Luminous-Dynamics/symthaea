# Validation Studies

*Empirical testing of Symthaea's claims.*

---

## Overview

We validate Symthaea through multiple approaches:
1. C. elegans connectome
2. EEG pattern analysis
3. PyPhi comparison
4. Internal benchmarks

---

## C. elegans Connectome

The nematode C. elegans has a completely mapped nervous system:
- 302 neurons
- ~7,000 synapses
- Ground truth for consciousness measures

**Status**: Example exists, validation ongoing.

```bash
cargo run --example c_elegans_validation --release
```

---

## EEG Pattern Analysis

EEG data provides empirical measures of brain integration:
- Compare Φ predictions with real neural activity
- Test theoretical predictions

**Status**: Example exists, validation ongoing.

```bash
cargo run --example eeg_pattern_generation --release
```

---

## PyPhi Comparison

PyPhi provides exact IIT calculations for small systems:
- Validate our approximations
- Requires Python + PyPhi installation

**Status**: Feature flag `pyphi` enables comparison.

```bash
cargo build --features pyphi
```

---

## Internal Benchmarks

Regular benchmarking ensures performance claims:

```bash
cargo bench
```

Expected:
| Operation | Time |
|-----------|------|
| HDC Encoding | 0.05ms |
| HDC Recall | 0.10ms |
| LTC Step | 0.02ms |
| Full Query | 0.50ms |

---

## Statistical Methods

- t-tests and ANOVA across topologies
- Multiple seeds for cross-validation
- Effect sizes, not just p-values
- Confidence intervals

---

*"Claims require evidence."*

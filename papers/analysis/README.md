# Five-Component Consciousness (FCC) Analysis Toolkit

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository contains the reference implementation for computing the five components of consciousness from neural data, as described in:

> **The Master Equation of Consciousness: A Unified Five-Component Framework**
> Tristan Stoltz, Luminous Dynamics
> Correspondence: tristan.stoltz@luminousdynamics.org

The framework quantifies consciousness as:

**C = min(Φ, B, W, A, R)**

where:
- **Φ (Integration)**: Lempel-Ziv complexity of neural signals
- **B (Binding)**: Gamma-band phase coherence (Kuramoto parameter)
- **W (Workspace)**: Global signal variance × connectivity
- **A (Attention)**: Beta/alpha ratio + arousal index
- **R (Recursion)**: Frontal-posterior theta phase-locking

## Installation

```bash
# Clone repository
git clone https://github.com/Luminous-Dynamics/five-component-consciousness.git
cd five-component-consciousness

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Optional: Install with MNE for EDF file support
pip install mne
```

## Quick Start

```python
from compute_components import compute_consciousness

# Your EEG data: shape (n_channels, n_samples)
eeg_data = load_your_data()
sfreq = 256  # Sampling frequency

# Compute all components
result = compute_consciousness(eeg_data, sfreq)

print(f"Φ (Integration): {result['phi']:.2f}")
print(f"B (Binding):     {result['binding']:.2f}")
print(f"W (Workspace):   {result['workspace']:.2f}")
print(f"A (Attention):   {result['attention']:.2f}")
print(f"R (Recursion):   {result['recursion']:.2f}")
print(f"C (Overall):     {result['C']:.2f}")
```

## Repository Structure

```
five-component-consciousness/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── LICENSE                      # MIT License
│
├── compute_components.py        # Main computation module
├── sleep_edf_analysis.py        # Sleep-EDF validation pipeline
│
├── data/                        # Sample data (not included in repo)
│   └── README.md               # Data download instructions
│
├── tests/                       # Unit tests
│   ├── test_components.py
│   └── test_synthetic.py
│
├── examples/                    # Example notebooks
│   ├── basic_usage.ipynb
│   ├── sleep_analysis.ipynb
│   └── clinical_application.ipynb
│
└── docs/                        # Documentation
    ├── API.md
    ├── METHODS.md
    └── VALIDATION.md
```

## Module Reference

### `compute_components.py`

Core computation functions:

| Function | Description | Input | Output |
|----------|-------------|-------|--------|
| `compute_phi(data, sfreq)` | Lempel-Ziv complexity | EEG array | 0-1 |
| `compute_binding(data, sfreq)` | Gamma coherence | EEG array | 0-1 |
| `compute_workspace(data, sfreq)` | Global signal | EEG array | 0-1 |
| `compute_attention(data, sfreq)` | Spectral ratio | EEG array | 0-1 |
| `compute_recursion(data, sfreq)` | Theta PLV | EEG array | 0-1 |
| `compute_consciousness(data, sfreq)` | All components | EEG array | dict |

### `sleep_edf_analysis.py`

Validation pipeline for Sleep-EDF database:

```bash
# Run validation
python sleep_edf_analysis.py

# Output: REAL_DATA_VALIDATION.md
```

## Data Requirements

- **Format**: NumPy array, shape (n_channels, n_samples)
- **Channels**: ≥2 EEG channels (more is better for frontal-posterior PLV)
- **Sampling Rate**: ≥100 Hz recommended (tested with 256 Hz)
- **Epoch Length**: 30 seconds recommended

## Validation Results

### Sleep Stages (Synthetic + Sleep-EDF)

| Stage | C (mean) | Expected Order |
|-------|----------|----------------|
| Wake  | 0.26     | 1st            |
| REM   | 0.21     | 2nd            |
| N1    | 0.09     | 3rd            |
| N2    | 0.05     | 4th            |
| N3    | 0.04     | 5th            |

Ordering confirmed: Wake > REM > N1 > N2 > N3 (p < 0.001)

### Anesthesia Depths (Synthetic)

| Depth    | C (mean) | Expected Order |
|----------|----------|----------------|
| Awake    | 0.20     | 1st            |
| Sedation | 0.20     | 1st (tie)      |
| Light    | 0.04     | 3rd            |
| Moderate | 0.01     | 4th            |
| Deep     | 0.01     | 4th (tie)      |

## Citation

If you use this code, please cite:

```bibtex
@article{fcc2025,
  title={The Master Equation of Consciousness: A Unified Five-Component Framework},
  author={[Authors]},
  journal={[Journal]},
  year={2025},
  doi={[DOI]}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.

## Contributing

Contributions welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) first.

## Acknowledgments

- Sleep-EDF database: PhysioNet
- Inspiration: IIT, GWT, HOT theory communities

## Contact

[corresponding.author@institution.edu]

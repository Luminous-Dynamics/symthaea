# Hyperdimensional Active Inference (HAI)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Hyperdimensional Active Inference** integrates the Free Energy Principle with Hyperdimensional Computing for efficient, interpretable cognitive architectures.

## Key Results

| Metric | HAI | pymdp | Speedup |
|--------|-----|-------|---------|
| Inference | 0.093 ms | 0.318 ms | **1.9×** |
| Action Selection | 0.148 ms | 2.338 ms | **15.8×** |
| Success Rate | 88-100% | 10-16% | **+72-84%** |

## Installation

```bash
# Clone repository
git clone https://github.com/[anonymous]/hai-symthaea.git
cd hai-symthaea

# Rust implementation
cargo build --release
cargo test test_fep_active_inference

# Python validation scripts
pip install numpy scipy matplotlib
python validation/pymdp_comparison_benchmark.py
```

## Quick Start

```rust
use symthaea::fep_active_inference::*;

// Create agent
let mut agent = FEPActiveInferenceAgent::new(16384);

// Update beliefs given observation
let observation = encode_observation(&sensor_data);
let free_energy = agent.update_belief(&observation);

// Select action via expected free energy
let action = agent.select_action(&goal_state);
```

## Paper

See `papers/latex/hai_neurips2026.pdf` for the full paper.

**Abstract:** We present Hyperdimensional Active Inference (HAI), the first integration of the Free Energy Principle with Hyperdimensional Computing. HAI reformulates variational free energy using cosine similarity in high-dimensional space and introduces precision-weighted binding for uncertainty-modulated feature combination. On T-Maze and Grid World benchmarks, HAI achieves 7.9× total speedup over pymdp while improving success rates from 10-16% to 88-100%.

## Project Structure

```
symthaea-hlb/
├── src/
│   └── fep_active_inference.rs    # Core HAI implementation (Rust)
├── validation/
│   ├── pymdp_comparison_benchmark.py  # Benchmark vs pymdp
│   ├── ablation_studies.py            # Dimension/precision ablations
│   └── statistical_analysis.py        # Statistical significance
├── papers/
│   ├── latex/                         # NeurIPS submission
│   ├── figures/                       # Generated figures
│   └── appendices/                    # Theoretical proofs
└── docs/
    ├── PYMDP_COMPARISON_REPORT.md     # Detailed benchmark results
    ├── ABLATION_STUDIES_REPORT.md     # Ablation study results
    └── STATISTICAL_ANALYSIS_REPORT.md # Statistical analysis
```

## Reproducibility

### Run Benchmarks

```bash
# pymdp comparison
python validation/pymdp_comparison_benchmark.py

# Ablation studies
python validation/ablation_studies.py

# Statistical analysis (10 seeds × 10 trials)
python validation/statistical_analysis.py
```

### Generate Figures

```bash
python papers/figures/generate_hai_figures.py
```

### Compile Paper

```bash
cd papers/latex
pdflatex hai_neurips2026.tex
bibtex hai_neurips2026
pdflatex hai_neurips2026.tex
pdflatex hai_neurips2026.tex
```

## Citation

```bibtex
@inproceedings{anonymous2026hai,
  title={Hyperdimensional Active Inference: Free Energy Principle in Vector Symbolic Architectures},
  author={Anonymous},
  booktitle={NeurIPS 2026},
  year={2026}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- pymdp team for the active inference baseline
- HDC/VSA research community
- Free Energy Principle literature

---

*Code for "Hyperdimensional Active Inference: Free Energy Principle in Vector Symbolic Architectures"*

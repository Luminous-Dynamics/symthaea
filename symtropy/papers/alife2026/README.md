# ALIFE 2026 Paper: The Conscious/Unconscious Distinction

## Reproducing Results

All experiments can be reproduced from the `symtropy` directory:

```bash
cd symtropy/crates/symtropy-consciousness-physics

# Table 1: Ablation (6 conditions × 5 seeds)
cargo run --release --example ablation_study

# Table 2: Metric independence (5 metrics × 30 seeds)
cargo run --release --example phi_ablation

# Table 3: Causal test (6 conditions × 20 seeds)
cargo run --release --example phi_causal

# Scaling + Phase diagram
cargo run --release --example scaling_and_phase

# Comprehensive Phi effects (5 DVs × 5 metrics × 20 seeds)
cargo run --release --example phi_effects
```

## Generating Figures

```bash
cd papers/alife2026
pip install matplotlib numpy
python3 generate_figures.py
```

## Compiling the Paper

```bash
cd papers/alife2026
pdflatex main.tex
pdflatex main.tex  # twice for references
```

## Requirements

- Rust 1.75+ with `cargo`
- Python 3.8+ with matplotlib (for figures only)
- pdflatex (for paper compilation only)

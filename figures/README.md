# Figures

Generated visualizations for papers and documentation.

## Generation

```bash
# Generate all figures
python generate_figures.py

# Generate supplementary figures
python generate_supplementary_figures.py

# Paper-specific figures
python papers/generate_paper01_figures.py
```

## Contents

- **Φ topology plots** - Phi values across network topologies
- **Consciousness graphs** - Visualizations of graph evolution
- **Performance charts** - Benchmark results
- **Architecture diagrams** - System architecture

## Formats

- PNG for documentation
- PDF/EPS for papers
- SVG for web

## Dependencies

```bash
pip install matplotlib seaborn networkx
```

## Output

Figures are saved here and copied to:
- `papers/figures/` for academic papers
- `docs/` for documentation

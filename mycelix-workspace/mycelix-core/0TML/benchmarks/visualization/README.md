# Visualization Scripts for Zero-TrustML Paper

Publication-ready figures for academic paper submission.

## Usage

```bash
# After MNIST experiment completes, generate all figures:
cd /srv/luminous-dynamics/Mycelix-Core/0TML/benchmarks/visualization
nix develop --command python plot_mnist_results.py
```

## Generated Outputs

### Figures (PDF + PNG)
- `mnist_accuracy_comparison.pdf` - Figure 1: Accuracy vs Rounds
- `mnist_convergence_comparison.pdf` - Figure 2: Convergence Speed
- `mnist_loss_curves.pdf` - Supplementary: Loss Curves

### Tables
- `mnist_results_table.tex` - LaTeX table for paper
- `mnist_results_table.md` - Markdown table for README

## Requirements

- matplotlib
- numpy
- Completed MNIST experiment results in `../results/`

## Publication Format

All figures generated in IEEE two-column format:
- Width: 3.5 inches (single column)
- DPI: 300 (publication quality)
- Font: Times New Roman
- Format: PDF (vector) + PNG (preview)

## Quick Preview

To test with existing results:
```bash
# Run visualization on any completed experiment
python plot_mnist_results.py
```

The script automatically finds the latest results files.

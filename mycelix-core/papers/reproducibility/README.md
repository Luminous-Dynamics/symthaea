# Mycelix Paper Reproducibility Kit

**Paper**: Mycelix: Byzantine-Resistant Federated Learning with Sub-Millisecond Aggregation
**Venue**: MLSys 2026
**Version**: 1.0.0

This kit enables full reproducibility of all experimental results presented in the paper.

---

## Quick Start

### Using Docker (Recommended)

```bash
# Build the reproducibility container
docker build -t mycelix-paper .

# Run all experiments and generate results
docker run -v $(pwd)/output:/app/output mycelix-paper python run_all.py

# Results will be available in:
# - output/results/       # Raw JSON/CSV results
# - output/figures/       # Generated figures (PNG)
# - output/tables/        # LaTeX tables
```

### Manual Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install exact dependencies
pip install -r requirements_exact.txt

# Run all experiments
python run_all.py
```

---

## Hardware Requirements

### Minimum (for validation)
- **CPU**: 4 cores, 2.0GHz+
- **RAM**: 8GB
- **Storage**: 10GB SSD
- **Expected Runtime**: ~30 minutes (quick mode)

### Recommended (for full reproduction)
- **CPU**: 8+ cores, 2.5GHz+
- **RAM**: 32GB
- **Storage**: 50GB SSD
- **Expected Runtime**: ~2-4 hours (full mode)

### Paper Results Hardware
- **CPU**: AMD EPYC 7763 (64 cores)
- **RAM**: 256GB
- **Storage**: 1TB NVMe SSD
- **Note**: Results may vary slightly on different hardware due to timing measurements

---

## Experiment Overview

### Table 1: Byzantine Detection Rate (Sign-Flip Attack)
**Script**: `experiments/table1_byzantine_detection.py`
**Runtime**: ~20 minutes (full) / ~5 minutes (quick)
**Description**: Measures detection rate across Byzantine ratios (10%, 20%, 33%, 45%)

| Byzantine Ratio | Expected Detection Rate |
|-----------------|------------------------|
| 10%             | 99.0% +/- 1.0%         |
| 20%             | 98.0% +/- 1.5%         |
| 33%             | 95.0% +/- 2.0%         |
| 45%             | 89.0% +/- 3.0%         |

### Table 2: Aggregation Latency
**Script**: `experiments/table2_latency.py`
**Runtime**: ~15 minutes (full) / ~3 minutes (quick)
**Description**: Measures aggregation latency for PoGQ, Krum, FLTrust, and FedAvg

| Method  | Expected Median (ms) |
|---------|---------------------|
| PoGQ    | 0.45 +/- 0.05       |
| Krum    | 2.85 +/- 0.20       |
| FLTrust | 0.39 +/- 0.05       |
| FedAvg  | 0.12 +/- 0.02       |

### Figure 1: Detection Rate vs Byzantine Ratio
**Script**: `experiments/figure1_detection_vs_ratio.py`
**Runtime**: ~25 minutes
**Description**: Line plot comparing detection rates across methods

### Figure 2: Convergence Over Rounds
**Script**: `experiments/figure2_convergence.py`
**Runtime**: ~30 minutes
**Description**: Model accuracy convergence with/without Byzantine attackers

### Figure 3: Scalability
**Script**: `experiments/figure3_scalability.py`
**Runtime**: ~45 minutes
**Description**: Latency vs node count (10 to 1000 nodes)

---

## Running Individual Experiments

```bash
# Table 1: Byzantine detection
python experiments/table1_byzantine_detection.py

# Table 2: Latency measurements
python experiments/table2_latency.py

# Figure 1: Detection vs ratio plot
python experiments/figure1_detection_vs_ratio.py

# Figure 2: Convergence plot
python experiments/figure2_convergence.py

# Figure 3: Scalability analysis
python experiments/figure3_scalability.py
```

---

## Validating Results

```bash
# Validate all results against expected values
python validate.py

# Example output:
# [PASS] Table 1 Byzantine Detection: Within tolerance
# [PASS] Table 2 Latency: Within tolerance
# [WARN] Figure 3 Scalability: 1000 nodes - 5% slower than expected (within tolerance)
```

---

## Output Structure

```
output/
├── results/
│   ├── table1_byzantine_detection.json
│   ├── table2_latency.json
│   ├── figure1_data.json
│   ├── figure2_data.json
│   └── figure3_data.json
├── figures/
│   ├── figure1_detection_vs_ratio.png
│   ├── figure2_convergence.png
│   └── figure3_scalability.png
└── tables/
    ├── table1_byzantine_detection.tex
    └── table2_latency.tex
```

---

## Configuration

All hyperparameters are defined in `config.yaml`. Key parameters:

```yaml
experiment:
  random_seed: 42              # For reproducibility
  num_trials: 10               # Trials per configuration

byzantine:
  ratios: [0.1, 0.2, 0.33, 0.45]
  attack_types: [sign_flip, scaling, label_flip, backdoor]

aggregation:
  methods: [pogq, krum, fltrust, fedavg]
  gradient_dim: 10000
```

---

## Troubleshooting

### Memory Issues
If running out of memory at large scales:
```bash
# Run with reduced node count
python run_all.py --max-nodes 100
```

### Timing Variance
Latency measurements may vary by +/- 20% depending on:
- CPU architecture (ARM vs x86)
- Background processes
- Memory speed

Run multiple trials for accurate statistics:
```bash
python experiments/table2_latency.py --trials 20
```

### Reproducibility Note
Results are deterministic with `random_seed=42`. If you get different results:
1. Verify NumPy version matches `requirements_exact.txt`
2. Check for GPU influence (set `CUDA_VISIBLE_DEVICES=""`)
3. Run in Docker for exact environment match

---

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{mycelix2026,
  title={Mycelix: Byzantine-Resistant Federated Learning with Sub-Millisecond Aggregation},
  author={[Authors]},
  booktitle={MLSys},
  year={2026}
}
```

---

## License

This reproducibility kit is released under the MIT License.

---

## Contact

For questions about reproducibility, please open an issue in the repository or contact [REDACTED].

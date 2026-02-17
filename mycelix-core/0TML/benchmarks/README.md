# Benchmarking Infrastructure

**Purpose**: Rigorous academic validation for peer-reviewed publication

## Structure

```
benchmarks/
├── datasets/           # Real dataset loaders (MNIST, CIFAR-10, FEMNIST)
├── baselines/          # Baseline FL algorithms (FedAvg, Multi-Krum, Bulyan)
├── experiments/        # Experiment scripts and configurations
└── results/           # Experiment results and analysis
```

## Budget: $0

All experiments run locally using:
- Real MNIST dataset (download once, ~50MB)
- Real CIFAR-10 dataset (download once, ~170MB)
- CPU-based training (no cloud costs)
- Free cloud credits if scale testing needed

## Experiments Planned

### 1. Accuracy vs Communication Rounds
**Compare**: Zero-TrustML (Krum) vs FedAvg vs Multi-Krum
**Datasets**: MNIST, CIFAR-10
**Metrics**: Model accuracy, convergence speed

### 2. Byzantine Detection Rate
**Compare**: Zero-TrustML (PoGQ + Credits) vs Krum vs Bulyan
**Attack Types**: Random noise, poisoning, coordinated
**Metrics**: Detection rate, false positive rate

### 3. Non-IID Performance
**Compare**: Zero-TrustML vs FedAvg on heterogeneous data
**Data Splits**: Dirichlet α ∈ {0.1, 0.5, 1.0, 5.0}
**Metrics**: Accuracy degradation, convergence time

### 4. Communication Overhead
**Measure**: Bytes transmitted per round
**Compare**: Zero-TrustML (O(N²d)) vs FedAvg (O(Nd))
**Analysis**: Trade-off for decentralization

## Running Experiments

```bash
# Download datasets (one-time)
python benchmarks/datasets/download_all.py

# Run single experiment
python benchmarks/experiments/mnist_accuracy.py

# Run full benchmark suite
./benchmarks/run_all_experiments.sh

# Analyze results
python benchmarks/analyze_results.py
```

## Target Metrics for Publication

### Table 1: Accuracy vs Rounds (MNIST)
| Algorithm | Round 10 | Round 50 | Round 100 | Final |
|-----------|----------|----------|-----------|-------|
| FedAvg    | TBD      | TBD      | TBD       | TBD   |
| Krum      | TBD      | TBD      | TBD       | TBD   |
| Zero-TrustML   | TBD      | TBD      | TBD       | TBD   |

### Table 2: Byzantine Detection (10 nodes, 3 Byzantine)
| Attack Type  | Krum | Bulyan | Zero-TrustML | Improvement |
|--------------|------|--------|---------|-------------|
| Random       | TBD  | TBD    | TBD     | TBD         |
| Poisoning    | TBD  | TBD    | TBD     | TBD         |
| Coordinated  | TBD  | TBD    | TBD     | TBD         |

### Table 3: Non-IID Performance (CIFAR-10)
| α (heterogeneity) | FedAvg | Zero-TrustML | Accuracy Gap |
|-------------------|--------|---------|--------------|
| 0.1 (high)        | TBD    | TBD     | TBD          |
| 0.5 (moderate)    | TBD    | TBD     | TBD          |
| 1.0 (low)         | TBD    | TBD     | TBD          |

## Status

- [x] Directory structure created
- [ ] Dataset loaders implemented
- [ ] Baseline algorithms implemented
- [ ] Experiment scripts created
- [ ] Results analysis pipeline ready

## Timeline

- Week 1-2: Dataset loaders + FedAvg baseline
- Week 3-4: MNIST experiments (accuracy + Byzantine)
- Week 5-6: CIFAR-10 experiments + non-IID
- Week 7-8: Analysis and paper figures

## References

- FedAvg: McMahan et al. (2017)
- Krum: Blanchard et al. (2017)
- Bulyan: Mhamdi et al. (2018)
- Non-IID: Hsu et al. (2019) - Dirichlet splits

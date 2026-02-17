# Non-IID Data Validation Experiments

This directory contains comprehensive experiments for validating Mycelix Byzantine-resilient federated learning under Non-IID (Non-Independent and Identically Distributed) data conditions.

## Background

In real-world federated learning deployments, data across nodes is rarely IID. Each participating device has unique usage patterns, demographics, or operating conditions that lead to heterogeneous data distributions. This heterogeneity can:

1. **Affect Byzantine Detection**: Non-IID data causes legitimate gradient variance that may be confused with malicious behavior
2. **Slow Convergence**: Models may oscillate or fail to converge when gradients point in vastly different directions
3. **Create Unfair Outcomes**: Nodes with minority data distributions may be systematically disadvantaged

## Non-IID Scenarios

### 1. Label Skew (`scenarios/label_skew.py`)

Each node has a different distribution of class labels, simulated using Dirichlet allocation.

| Alpha | Distribution | Description |
|-------|-------------|-------------|
| 0.1 | Extreme | Each node has 1-2 dominant classes |
| 0.5 | Moderate | Each node has 3-5 visible classes |
| 1.0 | Mild | Roughly uniform with some variance |

**Example**: Node A has 80% cats, 20% dogs; Node B has 90% planes, 10% ships.

### 2. Feature Skew (`scenarios/feature_skew.py`)

Different nodes receive data with different feature characteristics even for the same labels.

- **Brightness shift**: Some nodes have darker/lighter images
- **Rotation**: Some nodes have rotated samples
- **Noise injection**: Some nodes have noisier data
- **Domain shift**: Hospital A has different scanner characteristics than Hospital B

### 3. Quantity Skew (`scenarios/quantity_skew.py`)

Massive imbalance in data quantity across nodes.

| Node Type | Samples | Ratio |
|-----------|---------|-------|
| Large | 10,000 | 100x |
| Medium | 1,000 | 10x |
| Small | 100 | 1x |

This simulates scenarios like:
- Mobile devices with varying usage
- Hospitals with different patient volumes
- IoT sensors with different activity levels

### 4. Combined Skew (`scenarios/combined_skew.py`)

All three skew types applied simultaneously for maximum stress testing.

## Metrics Collected

Each experiment measures:

1. **Detection Accuracy**
   - True Positive Rate (Byzantine correctly identified)
   - False Positive Rate (honest nodes incorrectly flagged)
   - F1 Score

2. **Model Performance**
   - Test accuracy on IID holdout set
   - Per-class accuracy
   - Accuracy on minority vs majority classes

3. **Convergence**
   - Rounds to reach target accuracy
   - Gradient variance over time
   - Loss curve stability

4. **Fairness**
   - Reputation scores across nodes
   - Contribution weights
   - Gini coefficient of influence

## Running Experiments

### Quick Start

```bash
# Run all experiments with default settings
python experiments/non_iid/run_all.py

# Run specific scenario
python experiments/non_iid/scenarios/label_skew.py --alpha 0.1

# Run with custom configuration
python experiments/non_iid/run_all.py --config experiments/non_iid/config.yaml
```

### Configuration

Edit `config.yaml` to customize:

```yaml
# Number of federated nodes
num_nodes: 20

# Byzantine ratio (fraction of malicious nodes)
byzantine_ratio: 0.3

# Dataset: mnist or cifar10
dataset: cifar10

# Dirichlet alpha values for label skew
label_skew_alphas: [0.1, 0.5, 1.0]

# Number of FL rounds
num_rounds: 100

# Random seed for reproducibility
seed: 42
```

### Output Structure

Results are saved to:

```
results/non_iid/{scenario}/{timestamp}/
  metrics.json       # Raw metrics data
  plots/
    detection_accuracy.png
    convergence_curve.png
    fairness_distribution.png
    per_class_accuracy.png
  summary.md         # Human-readable summary
```

## Expected Results

Based on IID benchmarks (100% detection, 0.7ms latency), Non-IID scenarios may show:

| Scenario | Expected Detection | Expected Impact |
|----------|-------------------|-----------------|
| Label Skew (alpha=0.1) | 85-95% | High false positives possible |
| Label Skew (alpha=0.5) | 95-99% | Moderate impact |
| Label Skew (alpha=1.0) | 99-100% | Minimal impact |
| Feature Skew | 90-98% | Depends on shift magnitude |
| Quantity Skew | 95-99% | May bias toward large nodes |
| Combined | 80-95% | Cumulative challenges |

## Analysis Tools

### Convergence Analysis (`analysis/convergence.py`)

Tracks model convergence under Non-IID conditions:
- Loss curves per round
- Gradient norm statistics
- Client drift measurement

### Detection Analysis (`analysis/detection.py`)

Evaluates Byzantine detection under Non-IID:
- Per-class detection rates
- False positive analysis by data distribution
- Attack resilience comparison

### Fairness Analysis (`analysis/fairness.py`)

Measures fairness of the FL system:
- Reputation score distribution
- Contribution equality (Gini coefficient)
- Minority node treatment

## References

1. Zhao, Y., et al. "Federated Learning with Non-IID Data" (2018)
2. Li, T., et al. "Federated Optimization in Heterogeneous Networks" (2020)
3. Karimireddy, S.P., et al. "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning" (2020)

## Contributing

When adding new scenarios:

1. Follow the pattern in existing scenario files
2. Implement the `NonIIDScenario` interface
3. Add configuration to `config.yaml`
4. Update this README with expected results
5. Add tests to `tests/experiments/test_non_iid.py`

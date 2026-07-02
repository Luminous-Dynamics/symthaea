# Federated Learning Experiments Framework

Automated experiment execution for FL baseline comparisons.

---

## 📁 Directory Structure

```
experiments/
├── models/
│   └── cnn_models.py              # SimpleCNN, ResNet9, CharLSTM
├── utils/
│   ├── data_splits.py             # IID, Dirichlet, Pathological splits
│   └── analyze_results.py         # Result analysis and plotting
├── configs/
│   ├── mnist_iid.yaml             # MNIST IID experiment
│   ├── mnist_non_iid.yaml         # MNIST Non-IID experiment
│   ├── cifar10_iid.yaml           # CIFAR-10 IID experiment
│   └── mnist_byzantine.yaml       # Byzantine attack experiment
├── runner.py                      # Main experiment executor
└── README.md                      # This file
```

---

## 🚀 Quick Start

### 1. Validate Framework (Optional but Recommended)
```bash
cd /srv/luminous-dynamics/Mycelix-Core/0TML
nix develop

# Quick validation (completes in <30 seconds)
python -u test_minimal.py
```

### 2. Run Small Test First
```bash
# Test with 10 rounds and 3 baselines (50-100 minutes on CPU)
python -u experiments/runner.py --config experiments/configs/mnist_test.yaml
```

### 3. Run Full Experiment
```bash
# Full 100-round experiment (8-16 hours on CPU)
python -u experiments/runner.py --config experiments/configs/mnist_iid.yaml
```

### 4. Analyze Results
```bash
python experiments/utils/analyze_results.py \
  --results results/test/mnist_test_20251003_140530.json \
  --output results/analysis/mnist_test/
```

**Note**: Use `-u` flag for unbuffered output to see progress in real-time.

---

## 📊 Available Experiments

### MNIST IID
- **Config**: `configs/mnist_iid.yaml`
- **Purpose**: Compare all 7 baselines on IID data
- **Duration**: ~1 hour (100 rounds)
- **Expected**: All reach ~99% accuracy

### MNIST Non-IID
- **Config**: `configs/mnist_non_iid.yaml`
- **Purpose**: Test non-IID handling (Dirichlet α=0.1)
- **Duration**: ~2 hours (200 rounds)
- **Expected**: SCAFFOLD > FedProx > FedAvg

### CIFAR-10 IID
- **Config**: `configs/cifar10_iid.yaml`
- **Purpose**: More complex dataset baseline
- **Duration**: ~3 hours (200 rounds)
- **Expected**: FedAvg ~85%

### MNIST Byzantine
- **Config**: `configs/mnist_byzantine.yaml`
- **Purpose**: Test Byzantine defenses (2 attackers)
- **Duration**: ~1 hour (100 rounds)
- **Expected**: Robust methods work, FedAvg fails

---

## 🛠️ Creating Custom Experiments

### Step 1: Create Config File

```yaml
# experiments/configs/my_experiment.yaml

experiment_name: "my_custom_experiment"
output_dir: "results/custom"

# Dataset
dataset:
  name: "mnist"  # or "cifar10"
  data_dir: "datasets"

# Model
model:
  architecture: "simple_cnn"  # or "resnet9", "char_lstm"
  params:
    num_classes: 10

# Data split
data_split:
  type: "dirichlet"  # or "iid", "pathological"
  alpha: 0.5  # For dirichlet only
  seed: 42

# Federated learning
federated:
  num_clients: 10
  batch_size: 32
  learning_rate: 0.01
  local_epochs: 1
  fraction_clients: 1.0

# Training
training:
  num_rounds: 100
  eval_every: 10

# Baselines to run
baselines:
  - fedavg
  - fedprox
```

### Step 2: Run Experiment
```bash
python experiments/runner.py --config experiments/configs/my_experiment.yaml
```

---

## 📈 Analysis Tools

### Generate Full Report
```bash
python experiments/utils/analyze_results.py \
  --results results/my_experiment_20251003.json \
  --output results/analysis/my_experiment/
```

**Generates**:
- `comparison_table.csv` - Performance metrics
- `convergence_accuracy.png` - Training curves (accuracy)
- `convergence_loss.png` - Training curves (loss)
- `comparison_accuracy.png` - Bar chart (accuracy)
- `comparison_loss.png` - Bar chart (loss)

### Compare Multiple Experiments
```python
from experiments.utils.analyze_results import compare_experiments

compare_experiments(
    [
        'results/iid/mnist_iid.json',
        'results/non_iid/mnist_non_iid_alpha01.json',
        'results/non_iid/mnist_non_iid_alpha05.json'
    ],
    save_path='results/analysis/iid_vs_non_iid.png'
)
```

---

## 🎯 Configuration Parameters

### Dataset Options
- **name**: `"mnist"` | `"cifar10"`
- **data_dir**: Path to datasets (default: `"datasets"`)

### Model Options
- **architecture**: `"simple_cnn"` | `"resnet9"` | `"char_lstm"`
- **params**: Model-specific parameters

### Data Split Options
**IID**:
```yaml
data_split:
  type: "iid"
  seed: 42
```

**Dirichlet (Non-IID)**:
```yaml
data_split:
  type: "dirichlet"
  alpha: 0.1  # Lower = more non-IID
  seed: 42
```

**Pathological (Extreme Non-IID)**:
```yaml
data_split:
  type: "pathological"
  shards_per_client: 2  # Classes per client
  seed: 42
```

### Byzantine Attack Options
```yaml
federated:
  num_byzantine: 2
  byzantine_clients: [5, 7]  # Specific client IDs
```

### Baseline Options
Available baselines:
- `fedavg` - Standard federated averaging
- `fedprox` - Proximal term for non-IID
- `scaffold` - Control variates for client drift
- `krum` - Byzantine-robust single selection
- `multikrum` - Byzantine-robust multi-selection
- `bulyan` - Strongest Byzantine defense
- `median` - Simple Byzantine defense

---

## 📚 Code Structure

### Models (`models/cnn_models.py`)
```python
from experiments.models.cnn_models import create_model

model = create_model('simple_cnn', num_classes=10)
model = create_model('resnet9')
model = create_model('char_lstm', vocab_size=80)
```

### Data Splits (`utils/data_splits.py`)
```python
from experiments.utils.data_splits import (
    create_iid_split,
    create_dirichlet_split,
    create_pathological_split
)

# IID split
client_indices = create_iid_split(dataset, num_clients=10)

# Dirichlet split (non-IID)
client_indices = create_dirichlet_split(
    dataset, num_clients=10, alpha=0.1
)

# Pathological split (extreme non-IID)
client_indices = create_pathological_split(
    dataset, num_clients=10, shards_per_client=2
)
```

### Experiment Runner (`runner.py`)
```python
from experiments.runner import ExperimentRunner

runner = ExperimentRunner('experiments/configs/mnist_iid.yaml')
runner.run()
```

### Result Analysis (`utils/analyze_results.py`)
```python
from experiments.utils.analyze_results import ResultsAnalyzer

analyzer = ResultsAnalyzer('results/mnist_iid_20251003.json')
analyzer.generate_report('results/analysis/')
```

---

## 🔍 Troubleshooting

### Import Errors
Make sure you're in the nix develop environment:
```bash
nix develop
```

### Dataset Not Found
Datasets should be in `datasets/` directory. Download them:
```bash
python scripts/download_datasets.py
```

### Out of Memory
Reduce batch size in config:
```yaml
federated:
  batch_size: 16  # Instead of 32
```

### Slow Training
**Expected CPU Performance** (PyTorch 2.8.0):
- **Minimal test** (100 samples): <30 seconds
- **MNIST test** (10 rounds, 5 clients): 50-100 minutes
- **MNIST IID** (100 rounds): 8-16 hours
- **CIFAR-10 IID** (200 rounds): 33-50 hours

**Performance Tips**:
- Reduce number of rounds (e.g., 10 instead of 100)
- Use fewer clients (e.g., 5 instead of 10)
- Use smaller model (SimpleCNN < ResNet9)
- Run experiments overnight for full benchmarks
- Consider GPU if available (10-50x speedup)

---

## 📖 References

### Papers
- **FedAvg**: McMahan et al., AISTATS 2017
- **FedProx**: Li et al., MLSys 2020
- **SCAFFOLD**: Karimireddy et al., ICML 2020
- **Krum**: Blanchard et al., NeurIPS 2017
- **Bulyan**: El Mhamdi et al., ICML 2018
- **Median**: Yin et al., ICML 2018

### Documentation
- Main planning: `../PHASE_11_PLANNING.md`
- Progress reports: `../PHASE_11_DAY_*_PROGRESS.md`
- Baseline comparison: `../baselines/BASELINE_COMPARISON.md`

---

## 💡 Tips

### Best Practices
1. **Start Small**: Test with 10 rounds before full experiment
2. **Use Seeds**: Ensure reproducibility
3. **Save Configs**: Version control your YAML files
4. **Monitor Progress**: Watch training output
5. **Analyze Results**: Use automated tools

### Common Workflows
**Quick Test**:
```yaml
training:
  num_rounds: 10  # Quick test
  eval_every: 2
```

**Full Benchmark**:
```yaml
training:
  num_rounds: 100  # Full run
  eval_every: 10
```

**Parameter Sweep**:
Create multiple configs with different alpha values:
- `mnist_non_iid_alpha01.yaml`
- `mnist_non_iid_alpha05.yaml`
- `mnist_non_iid_alpha10.yaml`

---

## 🎯 Next Steps

After running experiments:
1. Analyze results with `analyze_results.py`
2. Compare different configurations
3. Generate tables for paper
4. Create publication-quality plots
5. Document findings

---

**Status**: Ready for experiments! 🚀

**Last Updated**: October 3, 2025

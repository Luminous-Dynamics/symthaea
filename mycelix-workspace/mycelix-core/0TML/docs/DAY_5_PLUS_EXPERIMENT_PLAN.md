# Day 5+ Experiment Plan: Comprehensive FL Evaluation

**Status**: Planning phase
**Prerequisites**: ✅ Day 4 GPU enablement complete
**Timeline**: Days 5-10 (1 week)

## Overview

This document outlines the comprehensive experiment plan following successful GPU acceleration and IID baseline validation. The goal is to systematically evaluate all baselines across diverse scenarios to validate the framework's robustness and identify optimal algorithms for different deployment contexts.

## Experiment Categories

### 1. Non-IID Data Experiments 🎯 **Priority: HIGH**

**Objective**: Evaluate baseline performance under realistic non-IID data distributions

**Motivation**: Real-world federated learning rarely has IID data. Clients have different:
- User preferences (personalization)
- Geographic locations (regional bias)
- Device types (sensor variations)
- Time zones (temporal patterns)

#### Experiment 1.1: Dirichlet Distribution (Label Skew)

**Config File**: `experiments/configs/mnist_non_iid_dirichlet.yaml`

```yaml
experiment_name: "mnist_non_iid_dirichlet"
output_dir: "results/non_iid"

dataset:
  name: "mnist"
  data_dir: "datasets"

model:
  architecture: "simple_cnn"
  params:
    num_classes: 10

data_split:
  type: "dirichlet"
  alpha: 0.5  # Lower = more non-IID
  seed: 42

federated:
  num_clients: 10
  batch_size: 32
  learning_rate: 0.01
  local_epochs: 1
  fraction_clients: 1.0

training:
  num_rounds: 150  # Non-IID needs more rounds
  eval_every: 10

baselines:
  - fedavg     # Baseline
  - fedprox    # Proximal term helps with non-IID
  - scaffold   # Control variates help with drift
  - krum       # Byzantine-robust
  - multikrum  # Multi-gradient averaging
```

**Variations to Test**:
- alpha = 0.1 (highly non-IID)
- alpha = 0.5 (moderately non-IID)
- alpha = 1.0 (mildly non-IID)
- alpha = 10.0 (almost IID)

**Expected Results**:
- FedAvg: Performance degrades significantly
- FedProx: 5-10% better than FedAvg
- SCAFFOLD: Best convergence speed (3-4x fewer rounds)
- Krum/Multi-Krum: Slower but stable

**Key Metrics**:
- Convergence speed (rounds to 95% accuracy)
- Final test accuracy
- Client accuracy variance
- Communication cost

#### Experiment 1.2: Pathological Distribution (Extreme Label Skew)

**Config**: Each client gets only 2 classes
```yaml
data_split:
  type: "pathological"
  shards_per_client: 2  # Each client gets 2 digit classes
  seed: 42
```

**Expected Challenge**: This is the most extreme non-IID scenario
- FedAvg may fail to converge
- SCAFFOLD should shine here

### 2. Byzantine Attack Experiments 🛡️ **Priority: MEDIUM**

**Objective**: Validate Byzantine-robust baselines under adversarial conditions

**Motivation**: Federated learning is vulnerable to malicious clients sending poisoned gradients. Byzantine-robust algorithms are critical for production deployment.

#### Experiment 2.1: Random Noise Attack

**Config File**: `experiments/configs/mnist_byzantine_noise.yaml`

```yaml
experiment_name: "mnist_byzantine_random_noise"
output_dir: "results/byzantine"

# ... (same dataset/model config as IID)

data_split:
  type: "iid"  # Start with IID to isolate Byzantine effect
  seed: 42

federated:
  num_clients: 10
  batch_size: 32
  learning_rate: 0.01
  local_epochs: 1
  fraction_clients: 1.0

byzantine:
  num_byzantine: 2  # 20% attackers
  attack_type: "random_noise"  # Large random gradients
  attack_params:
    noise_scale: 10.0

training:
  num_rounds: 100
  eval_every: 10

baselines:
  - fedavg      # Baseline (should fail)
  - krum        # Single gradient selection
  - multikrum   # Top-m averaging
  - bulyan      # Median-based
  - median      # Simplest robust method
```

**Attack Variations**:
- Random noise (Gaussian with large variance)
- Sign flipping (negate all gradients)
- Label flipping (train on flipped labels)
- Backdoor attack (inject trigger patterns)

**Expected Results**:
- FedAvg: Accuracy drops to ~70-80%
- Krum: Maintains ~95% accuracy
- Multi-Krum: ~96% accuracy (more stable)
- Bulyan/Median: ~94-96% accuracy

**Key Metrics**:
- Test accuracy under attack
- Attack success rate
- Detection rate
- Convergence stability

#### Experiment 2.2: Adaptive Attack (Intelligent Adversary)

**Advanced**: Attackers try to mimic honest clients
```yaml
byzantine:
  num_byzantine: 3  # 30% attackers
  attack_type: "adaptive"  # Try to evade detection
  attack_params:
    perturbation_scale: 0.1  # Subtle poisoning
```

**Expected Challenge**: Tests limits of Byzantine defenses

### 3. Privacy-Preserving FL Experiments 🔒 **Priority: LOW-MEDIUM**

**Objective**: Evaluate accuracy/privacy tradeoffs with differential privacy

**Motivation**: Production FL systems need privacy guarantees. Differential privacy (DP) provides formal guarantees but reduces accuracy.

#### Experiment 3.1: Differential Privacy with DP-SGD

**Config File**: `experiments/configs/mnist_privacy_dp.yaml`

```yaml
experiment_name: "mnist_privacy_dp"
output_dir: "results/privacy"

# ... (same dataset/model config)

federated:
  num_clients: 10
  batch_size: 32
  learning_rate: 0.01
  local_epochs: 1
  fraction_clients: 0.5  # Sampling for privacy

privacy:
  mechanism: "gaussian"  # DP mechanism
  noise_multiplier: 1.0  # Privacy parameter
  max_grad_norm: 1.0     # Gradient clipping
  target_epsilon: 10.0   # Privacy budget
  target_delta: 1e-5

training:
  num_rounds: 150  # DP needs more rounds
  eval_every: 10

baselines:
  - fedavg   # DP-SGD compatible
  - scaffold # DP-SCAFFOLD
```

**Privacy Variations**:
- epsilon = 1.0 (strong privacy, low accuracy)
- epsilon = 10.0 (weak privacy, high accuracy)
- epsilon = infinity (no DP, baseline)

**Expected Results**:
- epsilon=1.0: ~85% accuracy (privacy-first)
- epsilon=10.0: ~95% accuracy (balanced)
- No DP: ~99% accuracy (accuracy-first)

**Key Metrics**:
- Test accuracy vs epsilon
- Privacy budget consumption
- Convergence speed
- Utility/privacy tradeoff curve

### 4. Real-World Dataset Experiments 📊 **Priority: MEDIUM-HIGH**

**Objective**: Validate baselines on non-MNIST datasets

**Motivation**: MNIST is toy data. Real applications need image, text, time-series support.

#### Experiment 4.1: CIFAR-10 (Color Images)

**Config File**: `experiments/configs/cifar10_iid.yaml`

```yaml
experiment_name: "cifar10_iid"
output_dir: "results/cifar10"

dataset:
  name: "cifar10"  # 32x32 color images, 10 classes
  data_dir: "datasets"

model:
  architecture: "resnet18"  # Deeper model needed
  params:
    num_classes: 10
    pretrained: false

federated:
  num_clients: 20  # More clients for 50K training samples
  batch_size: 64
  learning_rate: 0.001  # Lower LR for ResNet
  local_epochs: 2
  fraction_clients: 0.5

training:
  num_rounds: 200  # CIFAR-10 needs more rounds
  eval_every: 10

baselines:
  - fedavg
  - fedprox
  - scaffold
```

**Expected Challenge**: Much harder than MNIST
- Requires deeper models (ResNet, MobileNet)
- Longer training time (~2-3 hours per baseline on GPU)
- Final accuracy: ~75-85% (vs 99% on MNIST)

#### Experiment 4.2: FEMNIST (Federated MNIST)

**Realistic FL Dataset**: Handwriting from 3,500 real users

```yaml
dataset:
  name: "femnist"  # Naturally partitioned by writer
  data_dir: "datasets/femnist"

data_split:
  type: "natural"  # Use existing user partitioning

federated:
  num_clients: 100  # Simulate 100 writers
  batch_size: 16   # Writers have 100-1000 samples each
  fraction_clients: 0.1  # 10 writers per round
```

**Realistic Characteristics**:
- Naturally non-IID (each writer has unique style)
- Highly unbalanced (writers have 100-1000 samples)
- Large-scale (3,500+ clients available)

### 5. Scalability Experiments ⚡ **Priority: LOW**

**Objective**: Stress-test framework with many clients

#### Experiment 5.1: 100 Clients

```yaml
federated:
  num_clients: 100
  fraction_clients: 0.1  # 10 clients per round

training:
  num_rounds: 500
```

**Questions to Answer**:
- Does performance scale linearly?
- Memory usage with 100 clients?
- Communication overhead?
- Convergence speed vs IID/non-IID?

#### Experiment 5.2: GPU Multi-Processing

**Goal**: Train multiple clients in parallel on GPU

```yaml
execution:
  parallel_clients: 5  # Train 5 clients simultaneously
  gpu_allocation: "shared"  # Share GPU memory
```

**Expected Speedup**: 3-4x (limited by GPU memory)

## Experiment Schedule

### Week 1 (Days 5-7)
- **Day 5**: Non-IID experiments (Dirichlet α=0.1, 0.5, 1.0)
- **Day 6**: Non-IID experiments (Pathological sharding)
- **Day 7**: Analysis and comparison of non-IID results

### Week 2 (Days 8-10)
- **Day 8**: Byzantine attack experiments (Random noise, sign flip)
- **Day 9**: CIFAR-10 baseline experiments
- **Day 10**: Comprehensive analysis and write-up

## Implementation Checklist

### Before Starting

- [ ] Verify GPU experiments from Day 4 completed successfully
- [ ] Analyze Day 4 results thoroughly
- [ ] Ensure all baselines work correctly
- [ ] Clean up any GPU device placement issues

### For Each New Experiment

- [ ] Create YAML config file
- [ ] Verify config loads correctly: `python experiments/runner.py --config <config> --dry-run`
- [ ] Run small test (10 rounds) to catch issues early
- [ ] Launch full experiment in background
- [ ] Monitor progress with `./monitor_experiment.sh`
- [ ] Run analysis immediately after completion
- [ ] Document results and observations

### New Features Needed

#### 1. Differential Privacy Support
**Status**: Not implemented yet
**Priority**: Medium
**Files to Create**:
- `baselines/dp_fedavg.py` - DP-SGD wrapper
- `baselines/dp_scaffold.py` - DP-SCAFFOLD variant
- `experiments/utils/privacy.py` - Privacy accounting

**Implementation Notes**:
```python
# Add Gaussian noise to gradients
def add_dp_noise(gradients, noise_multiplier, max_grad_norm):
    # Clip gradients
    clipped_grads = clip_gradients(gradients, max_grad_norm)

    # Add Gaussian noise
    noise = torch.randn_like(clipped_grads) * noise_multiplier * max_grad_norm
    return clipped_grads + noise
```

#### 2. Byzantine Attack Simulation
**Status**: Partially implemented (Client.is_byzantine exists)
**Priority**: High
**Enhancements Needed**:
- Sign flip attack
- Label flip attack
- Backdoor attack
- Adaptive attack

**Files to Modify**:
- All baseline Client classes (add attack methods)
- `experiments/runner.py` (parse byzantine config)

#### 3. Additional Datasets
**Status**: Only MNIST implemented
**Priority**: High
**Files to Create**:
- `experiments/datasets/cifar10.py`
- `experiments/datasets/femnist.py`
- `experiments/datasets/shakespeare.py` (for NLP)

#### 4. Advanced Models
**Status**: Only SimpleCNN implemented
**Priority**: Medium
**Files to Create**:
- `experiments/models/resnet.py` - ResNet18/34/50
- `experiments/models/mobilenet.py` - Efficient mobile models
- `experiments/models/lstm.py` - For text/time-series

## Expected Outputs

### Plots and Visualizations

For each experiment category, generate:

1. **Convergence Plots**
   - Test accuracy vs rounds (all baselines overlaid)
   - Train loss vs rounds
   - With error bars (stddev across clients)

2. **Comparison Tables**
   - Final test accuracy
   - Rounds to 95% accuracy
   - Communication cost
   - Training time
   - GPU memory usage

3. **Distribution Analysis** (for non-IID)
   - Client data distribution heatmaps
   - Class imbalance per client
   - Statistical heterogeneity metrics (KL divergence, Wasserstein distance)

4. **Attack Success Analysis** (for Byzantine)
   - Accuracy degradation under attack
   - Detection rate vs false positive rate (ROC curves)
   - Attack success rate vs number of attackers

5. **Privacy-Utility Tradeoff** (for DP)
   - Accuracy vs epsilon curves
   - Privacy budget consumption over time
   - Noise scale vs convergence speed

### Research Artifacts

1. **Comprehensive Report** (`EXPERIMENT_RESULTS.md`)
   - Executive summary
   - Detailed results for each category
   - Key findings and insights
   - Recommendations for deployment

2. **Publication Draft** (`PAPER_DRAFT.md`)
   - Abstract
   - Introduction
   - Methodology
   - Results
   - Discussion
   - Conclusion

3. **Deployment Guide** (`DEPLOYMENT_GUIDE.md`)
   - Which baseline for which scenario?
   - Hyperparameter recommendations
   - Performance benchmarks
   - Best practices

## Success Criteria

**Minimum Viable Evaluation**:
- [ ] ✅ IID baseline comparison (Day 4)
- [ ] Non-IID experiments (α=0.1, 0.5, 1.0)
- [ ] Byzantine experiments (2/10 attackers, 3 attack types)
- [ ] At least one real-world dataset (CIFAR-10 or FEMNIST)

**Comprehensive Evaluation**:
- [ ] All non-IID scenarios
- [ ] All Byzantine attack types
- [ ] Differential privacy experiments
- [ ] Multiple real-world datasets
- [ ] Scalability tests (100+ clients)
- [ ] Published results and artifacts

## Research Questions

Each experiment should answer specific questions:

### Non-IID Experiments
1. How does heterogeneity level (α) affect convergence speed?
2. Which baseline is most robust to non-IID data?
3. Is SCAFFOLD worth the complexity vs FedProx?
4. What's the minimum α where FedAvg still works?

### Byzantine Experiments
1. Can Krum/Multi-Krum really detect 20% attackers?
2. What's the maximum attacker fraction each baseline can tolerate?
3. Are multi-gradient methods (Multi-Krum, Bulyan) worth the overhead?
4. Can adaptive attacks evade detection?

### Privacy Experiments
1. What's the accuracy cost of ε=1.0 vs ε=10.0?
2. How many more rounds does DP-SGD need?
3. Can SCAFFOLD reduce privacy cost?
4. Is there a sweet spot for privacy-utility tradeoff?

### Real-World Experiments
1. Do insights from MNIST transfer to CIFAR-10?
2. How much does natural non-IID (FEMNIST) differ from synthetic?
3. What model architecture works best for FL?
4. Can we achieve competitive accuracy vs centralized training?

## Next Actions

**Immediate (Today)**:
1. ✅ Wait for Day 4 experiments to complete
2. ✅ Run analysis on IID results
3. ⏳ Review results and validate baselines

**Tomorrow (Day 5)**:
1. Create non-IID configs (Dirichlet α=0.1, 0.5, 1.0)
2. Launch non-IID experiments (3 configs × 7 baselines = 21 experiments)
3. Estimate: ~12 hours total runtime on GPU

**This Week**:
1. Complete non-IID and Byzantine experiments
2. Analyze all results comprehensively
3. Draft initial findings document

**Next Week**:
1. Add new datasets (CIFAR-10)
2. Implement DP if time permits
3. Write comprehensive report

## Resources and References

### Papers to Review
1. **SCAFFOLD**: Karimireddy et al., ICML 2020
2. **FedProx**: Li et al., MLSys 2020
3. **Krum**: Blanchard et al., NeurIPS 2017
4. **Multi-Krum**: Blanchard et al., NeurIPS 2017
5. **DP-FedAvg**: McMahan et al., AISTATS 2018

### Datasets
- **MNIST**: http://yann.lecun.com/exdb/mnist/
- **CIFAR-10**: https://www.cs.toronto.edu/~kriz/cifar.html
- **FEMNIST**: https://leaf.cmu.edu/

### Baselines to Benchmark Against
- **LEAF Benchmark**: https://leaf.cmu.edu/
- **FedML**: https://fedml.ai/
- **Flower**: https://flower.dev/

## Conclusion

This comprehensive experiment plan provides a systematic path from toy datasets (MNIST) to realistic scenarios (non-IID, Byzantine, privacy, real-world data). Each experiment builds on previous insights and validates the framework's robustness.

**Timeline**: 1-2 weeks
**Cost**: ~30-40 hours of GPU compute
**Output**: Publication-ready results and deployment guide

**Next Step**: Complete Day 4 analysis, then proceed with non-IID experiments (Day 5).

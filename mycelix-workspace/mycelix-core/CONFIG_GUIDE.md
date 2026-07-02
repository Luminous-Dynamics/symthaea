# Mycelix Configuration Guide

This document provides comprehensive documentation for all configuration options in the Mycelix Byzantine-resistant Federated Learning system.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Critical Parameters](#critical-parameters)
3. [Configuration Sources](#configuration-sources)
4. [Parameter Reference](#parameter-reference)
   - [Byzantine Detection](#byzantine-detection)
   - [Federated Learning](#federated-learning)
   - [Holochain Integration](#holochain-integration)
   - [Security](#security)
   - [Privacy](#privacy)
   - [Monitoring](#monitoring)
5. [Dataset-Specific Configuration](#dataset-specific-configuration)
6. [Common Mistakes](#common-mistakes)
7. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Minimum Required Setup

```bash
# 1. Always source the optimal environment file FIRST
source .env.optimal

# 2. Run your training/testing
poetry run python tests/test_30_bft_validation.py
```

### Using the Configuration Validator

```python
from config import load_and_validate_config

# Load with validation (fails fast on critical errors)
config = load_and_validate_config()

# Access parameters
cos_min = config.byzantine_detection.label_skew_cos_min
aggregation = config.federated_learning.aggregation

# With dataset-specific optimal parameters
config = load_and_validate_config(dataset="cifar10")
```

### CLI Validation

```bash
# Validate current configuration
python -m config.validator --summary

# Validate with specific dataset parameters
python -m config.validator --dataset cifar10 --summary

# Export as environment variables
python -m config.validator --export-env
```

---

## Critical Parameters

### The Most Important Parameters

These parameters have the highest impact on system performance. Incorrect values can cause **massive performance degradation**.

| Parameter | Optimal Value | Dangerous Values | Impact |
|-----------|--------------|------------------|--------|
| `LABEL_SKEW_COS_MIN` | **-0.5** | -0.3, -0.4 | **16x worse FP rate** (57-92% vs 3.55-7.1%) |
| `BEHAVIOR_RECOVERY_THRESHOLD` | **2** | 3, 4, 5 | Slower honest node recovery |
| `BEHAVIOR_RECOVERY_BONUS` | **0.12** | 0.08, 0.10 | Permanent honest node exclusion |

### Why LABEL_SKEW_COS_MIN = -0.5 is Critical

The cosine similarity threshold determines whether a node's gradient is considered "suspicious" compared to the aggregate. The difference between -0.5 and -0.3 is critical:

```
LABEL_SKEW_COS_MIN = -0.5 --> 3.55-7.1% False Positive Rate   [Correct]
LABEL_SKEW_COS_MIN = -0.3 --> 57-92% False Positive Rate      [16x WORSE!]
```

With -0.3, honest nodes with naturally diverse data distributions (label skew) are incorrectly flagged as Byzantine, destroying system performance.

### Setting Critical Parameters

**Environment Variables (Recommended):**
```bash
export BEHAVIOR_RECOVERY_THRESHOLD=2
export BEHAVIOR_RECOVERY_BONUS=0.12
export LABEL_SKEW_COS_MIN=-0.5
export LABEL_SKEW_COS_MAX=0.95
```

**Python:**
```python
import os
os.environ["LABEL_SKEW_COS_MIN"] = "-0.5"
os.environ["BEHAVIOR_RECOVERY_THRESHOLD"] = "2"
os.environ["BEHAVIOR_RECOVERY_BONUS"] = "0.12"
```

---

## Configuration Sources

Configuration is loaded from multiple sources in order of priority (later sources override earlier):

1. **Default values** (built into schema)
2. **Dataset-specific optimal parameters** (if dataset specified)
3. **Environment file** (`.env.optimal`)
4. **Config file** (`production_config.yaml`)
5. **Current environment variables** (highest priority)

### File Locations

| File | Purpose |
|------|---------|
| `.env.optimal` | Critical BFT parameters - **ALWAYS SOURCE THIS** |
| `.env.example` | Template for new deployments |
| `.env.adaptive` | Dataset-adaptive parameters |
| `production_config.yaml` | Full FL configuration |
| `config/node.yaml` (0TML) | Node-specific settings |
| `conductor-config.yaml` | Holochain conductor settings |

---

## Parameter Reference

### Byzantine Detection

The core parameters controlling Byzantine fault tolerance.

#### Cosine Similarity Thresholds

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `label_skew_cos_min` | float | **-0.5** | [-1.0, 0.0] | Minimum cosine similarity. CRITICAL - see above. |
| `label_skew_cos_max` | float | 0.95 | [0.5, 1.0] | Maximum cosine similarity. Detects overly-aligned attacks. |

#### Reputation System

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `behavior_recovery_threshold` | int | **2** | [1, 10] | Rounds of good behavior before recovery begins |
| `behavior_recovery_bonus` | float | **0.12** | [0.01, 0.5] | Reputation bonus per good round |
| `reputation_floor` | float | 0.01 | [0.0, 0.5] | Minimum reputation (prevents permanent exclusion) |
| `reputation_decay` | float | 0.95 | [0.5, 1.0] | Decay factor applied each round |
| `reputation_initial` | float | 0.5 | [0.0, 1.0] | Starting reputation for new nodes |
| `byzantine_penalty` | float | -0.2 | [-1.0, 0.0] | Reputation penalty for detected Byzantine behavior |

#### Proof of Gradient Quality (PoGQ)

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `pogq_threshold` | float | 0.7 | [0.0, 1.0] | Minimum PoGQ score for acceptance |
| `pogq_test_size` | float | 0.1 | [0.01, 0.5] | Fraction of data for validation |

#### Hybrid Detection

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `hybrid_suspicion_threshold` | float | 0.35 | Combined score threshold for Byzantine classification |
| `hybrid_pogq_weight` | float | 0.35 | Weight for PoGQ component |
| `hybrid_cos_weight` | float | 0.30 | Weight for cosine similarity |
| `hybrid_committee_weight` | float | 0.15 | Weight for committee voting |
| `hybrid_norm_weight` | float | 0.10 | Weight for gradient norm analysis |
| `hybrid_cluster_weight` | float | 0.10 | Weight for cluster analysis |

**Note:** Weights should sum to 1.0.

#### Attack Detection

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `zero_norm_threshold` | float | 1e-6 | Threshold for zero-gradient attack detection |
| `backdoor_spike_threshold` | float | 3.0 | Threshold for backdoor attack detection |
| `anomaly_sensitivity` | float | 2.5 | Standard deviations for anomaly detection |
| `anomaly_window_size` | int | 10 | Rounds to analyze for anomaly detection |
| `committee_reject_floor` | float | 0.25 | Minimum rejection rate to flag node (0.1 for label_skew) |

### Federated Learning

#### Aggregation

| Parameter | Type | Default | Options | Description |
|-----------|------|---------|---------|-------------|
| `aggregation` | string | "krum" | krum, median, trimmed_mean, fedavg, weighted | Gradient aggregation method |

**Security Note:** `fedavg` is NOT Byzantine-robust. Use only for baselines.

#### Training

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `num_rounds` | int | 100 | [1, 10000] | Total training rounds |
| `num_clients` | int | 10 | [2, 1000] | Total FL clients |
| `clients_per_round` | int | 5 | - | Clients per round |
| `min_clients` | int | 5 | [2, -] | Minimum clients for aggregation |
| `local_epochs` | int | 10 | [1, 100] | Local epochs per round |
| `learning_rate` | float | 0.05 | [0.0001, 1.0] | Client learning rate |
| `momentum` | float | 0.9 | [0.0, 0.99] | SGD momentum |
| `weight_decay` | float | 0.0001 | [0.0, 0.1] | L2 regularization |
| `gradient_clipping` | float | 10.0 | [0.1, 100.0] | Maximum gradient norm |

#### Data Distribution

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data_distribution.type` | string | "dirichlet" | iid, dirichlet, label_skew |
| `data_distribution.alpha` | float | 1.0 | Dirichlet alpha (lower = more heterogeneous) |

#### FedProx

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `fedprox.enabled` | bool | true | Enable proximal regularization |
| `fedprox.mu` | float | 0.01 | Proximal term weight |
| `fedprox.adaptive_mu` | bool | true | Adjust mu based on drift |
| `fedprox.mu_max` | float | 1.0 | Maximum mu value |
| `fedprox.mu_min` | float | 0.001 | Minimum mu value |

### Communication Efficiency

#### Quantization (4x compression)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `communication.quantization.enabled` | bool | true | Enable quantization |
| `communication.quantization.bits` | int | 8 | Bit depth (4, 8, 16) |
| `communication.quantization.dynamic` | bool | true | Dynamic range per layer |

#### Sparsification (10x compression)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `communication.sparsification.enabled` | bool | true | Enable sparsification |
| `communication.sparsification.sparsity` | float | 0.9 | Fraction to drop (0.9 = keep 10%) |
| `communication.sparsification.strategy` | string | "topk" | topk, random, threshold |
| `communication.sparsification.error_feedback` | bool | true | Accumulate dropped gradients |

**Combined:** With both enabled, expect ~40x compression.

### Holochain Integration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `holochain.enabled` | bool | false | Enable Holochain DHT |
| `holochain.conductor_url` | string | "ws://localhost:8888" | Conductor WebSocket URL |
| `holochain.admin_port` | int | 8888 | Admin interface port |
| `holochain.app_port` | int | 8889 | Application interface port |
| `holochain.dna_path` | string | "./holochain/dna/zerotrustml.dna" | DNA file path |
| `holochain.network.bootstrap_url` | string | "https://bootstrap-staging.holo.host" | Bootstrap server |
| `holochain.network.signal_url` | string | "wss://signal.holo.host" | WebRTC signaling server |

### Security

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `security.require_signatures` | bool | true | Require cryptographic signatures |
| `security.signature_algorithm` | string | "ed25519" | ed25519, secp256k1 |
| `security.encrypt_gradients` | bool | true | Encrypt gradients in transit |
| `security.encryption_algorithm` | string | "chacha20-poly1305" | Encryption algorithm |
| `security.rate_limiting.max_requests_per_minute` | int | 60 | Rate limit |
| `security.tls_enabled` | bool | false | Enable TLS |

**Warning:** In production mode, disabling `require_signatures` is a critical error.

### Privacy

#### Differential Privacy

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `privacy.differential_privacy.enabled` | bool | false | Enable DP |
| `privacy.differential_privacy.epsilon` | float | 1.0 | Privacy budget (lower = more private) |
| `privacy.differential_privacy.delta` | float | 1e-5 | Failure probability |

#### Zero-Knowledge Proof of Contribution

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `privacy.zkpoc.enabled` | bool | false | Enable ZK-PoC |
| `privacy.zkpoc.proof_generator` | string | "bulletproofs" | bulletproofs, zk-snarks, zk-starks |
| `privacy.zkpoc.hipaa_mode` | bool | false | Extra privacy for medical data |
| `privacy.zkpoc.gdpr_mode` | bool | false | Extra privacy for EU data |

### Monitoring

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `monitoring.prometheus_enabled` | bool | true | Enable Prometheus metrics |
| `monitoring.prometheus_port` | int | 9090 | Prometheus port |
| `monitoring.tensorboard` | bool | true | Enable TensorBoard |
| `monitoring.log_level` | string | "INFO" | DEBUG, INFO, WARNING, ERROR, CRITICAL |
| `monitoring.log_format` | string | "json" | json, text |
| `monitoring.checkpoint_interval` | int | 10 | Save model every N rounds |

---

## Dataset-Specific Configuration

Different datasets require different parameters due to varying gradient statistics.

### CIFAR-10 (High-dimensional CNN)
- Input: 32x32x3 RGB images
- Parameters: ~3K+
- Validated success rate: 77%

```bash
export BEHAVIOR_RECOVERY_THRESHOLD=2
export BEHAVIOR_RECOVERY_BONUS=0.12
export LABEL_SKEW_COS_MIN=-0.5
export LABEL_SKEW_COS_MAX=0.95
export COMMITTEE_REJECT_FLOOR=0.25
```

### EMNIST Balanced (Mid-dimensional CNN)
- Input: 28x28 grayscale
- Parameters: ~800+
- Validated success rate: 74%

```bash
export BEHAVIOR_RECOVERY_THRESHOLD=3
export BEHAVIOR_RECOVERY_BONUS=0.10
export LABEL_SKEW_COS_MIN=-0.4
export LABEL_SKEW_COS_MAX=0.96
export COMMITTEE_REJECT_FLOOR=0.30
```

### Breast Cancer (Low-dimensional Tabular)
- Input: 30 features
- Uses CIFAR-10 cosine thresholds (validated via parameter sweep)

```bash
export BEHAVIOR_RECOVERY_THRESHOLD=2
export BEHAVIOR_RECOVERY_BONUS=0.12
export LABEL_SKEW_COS_MIN=-0.5
export LABEL_SKEW_COS_MAX=0.95
export COMMITTEE_REJECT_FLOOR=0.25
```

### Using Dataset-Specific Parameters in Code

```python
from config import load_and_validate_config, get_dataset_optimal_params

# Method 1: Automatic loading
config = load_and_validate_config(dataset="cifar10")

# Method 2: Get parameters only
optimal = get_dataset_optimal_params("cifar10")
# Returns: {'behavior_recovery_threshold': 2, 'behavior_recovery_bonus': 0.12, ...}
```

---

## Common Mistakes

### Mistake 1: Wrong LABEL_SKEW_COS_MIN

```bash
# WRONG - causes 57-92% false positive rate
export LABEL_SKEW_COS_MIN=-0.3

# CORRECT - achieves 3.55-7.1% false positive rate
export LABEL_SKEW_COS_MIN=-0.5
```

### Mistake 2: Not Sourcing .env.optimal

```bash
# WRONG - runs with default (bad) parameters
poetry run python tests/test_30_bft_validation.py

# CORRECT - sources optimal parameters first
source .env.optimal
poetry run python tests/test_30_bft_validation.py
```

### Mistake 3: Using FedAvg in Production

```yaml
# WRONG - FedAvg is NOT Byzantine-robust
federated:
  aggregation: "fedavg"

# CORRECT - Use Byzantine-robust aggregation
federated:
  aggregation: "krum"  # or "median", "trimmed_mean"
```

### Mistake 4: Disabling Signatures in Production

```yaml
# WRONG - allows message spoofing
deployment:
  mode: "production"
security:
  require_signatures: false  # CRITICAL ERROR!

# CORRECT
security:
  require_signatures: true
```

### Mistake 5: Hybrid Weights Don't Sum to 1.0

```python
# WRONG - weights sum to 1.15
HYBRID_POGQ_WEIGHT = 0.35
HYBRID_COS_WEIGHT = 0.30
HYBRID_COMMITTEE_WEIGHT = 0.20  # Should be 0.15
HYBRID_NORM_WEIGHT = 0.15       # Should be 0.10
HYBRID_CLUSTER_WEIGHT = 0.15    # Should be 0.10

# CORRECT - weights sum to 1.0
HYBRID_POGQ_WEIGHT = 0.35
HYBRID_COS_WEIGHT = 0.30
HYBRID_COMMITTEE_WEIGHT = 0.15
HYBRID_NORM_WEIGHT = 0.10
HYBRID_CLUSTER_WEIGHT = 0.10
```

---

## Troubleshooting

### High False Positive Rate (>10%)

1. Check `LABEL_SKEW_COS_MIN`:
   ```bash
   echo $LABEL_SKEW_COS_MIN
   # Should be -0.5
   ```

2. Ensure you've sourced `.env.optimal`:
   ```bash
   source .env.optimal
   ```

3. Check hybrid weights sum:
   ```python
   from config import load_and_validate_config
   config = load_and_validate_config()
   bd = config.byzantine_detection
   total = bd.hybrid_pogq_weight + bd.hybrid_cos_weight + bd.hybrid_committee_weight + bd.hybrid_norm_weight + bd.hybrid_cluster_weight
   print(f"Weight sum: {total}")  # Should be 1.0
   ```

### Honest Nodes Permanently Excluded

Check reputation recovery parameters:
```bash
echo $BEHAVIOR_RECOVERY_THRESHOLD  # Should be 2
echo $BEHAVIOR_RECOVERY_BONUS      # Should be 0.12
echo $REPUTATION_FLOOR             # Should be 0.01
```

### Configuration Not Loading

Verify file paths:
```python
from config.validator import load_env_file, load_yaml_file
from pathlib import Path

# Check environment file
env = load_env_file(".env.optimal")
print(f"Loaded {len(env)} env vars")

# Check YAML file
yaml = load_yaml_file("production_config.yaml")
print(f"Loaded config sections: {list(yaml.keys())}")
```

### Validate Full Configuration

```bash
# Run validator with verbose output
python -m config.validator --summary --no-strict

# Check for specific dataset
python -m config.validator --dataset cifar10 --summary
```

---

## Environment Variable Reference

Quick reference for all environment variables:

```bash
# Byzantine Detection (CRITICAL)
LABEL_SKEW_COS_MIN=-0.5
LABEL_SKEW_COS_MAX=0.95
BEHAVIOR_RECOVERY_THRESHOLD=2
BEHAVIOR_RECOVERY_BONUS=0.12
REPUTATION_FLOOR=0.01
COMMITTEE_REJECT_FLOOR=0.25

# Hybrid Detection Weights
HYBRID_POGQ_WEIGHT=0.35
HYBRID_COS_WEIGHT=0.30
HYBRID_COMMITTEE_WEIGHT=0.15
HYBRID_NORM_WEIGHT=0.10
HYBRID_CLUSTER_WEIGHT=0.10
HYBRID_SUSPICION_THRESHOLD=0.35

# Attack Detection
POGQ_THRESHOLD=0.7
ZERO_NORM_THRESHOLD=1e-6
BACKDOOR_SPIKE_THRESHOLD=3.0
ANOMALY_SENSITIVITY=2.5

# System
SEED=42
DEVICE=cuda
NUM_WORKERS=4
LOG_LEVEL=INFO

# Database (ZeroTrustML)
ZEROTRUSTML_DB_HOST=postgres
ZEROTRUSTML_DB_NAME=zerotrustml
ZEROTRUSTML_DB_USER=zerotrustml
ZEROTRUSTML_DB_PASSWORD=<secret>

# Holochain
HOLOCHAIN_ADMIN_PORT=8888
HOLOCHAIN_APP_PORT=8889
```

---

## Related Files

| File | Location | Purpose |
|------|----------|---------|
| Schema | `config/schema.json` | JSON Schema for validation |
| Validator | `config/validator.py` | Python validation module |
| Optimal Params | `.env.optimal` | Critical BFT parameters |
| Adaptive Params | `0TML/.env.adaptive` | Dataset-adaptive parameters |
| Production Config | `production_config.yaml` | Full FL configuration |
| Node Config | `0TML/config/node.yaml` | Node-specific settings |

---

## Version History

- **1.0.0** (2026-01-08): Initial unified configuration system
  - Consolidated 5+ config sources into single validated schema
  - Added Python validation module with fail-fast behavior
  - Documented critical parameters and common mistakes
  - Created dataset-specific optimal parameter sets

---

## References

- `SESSION_STATUS_2025-10-28.md`: Original parameter optimization documentation
- `CI_CD_DATASET_STRATEGY.md`: CI/CD operationalization strategy
- `0TML/tests/test_30_bft_validation.py`: Test implementation
- `.env.adaptive`: Dataset-adaptive parameter research notes

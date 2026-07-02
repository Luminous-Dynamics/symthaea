# Attack & Defense Specification - Publication-Grade Canonical Presets

**Purpose**: Reproducible, canonical attack and defense configurations for Gen-4 evaluation
**Status**: Reference specification for all experiments
**Last Updated**: November 8, 2025

---

## 🎯 Attack Suite - Canonical Presets

All attacks use **exact parameter sets** for reproducibility across papers.

### Basic Gradient Manipulation

#### 1. Sign-Flip Attack
```python
canonical_sign_flip = {
    "name": "sign_flip",
    "lambda": -1.0,  # Full negation
    "sophistication": "low",
    "detectability": "very_high"
}
```

**Expected Behavior**: Gradient fully negated, easily detected by direction-based methods (FLTrust, PoGQ-v4)

#### 2. Gaussian Noise Attack
```python
canonical_gaussian = {
    "name": "gaussian",
    "sigma": 0.1,  # Standard deviation
    "sophistication": "low",
    "detectability": "very_high"
}
```

**Expected Behavior**: Zero-mean Gaussian noise, detected by magnitude checks

#### 3. Scaling Attack
```python
canonical_scaling = {
    "name": "scaling",
    "lambda": 10.0,  # Amplification factor
    "sophistication": "low",
    "detectability": "high"
}
```

**Expected Behavior**: Gradient amplified 10×, detected by magnitude and distance-based methods

### Targeted Poisoning

#### 4. Label-Flip Attack
```python
canonical_label_flip = {
    "name": "label_flip",
    "flip_rate": 0.3,  # 30% of labels flipped
    "mapping": {7: 1},  # 7→1 (FEMNIST: flip '7' to '1')
    "sophistication": "medium",
    "detectability": "medium"
}
```

**Expected Behavior**: Targeted class confusion, utility-based detection needed

#### 5. Backdoor (Triggered Patch)
```python
canonical_backdoor = {
    "name": "backdoor_patch",
    "trigger": "3x3_patch",  # 3×3 pixel patch trigger
    "trigger_position": "bottom_right",
    "target_class": 7,  # When trigger present → class 7
    "poison_rate": 0.1,  # 10% of training data poisoned
    "sophistication": "high",
    "detectability": "hard",
    "metrics": ["asr", "clean_acc"]  # Attack Success Rate + Clean Accuracy
}
```

**Expected Behavior**: High ASR, minimal clean accuracy degradation

**Evaluation Metrics**:
- **ASR (Attack Success Rate)**: % of triggered samples classified as target
- **Clean Accuracy**: Accuracy on non-triggered samples
- **Trade-off**: ASR ↓ while Clean Acc ↑ (defense success)

### Stateful/Adaptive Attacks

#### 6. Sleeper Agent Attack
```python
canonical_sleeper = {
    "name": "sleeper",
    "activation_round": 5,  # Honest for 5 rounds, then attack
    "honest_noise": 0.0,  # Perfect mimicry during honest phase
    "attack_mode": "sign_flip",  # What attack after activation
    "sophistication": "very_high",
    "detectability": "hard",
    "metrics": ["tpr", "fpr", "t2d"]  # Time-to-Detection critical
}
```

**Modes**: `sign_flip`, `noise_masked`, `scaling`, `adaptive_stealth`

**Expected Behavior**: Builds reputation, then strikes

**Evaluation Metrics**:
- **T2D (Time-to-Detection)**: Rounds until first detection after activation
- **Pre-activation FPR**: Should be ≈0 (perfect mimicry)
- **Post-activation TPR**: Ability to detect the switch

#### 7. Noise-Masked Attack
```python
canonical_noise_masked = {
    "name": "noise_masked",
    "sigma": 0.05,  # Masking noise std dev
    "poison_ratio": 0.3,  # 30% poison, 50% honest, 20% noise
    "mimic_target": "median",  # Try to match median gradient profile
    "sophistication": "high",
    "detectability": "medium"
}
```

**Expected Behavior**: Poison masked by noise to evade thresholds

### Coordinated Attacks

#### 8. Collusion Attack
```python
canonical_collusion = {
    "name": "collusion",
    "k": 4,  # 4 Byzantine nodes coordinate
    "near_duplicate": 0.99,  # Cosine similarity between colluders
    "sophistication": "high",
    "detectability": "medium"
}
```

**Expected Behavior**: Multiple nodes submit similar malicious gradients

#### 9. Sybil Attack
```python
canonical_sybil = {
    "name": "sybil",
    "k": 2,  # Single adversary, 2 identities
    "variation": 0.01,  # Tiny variation to appear different
    "sophistication": "high",
    "detectability": "medium"
}
```

**Expected Behavior**: Near-duplicate gradients from "different" clients

### Advanced Evasion

#### 10. Adaptive Stealth Attack
```python
canonical_adaptive = {
    "name": "adaptive_stealth",
    "epsilon": 0.01,  # Learning rate for threshold evasion
    "direction": "server_grad",  # Align with server gradient direction
    "rl_feedback": True,  # Adapt based on detection feedback
    "sophistication": "very_high",
    "detectability": "very_hard"
}
```

**Expected Behavior**: Learns to evade detection thresholds via RL

---

## 🛡️ Defense Registry - SOTA Baselines

All defenses implement the `Defense` protocol:

```python
class Defense(Protocol):
    name: str

    def aggregate(
        self,
        client_updates: List[np.ndarray],
        **context
    ) -> np.ndarray:
        """Aggregate client gradients with Byzantine filtering"""
        ...

    def explain(self) -> Dict[str, Any]:
        """Return per-client diagnostics (scores, weights, flags)"""
        ...
```

### Tier 1: Must-Have Baselines

#### 1. FLTrust (Direction-Based)
```python
fltrust_config = {
    "name": "fltrust",
    "server_validation_size": 256,  # Clean server validation set
    "cosine_threshold": 0.0,  # ReLU(cosine) filter
    "trust_score_method": "ts",  # Trust Score normalization
    "family": "direction_based"
}
```

**Strengths**: Excellent on sign-flip, scaling, direction attacks
**Weaknesses**: Requires clean server dataset

#### 2. RFA (Robust FedAvg / Geometric Median)
```python
rfa_config = {
    "name": "rfa",
    "method": "geometric_median",  # Weiszfeld algorithm
    "max_iters": 20,
    "tolerance": 1e-5,
    "family": "robust_location"
}
```

**Strengths**: Robust to outliers, no hyperparameters
**Weaknesses**: O(N²) pairwise distances

#### 3. Coordinate-wise Median
```python
median_config = {
    "name": "coord_median",
    "family": "coordinate_wise"
}
```

**Strengths**: Simple, robust, no hyperparameters
**Weaknesses**: Can fail under coordinated attacks

#### 4. Trimmed Mean
```python
trimmed_mean_config = {
    "name": "trimmed_mean",
    "trim_ratio": 0.1,  # Trim top/bottom 10%
    "family": "coordinate_wise"
}
```

**Strengths**: Balance of robustness and efficiency
**Weaknesses**: Sensitive to trim_ratio choice

#### 5. Krum
```python
krum_config = {
    "name": "krum",
    "f": None,  # Defaults to floor(bft_ratio * n)
    "family": "distance_based"
}
```

**Strengths**: Theoretically sound under f Byzantine assumption
**Weaknesses**: Discards most gradients (single selection)

#### 6. Multi-Krum
```python
multi_krum_config = {
    "name": "multi_krum",
    "f": None,  # Floor(bft_ratio * n)
    "m": 5,     # Average top-5 Krum selections
    "family": "distance_based"
}
```

**Strengths**: Better than Krum (averages multiple)
**Weaknesses**: Still discards many gradients

#### 7. Bulyan
```python
bulyan_config = {
    "name": "bulyan",
    "f": None,  # Floor(bft_ratio * n)
    "selection_size": "auto",  # n - 2f
    "family": "distance_based"
}
```

**Strengths**: Combines Krum + Trimmed Mean, robust
**Weaknesses**: High compute cost (Krum selection + trimming)

#### 8. FedGuard (Learned Filter)
```python
fedguard_config = {
    "name": "fedguard",
    "history_window": 5,  # Rounds of client history
    "anomaly_threshold": 0.8,  # Representation similarity threshold
    "magnitude_gate": 3.0,  # σ-based magnitude filter
    "family": "learned_filter"
}
```

**Strengths**: Adapts to client behavior patterns
**Weaknesses**: Requires warm-up period, more complex

### Tier 2: Nice-to-Have (Specialized)

#### 9. FoolsGold (Anti-Sybil)
```python
foolsgold_config = {
    "name": "foolsgold",
    "history_length": 10,  # Track client similarity history
    "uniqueness_power": 1.0,  # Weight scaling by uniqueness
    "family": "anti_sybil"
}
```

**Strengths**: Excellent against Sybil/collusion attacks
**Weaknesses**: Requires historical tracking

#### 10. BOBA (Label-Skew Aware)
```python
boba_config = {
    "name": "boba",
    "class_histograms": True,  # Requires per-client class distributions
    "reweight_by_class": True,
    "family": "label_skew_aware"
}
```

**Strengths**: Handles non-IID label skew gracefully
**Weaknesses**: Needs class histogram metadata

#### 11. FedAvg (Vanilla Baseline)
```python
fedavg_config = {
    "name": "fedavg",
    "family": "baseline"
}
```

**Purpose**: Sanity floor - should fail under all attacks

---

## 🔬 PoGQ-v4 Specification (Gen-4 Detector)

**Definition**: PoGQ-v4 = {Mondrian + Conformal + Hybrid(λ) + EMA + Direction-Prefilter}

```python
pogq_v4_config = {
    "name": "pogq_v4",

    # 1. Mondrian (Class-Aware Validation)
    "class_aware": True,
    "validate_on_client_classes": True,  # Only score on client's actual classes

    # 2. Conformal FPR Cap
    "conformal_threshold": True,
    "alpha": 0.10,  # FPR ≤ 10% guarantee
    "calibration_size": 256,  # Samples for quantile estimation

    # 3. Hybrid Score: λ·direction + (1-λ)·utility
    "hybrid_scoring": True,
    "lambda_direction": 0.7,  # 70% direction, 30% utility
    "direction_metric": "cosine",  # Cosine similarity to reference
    "utility_metric": "loss_reduction",  # Δloss on validation set

    # 4. Temporal EMA
    "temporal_ema": True,
    "ema_beta": 0.85,  # Historical weight

    # 5. Direction Prefilter
    "direction_prefilter": True,
    "min_cosine": 0.0,  # ReLU(cosine) - reject opposing directions

    # 6. Adaptive Threshold (Winsorized Dispersion)
    "adaptive_threshold": True,
    "dispersion_method": "biweight",  # Robust to outliers

    "family": "gen4_hybrid"
}
```

### Component Rationale

| Component | Purpose | Attack Coverage |
|-----------|---------|----------------|
| **Mondrian** | Reduce heterogeneity bleed | Label-flip, backdoor |
| **Conformal** | Hard FPR guarantee | All (prevents false accusations) |
| **Hybrid(λ)** | Direction + Utility fusion | Sign-flip, scaling, adaptive |
| **EMA** | Temporal stability | Sleeper, gradual attacks |
| **Direction-Prefilter** | Cheap wins on opposing grads | Sign-flip, scaling |
| **Winsorized** | Outlier-robust thresholding | Collusion, Sybil |

### Ablation Components (for Table)

Test removing each component to show contribution:

1. **PoGQ-v4-Full**: All 6 components
2. **PoGQ-v4-NoMondrian**: Remove class-aware validation
3. **PoGQ-v4-NoConformal**: Remove FPR cap
4. **PoGQ-v4-NoHybrid**: Use only utility (λ=0)
5. **PoGQ-v4-NoEMA**: Remove temporal smoothing
6. **PoGQ-v4-NoPrefilter**: Remove direction filter

**Evaluation**: FEMNIST @ 35% BFT, all 6 attacks, 1 seed

---

## 📊 Evaluation Matrix - Tighter Specification

### Primary Dataset: FEMNIST

**Focus**: Non-IID realistic federated learning scenario

```yaml
dataset: femnist
clients_total: 200
clients_per_round: 20
rounds: 10
non_iid_alpha: [0.1, 0.3, 0.5]  # Dirichlet concentration
server_validation_split: 256  # Clean samples for FLTrust/PoGQ-v4
```

### BFT Ratios (Byzantine Fault Tolerance)
```python
bft_ratios = [0.35, 0.50]  # 35% and 50% Byzantine clients
```

**Rationale**:
- **35%**: Challenging but realistic (near f=⅓ theoretical limit)
- **50%**: Stress test (detector inversion possible)

### Attacks (6 Canonical)
```python
attack_presets = [
    "sign_flip",
    "gaussian",
    "scaling",
    "label_flip",
    "backdoor_patch",
    "sleeper"
]
```

**Optionally add**: `noise_masked`, `collusion`, `sybil`, `adaptive_stealth` for appendix

### Detectors (8 Core + 1 Baseline)
```python
detectors = [
    "pogq_v4",        # Gen-4 (ours)
    "fltrust",        # Direction-based SOTA
    "rfa",            # Robust location
    "coord_median",   # Coordinate-wise
    "trimmed_mean",   # Coordinate-wise
    "multi_krum",     # Distance-based
    "bulyan",         # Distance + trimmed
    "fedguard",       # Learned filter
    "fedavg"          # Vanilla (sanity floor)
]
```

**Optional Tier 2**: `foolsgold` (for Sybil figure), `boba` (for label-skew appendix)

### Seeds (Reproducibility)
```python
seeds = [42, 123, 456]
```

### Full Matrix Size

**Primary Table (FEMNIST)**:
```
6 attacks × 8 detectors × 2 BFT ratios × 3 seeds × 3 α values
= 864 experiments
```

**Per-experiment**: ~10 rounds × 20 clients/round = ~200 evaluations per run

**Estimated Runtime**: ~25-30 hours on GPU (assuming 2 min/experiment)

**Storage**: ~864 JSON files × 2KB ≈ 1.7MB

---

## 📈 Metrics Schema - Canonical JSON

Every experiment produces **exactly one JSON file** with this structure:

```json
{
  "metadata": {
    "dataset": "femnist",
    "attack": "sign_flip",
    "attack_params": {"lambda": -1.0},
    "bft_ratio": 0.35,
    "seed": 42,
    "detector": "pogq_v4",
    "detector_config": {
      "class_aware": true,
      "alpha": 0.10,
      "lambda_direction": 0.7
    },
    "non_iid_alpha": 0.3,
    "rounds": 10,
    "clients_per_round": 20,
    "timestamp": "2025-11-08T14:30:00Z"
  },

  "metrics": {
    "detection": {
      "tpr": 0.91,
      "fpr": 0.07,
      "tnr": 0.93,
      "fnr": 0.09,
      "auroc": 0.92,
      "precision": 0.85,
      "f1": 0.88
    },
    "attack_specific": {
      "asr": null,
      "clean_acc": null,
      "t2d": null
    },
    "model_quality": {
      "final_test_acc": 0.82,
      "convergence_round": 8
    }
  },

  "calibration": {
    "alpha": 0.10,
    "n_calibration": 256,
    "empirical_fpr": 0.08,
    "threshold_value": 0.65
  },

  "timing": {
    "score_ms_per_client": 38,
    "detector_ms_per_round": 410,
    "total_experiment_sec": 124.5
  },

  "memory": {
    "peak_mb": 2048,
    "cache_mb": 512
  },

  "provenance": {
    "per_client_class_histograms": [...],
    "server_val_classes": [0,1,2,3,4,5,6,7,8,9],
    "server_val_disjoint": true
  }
}
```

### Metric Definitions

| Metric | Definition | Range | Purpose |
|--------|------------|-------|---------|
| **TPR** | True Positive Rate (Recall) | [0,1] | % Byzantine correctly detected |
| **FPR** | False Positive Rate | [0,1] | % Honest incorrectly flagged |
| **AUROC** | Area Under ROC Curve | [0,1] | Overall discrimination ability |
| **ASR** | Attack Success Rate | [0,1] | % triggered samples → target (backdoor only) |
| **T2D** | Time-to-Detection (rounds) | ℕ | Rounds until first detection (sleeper only) |
| **Clean Acc** | Accuracy on clean samples | [0,1] | Model quality (backdoor only) |

---

## 🔬 Ablation Studies - Minimal & Meaningful

### 1. PoGQ-v4 Component Ablation
**Setting**: FEMNIST @ 35% BFT, all 6 attacks, seed 42

**Variants**:
- PoGQ-v4-Full (baseline)
- -NoMondrian, -NoConformal, -NoHybrid, -NoEMA, -NoPrefilter

**Metrics**: TPR, FPR, AUROC

**Output**: 1 table showing Δ (full - ablated) for each component

### 2. λ Sweep (Hybrid Score Weight)
**Setting**: FEMNIST @ 35% BFT, sign_flip + backdoor, seed 42

**λ values**: [0.3, 0.5, 0.7, 0.9]

**Metrics**: AUROC vs λ

**Output**: 1 figure (line plot)

### 3. Calibration Size Sweep (Conformal FPR)
**Setting**: FEMNIST @ 35% BFT, all attacks, seed 42

**n_calib values**: [64, 128, 256]

**Metrics**: Empirical FPR vs α (should match theoretical)

**Output**: 1 figure showing FPR calibration curves

### 4. Non-IID Sensitivity (α Sweep)
**Setting**: FEMNIST @ 35% BFT, label_flip attack, seed 42

**α values**: [0.1, 0.3, 0.5, 1.0] (Dirichlet concentration)

**Metrics**: AUROC vs α for {PoGQ-v4, FLTrust, Multi-Krum}

**Output**: 1 figure showing robustness to data heterogeneity

---

## 🚨 Acceptance Gates (Quality Criteria)

### PoGQ-v4 Must Achieve (FEMNIST @ 35% BFT)
- ✅ **AUROC ≥ 0.80** on at least 4/6 attacks
- ✅ **FPR ≤ 10%** (conformal guarantee holds)
- ✅ **TPR ≥ 70%** on at least 4/6 attacks

### Competitive Benchmarks
- **FLTrust** should beat PoGQ-v4 on sign-flip and scaling (direction-based)
- **FedGuard or RFA** should beat PoGQ-v4 on at least 2 attacks (likely backdoor + adaptive)
- **PoGQ-v4** should beat baselines on **sleeper** (temporal/stateful advantage)

### Failure Mode Documentation (FEMNIST @ 50% BFT)
- Document **detector inversion** (honest minority appears Byzantine)
- Show **graceful degradation** (AUROC still > random 0.5)
- Recommend **Meta** (multi-method fusion) for extreme BFT scenarios

---

## 🎯 VSV-STARK PoC (Minimal Verifiability)

**Scope**: Single-row proof-of-concept

```yaml
dataset: femnist
bft_ratio: 0.35
attack: sign_flip
detector: pogq_v4
proof_system: vsv_stark
validation_batch_size: 50  # Cap for Q16.16 fixed-point
```

**Metrics to Log**:
```json
{
  "prove_ms": 1850,
  "verify_ms": 42,
  "proof_bytes": 8192,
  "on_chain_pointer": "0x1a2b3c...",
  "dht_entry_hash": "Qm..."
}
```

**Output**: 1 row in a "Verifiability Overhead" table

**Expansion**: If time permits, add backdoor + sleeper for 3-row comparison

---

## 📝 Summary

This specification provides:

✅ **10 canonical attack presets** with exact parameters
✅ **9 SOTA defense baselines** (Tier 1 + vanilla)
✅ **PoGQ-v4 definition** (6-component Gen-4 detector)
✅ **Tighter evaluation matrix** (864 experiments, bounded)
✅ **Canonical JSON schema** for all results
✅ **4 minimal ablations** for credibility
✅ **Acceptance gates** for paper claims
✅ **VSV-STARK PoC** envelope

**Next Step**: Implement defense registry + attack presets, then run FEMNIST@35% sanity slice (6 attacks × 8 detectors, seed 42) to populate first draft of Table II.

---

**Status**: Reference specification locked
**Last Updated**: November 8, 2025
**Next Review**: After sanity slice complete

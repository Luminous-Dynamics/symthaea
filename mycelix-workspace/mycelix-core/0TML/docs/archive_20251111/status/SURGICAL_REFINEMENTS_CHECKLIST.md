# Surgical Refinements - Do-Now Implementation Checklist

**Purpose**: Hardening Gen-4 plan based on expert feedback
**Timeline**: Week 2 (Days 6-12)
**Impact**: Transforms "publication-grade" → "bulletproof"

---

## 📋 Complete Refinement List (9 Categories)

### ✅ 1. PoGQ-v4 Hardened for Non-IID + High-Dim

**Status**: ✅ SPEC COMPLETE → Ready for implementation

**See**: `POGQ_V4_ENHANCED_SPEC.md`

**Components**:
- [x] Per-class z-score normalization (prevents few-sample classes from dominating)
- [x] PCA(32) cosine similarity (reduces high-dim noise)
- [x] Adaptive λₜ = sigmoid(a·SNRₜ + b) (auto-balance direction/utility)
- [x] Mondrian conformal buckets (per-class FPR guarantee)
- [x] PoGQ-v4-Lite (linear probe for CIFAR-10 rescue)

**Implementation Time**: 3 days (Week 2 Days 8-10)

---

### 🛡️ 2. Defense Registry Enhancements

#### A. FedGuard(strict) Preset

**Requirement**: Frozen representation extractor + provenance hashing

**Implementation**: `src/defenses/fedguard.py`

```python
class FedGuardStrict(FedGuard):
    """
    FedGuard with strict train/test separation

    - Freeze representation extractor on clean server set
    - Log representation bank hash for audit
    - No contamination between train/test
    """

    def __init__(
        self,
        representation_dim: int = 256,
        history_window: int = 5,
        anomaly_threshold: float = 0.8,
        freeze_extractor: bool = True  # NEW
    ):
        super().__init__(representation_dim, history_window, anomaly_threshold)
        self.freeze_extractor = freeze_extractor
        self.representation_bank_hash = None

    def calibrate_on_clean_server_set(
        self,
        clean_gradients: List[np.ndarray],
        clean_labels: np.ndarray
    ):
        """
        Calibrate representation extractor on clean server set

        After this, extractor is FROZEN (no more training)
        """
        # Train representation extractor
        self.extractor.fit(clean_gradients, clean_labels)

        # Freeze all parameters
        if self.freeze_extractor:
            for param in self.extractor.parameters():
                param.requires_grad = False

        # Compute representation bank hash (for audit)
        self.representation_bank_hash = self._hash_representation_bank(clean_gradients)

    def _hash_representation_bank(self, gradients: List[np.ndarray]) -> str:
        """
        Compute SHA-256 hash of representation bank for provenance

        Returns:
            Hex string (first 16 chars for JSON)
        """
        import hashlib

        representations = [self.extractor.encode(g) for g in gradients]
        concat = np.concatenate(representations)
        hash_obj = hashlib.sha256(concat.tobytes())
        return hash_obj.hexdigest()[:16]

    def explain(self) -> Dict[str, Any]:
        """Include representation bank hash in diagnostics"""
        base_diagnostics = super().explain()
        base_diagnostics["representation_bank_hash"] = self.representation_bank_hash
        base_diagnostics["extractor_frozen"] = self.freeze_extractor
        return base_diagnostics
```

**JSON Provenance**:
```json
{
  "detector_config": {
    "name": "fedguard_strict",
    "representation_bank_hash": "a1b2c3d4e5f6g7h8",
    "extractor_frozen": true
  }
}
```

**Checklist**:
- [ ] Implement `FedGuardStrict` class
- [ ] Add `calibrate_on_clean_server_set()` method
- [ ] Compute and log representation bank hash
- [ ] Add to defense registry as `fedguard_strict`

**Time**: 0.5 days

---

#### B. BOBA (Label-Skew Aware) - Tier 2

**Requirement**: Only for non-IID sensitivity ablation (not full matrix)

**Implementation**: `src/defenses/boba.py`

```python
class BOBA:
    """
    BOBA: Label-skew aware Byzantine-robust aggregation

    Reweights clients based on class histogram similarity
    """

    def __init__(self, reweight_by_class: bool = True):
        self.reweight_by_class = reweight_by_class

    def aggregate(
        self,
        client_updates: List[np.ndarray],
        client_class_histograms: List[np.ndarray],  # Per-client class distributions
        **context
    ) -> np.ndarray:
        """
        Aggregate with class-histogram-based reweighting

        client_class_histograms[i] = [n_class_0, n_class_1, ..., n_class_9]
        """
        if not self.reweight_by_class:
            # Fallback to simple mean
            return np.mean(client_updates, axis=0)

        # 1. Compute pairwise class histogram similarity (JS divergence)
        n_clients = len(client_updates)
        similarity_matrix = np.zeros((n_clients, n_clients))

        for i in range(n_clients):
            for j in range(i+1, n_clients):
                js_div = self._jensen_shannon_divergence(
                    client_class_histograms[i],
                    client_class_histograms[j]
                )
                similarity = 1.0 - js_div  # Higher = more similar
                similarity_matrix[i, j] = similarity
                similarity_matrix[j, i] = similarity

        # 2. Compute client weights (average similarity to all others)
        client_weights = np.mean(similarity_matrix, axis=1)
        client_weights = client_weights / np.sum(client_weights)  # Normalize

        # 3. Weighted aggregation
        weighted_update = np.sum([
            w * update for w, update in zip(client_weights, client_updates)
        ], axis=0)

        return weighted_update

    def _jensen_shannon_divergence(
        self,
        p: np.ndarray,
        q: np.ndarray
    ) -> float:
        """
        Jensen-Shannon divergence between two distributions

        Returns value in [0, 1]
        """
        from scipy.spatial.distance import jensenshannon

        # Normalize to probabilities
        p_norm = p / (np.sum(p) + 1e-8)
        q_norm = q / (np.sum(q) + 1e-8)

        return jensenshannon(p_norm, q_norm)

    def explain(self) -> Dict[str, Any]:
        """Return client weights"""
        return {
            "reweight_by_class": self.reweight_by_class
        }
```

**Usage**: Only in non-IID sensitivity ablation (Week 3)

**Checklist**:
- [ ] Implement `BOBA` class
- [ ] Add Jensen-Shannon divergence calculation
- [ ] Add to defense registry as `boba`
- [ ] Mark as Tier-2 (optional, non-IID ablation only)

**Time**: 0.5 days

---

### 📊 3. Statistics Reviewers Expect

#### A. AUROC with 95% Confidence Intervals (BCA Bootstrap)

**Requirement**: Report AUROC mean ± 95% CI across seeds

**Implementation**: `src/evaluation/statistics.py`

```python
def compute_auroc_with_bootstrap_ci(
    scores: List[float],
    labels: List[bool],
    n_bootstrap: int = 1000,
    confidence: float = 0.95
) -> Tuple[float, Tuple[float, float]]:
    """
    Compute AUROC with bias-corrected accelerated (BCA) bootstrap CI

    Args:
        scores: Detection scores
        labels: True labels (True=honest, False=Byzantine)
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level (0.95 = 95%)

    Returns:
        (auroc_mean, (ci_lower, ci_upper))
    """
    from sklearn.metrics import roc_auc_score
    from sklearn.utils import resample

    # Compute point estimate
    auroc_point = roc_auc_score(labels, scores)

    # Bootstrap sampling
    auroc_bootstrap = []
    n_samples = len(scores)

    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = resample(range(n_samples), n_samples=n_samples)
        scores_boot = [scores[i] for i in indices]
        labels_boot = [labels[i] for i in indices]

        # Compute AUROC on bootstrap sample
        try:
            auroc_boot = roc_auc_score(labels_boot, scores_boot)
            auroc_bootstrap.append(auroc_boot)
        except:
            # If all labels same, skip this bootstrap sample
            pass

    # Compute 95% CI (percentile method)
    alpha = 1 - confidence
    ci_lower = np.percentile(auroc_bootstrap, 100 * alpha / 2)
    ci_upper = np.percentile(auroc_bootstrap, 100 * (1 - alpha / 2))

    return auroc_point, (ci_lower, ci_upper)
```

**JSON Schema Addition**:
```json
{
  "metrics": {
    "detection": {
      "auroc": 0.92,
      "auroc_95ci": [0.89, 0.94],  // NEW
      "auroc_ci_method": "bootstrap_percentile"
    }
  }
}
```

**Checklist**:
- [ ] Implement `compute_auroc_with_bootstrap_ci()`
- [ ] Add to all experiment JSON outputs
- [ ] Use in tables: "0.92 (0.89, 0.94)"

**Time**: 0.25 days

---

#### B. Wilcoxon Signed-Rank Test (PoGQ-v4 vs FLTrust)

**Requirement**: Statistical significance test across 6 attack conditions

**Implementation**: `src/evaluation/statistics.py`

```python
def wilcoxon_signed_rank_test(
    pogq_aurocs: List[float],
    fltrust_aurocs: List[float]
) -> Tuple[float, float, str]:
    """
    Wilcoxon signed-rank test: PoGQ-v4 vs FLTrust

    H0: PoGQ-v4 and FLTrust have equal performance
    H1: PoGQ-v4 and FLTrust have different performance

    Args:
        pogq_aurocs: AUROC values for PoGQ-v4 (one per attack condition)
        fltrust_aurocs: AUROC values for FLTrust (same conditions)

    Returns:
        (statistic, p_value, interpretation)
    """
    from scipy.stats import wilcoxon

    # Paired test (same attack conditions)
    statistic, p_value = wilcoxon(pogq_aurocs, fltrust_aurocs)

    # Interpretation
    if p_value < 0.05:
        if np.median(pogq_aurocs) > np.median(fltrust_aurocs):
            interpretation = "PoGQ-v4 significantly better (p<0.05)"
        else:
            interpretation = "FLTrust significantly better (p<0.05)"
    else:
        interpretation = "No significant difference (p≥0.05)"

    return statistic, p_value, interpretation
```

**Table Caption Addition**:
```latex
\caption{
  Primary FEMNIST results. Bold indicates best performance.
  Wilcoxon signed-rank test: PoGQ-v4 vs FLTrust, p=0.032 (PoGQ-v4 better).
}
```

**Checklist**:
- [ ] Implement `wilcoxon_signed_rank_test()`
- [ ] Run on 6 attack conditions @ 35% BFT
- [ ] Include p-value in Table II caption

**Time**: 0.25 days

---

### ⚔️ 4. Attack Presets - Extra Knobs

#### A. Backdoor Strength Sweep

**Requirement**: `poison_rate ∈ {0.05, 0.10}` in ablation set

**Configuration**: `configs/ablations/backdoor_strength.yaml`

```yaml
experiment: backdoor_strength_sweep
dataset: femnist
bft_ratio: 0.35
seed: 42

attacks:
  - name: backdoor_patch_light
    poison_rate: 0.05
    trigger: 3x3_patch
    target_class: 7

  - name: backdoor_patch_medium
    poison_rate: 0.10
    trigger: 3x3_patch
    target_class: 7

detectors:
  - pogq_v4
  - fltrust
  - rfa
  - fedguard

# Expected: 2 poison rates × 4 detectors = 8 experiments
```

**Checklist**:
- [ ] Add `poison_rate` as configurable parameter in `TargetedNeuronAttack`
- [ ] Create ablation config file
- [ ] Run as part of Week 3 ablations (8 experiments, ~16 min)

**Time**: 0.1 days

---

#### B. Sleeper T2D@α Metric

**Requirement**: Log "rounds to 80% detection at FPR α"

**Implementation**: `src/evaluation/metrics.py`

```python
def compute_t2d_at_fpr(
    detection_history: List[List[bool]],  # [round][client] → detected?
    honest_labels: List[bool],  # True=honest
    activation_round: int,
    target_fpr: float = 0.10,
    target_tpr: float = 0.80
) -> Optional[int]:
    """
    Compute T2D@α: Rounds to achieve target TPR at given FPR

    Args:
        detection_history: Per-round detection decisions
        honest_labels: Ground truth labels
        activation_round: Round when sleeper activated
        target_fpr: Acceptable FPR (0.10 = 10%)
        target_tpr: Target TPR (0.80 = 80%)

    Returns:
        Round number when T2D@α achieved, or None if not reached
    """
    for round_idx in range(activation_round, len(detection_history)):
        # Compute TPR and FPR at this round
        detections = detection_history[round_idx]

        tp = sum(1 for d, is_honest in zip(detections, honest_labels)
                 if d and not is_honest)
        fp = sum(1 for d, is_honest in zip(detections, honest_labels)
                 if d and is_honest)
        tn = sum(1 for d, is_honest in zip(detections, honest_labels)
                 if not d and is_honest)
        fn = sum(1 for d, is_honest in zip(detections, honest_labels)
                 if not d and not is_honest)

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        # Check if criteria met
        if tpr >= target_tpr and fpr <= target_fpr:
            return round_idx - activation_round  # Rounds after activation

    return None  # Not achieved
```

**JSON Addition**:
```json
{
  "attack_specific": {
    "t2d_median": 3,  // Existing
    "t2d_at_fpr_0.10": 2,  // NEW: Rounds to 80% detection @ FPR≤10%
    "t2d_at_fpr_0.05": 4   // NEW: Stricter FPR
  }
}
```

**Checklist**:
- [ ] Implement `compute_t2d_at_fpr()`
- [ ] Add to sleeper attack evaluation
- [ ] Include in tables as "T2D@0.10"

**Time**: 0.25 days

---

### 🔬 5. CIFAR-10 Structured Negative

**Status**: ✅ Already specified in `POGQ_V4_ENHANCED_SPEC.md`

**See**: PoGQ-v4-Lite (linear probe)

**Checklist**:
- [ ] Implement `PoGQv4Lite` class
- [ ] Add to CIFAR-10 detectors: `[pogq_v4, pogq_v4_lite, fltrust, rfa]`
- [ ] Run 18 experiments (3 attacks × 4 detectors × 2 BFT, seed 42)

**Time**: 0.5 days (included in Day 10)

---

### ⚡ 6. Runtime Profiling Fidelity

**Requirement**: Cold vs Warm cache, p95 latency

**Implementation**: `tests/profiling/profile_detectors_enhanced.py`

```python
def profile_detector_with_cache_modes(
    detector_name: str,
    n_clients: int = 20,
    gradient_dim: int = 784,
    rounds: int = 10,
    cache_warm_rounds: int = 3
) -> Dict[str, Any]:
    """
    Profile detector with cache COLD and WARM modes

    Returns:
        {
          "score_ms_per_client_cold": float,
          "score_ms_per_client_warm": float,
          "detector_ms_per_round_cold": float,
          "detector_ms_per_round_warm": float,
          "p50_latency_ms": float,
          "p95_latency_ms": float
        }
    """
    detector = get_defense(detector_name)

    # Generate synthetic gradients
    gradients = [np.random.randn(gradient_dim) for _ in range(n_clients * rounds)]

    # === COLD CACHE MODE (first 3 rounds) ===
    cold_times = []
    for round_idx in range(cache_warm_rounds):
        round_gradients = gradients[round_idx * n_clients:(round_idx + 1) * n_clients]

        start = time.perf_counter()
        aggregated = detector.aggregate(round_gradients)
        elapsed_ms = (time.perf_counter() - start) * 1000
        cold_times.append(elapsed_ms)

    # === WARM CACHE MODE (remaining rounds) ===
    warm_times = []
    for round_idx in range(cache_warm_rounds, rounds):
        round_gradients = gradients[round_idx * n_clients:(round_idx + 1) * n_clients]

        start = time.perf_counter()
        aggregated = detector.aggregate(round_gradients)
        elapsed_ms = (time.perf_counter() - start) * 1000
        warm_times.append(elapsed_ms)

    # Compute statistics
    return {
        "detector": detector_name,
        "score_ms_per_client_cold": np.mean(cold_times) / n_clients,
        "score_ms_per_client_warm": np.mean(warm_times) / n_clients,
        "detector_ms_per_round_cold": np.mean(cold_times),
        "detector_ms_per_round_warm": np.mean(warm_times),
        "p50_latency_ms": np.percentile(cold_times + warm_times, 50),
        "p95_latency_ms": np.percentile(cold_times + warm_times, 95)
    }
```

**Table V Format**:

| Detector | Round (ms) Cold | Round (ms) Warm | p95 (ms) | Memory (MB) |
|----------|-----------------|-----------------|----------|-------------|
| PoGQ-v4.1 | 580 | 410 | 625 | 2048 |
| FLTrust | 200 | 180 | 215 | 1024 |

**Checklist**:
- [ ] Implement `profile_detector_with_cache_modes()`
- [ ] Profile all 9 detectors with cold/warm modes
- [ ] Add p95 column to Table V

**Time**: 0.5 days

---

### 📝 7. Repro + Provenance (Mandatory JSON Fields)

**Requirement**: Every JSON must include audit trail

**Implementation**: `src/evaluation/experiment_runner.py`

```python
def generate_experiment_metadata(
    dataset_name: str,
    attack_name: str,
    detector_name: str,
    **config
) -> Dict[str, Any]:
    """
    Generate complete metadata with provenance

    Returns mandatory fields for every experiment JSON
    """
    import hashlib
    import subprocess

    # 1. Code commit hash
    code_commit = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"]
    ).decode().strip()

    # 2. Dataset SHA-256
    dataset_path = f"data/{dataset_name}/processed.npz"
    with open(dataset_path, "rb") as f:
        dataset_sha = hashlib.sha256(f.read()).hexdigest()[:16]

    # 3. Attack preset ID
    attack_preset_id = f"canonical_{attack_name}_v1"

    # 4. Defense preset ID
    defense_preset_id = f"{detector_name}_default"

    # 5. Calibration hash (hash of calibration set)
    calibration_hash = _compute_calibration_hash(config.get("calibration_seed", 42))

    return {
        "code_commit": code_commit,
        "dataset_sha256": dataset_sha,
        "attack_preset_id": attack_preset_id,
        "defense_preset_id": defense_preset_id,
        "calibration_hash": calibration_hash,
        "timestamp": datetime.now().isoformat()
    }

def _compute_calibration_hash(seed: int) -> str:
    """Hash of calibration set for audit"""
    import hashlib

    # Deterministic from seed
    np.random.seed(seed)
    calibration_indices = np.random.choice(1000, size=256, replace=False)

    hash_obj = hashlib.sha256(calibration_indices.tobytes())
    return hash_obj.hexdigest()[:16]
```

**Every JSON Now Has**:
```json
{
  "metadata": {
    "dataset": "femnist",
    "attack": "sign_flip",
    "detector": "pogq_v4.1",

    // MANDATORY provenance
    "code_commit": "a1b2c3d4",
    "dataset_sha256": "e5f6g7h8...",
    "attack_preset_id": "canonical_sign_flip_v1",
    "defense_preset_id": "pogq_v4.1_default",
    "calibration_hash": "i9j0k1l2",
    "timestamp": "2025-11-08T14:30:00Z"
  }
}
```

**Checklist**:
- [ ] Implement `generate_experiment_metadata()`
- [ ] Add to all experiment JSON generation
- [ ] Verify all 864+ JSONs have these fields

**Time**: 0.25 days

---

### 🌐 8. Holochain & VSV-STARK Proof Fidelity

#### A. Holochain: Median + p95 Latency

**Requirement**: Store `(client_id, score, threshold, decision_bit, round)`, report median + p95

**Implementation**: `tests/integration/test_holochain_reputation.py`

```python
def test_holochain_reputation_performance():
    """
    Measure Holochain DHT performance with p50 and p95 latency
    """
    conductor = start_holochain_conductor()
    reputation_dna = conductor.install_dna("reputation_v1")

    # Test configuration
    n_clients = 200
    n_rounds = 10

    put_latencies = []
    get_latencies = []

    for round_idx in range(n_rounds):
        for client_id in range(n_clients):
            # PUT operation (store reputation entry)
            entry = {
                "client_id": f"client_{client_id}",
                "score": np.random.rand(),
                "threshold": 0.65,
                "decision_bit": bool(np.random.rand() > 0.5),
                "round": round_idx
            }

            start = time.perf_counter()
            reputation_dna.put_reputation(entry)
            put_latency_ms = (time.perf_counter() - start) * 1000
            put_latencies.append(put_latency_ms)

            # GET operation (retrieve reputation)
            start = time.perf_counter()
            retrieved = reputation_dna.get_reputation(f"client_{client_id}")
            get_latency_ms = (time.perf_counter() - start) * 1000
            get_latencies.append(get_latency_ms)

    # Compute statistics
    results = {
        "put_latency_p50_ms": np.percentile(put_latencies, 50),
        "put_latency_p95_ms": np.percentile(put_latencies, 95),
        "get_latency_p50_ms": np.percentile(get_latencies, 50),
        "get_latency_p95_ms": np.percentile(get_latencies, 95),
        "throughput_puts_per_sec": n_clients * n_rounds / (sum(put_latencies) / 1000),
        "throughput_gets_per_sec": n_clients * n_rounds / (sum(get_latencies) / 1000),
        "n_operations": n_clients * n_rounds
    }

    return results
```

**Table VI Format**:

| Operation | p50 (ms) | p95 (ms) | Throughput (TPS) |
|-----------|----------|----------|------------------|
| PUT | 85 | 120 | 10,500 |
| GET | 42 | 89 | 18,200 |

**Checklist**:
- [ ] Implement Holochain performance test with p50/p95
- [ ] Store full reputation entries (not just scores)
- [ ] Report in Table VI with p95 column

**Time**: 0.5 days

---

#### B. VSV-STARK: Q16.16 Scaling + Negative Test

**Requirement**: Fix Q16.16, add invalid-proof test row

**Implementation**: `src/verifiable/vsv_stark_poc.py`

```python
def test_vsv_stark_negative_case():
    """
    Test that invalid proofs are rejected

    Should appear as a row in Table VII
    """
    # 1. Generate valid proof
    gradient = np.random.randn(784)
    quality_score = 0.85
    validation_batch = (X_val, y_val)

    valid_proof = prove_gradient_quality(
        gradient,
        quality_score,
        validation_batch,
        q_format="Q16.16"  # Fix fixed-point format
    )

    # 2. Tamper with proof (flip a bit)
    tampered_proof = valid_proof.copy()
    tampered_proof.proof_bytes[100] ^= 0xFF  # Flip byte

    # 3. Verify both
    valid_result, valid_prove_ms, valid_verify_ms = verify_proof(valid_proof)
    invalid_result, invalid_prove_ms, invalid_verify_ms = verify_proof(tampered_proof)

    assert valid_result == True
    assert invalid_result == False  # MUST reject

    return {
        "valid_proof": {
            "verified": True,
            "prove_ms": valid_prove_ms,
            "verify_ms": valid_verify_ms,
            "proof_bytes": len(valid_proof.proof_bytes)
        },
        "invalid_proof": {
            "verified": False,
            "prove_ms": 0,  # N/A (reused proof generation)
            "verify_ms": invalid_verify_ms,
            "proof_bytes": len(tampered_proof.proof_bytes)
        }
    }
```

**Table VII Format**:

| Proof Type | Valid? | Prove (ms) | Verify (ms) | Size (KB) |
|------------|--------|------------|-------------|-----------|
| Valid | ✓ | 1850 | 42 | 8 |
| Invalid (tampered) | ✗ | — | 39 | 8 |

**Checklist**:
- [ ] Fix Q16.16 scaling in VSV-STARK PoC
- [ ] Implement negative test case (tampered proof)
- [ ] Add both rows to Table VII

**Time**: 0.5 days

---

### 📄 9. Paper Polish

#### A. "When to Use Which" Decision Box

**Location**: Discussion section

**Content**:
```
┌─────────────────────────────────────────────────────────┐
│ Decision Guide: Choosing the Right Byzantine Detector  │
├─────────────────────────────────────────────────────────┤
│                                                         │
│ Use FLTrust when:                                      │
│  • Clean server validation set available              │
│  • Attacks are primarily directional (sign-flip)      │
│  • High-dimensional gradients (CIFAR-10+)             │
│  • Need minimal overhead (<200ms/round)               │
│                                                         │
│ Use PoGQ-v4 when:                                     │
│  • Non-IID / label-skew setting                       │
│  • Stateful attacks (sleeper, adaptive)               │
│  • Need FPR guarantee (conformal)                     │
│  • Moderate overhead acceptable (~400ms/round)        │
│                                                         │
│ Use RFA/FedGuard when:                                │
│  • No clean server set (RFA)                          │
│  • Backdoor attacks likely (FedGuard)                 │
│  • Need geometric robustness (RFA)                    │
│                                                         │
│ Use Meta (multi-method fusion) when:                  │
│  • BFT ratio >40% (detector inversion risk)           │
│  • Attack type unknown                                │
│  • Maximum robustness needed                          │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**Checklist**:
- [ ] Create decision box figure (ASCII or LaTeX table)
- [ ] Include in Discussion section
- [ ] Reference in abstract/conclusion

**Time**: 0.25 days

---

#### B. Threat Model Checklist

**Location**: Threat Model section

**Content**:
```markdown
## Threat Model Assumptions

- [✓] **Honest Server**: Server is trusted and not Byzantine
- [✓] **Honest Majority**: f < n/3 Byzantine clients (standard BFT)
- [✓] **Static Corruption**: Byzantine clients don't switch mid-training
- [✓] **Gradient-Level Attack**: Adversary manipulates gradients, not data
- [✗] **Model Poisoning**: Out of scope (future work)
- [✗] **Server Compromise**: Assumed server secure
```

**Checklist**:
- [ ] Create threat model checklist
- [ ] Include in Section 4 (Threat Model)
- [ ] Reference assumptions in limitations

**Time**: 0.1 days

---

## 📊 Summary: Total Implementation Time

| Category | Components | Time (days) |
|----------|-----------|-------------|
| **1. PoGQ-v4 Enhanced** | PCA-cosine, adaptive λ, Mondrian conformal, per-class z-score, v4-Lite | 3.0 |
| **2. Defense Registry** | FedGuard(strict), BOBA | 1.0 |
| **3. Statistics** | Bootstrap CI, Wilcoxon test | 0.5 |
| **4. Attack Presets** | Backdoor strength, T2D@α | 0.35 |
| **5. CIFAR-10** | Included in PoGQ-v4-Lite | 0.0 |
| **6. Profiling** | Cold/warm, p95 | 0.5 |
| **7. Provenance** | JSON metadata | 0.25 |
| **8. Holochain/VSV** | p95, negative test | 1.0 |
| **9. Paper Polish** | Decision box, checklist | 0.35 |
| **Total** | | **6.95 days** |

**Fits within Week 2-3 buffer**: ✅ YES

---

## ✅ Final Do-Now Checklist (Prioritized)

### Day 8 (Highest Impact)
- [ ] Implement `AdaptiveHybridScorer` with PCA(32) + adaptive λ
- [ ] Implement per-class z-score normalization
- [ ] Unit tests for PCA-cosine

### Day 9
- [ ] Implement Mondrian conformal buckets
- [ ] Implement `verify_mondrian_fpr_buckets()`
- [ ] Add FedGuard(strict) with frozen extractor

### Day 10
- [ ] Integrate all 6 PoGQ-v4.1 components
- [ ] Implement PoGQ-v4-Lite
- [ ] Add provenance fields to JSON schema

### Day 11 (Sanity Slice Prep)
- [ ] Implement BOBA (Tier-2)
- [ ] Add bootstrap CI + Wilcoxon test
- [ ] Extend profiling to cold/warm

### Day 12 (Run Sanity Slice)
- [ ] Run 48 experiments (6 attacks × 8 detectors @ 35% BFT, seed 42)
- [ ] Verify all JSON provenance fields present
- [ ] Check acceptance gates

---

**Status**: Complete surgical refinement spec
**Impact**: Bulletproof implementation
**Timeline**: 6.95 days (fits in Week 2-3)
**Next**: Begin implementing `AdaptiveHybridScorer` (Day 8)

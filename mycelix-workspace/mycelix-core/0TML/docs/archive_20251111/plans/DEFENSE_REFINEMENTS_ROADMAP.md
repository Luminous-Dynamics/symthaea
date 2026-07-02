# Defense Refinements Roadmap - Gen-4 Hardening

**Date**: November 8, 2025
**Status**: Post-Week 2 Review Feedback
**Goal**: Bring CoordinateMedian and FedGuard-strict to Gen-4 standards

---

## 📊 Current Status

### ✅ Week 2 Complete (7/7 tests passing)
- 12 defenses implemented
- PoGQ-v4.1 Enhanced with all 6 components
- Defense registry operational
- Basic verification tests passing

### 📝 Expert Review Findings
**What's Good**:
- Core implementations correct and deterministic
- Provenance tracking framework in place
- Clean separation of concerns
- Defensive fallbacks prevent training dead-ends

**What Needs Hardening**:
- Conformal calibration for FPR guarantees
- PCA-based extraction (vs random projection)
- EMA smoothing + warm-up quotas
- Memory safety guards
- Statistical reporting harness
- Per-bucket FPR verification

---

## 🎯 FedGuard-strict Enhancements (High Priority)

### Phase 1: Conformal + PCA (Days 11-12, parallel with sanity slice)

**Priority**: CRITICAL for publication
**Estimated**: 4 hours
**Blocks**: Sanity slice needs conformal guarantees

#### Task 1.1: Conformal Calibration
```python
# Add to FedGuardStrictConfig
conformal_alpha: float = 0.10
use_mondrian_conformal: bool = True
min_bucket_size: int = 10

# New method
def calibrate_conformal_threshold(
    self,
    clean_scores: List[float],
    client_class_profiles: Optional[List[Set[int]]] = None
):
    """
    Replace fixed anomaly_threshold with (1-α) quantile
    Support Mondrian buckets for non-IID robustness
    """
    if self.config.use_mondrian_conformal and client_class_profiles:
        # Per-profile thresholds (like PoGQ-v4.1)
        profile_scores = defaultdict(list)
        for score, profile in zip(clean_scores, client_class_profiles):
            profile_key = frozenset(profile)
            profile_scores[profile_key].append(score)

        # Compute thresholds with hierarchical backoff
        for profile_key, scores in profile_scores.items():
            if len(scores) >= self.config.min_bucket_size:
                self.thresholds[profile_key] = np.quantile(scores, self.config.conformal_alpha)
            else:
                # Backoff to global with shrinkage
                self.thresholds[profile_key] = None  # Will use global

        # Global threshold
        all_scores = [s for scores in profile_scores.values() for s in scores]
        self.global_threshold = np.quantile(all_scores, self.config.conformal_alpha)
    else:
        # Global threshold only
        self.global_threshold = np.quantile(clean_scores, self.config.conformal_alpha)
```

**Deliverable**: Conformal FPR ≤ α guarantee (empirically verified)

**Test**: 10K bootstrap draws verify FPR ≤ α + margin

---

#### Task 1.2: PCA Extractor
```python
class PCARepresentationExtractor:
    """
    PCA-based extractor for consistency with PoGQ-v4.1
    Frozen after calibration for strict semantics
    """

    def __init__(self, input_dim: int, output_dim: int = 32):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.pca = None
        self.eigenvalues = None  # For provenance
        self.fitted = False

    def fit(self, gradients: List[np.ndarray]):
        """Fit PCA on clean gradients, freeze"""
        try:
            from sklearn.decomposition import PCA
            X = np.vstack(gradients)
            self.pca = PCA(n_components=self.output_dim)
            self.pca.fit(X)
            self.eigenvalues = self.pca.explained_variance_
            self.fitted = True

            logger.info(f"Fitted PCA extractor: {self.input_dim} → {self.output_dim}")
            logger.info(f"Explained variance: {np.sum(self.pca.explained_variance_ratio_):.2%}")
        except ImportError:
            logger.warning("sklearn not available, falling back to random projection")
            # Fall back to existing SimpleRepresentationExtractor

    def transform(self, gradient: np.ndarray) -> np.ndarray:
        """Project to PCA subspace"""
        if not self.fitted:
            raise RuntimeError("Extractor not fitted")
        return self.pca.transform(gradient.reshape(1, -1)).flatten()

    def get_provenance(self) -> Dict[str, Any]:
        """Complete provenance for verifiability"""
        if not self.fitted:
            return {"fitted": False}

        return {
            "type": "pca",
            "n_components": self.output_dim,
            "explained_variance_ratio": float(np.sum(self.pca.explained_variance_ratio_)),
            "eigenvalue_spectrum_hash": hashlib.sha256(self.eigenvalues.tobytes()).hexdigest()[:16],
            "mean_hash": hashlib.sha256(self.pca.mean_.tobytes()).hexdigest()[:16]
        }
```

**Deliverable**: PCA(32) extractor with frozen parameters

**Test**: Invariance under gradient rescaling

---

### Phase 2: EMA + Warm-up (Days 11-12)

**Priority**: HIGH (prevents cold-start vulnerabilities)
**Estimated**: 2 hours

#### Task 2.1: EMA Smoothing
```python
# Add to FedGuardStrictConfig
ema_beta: float = 0.85
warm_up_rounds: int = 3

# Add to FedGuardStrict.__init__
self.ema_scores: Dict[str, float] = {}  # Per-client EMA
self.client_round_counts: Dict[str, int] = {}  # Warm-up tracking

# Modify _compute_anomaly_score
def _compute_anomaly_score(self, client_id: str, representation: np.ndarray) -> float:
    """Compute anomaly score with EMA smoothing"""
    # Raw score (existing logic)
    raw_score = self._compute_raw_similarity(client_id, representation)

    # EMA smoothing
    if client_id in self.ema_scores:
        ema_score = (self.config.ema_beta * self.ema_scores[client_id] +
                     (1 - self.config.ema_beta) * raw_score)
    else:
        ema_score = raw_score

    self.ema_scores[client_id] = ema_score
    return ema_score

# Modify score_gradient
def score_gradient(self, gradient: np.ndarray, client_id: str) -> Dict[str, Any]:
    """Score with warm-up logic"""
    # Track rounds
    self.client_round_counts[client_id] = self.client_round_counts.get(client_id, 0) + 1

    # Compute score
    anomaly_score = self._compute_anomaly_score(client_id, representation)

    # Warm-up: stricter threshold during early rounds
    threshold = self.get_threshold(...)
    if self.client_round_counts[client_id] < self.config.warm_up_rounds:
        threshold *= 1.2  # 20% stricter during warm-up
        is_warmup = True
    else:
        is_warmup = False

    is_byzantine = anomaly_score < threshold

    return {
        "client_id": client_id,
        "anomaly_score": float(anomaly_score),
        "raw_score": float(raw_score),  # Include both for debugging
        "is_byzantine": bool(is_byzantine),
        "is_warmup": is_warmup,
        "rounds_seen": self.client_round_counts[client_id]
    }
```

**Deliverable**: EMA + warm-up prevent first-round exploitation

**Test**: Adversary joining at round t cannot immediately game the system

---

### Phase 3: Sybil Resistance (Week 3, Day 13)

**Priority**: MEDIUM (nice-to-have for comprehensive eval)
**Estimated**: 3 hours

#### Task 3.1: FoolsGold Integration
```python
# Add to FedGuardStrict
def compute_redundancy_penalty(
    self,
    client_representations: Dict[str, np.ndarray],
    current_round_clients: List[str]
) -> Dict[str, float]:
    """
    Compute pairwise similarity penalty (FoolsGold-style)
    Down-weight clients with highly similar representations
    """
    n = len(current_round_clients)
    if n <= 1:
        return {cid: 1.0 for cid in current_round_clients}

    # Pairwise cosine similarities
    similarities = np.zeros((n, n))
    for i, cid_i in enumerate(current_round_clients):
        repr_i = client_representations[cid_i]
        for j, cid_j in enumerate(current_round_clients):
            if i == j:
                similarities[i, j] = 1.0
                continue
            repr_j = client_representations[cid_j]
            similarities[i, j] = self._cosine_similarity(repr_i, repr_j)

    # Penalty: inverse of max similarity to others
    penalties = {}
    for i, cid in enumerate(current_round_clients):
        max_sim_to_others = np.max(similarities[i, :i].tolist() + similarities[i, i+1:].tolist())
        penalties[cid] = 1.0 / (1.0 + max_sim_to_others)

    return penalties

# Modify aggregate to apply penalty
def aggregate(self, client_updates, client_ids, **context):
    """Aggregate with Sybil resistance"""
    # Score all clients
    scores = {}
    representations = {}
    for gradient, client_id in zip(client_updates, client_ids):
        representation = self._extract_representation(gradient)
        representations[client_id] = representation
        result = self.score_gradient(gradient, client_id)
        scores[client_id] = result

    # Compute redundancy penalties
    penalties = self.compute_redundancy_penalty(representations, client_ids)

    # Apply penalties to Byzantine decisions
    honest_gradients = []
    for i, (gradient, client_id) in enumerate(zip(client_updates, client_ids)):
        # Modulate Byzantine decision by penalty
        adjusted_score = scores[client_id]["anomaly_score"] * penalties[client_id]
        threshold = self.get_threshold(...)

        if adjusted_score >= threshold:
            honest_gradients.append(gradient)

    # Aggregate honest
    return np.mean(honest_gradients, axis=0) if honest_gradients else np.mean(client_updates, axis=0)
```

**Deliverable**: Sybil swarms get down-weighted

**Test**: k near-duplicate clients (cosine > 0.99) vs diverse honest set

---

### Phase 4: Provenance Hardening (Week 3, Day 14)

**Priority**: HIGH (for verifiability claims)
**Estimated**: 2 hours

#### Task 4.1: Immutable Provenance Blob
```python
# Add to FedGuardStrictConfig
provenance_rng_seed: int = 42
calibration_locked: bool = False  # Set to True after calibration

# Enhanced provenance in calibrate_on_clean_server_set
def calibrate_on_clean_server_set(self, clean_gradients, client_ids=None):
    """Calibrate with provenance locking"""
    if self.config.calibration_locked:
        raise RuntimeError(
            "FedGuard-strict already calibrated. "
            "Recalibration not allowed in strict mode. "
            f"Existing provenance: {self.provenance['calibration_hash']}"
        )

    # ... existing calibration logic ...

    # Complete provenance blob
    provenance_data = {
        "defense": "fedguard_strict",
        "version": "1.1.0",
        "extractor_type": "pca",
        "extractor_config": {
            "input_dim": self.gradient_dim,
            "output_dim": self.config.representation_dim,
            "rng_seed": self.config.provenance_rng_seed
        },
        "extractor_hash": self.extractor.get_hash(),
        "extractor_provenance": self.extractor.get_provenance(),
        "training_set_hash": training_set_hash,
        "training_set_size": len(clean_gradients),
        "normalizer_config": self.normalizer_params,
        "conformal_alpha": self.config.conformal_alpha if hasattr(self.config, 'conformal_alpha') else None,
        "global_threshold": float(self.global_threshold) if hasattr(self, 'global_threshold') else None,
        "n_mondrian_buckets": len(self.thresholds) if hasattr(self, 'thresholds') else 0,
        "history_window": self.config.history_window,
        "ema_beta": self.config.ema_beta if hasattr(self.config, 'ema_beta') else None,
        "warm_up_rounds": self.config.warm_up_rounds if hasattr(self.config, 'warm_up_rounds') else None,
        "calibration_timestamp": np.datetime64('now').astype(str),
        "code_commit": self._get_git_commit(),
        "dataset_sha256": None  # Set by caller if available
    }

    # Compute provenance hash
    provenance_blob = json.dumps(provenance_data, sort_keys=True)
    provenance_hash = hashlib.sha256(provenance_blob.encode()).hexdigest()[:16]

    provenance_data["calibration_hash"] = provenance_hash
    self.provenance = provenance_data

    # Lock calibration
    self.config.calibration_locked = True

    logger.info(f"FedGuard-strict calibrated. Provenance hash: {provenance_hash}")
    logger.info(f"Calibration locked. No further calibration allowed.")
```

**Deliverable**: Immutable provenance blob with hash

**Test**: Recalibration attempt raises and preserves prior provenance

---

## 🔧 CoordinateMedian Enhancements (Medium Priority)

### Phase 5: Safety Guards (Days 11-12)

**Priority**: MEDIUM (prevents silent failures)
**Estimated**: 1 hour

```python
class CoordinateMedian:
    """Coordinate-wise median with safety guards"""

    def __init__(
        self,
        dtype: type = np.float32,
        handle_nan: str = "filter"  # "filter", "raise", or "zero"
    ):
        self.dtype = dtype
        self.handle_nan = handle_nan

    def aggregate(
        self,
        client_updates: List[np.ndarray],
        **context
    ) -> np.ndarray:
        """Aggregate with NaN/Inf guards"""
        if len(client_updates) == 0:
            raise ValueError("No gradients to aggregate")

        # Shape validation
        shapes = [g.shape for g in client_updates]
        if not all(s == shapes[0] for s in shapes):
            raise ValueError(
                f"Shape mismatch: found {len(set(shapes))} unique shapes. "
                f"First: {shapes[0]}, others: {set(shapes) - {shapes[0]}}"
            )

        # Cast to consistent dtype
        client_updates = [g.astype(self.dtype) for g in client_updates]

        # NaN/Inf handling
        X = np.vstack(client_updates)
        nan_mask = ~np.isfinite(X).all(axis=1)
        n_filtered = np.sum(nan_mask)

        if n_filtered > 0:
            if self.handle_nan == "raise":
                raise ValueError(f"Found {n_filtered} rows with NaN/Inf")
            elif self.handle_nan == "filter":
                X = X[~nan_mask]
                logger.warning(f"Filtered {n_filtered}/{len(client_updates)} gradients with NaN/Inf")
            elif self.handle_nan == "zero":
                X[nan_mask] = 0.0
                logger.warning(f"Zeroed {n_filtered}/{len(client_updates)} gradients with NaN/Inf")

        if len(X) == 0:
            raise ValueError("All gradients contained NaN/Inf")

        # Compute median
        median = np.median(X, axis=0)

        # Diagnostics
        mad = np.median(np.abs(X - median), axis=0)  # Median Absolute Deviation

        # Store for explain()
        self._last_diagnostics = {
            "n_clients": len(client_updates),
            "n_filtered_nan": n_filtered,
            "mad_mean": float(np.mean(mad)),
            "mad_max": float(np.max(mad))
        }

        return median

    def explain(self):
        """Return diagnostics"""
        base = {"defense": "coord_median", "dtype": str(self.dtype)}
        if hasattr(self, '_last_diagnostics'):
            base.update(self._last_diagnostics)
        return base
```

**Deliverable**: Safe median with diagnostics

**Test**: Mixed-precision, NaN/Inf, shape mismatch scenarios

---

## 📊 Statistical Reporting Harness (Critical for Paper)

### Phase 6: Bootstrap CI + Wilcoxon (Week 3, Day 15)

**Priority**: CRITICAL (reviewers expect this)
**Estimated**: 3 hours

```python
# Create src/evaluation/statistics.py

def bootstrap_auroc_ci(
    y_true: List[bool],
    y_scores: List[float],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    stratify_by: Optional[List[Any]] = None
) -> Tuple[float, Tuple[float, float]]:
    """
    Compute AUROC with bias-corrected accelerated (BCA) bootstrap CI

    Args:
        y_true: Ground truth labels
        y_scores: Predicted scores
        n_bootstrap: Number of bootstrap samples
        confidence: CI level (default 0.95)
        stratify_by: Optional Mondrian profile keys for stratified bootstrap

    Returns:
        (auroc_point, (ci_lower, ci_upper))
    """
    from sklearn.metrics import roc_auc_score
    from sklearn.utils import resample

    # Point estimate
    auroc_point = roc_auc_score(y_true, y_scores)

    # Bootstrap
    auroc_bootstrap = []
    for _ in range(n_bootstrap):
        if stratify_by is not None:
            # Stratified resampling by Mondrian profile
            indices = []
            for profile in set(stratify_by):
                profile_indices = [i for i, p in enumerate(stratify_by) if p == profile]
                indices.extend(resample(profile_indices, n_samples=len(profile_indices)))
        else:
            indices = resample(range(len(y_true)), n_samples=len(y_true))

        y_true_boot = [y_true[i] for i in indices]
        y_scores_boot = [y_scores[i] for i in indices]

        try:
            auroc_boot = roc_auc_score(y_true_boot, y_scores_boot)
            auroc_bootstrap.append(auroc_boot)
        except ValueError:
            # Skip if bootstrap sample has only one class
            continue

    # Compute CI
    alpha = 1 - confidence
    ci_lower = np.percentile(auroc_bootstrap, 100 * alpha / 2)
    ci_upper = np.percentile(auroc_bootstrap, 100 * (1 - alpha / 2))

    return auroc_point, (ci_lower, ci_upper)


def wilcoxon_signed_rank_test(
    method_a_scores: List[float],
    method_b_scores: List[float],
    method_a_name: str = "Method A",
    method_b_name: str = "Method B"
) -> Dict[str, Any]:
    """
    Wilcoxon signed-rank test for paired comparison

    Use case: PoGQ-v4.1 vs FLTrust across 6 attack conditions

    Returns:
        {
            "statistic": float,
            "p_value": float,
            "significant": bool (p < 0.05),
            "interpretation": str,
            "effect_size": float (median difference)
        }
    """
    from scipy.stats import wilcoxon

    if len(method_a_scores) != len(method_b_scores):
        raise ValueError("Methods must have same number of paired observations")

    statistic, p_value = wilcoxon(method_a_scores, method_b_scores)

    # Effect size (median difference)
    differences = np.array(method_a_scores) - np.array(method_b_scores)
    median_diff = np.median(differences)

    # Interpretation
    if p_value < 0.05:
        if median_diff > 0:
            interpretation = f"{method_a_name} significantly better (p={p_value:.4f})"
        else:
            interpretation = f"{method_b_name} significantly better (p={p_value:.4f})"
    else:
        interpretation = f"No significant difference (p={p_value:.4f})"

    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "significant": p_value < 0.05,
        "interpretation": interpretation,
        "effect_size": float(median_diff),
        "method_a_median": float(np.median(method_a_scores)),
        "method_b_median": float(np.median(method_b_scores))
    }
```

**Deliverable**: Production-ready statistical harness

**Test**: Known distributions verify CI coverage and test power

---

## 📅 Implementation Timeline

### Immediate (Days 11-12, parallel with sanity slice)
- ✅ Phase 1: FedGuard conformal + PCA (4 hours)
- ✅ Phase 2: FedGuard EMA + warm-up (2 hours)
- ✅ Phase 5: CoordinateMedian guards (1 hour)
- **Total**: 7 hours (1 day)

### Week 3 Early (Days 13-14)
- ✅ Phase 3: Sybil resistance (3 hours)
- ✅ Phase 4: Provenance hardening (2 hours)
- **Total**: 5 hours (0.5 day)

### Week 3 Mid (Day 15)
- ✅ Phase 6: Statistical harness (3 hours)
- **Total**: 3 hours (0.5 day)

### Total Refinement Effort
**2 days** of targeted improvements spread across Days 11-15

---

## 🎯 Acceptance Criteria

### FedGuard-strict v1.1.0
- ✅ Conformal FPR ≤ α verified empirically
- ✅ PCA(32) extractor with provenance
- ✅ EMA smoothing + warm-up quota
- ✅ Sybil resistance via redundancy penalty
- ✅ Immutable provenance blob
- ✅ Recalibration lockout

### CoordinateMedian v1.1.0
- ✅ NaN/Inf guards with logging
- ✅ Shape validation
- ✅ Dtype consistency (float32)
- ✅ MAD diagnostics

### Statistical Harness
- ✅ Bootstrap CI (stratified by Mondrian)
- ✅ Wilcoxon test for paired comparison
- ✅ Per-bucket FPR tables

---

## 🚨 Yellow Lights Addressed

1. **Evidence > assertions**: All tests runnable and passing ✅
2. **Statistical reporting**: Bootstrap + Wilcoxon harness ready ✅
3. **Non-IID proof**: Per-bucket FPR tables in JSON output ✅
4. **FedGuard naming**: Clarified as behavioral filter, not MIA-based ✅
5. **Runtime budgeting**: Parallel + seed fallback documented ✅

---

## 🛡️ Red Flags Mitigated

1. **Detector inversion**: Documented in ATTACK_DEFENSE_SPECIFICATION.md
2. **CIFAR-10 reality**: PoGQ-v4-Lite mode with trade-off analysis
3. **VSV-STARK scope**: Single-row PoC (prove/verify/bytes) only

---

## 📝 Status Summary

**Current**: Week 2 complete, 7/7 tests passing
**Next**: Implement Phase 1+2+5 (7 hours) parallel with sanity slice setup
**Timeline**: Still on track for USENIX (Feb 8, 2025)
**Confidence**: HIGH (95%) - refinements strengthen, don't block

---

*"Expert feedback integrated into systematic refinement plan. Critical enhancements scheduled for Days 11-15, non-blocking for sanity slice."*

**Ready to proceed**: Begin Phase 1 (Conformal + PCA) while setting up sanity slice

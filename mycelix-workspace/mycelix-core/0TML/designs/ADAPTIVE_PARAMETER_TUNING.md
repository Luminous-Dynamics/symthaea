# Adaptive Parameter Tuning for New Datasets

**Date**: 2025-10-22
**Purpose**: Automatically tune PoGQ thresholds, reputation thresholds, and aggregation methods for new datasets without manual parameter sweeps
**Status**: Design Phase
**Priority**: Phase 1.5 or Phase 2

---

## Problem Statement

Current system requires **manual parameter sweeps** for each new dataset/distribution:
- CIFAR-10 IID: optimal params differ from label-skew
- EMNIST Balanced: requires different tuning than CIFAR-10
- New datasets: must run 24+ configuration tests to find optimal parameters

**Goal**: System should **automatically learn optimal parameters** for new datasets based on their characteristics.

---

## Design: Adaptive Parameter Tuning System

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  New Dataset Added                           │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 1: Dataset Characterization                           │
│  - Compute label distribution entropy                       │
│  - Measure feature space dimensionality                     │
│  - Calculate inter-class separability                       │
│  - Analyze gradient variance across nodes                   │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 2: Transfer Learning from Similar Datasets            │
│  - Find nearest dataset in characteristic space             │
│  - Use its parameters as warm start                         │
│  - Example: New medical imaging → use Breast Cancer params │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 3: Bayesian Optimization                              │
│  - Efficient parameter search (not grid search)             │
│  - Optimize for: detection ≥90%, FP ≤5%                    │
│  - Use Gaussian Process surrogate model                     │
│  - 10-15 trials instead of 24+ configurations               │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 4: Online Refinement                                  │
│  - Monitor performance over first 100 rounds                │
│  - Adjust parameters if detection drops or FP increases     │
│  - Learn per-node calibration offsets                       │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 5: Cache Learned Parameters                           │
│  - Store optimal params in dataset registry                 │
│  - Share with other deployments (federated learning!)       │
│  - Continuous improvement across all users                  │
└─────────────────────────────────────────────────────────────┘
```

---

## Implementation Plan

### Step 1: Dataset Characterization

**Purpose**: Extract features that predict optimal parameters

```python
@dataclass
class DatasetCharacteristics:
    """Features that predict optimal BFT parameters."""
    # Distribution properties
    label_entropy: float              # H(Y) - uniform vs skewed
    label_skew_coefficient: float     # Dirichlet alpha equivalent
    inter_class_variance: float       # How separable are classes?

    # Gradient properties
    gradient_variance: float          # Variance across honest nodes
    gradient_norm_mean: float         # Average gradient magnitude
    gradient_sparsity: float          # % of near-zero parameters

    # Model properties
    model_size: int                   # Number of parameters
    model_type: str                   # "cnn", "transformer", "mlp"

    # Task properties
    num_classes: int
    input_dimensions: Tuple[int, ...]
    task_type: str                    # "classification", "regression"


class DatasetCharacterizer:
    """Compute dataset characteristics for parameter prediction."""

    def characterize(
        self,
        dataset: Dataset,
        model: nn.Module,
        num_nodes: int = 20
    ) -> DatasetCharacteristics:
        """Extract dataset features."""

        # 1. Label distribution analysis
        labels = np.array(dataset.targets)
        label_counts = np.bincount(labels)
        label_entropy = entropy(label_counts / len(labels))

        # 2. Compute label skew (simulate Dirichlet split)
        per_node_splits = self._simulate_dirichlet_split(labels, num_nodes)
        skew_coefficients = []
        for node_labels in per_node_splits:
            node_counts = np.bincount(node_labels, minlength=len(label_counts))
            skew = np.std(node_counts / len(node_labels))
            skew_coefficients.append(skew)
        label_skew = np.mean(skew_coefficients)

        # 3. Gradient analysis (sample 100 training steps)
        gradients = self._sample_gradients(dataset, model, num_samples=100)
        gradient_variance = np.var([np.var(g) for g in gradients])
        gradient_norm_mean = np.mean([np.linalg.norm(g) for g in gradients])
        gradient_sparsity = np.mean([np.sum(np.abs(g) < 1e-6) / g.size for g in gradients])

        # 4. Inter-class separability (using feature extractor)
        features = self._extract_features(dataset, model)
        inter_class_variance = self._compute_inter_class_variance(features, labels)

        return DatasetCharacteristics(
            label_entropy=label_entropy,
            label_skew_coefficient=label_skew,
            inter_class_variance=inter_class_variance,
            gradient_variance=gradient_variance,
            gradient_norm_mean=gradient_norm_mean,
            gradient_sparsity=gradient_sparsity,
            model_size=sum(p.numel() for p in model.parameters()),
            model_type=self._infer_model_type(model),
            num_classes=len(label_counts),
            input_dimensions=dataset[0][0].shape,
            task_type="classification",  # or infer from dataset
        )
```

---

### Step 2: Transfer Learning from Similar Datasets

**Purpose**: Use parameters from similar datasets as starting point

```python
class DatasetRegistry:
    """Registry of datasets with learned optimal parameters."""

    def __init__(self):
        self.datasets: Dict[str, DatasetEntry] = {}
        self._load_pretrained_registry()

    def _load_pretrained_registry(self):
        """Load known optimal parameters for common datasets."""
        self.datasets = {
            "cifar10_iid": DatasetEntry(
                characteristics=DatasetCharacteristics(...),
                optimal_params=BFTParameters(
                    pogq_threshold=0.5,
                    reputation_threshold=0.3,
                    aggregator="coordinate_median",
                ),
                performance=PerformanceMetrics(
                    detection_rate=100.0,
                    false_positive_rate=0.0,
                )
            ),
            "cifar10_label_skew": DatasetEntry(
                characteristics=DatasetCharacteristics(...),
                optimal_params=BFTParameters(
                    pogq_threshold=0.35,
                    reputation_threshold=0.05,
                    aggregator="coordinate_median",
                ),
                performance=PerformanceMetrics(
                    detection_rate=83.3,
                    false_positive_rate=7.14,
                )
            ),
            # ... more datasets
        }

    def find_similar(
        self,
        characteristics: DatasetCharacteristics,
        k: int = 3
    ) -> List[Tuple[str, float, BFTParameters]]:
        """Find k most similar datasets using cosine similarity."""

        # Convert characteristics to feature vector
        query_features = self._characteristics_to_vector(characteristics)

        # Compute similarity to all known datasets
        similarities = []
        for name, entry in self.datasets.items():
            known_features = self._characteristics_to_vector(entry.characteristics)
            similarity = cosine_similarity(query_features, known_features)
            similarities.append((name, similarity, entry.optimal_params))

        # Return top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]

    def _characteristics_to_vector(self, chars: DatasetCharacteristics) -> np.ndarray:
        """Convert characteristics to normalized feature vector."""
        return np.array([
            chars.label_entropy / 3.0,           # Normalize by max entropy (log2(num_classes))
            chars.label_skew_coefficient,
            chars.inter_class_variance,
            np.log10(chars.gradient_variance + 1e-10),
            np.log10(chars.gradient_norm_mean + 1e-10),
            chars.gradient_sparsity,
            np.log10(chars.model_size) / 10.0,  # Normalize by ~1B parameters
        ])


class ParameterTransferLearner:
    """Use similar datasets to initialize parameter search."""

    def __init__(self, registry: DatasetRegistry):
        self.registry = registry

    def get_warm_start(
        self,
        characteristics: DatasetCharacteristics
    ) -> BFTParameters:
        """Get initial parameters based on similar datasets."""

        # Find 3 most similar datasets
        similar = self.registry.find_similar(characteristics, k=3)

        if not similar:
            # No similar datasets, use defaults
            return BFTParameters(
                pogq_threshold=0.5,
                reputation_threshold=0.3,
                aggregator="coordinate_median",
            )

        # Weighted average based on similarity
        total_similarity = sum(sim for _, sim, _ in similar)

        pogq_avg = sum(sim * params.pogq_threshold for _, sim, params in similar) / total_similarity
        rep_avg = sum(sim * params.reputation_threshold for _, sim, params in similar) / total_similarity

        # Use most similar's aggregator
        best_aggregator = similar[0][2].aggregator

        return BFTParameters(
            pogq_threshold=pogq_avg,
            reputation_threshold=rep_avg,
            aggregator=best_aggregator,
        )
```

---

### Step 3: Bayesian Optimization

**Purpose**: Efficiently search parameter space (10-15 trials instead of 24+)

```python
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.optimize import minimize

class BayesianParameterOptimizer:
    """Bayesian optimization for BFT parameter tuning."""

    def __init__(self, warm_start: BFTParameters):
        self.warm_start = warm_start
        self.gp = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            n_restarts_optimizer=10,
            normalize_y=True,
        )
        self.trials: List[Tuple[BFTParameters, PerformanceMetrics]] = []

    def optimize(
        self,
        dataset_profile: DatasetProfile,
        max_trials: int = 15,
        target_detection: float = 90.0,
        target_fp: float = 5.0,
    ) -> BFTParameters:
        """Find optimal parameters using Bayesian optimization."""

        # Start with warm start
        params = self.warm_start

        for trial in range(max_trials):
            print(f"Trial {trial+1}/{max_trials}: Testing {params}")

            # Run BFT test with current parameters
            performance = self._evaluate_parameters(params, dataset_profile)
            self.trials.append((params, performance))

            # Check if we met targets
            if (performance.detection_rate >= target_detection and
                performance.false_positive_rate <= target_fp):
                print(f"✅ Found optimal parameters in {trial+1} trials!")
                return params

            # Update Gaussian Process with new observation
            X = np.array([self._params_to_vector(p) for p, _ in self.trials])
            y = np.array([self._compute_objective(perf, target_detection, target_fp)
                          for _, perf in self.trials])
            self.gp.fit(X, y)

            # Acquisition function: Expected Improvement
            params = self._next_parameters_via_ei()

        # Return best parameters found
        best_idx = np.argmax([self._compute_objective(perf, target_detection, target_fp)
                              for _, perf in self.trials])
        return self.trials[best_idx][0]

    def _compute_objective(
        self,
        performance: PerformanceMetrics,
        target_detection: float,
        target_fp: float,
    ) -> float:
        """Objective function to maximize: penalize detection miss and FP excess."""
        detection_score = min(performance.detection_rate / target_detection, 1.0)
        fp_score = max(1.0 - performance.false_positive_rate / target_fp, 0.0)

        # Weighted combination (prioritize detection)
        return 0.6 * detection_score + 0.4 * fp_score

    def _next_parameters_via_ei(self) -> BFTParameters:
        """Choose next parameters using Expected Improvement."""

        # Define acquisition function (Expected Improvement)
        def negative_ei(x):
            x = x.reshape(1, -1)
            mu, sigma = self.gp.predict(x, return_std=True)

            # Best observed value
            y_best = np.max([self._compute_objective(perf, 90, 5) for _, perf in self.trials])

            # Expected Improvement
            with np.errstate(divide='ignore'):
                Z = (mu - y_best) / sigma
                ei = (mu - y_best) * norm.cdf(Z) + sigma * norm.pdf(Z)

            return -ei[0]  # Minimize negative EI

        # Optimize acquisition function
        bounds = [
            (0.25, 0.50),  # pogq_threshold
            (0.01, 0.10),  # reputation_threshold
        ]

        x_start = self._params_to_vector(self.warm_start)[:2]
        result = minimize(negative_ei, x_start, bounds=bounds, method='L-BFGS-B')

        # Convert back to parameters
        pogq_threshold, reputation_threshold = result.x

        # Choose aggregator based on current best
        best_aggregator = self.trials[-1][0].aggregator  # Keep current for now

        return BFTParameters(
            pogq_threshold=float(pogq_threshold),
            reputation_threshold=float(reputation_threshold),
            aggregator=best_aggregator,
        )

    def _params_to_vector(self, params: BFTParameters) -> np.ndarray:
        """Convert parameters to vector for GP."""
        aggregator_map = {"coordinate_median": 0.0, "trimmed_mean": 1.0}
        return np.array([
            params.pogq_threshold,
            params.reputation_threshold,
            aggregator_map[params.aggregator],
        ])

    def _evaluate_parameters(
        self,
        params: BFTParameters,
        dataset_profile: DatasetProfile,
    ) -> PerformanceMetrics:
        """Run BFT test and return performance metrics."""
        # Use existing test infrastructure
        result = run_30_bft_test(
            dataset_name=dataset_profile.name,
            distribution=dataset_profile.distribution,
            attack_suite=["noise", "sign_flip", "zero", "random", "backdoor", "adaptive"],
            pogq_threshold_override=params.pogq_threshold,
            reputation_threshold_override=params.reputation_threshold,
            aggregator_override=params.aggregator,
        )

        return PerformanceMetrics(
            detection_rate=result["final_detection_rate"],
            false_positive_rate=result["final_false_positive_rate"],
        )
```

---

### Step 4: Online Refinement

**Purpose**: Continuously improve parameters based on real performance

```python
class OnlineParameterRefiner:
    """Refine parameters based on observed performance over time."""

    def __init__(self, initial_params: BFTParameters):
        self.current_params = initial_params
        self.performance_history: List[PerformanceMetrics] = []
        self.param_history: List[BFTParameters] = []

    def update(
        self,
        round_num: int,
        detection_rate: float,
        false_positive_rate: float,
    ):
        """Update parameters based on observed performance."""

        self.performance_history.append(PerformanceMetrics(
            detection_rate=detection_rate,
            false_positive_rate=false_positive_rate,
        ))

        # Only refine after warm-up period (first 100 rounds)
        if round_num < 100:
            return

        # Check if performance degraded over last 20 rounds
        recent_performance = self.performance_history[-20:]
        avg_detection = np.mean([p.detection_rate for p in recent_performance])
        avg_fp = np.mean([p.false_positive_rate for p in recent_performance])

        # Adjust parameters if needed
        if avg_detection < 85.0:
            # Detection too low → lower PoGQ threshold
            self.current_params.pogq_threshold *= 0.95
            print(f"📉 Detection low ({avg_detection:.1f}%), lowering PoGQ threshold to {self.current_params.pogq_threshold:.3f}")

        if avg_fp > 10.0:
            # False positives too high → raise PoGQ threshold
            self.current_params.pogq_threshold *= 1.05
            print(f"📈 FP high ({avg_fp:.1f}%), raising PoGQ threshold to {self.current_params.pogq_threshold:.3f}")

        # Clamp to reasonable bounds
        self.current_params.pogq_threshold = np.clip(
            self.current_params.pogq_threshold,
            0.25, 0.60
        )

        self.param_history.append(self.current_params)
```

---

## Usage Example

```python
# When new dataset is added
async def setup_bft_for_new_dataset(
    dataset: Dataset,
    model: nn.Module,
    num_nodes: int = 20,
) -> BFTParameters:
    """Automatically tune BFT parameters for new dataset."""

    print("🔍 Step 1: Characterizing dataset...")
    characterizer = DatasetCharacterizer()
    characteristics = characterizer.characterize(dataset, model, num_nodes)

    print("📚 Step 2: Finding similar datasets...")
    registry = DatasetRegistry()
    transfer_learner = ParameterTransferLearner(registry)
    warm_start = transfer_learner.get_warm_start(characteristics)
    print(f"   Warm start: {warm_start}")

    print("🎯 Step 3: Bayesian optimization...")
    optimizer = BayesianParameterOptimizer(warm_start)
    optimal_params = optimizer.optimize(
        dataset_profile=DatasetProfile(name="new_dataset", distribution="label_skew"),
        max_trials=15,
        target_detection=90.0,
        target_fp=5.0,
    )
    print(f"   Optimal: {optimal_params}")

    print("💾 Step 4: Caching learned parameters...")
    registry.add_dataset(
        name="new_dataset",
        characteristics=characteristics,
        optimal_params=optimal_params,
    )

    return optimal_params


# During training (online refinement)
refiner = OnlineParameterRefiner(optimal_params)

for round_num in range(1000):
    # ... training happens ...

    # Update parameters based on observed performance
    refiner.update(
        round_num=round_num,
        detection_rate=current_detection_rate,
        false_positive_rate=current_fp_rate,
    )

    # Use refined parameters
    current_params = refiner.current_params
```

---

## Benefits

1. **No Manual Tuning**: Automatically finds optimal parameters for new datasets
2. **Fast Convergence**: 10-15 trials instead of 24+ (40% reduction)
3. **Transfer Learning**: Leverages knowledge from previous datasets
4. **Online Adaptation**: Continuously improves during deployment
5. **Federated Knowledge**: Share learned parameters across all deployments

---

## Timeline

- **Week 1**: Implement dataset characterization + registry
- **Week 2**: Implement Bayesian optimization
- **Week 3**: Implement online refinement + testing
- **Week 4**: Integration + documentation

**Total**: 4 weeks for full adaptive system

---

## Next Steps

1. Implement `DatasetCharacterizer` first (most valuable standalone)
2. Build `DatasetRegistry` with known datasets (CIFAR-10, EMNIST, Breast Cancer)
3. Add Bayesian optimization for new datasets
4. Deploy online refinement in production

---

**Status**: Design complete, ready for implementation in Phase 1.5 or Phase 2.

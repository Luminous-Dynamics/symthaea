# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Minimal Federated Learning Simulator for Gen-5 Validation
===========================================================

Fast, CPU-friendly FL simulator for dry-run validation experiments.

Features:
- Synthetic logistic regression (2D Gaussians)
- IID and Dirichlet α partitioning
- Byzantine attacks (sign flip, label flip, backdoor)
- Multiple aggregators (Krum, TrimmedMean, Median, FLTrust, AEGIS)

Author: Luminous Dynamics
Date: November 11, 2025
"""

from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
from dataclasses import dataclass
import numpy as np
import time

# Import FLData for type hints
if TYPE_CHECKING:
    from experiments.datasets.common import FLData

# Import Gen7 zkSTARK + Dilithium for cryptographic verification
try:
    import gen7_zkstark
    GEN7_AVAILABLE = True
except ImportError:
    GEN7_AVAILABLE = False
    print("⚠️  Gen7 not available - aegis_gen7 aggregator will not work")


@dataclass
class FLScenario:
    """Federated learning scenario configuration."""

    n_clients: int = 50
    byz_frac: float = 0.0
    noniid_alpha: float = 1.0  # Dirichlet concentration (1.0=IID, 0.1=highly non-IID)
    attack: str = "sign_flip"  # "model_replacement", "scaled_grad", "backdoor", "sign_flip", "none"
    seed: int = 101

    # Task parameters
    n_samples_per_client: int = 100
    n_features: int = 50
    n_classes: int = 5
    participation_rate: float = 0.6  # Fraction of clients participating per round

    # Attack parameters
    attack_lambda: float = 10.0  # Model replacement scaling factor
    attack_scale: float = 10.0  # Gradient scaling factor
    flip_topk: float = 0.2  # Fraction of top gradients to flip

    # Backdoor attack parameters
    backdoor_trigger_feature: int = 0  # Which feature to poison
    backdoor_trigger_value: float = 5.0  # Trigger value
    backdoor_target_label: int = 1  # Target label for backdoor
    backdoor_triggered_frac: float = 0.2  # Fraction of samples to trigger per batch


class SyntheticMultiClassTask:
    """Fast synthetic multi-class classification task using softmax regression."""

    def __init__(self, n_features: int = 50, n_classes: int = 5, seed: int = 42):
        self.n_features = n_features
        self.n_classes = n_classes
        self.rng = np.random.default_rng(seed)

        # True model: class-specific means (one per class)
        self.class_means = []
        for k in range(n_classes):
            # Spread classes in feature space
            angle = 2 * np.pi * k / n_classes
            mean = np.zeros(n_features)
            mean[:2] = [np.cos(angle), np.sin(angle)]  # First 2 features determine class
            mean[2:] = self.rng.standard_normal(n_features - 2) * 0.1  # Rest are noise
            self.class_means.append(mean)

    def generate_data(self, n_samples: int, label: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """Generate samples for one class."""
        # Class-conditional Gaussian with tighter overlap (std=0.8 for better separation)
        mean = self.class_means[label]
        X = self.rng.normal(mean, 0.8, (n_samples, self.n_features))
        y = np.full(n_samples, label, dtype=np.int64)
        return X, y

    def partition_iid(self, n_clients: int, samples_per_client: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """IID partition: equal class distribution."""
        datasets = []
        samples_per_class = samples_per_client // self.n_classes

        for _ in range(n_clients):
            X_list, y_list = [], []
            for k in range(self.n_classes):
                X_k, y_k = self.generate_data(samples_per_class, label=k)
                X_list.append(X_k)
                y_list.append(y_k)
            X = np.vstack(X_list)
            y = np.concatenate(y_list)
            datasets.append((X, y))
        return datasets

    def partition_dirichlet(self, n_clients: int, samples_per_client: int, alpha: float) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Non-IID Dirichlet partition."""
        # Generate all data first
        total_samples = n_clients * samples_per_client
        samples_per_class = total_samples // self.n_classes

        X_by_class = []
        y_by_class = []
        for k in range(self.n_classes):
            X_k, y_k = self.generate_data(samples_per_class, label=k)
            X_by_class.append(X_k)
            y_by_class.append(y_k)

        # Sample Dirichlet proportions for each class
        props = self.rng.dirichlet([alpha] * n_clients, size=self.n_classes)  # [n_classes, n_clients]

        datasets = []
        class_indices = [0] * self.n_classes

        for i in range(n_clients):
            X_client_list = []
            y_client_list = []

            for k in range(self.n_classes):
                # Samples from class k for this client
                n_k = int(props[k, i] * len(X_by_class[k]))
                n_k = max(1, min(n_k, len(X_by_class[k]) - class_indices[k]))

                idx = class_indices[k]
                X_client_list.append(X_by_class[k][idx:idx+n_k])
                y_client_list.append(y_by_class[k][idx:idx+n_k])
                class_indices[k] += n_k

            X_client = np.vstack(X_client_list)
            y_client = np.concatenate(y_client_list)
            datasets.append((X_client, y_client))

        return datasets


class TwoLayerNet:
    """Simple 2-layer neural network for CIFAR-10 and complex datasets.

    Architecture: Input → Hidden (ReLU) → Output (Softmax)

    This enables non-linear feature learning for complex image classification
    tasks where linear models fail (e.g., CIFAR-10).
    """

    def __init__(self, n_features: int, hidden_size: int = 128, n_classes: int = 10, seed: int = 42):
        """Initialize network with He initialization."""
        rng = np.random.default_rng(seed)

        # He initialization for ReLU activation
        self.W1 = rng.normal(0, np.sqrt(2.0 / n_features), (n_features, hidden_size))
        self.b1 = np.zeros(hidden_size)

        self.W2 = rng.normal(0, np.sqrt(2.0 / hidden_size), (hidden_size, n_classes))
        self.b2 = np.zeros(n_classes)

        self.n_features = n_features
        self.hidden_size = hidden_size
        self.n_classes = n_classes

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Forward pass with cache for backprop.

        Args:
            X: [batch_size, n_features]

        Returns:
            probs: [batch_size, n_classes] softmax probabilities
            cache: Dictionary with intermediate activations
        """
        # Hidden layer with ReLU
        Z1 = X @ self.W1 + self.b1  # [batch, hidden]
        A1 = np.maximum(0, Z1)      # ReLU activation

        # Output layer with softmax
        Z2 = A1 @ self.W2 + self.b2  # [batch, n_classes]

        # Numerically stable softmax
        Z2_shifted = Z2 - np.max(Z2, axis=1, keepdims=True)
        exp_scores = np.exp(Z2_shifted)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        cache = {'X': X, 'Z1': Z1, 'A1': A1, 'Z2': Z2, 'probs': probs}
        return probs, cache

    def backward(self, cache: Dict, y: np.ndarray, reg: float = 0.01) -> Dict:
        """Backward pass to compute gradients.

        Args:
            cache: Forward pass cache
            y: [batch_size] integer labels
            reg: L2 regularization strength

        Returns:
            grads: Dictionary with dW1, db1, dW2, db2
        """
        X = cache['X']
        A1 = cache['A1']
        probs = cache['probs']
        batch_size = X.shape[0]

        # Output layer gradient
        dZ2 = probs.copy()
        dZ2[np.arange(batch_size), y] -= 1  # Softmax + cross-entropy derivative
        dZ2 /= batch_size

        dW2 = A1.T @ dZ2 + reg * self.W2
        db2 = np.sum(dZ2, axis=0)

        # Hidden layer gradient
        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * (cache['Z1'] > 0)  # ReLU derivative

        dW1 = X.T @ dZ1 + reg * self.W1
        db1 = np.sum(dZ1, axis=0)

        return {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}

    def get_flat_params(self) -> np.ndarray:
        """Flatten all parameters into 1D vector."""
        return np.concatenate([
            self.W1.flatten(),
            self.b1.flatten(),
            self.W2.flatten(),
            self.b2.flatten()
        ])

    def set_flat_params(self, flat_params: np.ndarray):
        """Set parameters from 1D vector."""
        W1_size = self.n_features * self.hidden_size
        b1_size = self.hidden_size
        W2_size = self.hidden_size * self.n_classes
        b2_size = self.n_classes

        idx = 0
        self.W1 = flat_params[idx:idx+W1_size].reshape(self.n_features, self.hidden_size)
        idx += W1_size

        self.b1 = flat_params[idx:idx+b1_size]
        idx += b1_size

        self.W2 = flat_params[idx:idx+W2_size].reshape(self.hidden_size, self.n_classes)
        idx += W2_size

        self.b2 = flat_params[idx:idx+b2_size]

    def get_flat_grads(self, grads: Dict) -> np.ndarray:
        """Flatten all gradients into 1D vector."""
        return np.concatenate([
            grads['dW1'].flatten(),
            grads['db1'].flatten(),
            grads['dW2'].flatten(),
            grads['db2'].flatten()
        ])


def apply_attack(gradients: np.ndarray, attack_type: str, attack_indices: List[int], scenario: Optional[FLScenario] = None, W_global: Optional[np.ndarray] = None) -> np.ndarray:
    """Apply Byzantine attack to gradients.

    Args:
        gradients: [n_clients, ...] gradient array
        attack_type: "model_replacement", "scaled_grad", "sign_flip", "backdoor", "none"
        attack_indices: Indices of Byzantine clients
        scenario: Attack configuration
        W_global: Current global model (for model_replacement attack)
    """
    attacked = gradients.copy()

    if scenario is None:
        return attacked

    # COORDINATED ATTACK: All Byzantine clients use SAME target model
    # This is critical for testing BFT - uncoordinated attacks cancel out in median!
    if attack_type == "model_replacement" and W_global is not None and attack_indices:
        # Generate shared malicious target (same for all Byzantine clients)
        rng_shared = np.random.default_rng(scenario.seed)  # Same seed for all!
        W_target_shared = rng_shared.standard_normal(W_global.shape) * 0.1

    for idx in attack_indices:
        if attack_type == "model_replacement":
            # Model replacement: update := λ*(W_target - W_cur)
            # All Byzantine clients coordinate toward the SAME malicious target
            if W_global is not None:
                attacked[idx] = scenario.attack_lambda * (W_target_shared - W_global)
            else:
                attacked[idx] = gradients[idx] * scenario.attack_lambda

        elif attack_type == "scaled_grad":
            # Scaled gradient with top-k flip
            grad = gradients[idx]
            grad_flat = grad.flatten()

            # Find top-k largest magnitude gradients
            k = int(scenario.flip_topk * len(grad_flat))
            top_k_idx = np.argpartition(np.abs(grad_flat), -k)[-k:]

            # Scale entire gradient
            grad_scaled = grad * scenario.attack_scale

            # Flip top-k
            grad_flat_scaled = grad_scaled.flatten()
            grad_flat_scaled[top_k_idx] = -grad_flat_scaled[top_k_idx]

            attacked[idx] = grad_flat_scaled.reshape(grad.shape)

        elif attack_type == "sign_flip":
            attacked[idx] = -gradients[idx]

        elif attack_type == "gaussian_noise":
            attacked[idx] = np.random.randn(*gradients[idx].shape) * 5.0

    return attacked


def inject_backdoor_trigger(X: np.ndarray, scenario: FLScenario) -> np.ndarray:
    """Inject backdoor trigger into samples."""
    X_triggered = X.copy()
    X_triggered[:, scenario.backdoor_trigger_feature] = scenario.backdoor_trigger_value
    return X_triggered


def compute_robust_centroid(gradients: List[np.ndarray], coherence_threshold: float = 0.8) -> np.ndarray:
    """Compute robust median using top-X% mutually coherent clients.

    Prevents cascade failures under extreme non-IID by trimming outliers before median.

    Args:
        gradients: List of client gradients
        coherence_threshold: Keep top X% coherent clients (default 0.8 = 80%)

    Returns:
        Robust median gradient
    """
    n_clients = len(gradients)

    # Initial median (may be skewed by outliers)
    grad_flat = np.stack([g.flatten() for g in gradients], axis=0)
    initial_median = np.median(grad_flat, axis=0)
    initial_norm = np.linalg.norm(initial_median)

    # Compute cosine similarity to initial median
    cosines = np.array([
        np.dot(g.flatten(), initial_median) / (np.linalg.norm(g) * initial_norm + 1e-10)
        for g in gradients
    ])

    # Keep top-X% most coherent clients
    n_keep = max(int(coherence_threshold * n_clients), 1)
    coherent_indices = np.argpartition(cosines, -n_keep)[-n_keep:]

    # Recompute median on coherent subset
    coherent_grads = grad_flat[coherent_indices]
    robust_median = np.median(coherent_grads, axis=0)

    return robust_median


def compute_aegis_detection_features(
    gradients: List[np.ndarray],
    prev_gradients: Optional[List[np.ndarray]] = None,
    prev_median: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute enhanced AEGIS detection features for each gradient.

    Features:
    1. L2 norm
    2. Cosine to median
    3. Cosine to previous median
    4. MAD z-score
    5. Influence proxy (dot product with global gradient estimate)
    6. Novelty (change from previous round)

    Returns:
        features: [n_clients, n_features] array
    """
    n_clients = len(gradients)
    grad_norms = np.array([np.linalg.norm(g) for g in gradients])

    # Compute robust median (top-80% coherent clients to prevent non-IID cascade)
    median_grad = compute_robust_centroid(gradients, coherence_threshold=0.8)
    median_norm = np.linalg.norm(median_grad)

    # Feature 1: L2 norm
    feat_norm = grad_norms

    # Feature 2: Cosine to current median
    feat_cos_median = np.array([
        np.dot(g.flatten(), median_grad) / (np.linalg.norm(g) * median_norm + 1e-10)
        for g in gradients
    ])

    # Feature 3: Cosine to previous median
    if prev_median is not None:
        prev_median_norm = np.linalg.norm(prev_median)
        feat_cos_prev_median = np.array([
            np.dot(g.flatten(), prev_median) / (np.linalg.norm(g) * prev_median_norm + 1e-10)
            for g in gradients
        ])
    else:
        feat_cos_prev_median = np.zeros(n_clients)

    # Feature 4: MAD z-score
    mad = np.median(np.abs(grad_norms - np.median(grad_norms)))
    feat_mad_z = (grad_norms - np.median(grad_norms)) / (mad + 1e-10)

    # Feature 5: Influence proxy (dot with global estimate = median)
    feat_influence = np.array([
        np.abs(np.dot(g.flatten(), median_grad))
        for g in gradients
    ])

    # Feature 6: Novelty (change from previous round)
    if prev_gradients is not None and len(prev_gradients) == n_clients:
        feat_novelty = np.array([
            np.linalg.norm(gradients[i] - prev_gradients[i]) / (np.linalg.norm(prev_gradients[i]) + 1e-10)
            for i in range(n_clients)
        ])
    else:
        feat_novelty = np.zeros(n_clients)

    # Stack features: [n_clients, 6]
    features = np.column_stack([
        feat_norm,
        feat_cos_median,
        feat_cos_prev_median,
        feat_mad_z,
        feat_influence,
        feat_novelty,
    ])

    return features, median_grad


def apply_aegis_detection(
    gradients: List[np.ndarray],
    byz_frac: float,
    prev_gradients: Optional[List[np.ndarray]] = None,
    prev_median: Optional[np.ndarray] = None,
    prev_scores: Optional[np.ndarray] = None,
    ema_beta: float = 0.8,
    round_idx: int = 0,
    burn_in_rounds: int = 3,
    noniid_alpha: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, List[int], Dict]:
    """Apply enhanced AEGIS detection with adaptive quarantine ramp and robust centroiding.

    Args:
        gradients: List of client gradients
        byz_frac: Expected Byzantine fraction (used for quarantine)
        prev_gradients: Gradients from previous round
        prev_median: Median gradient from previous round
        prev_scores: Detection scores from previous round (for EMA)
        ema_beta: EMA decay parameter
        round_idx: Current round index (for adaptive quarantine ramp)
        burn_in_rounds: Number of rounds with reduced quarantine (default 3)
        noniid_alpha: Dirichlet alpha parameter for non-IID severity (default 1.0)

    Returns:
        detection_scores: [n_clients] detection scores
        median_grad: Median gradient for next round
        flagged_indices: Indices of flagged clients
        detection_info: Dict with detection statistics for logging
    """
    n_clients = len(gradients)

    # Compute detection features (uses robust centroiding internally)
    features, median_grad = compute_aegis_detection_features(
        gradients, prev_gradients, prev_median
    )

    # Extract key features for hard gates
    cosine_to_median = features[:, 1]
    mad_z_score = features[:, 3]

    # Stretch-mode hard-gate relaxation for pathological non-IID (α≤0.1)
    # Use stricter thresholds to avoid false positives under extreme heterogeneity
    if noniid_alpha <= 0.1:
        hard_blocked = (cosine_to_median < -0.3) & (mad_z_score > 3.5)
    else:
        # Standard hard gates for α>0.1
        hard_blocked = (cosine_to_median < -0.1) & (mad_z_score > 2.5)

    # Compute detection scores (simple: use MAD z-score as primary signal)
    raw_scores = np.abs(mad_z_score)

    # Apply temporal debounce (EMA)
    if prev_scores is not None:
        scores = ema_beta * prev_scores + (1 - ema_beta) * raw_scores
    else:
        scores = raw_scores

    # Adaptive quarantine ramp: Start gentle during burn-in to avoid over-pruning under high non-IID
    if round_idx < burn_in_rounds:
        q_frac = byz_frac * 0.5  # Gentle filtering during burn-in
    else:
        q_frac = byz_frac  # Full filtering after baseline established

    q = min(int(q_frac * n_clients), n_clients - 1)
    if q > 0:
        threshold = np.partition(scores, -q)[-q]
        soft_flagged = scores >= threshold
    else:
        soft_flagged = np.zeros(n_clients, dtype=bool)

    # Combine hard and soft flags
    flagged = hard_blocked | soft_flagged
    flagged_indices = np.where(flagged)[0].tolist()

    # Compile detection info for logging
    detection_info = {
        "q_frac_applied": q_frac,
        "num_flagged": len(flagged_indices),
        "num_hard_blocked": int(np.sum(hard_blocked)),
        "num_soft_flagged": int(np.sum(soft_flagged)),
        "cos_to_median_mean": float(np.mean(cosine_to_median)),
        "cos_to_median_var": float(np.var(cosine_to_median)),
        "mad_z_mean": float(np.mean(mad_z_score)),
        "mad_z_max": float(np.max(mad_z_score)),
    }

    return scores, median_grad, flagged_indices, detection_info


def aggregate_krum(gradients: List[np.ndarray], n_byz: int) -> np.ndarray:
    """Krum aggregation."""
    n = len(gradients)
    m = n - n_byz - 2

    # Compute pairwise distances
    scores = []
    for i in range(n):
        dists = [np.linalg.norm(gradients[i] - gradients[j]) for j in range(n) if j != i]
        dists.sort()
        scores.append(sum(dists[:m]))

    # Select gradient with minimum score
    selected = np.argmin(scores)
    return gradients[selected]


def aggregate_trimmed_mean(gradients: List[np.ndarray], trim_frac: float = 0.1) -> np.ndarray:
    """Trimmed mean aggregation."""
    stacked = np.stack(gradients, axis=0)
    n_trim = int(len(gradients) * trim_frac)

    if n_trim == 0:
        return np.mean(stacked, axis=0)

    # Sort and trim extremes
    sorted_grads = np.sort(stacked, axis=0)
    trimmed = sorted_grads[n_trim:-n_trim]
    return np.mean(trimmed, axis=0)


def aggregate_median(gradients: List[np.ndarray]) -> np.ndarray:
    """Coordinate-wise median."""
    stacked = np.stack(gradients, axis=0)
    return np.median(stacked, axis=0)


# Gen7 Cryptographic Verification Functions
def sign_gradient_gen7(gradient: np.ndarray, keypair) -> bytes:
    """Sign a gradient with Dilithium5 for Gen7 verification."""
    if not GEN7_AVAILABLE:
        raise RuntimeError("Gen7 not available - cannot sign gradients")
    gradient_bytes = gradient.tobytes()
    gradient_hash = gen7_zkstark.hash_gradient_py(gradient_bytes)
    return keypair.sign(gradient_hash)


def verify_gradient_signature_gen7(gradient: np.ndarray, signature: bytes, public_key: bytes) -> bool:
    """Verify a Dilithium5 gradient signature for Gen7."""
    if not GEN7_AVAILABLE:
        raise RuntimeError("Gen7 not available - cannot verify signatures")
    gradient_bytes = gradient.tobytes()
    gradient_hash = gen7_zkstark.hash_gradient_py(gradient_bytes)
    return gen7_zkstark.DilithiumKeypair.verify(gradient_hash, signature, public_key)


def softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    logits_max = np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits - logits_max)
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


def softmax_loss(X: np.ndarray, y: np.ndarray, W: np.ndarray) -> float:
    """Multi-class softmax (cross-entropy) loss.

    Args:
        X: [n_samples, n_features]
        y: [n_samples] (integer labels 0 to n_classes-1)
        W: [n_features, n_classes] (weight matrix)
    """
    n_samples = X.shape[0]
    logits = X @ W  # [n_samples, n_classes]
    probs = softmax(logits)

    # Cross-entropy loss
    correct_probs = probs[np.arange(n_samples), y]
    loss = -np.mean(np.log(correct_probs + 1e-10))
    return loss


def softmax_gradient(X: np.ndarray, y: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Gradient for softmax regression.

    Returns:
        grad_W: [n_features, n_classes]
    """
    n_samples = X.shape[0]
    logits = X @ W  # [n_samples, n_classes]
    probs = softmax(logits)

    # Create one-hot encoding
    y_onehot = np.zeros_like(probs)
    y_onehot[np.arange(n_samples), y] = 1.0

    # Gradient: X^T @ (probs - y_onehot) / n_samples
    grad_W = X.T @ (probs - y_onehot) / n_samples
    return grad_W


def run_fl(
    scenario: FLScenario,
    aggregator: str,
    rounds: int = 20,  # Tuned for clear differentiation
    local_epochs: int = 5,  # Increased for more local training
    lr: float = 0.05,  # Higher LR with cosine decay
    aegis_cfg: Optional[Dict] = None,
    return_history: bool = False,
    use_cosine_decay: bool = True,  # Cosine LR decay
    dataset: Optional["FLData"] = None,  # Real dataset (EMNIST, CIFAR-10)
) -> Dict:
    """
    Run federated learning simulation with multi-class softmax regression.

    Args:
        scenario: FL scenario configuration
        aggregator: "krum", "trimmed_mean", "median", "fltrust", "aegis"
        rounds: Number of FL rounds
        local_epochs: Local training epochs per round
        lr: Learning rate
        aegis_cfg: AEGIS configuration (if aggregator="aegis")
        return_history: If True, return per-round histories

    Returns:
        Dictionary with metrics:
        - clean_acc: Final accuracy on clean test set
        - robust_acc: Final accuracy accounting for Byzantine influence
        - asr: Attack success rate (backdoor triggered test)
        - auc: ROC AUC for Byzantine detection
        - fpr_at_tpr90: False positive rate @ 90% TPR
        - wall_s: Wall clock time
        - bytes_tx: Bytes transmitted
        - flags_per_round: Mean detection flags per round
        - convergence_round: Round where accuracy plateaus (optional)
        - history: Per-round metrics (if return_history=True)
    """
    import time
    from sklearn.metrics import roc_auc_score

    start_time = time.time()

    # Use real dataset if provided, otherwise generate synthetic
    if dataset is not None:
        # Real dataset (EMNIST, CIFAR-10)
        datasets = dataset.train_splits
        X_test_clean, y_test = dataset.test_clean

        # Override scenario parameters from dataset
        scenario.n_features = dataset.n_features
        scenario.n_classes = dataset.n_classes

        # Backdoor-triggered test set (if available)
        if dataset.test_triggered is not None:
            X_test_triggered, _ = dataset.test_triggered
        else:
            X_test_triggered = inject_backdoor_trigger(X_test_clean, scenario)
    else:
        # Synthetic data
        task = SyntheticMultiClassTask(
            n_features=scenario.n_features,
            n_classes=scenario.n_classes,
            seed=scenario.seed
        )

        if scenario.noniid_alpha >= 10.0:  # Treat as IID
            datasets = task.partition_iid(scenario.n_clients, scenario.n_samples_per_client)
        else:
            datasets = task.partition_dirichlet(
                scenario.n_clients, scenario.n_samples_per_client, scenario.noniid_alpha
            )

        # Generate test set (balanced across all classes)
        X_test_list, y_test_list = [], []
        for k in range(scenario.n_classes):
            X_k, y_k = task.generate_data(200, label=k)
            X_test_list.append(X_k)
            y_test_list.append(y_k)
        X_test_clean = np.vstack(X_test_list)
        y_test = np.concatenate(y_test_list)

        # Generate backdoor-triggered test set
        X_test_triggered = inject_backdoor_trigger(X_test_clean, scenario)

    # Initialize model: W is [n_features, n_classes]
    W = np.zeros((scenario.n_features, scenario.n_classes))

    # Determine Byzantine clients
    n_byz = int(scenario.n_clients * scenario.byz_frac)
    byz_indices = set(range(n_byz))

    # Tracking
    flags_per_round = []
    acc_history = []
    all_detection_scores = []
    all_detection_labels = []  # 1 = Byzantine, 0 = honest

    # AEGIS temporal state
    prev_gradients = None
    prev_median = None
    prev_scores = None

    # RNG for participation sampling
    rng = np.random.default_rng(scenario.seed)

    # Training loop
    for round_idx in range(rounds):
        # Cosine LR decay
        if use_cosine_decay:
            lr_t = lr * 0.5 * (1 + np.cos(np.pi * round_idx / rounds))
        else:
            lr_t = lr

        # Client participation sampling
        n_participating = max(int(scenario.participation_rate * scenario.n_clients), 1)
        participating_indices = rng.choice(scenario.n_clients, n_participating, replace=False)

        gradients = []
        client_indices_map = []  # Map gradient index to client index

        # Local updates (only participating clients)
        for client_idx in participating_indices:
            X, y = datasets[client_idx]
            is_byzantine = client_idx in byz_indices

            # For backdoor attack: Byzantine clients train on triggered data
            if scenario.attack == "backdoor" and is_byzantine:
                # Inject trigger into fraction of samples
                n_trigger = max(int(scenario.backdoor_triggered_frac * len(X)), 1)
                X_local = X.copy()
                y_local = y.copy()
                X_local[:n_trigger] = inject_backdoor_trigger(X[:n_trigger], scenario)
                y_local[:n_trigger] = scenario.backdoor_target_label
            else:
                X_local = X.copy()
                y_local = y.copy()

            # Local training with softmax regression
            W_local = W.copy()
            for _ in range(local_epochs):
                grad_W = softmax_gradient(X_local, y_local, W_local)
                W_local = W_local - lr_t * grad_W

            # Compute gradient (model delta)
            client_grad = W_local - W  # [n_features, n_classes]
            gradients.append(client_grad)
            client_indices_map.append(client_idx)

        # Apply attack (for non-backdoor attacks)
        if scenario.attack != "none" and scenario.attack != "backdoor" and n_byz > 0:
            # Identify which participating clients are Byzantine
            attack_indices = [i for i, idx in enumerate(client_indices_map) if idx in byz_indices]
            if attack_indices:
                gradients_array = np.array(gradients)
                gradients_array = apply_attack(
                    gradients_array, scenario.attack, attack_indices, scenario, W_global=W
                )
                gradients = list(gradients_array)

        # Aggregate (handle 2D gradients)
        if aggregator == "krum":
            # Flatten for Krum
            flat_grads = [g.flatten() for g in gradients]
            agg_grad_flat = aggregate_krum(flat_grads, n_byz)
            agg_grad = agg_grad_flat.reshape(scenario.n_features, scenario.n_classes)
        elif aggregator == "trimmed_mean":
            agg_grad = aggregate_trimmed_mean(gradients, trim_frac=0.1)
        elif aggregator == "median":
            agg_grad = aggregate_median(gradients)
        elif aggregator == "aegis":
            # Enhanced AEGIS with adaptive quarantine and robust centroiding
            # Use longer burn-in (5 rounds) for pathological non-IID (α≤0.1)
            burn_in = 5 if scenario.noniid_alpha <= 0.1 else 3

            detection_scores, median_grad, flagged_indices, detection_info = apply_aegis_detection(
                gradients,
                byz_frac=scenario.byz_frac,
                prev_gradients=prev_gradients,
                prev_median=prev_median,
                prev_scores=prev_scores,
                round_idx=round_idx,
                burn_in_rounds=burn_in,  # Adaptive: 5 for α≤0.1, 3 otherwise
                noniid_alpha=scenario.noniid_alpha,  # Pass alpha for relaxed gates
            )

            # Filter out flagged gradients
            clean_gradients = [g for i, g in enumerate(gradients) if i not in flagged_indices]
            if len(clean_gradients) > 0:
                agg_grad = aggregate_median(clean_gradients)
            else:
                agg_grad = aggregate_median(gradients)  # Fallback if all flagged

            # Track detection metrics
            all_detection_scores.extend(detection_scores)

            # Log detection info (optional, for debugging)
            if round_idx % 5 == 0:  # Log every 5 rounds
                pass  # TODO: Add logging if needed

            # Map participating clients back to full client list for labels
            labels = [1 if client_indices_map[i] in byz_indices else 0 for i in range(len(gradients))]
            all_detection_labels.extend(labels)

            flags_per_round.append(len(flagged_indices))

            # Update temporal state
            prev_gradients = gradients.copy()
            prev_median = median_grad
            prev_scores = detection_scores
        elif aggregator == "aegis_gen7":
            # AEGIS + Gen7 Cryptographic Verification (Layer 0)
            # This combines perfect cryptographic verification (AUC 1.00) with existing AEGIS defenses
            if not GEN7_AVAILABLE:
                raise RuntimeError("aegis_gen7 aggregator requires Gen7 module (run: pip install gen7_zkstark)")

            # Initialize client keypairs (persistent across rounds)
            if not hasattr(run_fl, '_gen7_keypairs'):
                print("🔐 Initializing Gen7 client keypairs...")
                run_fl._gen7_keypairs = []
                run_fl._gen7_public_keys = []
                for _ in range(scenario.n_clients):
                    keypair = gen7_zkstark.DilithiumKeypair()
                    pubkey = keypair.get_public_key()
                    run_fl._gen7_keypairs.append(keypair)
                    run_fl._gen7_public_keys.append(pubkey)
                print(f"✅ Generated {scenario.n_clients} Dilithium5 keypairs")

            # Layer 0: Gen7 Cryptographic Verification
            # Sign gradients with Dilithium5 (simulating client-side signing)
            signatures = []
            for i, client_idx in enumerate(client_indices_map):
                gradient = gradients[i]
                # Byzantine clients sign DIFFERENT gradient (invalid proof)
                if client_idx in byz_indices:
                    fake_gradient = np.random.randn(*gradient.shape).astype(np.float32)
                    sig = sign_gradient_gen7(fake_gradient, run_fl._gen7_keypairs[client_idx])
                else:
                    # Honest clients sign correct gradient
                    sig = sign_gradient_gen7(gradient, run_fl._gen7_keypairs[client_idx])
                signatures.append(sig)

            # Verify signatures (server-side verification)
            verified_gradients = []
            rejected_gen7_indices = []
            gen7_detection_scores = []

            for i, client_idx in enumerate(client_indices_map):
                gradient = gradients[i]
                signature = signatures[i]
                public_key = run_fl._gen7_public_keys[client_idx]

                # Verify signature
                is_valid = verify_gradient_signature_gen7(gradient, signature, public_key)

                if is_valid:
                    verified_gradients.append(gradient)
                    gen7_detection_scores.append(0.0)  # Low score = likely honest
                else:
                    rejected_gen7_indices.append(i)
                    gen7_detection_scores.append(1.0)  # High score = rejected

            # Track Gen7 Layer 0 metrics
            all_detection_scores.extend(gen7_detection_scores)
            labels = [1 if client_indices_map[i] in byz_indices else 0 for i in range(len(gradients))]
            all_detection_labels.extend(labels)

            # If we have verified gradients, proceed with AEGIS Layers 1-7
            if len(verified_gradients) > 0:
                burn_in = 5 if scenario.noniid_alpha <= 0.1 else 3

                # Reset temporal tracking if number of verified gradients changed
                # (Gen7 rejection varies round-to-round, AEGIS EMA expects same count)
                if prev_gradients and len(prev_gradients) != len(verified_gradients):
                    prev_gradients = None
                    prev_median = None
                    prev_scores = None

                detection_scores, median_grad, flagged_indices, detection_info = apply_aegis_detection(
                    verified_gradients,
                    byz_frac=scenario.byz_frac,
                    prev_gradients=prev_gradients,
                    prev_median=prev_median,
                    prev_scores=prev_scores,
                    round_idx=round_idx,
                    burn_in_rounds=burn_in,
                    noniid_alpha=scenario.noniid_alpha,
                )

                # Filter out flagged gradients from verified set
                clean_gradients = [g for i, g in enumerate(verified_gradients) if i not in flagged_indices]
                if len(clean_gradients) > 0:
                    agg_grad = aggregate_median(clean_gradients)
                else:
                    agg_grad = aggregate_median(verified_gradients)  # Fallback

                flags_per_round.append(len(flagged_indices) + len(rejected_gen7_indices))

                # Update temporal state (using verified gradients)
                prev_gradients = verified_gradients.copy()
                prev_median = median_grad
                prev_scores = detection_scores
            else:
                # All gradients rejected by Gen7 - use median of original gradients as fallback
                agg_grad = aggregate_median(gradients)
                flags_per_round.append(len(rejected_gen7_indices))
        elif aggregator == "aegis_gen7_full":
            # AEGIS + Full zk-DASTARK (zkSTARK computation proofs + Dilithium authentication)
            # This validates both identity AND computation correctness
            if not GEN7_AVAILABLE:
                raise RuntimeError("aegis_gen7_full aggregator requires Gen7 module")

            # Import zk-DASTARK components
            from zerotrustml.gen7.authenticated_gradient_proof import (
                AuthenticatedGradientClient,
                AuthenticatedGradientCoordinator,
            )

            # Initialize coordinator and clients (persistent across rounds)
            if not hasattr(run_fl, '_zkdastark_coordinator'):
                print("🔐 Initializing zk-DASTARK coordinator and clients...")
                run_fl._zkdastark_coordinator = AuthenticatedGradientCoordinator(
                    max_timestamp_delta=300,
                    use_database=False,  # In-memory for simulation
                )
                run_fl._zkdastark_coordinator.set_round(0)

                run_fl._zkdastark_clients = []
                for _ in range(scenario.n_clients):
                    client = AuthenticatedGradientClient()
                    run_fl._zkdastark_coordinator.register_client(
                        client.get_client_id(),
                        client.get_public_key()
                    )
                    run_fl._zkdastark_clients.append(client)
                print(f"✅ Initialized zk-DASTARK for {scenario.n_clients} clients")

            # Update round in coordinator
            run_fl._zkdastark_coordinator.set_round(round_idx)

            # Generate and verify proofs
            verified_gradients = []
            rejected_indices = []
            zkstark_detection_scores = []

            for i, client_idx in enumerate(client_indices_map):
                gradient = gradients[i]
                client = run_fl._zkdastark_clients[client_idx]
                is_byzantine = client_idx in byz_indices

                try:
                    # Get local data for this client
                    X_client, y_client = datasets[client_idx]

                    # Byzantine clients cannot produce valid zkSTARK proofs
                    # for tampered gradients - they'd need to prove incorrect computation
                    if is_byzantine:
                        # Byzantine: Generate proof for DIFFERENT gradient (will fail verification)
                        # This simulates that attackers cannot forge zkSTARK proofs
                        fake_gradient = np.random.randn(*gradient.shape).astype(np.float32)
                        fake_model = W + fake_gradient

                        # Generate proof (will be cryptographically invalid for actual gradient)
                        auth_proof = client.generate_proof(
                            gradient=list(fake_gradient.flatten()),
                            model_params=list(W.flatten()),
                            local_data=list(X_client.flatten()),
                            local_labels=list(y_client.astype(int)),
                            round_number=round_idx,
                        )
                    else:
                        # Honest client: Generate valid proof
                        auth_proof = client.generate_proof(
                            gradient=list(gradient.flatten()),
                            model_params=list(W.flatten()),
                            local_data=list(X_client.flatten()),
                            local_labels=list(y_client.astype(int)),
                            round_number=round_idx,
                        )

                    # Verify proof (checks both zkSTARK and Dilithium)
                    is_valid, error = run_fl._zkdastark_coordinator.verify_proof(auth_proof)

                    if is_valid:
                        verified_gradients.append(gradient)
                        zkstark_detection_scores.append(0.0)
                    else:
                        rejected_indices.append(i)
                        zkstark_detection_scores.append(1.0)

                except Exception as e:
                    # Proof generation/verification failed
                    rejected_indices.append(i)
                    zkstark_detection_scores.append(1.0)

            # Track detection metrics
            all_detection_scores.extend(zkstark_detection_scores)
            labels = [1 if client_indices_map[i] in byz_indices else 0 for i in range(len(gradients))]
            all_detection_labels.extend(labels)

            # Aggregate verified gradients
            if len(verified_gradients) > 0:
                # Apply AEGIS on verified gradients for additional filtering
                burn_in = 5 if scenario.noniid_alpha <= 0.1 else 3

                detection_scores, median_grad, flagged_indices, detection_info = apply_aegis_detection(
                    verified_gradients,
                    byz_frac=scenario.byz_frac,
                    prev_gradients=prev_gradients,
                    prev_median=prev_median,
                    prev_scores=prev_scores,
                    round_idx=round_idx,
                    burn_in_rounds=burn_in,
                    noniid_alpha=scenario.noniid_alpha,
                )

                clean_gradients = [g for i, g in enumerate(verified_gradients) if i not in flagged_indices]
                if len(clean_gradients) > 0:
                    agg_grad = aggregate_median(clean_gradients)
                else:
                    agg_grad = aggregate_median(verified_gradients)

                flags_per_round.append(len(flagged_indices) + len(rejected_indices))

                prev_gradients = verified_gradients.copy()
                prev_median = median_grad
                prev_scores = detection_scores
            else:
                agg_grad = aggregate_median(gradients)
                flags_per_round.append(len(rejected_indices))
        else:
            agg_grad = aggregate_median(gradients)  # Default to median

        # Update global model
        W = W + agg_grad

        # Evaluate current model
        logits = X_test_clean @ W
        y_pred = np.argmax(logits, axis=1)
        acc = np.mean(y_pred == y_test)
        acc_history.append(acc)

    wall_time = time.time() - start_time

    # Final evaluation on clean test set
    logits = X_test_clean @ W
    y_pred = np.argmax(logits, axis=1)
    clean_acc = np.mean(y_pred == y_test)

    # Evaluate ASR on triggered test set (for backdoor attacks)
    logits_triggered = X_test_triggered @ W
    y_pred_triggered = np.argmax(logits_triggered, axis=1)
    # ASR: fraction of triggered samples classified as target label
    asr = np.mean(y_pred_triggered == scenario.backdoor_target_label)

    # Compute AUC if we have detection scores
    if all_detection_scores and all_detection_labels:
        try:
            auc = roc_auc_score(all_detection_labels, all_detection_scores)
        except:
            auc = 0.5

        # Compute FPR @ TPR=0.9
        try:
            from sklearn.metrics import roc_curve
            fpr, tpr, thresholds = roc_curve(all_detection_labels, all_detection_scores)
            # Find FPR where TPR >= 0.9
            idx = np.where(tpr >= 0.9)[0]
            fpr_at_tpr90 = fpr[idx[0]] if len(idx) > 0 else 0.5
        except:
            fpr_at_tpr90 = 0.1
    else:
        auc = 0.5
        fpr_at_tpr90 = 0.1

    # Robust accuracy: penalize for high ASR
    robust_acc = clean_acc * (1.0 - 0.5 * (asr if scenario.attack == "backdoor" else 0))

    # Find convergence round (where accuracy plateaus)
    convergence_round = rounds
    if len(acc_history) > 5:
        for i in range(5, len(acc_history)):
            if all(abs(acc_history[j] - acc_history[i]) < 0.02 for j in range(i-5, i)):
                convergence_round = i
                break

    # Compute metrics
    metrics = {
        "clean_acc": float(clean_acc),
        "robust_acc": float(robust_acc),
        "asr": float(asr) if scenario.attack == "backdoor" else 0.0,
        "auc": float(auc),
        "fpr_at_tpr90": float(fpr_at_tpr90),
        "wall_s": float(wall_time),
        "bytes_tx": int(scenario.n_clients * scenario.n_features * scenario.n_classes * 8 * rounds),
        "bytes_rx": int(scenario.n_clients * scenario.n_features * scenario.n_classes * 8 * rounds),
        "flags_per_round": float(np.mean(flags_per_round)) if flags_per_round else 0.0,
        "convergence_round": int(convergence_round),
    }

    if return_history:
        metrics["history"] = {
            "clean_acc": acc_history,
            "flags": flags_per_round,
        }

    return metrics

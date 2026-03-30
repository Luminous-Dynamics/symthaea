#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Sparse Autoencoder (SAE) for Monosemantic Feature Extraction

Method 2 of the Neural Bridge: Train a sparse autoencoder on LLM activations
to extract interpretable, monosemantic features that can be individually
mapped to HDC space.

Key Insight: LLM neurons are polysemantic (respond to multiple concepts).
SAE features are monosemantic (one concept per feature), making them
ideal for HDC mapping.

Architecture:
    activation [d_model] → encoder [d_sae] → ReLU → decoder [d_model]

Where:
    d_sae >> d_model (e.g., 768 → 8192 or 16384)

The encoder learns sparse feature activations that can be interpreted
and mapped to HDC space.

Usage:
    python sparse_autoencoder.py --activations data/sae/activations.npz \\
                                 --expansion-factor 16 \\
                                 --sparsity 0.01 \\
                                 --output models/sae/

References:
    - Anthropic's "Towards Monosemanticity" (2023)
    - OpenAI's "Language models can explain neurons in language models" (2023)
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Symthaea HDC dimension
HDC_DIMENSION = 16_384


class SparseAutoencoder:
    """
    Sparse Autoencoder for monosemantic feature extraction.

    Uses L1 regularization to encourage sparsity in the hidden layer.

    Architecture:
        encoder: d_model → d_sae (linear + bias)
        activation: ReLU
        decoder: d_sae → d_model (linear, no bias typically)
    """

    def __init__(
        self,
        d_model: int,
        d_sae: int,
        l1_coefficient: float = 0.01,
        learning_rate: float = 0.001,
    ):
        """
        Initialize SAE.

        Args:
            d_model: Input/output dimension (LLM hidden size)
            d_sae: SAE hidden dimension (typically 8-32x d_model)
            l1_coefficient: L1 sparsity penalty weight
            learning_rate: Learning rate for training
        """
        self.d_model = d_model
        self.d_sae = d_sae
        self.l1_coefficient = l1_coefficient
        self.learning_rate = learning_rate

        # Initialize weights (Kaiming initialization)
        self.W_enc = np.random.randn(d_sae, d_model).astype(np.float32)
        self.W_enc *= np.sqrt(2.0 / d_model)

        self.b_enc = np.zeros(d_sae, dtype=np.float32)

        # Decoder weights (tied or untied)
        self.W_dec = np.random.randn(d_model, d_sae).astype(np.float32)
        self.W_dec *= np.sqrt(2.0 / d_sae)

        self.b_dec = np.zeros(d_model, dtype=np.float32)

        # Normalize decoder columns (important for SAE training)
        self._normalize_decoder()

        # Training stats
        self.training_history: List[Dict] = []

    def _normalize_decoder(self):
        """Normalize decoder column norms to 1."""
        norms = np.linalg.norm(self.W_dec, axis=0, keepdims=True)
        self.W_dec = self.W_dec / (norms + 1e-8)

    def encode(self, x: np.ndarray) -> np.ndarray:
        """
        Encode input to sparse features.

        Args:
            x: Input activations [batch, d_model] or [d_model]

        Returns:
            Sparse feature activations [batch, d_sae] or [d_sae]
        """
        # Linear + ReLU
        pre_activation = x @ self.W_enc.T + self.b_enc
        return np.maximum(0, pre_activation)  # ReLU

    def decode(self, z: np.ndarray) -> np.ndarray:
        """
        Decode sparse features back to activation space.

        Args:
            z: Sparse features [batch, d_sae] or [d_sae]

        Returns:
            Reconstructed activations [batch, d_model] or [d_model]
        """
        return z @ self.W_dec.T + self.b_dec

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Full forward pass.

        Returns:
            (reconstruction, sparse_features, pre_activation)
        """
        pre_activation = x @ self.W_enc.T + self.b_enc
        sparse_features = np.maximum(0, pre_activation)
        reconstruction = sparse_features @ self.W_dec.T + self.b_dec
        return reconstruction, sparse_features, pre_activation

    def loss(self, x: np.ndarray) -> Tuple[float, float, float, np.ndarray]:
        """
        Compute loss components.

        Returns:
            (total_loss, reconstruction_loss, sparsity_loss, sparse_features)
        """
        reconstruction, sparse_features, _ = self.forward(x)

        # Reconstruction loss (MSE)
        reconstruction_loss = np.mean((x - reconstruction) ** 2)

        # L1 sparsity loss
        sparsity_loss = self.l1_coefficient * np.mean(np.abs(sparse_features))

        total_loss = reconstruction_loss + sparsity_loss

        return total_loss, reconstruction_loss, sparsity_loss, sparse_features

    def train_step(self, x: np.ndarray) -> Dict[str, float]:
        """
        Single training step with gradient descent.

        Args:
            x: Batch of activations [batch, d_model]

        Returns:
            Dict with loss components
        """
        batch_size = x.shape[0]

        # Forward pass
        reconstruction, sparse_features, pre_activation = self.forward(x)

        # Compute losses
        reconstruction_loss = np.mean((x - reconstruction) ** 2)
        sparsity_loss = self.l1_coefficient * np.mean(np.abs(sparse_features))
        total_loss = reconstruction_loss + sparsity_loss

        # Backward pass (manual gradients)
        # d_loss/d_reconstruction = 2 * (reconstruction - x) / (batch * d_model)
        d_reconstruction = 2 * (reconstruction - x) / (batch_size * self.d_model)

        # Decoder gradients
        d_W_dec = d_reconstruction.T @ sparse_features / batch_size
        d_b_dec = np.mean(d_reconstruction, axis=0)

        # Backprop through decoder
        d_sparse = d_reconstruction @ self.W_dec

        # L1 gradient (subgradient at 0)
        d_sparse += (
            self.l1_coefficient * np.sign(sparse_features) / (batch_size * self.d_sae)
        )

        # ReLU gradient
        d_pre = d_sparse * (pre_activation > 0).astype(np.float32)

        # Encoder gradients
        d_W_enc = d_pre.T @ x / batch_size
        d_b_enc = np.mean(d_pre, axis=0)

        # Update weights
        self.W_enc -= self.learning_rate * d_W_enc
        self.b_enc -= self.learning_rate * d_b_enc
        self.W_dec -= self.learning_rate * d_W_dec
        self.b_dec -= self.learning_rate * d_b_dec

        # Re-normalize decoder
        self._normalize_decoder()

        # Compute sparsity metrics
        active_features = np.mean(sparse_features > 0)
        mean_activation = (
            np.mean(sparse_features[sparse_features > 0]) if active_features > 0 else 0
        )

        return {
            "total_loss": float(total_loss),
            "reconstruction_loss": float(reconstruction_loss),
            "sparsity_loss": float(sparsity_loss),
            "active_features": float(active_features),
            "mean_activation": float(mean_activation),
        }

    def train(
        self,
        activations: np.ndarray,
        epochs: int = 100,
        batch_size: int = 64,
        verbose: bool = True,
    ) -> List[Dict]:
        """
        Train SAE on activation dataset.

        Args:
            activations: Training data [n_samples, d_model]
            epochs: Number of training epochs
            batch_size: Batch size
            verbose: Print progress

        Returns:
            Training history
        """
        n_samples = activations.shape[0]
        history = []

        for epoch in range(epochs):
            # Shuffle
            perm = np.random.permutation(n_samples)
            activations_shuffled = activations[perm]

            epoch_losses = []

            for i in range(0, n_samples, batch_size):
                batch = activations_shuffled[i : i + batch_size]
                metrics = self.train_step(batch)
                epoch_losses.append(metrics)

            # Average epoch metrics
            avg_metrics = {
                k: np.mean([m[k] for m in epoch_losses]) for k in epoch_losses[0].keys()
            }
            history.append(avg_metrics)

            if verbose and ((epoch + 1) % 10 == 0 or epoch == 0):
                print(
                    f"  Epoch {epoch+1:3d}: "
                    f"loss={avg_metrics['total_loss']:.4f}, "
                    f"recon={avg_metrics['reconstruction_loss']:.4f}, "
                    f"sparse={avg_metrics['sparsity_loss']:.4f}, "
                    f"active={avg_metrics['active_features']:.2%}"
                )

        self.training_history = history
        return history

    def get_feature_activations(self, x: np.ndarray) -> np.ndarray:
        """Get sparse feature activations for input."""
        return self.encode(x)

    def get_top_features(self, x: np.ndarray, k: int = 10) -> List[Tuple[int, float]]:
        """
        Get top-k active features for an input.

        Returns:
            List of (feature_idx, activation_value) tuples
        """
        features = self.encode(x)
        if features.ndim > 1:
            features = features.mean(axis=0)

        top_indices = np.argsort(features)[-k:][::-1]
        return [(int(i), float(features[i])) for i in top_indices if features[i] > 0]

    def save(self, path: Path):
        """Save SAE weights and config."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        np.save(path / "W_enc.npy", self.W_enc)
        np.save(path / "b_enc.npy", self.b_enc)
        np.save(path / "W_dec.npy", self.W_dec)
        np.save(path / "b_dec.npy", self.b_dec)

        config = {
            "d_model": self.d_model,
            "d_sae": self.d_sae,
            "l1_coefficient": self.l1_coefficient,
            "learning_rate": self.learning_rate,
        }
        with open(path / "config.json", "w") as f:
            json.dump(config, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "SparseAutoencoder":
        """Load SAE from saved weights."""
        path = Path(path)

        with open(path / "config.json") as f:
            config = json.load(f)

        sae = cls(**config)
        sae.W_enc = np.load(path / "W_enc.npy")
        sae.b_enc = np.load(path / "b_enc.npy")
        sae.W_dec = np.load(path / "W_dec.npy")
        sae.b_dec = np.load(path / "b_dec.npy")

        return sae


class SAEtoHDCBridge:
    """
    Bridge SAE features to HDC space.

    Each SAE feature gets a unique HDC target vector.
    The final HDC representation is the bundling (sum) of active
    feature vectors weighted by their activation.
    """

    def __init__(self, sae: SparseAutoencoder, seed: int = 42):
        """
        Initialize bridge.

        Args:
            sae: Trained sparse autoencoder
            seed: Random seed for generating feature HDC vectors
        """
        self.sae = sae
        self.seed = seed

        # Generate HDC vectors for each SAE feature
        rng = np.random.RandomState(seed)
        self.feature_vectors = rng.choice(
            [-1, 1], size=(sae.d_sae, HDC_DIMENSION)
        ).astype(np.float32)

    def project(self, activation: np.ndarray) -> np.ndarray:
        """
        Project LLM activation to HDC space via SAE features.

        1. Encode activation to sparse features
        2. Bundle feature HDC vectors weighted by activation

        Args:
            activation: LLM activation [d_model]

        Returns:
            HDC vector [HDC_DIMENSION]
        """
        # Get sparse feature activations
        features = self.sae.encode(activation)  # [d_sae]

        # Weighted bundle: sum of (feature_activation * feature_hdc)
        # features: [d_sae], feature_vectors: [d_sae, HDC_DIM]
        hdc = features @ self.feature_vectors  # [HDC_DIM]

        return hdc

    def project_to_bipolar(self, activation: np.ndarray) -> np.ndarray:
        """Project and convert to bipolar {-1, +1}."""
        hdc = self.project(activation)
        return np.sign(hdc)

    def explain_projection(self, activation: np.ndarray, k: int = 10) -> List[Dict]:
        """
        Explain which features contributed to the HDC projection.

        Returns list of top-k contributing features with their
        activation values and HDC influence.
        """
        top_features = self.sae.get_top_features(activation, k)

        explanations = []
        for feature_idx, activation_value in top_features:
            explanations.append(
                {
                    "feature_idx": feature_idx,
                    "activation": activation_value,
                    "contribution": activation_value
                    / max(sum(a for _, a in top_features), 1e-8),
                }
            )

        return explanations


def collect_activations_for_sae(
    model_name: str = "embeddinggemma:300m",
    n_samples: int = 10000,
) -> np.ndarray:
    """
    Collect diverse activations for SAE training.

    For a proper SAE, we need many diverse activations from
    running the model on varied text.
    """
    import json as json_module
    import urllib.request

    print(f"Collecting {n_samples} activations from {model_name}...")

    # Sample text sources (in practice, use a large corpus)
    text_sources = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning algorithms process data patterns.",
        "Democracy enables citizens to participate in governance.",
        "Mitochondria generate ATP through cellular respiration.",
        "NixOS provides declarative system configuration.",
        "Consciousness arises from complex neural activity.",
        "Evolution shapes species through natural selection.",
        "Cryptography secures digital communication channels.",
        "Photosynthesis converts sunlight into chemical energy.",
        "Justice requires fair and impartial treatment.",
    ] * (n_samples // 10 + 1)

    activations = []

    for i, text in enumerate(text_sources[:n_samples]):
        data = json_module.dumps({"model": model_name, "input": text}).encode("utf-8")
        req = urllib.request.Request(
            "http://localhost:11434/api/embed",
            data=data,
            headers={"Content-Type": "application/json"},
        )

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json_module.loads(resp.read().decode("utf-8"))
                embedding = np.array(result["embeddings"][0], dtype=np.float32)
                activations.append(embedding)

                if (i + 1) % 100 == 0:
                    print(f"  Collected {i + 1}/{n_samples} activations")

        except Exception as e:
            print(f"  Warning: Failed to get embedding: {e}")

    return np.array(activations)


def main():
    parser = argparse.ArgumentParser(
        description="Train Sparse Autoencoder for monosemantic features"
    )
    parser.add_argument(
        "--activations", type=Path, help="Path to activation dataset (or collect new)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/sae"),
        help="Output directory for trained SAE",
    )
    parser.add_argument(
        "--expansion-factor",
        type=int,
        default=16,
        help="d_sae / d_model ratio (default: 16)",
    )
    parser.add_argument(
        "--l1-coefficient",
        type=float,
        default=0.01,
        help="L1 sparsity penalty (default: 0.01)",
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Training epochs (default: 100)"
    )
    parser.add_argument(
        "--collect-samples",
        type=int,
        default=0,
        help="Collect this many new samples instead of loading",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="embeddinggemma:300m",
        help="Model for collecting activations",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  Sparse Autoencoder Training")
    print("  Method 2: Monosemantic Feature Extraction")
    print("=" * 60)
    print()

    # Load or collect activations
    if args.collect_samples > 0:
        activations = collect_activations_for_sae(args.model, args.collect_samples)
        d_model = activations.shape[1]
    elif args.activations and args.activations.exists():
        print(f"Loading activations from {args.activations}...")
        data = np.load(args.activations, allow_pickle=True)
        activations = data["activations"]
        d_model = activations.shape[1]
    else:
        # Use concept activations as training data
        print("Using concept activations for training...")
        concept_path = Path("data/neural_bridge/concept_activations_embeddinggemma.npz")
        if not concept_path.exists():
            print("ERROR: No activation data found. Run activation_extractor.py first.")
            return

        data = np.load(concept_path, allow_pickle=True)
        activations = data["activations"]
        d_model = activations.shape[1]

        # Expand by adding noise variations
        print("  Augmenting with noise variations...")
        augmented = [activations]
        for _ in range(9):  # 10x augmentation
            noisy = (
                activations
                + np.random.randn(*activations.shape).astype(np.float32) * 0.1
            )
            augmented.append(noisy)
        activations = np.vstack(augmented)

    print(f"  Samples: {activations.shape[0]}")
    print(f"  d_model: {d_model}")

    # Create SAE
    d_sae = d_model * args.expansion_factor
    print(f"\nCreating SAE:")
    print(f"  d_sae: {d_sae} ({args.expansion_factor}x expansion)")
    print(f"  L1 coefficient: {args.l1_coefficient}")

    sae = SparseAutoencoder(
        d_model=d_model,
        d_sae=d_sae,
        l1_coefficient=args.l1_coefficient,
        learning_rate=0.001,
    )

    # Train
    print(f"\nTraining for {args.epochs} epochs...")
    history = sae.train(activations, epochs=args.epochs, batch_size=32)

    # Save
    args.output.mkdir(parents=True, exist_ok=True)
    sae.save(args.output)

    print()
    print("=" * 60)
    print("  Training Complete!")
    print("=" * 60)
    print()
    print(f"SAE saved to: {args.output}")
    print(f"Final metrics:")
    print(f"  Reconstruction loss: {history[-1]['reconstruction_loss']:.4f}")
    print(f"  Sparsity loss: {history[-1]['sparsity_loss']:.4f}")
    print(f"  Active features: {history[-1]['active_features']:.2%}")

    # Test HDC bridge
    print("\nTesting SAE → HDC Bridge:")
    bridge = SAEtoHDCBridge(sae)

    test_activation = activations[0]
    hdc = bridge.project_to_bipolar(test_activation)

    print(f"  Input: {test_activation.shape}")
    print(f"  Output: {hdc.shape}")
    print(
        f"  Active features: {len(bridge.sae.get_top_features(test_activation, k=100))}"
    )

    # Show top features
    print("\n  Top contributing features:")
    for exp in bridge.explain_projection(test_activation, k=5):
        print(
            f"    Feature {exp['feature_idx']:4d}: activation={exp['activation']:.4f}, contribution={exp['contribution']:.1%}"
        )


if __name__ == "__main__":
    main()

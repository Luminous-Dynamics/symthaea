#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
BGE-M3 Hyperdimensional Probe Training

Trains a linear projection matrix W that maps BGE-M3 embeddings (1024-dim)
directly to Symthaea's 16,384-dimensional HDC space.

This is the Rust-native counterpart to train_probe.py - designed for the
BGE-M3 + Candle pipeline (Neural Bridge v2).

Usage:
    # Extract embeddings using BGE-M3
    python extract_bge_m3_embeddings.py --output data/neural_bridge/bge_m3_embeddings.npz

    # Train probe
    python train_probe_bge_m3.py --activations data/neural_bridge/bge_m3_embeddings.npz \
                                 --output models/neural_bridge/probe_weights_bge_m3.npy \
                                 --epochs 200
"""

import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# BGE-M3 embedding dimension
BGE_M3_DIMENSION = 1024

# Symthaea HDC dimension (must match symthaea-core/src/hdc/mod.rs)
HDC_DIMENSION = 16_384

# Check for PyTorch (optional - pure NumPy fallback available)
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("Note: PyTorch not available. Using NumPy-only training.")


def generate_target_hdc(concept_name: str, seed_base: int = 42) -> np.ndarray:
    """
    Generate a target HDC vector for a concept.

    Uses deterministic seeding so the same concept always
    gets the same target vector.

    This matches the logic in Symthaea's Rust code for generating
    semantic prime vectors.
    """
    # Create seed from concept name (xorshift-like mixing)
    seed = seed_base
    for c in concept_name:
        seed = ((seed * 31) + ord(c)) & 0xFFFFFFFF

    rng = np.random.RandomState(seed)

    # Generate random bipolar vector {-1, +1}
    return rng.choice([-1, 1], size=HDC_DIMENSION).astype(np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-8 or norm_b < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class HyperdimensionalProbeNumpy:
    """
    NumPy-only implementation of the hyperdimensional probe.

    Uses gradient descent with momentum.
    """

    def __init__(
        self, input_dim: int = BGE_M3_DIMENSION, output_dim: int = HDC_DIMENSION
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Initialize projection matrix with small random values
        # Xavier initialization: std = sqrt(2 / (fan_in + fan_out))
        scale = np.sqrt(2.0 / (input_dim + output_dim)) * 0.1
        self.W = np.random.randn(output_dim, input_dim).astype(np.float32) * scale

        # Momentum
        self.velocity = np.zeros_like(self.W)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Project activation(s) to HDC space. x: [batch, input_dim] or [input_dim]"""
        return x @ self.W.T  # [batch, output_dim]

    def to_bipolar(self, x: np.ndarray) -> np.ndarray:
        """Convert to bipolar {-1, +1}"""
        return np.sign(self.forward(x))

    def train_step(
        self, x: np.ndarray, y: np.ndarray, lr: float = 0.01, momentum: float = 0.9
    ) -> float:
        """
        Single training step using cosine loss.

        x: [batch, input_dim] - embeddings
        y: [batch, output_dim] - target HDC vectors

        Returns: loss value
        """
        # Forward
        pred = self.forward(x)  # [batch, output_dim]

        # Cosine loss: 1 - cosine_similarity
        pred_norm = pred / (np.linalg.norm(pred, axis=-1, keepdims=True) + 1e-8)
        y_norm = y / (np.linalg.norm(y, axis=-1, keepdims=True) + 1e-8)
        cos_sim = (pred_norm * y_norm).sum(axis=-1)  # [batch]
        loss = (1 - cos_sim).mean()

        # Gradient of cosine loss w.r.t. W
        # d(cos_sim)/d(pred) = y_norm / ||pred|| - pred_norm * cos_sim / ||pred||
        pred_norms = np.linalg.norm(pred, axis=-1, keepdims=True) + 1e-8
        d_pred = y_norm / pred_norms - pred_norm * cos_sim[:, np.newaxis] / pred_norms
        d_pred = -d_pred / len(x)  # Negative because we minimize 1 - cos_sim

        # d(loss)/d(W) = d_pred.T @ x
        grad = d_pred.T @ x  # [output_dim, input_dim]

        # Update with momentum (maintain float32 precision)
        self.velocity = (momentum * self.velocity - lr * grad).astype(np.float32)
        self.W = (self.W + self.velocity).astype(np.float32)

        return float(loss)

    def save(self, path: Path):
        """Save weights to numpy file (float32 for Rust compatibility)."""
        np.save(path, self.W.astype(np.float32))


def train_probe_numpy(
    embeddings: np.ndarray,
    names: List[str],
    output_path: Path,
    epochs: int = 200,
    lr: float = 0.01,
    batch_size: int = 32,
) -> Tuple[HyperdimensionalProbeNumpy, dict]:
    """Train probe using NumPy only."""
    print("Training with NumPy backend...")

    n_concepts, input_dim = embeddings.shape

    if input_dim != BGE_M3_DIMENSION:
        print(f"  Warning: Input dim is {input_dim}, expected {BGE_M3_DIMENSION}")

    # Generate target HDC vectors
    targets = np.stack([generate_target_hdc(name) for name in names])
    print(f"  Generated {len(names)} target HDC vectors")

    # Create probe
    probe = HyperdimensionalProbeNumpy(input_dim, HDC_DIMENSION)

    # Training loop
    for epoch in range(epochs):
        # Shuffle
        perm = np.random.permutation(n_concepts)
        X = embeddings[perm]
        Y = targets[perm]

        total_loss = 0.0
        n_batches = 0

        for i in range(0, n_concepts, batch_size):
            batch_x = X[i : i + batch_size]
            batch_y = Y[i : i + batch_size]

            loss = probe.train_step(batch_x, batch_y, lr=lr)
            total_loss += loss
            n_batches += 1

        avg_loss = total_loss / n_batches

        if (epoch + 1) % 10 == 0 or epoch == 0:
            # Evaluate
            pred_bipolar = probe.to_bipolar(embeddings)
            cos_sims = [
                cosine_similarity(pred_bipolar[i], targets[i])
                for i in range(n_concepts)
            ]
            avg_cos_sim = np.mean(cos_sims)
            print(
                f"  Epoch {epoch+1:3d}: loss={avg_loss:.4f}, cos_sim={avg_cos_sim:.4f}"
            )

    # Final evaluation
    pred_bipolar = probe.to_bipolar(embeddings)
    cos_sims = [
        cosine_similarity(pred_bipolar[i], targets[i]) for i in range(n_concepts)
    ]
    final_cos_sim = np.mean(cos_sims)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    probe.save(output_path)

    metadata = {
        "input_dim": input_dim,
        "output_dim": HDC_DIMENSION,
        "n_concepts_trained": n_concepts,
        "final_loss": avg_loss,
        "final_cos_sim": float(final_cos_sim),
        "epochs": epochs,
        "backend": "numpy",
        "model": "bge-m3",
    }

    return probe, metadata


def train_probe(
    embeddings_path: Path,
    output_path: Path,
    epochs: int = 200,
    lr: float = 0.01,
    batch_size: int = 32,
):
    """
    Main training function.

    Loads embeddings, trains probe, saves weights and metadata.
    """
    print("=" * 60)
    print("  BGE-M3 Hyperdimensional Probe Training")
    print("=" * 60)
    print()

    # Load embeddings
    print(f"Loading embeddings from {embeddings_path}...")
    data = np.load(embeddings_path, allow_pickle=True)
    embeddings = data["embeddings"]
    names = list(data["names"])

    n_concepts, input_dim = embeddings.shape
    print(f"  Concepts: {n_concepts}")
    print(f"  Embedding dim: {input_dim}")
    print(f"  Target HDC dim: {HDC_DIMENSION}")
    print()

    # Train
    probe, metadata = train_probe_numpy(
        embeddings, names, output_path, epochs=epochs, lr=lr, batch_size=batch_size
    )

    # Save metadata
    metadata_path = output_path.with_suffix(".json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print()
    print("=" * 60)
    print("  Training Complete!")
    print("=" * 60)
    print()
    print(f"Weights saved to: {output_path}")
    print(f"Metadata saved to: {metadata_path}")
    print()
    print("Metadata:")
    for k, v in metadata.items():
        print(f"  {k}: {v}")
    print()
    print("Next steps:")
    print(
        f'  1. Use in Rust: NeuralBridgeV2::load_default() with probe at "{output_path}"'
    )
    print("  2. Run: cargo test --features neural-bridge -p symthaea")


def main():
    parser = argparse.ArgumentParser(
        description="Train hyperdimensional probe for BGE-M3 Neural Bridge v2"
    )
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=Path("data/neural_bridge/bge_m3_embeddings.npz"),
        help="Path to extracted BGE-M3 embeddings",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/neural_bridge/probe_weights_bge_m3.npy"),
        help="Output path for weights",
    )
    parser.add_argument(
        "--epochs", type=int, default=200, help="Number of training epochs"
    )
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    args = parser.parse_args()

    train_probe(
        args.embeddings,
        args.output,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()

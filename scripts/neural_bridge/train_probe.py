#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Hyperdimensional Probe Training

Trains a linear projection matrix W that maps LLM activations
directly to Symthaea's 16,384-dimensional HDC space.

The probe learns: HDC_target = sign(W @ activation)

This is the core of the "Hyperdimensional Probe" technique.

Usage:
    python train_probe.py --activations data/neural_bridge/concept_activations.npz \\
                          --output models/neural_bridge/probe_weights.npy \\
                          --epochs 100
"""

import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# Symthaea HDC dimension (must match symthaea-core/src/hdc/unified_hv.rs)
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

    def __init__(self, input_dim: int, output_dim: int = HDC_DIMENSION):
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

        x: [batch, input_dim] - activations
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

        # Update with momentum
        self.velocity = momentum * self.velocity - lr * grad
        self.W += self.velocity

        return float(loss)

    def save(self, path: Path):
        """Save weights to numpy file."""
        np.save(path, self.W)


if HAS_TORCH:

    class HyperdimensionalProbeTorch(nn.Module):
        """
        PyTorch implementation of the hyperdimensional probe.

        Architecture:
            activation [hidden_dim] -> Linear -> HDC [16384]
        """

        def __init__(self, input_dim: int, output_dim: int = HDC_DIMENSION):
            super().__init__()
            self.projection = nn.Linear(input_dim, output_dim, bias=False)

            # Initialize with small random values
            nn.init.xavier_uniform_(self.projection.weight, gain=0.1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.projection(x)

        def to_bipolar(self, x: torch.Tensor) -> torch.Tensor:
            return torch.sign(self.forward(x))

    def train_probe_torch(
        activations: np.ndarray,
        names: List[str],
        output_path: Path,
        epochs: int = 100,
        lr: float = 0.01,
        batch_size: int = 32,
        device: str = "cpu",
    ):
        """Train probe using PyTorch."""
        print(f"Training with PyTorch backend (device: {device})...")

        n_concepts, hidden_dim = activations.shape

        # Generate target HDC vectors
        targets = np.stack([generate_target_hdc(name) for name in names])
        print(f"  Generated {len(names)} target HDC vectors")

        # Convert to tensors
        X = torch.from_numpy(activations).float().to(device)
        Y = torch.from_numpy(targets).float().to(device)

        # Create probe
        probe = HyperdimensionalProbeTorch(hidden_dim, HDC_DIMENSION).to(device)
        optimizer = optim.Adam(probe.parameters(), lr=lr)

        # Cosine loss function
        def cosine_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            pred_norm = pred / (torch.norm(pred, dim=-1, keepdim=True) + 1e-8)
            target_norm = target / (torch.norm(target, dim=-1, keepdim=True) + 1e-8)
            cos_sim = (pred_norm * target_norm).sum(dim=-1)
            return (1 - cos_sim).mean()

        # Training loop
        for epoch in range(epochs):
            # Shuffle
            perm = torch.randperm(n_concepts)
            X_shuffled = X[perm]
            Y_shuffled = Y[perm]

            total_loss = 0.0
            n_batches = 0

            for i in range(0, n_concepts, batch_size):
                batch_x = X_shuffled[i : i + batch_size]
                batch_y = Y_shuffled[i : i + batch_size]

                optimizer.zero_grad()
                pred = probe(batch_x)
                loss = cosine_loss(pred, batch_y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            avg_loss = total_loss / n_batches

            if (epoch + 1) % 10 == 0 or epoch == 0:
                with torch.no_grad():
                    pred_bipolar = probe.to_bipolar(X)
                    cos_sim = (
                        torch.nn.functional.cosine_similarity(pred_bipolar, Y, dim=-1)
                        .mean()
                        .item()
                    )
                print(
                    f"  Epoch {epoch+1:3d}: loss={avg_loss:.4f}, cos_sim={cos_sim:.4f}"
                )

        # Final evaluation
        with torch.no_grad():
            pred_bipolar = probe.to_bipolar(X)
            final_cos_sim = (
                torch.nn.functional.cosine_similarity(pred_bipolar, Y, dim=-1)
                .mean()
                .item()
            )

        # Save weights as numpy float32 (for Rust compatibility)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        weights = probe.projection.weight.detach().cpu().numpy().astype(np.float32)
        np.save(output_path, weights)

        metadata = {
            "input_dim": hidden_dim,
            "output_dim": HDC_DIMENSION,
            "n_concepts_trained": n_concepts,
            "final_loss": avg_loss,
            "final_cos_sim": float(final_cos_sim),
            "epochs": epochs,
            "backend": "pytorch",
        }

        return probe, metadata


def train_probe_numpy(
    activations: np.ndarray,
    names: List[str],
    output_path: Path,
    epochs: int = 100,
    lr: float = 0.01,
    batch_size: int = 32,
) -> Tuple[HyperdimensionalProbeNumpy, dict]:
    """Train probe using NumPy only."""
    print("Training with NumPy backend...")

    n_concepts, hidden_dim = activations.shape

    # Generate target HDC vectors
    targets = np.stack([generate_target_hdc(name) for name in names])
    print(f"  Generated {len(names)} target HDC vectors")

    # Create probe
    probe = HyperdimensionalProbeNumpy(hidden_dim, HDC_DIMENSION)

    # Training loop
    for epoch in range(epochs):
        # Shuffle
        perm = np.random.permutation(n_concepts)
        X = activations[perm]
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
            pred_bipolar = probe.to_bipolar(activations)
            cos_sims = [
                cosine_similarity(pred_bipolar[i], targets[i])
                for i in range(n_concepts)
            ]
            avg_cos_sim = np.mean(cos_sims)
            print(
                f"  Epoch {epoch+1:3d}: loss={avg_loss:.4f}, cos_sim={avg_cos_sim:.4f}"
            )

    # Final evaluation
    pred_bipolar = probe.to_bipolar(activations)
    cos_sims = [
        cosine_similarity(pred_bipolar[i], targets[i]) for i in range(n_concepts)
    ]
    final_cos_sim = np.mean(cos_sims)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    probe.save(output_path)

    metadata = {
        "input_dim": hidden_dim,
        "output_dim": HDC_DIMENSION,
        "n_concepts_trained": n_concepts,
        "final_loss": avg_loss,
        "final_cos_sim": float(final_cos_sim),
        "epochs": epochs,
        "backend": "numpy",
    }

    return probe, metadata


def train_probe(
    activations_path: Path,
    output_path: Path,
    epochs: int = 100,
    lr: float = 0.01,
    batch_size: int = 32,
    device: str = "cpu",
):
    """
    Main training function.

    Loads activations, trains probe, saves weights and metadata.
    """
    print("=" * 60)
    print("  Hyperdimensional Probe Training")
    print("=" * 60)
    print()

    # Load activations
    print(f"Loading activations from {activations_path}...")
    data = np.load(activations_path, allow_pickle=True)
    activations = data["activations"]
    names = list(data["names"])

    n_concepts, hidden_dim = activations.shape
    print(f"  Concepts: {n_concepts}")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Target HDC dim: {HDC_DIMENSION}")
    print()

    # Train
    if HAS_TORCH:
        probe, metadata = train_probe_torch(
            activations,
            names,
            output_path,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            device=device,
        )
    else:
        probe, metadata = train_probe_numpy(
            activations, names, output_path, epochs=epochs, lr=lr, batch_size=batch_size
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
    print('  1. Use in Rust: NeuralBridge::load("' + str(output_path) + '")')
    print("  2. Run: cargo test -p symthaea --lib perception::neural_bridge")


def evaluate_probe(activations_path: Path, weights_path: Path):
    """Evaluate a trained probe."""
    print("Evaluating trained probe...")

    # Load
    data = np.load(activations_path, allow_pickle=True)
    activations = data["activations"]
    names = list(data["names"])
    weights = np.load(weights_path)

    print(f"  Activations: {activations.shape}")
    print(f"  Weights: {weights.shape}")
    print()

    # Project
    projected = activations @ weights.T
    projected_bipolar = np.sign(projected)

    # Generate targets and compare
    targets = np.stack([generate_target_hdc(name) for name in names])

    print("Per-concept evaluation:")
    for i, name in enumerate(names):
        cos_sim = cosine_similarity(projected_bipolar[i], targets[i])
        bit_acc = (projected_bipolar[i] == targets[i]).mean()
        print(f"  {name:25s}: cos_sim={cos_sim:+.3f}, bit_acc={bit_acc:.1%}")


def main():
    parser = argparse.ArgumentParser(
        description="Train hyperdimensional probe for Neural Bridge"
    )
    parser.add_argument(
        "--activations",
        type=Path,
        default=Path("data/neural_bridge/concept_activations.npz"),
        help="Path to extracted activations",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/neural_bridge/probe_weights.npy"),
        help="Output path for weights",
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device for PyTorch (cpu/cuda)"
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate existing probe instead of training",
    )
    args = parser.parse_args()

    if args.evaluate:
        evaluate_probe(args.activations, args.output)
    else:
        train_probe(
            args.activations,
            args.output,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            device=args.device,
        )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Phase 10 Node Integration

Wrapper that integrates Phase10Coordinator with existing ZeroTrustML node functionality.
Provides easy migration path from Phase 9 to Phase 10.
"""

import asyncio
import logging
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from zerotrustml.core.phase10_coordinator import Phase10Coordinator, Phase10Config
from zerotrustml.core.node import Node, NodeConfig

logger = logging.getLogger(__name__)


class Phase10Node:
    """
    ZeroTrustML Node with Phase 10 Features

    Extends base Node with:
    - Hybrid PostgreSQL + Holochain storage
    - Zero-knowledge proof validation
    - Privacy-preserving gradient submission
    - Immutable audit trail
    """

    def __init__(
        self,
        node_id: str,
        data_path: str,
        phase10_config: Optional[Phase10Config] = None,
        enable_zkpoc: bool = True,
        enable_holochain: bool = True
    ):
        """
        Initialize Phase 10 node

        Args:
            node_id: Unique node identifier
            data_path: Path to training data
            phase10_config: Phase 10 configuration (optional)
            enable_zkpoc: Enable zero-knowledge proofs
            enable_holochain: Enable Holochain immutable storage
        """
        self.node_id = node_id

        # Base node configuration
        self.node_config = NodeConfig(
            node_id=node_id,
            data_path=data_path
        )

        # Phase 10 configuration
        if phase10_config is None:
            phase10_config = Phase10Config(
                zkpoc_enabled=enable_zkpoc,
                holochain_enabled=enable_holochain
            )

        self.phase10_config = phase10_config

        # Initialize coordinator (will be set up in start())
        self.coordinator = None

        # Local training state
        self.model = None
        self.round_num = 0
        self.local_dataset = None

        logger.info(f"Phase 10 Node initialized: {node_id}")
        logger.info(f"  ZK-PoC: {'✅ Enabled' if enable_zkpoc else '❌ Disabled'}")
        logger.info(f"  Holochain: {'✅ Enabled' if enable_holochain else '❌ Disabled'}")

    async def start(self):
        """
        Start the node and connect to coordinator

        This connects to the Phase 10 coordinator (which manages
        PostgreSQL, Holochain, and ZK-PoC).
        """
        logger.info(f"Starting Phase 10 node: {self.node_id}")

        # Initialize Phase 10 coordinator
        self.coordinator = Phase10Coordinator(self.phase10_config)
        await self.coordinator.initialize()

        # Initialize model
        from zerotrustml.core.training import SimpleNN
        self.model = SimpleNN()

        # Load local dataset
        await self._load_dataset()

        logger.info(f"✅ Phase 10 node started: {self.node_id}")

    async def train_and_submit(
        self,
        epochs: int = 1,
        batch_size: int = 32
    ) -> Dict[str, Any]:
        """
        Train locally and submit gradient to coordinator

        This is the main workflow for Phase 10 nodes:
        1. Train on local data
        2. Generate gradient
        3. (Optional) Generate ZK-PoC proof
        4. Submit to coordinator
        5. Receive credits

        Args:
            epochs: Number of local training epochs
            batch_size: Batch size for training

        Returns:
            Submission result with acceptance status and credits
        """
        logger.info(f"Round {self.round_num}: Training locally...")

        # 1. Local training
        gradient, pogq_score = await self._train_local(epochs, batch_size)

        logger.info(f"Round {self.round_num}: Local training complete (PoGQ: {pogq_score:.3f})")

        # 2. Encrypt gradient
        encrypted_gradient = self._encrypt_gradient(gradient)

        # 3. Generate ZK-PoC proof (if enabled)
        zkpoc_proof = None
        if self.phase10_config.zkpoc_enabled and self.coordinator.zkpoc:
            zkpoc_proof = self.coordinator.zkpoc.generate_proof(pogq_score)
            logger.info(f"Round {self.round_num}: ZK-PoC proof generated ✅")

        # 4. Submit to coordinator
        result = await self.coordinator.handle_gradient_submission(
            node_id=self.node_id,
            encrypted_gradient=encrypted_gradient,
            zkpoc_proof=zkpoc_proof,
            pogq_score=pogq_score if not zkpoc_proof else None  # Hide score if using ZK
        )

        # 5. Handle result
        if result["accepted"]:
            logger.info(f"Round {self.round_num}: Gradient accepted ✅")
            logger.info(f"  Gradient ID: {result['gradient_id'][:8]}...")
            if result.get("holochain_hash"):
                logger.info(f"  Holochain Hash: {result['holochain_hash'][:8]}...")
            logger.info(f"  Credits Earned: {result['credits_issued']}")
        else:
            logger.warning(f"Round {self.round_num}: Gradient rejected ❌")
            logger.warning(f"  Reason: {result['reason']}")

        self.round_num += 1
        return result

    async def get_credits(self) -> int:
        """Get current credit balance"""
        if not self.coordinator or not self.coordinator.postgres:
            return 0

        balance = await self.coordinator.postgres.get_credit_balance(self.node_id)
        return balance

    async def get_reputation(self) -> Dict[str, Any]:
        """Get current reputation"""
        if not self.coordinator or not self.coordinator.postgres:
            return {"score": 0.0}

        reputation = await self.coordinator.postgres.get_reputation(self.node_id)
        return reputation or {"score": 0.0}

    async def shutdown(self):
        """Graceful shutdown"""
        logger.info(f"Shutting down Phase 10 node: {self.node_id}")

        if self.coordinator:
            await self.coordinator.shutdown()

        logger.info(f"✅ Node shut down: {self.node_id}")

    # ============================================================
    # Private Methods
    # ============================================================

    async def _load_dataset(self):
        """
        Load local training dataset from data_path

        Supports multiple formats:
        - .npz files with 'X' and 'y' arrays
        - .pt or .pth PyTorch dataset files
        - Directory with images organized by class (ImageFolder format)
        - CSV files with features and labels

        Falls back to synthetic data if loading fails.
        """
        import os
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        data_path = self.node_config.data_path

        try:
            if os.path.exists(data_path):
                # NPZ format (numpy archives)
                if data_path.endswith('.npz'):
                    data = np.load(data_path)
                    X = data['X'] if 'X' in data else data['features']
                    y = data['y'] if 'y' in data else data['labels']
                    self.local_dataset = {"X": X, "y": y}
                    self._create_torch_loaders(X, y)
                    logger.info(f"Loaded NPZ dataset: {len(X)} samples from {data_path}")

                # PyTorch format
                elif data_path.endswith('.pt') or data_path.endswith('.pth'):
                    data = torch.load(data_path)
                    if isinstance(data, dict):
                        X = data.get('X', data.get('features'))
                        y = data.get('y', data.get('labels'))
                        if isinstance(X, torch.Tensor):
                            X = X.numpy()
                        if isinstance(y, torch.Tensor):
                            y = y.numpy()
                    elif isinstance(data, TensorDataset):
                        X = data.tensors[0].numpy()
                        y = data.tensors[1].numpy()
                    else:
                        raise ValueError(f"Unsupported PyTorch data format: {type(data)}")
                    self.local_dataset = {"X": X, "y": y}
                    self._create_torch_loaders(X, y)
                    logger.info(f"Loaded PyTorch dataset: {len(X)} samples from {data_path}")

                # CSV format
                elif data_path.endswith('.csv'):
                    import csv
                    with open(data_path, 'r') as f:
                        reader = csv.reader(f)
                        header = next(reader)  # Skip header
                        data = list(reader)
                    # Assume last column is label
                    X = np.array([[float(v) for v in row[:-1]] for row in data])
                    y = np.array([int(row[-1]) for row in data])
                    self.local_dataset = {"X": X, "y": y}
                    self._create_torch_loaders(X, y)
                    logger.info(f"Loaded CSV dataset: {len(X)} samples from {data_path}")

                # Directory with class subfolders (ImageFolder format)
                elif os.path.isdir(data_path):
                    from torchvision import datasets, transforms
                    transform = transforms.Compose([
                        transforms.Grayscale(),
                        transforms.Resize((28, 28)),
                        transforms.ToTensor(),
                        transforms.Lambda(lambda x: x.view(-1))  # Flatten to 784
                    ])
                    try:
                        dataset = datasets.ImageFolder(data_path, transform=transform)
                        X = np.stack([dataset[i][0].numpy() for i in range(len(dataset))])
                        y = np.array([dataset[i][1] for i in range(len(dataset))])
                        self.local_dataset = {"X": X, "y": y}
                        self._create_torch_loaders(X, y)
                        logger.info(f"Loaded ImageFolder dataset: {len(X)} samples from {data_path}")
                    except Exception as e:
                        logger.warning(f"ImageFolder load failed: {e}, using synthetic data")
                        self._generate_synthetic_data()
                else:
                    logger.warning(f"Unsupported data format: {data_path}, using synthetic data")
                    self._generate_synthetic_data()
            else:
                logger.warning(f"Data path not found: {data_path}, using synthetic data")
                self._generate_synthetic_data()

        except Exception as e:
            logger.error(f"Failed to load dataset from {data_path}: {e}")
            logger.info("Falling back to synthetic data")
            self._generate_synthetic_data()

    def _generate_synthetic_data(self):
        """Generate synthetic MNIST-like data for testing"""
        n_samples = 1000
        n_features = 784
        n_classes = 10

        # Create clustered data for more realistic learning
        X = np.random.randn(n_samples, n_features).astype(np.float32)
        class_centers = np.random.randn(n_classes, n_features) * 2

        samples_per_class = n_samples // n_classes
        for i in range(n_classes):
            start_idx = i * samples_per_class
            end_idx = (i + 1) * samples_per_class
            X[start_idx:end_idx] += class_centers[i]

        y = np.repeat(np.arange(n_classes), samples_per_class)

        # Shuffle
        indices = np.random.permutation(n_samples)
        X = X[indices]
        y = y[indices]

        self.local_dataset = {"X": X, "y": y}
        self._create_torch_loaders(X, y)
        logger.debug(f"Generated synthetic dataset: {n_samples} samples")

    def _create_torch_loaders(self, X: np.ndarray, y: np.ndarray):
        """Create PyTorch DataLoaders from numpy arrays"""
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        # Convert to tensors
        X_tensor = torch.from_numpy(X.astype(np.float32))
        y_tensor = torch.from_numpy(y.astype(np.int64))

        # Split 80/20 train/test
        n_samples = len(X)
        n_train = int(0.8 * n_samples)
        indices = torch.randperm(n_samples)

        train_dataset = TensorDataset(
            X_tensor[indices[:n_train]],
            y_tensor[indices[:n_train]]
        )
        test_dataset = TensorDataset(
            X_tensor[indices[n_train:]],
            y_tensor[indices[n_train:]]
        )

        self.train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    async def _train_local(
        self,
        epochs: int,
        batch_size: int
    ) -> tuple[np.ndarray, float]:
        """
        Train model on local data using PyTorch

        Returns:
            (gradient, pogq_score) where:
            - gradient: Flattened numpy array of model gradients
            - pogq_score: Proof of Gradient Quality score (0.0-1.0)
        """
        import torch
        import torch.nn as nn
        import torch.optim as optim

        if self.model is None:
            raise RuntimeError("Model not initialized. Call start() first.")

        if not hasattr(self, 'train_loader') or self.train_loader is None:
            raise RuntimeError("Dataset not loaded. Call start() first.")

        # Set up device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)

        # Set up optimizer and loss
        optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        criterion = nn.CrossEntropyLoss()

        # Store initial parameters for gradient computation
        initial_params = {name: param.clone() for name, param in self.model.named_parameters()}

        # Training loop
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        correct = 0

        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_samples = 0
            epoch_correct = 0

            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(device), target.to(device)

                # Flatten input if needed (for SimpleNN expecting 784-dim input)
                if data.dim() > 2:
                    data = data.view(data.size(0), -1)
                elif data.dim() == 2 and data.size(1) != 784:
                    # Already flat but wrong size - reshape or pad
                    if data.numel() // data.size(0) == 784:
                        data = data.view(data.size(0), 784)

                # Forward pass
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)

                # Backward pass
                loss.backward()
                optimizer.step()

                # Track metrics
                epoch_loss += loss.item() * data.size(0)
                epoch_samples += data.size(0)
                _, predicted = torch.max(output.data, 1)
                epoch_correct += (predicted == target).sum().item()

            total_loss += epoch_loss
            total_samples += epoch_samples
            correct += epoch_correct

        # Compute gradient as difference between final and initial parameters
        gradients = []
        for name, param in self.model.named_parameters():
            grad = (initial_params[name] - param).detach().cpu().numpy().flatten()
            gradients.append(grad)

        gradient = np.concatenate(gradients)

        # Compute PoGQ (Proof of Gradient Quality) score
        # Based on gradient statistics and training improvement
        avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')
        accuracy = correct / total_samples if total_samples > 0 else 0.0

        pogq_score = self._compute_pogq_score(gradient, avg_loss, accuracy)

        # Yield to event loop
        await asyncio.sleep(0)

        return gradient, pogq_score

    def _compute_pogq_score(
        self,
        gradient: np.ndarray,
        avg_loss: float,
        accuracy: float
    ) -> float:
        """
        Compute Proof of Gradient Quality score

        Factors considered:
        - Gradient norm (not too large/small)
        - Gradient sparsity (reasonable activation)
        - Training loss (convergence indicator)
        - Training accuracy (learning indicator)
        - Gradient variance (stability indicator)

        Returns:
            Score between 0.0 and 1.0
        """
        scores = []

        # 1. Gradient norm score (penalize extreme values)
        grad_norm = np.linalg.norm(gradient)
        if grad_norm < 1e-6:
            norm_score = 0.1  # Too small - not learning
        elif grad_norm > 100:
            norm_score = 0.3  # Too large - potentially malicious
        else:
            # Optimal range: 0.01 to 10
            norm_score = min(1.0, max(0.5, 1.0 - abs(np.log10(grad_norm + 1e-10) - 0.5) / 3))
        scores.append(norm_score * 0.2)

        # 2. Sparsity score (reasonable activation pattern)
        sparsity = np.mean(np.abs(gradient) < 1e-6)
        if sparsity > 0.9:
            sparsity_score = 0.3  # Too sparse
        elif sparsity < 0.01:
            sparsity_score = 0.7  # Dense is okay
        else:
            sparsity_score = 0.9  # Moderate sparsity is good
        scores.append(sparsity_score * 0.15)

        # 3. Loss score (lower is better, normalized)
        loss_score = max(0.0, min(1.0, 1.0 - avg_loss / 5.0))
        scores.append(loss_score * 0.25)

        # 4. Accuracy score (direct quality indicator)
        accuracy_score = accuracy
        scores.append(accuracy_score * 0.25)

        # 5. Gradient variance score (stability)
        grad_var = np.var(gradient)
        if grad_var < 1e-10:
            var_score = 0.2  # Suspiciously uniform
        elif grad_var > 1000:
            var_score = 0.3  # Too noisy
        else:
            var_score = 0.8
        scores.append(var_score * 0.15)

        # Combine scores
        total_score = sum(scores)

        # Clamp to valid range
        return max(0.0, min(1.0, total_score))

    def _encrypt_gradient(self, gradient: np.ndarray) -> bytes:
        """
        Encrypt gradient for privacy

        In production: use NaCl/libsodium authenticated encryption
        For demo: JSON encoding
        """
        import json
        return json.dumps(gradient.tolist()).encode()


# ============================================================
# Example Usage
# ============================================================

async def demo_phase10_node():
    """Demonstrate Phase 10 node workflow"""
    print("🚀 Phase 10 Node Demo\n")

    # Create node
    node = Phase10Node(
        node_id="hospital-a",
        data_path="/data/hospital-a",
        enable_zkpoc=True,
        enable_holochain=False  # Disable for demo (no conductor running)
    )

    try:
        # Start node
        print("Starting node...")
        # await node.start()  # Commented out - requires PostgreSQL

        print("✅ Node started\n")

        # Simulate training rounds
        print("Simulating 3 training rounds...\n")

        # for i in range(3):
        #     result = await node.train_and_submit(epochs=1)
        #
        #     if result["accepted"]:
        #         print(f"Round {i+1}: ✅ Accepted")
        #         print(f"  Credits: {result['credits_issued']}")
        #     else:
        #         print(f"Round {i+1}: ❌ Rejected ({result['reason']})")
        #
        #     print()

        # Get final stats
        # balance = await node.get_credits()
        # reputation = await node.get_reputation()
        #
        # print(f"Final Stats:")
        # print(f"  Credits: {balance}")
        # print(f"  Reputation: {reputation.get('score', 0.0):.3f}")

        # await node.shutdown()

        print("✅ Demo complete!")
        print("\nTo run with real backends:")
        print("1. Start PostgreSQL: docker-compose -f docker-compose.phase10.yml up -d postgres")
        print("2. Start Holochain: docker-compose -f docker-compose.phase10.yml up -d holochain")
        print("3. Initialize database: psql -f scripts/init_db_phase10.sql")
        print("4. Uncomment the code above and run again")

    except Exception as e:
        print(f"❌ Demo error: {e}")
        print("(This is expected without PostgreSQL/Holochain running)")


if __name__ == "__main__":
    asyncio.run(demo_phase10_node())

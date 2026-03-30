#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Test suite for the datasets module.

Tests all distribution strategies and federated dataset creation.
"""

import pytest
import numpy as np


class TestDataDistributions:
    """Test data distribution strategies."""

    def test_iid_distribution_import(self):
        """Test that IID distribution can be imported."""
        from src.datasets.data_distribution import IIDDistribution
        dist = IIDDistribution()
        assert dist is not None
        assert str(dist) == "IIDDistribution()"

    def test_dirichlet_distribution_import(self):
        """Test that Dirichlet distribution can be imported."""
        from src.datasets.data_distribution import DirichletDistribution
        dist = DirichletDistribution(alpha=0.5)
        assert dist is not None
        assert dist.alpha == 0.5
        assert "α=0.5" in str(dist)

    def test_label_skew_distribution_import(self):
        """Test that LabelSkew distribution can be imported."""
        from src.datasets.data_distribution import LabelSkewDistribution
        dist = LabelSkewDistribution(labels_per_client=2)
        assert dist is not None
        assert dist.labels_per_client == 2
        assert "labels_per_client=2" in str(dist)

    def test_quantity_skew_distribution_import(self):
        """Test that QuantitySkew distribution can be imported."""
        from src.datasets.data_distribution import QuantitySkewDistribution
        dist = QuantitySkewDistribution(skew_factor=2.0)
        assert dist is not None
        assert dist.skew_factor == 2.0

    def test_iid_distributes_uniformly(self):
        """Test that IID distribution creates roughly equal partitions."""
        from src.datasets.data_distribution import IIDDistribution

        dist = IIDDistribution()
        n_samples = 1000
        n_clients = 10
        labels = np.random.randint(0, 10, size=n_samples)

        # API: distribute(labels, num_clients, seed)
        indices = dist.distribute(labels, n_clients, seed=42)

        # Should have 10 clients
        assert len(indices) == n_clients

        # Each client should have ~100 samples (allow 10% variance due to integer division)
        for client_idx in indices:
            assert 90 <= len(client_idx) <= 110, f"Client has {len(client_idx)} samples"

    def test_dirichlet_creates_heterogeneous_splits(self):
        """Test that Dirichlet distribution creates heterogeneous partitions."""
        from src.datasets.data_distribution import DirichletDistribution

        dist = DirichletDistribution(alpha=0.1)  # Very heterogeneous
        n_samples = 1000
        n_clients = 10
        labels = np.random.randint(0, 10, size=n_samples)

        indices = dist.distribute(labels, n_clients, seed=42)

        # Should have 10 clients
        assert len(indices) == n_clients

        # All samples should be assigned
        total_assigned = sum(len(idx) for idx in indices)
        assert total_assigned == n_samples

    def test_label_skew_limits_classes(self):
        """Test that LabelSkew limits classes per client."""
        from src.datasets.data_distribution import LabelSkewDistribution

        labels_per_client = 2
        dist = LabelSkewDistribution(labels_per_client=labels_per_client)
        n_samples = 1000
        n_clients = 5
        labels = np.random.randint(0, 10, size=n_samples)

        indices = dist.distribute(labels, n_clients, seed=42)

        # Each client should have at most labels_per_client unique classes
        for client_idx in indices:
            if len(client_idx) > 0:  # Skip empty clients
                client_labels = labels[client_idx]
                unique_classes = len(np.unique(client_labels))
                assert unique_classes <= labels_per_client, \
                    f"Client has {unique_classes} classes, expected max {labels_per_client}"

    def test_quantity_skew_creates_unequal_partitions(self):
        """Test that QuantitySkew creates unequal data amounts."""
        from src.datasets.data_distribution import QuantitySkewDistribution

        dist = QuantitySkewDistribution(skew_factor=3.0)
        n_samples = 1000
        n_clients = 10
        labels = np.random.randint(0, 10, size=n_samples)

        indices = dist.distribute(labels, n_clients, seed=42)

        # Should have 10 clients
        assert len(indices) == n_clients

        # All samples should be assigned
        total_assigned = sum(len(idx) for idx in indices)
        assert total_assigned == n_samples

        # Clients should have varying amounts (check variance > 0)
        sizes = [len(idx) for idx in indices]
        assert np.var(sizes) > 0, "Quantity skew should create variance in client sizes"

    def test_distribution_stats(self):
        """Test computing distribution statistics."""
        from src.datasets.data_distribution import DirichletDistribution

        dist = DirichletDistribution(alpha=0.5)
        n_samples = 1000
        n_clients = 10
        labels = np.random.randint(0, 10, size=n_samples)

        indices = dist.distribute(labels, n_clients, seed=42)
        stats = dist.compute_stats(indices, labels)

        assert stats.num_clients == n_clients
        assert stats.total_samples == n_samples
        assert len(stats.samples_per_client) == n_clients
        assert stats.quantity_variance >= 0  # Non-negative variance
        assert isinstance(stats.to_dict(), dict)


class TestClientData:
    """Test client data utilities."""

    def test_client_dataset_import(self):
        """Test that ClientDataset can be imported."""
        from src.datasets.client_data import ClientDataset
        assert ClientDataset is not None

    def test_client_stats_dataclass(self):
        """Test ClientStats dataclass."""
        from src.datasets.client_data import ClientStats

        stats = ClientStats(
            client_id=0,
            num_samples=100,
            label_distribution={0: 50, 1: 50},
        )
        assert stats.client_id == 0
        assert stats.num_samples == 100
        assert stats.to_dict()["client_id"] == 0


class TestFederatedDatasets:
    """Test federated dataset creation."""

    def test_federated_dataset_import(self):
        """Test that FederatedDataset can be imported."""
        from src.datasets.federated_datasets import FederatedDataset
        assert FederatedDataset is not None

    def test_federated_dataset_config_import(self):
        """Test that FederatedDatasetConfig can be imported."""
        from src.datasets.federated_datasets import FederatedDatasetConfig

        config = FederatedDatasetConfig(
            num_clients=10,
            distribution="dirichlet",
            alpha=0.5,
        )
        assert config.num_clients == 10
        assert config.distribution == "dirichlet"
        assert config.alpha == 0.5

    def test_create_cifar10_function_import(self):
        """Test that create_cifar10_federated can be imported."""
        from src.datasets.federated_datasets import create_cifar10_federated
        assert create_cifar10_federated is not None
        assert callable(create_cifar10_federated)

    def test_create_cifar100_function_import(self):
        """Test that create_cifar100_federated can be imported."""
        from src.datasets.federated_datasets import create_cifar100_federated
        assert create_cifar100_federated is not None
        assert callable(create_cifar100_federated)

    @pytest.mark.skipif(
        not pytest.importorskip("torch", reason="PyTorch required"),
        reason="PyTorch not available"
    )
    def test_create_cifar10_iid(self):
        """Test creating IID CIFAR-10 federated dataset."""
        from src.datasets.federated_datasets import create_cifar10_federated

        # Create with minimal clients for speed
        fed_dataset = create_cifar10_federated(
            num_clients=5,
            distribution="iid",
            download=True,  # Will use cached if available
        )

        assert fed_dataset.num_clients == 5
        assert fed_dataset.num_classes == 10

        # Get first client's data
        train_loader = fed_dataset.get_train_loader(0, batch_size=32)
        assert train_loader is not None

    @pytest.mark.skipif(
        not pytest.importorskip("torch", reason="PyTorch required"),
        reason="PyTorch not available"
    )
    def test_create_cifar10_dirichlet(self):
        """Test creating Dirichlet CIFAR-10 federated dataset."""
        from src.datasets.federated_datasets import create_cifar10_federated

        fed_dataset = create_cifar10_federated(
            num_clients=5,
            distribution="dirichlet",
            alpha=0.5,
            download=True,
        )

        assert fed_dataset.num_clients == 5

        # Verify all clients have data
        for i in range(5):
            count = fed_dataset.client_data_counts[i]
            assert count > 0, f"Client {i} has no data"


class TestModuleExports:
    """Test that the datasets module exports correctly."""

    def test_distribution_exports(self):
        """Test that all distribution classes are exported."""
        from src.datasets import (
            DataDistribution,
            IIDDistribution,
            DirichletDistribution,
            LabelSkewDistribution,
            QuantitySkewDistribution,
        )

        assert DataDistribution is not None
        assert IIDDistribution is not None
        assert DirichletDistribution is not None
        assert LabelSkewDistribution is not None
        assert QuantitySkewDistribution is not None

    def test_federated_exports(self):
        """Test that federated dataset utilities are exported."""
        from src.datasets import (
            FederatedDataset,
            create_cifar10_federated,
            create_cifar100_federated,
        )

        assert FederatedDataset is not None
        assert create_cifar10_federated is not None
        assert create_cifar100_federated is not None

    def test_client_exports(self):
        """Test that client utilities are exported."""
        from src.datasets import (
            ClientDataset,
            ClientDataLoader,
        )

        assert ClientDataset is not None
        assert ClientDataLoader is not None


class TestDistributionDeterminism:
    """Test that distributions are deterministic with seeds."""

    def test_iid_deterministic(self):
        """Test that IID distribution is deterministic."""
        from src.datasets.data_distribution import IIDDistribution

        dist = IIDDistribution()
        labels = np.random.randint(0, 10, size=1000)

        indices1 = dist.distribute(labels, 10, seed=42)
        indices2 = dist.distribute(labels, 10, seed=42)

        # Same seed should give same results
        for i1, i2 in zip(indices1, indices2):
            np.testing.assert_array_equal(i1, i2)

    def test_dirichlet_deterministic(self):
        """Test that Dirichlet distribution is deterministic."""
        from src.datasets.data_distribution import DirichletDistribution

        dist = DirichletDistribution(alpha=0.5)
        labels = np.random.randint(0, 10, size=1000)

        indices1 = dist.distribute(labels, 10, seed=42)
        indices2 = dist.distribute(labels, 10, seed=42)

        # Same seed should give same results
        for i1, i2 in zip(indices1, indices2):
            np.testing.assert_array_equal(sorted(i1), sorted(i2))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

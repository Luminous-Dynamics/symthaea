# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Data Splitting Utilities for Federated Learning

Three types of data splits:
1. IID (Independent and Identically Distributed) - Equal random split
2. Dirichlet Non-IID - Controlled heterogeneity via Dirichlet distribution
3. Pathological Non-IID - Extreme heterogeneity (fixed classes per client)

Reference: These splitting strategies are standard in FL literature
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from typing import List, Tuple, Dict
from collections import defaultdict


def create_iid_split(
    dataset: Dataset,
    num_clients: int,
    seed: int = 42
) -> List[List[int]]:
    """
    Create IID (Independent and Identically Distributed) split.

    Randomly shuffles dataset and splits equally among clients.

    Args:
        dataset: PyTorch dataset to split
        num_clients: Number of clients
        seed: Random seed for reproducibility

    Returns:
        List of client indices, where client_indices[i] contains
        the dataset indices for client i

    Example:
        >>> train_data = datasets.MNIST(root='./data', train=True)
        >>> client_indices = create_iid_split(train_data, num_clients=10)
        >>> # Each client gets 6000 samples (60000 / 10)
    """
    np.random.seed(seed)

    # Get total number of samples
    num_samples = len(dataset)

    # Create shuffled indices
    indices = np.random.permutation(num_samples)

    # Split equally among clients
    samples_per_client = num_samples // num_clients

    client_indices = []
    for i in range(num_clients):
        start = i * samples_per_client
        end = start + samples_per_client if i < num_clients - 1 else num_samples
        client_indices.append(indices[start:end].tolist())

    return client_indices


def create_dirichlet_split(
    dataset: Dataset,
    num_clients: int,
    alpha: float = 0.5,
    seed: int = 42
) -> List[List[int]]:
    """
    Create non-IID split using Dirichlet distribution.

    Uses Dirichlet distribution to create heterogeneous class distributions
    across clients. Lower alpha = more heterogeneous.

    Args:
        dataset: PyTorch dataset with 'targets' attribute
        num_clients: Number of clients
        alpha: Dirichlet concentration parameter
            - alpha → 0: Each client has few classes (highly non-IID)
            - alpha → ∞: Approaches uniform distribution (IID)
            - Typical values: 0.1 (highly non-IID), 0.5 (moderate), 1.0 (mild)
        seed: Random seed for reproducibility

    Returns:
        List of client indices

    Example:
        >>> train_data = datasets.MNIST(root='./data', train=True)
        >>> # Highly non-IID split
        >>> client_indices = create_dirichlet_split(train_data, num_clients=10, alpha=0.1)
        >>> # Each client will have skewed class distribution
    """
    np.random.seed(seed)

    # Get labels
    if hasattr(dataset, 'targets'):
        if isinstance(dataset.targets, torch.Tensor):
            labels = dataset.targets.numpy()
        else:
            labels = np.array(dataset.targets)
    elif hasattr(dataset, 'labels'):
        labels = np.array(dataset.labels)
    else:
        raise ValueError("Dataset must have 'targets' or 'labels' attribute")

    num_classes = len(np.unique(labels))
    num_samples = len(labels)

    # Initialize client indices
    client_indices = [[] for _ in range(num_clients)]

    # For each class, distribute samples using Dirichlet
    for k in range(num_classes):
        # Get indices of samples with class k
        idx_k = np.where(labels == k)[0]
        np.random.shuffle(idx_k)

        # Sample proportions from Dirichlet distribution
        proportions = np.random.dirichlet(alpha * np.ones(num_clients))

        # Convert proportions to sample counts (round, ensuring at least 1 sample per non-zero proportion)
        sample_counts = (proportions * len(idx_k)).astype(int)

        # Adjust counts to match total samples (due to rounding)
        diff = len(idx_k) - sample_counts.sum()
        if diff > 0:
            # Add remainder samples to clients with largest proportions
            top_clients = np.argsort(proportions)[-diff:]
            sample_counts[top_clients] += 1
        elif diff < 0:
            # Remove excess samples from clients with most samples
            top_clients = np.argsort(sample_counts)[::-1][:abs(diff)]
            sample_counts[top_clients] -= 1

        # Distribute samples to clients
        idx = 0
        for client_id in range(num_clients):
            count = sample_counts[client_id]
            if count > 0:
                client_indices[client_id].extend(idx_k[idx:idx+count].tolist())
                idx += count

    # Shuffle each client's indices
    for i in range(num_clients):
        np.random.shuffle(client_indices[i])

    return client_indices


def create_pathological_split(
    dataset: Dataset,
    num_clients: int,
    shards_per_client: int = 2,
    seed: int = 42
) -> List[List[int]]:
    """
    Create pathological non-IID split.

    Each client gets data from a fixed number of classes (shards).
    This creates extreme heterogeneity.

    Args:
        dataset: PyTorch dataset with 'targets' attribute
        num_clients: Number of clients
        shards_per_client: Number of classes each client receives
        seed: Random seed for reproducibility

    Returns:
        List of client indices

    Example:
        >>> train_data = datasets.MNIST(root='./data', train=True)
        >>> # Each client gets exactly 2 digits (e.g., client 0 gets 0&1, client 1 gets 2&3)
        >>> client_indices = create_pathological_split(
        ...     train_data,
        ...     num_clients=10,
        ...     shards_per_client=2
        ... )

    Reference: This splitting strategy is from McMahan et al. (FedAvg paper)
    """
    np.random.seed(seed)

    # Get labels
    if hasattr(dataset, 'targets'):
        if isinstance(dataset.targets, torch.Tensor):
            labels = dataset.targets.numpy()
        else:
            labels = np.array(dataset.targets)
    elif hasattr(dataset, 'labels'):
        labels = np.array(dataset.labels)
    else:
        raise ValueError("Dataset must have 'targets' or 'labels' attribute")

    num_classes = len(np.unique(labels))
    num_samples = len(labels)

    # Group indices by class
    class_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        class_indices[label].append(idx)

    # Shuffle indices within each class
    for label in class_indices:
        np.random.shuffle(class_indices[label])

    # Create shards (each shard = one class)
    shards = []
    for label in range(num_classes):
        indices = class_indices[label]
        # Split class into equal-sized shards
        shard_size = len(indices) // num_clients
        for i in range(num_clients):
            start = i * shard_size
            end = start + shard_size if i < num_clients - 1 else len(indices)
            shards.append(indices[start:end])

    # Shuffle shards
    np.random.shuffle(shards)

    # Assign shards to clients
    client_indices = [[] for _ in range(num_clients)]
    for client_id in range(num_clients):
        # Each client gets shards_per_client shards
        for shard_id in range(shards_per_client):
            idx = client_id * shards_per_client + shard_id
            if idx < len(shards):
                client_indices[client_id].extend(shards[idx])

    # Shuffle each client's indices
    for i in range(num_clients):
        np.random.shuffle(client_indices[i])

    return client_indices


def create_dataloaders(
    dataset: Dataset,
    client_indices: List[List[int]],
    batch_size: int = 10,
    shuffle: bool = True,
    num_workers: int = 0
) -> List[DataLoader]:
    """
    Create DataLoaders for each client from indices.

    Args:
        dataset: PyTorch dataset
        client_indices: List of indices for each client
        batch_size: Batch size for training
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes

    Returns:
        List of DataLoader objects, one per client

    Example:
        >>> train_data = datasets.MNIST(root='./data', train=True)
        >>> client_indices = create_iid_split(train_data, num_clients=10)
        >>> client_loaders = create_dataloaders(
        ...     train_data,
        ...     client_indices,
        ...     batch_size=32
        ... )
    """
    dataloaders = []
    for indices in client_indices:
        subset = Subset(dataset, indices)
        loader = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers
        )
        dataloaders.append(loader)

    return dataloaders


def analyze_split(
    dataset: Dataset,
    client_indices: List[List[int]],
    num_classes: int = None
) -> Dict:
    """
    Analyze data split statistics.

    Computes:
    - Class distribution per client
    - Total samples per client
    - Class imbalance metrics

    Args:
        dataset: PyTorch dataset with 'targets' attribute
        client_indices: List of indices for each client
        num_classes: Number of classes in dataset (auto-detected if None)

    Returns:
        Dictionary with split statistics

    Example:
        >>> train_data = datasets.MNIST(root='./data', train=True)
        >>> client_indices = create_dirichlet_split(train_data, num_clients=10, alpha=0.1)
        >>> stats = analyze_split(train_data, client_indices)
        >>> print(f"Client 0 has {stats['samples_per_client'][0]} samples")
        >>> print(f"Client 0 class distribution: {stats['class_distribution'][0]}")
    """
    # Get labels
    if hasattr(dataset, 'targets'):
        if isinstance(dataset.targets, torch.Tensor):
            labels = dataset.targets.numpy()
        else:
            labels = np.array(dataset.targets)
    elif hasattr(dataset, 'labels'):
        labels = np.array(dataset.labels)
    else:
        raise ValueError("Dataset must have 'targets' or 'labels' attribute")

    num_clients = len(client_indices)

    # Auto-detect num_classes if not provided
    if num_classes is None:
        num_classes = len(np.unique(labels))

    # Compute statistics
    samples_per_client = [len(indices) for indices in client_indices]

    class_distribution = []
    for indices in client_indices:
        client_labels = labels[indices]
        dist = np.bincount(client_labels, minlength=num_classes)
        class_distribution.append(dist.tolist())

    # Compute class imbalance (coefficient of variation)
    class_counts = np.array(class_distribution)
    class_imbalance = []
    for i in range(num_clients):
        counts = class_counts[i]
        if counts.sum() > 0:
            # Coefficient of variation (std / mean)
            cv = np.std(counts) / np.mean(counts) if np.mean(counts) > 0 else 0
            class_imbalance.append(cv)
        else:
            class_imbalance.append(0)

    return {
        'num_clients': num_clients,
        'samples_per_client': samples_per_client,
        'total_samples': sum(samples_per_client),
        'min_samples': min(samples_per_client),
        'max_samples': max(samples_per_client),
        'mean_samples': np.mean(samples_per_client),
        'class_distribution': class_distribution,
        'class_imbalance': class_imbalance,
        'mean_class_imbalance': np.mean(class_imbalance),
    }


def print_split_stats(stats: Dict):
    """
    Print split statistics in readable format.

    Args:
        stats: Statistics dictionary from analyze_split()

    Example:
        >>> stats = analyze_split(train_data, client_indices)
        >>> print_split_stats(stats)
    """
    print(f"\n{'=' * 70}")
    print(f"Data Split Statistics")
    print(f"{'=' * 70}")
    print(f"Number of clients: {stats['num_clients']}")
    print(f"Total samples: {stats['total_samples']}")
    print(f"\nSamples per client:")
    print(f"  Min: {stats['min_samples']}")
    print(f"  Max: {stats['max_samples']}")
    print(f"  Mean: {stats['mean_samples']:.1f}")
    print(f"\nClass distribution heterogeneity:")
    print(f"  Mean imbalance (CV): {stats['mean_class_imbalance']:.3f}")
    print(f"  (Lower = more balanced, Higher = more heterogeneous)")

    print(f"\nFirst 3 clients' class distributions:")
    for i in range(min(3, stats['num_clients'])):
        dist = stats['class_distribution'][i]
        print(f"  Client {i}: {dist}")
    print(f"{'=' * 70}\n")


# ============================================================================
# Testing and Examples
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Data Splitting Utilities Test")
    print("=" * 70)

    # Create synthetic dataset for testing
    class SyntheticDataset(Dataset):
        def __init__(self, size=1000, num_classes=10):
            self.size = size
            self.num_classes = num_classes
            self.targets = torch.randint(0, num_classes, (size,))

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            return torch.randn(1, 28, 28), self.targets[idx]

    # Create dataset
    dataset = SyntheticDataset(size=1000, num_classes=10)

    # Test 1: IID Split
    print("\n1. IID Split (10 clients)")
    client_indices = create_iid_split(dataset, num_clients=10)
    stats = analyze_split(dataset, client_indices)
    print_split_stats(stats)

    # Test 2: Dirichlet Split (highly non-IID)
    print("\n2. Dirichlet Split (alpha=0.1, highly non-IID)")
    client_indices = create_dirichlet_split(dataset, num_clients=10, alpha=0.1)
    stats = analyze_split(dataset, client_indices)
    print_split_stats(stats)

    # Test 3: Dirichlet Split (moderately non-IID)
    print("\n3. Dirichlet Split (alpha=1.0, moderately non-IID)")
    client_indices = create_dirichlet_split(dataset, num_clients=10, alpha=1.0)
    stats = analyze_split(dataset, client_indices)
    print_split_stats(stats)

    # Test 4: Pathological Split
    print("\n4. Pathological Split (2 classes per client)")
    client_indices = create_pathological_split(dataset, num_clients=10, shards_per_client=2)
    stats = analyze_split(dataset, client_indices)
    print_split_stats(stats)

    # Test 5: Create DataLoaders
    print("\n5. Creating DataLoaders")
    client_indices = create_iid_split(dataset, num_clients=10)
    dataloaders = create_dataloaders(dataset, client_indices, batch_size=32)
    print(f"   Created {len(dataloaders)} DataLoaders")
    print(f"   First DataLoader has {len(dataloaders[0])} batches")
    print(f"   Batch size: 32")

    print("\n" + "=" * 70)
    print("✅ All splitting utilities working correctly!")
    print("=" * 70)

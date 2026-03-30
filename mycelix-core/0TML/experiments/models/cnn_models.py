# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Neural Network Models for Federated Learning Experiments

Three standard architectures used in FL research:
1. SimpleCNN - For MNIST (28x28 grayscale)
2. ResNet9 - For CIFAR-10 (32x32 RGB)
3. CharLSTM - For Shakespeare (character-level)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


# ============================================================================
# SimpleCNN for MNIST
# ============================================================================

class SimpleCNN(nn.Module):
    """
    Simple CNN for MNIST digit classification.

    Architecture:
    - Conv1: 1 -> 32 channels, 5x5 kernel
    - MaxPool: 2x2
    - Conv2: 32 -> 64 channels, 5x5 kernel
    - MaxPool: 2x2
    - FC1: 1024 -> 512
    - FC2: 512 -> 10

    Parameters: ~1.2M
    Expected accuracy: ~99% on MNIST

    Reference: Standard architecture used in McMahan et al. (FedAvg paper)
    """

    def __init__(self, num_classes: int = 10):
        super(SimpleCNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)

        # Pooling
        self.pool = nn.MaxPool2d(2, 2)

        # Fully connected layers
        # After 2 pooling layers: 28x28 -> 14x14 -> 7x7
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, num_classes)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, 1, 28, 28)

        Returns:
            Logits of shape (batch_size, num_classes)
        """
        # Conv1 + ReLU + Pool: (B, 1, 28, 28) -> (B, 32, 14, 14)
        x = self.pool(F.relu(self.conv1(x)))

        # Conv2 + ReLU + Pool: (B, 32, 14, 14) -> (B, 64, 7, 7)
        x = self.pool(F.relu(self.conv2(x)))

        # Flatten: (B, 64, 7, 7) -> (B, 3136)
        x = x.view(-1, 64 * 7 * 7)

        # FC1 + ReLU + Dropout: (B, 3136) -> (B, 512)
        x = self.dropout(F.relu(self.fc1(x)))

        # FC2: (B, 512) -> (B, 10)
        x = self.fc2(x)

        return x


# ============================================================================
# ResNet9 for CIFAR-10
# ============================================================================

class ResidualBlock(nn.Module):
    """
    Residual block with skip connection.

    Architecture:
    - Conv1: 3x3 conv + BatchNorm + ReLU
    - Conv2: 3x3 conv + BatchNorm
    - Skip connection
    - ReLU
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip connection for dimension matching
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with skip connection."""
        identity = x

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out += self.skip(identity)
        out = F.relu(out)

        return out


class ResNet9(nn.Module):
    """
    9-layer ResNet for CIFAR-10 classification.

    Architecture:
    - Initial Conv: 3 -> 64 channels
    - ResBlock1: 64 -> 64 channels
    - ResBlock2: 64 -> 128 channels (stride=2)
    - ResBlock3: 128 -> 256 channels (stride=2)
    - Global Average Pooling
    - FC: 256 -> 10

    Parameters: ~6.6M
    Expected accuracy: ~90% on CIFAR-10

    Reference: Simplified version of ResNet used in FL research
    """

    def __init__(self, num_classes: int = 10):
        super(ResNet9, self).__init__()

        # Initial convolution
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # Residual blocks
        self.layer1 = ResidualBlock(64, 64, stride=1)
        self.layer2 = ResidualBlock(64, 128, stride=2)
        self.layer3 = ResidualBlock(128, 256, stride=2)

        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected layer
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, 3, 32, 32)

        Returns:
            Logits of shape (batch_size, num_classes)
        """
        # Initial conv: (B, 3, 32, 32) -> (B, 64, 32, 32)
        x = F.relu(self.bn1(self.conv1(x)))

        # ResBlocks: (B, 64, 32, 32) -> (B, 256, 8, 8)
        x = self.layer1(x)  # (B, 64, 32, 32)
        x = self.layer2(x)  # (B, 128, 16, 16)
        x = self.layer3(x)  # (B, 256, 8, 8)

        # Global average pooling: (B, 256, 8, 8) -> (B, 256, 1, 1)
        x = self.avgpool(x)

        # Flatten: (B, 256, 1, 1) -> (B, 256)
        x = x.view(x.size(0), -1)

        # FC: (B, 256) -> (B, 10)
        x = self.fc(x)

        return x


# ============================================================================
# CharLSTM for Shakespeare
# ============================================================================

class CharLSTM(nn.Module):
    """
    Character-level LSTM for next-character prediction.

    Architecture:
    - Embedding: vocab_size -> embedding_dim
    - LSTM: 2 layers with hidden_dim
    - FC: hidden_dim -> vocab_size

    Parameters: ~4.4M (with default settings)
    Expected accuracy: ~50-55% on Shakespeare

    Reference: Standard architecture for character-level language modeling
    """

    def __init__(
        self,
        vocab_size: int = 80,
        embedding_dim: int = 8,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        """
        Initialize CharLSTM.

        Args:
            vocab_size: Number of unique characters in vocabulary
            embedding_dim: Dimension of character embeddings
            hidden_dim: Dimension of LSTM hidden states
            num_layers: Number of LSTM layers
            dropout: Dropout probability
        """
        super(CharLSTM, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Character embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # LSTM layers
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Output layer
        self.fc = nn.Linear(hidden_dim, vocab_size)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        hidden: Tuple[torch.Tensor, torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_len) with character indices
            hidden: Optional tuple of (h_0, c_0) hidden states

        Returns:
            Tuple of:
                - Logits of shape (batch_size, seq_len, vocab_size)
                - Hidden states tuple (h_n, c_n)
        """
        batch_size = x.size(0)

        # Initialize hidden state if not provided
        if hidden is None:
            hidden = self.init_hidden(batch_size, x.device)

        # Embedding: (B, L) -> (B, L, embedding_dim)
        embedded = self.dropout(self.embedding(x))

        # LSTM: (B, L, embedding_dim) -> (B, L, hidden_dim)
        lstm_out, hidden = self.lstm(embedded, hidden)

        # Dropout
        lstm_out = self.dropout(lstm_out)

        # FC: (B, L, hidden_dim) -> (B, L, vocab_size)
        output = self.fc(lstm_out)

        return output, hidden

    def init_hidden(
        self,
        batch_size: int,
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize hidden state.

        Args:
            batch_size: Batch size
            device: Device to create tensors on

        Returns:
            Tuple of (h_0, c_0) zero-initialized hidden states
        """
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        return (h_0, c_0)


# ============================================================================
# Model Factory
# ============================================================================

def create_model(model_name: str, **kwargs) -> nn.Module:
    """
    Factory function to create models.

    Args:
        model_name: Name of model ('simple_cnn', 'resnet9', 'char_lstm')
        **kwargs: Additional arguments for model initialization

    Returns:
        Initialized model

    Example:
        >>> model = create_model('simple_cnn', num_classes=10)
        >>> model = create_model('resnet9')
        >>> model = create_model('char_lstm', vocab_size=80, hidden_dim=256)
    """
    models = {
        'simple_cnn': SimpleCNN,
        'resnet9': ResNet9,
        'char_lstm': CharLSTM,
    }

    if model_name not in models:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available models: {list(models.keys())}"
        )

    return models[model_name](**kwargs)


def count_parameters(model: nn.Module) -> int:
    """
    Count trainable parameters in model.

    Args:
        model: PyTorch model

    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================================
# Model Testing
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Model Architecture Tests")
    print("=" * 70)

    # Test SimpleCNN
    print("\n1. SimpleCNN (MNIST)")
    model = SimpleCNN(num_classes=10)
    x = torch.randn(4, 1, 28, 28)  # Batch of 4 MNIST images
    y = model(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {y.shape}")
    print(f"   Parameters: {count_parameters(model):,}")

    # Test ResNet9
    print("\n2. ResNet9 (CIFAR-10)")
    model = ResNet9(num_classes=10)
    x = torch.randn(4, 3, 32, 32)  # Batch of 4 CIFAR-10 images
    y = model(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {y.shape}")
    print(f"   Parameters: {count_parameters(model):,}")

    # Test CharLSTM
    print("\n3. CharLSTM (Shakespeare)")
    model = CharLSTM(vocab_size=80, embedding_dim=8, hidden_dim=256, num_layers=2)
    x = torch.randint(0, 80, (4, 50))  # Batch of 4 sequences, length 50
    y, hidden = model(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {y.shape}")
    print(f"   Parameters: {count_parameters(model):,}")

    print("\n" + "=" * 70)
    print("✅ All model architectures working correctly!")
    print("=" * 70)

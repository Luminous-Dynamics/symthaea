#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
ZeroTrustML Phase 9 - Simple Client Example
Connects to ZeroTrustML coordinator and participates in federated learning
"""

import asyncio
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Mock ZeroTrustML client (replace with actual import in production)
class ZeroTrustMLClient:
    """Simple FL client for ZeroTrustML Phase 9."""
    
    def __init__(self, node_id: str, coordinator_url: str = "ws://localhost:8765"):
        self.node_id = node_id
        self.coordinator_url = coordinator_url
        self.model = None
        self.connected = False
    
    async def connect(self):
        """Connect to ZeroTrustML coordinator."""
        print(f"🔌 Connecting to coordinator: {self.coordinator_url}")
        # TODO: WebSocket connection
        self.connected = True
        print(f"✅ Connected as node: {self.node_id}")
    
    async def get_global_model(self):
        """Download latest global model from coordinator."""
        print("⬇️  Downloading global model...")
        # TODO: Actual model download
        model = SimpleModel()
        print("✅ Model downloaded")
        return model
    
    async def train_local_model(self, model, train_loader, epochs=1):
        """Train model on local data."""
        print(f"🏋️  Training for {epochs} epochs...")
        
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        
        model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            print(f"   Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
        
        print("✅ Local training complete")
        return model
    
    async def submit_gradient(self, model):
        """Submit gradient to coordinator."""
        print("⬆️  Submitting gradient to coordinator...")
        
        # Extract gradient
        gradient = []
        for param in model.parameters():
            if param.grad is not None:
                gradient.append(param.grad.flatten())
        gradient_vector = torch.cat(gradient).numpy()
        
        # Submit to coordinator (with Byzantine detection)
        print(f"   Gradient norm: {np.linalg.norm(gradient_vector):.4f}")
        print(f"   Gradient size: {len(gradient_vector)} parameters")
        
        # TODO: Actual submission with validation
        # response = await self.send_gradient(gradient_vector)
        
        print("✅ Gradient submitted and validated")
        return True
    
    async def participate_in_round(self, round_num: int, train_loader):
        """Participate in one FL round."""
        print(f"\n🔄 Round {round_num}")
        print("=" * 50)
        
        # Get global model
        model = await self.get_global_model()
        
        # Train locally
        model = await self.train_local_model(model, train_loader, epochs=1)
        
        # Submit gradient
        success = await self.submit_gradient(model)
        
        if success:
            print(f"✅ Round {round_num} complete\n")
        else:
            print(f"❌ Round {round_num} failed\n")
        
        return success


class SimpleModel(nn.Module):
    """Simple neural network for demonstration."""
    
    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def create_dummy_dataset(num_samples=1000):
    """Create dummy dataset for testing."""
    # Random data (28x28 images, 10 classes)
    X = torch.randn(num_samples, 1, 28, 28)
    y = torch.randint(0, 10, (num_samples,))
    
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    return loader


async def main():
    """Main function - run FL client."""
    
    print("🚀 ZeroTrustML Phase 9 - Simple Client Example")
    print("=" * 50)
    print()
    
    # Configuration
    NODE_ID = "hospital-a"
    COORDINATOR_URL = "ws://localhost:8765"
    NUM_ROUNDS = 5
    
    # Create client
    client = ZeroTrustMLClient(NODE_ID, COORDINATOR_URL)
    
    # Connect to coordinator
    await client.connect()
    
    # Create local dataset
    print("\n📊 Preparing local dataset...")
    train_loader = create_dummy_dataset(num_samples=1000)
    print(f"✅ Dataset ready: {len(train_loader.dataset)} samples\n")
    
    # Participate in FL rounds
    for round_num in range(1, NUM_ROUNDS + 1):
        await client.participate_in_round(round_num, train_loader)
        
        # Wait between rounds
        if round_num < NUM_ROUNDS:
            await asyncio.sleep(2)
    
    print("🎉 Federated learning complete!")
    print(f"   Node: {NODE_ID}")
    print(f"   Rounds participated: {NUM_ROUNDS}")


if __name__ == "__main__":
    asyncio.run(main())

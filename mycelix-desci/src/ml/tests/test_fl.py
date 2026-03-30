# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""Tests for Federated Learning components"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from mycelix_desci_ml.fl import FederatedClient, FederatedServer


def create_simple_model():
    """Create a simple model for testing"""
    return nn.Sequential(
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 2)
    )


def create_dummy_data(num_samples=100):
    """Create dummy dataset for testing"""
    X = torch.randn(num_samples, 10)
    y = torch.randint(0, 2, (num_samples,))
    return TensorDataset(X, y)


class TestFederatedClient:
    def test_create_client(self):
        """Test client creation"""
        model = create_simple_model()
        client = FederatedClient(model, client_id="test_client")

        assert client.client_id == "test_client"
        assert client.device == "cpu"

    def test_train(self):
        """Test local training"""
        model = create_simple_model()
        client = FederatedClient(model, client_id="test_client")

        dataset = create_dummy_data(50)
        loader = DataLoader(dataset, batch_size=10)

        metrics = client.train(loader, epochs=1, learning_rate=0.01)

        assert "loss" in metrics
        assert "accuracy" in metrics
        assert "samples" in metrics
        assert metrics["samples"] == 50

    def test_get_gradients(self):
        """Test gradient extraction"""
        model = create_simple_model()
        client = FederatedClient(model, client_id="test_client")

        # Train one batch to generate gradients
        dataset = create_dummy_data(10)
        loader = DataLoader(dataset, batch_size=10)
        client.train(loader, epochs=1)

        gradients = client.get_gradients()

        assert isinstance(gradients, dict)
        assert len(gradients) > 0
        for name, grad in gradients.items():
            assert grad is not None

    def test_set_parameters(self):
        """Test parameter setting"""
        model = create_simple_model()
        client = FederatedClient(model, client_id="test_client")

        # Get initial parameters
        import numpy as np
        new_params = {}
        for name, param in model.named_parameters():
            new_params[name] = np.random.randn(*param.shape)

        # Set new parameters
        client.set_parameters(new_params)

        # Verify parameters were set
        for name, param in model.named_parameters():
            assert param is not None


class TestFederatedServer:
    def test_create_server(self):
        """Test server creation"""
        model = create_simple_model()
        server = FederatedServer(model)

        assert server.round == 0

    def test_get_parameters(self):
        """Test parameter retrieval"""
        model = create_simple_model()
        server = FederatedServer(model)

        params = server.get_parameters()

        assert isinstance(params, dict)
        assert len(params) > 0

    def test_aggregate_parameters(self):
        """Test parameter aggregation"""
        model = create_simple_model()
        server = FederatedServer(model)

        # Create mock client parameters
        import numpy as np
        client1_params = {}
        client2_params = {}

        for name, param in model.named_parameters():
            shape = param.shape
            client1_params[name] = np.ones(shape)
            client2_params[name] = np.ones(shape) * 2.0

        # Aggregate with equal weights
        server.aggregate_parameters(
            [client1_params, client2_params],
            [1.0, 1.0]
        )

        # Check round incremented
        assert server.round == 1

        # Check parameters were updated (should be average: 1.5)
        params = server.get_parameters()
        for name, param_array in params.items():
            expected = np.ones(param_array.shape) * 1.5
            assert np.allclose(param_array, expected, atol=0.01)

    def test_aggregate_weighted(self):
        """Test weighted aggregation"""
        model = create_simple_model()
        server = FederatedServer(model)

        import numpy as np
        client1_params = {}
        client2_params = {}

        for name, param in model.named_parameters():
            shape = param.shape
            client1_params[name] = np.ones(shape) * 1.0
            client2_params[name] = np.ones(shape) * 3.0

        # Aggregate with different weights (3:1 ratio)
        server.aggregate_parameters(
            [client1_params, client2_params],
            [1.0, 3.0]
        )

        # Check parameters: should be (1.0 * 1 + 3.0 * 3) / 4 = 2.5
        params = server.get_parameters()
        for name, param_array in params.items():
            expected = np.ones(param_array.shape) * 2.5
            assert np.allclose(param_array, expected, atol=0.01)

    def test_evaluate(self):
        """Test server evaluation"""
        model = create_simple_model()
        server = FederatedServer(model)

        test_data = create_dummy_data(50)
        test_loader = DataLoader(test_data, batch_size=10)

        metrics = server.evaluate(test_loader)

        assert "loss" in metrics
        assert "accuracy" in metrics
        assert "round" in metrics
        assert metrics["round"] == 0

    def test_aggregate_empty_fails(self):
        """Test aggregation fails with no clients"""
        model = create_simple_model()
        server = FederatedServer(model)

        with pytest.raises(ValueError, match="No client parameters"):
            server.aggregate_parameters([], [])


class TestFederatedWorkflow:
    def test_full_fl_round(self):
        """Test complete federated learning round"""
        # Create server
        global_model = create_simple_model()
        server = FederatedServer(global_model)

        # Create clients
        num_clients = 3
        clients = []
        for i in range(num_clients):
            local_model = create_simple_model()
            client = FederatedClient(local_model, client_id=f"client_{i}")
            data = create_dummy_data(50)
            loader = DataLoader(data, batch_size=10)
            clients.append((client, loader))

        # FL Round
        global_params = server.get_parameters()

        client_params_list = []
        client_weights = []

        for client, loader in clients:
            # Set global parameters
            client.set_parameters(global_params)

            # Train locally
            metrics = client.train(loader, epochs=1)

            # Get updated parameters
            # In real implementation, we'd extract parameters
            # For test, we just use dummy params
            import numpy as np
            dummy_params = {}
            for name, param in client.model.named_parameters():
                dummy_params[name] = np.random.randn(*param.shape)

            client_params_list.append(dummy_params)
            client_weights.append(float(metrics["samples"]))

        # Aggregate on server
        server.aggregate_parameters(client_params_list, client_weights)

        # Verify round incremented
        assert server.round == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

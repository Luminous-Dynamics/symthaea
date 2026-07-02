# Mycelix-DeSci ML

Machine Learning and Federated Learning components for Mycelix-DeSci.

## Features

- **PoGQ (Proof of Gradient Quality)**: Byzantine-resistant gradient validation
- **Federated Learning**: Client/server implementation for collaborative training
- **Bio-specific tools**: Integration with BioPython for biomedical datasets

## Installation

```bash
cd src/ml
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -e .
```

## Usage

### PoGQ Validation

```python
from mycelix_desci_ml.pogq import PoGQValidator, GradientUpdate
import numpy as np

# Create validator
validator = PoGQValidator(bft_threshold=0.45)

# Simulate gradient updates
gradients = [
    GradientUpdate(
        participant_id=f"participant_{i}",
        gradients=np.random.randn(100),
        timestamp=float(i)
    )
    for i in range(10)
]

# Validate
scores = validator.validate_gradients(gradients)

# Detect Byzantine actors
byzantine = validator.detect_byzantine(gradients, scores)
print(f"Byzantine actors: {byzantine}")

# Aggregate valid gradients
aggregated = validator.aggregate_gradients(gradients, scores)
```

### Federated Learning

```python
from mycelix_desci_ml.fl import FederatedClient, FederatedServer
import torch
import torch.nn as nn

# Define model
model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

# Create client
client = FederatedClient(model, client_id="client_1")

# Train locally
metrics = client.train(train_loader, epochs=1)
print(f"Training metrics: {metrics}")

# Get gradients for sharing
gradients = client.get_gradients()
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black .

# Type checking
mypy mycelix_desci_ml
```

## License

MIT

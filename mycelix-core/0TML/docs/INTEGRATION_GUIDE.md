# 🔗 Zero-TrustML-Holochain Integration Guide

**Complete guide to adding Holochain credits to your Zero-TrustML federated learning system**

---

## 📋 Table of Contents

1. [Quick Start](#quick-start)
2. [Installation](#installation)
3. [Basic Integration](#basic-integration)
4. [Advanced Patterns](#advanced-patterns)
5. [Configuration](#configuration)
6. [Testing](#testing)
7. [Production Deployment](#production-deployment)
8. [Troubleshooting](#troubleshooting)

---

## 🚀 Quick Start

Add Holochain credits to your federated learning system in 3 steps:

```python
from holochain_credits_bridge import HolochainBridge

# 1. Initialize bridge
bridge = HolochainBridge("ws://localhost:8888", enabled=True)

# 2. Issue credits for valid contributions
action_hash = bridge.issue_credits(
    node_id=42,
    event_type="model_update",
    amount=100,
    pogq_score=0.95
)

# 3. Query node reputation
balance = bridge.get_balance(42)
print(f"Node reputation: {balance} credits")
```

**That's it!** Your system now has:
- ✅ Decentralized reputation tracking
- ✅ Byzantine node detection
- ✅ Trust-based node selection
- ✅ Persistent credit history

---

## 📦 Installation

### Prerequisites

- Python 3.8+
- Holochain conductor running (optional for mock mode)
- Rust toolchain (for building from source)

### Install from Source

```bash
# Clone the repository
git clone https://github.com/your-org/0TML.git
cd 0TML

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install maturin

# Build and install the Rust bridge
cd rust-bridge
maturin develop --release
cd ..

# Verify installation
python -c "from holochain_credits_bridge import HolochainBridge; print('✅ Bridge installed')"
```

### Install Pre-built Wheel (Coming Soon)

```bash
pip install holochain-credits-bridge
```

---

## 🔧 Basic Integration

### Step 1: Add Bridge to Your Coordinator

```python
from holochain_credits_bridge import HolochainBridge

class FederatedCoordinator:
    def __init__(self, conductor_url: str = "ws://localhost:8888"):
        # Initialize Holochain bridge
        self.bridge = HolochainBridge(conductor_url, enabled=True)
        print(f"✅ Holochain bridge initialized")

        # Your existing initialization code
        self.nodes = []
        self.global_model = YourModel()
```

### Step 2: Issue Credits After Validation

```python
def validate_and_reward(self, node_id: int, gradient: np.ndarray):
    """Validate gradient and issue Holochain credits"""

    # Your existing validation logic
    is_valid, quality_score = self.validate_gradient(gradient)

    if is_valid:
        # Calculate credits based on quality (0-100)
        credits_earned = int(quality_score * 100)

        # Issue credits via Holochain
        try:
            action_hash = self.bridge.issue_credits(
                node_id=node_id,
                event_type="gradient_contribution",
                amount=credits_earned,
                pogq_score=quality_score
            )
            print(f"✅ Issued {credits_earned} credits to node {node_id}")
        except Exception as e:
            print(f"⚠️  Credit issuance failed: {e}")
            # Fall back to local tracking if needed
```

### Step 3: Use Credits for Trust-Based Selection

```python
def select_trusted_nodes(self, min_credits: int = 500, count: int = 10):
    """Select most trusted nodes based on credit balance"""

    # Get credit balances for all nodes
    node_trust = []
    for node_id in self.node_ids:
        try:
            balance = self.bridge.get_balance(node_id)
            node_trust.append((node_id, balance))
        except Exception as e:
            print(f"⚠️  Failed to get balance for node {node_id}: {e}")
            node_trust.append((node_id, 0))

    # Filter and sort by trust
    trusted = [(nid, bal) for nid, bal in node_trust if bal >= min_credits]
    trusted.sort(key=lambda x: x[1], reverse=True)

    return [node_id for node_id, _ in trusted[:count]]
```

---

## 🎯 Advanced Patterns

### Pattern 1: Trust-Weighted Aggregation

Weight model updates by node reputation:

```python
def aggregate_with_trust(self, gradients: List[Tuple[int, np.ndarray]]):
    """Aggregate gradients weighted by node trust"""

    # Get trust scores (credit balances)
    trust_scores = {}
    for node_id, _ in gradients:
        trust_scores[node_id] = self.bridge.get_balance(node_id)

    # Normalize trust scores to weights
    total_trust = sum(trust_scores.values())
    weights = {nid: score / total_trust
               for nid, score in trust_scores.items()}

    # Weighted aggregation
    aggregated = np.zeros_like(gradients[0][1])
    for node_id, gradient in gradients:
        aggregated += weights[node_id] * gradient

    return aggregated
```

### Pattern 2: Dynamic Trust Thresholds

Adjust trust requirements based on network conditions:

```python
def adaptive_trust_threshold(self):
    """Calculate adaptive minimum trust threshold"""

    # Get all node balances
    balances = []
    for node_id in self.node_ids:
        balances.append(self.bridge.get_balance(node_id))

    # Set threshold at 60th percentile
    balances.sort()
    threshold_idx = int(len(balances) * 0.6)
    min_threshold = balances[threshold_idx]

    print(f"📊 Adaptive threshold: {min_threshold} credits")
    return min_threshold
```

### Pattern 3: Byzantine Node Isolation

Automatically isolate nodes with consistently low trust:

```python
def check_and_isolate_byzantine(self):
    """Identify and isolate potential Byzantine nodes"""

    isolated_nodes = []
    for node_id in self.node_ids:
        balance = self.bridge.get_balance(node_id)
        history = self.bridge.get_history(node_id)

        # Byzantine detection criteria
        if balance < 100 and len(history) > 5:
            isolated_nodes.append(node_id)
            print(f"🚨 Node {node_id} isolated (balance: {balance})")

    # Remove from active pool
    self.active_nodes = [n for n in self.active_nodes
                         if n not in isolated_nodes]

    return isolated_nodes
```

### Pattern 4: Reputation Decay

Implement time-based reputation decay:

```python
def apply_reputation_decay(self, decay_rate: float = 0.95):
    """Apply decay to node reputations over time"""

    for node_id in self.node_ids:
        current_balance = self.bridge.get_balance(node_id)

        # Calculate decayed balance
        decayed = int(current_balance * decay_rate)
        decay_amount = current_balance - decayed

        if decay_amount > 0:
            # Issue negative credits for decay (if supported)
            # Or track separately
            self.local_decay_tracking[node_id] = decay_amount
```

---

## ⚙️ Configuration

### Bridge Configuration Options

```python
bridge = HolochainBridge(
    conductor_url="ws://localhost:8888",  # Conductor WebSocket URL
    enabled=True,                         # Enable/disable Holochain backend
    timeout=30.0,                         # Connection timeout (seconds)
    retry_attempts=3,                     # Retry failed operations
)
```

### Environment Variables

```bash
# Conductor URL
export HOLOCHAIN_CONDUCTOR_URL="ws://localhost:8888"

# Enable/disable Holochain backend
export HOLOCHAIN_ENABLED="true"

# Connection timeout
export HOLOCHAIN_TIMEOUT="30"
```

### Mock Mode (No Conductor Required)

```python
# Bridge automatically falls back to mock mode if conductor unavailable
bridge = HolochainBridge("ws://localhost:8888", enabled=True)

# Check if running in mock mode
if not bridge.is_connected():
    print("⚠️  Running in mock mode (no conductor)")
```

---

## 🧪 Testing

### Unit Tests

```python
import pytest
from holochain_credits_bridge import HolochainBridge

def test_credit_issuance():
    """Test credit issuance and balance tracking"""
    bridge = HolochainBridge("ws://localhost:8888", enabled=True)

    # Issue credits
    action_hash = bridge.issue_credits(
        node_id=1,
        event_type="test",
        amount=100,
        pogq_score=0.95
    )

    assert action_hash is not None
    assert len(action_hash) > 0

    # Check balance
    balance = bridge.get_balance(1)
    assert balance == 100

def test_history_retrieval():
    """Test credit history retrieval"""
    bridge = HolochainBridge("ws://localhost:8888", enabled=True)

    # Issue multiple credits
    for i in range(5):
        bridge.issue_credits(
            node_id=2,
            event_type="test",
            amount=10,
            pogq_score=0.9
        )

    # Retrieve history
    history = bridge.get_history(2)
    assert len(history) == 5
    assert all(event.amount == 10 for event in history)
```

### Integration Tests

```python
def test_federated_learning_with_credits():
    """Test full federated learning round with credits"""
    coordinator = FederatedCoordinator()

    # Register nodes
    for i in range(10):
        coordinator.register_node(NodeID=i)

    # Run training round
    coordinator.training_round()

    # Check that honest nodes got credits
    for node_id in range(8):
        balance = coordinator.bridge.get_balance(node_id)
        assert balance > 0

    # Check that Byzantine nodes got no credits
    for node_id in range(8, 10):
        balance = coordinator.bridge.get_balance(node_id)
        assert balance == 0
```

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=holochain_credits_bridge --cov-report=html
```

---

## 🚀 Production Deployment

### Conductor Setup

1. **Install Holochain conductor**:
```bash
# Using Nix
nix-shell -p holochain

# Or download binary
wget https://github.com/holochain/holochain/releases/latest/download/holochain
chmod +x holochain
```

2. **Create conductor configuration**:
```yaml
# conductor-config.yaml
---
network:
  bootstrap_service: https://bootstrap.holo.host
  transport_pool:
    - type: webrtc
      signal_url: wss://signal.holo.host

admin_interfaces:
  - driver:
      type: websocket
      port: 8888

app_interfaces:
  - driver:
      type: websocket
      port: 8889
```

3. **Start conductor**:
```bash
holochain --structured -c conductor-config.yaml
```

### Install Zero-TrustML DNA

```python
# install_happ.py
import asyncio
from install_happ import install_happ

async def main():
    await install_happ(
        conductor_url="ws://localhost:8888",
        happ_path="zerotrustml-dna/zerotrustml.happ"
    )

if __name__ == "__main__":
    asyncio.run(main())
```

### Docker Deployment

```dockerfile
# Dockerfile
FROM rust:1.70 as builder

# Install Holochain
RUN cargo install holochain --version 0.2.4

# Build Zero-TrustML bridge
WORKDIR /app
COPY rust-bridge rust-bridge
RUN cd rust-bridge && cargo build --release

FROM python:3.11

# Copy Holochain and bridge
COPY --from=builder /usr/local/cargo/bin/holochain /usr/local/bin/
COPY --from=builder /app/rust-bridge/target/release/*.so /usr/local/lib/

# Install Zero-TrustML
COPY . /app
WORKDIR /app
RUN pip install -e .

# Run conductor and Zero-TrustML
CMD holochain -c conductor-config.yaml & \
    sleep 5 && \
    python install_happ.py && \
    python your_federated_learning_app.py
```

### Monitoring

```python
def monitor_network_health(bridge: HolochainBridge):
    """Monitor Holochain network health"""

    stats = bridge.get_system_stats()

    print(f"Network Statistics:")
    print(f"  Total nodes: {stats['total_nodes']}")
    print(f"  Total credits: {stats['total_credits']}")
    print(f"  Average balance: {stats['avg_balance']:.1f}")

    # Alert if network unhealthy
    if stats['total_nodes'] < 5:
        print("⚠️  WARNING: Low node count")

    if stats['avg_balance'] < 100:
        print("⚠️  WARNING: Low average reputation")
```

---

## 🔍 Troubleshooting

### Issue: Bridge fails to connect

**Symptoms**: `ConnectionError: Failed to connect to conductor`

**Solutions**:
1. Check conductor is running:
   ```bash
   ps aux | grep holochain
   ```
2. Verify conductor URL:
   ```python
   bridge = HolochainBridge("ws://localhost:8888", enabled=True)
   ```
3. Check firewall settings
4. Fall back to mock mode:
   ```python
   # Mock mode works without conductor
   bridge = HolochainBridge("ws://localhost:8888", enabled=False)
   ```

### Issue: Credit issuance fails

**Symptoms**: `Exception: Failed to issue credits`

**Solutions**:
1. Check DNA is installed:
   ```bash
   python install_happ.py
   ```
2. Verify node_id format (must be integer)
3. Check pogq_score is 0.0-1.0 range
4. Review conductor logs

### Issue: Balance queries return 0

**Symptoms**: All nodes have 0 credits

**Solutions**:
1. Verify credits were issued successfully
2. Check correct node_id being queried
3. Ensure DNA is installed and activated
4. Review credit issuance logs

### Issue: Mock mode performance

**Symptoms**: Slow operations in mock mode

**Solutions**:
Mock mode should be instant (<1ms). If slow:
1. Check Python version (3.8+ required)
2. Reduce number of nodes
3. Clear history periodically
4. Use batch operations

---

## 📚 API Reference

### HolochainBridge

```python
class HolochainBridge:
    def __init__(self, conductor_url: str, enabled: bool = True) -> None:
        """Initialize bridge to Holochain conductor"""

    def issue_credits(
        self,
        node_id: int,
        event_type: str,
        amount: int,
        pogq_score: Optional[float] = None,
        verifiers: Optional[List[int]] = None
    ) -> str:
        """Issue credits to a node

        Args:
            node_id: Node identifier (integer)
            event_type: Type of event ("gradient", "model_update", etc.)
            amount: Number of credits to issue
            pogq_score: Proof of Gradient Quality score (0.0-1.0)
            verifiers: Optional list of verifier node IDs

        Returns:
            Action hash (string)
        """

    def get_balance(self, node_id: int) -> int:
        """Get total credit balance for a node"""

    def get_history(self, node_id: int) -> List[CreditIssuance]:
        """Get credit history for a node"""

    def get_system_stats(self) -> Dict[str, Any]:
        """Get network-wide statistics"""

    def is_connected(self) -> bool:
        """Check if connected to conductor"""
```

### CreditIssuance

```python
@dataclass
class CreditIssuance:
    action_hash: str
    node_id: int
    event_type: str
    amount: int
    pogq_score: float
    timestamp: int
```

---

## 🎓 Best Practices

### 1. Credit Amounts

- Use 0-100 range for easy interpretation
- Base on quality score: `credits = int(quality * 100)`
- Consider magnitude of contributions
- Keep consistent across event types

### 2. Trust Thresholds

- Start with low thresholds (e.g., 100 credits)
- Increase as network matures
- Use adaptive thresholds based on percentiles
- Monitor threshold effectiveness

### 3. Event Types

Use descriptive event types:
- `gradient_contribution`: Federated learning gradients
- `model_update`: Full model updates
- `validation_work`: Validation contributions
- `data_contribution`: Dataset contributions

### 4. Error Handling

Always wrap bridge calls in try-except:
```python
try:
    action_hash = bridge.issue_credits(...)
except Exception as e:
    logger.error(f"Credit issuance failed: {e}")
    # Fall back to local tracking
```

### 5. Testing

- Test with mock mode first
- Test Byzantine scenarios
- Test network partitions
- Load test with many nodes

---

## 🔗 Resources

- **Documentation**: [/docs](/docs)
- **Examples**: [/examples](/examples)
- **API Reference**: [/docs/API.md](/docs/API.md)
- **Holochain Docs**: https://docs.holochain.org
- **GitHub Issues**: https://github.com/your-org/0TML/issues

---

## 📝 License

MIT License - See [LICENSE](LICENSE) for details

---

**Questions?** Open an issue or join our Discord server!

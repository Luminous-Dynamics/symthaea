# Zero-TrustML Credits Tutorial: Getting Started

**Welcome to Zero-TrustML Credits!** 🎉

This tutorial will guide you through understanding and using the Zero-TrustML Credits system, from basic concepts to running your first integration.

---

## What is Zero-TrustML Credits?

Zero-TrustML Credits is an **economic incentive layer** for distributed machine learning. It automatically rewards participants for:

- **Quality contributions** to the model (good gradients)
- **Network security** (detecting malicious nodes)
- **Validation work** (checking others' contributions)
- **Network reliability** (staying online)

Credits are stored on a **decentralized ledger** (Holochain), ensuring transparency and preventing cheating.

---

## 🎓 Core Concepts

### 1. Credits = Digital Currency

Think of credits like points in a game:
- **Earn** credits by contributing to the network
- **Spend** credits on services (future feature)
- **Transfer** credits to other users (future feature)

### 2. Four Ways to Earn Credits

| Event Type | What You Do | Credits Earned |
|------------|-------------|----------------|
| **Quality Gradient** | Submit good ML gradients | 0-100 (based on quality) |
| **Byzantine Detection** | Catch malicious nodes | 50 (fixed reward) |
| **Peer Validation** | Validate others' work | 10 (fixed reward) |
| **Network Uptime** | Stay online | 1 per hour (up to 24/day) |

### 3. Reputation Multipliers

Your reputation affects how many credits you earn:

| Reputation | Multiplier | Meaning |
|------------|-----------|---------|
| BLACKLISTED | 0.0× | No credits (banned) |
| CRITICAL | 0.5× | Half credits (under review) |
| WARNING | 0.75× | Reduced credits |
| NORMAL | 1.0× | Standard rate |
| TRUSTED | 1.2× | +20% bonus |
| ELITE | 1.5× | +50% bonus |

**Example**: If you earn 100 credits with ELITE reputation, you get 150 credits (100 × 1.5).

### 4. Rate Limiting (Anti-Spam)

To prevent abuse, there are limits:
- **Quality Gradients**: 10,000 credits per hour
- **Byzantine Detection**: 2,000 credits per day
- **Peer Validation**: 1,000 credits per hour
- **Network Uptime**: 24 credits per day

---

## 🚀 Quick Start: Run the Demo

The easiest way to learn is by running the interactive demo!

### Step 1: Navigate to Project

```bash
cd /srv/luminous-dynamics/Mycelix-Core/0TML
```

### Step 2: Enter Development Environment

```bash
nix develop
```

This gives you Python 3.13 + PyTorch + all dependencies.

### Step 3: Run the Demo

```bash
python3 demos/demo_zerotrustml_credits_integration.py
```

**What You'll See**:
- 4 demos showing each credit type
- Audit trails for each participant
- Statistics and rate limiting in action
- Complete summary of credits earned

**Time**: ~30 seconds

---

## 📖 Understanding the Demo

Let's break down what happens in the demo:

### Demo 1: Quality Gradient Credits

```
1️⃣ Node 'alice' - ELITE reputation, excellent gradient (PoGQ=0.98)
   • PoGQ Score: 0.98
   • Reputation: ELITE (1.5x multiplier)
   • Base Credits: 0.98 * 100 = 98.0
   • Final Credits: 98.0 * 1.5 = 147.0
```

**What's happening**:
1. Alice submits a gradient with quality score 0.98 (excellent!)
2. Base credits = 0.98 × 100 = 98 credits
3. Alice has ELITE reputation (1.5× multiplier)
4. Final credits = 98 × 1.5 = **147 credits**

### Demo 2: Byzantine Detection Rewards

```
1️⃣ Node 'alice' detects Byzantine node 'mallory'
   • Detector Reputation: ELITE (1.5x multiplier)
   • Base Reward: 50 credits
   • Final Reward: 50 * 1.5 = 75.0 credits
```

**What's happening**:
1. Alice catches a malicious node (mallory) misbehaving
2. Base reward = 50 credits (fixed for Byzantine detection)
3. With ELITE reputation: 50 × 1.5 = **75 credits**

### Demo 3: Peer Validation Credits

```
1️⃣ Node 'charlie' validates node 'alice'
   • Reputation: NORMAL (1.0x multiplier)
   • Base Credits: 10
   • Final Credits: 10 * 1.0 = 10.0
```

**What's happening**:
1. Charlie validates Alice's gradient
2. Base reward = 10 credits (fixed for validation)
3. With NORMAL reputation: 10 × 1.0 = **10 credits**

### Demo 4: Network Uptime Credits

```
1️⃣ Node 'alice' - 99% uptime (excellent)
   • Uptime: 99% (above 95% threshold)
   • Reputation: ELITE (1.5x multiplier)
   • Base Credits: 1.0 credit/hour
   • Final Credits: 1.0 * 1.5 = 1.5
```

**What's happening**:
1. Alice maintained 99% uptime (above 95% threshold)
2. Base reward = 1 credit per hour
3. With ELITE reputation: 1 × 1.5 = **1.5 credits**

---

## 💻 Your First Integration

Let's write code to issue credits!

### Basic Example: Reward Quality Gradient

```python
import asyncio
from zerotrustml_credits_integration import (
    Zero-TrustMLCreditsIntegration,
    CreditIssuanceConfig
)
from holochain_credits_bridge import HolochainCreditsBridge

async def reward_quality_gradient():
    # Initialize (mock mode for learning)
    bridge = HolochainCreditsBridge(enabled=False)  # Mock mode
    integration = Zero-TrustMLCreditsIntegration(bridge)

    # Issue credits for quality gradient
    credit_id = await integration.on_quality_gradient(
        node_id="alice",
        pogq_score=0.95,         # 95% quality
        reputation_level="NORMAL",
        verifiers=["bob", "charlie"]  # Who validated this
    )

    print(f"✅ Issued credit ID: {credit_id}")
    # Output: ✅ Issued credit ID: 95

# Run it
asyncio.run(reward_quality_gradient())
```

**Result**: Alice earns 95 credits (0.95 × 100 × 1.0).

### Example: Detect Byzantine Node

```python
async def reward_byzantine_detection():
    bridge = HolochainCreditsBridge(enabled=False)
    integration = Zero-TrustMLCreditsIntegration(bridge)

    # Reward detector
    credit_id = await integration.on_byzantine_detection(
        detector_node_id="bob",
        detected_node_id="mallory",
        reputation_level="TRUSTED",  # Bob is trusted
        evidence={
            "consecutive_failures": 20,
            "pogq_score": 0.15
        }
    )

    print(f"✅ Bob earned reward: {credit_id}")
    # Bob earns 60 credits (50 × 1.2)

asyncio.run(reward_byzantine_detection())
```

### Example: Custom Configuration

```python
# Create custom configuration
config = CreditIssuanceConfig(
    enabled=True,
    max_quality_credits_per_hour=5000,  # Lower limit
    min_pogq_score=0.8,                  # Higher quality bar
    reputation_multipliers={
        "NORMAL": 1.0,
        "TRUSTED": 1.3,  # Custom multiplier
        "ELITE": 2.0     # Higher elite bonus
    }
)

bridge = HolochainCreditsBridge(enabled=False)
integration = Zero-TrustMLCreditsIntegration(bridge, config)
```

---

## 🔍 Common Scenarios

### Scenario 1: Node Below Quality Threshold

```python
credit_id = await integration.on_quality_gradient(
    node_id="eve",
    pogq_score=0.65,  # Below 0.7 threshold
    reputation_level="NORMAL",
    verifiers=["alice"]
)

# credit_id = None (no credits issued)
```

**Why**: PoGQ score (0.65) is below minimum (0.7).

### Scenario 2: Rate Limit Hit

```python
# Try to issue too many credits rapidly
for i in range(200):
    credit_id = await integration.on_quality_gradient(
        node_id="spam_node",
        pogq_score=1.0,
        reputation_level="NORMAL",
        verifiers=["validator"]
    )
    if not credit_id:
        print(f"Rate limited after {i} credits")
        break
```

**Why**: Hourly limit (10,000 credits) reached.

### Scenario 3: Blacklisted Node

```python
credit_id = await integration.on_quality_gradient(
    node_id="banned_node",
    pogq_score=0.99,
    reputation_level="BLACKLISTED",  # Banned!
    verifiers=["alice"]
)

# credit_id = None (zero multiplier)
```

**Why**: BLACKLISTED reputation has 0.0× multiplier.

---

## 📊 Monitoring and Audit

### View Audit Trail

```python
# Get complete history for a node
audit = await integration.get_audit_trail("alice")

for record in audit:
    print(f"Event: {record['event_type']}")
    print(f"Credits: {record['credits_issued']:.2f}")
    print(f"Time: {record['timestamp']}")
```

### Get Statistics

```python
stats = integration.get_integration_stats()

print(f"Total Credits Issued: {stats['total_credits_issued']:.2f}")
print(f"Total Events: {stats['total_events']}")
print(f"Total Nodes: {stats['total_nodes']}")
```

### Check Rate Limits

```python
rate_stats = integration.get_rate_limit_stats("alice")

for event_type, stats in rate_stats['by_event_type'].items():
    print(f"{event_type}:")
    print(f"  Hourly: {stats['hourly']:.2f} credits")
    print(f"  Daily: {stats['daily']:.2f} credits")
```

---

## 🎯 Best Practices

### 1. Always Use Accurate Reputation

```python
# ❌ BAD: Hardcoded reputation
reputation_level = "NORMAL"

# ✅ GOOD: Get from reputation system
reputation_level = get_node_reputation(node_id)
```

### 2. Provide Evidence for Byzantine Detection

```python
# ❌ BAD: No evidence
await integration.on_byzantine_detection(
    detector_node_id="alice",
    detected_node_id="mallory",
    reputation_level="NORMAL"
)

# ✅ GOOD: Detailed evidence
await integration.on_byzantine_detection(
    detector_node_id="alice",
    detected_node_id="mallory",
    reputation_level="NORMAL",
    evidence={
        "attack_type": "gradient_poisoning",
        "consecutive_failures": 15,
        "pogq_score": 0.18,
        "anomaly_score": 0.94
    }
)
```

### 3. Monitor Rate Limits

```python
# Check before expensive operations
stats = integration.get_rate_limit_stats(node_id)
quality_credits = stats['by_event_type']['quality_gradient']

if quality_credits['hourly'] > 9000:  # Near 10k limit
    print(f"⚠️  Warning: Near rate limit")
```

### 4. Use Mock Mode for Development

```python
# Development/Testing
bridge = HolochainCreditsBridge(enabled=False)  # Mock mode

# Production
bridge = HolochainCreditsBridge(
    conductor_url="ws://production-server:8888",
    enabled=True
)
await bridge.connect()
```

---

## 🔧 Troubleshooting

### Problem: No credits issued

```python
credit_id = await integration.on_quality_gradient(...)
# credit_id is None
```

**Possible Causes**:
1. PoGQ score below threshold (0.7)
2. Rate limit exceeded
3. Reputation is BLACKLISTED
4. System disabled (`config.enabled = False`)

**Solution**:
```python
# Check configuration
print(f"Enabled: {integration.config.enabled}")
print(f"Min PoGQ: {integration.config.min_pogq_score}")

# Check rate limit
stats = integration.get_rate_limit_stats(node_id)
print(f"Hourly usage: {stats['by_event_type']['quality_gradient']['hourly']}")

# Check reputation multiplier
multiplier = integration.config.reputation_multipliers[reputation]
print(f"Reputation multiplier: {multiplier}")
```

### Problem: Can't connect to Holochain

```python
success = await bridge.connect()
# success is False
```

**Solutions**:
1. **Check Holochain is running**:
   ```bash
   hc sandbox call --running
   ```

2. **Verify conductor URL**:
   ```python
   bridge = HolochainCreditsBridge(
       conductor_url="ws://localhost:8888"  # Check port!
   )
   ```

3. **Use mock mode for testing**:
   ```python
   bridge = HolochainCreditsBridge(enabled=False)
   ```

---

## 🎓 Next Steps

### 1. Read the API Documentation

See `ZEROTRUSTML_CREDITS_API_DOCUMENTATION.md` for complete API reference.

### 2. Review Test Examples

Check `tests/test_zerotrustml_credits_integration.py` for comprehensive examples.

### 3. Run Real Integration

Once comfortable with mock mode, connect to real Holochain:

```bash
# Start Holochain conductor
hc sandbox run

# In your code
bridge = HolochainCreditsBridge(enabled=True)
await bridge.connect()
```

### 4. Monitor Production

Set up monitoring for:
- Credit issuance rate
- Rate limit violations
- Byzantine detections
- Node reputations

---

## 💡 Key Takeaways

1. **Four credit types**: Quality, Byzantine detection, Validation, Uptime
2. **Reputation matters**: 0.0× - 1.5× multiplier affects earnings
3. **Rate limits prevent spam**: Hourly/daily caps on each event type
4. **Mock mode for learning**: Test without real Holochain
5. **Audit trails**: Complete history for transparency
6. **Evidence helps**: Document Byzantine detections thoroughly

---

## 📚 Resources

- **API Documentation**: `ZEROTRUSTML_CREDITS_API_DOCUMENTATION.md`
- **Demo Script**: `demos/demo_zerotrustml_credits_integration.py`
- **Integration Tests**: `tests/test_zerotrustml_credits_integration.py`
- **Source Code**: `src/zerotrustml_credits_integration.py`

---

## 🤝 Getting Help

**Questions?** Check the troubleshooting section or refer to the API documentation.

**Found a Bug?** Report it to the Zero-TrustML development team with:
- Code that reproduces the issue
- Expected vs actual behavior
- Logs and error messages

---

## 🎉 Congratulations!

You now understand the Zero-TrustML Credits system! You can:
- ✅ Run the interactive demo
- ✅ Issue credits for different events
- ✅ Configure policies and rate limits
- ✅ Monitor audit trails and statistics
- ✅ Troubleshoot common issues

**Ready to build?** Start with the demo, then integrate into your federated learning workflow!

---

*Happy coding! 🚀*

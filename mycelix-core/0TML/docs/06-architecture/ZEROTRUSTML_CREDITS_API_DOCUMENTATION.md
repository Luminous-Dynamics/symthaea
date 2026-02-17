# Zero-TrustML Credits API Documentation

**Version**: 1.0
**Last Updated**: October 1, 2025
**Status**: Production Ready

---

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Core Classes](#core-classes)
4. [Event Handlers](#event-handlers)
5. [Configuration](#configuration)
6. [Rate Limiting](#rate-limiting)
7. [Usage Examples](#usage-examples)
8. [Best Practices](#best-practices)
9. [Troubleshooting](#troubleshooting)

---

## Overview

The Zero-TrustML Credits system provides economic incentives for participation in the Zero-TrustML federated learning network. It automatically issues decentralized credits (stored on Holochain DHT) for positive contributions:

- **Quality Gradients**: Reward high-quality machine learning contributions
- **Byzantine Detection**: Reward nodes that catch malicious behavior
- **Peer Validation**: Reward nodes that validate others' contributions
- **Network Contribution**: Reward consistent network participation

### Architecture

```
┌─────────────────────────────────────────────────┐
│           Zero-TrustML Application Layer             │
│  (Federated Learning, Byzantine Detection)      │
└───────────────────┬─────────────────────────────┘
                    │
                    │ Events
                    ↓
┌─────────────────────────────────────────────────┐
│     Zero-TrustMLCreditsIntegration Layer             │
│  • Economic Policies (rate limits, multipliers) │
│  • Audit Trail                                  │
│  • Statistics                                   │
└───────────────────┬─────────────────────────────┘
                    │
                    │ Credit Issuance
                    ↓
┌─────────────────────────────────────────────────┐
│       HolochainCreditsBridge Layer              │
│  • WebSocket connection to Holochain            │
│  • DNA function calls (create_credit, etc.)     │
│  • Balance queries                              │
└───────────────────┬─────────────────────────────┘
                    │
                    │ P2P DHT
                    ↓
┌─────────────────────────────────────────────────┐
│        Holochain Credits DNA (Rust)             │
│  • Decentralized credit ledger                  │
│  • PoGQ validation                              │
│  • Byzantine-resistant aggregation              │
└─────────────────────────────────────────────────┘
```

---

## Quick Start

### Installation

```bash
# Install dependencies
pip install holochain-client-py  # For Holochain integration
# or use in mock mode (no Holochain required)
```

### Basic Usage

```python
from zerotrustml_credits_integration import (
    Zero-TrustMLCreditsIntegration,
    CreditIssuanceConfig
)
from holochain_credits_bridge import HolochainCreditsBridge

# Initialize (mock mode for development/testing)
bridge = HolochainCreditsBridge(enabled=False)
integration = Zero-TrustMLCreditsIntegration(bridge)

# Issue credits for quality gradient
credit_id = await integration.on_quality_gradient(
    node_id="alice",
    pogq_score=0.95,
    reputation_level="NORMAL",
    verifiers=["bob", "charlie"]
)

# credit_id: Unique identifier for this credit issuance
```

### Production Mode

```python
# Connect to real Holochain conductor
bridge = HolochainCreditsBridge(
    conductor_url="ws://localhost:8888",
    app_id="zerotrustml",
    enabled=True
)

# Connect to Holochain
await bridge.connect()

# Use with Zero-TrustMLCreditsIntegration
integration = Zero-TrustMLCreditsIntegration(bridge)
```

---

## Core Classes

### Zero-TrustMLCreditsIntegration

Main integration layer for the credits system.

```python
class Zero-TrustMLCreditsIntegration:
    """
    Integration between Zero-TrustML and Holochain Credits

    Attributes:
        bridge: HolochainCreditsBridge instance
        config: CreditIssuanceConfig for policies
        rate_limiter: RateLimiter for anti-spam
        issuance_log: List[CreditIssuanceRecord] audit trail
    """
```

#### Constructor

```python
def __init__(
    self,
    credits_bridge: HolochainCreditsBridge,
    config: Optional[CreditIssuanceConfig] = None
)
```

**Parameters:**
- `credits_bridge`: Configured HolochainCreditsBridge instance
- `config`: Optional configuration (uses defaults if None)

**Example:**
```python
bridge = HolochainCreditsBridge(enabled=False)
config = CreditIssuanceConfig(
    max_quality_credits_per_hour=5000,
    min_pogq_score=0.8
)
integration = Zero-TrustMLCreditsIntegration(bridge, config)
```

---

### HolochainCreditsBridge

Bridge to Holochain Credits DNA.

```python
class HolochainCreditsBridge:
    """
    WebSocket connection to Holochain conductor

    Attributes:
        conductor_url: WebSocket URL
        app_id: Holochain app identifier
        zome_name: Zome name for credits
        enabled: Whether to connect to real Holochain
    """
```

#### Constructor

```python
def __init__(
    self,
    conductor_url: str = "ws://localhost:8888",
    app_id: str = "zerotrustml",
    zome_name: str = "zerotrustml_credits",
    enabled: bool = True
)
```

**Parameters:**
- `conductor_url`: WebSocket URL for Holochain conductor
- `app_id`: Installed Holochain app ID
- `zome_name`: Name of credits zome
- `enabled`: Set to False for mock mode

#### Methods

##### `connect() -> bool`

Connect to Holochain conductor.

```python
success = await bridge.connect()
if success:
    print("✓ Connected to Holochain")
else:
    print("✗ Failed to connect")
```

##### `issue_credits(...) -> int`

Low-level credit issuance (called by integration layer).

```python
await bridge.issue_credits(
    node_id=42,
    event_type="quality_gradient",
    pogq_score=0.95,
    verifiers=[1, 2, 3]
)
```

---

## Event Handlers

### 1. Quality Gradient Credits

Reward nodes for submitting high-quality gradients.

```python
async def on_quality_gradient(
    self,
    node_id: str,
    pogq_score: float,
    reputation_level: str,
    verifiers: List[str],
    additional_data: Optional[Dict[str, Any]] = None
) -> Optional[str]
```

**Parameters:**
- `node_id`: Unique node identifier (e.g., "node_123", "alice")
- `pogq_score`: Proof of Quality Gradient score (0.0-1.0)
- `reputation_level`: Node's reputation ("NORMAL", "TRUSTED", "ELITE", etc.)
- `verifiers`: List of node IDs that validated this gradient
- `additional_data`: Optional metadata for audit trail

**Returns:**
- `str`: Credit ID if issued successfully
- `None`: If credits not issued (below threshold, rate limited, or disabled)

**Credit Calculation:**
```python
base_credits = pogq_score * 100  # 0-100 credits
multiplier = reputation_multipliers[reputation_level]  # 0.5-1.5x
final_credits = base_credits * multiplier
```

**Example:**
```python
# ELITE node with excellent gradient
credit_id = await integration.on_quality_gradient(
    node_id="alice",
    pogq_score=0.98,
    reputation_level="ELITE",  # 1.5x multiplier
    verifiers=["bob", "charlie", "david"]
)
# Result: 98 * 1.5 = 147 credits

# NORMAL node with good gradient
credit_id = await integration.on_quality_gradient(
    node_id="bob",
    pogq_score=0.85,
    reputation_level="NORMAL",  # 1.0x multiplier
    verifiers=["alice"]
)
# Result: 85 * 1.0 = 85 credits

# Node below threshold
credit_id = await integration.on_quality_gradient(
    node_id="eve",
    pogq_score=0.65,  # Below 0.7 threshold
    reputation_level="NORMAL",
    verifiers=["alice"]
)
# Result: None (below min_pogq_score)
```

**Rate Limiting:**
- **Hourly Cap**: 10,000 credits by default
- **Prevents**: Spam submissions and exploitation

---

### 2. Byzantine Detection Rewards

Reward nodes for detecting malicious behavior.

```python
async def on_byzantine_detection(
    self,
    detector_node_id: str,
    detected_node_id: str,
    reputation_level: str,
    evidence: Optional[Dict[str, Any]] = None
) -> Optional[str]
```

**Parameters:**
- `detector_node_id`: Node that detected Byzantine behavior
- `detected_node_id`: Node exhibiting Byzantine behavior
- `reputation_level`: Detector's reputation level
- `evidence`: Optional evidence data (for audit and validation)

**Returns:**
- `str`: Credit ID if issued
- `None`: If rate limited or disabled

**Credit Calculation:**
```python
base_credits = 50.0  # Fixed reward
multiplier = reputation_multipliers[reputation_level]
final_credits = base_credits * multiplier
```

**Example:**
```python
# ELITE detector
credit_id = await integration.on_byzantine_detection(
    detector_node_id="alice",
    detected_node_id="mallory",
    reputation_level="ELITE",  # 1.5x multiplier
    evidence={
        "consecutive_failures": 20,
        "pogq_score": 0.15,
        "anomaly_score": 0.95
    }
)
# Result: 50 * 1.5 = 75 credits

# TRUSTED detector
credit_id = await integration.on_byzantine_detection(
    detector_node_id="bob",
    detected_node_id="sybil",
    reputation_level="TRUSTED",  # 1.2x multiplier
    evidence={"attack_type": "coordinated"}
)
# Result: 50 * 1.2 = 60 credits
```

**Rate Limiting:**
- **Daily Cap**: 2,000 credits by default
- **Prevents**: Spam reporting and false accusations

---

### 3. Peer Validation Credits

Reward nodes for validating peers' contributions.

```python
async def on_peer_validation(
    self,
    validator_node_id: str,
    validated_node_id: str,
    reputation_level: str,
    gradient_hash: Optional[str] = None
) -> Optional[str]
```

**Parameters:**
- `validator_node_id`: Node performing validation
- `validated_node_id`: Node being validated
- `reputation_level`: Validator's reputation
- `gradient_hash`: Optional hash of validated gradient

**Credit Calculation:**
```python
base_credits = 10.0  # Small reward for validation work
multiplier = reputation_multipliers[reputation_level]
final_credits = base_credits * multiplier
```

**Example:**
```python
credit_id = await integration.on_peer_validation(
    validator_node_id="alice",
    validated_node_id="bob",
    reputation_level="NORMAL",
    gradient_hash="abc123def456"
)
# Result: 10 * 1.0 = 10 credits
```

**Rate Limiting:**
- **Hourly Cap**: 1,000 credits by default

---

### 4. Network Contribution Credits

Reward nodes for consistent uptime and participation.

```python
async def on_network_contribution(
    self,
    node_id: str,
    uptime_hours: int,
    reputation_level: str
) -> Optional[str]
```

**Parameters:**
- `node_id`: Contributing node
- `uptime_hours`: Hours of continuous uptime
- `reputation_level`: Node's reputation level

**Credit Calculation:**
```python
base_credits = 1.0 per hour (up to 24/day)
multiplier = reputation_multipliers[reputation_level]
final_credits = base_credits * uptime_hours * multiplier
```

**Example:**
```python
# 24 hours uptime
credit_id = await integration.on_network_contribution(
    node_id="alice",
    uptime_hours=24,
    reputation_level="NORMAL"
)
# Result: 1 * 24 * 1.0 = 24 credits

# Partial day
credit_id = await integration.on_network_contribution(
    node_id="bob",
    uptime_hours=12,
    reputation_level="TRUSTED"
)
# Result: 1 * 12 * 1.2 = 14.4 credits
```

**Rate Limiting:**
- **Daily Cap**: 24 credits by default (1 per hour max)

---

## Configuration

### CreditIssuanceConfig

Configure economic policies and limits.

```python
@dataclass
class CreditIssuanceConfig:
    # Enable/disable system
    enabled: bool = True

    # Rate limits (anti-spam)
    max_quality_credits_per_hour: int = 10000
    max_byzantine_credits_per_day: int = 2000
    max_validation_credits_per_hour: int = 1000
    max_network_credits_per_day: int = 24

    # Quality thresholds
    min_pogq_score: float = 0.7
    min_uptime_percentage: float = 0.95

    # Reputation multipliers
    reputation_multipliers: Dict[str, float] = {
        "BLACKLISTED": 0.0,   # No credits
        "CRITICAL": 0.5,       # Half credits
        "WARNING": 0.75,       # 75% credits
        "NORMAL": 1.0,         # Standard rate
        "TRUSTED": 1.2,        # 20% bonus
        "ELITE": 1.5           # 50% bonus
    }
```

**Custom Configuration Example:**
```python
config = CreditIssuanceConfig(
    enabled=True,
    max_quality_credits_per_hour=5000,  # Stricter limit
    min_pogq_score=0.8,                  # Higher quality bar
    reputation_multipliers={
        "BLACKLISTED": 0.0,
        "NORMAL": 1.0,
        "TRUSTED": 1.3,   # Custom multiplier
        "ELITE": 2.0      # Higher elite bonus
    }
)

integration = Zero-TrustMLCreditsIntegration(bridge, config)
```

---

### ReputationLevel Enum

Pre-defined reputation levels with multipliers.

```python
class ReputationLevel(Enum):
    BLACKLISTED = ("BLACKLISTED", 0.0)   # Banned
    CRITICAL = ("CRITICAL", 0.5)          # Under review
    WARNING = ("WARNING", 0.75)           # Degraded service
    NORMAL = ("NORMAL", 1.0)              # Full service
    TRUSTED = ("TRUSTED", 1.2)            # Enhanced privileges
    ELITE = ("ELITE", 1.5)                # Top contributors
```

**Usage:**
```python
level = ReputationLevel.ELITE
print(level.label)      # "ELITE"
print(level.multiplier) # 1.5
```

---

### CreditEventType Enum

Types of credit-earning events.

```python
class CreditEventType(Enum):
    QUALITY_GRADIENT = "quality_gradient"
    BYZANTINE_DETECTION = "byzantine_detection"
    PEER_VALIDATION = "peer_validation"
    NETWORK_CONTRIBUTION = "network_contribution"
```

---

## Rate Limiting

### RateLimiter

Sliding-window rate limiter for anti-spam.

```python
class RateLimiter:
    """
    Track credit issuances and enforce limits

    Uses sliding window algorithm for accurate rate limiting
    """
```

#### Methods

##### `check_limit(...) -> tuple[bool, str]`

Check if issuing credits would exceed rate limit.

```python
allowed, reason = rate_limiter.check_limit(
    node_id="alice",
    event_type=CreditEventType.QUALITY_GRADIENT,
    credits=100.0,
    hourly_limit=10000
)

if allowed:
    # Proceed with issuance
    pass
else:
    print(f"Rate limited: {reason}")
```

**Returns:**
- `(True, "OK")`: Credits allowed
- `(False, "Hourly limit exceeded (9500/10000)")`: Rate limited

##### `get_stats(...) -> Dict[str, float]`

Get credit statistics for a node.

```python
stats = rate_limiter.get_stats(
    node_id="alice",
    event_type=CreditEventType.QUALITY_GRADIENT
)
# {
#     "hourly": 1500.0,
#     "daily": 8000.0,
#     "total": 50000.0,
#     "count": 127
# }
```

---

## Usage Examples

### Example 1: Federated Learning Workflow

```python
import asyncio
from zerotrustml_credits_integration import Zero-TrustMLCreditsIntegration
from holochain_credits_bridge import HolochainCreditsBridge

async def federated_learning_round():
    # Initialize
    bridge = HolochainCreditsBridge(enabled=True)
    await bridge.connect()
    integration = Zero-TrustMLCreditsIntegration(bridge)

    # Training round
    nodes = ["alice", "bob", "charlie", "david"]

    for node_id in nodes:
        # Node computes gradient
        gradient = compute_gradient(node_id)

        # Validate gradient quality
        pogq_score = validate_quality(gradient)

        # Get node reputation
        reputation = get_reputation(node_id)

        # Issue credits
        credit_id = await integration.on_quality_gradient(
            node_id=node_id,
            pogq_score=pogq_score,
            reputation_level=reputation,
            verifiers=[v for v in nodes if v != node_id]
        )

        if credit_id:
            print(f"✓ {node_id}: {pogq_score:.2f} → {credit_id}")
        else:
            print(f"✗ {node_id}: Below threshold or rate limited")

asyncio.run(federated_learning_round())
```

### Example 2: Byzantine Detection Pipeline

```python
async def byzantine_detection_workflow():
    integration = Zero-TrustMLCreditsIntegration(bridge)

    # Detect Byzantine behavior
    byzantine_nodes = detect_byzantine_nodes(network)

    for detector, malicious_node in byzantine_nodes:
        # Get detector reputation
        reputation = get_reputation(detector)

        # Gather evidence
        evidence = {
            "consecutive_failures": 15,
            "anomaly_score": 0.92,
            "attack_type": "coordinated"
        }

        # Issue reward
        credit_id = await integration.on_byzantine_detection(
            detector_node_id=detector,
            detected_node_id=malicious_node,
            reputation_level=reputation,
            evidence=evidence
        )

        if credit_id:
            print(f"✓ {detector} caught {malicious_node}: {credit_id}")

asyncio.run(byzantine_detection_workflow())
```

### Example 3: Audit Trail Analysis

```python
def analyze_credit_issuances(integration):
    """Analyze credit issuance patterns"""

    # Get all issuances
    for record in integration.issuance_log:
        print(f"{record.timestamp}: {record.node_id}")
        print(f"  Event: {record.event_type.value}")
        print(f"  Credits: {record.base_credits} × {record.multiplier} = {record.final_credits}")
        if record.error:
            print(f"  Error: {record.error}")
        print()

    # Group by event type
    by_event = {}
    for record in integration.issuance_log:
        event = record.event_type.value
        if event not in by_event:
            by_event[event] = []
        by_event[event].append(record.final_credits)

    # Statistics
    for event, credits in by_event.items():
        print(f"{event}:")
        print(f"  Count: {len(credits)}")
        print(f"  Total: {sum(credits):.0f}")
        print(f"  Average: {sum(credits)/len(credits):.1f}")
```

---

## Best Practices

### 1. Use Reputation Multipliers

Always include accurate reputation levels:

```python
# ❌ BAD: Hardcoded reputation
await integration.on_quality_gradient(
    node_id="alice",
    pogq_score=0.95,
    reputation_level="NORMAL",  # Always same
    verifiers=["bob"]
)

# ✅ GOOD: Dynamic reputation from system
reputation = get_node_reputation("alice")
await integration.on_quality_gradient(
    node_id="alice",
    pogq_score=0.95,
    reputation_level=reputation,
    verifiers=["bob"]
)
```

### 2. Provide Evidence for Byzantine Detection

Include detailed evidence for audit trail:

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
        "consecutive_failures": 20,
        "pogq_score": 0.12,
        "anomaly_score": 0.98,
        "attack_type": "sign_flip",
        "detection_method": "PoGQ+statistical"
    }
)
```

### 3. Handle Rate Limiting Gracefully

Check rate limits before expensive operations:

```python
# Check if node is near limit
stats = integration.rate_limiter.get_stats(
    "alice",
    CreditEventType.QUALITY_GRADIENT
)

if stats["hourly"] > 9000:  # Near 10k limit
    print(f"⚠️  alice near hourly limit ({stats['hourly']}/10000)")
    # Maybe skip or throttle
```

### 4. Monitor Audit Logs

Regularly review issuance patterns:

```python
# Daily audit
def daily_audit(integration):
    recent = [r for r in integration.issuance_log
              if r.timestamp > datetime.now() - timedelta(days=1)]

    total_issued = sum(r.final_credits for r in recent if not r.error)
    total_rejected = len([r for r in recent if r.error])

    print(f"Last 24h:")
    print(f"  Issued: {total_issued:.0f} credits")
    print(f"  Rejected: {total_rejected} attempts")
```

### 5. Use Mock Mode for Testing

Always test with mock mode first:

```python
# Development/Testing
bridge = HolochainCreditsBridge(enabled=False)  # Mock
integration = Zero-TrustMLCreditsIntegration(bridge)

# Production
bridge = HolochainCreditsBridge(
    conductor_url="ws://production-conductor:8888",
    enabled=True
)
await bridge.connect()
integration = Zero-TrustMLCreditsIntegration(bridge)
```

---

## Troubleshooting

### Issue: No credits issued despite high PoGQ

**Symptoms:**
```python
credit_id = await integration.on_quality_gradient(...)
# credit_id is None
```

**Possible Causes:**
1. **Below threshold**: PoGQ < 0.7 (default `min_pogq_score`)
2. **Rate limited**: Exceeded hourly cap
3. **Zero multiplier**: Reputation = "BLACKLISTED"
4. **System disabled**: `config.enabled = False`

**Solution:**
```python
# Check configuration
print(f"Enabled: {integration.config.enabled}")
print(f"Min PoGQ: {integration.config.min_pogq_score}")
print(f"Reputation multiplier: {integration.config.reputation_multipliers[reputation]}")

# Check rate limit
stats = integration.rate_limiter.get_stats(node_id, CreditEventType.QUALITY_GRADIENT)
print(f"Hourly usage: {stats['hourly']}/{integration.config.max_quality_credits_per_hour}")

# Review audit log
for record in integration.issuance_log:
    if record.node_id == node_id and record.error:
        print(f"Error: {record.error}")
```

### Issue: Holochain connection fails

**Symptoms:**
```python
success = await bridge.connect()
# success is False
```

**Possible Causes:**
1. **Conductor not running**: `hc sandbox run` not active
2. **Wrong URL**: Conductor on different port
3. **App not installed**: Zero-TrustML app not installed in conductor
4. **Missing client library**: `holochain-client-py` not installed

**Solution:**
```bash
# Check conductor status
hc sandbox call --running
# If not running:
hc sandbox run

# Verify app installed
hc app list
# Should show "zerotrustml" app

# Check connection manually
wscat -c ws://localhost:8888
```

### Issue: Rate limit too restrictive

**Symptoms:**
```python
# Many "rate limit exceeded" messages
```

**Solution:**

```python
# Adjust limits in configuration
config = CreditIssuanceConfig(
    max_quality_credits_per_hour=20000,  # Doubled
    max_byzantine_credits_per_day=5000,  # Increased
)
integration = Zero-TrustMLCreditsIntegration(bridge, config)
```

---

## API Reference Summary

### Zero-TrustMLCreditsIntegration

| Method | Purpose | Returns |
|--------|---------|---------|
| `on_quality_gradient()` | Issue credits for quality gradients | `Optional[str]` |
| `on_byzantine_detection()` | Reward Byzantine detection | `Optional[str]` |
| `on_peer_validation()` | Reward peer validation | `Optional[str]` |
| `on_network_contribution()` | Reward network uptime | `Optional[str]` |
| `get_stats()` | Get statistics for node | `Dict` |
| `export_audit_trail()` | Export audit log | `List[CreditIssuanceRecord]` |

### HolochainCreditsBridge

| Method | Purpose | Returns |
|--------|---------|---------|
| `connect()` | Connect to Holochain conductor | `bool` |
| `issue_credits()` | Low-level credit issuance | `int` |
| `get_balance()` | Query node's credit balance | `int` |
| `get_credit_history()` | Get issuance history | `List` |

### Configuration Classes

| Class | Purpose |
|-------|---------|
| `CreditIssuanceConfig` | Economic policies and limits |
| `ReputationLevel` | Reputation tiers with multipliers |
| `CreditEventType` | Types of credit events |
| `CreditIssuanceRecord` | Audit trail record |

---

## Credits Economics Summary

| Event Type | Base Credits | Rate Limit | Typical Use |
|------------|--------------|------------|-------------|
| **Quality Gradient** | PoGQ × 100 (0-100) | 10k/hour | Every training round |
| **Byzantine Detection** | 50 (fixed) | 2k/day | When malicious node found |
| **Peer Validation** | 10 (fixed) | 1k/hour | Every validation |
| **Network Contribution** | 1/hour (up to 24/day) | 24/day | Daily uptime |

**Reputation Multipliers:**
- BLACKLISTED: 0.0× (no credits)
- CRITICAL: 0.5× (reduced)
- WARNING: 0.75× (reduced)
- NORMAL: 1.0× (standard)
- TRUSTED: 1.2× (+20%)
- ELITE: 1.5× (+50%)

---

*For additional support or questions, contact the Zero-TrustML development team.*

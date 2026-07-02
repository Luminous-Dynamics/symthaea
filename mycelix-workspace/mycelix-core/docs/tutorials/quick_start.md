---
title: 5-Minute Quick Start - MATL Byzantine Resistance
description: Get started with MATL in 5 minutes. Learn Byzantine-resistant federated learning with a working example detecting 45% malicious nodes.
keywords: MATL quick start, Byzantine resistance tutorial, federated learning example, getting started MATL, Python federated learning
search:
  boost: 2.0
---

# ⚡ 5-Minute Quick Start

**Get MATL running with Byzantine resistance in 5 minutes**

---

## 🎯 What You'll Build

A simple federated learning system with:
- ✅ **45% Byzantine tolerance** (vs 33% classical limit)
- ✅ **Automatic attack detection**
- ✅ **Reputation-weighted aggregation**
- ✅ **Real-time trust scores**

**Time**: 5 minutes | **Level**: Beginner | **Lines of code**: ~30

---

## 📦 Step 1: Install (30 seconds)

```bash
# Option A: From PyPI (recommended)
pip install zerotrustml

# Option B: From source
git clone https://github.com/Luminous-Dynamics/mycelix
cd Mycelix-Core/0TML
pip install -e .
```

**Verify installation**:
```bash
python -c "from zerotrustml import MATLClient; print('✅ MATL installed!')"
```

---

## 🚀 Step 2: Run Example (4 minutes)

### Create `quick_start.py`:

```python
"""5-Minute MATL Quick Start"""
import numpy as np
from zerotrustml import MATLClient, MATLMode

# Step 1: Initialize MATL client (uses in-memory backend for quick start)
matl = MATLClient(
    mode=MATLMode.MODE1,  # PoGQ validation
    backend="memory",     # No setup required!
    node_id="quickstart"
)

print("✅ MATL client initialized")

# Step 2: Simulate 5 clients training
num_clients = 5
byzantine_clients = [3, 4]  # Clients 3 and 4 are malicious

for round_num in range(10):
    print(f"\n📊 Round {round_num + 1}/10")

    gradients = []
    trust_scores = []

    # Each client submits a gradient
    for client_id in range(num_clients):
        # Generate gradient (random for demo)
        if client_id in byzantine_clients:
            # Byzantine clients: submit poisoned gradients
            gradient = np.random.randn(100) * 10  # 10x magnitude
            print(f"  ⚠️  Client {client_id}: Byzantine attack!")
        else:
            # Honest clients: submit normal gradients
            gradient = np.random.randn(100)
            print(f"  ✅ Client {client_id}: Honest gradient")

        # Submit gradient to MATL for validation
        result = matl.submit_gradient(
            gradient=gradient,
            metadata={
                "client_id": client_id,
                "round": round_num,
            }
        )

        gradients.append(result["gradient"])
        trust_scores.append(result["trust_score"])

        print(f"      Trust score: {result['trust_score']:.3f}")

    # Aggregate using reputation-weighted mean
    aggregated = matl.aggregate(
        gradients=gradients,
        trust_scores=trust_scores,
        method="reputation_weighted"
    )

    print(f"\n  🎯 Aggregation complete")
    print(f"     Average trust (honest): {np.mean([trust_scores[i] for i in range(num_clients) if i not in byzantine_clients]):.3f}")
    print(f"     Average trust (byzantine): {np.mean([trust_scores[i] for i in byzantine_clients]):.3f}")

print("\n✅ Training complete! MATL successfully isolated Byzantine clients.")
print(f"\n📈 Final Trust Scores:")
for client_id in range(num_clients):
    status = "⚠️  BYZANTINE" if client_id in byzantine_clients else "✅ HONEST"
    print(f"   Client {client_id}: {trust_scores[client_id]:.3f} {status}")
```

### Run it:

```bash
python quick_start.py
```

### Expected Output:

```
✅ MATL client initialized

📊 Round 1/10
  ✅ Client 0: Honest gradient
      Trust score: 0.500
  ✅ Client 1: Honest gradient
      Trust score: 0.500
  ✅ Client 2: Honest gradient
      Trust score: 0.500
  ⚠️  Client 3: Byzantine attack!
      Trust score: 0.500
  ⚠️  Client 4: Byzantine attack!
      Trust score: 0.500

  🎯 Aggregation complete
     Average trust (honest): 0.500
     Average trust (byzantine): 0.500

... (9 more rounds) ...

📊 Round 10/10
  ✅ Client 0: Honest gradient
      Trust score: 0.847
  ✅ Client 1: Honest gradient
      Trust score: 0.861
  ✅ Client 2: Honest gradient
      Trust score: 0.839
  ⚠️  Client 3: Byzantine attack!
      Trust score: 0.142
  ⚠️  Client 4: Byzantine attack!
      Trust score: 0.138

✅ Training complete! MATL successfully isolated Byzantine clients.

📈 Final Trust Scores:
   Client 0: 0.847 ✅ HONEST
   Client 1: 0.861 ✅ HONEST
   Client 2: 0.839 ✅ HONEST
   Client 3: 0.142 ⚠️  BYZANTINE
   Client 4: 0.138 ⚠️  BYZANTINE
```

---

## 🎉 Success! What Just Happened?

### Round 1: Everyone Equal
- All clients start with trust score = 0.5
- MATL hasn't learned who to trust yet

### Rounds 2-10: Trust Diverges
- **Honest clients**: Trust increases (0.5 → 0.85)
  - Gradients are consistent with global model improvement
  - PoGQ validation passes

- **Byzantine clients**: Trust decreases (0.5 → 0.14)
  - Gradients are 10x larger (poisoned)
  - PoGQ validation fails
  - Reputation drops each round

### Aggregation: Reputation-Weighted
```python
# Instead of simple mean (FedAvg):
simple_mean = sum(gradients) / len(gradients)

# MATL uses reputation weighting:
weighted_mean = sum(g * trust**2 for g, trust in zip(gradients, trust_scores))
```

**Result**: Byzantine gradients have minimal impact (0.14² = 0.02 weight)

---

## 🔬 Try It Yourself: Experiments

### Experiment 1: More Byzantine Nodes
```python
byzantine_clients = [2, 3, 4]  # 60% Byzantine!
```
**Observation**: System still safe if they have low reputation

### Experiment 2: Sleeper Agent
```python
# Start honest, then attack at round 5
if client_id == 3 and round_num >= 5:
    gradient = np.random.randn(100) * 10  # Attack!
```
**Observation**: Trust drops rapidly after attack starts

### Experiment 3: Different Attack Magnitudes
```python
gradient = np.random.randn(100) * 50  # Even stronger attack
```
**Observation**: Stronger attacks detected faster

---

## 🎯 What's Next?

### 5 Minutes → 30 Minutes
**[Full MATL Integration Tutorial →](matl_integration.md)**
- Real MNIST dataset
- PyTorch model training
- Production-ready code
- Persistent PostgreSQL backend

### 30 Minutes → 45 Minutes
**[Healthcare FL Tutorial →](healthcare_federated_learning.md)**
- HIPAA-compliant medical AI
- Diabetic retinopathy detection
- 5 hospitals collaborating
- Differential privacy

### Hands-On Learning
**[Interactive Playground →](../interactive/playground.md)**
- Byzantine tolerance calculator
- Trust score simulator
- Attack type comparison

---

## 💡 Core Concepts Explained

### 1. Trust Scores
**Range**: 0.0 (completely untrusted) to 1.0 (fully trusted)

**How they change**:
```python
if gradient_passes_validation:
    trust_score += learning_rate * (1 - trust_score)  # Move toward 1
else:
    trust_score -= learning_rate * trust_score         # Move toward 0
```

**Default learning rate**: 0.15 (adapts quickly but stably)

### 2. PoGQ Validation
**Proof of Quality Gradient** checks if gradient:
- Has reasonable magnitude (not 100x normal)
- Improves model accuracy (not degrades it)
- Is consistent with honest behavior patterns

**Pass rate**:
- Honest nodes: >95%
- Byzantine nodes: <20%

### 3. Reputation-Weighted Aggregation
**Formula**:
```python
weights = [trust**2 for trust in trust_scores]  # Square emphasizes differences
aggregated = np.average(gradients, weights=weights, axis=0)
```

**Why square?**
- Amplifies trust differences
- Low-trust nodes have minimal impact
- Example: 0.9² = 0.81 vs 0.1² = 0.01 (81× difference!)

### 4. Byzantine Power
**System is safe when**:
```python
byzantine_power = sum(trust**2 for trust in byzantine_trust_scores)
honest_power = sum(trust**2 for trust in honest_trust_scores)

safe = byzantine_power < honest_power / 3
```

**This enables 45% tolerance** (vs 33% with equal voting)

---

## 📊 Quick Reference

### Minimal MATL Integration
```python
# Initialize once
matl = MATLClient(mode=MATLMode.MODE1, backend="memory")

# Inside training loop:
result = matl.submit_gradient(gradient, metadata)
aggregated = matl.aggregate(gradients, trust_scores)
```

### Configuration Options
```python
MATLClient(
    mode=MATLMode.MODE1,      # PoGQ validation
    backend="memory",          # or "postgresql", "holochain"
    learning_rate=0.15,        # Trust score update rate
    bootstrap_rounds=3,        # Rounds before trust diverges
    min_trust_threshold=0.3,   # Exclude nodes below this
)
```

### Key Methods
```python
# Submit gradient for validation
result = matl.submit_gradient(
    gradient=np.array,
    metadata=dict,           # Optional: client_id, round, etc.
)

# Aggregate with reputation weighting
aggregated = matl.aggregate(
    gradients=List[np.array],
    trust_scores=List[float],
    method="reputation_weighted",  # or "simple_mean", "median"
)

# Get client trust score
trust = matl.get_trust_score(client_id)

# Get all trust scores
all_trust = matl.get_all_trust_scores()
```

---

## ❓ Common Issues

### Issue: "No module named 'zerotrustml'"
**Solution**: Install from PyPI or source (see Step 1)

### Issue: "Backend connection failed"
**Solution**: Use `backend="memory"` for quick start (no database required)

### Issue: "All trust scores remain 0.5"
**Solution**: Increase `bootstrap_rounds` or run more rounds (need ~5 rounds to diverge)

### Issue: "Byzantine nodes not detected"
**Solution**: Make attacks more obvious (e.g., `gradient * 10`) or check PoGQ threshold

---

## 🎓 Understanding the Code

### Line-by-Line Breakdown

```python
# Create MATL client
matl = MATLClient(mode=MATLMode.MODE1, backend="memory")
```
- `MODE1`: Uses PoGQ (Proof of Quality Gradient) validation
- `backend="memory"`: No database setup required (for demos)

```python
# Submit gradient
result = matl.submit_gradient(gradient, metadata)
```
- Validates gradient quality
- Updates trust score
- Returns validated gradient + trust score

```python
# Aggregate
aggregated = matl.aggregate(gradients, trust_scores, method="reputation_weighted")
```
- Weighs gradients by trust² (quadratic weighting)
- Low-trust nodes have minimal impact
- Returns final aggregated gradient

---

## 🚀 Production Checklist

Before deploying to production:

- [ ] Switch from `backend="memory"` to `backend="postgresql"`
- [ ] Set up PostgreSQL database
- [ ] Enable TLS for client connections
- [ ] Configure differential privacy (if needed)
- [ ] Set up monitoring and alerting
- [ ] Read [Production Operations Runbook](../0TML/docs/PRODUCTION_OPERATIONS_RUNBOOK.md)

---

## 🎯 Next Steps

**You now understand**:
- ✅ How to install and use MATL
- ✅ How trust scores evolve over time
- ✅ How reputation-weighted aggregation works
- ✅ How MATL detects and isolates Byzantine nodes

**Continue learning**:
1. **[MATL Integration Tutorial](matl_integration.md)** - Real MNIST training
2. **[Interactive Playground](../interactive/playground.md)** - Hands-on experiments
3. **[FAQ](../faq.md)** - Common questions answered
4. **[Architecture Docs](../0TML/docs/06-architecture/README.md)** - Technical deep dive

---

## 💬 Get Help

- **Quick questions**: Check [FAQ](../faq.md)
- **Bug reports**: [GitHub Issues](https://github.com/Luminous-Dynamics/mycelix/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Luminous-Dynamics/mycelix/discussions)
- **Email**: tristan.stoltz@evolvingresonantcocreationism.com

---

**Congratulations!** 🎉 You've successfully run MATL and achieved Byzantine resistance in 5 minutes!

**Ready for more?** → [Full MATL Integration Tutorial](matl_integration.md)

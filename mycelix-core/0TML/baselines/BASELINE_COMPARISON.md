# Federated Learning Baseline Comparison

Quick reference guide for all 7 implemented baselines.

---

## 📊 At-a-Glance Comparison

| Baseline | Paper | Year | Non-IID | Byzantine | Complexity | Use Case |
|----------|-------|------|---------|-----------|------------|----------|
| **FedAvg** | McMahan et al. | 2017 | ❌ Poor | ❌ No | ⭐ Simple | IID data, honest clients |
| **FedProx** | Li et al. | 2020 | ✅ Good | ❌ No | ⭐⭐ Medium | Non-IID data, system heterogeneity |
| **SCAFFOLD** | Karimireddy et al. | 2020 | ✅ Best | ❌ No | ⭐⭐⭐ Complex | Non-IID data, fast convergence |
| **Krum** | Blanchard et al. | 2017 | ⚠️ OK | ✅ f<n/2 | ⭐⭐ Medium | Byzantine attacks |
| **Multi-Krum** | Blanchard et al. | 2017 | ⚠️ OK | ✅ f<n/2 | ⭐⭐ Medium | Byzantine attacks, smoother |
| **Bulyan** | El Mhamdi et al. | 2018 | ⚠️ OK | ✅ f<n/3 | ⭐⭐⭐ Complex | Strong Byzantine defense |
| **Median** | Yin et al. | 2018 | ⚠️ OK | ✅ f<n/2 | ⭐ Simple | Byzantine attacks, simple |

---

## 🎯 When to Use Each Baseline

### FedAvg (Federated Averaging)
**Use when:**
- Data is IID (independently and identically distributed)
- All clients are honest (no Byzantine attacks)
- You need fast convergence
- You want the simplest baseline

**Don't use when:**
- Data is non-IID
- Byzantine clients present
- Need provable guarantees

**Key Formula:** `w_global = Σ (n_k / n_total) * w_k`

---

### FedProx (Federated Proximal)
**Use when:**
- Data is non-IID across clients
- Clients have different compute capabilities (system heterogeneity)
- Some clients drop out during training
- You want stable convergence

**Don't use when:**
- Byzantine attacks are present
- Data is perfectly IID (FedAvg is faster)

**Key Formula:** `min F_k(w) + (μ/2) ||w - w_global||²`

**Hyperparameter:** `μ = 0.01` (default) - higher = more regularization

---

### SCAFFOLD (Stochastic Controlled Averaging)
**Use when:**
- Data is highly non-IID
- You need 3-4x faster convergence than FedAvg
- You can afford the extra memory (control variates)
- Partial client participation

**Don't use when:**
- Memory constrained (needs control variates per client)
- Byzantine attacks present
- Data is IID (overhead not worth it)

**Key Formula:** `w_{t+1} = w_t - η(∇F(w) - c_i + c)`

**Advantage:** Provable variance reduction via control variates

---

### Krum (Byzantine-Robust Selection)
**Use when:**
- Byzantine clients present (up to f < n/2)
- You need provable Byzantine tolerance
- Single "representative" gradient is acceptable

**Don't use when:**
- No Byzantine threat (FedAvg is better)
- More than n/2 Byzantine clients
- You need smoother updates (use Multi-Krum)

**Key Formula:** `score(i) = Σ_{j∈N(i,k)} ||g_i - g_j||²` → select argmin

**Byzantine Tolerance:** f < n/2 - 1

---

### Multi-Krum (Byzantine-Robust Averaging)
**Use when:**
- Byzantine clients present (up to f < n/2)
- You want smoother convergence than single Krum
- You can tolerate slight compute overhead

**Don't use when:**
- No Byzantine threat (FedAvg is better)
- Maximum Byzantine tolerance needed (use Bulyan)

**Key Formula:** Average top-m gradients with lowest Krum scores

**Byzantine Tolerance:** f < n/2 - 1

**Hyperparameter:** `m = n - f - 2` (default) - number to average

---

### Bulyan (Strongest Byzantine Defense)
**Use when:**
- Strong Byzantine attacks (up to f < n/3)
- You need provably strongest defense
- Coordinate-level attacks possible
- Can afford two-phase aggregation

**Don't use when:**
- No Byzantine threat (expensive overhead)
- Need fast aggregation (two phases slow)
- f ≥ n/3 Byzantine clients

**Key Algorithm:**
1. Phase 1: Multi-Krum selection (θ = n - 2f gradients)
2. Phase 2: Coordinate-wise trimmed mean (β = f/2)

**Byzantine Tolerance:** f < n/3 (strongest!)

---

### Median (Simple Byzantine Defense)
**Use when:**
- Byzantine clients present (up to f < n/2)
- You want simplest possible defense
- No hyperparameters to tune
- Fast computation needed (O(n log n) per coordinate)

**Don't use when:**
- No Byzantine threat (FedAvg converges faster)
- Need coordinate correlations (median ignores them)

**Key Formula:** `w_global[i] = median(w_1[i], ..., w_n[i])` for each coordinate i

**Byzantine Tolerance:** f < n/2

**Advantage:** Parameter-free, simple, fast

---

## 🔬 Experimental Scenarios

### Scenario 1: IID Data, Honest Clients
**Recommended Order:**
1. **FedAvg** - fastest convergence
2. FedProx - nearly as fast
3. SCAFFOLD - slight overhead
4. Krum/Multi-Krum/Bulyan/Median - unnecessary overhead

**Expected Results:** All should reach similar accuracy, FedAvg fastest

---

### Scenario 2: Non-IID Data (Dirichlet α=0.1), Honest Clients
**Recommended Order:**
1. **SCAFFOLD** - best performance (3-4x faster)
2. **FedProx** - good performance
3. FedAvg - struggles with non-IID
4. Krum/Multi-Krum/Bulyan/Median - defense overhead, not needed

**Expected Results:** SCAFFOLD > FedProx > FedAvg by 5-15% accuracy

---

### Scenario 3: IID Data, Byzantine Clients (f=2, n=10)
**Recommended Order:**
1. **Bulyan** - strongest defense (f < n/3)
2. **Multi-Krum** - good defense, smoother
3. **Median** - simple defense
4. Krum - works but less smooth
5. FedAvg/FedProx/SCAFFOLD - no defense, will fail

**Expected Results:** Byzantine-robust methods maintain accuracy, others collapse

---

### Scenario 4: Non-IID Data, Byzantine Clients
**Recommended:**
- **Bulyan** if f < n/3 and strong defense needed
- **Multi-Krum** if f < n/2 and prefer smoother convergence
- **Median** if want simplest solution

**Challenge:** Combining non-IID handling with Byzantine robustness is active research

---

## 📈 Performance Characteristics

### Convergence Speed (Honest Setting)
```
SCAFFOLD     ████████████████████ (fastest on non-IID)
FedAvg       ██████████████████   (fastest on IID)
FedProx      ████████████████     (good on non-IID)
Multi-Krum   ██████████           (defense overhead)
Krum         ████████             (selection overhead)
Bulyan       ██████               (two-phase overhead)
Median       ████████             (converges slower)
```

### Byzantine Robustness
```
Bulyan       ████████████████████ (f < n/3, strongest)
Multi-Krum   ████████████████     (f < n/2, smooth)
Krum         ██████████████       (f < n/2, single)
Median       ████████████         (f < n/2, simple)
FedProx      ░░░░░░░░             (no defense)
SCAFFOLD     ░░░░░░░░             (no defense)
FedAvg       ░░░░░░░░             (no defense)
```

### Computational Overhead
```
FedAvg       █                    (O(1) per client)
FedProx      ██                   (proximal term)
Median       ███                  (O(n log n))
Krum         ████                 (O(n²) distances)
SCAFFOLD     ████                 (control variates)
Multi-Krum   █████                (O(n²) + selection)
Bulyan       ██████               (two-phase)
```

---

## 🛠️ Code Usage Examples

### Basic FedAvg
```python
from baselines.fedavg import create_fedavg_experiment, FedAvgConfig

config = FedAvgConfig(
    learning_rate=0.01,
    local_epochs=1,
    num_clients=10
)

server, clients, test_data = create_fedavg_experiment(
    model_fn=lambda: SimpleCNN(),
    train_data_splits=train_splits,
    test_data=test_data,
    config=config
)

for round in range(100):
    stats = server.train_round(clients)
    if round % 10 == 0:
        test_stats = evaluate_global_model(server.model, test_data)
        print(f"Round {round}: Test Acc = {test_stats['test_accuracy']:.3f}")
```

### FedProx for Non-IID
```python
from baselines.fedprox import create_fedprox_experiment, FedProxConfig

config = FedProxConfig(
    learning_rate=0.01,
    local_epochs=5,  # More epochs for non-IID
    num_clients=10,
    mu=0.01  # Proximal term coefficient
)

server, clients, test_data = create_fedprox_experiment(
    model_fn=lambda: SimpleCNN(),
    train_data_splits=non_iid_splits,  # Non-IID data
    config=config
)

# Training loop same as FedAvg
```

### Byzantine Defense with Bulyan
```python
from baselines.bulyan import create_bulyan_experiment, BulyanConfig

config = BulyanConfig(
    learning_rate=0.01,
    local_epochs=1,
    num_clients=10,
    num_byzantine=2  # Tolerate 2 Byzantine (f < n/3)
)

server, clients, test_data = create_bulyan_experiment(
    model_fn=lambda: SimpleCNN(),
    train_data_splits=train_splits,
    byzantine_clients=[5, 7],  # Make clients 5 and 7 Byzantine
    config=config
)

# Training loop same, but now robust to attacks
```

---

## 📚 Paper References

1. **FedAvg**: [McMahan et al., AISTATS 2017](https://arxiv.org/abs/1602.05629)
2. **FedProx**: [Li et al., MLSys 2020](https://arxiv.org/abs/1812.06127)
3. **SCAFFOLD**: [Karimireddy et al., ICML 2020](https://arxiv.org/abs/1910.06378)
4. **Krum**: [Blanchard et al., NeurIPS 2017](https://arxiv.org/abs/1703.02757)
5. **Multi-Krum**: [Blanchard et al., NeurIPS 2017](https://arxiv.org/abs/1703.02757)
6. **Bulyan**: [El Mhamdi et al., ICML 2018](https://arxiv.org/abs/1802.07927)
7. **Median**: [Yin et al., ICML 2018](https://arxiv.org/abs/1803.01498)

---

## 🎯 Quick Decision Matrix

**Question 1: Are there Byzantine (malicious) clients?**
- **No** → Go to Question 2
- **Yes** → Go to Question 3

**Question 2: Is the data non-IID (heterogeneous across clients)?**
- **No (IID)** → Use **FedAvg** (simplest, fastest)
- **Yes (non-IID)** → Go to Question 4

**Question 3: How many Byzantine clients? (f out of n clients)**
- **f < n/3** → Use **Bulyan** (strongest defense)
- **f < n/2** → Use **Multi-Krum** (good defense, smoother than Krum)
- **Want simplest** → Use **Median** (parameter-free)

**Question 4: How non-IID is the data?**
- **Mildly non-IID** → Use **FedProx** (simple, effective)
- **Highly non-IID** → Use **SCAFFOLD** (3-4x faster convergence)
- **System heterogeneity** (different devices) → Use **FedProx**

---

## 💡 Pro Tips

1. **Start with FedAvg**: Always establish FedAvg baseline first to understand the problem
2. **Measure non-IID**: Use Dirichlet α parameter (α→0 is more non-IID)
3. **Test defenses**: Even if no Byzantine expected, test robustness anyway
4. **Combine methods**: Some research combines FedProx with Krum (not implemented here)
5. **Tune hyperparameters**: Default values work but tuning can improve significantly

---

**Created**: October 3, 2025
**Status**: All 7 baselines implemented and ready for experiments

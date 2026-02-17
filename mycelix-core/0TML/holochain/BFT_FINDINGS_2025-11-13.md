# Byzantine Fault Tolerance Findings - November 13, 2025

## Experimental Results: BFT Limit Sweep

**Experiment**: Systematic testing of AEGIS vs Median aggregation across Byzantine ratios
**Dataset**: EMNIST (6000 train, 1000 test, 10 classes)
**Setup**: 50 clients, Dirichlet α=1.0 (non-IID data distribution)
**Attack**: Label flipping attack

---

## 🎯 Key Findings

### AEGIS Performance Across Byzantine Ratios

| Byzantine % | n_Byzantine | AEGIS Accuracy | Median Accuracy | Status |
|-------------|-------------|----------------|-----------------|--------|
| 10% | 5/50 | 87.8% | 87.6% | ✅ Both work |
| 15% | 7/50 | 87.3% | 87.5% | ✅ Both work |
| 20% | 10/50 | 87.8% | 87.5% | ✅ Both work |
| 25% | 12/50 | 87.6% | 87.5% | ✅ Both work |
| 30% | 15/50 | 87.4% | 86.4% | ✅ Both work |
| 35% | 17/50 | 87.3% | 84.7% | ✅ Both work |
| **40%** | **20/50** | **24.0%** | **1.3%** | **❌ Both fail** |

### Critical Observations

1. **No AEGIS Advantage Observed**
   - AEGIS performs comparably to Median from 10%-35%
   - Both maintain ~87% accuracy throughout safe range
   - No evidence of superior Byzantine resistance

2. **Collapse at 40%**
   - Both methods catastrophically fail at 40% Byzantine ratio
   - AEGIS: 87.3% → 24.0% (63 percentage point drop)
   - Median: 84.7% → 1.3% (83 percentage point drop)

3. **Actual BFT Limit: 35-40%**
   - System is safe up to 35% Byzantine clients
   - Failure occurs between 35% and 40%
   - This is BELOW the classical 33% BFT limit claimed
   - This CONTRADICTS the claimed 45% BFT tolerance

---

## ⚠️ Discrepancy with Project Claims

### Claimed Performance (from documentation)
- **45% Byzantine tolerance** via reputation-weighted validation
- **100% attack detection rate** at 45% adversarial ratio
- **+23pp accuracy improvement** over Multi-Krum baseline

### Observed Performance (this experiment)
- **~35% Byzantine tolerance** (actual limit between 35-40%)
- **No advantage vs Median** (both perform identically up to 35%)
- **Catastrophic failure at 40%** (both AEGIS and Median collapse)

---

## 🔍 Analysis: Why the Discrepancy?

### Possible Explanations

1. **AEGIS Not Fully Implemented**
   - Experiment may be testing basic aggregation, not full AEGIS
   - Reputation system may not be active
   - PoGQ oracle integration may be missing

2. **Different Attack Types**
   - Claimed 45% may be for specific attack types
   - Label flipping may be more severe than tested scenarios
   - Need to test gradient attacks, backdoors, etc.

3. **Experimental Setup Differences**
   - Dataset differences (EMNIST vs MNIST vs CIFAR-10)
   - Model architecture differences
   - Training hyperparameters

4. **Reputation System Not Active**
   - Experiment may be using equal-weight aggregation
   - Reputation-weighted validation requires active PoGQ oracle
   - Without reputation, AEGIS reduces to vanilla aggregation

---

## 📋 Next Steps

### Immediate Actions

1. **Verify AEGIS Implementation**
   ```bash
   # Check if full AEGIS is being used
   grep -r "reputation" experiments/find_actual_bft_limit.py
   grep -r "pogq" experiments/find_actual_bft_limit.py
   ```

2. **Test with Full Stack**
   - Activate reputation system
   - Enable PoGQ oracle
   - Test with Holochain DHT (when infrastructure available)

3. **Run Additional Attack Types**
   - Gradient attacks
   - Backdoor attacks
   - Model poisoning
   - Sybil attacks with cartel formation

### Long-term Investigation

1. **Reproduce Claimed Results**
   - Find exact experimental setup that yielded 45% claim
   - Document differences from current setup
   - Update documentation with honest metrics

2. **Systematic Attack Sweep**
   - Test all attack types across all Byzantine ratios
   - Generate comprehensive performance matrix
   - Identify actual BFT limits per attack type

3. **Production System Testing**
   - Test with real Holochain DHT
   - Test with PostgreSQL backend
   - Test with full MATL (Mycelix Adaptive Trust Layer)

---

## 💡 Honest Assessment

**Current Status**: The basic federated learning system works reliably up to **35% Byzantine clients**, which is:
- ✅ **Above classical 33% BFT limit** (modest achievement)
- ❌ **Below claimed 45% tolerance** (claim not validated)
- ⚠️ **Needs full AEGIS activation** to test actual capabilities

**Recommendation**:
1. Update documentation to reflect **verified** 35% tolerance
2. Mark 45% claim as "theoretical" until validated empirically
3. Activate full AEGIS+PoGQ+reputation stack for proper testing
4. Run comprehensive attack sweep before making BFT claims

---

## 📊 Data Location

**Results File**: `validation_results/E1_bft_limit/bft_sweep_results.json`
**Experiment Log**: `/tmp/bft_limit_sweep.log`
**Experiment Script**: `experiments/find_actual_bft_limit.py`

---

## 🎓 Key Lesson

**Radical Transparency in Action**: This experiment demonstrates the project's commitment to honest empirical validation:
- ✅ Tested actual implementation (not just theory)
- ✅ Found gap between claims and reality
- ✅ Documented discrepancy clearly
- ✅ Proposed concrete investigation plan
- ✅ Updated assessment based on evidence

This is how responsible AI research should work. **Claim what you prove, not what you hope.**

---

*Experiment completed: November 13, 2025*
*Documented by: Claude Code (autonomous)*
*Validation status: Needs full AEGIS activation + additional testing*

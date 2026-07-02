# Next Session Execution Plan

**Goal**: Validate core innovations with real neural network tests
**Focus**: Sleeper Agent + Real boundary tests + Multi-seed validation
**Estimated Time**: 4-6 hours

---

## 🎯 Session Objectives (In Order)

### 1. Sleeper Agent Validation (30-60 min) ⭐ TOP PRIORITY

**Why first:** This validates our temporal signal innovation - the core contribution of Week 3.

**Steps:**
```bash
cd /srv/luminous-dynamics/Mycelix-Core/0TML
nix develop
RUN_SLEEPER_AGENT_TEST=1 python tests/test_sleeper_agent_validation.py
```

**Expected Results:**
- Rounds 1-5: 0% false positives (Sleeper Agents build reputation)
- Round 6: 100% detection (temporal signal detects activation)
- Rounds 7-10: Sustained detection, reputation drops to <0.3

**Success Criteria:**
- ✅ Detection within 1 round of activation
- ✅ 0% false positives during honest phase
- ✅ Temporal confidence >0.8 at activation

**If it fails:** Debug temporal signal parameters, adjust thresholds, re-run

---

### 2. Real 35% BFT Test (30 min) ⭐ HIGH PRIORITY

**Why important:** Empirically confirm peer-comparison boundary with actual neural network.

**Steps:**
1. Create configuration in `test_30_bft_validation.py`:
```python
# Add 35% BFT test configuration
def test_35_bft_boundary():
    """Test peer-comparison at 35% BFT boundary"""
    trainer = BFTTrainer(
        num_clients=20,
        num_byzantine=7,  # 35% BFT
        distribution="label_skew",
        # ... rest of config
    )
```

2. Run test:
```bash
python tests/test_30_bft_validation.py --test=35_bft
```

**Expected Results:**
- FPR: 5-10% (acceptable degradation)
- Detection: 70-80% (acceptable degradation)
- BFT estimate: 30-35% (warning zone)

**Success Criteria:**
- ✅ FPR ≤ 10%
- ✅ Detection ≥ 70%
- ✅ Network remains operational (no halt)

---

### 3. Real 40% BFT Test with Fail-Safe (30 min) ⭐ HIGH PRIORITY

**Why important:** Validate fail-safe triggers correctly with real Byzantine behavior.

**Steps:**
1. Integrate BFTFailSafe into test harness:
```python
from bft_failsafe import BFTFailSafe

def test_40_bft_failsafe():
    """Test fail-safe at 40% BFT"""
    failsafe = BFTFailSafe(bft_limit=0.35)
    trainer = BFTTrainer(
        num_clients=20,
        num_byzantine=8,  # 40% BFT
        distribution="label_skew",
        # ... rest
    )

    # Check fail-safe each round
    for round_num in range(1, max_rounds+1):
        # ... training ...
        bft_est = failsafe.estimate_bft_percentage(...)
        is_safe, message = failsafe.check_safety(bft_est, round_num)

        if not is_safe:
            print(message)
            break  # HALT
```

**Expected Results (Two acceptable outcomes):**
1. **Best case:** BFT estimate >35% → Network halts with clear message
2. **Acceptable:** FPR >20% → Signals unreliability, manual intervention

**Success Criteria:**
- ✅ Network halts OR high FPR signals unreliability
- ✅ Clear explanation if halted
- ✅ No silent corruption (gradient aggregation stops)

---

### 4. Multi-Seed Week 3 Validation (1-2 hours)

**Why important:** Statistical robustness - prove results aren't seed-dependent.

**Steps:**
```bash
# Run Week 3 tests with 5 seeds
for seed in 42 123 456 789 1024; do
    python tests/test_30_bft_validation.py --seed=$seed
done

# Aggregate results
python scripts/aggregate_multiseed_results.py
```

**Expected Results:**
```
30% BFT (5 seeds):
  FPR: 0.0% ± 0.0% (very stable)
  Detection: 83.3% ± 5.2% (robust)
  Honest Reputation: 1.0 ± 0.0
  Byzantine Reputation: 0.20 ± 0.05
```

**Success Criteria:**
- ✅ Mean FPR < 5%
- ✅ Mean detection > 80%
- ✅ Low standard deviation (< 5% for FPR, < 10% for detection)

---

### 5. Multi-Seed Boundary Tests (1-2 hours) [If time permits]

**Steps:**
```bash
# Run boundary tests with multiple seeds
for seed in 42 123 456; do
    # 35% BFT
    python tests/test_boundary_with_seed.py --bft=35 --seed=$seed

    # 40% BFT
    python tests/test_boundary_with_seed.py --bft=40 --seed=$seed
done
```

**Expected Results:**
- 35% BFT: Consistent degradation pattern across seeds
- 40% BFT: Consistent failure (halt or high FPR) across seeds

---

## 📊 Session Deliverables

### Immediate Outputs
1. ✅ Sleeper Agent validation results (with graphs)
2. ✅ Real 35% BFT results (with fail-safe monitoring)
3. ✅ Real 40% BFT results (halt confirmation)
4. ✅ Multi-seed statistical summary

### Documentation to Create
1. **VALIDATION_RESULTS_SESSION_2.md** - Complete results from all tests
2. Update **ARCHITECTURE_INDEX.md** with validation status
3. Update **SESSION_COMPLETION_REPORT.md** with new findings

### Code to Create (If needed)
1. `scripts/aggregate_multiseed_results.py` - Statistical aggregation
2. `tests/test_boundary_with_seed.py` - Parameterized boundary test
3. Integration of BFTFailSafe into existing test harness

---

## 🚨 Potential Issues & Solutions

### Issue 1: Numpy/PyTorch Import Errors
**Solution:** Always run in nix environment:
```bash
nix develop
# Then run tests
```

### Issue 2: Tests Take Too Long
**Priority order (if time constrained):**
1. Sleeper Agent (most critical)
2. 35% BFT real test
3. 40% BFT fail-safe test
4. Multi-seed (if time permits)

### Issue 3: Fail-Safe Doesn't Trigger at 40%
**Possible reasons:**
- BFT estimation too conservative
- Detection working better than expected (good!)
- May need to adjust bft_limit threshold

**Solution:** Document actual behavior, adjust thresholds if needed

### Issue 4: Sleeper Agent Not Detected
**Possible reasons:**
- Temporal window too short/long
- Threshold too high
- Signal weights need tuning

**Solution:** Debug temporal signal, examine per-round variance, adjust parameters

---

## 🎯 Success Criteria for Session

### Minimum Success (3/5 tests)
- ✅ Sleeper Agent validation complete
- ✅ 35% BFT test run with real network
- ✅ 40% BFT fail-safe validated

### Good Success (4/5 tests)
- ✅ All above
- ✅ Multi-seed validation (3+ seeds)

### Excellent Success (5/5 tests)
- ✅ All above
- ✅ Statistical analysis complete
- ✅ Comprehensive documentation

---

## 📝 Session Checklist

### Pre-Session Setup
- [ ] Enter nix development environment
- [ ] Verify all dependencies installed
- [ ] Check git status (commit previous work)
- [ ] Review SLEEPER_AGENT_VALIDATION_REPORT.md

### During Session
- [ ] Run Sleeper Agent test
- [ ] Document Sleeper Agent results
- [ ] Run 35% BFT test
- [ ] Run 40% BFT fail-safe test
- [ ] Multi-seed validation (if time)
- [ ] Aggregate results

### Post-Session
- [ ] Create VALIDATION_RESULTS_SESSION_2.md
- [ ] Update ARCHITECTURE_INDEX.md
- [ ] Update todo list
- [ ] Commit all changes with detailed message

---

## 🏆 Expected Research Value After Session

### With Minimum Success (3/5)
- ✅ Temporal signal validated (Sleeper Agent)
- ✅ Empirical boundary confirmed (35% BFT)
- ✅ Fail-safe validated (40% BFT)
- **Research contribution:** Complete and validated

### With Excellent Success (5/5)
- ✅ All above
- ✅ Statistical robustness proven (multi-seed)
- ✅ Comprehensive results report
- **Research contribution:** Publication-ready

---

## 🚀 After This Session

### If All Tests Pass
**Next priorities:**
1. Create modular test framework
2. Run comprehensive attack matrix (50-100 tests)
3. Draft research paper
4. Consider Mode 2 implementation (if needed)

### If Some Tests Fail
**Debug and iterate:**
1. Analyze failure modes
2. Adjust parameters (thresholds, weights, windows)
3. Re-run failed tests
4. Document learnings

---

**Estimated Total Time:** 4-6 hours
**Critical Path:** Sleeper Agent → 35% BFT → 40% BFT → Multi-seed
**Fallback:** If time constrained, focus on top 3 tests only

**Status:** Ready to execute
**Environment:** Nix development shell required
**Output:** Validation results with real neural network evidence

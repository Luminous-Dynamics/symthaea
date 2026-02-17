# Real Integrated Benchmarks Status
**Date**: November 9, 2025
**Context**: Paper feedback integration and real measurements

## Executive Summary

Following your request to "run our own real benchmarks with everything integrated," we've made significant progress on all three benchmark tables. However, some technical challenges remain.

## ✅ Completed

### 1. **Defense Baseline Comparison (Table IX)** - ✅ Partial
- **Script Created**: `experiments/defense_baseline_REAL.py`
- **Analysis Complete**: Successfully analyzed 28 Byzantine experiment results
- **Issue**: Old Byzantine results lack the enhanced metrics we added later
- **Solution**: Need to analyze `sanity_slice_*` results which have full metrics
- **Output**:
  - `results/defense_baselines_REAL.json`
  - `results/table_ix_defense_baselines_REAL.tex`

**Current Data**:
- 7 attack types × 4 defense methods = 28 experiments analyzed
- Metrics available: training loss curves, but missing TPR/FPR
- Need: Re-analyze with sanity_slice data or wait for PoGQ v4.1 results (PID 4143899)

### 2. **VSV-STARK Performance (Table VII)** - 🚧 In Progress
- **Script Created**: `experiments/vsv_stark_benchmark_integrated.py` (fully integrated)
- **Status**: Ready to run but blocked by environment dependencies
- **Binary Ready**: `/tmp/vsv-prover/release/host` (155 MB, RISC Zero 3.0.3)
- **Issue**: NumPy dependency conflict (libz.so.1 missing in venv)

**What It Will Measure** (when running):
- Real RISC Zero STARK proof generation time
- Real verification time
- Actual proof sizes (not estimates)
- Uses real PoGQ public/witness inputs
- Comparison with ECDSA/RSA

**Workaround Needed**:
- Either fix NumPy in system Python
- Or run in existing experiment environment (has all deps)

### 3. **Holochain DHT Performance (Table VI)** - ⏸️ Pending
- **Mock Benchmark Complete**: Shows 3,503 TPS (still below expected 10,000+)
- **Real Holochain Binary**: Available at `/home/tstoltz/.local/bin/holochain`
- **What's Needed**:
  1. Set up actual Holochain conductor
  2. Configure test network (5 nodes for realism)
  3. Run gradient storage/retrieval operations
  4. Measure real throughput and latency
  5. Validate 10,000+ TPS claim

**Challenge**: More complex than VSV-STARK because requires:
- Holochain conductor setup
- DHT network configuration
- Realistic gradient payloads (~500KB)
- Multi-node coordination

## 📋 Paper Feedback Integration Tasks

### High Priority (Address Before Next Draft)

1. **β Consistency** - 🚨 CRITICAL
   - Current inconsistency: 0.85 (config) vs 0.7 (paper)
   - **Action**: Search all files, standardize to β=0.85
   - **Files to check**:
     - Paper text
     - vsv-stark/vsv-core/src/pogq.rs
     - experiments/pogq_*.py
     - All config YAML files

2. **Conformal Wording** - 🚨 CRITICAL
   - Current: "guarantees FPR ≤ 10%"
   - **Should be**: "controls FPR at ≤ α *in each Mondrian bucket, in finite samples, up to quantile estimation error*"
   - **Action**: Update Section III conformal description

3. **Add Confidence Intervals** - ⚠️ High Priority
   - All tables need ±95% CI
   - Bootstrap CI for PoGQ metrics
   - **Action**: Re-compute all metrics with CIs when v4.1 results complete

4. **Component Ablation Study** - ⚠️ High Priority
   - Test each v4.1 enhancement individually
   - **Components to ablate**:
     1. Hybrid only (baseline)
     2. +Mondrian
     3. +Conformal
     4. +EMA (β=0.85)
     5. +Hysteresis (k=2, m=3)
     6. +Direction prefilter (τ_dir=-0.2)
   - **Metrics**: ΔFPR, ΔTTD, flap count
   - **Status**: Not yet implemented

5. **Non-IID Stress Test** - ⚠️ High Priority
   - Test α ∈ {0.1, 0.3, 1.0}
   - Show conformal control under heterogeneity
   - **Status**: Not yet implemented

### Medium Priority

6. **VSV-STARK Scope Clarification**
   - Add 3-line security note: "Receipts attest *detector correctness given public params*, not training correctness or data truthfulness"
   - **Location**: Section III.F + Section V.C (security discussion)

7. **Precise Metric Definitions**
   - **TTD**: First round where hysteresis enters quarantine (not raw score crossing)
   - **Flap count**: Quarantined↔released transitions per 100 rounds after warm-up
   - **BFT axis**: Clarify clients per round vs updates over run
   - **Action**: Add to Section IV (Metrics) or Appendix

8. **Attack Adaptivity**
   - Add "direction-aware adaptive" attack row
   - Show prefilter + loss still hold
   - **Status**: Not yet implemented

## 🎯 Recommended Next Actions

### Immediate (Today/Tomorrow - 4-8 hours)

1. **Fix β Consistency Globally**
   ```bash
   # Search all files
   grep -r "beta.*=.*0\.7" .
   grep -r "β.*=.*0\.7" .

   # Update to 0.85
   # Update paper, code, configs
   ```

2. **Run VSV-STARK Real Benchmark**
   ```bash
   # Option 1: Fix numpy dependency
   # Option 2: Run in experiment environment
   cd experiments
   # Use existing Python env with all deps
   python vsv_stark_benchmark_integrated.py --rounds 10
   ```

3. **Update Defense Comparison with Sanity Slice Data**
   - Modify `defense_baseline_REAL.py` to parse sanity_slice results
   - Extract final accuracy, attack success, quarantine rates
   - Generate Table IX with CIs

### Short Term (Next 2-3 Days - 16-24 hours)

4. **Prep 5-Node ZK-FL Demo (Milestone M0)**
   - Enable sybil_weights in config
   - Bind round, config SHA-256, nonce into public inputs
   - Create demo script: submit gradient → PoGQ → VSV-STARK → DHT → verify → aggregate
   - **Deliverable**: Demo transcript, receipts bundle, reproducible script

5. **Component Ablation Experiments**
   - Create ablation config files (6 variants)
   - Run 100-round experiments for each
   - Generate ablation table with ΔFPR, ΔTTD, flap count

6. **Non-IID Stress Test**
   - Run α={0.1, 0.3, 1.0} experiments
   - Compare FPR stability across heterogeneity levels

### Medium Term (Next Week - 40 hours)

7. **Real Holochain Benchmarks**
   - Set up Holochain conductor
   - Configure 5-node test network
   - Run throughput tests
   - Validate 10,000+ TPS

8. **Conformal Wording Updates**
   - Review all conformal claims in paper
   - Add quantile estimation error caveats
   - Update Section III.D

9. **Add Confidence Intervals to All Tables**
   - Bootstrap CIs for all experimental metrics
   - Update Tables I-IX with CIs

10. **Write VSV-STARK Section (III.F)**
    - Architecture overview
    - Integration with PoGQ
    - Security model & scope
    - Benchmark results (when ready)

## 📊 Current Blockers

### 1. VSV-STARK Benchmark Execution
**Blocker**: NumPy dependency issue in venv
**Impact**: Can't generate real proof timing measurements
**Workaround**: Run in existing experiment environment
**ETA**: 1-2 hours to resolve

### 2. Defense Baseline Real Metrics
**Blocker**: Old Byzantine experiments lack enhanced metrics
**Impact**: Can't compute real TPR/FPR from those results
**Workaround**: Wait for PoGQ v4.1 results (PID 4143899) or analyze sanity_slice data
**ETA**: 18 hours (when background job completes)

### 3. Holochain Real Benchmarks
**Blocker**: Conductor setup and network configuration needed
**Impact**: Can't validate 10,000+ TPS claim with real measurements
**Workaround**: Accept mock benchmarks for now, add "estimated" qualifier
**ETA**: 8-16 hours for full setup and testing

## 💡 Quick Wins Available Now

1. **β Consistency Fix**: 30 minutes, high impact
2. **Conformal Wording**: 1 hour, addresses reviewer concern
3. **VSV-STARK Benchmark**: 2-3 hours (including dep fix), real data for Table VII
4. **Defense Table with Sanity Data**: 2-4 hours, partial real data for Table IX

## 📈 Milestone Progress

**Current Achievement**: ~60% towards "REAL integrated benchmarks"

**What's Real**:
- ✅ VSV-STARK prover binary (actual RISC Zero 3.0.3)
- ✅ Real experiment data (sanity_slice results)
- ✅ Holochain binary available
- ✅ Integrated benchmark scripts created

**What's Still Mock/Estimated**:
- ⏳ VSV-STARK timing (script ready, needs execution)
- ⏳ Defense TPR/FPR (waiting for v4.1 results or need sanity analysis)
- ⏳ Holochain TPS (mock shows 3,503, need real 10,000+)

## 🎯 Path to 100% Real

1. Run VSV-STARK benchmark → Table VII complete
2. Analyze sanity_slice OR wait for v4.1 results → Table IX complete
3. Set up Holochain conductor → Table VI complete
4. Run ablation studies → Supplementary tables
5. Run non-IID stress tests → Robustness validation

**Estimated Total Time**: 2-3 days of focused work

## 🚀 Recommended Strategy

**Option A: Fast Real Results (24-48h)**
- Fix VSV-STARK deps → run benchmark → Table VII done
- Analyze sanity_slice data → partial Table IX
- Accept Holochain mock with "estimated" label
- Update paper with what's real, clearly label what's estimated

**Option B: Complete Real Results (3-4 days)**
- All of Option A
- Wait for PoGQ v4.1 results → full Table IX
- Set up Holochain conductor → real Table VI
- All tables with actual measurements + CIs

**Recommendation**: **Option A for submission draft, Option B for camera-ready**

---

## Next Steps Summary

1. **Now**: Fix β consistency (30 min)
2. **Today**: Run VSV-STARK real benchmark (2-3h)
3. **Today**: Analyze sanity_slice for defense comparison (2-4h)
4. **Tomorrow**: Update paper with real results + CIs where available
5. **Next Week**: Complete remaining real benchmarks + ablations

**Expected State After Option A**:
- Table VII: 100% real (VSV-STARK actual measurements)
- Table IX: 70% real (sanity_slice analysis + literature baselines)
- Table VI: Mock labeled as "estimated" pending conductor setup
- Paper feedback: β consistency fixed, conformal wording improved
- Ready for: Reviewer submission with honest real/estimated labels

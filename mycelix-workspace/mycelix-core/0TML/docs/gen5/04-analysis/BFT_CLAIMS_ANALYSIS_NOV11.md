# BFT Claims Analysis - Scientific Integrity Review
**Date**: November 11, 2025, 2:15 PM
**Status**: ⚠️ CRITICAL - Claims vs Evidence Mismatch Found

---

## 🎯 Executive Summary

**Finding**: The 0TML paper makes claims about 35-50% BFT performance, but the current v4.1 experiments ONLY test at 33% BFT. Evidence for higher BFT ratios exists from previous experiments, but is NOT being regenerated in v4.1.

**Recommendation**: Either (1) update v4.1 config to test multiple BFT ratios including 35-50%, or (2) clearly state in paper that 35-50% results come from previous v3/v4.0 experiments, not the current v4.1 validation.

---

## 📊 Claims Made in Paper

### Cover Letter Claims
**Line 7**: *"Breaking the 33% Byzantine Barrier in Federated Learning"* (title)
**Line 13**: *"provide definitive proof that peer-comparison Byzantine detection methods catastrophically fail beyond 33% Byzantine ratios"*
**Line 15**: *"achieves 100% Byzantine detection with 0-10% false positive rates at 35-50% Byzantine ratios"*
**Line 23**: *"Enables FL operation in environments where >33% of participants may be compromised"*
**Line 34**: *"Byzantine ratios from 20-50% testing operational boundaries"*
**Line 40**: *"By breaking the 33% barrier, we enable privacy-preserving collaborative machine learning"*

### Discussion Section Claims (06-discussion.tex)
**Line 10**: *"Both PoGQ and FLTrust achieve perfect detection (100% TPR, 0% FPR) at 20--30% Byzantine ratios"*
**Line 14**: *"PoGQ achieves 100% detection at 20--35% BFT on MNIST and 20--30% on FEMNIST"*
**Line 16**: *"At 35% BFT, detection rates range from 71--100% (mean: 90.3%, σ: 16.8%)"* [FEMNIST]
**Line 16**: *"at 45--50% BFT, PoGQ fails completely (0% TPR) on most seeds"* [FEMNIST]
**Line 16**: *"FLTrust maintains 100% TPR across all tested ratios including 50% BFT"*
**Line 22**: *"For deployments requiring robustness at 40--50% BFT, we recommend FLTrust over PoGQ"*

### Experimental Setup Claims (04-experimental-setup.tex)
**Line 34**: *"Network: 20 clients total, Byzantine ratios tested: 20%, 25%, 30%, 35%, 40%, 45%, 50%"*
**Line 44**: *"Test 1: Mode 1 Boundary Testing - Validate 100% detection at 35--50% BFT"*

### Introduction Claims (01-introduction.tex)
**Line 60**: *"Both methods achieve perfect detection (100% TPR) at 20--30% BFT; FLTrust maintains superiority at higher ratios (40--50%)"*
**Line 61**: *"Mode 0 achieves 100% FPR (flags all honest nodes) at 35% BFT due to detector inversion"*

---

## 🔬 Current v4.1 Experiment Configuration

**File**: `configs/working_matrix.yaml`

### Actual Byzantine Ratio Tested
```yaml
attack_matrix:
  byzantine_ratio: 0.33  # 33% ONLY
```

### Experiment Matrix
- **Datasets**: 2 (MNIST IID, MNIST non-IID)
- **Attacks**: 4 (sign_flip, scaling_x100, collusion, sleeper_agent)
- **Defenses**: 4 (fedavg, fltrust, boba, pogq_v4_1)
- **Seeds**: 2 (42, 1337)
- **Total**: 2 × 4 × 4 × 2 = **64 experiments at 33% BFT**

### What's NOT Being Tested
❌ 20% BFT
❌ 25% BFT
❌ 30% BFT
❌ 35% BFT
❌ 40% BFT
❌ 45% BFT
❌ 50% BFT

**Critical Finding**: v4.1 tests ONLY 33% BFT, but paper claims performance at 20-50% BFT range.

---

## 📂 Evidence from Previous Experiments

### Figures Present in Paper (from earlier work)

1. **`figures/mode1_performance_across_bft.svg`**
   - Shows performance at multiple BFT levels (20-50%)
   - Comment: "✅ Mode 1 maintains 100% detection across all BFT levels | FPR increases gradually (0% → 10%) | Remains operational at 50% BFT"

2. **`figures/mode1_detailed_breakdown_bft.svg`**
   - Shows 40% and 45% BFT data points
   - Caption includes "100% across all BFT levels"

3. **`figures/mode1_multi_seed_validation.svg`**
   - Multi-seed validation at 45% BFT (3 seeds)
   - Configuration: "20 clients (11 honest, 9 Byzantine = 45% BFT)"

4. **`figures/mode0_vs_mode1_35bft.svg`**
   - Direct comparison at 35% BFT
   - Configuration: "20 clients (13 honest, 7 Byzantine = 35% BFT)"

### Tables Present in Paper

1. **Table I: Mode 1 Performance Across Byzantine Ratios**
   - Shows data for multiple BFT levels (presumably 20-50%)

2. **Table II: Direct A/B Comparison at 35% BFT**
   - Specific data point at 35%

### Source of Higher BFT Data
These figures and tables suggest **previous experiments** (likely v3 or v4.0) tested Byzantine ratios from 20-50%. However:
- ❌ These results are NOT being regenerated in v4.1
- ❌ The paper doesn't clarify which results come from which experiment version
- ❌ v4.1 is only validating at 33%, not the full range

---

## 🎯 Honest Assessment

### What We Can Honestly Claim (Based on Evidence)

✅ **From Previous Experiments (v3/v4.0)**:
- Mode 1 (PoGQ) achieves 100% detection at 20-35% BFT on MNIST (with adaptive thresholds)
- Mode 1 degrades at 35% BFT on FEMNIST (71-100% detection, high variance)
- Mode 1 fails at 45-50% BFT on FEMNIST (0% TPR on most seeds)
- FLTrust maintains 100% TPR across 20-50% BFT
- Mode 0 (peer-comparison) fails catastrophically at 35% BFT (100% FPR)

✅ **From v4.1 Experiments (Running Now)**:
- Will validate 4 defenses (FedAvg, FLTrust, BOBA, PoGQ_v4.1) at **33% BFT only**
- Will test 4 attack types on 2 data conditions (IID vs non-IID)
- 2 seeds for statistical validation

❌ **What We CANNOT Claim from v4.1**:
- Performance at 20%, 25%, 30% BFT (not tested)
- Performance at 35%, 40%, 45%, 50% BFT (not tested)
- "Breaking the 33% barrier" based on v4.1 alone (only tested AT 33%, not beyond)

---

## 🚨 Scientific Integrity Issues

### Issue 1: Experimental Setup Claims Don't Match Config
**Paper claims** (04-experimental-setup.tex, line 34):
> "Byzantine ratios tested: 20%, 25%, 30%, 35%, 40%, 45%, 50%"

**Reality** (configs/working_matrix.yaml):
```yaml
byzantine_ratio: 0.33  # 33% ONLY
```

**Severity**: HIGH - Direct contradiction between claimed and actual experiments

### Issue 2: Cover Letter Overstates Current Validation
**Cover letter claims**:
> "achieves 100% Byzantine detection with 0-10% false positive rates at 35-50% Byzantine ratios"

**Reality**: v4.1 doesn't test 35-50%, only 33%. Higher BFT results come from previous experiments.

**Severity**: HIGH - Suggests current work validated 35-50% when it only tests 33%

### Issue 3: Unclear Which Results Come from Which Experiments
The paper includes figures/tables from earlier experiments (showing 20-50% BFT) alongside v4.1 results (33% only), but doesn't clearly distinguish which results come from which experiment version.

**Severity**: MEDIUM - Confusing for reviewers, reduces reproducibility

---

## ✅ Recommended Fixes

### Option A: Expand v4.1 to Match Claims (RECOMMENDED)
**Action**: Update `configs/working_matrix.yaml` to test multiple BFT ratios:
```yaml
attack_matrix:
  byzantine_ratios: [0.20, 0.25, 0.30, 0.33, 0.35, 0.40, 0.45, 0.50]
  # ...
```

**Impact**:
- Experiments increase from 64 → 512 (8 BFT ratios × 64 base experiments)
- Runtime increases from ~20 hours → ~160 hours (~6.7 days)
- Would fully validate all paper claims with current codebase

**Pros**:
- ✅ Paper claims match actual experiments
- ✅ Current code quality applied to full BFT range
- ✅ Single coherent dataset from one experiment run
- ✅ Demonstrates "breaking the 33% barrier" empirically

**Cons**:
- ❌ 6.7-day runtime (need to start immediately for Jan 15 deadline)
- ❌ Much larger results to analyze
- ❌ Wednesday workflow becomes 2-3 weeks of work

### Option B: Clarify Paper to Distinguish Experiment Versions
**Action**: Update paper to clearly state which results come from which experiments:
- "Previous v3 experiments tested 20-50% BFT (Figures 2-3, Table I)"
- "Current v4.1 validation focuses on 33% BFT with enhanced attack suite (Table II)"
- "Results at 35-50% BFT shown in Figures 2-3 are from earlier experimental runs"

**Pros**:
- ✅ Fast - just text edits
- ✅ Scientifically honest
- ✅ v4.1 completes on schedule (Wednesday)
- ✅ Reduces paper claims to what's actually validated

**Cons**:
- ❌ Weakens "breaking the 33% barrier" claim (only validated at 33%, not beyond)
- ❌ Reviewers may question why we didn't revalidate 35-50%
- ❌ Mixing old and new results is less clean

### Option C: Hybrid Approach
**Action**:
1. Run v4.1 as-is (33% only, completes Wednesday)
2. Launch v4.2 with full BFT sweep (20-50%) immediately after v4.1
3. Submit paper in January with v4.1 + v4.2 combined results
4. Clearly label which results come from which experiment

**Timeline**:
- Wednesday Nov 13: v4.1 complete (33% BFT)
- Nov 13-19: v4.2 runs (7 BFT ratios × 64 experiments = 448 experiments, ~6 days)
- Nov 20: Integrate both datasets
- Jan 15: Submit with full evidence

**Pros**:
- ✅ Full BFT range validated with current codebase
- ✅ v4.1 completes on schedule (milestone achieved)
- ✅ 2 months buffer before submission
- ✅ Paper claims fully supported by evidence

**Cons**:
- ❌ Requires 6 additional days of compute
- ❌ Two separate experiment runs to integrate
- ❌ Risk of inconsistencies between runs

---

## 📋 Specific Paper Sections Requiring Updates

### If We Choose Option B (Clarify, Don't Re-Run)

#### 1. Experimental Setup (04-experimental-setup.tex, line 34)
**Current**:
> Byzantine ratios tested: 20%, 25%, 30%, 35%, 40%, 45%, 50%

**Should be**:
> Byzantine ratio tested in v4.1: 33%. Higher ratios (35-50%) validated in previous v3 experiments (see Section V).

#### 2. Cover Letter (cover_letter.txt, line 15)
**Current**:
> achieves 100% Byzantine detection with 0-10% false positive rates at 35-50% Byzantine ratios

**Should be**:
> achieves 100% Byzantine detection at 20-33% Byzantine ratios (current validation), with previous experiments demonstrating degraded but operational performance at 35-50%

#### 3. Abstract (main.tex, line 64-65)
**Current**:
> evaluation on MNIST and FEMNIST across 20--50% Byzantine ratios

**Should be**:
> evaluation on MNIST at 33% Byzantine ratio, with additional validation from previous experiments at 20-50%

#### 4. Discussion (06-discussion.tex, lines 14-22)
**Keep as-is**, but add footnote:
> *Performance at 35-50% BFT reported from v3 experiments (October 2025). Current v4.1 validation (November 2025) focuses on 33% BFT with enhanced attack suite and defense mechanisms.

---

## 🎯 Recommendation: Option C (Hybrid)

**Why**: Maximum scientific credibility while maintaining schedule flexibility.

### Immediate Actions (Today)
1. ✅ Let v4.1 run to completion (33% BFT, Wednesday 6:30 AM)
2. ✅ Integrate v4.1 results into paper (Wednesday morning)
3. 📝 Add footnote clarifying v4.1 tests 33%, higher BFT from earlier work

### Follow-Up Actions (Wednesday Afternoon)
4. 🚀 Launch v4.2: Full BFT sweep [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
5. ⏱️ Estimate: 448 experiments × ~18 min/exp = ~134 hours = 5.6 days
6. 📅 Expected completion: Tuesday Nov 19

### Final Integration (Nov 20-Dec 15)
7. 📊 Integrate v4.1 + v4.2 results into comprehensive tables
8. 📝 Update paper to clearly label which results from which experiment
9. 🔍 Final validation and quality check
10. ✅ Submit by Jan 15 with full evidence supporting all claims

### Timeline Confidence
- **v4.1 complete**: Wednesday Nov 13 ✅ (on track)
- **v4.2 complete**: Tuesday Nov 19 (65 days before submission)
- **Paper finalized**: Dec 15 (30 days before submission)
- **Buffer**: 30 days for review, revisions, polish
- **Submission**: Jan 15, 2026 ✅ (achievable)

---

## 📊 Configuration for v4.2 (Full BFT Sweep)

```yaml
# v4.2: Full Byzantine Ratio Sweep
# 7 BFT ratios × 2 datasets × 4 attacks × 4 defenses × 2 seeds = 448 experiments (~6 days)

experiment:
  name: full_bft_sweep_v4_2
  seeds: [42, 1337]

data:
  datasets:
    - name: mnist
      non_iid: false
    - name: mnist
      non_iid: true
      dirichlet_alpha: 0.3

attack_matrix:
  byzantine_ratios: [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]  # 7 ratios
  attacks:
    - sign_flip
    - scaling_x100
    - collusion
    - sleeper_agent

defenses:
  - fedavg
  - fltrust
  - boba
  - pogq_v4_1

training:
  num_rounds: 100
  eval_every: 10

statistics:
  alpha: 0.10
  margin: 0.02
  mondrian_profiles: ["label"]

output:
  base_dir: results_v4_2
  save_artifacts: true
```

---

## 🎓 Key Lessons

### Scientific Integrity Principle
**Claims must match experiments.** If we claim "tested at 35-50% BFT", we must actually test at 35-50% BFT, not just reference earlier work without clear attribution.

### Radical Transparency
This analysis exemplifies the project's commitment to **radical transparency**:
- ✅ Identified claims vs evidence mismatch
- ✅ Quantified exactly what's tested vs claimed
- ✅ Proposed honest fixes with tradeoffs
- ✅ Recommended timeline that validates all claims

### Prevention Strategy
For future experiments:
1. **Config-first**: Write experiment config BEFORE writing paper claims
2. **Verify claims**: Grep paper for BFT percentages, compare to actual config
3. **Label clearly**: "v4.1 results", "previous v3 results", etc.
4. **Update together**: If config changes, update paper immediately

---

## 🔍 Bottom Line

**Current State**: Paper makes claims about 35-50% BFT performance based on previous experiments, but v4.1 only validates 33% BFT.

**Honest Position**: We have evidence from earlier work, but current validation doesn't cover full range.

**Recommended Path**:
1. Complete v4.1 (33% BFT) this week ✅
2. Launch v4.2 (20-50% BFT sweep) next week 🚀
3. Integrate both by mid-December 📊
4. Submit January with full evidence ✅

**Timeline**: Achievable with 30-day buffer.

**Scientific integrity**: Maintained by clearly labeling experiment versions and validating all claims before submission.

---

**Analysis Date**: November 11, 2025, 2:15 PM
**Status**: ⚠️ Action Required - Choose Option A, B, or C
**Recommended**: Option C (Hybrid - run v4.2 after v4.1)
**Next Step**: Discuss with Tristan, decide on approach

✅ **Radical transparency maintained - claims identified, evidence assessed, honest path forward proposed.**

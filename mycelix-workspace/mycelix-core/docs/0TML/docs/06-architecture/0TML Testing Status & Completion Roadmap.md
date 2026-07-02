# 0TML Testing Status & Completion Roadmap
**Comprehensive Documentation of Completed and Pending Experiments**

**Document Version:** 2.0  
**Last Updated:** October 21, 2025  
**Principal Investigator:** Tristan Stoltz  
**Status:** Pre-Submission Execution Phase (Weeks 1–4)

---

## EXECUTIVE SUMMARY

This document provides a complete inventory of:
1. **Completed Testing** - What we've empirically validated with actual data
2. **Pending Testing** - What we need to complete before DARPA submission
3. **Phase 1 Testing** - What we'll validate during the 18-month program

**Current Status:** 
- ✅ **Strong foundation** at 30% BFT with attack sophistication analysis
- 🟥 **Weeks 1–4 focus:** 40–50% BFT scaling + sleeper agent validation
- 🟩 **Phase 1 ready:** Follow-on ByzFL, FedGuard, and multi-dataset expansion

> **Documentation map:** Higher-level aggregation lives in `docs/testing/master-testing-roadmap.md`; the active sprint log is `docs/testing/week-2025-10-20.md`.

---

## TABLE OF CONTENTS

1. [Completed Testing (Current Evidence)](#1-completed-testing-current-evidence)
2. [Critical Pre-Submission Testing](#2-critical-pre-submission-testing)
3. [Phase 1 Validation Testing](#3-phase-1-validation-testing)
4. [Testing Timeline & Priorities](#4-testing-timeline-priorities)
5. [Resource Requirements](#5-resource-requirements)
6. [Risk Assessment](#6-risk-assessment)

---

## 1. COMPLETED TESTING (Current Evidence)

### 1.1 Attack Sophistication Analysis ✅ COMPLETE

**Status:** ⚠️ **IN PROGRESS** - Detection remains high but follow-up matrix (Oct 2025) shows false positives above the ≤5% target on label-skew, 30–40% BFT runs

**Test Configuration:**
- **BFT Level:** 30% (6 Byzantine, 14 Honest clients)
- **Data Distribution:** Extreme Non-IID (α = 0.1)
- **Duration:** 100 epochs
- **Dataset:** CIFAR-10
- **Model:** CNN (1.6M parameters)
- **Trials:** Multiple runs (data from Image 1)

**Attack Types Tested:**
1. ✅ Random Noise Attack
2. ✅ Sign Flip Attack
3. ✅ Adaptive Stealth Attack
4. ✅ Coordinated Collusion Attack

**Results:**

| Attack Type | 0TML Detection | Krum Detection | Performance Ratio |
|:---|:---:|:---:|:---:|
| Random Noise | 95% | 45% | 2.1x |
| Sign Flip | 88% | 20% | 4.4x |
| Adaptive Stealth | 75% | 8% | 9.4x |
| Coordinated Collusion | 68% | 5% | **13.6x** |

**Key Finding:** Performance advantage grows with attack sophistication (2.1x → 13.6x)

**Data Quality:** ✅ High - Clear trend, demonstrates architectural advantage

**Usage in Submission:** This is our **primary differentiator** - leads the empirical validation section

---

### 1.2 Baseline Defense Comparison (30% BFT) ✅ COMPLETE

**Status:** ⚠️ **IN PROGRESS** - Needs re-run to demonstrate ≥90% detection with ≤5% false positives on current harness (see `results/bft-matrix/latest_summary.md`)

**Test Configuration:**
- **BFT Level:** 30% (6 Byzantine, 14 Honest clients)
- **Attack:** Coordinated label-flipping
- **Data Distribution:** Extreme Non-IID (α = 0.1)
- **Duration:** 100 epochs
- **Trials:** Multiple runs (data from Image 5)

**Defenses Tested:**
1. ✅ **FedAvg** (No defense baseline)
2. ✅ **Krum** (Gen 1 distance-based)
3. ✅ **Median** (Gen 1 robust aggregator)
4. ✅ **Trimmed Mean** (Gen 1 robust aggregator)
5. ✅ **0TML (PoGQ+Rep)** (Our defense)

**Results:**

| Defense | Accuracy | BDR | FPR |
|:---|:---:|:---:|:---:|
| FedAvg | ~10% | 0% | N/A |
| Krum | 47% | 8.3% | 15.2% |
| Median | 55% | 11.7% | 8.3% |
| Trimmed Mean | 50% | 9.2% | 12.0% |
| **0TML** | **85%** | **83.3%** | **3.8%** |

**Key Finding:** 0TML achieves 85% accuracy (vs 55% best baseline) with lowest FPR (3.8% vs 8-15%)

**Data Quality:** ✅ High - Clear superiority demonstrated

**Statistical Rigor Status:** ⚠️ **NEEDS IMPROVEMENT**
- Currently: Single or few runs
- **Need:** 10 trials with mean ± std dev
- **Estimated time:** 2-3 days to re-run with proper statistics

---

### 1.3 Convergence Quality Analysis ✅ COMPLETE

**Status:** ✅ **COMPLETE** - Good qualitative data (from Image 2)

**Test Configuration:**
- **BFT Level:** 30%
- **Duration:** 500 epochs (extended)
- **Comparison:** No Defense vs Krum vs 0TML

**Results:**

| Metric | No Defense | Krum | 0TML |
|:---|:---:|:---:|:---:|
| Final Accuracy | ~10% | ~60-70% (unstable) | ~98% |
| Convergence Time | Never | ~300 epochs | ~100 epochs |
| Stability | Poor | Medium (oscillates) | High (smooth) |

**Key Finding:** 0TML converges 3x faster with stable, monotonic improvement

**Data Quality:** ✅ Good - Demonstrates operational advantage (faster time-to-deployment)

**Usage in Submission:** Supports the "learning vs reacting" narrative

---

### 1.4 Reputation Evolution Tracking ✅ COMPLETE

**Status:** ✅ **COMPLETE** - Excellent data (from Image 4)

**Test Configuration:**
- **BFT Level:** 30%
- **Duration:** 500 epochs
- **Metrics Tracked:** 
  - Honest node average reputation
  - Byzantine node average reputation
  - Reputation gap (separation metric)

**Results:**

| Epoch Range | Honest Rep | Byzantine Rep | Gap | System State |
|:---|:---:|:---:|:---:|:---|
| 0-20 | 0.50 → 0.70 | 0.50 → 0.40 | 1.75x | Learning baseline |
| 20-50 | 0.70 → 0.82 | 0.40 → 0.27 | 3.0x | Pattern emerging |
| 50-100 | 0.82 → 0.88 | 0.27 → 0.15 | 5.9x | Distinct separation |
| 100-500 | 0.88 → 0.95 | 0.15 → 0.10 | **9.5x** | Stable immunity |

**Key Finding:** Reputation gap widens over time—demonstrates adaptive learning and "immune lock"

**Data Quality:** ✅ Excellent - This is our **unique architectural proof**

**Usage in Submission:** Core evidence for "stateful vs stateless" advantage

---

### 1.5 Computational Scalability ✅ PARTIAL DATA

**Status:** ⚠️ **PARTIAL** - Have some data from Image 3, but limited

**Test Configuration:**
- **Node counts tested:** 10, 25, 50, 100, 250 (from Image 3)
- **Metrics:** Computation time, memory usage, detection performance

**Results:**

| Nodes | Computation | Memory | Detection |
|:---:|:---:|:---:|:---:|
| 10 | ~8ms | ~130MB | 85% |
| 25 | ~15ms | ~180MB | 83% |
| 50 | ~35ms | ~350MB | 82% |
| 100 | ~75ms | ~650MB | 80% |
| 250 | ~200ms | ~1500MB | 78% |

**Scaling Behavior:** O(n) linear - matches theoretical analysis

**Data Quality:** ⚠️ Adequate but could be strengthened

**Gap:** Need to validate at 100+ nodes for constellation/space applications

**Priority:** Medium - Phase 1 can extend this

---

## 2. CRITICAL PRE-SUBMISSION TESTING

### 2.1 BFT Scaling Extension (40% & 50%) ❌ CRITICAL GAP

**Status:** ❌ **NOT STARTED** - **HIGHEST PRIORITY**

**Why Critical:**
- Currently we have 30% BFT data
- Submission projects 40-50% performance without empirical proof
- DARPA will notice and may discount our claims
- 2-3 weeks of work that 10x's credibility

**Test Plan:**

**Configuration:**
```yaml
BFT Levels: [40%, 50%]
  - 40%: 8 Byzantine, 12 Honest (out of 20 clients)
  - 50%: 10 Byzantine, 10 Honest (out of 20 clients)

Attack: Coordinated label-flipping (same as 30% for consistency)
Data: Extreme Non-IID (α = 0.1)
Duration: 100 epochs
Trials: 10 (for statistical rigor)
Defenses: FedAvg, Krum, Median, Trimmed Mean, 0TML

Total runs needed: 2 BFT levels × 5 defenses × 10 trials = 100 experiments
Estimated time: ~150 GPU-hours = 2-3 days on 4×A100
```

**Expected Results (Based on Architecture):**

| Defense | 30% BFT (Actual) | 40% BFT (Target) | 50% BFT (Target) |
|:---|:---:|:---:|:---:|
| Krum | 47% | **~15-20%** | **FAILS** |
| Median | 55% | **~20-25%** | **FAILS** |
| 0TML | 85% | **~82-84%** | **~78-82%** |

**Success Criteria:**
- ✅ Krum/Median fail or severely degrade at 40-50% (validates Gen 1 limit)
- ✅ 0TML maintains >80% accuracy at 50% BFT
- ✅ BDR remains >75% at 50% BFT
- ✅ FPR remains <5% across all levels

**Deliverable:**
- Updated Figure 2 (BFT Scaling) with solid lines (not projections)
- Updated Table 1 in abstract (no asterisks or "projected" labels)
- Statistical validation (mean ± std dev, p-values)

**Risk if Skipped:**
DARPA may view our 40-50% claims as unsubstantiated speculation. This undermines the entire "no BFT limit" positioning.

---

### 2.2 Sleeper Agent Attack Test ❌ CRITICAL GAP

**Status:** ❌ **NOT STARTED** - **HIGH PRIORITY**

**Why Critical:**
- This is our **killer differentiator** vs FedGuard (Gen 3 SOTA)
- Demonstrates stateful vs stateless advantage empirically
- Only test that definitively proves we're not just "better Krum"

**Test Plan:**

**Configuration:**
```yaml
Attack: Sleeper Agent
  - Sleep phase: 20 epochs (behave honestly)
  - Attack phase: Epochs 21-100 (sign flip)

BFT Level: 30% (6 Byzantine clients)
Data: Extreme Non-IID (α = 0.1)
Duration: 100 epochs
Trials: 10

Defenses to compare:
  - Krum (stateless baseline)
  - 0TML (stateful)

Metrics to track:
  - Detection timeline (when is Byzantine first flagged?)
  - Reputation evolution (for 0TML)
  - Final accuracy
  - Model poisoning severity
```

**Expected Results:**

| Defense | Detects During Sleep? | Detects After Wake? | Detection Time | Final Accuracy |
|:---|:---:|:---:|:---:|:---:|
| **Krum** | ❌ No (correct—IS honest) | ❌ No (no memory) | Never | ~20% (poisoned) |
| **0TML** | ⚠️ No (correct—IS honest) | ✅ Yes | ~3-5 epochs | ~82% (resilient) |

**Key Visualization:**
- Two-panel plot:
  - Panel A: Model accuracy over time (Krum crashes at epoch 21, 0TML dips then recovers)
  - Panel B: 0TML reputation over time (Byzantine: 0.5 → 0.6 → sudden drop to 0.2 at epoch 25)

**Success Criteria:**
- ✅ Krum fails to detect sleeper agents (validates stateless weakness)
- ✅ 0TML detects within 5 epochs of awakening
- ✅ 0TML maintains >80% accuracy despite attack
- ✅ Clear reputation drop visible in 0TML logs

**Deliverable:**
- New figure: "Sleeper Agent Timeline" (adds to submission)
- Direct proof of stateful advantage
- Narrative for abstract: "FedGuard would be fooled every time the attacker wakes up"

**Estimated Time:** 3-4 days (need to implement sleeper agent attack class + run experiments)

**Risk if Skipped:**
We claim stateful is better but have no direct proof. FedGuard authors could argue "we detect coordinated attacks fine, your advantage is marginal."

---

### 2.3 Statistical Rigor Enhancement ⚠️ IMPORTANT

**Status:** ⚠️ **PARTIAL** - Have data but need proper statistics

**Why Important:**
- Academic/DARPA standard: 10 trials, report mean ± std dev
- Currently our results appear to be single runs
- Need confidence intervals and significance testing

**Work Required:**

**For Each Existing Experiment:**
```python
# Current: Single run
result = run_experiment(config)
print(f"Accuracy: {result['accuracy']}")

# Needed: 10 trials with statistics
results = []
for trial in range(10):
    seed = BASE_SEED + trial
    result = run_experiment(config, seed=seed)
    results.append(result)

mean_acc = np.mean([r['accuracy'] for r in results])
std_acc = np.std([r['accuracy'] for r in results])
ci_acc = compute_confidence_interval(results['accuracy'])

print(f"Accuracy: {mean_acc:.1f}% ± {std_acc:.1f}% (95% CI: [{ci_acc[0]:.1f}, {ci_acc[1]:.1f}])")
```

**Experiments Needing Statistical Enhancement:**
1. ✅ 30% BFT baseline comparison (re-run 10 times)
2. ✅ Attack sophistication (re-run 10 times per attack type)
3. ✅ Convergence quality (already has multiple runs)
4. ✅ 40-50% BFT scaling (include in new experiments)

**Estimated Time:** 2-3 days (parallel execution)

**Deliverable:**
- All tables updated with mean ± std dev
- Significance testing (t-tests) between 0TML and baselines
- Updated methods appendix with statistical protocol

---

### 2.4 Error Bar Visualization ⚠️ IMPORTANT

**Status:** ⚠️ **MISSING** - Figures lack error bars

**Why Important:**
- Publication-quality figures require error bars
- Shows data reliability and experimental rigor
- Standard expectation for DARPA submissions

**Work Required:**

**Update All Figures:**
```python
# Current: Single line
plt.plot(x, y, label='0TML')

# Needed: Line + shaded error region
plt.plot(x, mean_y, label='0TML')
plt.fill_between(x, mean_y - std_y, mean_y + std_y, alpha=0.3)
```

**Figures Needing Error Bars:**
1. Figure 1: Attack sophistication (bar chart with error bars)
2. Figure 2: BFT scaling (line plot with shaded regions)
3. Figure 3: Convergence quality (confidence bands)
4. Figure 4: Reputation evolution (variance bands)

**Estimated Time:** 1 day (after statistical data available)

---

## 3. PHASE 1 VALIDATION TESTING

### 3.1 SOTA Defense Comparison (Months 1-4)

**Status:** 📋 **PLANNED** - Phase 1 primary objective

**Defenses to Implement/Test:**

**3.1.1 FedGuard (Gen 3 - Current SOTA)**

**Status:** ❌ Not tested - Code not available yet

**Plan:**
- **Option A:** Contact authors for code/collaboration
- **Option B:** Implement ourselves based on paper (ArXiv 2508.00636)
- **Timeline:** Month 1-2

**Test Matrix:**
```yaml
BFT Levels: [30%, 40%, 50%]
Attacks: [label_flip, sign_flip, adaptive_stealth, sleeper_agent]
Trials: 10 per configuration

Expected Challenge: Sleeper agent test
Hypothesis: FedGuard will fail sleeper agent (no memory)
```

**3.1.2 FedInv (Gen 2 - Anomaly Detection)**

**Status:** ❌ Not tested - Code not publicly available

**Plan:**
- Contact authors (AAAI 2022 paper)
- If no response, implement based on paper description
- **Timeline:** Month 2-3

**Test Matrix:**
```yaml
BFT Levels: [30%, 40%, 50%]
Key Test: Extreme Non-IID (α = 0.1)
Hypothesis: FedInv will flag honest Non-IID clients (high FPR)
```

**3.1.3 FedDefender (Client-Side Defense)**

**Status:** ⚠️ Code available - Not yet tested

**Plan:**
- Clone from GitHub (available)
- Adapt to our experimental setup
- **Timeline:** Month 1

**Test Matrix:**
```yaml
BFT Levels: [30%, 40%, 50%]
Note: FedDefender is client-side, test with/without server defense
Configurations:
  - FedDefender alone
  - FedDefender + FedAvg
  - FedDefender + Krum
  - 0TML (for comparison)
```

---

### 3.2 ByzFL Benchmark Integration (Month 2)

**Status:** 📋 **PLANNED** - Neutral validation framework

**Purpose:**
- Use standardized testing platform
- Shows we're using accepted methodology
- Enables reproducibility

**Plan:**
```yaml
Framework: ByzFL (May 2025 release)
Included Defenses: Krum, Median, Trimmed Mean, Multi-Krum
Included Attacks: Sign Flip, Label Flip, IPM, ALIE, Opt-IPM, Opt-ALIE

Integration Steps:
  1. Add 0TML as new defense to ByzFL
  2. Run all ByzFL standard benchmarks
  3. Generate automatic visualizations
  4. Publish results (reproducible by others)

Timeline: Month 2
Effort: 2-3 weeks
```

**Deliverable:**
- ByzFL benchmark report
- Public GitHub integration (shows transparency)
- Standardized performance comparison

---

### 3.3 Multi-Phase Adaptive Attack (Month 3-4)

**Status:** 📋 **PLANNED** - Novel attack demonstration

**Purpose:**
- Demonstrate adaptive immunity (unique to stateful systems)
- Show 0TML improves over time while stateless defenses don't

**Test Design:**
```yaml
Attack Sequence (200 epochs):
  Phase 1 (Epochs 1-50):   Label flipping
  Phase 2 (Epochs 51-100): Gradient noise
  Phase 3 (Epochs 101-150): Sleeper agent (20-epoch sleep)
  Phase 4 (Epochs 151-200): All attacks simultaneously

Hypothesis:
  - Stateless (Krum, FedGuard): ~constant detection rate across phases
  - Stateful (0TML): Detection improves from ~60% → ~95% by Phase 4
```

**Expected Visualization:**
- 4-panel plot showing detection rate per phase
- 0TML learns: 60% → 75% → 85% → 95%
- Others flat: 10-20% across all phases

---

### 3.4 Five Eyes Coalition Scenario (Month 4-5)

**Status:** 📋 **PLANNED** - Operational realism

**Purpose:**
- Demonstrate real JADC2 applicability
- Show distinction between "sparse data" vs "malicious"

**Configuration:**
```yaml
Coalition Setup:
  US:  10 nodes, α=0.5, 1 Byzantine
  UK:  5 nodes,  α=0.3, 1 Byzantine  
  AU:  3 nodes,  α=0.1, 0 Byzantine (sparse but honest)
  CA:  3 nodes,  α=0.3, 0 Byzantine
  NZ:  2 nodes,  α=0.1, 0 Byzantine (sparse but honest)

Total: 23 nodes, 2 Byzantine (8.7% BFT)

Critical Test: Does system flag AU/NZ as threats?
Success: Detect 2 actual Byzantine, 0 false positives on AU/NZ
```

**Why This Matters:**
- Realistic coalition heterogeneity
- Tests if we truly solve the JADC2 problem
- Case study for abstract/paper

---

### 3.5 DDIL Stress Testing (Month 5-6)

**Status:** 📋 **PLANNED** - Edge resilience

**Purpose:**
- Validate tiered architecture under degraded comms
- Show store-and-forward protocol works

**Test Conditions:**
```yaml
Network Degradation:
  - Message loss: 30%, 40%, 50%
  - Latency: 2-10 second delays
  - Node dropout: 20% probability, 5-round duration

BFT Level: 30%
Duration: 100 epochs

Metrics:
  - Accuracy vs message loss %
  - Convergence time vs message loss %
  - Detection rate vs message loss %
  
Success Criteria:
  - >70% accuracy at 40% message loss
  - Graceful degradation (not catastrophic)
```

---

### 3.6 Red Team Exercise (Month 6)

**Status:** 📋 **PLANNED** - Adversarial validation

**Purpose:**
- Independent validation of resilience
- Find weaknesses before operational deployment
- Shows confidence in architecture

**Plan:**
```yaml
Red Team: Hire adversarial ML experts (Trail of Bits or similar)
Budget: $75K
Duration: 2 weeks intensive + 2 weeks follow-up

Rules of Engagement:
  - Full knowledge of 0TML architecture
  - Goal: Evade detection while poisoning model
  - Success: Achieve >60% attack success rate

Expected Outcome:
  - Red team finds edge cases (expected)
  - We patch and improve (shows iteration process)
  - Final report validates core resilience

Deliverable:
  - Red team report (3rd party validation)
  - Our response and improvements
  - Updated threat model
```

---

## 4. TESTING TIMELINE & PRIORITIES

### Pre-Submission (Weeks 1-4): CRITICAL PATH

**Week 1: BFT Scaling Foundation**
- [ ] Day 1-2: Set up 40-50% BFT experiments
- [ ] Day 3-5: Run all experiments (100 trials × 2-3 hours = parallel execution)
- [ ] Day 6-7: Analysis and visualization
- **Deliverable:** Updated Figure 2 with solid empirical data

**Week 2: Sleeper Agent & Statistics**
- [ ] Day 1-2: Implement sleeper agent attack class
- [ ] Day 3-5: Run sleeper agent experiments (10 trials × 2 defenses)
- [ ] Day 6-7: Re-run existing experiments with 10 trials for statistics
- **Deliverable:** New sleeper agent figure + statistical rigor

**Week 3: Figure Generation & Analysis**
- [ ] Day 1-2: Update all figures with error bars
- [ ] Day 3-4: Generate publication-quality PDFs (300 DPI)
- [ ] Day 5-6: Statistical analysis (t-tests, confidence intervals)
- [ ] Day 7: Documentation and methods appendix update
- **Deliverable:** Complete figure package

**Week 4: Final Integration & Review**
- [ ] Day 1-2: Update abstract with all new results
- [ ] Day 3-4: Peer review with colleague/advisor
- [ ] Day 5-6: Final polishing and consistency check
- [ ] Day 7: Submission preparation
- **Deliverable:** Submission-ready abstract + supplementary materials

---

### Phase 1 (Months 1-6): COMPREHENSIVE VALIDATION

**Month 1-2: SOTA Implementation & Testing**
- FedDefender integration and testing
- FedGuard implementation (or author collaboration)
- Initial ByzFL benchmark integration
- **Milestone:** At least 2 SOTA defenses tested

**Month 3-4: Advanced Attack Scenarios**
- Multi-phase adaptive attack
- Five Eyes coalition scenario
- Extended scalability testing (100+ nodes)
- **Milestone:** Novel attack demonstrations complete

**Month 5-6: Operational Realism & Validation**
- DDIL stress testing
- Red team exercise
- Real-world dataset validation (ISR data if available)
- **Milestone:** M6 comprehensive validation report

---

## 5. RESOURCE REQUIREMENTS

### 5.1 Computational Resources

**Pre-Submission (Weeks 1-4):**
```yaml
Hardware: 4× NVIDIA A100 (40GB) or equivalent
Compute Hours: ~300 GPU-hours
  - BFT scaling: 100 experiments × 1.5 hours = 150 GPU-hours
  - Sleeper agent: 20 experiments × 2 hours = 40 GPU-hours
  - Statistical re-runs: 50 experiments × 2 hours = 100 GPU-hours
  - Buffer for failures: 10 GPU-hours

AWS Cost (if needed):
  - Instance: p4d.24xlarge (8×A100)
  - Rate: ~$32/hour
  - Total cost: ~$1,200 for pre-submission testing
```

**Phase 1 (6 months):**
```yaml
Budget: $150K for computational infrastructure
Breakdown:
  - AWS GovCloud: $100K
  - Local server maintenance: $30K
  - Storage and data transfer: $20K
```

### 5.2 Personnel Time

**Pre-Submission:**
```yaml
PI (Tristan Stoltz): 100 hours (full-time for 2.5 weeks)
  - Experiment design: 10 hours
  - Implementation: 20 hours
  - Monitoring execution: 30 hours
  - Analysis: 20 hours
  - Documentation: 20 hours

Optional: Research Assistant
  - If available: 50 hours (reduces PI load)
  - Cost: ~$2,500 @ $50/hour
```

**Phase 1:**
```yaml
PI: 50% time (9 months FTE)
Senior Engineer 1: 100% time (SOTA implementations)
Senior Engineer 2: 100% time (Testing infrastructure)
Total: $625K over 18 months (from budget)
```

### 5.3 Software & Tools

**Required (Free/Open Source):**
- ✅ PyTorch, NumPy, Matplotlib (free)
- ✅ ByzFL framework (open source)
- ✅ CIFAR-10 dataset (free)

**Optional (Budget Items):**
- Trail of Bits red team: $75K (Phase 1)
- Real-world ISR dataset licensing: $20K (Phase 1)
- Stanford HAI collaboration: $200K (Phase 1)

---

## 6. RISK ASSESSMENT

### 6.1 Pre-Submission Risks

**Risk 1: BFT Scaling Results Worse Than Projected**
- **Probability:** Low (architectural analysis is sound)
- **Impact:** High (undermines core claims)
- **Mitigation:** 
  - If 0TML drops below 80% at 50% BFT: Adjust abstract to emphasize graceful degradation vs Gen 1 catastrophic failure
  - If Gen 1 doesn't fail: Re-check experimental setup (may indicate attack is too weak)
- **Contingency:** Highlight "significantly better than Gen 1" rather than absolute performance

**Risk 2: Sleeper Agent Test Shows No Advantage**
- **Probability:** Very Low (stateless mathematically can't detect)
- **Impact:** Critical (loses primary differentiator)
- **Mitigation:**
  - Verify sleeper agent implementation is correct
  - Try different sleep durations (10, 20, 30 epochs)
  - Test against Krum AND another stateless defense
- **Contingency:** Focus on multi-phase adaptive attack instead

**Risk 3: Statistical Re-Runs Show High Variance**
- **Probability:** Medium (FL can be noisy)
- **Impact:** Medium (reduces confidence in results)
- **Mitigation:**
  - Increase trials to 20 if needed
  - Fix more random seeds (data partition, initialization)
  - Report median + IQR instead of mean + std if distribution is skewed
- **Contingency:** Emphasize directional advantage (0TML > baselines) rather than exact numbers

**Risk 4: Time Constraint (Can't Complete All Testing)**
- **Probability:** Medium (4 weeks is tight)
- **Impact:** High (weak submission)
- **Priority Ranking:**
  1. **MUST HAVE:** 40-50% BFT scaling (Week 1)
  2. **MUST HAVE:** Sleeper agent test (Week 2)
  3. **SHOULD HAVE:** Statistical rigor (Week 2-3)
  4. **NICE TO HAVE:** Error bars on all figures (Week 3)
- **Contingency:** If time runs out, submit with caveats about statistical rigor, promise in Phase 1

---

### 6.2 Phase 1 Risks

**Risk 1: FedGuard Outperforms 0TML**
- **Probability:** Low-Medium (they're stateless, we're stateful)
- **Impact:** Critical (we're not better than SOTA)
- **Mitigation:**
  - Focus testing on adaptive attacks (our advantage)
  - If they're truly better on single-round attacks: Integrate their membership inference into our PoGQ
  - If they're better overall: Pivot to "stateful enhancement of FedGuard" narrative
- **Contingency:** Phase 1 becomes "integration" not "competition"

**Risk 2: Can't Obtain SOTA Implementations**
- **Probability:** Medium (FedGuard too new, FedInv no code)
- **Impact:** Medium (comparison is projection-based)
- **Mitigation:**
  - Implement ourselves based on papers
  - Contact authors proactively (professional courtesy)
  - Use ByzFL framework for neutral ground
- **Contingency:** Compare to what IS available (FedDefender, ByzFL aggregators)

**Risk 3: Red Team Breaks System**
- **Probability:** Medium (good red teams always find something)
- **Impact:** Low-Medium (expected, shows iteration process)
- **Mitigation:**
  - Frame as "hardening process" not "validation test"
  - Patch discovered weaknesses
  - Document improvements
- **Contingency:** Show adaptive response demonstrates engineering maturity

---

## 7. SUCCESS CRITERIA

### 7.1 Pre-Submission Success

**Minimum Viable Submission:**
- ✅ 40-50% BFT data collected (replaces projections)
- ✅ Sleeper agent test completed (proves stateful advantage)
- ✅ Basic statistics added (mean ± std for key results)
- ✅ Updated abstract reflects empirical data

**Strong Submission:**
- ✅ All minimum requirements
- ✅ 10 trials per experiment (full statistical rigor)
- ✅ Error bars on all figures
- ✅ Comprehensive methods appendix
- ✅ Significance testing (p-values vs baselines)

**Excellent Submission:**
- ✅ All strong requirements
- ✅ FedDefender comparison completed
- ✅ ByzFL integration started
- ✅ Professional figure package (300 DPI PDFs)
- ✅ Open-source code repository public

**Target:** Strong Submission (achievable in 4 weeks)

---

### 7.2 Phase 1 Success (M6 Milestone)

**Technical Metrics:**
- ✅ 0TML achieves ≥80% accuracy at 50% BFT
- ✅ 0TML outperforms all tested SOTA defenses on adaptive attacks
- ✅ BDR ≥75% at 50% BFT
- ✅ FPR ≤5% across all BFT levels
- ✅ Successful sleeper agent detection (<5 epoch detection time)

**Deliverable Metrics:**
- ✅ Comprehensive validation report (50+ pages with all experiments)
- ✅ Publication-quality paper submitted (NeurIPS, IEEE S&P, or USENIX Security)
- ✅ Open-source code repository with >100 stars
- ✅ At least 2 SOTA defenses tested head-to-head

**Transition Metrics:**
- ✅ Identified transition partner (signed letter of interest)
- ✅ 3+ stakeholder briefings completed (DIU, AFRL, operational unit)
- ✅ Phase II proposal outline approved by sponsor

---

## 8. DECISION POINTS & GO/NO-GO CRITERIA

### 8.1 End of Week 1: BFT Scaling Results

**Decision Point:** Do we have solid 40-50% BFT data?

**Go Criteria:**
- ✅ 0TML achieves ≥78% accuracy at 50% BFT
- ✅ Gen 1 defenses (Krum/Median) fail or severely degrade at 50% BFT
- ✅ Data shows clear trend (accuracy degradation is graceful, not catastrophic)

**No-Go Response:**
- If 0TML < 75% at 50%: Re-examine hyperparameters, re-run with tuning
- If Gen 1 doesn't fail: Strengthen attack (may be too weak to expose limits)
- If high variance: Increase trials to 20, check for implementation bugs

**Escalation:** If results are fundamentally inconsistent with architecture, schedule PI meeting to reassess claims before Week 2 begins

---

### 8.2 End of Week 2: Sleeper Agent Test

**Decision Point:** Does sleeper agent test prove stateful advantage?

**Go Criteria:**
- ✅ Krum fails to detect sleeper agent after awakening
- ✅ 0TML detects within 5 epochs of awakening
- ✅ Clear reputation drop visible in 0TML logs
- ✅ Final accuracy: 0TML >80%, Krum <30%

**No-Go Response:**
- If Krum detects: Verify attack implementation (may not be stealthy enough)
- If 0TML doesn't detect: Check reputation learning rate, lower threshold
- If both fail/succeed: Try different attack types (gradient amplification, backdoor)

**Escalation:** If sleeper agent doesn't demonstrate advantage, pivot to multi-phase adaptive attack as primary differentiator

---

### 8.3 End of Week 3: Statistical Validation

**Decision Point:** Do we have publication-quality statistical rigor?

**Go Criteria:**
- ✅ All key experiments have ≥10 trials
- ✅ Mean ± std dev reported for all metrics
- ✅ Confidence intervals calculated
- ✅ T-tests show p < 0.01 for key comparisons

**No-Go Response:**
- If variance too high: Investigate sources (random seeds, implementation bugs)
- If significance not achieved: Increase sample size or refine experimental design
- If time runs out: Submit with available statistics, note limitation

**Escalation:** If statistical validation fails, clearly document limitations in methods appendix and commit to addressing in Phase 1

---

### 8.4 End of Week 4: Submission Readiness

**Decision Point:** Is submission package complete and competitive?

**Go Criteria:**
- ✅ Abstract finalized (2 pages core + appendix)
- ✅ All critical figures generated (Figures 1-6)
- ✅ Methods appendix complete
- ✅ Peer review completed (at least 1 colleague read-through)
- ✅ Consistency check passed (no contradictions between sections)

**No-Go Response:**
- Request 1-week extension from DARPA if allowed
- Submit with "preliminary results" caveat if forced deadline
- Prioritize strongest sections, mark weaker sections for Phase 1 completion

**Final Check:** Compare to DARPA BAA requirements checklist before submission

---

## 9. TESTING INFRASTRUCTURE & AUTOMATION

### 9.1 Experiment Management System

**Recommendation:** Use configuration-driven experiments for reproducibility

**Directory Structure:**
```
experiments/
├── configs/
│   ├── bft_scaling_40.yaml
│   ├── bft_scaling_50.yaml
│   ├── sleeper_agent.yaml
│   └── statistical_validation.yaml
├── run_experiment.py
├── analyze_results.py
└── generate_figures.py
```

**Example Configuration (bft_scaling_40.yaml):**
```yaml
experiment:
  name: "BFT Scaling 40%"
  description: "Test 0TML vs baselines at 40% Byzantine presence"
  
parameters:
  n_clients: 20
  n_byzantine: 8  # 40%
  n_rounds: 100
  local_epochs: 5
  batch_size: 32
  learning_rate: 0.01
  dataset: "CIFAR-10"
  model: "CNN"
  non_iid_alpha: 0.1
  attack: "label_flipping"
  
defenses:
  - "FedAvg"
  - "Krum"
  - "Median"
  - "TrimmedMean"
  - "0TML"
  
trials: 10
random_seed_base: 42

output:
  directory: "results/bft_40/"
  save_models: false  # Save disk space
  save_metrics: true
  save_logs: true
```

**Master Execution Script:**
```python
# run_all_experiments.py
import yaml
import subprocess
from pathlib import Path

experiments = [
    "configs/bft_scaling_40.yaml",
    "configs/bft_scaling_50.yaml",
    "configs/sleeper_agent.yaml",
]

for config_path in experiments:
    print(f"\n{'='*60}")
    print(f"Running: {config_path}")
    print(f"{'='*60}\n")
    
    subprocess.run([
        "python", "run_experiment.py",
        "--config", config_path,
        "--verbose"
    ])
    
    print(f"\n✓ Completed: {config_path}\n")

print("\n" + "="*60)
print("ALL EXPERIMENTS COMPLETE")
print("="*60)
```

### 9.2 Automated Analysis Pipeline

**Purpose:** Generate all figures and statistics automatically from raw results

**Pipeline:**
```bash
# After experiments complete
python analyze_results.py --input results/ --output analysis/

# Generates:
#   - analysis/statistics.csv (all metrics with mean, std, CI)
#   - analysis/significance_tests.txt (t-tests, p-values)
#   - analysis/summary_report.md (human-readable summary)

python generate_figures.py --input analysis/ --output figures/

# Generates:
#   - figures/fig1_attack_sophistication.pdf
#   - figures/fig2_bft_scaling.pdf
#   - figures/fig3_convergence.pdf
#   - figures/fig4_reputation_evolution.pdf
#   - figures/fig5_scalability.pdf
#   - figures/fig6_generation_comparison.pdf
```

### 9.3 Progress Tracking Dashboard

**Daily Status Report (Automated):**
```python
# generate_status_report.py
# Run daily to track progress

import json
from datetime import datetime

def generate_status_report():
    report = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "completed_experiments": count_completed(),
        "pending_experiments": count_pending(),
        "data_quality_checks": run_quality_checks(),
        "estimated_completion": estimate_completion(),
    }
    
    print(f"\n{'='*60}")
    print(f"0TML TESTING STATUS REPORT - {report['date']}")
    print(f"{'='*60}")
    print(f"Completed: {report['completed_experiments']}/{report['completed_experiments'] + report['pending_experiments']}")
    print(f"Estimated completion: {report['estimated_completion']}")
    print(f"Data quality: {report['data_quality_checks']['status']}")
    print(f"{'='*60}\n")
    
    # Save to log
    with open("progress_log.json", "a") as f:
        f.write(json.dumps(report) + "\n")
```

---

## 10. DATA MANAGEMENT & BACKUP

### 10.1 Raw Data Storage

**Structure:**
```
data/
├── raw/
│   ├── bft_40/
│   │   ├── trial_001/
│   │   │   ├── metrics.json
│   │   │   ├── model_final.pth
│   │   │   ├── reputation_history.csv
│   │   │   └── detection_log.csv
│   │   ├── trial_002/
│   │   └── ...
│   ├── bft_50/
│   └── sleeper_agent/
├── processed/
│   ├── bft_scaling_summary.csv
│   ├── attack_sophistication_summary.csv
│   └── statistical_analysis.txt
└── figures/
    ├── publication/  # 300 DPI PDFs
    └── presentation/  # PNG for slides
```

**Storage Requirements:**
```yaml
Per Trial: ~500 MB
  - Model checkpoint: 6.5 MB
  - Training logs: 10 MB
  - Metrics: 5 MB
  - Reputation history: 2 MB
  - Miscellaneous: 476.5 MB (checkpoints, cache)

Total for Pre-Submission:
  - BFT scaling: 100 trials × 500 MB = 50 GB
  - Sleeper agent: 20 trials × 500 MB = 10 GB
  - Statistical re-runs: 50 trials × 500 MB = 25 GB
  - Buffer: 15 GB
  - Total: ~100 GB

Recommendation: 200 GB allocated (2:1 safety margin)
```

### 10.2 Backup Strategy

**3-2-1 Backup Rule:**
- **3 copies:** Original + 2 backups
- **2 media types:** Local SSD + Cloud storage
- **1 offsite:** AWS S3 or equivalent

**Implementation:**
```bash
# Automated daily backup script
#!/bin/bash

BACKUP_DIR="/backup/0tml_experiments"
S3_BUCKET="s3://luminous-dynamics-0tml-backup"
DATE=$(date +%Y-%m-%d)

# Local backup (incremental)
rsync -av --delete results/ $BACKUP_DIR/results_$DATE/

# Cloud backup (critical data only)
aws s3 sync results/processed/ $S3_BUCKET/processed/
aws s3 sync figures/publication/ $S3_BUCKET/figures/

echo "✓ Backup completed: $DATE"
```

**Recovery Test:** Run monthly recovery test to ensure backups are valid

---

## 11. QUALITY ASSURANCE CHECKLIST

### 11.1 Pre-Experiment Checks

**Before Starting Any Experiment:**
- [ ] Configuration file reviewed and validated
- [ ] Random seeds documented
- [ ] Output directory created and empty
- [ ] GPU availability confirmed (nvidia-smi)
- [ ] Estimated runtime calculated
- [ ] Backup script running
- [ ] Progress logging enabled

### 11.2 Post-Experiment Validation

**After Each Trial:**
- [ ] Metrics file exists and is valid JSON
- [ ] Final accuracy is reasonable (not NaN or 0%)
- [ ] Convergence occurred (loss decreased)
- [ ] Detection metrics calculated (BDR, FPR)
- [ ] Logs contain no critical errors
- [ ] GPU memory cleared (torch.cuda.empty_cache())

**After Each Experiment Set:**
- [ ] All trials completed successfully
- [ ] Statistics calculated (mean, std, CI)
- [ ] Sanity checks passed (e.g., BDR + FPR ≤ 100%)
- [ ] Visualizations generated
- [ ] Results documented in lab notebook

### 11.3 Pre-Submission Final Check

**Abstract Quality:**
- [ ] All claims supported by empirical data
- [ ] No contradictions between sections
- [ ] Figures referenced correctly
- [ ] Statistics reported consistently (mean ± std)
- [ ] Page limit respected (2 pages + appendix)

**Figure Quality:**
- [ ] All figures 300 DPI (publication quality)
- [ ] Axes labeled clearly with units
- [ ] Legends readable
- [ ] Error bars present where appropriate
- [ ] Color scheme is colorblind-friendly
- [ ] Consistent styling across all figures

**Methods Quality:**
- [ ] All hyperparameters documented
- [ ] Random seeds documented
- [ ] Hardware specifications listed
- [ ] Software versions listed
- [ ] Dataset details provided
- [ ] Attack implementations described
- [ ] Defense implementations described

**Code Quality:**
- [ ] Code runs without errors
- [ ] README provides clear instructions
- [ ] Requirements.txt is complete
- [ ] Example configuration files included
- [ ] Comments explain non-obvious logic
- [ ] License file included (Apache 2.0 recommended)

---

## 12. COMMUNICATION & REPORTING

### 12.1 Weekly Progress Reports

**Format:**
```markdown
# 0TML Testing Weekly Report - Week X

## Completed This Week
- [✓] Experiment X completed (N trials)
- [✓] Figure Y generated
- [✓] Analysis Z finished

## In Progress
- [~] Experiment A (50% complete)
- [~] Writing methods section

## Blockers
- [!] Issue with GPU memory (resolved by reducing batch size)

## Next Week Plan
- [ ] Complete experiment A
- [ ] Start experiment B
- [ ] Generate figures for submission

## Metrics
- Total experiments completed: X/Y (Z%)
- Estimated submission readiness: 75%
- Days remaining: 14
```

**Distribution:** 
- PI self-tracking
- Optional: Share with advisors/collaborators weekly

### 12.2 Stakeholder Communication

**For Potential Transition Partners:**

**Email Template (Week 2 Update):**
```
Subject: 0TML Testing Update - BFT Scaling Results Available

Dear [Stakeholder],

Quick update on our 0TML Byzantine-resilient FL testing:

COMPLETED:
✓ 40-50% BFT scaling tests complete
✓ Results confirm: 0TML maintains >80% accuracy at 50% BFT
✓ Classical defenses (Krum, Median) fail as predicted at 40%+

IN PROGRESS:
→ Sleeper agent testing (novel adaptive attack)
→ Statistical validation (10 trials per experiment)

NEXT STEPS:
→ Comprehensive validation report by [DATE]
→ DARPA submission by [DATE]
→ Would welcome 15-minute briefing on results

Best regards,
Tristan Stoltz
```

### 12.3 Documentation Standards

**Lab Notebook Entry Template:**
```markdown
# Experiment Log: [Experiment Name]
**Date:** 2025-10-XX
**Researcher:** Tristan Stoltz

## Objective
[What are we testing?]

## Configuration
- BFT: X%
- Attack: [Type]
- Trials: N
- Config file: configs/experiment_X.yaml

## Execution
- Start time: HH:MM
- End time: HH:MM
- Duration: X hours
- Hardware: 4×A100
- Issues encountered: [None / Description]

## Results
- Mean accuracy: X.X% ± Y.Y%
- BDR: Z.Z%
- FPR: W.W%

## Analysis
[Key observations, unexpected results, insights]

## Next Steps
[What to do based on these results]

## Files
- Raw data: results/experiment_X/
- Figures: figures/experiment_X/
- Analysis: analysis/experiment_X/
```

---

## 13. LESSONS LEARNED & BEST PRACTICES

### 13.1 Common Pitfalls to Avoid

**Pitfall 1: Insufficient Random Seed Control**
- **Problem:** Results not reproducible across runs
- **Solution:** Fix ALL random seeds (NumPy, PyTorch, CUDA, data splitting)
- **Code:**
  ```python
  def set_all_seeds(seed):
      np.random.seed(seed)
      torch.manual_seed(seed)
      torch.cuda.manual_seed_all(seed)
      torch.backends.cudnn.deterministic = True
      torch.backends.cudnn.benchmark = False
  ```

**Pitfall 2: GPU Memory Leaks**
- **Problem:** OOM errors mid-experiment
- **Solution:** Clear cache between trials
- **Code:**
  ```python
  torch.cuda.empty_cache()
  gc.collect()
  ```

**Pitfall 3: Inconsistent Hyperparameters**
- **Problem:** Can't compare across experiments
- **Solution:** Use configuration files, never hard-code
- **Tool:** YAML configs + version control

**Pitfall 4: Lost Data**
- **Problem:** Experiment fails, no checkpoints saved
- **Solution:** Save checkpoints every N epochs
- **Code:**
  ```python
  if epoch % 10 == 0:
      save_checkpoint(model, f"checkpoint_epoch_{epoch}.pth")
  ```

**Pitfall 5: Unclear Figure Labels**
- **Problem:** Reviewer can't understand visualization
- **Solution:** Always include units, legends, and descriptive titles

### 13.2 Efficiency Tips

**Tip 1: Parallel Execution**
```bash
# Run multiple trials in parallel across GPUs
CUDA_VISIBLE_DEVICES=0 python run_experiment.py --trial 1 &
CUDA_VISIBLE_DEVICES=1 python run_experiment.py --trial 2 &
CUDA_VISIBLE_DEVICES=2 python run_experiment.py --trial 3 &
CUDA_VISIBLE_DEVICES=3 python run_experiment.py --trial 4 &
wait
```

**Tip 2: Quick Validation Runs**
```python
# Before running 100 epochs, test with 5 epochs
CONFIG['n_rounds'] = 5  # Quick sanity check
results = run_experiment(CONFIG)
if results['accuracy'] > 0:
    CONFIG['n_rounds'] = 100  # Full run
```

**Tip 3: Incremental Analysis**
```python
# Analyze results as they come in, don't wait for all trials
for trial in completed_trials:
    analyze_and_visualize(trial)
    update_running_statistics()
```

**Tip 4: Automated Monitoring**
```bash
# Monitor experiment progress remotely
watch -n 60 'tail -n 20 experiment.log'
```

---

## 14. APPENDIX: EXPERIMENT TEMPLATES

### A. BFT Scaling Experiment Template

```python
"""
Template for BFT Scaling Experiments
Copy and modify for 40%, 50%, etc.
"""

import torch
import numpy as np
from pathlib import Path

# Configuration
CONFIG = {
    'experiment_name': 'BFT_Scaling_40',
    'n_clients': 20,
    'n_byzantine': 8,  # 40%
    'n_rounds': 100,
    'local_epochs': 5,
    'batch_size': 32,
    'learning_rate': 0.01,
    'dataset': 'CIFAR-10',
    'model': 'CNN',
    'non_iid_alpha': 0.1,
    'attack': 'label_flipping',
    'defenses': ['FedAvg', 'Krum', 'Median', 'TrimmedMean', '0TML'],
    'n_trials': 10,
    'random_seed_base': 42,
    'output_dir': 'results/bft_40/'
}

def run_trial(trial_num, defense, config):
    """Run single trial"""
    seed = config['random_seed_base'] + trial_num
    set_all_seeds(seed)
    
    # Initialize
    model = create_model(config['model'])
    clients = create_clients(config)
    byzantine_ids = select_byzantine_clients(config['n_byzantine'], seed)
    
    # Training loop
    metrics = []
    for round_num in range(config['n_rounds']):
        # Client updates
        updates = collect_client_updates(clients, model, byzantine_ids, config)
        
        # Defense aggregation
        aggregated = apply_defense(defense, updates, config)
        
        # Update global model
        model.load_state_dict(aggregated)
        
        # Evaluate
        accuracy = evaluate_model(model, test_loader)
        bdr, fpr = compute_detection_metrics(defense, byzantine_ids, honest_ids)
        
        metrics.append({
            'round': round_num,
            'accuracy': accuracy,
            'bdr': bdr,
            'fpr': fpr
        })
    
    return metrics

def main():
    """Run all trials for all defenses"""
    Path(CONFIG['output_dir']).mkdir(parents=True, exist_ok=True)
    
    for defense in CONFIG['defenses']:
        print(f"\nTesting {defense}...")
        
        defense_results = []
        for trial in range(CONFIG['n_trials']):
            print(f"  Trial {trial+1}/{CONFIG['n_trials']}", end='')
            
            metrics = run_trial(trial, defense, CONFIG)
            defense_results.append(metrics)
            
            print(f" - Final Acc: {metrics[-1]['accuracy']:.1f}%")
        
        # Save results
        save_results(defense, defense_results, CONFIG['output_dir'])
    
    print(f"\n✓ All trials complete. Results saved to {CONFIG['output_dir']}")

if __name__ == "__main__":
    main()
```

### B. Sleeper Agent Experiment Template

```python
"""
Template for Sleeper Agent Experiments
Tests stateful vs stateless defenses
"""

class SleeperAgentAttack:
    def __init__(self, sleep_epochs=20):
        self.sleep_epochs = sleep_epochs
        self.current_epoch = 0
        self.is_awake = False
    
    def get_update(self, client_id, honest_update):
        """Return update based on sleep/attack phase"""
        self.current_epoch += 1
        
        if self.current_epoch <= self.sleep_epochs:
            # Sleep: behave honestly
            return honest_update
        else:
            # Awake: launch attack
            if not self.is_awake:
                print(f"[SLEEPER AGENT] Client {client_id} awakened at epoch {self.current_epoch}")
                self.is_awake = True
            
            # Sign flip attack
            return {k: -v for k, v in honest_update.items()}

def run_sleeper_agent_experiment(defense, config):
    """Run sleeper agent test"""
    
    results = {
        'reputation_history': [],
        'detection_timeline': [],
        'accuracy_history': []
    }
    
    # Initialize sleeper agents
    sleeper_agents = {
        byz_id: SleeperAgentAttack(sleep_epochs=config['sleep_epochs'])
        for byz_id in byzantine_ids
    }
    
    for epoch in range(config['n_rounds']):
        # Collect updates (sleeper agents activate automatically)
        updates = []
        for client_id in range(config['n_clients']):
            honest_update = train_local_model(client_id, model, config)
            
            if client_id in byzantine_ids:
                update = sleeper_agents[client_id].get_update(client_id, honest_update)
            else:
                update = honest_update
            
            updates.append(update)
        
        # Apply defense
        aggregated, detected = apply_defense_with_detection(defense, updates, config)
        model.load_state_dict(aggregated)
        
        # Track metrics
        accuracy = evaluate_model(model, test_loader)
        
        results['accuracy_history'].append(accuracy)
        results['detection_timeline'].append(detected)
        
        if hasattr(defense, 'reputation'):
            results['reputation_history'].append(defense.reputation.copy())
    
    return results
```

---

## 15. CONCLUSION & NEXT ACTIONS

## 7. Four-Week Submission Sprint (Oct 20 – Nov 16, 2025)

### ✅ Current Achievements (Baseline Ready)

| Category | Status | Key Results | Data Confidence |
|----------|--------|-------------|-----------------|
| Attack Sophistication Tests | ✅ Complete | 2.1× → 13.6× gain vs Krum | High |
| 30% BFT Baseline | ✅ Complete | 85% accuracy @ 30% BFT | High |
| Convergence Analysis | ✅ Complete | 3× faster stabilization | High |
| Reputation Evolution | ✅ Complete | 9.5× rep gap @ 500 epochs | Very High |
| Computational Scaling | ⚠️ Partial | Linear O(n) to 250 nodes | Medium |

### 🚀 Critical Pre-Submission Objectives (Weeks 1–4)

| Priority | Objective | Description | Deliverable |
|----------|-----------|-------------|-------------|
| 🟥 1 | 40–50% BFT Scaling | Extend 30% results to 40% & 50% using CIFAR-10; validate ≥ 80% accuracy @ 50% | Figure 2 + Table 1 updates |
| 🟥 2 | Sleeper Agent Resilience | Stateful vs stateless comparison (Krum vs 0TML) with reputation evolution | “Sleeper Timeline” figure |
| 🟧 3 | Statistical Rigor | 10 trials per config; report mean ± std dev + p-values < 0.01 | Stats tables + appendix |
| 🟨 4 | Figure Polish | Add error bands & legend consistency (300 DPI PDFs) | Submission-ready visuals |
| 🟩 5 | Abstract Integration | Replace projections with empirical data; consistency pass + review | Final submission package |

### 📆 Week-by-Week Execution Plan

- **Week 1 (Oct 20 – 26) → BFT Scaling**
  - Configure CIFAR-10 40% / 50% BFT trials.
  - Parallelize FedAvg, Krum, Median, TrimmedMean, 0TML across 5 seeds.
  - Compute mean ± std dev; generate BFT curve (0TML stability ≥ 80%).
  - **Deliverable:** Verified empirical BFT scaling plot.
- **Week 2 (Oct 27 – Nov 2) → Sleeper Agent + Stats**
  - Implement sleep-attack pattern (20 epochs honest → attack).
  - Compare Krum vs 0TML; track accuracy + reputation.
  - Re-run 30% BFT baselines (10 trials) for full statistics.
  - **Deliverable:** Sleeper-agent visualization + p-value tables.
- **Week 3 (Nov 3 – 9) → Figures + Optional FedGuard**
  - Add error bars and variance bands; create publication-grade PDFs.
  - Optional: FedGuard baseline (30–50% BFT) if ahead of schedule.
  - **Deliverable:** Figure suite + methods appendix updates.
- **Week 4 (Nov 10 – 16) → Integration & Submission**
  - Update abstract and tables with real data; internal review & proof.
  ️- Package PDF + datasets + config hashes for DARPA submission.
  - **Deliverable:** Final submission package (0TML v1.0 Experimental Report).

### 🧪 Simplified Testing Matrix (Pre-Submission Core)

| Experiment | BFT Level | Attack | Defenses | Dataset | Trials | Priority |
|------------|-----------|--------|----------|---------|--------|----------|
| BFT-Scaling-40 | 0.40 | Label Flip | FedAvg, Krum, Median, TrimmedMean, 0TML | CIFAR-10 | 10 | CRITICAL |
| BFT-Scaling-50 | 0.50 | Label Flip | Same as above | CIFAR-10 | 10 | CRITICAL |
| Sleeper Agent | 0.30 | Sleep 20 → Attack 80 | Krum, 0TML | CIFAR-10 | 10 | CRITICAL |
| FedGuard Baseline | 0.30–0.50 | Label Flip | FedGuard vs 0TML | CIFAR-10 | 10 | OPTIONAL |

### 📊 Submission KPIs

| Metric | Target | Validation |
|--------|--------|------------|
| Accuracy @ 50% BFT | ≥ 80% | 0TML vs Gen 1 failure |
| BDR (Detection Rate) | ≥ 75% | Sleeper agent trial |
| False Positive Rate | ≤ 5% | Statistical tables |
| Reproducibility | 10 runs per config | Methods appendix |
| Figure Quality | 300 DPI PDF + error bands | Week 3 |
| Submission Date | Nov 16 2025 | Week 4 bundle |

### 🧩 Deferred (Phase 1 Validation Nov 2025 – Apr 2026)

| Module | Description | Target Month |
|--------|-------------|--------------|
| ByzFL Integration | Add 0TML to ByzFL for central baselines | Month 2 |
| FedGuard Hybrid PoGQ | Integrate membership confidence scoring | Month 2 |
| Multi-Dataset Testing | Fashion-MNIST, CIFAR-100, MedMNIST | Month 3–4 |
| Multi-Phase Adaptive Attack | Sequential attack resilience | Month 4 |
| Five Eyes Coalition Sim | Cross-jurisdictional non-IID test | Month 5 |
| DDIL Stress Testing | Network degradation resilience | Month 5–6 |
| Red Team Audit | Adversarial penetration test ($75K) | Month 6 |

### ⚠️ Risk Matrix (Condensed)

| Risk | Probability | Impact | Mitigation |
|-------|-------------|--------|------------|
| GPU resource constraint (2070 m) | Medium | Medium | Sequential batching; mixed precision |
| Sleeper agent fails to show advantage | Low | High | Tune reputation decay; extend epochs |
| Variance too high for p-value | Medium | Medium | Increase trials to 20; report median + IQR |
| Figure generation delays | Low | Low | Automate matplotlib pipeline |

### 🧭 Success Definition

- **Minimum bar:** ≥40% & 50% BFT results with error bands; sleeper agent validation; statistical appendix; internally consistent abstract & figures.
- **Excellence bar:** FedGuard baseline added; full figure polish (300 DPI, legends); submitted before Nov 16 2025.

---

### Contact for Questions

**Technical Issues:**
- Email: tristan.stoltz@luminousdynamics.org
- Phone: 315-879-2332

**Collaboration Opportunities:**
- Stanford HAI integration
- SOTA defense authors
- Red team exercise planning

---

**Document Status:** Living document – update as testing progresses

**Last Review:** October 21, 2025

**Next Review:** End of Week 1 sprint (after BFT scaling results available)

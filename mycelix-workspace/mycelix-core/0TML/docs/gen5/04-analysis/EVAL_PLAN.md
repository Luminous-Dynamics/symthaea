# AEGIS Gen-5 — Evaluation Plan

**Date**: November 12, 2025
**Status**: Ready for execution
**Validation ID**: Will be generated on first run

---

## Goals

Quantify:
1. **Detection quality**: Byzantine detection accuracy across adversary rates (0-50%)
2. **Speed/compute tradeoffs**: Active learning query budget vs. accuracy/speedup
3. **Validator overhead**: Secret sharing latency with varying (n,t) configurations
4. **Self-healing performance**: Recovery time from Byzantine surges

---

## Datasets & Partitions

### Primary Datasets
- **CIFAR-10**: RGB images, 10 classes
  - IID partition (uniform distribution)
  - Non-IID Dirichlet partition (α ∈ {0.1, 1.0})

### Secondary Datasets (Future)
- **FEMNIST**: Handwritten characters, non-IID by user

---

## Experiment Matrix (E1–E9)

| Exp | Purpose | Config(s) | Seeds | Runs | Figure |
|-----|---------|-----------|-------|------|--------|
| **E1** | Byzantine tolerance curves | `gen5_eval_{iid,nonIID}_B{15,25}.yaml` | 101–505 | 120 | **F1** |
| **E2** | Sleeper detection (KM survival) | `gen5_temporal_sleeper_{short,med,long}.yaml` | 101–505 | 40 | **F2** |
| **E3** | Coordination detection | `gen5_coordination_{rho0.7,0.9}.yaml` | 101–505 | 24 | **F3** |
| **E4** | L5 speed/accuracy vs budget | `gen5_eval_iid_B{5,10,15,25}.yaml` | 101–505 | 30 | **F4** |
| **E5** | Convergence (FedAvg vs FedMDO) | `gen5_fedmdo_vs_fedavg.yaml` | 101–505 | 30 | **F5** |
| **E6** | Privacy-utility (ε∈{1,4,8,16}) | `gen5_privacy_eps.yaml` | 101–505 | 24 | **F6** |
| **E7** | Validator overhead (n,t grid) | `gen5_validator_n{5,7,9}_t{3,4,5}.yaml` | 101–505 | 12 | **F7** |
| **E8** | Self-healing recovery | `gen5_recovery_{spike,sleeper}.yaml` | 101–505 | 12 | **F8** |
| **E9** | Secret sharing BFT limits | `gen5_validator_byz.yaml` | 101–505 | 8 | **F9** |

**Total**: 300 runs across 9 experiment types

---

## Detailed Experiment Specifications

### E1: Byzantine Tolerance Curves
**Purpose**: Measure TPR, FPR, F1 across increasing adversary rates

**Parameters**:
- Adversary rates: 0%, 10%, 20%, 30%, 40%, 50%
- Attack types: sign_flip, scaling, gaussian_noise, backdoor
- Seeds: 101, 202, 303, 404, 505

**Expected Results**:
- TPR > 95% @ 0% Byzantine (baseline)
- TPR > 85% @ 30% Byzantine (above classical limit)
- TPR > 75% @ 45% Byzantine (AEGIS target) ✨
- TPR > 60% @ 50% Byzantine (graceful degradation)

**Figure F1**: 3-panel plot (TPR, FPR, F1) vs. Byzantine fraction

---

### E2: Sleeper Agent Detection
**Purpose**: Measure time-to-detection for dormant-then-active attackers

**Parameters**:
- Activation rounds: 30 (short), 50 (medium), 75 (long)
- Stealth levels: 0.0 (blatant), 0.5 (stealthy)
- Detection method: CUSUM with fixed baseline

**Expected Results**:
- Low stealth: Detection within 5-10 rounds
- High stealth: Detection within 15-20 rounds
- Detection rate: > 90% within 25 rounds

**Figure F2**: Kaplan-Meier survival curves (fraction undetected vs. rounds)

---

### E3: Coordination Detection
**Purpose**: Detect synchronized attacks via cross-correlation

**Parameters**:
- Coordinated agents: 5, 10
- Correlation strength (ρ): 0.7, 0.9
- Background honest: 15-20 agents

**Expected Results**:
- TPR ≥ 85% @ ρ=0.9 (strong coordination)
- TPR ≥ 70% @ ρ=0.7 (moderate coordination)
- FPR < 10% (few false accusations)

**Figure F3**: Bar chart of coordination TPR by correlation strength

---

### E4: Active Learning Label Efficiency
**Purpose**: Validate 6-10× speedup claim with minimal accuracy loss

**Parameters**:
- Query strategies: uncertainty, margin, diversity
- Budget fractions: 0.05, 0.10, 0.15, 0.25
- Full detection baseline: 1000 gradients

**Expected Results**:
- Speedup ≥ 6-10× @ budget 0.15-0.25
- Accuracy loss < 1% compared to full detection
- Uncertainty + diversity strategy performs best

**Figure F4**: 2-panel (Accuracy + Speedup vs. Budget)

---

### E5: Federated Convergence
**Purpose**: Compare FedAvg vs. FedMDO convergence speed

**Parameters**:
- Optimizers: FedAvg, FedMDO
- Byzantine fraction: 25%
- Rounds: 50

**Expected Results**:
- FedMDO: 45-55% loss reduction in 50 rounds
- FedAvg: 40-50% loss reduction
- FedMDO converges 10-15% faster

**Figure F5**: Scatter plot (Convergence round vs. Final loss)

---

### E6: Privacy-Utility Tradeoff
**Purpose**: Quantify DP noise impact on accuracy

**Parameters**:
- Epsilon values: 1.0, 4.0, 8.0, 16.0
- Delta: 0.0 (pure DP)
- Byzantine fraction: 25%

**Expected Results**:
- Accuracy @ ε=1.0: ≥ 85% (strong privacy)
- Accuracy @ ε=8.0: ≥ 92% (moderate privacy)
- Accuracy @ ε=16.0: ≥ 95% (weak privacy)

**Figure F6**: Pareto frontier (Privacy budget ε vs. Accuracy) with zones

---

### E7: Distributed Validation Overhead
**Purpose**: Measure secret sharing latency

**Parameters**:
- Validators (n): 5, 7, 9
- Threshold (t): 3, 4, 5
- Byzantine fraction: 14% (1 of 7)

**Expected Results**:
- Share generation: < 5ms
- Reconstruction: < 10ms
- Total overhead: < 20ms @ (n=7, t=4)

**Figure F7**: Grouped bar chart (Latency components by n_validators)

---

### E8: Self-Healing Recovery
**Purpose**: Measure Mean Time To Recovery (MTTR) from Byzantine surges

**Parameters**:
- Attack types: poison_spike, sleeper_agent
- Surge magnitude: 60% Byzantine
- Healing threshold: 40%

**Expected Results**:
- MTTR (poison spike): 8-12 rounds
- MTTR (sleeper agent): 15-20 rounds
- Recovery success: > 95%

**Figure F8**: Violin plots of MTTR distribution by attack type

---

### E9: Secret Sharing Byzantine Tolerance
**Purpose**: Verify BFT limit for distributed validators

**Parameters**:
- Byzantine validators: 0, 1, 2, 3
- Configuration: (n=7, t=4)
- BFT limit: n - t = 3 validators

**Expected Results**:
- Success @ 0-2 Byzantine: > 95%
- Success @ 3 Byzantine: < 20% (BFT limit exceeded correctly)

**Figure F9**: Bar chart (Reconstruction success rate vs. Byzantine count)

---

## Statistics & Methodology

### Reproducibility
- **Manifest**: {git commit, state hash, seeds, environment} committed with results
- **Drift detection**: State hash verified every 5 runs; abort on mismatch
- **Checkpointing**: Save progress every 5 runs for interruption recovery

### Statistical Rigor
- **Bootstrap CI**: 10,000 resamples for robust uncertainty quantification
- **Cliff's Delta**: Nonparametric effect size measure
- **Kaplan-Meier**: Survival analysis for sleeper detection (E2)
- **Fixed seeds**: [101, 202, 303, 404, 505] for cross-run comparability

### Anti-P-Hacking
- **Pre-registered matrix**: All 300 configurations defined before execution
- **No cherry-picking**: Results include all runs; failures preserved
- **Multiple comparison**: Benjamini-Hochberg correction (q=0.1) when needed

---

## Acceptance Criteria

### Layer 5: Active Learning Inspector
✅ **Speedup**: ≥ 6-10× @ budget 0.15-0.25
✅ **Accuracy**: < 1% loss compared to full detection
✅ **Strategy**: Uncertainty + diversity outperforms margin

### Layer 6: Multi-Round Temporal Detection
✅ **Sleeper detection**: ≤ 20 rounds to detection
✅ **Coordination detection**: TPR ≥ 85% @ ρ=0.9
✅ **False positive rate**: < 10%

### Layer 4: Federated Validator
✅ **Overhead**: < 20ms @ (n=7, t=4)
✅ **BFT tolerance**: 1 of 7 validators can be Byzantine
✅ **Reconstruction**: > 95% success when within BFT limit

### Layer 7: Self-Healing Mechanism
✅ **MTTR**: < 20 rounds for Byzantine surges
✅ **Recovery success**: > 95% from 60% Byzantine spikes
✅ **Stability**: No oscillation after recovery

### Overall System
✅ **Byzantine tolerance**: TPR > 75% @ 45% adversary rate
✅ **Perfect test coverage**: 147/147 tests passing
✅ **Convergence**: FedMDO 10-15% faster than FedAvg

---

## Reproducibility Guarantees

### Manifest Structure
```json
{
  "validation_id": "AEGIS-{mode}-{timestamp}",
  "git_commit": "09762061...",
  "state_hash": "8a3f9c2d...",
  "timestamp": "2025-11-12T23:59:00",
  "python_version": "3.13.5",
  "numpy_version": "2.3.4",
  "seeds": [101, 202, 303, 404, 505],
  "experiment_matrix": { ... },
  "mode": "dry-run | full",
  "total_runs": 10 | 300
}
```

### Figure Captions (Template)
```
Figure F1: Byzantine tolerance curves for AEGIS Gen-5 across adversary rates
(0-50%) and attack types. Each point represents mean ± std across 5 seeds
(101-505). Classical BFT limit (33%) and AEGIS target (45%) shown as reference
lines. Configuration: gen5_eval_nonIID_alpha1_B20.yaml. Validation ID:
AEGIS-full-2025-11-13-080000. Manifest hash: 8a3f9c2d.
```

### Reference in Papers
```latex
\textit{All experiments used fixed seeds [101, 202, 303, 404, 505] and were
validated against an immutable manifest (Git commit: 09762061, State hash:
8a3f9c2d). Complete experimental configurations and raw results are available
at [repository URL].}
```

---

## Execution Timeline

### Dry-Run (~30 minutes)
- **Runs**: 10 experiments
- **Purpose**: Smoke test infrastructure
- **Configs**: E1 @ {0%, 30%, 50%}, E5 both optimizers, E8 poison spike

### Full Validation (~8 hours)
- **Runs**: 300 experiments
- **Purpose**: Generate publication results
- **Output**: results_complete.json + summary.txt

### Figure Generation (~5 minutes)
- **Input**: results_complete.json
- **Output**: F1-F9 (PDF + PNG, 300 DPI)

---

## Paper Integration Plan

### Methods Section (L1-L7)
One paragraph per layer with references to:
- Class diagrams (design docs)
- Algorithm pseudocode
- Theoretical properties (convergence, privacy, BFT)

**Architecture Figure**: Detection → Validation → Recovery pipeline showing all 7 layers

### Validation Framework Section
- Manifest structure and immutability guarantees
- Bootstrap CI / Cliff's Delta / Kaplan-Meier methodology
- Drift detection and checkpointing
- Anti-p-hacking guardrails

### Results Section
- **Section 4.1**: Byzantine Tolerance (F1)
- **Section 4.2**: Temporal Detection (F2, F3)
- **Section 4.3**: Active Learning Efficiency (F4)
- **Section 4.4**: Convergence & Privacy (F5, F6)
- **Section 4.5**: Distributed Validation (F7)
- **Section 4.6**: Self-Healing Performance (F8, F9)

Each section:
- References exact config + manifest ID
- Includes acceptance criteria validation
- Discusses practical implications

### Discussion Section
- **ZTKN Sidebar**: Optional Layer 8 (verifiable detection)
- Deployment pathways (healthcare, finance, federated AI)
- Limitations and future work

---

## Quality Assurance Checklist

### Pre-Flight (Before Launch)
- [ ] `nix develop` opens successfully
- [ ] `python -c "import numpy; print(numpy.__version__)"` works
- [ ] `validation_results/` cleared or archived
- [ ] `figures/` cleared
- [ ] Manifest seeds [101, 202, 303, 404, 505] committed to repo
- [ ] Environment variables set: `OMP_NUM_THREADS=2`, `PYTHONHASHSEED=0`

### Post-Run (After Completion)
- [ ] Manifest triplet {git_commit, state_hash, seeds} identical in all files
- [ ] N=300 coverage: Each E1-E9 subfolder has expected run counts
- [ ] Figures sanity:
  - [ ] F1: TPR degrades gracefully with Byzantine rate
  - [ ] F4: Speedup ≥ 6-10× @ budget 0.15-0.25
  - [ ] F7: Validator overhead < 20ms @ (n=7, t=4)
  - [ ] F8: MTTR < 20 rounds
- [ ] No drift detected during run (state hash stable)
- [ ] Checkpoints saved every 5 runs
- [ ] results_complete.json contains all 300 runs

---

**Status**: Ready for execution
**Next**: Run dry-run validation to verify infrastructure
**Target**: MLSys / ICML 2026 submission (January 15, 2026)

🎯 **Evaluation plan locked - ready to generate publication-quality results!** 🎯

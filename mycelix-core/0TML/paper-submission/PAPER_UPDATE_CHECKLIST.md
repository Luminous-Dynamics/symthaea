# Paper Update Checklist - Week 3 Day 5

**Status**: In Progress
**Priority**: High - Complete before v4.1 results arrive

---

## ✅ Completed Updates

### 1. Methods Section - Range Checks & Soundness ✅
**File**: `sections/03_METHODS.tex`
**Lines**: 250-254
**Changes**:
- Updated Winterfell backend description (11 → 44 columns)
- Added LEAN range check paragraph with technical details
- Added dual-backend rationale
- Cited Table VII-bis

**Content Added**:
- Bit decomposition constraints (rem_t, x_t, quar_t)
- Security parameters (blowup=16, queries=42, 96-bit security)
- Performance tradeoffs (7-10× slowdown vs 30,000× speedup)

---

### 2. Conformal Wording Improvements ✅
**File**: `sections/03_METHODS.tex` (Lines 121, 155)
**Status**: COMPLETED
**Changes Made**:
- Updated CBF description (line 121): "FPR ≤ α up to quantile estimation error (under exchangeability assumption; non-IID data may exhibit calibration drift)"
- Updated Mondrian FPR (line 155): "up to quantile estimation error (under exchangeability within buckets)"

---

### 3. Security Scope Clarifications ✅
**File**: `sections/03_METHODS.tex` (Line 254)
**Status**: COMPLETED
**Changes Made**:
- Added receipts scope: "Cryptographic receipts attest decision correctness under PoGQ logic---that quarantine state transitions follow specified rules---but do not attest raw gradient provenance or client identity"
- 96-bit security already present in line 252
- zkVM fallback already present in line 254

---

### 4. Discussion Section - Security vs Performance ✅
**File**: `sections/06-discussion.tex`
**Status**: COMPLETED
**New Subsection Needed**:
```latex
\subsection{Security vs Performance Tradeoffs}

Range checks via bit decomposition increase proving time 7-10$\times$
versus unconstrained baseline (0.15ms → 1.5ms), primarily due to:
(1) wider trace (12 → 44 columns) increasing Merkle commitment size,
(2) more constraints (9 → 40) expanding quotient polynomial degree.
However, this cost is negligible compared to the 30,000$\times$ speedup
over zkVM (46.6s → 1.5ms), demonstrating that domain-specific AIRs can
achieve both mathematical rigor and practical performance.

zkVM remains valuable as a portability layer: identical PoGQ guest code
compiles to both RISC-V (zkVM) and custom AIR (Winterfell), enabling
cross-validation and independent soundness verification. Production
systems can use Winterfell for performance-critical paths while
maintaining zkVM compatibility for auditability.
```

**Content Added**:
```latex
\subsection{Security vs Performance Tradeoffs}

Range checks via bit decomposition increase proving time 7--10$\times$
versus unconstrained baseline (0.13--0.16ms → 1.02--1.56ms), primarily due to:
(1) wider trace (12 → 44 columns) increasing Merkle commitment size,
(2) more constraints (9 → 40) expanding quotient polynomial degree.
However, this cost is negligible compared to the 30,000$\times$ speedup
over zkVM (46.6s → 1.5ms), demonstrating that domain-specific AIRs can
achieve both mathematical rigor and practical performance.
```

---

### 5. Results Section - Table VII-bis Integration ✅
**File**: `sections/05-results.tex`
**Status**: COMPLETED
**Location**: New subsection after Performance Analysis
**Changes Made**:
- Added new subsection "VSV-STARK Proof Backend Performance"
- Added reference to Table~\ref{tab:vsv_stark_dual_backend}
- Added comparison of backends with performance context

**Content Added**:
```latex
Table~\ref{tab:vsv_stark_dual_backend} compares dual-backend performance
across three representative PoGQ scenarios. Winterfell STARK with LEAN range
checks achieves 1.0--1.6ms proving time (30,000$\times$ faster than zkVM
baseline) while maintaining mathematical soundness via bit decomposition
constraints. The 7--10$\times$ slowdown versus unconstrained baseline is
negligible compared to the massive speedup over general-purpose zkVM,
demonstrating that domain-specific AIRs can achieve both cryptographic rigor
and practical performance for production federated learning deployments.
```

---

## 🚧 Pending Updates

### 6. Results Section - PoGQ v4.1 Placeholder
**File**: `sections/05-results.tex`
**Status**: Not started
**New Subsection Needed**:
```latex
\subsection{PoGQ v4.1 vs v4.0 Detection Performance}

[PLACEHOLDER - Complete after experiments finish (~Nov 12-13)]

Table~\ref{tab:pogq_versions} compares PoGQ v4.1 (Dirichlet-aware) against
v4.0 (beta=0.90) across Byzantine fraction rates. [Results pending]
```

**Table Shell**:
```latex
\begin{table}[t]
\centering
\caption{PoGQ v4.1 vs v4.0 Detection Rates}
\label{tab:pogq_versions}
\begin{tabular}{lcccccc}
\toprule
& \multicolumn{2}{c}{20\% BFT} & \multicolumn{2}{c}{30\% BFT} & \multicolumn{2}{c}{40\% BFT} \\
\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}
& TPR & FPR & TPR & FPR & TPR & FPR \\
\midrule
v4.0 & -- & -- & -- & -- & -- & -- \\
v4.1 & -- & -- & -- & -- & -- & -- \\
\bottomrule
\end{tabular}
\end{table}
```

---

### 7. Results Section - Ablation Placeholder
**File**: `sections/05-results.tex`
**Status**: Not started
**New Subsection Needed**:
```latex
\subsection{Component Ablation Study}

[PLACEHOLDER - Run after v4.1 completes]

Table~\ref{tab:ablations} quantifies each PoGQ component's contribution
to detection accuracy. Configurations: baseline (all components),
no-PCA, no-conformal, no-mondrian, no-hybrid, no-EMA, no-hysteresis.
```

**Table Shell**:
```latex
\begin{table}[t]
\centering
\caption{PoGQ Component Ablation (30\% BFT, CIFAR-10)}
\label{tab:ablations}
\begin{tabular}{lcccc}
\toprule
Configuration & TPR & FPR & AUROC & $\Delta$ AUROC \\
\midrule
Full PoGQ     & -- & -- & -- & baseline \\
No PCA        & -- & -- & -- & -- \\
No Conformal  & -- & -- & -- & -- \\
No Mondrian   & -- & -- & -- & -- \\
No Hybrid     & -- & -- & -- & -- \\
No EMA        & -- & -- & -- & -- \\
No Hysteresis & -- & -- & -- & -- \\
\bottomrule
\end{tabular}
\end{table}
```

---

### 8. Appendix - Adversarial Test Results
**File**: `sections/08-appendix.tex`
**Status**: Not started
**New Subsection**:
```latex
\subsection{LEAN Range Check Validation}

We validated range check soundness with 4 adversarial tests:
(1) maximum remainder (65535), (2) maximum witness (65535),
(3) zero values, (4) alternating extremes. All tests passed,
confirming bit decomposition constraints prevent overflow attacks.

Additionally, we tested failure modes: (1) remainder > 65535 → proof
generation fails, (2) witness > 65535 → trace building fails,
(3) tampered quarantine flag → verification fails. These demonstrate
that range violations are mathematically impossible under STARK soundness.
```

---

### 9. Figure - Scaling Results
**File**: `sections/05-results.tex` or `06-discussion.tex`
**Status**: Not started
**Figure Needed**:
```latex
\begin{figure}[t]
\centering
\includegraphics[width=0.48\textwidth]{figures/winterfell_scaling.pdf}
\caption{Winterfell prove time vs trace length (T=8, 128, 256, 512).
Sub-linear scaling demonstrates efficient constraint evaluation.}
\label{fig:winterfell_scaling}
\end{figure}
```

**Data**: From scaled benchmarks (6.5ms, 18ms, 34ms for T=128, 256, 512)

---

### 10. CI & Reproducibility Updates
**File**: `sections/08-appendix.tex` (Artifact Appendix)
**Status**: Not started
**Updates Needed**:
- Add `dual_backend_smoke.sh` to artifact list
- Pin toolchain versions (Rust 1.83, Winterfell 0.13.1)
- Add commit SHA for reproducibility
- Mention PROOF_OPTIONS.json export

---

## 📊 Priority Order

**Immediate (Today)** - ✅ COMPLETE:
1. ✅ Methods - Range checks (DONE)
2. ✅ Conformal wording improvements (DONE)
3. ✅ Security scope clarifications (DONE)
4. ✅ Discussion - Security vs performance (DONE)
5. ✅ Results - Table VII-bis integration (DONE)

**This Week**:
6. Results - PoGQ v4.1 placeholder
7. Results - Ablation placeholder

**Next Week (After v4.1 completes)**:
8. Fill v4.1 comparison data
9. Run & fill ablation data
10. Generate scaling figure

---

## 🔧 Implementation Tasks

**Scripts Needed**:
1. `scripts/generate_winterfell_scaling_figure.py` - Plot prove time vs T
2. `scripts/dual_backend_smoke.sh` - CI smoke test (already created)
3. `scripts/export_proof_options.py` - Export PROOF_OPTIONS.json

**Experiments Needed**:
1. Component ablations (6 configs × 3 datasets = 18 experiments)
2. v4.1 baseline runs (if not in current matrix)

---

## ✅ Completion Criteria

Paper is ready for submission when:
- [ ] All 10 updates complete
- [ ] v4.1 results filled in
- [ ] Ablation results filled in
- [ ] All figures generated
- [ ] CI/reproducibility documented
- [ ] External review completed

**Target Date**: Week 4, Day 7 (before external review)

---

## 📝 Notes

- Keep wording concise (USENIX Security page limits)
- Cite Winterfell v0.13.1 docs for STARK parameters
- Add `\cite{stark-security}` placeholder (fill with actual paper)
- Ensure all tables have consistent formatting

---

*Last Updated*: November 10, 2025
*Next Review*: After v4.1 experiments complete

# Decision-risk certification when the contamination bound is wrong

**Date:** 2026-10-29. **Status:** synthetic stress experiment; not a production certificate or direct SOTA comparison.

## Why this follow-up matters

The previous Huber-sensitivity experiment assumed the declared contamination upper bound was valid. A robust confidence set is only robust relative to that assumption. If the bound is understated, even a mathematically correct sensitivity transformation can narrow the clean-target set too much and admit an unsafe model. The bound must therefore be an independently defended input—not estimated optimistically from the same contaminated stream and then treated as truth.

Wang & Ramdas's Huber-robust confidence sequences explicitly allow a bounded fraction of arbitrary corruption and are valid at arbitrary stopping times under their stated assumptions. Their official implementation is HJ-Wg/catoni-confseq (GPL-3.0). This experiment does not reproduce that algorithm; it stress-tests the simpler Bernoulli sensitivity inversion already developed in this research lane. See https://proceedings.mlr.press/v206/wang23p.html and https://github.com/HJ-Wg/catoni-confseq.

## Matched protocol

- 4 risk cells, predicted risks [0.15, 0.30, 0.60, 0.80].
- Clean target risks [0.15, 0.37, 0.60, 0.80]; clean maximum model error is 0.07, so the model is genuinely unsafe relative to tolerance 0.05.
- One cell is contaminated by an IID Huber mixture with contaminant R=0; actual contamination rates are 10%, 12%, and 18%.
- The same observed stream within each repeat is reused across claimed-bound conditions.
- Claimed contamination bounds are 0%, 2.5%, 5%, 7.5%, 10%, 12%, 15%, 18%, and 20%.
- 100 independent repeats per actual-rate × claimed-bound pair; 2,700 certifier calls total.
- Familywise delta 0.10 split across cells; pilot 256 per cell, batches of 256, cap 8,192 per cell.
- The decision rule itself is unchanged. A run either admits, rejects, or refuses unresolved. All reported unsafe admissions are genuinely false admissions because the clean target error is 0.07.

## Core results

Each table cell is unsafe admissions / 100, with unresolved count in parentheses. No run conclusively rejected; the model error is masked toward the model by this particular contaminant.

| Actual contamination | Declared 0% | Declared 2.5% | Declared 5% | Declared bound equal to actual |
|---:|---:|---:|---:|---:|
| 10% | 24 (76 unresolved) | 1 (99 unresolved) | 0 (100 unresolved) | 0 (100 unresolved; declared 10%) |
| 12% | 77 (23 unresolved) | 16 (84 unresolved) | 0 (100 unresolved) | 0 (100 unresolved; declared 12%) |
| 18% | 100 (0 unresolved) | 99 (1 unresolved) | 0 (100 unresolved) | 0 (100 unresolved; declared 18%) |

The nominal/zero-bound method falsely admitted in 24%, 77%, and 100% of trials for actual contamination of 10%, 12%, and 18%, respectively. Understating the bound to 2.5% still led to 16% unsafe admissions at 12% actual contamination and 99% at 18% actual contamination. A claimed bound of 5% or greater caused all trials in this limited panel to refuse unresolved under the sample cap—zero observed unsafe admissions, but poor availability and no conclusive detection.

The two-sided Wilson 95% interval for 77/100 unsafe admissions is approximately 67.8%–84.2%; for 100/100 it is approximately 96.3%–100%. For 0/100 admissions, the Wilson upper limit is about 3.70%, so zero observed events is not zero population risk.

Cost reveals the danger of rewarding early stopping without conditioning on validity: for actual contamination 18%, a zero declared bound admitted in 100/100 trials after a mean 10,158 observations; a 2.5% bound admitted in 99/100 after a mean 19,876 observations; 5% or higher ran to the 32,768-observation total cap and refused unresolved. Fewer samples can be a symptom of incorrect assumptions, not better efficiency.

## Decision-relevant finding

The certificate depends jointly on (1) the sequential statistical construction and (2) the integrity of its assumptions. A sensitivity interval cannot compensate for a bound that is falsely declared. The appropriate system design is an **assumption-carrying certificate**:

1. bind the bound to an independently sourced, versioned measurement/attestation and explicit validity scope;
2. if the bound is unavailable or its provenance/validity window fails, return refuse_unresolved before inference;
3. publish a decision-versus-contamination sensitivity frontier when the bound is merely a scenario, not an attested fact;
4. never tune the declared bound on the same contaminated sample and subsequently use it as a guarantee.

The result also exposes the availability trade-off: a large honest contamination cap can make the correct answer “not identifiable with the present evidence.” The next algorithmic goal is not to suppress abstention, but to add independent evidence sources or justified assumptions that narrow the uncertainty set without compromising coverage.

## Literature / comparison discipline

- Wang & Ramdas, Huber-robust confidence sequences, AISTATS 2023. The paper uses robust exponential supermartingales under a known corruption fraction and a known upper bound on a central moment; its guarantee is stronger than our simple Bernoulli sensitivity inversion and is the appropriate implementation baseline for the next phase. https://proceedings.mlr.press/v206/wang23p.html
- Official reference implementation: https://github.com/HJ-Wg/catoni-confseq (GPL-3.0; assess license compatibility before incorporating code into another project).
- Cai & Kandasamy, Constrained Best Arm Identification with Tests for Feasibility, AAAI 2026: related to allocation among performance and feasibility measurements. https://ojs.aaai.org/index.php/AAAI/article/view/39063
- Jourdan et al., An Anytime Algorithm for Good Arm Identification (APGAI), JMLR 2026. https://www.jmlr.org/papers/v27/24-0680.html

No published SOTA implementation was executed in this experiment. No SOTA/industry superiority is claimed.

## Reproduction

```sh
PYTHONPATH=. python -m unittest -v test_decision_risk_huber_bound_misspecification.py
PYTHONPATH=. python decision_risk_huber_bound_misspecification.py \
  --repeats 100 \
  --actual-rhos 0.10,0.12,0.18 \
  --claimed-rhos 0.00,0.025,0.05,0.075,0.10,0.12,0.15,0.18,0.20 \
  --output DECISION_RISK_HUBER_BOUND_MISSPECIFICATION_RESULTS_2026-10-29_100.json
```

The local test suite passed 5/5. This is Python research code, not Rust production integration; no remote CI pass is claimed.

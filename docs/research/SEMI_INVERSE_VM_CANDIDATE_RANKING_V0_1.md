# SEMI inverse-VM feasible top-K candidate ranking v0.1

**Issue:** #13619  
**Implementation:** `benchmark::inverse_vm_candidate_ranking`  
**Status:** Rust candidate feasibility/ranking primitive with authored tests; exact-head validation pending.

## Research comparator

Lin et al., *Feasibility-Aware Inverse Virtual Metrology for Top-K Recipe Recommendation in Semiconductor Process Correction*, IEEE Transactions on Automation Science and Engineering, 2026-07-20, DOI 10.1109/TASE.2026.3715101.

The public abstract reports a production-scale single-process EPI dataset with 2,140 historical correction events. On its time-based split, the paper reports:
- Success@5 = 0.742;
- Feasible@5 = 0.946;
- strongest overall baseline: Success@5 = 0.647 and Feasible@5 = 0.901.

These are task- and dataset-specific offline results. They are not a cross-process SOTA constant, and our implementation has not reproduced them.

## Paradigm shift

A VM point estimate is not a corrective action. The new unit of output is a **feasibility-qualified candidate** tied to:
- a declared recipe/control schema;
- a frozen forward-model artifact and prediction receipt;
- uncertainty on target and side-constraint predictions;
- a historical-support manifest and per-candidate support receipt;
- robust target-attainment and side-constraint tests;
- deterministic diversity-aware ranking.

The module ranks supplied candidate vectors; candidate generation and the full forward model are external inputs in v0.1. It emits a human-review shortlist and never writes process settings.

## Robust feasibility rules

For target (j), a candidate is admitted only if the entire declared uncertainty interval fits inside the requested target tolerance:

[
|\hat y_j-y_j^*|+z\sigma_j\leq \epsilon_j.
]

For a side constraint with allowed interval ([L_k,U_k]), require:

[
\hat c_k-z\sigma_k\geq L_k,
\qquad
\hat c_k+z\sigma_k\leq U_k.
]

Candidate controls must remain inside every declared recipe-control bound. Historical support distance must lie within the declared support radius and meet the minimum support count.

A candidate with current but out-of-domain support is **UNSUPPORTED**. Insufficient support or unavailable/unknown model qualification is **UNKNOWN**. A robust target/constraint failure is **INFEASIBLE**. Malformed/tampered identity or schema is **REJECTED**. These states are not collapsed into one score.

## Ranking

Only candidates that pass all hard gates are ranked. The base score combines mean robust target-risk ratio, historical-support ratio and a side-constraint margin penalty. A deterministic greedy selection then applies a capped diversity bonus based on normalized Euclidean distance in recipe-control space. The report preserves:
- robust target risk;
- support-distance ratio;
- minimum normalized side-constraint margin;
- base score;
- diversity bonus;
- final selection score and rank;
- every candidate rejection reason.

The requested top-K may be only partially filled; the report states that explicitly. It does not invent out-of-support candidates to fill K.

## Evidence contract

Each candidate binds:
- recipe schema digest;
- model artifact digest;
- forward prediction receipt digest;
- model-evidence receipt digest/status and its availability frontier;
- historical-support manifest and candidate support receipt;
- control values and every target/constraint estimate including standard uncertainty.

The input and output are digest-bound and canonicalized across candidate/target/control ordering. Digests establish content identity, not source authenticity or a physically valid model. The report explicitly sets source authenticity, physical success, recipe write, APC/R2R and equipment authority to false.

## Authored known-answer tests

Ten tests cover:
1. robust target-interval and side-constraint rejection;
2. unsupported and unknown evidence never entering top-K;
3. insufficient historical support returning UNKNOWN;
4. model evidence unavailable at the knowledge cutoff returning UNKNOWN;
5. control value outside recipe bounds returning INFEASIBLE;
6. candidate payload tampering returning REJECTED;
7. diversity changing top-K without changing feasibility;
8. candidate/input order-invariant ranking receipt;
9. partial top-K fill and human-review-only authority flags;
10. disabling diversity permits a zero diversity-distance threshold without NaN ranking scores.

Tests are authored; exact-head rustfmt, strict Clippy, crate tests and SEMI-BENCH-001 reference validation remain pending.

## Benchmark plan

Reproduce the paper on the same licensed EPI data, time split and independent evaluation stack before any parity claim. Then compare:
- inverse candidate generation + historical support;
- forward-only ranking;
- no support gate;
- no uncertainty/robust constraints;
- no diversity;
- evidence-qualified model/label/calibration gate;
- causal-drift-conditioned candidate or discriminative-metrology selection.

Report Success@1/3/5/10, Feasible@1/3/5/10, unsupported-action rate, side-constraint violation rate, candidate diversity, partial top-K rate, and time-split/worst-period results. Compare equal qualified-label budgets and publish paired bootstrap confidence intervals.

Proposed targets after faithful reproduction (not achieved): Success@5 >= 0.76, Feasible@5 >= 0.97, zero out-of-support items in surfaced top-K, 100% rejection/UNKNOWN for invalid or unavailable evidence, and deterministic replay.

## Claim ceiling

This is offline candidate ranking under declared model predictions, uncertainties, constraints and historical support. It does not prove physical success, demonstrate cross-fab generalization, execute a recipe, write APC/R2R settings, or authorize process control. Human process engineering and independent physical validation remain required.

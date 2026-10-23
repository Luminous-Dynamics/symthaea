# VIS-DVPC-009: Safety- and Tail-Constrained Exact Oracle v1

**Status:** Draft evaluator-only reference and synthetic adversarial tests.  
**Issue:** [#8541](https://github.com/Luminous-Dynamics/symthaea/issues/8541)  
**Existing utility/resource scorer:** [PR #8506](https://github.com/Luminous-Dynamics/symthaea/pull/8506)  
**Online vector budgets:** [PR #8518](https://github.com/Luminous-Dynamics/symthaea/pull/8518)  
**Prior art:** [DEEP (2026)](https://arxiv.org/abs/2609.35910); [Conformal Policy Control, ICML 2026](https://proceedings.mlr.press/v306/prinster26a.html)

## Research question

The existing paired-trace oracle maximizes group-balanced utility subject to total latency and energy caps. It is correctly described as a *utility-only upper reference*, but it may choose an action that increases safety violations or tail decision loss. Reporting safety/CVaR next to the oracle does not make the oracle itself risk-constrained.

This tool creates a second, distinct reference:

> What is the highest equal-group-weighted utility achievable on these post-commit paired outcomes while all declared resource, safety-count, and tail-loss caps are simultaneously satisfied?

DEEP motivates decision-level evaluation under resource budgets and warns that higher-fidelity perception can worsen downstream decisions. Conformal Policy Control studies a different but complementary idea: regulating an optimized policy against a safe reference with finite-sample risk control. This implementation reproduces neither method; it supplies an exact small-instance evaluator to clarify the frontier on Symthaea's paired outcomes.

## Exact selector

Source: `tools/vision-dvpc-safety-tail-oracle.rs`

For each trace the evaluator provides cheap/full utilities, cheap/full safety-violation counts, source-group ID, and incremental full-mode latency/energy. The selector enumerates every cheap/full assignment up to the fixed cap of 16 traces (65,536 assignments).

Among assignments that satisfy every cap, it maximizes equal-group-weighted utility. It enforces independently:
- total incremental latency;
- total incremental energy;
- total chosen-action safety-violation count;
- CVaR95 of per-trace incremental decision loss relative to cheap mode.

For a full-mode choice, the per-trace incremental loss is `cheap_utility - full_utility`; for cheap mode it is zero. CVaR95 is computed as the mean of the worst `ceil(n / 20)` such losses. Positive values indicate harm. The selector does not trade a violation of any hard cap against utility.

Deterministic tie-breaking prioritizes lower CVaR95, fewer safety violations, lower resource use, fewer escalations, and then the lexicographically smaller chosen trace-ID list.

### Exactness and infeasibility

The exact search is bounded at 16 traces to keep runtime and auditability predictable. More than 16 traces yields `InstanceTooLarge`. The tool never silently switches to a heuristic and labels its result exact. If no assignment satisfies all caps, it returns `NoFeasiblePolicy`; it does not relax the caps or invent a score.

## Claim boundaries

This is **post-commit evaluator truth**. The function consumes paired counterfactual outcomes and cannot be used as a deployable online selector. Its output is an upper reference for the finite observed trace set, not evidence of achievable online performance, expected future safety, or a statistical guarantee. The safety cap here is a count constraint on observed violations, not a calibrated probability-of-harm bound.

The current receipt records zero oracle allocator overhead by design; a deployable policy must be scored separately with its measured overhead included under the same total resource budgets. Do not compare an overhead-free oracle to a policy without showing this difference.

The standalone function receives typed in-memory records and does not parse/verify dataset manifests, SHA-256 media digests, or licenses. A production adapter needs schema validation, frozen provenance, split integrity, and approved data rights. The source is deliberately not wired into the online calibrated router.

## Adversarial tests

The twelve authored unit tests cover:
1. an unconstrained utility optimum that violates the safety cap, while the constrained reference chooses the best admissible alternative;
2. latency and energy caps that bind independently;
3. a harmful tail action excluded by the CVaR95 cap;
4. group-balanced selection that differs from record-weighted mean;
5. explicit infeasibility rather than relaxed constraints;
6. row-order-invariant selection and deterministic receipts;
7. fail-closed exact-search cap;
8. duplicate IDs and non-finite utility rejection;
9. aggregate resource-cost overflow;
10. aggregate safety-count overflow;
11. invalid identifiers and invalid constraints;
12. receipt fields, constraint values and enumerated-subset count.

Synthetic tests validate selector mechanics only; they do not establish safe perception, calibrated risk, a real-trace gain, or SOTA.

## Benchmark plan

On a frozen, rights-cleared trace set, report three *different* references:
1. the existing resource-constrained utility-only oracle;
2. this resource/safety/tail-constrained exact oracle;
3. deployed candidates scored with actual allocator overhead.

Then compare all-cheap, all-full, matched-budget random, point-threshold, raw-value and calibrated-lower-bound policies at identical latency/energy ceilings and the same pre-registered safety/CVaR limits. Report group-balanced utility, CVaR95, safety-violation counts, abstentions, allocator overhead, total latency/energy, budget violations, worst-group deltas, and paired confidence intervals. Keep invalid and over-budget rows visible. Do not treat the exact oracle gap as a deployable improvement.

A credible benchmark advance is a candidate that reduces paired downstream decision loss against the strongest matched-budget lightweight baseline without violating pre-registered safety/tail limits on a frozen test split and a held-out domain. A synthetic oracle test is not such a result.

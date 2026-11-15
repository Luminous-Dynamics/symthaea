# EGDDM ShiftBench D6.6 — gold-label selective-revalidation oracle

**Version:** 0.1  
**Date:** 2026-11-15  
**Status:** Rust scorer, known-answer fixture, tests and protocol authored. No Rust compile/test, agent rollout or empirical result is claimed.

## 1. Research question

D6 already compares agent revalidation cost with a full-rerun reference. That baseline asks: *how much work can be avoided compared with rerunning every active check?* It does not answer a more precise question:

> What is the smallest revalidation set that would still refresh every newly introduced or gold-labeled affected active obligation?

D6.6 adds an evaluator-side oracle for this lower bound. It makes selective revalidation measurable against both:
- **over-revalidation:** the agent refreshes obligations that did not need invalidation; and
- **under-revalidation:** the agent fails to refresh one or more newly introduced or gold-affected active obligations.

The distinction matters. A method that skips every test appears cheap under a cost-only metric. A method that reruns everything is safe but wasteful. D6.6 measures the set difference between the agent's freshness choices and the gold-label minimum.

## 2. Oracle definition

For intervention epoch \(t\):

- \(A_t\): active obligations after the intervention.
- \(P_t\): obligations active before the intervention.
- \(N_t = A_t \setminus P_t\): newly introduced active obligations.
- \(G_t\): evaluator-side gold obligations affected by the intervention.
- \(R_t^* = (N_t \cup G_t) \cap A_t\): minimum required fresh-revalidation set, assuming the gold impact labels are complete.
- \(R_t\): obligations with a receipt whose \`source_epoch == t\` at the matching checkpoint.
- \(V_t\): obligations in \(R_t\) whose current-epoch receipt outcome is \`pass\`.

Then:

\[
\text{over}_t = |R_t \setminus R_t^*|
\]

\[
\text{under}_t = |R_t^* \setminus R_t|
\]

\[
\text{required-without-pass}_t = |R_t^* \setminus V_t|
\]

\[
\text{set-distance}_t = \text{over}_t + \text{under}_t
\]

The full-rerun reference is \(|A_t|\) count units. The oracle minimum is \(|R_t^*|\). The theoretical count-based savings ceiling is \(1 - |R_t^*|/|A_t|\). The agent refresh fraction is \(|R_t|/|A_t|\). Precision and recall report the overlap between \(R_t\) and \(R_t^*\); they do not replace the independent fresh-pass result.

A newly introduced obligation is always in the minimum set even if a flawed gold annotation omitted it. A gold-affected obligation is only required while active; a removed obligation does not require a fresh current check.

## 3. Why this is a paradigm shift

The benchmark can now distinguish three operating policies:

1. **Full rerun:** broad refresh with low under-revalidation risk but high unnecessary work.
2. **Naive selective reuse:** lower refresh count, but may miss stale or newly required obligations.
3. **Evidence-scoped revalidation:** selective refresh that approaches the gold minimum while preserving current passing evidence for every required obligation.

The target is not to maximize reuse in isolation. It is to approach the minimum safe evidence set while maintaining full required coverage, fresh passing outcomes, and the critical-risk limits from D6/D6.5.

This turns revalidation into an auditable constrained optimization problem. Future versions can introduce a dependency hypergraph and heterogeneous check costs, but the v0.1 count-based oracle is deliberately simple and falsifiable.

## 4. Implementation

The branch adds:
- \`tools/development-map-validator/src/selective_revalidation.rs\`
- CLI command \`development-map-validator revalidation-oracle <scenario.json>\`
- known-answer output at \`docs/development-map/fixtures/egddm-selective-revalidation-oracle-v0.1.expected.json\`

The command first invokes the D6 trace validator, then computes the oracle metrics. It reports per-epoch and per-arm:
- gold minimum set;
- agent refreshed set;
- fresh passing set;
- unnecessary and missing obligation IDs;
- required obligations without a fresh passing result;
- full-rerun, oracle-minimum and agent-refresh counts;
- over/under/set-distance units;
- oracle savings ceiling, observed refresh fraction, precision and recall.

The module authors 12 tests for the committed known-answer fixture, clean-arm behavior, unnecessary refresh, missing required refresh, fresh failure versus missing refresh, unknown gold obligation, unknown receipt obligation, future source epoch, duplicate active IDs, malformed scenario and claim-ceiling behavior.

## 5. Synthetic known answer

The existing D6 fixture has one shifted change:
- \`api-contract\` is marked affected;
- \`audit-requirement\` is newly introduced;
- \`build-repro\` remains unaffected.

Therefore the oracle requires two refreshes out of three active obligations. The synthetic agent refreshes exactly those two, producing:
- oracle minimum: 2;
- full-rerun reference: 3;
- unnecessary refreshes: 0;
- missing refreshes: 0;
- fresh passing required obligations: 2/2;
- count-based savings ceiling: one-third.

The clean arm requires zero post-baseline refreshes. These are fixture arithmetic results, not measured agent performance.

## 6. Assumptions and limits

The result is only as sound as its gold affected-obligation labels. The lower-bound interpretation requires:
1. the labels include every affected active obligation, including shared dependencies and common-cause impacts;
2. new obligations are explicitly represented in the active-obligation timeline;
3. obligation identity and check scope are stable across the trace;
4. one count unit is an acceptable first approximation for each obligation's revalidation cost;
5. the D6 receipt timing/source rules remain valid.

A count-based lower bound is not a real cost lower bound when checks vary substantially in compute, latency, hardware, service charges or reviewer effort. The output therefore keeps count metrics separate from D6's observed cost units. Future work should attach pinned per-check cost distributions, report uncertainty, and model shared setup costs and common-cause invalidation explicitly.

The scorer does not authenticate gold labels, execute checks, fetch artifacts, verify signatures, establish independent custody, or prove that a reused receipt is semantically correct. Gold-label outputs must remain evaluator-side and must never be sent to candidate agents.

## 7. How to compare against industry

Static issue benchmarks such as SWE-Bench Pro V2 measure issue resolution; SWE-Shift measures recovery under mid-execution changes; ChainSWE adds chronological multi-bug maintenance; SWE-Gate adds review constraints. D6.6 is complementary: it measures the *selection quality* of evidence refreshes and the efficiency gap to an oracle minimum under a matched state transition.

A credible advantage requires full EGDDM to:
- match or improve shifted recovery;
- maintain the critical unverified-exposure and worst-gap gates;
- reduce unnecessary revalidation relative to full rerun and static dependency traversal;
- approach the oracle minimum without under-revalidating;
- lower full end-to-end cost after runner and human-review effort are included;
- reproduce on held-out repositories, intervention families and future tasks.

Do not compare the oracle ceiling with a published SWE-Bench pass rate. They are different metrics and different experiments.

## 8. Claim ceiling

D6.6 implements a deterministic, count-based lower-bound analysis over a D6-validated trace and supplied evaluator-side gold labels. It is not an execution oracle, causal proof, authenticated label set, empirical agent result, SOTA result, or production safety guarantee. Rust tests are authored but have not been compiled or run in this environment.

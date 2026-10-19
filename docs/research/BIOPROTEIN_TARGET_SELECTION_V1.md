# BIOPROTEIN-MVP-001: evidence-gated ingredient target selection

**Status:** executable synthetic-first Rust contract; exact-head workflow and receipt included.  
**Tracking:** [issue #8118](https://github.com/Luminous-Dynamics/symthaea/issues/8118).  
**Prior decision machinery:** [base EVSI](https://github.com/Luminous-Dynamics/symthaea/pull/8098), [joint-scenario EVSI](https://github.com/Luminous-Dynamics/symthaea/pull/8114), [fixed-budget regret](https://github.com/Luminous-Dynamics/symthaea/pull/8116).

## Paradigm shift

Do not begin by maximizing strain titer and ask later whether the ingredient is useful or manufacturable. Start from a narrowly specified functional job and compare the product opportunity and route together: product function, compliant usable output, recovery, authorization, scale transfer, cost, energy, water, time, evidence rights, and explicitly bounded TEA/LCA scenarios.

This matches the 2026 precision-fermentation MVP review, which frames target selection around a value proposition linking ingredient class, volume, price and functional role, and the 2026 scale-up-gated DBTL review, which argues that downstream-process compatibility and industrial robustness must gate progress while TEA/LCA—not titer plateaus—should be convergence criteria:
- A. et al., *Minimum viable products in precision fermentation: defining translational milestones for food biotechnology*, Current Opinion in Biotechnology 100 (August 2026), [doi:10.1016/j.copbio.2026.103515](https://doi.org/10.1016/j.copbio.2026.103515).
- He et al., *Closing the Loop with Gates: A Scale-up-Gated Design–Build–Test–Learn Framework for Industrial Fermentation*, Microorganisms 14(8), 1830 (19 August 2026), [doi:10.3390/microorganisms14081830](https://doi.org/10.3390/microorganisms14081830).
- ISPOR good-practice reports explain why parameter dependencies and structural uncertainty belong in decision analysis, rather than being silently treated as independent or known: [Value of Information Analysis – Methods](https://pmc.ncbi.nlm.nih.gov/articles/PMC7373630/).

## Contract

A target brief contains a product/application/function, quality and purity basis, functional unit, target volume band, price/value basis and vintage, biology endpoint, process-closure and scale-transfer references, and TEA/LCA plus regional feedstock/electricity scenario digests. All identifiers are provenance hooks, not proof by themselves.

Each target has seven explicit gates: food safety, authorization, source rights, market evidence freshness/sufficiency, product-function evidence, process closure, and scale transfer. A failed gate blocks the target; unknown or not-assessed evidence makes the target indeterminate. High estimated value, titer, or another objective cannot compensate for a failed gate. Before real deployment, gate state must be supplied by a named evidence-verification process; this oracle does not certify the underlying source.

## Interval-aware Pareto semantics

The objectives are typed intervals with explicit direction and unit. Maximization objective A robustly beats B only when A.low is at least B.high; minimization requires A.high to be at most B.low. At least one objective must be strictly better. Overlap is uncertainty—not a tie-break invitation. All objective sets, units, functional units, product specification, price basis/vintage, TEA/LCA boundary, and regional/feedstock/electricity scenario digests must match. Otherwise the case is incomparable and no Pareto frontier is emitted. The contract performs no implicit conversions, imputation, or scalar weighting across unlike objectives.

Only eligible targets enter the Pareto relations. A failed or indeterminate target can never dominate an eligible target or appear on the frontier. The frontier is a set, not a forced single winner. EVSI can then prioritize the next evidence acquisition; the decision-regret benchmark can compare that plan to simple baselines under the same budget.

## Frozen synthetic corpus

Eleven cases exercise:
1. robustly dominated product target;
2. attractive market/metric proxy blocked by authorization;
3. high nominal titer with poor recovered usable output;
4. mismatched functional units;
5. unpermissioned source;
6. unknown scale-transfer evidence;
7. regional/feedstock/electricity scenario mismatch;
8. overlapping metric intervals;
9. synthetic fixture mislabeled as measured (negative control);
10. stale market-value evidence;
11. objective unit mismatch.

All data, ranges, values, digests and gate states in the corpus are invented. Pass means the encoded contract matches this frozen fixture—not that any real target is viable.

## Benchmarks

| Benchmark | Contract target |
|---|---:|
| Frozen-case disposition/frontier agreement | 100% |
| False accepts across hard-gate negative controls | 0 |
| Synthetic-as-measured acceptance | 0 |
| Silent unit conversion or missing-value imputation | 0 |
| Pareto relations across overlapping intervals | 0 |
| Frontier emitted for incomparable contexts | 0 |
| Exact-head mismatch accepted | 0 |
| Receipt includes corpus SHA-256 and per-case reasons | 100% |

Next empirical tier (not implemented or claimed as achieved): independently source-check data and rights; preregister product specification and region; use blinded held-out candidate/region combinations; compare target choices against titer-only, price-only, and human-expert baselines; report Pareto stability and decision regret under joint feedstock/electricity/recovery shocks; then validate with measured usable output and audited complete-cycle economics/LCA. Do not set a universal “20% better” claim until the comparison metric, sample size, evaluator, and test set are preregistered.

## Scope boundaries

This is a decision-contract tool, not a biological design, wet-lab protocol, safety certification, regulatory opinion, validated TEA/LCA engine, or commercial recommendation. Symtropy should provide process closure/recovery evidence; Sol Atlas regional inputs should be licensed and dated; Mycelix should preserve source permissions, lot/assay lineage, corrections, and claim provenance. Real records must carry original units, uncertainty, provenance, and evidence label; synthetic data must never be upgraded by this tool.

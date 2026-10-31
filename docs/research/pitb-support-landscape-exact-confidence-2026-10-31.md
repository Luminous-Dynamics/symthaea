# PITB exact support-landscape confidence and power (2026-10-31)

**Current recommended statistical track:** support landscape v2 uses exact one-sided Clopper-Pearson lower confidence bounds with Bonferroni simultaneous adjustment. The previous v1 Wilson implementation and power screen remain as historical research artifacts; do not mix their outputs with v2.

## Exact simultaneous reliability gate

For each well-formed context, let X be the number of exact-target successes in n independently seeded, identically configured attempts. A success requires both trusted-kernel acceptance and semantic-gate acceptance. The one-sided Clopper-Pearson lower bound L is obtained by inverting the exact Binomial upper tail:

\[
P_{p=L}(X \ge x)=\alpha/m
\]

where m is the number of well-formed contexts in the frozen lattice and alpha is the familywise error budget. Each context qualifies only if L is at least the preregistered reliability floor and n meets the minimum replicate count.

By Bonferroni's inequality, the familywise probability of at least one false reliability qualification is at most alpha when the per-context exact one-sided tests are valid. This guarantee does not require independence between different contexts. It does require the per-context rollout trials to satisfy the assumed Bernoulli sampling model; seed dependence, model-version changes, nonstationarity, or adaptive outcome-dependent reruns would invalidate that interpretation.

The v2 analyzer computes the lower bound by numerically inverting the Binomial tail, not by a normal approximation. The two-sided Wilson interval remains a descriptive display only and is not used for qualification. The trial schema and exact v2 workflow are separate from the v1 artifacts so the statistical contracts cannot be silently mixed.

## Design-only sample-size calculation

Planning parameters:
- reliability floor: 0.80;
- design alternative: true success probability 0.90;
- familywise alpha: 0.05;
- target probability of qualification at that design alternative: 0.80.

| Well-formed contexts | Attempts per context | Successes required | Threshold observed rate | Total attempts |
|---:|---:|---:|---:|---:|
| 4 | 126 | 111 | 88.10% | 504 |
| 8 | 145 | 128 | 88.28% | 1,160 |
| 16 | 171 | 151 | 88.30% | 2,736 |
| 64 | 216 | 191 | 88.43% | 13,824 |
| 256 | 255 | 226 | 88.63% | 65,280 |

These values are computed by finding the minimum n and critical success count satisfying the exact Clopper-Pearson lower-bound gate and the Binomial alternative power target. They are design calculations, not observed prover performance. The iid Bernoulli model and finite-sample operating characteristics still need independent simulation and statistical review before confirmatory preregistration.

## Reproduction

The exact planner is scripts/research/pitb_support_landscape_power_v1.py. Its local exact-source copy passed:
- Python byte-compilation;
- five fixed context-count sample-size anchors;
- Clopper-Pearson boundary checks (all successes and zero successes);
- the default 4/8/16/64/256 scenario calculation.

The corresponding focused workflow is .github/workflows/pitb-support-landscape-power-v1.yml. No GitHub Actions result is implied by local checks.

## Consequence for study design

A full eight-variable powerset can have 256 subsets before invalid contexts are removed. Even under a true 90% per-context success probability, the exact simultaneous design can require over 65,000 prover attempts per theorem. Use a discovery/confirmation split:
1. explore the valid context landscape with a small, fixed exploratory schedule;
2. freeze a candidate route family, required proper subcontexts, full-context control, dependency manifest, model/checker/policy/budget and fresh confirmation seeds;
3. confirm those frozen contexts with exact simultaneous inference.

This supports a confirmatory claim about the frozen candidate family. It does not prove that the entire powerset antichain is complete unless every valid subset is assessed or a separate pruning certificate proves that untested subsets cannot be minimal.

## Research boundary

Exact finite-sample binomial inference is stronger than the previous Wilson approximation, but it does not solve benchmark contamination, semantic statement alignment, or dependence between rollout seeds. It does not replace component-level generalization analysis. The v0 source split remains unqualified as an independent mathematical holdout until the real cross-family relationship audit and reviewer coverage are complete.

Related work:
- Fixed-rollout pass@k non-identifiability: https://arxiv.org/abs/2609.09245
- Adaptive context-dependent Lean search: https://arxiv.org/abs/2608.18084
- Lean dependency cleanup baseline: https://lean-lang.org/doc/api/Lean/Meta/Tactic/Cleanup.html

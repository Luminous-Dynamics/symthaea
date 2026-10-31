# PITB support-landscape reliability power screen (2026-10-31)

**Status:** design-only computation. The script is locally executable; no real theorem-agent outcomes are used.

## Why the number of contexts changes sample size

The probabilistic support-landscape analyzer qualifies a context only when a one-sided Wilson lower confidence bound, Bonferroni-adjusted over all well-formed contexts, reaches a reliability floor of 0.80.

A minimum of 20 attempts per context is only a structural pilot floor. It does not imply that the study can qualify a context. For four or more simultaneously evaluated contexts, even 20/20 successes do not reach the 0.80 lower-bound requirement under the current adjustment.

To expose this before collecting expensive model runs, the research branch now includes:
- scripts/research/pitb_support_landscape_power_v0.py
- .github/workflows/pitb-support-landscape-power.yml

The planner finds the smallest replicate count where the Bonferroni-Wilson rule both has 80% probability of qualifying a context under a design alternative with true success probability 0.90 and requires a specified observed success count. It uses the exact Binomial upper tail for the design power calculation, not a pass@k extrapolation.

## Design screen results

Parameters: reliability floor 0.80, true per-context success probability 0.90, familywise alpha 0.05, target power 0.80.

| Well-formed contexts | Minimum paired attempts per context | Successes needed to qualify | Observed rate at boundary | Total attempts across lattice |
|---:|---:|---:|---:|---:|
| 4 | 126 | 111 | 88.10% | 504 |
| 8 | 152 | 134 | 88.16% | 1,216 |
| 16 | 172 | 152 | 88.37% | 2,752 |
| 64 | 225 | 199 | 88.44% | 14,400 |
| 256 | 272 | 241 | 88.67% | 69,632 |

This shows the cost of evaluating many contexts simultaneously. An eight-variable powerset contains 256 candidate subsets before dependency-invalid rows are removed. A fully repeated reliability landscape at that scale is not a cheap pilot; it can require tens of thousands of solver attempts per theorem even under optimistic success assumptions.

The calculation is a planning screen for the current Wilson decision rule, not a guarantee of actual power. Its binomial assumptions and finite-sample operating characteristics must be independently Monte Carlo checked before confirmatory preregistration.

## Recommended discovery / confirmation protocol

Do not spend 272 attempts on every possible context just to discover which few routes deserve attention.

**Stage A — discovery (exploratory only).** Enumerate well-formed subsets and use a small, fixed seed schedule, perhaps 5–10 attempts per context, to identify candidate minimal routes, dependency patterns, context-invalid interventions, and cost outliers. This stage may rank candidates but must not claim 0.80 reliable sufficiency.

**Stage B — freeze the candidate proof routes.** Before new outcomes are observed, freeze:
- candidate minimal routes;
- all proper subcontexts needed to test their minimality;
- the exact full-context control;
- the dependency manifest, theorem digest, model/checker, solver policy, budgets, and independent confirmation seed schedule.

The candidate set should include every proposed minimal route and its relevant proper subsets. If two candidates share subcontexts, reuse the same independently scheduled control runs. Do not use discovery seeds for confirmation.

**Stage C — confirm with simultaneous inference.** Use fresh seeds and simultaneous multiplicity control across the frozen candidate contexts. The confirmatory claim is limited to the candidate family frozen in Stage B; it is not a claim that no other reliable route exists in the untested part of the full powerset. To claim a globally complete antichain, the full valid lattice must be tested or covered by a separately proved pruning certificate.

This separates efficient exploration from defensible confirmation, while avoiding post-hoc deletion of inconvenient contexts.

## Related work and comparison

- *What Fixed-Rollout pass@k Evaluations Can Identify* (September 2026) shows that fixed-n rollout counts identify direct pass@k only up to observed n in its nonparametric setting; extrapolation beyond n requires additional assumptions. PITB therefore reports direct empirical success frequencies and intervals rather than extrapolated pass@k. [Paper](https://arxiv.org/abs/2609.09245).
- *Compiler-Guided Adaptive Proof Search with Cross-Model Synergy* reports +12.8 percentage points average pass rate and 21.9% fewer LLM calls than pass@k baselines within a pass@32 budget on seven real Lean projects. PITB should compare context-conditioned route scheduling with that kind of adaptive search, not just independent sampling. [Paper](https://arxiv.org/abs/2608.18084).
- Lean already provides MVarId.cleanup, which removes declarations according to dependency relevance to the target and preserved variables. Use it as a deterministic context-pruning baseline; its relevance criterion is not itself a stochastic prover-reliability result. [Official Lean cleanup API](https://lean-lang.org/doc/api/Lean/Meta/Tactic/Cleanup.html).

## Immediate benchmark decision

For the first calibration, keep the declaration universe small (2–4 mutable declarations) and use the power screen to choose the seed count for the preregistered candidate family. Use the full exhaustive reliability landscape as a later high-compute cohort. The primary transfer endpoint and its component-clustered inference remain separate.

No model result, real support landscape, transfer improvement or SOTA claim is implied by this design screen.

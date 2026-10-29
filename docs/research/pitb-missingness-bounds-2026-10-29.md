# PITB paired-outcome missingness bounds and tipping-point diagnostic

**Date:** 2026-10-29  
**Status:** implemented as a research-only diagnostic and tested locally on synthetic fixtures; not a confirmatory analyzer and not a proof-transfer result.

## Why this is different from the existing missingness simulation

The earlier Monte Carlo screen showed how complete-case deletion can bias a paired effect when missingness depends on outcomes. This new diagnostic addresses a complementary question: given a fixed, fully enumerated target set and observed binary outcomes for two matched method arms, what effect values remain possible without assuming how the unknown outcomes would have resolved?

For each frozen target, status is one of:

- `success`: observed operational outcome (Y=1);
- `non_solve`: observed operational non-solve (Y=0), including timeout, unavailable tool, not attempted or proof rejected under the preregistered operational endpoint;
- `unknown`: the outcome evidence is genuinely unavailable, so the latent binary outcome is in ([0,1]).

An absent target row is not permitted: the script rejects it. An `unknown` outcome is never dropped from the denominator. This is a bounded-outcome, finite-cohort identification calculation, not a confidence interval or estimator of a superpopulation effect.

## Exact paired bounds

Let the treatment and control outcomes for target i be Y_Ti and Y_Ci in the frozen finite cohort of N targets. The quantity of interest is:

`Delta_N = (1/N) * sum_i(Y_Ti - Y_Ci)`

For targets with both outcomes observed, the contribution is exact. If only control is observed, the missing treatment outcome is bounded in `[0,1]`; if only treatment is observed, the control outcome is bounded in `[0,1]`; if both are unknown, the difference is in `[-1,1]`. Summing these per-target minima and maxima and dividing by the *full frozen target count* gives the sharp finite-cohort identification interval under only binary outcome bounds. Every possible completion of unknown outcomes must lie inside this interval.

The report also provides a 5×5 assumption grid over success probabilities among unknown treatment and control outcomes (0, 0.25, 0.5, 0.75, 1). These are explicit sensitivity assumptions—not imputed facts. A complete-case paired effect is shown for diagnosis only and must never replace the fixed-cohort bound.

## Synthetic negative-control fixture

Reproduce from the repository root:

```bash
python3 scripts/research/pitb_missingness_bounds_v0.py --self-test
python3 scripts/research/pitb_missingness_bounds_v0.py \
  data/benchmarks/proof-insight-transfer-v0/fixtures/paired-outcome-ledger.missingness.synthetic.json \
  /tmp/pitb-missingness-bounds-report.json
```

The synthetic fixture contains 8 fixed targets across 4 component labels, with 5 targets having at least one unknown outcome. The observed complete-pair effect is 0 percentage points on only 3 of 8 pairs (37.5% coverage). The sharp worst-case finite-cohort identification interval is **−37.5 to +50 percentage points**. Setting the assumed success probability of every unknown outcome to 0.5 produces +6.25 points; assuming 1 for both arms yields +12.5 points. These different assumptions illustrate why the observed complete-pair estimate alone cannot identify the full-cohort result. The input fixture is intentionally incomplete in its outcome evidence and **must** remain confirmatorily blocked.

The local self-test suite has seven tests: complete-data interval collapse, exhaustive unknown-outcome completion containment, non-solve semantics, missing frozen-row rejection, duplicate-row rejection, invalid-status rejection, and non-string-status rejection. Local syntax compilation and these seven tests passed for the authoring copy. The exact-head GitHub Actions result remains unverified.

## What this does not permit

- It does not make a missing-ledger row acceptable; the expected target IDs and rows must reconcile exactly.
- It does not allow timeouts or failed proofs to be relabelled as unknown to gain wider bounds. Those are operational non-solves if that is the preregistered primary endpoint.
- It does not authorize confirmatory use. The script always emits `confirmatory_eligible: false`; any unknown outcomes set `confirmatory_status: blocked_unknown_outcomes`.
- It does not provide frequentist confidence coverage, solve cluster dependence, establish statement correctness, validate the dependency graph, or replace the primary analyzer's receipts and power gate.
- It does not recover a latent "capability if the run had not failed" estimand. That is a distinct question requiring assumptions about why outcomes are absent.

## Why this is methodologically useful

Partial-identification methods make the role of assumptions visible instead of selecting one missing-data model and treating its imputation as observed fact. Manski's overview of partial identification with missing data describes this identified-set approach ([overview](https://www.sciencedirect.com/science/article/pii/S0888613X04001124)). Recent literature presents tipping-point sensitivity as a systematic exploration of assumptions that can overturn a conclusion rather than a few convenient alternatives ([tipping-point method, 2025](https://pubmed.ncbi.nlm.nih.gov/40140629/)). A 2026 paper on incomplete outcomes in cluster-randomized studies similarly derives doubly robust methods under specified missing-at-random conditions and adds a sensitivity framework for violations of those assumptions ([open article](https://pmc.ncbi.nlm.nih.gov/articles/PMC13108325/)). Research on informative cluster size emphasizes that individual-weighted and cluster-weighted estimands may differ and should not be swapped silently ([Statistics in Medicine, first published 9 October 2026](https://onlinelibrary.wiley.com/doi/10.1002/sim.70762)).

PITB is not a randomized clinical trial, and its proof-method comparison does not inherit trial randomization assumptions. These sources inform the general discipline: define the estimand, state the assumptions, show how conclusions depend on missing outcomes, and keep the diagnostic distinct from confirmatory proof evaluation.

## Decision and next gate

Use this tool as an audit/sensitivity companion to the fail-closed outcome ledger. The primary evaluation should still block if an expected result row or proof receipt is absent. The bounds are useful to quantify how much the remaining evidence leaves unidentified—not to excuse missing artifacts. Before claiming a transfer advantage, retain complete run receipts, qualify the actual dependency components, apply the preregistered cluster inference procedure and replicate the exact-target semantic-gate comparison.

**No model result or SOTA claim is made.**

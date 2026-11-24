# PCIR Posterior-Draw Cohort Evaluator v0.1

**Status:** executable evaluator-side cohort adapter; exact-head Rust qualification pending.  
**Parent:** #13543 — Posterior-Coupled Individual Rollouts.  
**Consumes:** #13520 — individual counterfactual effect retention.

## Purpose

#13520 evaluates one individual/action pair and separates:
- independently measured realized effect;
- frozen-latent predicted effect;
- imagined-rollout predicted effect.

It deliberately rejects duplicate individual identities within one cohort and provides exact finite arithmetic. PCIR needs a further unit of organization: multiple posterior hypotheses about the *same* individual and the *same* already-executed pair of action branches.

This module supplies that adapter. It calls #13520 once per posterior draw, reusing its arithmetic and branch validation, then groups the per-draw predictions under the original individual identity.

## Input semantics

For an individual i, action pair (a0,a1), and posterior draw k, the input binds:
- one posterior draw ID, SHA-256 digest and integer PPM weight;
- one baseline branch prediction;
- one alternative branch prediction;
- the existing #13520 context and branch records.

Both branches must reference the exact same draw ID and digest. The candidate's draw-conditioned latent-state digest and predicted utilities may differ between posterior draws.

The actual branch evidence may **not** differ between posterior draws. Baseline/alternative source run IDs, action IDs, applied timestamps, realized utilities, factual-history digest, branch-point digest, exogenous-forcing manifest, model configuration and latent representation profile must remain fixed. The same actual pair is not re-executed just because the posterior is sampled more than once.

The module also requires:
- nonzero unique draw IDs and draw digests within each individual;
- positive integer weights summing exactly to 1,000,000 PPM for each individual;
- unique individual IDs and pair IDs in the outer cohort;
- a fixed task, action-pair, loss-scale, evaluator, split, model, representation, readout, rollout and pairing-verifier profile across the compared individual cohort.

A missing/non-applied physical branch is passed to #13520 and returned as its typed error. It is never silently dropped or assigned zero.

## Output semantics

For each individual, the evaluator returns:
- the one independently measured realized effect and its application timestamps;
- separate latent-readout and imagined-rollout posterior-sample distributions;
- weighted empirical 5th, 50th and 95th percentiles;
- min/max sample support;
- exact weighted-effect and weighted-absolute-error numerators;
- negative/zero/positive weight mass;
- sign-match weight;
- every draw's ID, digest, weight, prediction, error and sign comparison.

The weighted mean is represented as an exact rational with denominator 1,000,000. The weighted empirical quantiles use the lower inverse-CDF convention. The result is kept **per individual**: the number of posterior draws never increments individual_count and must never be used as the independent bootstrap unit.

These outputs describe the finite supplied draw set. They do not establish that the learned posterior is calibrated, that its support is complete, that a 5–95% sample interval has 90% coverage, or that a sampled effect range is a causal identified set. Calibration and structural support remain separate owners.

## Why this is not a second effect evaluator

The adapter calls evaluate_individual_effect_retention from #13520 for every posterior draw. It does not reproduce its:
- branch provenance checks;
- task/loss-scale pairing rules;
- treatment-effect arithmetic;
- non-application handling;
- latent/rollout absolute-error calculations;
- sign comparison.

It adds only draw-level coupling, cross-draw fixed-realization checks, exact posterior-weight aggregation and per-individual cohort grouping.

It does not authenticate the pairing receipt bytes. The trusted launcher must validate #13377/Symtropy's branch/event correspondence evidence before supplying records. #9437 process/OS isolation remains mandatory.

## Tests

Eighteen deterministic Rust tests cover:
1. bimodal effect distribution with zero mean but nonzero sign ambiguity;
2. draw-ID mismatch between siblings;
3. draw-digest mismatch between siblings;
4. fixed actual branch evidence reused across draws;
5. realized-effect drift across draws;
6. common-mode branch utility shifts that leave the effect unchanged;
7. application-time drift across draws;
8. source run-ID drift across draws;
9. exact PPM weight-sum enforcement;
10. duplicate draw-ID rejection;
11. duplicate draw-digest rejection;
12. posterior draws not inflating individual count;
13. model-profile drift rejection;
14. typed non-application propagation;
15. latent and rollout distributions remaining distinct;
16. exact weighted mean and error numerators;
17. canonical output under reordered individual/draw input;
18. empty draw-set rejection.

These tests are authored in source. No Rust test/format/Clippy result is claimed until the exact-head workflow completes successfully.

## Claim boundary

This adapter is a deterministic evaluator-side aggregation primitive only. It does not implement posterior inference, train HDC/CfC/LTC, run DMC, compute the abducible-share ladder, calibrate posterior coverage, establish causal identification, calculate regret, or authorize an action.

No DMC reproduction, model gain, benchmark result, resource improvement, physical result or SOTA claim is attached to this change.

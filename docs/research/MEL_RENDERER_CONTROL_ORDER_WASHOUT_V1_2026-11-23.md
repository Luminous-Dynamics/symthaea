# Melothaea renderer-control arm order and session carryover v1

**Date:** 23 November 2026  
**Parent evaluator:** [#13362 — counterfactual renderer-control effect qualification](https://github.com/Luminous-Dynamics/symthaea/pull/13362)  
**Capability contract:** [#10202](https://github.com/Luminous-Dynamics/symthaea/pull/10202)  
**Owner thread:** [#10196](https://github.com/Luminous-Dynamics/symthaea/issues/10196)

## Why this increment exists

The baseline/sham/active/reversal evaluator qualifies whether a control has an observed effect under matched backend/version/adapter/source/seed. That alone does not eliminate two residual confounds:

- **Order effects:** the active arm may always run after baseline/sham, so time drift or model/session adaptation can masquerade as a control effect.
- **Carryover:** a prior active or reversal setting may persist into the next arm, even when the API reports a reset or the next request supplies a different value.

The evaluator must not infer a causal control effect from four observations if their order and session state are unknown.

## Implemented

The companion Rust module is crates/domains/symthaea-muse/src/realization_effect_order.rs.

### 1. Deterministically randomized, fully counterbalanced order

Each 24-probe block uses all 24 possible permutations of:

- baseline;
- sham/no-op;
- active;
- reversal.

The schedule is generated from a preregistered SHA-256 seed commitment and block ID. Each order plan binds the probe ID, block ID/index, seed commitment and four-arm sequence. The schedule is deterministic for replay, but hash-ranked so arm order is not fixed across every probe. A block validator proves that the 24 permutations occur exactly once and rejects missing, duplicated, mixed-seed, modified or out-of-block plans.

This is a counterbalancing mechanism, not a claim of cryptographic randomness. The exact seed commitment and schedule receipt must be included in the experiment manifest before any outcomes are inspected.

### 2. Per-arm state-transition receipts

Each arm carries a digest-bound session transition receipt with:

- session ID and previous/current arm identity;
- previous/current state epoch;
- exact previous-arm output artifact digest;
- reset event digest, reset observer ID and observer identity digest;
- washout duration;
- pre-arm baseline target-metric value and its independent observation/observer identity.

This creates an explicit chain from the previous arm's output to the next arm's starting state.

### 3. Carryover and washout gates

The probe specification pins:
- whether an explicit reset is required before each arm;
- minimum washout duration;
- maximum allowed pre-arm baseline drift;
- minimum state-epoch increment.

The ordered evaluator rejects before scoring when an arm is out of order, a reset is missing, the washout is too short, the state epoch does not advance, the previous-output link is wrong, the reset/baseline observer is the output producer, or the pre-arm sentinel measurement drifts beyond its preregistered bound.

Only after all order/session conditions pass does the evaluator delegate to the existing baseline/sham/active/reversal effect scorer. The original effect verdict remains unchanged and is nested in an order-qualified report.

## Negative controls

The new tests cover:
- deterministic 24-order schedule generation and exact permutation coverage;
- missing/duplicated block assignment;
- valid ordered probe with chained reset/washout receipts;
- arm sequence mismatch;
- missing explicit reset;
- insufficient washout;
- pre-arm baseline drift indicating carryover;
- previous arm/output/state chain mismatch;
- reset observer equal to output producer.

The existing 11 control-effect tests remain in the workflow as regression gates.

## Initial engineering targets

These are prospective targets, not external or production measurements:

| Invariant | Target |
|---|---:|
| Every complete 24-probe block contains all 24 permutations once | 100% |
| Scored arms match preregistered within-probe order | 100% |
| Arms without valid transition/reset/washout evidence scored | 0 |
| Pre-arm baseline drift beyond threshold credited as causal effect | 0 |
| Previous-output/session-epoch linkage missing or substituted | 0 |
| Existing active/sham/reversal/collateral controls still pass | 100% synthetic known-answer fixtures |

Real renderer studies must set washout duration and drift tolerance from pilot stability measurements, not select them after looking at treatment effects. If a renderer cannot reliably reset, use an independently justified washout protocol and report the residual carryover bound; otherwise the probe is unqualified.

## Remaining limits

- The order generator is deterministic hash-ranked assignment, not an external randomization service.
- Session reset and baseline receipts are digest-linked declarations; this module does not authenticate the reset observer or physically inspect hidden backend state.
- A pre-arm baseline sentinel catches measured drift only in the selected metric; it cannot prove all latent state has been cleared.
- Production capture/observation remains owned by [#13349](https://github.com/Luminous-Dynamics/symthaea/issues/13349) and [#13348](https://github.com/Luminous-Dynamics/symthaea/pull/13348). Signed actor/runtime provenance remains under [#13254](https://github.com/Luminous-Dynamics/symthaea/issues/13254).

**Claim ceiling:** this is counterbalancing and state-carryover qualification logic for the existing synthetic effect evaluator. It is not proof that a commercial renderer changes sound as requested, nor a result from musician/listener sessions.

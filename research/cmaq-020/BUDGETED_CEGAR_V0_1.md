# CMAQ-021 — Budgeted CEGAR and candidate-visible probe receipts v0.1

**Status:** finite deterministic evaluator extension, stacked on CMAQ-020 / draft PR #13747.  
**Claim ceiling:** budget arithmetic, receipt validation and synthetic partition-refinement behavior only. No external model result, causal-identification result, global state-sufficiency proof or SOTA claim.

## Motivation

CMAQ-020 already selects a low-cost separating intervention/query/horizon identity from candidate-merged state pairs. A query identity is not the same thing as evidence acquisition: the evaluator must also account for what the candidate actually observed, whether that observation is bound to the selected pair and frozen source, how much budget it consumed, and whether the candidate used it productively.

The extension keeps three events distinct:

1. **Counterexample selection:** the evaluator returns an opaque state pair and the cheapest registered response key. No numeric response or invariant value is included.
2. **Probe execution:** the candidate executes that permitted intervention/query/horizon against both states. The environment returns the two observations.
3. **Candidate refinement and evaluation:** the candidate submits a new partition under a new checkpoint digest. The evaluator checks the observation receipt and compares before/after partitions against unchanged reference records.

## Receipt contract

`ProbeObservationReceiptV1` binds:
- the frozen contract ID;
- left/right state IDs and exact `ResponseKey`;
- the two values actually returned by the environment;
- the frozen source-manifest digest;
- the digest of the exact receipt bytes.

The candidate receives the observed values as normal results of the permitted probe. It does not receive the evaluator's full response table, reference partition, or any unselected response values through the receipt. The receipt and request bind to the candidate-checkpoint SHA-256 as well as the frozen contract/source manifest, pair and response key, preventing an observation from an earlier candidate partition from being silently reused after refinement. The evaluator accepts only the probe currently selected from the candidate's existing merged pairs, requires the response key and cost to match the contract, validates the observation values against the deterministic reference result, and rejects malformed or replayed receipt digests.

The source and receipt digests are only syntax-checked here. The runner must hash exact raw bytes, authenticate the originating environment, and attach signed custody receipts (#9437). This API is not a cryptographic verifier.

## Budget ledger

`ProbeBudgetLedger` is scoped to one frozen contract and one candidate session. It binds to the contract ID, canonical contract SHA-256 and source-manifest SHA-256; reusing a human-readable contract ID with a different contract or source manifest is rejected. Cost comes only from the frozen `probe_cost_units` value; the candidate cannot lower or override it. The ledger:
- uses checked integer addition;
- rejects a probe that would exceed the registered budget;
- rejects reused receipt digests;
- does not mutate on invalid receipt, wrong contract, wrong selected probe or budget exhaustion;
- never refunds a probe already executed if the candidate's subsequent partition update is malformed or ineffective.

Invariant-only witnesses have no intervention/query cost in this version. Source reconstruction and invariant verification must be charged through a separate ledger before that path is promoted.

## Refinement effectiveness

For each accepted probe, the evaluator reports before/after `AbstractionScore` values and distinct deltas:
- false merges resolved and newly introduced;
- false splits resolved and newly introduced;
- whether the exact selected state pair was separated;
- probe cost, cumulative cost and remaining budget;
- cost units per false merge removed (undefined when none were removed).

The after-record set must preserve the exact evaluated state IDs, source-state digests, domain/family/cluster identity, invariant signatures and evaluator response signatures. It must carry a new candidate-checkpoint digest. The candidate can refine poorly or not at all; the evaluator reports that outcome rather than auto-splitting states.

The scalar efficiency ratio is descriptive only. It must never hide newly introduced false splits or false merges. Future experiments should report the full cost-versus-(false merges, false splits, held-out query regret) frontier.

## Synthetic known-answer study

The deterministic fixtures cover:

- one cheapest probe that separates a false-merged pair, with exact charge and zero new false splits;
- one probe that can resolve two false merges when the candidate partitions all three states, but also introduces a false split between two response-equivalent nuisance variants;
- budget exhaustion without charge;
- malformed/incorrect observation values without charge;
- duplicate receipt replay without double charging;
- evaluator-reference mutation in the candidate's after-record set (the executed probe remains charged);
- missing checkpoint advancement;
- a candidate that receives a valid probe but makes no partition progress.

An error-migration fixture keeps aggregate false-merge and false-split counts unchanged while replacing the identities of the erroneous pairs; the pair-level ledger must report both the resolved old errors and the newly introduced ones. These are unit tests of the protocol and scoring logic, not empirical model results.

## Critical evaluation caveat: counterexample identity is an oracle hint

The selected pair and response-key identity reveal which query is predicted to distinguish the pair. Numeric outcomes remain undisclosed until the probe runs, but the candidate has still received a counterexample hint. This setting must not be described as oracle-blind. Fair comparisons must give all methods the same hint and the same paid probe budget. A separate blind active-query track should measure how candidates discover useful queries without the evaluator selecting the witness.

## Separate oracle-blind probe acquisition (CMAQ-023)

CMAQ-023 adds a second acquisition path with candidate-selected `BlindProbeRequest`. It is intentionally separate from `charge_probe`:

- The candidate selects a canonical state pair that is currently merged in its own partition.
- The candidate selects any registered intervention/query/horizon key in the frozen contract.
- The ledger validates the selected pair, contract, current candidate-checkpoint digest, receipt identity, source manifest, observed values, digest syntax, replay status and frozen cost.
- The blind path does **not** call `select_counterexample_probe` and does not require the selected response to differ between the pair.
- A valid non-separating observation is still an executed experiment and consumes the registered cost. No refund is made because the candidate chose an uninformative query.
- Invalid pair/key/receipt, replay and budget exhaustion do not mutate the ledger.

The hinted `charge_probe` path remains useful as an evaluator-assisted CEGAR upper bound: the evaluator selects the cheapest known separating pair/query identity. It must not be presented as autonomous query discovery.

### Matched comparison protocol

Run hinted and blind acquisition with the same frozen contract, initial candidate partition, environment/mechanism split, total probe-cost budget, receipt-verification process and candidate refinement scorer. Add uniform-cost, random, uncertainty/information-gain and policy-selected controls. Do not give a blind candidate the hinted path's chosen witness, pair or key.

Report per independent environment/mechanism cluster:

- total cost and number of probes, including non-separating probes;
- number and rate of useful separations discovered;
- pair-level false merges resolved/introduced and false splits resolved/introduced;
- held-out response regret and downstream task utility;
- abstention/UNKNOWN rate and worst-family performance;
- all-in state/checkpoint bytes, model update compute and query latency.

The hinted track is a reference upper bound under evaluator assistance, not a matched autonomous baseline. Report the full cost-versus-utility frontier rather than only cost per successful separation; the latter omits false splits, uninformative queries and task consequences.

## Promotion requirements

Before external promotion:
1. #9437 must recompute receipt hashes from exact environment-returned bytes and authenticate probe execution independently.
2. #8578 must own cluster-level uncertainty, paired resampling and multiplicity control.
3. The experiment must report independent environment/mechanism clusters, not task rows, as inferential units.
4. Add held-out intervention families and nuisance shifts so that a candidate cannot optimize only the probe identities it has already seen.
5. Compare cheapest-first CEGAR with random, uniform-cost, information-gain and blind-query controls at matched total cost.
6. Report false merges, false splits, changed-consequence correctness, response regret, task utility and all-in memory/compute separately.
7. Keep the evaluator-hinted CEGAR and candidate-chosen oracle-blind protocols separate; match their total registered probe-cost budgets and show all non-separating probes charged.

No global predictive sufficiency follows from a finite set of probes. The contract establishes only response equivalence for its registered domains, interventions, queries and horizons.

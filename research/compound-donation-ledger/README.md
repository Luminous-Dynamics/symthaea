# Compound donation ledger — reference core

This is a standalone, dependency-free Rust research crate for Issue #9894.
It is intentionally outside the main Cargo workspace until the mathematical
contract and API stabilize.

## Scope

The crate implements two bounded pieces:

1. A fixed-point, append-only donation ledger. Each donor automatically
   reserves a precommitted fraction of its **own new adjusted-evidence
   increments**. Incoming donations cannot be re-donated. Reservations become
   source-tagged escrow tokens; tokens can be credited once (or split without
   exceeding the original amount) only to the donor's precommitted recipient
   slot, once that recipient is active and has positive weight.
2. A finite staircase calibrator. It evaluates the threshold-budget function
   A_pi(x) = sum_j pi_j K_j 1{x >= K_j} using integer arithmetic and a
   precommitted prior-mass budget.

The ledger receives already-calibrated, nondecreasing evidence values. It does
not compute likelihood ratios, test martingales, e-processes, null models,
p-values, or multiple-testing rejections.

## Accounting units

- Candidate weights are integer ticks on WEIGHT_SCALE; their sum may not
  exceed one.
- Evidence values are integer ticks on EVIDENCE_ONE; EVIDENCE_ONE represents
  the value one.
- Weighted capital is weight_ticks * evidence_ticks.
- Public scores are stored as weighted scores gamma_i * C_i, so the
  implementation never divides by a candidate weight.
- Reserve rates are fixed rational values in each candidate specification.
  The rate remainder is carried forward to avoid systematic per-round
  truncation. Outgoing capital is additionally capped by
  weight * min(adjusted_evidence, 1).

A reserve is generated deterministically on an evidence update. This makes
the donor's withdrawal path independent of the currently declared future
recipient roster. Each donor's destination slot is also committed in its
candidate specification before evidence is observed. If that destination is
outside the current horizon or remains dormant, the token stays in escrow.
Routing never rewrites a previous donor withdrawal or recipient credit.
This fixed-destination rule is the exact projective reference policy; adaptive
recipient selection remains a separate research problem.

## Event order and predictability

Activation events must be supplied before evidence updates for that candidate.
The ledger checks event order and monotone round numbers, but a serialized
event log cannot prove that a real-world arrival decision was genuinely made
before an observation. Production use must bind each round's roster to the
pre-observation arrival-policy receipt and source-event attestation; see
Symthaea Issue #9809. This crate does not authenticate timestamps or source
completeness.

## Replay and trust boundary

verify_receipt independently reconstructs all derived reserve events from the
policy and evidence-update events, checks that the complete supplied audit
log equals the reconstructed log, and optionally compares the exact terminal
snapshot. It detects edits, omitted/duplicated reserve records, invalid
credits, backward routing, overdraws, and inconsistent final state.

This is deterministic replay, not a digital signature, Merkle proof, source
attestation, or proof that the evidence itself is statistically valid. Those
layers require separate authenticated envelopes and independently verified
global-filtration evidence inputs.

## Run

From this directory:

~~~sh
cargo test --offline
~~~

No external crates are required. The crate is not yet added to the workspace,
and no repository CI result is claimed by its existence.

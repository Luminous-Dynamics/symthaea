# Symthaea pre-label priority-to-q audit policy v1

Status: research implementation; exact-head Rust qualification pending.

## Why add a q-policy

SOWRC requires a conditional audit inclusion probability q_t known before the current label, with a deterministic positive lower bound q_min. Supplying arbitrary q_t values is sufficient as a data interface, but it leaves the actual allocation policy as an opaque external assertion.

This increment adds one optional deterministic policy mapping a pre-label priority score to an inclusion probability. The priority score may summarize reviewer workload, domain scarcity, model uncertainty, novelty, risk class or previous-label history, but it must be computed by the independent audit custodian from information available before the current hidden outcome.

## Exact representation

Use a dyadic numerator n over denominator 2^64:

`q = n / 2^64`.

The numerator is restricted to multiples of 2^11 so the exact binary64 value is:

`q_f64 = (n >> 11) / 2^53`.

The allowed numerator range is 2^11 through 2^64 inclusive. The endpoint 2^64 is certainty. The selection decision uses only integer arithmetic:

`selected = (random_word_u64 as u128) < n`.

No floating-point comparison decides the random draw.

## Linear pre-label priority policy

Let p be an integer priority score in [0,2^53]. Let q_min and q_max be frozen dyadic endpoints satisfying 0 < q_min <= q_max <= 1.

The raw interpolation is:

`q_raw = q_min + (q_max - q_min) * p / 2^53`.

The implementation rounds the numerator down to the nearest multiple of 2^11. Since q_min is already on this grid, the result can never fall below q_min. It is monotone in p and exactly represented by binary64.

The per-event record carries:
- the exact pre-label history digest;
- the priority score p;
- the resulting q_t;
- the custodian random-stream index.

The receipt preserves the priority score so the q-policy can be replayed later. The receipt is custodian-private because it also contains selected flags; it must never be sent to the candidate researcher.

## Why this is safe only under an external boundary

The mapping is mechanically replayable, but the module cannot prove:
- the priority score used only pre-label information;
- the score did not encode or leak the current hidden outcome;
- the custodian's random u64 word is uniform and independent;
- selected label requests yield an independent observed label;
- conditional ignorability holds.

The q-policy ID/version, interpolation endpoints, priority-score source, pre-label history manifest, random-stream identity and q_t schedule must be committed by an independently audited design manifest. A selected request with no completed independent label is a second observation mechanism and is not silently counted as an observed label.

## Relationship to other audit designs

- The fixed-quota stratified SRSWOR design (#14101) provides pi_i and pi_ij for design-based finite-population estimation. It must not be substituted directly for predictable q_t.
- The external prelabel schedule mode remains available when the custodian uses a more sophisticated audited policy.
- The linear priority policy is an explicit deterministic option, not a claim that linear allocation is optimal.
- The candidate remains blind to random words, selected flags, and audit labels until predictions are frozen.

## Tests authored

The production module's 14 tests include exact dyadic conversion and Bernoulli thresholds, q=1 certainty, positive q_min, history and manifest digests, sequence and identity checks, raw-draw non-disclosure, history-varying externally supplied q, deterministic linear q monotonicity, exact event policy replay, mismatch rejection, invalid score/range checks, and fail-closed conversion to SOWRC observations. They are authored and imported by path into the focused Rust 1.96 harness; no pass is claimed until exact-head qualification.

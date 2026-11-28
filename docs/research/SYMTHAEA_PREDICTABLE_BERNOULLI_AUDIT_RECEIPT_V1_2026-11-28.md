# Symthaea predictable Bernoulli audit receipt v1

Status: research implementation; exact-head Rust qualification pending.

## Why this design is separate from SRSWOR

The fixed-quota stratified audit (#14101) is simple random sampling without replacement within frozen strata. It yields first- and second-order inclusion probabilities for design-based point estimates and variance.

SOWRC instead assumes a predictable per-event conditional inclusion probability q_t, known before the current label and bounded below by q_min>0. q_t may depend on recorded pre-label history. The marginal pi_i from a fixed-quota sample is not, without a separate sequential derivation, the predictable q_t required by SOWRC.

This module implements an eventwise Bernoulli audit request receipt, with the random stream and decision kept in custodian custody.

## Exact probability representation

Represent q_t by an integer numerator over denominator 2^64:

`q_t = numerator / 2^64`.

To preserve exact f64 representation, numerator must be a multiple of 2^11. Therefore:

`q_t = (numerator >> 11) / 2^53`.

The numerator is restricted to 1 through 2^64 inclusive. The upper endpoint encodes certainty. Given an independent uniform random u64 word U_t, select the audit exactly when:

`U_t < numerator`.

No floating-point comparison is used for the Bernoulli decision, and the returned f64 q_t is exact on the permitted dyadic grid. The probability floor q_min is represented using the same dyadic type.

## Custodian event receipt

Every event records:
- contiguous sequence number and stable record ID;
- SHA-256 of the exact pre-label history used by the q-policy;
- an integer pre-label priority score on [0, 2^53];
- exact dyadic q_t;
- external random-word index.

The optional `LinearPrelabelPriority` q-policy maps the pre-label priority score to a frozen interval [q_min, q_max] and verifies the event q_t matches. This provides deterministic risk-stratified audit intensity while retaining q_min support. The score itself is a custodian-side pre-label value; the module cannot prove it excludes current-outcome leakage. The policy rule, score source and history digest must all be included in the audit manifest.

The custodian-only output records whether the audit request was selected, the inclusion probability, history digest and random-stream index. It does not serialize the raw random word. The receipt binds the audit-design and random-stream manifest references and SHA-256 digests.

The candidate researcher must not receive this receipt or selected flags. It can receive only ordinary task/evidence observations under the separately owned AgentView protocol, after the relevant candidate output has been frozen for evaluation.

## Predictability and external assumptions

The module validates that each event has a pre-label history digest and a q_t not below q_min. It cannot prove the q-policy function was chosen from only pre-label history. A custodian/reviewer must replay q_t against the committed policy and history manifests and verify the random stream is independent of the current hidden label.

The Bernoulli selection rule is exact conditional on the supplied uniform u64 word. The module does not generate entropy. The random words must come from an independent cryptographically secure custodian, whose stream manifest is retained.

The q_t values represent the probability that an independent label is actually observed, not simply the probability that an adjudication request was dispatched. The current adapter requires every selected event to have one complete independent nonconformity score. Selected-label nonresponse is a second selection mechanism; it must be modelled before SOWRC can be applied. This module fails closed rather than treating nonresponse as random.

## SOWRC adapter

The custodian-only receipt can be combined with independent adjudication scores to produce the ordered SOWRC input:
- selected events require exactly one finite score in [0,1];
- nonselected events must have no score;
- score IDs must be known and unique;
- q_t is passed into the SOWRC observation as the eventwise conditional inclusion probability;
- the order and record IDs are preserved exactly.

The adapter cannot authenticate the label adjudicator or prove conditional ignorability. Those remain independent custody obligations.

## Tests authored

14 production-source tests cover dyadic probability exactness, half-probability selection boundaries, q=1 certainty, invalid dyadic numerators, q_min enforcement, manifest/history digest validation, sequence and duplicate-ID protection, non-disclosure of raw random words, history-adaptive q_t, and successful/failing SOWRC adapter cases. They are imported by path into the Rust 1.96 standalone harness and remain authored until exact-head workflow qualification.

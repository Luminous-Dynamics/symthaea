# Auditable pre-observation replay for predictable random arrivals

**Status:** research implementation on top of the countable global-projective e-closure branch. The ledger checks a declared simulator/arrival policy and event order. It does not prove real-world timestamp truth, source authenticity, filtration completeness, or stochastic e-process validity.

## 1. Why nested paths are not enough

The global-weight affine certificate has a pathwise dormant-one identity for nested active sets:

\`\`\`
Z_t^{N(P),w} = E_t^{N(P) intersect I_t,w}.
\`\`\`

That identity holds for any nested path. But a finite path oracle that receives \`I_t\` as an input does not establish that \`I_t\` was chosen predictably. A caller could supply a nested sequence selected after seeing the current-round score and still pass an ordinary nesting check.

The missing computational contract is event order:
1. decide the active set using only information available through t-1;
2. commit the decision;
3. expose current-round observations for the decided active set;
4. update the history used for the next arrival decision.

The policy's measurability is a mathematical assumption. This ledger makes the corresponding simulator rule explicit and replayable.

## 2. Pinned arrival policy

The first policy is intentionally simple and completely deterministic:

\`\`\`
A_t = A_(t-1) union { next_id }
\`\`\`

where \`next_id\` is activated only when the most recently active coordinate has at least k prior observations and its pre-round sample mean is strictly greater than a fixed threshold. At most one new coordinate activates per round. Active sets are nested and IDs have a fixed canonical order.

The policy's executable contract is hashed in the ledger header. Parameters include initial active count, minimum prior observation count and threshold encoded with canonical float hex. The event records:
- round number;
- \`decision_phase=PRE_OBSERVATION\`;
- \`observation_phase=POST_DECISION\`;
- active IDs before, newly activated and active IDs after;
- digest of all previously observed coordinate histories;
- decision-input and decision-output digests;
- previous event hash;
- current-round observations for exactly the active IDs;
- current event hash.

All score values are finite and encoded via canonical \`float.hex()\`. The verifier recomputes the history digest and expected policy decision from its own replay state before accepting any current-round score. It then validates the observation roster, recomputes the event hash and appends the new scores to replay history.

## 3. Streaming source interface

The preferred builder is:

\`\`\`python
build_predictable_arrival_ledger_from_source(
    score_source,
    rounds,
    hypothesis_ids,
    initial_active_count=1,
    min_observations=5,
    mean_threshold=0.0,
)
\`\`\`

The callback is invoked as \`score_source(round_index, active_ids)\` **after** the decision for that round. It must return exactly the active IDs and their scores; the builder does not request dormant coordinates. The test callback records the active roster at invocation to verify the event order.

The matrix-based helper remains useful for deterministic negative controls. It can construct a hash-consistent but statistically invalid ledger where the current-round score affects activation. The independent verifier rejects that ledger because replaying the pinned policy from prior history produces a different activation.

## 4. Why same-round lookahead is mathematically unsafe

The issue is not merely a bookkeeping or hash-chain violation. Same-round adaptive activation can destroy the one-step e-factor expectation.

Let Z be standard normal under a null and let a process use the e-factor exp(Z-1/2), which has expectation one. Suppose the process is activated only after observing Z>0; otherwise the coordinate remains dormant and contributes one. The post-observation selector has expectation

```
E[ 1{Z <= 0} + 1{Z > 0} exp(Z - 1/2) ]
  = 1/2 + Phi(1)
  = 1.341344746... > 1.
```

So it is not an e-factor. By contrast, a decision made from F_(t-1) before Z_t is revealed can be conditioned on, and the ordinary conditional-mgf proof applies. The test suite includes this analytic negative control in addition to a hash-consistent event ledger whose activation is rejected by replay.

## 5. Verification and trust boundary

The verifier checks:
- exact schema and pinned policy-specification digest;
- finite canonical score encodings;
- contiguous rounds;
- PRE_OBSERVATION/POST_DECISION phase labels;
- active-before state and nested activation;
- previous history digest;
- decision input/output digests;
- replayed arrival decision;
- exact observed ID roster and ordering;
- event hash chain and final root.

Negative controls cover same-round lookahead, history-digest mutation, missing observations, reordered observations, event-chain tampering, altered policy specification, unexpected event fields, non-finite scores and wrong callback rosters.

**Important:** a hash chain alone does not prove when an external event really happened. An untrusted producer can hash fabricated timestamps or omit records. Production integration requires independently authenticated source-event receipts, trusted monotonic sequence/time anchors, an independently validated active-hypothesis manifest, and a global filtration policy that specifies data arrival/observation visibility. This prototype only certifies that its own declared in-memory replay is internally consistent.

The current implementation stores a full finite score row for its matrix helper. The streaming callback avoids fetching dormant coordinates, but the compact event ledger should ultimately replace full-row materialization for large event campaigns.

## 6. Bridge to global-projective closure

The test suite takes the verified active-set path and feeds it into the finite countable global-projective closure oracle:
- all hypotheses have stable IDs;
- never-yet-active coordinates are exactly one;
- a finite nonnegative algebraic coordinate path is constructed for the example;
- the full true-set certificate equals the active true-null restriction at every time;
- local affine closure constraints are checked with the oracle's exact O(n) separator.

This is an algebraic integration test only. The synthetic values in this fixture are not presented as statistically valid e-processes. The stopping-time proof requires each real coordinate process to be valid in the same global filtration.

The countable certificate argument remains:

\`\`\`
Z_t = 1 - sum_(i in N(P)) w_i + sum_(i in N(P)) w_i M_i,t.
\`\`\`

With fixed ex-ante weights summing to at most one, component e-processes in a common filtration and neutral-one dormant padding, nonnegative countable mixing plus Tonelli gives the candidate stopped-FDR certificate for every a.s.-finite global stopping time. The local closure family remains finite at each round because the active family is finite.

SupFDR is separate. Ordinary e-process validity does not imply expected path-supremum control. It still requires a common projectivity-preserving running-maximum adjuster or pathwise nondecreasing certificate, together with the corresponding all-time adjusted local constraints.

## 7. Files and tests

- \`validation/predictable_arrival_ledger.py\`
- \`validation/test_predictable_arrival_ledger.py\`

The tests include:
- valid lagged-policy replay;
- a same-round lookahead negative control with internally consistent hashes;
- streaming callback invocation after the active set is fixed;
- callback roster, history digest, event hash, schema and observation-order tampering;
- 100 deterministic randomized nested-arrival replays;
- exact full-true-set / active-true-set projective certificate equality;
- connection to the finite global-projective closure oracle.

Tests are authored but not executed from a connected repository checkout in this environment. No CI pass is claimed.

## 8. Next production-grade steps

1. Replace the matrix helper's full score-row input with a source-backed event stream and per-source signed receipts.
2. Bind the arrival-policy binary/source digest, configuration digest, schema version, clock/sequence attestation and active-roster manifest root into each decision receipt.
3. Add missing, duplicated, delayed, out-of-order and conflicting evidence events; specify whether late evidence is excluded, queued or incorporated in a later filtration time.
4. Define a signed monotonic checkpoint or append-only transparency log to prevent rollback and equivocation.
5. Add randomized conditional-e-process generators and show the common-filtration supermartingale property under the actual arrival mechanism.
6. Run full stopped-FDR comparisons against the deterministic-envelope track (#9648), countable global-projective track (#9695), countable prior mixture (#9684), and online e-closure / donation baselines.
7. Separately integrate a valid running-maximum adjuster and benchmark SupFDR. Do not reuse the stopped-FDR claim.

The paradigm shift is to treat predictable activation as an **auditable event protocol**, not a caller-supplied list of active sets. That makes a hidden statistical assumption visible and testable without pretending that software hashes replace theorem assumptions or trusted data provenance.

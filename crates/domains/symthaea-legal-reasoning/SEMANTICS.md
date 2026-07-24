# Semantic contract

This document is normative for the behavior of the dependency-free kernel.

## 1. Facts and rules

A legacy `Rule` has finite condition, exception, and conclusion atom strings.
A rule fires when every condition is present and no exception is present.
Initial facts are always retained.

## 2. Local stratification

The engine assigns every atom a non-negative integer stratum subject to:

- `stratum(conclusion) >= stratum(condition)`;
- `stratum(conclusion) > stratum(exception)`.

If no finite assignment satisfies those constraints, the theory is rejected as
`DerivationError::NonStratified`. This includes every dependency cycle that
contains an exception edge.

Evaluation proceeds from the lowest stratum upward. Within each stratum,
positive rules close to a fixpoint. Input rules are structurally ordered before
evaluation, and conclusions are admitted in canonical lexical order.

## 3. Consequences

For accepted theories:

- rule permutation does not affect the fact set or proof trace;
- adding duplicate rules does not affect entailment;
- adding rules irrelevant to a queried component does not affect that query;
- a derived exception is established before any dependent default is tested;
- every derivation terminates because facts can only be added from the finite
  set of rule conclusions.

## 4. Explanation

`try_derive_with_trace` records the stratum and every canonical rule supporting
a conclusion when it first enters the fact set. `try_why_not` reports directly
relevant rules together with missing conditions and active exceptions.

A `why_not` result is local, not a minimal counterfactual proof. It does not yet
search recursively for the smallest set of missing premises.

## 5. Deontic interpretation

The kernel derives permission from obligation (`O(a) -> P(a)`) but distinguishes:

- explicit permission;
- permission implied by obligation;
- prohibition;
- contradictory positive and negative support;
- absence of a determination.

The compatibility Boolean `is_permitted` is fail-closed: conflicted,
forbidden, and undetermined all return `false`.

## 6. Hohfeldian interpretation

`Jural::correlative` changes party perspective. A relational correlative swaps
holder and counterparty while preserving action and source. `Jural::opposite`
keeps the same parties and returns the contradictory position.

## 7. Legal context

`TemporalScope` uses inclusive bounds for one legal interval.
`TemporalDimensions` separately represents enactment, publication, legal
effectiveness, and applicability to underlying acts or events. Decision time and
event time remain explicit query inputs to temporal revision selection.

The kernel represents these dates but does not infer whether retroactivity is
lawful, whether publication was sufficient, or which jurisdiction-specific
choice-of-law rule should control. Those judgments must be formalized upstream.

## 8. Profile boundaries and excluded semantics

The crate exposes several named and deliberately different profiles:

- the compatibility calculus accepts only locally stratified exception theories
  and rejects cycles through exceptions;
- `resolve_literal` resolves direct typed conflicts against an already-closed
  fact set and performs no recursive derivation;
- the recursive typed profiles use bounded whole-state grounded iteration with
  an explicit ambiguity-blocking or ambiguity-propagating policy.

No profile silently supplies natural-language interpretation, precedent
weighting, evidence authentication or sufficiency, jurisdiction-specific source
hierarchies, open-world negation beyond explicit `Literal` values, or a minimal
counterfactual explanation. Rejection or resource-limit failure means the
selected formal semantics did not produce a legal result; it is not itself a
legal answer.

## 9. Typed rule packs and superiority

A `FormalRule` has a stable ID, kind, canonical premise and exception sets,
explicitly signed conclusion, and optional source provision. `RulePack`
requires unique IDs and an acyclic superiority graph.

Priority is never inferred from slice order. Every direct priority relation
records a legal basis. Transitive reachability is used when determining whether
one rule outranks another.

## 10. Direct typed conflict resolution

`resolve_literal` considers only rules that are directly applicable to the
supplied fact set and conclude either the query literal or its explicit
opposite.

- Strict rules defeat contrary non-strict rules.
- Non-strict defeat requires an explicit superiority path.
- Non-strict rules cannot defeat strict rules.
- Defeaters may defeat contrary non-strict rules but cannot establish their own
  conclusion.
- Incomparable undefeated support on both sides yields `Both`.
- If defeaters remove all establishing support on both sides, the result is
  `Undetermined`.

This is skeptical direct resolution, not recursive typed derivation. No caller
should treat absence of direct support as proof of an opposite proposition.

## 11. Legal time

Legal effect and applicability to underlying events are separate inclusive
intervals. A revision governs only when it is effective on the decision date
and applicable to the event date. If multiple revisions govern and the caller
requests a unique revision, selection fails explicitly with `TemporalOverlap`.

Potential retroactivity is a review signal, not an automatic invalidity rule.

## 12. Norm lifecycle

Lifecycle assessment consumes already-formalized events occurring on or before
the query date. Waiver takes precedence over performance in the current profile.
Only events inside the norm's validity interval are considered. A matching
action fulfills an obligation, violates a prohibition, or exercises a
permission. An unperformed obligation becomes violated only after its explicit
deadline. Late performance is reported separately and does not erase the prior
violation signal. Same-day action and waiver events are temporally ambiguous
when no finer ordering was formalized. A supplied reparation activates on
violation or late fulfillment.

The kernel does not decide whether an event legally counts as performance or
whether a waiver is valid.

## 13. Hohfeldian transitions

A legal-position state contains each assertion and its correlative. Opposite
positions for the same holder, counterparty, and action are rejected. A power
exercise is accepted only when its exact authorizing `Power` relation is held.
Mutation is transactional and preserves correlative closure.

The effects asserted by a power exercise are inputs; the kernel does not infer
them from legal text.

## 14. Canonical evidence

Canonical evidence is schema-tagged and length-prefixed. Evidence manifests bind
results to the crate version, named semantic profile, rule pack, and query.
Canonical bytes are suitable as inputs to external cryptographic hashing and
signing, but are not themselves a signature or security digest.

## 15. Validation

Rule-pack validation is advisory. Findings are deterministic and cannot alter
legal results. An approval workflow may independently choose to reject packs
containing warnings or errors.

## 16. Provenance-bearing facts

`FactBase` is a canonical set of `FactAssertion` values. Assertions classify a
literal as observed, stipulated, or assumed and may bind a source provision and
asserting authority. Multiple assertions for the same literal are retained.
Explicitly opposite assertions are also retained and exposed as conflict; the
fact layer does not authenticate or reconcile them.

## 17. Recursive typed grounded profiles

`infer` uses whole-state iteration. The immutable factual base is retained in
every round. All other conclusions are recomputed from the previous round's
state using the same strictness and superiority defeat semantics as direct
resolution. A stable repeated state is the final result.

`typed-grounded-blocking-v1` requires a premise to be supported without its
explicit opposite. `typed-grounded-propagating-v1` allows each supported sign to
satisfy premises independently. Under both profiles, any support for an explicit
exception blocks its rule.

A non-consecutive repeated state is `InferenceError::Oscillation`; no member of
the cycle is selected as the legal result. Configured round and literal limits
also fail explicitly. Rule iteration order, fact assertion order, and map
iteration order are not semantic inputs.

## 18. Recursive explanations and proof graphs

`explain_query` is a local final-state explanation. It reports supporting and
opposing rules plus missing premises, conflicted premises, active exceptions,
explicit defeats, and defeater-only status. It is not a minimal counterfactual
search.

`ProofGraph` contains initial assertion provenance, undefeated final rule
applications, premise-to-conclusion edges, and `ProofGuard` records for explicit
exceptions that each application required to remain absent. `slice_for` follows
transitive support backward with cycle-safe traversal. A proof graph explains
the selected formalization; it does not establish that a source, fact, or
formalization was legally correct.

## 19. Temporal typed inference

`infer_at` first selects exactly one `TemporalRevision<RulePack>` using decision
and event dates, then runs the named recursive profile. No governing revision is
reported as absence, while overlap and inference failure remain distinct errors.
`overlapping_revisions` conservatively reports revision pairs whose effective
and applicability intervals can govern the same query.

## 20. Explicit event sequence

Civil dates order events across days. Same-day action and waiver events remain
ambiguous unless the caller supplies a validated `EventOrder` with unique
within-day positions. Event IDs never imply chronology. An incomplete order
cannot silently break a same-day tie.

## 21. Sessions, deltas, and impact analysis

`EvaluationSession` applies a factual change to a candidate fact base, recomputes
the bounded grounded result, and commits only on success. Its delta is the
canonical set difference between accepted states. The implementation is not an
incremental optimizer.

`RuleDependencyIndex` provides conservative direct and transitive impact sets.
It may over-approximate affected rules and must not be used to skip evaluation
until an incremental algorithm has a separate equivalence proof and conformance
gate.

# EGDDM ShiftBench D6.10 — Partial-order histories, branch joins and causal confluence v0.1

**Date:** 2026-11-16  
**Status:** evaluator-only prototype; synthetic known-answer fixture; not independently adjudicated  
**Related work:** [Issue #12418](https://github.com/Luminous-Dynamics/symthaea/issues/12418), D6.8 multi-shift continuity, D6.9 exploit migration

## Research question

Can an engineering agent preserve justified knowledge when several events are concurrent, clocks are skewed, observations arrive late, work proceeds on divergent branches, and those branches later join?

D6.8 v0.1 deliberately scores a strictly ordered event sequence. D6.10 treats history as a partial order. An event timestamp is a fact about a clock, not proof that one event caused another. Causal order must come from explicit, independently grounded happens-before evidence.

## Paradigm shift

Move from **timeline replay** to **causal history plus branch-local epistemic state**:

1. Represent event history as a directed acyclic graph (DAG), not a timestamp-sorted list.
2. Keep event time, observation time, and candidate-visible availability distinct.
3. Preserve incomparable events as concurrent; do not impose arbitrary serialization for deterministic presentation.
4. Treat branch-specific receipts as scoped knowledge, not globally portable facts.
5. Treat a join as a new verification context. Requalify every obligation affected by either parent branch before claiming it current.
6. Separate commutativity from mere concurrent arrival: two events can be concurrent but non-commuting.

This is a benchmark/evaluator contract, not a claim that all real-world causality can be reconstructed.

## v0.1 candidate-visible contract

The candidate view contains events, branch ancestry, dependency dimensions, context digests, obligations, and evidence receipts. The evaluator-only overlay contains adjudicated/synthetic happens-before edges, commuting and non-commuting pairs, and the expected join-requalification denominator.

Each candidate trace declares:
- asserted happens-before edges;
- event pairs claimed confluent;
- one decision cutoff, visible-event set, current-obligation set, and receipt-use set per branch;
- fresh receipt IDs used to requalify each join.

All objects are closed: unknown fields are rejected. Event IDs, branch IDs, obligation IDs, and receipt IDs are unique. Missing parents, duplicate ownership, ancestry cycles, event-DAG cycles, invalid pair references, and inconsistent join-impact labels reject the bundle.

## Metrics

- **Causal relation fidelity:** reference/declared ordered pair counts, correctly ordered pairs, false serialization, missing order, and wrong direction.
- **False serialization:** an asserted order between events independently adjudicated as incomparable. This is a key anti-timestamp-overreach measure.
- **Missing order:** failure to assert a real happens-before relation. A benchmark should also report correct abstention on truly incomparable pairs.
- **Confluence precision/recall:** true, false, missed, and explicitly false non-commuting confluence claims.
- **Availability leakage:** backdated or otherwise late-arriving events/receipts used before their candidate-visible availability.
- **Branch contamination:** receipts used outside their branch ancestry, with branch-only scope violations, or across a context-changing join.
- **Join requalification completeness:** required, fresh, missing, and invalid requalification obligations per join.
- **Unsupported claims:** current obligations asserted without a qualified receipt; critical unsupported claims remain separately visible.

No composite score may trade a critical unsupported claim, future-evidence leak, false serialization, or branch-contamination event for speed.

## Synthetic known-answer controls

The fixture contains:
- two events with identical event timestamps but an explicit causal edge, which must remain ordered;
- disjoint dependency changes in two branches with different wall-clock timestamps, which remain concurrent and are marked commuting;
- overlapping policy changes in the same branches, which are concurrent but marked non-commuting;
- a backdated observation whose availability is later than an earlier branch decision cutoff;
- a branch-A receipt attempted on sibling branch B;
- branch-A and branch-B receipts attempted at their join, where the combined context requires fresh qualification;
- a conformant trace that requalifies all three join obligations and an adversarial trace that does not.

The expected output is stored beside the input fixture. The Rust tests compare the complete serialized report, not only a few headline values.

## Baselines and experiment design

Compare, at identical event sets and budgets:
- timestamp-sorted total order (with stable-ID tie-breaking);
- arbitrary total-order replay;
- explicit DAG replay;
- branch-isolated evidence with no join transfer;
- branch-inheriting receipt reuse without requalification;
- global reset/revalidate at every event and at every join;
- D6.8 linearization;
- independent expert/adjudicator reference.

A validity pilot should include 12 intervention families × 6 composition families × 5 independently authored seeds (360 tasks) only as a research target, after the pilot demonstrates label quality and candidate/evaluator separation. Add topological-order permutations for commuting pairs, non-commuting concurrent pairs, causal edges at tied timestamps, delayed/backdated observations, branch receipt attacks, missing ancestry, cycles, and common-cause invalidation.

## Proposed hard gates

- Zero causal edges inferred solely from event timestamps.
- 100% correct handling of deterministic DAG controls, including ties and late arrivals.
- Zero false-confluence claims on registered non-commuting controls.
- Zero branch-receipt contamination on registered critical negative controls.
- Zero future-evidence use before availability.
- Zero join claims for required obligations without fresh, merge-bound, context-matching receipts.
- Cycles and missing ancestry never yield a qualified result.
- No loss of legitimate capability: matched valid branch-local checks continue to pass.
- Report per-task outcomes and confidence intervals; do not collapse safety failures into a scalar leaderboard score.

These are acceptance targets for future validation, not empirical results.

## Synthetic fixture result

The deterministic fixture contains 9 events, 4 branches, 3 obligations and 7 receipts.

- Conformant trace: 30/30 reference ordered pairs recovered, zero false serialization, zero missing order; the commuting source/docs pair is recognized; all 3 join obligations receive fresh requalification; no availability leaks or receipt contamination.
- Adversarial trace: 2 false serializations, a false confluence claim for a non-commuting policy pair, one late/backdated event used before availability, two contaminated receipts, and two unsupported join claims (one critical).

These are expected synthetic scorer outputs, not model trials, independent causal adjudication or production assurance.

## Claim ceiling and next gates

D6.10 v0.1 only checks deterministic conformance to a supplied synthetic DAG, branch manifest, receipt inventory and adjudication overlay. It does not authenticate source events, prove registry completeness, establish physical concurrency correctness, qualify a production merge, or demonstrate SOTA performance.

Before external benchmark claims: compile and run Rust tests; validate both schemas with an independent Draft 2020-12 engine; create independent DAG labels; test two or more implementations against exhaustive topological-order controls; run blinded matched-agent trials; publish cost and latency distributions; and obtain external replication.

# Symthaea Consolidation Governance Bench v0.1

**Status:** Exploratory deterministic synthetic harness. Not a production memory controller, not a public benchmark score, and not evidence of real-world transfer.

**Owner:** Symthaea issue [#9139 — govern consolidation, promotion, compression, and forgetting](https://github.com/Luminous-Dynamics/symthaea/issues/9139)

## Why this exists

The first executable slice in #9139 asks whether memory lifecycle decisions can be tested independently of the live cognitive loop. This harness creates a small, deterministic synthetic corpus with evaluator-owned outcomes and compares simple memory policies.

It deliberately does **not** modify `symthaea-memory`, create a new persistent memory store, or introduce a second provenance/epistemic authority. It is a research fixture and evaluator only.

## Run

This is a standalone, standard-library-only Rust package. It is deliberately not a member of the large Symthaea Cargo workspace.

```bash
cargo run --manifest-path research/consolidation-governance-bench-v0-1/Cargo.toml
cargo test --manifest-path research/consolidation-governance-bench-v0-1/Cargo.toml
```

The executable prints two deterministic CSV sections: aggregate policy metrics and per-case governed-policy decisions. No network, model API, database, runtime memory, or benchmark service is required.

## Fixture composition

The v0.1 corpus contains eleven cases:

1. Three independent observed successes that support a reusable procedure.
2. Eight correlated duplicate episodes from one source family (frequency is not independence).
3. A poisoned, contradictory, unverified episode.
4. A domain-shift case where prior success must not become a universal rule.
5. A rare safety-critical detail that lossy compression would destroy.
6. A source retraction that requires supersession.
7. A privacy erasure request that must remove derived descendants.
8. An under-supported candidate that should remain provisional.
9. A safe compression case.
10. Five independent supports for a reusable rule.
11. A non-load-bearing item that may leave the active index without deleting its source payload.

The candidate policy API receives only `Observation`. Evaluator-only `OracleTruth` is a separate type and is only consumed by the scoring/oracle functions. This is an architectural anti-leakage guard in the harness, not a substitute for a sealed external evaluator.

## Compared policies

- `raw_only`: never consolidates.
- `eager_compression`: compresses every case.
- `frequency_only`: promotes after three episodes, regardless of provenance independence.
- `source_diversity_only`: checks source-family diversity but ignores poison and distribution-shift controls.
- `governed`: applies erasure, retraction, contradiction/trust, rare-detail, scope, independent-support and pre-registration gates.

The oracle is an evaluator-only upper bound. It is not a candidate policy and must never be made available to the evaluated agent.

## Metrics (kept separate)

- Exact agreement with evaluator oracle.
- Mean evaluator utility and mean decision regret.
- Aggregate storage units.
- Harmful promotions.
- Poisoned-case escapes.
- Rare safety-detail losses.
- Stale-source reuse.
- Missed privacy erasure.

Utility, compute/storage cost and governance are separate columns. There is no weighted composite score.

The harness also tests a minimal in-memory projection model:
- reversible consolidation projections can be restored exactly from a snapshot;
- `ForgetFromActiveIndex` does not erase source payload;
- `EraseForPrivacy` cascades through descendants and cannot be rolled back.

These are logical fixture invariants only. They do not establish physical secure erasure, filesystem behavior, database recovery, or Mycelix authorization.

## v0.1 expected properties

The governed policy is intentionally constructed to match the symbolic oracle on these eleven transparent fixtures. The unit tests assert zero known governance failures and zero decision regret on this fixture. Its 100% agreement in this tiny synthetic suite is a regression property of the fixture, **not** an empirical performance claim or evidence that the policy generalizes.

The less careful baselines are expected to expose distinct failures:
- frequency-only promotion should fail on correlated duplicates, poison, domain shift, stale sources, rare details and privacy erasure;
- source-diversity-only should still fail when it ignores contradiction/trust and scope;
- eager compression should lose rare details and miss governance actions;
- raw-only preserves information but pays the largest storage cost and ignores lifecycle/erasure decisions.

## Limitations and next gates

- Eleven hand-authored cases are insufficient for statistical claims.
- The observation fields are idealized and do not yet consume the canonical bitemporal/evidence types from #9075, #8265, #6481 or #9177.
- The evaluator is co-located for easy inspection; a serious benchmark must separately build, seal and control its oracle.
- The policy is deterministic and hand-authored; there is no learned consolidation model.
- Utility values are synthetic ordinal scores. They are not calibrated user utility or downstream task success.
- Storage units are relative fixture costs, not measured bytes, latency, energy or monetary cost.
- The transaction projection models logical deletion only; secure erasure needs storage-level qualification.

Next steps should be owned by #9139 and its related work: grow source-held-out scenario families, add paired real task transfer, enforce bitemporal semantics, inject hidden poison/late corrections, test distribution shift, add independent evaluator custody, and measure total lifecycle cost (ingestion, maintenance, consolidation, retrieval, rollback and privacy operations).

## Related canonical owners to compose with

- [#9075](https://github.com/Luminous-Dynamics/symthaea/issues/9075): bitemporal source-event ledger.
- [#8265](https://github.com/Luminous-Dynamics/symthaea/issues/8265): claim identity versus HDC similarity.
- [#9177](https://github.com/Luminous-Dynamics/symthaea/issues/9177): proof-aware minimal sufficient context.
- [#8445](https://github.com/Luminous-Dynamics/symthaea/issues/8445): provisional-to-durable consolidation benchmark.
- [#9016](https://github.com/Luminous-Dynamics/symthaea/issues/9016): subject, authority and audience boundaries.
- [#11816](https://github.com/Luminous-Dynamics/symthaea/issues/11816): benchmark oracle independence.

Do not add a competing memory store, evidence graph, receipt schema, policy engine, or global truth score.

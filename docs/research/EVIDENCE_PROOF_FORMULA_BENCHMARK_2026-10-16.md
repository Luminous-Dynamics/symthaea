# Symthaea Alternative and Joint Evidence Proofs

**Date:** 2026-10-16  
**Status:** bounded pure-Rust proof-formula prototype, unit tests, and opt-in scaling benchmark authored; not connected to production retrieval; no test/performance results claimed  
**Implementation:** src/knowledge/evidence_proof_formula.rs

## Paradigm shift: evidence as a derivation, not a vote

A flat list of sources or root IDs loses an important distinction:

- **Joint evidence:** one explanation requires A and B.
- **Alternative evidence:** A or B provides separate sufficient derivations.
- **Redundant derivations:** A alone suffices, so a larger A+B proof is not minimal.
- **Shared dependencies:** two articles may look separate but both depend on A upstream.

Database provenance work offers algebraic models for representing why a query result exists. Green, Karvounarakis, and Tannen's 2007 paper, *Provenance Semirings*, uses semiring annotations to unify several forms of database provenance. The present contract is a bounded minimal-witness evaluator, not a semiring implementation, probability model, or truth calculus. [Paper](https://www.cs.ucdavis.edu/~green/papers/pods07.pdf).

A recent 2026 paper on smallest witnesses for conjunctive queries formalizes the cost/complexity of finding minimal sub-databases that preserve a query result. Its setting is database queries without self-joins, not Symthaea retrieval, but it is directly relevant to the need to measure witness minimality and expansion cost rather than assuming explanation extraction is free. [Paper](https://doi.org/10.1145/3786785).

## Typed contract

`EvidenceProofFormula` is an explicit expression:

- `Evidence(id)`: one stable, claim-specific evidence witness.
- `All(children)`: all children are jointly required for this derivation.
- `Any(children)`: each child is an alternative derivation.

The evaluator emits a canonical antichain of **minimal witness sets**: no returned set is a strict superset of another returned set. Witness atoms are claim-specific evidence-item IDs, not URLs or publisher IDs. After evaluation, each atom can be linked to the source-lineage graph to inspect origin dependencies. Distinct formulas and witness sets are not truth scores.

Examples:

| Formula | Minimal witness sets | Meaning |
|---|---|---|
| A AND B | {A, B} | Both witnesses are required |
| A OR B | {A}, {B} | Either route can support the derivation |
| A OR (A AND B) | {A} | The larger route is redundant |
| (A AND B) OR (A AND C) | {A, B}, {A, C} | Two alternative explanations share dependency A |

## Fail-closed resource contract

Alternative expansions can grow exponentially. The evaluator therefore validates node count and depth before evaluation, then bounds the number of incomparable witness sets and intermediate conjunction combinations. Invalid empty connectors and blank witness IDs are rejected. Exceeding any bound returns an abstention with **no partial witness-set list**, preventing a truncated result from masquerading as the complete explanation. The witness-frontier cap applies during evaluation, not only to the final minimized set: an expression can abstain if its intermediate frontier exceeds the cap even if a later alternative could reduce it. This conservative order-sensitive resource guard must be included in coverage measurements; canonical complete outputs, however, should be invariant to alternative ordering.

Defaults are implementation resource limits (200,000 expression nodes, depth 128, 1,024 minimal witness sets, and 200,000 intermediate combinations). They are not empirically tuned; benchmark them and retain explicit per-query overrides. The evaluator validates depth iteratively before a depth-bounded recursive pass. Conjunctions whose children each have one witness use a specialized union path to avoid repeatedly cloning an ever-growing set.

## Adversarial tests

| Case | Required behavior |
|---|---|
| Joint A AND B | One witness set containing A and B |
| Alternative A OR B | Two singleton witness sets |
| A OR (A AND B) | Remove the non-minimal superset |
| Repeated A AND A | Same atom does not appear twice in a set |
| Nested AND/OR | Stable, sorted, deduplicated minimal antichain |
| Empty AND or OR | Abstain; do not use vacuous truth as evidence |
| Blank evidence identity | Abstain |
| Expression node or depth excess | Abstain before evaluation |
| Too many witness sets | Abstain with no partial result |
| Cartesian-product explosion | Abstain with no truncated explanation |
| Large conjunction | One complete witness set with all unique atoms |

The module tests these cases and contains an ignored release-mode scaling probe for 1k, 10k, and 100k atom conjunctions. No runtime measurement is claimed.

## Proposed benchmarks and gates

These are targets, not measured Symthaea results.

| Metric | Proposed target |
|---|---:|
| Conjunction/disjunction semantic regression cases | 100% correct |
| Returned witness sets containing a strict superset of another returned set | 0 |
| Duplicate witness atom counted twice in one set | 0 |
| Budget overflow returning a partial explanation | 0 |
| Complete results that differ across repeated pinned executions | 0 |
| 100k atom conjunction | Exactly one complete witness set; no recursion overflow |
| Output witness frontier | p50/p95 at 1k, 10k, 100k atoms; exact release environment |
| Cartesian explosion | Deterministic abstention at configured budget |
| Gold witness family false split/merge | Measure on independent labeled families before setting a production threshold |
| Answerable task coverage vs. abstention | Always report both; never optimize zero unsupported completions alone |

### Reproducible scaling probe

Run explicitly in release mode:

    cargo test --release --lib knowledge::evidence_proof_formula::tests::benchmark_minimal_witness_frontier_scaling -- --ignored --nocapture

The expression is built outside the timed region. Each timed assessment includes input validation and full minimal-witness evaluation. Report the exact commit, Rust version, feature set, hardware, p50/p95, allocation behavior if available, and whether the result is complete. This probe uses an all-of conjunction; it does not measure worst-case disjunction/frontier explosion, which is evaluated by deterministic budget tests. It also excludes retrieval, source-lineage extraction, and persistence I/O.

## Integration sequence

1. Land the formula contract and benchmark independently of CI for the prior provenance graph.
2. Build gold formulas for claim-specific cases where annotated proofs distinguish joint and alternative support.
3. Connect formula leaves to evidence IDs and then to the provenance derivation graph; count shared upstream roots without replacing the logical AND/OR structure.
4. Compare exact symbolic evaluation with any future HDC/probabilistic approximation. The approximation must never be the only mechanism enforcing resource bounds or proof identity.
5. Add bounded explanation APIs to query reports, preserving reason codes for expansion limits.
6. Only then evaluate end-to-end selective retrieval, calibration, source-shift robustness, and latency.

## Limitations

- Minimal witness sets represent syntactic derivation under the provided formula; they do not validate the extractor's correctness.
- Whether evidence atoms are logically required or alternative is itself a claim-specific annotation/inference task.
- Identical evidence IDs are treated as the same atom. Callers must use stable claim-specific evidence identity rather than a shared source URL.
- A set of different evidence IDs may still share one source lineage. Resolve lineage separately; do not equate witness-set cardinality with independent-source count.
- This module does not calculate Bayesian probabilities, truth, reliability, counterfactual causal effects, or proof certificates.
- Limits deliberately fail closed; the abstention/coverage tradeoff must be measured per domain.

## References

- Todd J. Green, Grigoris Karvounarakis, Val Tannen, [Provenance Semirings](https://www.cs.ucdavis.edu/~green/papers/pods07.pdf), PODS 2007.
- Xiao Hu, Stavros Sintos, [Finding Smallest Witnesses for Conjunctive Queries](https://doi.org/10.1145/3786785), ACM Transactions on Database Systems, published June 15, 2026.
- W3C, [PROV-O: The PROV Ontology](https://www.w3.org/TR/prov-o/), for entity derivation relationships (used by the adjacent provenance graph contract).

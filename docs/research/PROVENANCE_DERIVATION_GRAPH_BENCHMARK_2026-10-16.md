# Symthaea Explicit Provenance Derivation Graph

**Date:** 2026-10-16  
**Status:** typed in-memory graph and tests authored; not integrated with persistent storage or production retrieval; no tests or performance measurements claimed  
**Implementation:** src/knowledge/evidence_provenance_graph.rs

## Why this next step matters

Flat origin identifiers help deduplicate known syndication, but real evidence is not always a single source -> one article. A later article can revise an earlier one; a summary can quote a report; a technical conclusion can synthesize multiple papers and a measurement. Counting a derived document as one new independent source loses those dependencies. Assigning it one origin can also destroy legitimate multi-parent lineage.

The shift is from **source labels to explicit derivation structure**. Record evidence as entities/revisions, record directed derivation edges, validate the graph, and resolve the terminal attributed origins for the evidence set used for one claim. Keep unresolved terminals visible rather than inventing their ancestry.

## Standards and theory

The [W3C PROV-O ontology](https://www.w3.org/TR/prov-o/) defines general derivation plus more specific relationships including quotation, revision, and primary-source relationships. The [W3C PROV Constraints Recommendation](https://www.w3.org/TR/prov-constraints/) describes validity checks for provenance records, including uniqueness and event-order consistency. Symthaea's contract borrows the entity/derivation concepts and uses cycle rejection as an explicit fail-closed graph-consistency policy; it does not claim complete PROV conformance.

Database provenance research offers a further paradigm shift. Green, Karvounarakis, and Tannen's *Provenance Semirings* (PODS 2007) shows how algebraic annotations can represent why query outputs exist and how multiple derivations combine. This motivates a future distinction between alternative derivations and evidence that is jointly required. The current graph reports upstream origin sets and preserves derivation edges; it does not yet implement semiring evaluation, probability, truth, or independence scoring. [Paper](https://www.cs.ucdavis.edu/~green/papers/pods07.pdf).

## Typed graph contract

- Nodes have stable entity/revision IDs and one declared kind: explicitly attributed primary origin, derived artifact, or unattributed terminal.
- Edges state which upstream entity a derived entity was derived from, using a typed relation: general derivation, quotation, revision, or declared primary source.
- Every derived artifact requires at least one known parent.
- Primary-origin and unknown-origin nodes are terminal in this initial model; a node classified as either but with parents is a role conflict.
- Missing endpoints, duplicate IDs, empty IDs, malformed graph structure, and cycles cause abstention.
- The graph is traversed without recursion, so a long derivation chain does not consume the runtime call stack.
- The origins of multiple selected evidence records are unioned; shared ancestors are counted once.
- Unknown terminal nodes are never counted as origins. A policy may require every terminal lineage to resolve or permit review when enough known roots exist while still returning unknown terminal IDs.
- Even two explicitly attributed origins are not proof of independence, trustworthiness, or truth. Those are separate, domain-specific judgments.

## Benchmark suite

| Case | Synthetic construction | Expected result |
|---|---|---|
| G0 | Two selected records descend from two distinct attributed roots | Two roots, eligible under the two-origin policy |
| G1 | Ten secondary articles descend from one primary report | One distinct root; abstain for multi-origin policy |
| G2 | One synthesis has two parent sources | Both upstream roots retained |
| G3 | Several selected articles have overlapping paths to the same root | Count the root once |
| G4 | A synthesis descends from a known root and an unknown terminal | Strict policy abstains and names the unknown terminal |
| G5 | Parent edge references a missing entity | Abstain with missing-source reason |
| G6 | Directed cycle, including revision cycles, even in a disconnected component | Abstain for the invalid graph snapshot |
| G7 | Duplicate/empty entity IDs or role/edge incompatibility | Abstain |
| G8 | Duplicate requested evidence IDs | Report raw and distinct requested counts separately; origin count unchanged |
| G9 | A valid graph contains a very deep chain | No recursion overflow; measure memory and latency |
| G10 | A graph has many irrelevant components and one valid query path | Current implementation globally validates; quantify the cost and design a validated-snapshot strategy before production |

G10 is a deliberate performance/architecture question. Global validation protects against quietly consuming an invalid provenance document, but can make each query proportional to the whole graph. The next production design should use a persisted validation/version receipt or incrementally maintained graph invariants—not skip validation without a proof that the relevant snapshot is valid.

## Metrics and proposed gates

Targets below are hypotheses, not measured results.

| Metric | Proposed target |
|---|---:|
| Shared-origin citation inflation | 0 additional root origins |
| Multi-parent root preservation | 100% on deterministic fixtures |
| Duplicate shared ancestors | 0 double-counting |
| Dangling-edge, duplicate-ID, role-conflict, and cycle escape | 0 |
| Unknown terminal roots credited as attributed | 0 |
| Deterministic fixture decision/root parity | 100% |
| Lineage graph larger than call stack | No recursion overflow at 100k nodes |
| Graph-validation/root-resolution runtime | Record p50/p95 at 1k, 10k, 100k evidence items (each item has a primary-origin node and a derived-evidence node) |
| Source-family false-split / false-merge | ≤1% pilot targets only after gold lineage annotation |
| Origin extraction precision and recall | Report separately from this graph contract |
| Audit replay | 100% identical root set and decision from pinned graph/policy/input |

The ignored release-mode benchmark creates 1k, 10k, and 100k primary-origin/evidence pairs (2x entities, 1x edge per pair), checks root-count parity, and reports p50/p95 runtime across seven measured repetitions. It includes graph construction only outside the timing loop; each timed assessment rebuilds indexes, validates the graph, and resolves selected evidence roots. It does not include provenance extraction, database I/O, cryptographic verification, or network access.

Run explicitly:

    cargo test --release --lib knowledge::evidence_provenance_graph::tests::benchmark_lineage_graph_scaling -- --ignored --nocapture

No latency result is claimed before the benchmark is run against a recorded commit, compiler, feature set, and hardware.

## Next experiments

1. Compare raw citation count, unique URL count, flat origin-ID grouping, and explicit derivation graph on the same gold lineage families.
2. Add domain-specific extraction of PROV-style derivation edges with calibrated unknown labels. Report edge precision/recall and root-set precision/recall.
3. Add claim-specific source support edges; do not infer that a source is relevant to every proposition it mentions.
4. Represent alternative derivations separately from jointly-required provenance. Explore semiring-style annotations while keeping truth and source independence separate.
5. For production scale, replace rebuild-on-query indexing with versioned graph snapshots or incremental consistency receipts; retain a fail-closed response for stale/invalid receipts.
6. Integrate this graph with the provenance gate only after canonical source/status ownership and persistence migration semantics are decided.

## References

- W3C, [PROV-O: The PROV Ontology](https://www.w3.org/TR/prov-o/), including derivation, quotation, revision, and primary-source relations.
- W3C, [Constraints of the PROV Data Model](https://www.w3.org/TR/prov-constraints/), provenance graph consistency.
- Todd J. Green, Grigoris Karvounarakis, Val Tannen, [Provenance Semirings](https://www.cs.ucdavis.edu/~green/papers/pods07.pdf), PODS 2007.

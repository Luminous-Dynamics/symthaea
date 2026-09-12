# symthaea-evidence-dependency

Acyclic provenance and anti-circularity assurance for safety evidence.

The core invariant is:

> evidence used to justify a safety decision must not ultimately depend on that same safety decision.

The graph records typed evidence nodes and explicit dependency edges. It rejects:

- malformed or duplicate nodes
- malformed, duplicate, self, or dangling edges
- dependency cycles
- a receipt/qualification for safety contract `X` that transitively depends on a readiness decision for contract `X`

The same-contract back-dependency check is transitive, so hiding a readiness dependency behind a test artifact, verifier, or intermediate qualification does not make it acceptable.

Cross-contract dependencies are not automatically rejected in this tranche, but remain explicit and auditable in the graph.

This crate performs assurance analysis only. It never discharges a safety obligation or grants physical authority.

```bash
cargo test -p symthaea-evidence-dependency
```

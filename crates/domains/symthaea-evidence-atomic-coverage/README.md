# symthaea-evidence-atomic-coverage

Atomic sub-claim coverage assurance for safety evidence.

A broad safety obligation can contain several independently meaningful assertions. Ordinary strict readiness intentionally treats the obligation as one controlled claim, so this layer adds an optional reviewed decomposition when a deployment needs stronger evidence accounting.

Each required facet has:

- a stable facet id
- the parent obligation key
- controlled facet text
- reviewed evidence references
- an explicit minimum number of distinct receipts
- an explicit minimum number of distinct facet evidence objects

Coverage is never inherited implicitly from the parent receipt. Each facet must be bound explicitly through a `FacetEvidenceBinding` to an exact sub-artifact reference and digest.

One receipt may support several facets only when each mapping is explicitly declared. Reusing one sub-artifact digest cannot satisfy a policy that requires multiple distinct evidence objects.

The gate is monotonic: it can retain or reduce ordinary strict readiness but cannot upgrade a blocked or invalid safety case.

This crate grants no physical authority.

```bash
cargo test -p symthaea-evidence-atomic-coverage
```

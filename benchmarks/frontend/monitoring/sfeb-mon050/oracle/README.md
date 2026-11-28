# MON-050 dependency-impact oracle

Standard-library-only Go reference implementation for deterministic claim-invalidation semantics.

Dependency edges point **from a dependent node to its dependency**. Only `required` edges propagate invalidation; `context` edges are intentionally excluded. Events are typed and must target nodes of the corresponding kind.

## Run

```sh
./CHECKS.sh
go test -run '^$' -bench BenchmarkImpactedClaims -benchmem -benchtime=1x
```

The frozen six-scenario corpus is in `../fixtures/SCENARIOS-v1.json`; its JSON Schema is in `../schema/DEPENDENCY-IMPACT-CORPUS-v1.schema.json`. The scenario test pins the exact corpus bytes with SHA-256.

The evaluator validates a typed DAG, node and edge types, edge uniqueness, endpoint closure, self-dependency and acyclicity. It computes reverse transitive closure from an event target over mandatory edges, then reports exact-set correctness, invalidation precision/recall, false invalidation and missed invalidation. Optional context edges do not propagate.

## Explicit limits

This is a deterministic graph-semantics oracle, not a distributed revocation service, production admission boundary or live recovery test. It does not establish authenticated event ordering, durable event storage, temporal lease evaluation, independent source-state completeness, production safety, or vendor certification. Smoke timings are not stable estimates or SOTA claims. `production_use_permitted` is always false.

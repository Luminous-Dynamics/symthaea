# MON-004 PostgreSQL outbox claim/fence bounded model

The exact tested source is stored as deterministic gzip:

- `mon004-postgres-claim-model-v1.mjs.gz`
- Uncompressed source SHA-256: `e459a3cf14cdff40a221a35e5d6d34672188c6a649acacac752e76797360346d`
- Compressed artifact SHA-256: `0ed8ff774a709ba2d6883356655cdd3fc9f4671eade8f96dd112fa42f792ad8b`

Run from the repository root:

```sh
gzip -dc benchmarks/frontend/monitoring/mon004-postgres-claim-model-v1.mjs.gz | node --check -
gzip -dc benchmarks/frontend/monitoring/mon004-postgres-claim-model-v1.mjs.gz | node -
```

The current executed result is **21/21 deterministic cases** and **5,040 bounded sequential operation orderings**, with zero duplicate mock effects and zero modeled invariant violations.

This is a pure in-memory model, not PostgreSQL execution, a concurrent memory model, network-partition qualification or a production reliability estimate. The SQL contract remains unexecuted until a pinned PostgreSQL runtime is available.

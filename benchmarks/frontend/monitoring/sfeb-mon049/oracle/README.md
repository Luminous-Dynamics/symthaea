# MON-049 — Evaluator challenge runner

A standard-library-only Go benchmark harness for the MON-048 v1 deterministic challenge manifest. It validates the pinned corpus, strictly parses candidate dispositions, reports false qualification / false blocking / indeterminate / decision-coverage metrics, and rejects uniform all-indeterminate, all-blocked and all-qualified strategies.

## Run

```sh
go test ./...
go vet ./...
go run . -manifest ../../sfeb-mon048/CHALLENGES-v1.json -candidate candidate-results.json -expected-manifest-sha256 <PRE-REGISTERED-SHA256> -out score-receipt.json
```

The expected manifest SHA-256 is supplied out-of-band and hashes exact manifest bytes. The runner rejects duplicate JSON member names, unknown fields, trailing values, duplicate/missing/extra candidate IDs, invalid dispositions, missing reason codes, and incorrect manifest cardinality.

This scores only the deterministic MON-048 corpus. It does not establish statistical reliability, run restores, verify custody evidence, authenticate source inventory, exercise production admission, or permit production use. `production_use_permitted` is permanently false. Dependency invalidation precision/recall and mutation operator coverage require a versioned manifest extension carrying explicit dependency graphs and operator identities.

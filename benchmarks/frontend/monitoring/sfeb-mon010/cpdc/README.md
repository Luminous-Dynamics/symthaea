# Claim Provenance and Drift Challenge (CPDC) — synthetic preflight

This is the deterministic CPDC v1.1 fixture set. It is a semantic qualification harness for MON-010 requirements, not an external benchmark and not an implementation of an Internet-Draft wire format.

## Run

```sh
go test -race -count=5 ./...
go vet ./...
go run . cases-v1.json
```

The corpus contains 15 adversarial cases covering publisher correction/retraction, independent dispute, unavailability/restoration, historical as-of evaluation, a pending action evaluated before a later correction, unrelated-source isolation, unit/population substitution, decision-byte mismatch, authority-free measurement, and failed source authentication.

## Baselines

- `citation-only`: a stable citation or URL with no correction, semantic or authority binding.
- `snapshot-only`: immutable source snapshot plus current-state and separate-decision checks, but no downstream correction propagation.
- `lineage-aware`: source state plus semantic identity, bitemporal query semantics, dependency impact, reauthorization, immutable executed-action history, and independent risk review.

These are explicit policy baselines, **not** measurements of DBOS, Temporal, Rekor, SCITT, or another product. Product adapters must be implemented and run against the same fixture corpus before any industry comparison is claimed.

## Metrics

`false_current`, `false_stale`, `missed_reauthorizations`, `unauthorized_allows`, `missed_current_risk_reviews`, and `historical_as_of_errors` are safety metrics (lower is better). `exact_expected_decisions` is a fixture exact-match count (higher is better). The lineage-aware policy is expected to score 15/15; this checks the harness and encoded rules, not an independent system.

## Limits

No cryptographic publisher-signature, RFC 9942/9943, SCITT, COSE, external source retrieval, trusted timestamp, or live action adapter is implemented here. The corpus is synthetic and intentionally small. The next step is to pin official profile vectors and independently authored adversarial cases, then run matched adapters and mutation tests.

# Claim Provenance and Drift Challenge (CPDC) — synthetic preflight

This is the first executable, deterministic CPDC fixture set. It checks MON-010 semantic requirements; it is not an external benchmark or an implementation of an Internet-Draft wire format.

## Run

```sh
sh reproduce.sh
```

The corpus contains 14 adversarial cases: publisher correction/retraction, independent dispute, unavailability/restoration, historical as-of evaluation, unrelated-source isolation, unit/population substitution, approval-byte mismatch, authority-free measurement and failed publisher authentication.

## Baselines

- `citation-only`: stable URL/citation without correction, semantic or approval binding.
- `snapshot-only`: immutable source snapshot plus current-state and separate-decision checks, but no downstream correction propagation.
- `lineage-aware`: source status, semantic identity, bitemporal evaluation, dependency impact, reauthorization, immutable executed-action history and independent current-risk review.

These are explicit policy baselines, **not** measurements of DBOS, Temporal, Rekor, SCITT or another product. Product adapters must run against the same corpus before any industry comparison is claimed.

## Metrics

`false_current`, `false_stale`, `missed_reauthorizations`, `unauthorized_allows`, `missed_current_risk_reviews` and `historical_as_of_errors` are safety metrics (lower is better). `exact_expected_decisions` is a fixture exact-match count (higher is better). The lineage-aware policy is expected to score 14/14; that validates these encoded rules, not an independent system.

## Limits and next work

No publisher-signature verifier, RFC 9942/9943, COSE, SCITT, external source retrieval, trusted timestamp or live action adapter is implemented. This small synthetic corpus must be expanded with official profile vectors and independently authored mutations. Then compare maintained profile adapters and product integrations under the same event traces, faults and resource budgets.

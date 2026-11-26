# MON-020 event-order oracle

Standard-library Go semantic reference for qualifying required event order in a provenance history.

It verifies normalized reports for source identities, event records and cross-source order witnesses against pinned verifier ID/version/build/trust-store values. It derives source-local order from authenticated strictly increasing source sequence values. For cross-source requirements, it requires an exact-event-bound witness with a control root distinct from both endpoint sources.

It enforces:
- generation before use and invalidation;
- all entity uses before invalidation;
- activity start before internal events and internal events before activity end;
- acyclic order graph;
- no duplicate source sequence position or event ID equivocation;
- no cross-source timestamp-only ordering.

The package does not verify cryptography itself or authenticate report transport. It does not establish event-source completeness, factual truth, destination effect or physical outcome. A signed source can still omit events; MON-016/017 own bounded capture/completeness and reconciliation.

Run: `go test -race -count=5 ./...`, `go vet ./...`, `go test -cover ./...`, `go test -run=^$ -fuzz=FuzzSequenceOrder -fuzztime=2s`.

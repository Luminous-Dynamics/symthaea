# MON-020 event-order oracle

Standard-library Go semantic reference for qualifying required event order in a provenance history.

It verifies normalized reports for source identities, separate control-root membership evidence, event records and cross-source order witnesses against pinned verifier ID/version/build/trust-store values and the exact `mon020-normalized-evidence-v1` claim projection. Source, event and witness reports must include the exact parsed fields used in the order decision. The source ID names one monotonic sequence incarnation; a reset requires a new ID. A used or invalidated entity without exactly one authenticated generation event remains indeterminate. It derives source-local order from authenticated strictly increasing source sequence values. For cross-source requirements, it requires an exact-event-bound witness with a control root distinct from both endpoint sources.

It enforces:
- generation before use and invalidation;
- all entity uses before invalidation;
- activity start before internal events and internal events before activity end;
- acyclic order graph;
- no duplicate source sequence position or event ID equivocation;
- no cross-source timestamp-only ordering.

The package does not verify cryptography itself or authenticate report transport. It does not establish event-source completeness, factual truth, destination effect or physical outcome. A signed source can still omit events; MON-016/017 own bounded capture/completeness and reconciliation.

Run: `go test -race -count=5 ./...`, `go vet ./...`, `go test -cover ./...`, `go test -run=^$ -fuzz=FuzzSequenceOrder -fuzztime=2s`.


## Normalized verifier claim profile

Each verifier report must identify `mon020-normalized-evidence-v1` and bind the exact normalized source identity/control-root membership, event fields (ID, kind, subject, activity, source incarnation, sequence), or witness fields (exact predecessor, successor, witness source). A signature over record bytes alone is not enough if the parsed event projection supplied to the order evaluator differs. This semantic join consumes verifier reports; it does not verify the underlying signature or authenticate report transport.

The implementation also requires exactly one authenticated generation event for any entity with use or invalidation events. Missing generation remains indeterminate.

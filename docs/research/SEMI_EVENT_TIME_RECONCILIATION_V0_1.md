# SEMI-EVENT-TIME-RECON-001 — dual-clock streaming reconstruction v0.1

**Issue:** #13654  
**Implementation:** symthaea-semiconductor::benchmark::event_time_reconciliation  
**Status:** Rust event-log/reconstruction primitive with authored tests; exact-head validation pending.

## Two clocks, two histories

Each event envelope binds:
- event ID;
- event time (when the source event is declared to have occurred);
- ingestion time (when the replay system learned it);
- one explicit clock domain;
- mandatory source sequence;
- payload digest and envelope digest.

The replay generation also binds a knowledge cut-off. Events ingested after that cut-off are excluded from the generation; events from another clock domain return UNKNOWN. The implementation does not infer clock mappings or invent missing source sequence values.

The original arrival-order adapter and its receipts remain separate. This module starts from a declared initial engine and produces a new reconstruction generation with an independent event-log digest, canonical replay digest, ordered event IDs and state receipts.

## Event-time ordering

Eligible events are sorted by:
1. event time ascending;
2. source sequence ascending for equal event times.

Source sequence must be present and unique within the declared clock domain. Event ID, sample ID, label ID or content digest is not used as a fallback tie-break. A missing source sequence produces UNKNOWN; duplicate event IDs/sequences are rejected.

Prediction envelope time must match prediction time. Delivery envelope time must match delivery time. Eligibility-correction envelope time must match the corrected report's availability time, and that report availability cannot precede its declared valid-time frontier.

## Late correction behavior

An eligibility correction carries:
- sample, label and report-attested revision;
- measurement event ID;
- valid-time effective_at;
- report-availability time;
- complete K1 report and digest.

If the correction is available before a future delivery and its valid-time frontier applies, that delivery uses the corrected report. If a correction is learned after the corresponding label revision has already updated the reconstructed adapter, the report records:
- the affected label ID;
- affected measurement-event ID;
- REBUILD_REQUIRED status.

It does not reverse or rewrite the earlier receipt. The report explicitly records automatic_rebuild_executed=false; a separate snapshot/model requalification and rebuild generation is required.

## Receipt and digest behavior

The report stores:
- full input event-log digest (so different ingestion orders remain auditable);
- canonical causal replay digest (event time + source sequence + payload identity + resulting disposition);
- initial and final adapter-state digests;
- per-event before/after state digests;
- original source event IDs and payload digests;
- disposition for prediction, label delivery and correction events;
- late-corrected label and measurement IDs;
- source_events_mutated=false, prior_receipts_mutated=false.

A transport-order permutation may change the audit log digest because the ingest sequence differs, but it must not change the canonical replay digest or final adapter state when event times and source sequences are the same.

## Authored tests

The initial test set covers:
1. transport input permutation preserves canonical replay and final state while the audit log digest remains distinguishable;
2. equal event times use source sequence to order a prediction before its label delivery;
3. missing source sequence returns UNKNOWN instead of inventing an ordering;
4. mixed clock domains return UNKNOWN;
5. duplicate event IDs or source sequences are rejected;
6. a correction available before a future delivery replaces the old K1 report;
7. a late revocation after an accepted update marks the affected label and measurement as REBUILD_REQUIRED without mutating prior receipts;
8. events ingested after the generation knowledge cut-off are excluded;
9. corrupted payload digest is rejected.

These tests are authored; exact-head rustfmt, strict Clippy, crate tests and SEMI-BENCH-001 reference validation are pending.

## Claim ceiling

This is bitemporal event-history reconstruction software. It does not prove that the source clock is accurate, authenticate clock mappings or issuers, rewrite physical truth, retroactively mutate prior online receipts, execute a model rebuild/rollback, or authorize fab control.

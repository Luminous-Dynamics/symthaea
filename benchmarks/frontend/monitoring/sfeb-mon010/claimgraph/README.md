# MON-010 bitemporal claim-lineage preflight

**Status:** local executable reference prototype; not a SCITT profile implementation or source-truth verifier.

This standard-library Go package demonstrates an append-only, bitemporal evidence graph for source assertions and their downstream claims, transformations, decisions and actions.

## Core behavior

- Source assertions bind publisher identity, profile, document edition, exact statement/artifact SHA-256 digests, subject/value/unit, numerator/denominator, population, geography, period, methodology, applicable interval, publication time and retrieval/observation time.
- The event ledger records source publication, artifact declaration, publisher supersession, publisher retraction, independent dispute, source-status unavailability and restoration as append-only events.
- `Evaluate(source, asOfRecordedAt, applicableAt)` answers what evidence status was knowable at one recorded time for one source-valid time. A correction recorded later is not projected backward into that historical result.
- `CurrentStatus` separately evaluates the latest recorded source state.
- A supersession must be issued under the same publisher identity and preserve figure semantics. Cross-issuer disagreement is not silently treated as publisher correction.
- A dispute requires a distinct measurement artifact that does not depend on the disputed source.
- `ImpactClosure` walks all dependent claims, transformations, decisions and actions. Pending actions require reauthorization; executed actions remain immutable historical records and trigger a separate current-risk review.
- Every event is bound into a domain-separated SHA-256 hash chain. This provides tamper-evidence for the local event sequence; it is not a digital signature, trusted timestamp or SCITT registration.

## Reproduce

Run `./reproduce.sh`. It expands the committed gzip source snapshots into a temporary directory, then runs five race-enabled test passes, `go vet`, and coverage.

Local environment used: Go 1.23.2, linux/amd64. Local preflight: 14 named test functions; five race-enabled suite runs passed; `go vet` passed; statement coverage 81.4%.

## Trust boundaries

Publisher signatures, source retrieval snapshots, status/supersession credentials and Measurement Capsule evidence must be independently verified before their events are appended. This package models append-only graph and correction semantics; it does not fetch the web, authenticate publishers, verify COSE/SCITT receipts, prove source figures are true, judge decision correctness, or authorize an action.

A hash chain does not prevent a malicious operator from rewriting the entire chain unless its head is anchored outside that operator's domain. Production use needs independently provisioned trust roots and external checkpoint/witness history (MON-008), and it must compose with MON-009 sink receipt/readback for real effects.

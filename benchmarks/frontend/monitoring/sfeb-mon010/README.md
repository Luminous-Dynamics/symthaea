# SFEB MON-010 — source-claim provenance and correction propagation

**Status:** research protocol; no source importer, decision verifier or CPDC benchmark is implemented in this PR.  
**Tracking:** [Issue #13679](https://github.com/Luminous-Dynamics/symthaea/issues/13679).  
**Composes with:** [MON-009 sink-boundary qualification](https://github.com/Luminous-Dynamics/symthaea/pull/13639).

MON-010 adds the epistemic inputs behind agent decisions and actions: publisher/version identity, exact source evidence, claim semantics, transformations, decision/approval records and later correction/supersession events.

Its central design shift is bitemporal append-only claim lineage. Historical evidence remains reproducible as-of the decision time while current status can become superseded, retracted, disputed, unavailable or indeterminate. Corrections propagate through dependent claims, calculations, decisions and pending actions without rewriting the historical record.

See [SOURCE-CLAIM-PROVENANCE-v1.md](SOURCE-CLAIM-PROVENANCE-v1.md) for the graph model, source/claim/decision semantics, correction rules, adversarial corpus and Claim Provenance and Drift Challenge (CPDC) benchmarks.

MON-010 reuses active SCITT profiles and RFC 9942/9943. It does not invent another receipt format or claim that a source signature proves factual truth.


## Executable bitemporal claimgraph preflight

The initial standard-library Go prototype is in [claimgraph/](claimgraph/). It records source publication, supersession, retraction, independent dispute and availability events in an append-only hash-chained history; evaluates historical status as-of recorded time separately from current source status; and computes transitive impact closure while preserving executed-action records.

Run `claimgraph/reproduce.sh` for five race-enabled test runs, `go vet` and coverage. The source/test snapshots are committed as gzip files and expanded into a temporary directory by the runner.

Local preflight on Go 1.23.2/linux-amd64: 14 named tests, five race-enabled runs passed, vet passed, 81.4% statement coverage. This is an executable reference model, not publisher-signature verification, a SCITT adapter, source truth oracle or production authorization mechanism.

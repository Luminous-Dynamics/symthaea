# SFEB MON-010 — source-claim provenance and correction propagation

**Status:** research protocol; no source importer, decision verifier or CPDC benchmark is implemented in this PR.  
**Tracking:** [Issue #13679](https://github.com/Luminous-Dynamics/symthaea/issues/13679).  
**Composes with:** [MON-009 sink-boundary qualification](https://github.com/Luminous-Dynamics/symthaea/pull/13639).

MON-010 adds the epistemic inputs behind agent decisions and actions: publisher/version identity, exact source evidence, claim semantics, transformations, decision/approval records and later correction/supersession events.

Its central design shift is bitemporal append-only claim lineage. Historical evidence remains reproducible as-of the decision time while current status can become superseded, retracted, disputed, unavailable or indeterminate. Corrections propagate through dependent claims, calculations, decisions and pending actions without rewriting the historical record.

See [SOURCE-CLAIM-PROVENANCE-v1.md](SOURCE-CLAIM-PROVENANCE-v1.md) for the graph model, source/claim/decision semantics, correction rules, adversarial corpus and Claim Provenance and Drift Challenge (CPDC) benchmarks.

MON-010 reuses active SCITT profiles and RFC 9942/9943. It does not invent another receipt format or claim that a source signature proves factual truth.

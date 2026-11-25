# Typed Public ResearchState — Evidence Projection v0.1

Owner: [#13663](https://github.com/Luminous-Dynamics/symthaea/issues/13663), composed into [#11171](https://github.com/Luminous-Dynamics/symthaea/issues/11171).

## Purpose

A research policy should not receive "claim confidence = 0.83" as if it fully described the evidence. It needs a public, versioned projection that distinguishes artifact retrieval, exact attribution, a claim-source classifier's output, independent truth adjudication, and decision-specific usability.

ResearchState is a **read-only projection** from existing custody, provenance and evaluation owners. It is not a second evidence store or authority.

## Paradigm shift: four epistemic layers plus decision context

| Layer | Question | State examples | What it does not establish |
|---|---|---|---|
| Retrieval / attribution | Was the artifact found, captured and bound to exact bytes and quote spans? | NOT_FOUND, FETCHED, CAPTURED, DIGEST_MISMATCH, UNATTRIBUTED | Claim support or truth |
| Claim-source relation | Does this exact artifact passage support, refute, qualify or fail to address this exact scoped claim? | UNCLASSIFIED, SUPPORTS, REFUTES, QUALIFIES, IRRELEVANT, UNRESOLVED | Whether the claim itself is true |
| Independent adjudication | Has an independent authority adjudicated the proposition? | NOT_ADJUDICATED, ADJUDICATED(SUPPORTED/REFUTED/MIXED/UNDECIDABLE) | Whether it remains usable for another time/scope/decision |
| Decision usability | Is the state sufficient for this specific decision and risk policy? | UNASSESSED, SUFFICIENT, INSUFFICIENT, CONFLICTED, STALE, OUT_OF_SCOPE | A global or context-free fact score |

A claim may be retrieved but unclassified; classified as supported but not adjudicated; adjudicated as historically true but stale for today's decision; or supported and adjudicated yet insufficient for a different high-risk decision. These are valid distinct states.

## Contract

### Episode and evidence prefix

- `research_episode_id` identifies the evaluation episode, not the evidence store.
- `evidence_prefix.prefix_id` and `prefix_sha256` bind the exact visible evidence history at decision time.
- `lineage_snapshot_ref` points to the upstream lineage/custody snapshot; it is a reference, not a copied graph.
- `sequence_number` is monotone within the episode.
- The exact serialized ResearchState digest must be written into the existing #11015 prospective action trace. Candidate and evaluator state must never share a digest field.

### Claim scope

Every claim includes population, outcome/measure, quantifier/effect, domain/time boundary and a scope note. A scope-free claim should not be promoted to decision-grade status.

### Claim-source relation

Each relation binds one claim ID to one evidence ID. A classified relation requires classifier ID/version, output digest, classifier-specific confidence, emission time, the exact evidence-prefix hash used, and a digest over the canonical claim manifest {claim_id, statement, scope, reference_time}. It also carries an exact quote span represented by byte offsets and a digest of quoted bytes. This prevents classifier output from being replayed against modified claim text or scope while the evidence set remains unchanged.

An UNCLASSIFIED relation must not carry classifier provenance or a quote span presented as classified support. The candidate policy may request classification or more evidence but cannot manufacture this state from retrieval relevance.

### Truth adjudication

Truth adjudication is a separate field with result, authority ID, evidence reference and time. It is not set by the claim-source classifier, local verifier confidence or a language-model judge sharing the same evidence path. Synthetic fixture authorities are test-only.

### Decision usability

Usability is scoped to `decision_scope_id` and `policy_id`. It can be SUFFICIENT for one low-risk decision and INSUFFICIENT for a high-risk decision. Stale, conflicted, unknown-lineage or integrity-failed evidence remains explicit.

## Ownership boundaries

- Evidence/custody owners supply immutable artifact references/digests, capture receipts and lineage snapshot references.
- The claim-source classifier supplies pair-specific relation outputs and versioned provenance; this is not final truth adjudication.
- The independent adjudicator supplies truth outcomes and durable receipts; only that owner may create eligible calibration labels under #13605/#13612.
- The decision-usability assessor supplies policy-scoped sufficiency state; it does not modify source truth.
- #11171 adapts only the public ResearchState view into the pinned candidate policy.
- #11015 binds the exact state digest to prospective decisions and outcomes.
- #13567/#13571 remain evaluator-only oracle/score owners.

Do not create a duplicate evidence database, source-trust service, graph traversal engine, posterior engine or scheduler in this projection.

## Metrics and qualification

The first conformance layer checks schema, IDs, cross-references, evidence-prefix identity, quote-span binding and cross-layer separation. It does not claim natural-language entailment or source authentication.

The next evaluation should measure false promotion rates at every transition, evidence-prefix mismatches, stale-claim leakage, conflict preservation, source-family duplicate credit, scope/temporal errors, false-safe usability outputs, decision regret and complete costs. A high classifier F1 cannot compensate for false adjudication promotion or false-safe decision usability.

## Synthetic conformance examples

`fixtures/research-state-examples-v0.1.json` contains four examples: retrieved/unclassified, classified/not adjudicated, independent-source conflict, and historically adjudicated but stale for current use. They demonstrate valid combinations, not real facts, benchmark scores or expert gold.

## Claim boundary

This contract does not implement the missing production sequential researcher, a claim-entailment model, a truth oracle or an evidence ledger. It specifies the state surface #11171 must consume without creating new authority.

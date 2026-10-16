# Symthaea Evidence Provenance and Revocation Contract

**Date:** 2026-10-16  
**Status:** typed policy prototype and unit tests authored; no production retrieval integration, persistence migration, external verifier, or measured benchmark result claimed  
**Implementation:** `src/knowledge/provenance.rs`; exports in `src/knowledge/mod.rs`

## Paradigm shift: citations are not provenance

A source URL is a locator, not an immutable identity. A revision label is not a digest. A digest proves byte equality only when independently computed and compared; it does not authenticate the publisher. Publisher authentication does not prove truth. A publisher withdrawing a source does not erase the historical fact that the source once made a claim. These are different questions and should not collapse into one confidence number.

The proposed model separates five layers:

1. **Source anchor:** stable source and publisher IDs, revision, exact content digest, and a locator to the relevant passage/field.
2. **Observation time:** when this exact source representation was retrieved, independent of publisher-declared publication time.
3. **Integrity/authenticity observation:** result and method reported by an external verifier, with time. The current module records the result; it does not implement hashing, signatures, PKI, or issuer trust.
4. **Lifecycle status:** active, unknown, unavailable, withdrawn, or revoked. Revocation/withdrawal blocks current advancement without deleting history.
5. **Claim-validity interval:** when the proposition is asserted to apply in the world, separate from when Symthaea learned about it.

This model should be treated as a provenance gate, not a truth detector. `EligibleForReview` is intentionally weaker than `Supported`, `True`, or `CausallyValid`. Semantic entailment, source independence, calibration, and causal identification require separate evaluations.

## Relationship to existing standards

- [W3C PROV Primer](https://www.w3.org/TR/prov-primer/) models provenance through entities, activities, and agents. Symthaea should eventually represent the source artifact, retrieval/extraction activity, and attributed publisher/agent separately, rather than treating a free-form citation string as the complete lineage.
- [in-toto Attestation Framework v1.2](https://github.com/in-toto/attestation/tree/main/spec/v1) binds attestations to artifact subjects and digests, while separating the statement, predicate, and authentication envelope. This is a useful design precedent for separating a source digest from the claims and the verifier's proof.
- [W3C Verifiable Credentials Data Model 2.0](https://www.w3.org/TR/vc-data-model-2.0/) defines a status mechanism that can report credential suspension or revocation. Symthaea should consume a source/status result from an appropriate verifier/registry; it should not invent a parallel issuer-of-record or claim that this module verifies credentials.

These are interoperability precedents, not a claim that the present prototype conforms to any of the standards.

## Prototype contract

`EvidenceSourceReference` keeps source ID, publisher ID, source revision, digest, locator, retrieval time, publisher-declared time, and rights reference distinct. `IntegrityStatus` tracks unknown/verified/mismatch/revoked. `SourceStatus` tracks active/unknown/unavailable/withdrawn/revoked. `ClaimValidity` uses a half-open interval: valid-from is inclusive and valid-until exclusive.

The strict default currently requires publisher identity, revision, digest, locator, retrieval time, a recorded integrity verification, and a fresh active-source status. Its age defaults—30 days for retrieval observation and 24 hours for status/integrity checks—are explicit placeholders, not scientifically tuned constants. The target domain must tune and preregister them. Rights references are represented but not universally required because availability and legal interpretation vary by source class; applications that require rights metadata must enable that gate.

The assessment is deterministic and side-effect free. It checks digest syntax, not digest correctness. It does not retrieve a page, interpret a licensing URL, validate a signature, establish publisher identity, or verify whether an issuer was authorized. Such results must come from a separate adapter that provides auditable outputs.

### Non-negotiable failure semantics

- Missing required provenance or unknown required status leads to abstention, not an implicit trust score.
- Integrity mismatch, integrity-attestation revocation, source withdrawal, and source revocation cannot be bypassed by a permissive completeness policy.
- Stale status is not treated as current status; status unavailable is not treated as active under the strict policy.
- Future timestamps and invalid validity intervals are rejected.
- Historical observations remain representable. Revocation affects eligibility for new/current decisions; it must not silently rewrite the historical record.
- Provenance quality is not semantic truth. A fully attested source can make a false claim, and a true claim can have incomplete provenance.

## Evaluation protocol

### Adversarial strata

Build examples around a licensed/public-domain corpus and keep a source-family manifest. Each base example gets one-axis counterfactual variants; variants from the same source/revision/family must never cross dataset splits.

| Stratum | Perturbation | Required behavior under strict policy |
|---|---|---|
| P0 Complete / active | Revision, digest, locator, publisher, recent verifier and active status all present | Eligible for review, never auto-accepted as true |
| P1 Missing anchor | Missing publisher, revision, digest, locator, or retrieval time | Abstain with the exact missing-field reason |
| P2 Mutable locator | Same URL serves changed bytes or a new revision | Digest/revision mismatch must be observable; no URL-only equivalence |
| P3 Integrity failure | Digest/signature mismatch | Abstain; zero escapes |
| P4 Revoked/withdrawn | Previously valid source or attestation is later withdrawn/revoked | Abstain for current advancement while preserving historical status |
| P5 Stale / unavailable status | Status check exceeds the policy age or current check cannot be fetched | Abstain under strict policy, not implicit active |
| P6 Temporal boundary | Future observation/publication/verification time; not-yet-valid or expired claim | Abstain with a temporal reason |
| P7 Semantically false but well-provenanced | Correctly hashed and authenticated source asserts a false proposition | Provenance gate may pass; a separate semantic/source-evaluation benchmark must catch or appropriately qualify the claim |
| P8 True claim, poor provenance | Claim is independently known true but source anchor is missing or revoked | Provenance gate abstains; measure this false-abstention cost separately |

P7 and P8 are crucial negative controls: otherwise a provenance completeness metric can be confused with truth accuracy.

### Dataset and leakage controls

1. Start with at least 1,000 independent source/query families for pilot characterization. Expand to 5,000 or more labeled query families before adopting release thresholds; ensure P3/P4/P5/P6 each have enough positive cases for useful confidence intervals.
2. Split by source, publisher where practical, revision lineage, and counterfactual family. Keep derived variants together. Reserve an out-of-domain publisher/source-shift set.
3. Separate gold-structured-provenance evaluation from end-to-end extraction/verification. Report extraction/identity resolution error separately; do not feed gold data into the production-path score and present it as end-to-end.
4. Have two independent annotators label expected status, time semantics, rights-reference expectations, and gate action; adjudicate disagreement.
5. Publish immutable dataset, code revision, policy, environment, and output manifests with SHA-256 digests. Respect source licences; retain only permitted text and metadata.
6. Report every policy threshold before examining the held-out split. Age thresholds are domain-dependent: news, scientific literature, and slowly changing reference documents should not necessarily share one freshness window.

### Required metrics and proposed gates

These are **targets to evaluate**, not measured Symthaea results.

| Metric | Proposed gate | Why |
|---|---:|---|
| Explicit mismatch/revocation escapes | 0 | Hard safety invariant, including permissive-policy tests |
| Reason-coded unit-contract cases | 100% expected decisions/reasons | Deterministic module contract |
| Required-provenance false eligibility | 0 on held-out P1/P3/P4/P6 cases with unambiguous labels | Directly measures fail-open defects |
| Correct active/complete acceptance | Report sensitivity; initial target ≥95% on P0 | Detects an over-conservative gate |
| False abstention on adjudicated reviewable, active evidence | Initial target ≤5%; publish per-domain breakdown and confidence interval | Makes operational rejection cost visible |
| Source-status revocation detection | ≥99% on a test feed with known timestamps and status history | Tests status ingestion, not the current pure policy function |
| Time-to-block after revocation event | Report p50/p95/max under cache and outage conditions; choose SLO after pilot | Measures real revocation propagation rather than only the local enum check |
| Provenance-gate overhead | Measure p50/p95/max at 1k / 10k / 100k evidence items; compare to provenance-disabled baseline | Avoids inventing a latency SLO before a baseline exists |
| Audit completeness | 100% of decisions emit policy ID, assessment time, source revision/digest refs, status-check times, decision, and reason codes | Makes decisions reproducible |

The deterministic policy function has no graph scan and should be approximately constant-time per record aside from field validation; that is a hypothesis. Real bottlenecks are likely source-status fetch/cache behavior, digesting large payloads, signature verification, and provenance-store lookups. Benchmark those separately. Never combine fixture arithmetic with population-level accuracy.

### Mandatory ablations

Compare (a) citation/URL only, (b) revision without content digest, (c) digest without status freshness, (d) full strict policy, and (e) each field gate ablated one at a time. Include a deliberately permissive policy to prove that explicit mismatch/revocation remains blocking. Report coverage alongside error rates; the gate can achieve zero false eligibility by abstaining on everything, which is not an acceptable success result.

## Integration sequence

1. **Current slice:** typed model, pure deterministic assessment, reason codes, and tests. Not connected to graph persistence or production retrieval.
2. **Next:** decide canonical provenance/status ownership. Prefer a single trust/provenance authority and store stable references from Symthaea facts; do not copy an evolving Mycelix trust score into a second source of authority.
3. Add a versioned persistence migration only after source ownership, backward compatibility, and unknown-status behavior are decided. Legacy facts without provenance must remain explicitly unknown; do not backfill invented publisher, revision, or status values.
4. Add a provenance requirement to the relation-constrained query report as a separate gate, preserving its semantic relation report. Test stale, revoked, conflicting, and partially sourced evidence independently.
5. Add source-status fixtures and a reproducible offline benchmark harness. Measure the pure gate locally before introducing networked status-feed benchmarks.
6. Only after parity and outage tests should any production caller opt into the policy. No default search-path behavior is changed by this PR.

## Known limitations / open risks

- A public key or signature method name is not proof that key resolution, certificate-chain validation, or revocation checking occurred. Those belong to the verifier adapter contract.
- An issuer being active does not make every claim from that issuer true.
- A source may be valid for historical reconstruction but not for a current decision.
- Source rights are not safely inferred from URL, open accessibility, or publisher identity.
- Exact-content hashing is sensitive to canonicalization; a web page's raw bytes, rendered text, and extracted passage need distinct digest semantics. The digest scope must be pinned in the versioned format.
- Same-source corroboration is not independent corroboration. Downstream reasoning should group evidence by source lineage, not count duplicate URLs or syndicated copies as independent witnesses.

## References

- W3C, [PROV Model Primer](https://www.w3.org/TR/prov-primer/), 2013.
- in-toto, [Attestation Framework v1.2](https://github.com/in-toto/attestation/tree/main/spec/v1).
- W3C, [Verifiable Credentials Data Model v2.0](https://www.w3.org/TR/vc-data-model-2.0/), Recommendation, 15 May 2025.

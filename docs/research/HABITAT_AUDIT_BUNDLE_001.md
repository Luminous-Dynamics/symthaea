# HABITAT-AUDIT-BUNDLE-001 — cross-layer evidence replay

**Status:** synthetic audit bundle and Rust qualifier; no signatures, trusted timestamps, real measurements, or physical-source authentication are connected.

## Paradigm shift

Individually correct modules can still disagree about what the same production history means. One validator may accept an event stream, another may accept a quantity ledger whose source references have changed, and a third may accept measurement decisions based on a different fixture revision. The integration unit should therefore be a **pinned, replayable evidence bundle**: exact inputs, independently rerun component validators, and a separate manifest of expected outputs.

This is a consumer of the existing structural, event-replay, material-ledger, measurement-gate, recall-response, and quantity-genealogy owners. It does not create another event system, provenance authority, scheduler, inventory ledger, or metrology engine.

## Provenance and integrity boundary

The [W3C PROV data model](https://www.w3.org/TR/prov-dm/) provides interoperable concepts for entities, activities, derivation, responsibility, and provenance bundles. The [PROV primer](https://www.w3.org/TR/prov-primer/) explains that provenance records can support trust and reproducibility assessments. A record of origin is still a claim about how information was produced; it is not by itself proof that an underlying physical event or measurement was true.

This implementation pins six source fixtures by SHA-256 over exact UTF-8 bytes, including JSON whitespace and final newline. A digest mismatch detects bytes that differ from the manifest. A digest match does **not** authenticate an author, validate a supplier, verify a measurement, establish a trusted timestamp, or prove physical truth. No signature, key-management protocol, transparency log, or trusted-clock service is implemented.

## What is included

- `docs/research/fixtures/habitat-audit-bundle-v0.1.json`: exact source paths, UTF-8 byte lengths, SHA-256 digests, explicit claim ceilings, and independently declared expected component summaries.
- `crates/domains/symthaea-habitat-vertical-bench/src/audit_bundle.rs`: verifies artifact set/path/schema/byte length/digest, reruns the structural, event-replay, material-ledger, measurement-gate, recall-response and quantity-genealogy validators, checks the expected result counts and closure state, and ensures that each fixture remains on its stated synthetic-evidence boundary.
- The default `validate-habitat-vertical` CLI executes the bundle audit after the existing component checks. A successful audit means the pinned synthetic inputs and known answers agree; it is not physical commissioning or product qualification.

## Proposed benchmarks (not achieved)

1. **Pinned-byte identity:** 100% of unmodified fixture bytes match the bundle's declared SHA-256 and length; every seeded byte mutation fails the fingerprint gate.
2. **Component replay coverage:** 100% of required component validators rerun on each audit; zero component errors may be hidden by a valid manifest.
3. **Cross-layer reference integrity:** zero unresolved source-event, operation, account, traceability-node, and measurement-case references in admitted bundles.
4. **Known-answer agreement:** 100% match with the separate manifest for event count/duplicates, transaction/account totals, opening/closing mass, and each measurement disposition.
5. **Determinism:** identical inputs yield identical sorted findings and component summaries regardless of artifact-array order.
6. **Tamper detection:** 100% rejection of changes to a pinned source fixture, path, length, digest, or manifest expected result.
7. **Honest claim boundary:** zero upgrades from synthetic to physical evidence merely because fingerprints match; future verified signatures and source attestations must be a separate protocol with its own negative controls.
8. **Scale:** after implementation is compiled and profiled, report p50/p95 bundle validation time and peak memory on pinned hardware for increasingly large event/ledger fixtures.

## Next integration gate

After Rust execution and code review, generate a machine-readable audit receipt containing the exact source digests, validator/evaluator revision, component-level outcomes, and a fail-closed overall disposition. Do not treat that receipt as signed or trusted until a separate signing/attestation design defines principals, key rotation/revocation, time source, replay prevention, custody, and independent verification.


## Recall-response integration

The bundle now pins the recall-response fixture as a fifth input by exact UTF-8 byte length and SHA-256. It reruns the recall qualifier over the already validated event stream and checks the independently declared affected-target count, terminal target count, response count, contained count, unresolved count, false-positive control count and closure state. The expected state remains **OPEN_UNRESOLVED**; the bundle is valid only when the fixture, component output and manifest agree about that open state. A valid audit bundle is not a closed recall or authenticated physical evidence.

Research anchors: [ISO 10393:2013 consumer product recall guidance](https://www.iso.org/standard/45968.html), which ISO confirmed as current in 2025; [GS1 EPCIS/CBV 2.0.1 artefacts](https://ref.gs1.org/standards/epcis/artefacts); and [NISTIR 8620 (2026)](https://www.nist.gov/publications/digital-twins-workshops-summary-report), which highlights interoperability, VVUQ and cybersecurity needs. These inform the test problem but are not conformance claims.


## Quantity-aware genealogy integration

The bundle pins a sixth source fixture for the split/merge and partial-consumption challenge. It reruns the quantity-genealogy qualifier and compares operation count, opening and ending mass, terminal count and the exact affected-terminal set against the separate manifest. The synthetic result preserves 2,700 mg of opening mass as 2,700 mg of remaining inventory, WIP, scrap and packaged units; PACKED-ARTICLE-A and PACKED-ARTICLE-B are affected, while the clean control is not. No physical lineage is established.

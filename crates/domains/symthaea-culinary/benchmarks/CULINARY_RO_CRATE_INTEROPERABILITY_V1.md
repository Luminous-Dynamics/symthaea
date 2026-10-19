# Culinary RO-Crate 1.3 interoperability benchmark v1

**Status:** implementation draft; synthetic test fixtures only. Rust compilation, test execution, and validation by an independent RO-Crate 1.3 validator have not been evidenced for this change. This is an interoperability adapter, not a food-safety or scientific-validity certification.

## Paradigm shift: verify once, exchange in standard formats

The content-addressed Culinary manifest remains the domain-specific verification contract. After it passes, this adapter emits an RO-Crate 1.3 JSON-LD metadata file plus a deterministic in-memory file map containing the original manifest and the exact source payload bytes at their declared relative paths.

This separates three claims that must not be conflated:

1. **Integrity:** supplied bytes match a separately pinned SHA-256 digest.
2. **Interoperability:** the files and their metadata can be consumed through a published research-object packaging format.
3. **Authenticity and reality:** who captured the evidence, who held it, whether a sensor was calibrated, whether the procedure was followed, and whether a scientific interpretation is true.

This increment addresses (1) and establishes a testable route toward (2). It does not establish (3).

## Export contract

Entry point: `export_culinary_evidence_bundle_ro_crate_v1`.

The exporter returns an opaque `CulinaryRoCrateV1` with:
- a RO-Crate 1.3 JSON-LD metadata document at `ro-crate-metadata.json`;
- original manifest bytes unchanged at `culinary-evidence-manifest.json`;
- every verified payload unchanged at its manifest-declared normalized relative path;
- the verified exact-manifest digest and getters for inspection.

The graph's root Dataset includes the experiment ID, bundle ID, protocol digest, a verification-profile declaration, and references to the manifest and every artifact. Each file record carries its path, content size, SHA-256, artifact kind, source reference and source version. Reserved filename collisions fail closed.

The adapter does not sign artifacts, add custody attestations, assign an authenticated identity to a source reference, or claim RO-Crate conformance until an independent validator has checked the serialized package. It currently exports; importing arbitrary RO-Crates back into trusted Culinary evidence is deliberately out of scope.

## Industry comparison

| Reference | Established strength | Scope here | What is required before a stronger claim |
|---|---|---|---|
| [RO-Crate 1.3](https://www.researchobject.org/ro-crate/specification/1.3/introduction.html) | Machine-readable aggregation and description of research files, context, agents, equipment and provenance. | Emits a RO-Crate-shaped JSON-LD graph with explicit file metadata while preserving domain-level verification. | Validate packages using an independent RO-Crate 1.3 validator; test at least two independent consumers; document unsupported profiles and round-trip limits. |
| [W3C PROV](https://www.w3.org/TR/prov-primer/) | Shared model for entities, activities, agents and derivation. | Artifacts and experiment identity remain addressable; richer PROV relations are not yet synthesized. | Add explicit capture activities, responsible agents, calibration entities, derived-from relations and provenance for the assertions themselves—only where evidence supports them. |
| [ISO 22005:2007](https://www.iso.org/standard/36297.html) | Principles and requirements for traceability-system design in the food/feed chain. | Byte-level research/experiment artifacts, not a supply-chain traceability management system. | Model ingredient lot IDs, transformations, custody handoffs, locations, quantities, recall traversal and operational procedures; commission a separate conformance assessment. |
| [GS1 traceability / EPCIS](https://www.gs1.org/standards/traceability) | Interoperable Critical Tracking Events and Key Data Elements across supply-chain participants. | Process traces and measurements are files linked to an experimental bundle. No EPCIS event exchange is implemented. | Define a reviewed mapping for event time/timezone, what, who, where, business step, disposition and input/output transformations; run producer/consumer interoperability tests. |
| [CONSORT 2025](https://www.bmj.com/content/389/bmj-2024-081124) | Reporting expectations for randomized trials, including pre-specified outcomes and transparent effect/precision reporting. | Separate preregistration, paired descriptive effects and measurement bounds. | Keep experimental-design, estimation, uncertainty and reporting qualification as distinct gates; RO-Crate export cannot upgrade a synthetic fixture into empirical evidence. |

These references establish comparison targets, not a claim that Culinary conforms to or is superior to the referenced systems.

## Acceptance benchmark

### Integrity and fail-closed checks

- 100% rejection of a changed payload byte, changed manifest byte, wrong expected digest, missing payload, or undeclared payload (inherited from the verifier).
- 100% rejection of duplicate artifact IDs/paths/source refs, duplicate JSON keys, unbound outcome sources, invalid trace IDs, unsafe paths and reserved metadata filename collisions.
- 100% preservation of each accepted source payload: exported bytes equal the original byte-for-byte.
- 100% preservation of original manifest bytes and digest.
- Zero payload paths that alias either `ro-crate-metadata.json` or `culinary-evidence-manifest.json`.

### Interoperability and determinism

- 100% of manifested payloads represented exactly once in the RO-Crate graph and reachable from the root Dataset.
- 100% of graph file identifiers resolve to a file present in the returned package.
- Repeated exports with identical inputs yield byte-identical files and metadata.
- Independent RO-Crate 1.3 validator: zero errors on a synthetic valid fixture and rejection/reporting of deliberately broken metadata.
- At least two independent consumer implementations can enumerate all files and recover the SHA-256 values without Culinary-specific code.
- A round-trip test must preserve file hashes and explicitly report metadata that cannot round-trip; never silently discard provenance fields.

### Performance

Report input payload count and total bytes, wall-clock export time, peak memory if measurable, JSON-LD metadata size, and throughput. Publish raw measurements, machine/runtime details and the exact commit SHA. Do not claim a speedup until an apples-to-apples baseline is measured.

Initial engineering targets (targets, not measured results):
- 100% byte preservation for valid inputs.
- zero accepted integrity mutations across the adversarial suite.
- deterministic output for identical ordered manifest bytes and payload bytes.
- linear scaling in total serialized bytes plus graph entries, documented with measurements at 1, 10, 100, 1,000 and 10,000 artifacts.

## Next frontier after this adapter

1. A signed capture receipt using an established envelope/attestation format, whose signed subject is the manifest digest—not a self-asserted identity string embedded inside the manifest.
2. Key discovery, rotation/revocation, trust policy and independent receipt verification.
3. Append-only custody events for collection, transfer, storage, calibration and analysis; distinguish a cryptographic signature from evidence that the described physical handoff occurred.
4. Optional GS1 EPCIS mapping for ingredient lots and physical transformations, without forcing research datasets to become a supply-chain system.
5. Separate physical validation studies: instrument calibration, repeatability/reproducibility, missingness and uncertainty, protocol deviations, contamination controls, food-safety review, sensory panel blinding, and out-of-sample recipe performance.

The next paradigm shift is not a larger synthetic benchmark score. It is a reproducible evidence path from the preregistered protocol through physical run, measurements, custody and independent analysis—while every layer states exactly what it can and cannot prove.

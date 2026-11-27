# Scientific artifact intake contract v0

Status: schema/fixture proposal only. No parser, runtime, storage layer, security boundary or benchmark adapter is implemented by these files.

## Purpose

This is the first contract slice for issue #13900. It separates an immutable captured artifact from the parser output derived from it, and binds observations to typed source locations. It intentionally composes with WebResearcher acquisition receipts rather than creating another retrieval subsystem or epistemic ledger.

## Files

- SCIENTIFIC_ARTIFACT_MANIFEST_SCHEMA_V0.json — JSON Schema Draft 2020-12 contract.
- SCIENTIFIC_ARTIFACT_MANIFEST_FIXTURES_V0.json — synthetic CSV, JSON, unsupported-PDF and digest-mismatch examples.

## Contract semantics

The raw artifact is content-addressed. The manifest's artifact_id is defined as sha256:<raw.sha256>; the acquisition receipt's content digest must equal that same raw digest. JSON Schema cannot enforce equality between these fields, so consumers must enforce the listed cross-field invariants.

PARSED means the pinned parser completed under declared limits and emitted a digest-bound derived representation. It does not mean the source is authentic, the extracted observation is correct, the observation supports a claim, or the claim is true. Claim/source relation adjudication remains #13796; source identity and graph provenance remain #13792.

PARTIAL is diagnostic only. UNSUPPORTED, MALFORMED, LIMIT_EXCEEDED and INTEGRITY_FAILED carry no derived-payload digest or observations. No failure state may be turned into a successful empty artifact.

The v0 locators are intentionally narrow: CSV cell and JSON Pointer. PDF, XLSX, Parquet, FASTA/GenBank and PDB/mmCIF need separately reviewed parser adapters and locator contracts. The fixtures do not imply that those adapters exist.

## Conformance rules beyond JSON Schema

A validator must additionally check:
1. artifact_id equals sha256: plus raw.sha256;
2. acquisition.receipt_content_sha256 equals raw.sha256;
3. observed usage is within every declared limit;
4. observations_emitted equals the observations array length;
5. byte spans are ordered and within the raw artifact;
6. parser ID/version/configuration and runtime identity are pinned;
7. observations from non-PARSED states cannot be promoted into claim support or decision inputs.

Keep raw lexical numeric values and unit strings. Do not coerce locale-specific numbers, units, nulls or sentinel values without an explicit, versioned transformation.

## Qualification boundary

The synthetic fixtures are contract examples only. Before implementation, agree on parser/API ownership, content-addressed byte retention, sandboxing and resource limits. Before any LifeSciBench or DISCERN claim, freeze exact task/data revisions, data rights, parser revisions, tool/model budget and official scoring/adjudication rules.

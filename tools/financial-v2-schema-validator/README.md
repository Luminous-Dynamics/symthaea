# Financial V2 Schema Validator

Standalone offline Rust validator for the proof-bound financial evidence v2 schemas.

## Why this exists

A JSON parse check is not JSON Schema validation. A schema file can parse while containing an invalid Draft 2020-12 construct; a manifest can parse while violating its schema; and a schema-valid manifest can still contain cross-field inconsistencies such as a rights principal different from the manifest principal.

This package adds three distinct checks:

1. **Draft 2020-12 meta-schema validation** for all registered schemas.
2. **Instance validation** for research and quant examples, with local-only \`$ref\` resolution and format assertion enabled.
3. **Cross-field semantic checks** for principal/rights-profile consistency, qualification/mode compatibility, availability precision, entitlement validity, future-known evidence, proof state and duplicate evidence IDs.

## Offline reference handling

The validator uses a small in-memory schema registry. The quant schema's relative reference to the shared evidence-v2 schema resolves only to a schema already loaded from the checked-out repository. An unknown reference returns an error; no HTTP or filesystem fallback is permitted by the custom retriever.

The \`jsonschema\` dependency is pinned to 0.49.8 and configured for Draft 2020-12. \`format\` validation is explicitly enabled; JSON Schema treats format behavior as configurable, so a parser-only check is insufficient.

## Proof-binding integrity

The evidence-v2 schema now references `evidence-proof-binding-v1.schema.json`. Availability and rights use distinct domain-separated binding preimages. The validator checks that every bound identity, version, content digest, principal, rights profile, availability field or entitlement field matches the enclosing evidence record, then recomputes `binding_sha256`.

The v1 binding preimage is restricted to ASCII strings, nested objects and null values (no numbers or arrays). With `serde_json`'s default sorted object map and no `preserve_order` feature, its compact UTF-8 serialization is the constrained RFC 8785 JCS form used for the SHA-256 preimage. The validator includes property-order invariance and digest-mutation tests. Do not broaden the binding schema to numbers, arrays, or non-ASCII data without adopting and testing a full JCS implementation.

This verifies only the declared binding digest and field equality. It does not authenticate the referenced source proof, entitlement document or verifier receipt; those remain part of the trusted Mycelix verification path.

## Current qualification boundary

The only committed examples are \`synthetic_development_only\`. Their \`synthetic_fixture_only\` proof markers are intentionally not accepted for historical confirmation or prospective shadow. This validator does not authenticate the artifacts, signatures, licences, availability receipts or verifier receipts represented by those references. It does not create \`AvailabilityProof::Verified\` or \`RightsProof::Verified\` values. That remains the responsibility of the trusted Mycelix/provenance adapter in #11264.

The validator rejects:
- self-asserted or missing proof states in qualification cohorts;
- a rights entitlement that is not yet valid or has expired at the decision frontier;
- availability timestamps after the historical/streaming frontier;
- exact precision without an exact instant;
- bounded intervals with reversed endpoints or a frontier-straddling interval;
- source-calendar intervals without an explicit timezone;
- unknown precision in qualification cohorts;
- principal/rights-profile mismatch and duplicate evidence IDs.

For bounded intervals, this development validator uses the fail-closed rule from #11256: a frontier strictly before the earliest possible time is future-known; a frontier at or after the latest possible time can be admissible; a frontier in between is indeterminate and rejected from qualification. Full timezone/DST and proof-bound interval semantics remain part of #11256.

## Run

From this directory:

\`cargo fmt -- --check\`

\`cargo test\`

\`cargo run\`

The GitHub Actions workflow separates formatting from schema validation. A formatting failure does not prevent schema tests from executing.

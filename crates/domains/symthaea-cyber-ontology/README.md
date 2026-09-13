# symthaea-cyber-ontology

Versioned external-cyber-knowledge bridge for Symthaea.

The crate is intentionally small. It does **not** vendor the MITRE ATT&CK or D3FEND corpus. Instead it binds compact object identities and mapping assertions to immutable `TechnicalSourceSnapshotV1` records in `symthaea-support`.

## Reference upstreams at implementation time

Verified 2026-09-11:

- MITRE ATT&CK current content release: **v19.2** (August 2026 agile release).
- MITRE D3FEND ontology: **v1.6.0**, released 2026-08-31.
- D3FEND publishes ontology artifacts and official ATT&CK↔D3FEND mapping resources.

These values are examples, not hard-coded truth. Historical and future releases are represented by their own exact source/framework snapshots.

## Core boundary

```text
immutable MITRE source snapshot
             ↓
 exact framework release identity
             ↓
 exact object identity within release
             ↓
 externally sourced mapping assertion
             ↓
 defensive candidate / analytic context
             ↓
 local applicability + evidence + policy
             ↓
 bounded recommendation
```

Never collapse:

```text
ATT&CK technique != observed compromise
D3FEND technique != proven effective control
official mapping != universal effectiveness
same public ID != same versioned object
revoked/deprecated != nonexistent historical knowledge
mapping evidence != execution authority
```

## Why exact release identity matters

ATT&CK and D3FEND both evolve. Object definitions, object versions, tactics, mappings, deprecations and revocations can change independently of a stable public identifier.

`CyberObjectKeyV1` therefore binds:

```text
framework snapshot + external object ID
```

rather than using `Txxxx` / `D3-*` alone.

## Mapping provenance

Each `CyberMappingAssertionV1` binds exact source and target object keys plus an immutable technical-source snapshot that published or asserted the mapping. Extraction quality measures normalization quality only; it is not a control-effectiveness probability.

V1 accepts MITRE-backed mapping sources for ATT&CK/D3FEND integration. Future framework adapters should add their own explicit publisher/source policy rather than disguising third-party inference as MITRE mapping authority.

## Defensive queries

`defensive_candidates_for_attack()` returns exact D3FEND technique candidates only for explicitly defensive mapping relations (`Counters`, `Mitigates`, `Prescribes`). Generic `MapsTo`/`RelatedTo` relations do not silently become recommendations.

Returned candidates preserve mapping-source lifecycle, authority and stability. Revoked targets are excluded by default; historical queries can opt in deliberately.

## Future ingestion

A later ingestion adapter can consume version-pinned ATT&CK STIX and D3FEND RDF/JSON-LD/CSV artifacts, validate their cryptographic/source snapshot identity, and populate this bridge. The ingestion layer should remain separate from the semantic model and should never auto-update a qualified corpus in place.

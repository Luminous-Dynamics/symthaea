# ADR-020 — RSK Shared Schema-Adapter Golden Corpus

- **Status:** Accepted for reference evidence; production admission denied
- **Change Class:** A
- **Scope:** Replicator Safety Kernel semantic adapter evidence
- **Related:** #1678, #1679, #1867, #1969, #2035, #2038, #2057

## Context

The deterministic schema-to-type-state adapter introduced in #2057 originally encoded its expected capability/resource rule tables directly inside Python test assertions.

That proves the Python implementation against itself, but it is weaker as a future cross-language contract: a Rust implementation would have to infer the expected output from Python test source rather than consume one stable shared artifact.

## Decision

Commit a separate canonical adapter golden corpus:

`docs/architecture/replicator-safety/golden/RSK_SCHEMA_ADAPTER_GOLDEN_V0_1.json`

The corpus binds:

- the source semantic golden corpus;
- the source capability/resource schema IDs;
- the exact deterministic capability validator rule table;
- the exact deterministic resource validator rule table;
- canonical SHA-256 identities of both derived rule tables.

Python tests consume the committed corpus instead of duplicating those rule tables in test source.

## Canonical adapter-output identities

For the abstract test schemas:

```text
capability rule table SHA-256:
a6fec0f37b92fc90ccf01d87b2e19ab69533d8b43da3e968938d92a18d93e99d

resource rule table SHA-256:
211d35785c8e174eac57cf29914fb06be1a3b36d730baeda634c9a5d73841335
```

These are semantic test vectors only. They do not encode physical replication capability or physical resource semantics.

## Why a separate corpus

The semantic schema digest answers:

```text
What exact schema bytes define meaning?
```

The adapter-output digest answers:

```text
What exact structural-validator configuration is the deterministic image of those bytes?
```

Keeping both identities allows independent implementations to detect disagreement in parsing/derivation even when they agree on the source schema digest.

## Required invariant

```text
same verified schema bytes
    -> same schema ID
    -> same adapter golden output
    -> same adapter-output digest
```

Any disagreement fails closed for production qualification.

## Non-amplification

A matching adapter-output digest does not establish:

- registry signature validity;
- trust-snapshot freshness;
- signer lifecycle or failure-domain independence;
- grant/quorum authority;
- trusted time;
- containment state;
- cross-schema translation authority;
- production admission.

It proves only deterministic schema-to-validator derivation.

## Future Rust requirement

Before schema-derived validators enter production authority paths, Rust must independently derive the same tables from the same canonical schema bytes and match the committed adapter-output digests.

The Rust implementation should not deserialize a Python-produced rule table and call that parity; it must recompute independently.

## Evidence status

The golden corpus and Python self-test changes are authored and statically reviewed. Exact-head workflow execution remains pending GitHub runner availability.

## Production status

**DENIED / NOT YET ELIGIBLE**.

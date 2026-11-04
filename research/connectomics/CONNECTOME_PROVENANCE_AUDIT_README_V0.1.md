# Provenance-completeness audit fixture V0.1

This is a synthetic format and sampling fixture. It is not a real connectome audit and contains no biological data.

## Files

- `connectome-provenance-audit-input-v0.1.schema.json`: strict structural input schema.
- `fixtures/connectome-provenance-audit-declared-bundle-v0.1.json`: producer-declared evidence bundle.
- `fixtures/connectome-provenance-audit-known-answer-v0.1.json`: finite frame, deterministic sample and five synthetic audit dispositions.

## Digest conventions used by this fixture

The dependence-bundle digest is SHA-256 over the exact UTF-8 bytes of `connectome-provenance-audit-declared-bundle-v0.1.json`, including the final newline.

The frame digest is SHA-256 over the canonical JSON encoding of the frame payload with the `frame_sha256` member omitted. Canonical encoding for this fixture is UTF-8 JSON with object keys sorted lexicographically, no insignificant whitespace, and arrays retained in their authored order. This fixture does not claim to be a general canonical JSON standard; the executable verifier must pin and test its exact byte algorithm.

The seed commitment is:

```
SHA256(
  UTF8("connectome-provenance-audit-seed-commitment-v0.1\0")
  || frame_sha256_bytes
  || seed_bytes
)
```

The sample-ranking score for one assertion is:

```
SHA256(seed_bytes || 0x00 || UTF8(stratum_id) || 0x00 || UTF8(assertion_id))
```

Within each stratum, sort by the lowercase hexadecimal score ascending (assertion ID ascending as a deterministic tie-breaker), then select the first `sample_size` assertions. This is the fixture's deterministic SHA256_RANK_V1 sample-selection convention. It makes the fixture reproducible; it does not authenticate the entropy or independence of the seed source.

## Expected sample membership

| Stratum | Frame N | Sample n | Selected assertion IDs |
|---|---:|---:|---|
| high_materiality | 4 | 3 | a-04, a-01, a-03 |
| routine | 4 | 2 | b-02, b-04 |

Expected observed dispositions:
- CONTRADICTED_DEPENDENCE: 1
- NO_SHARED_ANCESTOR_FOUND_WITHIN_AUDITED_SCOPE: 3
- UNRESOLVED_SOURCE_ACCESS: 1
- AMBIGUOUS_LINEAGE: 0
- PROTOCOL_VIOLATION: 0

The critical stratum has one contradiction and one unresolved assertion in only three sampled units. The query risk budget is intentionally not met; this small fixture must not produce an independence promotion or a certified claim.

## Limitations

The fixture illustrates deterministic frame/sample accounting and conservative outcome vocabulary. It does not implement hypergeometric confidence bounds, prove source records truthful, measure natural provenance completeness, or establish biological independence. The source references and record digests are synthetic placeholders for schema testing only.

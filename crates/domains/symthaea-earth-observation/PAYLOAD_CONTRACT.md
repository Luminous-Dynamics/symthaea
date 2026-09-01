# Canonical raster payload contract

This document freezes the interpretation order for `RasterPayloadDescriptor`.

## Scope

The descriptor applies to a decoded, canonical, packed scientific buffer. Provider container details such as GeoTIFF/JP2 compression, tiles, strips, SAFE layout, padding, caches, and transport framing are outside this contract. A provider adapter must materialize the canonical buffer before scientific feature extraction.

## Deterministic interpretation order

For each addressed sample:

1. resolve `(row, col, band)` to an exact byte offset using the declared BSQ/BIL/BIP layout;
2. decode the stored scalar using the declared sample type and canonical byte order;
3. compare the **stored value** against the band's typed NoData sentinel before scale/offset is applied;
4. evaluate any declared validity mask independently;
5. if either NoData or the validity mask marks the sample invalid, do not emit a physical value from that sample;
6. otherwise compute `physical = stored * scale + offset`.

NoData is therefore a storage-domain predicate, not a physical-value predicate. A scaled physical value that happens to equal the numerical value of a NoData sentinel is not automatically invalid.

For floating NoData, equality is by the exact declared IEEE-754 bit pattern. This permits a specific NaN payload or signed zero to remain reproducible rather than relying on language-level floating comparison semantics.

## Validity masks

A validity mask is independent evidence of sample usability.

- `EmbeddedBand` identifies an existing band in the same canonical payload.
- `External` names a separate mask input whose actual bytes and content identity must be bound by the evidence/fixture layer.
- `valid_when_nonzero` is evaluated on the mask's stored value before any scientific-band scale/offset transform.

A mask does not silently repair or replace NoData. Either invalidation path is sufficient to withhold the sample.

## Identity-significant fields

The eventual frozen payload identity must treat at least the following as significant:

- shape;
- sample type;
- byte order;
- ordered band list;
- band interleave;
- per-band scale and offset by exact IEEE-754 bits;
- typed NoData representation;
- validity-mask semantics;
- exact packed byte length;
- exact content digest.

Changing any of these means the scientific payload interpretation changed even if the raw byte array did not.

## Relationship to PP-06.1

PP-06.1 exact-window evidence answers **where the bytes came from**. PP-06.2 answers **what those bytes mean**.

A future integration receipt must fail unless:

- PP-06.1 output content digest equals PP-06.2 payload content digest;
- PP-06.1 output raster shape equals PP-06.2 payload shape;
- both receipts refer to the same canonical derived artifact lineage.

Only after that joint verification should feature extraction consume the payload.

## Non-scope

This contract does not define provider decoding, raster reprojection, interpolation/resampling, radiometric calibration, terrain correction, cloud classification, feature extraction, model fitting, or predictive evaluation. Each materially different transform remains an explicit evidence-bearing step.

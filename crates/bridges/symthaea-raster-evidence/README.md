# Symthaea Raster Evidence Bridge

This crate joins two independently reviewable Planetary Perception contracts:

- PP-06.1 / #220: exact-window provenance — which immutable fixture/source/pixels produced the output bytes;
- PP-06.2 / #237: canonical payload semantics — how those exact bytes are interpreted as stored samples and physical values.

The bridge accepts a `FrozenSentinelFixtureManifest`, `ExactPixelWindowEvidence`, and `RasterPayloadDescriptor`, then fail-closes unless:

1. the exact-window receipt verifies against the authoritative fixture;
2. the payload content digest exactly matches the exact-window output digest;
3. payload rows/columns exactly match the exact-window output shape;
4. known byte lengths match;
5. the payload interpretation receives a stable, domain-separated semantic digest using explicit tags and exact IEEE-754 scale/offset/NoData bits.

The resulting `CanonicalRasterEvidence` is content-addressed and verifies its own persisted representation, but self-consistency is not authority: callers must still run `verify_against(...)` before using a loaded receipt as evidence.

## Non-scope

No provider I/O, GeoTIFF/JP2 decoding, feature extraction, masking, resampling, reprojection, calibration, terrain correction, model fitting, forecasting, or scientific capability claim lives here.

## Qualification

This integration must remain draft until #220 and #237 have executable qualification evidence. The current GitHub Actions queue is not a green gate. #202's named 256-bit digest-length blocker also remains upstream and must be closed before promotion.

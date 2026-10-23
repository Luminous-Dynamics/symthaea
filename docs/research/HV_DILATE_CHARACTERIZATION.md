# HDC qualification scaffold

Experimental / qualification-only. No production ABI or default representation change is implied.

## Contract

| Dimension | Requirement |
|---|---|
| identity | Representation family, algorithm/version, dimension, encoding and serialization are explicit |
| geometry | Metric and normalization are declared; incompatible spaces do not compare implicitly |
| algebra | Bind, inverse/unbind, bundle and permutation laws are classified exact, approximate, or N/A |
| dimension | No silent resize, truncation, padding, or resampling |
| determinism | Seed/generator identity is explicit and replayable |
| storage | Resident bytes and materialization bytes are separately accounted |
| conversion | Named conversion family, source/target descriptors, and measured distortion |
| evidence | Results carry workload, seed, hardware, implementation and provenance |

## Qualification matrix

Record results across 1K, 2K, 4K, 8K, 16K, 32K and 64K where supported. Unsupported cells are `not_applicable`, never zero.

## Negative controls

- mismatched dimensions;
- mismatched representation families;
- hidden allocation/conversion;
- serialization round-trip drift;
- nondeterministic generation;
- accidental fallback to dense 16K BinaryHV.

## Exit criteria

A machine-readable companion result distinguishes exact invariants from empirical tolerances and can be consumed by adaptive-representation and cost-quality benchmarks.

## Related

#6531 #6533 #6535 #6536 #6537 #6540 #6541 #6542 #6543 #6545 #6546

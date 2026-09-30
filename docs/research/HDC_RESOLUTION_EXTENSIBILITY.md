# HDC Resolution Extensibility Contract

## Status

Research infrastructure only. This document does not authorize production defaults above 64K.

## Design rule

The representation space is open-ended, while the empirical ladder is finite and versioned.

- Canonical: 1K, 2K, 4K, 8K, 16K, 32K, 64K.
- Exploratory: 128K, 256K.
- Custom: any positive power-of-two dimension accepted by the validated metadata type.

Architectural extensibility is not evidence that larger dimensions improve task quality.

## Resource accounting

For f32 ContinuousHV:

| Dimension | One vector |
|---:|---:|
| 64K | 256 KiB |
| 128K | 512 KiB |
| 256K | 1 MiB |
| 512K | 2 MiB |
| 1M | 4 MiB |

For packed binary HDC:

| Dimension | One vector |
|---:|---:|
| 64K | 8 KiB |
| 128K | 16 KiB |
| 256K | 32 KiB |
| 512K | 64 KiB |
| 1M | 128 KiB |

These are representation sizes, not measurements of physical cache traffic or energy.

## Acceptance boundary

A larger resolution should become an evidence-backed tier only when measurements establish its quality/cost frontier relative to the current tier.

Required evidence:

1. representation fidelity;
2. binding, bundling, permutation and similarity fidelity;
3. liquid trajectory fidelity;
4. deterministic transition behavior;
5. working-set and allocation cost;
6. SIMD/scalar numerical conformance;
7. throughput and logical byte traffic;
8. quality per byte and, when hardware telemetry exists, quality per joule.

## Current implementation

HdcResolution validates positive power-of-two dimensions without a fixed upper bound and exposes checked byte-size accounting. Canonical and exploratory classifications are explicit.

The extended SIMD conformance test exercises 128K and 256K. The extended benchmark group measures dot product and norm at those dimensions without expanding the expensive bundle sweep.

This keeps the default controlled ladder intact while making the research space extensible.

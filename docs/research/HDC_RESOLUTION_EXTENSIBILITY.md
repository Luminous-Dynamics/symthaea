# HDC Resolution Extensibility

## Research status

This is research infrastructure. It does not make >64K a production default.

## Two separate concepts

The representation space is open-ended: any positive power-of-two dimension can be represented by the validated resolution type.

The evidence ladder is finite and explicit:

- Canonical: 1K, 2K, 4K, 8K, 16K, 32K, 64K.
- Exploratory: 128K, 256K.
- Custom: future power-of-two dimensions that have not yet earned empirical status.

This prevents a benchmark ladder from silently becoming an architectural maximum.

## Memory accounting

One f32 ContinuousHV:

| Dimension | Bytes |
|---:|---:|
| 64K | 256 KiB |
| 128K | 512 KiB |
| 256K | 1 MiB |
| 512K | 2 MiB |
| 1M | 4 MiB |

One packed binary hypervector:

| Dimension | Bytes |
|---:|---:|
| 64K | 8 KiB |
| 128K | 16 KiB |
| 256K | 32 KiB |
| 512K | 64 KiB |
| 1M | 128 KiB |

These are representation sizes, not physical cache traffic or energy measurements.

## Evidence gate for larger tiers

Moving a tier from exploratory to canonical requires evidence across:

1. representation fidelity;
2. binding, bundling, permutation and similarity fidelity;
3. liquid trajectory fidelity;
4. deterministic conversion and round-trip behavior;
5. resident working-set and allocation cost;
6. scalar/SIMD numerical conformance;
7. throughput and logical byte traffic;
8. task quality per byte and, where telemetry is available, per joule.

The controller should therefore optimize minimum sufficient resolution rather than assume that more dimensions are better.

## Why this matters now

The current HDC-LTC trajectory work treats 1K..64K as the controlled matrix. This change makes 128K and 256K possible without changing that evidence boundary, so the next experiment can measure whether extra dimensionality buys enough quality to justify its working-set cost.

As a reference point, current accelerator hardware can hold substantially larger working sets; AMD lists 192 GB HBM3 and 5.3 TB/s peak bandwidth for MI300X. Hardware capacity therefore does not by itself establish that larger HDC vectors are useful—the relevant question remains quality and cost per operation. 


## Resource admissibility is a separate contract

`HdcResolution` answers whether a dimension is mathematically representable in
the research space. It does **not** authorize an allocation or imply that the
dimension is experimentally qualified.

`ResolutionBudget` provides the next boundary:

- per-vector byte ceiling;
- resident working-set ceiling;
- checked multiplication for resident-vector counts;
- representation-aware f32 and binary sizing.

This separation is deliberate:

| Question | Contract |
| --- | --- |
| Is the dimension valid? | `HdcResolution` |
| Has the dimension been empirically qualified? | canonical/exploratory/custom evidence class |
| Can this workload fit the declared resource envelope? | `ResolutionBudget` |
| Is the dimension preferable for a task? | future quality/cost evidence |

Rust's checked integer multiplication returns `None` on overflow, which is the
required behavior for converting an open resolution space into bounded working
set accounting. citeturn0search0

The budget is research infrastructure, not a production adaptive policy.
Future adaptive selection should consume measured quality and cost evidence
alongside such a resource envelope rather than treating a valid resolution as
automatically admissible.

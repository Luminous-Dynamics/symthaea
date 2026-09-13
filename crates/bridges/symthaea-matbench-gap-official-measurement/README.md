# Symthaea Matbench Gap Official Measurement

This crate is the source-authenticated execution facade for the Matbench experimental-gap Benchmark Zero stack.

It exists because the lower-level `symthaea-matbench-gap-frozen-measurement` crate intentionally accepts a parsed `MatbenchGapDataset`. That kernel can prove internal coherence of the parsed dataset, exposure plan, frozen comparison, restricted truth and exact-universe metrics, but `MatbenchGapDataset` is publicly constructible and therefore cannot itself prove that its values came from the official compressed artifact bytes.

The authoritative entry point here closes that gap:

```text
compressed bytes
    ↓
parse_official_matbench_expt_gap
    ↓ exact pinned SHA-256 before decompression
verified parsed dataset
    ↓
post-freeze truth restriction
    ↓
frozen exact-universe measurement
    ↓
official source-authenticated receipt
```

## Public authority path

`measure_official_matbench_bytes(plan, freeze, compressed_bytes)` accepts no caller-supplied parsed dataset. It invokes #1722's official parser internally and only then passes the parser-produced value to the lower-level measurement kernel.

`verify_official_matbench_measurement(...)` repeats that full path from the compressed bytes and requires exact final receipt equality.

This distinction is deliberate:

- `OfficialFrozenMeasurementReceipt::validate()` proves internal receipt composition.
- `verify_official_matbench_measurement(...)` proves source-byte replay through the official parser.
- A serialized receipt by itself does **not** authenticate remote provenance or source bytes.

## Nested evidence binding

The official receipt requires the nested restricted-truth and measurement receipts to agree on:

- exact official compressed SHA-256;
- source exposure-plan identity;
- comparison-subject identity;
- restricted-truth content identity;
- candidate-universe identity and count;
- restricted-truth receipt identity.

Two individually valid receipts from different comparison subjects therefore cannot be combined into one valid official receipt.

## Pinned source

Current official artifact SHA-256:

`783e7d1461eb83b00b2f2942da4b95fda5e58a0d1ae26b581c24cf8a82ca75b2`

The actual parser in `symthaea-matbench-gap` is the authority for the complete dataset pin, schema, row count and bounded decompression rules.

## Non-claims

This facade does not establish:

- absence of all historical/manual/near-neighbor leakage;
- calibrated uncertainty;
- statistical significance;
- model or policy superiority;
- material novelty, feasibility or safety;
- experiment or deployment authority.

It only upgrades the final Benchmark Zero execution path from "coherent parsed dataset" to "replayed from the exact SHA-pinned compressed source bytes."
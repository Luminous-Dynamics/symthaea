# Exact HDC word-level primitives

## `hdc_bind_xor.sv`

A parameterized word-level XOR bind/unbind primitive, with default width 256 and a single-entry ready/valid output register.

## `hdc_hamming_count.sv`

A parameterized word-level Hamming count primitive: count set bits in `operand_a XOR operand_b`. The default result width is `$clog2(DATA_W + 1)`. For a canonical 16,384-bit BinaryHV, a 256-bit datapath produces a partial count for each of 64 word operations; a separate vector controller/accumulator is still required.

The first popcount implementation is deliberately a straightforward combinational loop. It is a functional baseline, **not** an optimized tree, timing-closed design or proven low-power implementation. Next architecture sweep should compare balanced/grouped popcount and pipelined variants at identical semantics.

## Intended simulation

With the pinned HDL environment:

```sh
nix develop .#hdl
bash crates/domains/symthaea-silicon/eda/run_hdc_smoke.sh
```

The HDC smoke runner should produce independent per-width receipts and logs. No simulation should be claimed until the pinned tool actually runs and the expected PASS marker is observed.

## Qualification boundary

RTL source presence != elaborated RTL != passing differential simulation != formal proof != synthesis/PPA != FPGA hardware observation != ASIC qualification. These modules do not yet implement vector DMA, memory control, CPU/GPU, runtime integration or a complete accelerator.

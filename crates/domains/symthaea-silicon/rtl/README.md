# HDC RTL reference primitives

## `hdc_bind_xor`

This is the first synthesizable RTL primitive in the NEURO-SILICON-001B line. It implements the canonical `BinaryHV` bind rule:

```text
bind(a, b) = a XOR b
```

The default width is 16,384 bits (2,048 bytes), matching `symthaea_core::hdc::binary_hv::BinaryHV`. The module is deliberately combinational and has no clock, protocol, buffering, memory, or control authority. It is a semantic reference point; it is not presented as the best physical architecture.

## Testbench

`hdc_bind_xor_tb.sv` checks zero/all-one boundaries, complementary structured vectors, a deterministic full-width pattern, and a walk-one at every bit on both input ports, including same-bit cancellation.

Using Icarus Verilog:

```sh
iverilog -g2012 -Wall -s hdc_bind_xor_tb -o /tmp/hdc_bind_xor_tb.vvp \
  crates/domains/symthaea-silicon/rtl/hdc_bind_xor.sv \
  crates/domains/symthaea-silicon/rtl/hdc_bind_xor_tb.sv
vvp /tmp/hdc_bind_xor_tb.vvp
```

An optional Verilator route, if installed with timing simulation support:

```sh
verilator --binary --timing --top-module hdc_bind_xor_tb -Wall \
  crates/domains/symthaea-silicon/rtl/hdc_bind_xor.sv \
  crates/domains/symthaea-silicon/rtl/hdc_bind_xor_tb.sv
./obj_dir/Vhdc_bind_xor_tb
```

The expected terminal marker is `PASS: hdc_bind_xor_tb; WIDTH=16384; all directed and per-bit checks completed`. Until a simulator run is captured with the exact source SHA, tool version, command, exit code, and output, this is testbench intent—not a simulation PASS.

## Semantic/evidence boundary

- The canonical Rust `BinaryHV` remains the source of HDC semantics.
- This initial bench is a directed RTL self-check; it is not yet a Rust-oracle differential run.
- Next step: consume source-bound vectors produced by `examples/hdc_rtl_reference_corpus.rs` and compare exact RTL outputs with those expected values.
- A simulation PASS would not establish FPGA implementation, physical-board observation, speedup, lower system energy, PPA closure, ASIC readiness, fabricated silicon, or neuromorphic advantage.
- Compare narrower 512/256/128/64-bit streaming and storage profiles only after primitive semantics are frozen. Count buffering, transfer, and protocol overhead in any eventual end-to-end benchmark.

## Source anchors

The canonical Rust implementation is `crates/core/symthaea-core/src/hdc/binary_hv.rs`, blob SHA `22a56cfaedf5b0cb7d8ff8b73c41ed4caf8d7056` on the parent branch used for this tranche. The software-oracle corpus generator is introduced by [PR #5813](https://github.com/Luminous-Dynamics/symthaea/pull/5813), pinned to oracle commit `458c7b98d81c64b9361e252f85ef9d45132e6682`.

# HDC RTL reference primitives

## `hdc_bind_xor`

This is the first synthesizable RTL primitive in the NEURO-SILICON-001B line. It implements the canonical `BinaryHV` bind rule:

```text
bind(a, b) = a XOR b
```

The default width is 16,384 bits (2,048 bytes), matching `symthaea_core::hdc::binary_hv::BinaryHV`. The module is deliberately combinational and has no clock, protocol, buffering, memory, or control authority. It is a semantic reference point; it is not presented as the best physical architecture.

## Directed RTL testbench

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

## Canonical Rust-oracle differential run

The stronger runner below is tracked under [NEURO-SILICON-001B-C #7716](https://github.com/Luminous-Dynamics/symthaea/issues/7716). It does not reimplement HDC algebra. It consumes the frozen corpus generated from the canonical `BinaryHV`, preserves the oracle and benchmark-contract pins, reverses complete bytes to map canonical byte 0 to RTL bus bits `[7:0]`, and checks exact expected outputs for the `bind` and `bind_chain` fixtures.

Run it from a clean checkout at the exact revision being qualified:

```sh
cargo run --locked -p symthaea-silicon --example hdc_rtl_reference_corpus > /tmp/hdc-corpus.json
cargo run --locked -p symthaea-silicon --example hdc_bind_rtl_simulator_diff -- /tmp/hdc-corpus.json
```

The tool writes a JSON receipt to stdout containing the repository head, oracle and contract pins, Git blob IDs for the RTL/oracle/corpus/generated bench, simulator version, commands, exit codes, warnings, tested fixture IDs, raw output, and the retained evidence directory. It fails closed if a required tool is missing, the source is dirty/untracked, a pin is wrong, a fixture is missing/malformed, the simulator fails, or compiler warnings require review. Other corpus operations are reported as unsupported; they are not run through an XOR module.


The differential runner additionally regenerates the corpus from the clean, tracked Rust example and requires byte-for-byte equality with the supplied file. It compares the current canonical oracle and benchmark-contract file blobs to those stored in their pinned commits; merely carrying matching commit-ID strings is not sufficient. Receipt fields include the generator source blob and the regenerated corpus hash.


## Ready/valid streaming XOR profiles

`hdc_bind_xor_stream.sv` applies the same exact XOR operation to one lane per accepted transfer. The one-entry elastic stage uses a ready/valid handshake; an output beat must remain valid and stable while the consumer is stalled. The final-beat marker is forwarded unchanged.

The lane sweep bench runs each profile over the same 16,384-bit vectors:

| Lane width | Transfers per vector |
|---:|---:|
| 64 bits | 256 |
| 128 bits | 128 |
| 256 bits | 64 |
| 512 bits | 32 |
| 16,384 bits | 1 |

These counts follow directly from the declared vector/lane widths. They are not latency, frequency, area, power or energy measurements. The bench applies the same deterministic downstream-stall schedule, reconstructs the full output vector, asserts the `last` marker, and checks output stability during backpressure.

Run with Icarus Verilog:

```sh
iverilog -g2012 -Wall -s hdc_bind_xor_stream_tb -o /tmp/hdc_bind_xor_stream_tb.vvp \\
  crates/domains/symthaea-silicon/rtl/hdc_bind_xor_stream.sv \\
  crates/domains/symthaea-silicon/rtl/hdc_bind_xor_stream_tb.sv
vvp /tmp/hdc_bind_xor_stream_tb.vvp
```

Only a captured exact-head simulator receipt qualifies the RTL/protocol bench. The next comparison should use frozen oracle vectors and then pinned synthesis/implementation flows; simulated cycles under artificial backpressure are not physical performance.

## Semantic/evidence boundary

- The canonical Rust `BinaryHV` remains the source of HDC semantics.
- The directed testbench is not a Rust-oracle differential run. The differential runner is a separate, stronger gate.
- A simulation PASS would not establish FPGA implementation, physical-board observation, speedup, lower system energy, power/area/timing closure, ASIC readiness, fabricated silicon, or neuromorphic advantage.
- Compare narrower 512/256/128/64-bit streaming and storage profiles only after primitive semantics are frozen. Count buffering, transfer, and protocol overhead in any eventual end-to-end benchmark.

## Source anchors

The canonical Rust implementation is `crates/core/symthaea-core/src/hdc/binary_hv.rs`, blob SHA `22a56cfaedf5b0cb7d8ff8b73c41ed4caf8d7056` on the parent branch used for this tranche. The software-oracle corpus generator is introduced by [PR #5813](https://github.com/Luminous-Dynamics/symthaea/pull/5813), pinned to oracle commit `458c7b98d81c64b9361e252f85ef9d45132e6682` and benchmark-contract commit `68370c5431c8843553380c70d21952db5696cb50`.

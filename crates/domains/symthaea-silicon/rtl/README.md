# Exact HDC word-level primitives

## `hdc_bind_xor.sv`

Parameterized exact XOR bind/unbind primitive with a one-entry ready/valid output register.

## `hdc_hamming_count.sv`

Parameterized exact per-word Hamming count: count set bits in `operand_a XOR operand_b`. The output width defaults to `$clog2(DATA_W + 1)`.

It has two selectable combinational architectures:
- `BALANCED_TREE=0`: linear accumulation chain; deliberately simple reference.
- `BALANCED_TREE=1`: balanced adder tree; a different timing/area trade-off under the same semantics.

For the 16,384-bit canonical BinaryHV, a 256-bit datapath needs 64 ordered words for the complete vector. These modules do not yet implement the vector controller, 64-part accumulator, local memory, DMA, permutation unit, CPU, GPU or OS/runtime integration. The popcount block returns one word's partial count.

## Intended correctness sweep

From the repository root:

```sh
nix develop .#hdl
bash crates/domains/symthaea-silicon/eda/run_hdl_smoke.sh
```

The runner is designed to execute twelve cases:
- XOR at 64, 128, 256 and 512 bits;
- linear-popcount at the same four widths;
- balanced-tree-popcount at the same four widths.

Each case must lint, compile and simulate successfully and emit its expected PASS marker. Receipts bind source hashes, runner/manifest/flake/lock hashes, repository HEAD, simulator version, per-case exit codes and hashed logs. The missing-Verilator path was exercised locally on a reconstructed harness and returned exit 2 with a valid `unqualified_tool_missing` JSON receipt. The real RTL sweep has not run.

## How to interpret results

This is **correctness qualification, not a speed benchmark**. A balanced tree is hypothesized to reduce combinational logic depth relative to a linear addition chain, but area, timing and power have not been measured. Run the exact variants through Yosys/target-specific synthesis only after simulator qualification; bind PDK/library/constraints/corners before comparing physical-design estimates. FPGA measurements require a named board and a distinct physical-observation receipt.

## Evidence boundary

`RTL source` != `elaborated RTL` != `passing simulation` != `formal proof` != `synthesis/PPA` != `FPGA hardware observed` != `ASIC-qualified` != `measured product advantage`.

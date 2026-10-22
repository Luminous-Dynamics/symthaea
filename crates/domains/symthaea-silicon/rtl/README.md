# HDC XOR RTL slice (prototype)

This directory begins the #5808 hardware-equivalence path with one synthesizable word-level primitive.

## Semantics

- Each accepted input word produces exactly `operand_a XOR operand_b`.
- It is parameterized by datapath width; default is 256 bits.
- The canonical Symthaea hypervector is 16,384 bits / 2,048 bytes, so a 256-bit implementation requires 64 ordered word operations per vector. This module has no vector-address/controller logic and does not itself implement permutation, Hamming distance, bundling, DMA or local memory.
- One output register supplies a ready/valid contract. While output is blocked, `out_valid` and `result` must remain stable.

## Intended first simulation

With a compatible Verilator installation:

```sh
verilator --binary --timing -Wall \
  --top-module tb_hdc_bind_xor \
  crates/domains/symthaea-silicon/rtl/hdc_bind_xor.sv \
  crates/domains/symthaea-silicon/rtl/tb_hdc_bind_xor.sv
./obj_dir/Vtb_hdc_bind_xor
```

Tool availability and command syntax must be rechecked against the pinned tool version before treating this as a run receipt. The testbench has not been executed during authoring: Verilator/Yosys/iverilog and Rust tools are not installed in the available execution environment.

## Qualification boundary

RTL source presence != parsed/elaborated RTL != passing simulation != formal proof != synthesis/PPA != FPGA hardware observed != ASIC qualification.

This is a starter artifact only. It does not prove CPU/GPU performance, area, frequency, power/energy efficiency, timing closure, or product readiness.

# HDL qualification dev shell and correctness sweep

The root flake exposes `nix develop .#hdl` for the Symthaea silicon word-primitive smoke sweep. It uses the repository's existing `flake.lock`; it does not add an unpinned Nixpkgs input.

From the repository root:

```sh
nix develop .#hdl
bash crates/domains/symthaea-silicon/eda/run_hdl_smoke.sh
```

The runner defines twelve cases:

- exact HDC XOR at 64/128/256/512-bit widths;
- linear-accumulation Hamming count at the same widths;
- balanced-tree Hamming count at the same widths.

Each case must pass lint, build and simulation, exit successfully and emit the exact expected testbench PASS marker. The JSON receipt records per-case status, source hashes for modules/testbenches/runner/manifest, repository HEAD, root `flake.nix` and `flake.lock` hashes, simulator version, and hashed logs. Missing tools yield `unqualified_tool_missing`; failed cases yield a non-pass status.

Local checks performed: a copy of the current runner passed `bash -n`; a harness with the Verilator tool intentionally absent exited 2 and emitted a valid JSON receipt that named Verilator as missing. A separate mathematical reference check compared linear and balanced popcount results over 144 patterned/generated cases across the four widths without arithmetic mismatches.

**The RTL itself has not been compiled or simulated** because Verilator is not available in the authoring environment. The repository flake has not been evaluated here either. The next qualification gate is to run this exact sweep within the pinned Nix shell on a host where Nix is installed.

This is not a full ASIC signoff environment. These tests do not establish RTL formal proof, synthesis/PPA, FPGA hardware behavior, physical energy or product/SOTA superiority.

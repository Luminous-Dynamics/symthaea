# HDL qualification dev shell

The root flake exposes `nix develop .#hdl` for the first Symthaea silicon RTL smoke test. It uses the repository's existing `flake.lock`; no separate unpinned Nixpkgs input is introduced.

From the repository root:

```sh
nix develop .#hdl
bash crates/domains/symthaea-silicon/eda/run_hdl_smoke.sh
```

A successful receipt means only that the exact HDC XOR RTL/testbench passed the pinned Verilator lint/build/smoke command. The receipt records source hashes, repository head, simulator version, stage exit codes, log hashes and a claim ceiling. Missing tools or failures produce a non-pass status. Outputs are under the ignored `target/hdl-qualification/` directory; preserve the output directory when retaining evidence.

This shell does not yet include a complete qualified FPGA or ASIC implementation flow. Yosys is provided for the next stage, but synthesis results are not produced by this smoke script.

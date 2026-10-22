# HDL qualification dev shell

The root flake exposes `nix develop .#hdl` for the first Symthaea silicon RTL smoke sweep. It uses the repository's existing `flake.lock`; no separate unpinned Nixpkgs input is introduced.

From the repository root:

```sh
nix develop .#hdl
bash crates/domains/symthaea-silicon/eda/run_hdl_smoke.sh
```

The runner checks the HDC XOR RTL and parameterized testbench at 64-, 128-, 256- and 512-bit datapath widths. Each width gets independent lint/build/simulation logs and an exit-code result; the aggregate receipt passes only when all widths pass and the expected PASS marker appears.

A successful receipt means only that the exact word-level RTL/testbench passed this pinned Verilator smoke command. It records source hashes, repository head, simulator version, width-specific exit codes and a claim ceiling. Missing tools or failures produce a non-pass status. Outputs are under the ignored `target/hdl-qualification/` directory; retain the full output directory when retaining evidence.

The local authoring environment did not have Verilator; I validated the runner's Bash syntax and its missing-tool path, but **have not compiled or simulated the RTL**. The root flake has not yet been evaluated by Nix in this environment. This is not a qualified ASIC or FPGA toolchain. Yosys is included for a subsequent stage, but the smoke script does not yet run synthesis.

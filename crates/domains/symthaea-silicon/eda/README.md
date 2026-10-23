# HDL qualification dev shell and correctness sweep

The root flake exposes `nix develop .#hdl` for the Symthaea silicon word-primitive sweep. It uses the repository's existing `flake.lock`; it does not add an unpinned Nixpkgs input.

From the repository root:

```sh
nix develop .#hdl
bash crates/domains/symthaea-silicon/eda/run_hdl_smoke.sh
bash crates/domains/symthaea-silicon/eda/validate_hdl_receipt.sh target/hdl-qualification/hdc/<run>/receipt.json
```

The runner defines twelve cases:

- exact HDC XOR at 64/128/256/512-bit widths;
- linear-accumulation Hamming count at the same widths;
- balanced-tree Hamming count at the same widths.

Each case must pass lint, build and simulation, exit successfully and emit the exact expected testbench PASS marker. The JSON receipt records per-case status, source hashes for modules/testbenches/runner/manifest/schema/case matrix/validator, repository HEAD, root `flake.nix` and `flake.lock` hashes, simulator version, and hashed logs. Missing tools yield `unqualified_tool_missing`; failed cases yield a non-pass status.

`hdl_expected_case_matrix_v1.json` freezes the exact twelve case IDs. `hdl_run_receipt_v1.schema.json` describes receipt shape and status invariants. `validate_hdl_receipt.sh` separately checks that the receipt covers each expected case exactly once, all stage codes/pass markers succeed, and the recorded source/configuration hashes still match the current checkout. Use the exact recorded Git commit to revalidate a historical receipt; a newer source tree should not silently validate an old run.

## Checks completed while tools are unavailable

- A local copy of the runner passed `bash -n`.
- The runner's missing-Verilator negative control exited 2 and produced a valid `unqualified_tool_missing` receipt.
- A local copy of the receipt validator accepted a synthetic 12-case complete receipt and rejected a synthetic receipt missing one case.
- A mathematical reference check compared linear and balanced popcount results over 144 generated/patterned cases across the four widths with no arithmetic mismatches.

These checks do **not** simulate or compile the RTL. The RTL sweep, root flake evaluation, formal proof, synthesis/PPA and physical FPGA measurements remain unexecuted because Verilator/Nix are unavailable on the authoring host. This is not a full ASIC signoff environment or evidence of a product/SOTA advantage.

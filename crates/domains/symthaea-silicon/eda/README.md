# HDL qualification dev shell and canonical-oracle correctness sweep

The root flake exposes `nix develop .#hdl` for the Symthaea silicon word-primitive sweep. It reuses the repository's locked Rust toolchain and `flake.lock`, and includes Verilator plus the small JSON/hash utilities used by the receipt path.

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

Before RTL simulation, it runs `hdc_rtl_oracle.rs` using `cargo run --locked --offline`. The generator consumes the canonical `symthaea_core::hdc::BinaryHV` APIs, writes per-width `$readmemh` files for XOR expected results and per-word Hamming counts, and checks that the sum of word counts equals the canonical full-vector Hamming distance. The generator binds the current Git HEAD, `binary_hv.rs` SHA-256, root `Cargo.lock` SHA-256, and each emitted memory-file hash. An identity mismatch or fixture hash mismatch blocks simulation.

Each RTL case must pass lint, build and simulation, exit successfully, emit its local PASS marker, and emit the software-oracle PASS marker. The versioned JSON receipt records per-case status, source hashes for modules/testbenches/generator/Cargo manifest/canonical HDC source/lock file/runner/manifest/schema/case matrix/validator/flake inputs, the generated oracle-receipt hash, Verilator version and hashed logs.

`hdl_expected_case_matrix_v1.json` freezes the exact twelve case IDs. `hdl_run_receipt_v1.schema.json` requires all twelve case results to pass, including oracle markers. `validate_hdl_receipt.sh` checks that each expected case appears exactly once, verifies source/configuration hashes and Git HEAD, and recomputes the hashes of all software-oracle memory files.

## Validation status for this draft

The environment used to author this change did not have Cargo, Rust, Nix or Verilator. Consequently the new Rust oracle generator has **not** been compiled and the twelve RTL cases have **not** been simulated. Earlier shell/receipt-only checks on the harness don't validate this new source-to-test integration. The immediate next gate is to evaluate `nix develop .#hdl`, compile the fixture generator, generate the mem files, and run all twelve cases. Any compile, file-format, or simulation failure must be fixed before claiming functional equivalence.

This is still a word-level correctness experiment, not synthesis/PPA, formal proof, FPGA observation, physical energy, ASIC qualification or evidence of product/SOTA superiority.

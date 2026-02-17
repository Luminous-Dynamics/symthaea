# Gen7 zkSTARK Stack Status (December 2025)

**Last updated:** 12 Nov 2025  
**Owner:** Gen7 zk / bindings team  

## TL;DR
- `gen7-zkstark/` is a RISC Zero–first code path that bundles host, guest, and PyO3 bindings for gradient-provenance proofs.
- It does **not** contain Winterfell AIR sources; the only `winterfell-pogq` artifact is another `DEBUGGING_GUIDE.md`, so the missing crate cannot be recovered from this tree.
- Python bindings (`gen7_zkstark`) expose two entry points—`prove_gradient_zkstark` and `verify_gradient_zkstark`—implemented in `src/lib.rs`.

## Layout Highlights
| Component | Path | Notes |
| --- | --- | --- |
| PyO3 bindings | `gen7-zkstark/src/lib.rs` | Defines `prove_gradient_zkstark` / `verify_gradient_zkstark`, hashes model + gradient, pushes witness into zkVM. |
| RISC Zero host | `gen7-zkstark/host/src/main.rs` | Standard zkVM host wrapper compiled via `cargo run`. |
| Methods (guest) | `gen7-zkstark/methods/guest/src/main.rs` | zkVM guest logic executed inside the receipt. |
| Python package | `gen7-zkstark/bindings/` | `pyproject.toml`, `gen7_zkstark/__init__.py`, and tests for bindings. |
| Docs | `gen7-zkstark/HANDOFF.md`, `README.md` | Currently the stock RISC Zero starter README; needs customization. |

## Status & Next Steps
1. **Validation**: Ensure `poetry run maturin develop` (or equivalent) builds the bindings, and add CI coverage for `gen7_zkstark/python/tests/test_bindings.py`.
2. **Documentation**: Replace the template README with project-specific instructions (host/guest build, Python usage, integration with 0TML coordinator).
3. **Integration**: Wire the new bindings into the main repo once feature-parity with existing zkVM backend is confirmed.

## Relationship to Winterfell
The `gen7-zkstark` repository does not contain any of the missing Winterfell AIR sources (`air.rs`, `trace.rs`, `prover.rs`). Recovery of `vsv-stark/winterfell-pogq` still requires external backups; this folder cannot be used for that purpose.

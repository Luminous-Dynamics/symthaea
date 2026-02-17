# Day 2 Progress: Fixed-Point Loss Stack (Nov 7, 2025)

## ✅ Deliverables Completed
- Implemented canonical MSE-on-logits loss stack (`vsv-core/src/loss.rs`) and re-exported helpers through the crate root.
- Added Q16.16 integer constructors in `Fixed` to support exact divisors and loss normalisation.
- Generated deterministic pretrained weights and calibration fixtures via `scripts/export_weights.py` (`--samples 8`) → `vsv-core/src/weights.rs`, `vsv-core/src/calibration.rs`.
- Wired CanaryCNN to the generated weights (`CanaryCNN::pretrained`/`Default`) and added calibration-aligned regression tests (ε\_logit ≤ 0.05, ε\_loss ≤ 0.02).
- Introduced Criterion benches (`benches/loss.rs`) for `mse_single` and `loss_delta_mse`; integrated `criterion` dev-dependency and bench target.
- Verified zkVM FHS devShell bootstrap with `nix develop 'path:/srv/luminous-dynamics/Mycelix-Core/0TML/vsv-stark'#fhs --command bash -lc '...'`.

## 🧪 Validation
- `cargo test -p vsv-core` (rustc 1.91.0) — 23 unit tests + 4 doc tests passing; calibration tests confirm Q16.16 forward pass matches PyTorch logits within tolerance.
- `cargo bench -p vsv-core --bench loss -- --warm-up-time 1`
  - `mse_single`: 34.0 ns ±0.3 ns
  - `loss_delta_mse`: 68.0 ns ±0.4 ns
- Export script validation: max weight conversion error 1.5e-5 (matches Q16.16 resolution).

## 📊 Benchmarks Snapshot
- Loss delta throughput: ~14.6M ops/sec on amd64 dev shell.
- Calibration fixtures: 8 samples (inputs, logits, labels) embedded for deterministic tests.
- Criterion HTML reports stored under `target/criterion/` (not checked in).

## 🚧 Outstanding Tasks
1. Generate trained CanaryCNN weights (replace random seed artefact) and refresh calibration set.
2. Add Criterion run to CI and capture baseline numbers in docs.
3. Port loss + CanaryCNN modules into `methods/guest` once RISC Zero toolchain installed (Day 3).
4. Document `nix develop` usage in README (include `path:` form to bypass git tracking requirement).

## 📝 Notes
- MSE-on-logits remains canonical through Phase 1; any cross-entropy exploration tracked in `docs/PHASE2_CE_LOSS_STUDY.md`.
- All APIs are slice-based (`&[Fixed]`), making the zkVM port a direct lift once guest environment is ready.

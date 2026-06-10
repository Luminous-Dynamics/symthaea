# Symthaea Repo Maturity Note

This note documents the engineering discipline and baseline stability of the Symthaea workspace.

## 1. Dependency Hygiene
*   **Canonical Path Strategy:** All shared workspace dependencies (e.g., `mycelix-zkp-core`, `symthaea-core`) are canonicalized in the root `Cargo.toml`.
*   **Collision Prevention:** Member crates inherit dependencies via `workspace = true`, eliminating duplicate packages in the lockfile.
*   **Offline Reproducibility:** The workspace is verified to compile and test cleanly in `--offline` mode, ensuring no hidden network dependencies.

## 2. Blast Radius Rule
*   Modifications are strictly isolated to target crates.
*   Patched crypto/network dependencies (e.g., `iroh`, `ed25519-dalek`) are never modified from feature branches.
*   Experimental modules (e.g., `Morphogenesis`) are quarantined in separate crates with explicit stability warnings.

## 3. Automated Validation
*   **Core Science:** Over 5,500 unit tests in the core and member crates.
*   **Integration Checks:** Dedicated integration suites for hardware stress and perceptual loop closure.
*   **Numerical Integrity:** Physics solvers include convergence audits (L2/Energy norm) as part of the test suite.

## 4. Verification Checkpoint (2026-06-06)
*   `cargo metadata`: **PASS**
*   `cargo check -p symthaea --lib --offline`: **PASS**
*   `cargo test -p symthaea-core --lib hdc::fem::tests --offline`: **PASS**
*   `cargo test --lib cognitive_loop::consciousness_engine::tests::test_substrate_stress_multi_axis --offline`: **PASS**

**Baseline Tag:** `maturation-checkpoint-pr4-dependency-hygiene`

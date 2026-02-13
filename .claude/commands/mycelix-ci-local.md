---
description: "Run local CI validation before pushing - catch issues before GitHub Actions"
---

# Mycelix Local CI Validation

Run the same checks that CI would catch, locally, before pushing. Focuses on changed files to keep it fast.

Arguments: $ARGUMENTS

If "full" is passed, run everything. Otherwise, scope to changes.

## Step 1: Detect What Changed

!git diff --name-only HEAD~1..HEAD 2>/dev/null; git diff --name-only 2>/dev/null; git diff --name-only --cached 2>/dev/null

## Step 2: Determine Scope

Based on changed files, determine which checks to run:

- **Rust changes** (any `.rs` files in mycelix-*/) -> Rust checks
- **TypeScript changes** (any `.ts` files in sdk-ts/) -> TS checks
- **Python changes** (any `.py` files in sdk-python/) -> Python checks
- **Cargo.toml changes** -> Full Rust rebuild check
- **Bridge changes** -> Bridge + both cluster checks

## Step 3: Run Checks (in order)

### 3a. Rust Formatting (if Rust changed)

For each changed Rust workspace:
```bash
cargo fmt --check
```

Run in: mycelix-commons, mycelix-civic, crates/mycelix-bridge-common, mycelix-workspace/sdk, or whichever workspace has changes.

### 3b. Rust Clippy (if Rust changed)

```bash
cargo clippy --all-targets -- -D warnings
```

Run in the same workspaces as formatting.

### 3c. Rust Unit Tests (if Rust changed)

```bash
cargo test --lib
```

Run in changed workspaces only.

### 3d. WASM Compilation Check (if zome code changed)

```bash
cargo build --target wasm32-unknown-unknown --release -p {changed_zome_integrity} -p {changed_zome_coordinator}
```

Only if zome source files changed. This catches WASM-specific compilation issues (e.g., getrandom, std dependencies).

### 3e. TypeScript Checks (if TS changed)

```bash
cd mycelix-workspace/sdk-ts
npm run typecheck
npm run lint
npm test
```

### 3f. Python Checks (if Python changed)

```bash
cd mycelix-workspace/sdk-python
ruff check .
pytest -x
```

## Step 4: Report

Output a summary table:

```
Check                    | Status | Duration
-------------------------|--------|----------
Rust fmt (commons)       | PASS   | 1s
Rust clippy (commons)    | PASS   | 15s
Rust tests (commons)     | PASS   | 5s
WASM build (2 zomes)     | PASS   | 30s
TS typecheck             | PASS   | 8s
TS lint                  | PASS   | 3s
TS tests                 | PASS   | 45s
-------------------------|--------|----------
Total                    | PASS   | 1m 47s
```

If any check fails:
- Show the error output
- Suggest how to fix it
- Do NOT proceed to remaining checks for that language (fail fast per language, but check all languages)

## Full Mode

If `$ARGUMENTS` contains "full":
1. Run ALL Rust workspaces (not just changed)
2. Run ALL TypeScript tests
3. Run Python tests
4. Run WASM compilation for all zomes (`just build-wasm-all`)
5. Run sweettest if conductor available

## Quick Reference

This mirrors the CI pipeline in `.github/workflows/mycelix-ci.yml`:
- Formatting -> `cargo fmt --check`
- Linting -> `cargo clippy`
- Type checking -> `npm run typecheck`
- Unit tests -> `cargo test --lib` / `npm test` / `pytest`
- WASM builds -> `cargo build --target wasm32-unknown-unknown`

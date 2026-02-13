---
description: "Smart test runner - detect what changed and run the right Mycelix tests"
---

# Mycelix Smart Test Runner

Detect what files changed and run the minimal set of tests needed for confidence.

Arguments: $ARGUMENTS

If arguments are provided, use them as a scope hint (e.g., "commons", "sdk-ts", "bridge", "all").
If no arguments, auto-detect from git changes.

## Step 1: Detect Changes

!git diff --name-only HEAD 2>/dev/null || git diff --name-only --cached 2>/dev/null || echo "no changes detected"

Also check unstaged:
!git status --porcelain 2>/dev/null | head -30

## Step 2: Map Changes to Test Suites

Use this mapping to determine which tests to run:

### Rust Crate Changes
| Changed Path Pattern | Tests to Run |
|---------------------|-------------|
| `crates/mycelix-bridge-common/**` | `cargo test -p mycelix-bridge-common` + both cluster tests + SDK TS bridge tests |
| `crates/mycelix-bridge-entry-types/**` | `cargo test -p mycelix-bridge-entry-types` + both cluster tests |
| `mycelix-commons/zomes/**` | `cd mycelix-commons && cargo test` |
| `mycelix-civic/zomes/**` | `cd mycelix-civic && cargo test` |
| `mycelix-commons/zomes/commons-bridge/**` | Commons tests + cross-cluster SDK tests |
| `mycelix-civic/zomes/civic-bridge/**` | Civic tests + cross-cluster SDK tests |
| `mycelix-core/**` | `cd mycelix-core && cargo test` |
| `mycelix-identity/**` | `cd mycelix-identity && cargo test` |
| `mycelix-workspace/sdk/**` (Rust SDK) | `cd mycelix-workspace/sdk && cargo test --features simulation` |

### TypeScript Changes
| Changed Path Pattern | Tests to Run |
|---------------------|-------------|
| `mycelix-workspace/sdk-ts/src/**` | `cd mycelix-workspace && npm test --prefix sdk-ts` |
| `mycelix-workspace/sdk-ts/src/integrations/commons/**` | TS tests filtered to commons |
| `mycelix-workspace/sdk-ts/src/integrations/civic/**` | TS tests filtered to civic |
| `mycelix-workspace/sdk-ts/src/matl/**` | `npm test -- --grep matl` |
| `mycelix-workspace/sdk-ts/src/epistemic/**` | `npm test -- --grep epistemic` |
| `mycelix-workspace/sdk-ts/tests/**` | Relevant test file directly |

### Python Changes
| Changed Path Pattern | Tests to Run |
|---------------------|-------------|
| `mycelix-workspace/sdk-python/**` | `cd mycelix-workspace/sdk-python && pytest` |

### Sweettest Changes
| Changed Path Pattern | Tests to Run |
|---------------------|-------------|
| `mycelix-workspace/tests/sweettest/**` | `cd mycelix-workspace && cargo test --release -p sweettest -- --ignored` |

## Step 3: Run Tests

Based on the mapping above, run the appropriate tests. For each suite:

1. Print what's being tested and why
2. Run the test command
3. Report pass/fail with counts

### Quick Mode (default)
Run only unit tests (no conductor, no release builds). This is fast (~seconds to minutes).

### Full Mode (if user passes "full" or "all")
Also run:
- Sweettest: `cargo test --release -- --ignored --nocapture` (requires release build)
- Conductor tests: `CONDUCTOR_AVAILABLE=true npm run test:conductor` (requires running conductor)

## Step 4: Summary

Output a table:
```
Suite               | Tests | Status | Duration
--------------------|-------|--------|----------
Bridge Common       |    14 | PASS   | 2s
Commons Cluster     |   127 | PASS   | 5s
TypeScript SDK      | 6316  | PASS   | 45s
...
```

## Scope Shortcuts

- `commons` - Run only mycelix-commons tests
- `civic` - Run only mycelix-civic tests
- `bridge` - Run bridge-common + both cluster bridge tests
- `sdk` or `sdk-rust` - Rust SDK only
- `sdk-ts` - TypeScript SDK only
- `sdk-python` or `python` - Python SDK only
- `sweettest` - Run sweettest suite (requires --release)
- `all` or `full` - Everything
- No args - Auto-detect from git diff

## Notes

- Rust tests: use `cargo test --lib` for speed (skip doc tests)
- TypeScript: `npm test` in sdk-ts excludes conductor tests by default
- Sweettest MUST use `--release` (debug mode nonce lifetime too short)
- Python: `cd mycelix-workspace/sdk-python && pytest -v`

# Tight Test Loop: `cargo test --no-run` + Direct Binary Execution

**Date**: 2026-04-13
**Origin**: Phase I.A.5 hardening — Track 4.2 deliverable.

## Why this exists

`cargo test` re-acquires the global `~/.cargo/.package-cache-mutate` lock on
every invocation, even when the test binary is already built and only one
assertion changed. In a monorepo with 12+ concurrent Claude sessions, that
lock serializes test runs across **all** sessions — a single test can wait
5–15 minutes in the queue before it even starts, and `pkill` on a stuck
cargo can cascade into killing other sessions' work.

The fix is the standard Cargo two-phase pattern that every Rust dev knows but
nobody writes down: **compile the test binary once, then execute it directly
many times**, bypassing cargo entirely on subsequent runs.

## The pattern

### Step 1 — Build the test binary (one-time cost)

```bash
cargo test --no-run --test integration_rdp_wire --features mesh-encryption
```

This compiles the test crate but does NOT execute. The output names the
binary path:

```
   Compiling symthaea v2.0.0 (...)
    Finished `dev` profile [...] target(s) in 1m 41s
  Executable tests/integration_rdp_wire.rs (target/debug/deps/integration_rdp_wire-7a3f9b...)
```

Capture that path. The hash suffix changes when the test source changes; it
stays stable when only deps change.

### Step 2 — Execute directly (~0.5 sec per run)

```bash
target/debug/deps/integration_rdp_wire-7a3f9b... --nocapture
```

Or to filter:

```bash
target/debug/deps/integration_rdp_wire-7a3f9b... --nocapture wire_envelope
```

No cargo, no lock, no contention. The first run is the slow one; iteration
9 is identical to iteration 2.

### Step 3 — When source changes

After editing the test (e.g. tightening an assertion), step 1 must run again
to recompile. But because the project deps are already cached, this rebuild
is incremental and typically takes <30 seconds — far less than the original
1m 41s and far less than waiting in the cargo lock queue.

## When to use this pattern

- **Tight assertion-tuning loops** (the case Phase I.A.5 was designed around):
  one-line constant change, want to re-run the test 5 times to compare values.
- **Concurrent-session environments**: any time `pgrep -af cargo | wc -l`
  returns more than 4–5, the global lock is contended and `cargo test`
  invocations will queue.
- **Headless verification pre-commit**: when you just want to confirm a test
  still passes after an edit, without paying cargo's setup cost.

## When NOT to use this pattern

- **First-time test compilation**: of course you need `cargo test --no-run`
  the first time. The pattern only helps for re-runs.
- **Workspace-wide test suites**: `cargo test` discovers all bin/test targets
  across the workspace; running each test binary by hand is tedious. Use this
  pattern for targeted iteration on one or two tests, not full suite sweeps.
- **CI**: CI runs in a clean environment with no contention. Use plain
  `cargo test` there.

## Locating test binaries

The hash-suffix names make discovery non-trivial. A reliable lookup:

```bash
find target/debug/deps -name 'integration_rdp_wire-*' -executable -newer Cargo.lock | sort | tail -1
```

Or, more portably, use `cargo test --no-run` with `--message-format=json` and
parse the `executable` field from the JSON output:

```bash
cargo test --no-run --test integration_rdp_wire --features mesh-encryption \
    --message-format=json 2>/dev/null \
    | jq -r 'select(.executable != null) | .executable'
```

## Worked example: Phase I.A integration test rerun

This is the exact recipe that closes Phase I.A Track 1 step 2:

```bash
# 1. Build
cd /srv/luminous-dynamics/symthaea
cargo test --no-run --test integration_rdp_wire --features mesh-encryption

# 2. Locate
BIN=$(find target/debug/deps -name 'integration_rdp_wire-*' -executable | sort | tail -1)

# 3. Run with output
$BIN --nocapture

# Expected:
#   running 5 tests
#   test wrong_key_fails_to_open ... ok
#   test seal_open_latency_under_5ms ... ok
#   test full_frame_seal_open_reconstructs ... ok
#   test delta_frame_seal_open_reconstructs ... ok
#   [rdp_wire] envelope bandwidth: sealed=65828 bytes json=197332 bytes ratio=2.997×
#   test wire_envelope_beats_json_by_3x ... ok
#   test result: ok. 5 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
```

## Relationship to worktrees

This pattern is **complementary** to `./scripts/session-worktree.sh create <name>`,
not a replacement. Worktrees give you source-level isolation (different files,
different target dir, no concurrent edits). The `--no-run` pattern gives you
runtime-loop iteration speed within whatever isolation you have.

For a one-day Phase I.A.5 hardening sprint, the right combination is:
worktree at session start (Track 4.1) + `--no-run` for tight iteration loops
within that worktree.

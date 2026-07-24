# Symthaea Browser Hardening Verification — Patch Sets 13–18

Date: 2026-07-21

## Scope

Campaign III extends the Patch Set 12 hardened snapshot with bounded autonomy and
recoverable execution:

- session-scoped observations, element references, and receipts;
- configurable same-host or same-origin navigation confinement;
- total/mutating action budgets and a consecutive-failure circuit breaker;
- privacy-preserving SHA-256-chained action traces;
- exact, expiring, session- and action-bound approvals; and
- closure of direct high-consequence dispatch bypasses plus deterministic
  checkpoint/recovery directives.

## Authored commits

| Set | Commit | Subject |
|---|---|---|
| 13 | `f791377b6cd8bcb903b8f67b2979685f0794a828` | Bind observed targets to CDP session identity |
| 14 | `bbbb4d581a9d855c8702030456feb9439306148b` | Confine final navigation by host or origin scope |
| 15 | `95ad1458d23a959390d45422833479af660e5f48` | Add runtime budgets and failure circuit breaker |
| 16 | `deaa18221bbc9ffb35f78a2e667da1e4fc49dc8b` | Add privacy-preserving chained action traces |
| 17 | `06c906edd296dce12cc6f11cbfbe0d37dae7ef69` | Require exact expiring approvals for consequential actions |
| 18 | `1024f85db7d469a5705e0047ccc7ee22857aa585` | Close approval bypasses and add deterministic recovery |

Patch Set 12 parent: `99b7b462a3e2e411609f832a47090d4e706645f5`

Final authored Git tree: `6df193f66ff8aacdb30cb2e274a564b7a3524cfd`

## Verification performed

1. Every individual Patch Set 13–18 passed `git apply --check` against its
   declared parent.
2. The complete 13–18 series replayed with `git am` from the Patch Set 12
   parent and reproduced the final authored Git tree exactly.
3. The complete 01–18 series replayed with `git am` from the original uploaded
   `symthaea-browser.tar.gz` source and reproduced the same final Git tree
   exactly.
4. `git diff --check` passed for the complete authored range.
5. `git fsck --no-dangling` passed for the authored repository.
6. `Cargo.toml` parsed successfully with Python's TOML parser.
7. `bash -n scripts/run-hostile-browser-lab.sh` passed.
8. Conflict-marker and required-invariant scans passed for:
   - session-bound reference rejection;
   - high-consequence raw-dispatch denial;
   - navigation-scope denial;
   - runtime failure limits;
   - chained trace hashes; and
   - recovery directives.
9. The cumulative archive contains 18 ordered patch entries.

## Important enforcement behavior

`BrowserExecutor::execute()` now fails closed for `Click` and `Type`. Those
operations must be submitted through `execute_proposal()` with an approval that
matches the current CDP session, exact serialized action digest, consequence
scope, and expiry. This prevents callers from bypassing the approval path by
using the lower-level executor method directly.

Action traces store action and output digests rather than typed text, extracted
page contents, or screenshot bytes. Query strings and URL credentials are not
retained in trace records.

## Verification not performed

Cargo compilation, formatting, Clippy, unit tests, and the real-Chromium hostile
browser lane were not run. The execution environment has no Rust toolchain, and
the standalone crate still references the absent workspace path dependency
`../../core/symthaea-core`.

Required workspace gates remain:

```bash
cargo fmt --check -p symthaea-browser
cargo clippy -p symthaea-browser --all-targets --all-features -- -D warnings
cargo test -p symthaea-browser --all-targets
cargo test -p symthaea-browser --test hostile_browser_lab -- --ignored --nocapture
```

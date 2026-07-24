# symthaea-browser Hardening Verification — Patch Sets 19–24

## Scope

Campaign IV extends the Patch Set 18 hardened snapshot with six sequential,
independently reviewable commits:

1. Patch 19 — multi-tab governance
2. Patch 20 — upload/download transfer containment
3. Patch 21 — privacy-preserving deterministic replay plans
4. Patch 22 — crash-durable tamper-evident checkpoints
5. Patch 23 — privacy-safe operational metrics
6. Patch 24 — expiring mission leases for bounded autonomy

The campaign changes 11 files with 1,643 insertions and 10 deletions, adds six
new Rust modules, and increases Rust test markers from 60 to 75.

## Reproducibility results

- Patch Set 18 parent tree: `6df193f66ff8aacdb30cb2e274a564b7a3524cfd`
- Patch Set 24 authored tree: `960d5b5c6eee3649cb9beae7ddb53dd4c8edd504`
- Replayed Patch Sets 19–24 tree: `960d5b5c6eee3649cb9beae7ddb53dd4c8edd504`
- Replayed full Patch Sets 01–24 tree: `960d5b5c6eee3649cb9beae7ddb53dd4c8edd504`

Both replay paths reproduce the authored Patch Set 24 tree exactly.

### Replay A — Campaign IV only

The Patch Set 18 source archive was committed as a fresh baseline. Each Patch
Set 19–24 mail patch first passed `git apply --check`, then applied in order
with `git am`. The resulting Git tree equals the authored final tree.

### Replay B — Complete campaign

The original uploaded `symthaea-browser.tar.gz` was committed as a fresh
baseline. Every Patch Set 01–24 mail patch passed `git apply --check` at its
sequential parent and applied with `git am`. The resulting Git tree also
matches the authored final tree.

## Additional checks completed

- `git diff --check` passed for every Patch Set 19–24 commit.
- `git fsck --no-dangling` passed for the authoring repository.
- All generated tar.gz archives passed `tar -tzf` integrity checks.
- The hardened source archive contains no `.git` metadata.
- `bash -n scripts/run-hostile-browser-lab.sh` passed.
- Naive brace, parenthesis, and bracket balance passed across all Rust source
  and test files.
- Individual and cumulative patch archives include parent/result commit IDs and
  ordered series manifests.

## Security properties added

- Script popups and cross-origin auxiliary tabs fail closed by default.
- Tab registries retain canonical origins rather than credential-, path-, query-,
  or fragment-bearing URLs.
- File transfers are bounded and content-addressed; executable types are denied
  and unknown types are quarantined.
- Replay plans contain digests and external bindings rather than typed secrets.
- Checkpoint persistence verifies a SHA-256 chain before atomic replacement.
- Metrics expose stable labels and aggregates without page or user payloads.
- Mission leases constrain session, time, origin, capability, consequence, and
  action count independently of Phi and webpage instructions.

## Unexecuted workspace gates

Cargo compilation, rustfmt, Clippy, unit tests, and the real-Chromium hostile
laboratory were not run in this environment because no Rust toolchain is
installed and the standalone archive does not contain the required
`../../core/symthaea-core` path dependency.

Run the following from the complete Symthaea workspace:

```bash
cargo fmt --check -p symthaea-browser
cargo clippy -p symthaea-browser --all-targets --all-features -- -D warnings
cargo test -p symthaea-browser --all-targets
cargo test -p symthaea-browser --test hostile_browser_lab -- --ignored --nocapture
```

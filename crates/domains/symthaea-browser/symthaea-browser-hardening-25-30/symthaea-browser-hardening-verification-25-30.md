# symthaea-browser hardening verification — Patch Sets 25–30

## Authored range

- Parent commit: `d1045eabb01b40e35802781d1df42c3d10cbc08e`
- Result commit: `94acd99e56763eef2961229b3410634ed2c9ac3a`
- Parent tree: `960d5b5c6eee3649cb9beae7ddb53dd4c8edd504`
- Result tree: `cda143df01a8d0e59f800009f7546edc2c921168`
- Campaign diff: 15 files changed, 2087 insertions(+)
- Final Rust source: 32 files, 9992 lines
- Rust test markers: 94

## Verified in this environment

- Every 25–30 mailbox patch passed sequential `git apply --check`.
- Patch Sets 25–30 replayed with `git am` from the Patch Set 24 tree.
- The complete Patch Sets 01–30 series replayed with `git am` from the original uploaded source.
- Both replays produced the authored result tree exactly: `cda143df01a8d0e59f800009f7546edc2c921168`.
- `git diff --check` passed for the complete Campaign V range.
- The release helper passed `bash -n` syntax validation.
- All Rust files passed a lexical delimiter/comment/string integrity scan.
- The hardened source archive is generated directly from the verified Git tree.

## Not executable here

This container has no Cargo, Rust compiler, or rustfmt, and the standalone crate
still references `../../core/symthaea-core`. Therefore the following are
**release gates, not claimed passes**:

```bash
cargo fmt --check -p symthaea-browser
cargo clippy -p symthaea-browser --all-targets --all-features -- -D warnings
cargo test -p symthaea-browser --all-targets
SYMTHAEA_BROWSER_REAL_CHROMIUM=1 \
  ./crates/symthaea-browser/scripts/verify-browser-release.sh
```

Patch Set 30 intentionally makes this distinction explicit: replay and static
integrity prove package continuity, while compilation and live Chromium behavior
must be established in the complete workspace.

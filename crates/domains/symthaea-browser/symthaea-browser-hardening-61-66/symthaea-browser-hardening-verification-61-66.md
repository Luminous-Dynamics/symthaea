# symthaea-browser Campaign XI verification

## Scope

- Parent commit: `07c4e1c295a7e3c20f79636fcab915afa3a50de9` (Patch Set 60)
- Final commit: `d218bf8837d2a8b3c2606c525e8d47dad102da4d`
- Final Git tree: `ac7e8c4d30cde25d212b00dc86459fea2977124d`
- New patches: 61–66

## Replay integrity

Each Patch Set 61–66 mail patch passed:

`git apply --check --whitespace=error-all`

The six patches then replayed sequentially with `git am` from the exact Patch
Set 60 parent and reproduced the authored final Git tree.

The complete Patch Sets 01–66 series also replayed from the original uploaded
source and reproduced the same final tree:

`ac7e8c4d30cde25d212b00dc86459fea2977124d`

## Static verification performed

- Cargo.toml parsed with Python `tomllib`.
- All 58 declared Rust modules have matching source files.
- Tree-sitter parsed all 62 Rust source, test, and example files without syntax errors.
- Both shell scripts passed `bash -n`.
- The static verifier passed and confirms release schema v6 and every Campaign XI gate.
- Conservative credential scan found no private-key blocks, AWS access keys,
  bearer-token patterns, or GitHub-token patterns.
- The authored Git working tree was clean before packaging.

## Campaign size

- 6 sequential commits
- 13 changed files
- 3,154 insertions and 2 deletions
- 25,276 Rust lines in `src`, `tests`, and `examples`
- 210 `#[test]` markers
- 57 public module declarations

## Executable gates not run

Cargo, rustfmt, Clippy, Rust tests, live Chromium, real operator-key/quorum
integration, revocation propagation, root-rotation integration, and the live
incident-containment drill were not run. This environment has no Rust toolchain
and the standalone crate lacks `../../core/symthaea-core`.

`scripts/verify-browser-release.sh` exited with status `127` and reported
that Cargo is required, rather than representing unavailable gates as passed.

Static parsing and patch replay establish structure and delivery integrity; they
do not substitute for compilation or live integration evidence.

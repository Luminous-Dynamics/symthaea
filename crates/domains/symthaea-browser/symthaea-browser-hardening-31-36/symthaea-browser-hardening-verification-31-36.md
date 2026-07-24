# symthaea-browser hardening verification — Patch Sets 31–36

## Authored campaign

- Parent source: Patch Set 30 hardened snapshot
- Final commit: `b779e43d31301e617cdf0d78b7cd65b85ee2a1a0`
- Final Git tree: `629e1298c2cf6c3ab28386aece8d89ff32a80d9c`
- Rust lines across `src/` and `tests/`: 11082
- Rust `#[test]` markers: 114
- Campaign diff: 2104 insertions, 4 deletions

## Patch replay

- Every Patch Set 31–36 patch passed strict `git apply --check --whitespace=error-all` against its sequential parent.
- Patch Sets 31–36 replayed from the supplied Patch Set 30 snapshot.
- The full Patch Sets 01–36 series replayed from the original uploaded source.
- Both replay routes produced the exact authored Git tree: `629e1298c2cf6c3ab28386aece8d89ff32a80d9c`.

## Static checks completed

- `git diff --check` passed for the full Campaign VI diff.
- Bash syntax checks passed for both release and hostile-browser scripts.
- All individual, cumulative, full-series, hardened-source, and delivery archives passed gzip/tar listing checks.
- Basic Rust delimiter-count scanning found no unmatched aggregate delimiters.

## Executable verification status

The release helper exited with status **127** in this environment. Its output was:

```text
error: cargo is required for release verification
```

Cargo, rustfmt, Clippy, unit tests, and live Chromium tests were therefore **not run** here. This is not represented as a pass. The standalone crate also depends on `../../core/symthaea-core`, so executable verification belongs in the full Symthaea workspace.

## Notable correction

Patch Set 35 removes a duplicated `before_url` field in an `OriginTransition` struct literal that was present in the Patch Set 30 snapshot and would prevent Rust compilation.

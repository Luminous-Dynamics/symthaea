# Symthaea Browser Hardening Verification — Patch Sets 79–84

## Scope

Campaign XIV extends the Patch Set 78 source with:

1. short-lived freshness leases for composed authority proofs;
2. automatic emergency-authority decay;
3. federated, organization-diverse audit verification;
4. deterministic rollback rehearsal;
5. continuous runtime assurance checkpoints; and
6. deployment-profile, migration, and release-evidence schema v9 closure.

## Authored commits

- `18e72401433cc8d0976769f4dc87cb24cd4c7265` — browser: add composed proof freshness leases
- `40c75d175f9d273c4c0ef33c10e80f8e5f9063dc` — browser: add automatic emergency authority decay
- `d402726d8024b53263f0ae2ce1d1e96a85b78e87` — browser: add federated audit verification
- `79748d89106880b8cf7f8213e6fe98fabd76df5f` — browser: add deterministic rollback rehearsal
- `f8efea485e374cef57eb113028523c470781bbef` — browser: add continuous assurance checkpoints
- `ef1c84cdf2b1150f936e435f5ce249c35bfc6204` — browser: close campaign XIV profile and release evidence

## Replay verification

Every Patch Set 79–84 mail patch passed:

`git apply --check --whitespace=error-all`

The patches were replayed sequentially with `git am` through two paths:

- Patch Sets 79–84 from the hardened Patch Set 78 source;
- Patch Sets 01–84 from the original uploaded `symthaea-browser.tar.gz` source, using the previously verified 01–78 series followed by 79–84.

Both paths reproduced the authored final Git tree exactly:

- base Patch Set 78 tree: `60f9c8a9b02346d5c045cc209e254da446ea7daf`
- final Patch Set 84 commit: `ef1c84cdf2b1150f936e435f5ce249c35bfc6204`
- final Patch Set 84 tree: `293ffef54014406515ce4b0b14e13c300f6d8e5b`
- 79–84 replay tree: `293ffef54014406515ce4b0b14e13c300f6d8e5b`
- 01–84 replay tree: `293ffef54014406515ce4b0b14e13c300f6d8e5b`

The hardened source archive was independently extracted and reconstructed during packaging.

## Static verification performed

- `scripts/verify-static-structure.py`: passed across 73 Rust modules.
- Tree-sitter Rust parsing: passed for all 77 Rust source, test, and example files.
- `bash -n`: passed for both shell scripts.
- Python TOML parsing: `Cargo.toml` passed.
- `git diff --check`: passed for the complete campaign diff.
- Conservative credential/private-key token scan: passed.
- Archive layout and SHA-256 manifest checks: passed during packaging.

## Size

- Campaign diff: 14 files changed, 2960 insertions, 7 deletions.
- Rust lines in `src/`: 33,068.
- Rust test markers: 241.
- Public module declarations: 72.

## Executable gates not run

The release helper exited with status `127` because Cargo is unavailable in this environment. Consequently, the following are **not claimed as passed**:

- rustfmt;
- Clippy;
- Cargo compilation;
- unit and integration tests;
- live Chromium tests;
- live emergency-authority decay drills;
- real federated-auditor signature integration;
- isolated rollback and continuous-assurance drills;
- full-workspace verification against `../../core/symthaea-core`.

The release script correctly failed before producing misleading executable evidence.

## Release-helper output

```text
error: cargo is required for release verification
```

# Repository-contained research qualification harness

Status: infrastructure specification and executable harness. This document does not qualify any scientific claim.

## Why this exists

Research qualification must not depend on GitHub Actions YAML being the only executable copy of the theorem. A runner scheduler is infrastructure; it is not scientific authority.

`scripts/qualify-research-crate.sh` moves the reusable bootstrap qualification logic into the repository so the same exact checks can execute under:

- GitHub-hosted CI;
- a local checkout;
- a pinned Nix development environment;
- another independent CI provider.

The environment still matters and must be recorded. Portability does **not** mean results from different environments are silently equivalent.

## Boundary

The harness establishes only **bootstrap source qualification** for a qualifier commit whose immediate parent is the frozen semantic source subject.

It proves, when it passes:

1. the exact qualifier head is the requested head;
2. its immediate parent is the requested frozen source subject;
3. the qualifier commit changed only explicitly allowed qualifier paths;
4. the named source paths are byte-identical to the frozen parent;
5. the requested Rust/Cargo version is active;
6. Cargo generated a non-empty additive-only lock candidate;
7. every requested package passed `cargo check`, rustfmt, unit tests, and strict all-target Clippy;
8. the named source paths remained immutable after execution;
9. the original working-tree `Cargo.lock` was restored for portable/local use;
10. the exact generated lock candidate, patch, hashes, toolchain, source objects, and qualifier identity were retained in a receipt.

A green bootstrap run does **not** establish merge readiness. The generated lock candidate still requires inspection, exact replay into the semantic line, and a final `--locked` qualification.

A green bootstrap run also does not establish any mathematical, physical, causal, empirical, or other scientific claim.

## Example

```bash
scripts/qualify-research-crate.sh \
  --program SCI-001A \
  --source-parent aeeea202c89d0ab96166bec5674d95695df5f56c \
  --expected-head "$QUALIFIER_HEAD" \
  --expected-rust 1.96.0 \
  --package symthaea-science-research \
  --source-path crates/core/symthaea-science-research \
  --qualifier-path .github/workflows/sci-001a-science-research-kernel.yml \
  --output-dir /tmp/sci-001a-evidence
```

Multiple packages and source paths are expressed by repeating their flags.

## CI design rule

Future focused qualification workflows should become thin adapters:

1. checkout exact head;
2. install/enter the pinned environment;
3. invoke this repository-contained harness;
4. upload the emitted evidence directory.

The workflow file should not independently reimplement authority rules already owned by the harness.

## Fail-closed properties

The harness refuses:

- wrong head or parent;
- dirty starting checkouts;
- extra files in the qualifier diff;
- altered frozen source paths;
- wrong Rust/Cargo version;
- empty lock materialization;
- non-additive bootstrap lock patches;
- package build/test/lint failures;
- unexpected postflight mutations.

It deliberately restores `Cargo.lock` after retaining its generated candidate so a local qualification does not silently mutate the research subject.

## Follow-on work

After this harness itself is reviewed and qualified, migrate focused research workflows to call it rather than carrying duplicated shell logic. A later final-qualification harness should accept an already replayed exact `Cargo.lock` and require `--locked` throughout rather than materializing a candidate.

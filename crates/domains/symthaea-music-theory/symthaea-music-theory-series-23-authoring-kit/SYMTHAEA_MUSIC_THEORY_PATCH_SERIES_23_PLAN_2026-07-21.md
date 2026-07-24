# Symthaea Music Theory Patch Series 23 Plan

**Date:** 2026-07-21  
**Base:** exact verified Patch Series 22 final tree  
**Theme:** Cumulative replay, build truth, independent-verifier execution, and reproducible release evidence

## Executive summary

Series 23 is the execution-truth campaign deliberately deferred after Series 22. It does not create another trust abstraction. It proves—or falsifies—the accumulated implementation claims in one reproducible lane.

The campaign begins from a byte-exact source baseline, replays the complete numbered patch lineage, verifies the resulting Git tree, exercises all declared Cargo and Nix lanes, runs the frozen conformance corpus through Rust and independent implementations, rebuilds every public artifact, and generates a claim matrix from the observed evidence rather than from prose.

## Release invariants

1. **Exact ancestry:** every numbered patch applies to the advertised predecessor without manual edits.
2. **Exact final tree:** authored and independently replayed trees are byte-identical.
3. **All targets are real:** every declared binary, example, test, benchmark, feature, and optional dependency is either built or explicitly excluded with a checked reason.
4. **No hidden workspace state:** generated files, ignored files, local environment variables, and untracked sources cannot affect release output.
5. **Independent agreement:** Rust and at least one non-linked verifier agree on the frozen Series 22 corpus.
6. **Reproducible public artifacts:** source archives, patch archives, vector kits, and manifests rebuild byte-for-byte.
7. **Negative controls work:** the lane must fail when a known patch, vector, manifest entry, or expected result is deliberately corrupted.
8. **Claims are derived:** documentation status is generated from evidence records and cannot independently promote a capability.
9. **No warning debt:** format, check, test, Clippy, documentation, and Nix gates are clean on the exact release tree.
10. **Failures remain evidence:** failed lanes produce bounded diagnostic bundles and are never rewritten as skipped success.

## Deliverables

- Machine-readable cumulative patch and artifact ledger.
- Clean-room replay tool and final-tree identity check.
- Cargo target/feature inventory and build matrix.
- Nix clean-build and reproducibility lane.
- Independent conformance execution report.
- Deterministic archive reproduction report.
- Negative-control evidence.
- Generated claim matrix and implementation-status document.
- Complete release evidence bundle with externally stored outer digest.

## Explicit non-goals

Series 23 does not change authorization semantics, witness policy, incident recovery, publication resumption, canonical encoding, or verifier trust roots except where a discovered implementation defect must be corrected to satisfy an already documented Series 16–22 contract.

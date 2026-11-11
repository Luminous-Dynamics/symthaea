# MACE-CoS Guard v0.1 — evidence status

Date: 2026-11-11.

## Public branch contents

Observed branch head after sink-binding hardening:
`247945c1c9631b913fcf0f23ea292a9779d9c2d7`.

The public branch contains:
- typed Go request/skill/artifact/step/sink schema;
- deterministic composition validator and strict JSON CLI;
- protocol, threat model and benchmark matrix;
- 13 named Go test functions: 7 focused validator tests, 1 generated two-step lattice-oracle test, and 4 sink/capability-binding regression tests.

This is a research-only Go reference. No Rust integration is included.

## Previously reported local bundle — identity unresolved

The initial handoff contains conflicting records for the local verification archive:

- The earlier PR/issue description reports 32 named test functions and ZIP SHA-256 `7e437113fd8735b1251951dbd049b5318e10fe19d006d3f64713c9f0d8901327`.
- The earlier version of this RESULTS.md reported 31 named test functions and ZIP SHA-256 `899a4c9b036c6652e4d045355416f24a06cdecae1eed06ea44e2e25ed21024a8`.

The archive bytes were not available in the active workspace during this review, so neither hash was independently rechecked and the two records are not assumed to identify the same archive. Do not publish either as the canonical current bundle until the archive is restored, rehashed, and its test inventory reconciled.

The earlier handoff reports a Go 1.23.2 local run with race detector, `go vet`, formatting, 79.4% statement coverage, five CLI fixtures, 16 label-monotonicity pairs, 256 two-step lattice-oracle combinations, and 256 irrelevant-capability mutations. Those are historical reported results, not a verification of the current branch head.

## Exact-head verification status

The four new sink/capability regression tests were committed after the previously reported local bundle. They have not been executed against the exact public branch head in this review.

For commit `247945c1c9631b913fcf0f23ea292a9779d9c2d7`:
- GitHub combined status contains no status checks.
- Associated CI, PR Governance, Showroom Integrity and Benchmark Suite workflow runs completed with conclusion `skipped`.
- No GitHub Actions pass or exact-head local Go test pass is claimed.

## Not yet established

- No Rust compile or production integration.
- No CoSA-Bench, SkillsBench, SkillFusion, SkillAlign or SkillShift-Bench evaluation.
- No persistent cross-request nonce ledger or pre-dispatch TOCTOU enforcement in this module.
- No natural-language semantic/implicit-flow assurance, source authentication, signature verification or independent proof that `verifier_status=verified` is truthful.
- No SOTA, security certification, production enforcement or real-world impact claim.

## Next qualification gate

1. Restore and reconcile the local archive identity and test count.
2. Run `go test -count=1 ./...`, `go test -race -count=1 ./...`, `go vet ./...` and formatting on the exact branch head.
3. Add independent multi-input/branching composition and sink-binding oracle cases.
4. Begin matched CoSA-Bench reproduction only after the official benchmark harness and published guard are available and frozen.

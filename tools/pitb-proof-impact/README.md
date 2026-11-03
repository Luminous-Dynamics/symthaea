# PITB proof impact and maintenance analyzer

Standalone Rust snapshot-diff tool for the proposed Change-Impact and Proof-Maintenance Benchmark (CIPM).

It compares two **typed dependency snapshots** and computes a conservative impact plan. The source convention is the same as the PITB typed graph contract: an edge `source_node_id -> target_node_id` means the source is the dependent and the target is the dependency.

## Run

From the repository root:

```sh
cargo fmt --check --manifest-path tools/pitb-proof-impact/Cargo.toml
cargo test --locked --manifest-path tools/pitb-proof-impact/Cargo.toml
cargo run --locked --manifest-path tools/pitb-proof-impact/Cargo.toml -- --self-test
cargo run --locked --manifest-path tools/pitb-proof-impact/Cargo.toml -- \
  tools/pitb-proof-impact/fixtures/old-snapshot.synthetic.json \
  tools/pitb-proof-impact/fixtures/new-snapshot.synthetic.json \
  /tmp/pitb-proof-impact-report.json
```

## Impact policy

| Change | Direct consumer action | Reason |
|---|---|---|
| Statement/signature/type, typeclass inheritance, structure field type, instance relation | `reprove` | Elaboration or proof compatibility may have changed |
| Transparent definition body | `reprove` | Reduction can affect downstream proof terms |
| Opaque theorem/proof body or axiom manifest | `revalidate` | The declaration type may be unchanged, but proof trust evidence changed |
| Proof-term reference | `revalidate` | Observed support must be replayed; it is not a logical-necessity claim |
| Module/import/environment dependency | `rebuild` | Build/elaboration context may have changed |
| Documentation, Blueprint, informal candidate, cross-formality link | `retrieval_refresh` | Search evidence changed; formal proof claims do not automatically invalidate |
| Source bytes change without statement/body/proof digest change | `source_revalidate` (or retrieval refresh for docs) | Reconfirm source coordinates and correspondence |
| Global Lean environment digest changes | `rebuild` every formal declaration | Environment identity is a global trust dependency |
| Incomplete old or new dependency inventory | `unknown_conservative` every formal declaration | Missing edges cannot justify optimistic reuse |

The analyzer uses the strongest action required by any change. It propagates changed proof evidence through observed proof-term references, and transparent body changes through definition-body references. It does not treat proof-term occurrence as proof that a premise is logically necessary.

## Safety boundary

The tool computes a **planning classification**, not a proof certificate. `retain` means only that this snapshot diff did not identify a reason to revalidate the artifact under the registered graph and policy. It does not independently authenticate either snapshot, validate receipts, rerun Lean, or prove the dependency inventory complete. Use snapshots whose identities and provenance have been verified by the separate PITB graph-evidence and signed snapshot-attestation workstreams.

Never release an affected proof merely because this tool has listed an action. Affected artifacts must undergo the required reproof/revalidation/rebuild and independent clean-checkout replay. Missing, stale or incomplete dependency information must fail closed.

## Synthetic controls

The unit/self-test suite exercises:
- no-op snapshot preservation;
- signature change -> direct consumer reproof -> downstream proof-user revalidation;
- proof-body-only change preserving type-only dependents;
- transparent definition-body invalidation;
- global environment rebuild;
- documentation-only retrieval refresh;
- incomplete inventory -> conservative unknown classification;
- action precedence and retrieval-only relation policies.

The checked-in snapshots are synthetic fixtures. They are not a live Lean repository capture, measured maintenance result, or evidence of benchmark performance.

## Relationship to CIPM and the wider PITB plan

CIPM asks whether a system can identify exactly which claims and receipts became stale, preserve unaffected proof work, repair only the necessary closure, and still independently replay the full required repository state. This utility implements an initial typed-graph invalidation calculation; it does not yet implement a live Lean extractor, proof repair, receipt authentication or whole-repository replay.

Next qualification stages:
1. capture a real parent snapshot from the pinned Lean environment;
2. apply controlled changes to a shared helper, instance, module option and transparent definition;
3. independently derive the gold impact set;
4. compare this analyzer against full-rebuild, changed-file heuristic and graph-only baselines;
5. measure stale-proof acceptance (must be zero), impact precision/recall, unaffected-proof preservation, repair success, revalidated count, build cost, p50/p95 change-to-verified-green time and full-repository replay.

Original design: `docs/research/pitb-change-impact-proof-maintenance-v0.1.md`.

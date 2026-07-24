# Symthaea Aesthetic Operational Evolution Patch Series

This series extends the API-maturity baseline with explicit compatibility,
migration, regression, rollout, fault-injection, and rollback evidence. It does
not add another aesthetic scoring heuristic. Its purpose is to make future
changes observable, reviewable, reversible, and safe to deploy.

## Bundle A — Contract evolution

1. `feat(evolution): add semantic contract compatibility diffs`
2. `feat(migration): add explicit contract migration plans`
3. `feat(upgrade): add fail-closed upgrade assessment`

The bundle compares two complete contract snapshots and classifies API,
schema, registry, extractor, modality, channel, determinism, and build changes as
compatible, reviewable, or breaking. Migration plans cover identified changes
with explicit actions and verification procedures. Upgrade policies can reject
breaking changes, missing coverage, irreversible migrations, or excessive
change counts.

## Bundle B — Golden regression and rollout

4. `feat(golden): add self-verifying evaluation corpora`
5. `feat(regression): compare candidate golden corpora`
6. `feat(rollout): add deterministic canary plans`

Golden corpora contain complete self-verifying evaluation archives rather than
only expected scalar values. Regression reports compare lineage, artifact
identity, utility, confidence, and intrinsic evidence under declared budgets.
Canary assignment is stable, deterministic, and nested across rollout phases.

## Bundle C — Fault resilience

7. `feat(fault): add deterministic archive fault injection`
8. `feat(resilience): add archive fault campaigns`

Data-only fault plans tamper with cloned archives without executing arbitrary
code. Campaigns report whether receipt, lineage, schema, and consistency checks
actually reject the injected faults. Fault plans that make no observable change
are rejected instead of producing misleading resilience evidence.

## Bundle D — Promotion and rollback closure

9. `feat(rollback): add auditable rollback capsules`
10. `feat(promotion): add release evolution bundles and gates`
11. `feat(api): publish operational evolution capabilities and schemas`
12. `fix(evolution): bind rollback baselines and reject no-op faults`
13. `docs(evolution): document operational evolution series`

A release bundle binds its contract diff, migration, upgrade assessment, golden
regression, fault campaign, rollout plan, and rollback capsule. Promotion fails
closed when these records disagree. Rollback must bind the exact prior contract
and baseline golden corpus used by the release evidence.

## Application order

Apply all patches in numerical order with `git am`:

```bash
git am /path/to/0001-*.patch \
       /path/to/0002-*.patch \
       /path/to/0003-*.patch
```

The bundle archives already contain ordered patch filenames. They are based on
the tree from `symthaea-aesthetic-api-maturity-patched.tar.gz`.

## Integration sequence

1. Capture current and candidate `ContractSnapshot` values.
2. Build a `ContractDiff` and `MigrationPlan`.
3. Run `assess_upgrade` under an explicit `UpgradePolicy`.
4. Evaluate the same request set under both releases and capture `GoldenCorpus`
   values.
5. Run `compare_golden_corpora` with release-specific numeric and lineage
   budgets.
6. Run an archive `FaultPlan` campaign.
7. Create a deterministic `RolloutPlan` and a baseline-bound
   `RollbackCapsule`.
8. Construct `EvolutionReleaseBundle` and call `evaluate_promotion`.
9. During rollout, call `assess_phase`; activate the bound rollback capsule if a
   critical trigger occurs.

## Compatibility notes

- The public aesthetic API advances from `1.0.0` to `1.1.0`.
- Existing API capabilities remain present; five operational-evolution
  capabilities are added.
- Six new persisted schema families are added at version 1.
- Existing persisted schema versions are unchanged.
- Integrity identifiers remain deterministic FNV-1a compatibility identifiers,
  not cryptographic signatures.

## Required parent-workspace verification

The patch environment did not provide Cargo or rustc and DNS prevented a
minimal toolchain bootstrap. Before merge, the parent workspace must run:

```bash
cargo fmt --all -- --check
cargo check -p symthaea-aesthetic --all-features
cargo test -p symthaea-aesthetic --all-features
cargo clippy -p symthaea-aesthetic --all-targets --all-features -- -D warnings
```

It should also run actual two-release corpus capture, fault campaigns, and a
staged rollback rehearsal using production extractor descriptors.

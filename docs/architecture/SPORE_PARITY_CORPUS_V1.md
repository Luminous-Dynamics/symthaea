# Spore Behavioral Parity Corpus v1

Status: **pre-extraction behavioral contract**

This corpus freezes the behaviors that must survive the standalone Spore extraction. It is deliberately a **semantic catalog**, not a claim that the destination repository has qualified.

The machine-readable source of truth is:

- `docs/architecture/spore-parity-corpus-v1.json`
- validated by `scripts/check_spore_parity_corpus.py`
- regression-tested by `tests/test_spore_parity_corpus.py`

## Why this exists

Repository extraction can preserve bytes while accidentally changing meaning: a systemd dependency can gain authority, a live activation can be confused with a physical boot, a helper timeout can become boot-blocking, or a failed candidate can overwrite recovery truth.

The parity corpus therefore binds each required behavior to an exact historical test fixture already recorded in the migration manifest. The manifest supplies repository, commit, path, and source-blob provenance; the corpus supplies the behavioral selector and the constitutional invariants that behavior protects.

This creates a two-part migration contract:

```text
exact historical source provenance
              +
frozen behavioral expectation
              |
              v
deterministic destination migration
              |
              v
fresh destination execution
              |
              v
new qualification lineage
```

Historical PASS evidence may explain why a behavior is in the corpus. It never becomes destination qualification.

## Frozen domains

v1 requires coverage of availability, physical boot identity, qualification, LKG semantics, helper expendability, lifecycle behavior, firmware recovery, and effective systemd authority.

The current source fixtures are the exact manifest entries:

- `fail-open-vm`
- `helper-expendability-vm`
- `ovmf-recovery-vm`
- `systemd-authority-tests`

A future source fixture may be added only with explicit migration provenance and destination qualification requirements.

## Execution rule after extraction

Each migrated fixture must execute against **exact committed destination bytes**. A test copied into the new repository is not evidence until its destination source and product closure are qualified there.

For every corpus behavior, destination evidence should eventually record at minimum:

```text
behavior_id
destination_commit
destination_tree
product_derivation_or_closure
test_fixture_digest
runner_environment
result
artifact_digest
```

If implementation changes intentionally alter a historical behavior, that change must land as a separately reviewed semantic change after parity has been established. The extraction itself must not silently redefine the contract.

## Scope

This corpus intentionally does not add Recovery Capsule v1, new boot-attempt authority, TPM/Secure Boot redesign, fleet orchestration, remote recovery, installer features, or new presentation behavior. Those remain post-extraction work.

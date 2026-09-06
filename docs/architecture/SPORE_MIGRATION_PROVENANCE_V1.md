# Spore Migration Provenance Ledger v1

Status: **Seed ledger — pre-extraction**

This ledger records exact source identities that are candidates for migration into the future independent `Luminous-Dynamics/spore` repository.

A ledger entry is provenance only. It is not destination qualification.

## Rules

For each migrated artifact, preserve:

- source repository,
- exact source commit,
- exact source tree,
- exact source blob,
- source path,
- source branch/PR context,
- destination path,
- transformation status,
- destination blob/commit once created,
- and fresh destination qualification status.

Git blob SHA values below identify exact source file bytes inside the recorded tree. A future migration step SHOULD additionally record an implementation-independent content digest (for example SHA-256) in the destination ledger/attestation so the evidence model is not coupled solely to Git object hashing.

## Source lineage A — NixOS host runtime-expendability proof

Repository:

`Tristan-Stoltz-ERC/nixos-config`

Source branch:

`spore/runtime-expendability-v1.3.2-proof`

Exact branch head inspected:

`5d80360768ee329c50756e71fbce4692ac3a8e45`

Commit message:

`fix(spore): make authority collector systemctl-portable`

Exact Git tree:

`51c04910b3a97586ecf88a46699b7de22e3e1b0b`

Git commit verification observed during ledger creation:

`verified=false`, reason `unsigned`.

This is recorded as a provenance fact only. A signed Git commit alone is not intended to become Spore's product/root-of-trust model.

### Candidate artifacts

| Source path | Exact Git blob SHA | Current role | Intended destination ownership | Migration status |
|---|---|---|---|---|
| `modules/system/symthaea-boot.nix` | `6a2c415e776139d202f0d44d719fa1d242186ad1` | Host module containing generic Spore boot/recovery integration plus host policy | Generic portion -> Spore NixOS module; host-specific policy remains in `nixos-config` | NOT MIGRATED |
| `scripts/check_spore_systemd_authority.py` | `7b587682011939479d6db6d903d62f3187d6fedd` | Effective systemd authority-graph checker | Spore qualification tooling | NOT MIGRATED |
| `scripts/spore_runtime_expendability_v13.py` | `fc45908f5f32a258c4e79cf87862b630e9ef385a` | Guarded deterministic transformation that materializes runtime hardening in qualification workspace | Historical migration/evidence tooling; stabilized product result should become committed source | NOT MIGRATED |
| `tests/spore-boot-fail-open.nix` | `4fcc1618b2e993ca38f0c7a988a7afae65dd9ede` | Fail-open VM proof | Spore NixOS qualification tests | NOT MIGRATED |
| `tests/spore-boot-helper-expendability.nix` | `0fc089e9677d0398d79028baf65ddfe0682a4493` | Hung-helper/sandbox/lifecycle expendability VM | Spore NixOS qualification tests | NOT MIGRATED |
| `tests/spore-boot-ovmf-recovery.nix` | `b2f645694c53c579d43b2b8f8781b0e6b8a4fbd8` | OVMF recovery qualification fixture | Spore NixOS qualification tests | NOT MIGRATED |
| `tests/test_spore_systemd_authority.py` | `4db5d9af7f942a978c2c786ffcbb8f6b1ad35e26` | Authority-checker regression suite | Spore qualification tests | NOT MIGRATED |
| `.github/workflows/spore-runtime-v132-proof.yml` | `4b4f9f1d9db9a67bf3cc7d4e9d36694726eb93be` | v1.3.2 qualification experiment | Historical evidence only; destination workflow must be rebuilt around exact committed bytes | NOT MIGRATED |
| `.github/workflows/spore-runtime-v13-autopatch.yml` | `43d902c0fb0a763fad14d37d0727ebd76f8eac30` | guarded runtime hardening/autopatch workflow | Historical evidence/migration context only | NOT MIGRATED |

## Critical evidence boundary for lineage A

The `v1.3.2` source tree above is **not equivalent to the source bytes actually exercised after workspace transformation** by the qualification workflow.

The observed structure is:

```text
exact checked-out tree
        |
        v
workspace edits / guarded transformation
        |
        v
formatted materialized candidate
        |
        v
qualification
```

Therefore this ledger MUST NOT encode the historical result as:

```text
QUALIFIED(5d80360768ee329c50756e71fbce4692ac3a8e45)
```

unless an evidence record separately proves that statement for those exact committed product bytes.

The safe interpretation is:

```text
5d803607... = exact source lineage of the experiment

transformed candidate = subject actually exercised by the runtime-expendability lane

destination exact-source qualification = NOT YET ESTABLISHED
```

The destination repository should eliminate this ambiguity by committing the stabilized implementation and qualifying it without product-source mutation after checkout.

## Source lineage B — Symthaea presentation/recovery-adjacent work

Repository:

`Luminous-Dynamics/symthaea`

One inspected historical branch is:

`feat/spore-clean-host-rehydration-v1`

Exact branch head observed during ledger creation:

`937b2247e95c766e44cb2c0cbc2038019e671f92`

Exact tree:

`16c3a737decc394da9219f0fb749c8c55a8f6356`

Commit message:

`ci(spore): qualify exact repaired source head`

This entry intentionally does **not** yet enumerate destination migration blobs.

Reason: the extraction boundary distinguishes two classes that must not be conflated:

1. generic recovery/qualification implementation that belongs in independent Spore, and
2. Boot Ecology/presentation implementation that remains in Symthaea.

Before moving any Symthaea file, the migration PR must identify its exact role and exact blob and prove that it belongs on the recovery side rather than the presentation side.

### Explicit non-migration

`crates/domains/symthaea-spore` is not implicitly part of this recovery extraction. It is an existing Symthaea domain/consciousness component and remains under Symthaea ownership unless a later, separately reviewed design explicitly changes that meaning.

## Destination fields

Once `Luminous-Dynamics/spore` exists, each migrated row must be extended with:

```text
destination_repository
destination_path
destination_commit
destination_blob
destination_sha256
migration_transform
parity_evidence
qualification_evidence
```

No row may change from `NOT MIGRATED` to `QUALIFIED` merely because bytes were copied.

Recommended progression:

```text
NOT MIGRATED
    -> MIGRATED_UNQUALIFIED
    -> PARITY_PROVEN
    -> EXACT_SOURCE_QUALIFIED
```

If destination behavior intentionally changes, use:

```text
MIGRATED_CHANGED_UNQUALIFIED
```

rather than claiming parity.

## Next ledger action

After the independent repository is created, the first destination PR should import this ledger and add only repository scaffolding plus a machine-checkable manifest/schema for these provenance records. Product implementation should move in later PRs after the golden parity corpus is established.

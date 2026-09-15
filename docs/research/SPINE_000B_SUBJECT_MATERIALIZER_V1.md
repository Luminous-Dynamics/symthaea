# SPINE-000B-M1A Subject Materializer v1

Status: **IMPLEMENTED / RUNTIME CAMPAIGN USE PENDING QUALIFICATION**

Authority: `measurement-only`

M1A turns a clean checkout plus frozen campaign inputs into the canonical M1 subject consumed by C2 observation-chain genesis.

## Command

```text
python scripts/spine_000b_subject_materializer_v1.py materialize \
  --config <campaign-input.json> \
  --out-dir <directory-outside-repository>
```

It produces:

- `subject-manifest.json` — human-readable semantic transport;
- `subject-manifest.bin` — canonical M1 bytes;
- `subject-manifest.sha256` — SHA-256 of canonical bytes;
- `materialization-receipt.json` — non-authoritative preflight metadata.

The output directory must resolve outside the repository.

## Derived checkout identity

M1A derives rather than accepts:

- exact Git HEAD;
- exact committed `HEAD^{tree}`;
- clean worktree status;
- observed `rustc --version` / `cargo --version` / rustc host target;
- SHA-256 of required bound sources.

It never uses `git write-tree` as a working-tree identity.

## Campaign-supplied identity

The campaign config supplies only semantic inputs that cannot be inferred safely:

- runtime profile;
- explicit Cargo features + default-feature policy;
- protocol IDs;
- workload ID/digest;
- named seeds;
- cycle start/count and stopping rule;
- fixed observer capacities;
- qualification-policy digest;
- additional repo-relative files whose semantics the campaign relies on.

Unknown Cargo features, duplicate semantic entries, missing/noncanonical bound paths, or invalid M1 fields fail closed.

## Two-phase source check

M1A hashes every bound file, encodes the subject, then immediately rechecks:

- HEAD/tree;
- worktree cleanliness;
- observed toolchain/target;
- every bound file hash.

Any change during materialization rejects the subject.

## Drift verification

```text
python scripts/spine_000b_subject_materializer_v1.py verify \
  --manifest subject-manifest.json \
  --bin subject-manifest.bin \
  --digest subject-manifest.sha256
```

Verification never updates the frozen subject. Drift requires a new manifest/genesis lineage.

## Genesis law

Only a successful M1A digest may populate C2's `subject_manifest_digest` for a qualified runtime campaign. M1A does not itself create runtime observation evidence.

## Claim boundary

M1A establishes subject preparation and drift detection only. It does not establish observer non-interference, execution/application completeness, causal load, benefit, or authority.
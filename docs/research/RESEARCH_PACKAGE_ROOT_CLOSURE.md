# Research Package-to-Source-Root Closure

`SCI-INFRA-001E2` adds a narrow static theorem above canonical research program
profiles. It does not alter any existing profile or qualification receipt.

## Why this layer exists

Program profile v1 truthfully records two independent lists:

```text
packages
source_paths
```

C2 proves that a candidate qualifier exactly matches those trusted lists. It does
not prove which declared source root owns which declared Cargo package.

E2 closes only that relationship.

## The theorem

For the exact candidate `source_parent` and qualifier head, E2:

1. resolves the already-trusted C2 program profile;
2. independently requires every declared profile source path to be unchanged
   across the qualifier-only commit;
3. inspects only declared directory source paths;
4. reads `<source-path>/Cargo.toml` as exact bytes directly from the frozen
   source-parent Git tree without executing Cargo or candidate code;
5. parses `[package].name` with Python's TOML parser;
6. requires every declared package to map to exactly one declared source root;
7. rejects a declared source root whose Cargo package is absent from the profile;
8. rejects duplicate roots for one declared package;
9. binds the exact package-root tree object and exact Cargo.toml blob/SHA-256.

The resulting closure is deterministic and content-addressed as:

```text
schema=symthaea.research-package-root-closure.v1
authority=package-root-binding-only
scientific_claim=NONE
```

## Why source_parent is authoritative here

Canonical profiles describe the source subject being qualified. The research
source does not need to exist on the infrastructure-policy branch.

Therefore package/root identity is derived from the exact candidate
`source_parent` Git tree, after C2 has established the trusted profile and B2's
single-manifest qualifier shape. The qualifier head must expose the same source
objects.

## Exact-byte boundary

Cargo manifest SHA-256 is computed over the literal bytes returned by Git. Text
normalization, newline trimming, and reserialization are forbidden. TOML parsing
is used only to derive the semantic `[package].name` after the exact bytes have
been bound.

## What one package-root record binds

Each declared package produces one record containing:

```text
package
source_root
source_tree_git_object
cargo_manifest_path
cargo_manifest_git_blob
cargo_manifest_sha256
```

Records preserve the canonical profile package order.

## Non-package source paths

A profile may also bind individual source files or non-package directories.
Those remain source evidence but do not need to contain `Cargo.toml`.

This matters for SCI-003A, whose profile binds five borrowed aesthetic
scientific-validation files in addition to its two package roots.

## Fail-closed behavior

E2 rejects:

- missing profile source paths in the frozen source parent;
- source-object drift in the qualifier commit;
- non-regular Cargo manifests;
- malformed or non-UTF-8 TOML;
- non-string package names;
- a declared package with no declared package root;
- multiple declared roots claiming the same package;
- a declared package root naming a package absent from the profile.

## Deliberate boundaries

E2 does **not** establish:

- Cargo workspace membership;
- dependency graph closure;
- feature resolution;
- build-script or proc-macro behavior;
- generated source identity;
- external tool/system-library identity;
- Nix closure identity;
- runtime resource/input closure;
- that the selected package/source scope is scientifically sufficient;
- scientific correctness, replication, independence, or causality.

Those are separate theorems.

## Integration direction

E2 should remain a third independent evidence layer rather than modifying B2 or
C2 historical receipts:

```text
B2 manifest-binding
  != C2 canonical-program-profile binding
  != E2 package-root closure
```

A later workflow activation should derive E2 before candidate execution, rederive
it after execution, independently derive it on the fresh sealing runner, and
publish a separate content-addressed closure/binding only after B2 and C2 have
sealed successfully.

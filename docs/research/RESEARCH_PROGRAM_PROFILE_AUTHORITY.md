# Canonical Research Program Profiles

`SCI-INFRA-001C2` adds a trusted program-scope theorem above the B2 research
qualification adapter. It does not replace or reinterpret B2 evidence.

## Why this layer exists

B2 proves that one exact qualifier manifest declared a scope and that the exact
scope was executed and sealed. Its final receipt therefore says:

```text
scope_authority=manifest-declared-only
```

That is intentionally weaker than proving that a human-readable program name
such as `SCI-001A` has a canonical toolchain/package/source scope.

C2 closes only that gap.

## Trusted profile registry

Canonical profiles live in the trusted default-branch policy checkout at:

```text
.github/research-program-profiles/<PROGRAM>.json
```

Profile schema:

```json
{
  "schema": "symthaea.research-program-profile.v1",
  "program": "SCI-001A",
  "expected_rust": "1.96.0",
  "packages": ["symthaea-science-research"],
  "source_paths": ["crates/core/symthaea-science-research"]
}
```

The profile is policy, not candidate data. A qualifier PR cannot add or modify
its own trusted profile because the qualification workflow reads profiles only
from the exact trusted policy checkout.

## Exact scope equality

A manifest is profile-admissible only when all four scope fields equal the
trusted profile exactly and in order:

```text
program
expected_rust
packages
source_paths
```

The candidate-specific `source_parent` remains in the manifest and is still
proved by A2/B2. It is deliberately not part of the canonical program profile.

C2 does not infer aliases, normalize package sets, widen source roots, or treat
set-equivalent ordering as equal. Policy changes require a new trusted profile
commit and therefore a new trusted policy identity.

## Trusted profile object theorem

Before a profile can authorize scope, C2 requires:

- the trusted checkout HEAD equals the exact policy SHA;
- the B2 workflow, adapter, harness, subject guard, and B2 sealer are exact tracked `100644/blob` policy files;
- the B2 adapter is verified before it is imported;
- the C2 validator/sealer are exact tracked `100644/blob` policy files;
- the profile path is canonical and derived from `program`;
- the profile is a tracked `100644/blob` Git object;
- the checked-out file is regular and not a symlink;
- worktree bytes hash to the exact Git blob at trusted HEAD;
- JSON has no duplicate keys;
- only the closed v1 field set is present;
- program/toolchain/package/path identities are canonical and bounded;
- package and source-path lists are duplicate-free;
- manifest scope equals profile scope exactly.

The validator exposes both the exact Git blob identity and SHA-256 identities
for the profile bytes and canonical semantic profile.

## Two independent final receipts

B2 remains unchanged and writes:

```text
manifest-binding.json
schema=symthaea.research-qualifier-binding.v2
scope_authority=manifest-declared-only
```

C2 writes a second file after B2 sealing on the fresh authenticated runner:

```text
program-profile-binding.json
schema=symthaea.research-program-profile-binding.v1
authority=program-scope-binding-only
scope_authority=trusted-program-profile-v1
scientific_claim=NONE
```

The C2 binding commits to the SHA-256 of the exact B2 binding, exact qualifier
manifest identity, exact source parent/head, trusted policy SHA, trusted program
profile path/blob/file/canonical digests, trusted B2 workflow/adapter/harness/guard/sealer
identities, trusted C2 validator/sealer identities, admission mode, live-PR postflight state,
and GitHub run identity. The B2 binding must carry the same policy-object digests and exact
run/admission identity before C2 can mint stronger scope authority.

Strong canonical-scope interpretation requires both receipts. The B2 receipt is
never silently upgraded or rewritten.

## Candidate-execution boundary

The trusted profile is checked before candidate execution, rechecked after
candidate execution without repository credentials, and rechecked again on the
fresh authenticated sealing runner.

Candidate code therefore cannot gain canonical-scope authority by mutating a
same-UID trusted checkout during Cargo/build-script execution. The validator
bootstraps Git/object verification without importing the B2 adapter first, so a
mutated adapter cannot redefine the postflight checks. A changed adapter or
profile fails closed, and the fresh sealing runner derives the trusted objects
independently from policy again.

This is still not a hostile-code sandbox theorem.

## Profile changes

A profile change is a policy change. It must be reviewed as such and must not be
smuggled into a candidate qualifier manifest.

Changing any canonical scope field changes the canonical profile SHA-256. Old
qualification evidence remains evidence about the old profile/policy identity;
it does not migrate by similarity or Git ancestry.

## Nonclaims

C2 does not establish:

- scientific correctness, novelty, causality, replication, or independence;
- final `Cargo.lock` acceptance or reproducible binaries;
- hostile-code containment;
- scheduler or GitHub platform trust beyond the already-recorded execution;
- that a canonical program scope is scientifically sufficient;
- that package/source scope captures every transitive build or runtime input;
- merge enforcement or branch protection.

It proves only that the exact scope B2 executed equals the exact canonical scope
published for that program by the trusted policy revision.

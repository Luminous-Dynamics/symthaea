# Authenticated Research Package-Root Seal

`SCI-INFRA-001F2` adds the fresh-runner authenticated publication layer for the
E2 package-to-source-root closure theorem.

It does not modify B2, C2, or E2 evidence semantics.

## Three independent receipts

The research bootstrap stack remains explicitly compositional:

```text
B2 manifest-binding.json
  authority=adapter-binding-only
  scope_authority=manifest-declared-only

C2 program-profile-binding.json
  authority=program-scope-binding-only
  scope_authority=trusted-program-profile-v1

F2 package-root-binding.json
  authority=package-root-binding-only
```

F2 also publishes the exact deterministic E2 closure as:

```text
package-root-closure.json
```

## Fresh-runner theorem

On the authenticated sealing runner F2:

1. verifies the exact trusted policy HEAD;
2. independently verifies exact Git object/worktree identity for B2, C2, E2,
   and F2 policy helpers before import;
3. independently re-derives the E2 closure from the exact candidate source
   parent/head and trusted canonical profile;
4. reads the already-sealed B2 and C2 receipts;
5. requires the exact closed C2 v1 field set;
6. requires C2 to bind the exact B2 receipt SHA-256;
7. requires C2's recorded B2/C2 policy object identities to equal the exact
   current trusted policy objects;
8. requires program/manifest/profile/source/run/admission identities to agree;
9. re-authenticates the live PR when operating under `pull_request_target`;
10. re-reads every trusted policy object and re-derives E2 immediately before
    publication;
11. writes the deterministic closure first;
12. writes `package-root-binding.json` last.

A workflow must upload final evidence only after the final F2 binding exists.

## Binding contents

The F2 binding commits to:

```text
schema
seal_profile
authority
scientific_claim
program
manifest_path + manifest_sha256
source_parent + qualifier_head
trusted_policy_sha
manifest_binding_sha256
program_profile_binding_sha256
program_profile_canonical_sha256
package_root_closure_sha256
package_root_closure_file_sha256
package_root_count
trusted E2 validator/sealer Git blobs + SHA-256
admission/live-PR/run identity
```

## Why C2 is checked as a closed schema

A shape-compatible JSON object with extra authority-like fields must not be
promoted merely because a few expected keys match. F2 therefore requires the
complete current C2 binding field set before relying on it.

This couples F2 to C2 binding schema v1 deliberately. A future C2 schema must
produce a new explicitly reviewed F2 compatibility path.

## Why no same-runner post-candidate helper execution

Candidate Cargo/build-script code runs under the same OS identity as the
execution-runner checkout. This stack is not a hostile-code sandbox theorem.

Therefore activation should not execute a trusted Python helper *after* candidate
code on that same runner merely to re-check E2: a hostile candidate could have
rewritten that helper before process start.

The safe shape is:

```text
admission: derive E2
execution runner: rederive E2 BEFORE candidate code
B2 in-process postflight: retain source/worktree invariants
fresh seal runner: independently rederive E2 and seal
```

The fresh runner is the authenticated post-candidate boundary.

## Deliberate boundaries

F2 authenticates package/root closure publication. It still does not establish:

- workspace membership;
- transitive Cargo dependency closure;
- feature/build-script/proc-macro closure;
- Nix/store closure;
- runtime input closure;
- scientific sufficiency of the chosen scope;
- scientific correctness, replication, independence, or causality.

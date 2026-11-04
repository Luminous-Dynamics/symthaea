# CEB receipt verifier

This is a standalone Rust CLI for checking a CEB primary receipt and its pristine
regrade. It is intentionally outside the root Cargo workspace. At runtime the
verifier does not shell out, open network connections, invoke candidate builds or
tests, or execute candidate code. It does create a private temporary bare libgit2
object database to reconstruct the patch result; the temporary directory is cleaned
up automatically. Compiling the tool with Cargo may run dependency
build scripts; its focused CI job has read-only repository permissions and no secrets.

The three Draft 2020-12 schemas are embedded at compile time from
`benchmarks/ceb/`. Their exact SHA-256 digests are included in every report.
Schema validation uses the pinned `jsonschema 0.33.0` crate with default features
disabled: the verifier cannot retrieve remote or local external references.
Date-time formats are asserted.

## Decision inputs

The verifier needs:

- The exact task contract JSON bytes and its out-of-band SHA-256 pin.
- The exact candidate patch bytes.
- Read-only source and result snapshot directories, each independently hashed by CEB-TREE-V1.
- A primary DSSE envelope and a pristine-regrade DSSE envelope.
- A separately provisioned trust-policy JSON file and its out-of-band SHA-256 pin.
- Separate, immutable artifact roots for the two executions.

Trust policy shape:

```json
{
  "schema_version": 0,
  "policy_id": "ceb-runner-trust-2026-11",
  "minimum_signers": 1,
  "keys": [
    {
      "key_id": "runner-primary-key-01",
      "algorithm": "ed25519",
      "public_key_base64": "<canonical padded Base64 of 32 public-key bytes>",
      "role": "primary_runner",
      "trust_domain": "runner-control-plane-a",
      "status": "active"
    },
    {
      "key_id": "runner-regrade-key-01",
      "algorithm": "ed25519",
      "public_key_base64": "<canonical padded Base64 of 32 public-key bytes>",
      "role": "regrade_runner",
      "trust_domain": "runner-control-plane-b",
      "status": "active"
    }
  ]
}
```

The trust-policy digest must be provisioned independently. A trust file cannot
bootstrap its own authenticity. DSSE `keyid` values are only lookup hints; an
envelope counts only when its signature verifies against the public key in the
pinned trust policy and the key has the required stage role.

## CLI

Compute a CEB-TREE-V1 digest for a clean Unix snapshot:

```sh
cargo run --manifest-path tools/ceb-receipt-verifier/Cargo.toml --release -- hash-tree \
  --directory ./source-snapshot
```

This prints the algorithm identifier and lowercase SHA-256. The resulting digest
must be committed to the task contract by the trusted task registry; do not accept
a task and its purported digest from the same untrusted source.


Verify a source-only public-development agent view. The manifest SHA-256 and
CEB-TREE-V1 digest must both be supplied out-of-band by the task registry:

```sh
cargo run --manifest-path tools/ceb-receipt-verifier/Cargo.toml --release -- verify-agent-view \\
  --directory ./agent-view/workspace \\
  --manifest ./agent-view/manifest.v0.json \\
  --task-id CEB-REPO-001 \\
  --expected-manifest-sha256 <64-lowercase-hex> \\
  --expected-tree-sha256 <64-lowercase-hex>
```

The V0 verifier accepts only a strict public-development manifest, exact
regular-file inventories, `100644`/`100755` modes, per-file SHA-256 digests and
the existing CEB-TREE-V1 digest. It rejects symlinks, special files, `.git`,
non-ASCII and unsafe/non-portable names, case-fold collisions, extra/missing
files, unsorted/duplicate manifest entries, unpinned manifests, and any
manifest claiming sealed status. The manifest is metadata outside the hashed
`workspace` payload so the digest is not self-referential.

```sh
cargo run --manifest-path tools/ceb-receipt-verifier/Cargo.toml --release -- verify-pair \
  --task-contract task.json \
  --task-id CEB-RUST-001 \
  --expected-task-contract-sha256 <64-lowercase-hex> \
  --source-tree-dir source-snapshot \
  --result-tree-dir result-snapshot \
  --patch candidate.patch \
  --primary primary.dsse.json \
  --primary-artifacts-dir primary-artifacts \
  --regrade pristine-regrade.dsse.json \
  --regrade-artifacts-dir regrade-artifacts \
  --trust-policy runner-trust.json \
  --expected-trust-policy-sha256 <64-lowercase-hex>
```

The machine-readable decision is written to stdout. Exit code 0 means
`accepted`; any other decision exits nonzero. The report has no current-time
field, so the same immutable inputs produce byte-stable JSON.

## Fail-closed rules

- Schema revisions are embedded, not selected by untrusted receipt fields.
- Duplicate JSON object members, unknown schema fields and non-canonical
  Base64 are rejected.
- DSSE signatures cover the DSSE v1 PAE bytes, including payload type.
- Primary and regrade receipts must bind the same task, task-contract digest,
  fixture revision, source tree, result tree, patch, agent/harness configuration,
  grader, image digest, toolchain lock, runner policy and task budget. The receipt runtime
  profile, network policy, image digest, toolchain digest and runner-policy digest
  must each equal the corresponding value pinned by the task contract.
- The task contract hash and trust-policy hash are pinned out of band by the host control plane. The source snapshot must match the contract's tree digest; the result snapshot is hashed independently by the verifier. The custom tree algorithm sorts UTF-8 relative paths and commits each entry type, byte length, executable bit and content digest; symlink target bytes are hashed without following links. Empty directories are ignored; .git metadata, non-UTF-8 names and special files are rejected.
- The verifier computes the exact set of paths changed between source and result snapshots. The signed receipt's changed_paths and the Git patch's file headers must equal that set; quoted/space-containing patch paths are deliberately unsupported in V0.
- It then reconstructs the exact result independently: builds the source and result snapshots as Git trees in a private temporary bare libgit2 repository, parses the submitted diff with `Diff::from_buffer`, applies it with `Repository::apply_to_tree`, and compares the reconstructed tree object with the tree built from the supplied result snapshot. No worktree, shell, build, test or candidate process is created. Temporary object storage is cleaned up by RAII.
- Patch reconstruction is limited to UTF-8 textual Git diffs with supported unquoted path/hunk syntax. Binary patches and quoted/ambiguous paths return unresolved in CEB V0. The combined source/result input cap for reconstruction is 512 MiB.
- The verifier creates Git trees from both snapshots in a private temporary bare repository, parses the submitted patch with libgit2, applies it with `Repository::apply_to_tree`, and compares the resulting Git tree object with the expected result tree. The submitted patch must actually transform the source snapshot into the result snapshot. Binary patch reconstruction is unresolved in V0.
- The regrade links the SHA-256 of the exact primary envelope bytes; run IDs,
  receipt IDs, signer trust domains and check-evidence sets must be distinct.
- Receipt check IDs, kinds, required flags and explicit authorities are bound to the exact acceptance-check set in the task contract. Required-check aggregates are recomputed from rows, not trusted from totals.
- Every receipt must retain the exact patch, at least one execution artifact and
  exactly one stage-specific report (`primary_report` or `regrade_report`).
  Every path is relative, remains beneath its supplied root, resolves without
  symlink components, and matches declared size and digest. The total artifact
  input is capped at 512 MiB. Primary/regrade execution artifacts are disjoint.
- Usage is checked against the task contract's wall-clock, turn, tool-call,
  token and cost budgets.
- Simulation/host-development execution, unqualified graders, unrestricted
  network, missing artifacts, absent usage measurements and non-qualified
  task contracts cannot produce an accepted score.

## Claim boundary

The `verify-agent-view` command validates only static package integrity:
manifest pin, task binding, exact path inventory, per-file digests/modes and
CEB-TREE-V1 payload digest. It does not prove answer secrecy, evaluator
independence, licensing rights, sealed-holdout custody or runner isolation.
This verifier validates evidence artifacts. It does not create a secure runner,
prove that a claimed model provider served a named model, or turn signatures
into proof that an untrusted machine was isolated. The runner, grader, signing
key provisioning and artifact roots remain host-control-plane responsibilities.
Issue #10064 remains a hard prerequisite before executing any untrusted task,
build script, test, or agent-generated program.

The code is a first implementation slice. It must still pass Rust compilation,
the included unit tests, Clippy, formatting, schema conformance tests and the
external negative-control corpus before it can be qualified for scored benchmark use. Patch reconstruction is currently limited to UTF-8 textual Git diffs, and the source/result snapshots must remain immutable during hashing and reconstruction. Direct dependency versions are exact-pinned, but no Cargo.lock has been generated yet; transitive dependency resolution is therefore not locked.


## Current implementation status (2026-11-03)

The verifier source and focused CI workflow are in draft PR [#10212](https://github.com/Luminous-Dynamics/symthaea/pull/10212), stacked on CEB schema PR #10115. Nineteen Rust unit tests are authored, including strict JSON, DSSE PAE/signature verification, role-separated signers, task-check parity, forged counter rejection, budget enforcement, runtime pinning, artifact confinement/hash checks, deterministic tree hashing, source/result change-path checks, libgit2 patch reconstruction (valid transformation and wrong-result rejection), and a complete signed primary/regrade pair with detached-link and evidence-reuse negative controls.

**None of these Rust tests have been run in this environment.** There is no `cargo` or `rustc` toolchain available here; no Rust build, rustfmt, Clippy, Cargo.lock generation, schema conformance or end-to-end receipt-pair result is claimed. Direct dependency versions are exact-pinned, but transitive dependency versions remain unlocked until a Cargo.lock is generated. The dedicated GitHub workflow is queued, not passed.

The verifier intentionally returns unresolved for all current CEB catalog tasks because they remain `design_only`. Before any task can be scored, issue #10064 must qualify OS-level isolation, the task contract must be promoted with a frozen source and sealed split, and a real trusted signer policy must be provisioned outside the verifier inputs.

## CEB-TREE-V1 details

The digest is not a Git object ID and must not be compared directly with Git's
tree hash. It is a CEB-specific snapshot commitment:

1. Recursively enumerate regular files and symlinks; reject non-UTF-8 paths,
   special files and any .git path component. Empty directories are omitted.
2. Sort entries by UTF-8 relative-path bytes.
3. Begin SHA-256 with the domain separator `CEB-TREE-V1\0` and the entry count.
4. For every entry, hash path length and bytes, target/file byte length, entry
   type, executable bit, and SHA-256 of file contents or symlink target bytes.
5. Do not follow symlinks.

The V0 digest uses Unix executable-bit semantics and therefore requires the verifier to run on a Linux/Unix host compatible with the task's OCI environment.

Snapshots must be mounted read-only and immutable for the entire verification
operation. The standard-library path walk is not a defense against a concurrent
attacker mutating the tree during hashing or reconstruction; the host control plane
must enforce snapshot immutability.

The verifier now reconstructs the result for supported patches: it builds source
and result snapshots as Git trees inside a private temporary bare libgit2 repository,
applies the submitted UTF-8 textual Git diff to the source tree, and requires the
resulting Git tree object to match the independently built result snapshot. This
closes the path-only consistency gap. Binary patches and quoted/ambiguous paths
remain unresolved in CEB V0. The reconstruction limit is 512 MiB combined source
and result snapshot bytes.

A separate public-development agent-view inventory verifier and `verify-agent-view`
CLI have been added on a follow-on branch. It uses the same CEB-TREE-V1 commitment
and requires the manifest and tree digests to be pinned out of band. It is not
part of the currently merged verifier and is not compiled or tested yet.
That follow-on branch adds ten inventory-specific unit tests (31 total `#[test]` functions in the current verifier source by static count). Coverage includes parity with the original CEB-TREE-V1 implementation, exact inventory, external pins, extra/missing paths, content/mode changes, unsafe Windows names and control characters, case-fold collisions, symlinks and symlink roots, duplicate/unsorted entries, sealed-status rejection and unknown fields. None of the 31 tests have been compiled or executed in this work session.

The verifier still does not establish runner isolation, independently adjudicate
test semantics, or authenticate that a named model provider served the claimed
model. The temporary Git object database is an implementation detail of patch
reconstruction, not a security boundary; runner isolation remains governed by
issue #10064.

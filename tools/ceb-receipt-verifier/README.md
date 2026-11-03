# CEB receipt verifier

This is a standalone Rust CLI for checking a CEB primary receipt and its pristine
regrade. It is intentionally outside the root Cargo workspace. At runtime the
verifier does not shell out, open network connections, invoke candidate builds or
tests, or execute candidate code. Compiling the tool with Cargo may run dependency
build scripts; its focused CI job has read-only repository permissions and no secrets.

The three Draft 2020-12 schemas are embedded at compile time from
`benchmarks/ceb/`. Their exact SHA-256 digests are included in every report.
Schema validation uses the pinned `jsonschema 0.33.0` crate with default features
disabled: the verifier cannot retrieve remote or local external references.
Date-time formats are asserted.

## Decision inputs

The verifier needs:

- The exact task contract JSON bytes.
- The exact candidate patch bytes.
- The expected source-tree digest, supplied out of band.
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

```sh
cargo run --manifest-path tools/ceb-receipt-verifier/Cargo.toml --release -- verify-pair \
  --task-contract task.json \
  --task-id CEB-RUST-001 \
  --source-tree-sha256 <64-lowercase-hex> \
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
  fixture revision, source tree, patch, agent/harness configuration, grader,
  image digest, toolchain lock, runner policy and task budget. The receipt runtime
  profile, network policy, image digest, toolchain digest and runner-policy digest
  must each equal the corresponding value pinned by the task contract.
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

This verifier validates evidence artifacts. It does not create a secure runner,
prove that a claimed model provider served a named model, or turn signatures
into proof that an untrusted machine was isolated. The runner, grader, signing
key provisioning and artifact roots remain host-control-plane responsibilities.
Issue #10064 remains a hard prerequisite before executing any untrusted task,
build script, test, or agent-generated program.

The code is a first implementation slice. It must still pass Rust compilation,
the included unit tests, Clippy, formatting, schema conformance tests and the
external negative-control corpus before it can be qualified for scored benchmark use.


## Current implementation status (2026-11-03)

The verifier source and focused CI workflow are in draft PR [#10212](https://github.com/Luminous-Dynamics/symthaea/pull/10212), stacked on CEB schema PR #10115. The source currently contains unit tests for duplicate JSON members, DSSE PAE and signature checks, trust roles, task-check parity, budget enforcement, artifact digest/path validation, traversal and symlink rejection. These tests are authored but have **not** been run; no build, rustfmt, Clippy, schema-conformance or end-to-end receipt-pair result is claimed. No Cargo.lock has yet been generated in this environment.

The implementation intentionally returns unresolved for all current CEB catalog tasks because they remain `design_only`. Before any task can be scored, issue #10064 must qualify OS-level isolation, the task contract must be promoted with a frozen source and sealed split, and a real trusted signer policy must be provisioned outside the verifier inputs.

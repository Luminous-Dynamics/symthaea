# Native Interoception v0.1 — Qualification Gate Evidence Contract

Status: **mechanical qualification infrastructure; not scientific result evidence**

`QualificationReceipt` schema v2 replaces free-form gate evidence strings with typed,
source-bound evidence identities. The purpose is to make stale, cross-head, malformed,
or category-incompatible qualification evidence detectable before a v0.1 baseline can
be promoted.

## Required invariant

Every evidence-bearing qualification gate identifies the exact Git source commit that
was the subject of the check. That `subject_commit` must equal the enclosing
`QualificationReceipt.source_commit`.

A gate from another source head is invalid even if its reported status is `Passed`.
A later source change therefore cannot inherit a previous head's local or CI pass by
copying an evidence locator into a new receipt.

## Local command evidence

The local gates use `QualificationGateEvidence::LocalCommand` and record:

- exact subject source commit;
- executed command identity;
- SHA-256 of the captured qualification environment descriptor;
- SHA-256 of the command transcript/result artifact.

The fixed required local gates are:

- `local_fmt`;
- `local_test`;
- `local_clippy`.

Their evidence kind must be `LocalCommand`.

The environment descriptor should itself record, at minimum, the identities required
by the v0.1 qualification capsule: locked dependencies, Rust toolchain, target/host,
architecture, relevant flags, and other execution-semantic inputs. Its canonical
representation and hashing procedure belong to the qualification harness, not to the
native regulatory runtime.

## GitHub Actions evidence

The repository gates use `QualificationGateEvidence::GitHubActions` and record:

- exact subject source commit;
- workflow identity;
- GitHub Actions run ID;
- run attempt.

The fixed required repository gates are:

- `workspace_ci`;
- `showroom_integrity`.

Their evidence kind must be `GitHubActions`. The optional `benchmark_suite` observation
uses the same evidence kind when recorded. A skipped benchmark remains skipped and
never becomes benchmark success.

## Pending evidence

A `Pending` gate may omit evidence entirely. This supports a truthful receipt while a
runner is queued or infrastructure is unavailable.

`Pending` never satisfies qualification. A gate cannot become qualified merely by
having a well-formed locator; its status must be explicitly `Passed` and the complete
receipt/bundle must validate.

## External-verification boundary

Typed evidence is a **provenance-consistency mechanism**, not an authentication oracle.
The crate does not contact GitHub, inspect a local filesystem, verify a command exit
status, or prove that a caller truthfully labeled a run `Passed`.

Before constructing a final passing receipt, the qualification harness must resolve
and verify the referenced evidence:

1. verify that the GitHub Actions run actually belongs to `subject_commit`, the
   expected repository/workflow, and the recorded run attempt;
2. verify the observed GitHub conclusion/status rather than trusting a caller-provided
   paraphrase;
3. verify that a local transcript artifact hashes to `transcript_sha256`;
4. verify that its captured environment descriptor hashes to `environment_sha256`;
5. verify the command and exit result represented by that transcript;
6. only then construct the corresponding evidence-bearing gate status.

A future stronger layer may use signed attestations or transparency-log inclusion
proofs. That is not claimed by v0.1.

## Bundle boundary

`QualificationEvidenceBundle` schema v2 embeds the typed qualification receipt and the
`EvidenceCapsuleManifest` under one exact source/model-semantics lineage.

A bundle is qualified only when:

- both embedded artifacts validate;
- all source/model identities agree;
- every fixed required gate explicitly reports `Passed`;
- each evidence-bearing gate has valid typed evidence for the same source commit.

This prevents two classes of accidental promotion:

1. cross-pairing a qualification receipt and evidence capsule from different heads;
2. carrying a local/CI evidence locator forward from an older head inside an otherwise
   current receipt.

It still does not replace independent verification of external evidence.

## Scientific claim boundary

Passing this contract means only that the mechanical qualification lineage is
internally coherent and that the referenced gate evidence has been represented in a
machine-auditable form. It does not establish regulatory affect, emotion, subjective
feeling, sentience, or consciousness.

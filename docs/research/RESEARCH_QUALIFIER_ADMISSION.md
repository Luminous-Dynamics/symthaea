# Research Qualifier Admission v1

Status: source contract only. This document does not claim that the manifest runner, stable workflow, or any research subject is qualified until the relevant exact head executes successfully.

## Purpose

Research qualification should be executable independently of GitHub Actions, while GitHub remains one scheduler for the same repository-contained theorem.

SCI-INFRA-001A places that theorem in `scripts/qualify-research-crate.sh`. This tranche standardizes a trusted GitHub admission boundary around **data-only qualifier manifests** rather than creating a new executable workflow for every research program.

The separation is:

```text
qualification theorem
    !=
qualifier manifest
    !=
scheduler admission
    !=
qualification evidence
    !=
scientific authority
```

## End state

The intended bootstrap qualifier shape is:

```text
trusted stable workflow
        +
.github/research-qualifiers/<PROGRAM>.json
        |
        v
pre-helper exact-subject preflight
        |
        v
strict manifest validation
        |
        v
SCI-INFRA-001A portable harness
        |
        v
bootstrap qualification evidence
```

A new qualifier does not introduce executable CI YAML. Program-specific values are data. Scheduler and qualification logic remain centralized and reviewable.

## Trusted workflow boundary

The pull-request entry point uses `pull_request_target`, not ordinary `pull_request`.

That distinction is deliberate. An ordinary pull-request workflow may execute the workflow definition from the pull-request merge context. A qualifier that changed both its manifest and its workflow could therefore attempt to change the code that judges the same qualifier.

`pull_request_target` keeps the workflow definition on the trusted base/default-branch side. The workflow then checks out the exact qualifier head with persisted credentials disabled and repository permissions restricted to `contents: read`.

Because checking out a PR head under `pull_request_target` becomes unsafe if arbitrary checked-out code is executed, v1 adds two hard boundaries:

1. fork qualifiers are not admitted; the PR head repository must equal the repository executing the workflow;
2. before any repository helper or source code from the checked-out head executes, trusted inline workflow shell proves that the exact PR delta is one canonical qualifier manifest and nothing else.

Only after those facts hold may helper code from the qualifier head execute. Since the qualifier delta is exactly one manifest, the helper/harness bytes in the head are identical to its exact base.

No repository or organization secret is referenced by this workflow. The workflow token is explicitly reduced to `contents: read`.

## Pre-helper exact-subject proof

For an admitted pull request, the trusted workflow requires:

1. checked-out `HEAD` equals the event's exact PR head SHA;
2. `HEAD^` equals the exact event PR base SHA;
3. the base and head are one immediate-parent edge apart;
4. `git diff --name-only <base> <head>` contains exactly one path;
5. that sole path matches `.github/research-qualifiers/<PROGRAM>.json`.

Only after these checks pass does the workflow execute `scripts/run-research-qualification-manifest.py` from the checked-out head.

Consequently the admitted qualifier cannot modify its harness, manifest parser, workflow, Rust source, or any other repository path in the same subject.

This inline shell is an admission boundary, not a second qualification theorem.

## Manifest schema

The v1 schema is:

```text
symthaea.research-qualifier-manifest.v1
```

The manifest contains exactly:

```text
schema
program
source_parent
expected_rust
packages
source_paths
```

Unknown fields are rejected. Duplicate JSON object keys are rejected rather than accepting JSON's usual last-key-wins ambiguity. In particular, a manifest cannot add fields such as `authority`, `qualified`, `scientific_claim`, or policy overrides and have an older evaluator silently ignore them.

The canonical path is:

```text
.github/research-qualifiers/<PROGRAM>.json
```

The filename must equal the validated `program` value plus `.json`.

## Manifest validation

`run-research-qualification-manifest.py` fails closed on:

- wrong, missing, unknown, or duplicate JSON fields;
- malformed or overlong program identity;
- non-40-hex source parent;
- malformed Rust version;
- empty, duplicate, or oversized package/source-path lists;
- unsupported Cargo package characters;
- absolute, traversing, normalized-but-not-canonical, or unsupported repository paths;
- manifest paths outside `.github/research-qualifiers/`;
- filename/program disagreement.

The input language is intentionally narrow because these values eventually become arguments to qualification tooling.

## Exact manifest identity

After validation, the manifest is canonicalized as compact sorted JSON and identified by:

```text
manifest_sha256 = SHA256(canonical validated manifest JSON)
```

The digest is an **identity**, not evidence that the manifest is correct, qualified, safe, or scientifically authoritative.

## Subject continuity

The manifest runner independently re-checks the same exact-subject facts before either resolving scheduler metadata or invoking qualification:

```text
actual HEAD == expected PR head
actual HEAD^ == expected PR base
manifest.source_parent == expected PR base
diff(base, head) == [manifest path]
```

For the standard bootstrap path this means the qualifier subject is exactly one data-only child of the frozen source subject.

SCI-INFRA-001A then receives:

- the manifest's exact `source_parent`;
- the exact qualifier head;
- the manifest's package set;
- the manifest's source-path set;
- the manifest path as the only allowed qualifier-path delta;
- the manifest's Rust version.

The portable harness remains responsible for source byte identity, Cargo/rustfmt/test/Clippy gates, bootstrap lock handling, postflight immutability, and its own bounded qualification receipt.

## Draft-safe scheduler lifecycle

The trusted workflow listens for:

```text
opened
synchronize
reopened
ready_for_review
converted_to_draft
closed
```

Its runner-backed qualification job is admitted only when either:

- the event is an explicit manual invocation; or
- the `pull_request_target` subject is open, non-draft, and comes from the same repository.

Therefore:

```text
draft open / synchronize / reopen
    -> workflow may resolve
    -> qualification job skipped before runner-backed steps

same-repository ready_for_review
    -> exact ready head may qualify

same-repository ready + synchronize
    -> new exact ready head may qualify

converted_to_draft / closed / fork PR
    -> new event is non-admitted
```

The workflow also uses same-PR concurrency with `cancel-in-progress: true`. Actual cancellation or supersession is a GitHub scheduler property and must be observed before it is claimed for a specific subject.

## Stable workflow execution

`.github/workflows/research-bootstrap-qualification.yml` uses:

- `pull_request_target` for the trusted workflow definition;
- same-repository PR admission only;
- repository permission `contents: read` only;
- pinned checkout, Rust toolchain, and upload-artifact actions;
- exact-head checkout with persisted credentials disabled;
- pre-helper one-manifest diff proof;
- a bounded job timeout;
- the manifest-selected Rust toolchain.

The manifest wrapper invokes the portable shell harness explicitly through `bash`; qualification does not depend on the harness file's executable mode.

The stable workflow is intentionally generic. Program-specific semantics belong in the frozen source subject, manifest, and portable harness—not copied scheduler YAML.

## Evidence binding

After the portable harness returns success and emits `receipt.txt`, the manifest wrapper writes:

```text
symthaea.research-qualifier-manifest-binding.v1
```

The binding includes:

- `authority = adapter-binding-only`;
- `scientific_claim = NONE`;
- program identity;
- manifest path and canonical manifest SHA-256;
- frozen source parent;
- exact qualifier head;
- SHA-256 of the portable harness receipt.

This binds identities together without promoting their authority.

The harness receipt remains the owner of bootstrap-source qualification semantics. A binding receipt is not a replacement for that receipt and cannot upgrade it.

## Manual execution

`workflow_dispatch` remains available as a scheduler convenience. It does not weaken the exact one-parent/one-manifest subject contract: the workflow derives the base from `HEAD^` and requires the supplied manifest input to equal the sole changed qualifier path.

The portable harness can also be executed outside GitHub Actions, which is why qualification semantics remain outside the workflow in the first place.

## Fork boundary

The v1 GitHub adapter intentionally does not qualify fork-origin pull requests. This avoids combining a privileged/base-context event with execution of arbitrary fork-controlled source.

A forked contribution can still be inspected, reproduced into a trusted repository branch, or qualified through the portable harness in an appropriately isolated environment. Such a reconstruction creates a new explicit subject identity; it does not inherit qualification from the fork by similarity.

## Relationship to the runner backlog

The repository has accumulated hundreds of queued workflow runs, including bespoke research/qualification workflows created on feature branches. Main-level full-CI draft guards cannot prevent a newly introduced branch-local workflow from allocating a runner.

This architecture attacks the source of that problem for future research bootstrap qualification:

```text
one stable draft-safe workflow
+
data-only qualifier manifests
```

rather than:

```text
one new pull-request workflow per qualifier
```

It does not cancel historical queued runs or prove hosted-runner capacity is healthy.

## Migration

After SCI-INFRA-001A/B are qualified and the trusted workflow is present on the repository's default branch, new bootstrap qualifier subjects should prefer the manifest path.

Historical frozen qualifier heads should not be rewritten merely for stylistic consistency. Rewriting them would change the evidence subject. If an old lineage requires a genuine replacement qualifier, the replacement can adopt the manifest path when the centralized infrastructure is available for that lineage; otherwise the portable harness remains usable locally or through another appropriately bound scheduler.

No Git ancestry relationship by itself transfers qualification.

## Platform-policy dependency

`pull_request_target` is a privileged GitHub event and is subject to GitHub Actions event policy. Repository/organization policy must permit this narrowly hardened workflow for it to remain an available scheduler. A platform policy allowing the event is not itself evidence that this workflow is safe or qualified.

If the platform blocks the event, the portable qualification theorem remains available outside this scheduler adapter.

## Nonclaims

The existence of a manifest, stable workflow, manifest digest, or binding receipt does **not** establish:

- that source code is correct;
- that the qualifier executed;
- that Rust/Cargo/rustfmt/tests/Clippy passed;
- that a bootstrap lock candidate is acceptable;
- that final locked qualification passed;
- that GitHub concurrency cancellation occurred;
- that GitHub merge admission is enforced;
- that GitHub permits `pull_request_target` under the active Actions event policy;
- that hosted runner capacity is healthy;
- that scientific replication, independence, causality, or correctness exists.

The architecture establishes a narrow, fail-closed admission path from a same-repository data-only qualifier subject into the separate repository-contained qualification theorem.

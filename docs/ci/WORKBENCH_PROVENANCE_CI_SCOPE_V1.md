# Workbench Provenance CI Scope v1

Status: **candidate CI-routing theorem only; does not change scientific authority**

Schema: `symthaea-workbench-provenance-ci-scope-v1`

## Problem

The repository's global `CI` workflow currently runs on every pull request. A theorem-sized Workbench provenance PR therefore queues the full Rust/workspace matrix in addition to its dedicated focused qualification lane.

On the #690 exact-head inspection that motivated this tranche, the ordinary global workflow materialized 30 queued jobs while the dedicated Workbench Root NAR Membership job also waited for a runner.

That creates the wrong resource relationship:

```text
small dependency-free provenance theorem
    -> one focused qualification job
    + unrelated repository-wide Rust/workspace matrix
```

Repeated across a stacked provenance series, the unrelated matrix fan-out can delay the focused evidence needed to decide whether the theorem itself passes.

## Routing theorem

Let `D` be the non-empty set of changed repository paths and `W` the narrow Workbench provenance language recognized by the classifier.

```text
GlobalCiMaySkip(D) := (D != empty) AND (for every p in D: p in W)
```

`W` admits only canonical repository-relative paths shaped as:

```text
.github/workflows/workbench-<lowercase-hyphen-name>.yml

data/neuroscience/workbench_<lowercase_underscore_name>.json

docs/neuroscience/WORKBENCH_<UPPERCASE_UNDERSCORE_NAME>.md

scripts/workbench_<lowercase_underscore_name>.py
scripts/verify_workbench_<lowercase_underscore_name>.py
scripts/test_workbench_<lowercase_underscore_name>.py
scripts/test_verify_workbench_<lowercase_underscore_name>.py
scripts/check_workbench_<lowercase_underscore_name>.py
```

The classifier does not treat broad regions such as `docs/neuroscience/**`, `scripts/**`, `crates/**`, or `.github/workflows/**` as exempt.

Therefore:

```text
one disallowed path -> full CI
unknown spelling    -> full CI
mixed Rust + WB     -> full CI
Cargo change        -> full CI
global CI change    -> full CI
empty/ambiguous set -> full CI
```

There is no percentage threshold and no "mostly documentation" exception.

## Deployment is intentionally stricter than the language

The v1 GitHub trigger must **not** deploy the classifier's whole syntactic language as globs.

Instead, the deployment renderer contains the exact union of the 28 currently reviewed files from the six Workbench provenance PRs:

```text
#624 execution capsule profile
#629 Nix closure identity
#638 Nix closure capture
#667 independent closure-capture verifier
#681 invocation isolation profile
#690 root NAR membership
```

Thus:

```text
ExactDeploymentSet_v1 subset-of W
```

and intentionally not the reverse.

A future file that merely has a Workbench-looking name therefore receives full global CI until the reviewed deployment set is explicitly revised. This trades a harmless false negative (extra CI) for protection against an accidental false positive (silently skipping CI for an unreviewed surface).

## Exact deployment transformation

The deployment renderer is bound to the reviewed current main-branch `ci.yml` Git blob:

```text
a48366076b30eb8e12d22c927a3b8bf333181409
```

It accepts only those exact source bytes and replaces exactly one bare:

```text
  pull_request:
```

with:

```text
  pull_request:
    paths-ignore:
      - <exact reviewed path 1>
      - ...
      - <exact reviewed path 28>
```

Everything else in `ci.yml` must remain byte-identical.

The renderer rejects:

- an input whose Git blob is not the reviewed source;
- a missing or duplicate bare pull-request stanza;
- a source that already has the Workbench `paths-ignore` prefix;
- candidate output that differs from the exact transformation;
- attempts to overwrite an existing output artifact.

This makes concurrent edits to the global workflow a re-review event rather than silently applying the optimization to changed CI semantics.

## Pull-request integration ratchet

The focused qualification workflow has an additional deployment check.

When `.github/workflows/ci.yml` is changed in a pull request, it obtains the exact base-commit version with Git and requires:

```text
CandidateCiYml
    ==
Render(BaseCiYml)
```

using the same renderer.

Therefore a deployment PR cannot combine the routing change with an unrelated edit elsewhere in the 66-KB global workflow while still passing the focused routing lane.

When `ci.yml` is not changed, that step records that only the routing theorem itself is under qualification.

## Why `paths-ignore` has the right mixed-diff behavior

GitHub's pull-request `paths-ignore` rule skips a workflow only when all changed paths match the ignored set. If a pull request also changes any non-ignored file, the global workflow runs.

For the exact v1 deployment this gives:

```text
only exact reviewed Workbench provenance files
    -> global CI may omit

one Cargo/Rust/other file added
    -> global CI runs
```

The dedicated Workbench workflow remains independently triggered by the files it qualifies.

## Real-stack reconciliation

The classifier regression corpus pins the exact changed-file sets of:

- #624 — Workbench execution capsule profile;
- #629 — Nix closure identity;
- #638 — Nix closure capture;
- #667 — independent closure-capture verifier;
- #681 — invocation isolation profile;
- #690 — root NAR program-membership verifier.

Every current theorem-sized diff is accepted as Workbench-provenance-only. Their combined union is accepted as well, while adding even one unrelated path forces full CI.

The renderer separately freezes that same current union to exactly 28 deployable paths.

## Explicit non-exemptions

v1 intentionally does not exempt:

```text
Cargo.toml
Cargo.lock
crates/**
src/**
.github/workflows/ci.yml
scripts/derive_hcpmmp1_*.py
docs/neuroscience/NEURAL_*.md
scripts/workbench_*.sh
.github/workflows/workbench-*.yaml
```

A future desire to exempt another domain or another Workbench file requires a reviewed policy/deployment revision rather than broadening the exception by analogy.

## Path canonicality

The classifier rejects malformed path sets rather than normalizing them into the exempt language:

- absolute paths;
- `.` or `..` components;
- duplicate separators;
- backslashes;
- NUL/newline/carriage-return characters;
- duplicate changed paths;
- non-string entries;
- non-array inputs.

The changed-path set is sorted before output so classification evidence is deterministic.

## Current repository-protection assumption

At the time this profile was authored, `main` reports no required status-check contexts and no enabled branch protection through the available repository API.

That operational fact is not encoded as a permanent truth. If branch protection or repository rules later require the global `CI` workflow, the routing policy must be re-reviewed before relying on skipped runs.

## Scientific boundary

This tranche is not a neuroscience qualification tool.

```text
GlobalCiMaySkip
    != FocusedQualificationPassed
    != WorkbenchExecutionQualified
    != TransformExecuted
    != AtlasCorrectness
    != FMQ010
    != NeuralAlignment
```

Even when global CI is omitted, the PR's dedicated Workbench workflow remains required to execute its exact theorem.

The optimization is therefore:

```text
Workbench-provenance-only diff
        -> dedicated Workbench qualification
        -> omit unrelated full Rust matrix
```

not:

```text
Workbench-provenance-only diff
        -> no qualification
```

## Authority state

The classifier emits only CI-routing facts:

```text
global_ci_may_skip
focused_workbench_qualification_still_required
```

The renderer emits only a deterministic repository-workflow transformation. Neither emits any scientific authority field.

## Qualification

The v1 authored surface contains 46 adversarial contracts:

- 28 classifier contracts covering all six real-stack surfaces, mixed-diff escalation, canonical path handling, near-name attacks, duplicate/control-character rejection, deterministic ordering, and CLI exit semantics;
- 18 renderer contracts covering exact deployment-set identity/order, reviewed input binding, byte-preserving transformation, already-patched rejection, candidate tampering/injection, output overwrite protection, and Git-blob identity changes.

Both exact locally reconstructed GitHub source blobs have executed successfully as **46/46 local contracts** after self-review repaired two renderer defects (noncanonical exempt-path ordering and repeated patching of an already-patched stanza).

Hosted qualification is still a separate evidence boundary and must not be claimed until the exact focused workflow executes successfully.

The focused workflow runs all 46 contracts, verifies the optional `ci.yml` deployment transformation against the pull request's base commit, and statically forbids subprocess/network/scientific-execution imports from the routing implementation surface.

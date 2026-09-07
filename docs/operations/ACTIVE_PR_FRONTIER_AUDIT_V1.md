# Active PR Frontier Audit v1

## Purpose

This document freezes the interpretation boundary for `scripts/audit_pr_frontier.py`.

The audit exists to measure Symthaea's active review/scheduler surface without turning repository topology into authority to close, merge, retarget, or rewrite research work.

The central rule is:

```text
repository relationship evidence
    != repository mutation authority
```

The v1 tool is therefore read-only and uses GitHub GET endpoints only.

## Core non-equivalences

```text
research theorem != branch
branch != pull request
commit lineage != active review surface
qualification capsule != product PR
strict descendant != independent merge boundary
closed-as-superseded != erased history
exact ancestry != recommendation to close
queued workflow != executed evidence
```

A smaller active PR surface can improve scheduler/review efficiency while preserving every branch, commit, exact SHA, workflow result, negative result, and issue record.

## Exact open-parent theorem

The strongest ancestry relation emitted by v1 is intentionally narrow.

For an open child PR `C` and open candidate parent PR `P`, v1 calls `P` the child's `exact_open_parent` only when both are true:

```text
C.base.ref == P.head.ref
AND
C.base.sha == P.head.sha
```

The first equality establishes that the child targets the exact branch advertised by the parent.

The second equality establishes that the child's recorded base generation is the exact current parent head generation.

Both are required.

### Branch-name equality is insufficient

If:

```text
C.base.ref == P.head.ref
```

but:

```text
C.base.sha != P.head.sha
```

v1 emits:

```text
base_ref_sha_drift
```

not ancestry credit.

This may mean the parent advanced after the child was created, the child intentionally targets an older generation, or another repository operation changed the relationship. The audit does not guess which explanation is correct.

### Ambiguous branch ownership is insufficient

If multiple open PRs advertise the same `head.ref`, v1 reports `ambiguous_base` rather than selecting one by recency, PR number, title, body text, or SHA coincidence.

### Missing open parent is not failure

A non-main base branch that is not the head of another currently open PR is `unresolved_base`.

That may represent a preserved branch, closed/superseded PR, externally managed integration branch, or another legitimate state. V1 does not infer abandonment or error.

## Root and depth semantics

Stack depth is computed only through consecutive `exact_open_parent` relations.

Example:

```text
#10 head=a@A
#11 base=a@A, head=b@B
#12 base=b@B, head=c@C
```

produces:

```text
#10 depth 0
#11 depth 1
#12 depth 2
```

If any edge becomes drifted, ambiguous, or unresolved, exact-depth propagation stops at that boundary.

A graph cycle is treated as malformed topology and receives no ordinary root/depth interpretation.

## Qualification-only detection has two evidence tiers

V1 deliberately distinguishes textual intent from changed-file proof.

### Tier H — self-declared heuristic

A PR may be marked `self_declared_qualification_only` when its title/body contains narrow phrases such as:

```text
qualification-only
no product source
exact candidate patch
```

This is a navigation heuristic only.

```text
PR description says qualification-only
    !=
changed files prove qualification-only
```

The field must never be used alone for automated closure, merge, execution, or authority decisions.

### Tier F — strict workflow + patch capsule morphology

When file enrichment is requested, v1 may mark `workflow_patch_capsule=true` only when every changed path is one of:

```text
.github/workflows/*.yml|yaml
docs/release/evidence/*.patch
```

and the PR contains at least one workflow and at least one patch artifact.

Any product source, manifest, lockfile, documentation outside the patch evidence surface, script, generated output, or other file causes the strict capsule test to return false.

This proves only file morphology:

```text
workflow + immutable patch shape
    !=
workflow executed
    !=
candidate qualified
    !=
product repair materialized
```

## Workflow-demand enrichment

Optional run enrichment queries workflow runs attached to an exact PR head SHA and reports current aggregate counts.

These counts are scheduler observations only.

```text
run exists != job assigned
queued != PASS
in_progress != meaningful progress without job evidence
cancelled != FAIL
workflow success != every scientific claim established
```

V1 does not cancel or rerun workflows.

Because per-PR file/run enrichment costs API calls, the tool defaults to metadata-only operation. Enrichment is bounded by `--enrich-limit`, and unauthenticated enrichment is refused unless the operator explicitly opts into the lower-rate-limit mode.

## Read-only invariant

The script contains no GitHub mutation endpoint.

It must not acquire code that performs:

- PR closure/reopening;
- retargeting base branches;
- merging;
- branch deletion;
- force pushing;
- labeling/assigning;
- issue or PR commenting;
- workflow dispatch/cancellation;
- branch creation;
- repository settings changes.

A future mutation tool, if ever justified, must be a separate executable and review boundary. Do not add a `--apply`, `--close`, or equivalent flag to this auditor.

## No automatic closure theorem

The following is explicitly invalid:

```text
exact_open_parent
    -> close child
```

An independently active PR can remain justified because it has:

- a genuine independent merge decision;
- separate reviewer ownership;
- an executable gate that should run independently now;
- a safety/security isolation boundary;
- an alternative architecture being compared against siblings;
- an external collaboration/release contract;
- a diff that would become unreviewable when consolidated;
- already-published evidence/results that deserve a stable review surface.

Therefore the audit supplies topology evidence to a later human/governance decision; it does not make that decision.

## Supersession preservation requirements

If a later explicit decision closes an intermediate PR as superseded, preserve at least:

```text
superseded PR number
final exact head SHA
superseding issue/PR
whether commits remain ancestors of the active frontier
existing exact workflow results
negative/inconclusive evidence
branch-retention status
reason the independent active review boundary is no longer needed
```

Do not squash, force-rewrite, or delete research history merely to reduce Actions demand.

## Relationship to #744

Issue #744 owns active-review-frontier governance.

This audit is its first read-only instrument. Its job is to answer questions such as:

```text
How many open PRs are exact descendants of other open PRs?
How deep are current exact stacks?
Where has a parent branch advanced after children were based?
Which PRs merely describe themselves as qualification-only?
Which enriched PRs actually have workflow+patch-only file shape?
How much workflow demand is attached to sampled exact heads?
```

It does not answer:

```text
Which PR should be closed?
Which theorem is scientifically valid?
Which branch should merge first?
Which workstream deserves more resources?
```

## Relationship to #217

Issue #217 owns CI lifecycle tiering and the proposed Tier-Q exact qualification path.

The intended future composition is:

```text
issue/theorem
    ↓
branch + exact candidate
    ↓
Tier-Q focused qualification without product PR fan-out
    ↓
PASS / FAIL / INCONCLUSIVE
    ↓
active product PR only when an independent review/merge boundary is justified
```

This audit can measure whether that architecture reduces active PR and Actions fan-out, but it must not be the scheduler itself.

## Relationship to #75

Issue #75 owns runner/account capacity and the narrow trusted CPU recovery substrate.

PR-topology reduction cannot substitute for adequate runner capacity, and additional runner capacity cannot justify unnecessary PR/workflow fan-out.

They are complementary controls:

```text
#75  supply / execution capacity
#217 demand / CI lifecycle
#744 review-topology admission
```

## Interpretation of a future baseline

A useful report should separate at least:

```text
open PR count
ready vs draft count
exact open-parent edge count
exact multi-PR lineage count
maximum exact depth
base-ref/SHA drift count
ambiguous/unresolved bases
self-declared qualification-only count
strict workflow+patch capsule count among enriched PRs
workflow demand among enriched exact heads
```

If only a subset is enriched, the report must label that subset. Do not extrapolate strict capsule/run statistics to all open PRs without measuring them.

## Acceptance criteria

V1 is acceptable when:

1. every exact parent edge requires ref and SHA equality;
2. branch-name/SHA drift is visible rather than silently accepted;
3. ambiguous parents receive no ancestry credit;
4. textual qualification intent remains explicitly heuristic;
5. strict workflow+patch capsule morphology is independently file-derived;
6. enrichment is optional and rate-limit-conscious;
7. output includes explicit interpretation boundaries;
8. no GitHub mutation operation exists in the tool;
9. fixture tests freeze the above invariants;
10. no report is presented as authority to close or merge a PR.

The metric to optimize is not minimum PR count. It is:

> minimum active scheduler/review surface consistent with clear independent decisions, preserved evidence, and auditable research lineage.

# HUM-OBS-002B — BEHAVIOR 2026 Submission-Set Evidence

Status: source-design candidate
Issue: #4898
Parent: HUM-OBS-002A / #4890
Authority: aggregate benchmark evidence only; **no leaderboard, physical, safety, or deployment authority**

## Purpose

Prevent a small or cherry-picked collection of individually valid BEHAVIOR rollouts from being represented as a complete 2026 public-report evaluation.

## Prescribed public set

The current 2026 public-report profile requires:

- BEHAVIOR evaluator/repository version `v3.9.2`;
- exactly 100 challenge tasks;
- exactly 10 prescribed public-report instances per task (`0..=9`);
- exactly one planned rollout for each task/instance pair;
- 1,000 expected rollout results in a complete set.

Partial sets remain valid evidence, but missing prescribed entries receive zero credit in the reconstructed primary score and coverage remains independently visible.

## Two-level identity

`Behavior2026SubmissionManifestV1` now separates the benchmark task-set identity from the subject-specific run manifest.

### Task-set commitment

`task_set_commitment` binds only the evaluation semantics that should remain identical across competing subjects:

- BEHAVIOR version `v3.9.2`;
- public-report split;
- task-set label;
- canonical ordered list of 100 task IDs.

It uses its own domain-separated BLAKE3 commitment. Reordering or changing any task changes this identity.

### Subject/run manifest commitment

`manifest_commitment` then binds:

- the exact `task_set_commitment`;
- exact Symthaea source/model/morphology/sensor-actuator subject;
- wrapper configuration reference;
- robot configuration reference.

This lets two systems prove they ran against the same benchmark task set while retaining different subject/run identities.

## Planned rollout identity

Every prescribed task/instance pair gets a deterministic rollout ID derived from:

```text
manifest commitment
+ task ID
+ instance index
```

The submission evaluator rejects a result whose rollout ID differs from the predeclared planned ID. This makes best-of-retry substitution detectable **inside the governed pipeline**.

It does not prove that no additional unobserved external attempts were executed. That limitation is preserved explicitly.

## Set dispositions

### CompletePrescribedSet

Every one of the 1,000 prescribed task/instance entries appears exactly once and no set-level violation exists.

### PartialPrescribedSet

The received rollouts are valid and conform to the manifest, but one or more prescribed entries are missing. Missing entries contribute zero to primary-score reconstruction.

### InvalidSet

At least one structural set violation exists, including duplicate prescribed entries, wrong split/version/task-set/configuration, unexpected task identity, malformed rollout evidence, or rollout-plan mismatch.

An invalid set does not expose an aggregate `q_score`.

Identity/version diagnostics are evaluated before generic structural validation where possible, so an artifact with the wrong BEHAVIOR version is reported specifically as `WrongBenchmarkVersion` rather than being collapsed into `MalformedRollout`.

## Aggregate score reconstruction

For a non-invalid set:

```text
aggregate_q_score = sum(received q_score) / 1000
```

because missing prescribed entries contribute zero. Since every task has the same 10 expected instances, this is equivalent to averaging the ten-instance task means across the 100 tasks.

Coverage is reported separately:

```text
received unique prescribed rollouts / 1000
```

A high score cannot erase low coverage.

## Tests

Source tests cover:

- exactly 100 unique task identities required;
- deterministic and distinct task-set + subject-manifest commitments;
- task reordering changes the task-set commitment;
- deterministic rollout commitments;
- complete 1,000-rollout set recognition;
- partial-set zero-credit reconstruction and visible coverage;
- duplicate prescribed entry invalidation;
- nonplanned rollout identity invalidation;
- wrong benchmark version receives a specific violation;
- robot configuration mismatch invalidation.

## Nonclaims

A complete internally reconstructed set is not an organizer-accepted submission, official leaderboard score, proof that no out-of-band retries occurred, real-robot performance, household deployment readiness, safety qualification, or product authority.

# HUM-OBS-003B — Predeclared HumanoidBench Evaluation Matrix

Status: source-design candidate
Issue: #4939
Parent: HUM-OBS-001D / #4968
Authority: external simulation evaluation evidence only; **no physical, safety, or deployment authority**

## Purpose

Prevent a selected subset of HumanoidBench episodes from being presented as a broader evaluation after the results are known.

This is a **Symthaea evaluation protocol** layered over the pinned upstream benchmark. It is not an official upstream scoring protocol and it does not invent a universal cross-task score.

## Pinned upstream execution vocabulary

The parent adapter remains pinned to:

`carlosferrazza/humanoid-bench@cb1189039151c8aadaaa987b442da54383c87fab`

At that exact source revision, `humanoid_bench/__init__.py` registers the full Cartesian product of the declared robot registry and task registry. It selects torque control for `g1` and position control for the other robots present in the pinned registry.

003B therefore validates the exact pinned registry/control semantics. Registration still does **not** imply successful execution, useful performance, physical transfer, or deployment readiness.

## Predeclared matrix

`HumanoidBenchEvaluationMatrixV1` binds one evaluated subject and evidence profile to an exact ordered case list:

```text
matrix identity
+ exact upstream commit
+ exact task-registry commitment
+ exact subject identity
+ authority profile
+ evidence profile
+ runner configuration reference
+ ordered cases
```

Each case binds:

- task ID;
- robot ID;
- pinned control mode;
- `Specified(u64)` or `Unspecified` seed state;
- observation/sensor profile identity;
- environment/configuration identity.

Identical duplicate cases fail closed. Distinct seed states remain distinct cases.

## Matrix identity

The matrix receives a domain-separated BLAKE3 commitment before results are admitted.

Case ordering is intentionally significant. Reordering a case list produces a different matrix commitment even if the set of cases is otherwise identical.

This makes the evaluation plan itself evidence rather than an after-the-fact description.

## Planned episode identity

Every case receives a deterministic planned episode ID derived from:

```text
matrix commitment
+ case index
+ exact case semantics
```

The external episode result must use that exact planned ID. An unplanned ID is invalid matrix evidence and never becomes extra coverage.

## Result binding

`HumanoidBenchMatrixCaseResultV1` binds the imported HUM-OBS-003A episode to:

- exact matrix commitment;
- exact runner configuration reference;
- exact observation profile;
- exact capability subject.

The assessment then requires exact agreement with the corresponding planned case for task, robot, control mode, seed state, environment, authority profile, and evidence profile.

A substituted policy, runner configuration, seed, task, robot, profile, or environment is recorded as invalid matrix evidence rather than silently accepted.

## Coverage states

A result set is classified as:

- `CompletePredeclaredMatrix` — every planned case has one structurally valid submitted result;
- `PartialPredeclaredMatrix` — no invalid evidence is present, but some planned cases are missing;
- `InvalidMatrixEvidence` — duplicate, unplanned, substituted, or otherwise inconsistent result evidence is present.

`CompletePredeclaredMatrix` means **evidence-set coverage**, not successful task performance.

For example:

```text
all planned cases supplied
+
all runner attempts infrastructure-indeterminate
=
complete matrix evidence set
!= complete demonstrated capability
```

Infrastructure-indeterminate planned cases remain visible as such and contribute no completed return sample.

## Reporting without a global score

The coverage receipt reports separately:

- expected case count;
- supplied result count;
- admitted planned-case count;
- exact missing planned episode IDs;
- exact evidence violations;
- expected/admitted coverage by Symthaea task family;
- per-task completed return samples;
- per-task termination/truncation counts;
- per-task infrastructure-indeterminate counts.

Completed returns remain samples tied to exact planned episode IDs. A negative or otherwise poor return remains valid evidence about completed execution; it is not rewritten as an infrastructure failure.

No arithmetic combines unrelated tasks into one HumanoidBench-wide score.

## Coverage receipt commitment

The receipt receives a domain-separated commitment over matrix identity, coverage status, counts, missing IDs, violations, family coverage, per-task return samples, and termination/infrastructure evidence.

Per-task return samples are deterministically ordered by planned episode identity before commitment.

## Typed-evidence relationship

003B is stacked on HUM-OBS-001D so imported episodes can also preserve exact seeds, booleans, and artifact references without numeric coercion.

The matrix protocol itself still validates the original lossless episode record rather than treating the common Observatory summary as the sole source of truth.

## Tests

The first source campaign covers:

- deterministic matrix commitment;
- case-order sensitivity;
- duplicate-case rejection;
- pinned robot/control semantics;
- `seed=0` versus `seed unspecified`;
- complete matrix coverage;
- partial matrix coverage;
- low/negative return remaining completed evidence;
- runner-configuration substitution rejection;
- unplanned episode rejection;
- infrastructure failure remaining explicitly indeterminate.

## Nonclaims

This protocol establishes no official upstream leaderboard standing, universal humanoid score, physical humanoid performance, sim-to-real transfer, safety qualification, household readiness, or deployment authority.

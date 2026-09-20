# HUM-OBS-003C — HumanoidBench Runner Subject + Execution Receipts

Status: source-design candidate
Issue: #4980
Parent: HUM-OBS-003B / #4978
Authority: execution provenance only; **no physical, safety, or deployment authority**

## Purpose

HUM-OBS-003B proves what evaluation cases were predeclared and whether imported result metadata matches those cases. It does not prove that the referenced runner actually executed the claimed policy/model/configuration bytes.

The missing theorem is:

```text
result metadata matches matrix
!=
claimed subject bytes actually executed
```

003C adds a pre-execution subject manifest and a post-execution receipt.

## Pre-execution subject manifest

`HumanoidBenchRunnerSubjectV1` binds:

- exact 003B matrix commitment;
- exact case index and deterministic planned episode ID;
- exact Symthaea source head;
- model/policy ID, artifact ref, and commitment;
- morphology ID, artifact ref, and commitment;
- sensor/actuator profile ID;
- exact simulation-profile ref and commitment;
- pinned HumanoidBench upstream commit;
- exact task / robot / control / seed / observation / environment semantics;
- runner artifact ref and commitment;
- runner configuration ref and commitment;
- MuJoCo/runtime ref and commitment;
- execution-environment identity.

The subject is validated against the predeclared matrix before it can receive a domain-separated commitment.

Changing the policy artifact commitment while leaving the human-readable policy ID unchanged therefore changes the subject manifest.

## Hermeticity

Execution environment identity is explicit:

```text
Hermetic { environment_ref, environment_commitment }
```

or:

```text
ResidualUncertainty {
    environment_ref,
    environment_commitment,
    uncertainty_ref,
}
```

A partially controlled Python/GPU/driver environment may still produce evidence, but it must disclose the residual uncertainty rather than claiming hermetic reproducibility.

## Phase-aware execution receipts

The runner records its strongest established phase:

```text
Prepared
EnvironmentConstructed
EpisodeStarted
EpisodeCompleted
```

and separately records disposition:

```text
Completed
InfrastructureFailure
```

This distinguishes, for example:

```text
runner prepared but environment construction failed
```

from:

```text
environment constructed, episode began, then infrastructure failed
```

and from:

```text
episode completed
```

A completed disposition is valid only with `EpisodeCompleted` plus raw-result and adapter-input evidence.

## Result binding

For completed results, the receipt binds:

- exact subject-manifest commitment;
- exact planned episode ID;
- clock domain and start/end times;
- raw result artifact ref + commitment;
- canonical adapter-input commitment;
- bounded log references;
- exact receipt commitment.

The adapter input itself is committed with domain-separated BLAKE3 over the validated serialized `HumanoidBenchEpisodeResultV1`.

The receipt then verifies that the adapter input matches the subject's task, robot, control mode, seed state, environment, upstream commit, planned episode, and runner artifact.

## Separation of claims

```text
predeclared case
!= runner subject

runner subject
!= execution receipt

execution receipt
!= good benchmark performance

good benchmark performance
!= physical-world capability

physical-world capability
!= deployment authority
```

003C strengthens provenance only.

## Tests

Source tests cover:

- deterministic subject commitment;
- policy identity substitution rejection;
- exact planned episode binding;
- exact seed binding;
- completed receipt / adapter-input binding;
- raw result substitution changing receipt identity;
- infrastructure failure before episode start;
- residual environment uncertainty disclosure.

## Next integration boundary

A future 003B revision should require a valid 003C receipt for high-assurance matrix admission. Until then, 003C remains an independently reviewable execution-provenance candidate rather than silently changing 003B's qualified semantics.

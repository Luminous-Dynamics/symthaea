# Symthaea Robotics Recovery Audit

A dependency-free Rust reference kernel for one narrow obligation in ROB-RECOVERY-ASSURANCE-001: **do not count simulator state rewind as physically realizable robot recovery**.

The crate audits trial identities and separates:

- safe task completion through a physically realizable intervention;
- successful-but-unsafe completion;
- simulator-only completion after state rewind;
- safe hold;
- human handoff;
- task failure and infrastructure-indeterminate outcomes.

It also rejects duplicate trial IDs, blank identity fields, malformed expected-trial manifests (duplicate or blank IDs), and trials outside a frozen manifest. Missing expected trials remain visible instead of disappearing from the denominator. Raw observed rewind and safety-violation markers remain visible even when a row is excluded from confirmatory accounting.

## Run

    cargo test -p symthaea-robotics-recovery-audit
    cargo fmt --manifest-path crates/domains/symthaea-robotics-recovery-audit/Cargo.toml -- --check

Thirty-four unit tests are included across the audit kernel and RoboRecover scorer. They have not yet been executed in this environment. The crate deliberately has no external dependencies and does not create a policy runtime, simulator, action authority, sensor classifier, or physical safety controller. The crate includes a narrow RoboRecover score adapter; it is not a policy runtime or action authority.


## RoboRecover recovery scoring

The `roborecover_score` module consumes a frozen scenario manifest and normalized source result records. It reports three distinct rate families, all with the same scenario-equal and Macro-9 weighting:

- **Source environment-completion RSR:** raw source completion evidence. This may include completion during replay before a policy has acted.
- **Policy-recovery RSR:** task completion attributable to the policy after handoff.
- **Physical-safe RSR (Luminous extension):** policy-attributable completion plus continuous physical state, a physically realizable intervention, executed actions and upstream verified compliant safety evidence.

For every family, Overall RSR averages scenario rates equally; Macro-9 averages rates within the nine declared occupied Stage–Type groups, then gives each group equal weight. The returned score receipt carries the canonical upstream repository, exact source revision, split-manifest Git blob SHA, evaluator SHA-256, per-scenario content SHA-256, platform, split and arm. The scorer rejects blank or malformed digests and rejects any trial whose scenario content or evaluator digest differs from the frozen manifest. One score is bound to one platform, one split and one policy/intervention `arm_id`. Do not pool base-policy, correction, monitor or rewind arms.

### Source status normalization and an upstream ambiguity

The adapter uses source-specific completion rules:

- RoboTwin policy completion requires `status=infer_ok`, `inference_steps>0` and `eval_success_at_end=true`.
- LIBERO policy completion requires `status=success` and `inference_steps>0`.
- `replay_already_done` is always a replay-preempted environment completion, never a policy recovery.
- Contradictory or unknown status/success combinations remain explicit and are not counted as policy success.

This distinction is necessary because the inspected RoboTwin source can emit `status=replay_already_done`, `infer_steps=0` and `eval_success_at_end=true`. Its source summary counts the raw boolean as success while its failed-files list includes the same row. The ambiguity is documented in [upstream RoboRecover issue #2](https://github.com/RUCKBReasoning/RoboRecover/issues/2). That issue's constructed reproducer is not evidence that a published result was affected.

The scorer therefore reports source environment completion, replay-preempted rows, policy-attributable recovery, unknown outcomes and contradictory outcomes separately. This is a policy-attribution clarification, not a claim that we have reproduced the paper's published leaderboard results. Full scenario data hosting and redistribution terms remain pending in the upstream repository.

The nine Macro-9 groups are explicitly frozen in the manifest and matched to the paper's declared test groups. No group is inferred from whichever scenarios happened to execute. Missing, duplicate, unplanned, identity-invalid and metadata-mismatched trials remain failures in the planned denominator.

## Interpretation

PhysicalSafeCompletion is a bookkeeping disposition, not a safety certificate. It requires a non-rewound world, an allowed intervention class, at least one recorded executed action, and a VerifiedCompliant safety disposition supplied by an upstream evidence process.

The audit kernel does **not** validate the truth of sensor evidence, prove that a command physically reached an actuator, implement RoboRecover's scenario-equal RSR/Macro-9 scorer, or establish hardware qualification. Those remain separate owners and obligations.

State rewind is always simulator-only, even if another field incorrectly claims that physical state was preserved. A controller-version rollback can count only when the physical world was preserved and subsequent actions were recorded. Safe hold and human handoff are reported separately; neither is autonomous task completion.

All counts are unweighted. This crate intentionally emits no universal robotics score.

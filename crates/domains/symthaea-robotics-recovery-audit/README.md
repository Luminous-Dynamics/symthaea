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

Thirty-one unit tests cover the fail-closed audit, manifest and source-adapter scoring cases. The crate deliberately has no external dependencies and does not create a policy runtime, simulator, action authority, sensor classifier, or physical safety controller. The crate includes a narrow RoboRecover score adapter; it is not a policy runtime or action authority.


## RoboRecover policy-recovery scoring

The `roborecover_score` module consumes a frozen scenario manifest and normalized source result records.

- **Policy-recovery RSR:** calculate each scenario's success rate using policy-attributable task completion, then average scenarios equally.
- **Macro-9 RSR:** average scenario-equal rates within each of the nine declared occupied Stage–Type groups, then average the nine groups equally.
- **Physical-safe RSR (Luminous extension):** apply the same scenario/group weighting, but count only physically continuous, executed, upstream-verified compliant completions.

Each score is bound to one platform, one split and one policy/intervention `arm_id`. Never pool base-policy, correction, monitor or state-rewind arms into one result. Every planned trial remains in the denominator; missing, duplicate, unplanned, identity-invalid and metadata-mismatched trials do not become successes.

### Source status normalization and an upstream ambiguity

The adapter deliberately distinguishes three things:

1. **Source environment completion:** RoboTwin's raw `eval_success_at_end` or LIBERO's completion status.
2. **Policy-attributable completion:** RoboTwin requires `status=infer_ok`, `inference_steps>0` and `eval_success_at_end=true`; LIBERO requires `status=success` and `inference_steps>0`. `replay_already_done` is a separate replay-preempted result, never policy recovery.
3. **Physical-safe completion:** additionally requires continuous physical state, a physically realizable intervention, executed actions and upstream verified compliant safety evidence.

This distinction is necessary because the inspected RoboTwin source can emit `status=replay_already_done`, `infer_steps=0` and `eval_success_at_end=true`. Its source summary counts the raw boolean as success while its failed-files list includes the same row. The ambiguity is documented in [upstream RoboRecover issue #2](https://github.com/RUCKBReasoning/RoboRecover/issues/2). The issue's constructed reproducer is not evidence that any published leaderboard result was affected. We therefore report source environment completions, replay-preempted rows, policy-attributable recoveries, unknowns and contradictions separately instead of silently choosing one meaning.

The nine Macro-9 groups are frozen explicitly in the manifest and matched to the paper's declared test groups. No group is inferred from whichever scenarios happened to execute. This adapter does not claim to replace the upstream scorer or reproduce published results until exact source revision, evaluator behavior and complete licensed scenario data are pinned and tested.

## Interpretation

PhysicalSafeCompletion is a bookkeeping disposition, not a safety certificate. It requires a non-rewound world, an allowed intervention class, at least one recorded executed action, and a VerifiedCompliant safety disposition supplied by an upstream evidence process.

The audit kernel does **not** validate the truth of sensor evidence, prove that a command physically reached an actuator, implement RoboRecover's scenario-equal RSR/Macro-9 scorer, or establish hardware qualification. Those remain separate owners and obligations.

State rewind is always simulator-only, even if another field incorrectly claims that physical state was preserved. A controller-version rollback can count only when the physical world was preserved and subsequent actions were recorded. Safe hold and human handoff are reported separately; neither is autonomous task completion.

All counts are unweighted. This crate intentionally emits no universal robotics score.

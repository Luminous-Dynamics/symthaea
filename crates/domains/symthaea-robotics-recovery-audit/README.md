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

Twenty-four unit tests cover the fail-closed audit and scoring cases. The crate deliberately has no external dependencies and does not create a policy runtime, simulator, action authority, benchmark-specific score implementation, sensor classifier, or physical safety controller.


## RoboRecover native scoring

The public \`roborecover_score\` module adds a typed scorer over a frozen
scenario manifest:

- **Overall RSR:** compute each scenario's mean task success across its
  expected trials, then average equally across scenarios.
- **Macro-9 RSR:** compute scenario-equal rates within each explicitly
  declared occupied Stage–Type group, then average the nine groups equally.
  The manifest must supply exactly nine unique groups and every planned
  scenario must belong to one of them. The scorer does not guess which groups
  the benchmark release contains.
- **Physical-safe RSR (Luminous extension):** use the same scenario/group
  weighting, but only count \`PhysicalSafeCompletion\` dispositions.

A manifest is bound to one platform, one split and one policy/intervention
\`arm_id\`. Do not pool base-policy, correction, monitor, or simulator-rewind
arms into one result. The native task-success metric can count a simulator
rewind or an unsafe completion as a task success, as the benchmark's task
metric may; the audit separately reports rewind and safety violations, while
physical-safe RSR excludes both. A reversion arm therefore cannot be presented
as a physically realizable recovery result.

Missing, duplicate, unplanned, identity-invalid and metadata-mismatched trial
rows are not successes, but their planned slots remain in the denominator.
The native RSR/Macro-9 and Luminous physical-safe extensions must be published
side by side. This scorer does not replace the official RoboRecover evaluator
or independently establish physical safety.

## Interpretation

PhysicalSafeCompletion is a bookkeeping disposition, not a safety certificate. It requires a non-rewound world, an allowed intervention class, at least one recorded executed action, and a VerifiedCompliant safety disposition supplied by an upstream evidence process.

The audit kernel does **not** validate the truth of sensor evidence, prove that a command physically reached an actuator, implement RoboRecover's scenario-equal RSR/Macro-9 scorer, or establish hardware qualification. Those remain separate owners and obligations.

State rewind is always simulator-only, even if another field incorrectly claims that physical state was preserved. A controller-version rollback can count only when the physical world was preserved and subsequent actions were recorded. Safe hold and human handoff are reported separately; neither is autonomous task completion.

All counts are unweighted. This crate intentionally emits no universal robotics score.

# Symthaea Robotics Recovery Audit

A dependency-free Rust reference kernel for one narrow obligation in ROB-RECOVERY-ASSURANCE-001: **do not count simulator state rewind as physically realizable robot recovery**.

The crate audits trial identities and separates:

- safe task completion through a physically realizable intervention;
- successful-but-unsafe completion;
- simulator-only completion after state rewind;
- safe hold;
- human handoff;
- task failure and infrastructure-indeterminate outcomes.

It also rejects duplicate trial IDs, blank identity fields, and trials outside an optional frozen expected-trial manifest. Missing expected trials remain visible instead of disappearing from the denominator.

## Run

    cargo test -p symthaea-robotics-recovery-audit
    cargo fmt --manifest-path crates/domains/symthaea-robotics-recovery-audit/Cargo.toml -- --check

The crate deliberately has no external dependencies and does not create a policy runtime, simulator, action authority, benchmark-specific score implementation, sensor classifier, or physical safety controller.

## Interpretation

PhysicalSafeCompletion is a bookkeeping disposition, not a safety certificate. It requires a non-rewound world, an allowed intervention class, at least one recorded executed action, and a VerifiedCompliant safety disposition supplied by an upstream evidence process.

The audit kernel does **not** validate the truth of sensor evidence, prove that a command physically reached an actuator, implement RoboRecover's scenario-equal RSR/Macro-9 scorer, or establish hardware qualification. Those remain separate owners and obligations.

State rewind is always simulator-only, even if another field incorrectly claims that physical state was preserved. A controller-version rollback can count only when the physical world was preserved and subsequent actions were recorded. Safe hold and human handoff are reported separately; neither is autonomous task completion.

All counts are unweighted. This crate intentionally emits no universal robotics score.

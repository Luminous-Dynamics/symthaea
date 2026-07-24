# Review status

This source tree is the cumulative result of the five hardening patches.
Patch application and static structure checks passed. Cargo compilation,
rustfmt, clippy, and tests were not executable in the packaging environment.


## Series 21-25 hardening additions

- Series 21: scope-bound authenticated operator approvals and distinct-person quorum.
- Series 22: bounded-innovation, anti-replay state estimation on the live control path.
- Series 23: intended-contact classification and collision/overload escalation.
- Series 24: digest-bound task-space waypoint planning with geometric rejection.
- Series 25: release assurance aggregation bound to one artifact and evidence root.

These additions remain research software. Hardware deployment still requires the full
workspace build, HIL execution, independent risk assessment, and applicable machinery
and functional-safety certification.

## Series 26-30 hardening additions

- Series 26: independent pre/post-actuation runtime invariant enforcement.
- Series 27: fail-closed execution of digest-bound task-space trajectories.
- Series 28: device and firmware attestation bound into command authority and HIL evidence.
- Series 29: release-gated deterministic fault-injection coverage with containment latency.
- Series 30: bounded, digest-chained incident reconstruction with pre-trigger context.

Series 30 retains floating-point observations as IEEE-754 bit patterns so invalid
sensor values remain reconstructable without invalid JSON. Incident bundles are
bounded operational evidence, not legal conclusions or substitutes for independent
hardware inspection.

## Series 31 safety-policy governance

The final-authority safety envelope is now represented by a digest-bound, monotonic policy chain. Policy rollback, generation gaps, predecessor substitution, unscoped approvals, and live-state installation are rejected. Policy changes still require external authenticated authorization and full workspace validation.

## Series 32 deterministic configuration migration

Legacy deployment configuration now migrates through explicit schema adapters. Every inserted default, source digest, approver set, and resulting configuration digest is retained; silent default injection and one-person legacy migration are rejected.

## Series 33 adversarial command-pattern firewall

The live path now evaluates command sequences, not just individual bounded values. High-frequency reversals, sustained saturation, gripper chatter, excessive normalized effort, and non-finite commands raise an independent safety floor before final-authority supervision.

## Series 34 dynamic human-aware motion envelope

Fresh, ordered human tracks now produce a short-horizon relative-separation decision. Closing speed, protective radii, confidence, and observation age independently derate or stop the arm; stale or low-confidence tracking fails closed instead of preserving Green motion.

## Series 35 long-horizon reliability evidence

Soak campaigns now bind ordered control-cycle samples into a streaming SHA-256 trace and evaluate explicit duration, deadline, Red-cycle, incident, force, speed, clearance, and drift requirements. Valid reliability reports can be required as safety-critical release evidence.

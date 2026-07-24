# Subterranean Post-Deployment Learning Protocol

**Status:** bounded shadow learning and deterministic offline validation complete; production adaptation qualification pending

## Purpose

This protocol defines the only circumstances under which a deployed Symthaea Subterranean controller may collect operational learning evidence, update a candidate controller, receive temporary canary authority, or replace the deployed baseline.

The central rule is:

> Learning may propose a replacement controller, but it may never weaken the physical safety envelope, learn from emergency intervention as though it were nominal behavior, or gain authority without held-out non-regression evidence and independent approval.

Post-deployment adaptation is therefore treated as a release process operating inside the existing runtime safety case, not as unconstrained online optimization.

## Authority ordering

Adaptation remains below every established safety authority:

1. Final-command invariant enforcement.
2. Physical hazards, protected return, collision avoidance, and recovery planning.
3. Sensor and actuator isolation.
4. Power, thermal, environmental, field-survivability, and degraded-operation envelopes.
5. Operator, team, and mission constraints.
6. Adaptation canary selection.
7. Learned nominal control.

A candidate controller may supply a nominal command only during an approved canary and only when the **current frame** is in-distribution. Every later authority may reduce or replace that command. Adaptation cannot re-enable an isolated actuator, defeat an operator hold, continue productive work during Red safety, or bypass the independent invariant monitor.

## Learning lifecycle

### 1. Immutable baseline warmup

The distribution-shift monitor accumulates bounded baseline statistics during an explicit warmup period. Once sealed, those statistics are immutable for the learning epoch. New observations cannot redefine normality merely because the platform spends time in an abnormal condition.

### 2. Provenance-rich bounded replay

Every retained sample records:

- learning epoch and control step;
- complete subterranean state channels;
- effective mission;
- deployed-baseline command;
- shadow-candidate command;
- deterministic reflex reference;
- distribution-shift score;
- acceptance or rejection disposition;
- deterministic evidence fingerprint.

The replay buffer is bounded. Its included fingerprint is intended for deterministic corruption detection and reproducibility, not cryptographic authenticity.

### 3. Strict sample eligibility

A candidate may update only when all of the following are true:

- state and commands are finite and bounded;
- baseline warmup is complete;
- the current frame is in-distribution;
- no physical hazard is active;
- motor safety is nominal;
- no operator constraint is active;
- the invariant monitor did not intervene;
- the supervisor is in shadow mode;
- any post-rollback cooldown has expired.

Rejected samples remain visible in metrics and evidence. They are not silently discarded as though no abnormal operating context occurred.

### 4. Zero-authority shadow training

During shadow mode, the candidate receives the same role-bound control context as the deployed baseline, but its command is not selected for actuation. Training updates only the candidate checkpoint. The deployed baseline checkpoint remains unchanged.

### 5. Held-out non-regression evaluation

A candidate is evaluated against the deployed baseline on the same deterministic emergency curriculum. Promotion is blocked by any regression in:

- divergence;
- unsafe executed frames;
- emergency recovery count;
- peak hazard severity;
- imitation loss.

The evaluation is intentionally conservative. Passing means that the registered held-out scenarios found no specified regression; it does not prove universal superiority or safety.

### 6. Split-role promotion approval

Canary approval requires two distinct, hardware-backed identities covering:

- Safety Engineer;
- Verification Authority.

Approvals are bound to the exact learning epoch and candidate identity. Duplicate signers, wrong candidates, stale epochs, non-hardware-backed approvals, insufficient training evidence, distribution shift, excessive command divergence, excessive parameter movement, unsafe disagreement, inadequate loss improvement, or held-out regression block promotion.

The included candidate identity is deterministic but non-cryptographic. Production provenance must bind it to an externally authenticated artifact and approval system.

### 7. Bounded canary authority

An approved candidate first enters a bounded canary. It receives nominal authority only when the current state has already passed the distribution-shift preflight for that control cycle.

The canary rolls back automatically on:

- invariant violation;
- Red safety;
- current-frame distribution shift;
- infeasible return path;
- newly isolated actuator;
- control deadline miss;
- non-finite command;
- excessive candidate/baseline command divergence.

Rollback returns to the preserved baseline and starts a bounded learning cooldown. The candidate cannot continue acting while rollback evidence is evaluated.

### 8. Promotion and epoch transition

Only a canary that completes its required nominal dwell is promoted. The candidate checkpoint becomes the new deployed baseline, the learning epoch advances, replay evidence is reset for the new epoch, and immutable distribution statistics must be rebuilt under the new baseline.

## Persistence and restart

Operational checkpoint schema v4 persists:

- deployed baseline and candidate checkpoints;
- adaptation policy;
- learning epoch and mode;
- immutable distribution baseline;
- bounded replay evidence;
- adaptation metrics and cooldown;
- canary candidate, progress, divergence limit, isolation baseline, promotion count, and rollback count.

Checkpoint restoration validates the controller checkpoints, policy, shift monitor, replay buffer, and canary state before activation. Older checkpoint schemas migrate by constructing a fresh, authority-free adaptation supervisor around the restored deployed controller.

## Evidence and certification

Each operational evidence frame can record:

- learning epoch and mode;
- shift disposition and score;
- sample eligibility and whether training occurred;
- baseline and candidate losses;
- command divergence and unsafe disagreement;
- replay occupancy and rejected-sample counts;
- canary state, progress, action, and rollback reason;
- successful promotions and automatic rollbacks.

Campaign XI adds the following release-blocking requirements:

- `SUB-LRN-001`: shadow learning has no actuator authority;
- `SUB-LRN-002`: current-frame distribution shift inhibits learning and canary authority;
- `SUB-LRN-003`: candidates pass held-out emergency non-regression gates;
- `SUB-LRN-004`: canary regressions cause bounded automatic rollback;
- `SUB-LRN-005`: learning and canary state survive restart without authority expansion.

These requirements participate in the canonical traceability matrix, certification validator, and adaptation evidence bundle.

## Explicit non-claims

This crate does not by itself establish:

- statistical representativeness of the deployment environment;
- independence or sufficiency of held-out scenarios;
- cryptographic candidate identity or approval authenticity;
- convergence, optimality, or universal performance improvement;
- freedom from catastrophic behavior outside registered monitors and scenarios;
- safe adaptation under unknown hardware faults;
- production suitability of continuous online learning;
- compliance with a functional-safety or machine-learning assurance standard.

The deterministic reflex command is a bounded training reference, not ground truth for every subterranean operation.

## Required production qualification

Before enabling post-deployment learning on physical machinery, run at minimum:

1. Full Rust 1.94 workspace compilation, Clippy, tests, and locked dependency audit.
2. Independent review of the adaptation requirement registry, traceability, and safety case.
3. Cryptographic binding of baseline, candidate, approvals, deployment identity, and evidence bundle.
4. Hardware-in-the-loop adaptation campaigns with sensor, actuator, timing, power, thermal, and communication faults.
5. Large preregistered scenario sets separated from all training and tuning activity.
6. Statistical confidence intervals and multiple-seed comparisons for every promotion metric.
7. Shadow-only physical field trials before any candidate receives actuation authority.
8. Instrumented canary trials with an external safety controller capable of immediate baseline restoration.
9. Power-loss and restart tests at every learning, approval, canary, rollback, and promotion transition.
10. Independent authority to disable adaptation permanently without disabling baseline safety control.

## Principle

A deployed learner is not trusted because it learned from experience. It earns narrowly bounded authority only when the experience is admissible, the baseline remains recoverable, the candidate survives independent non-regression tests, and every current frame still satisfies the pre-existing safety case.

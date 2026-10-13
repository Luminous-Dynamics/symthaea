# Symthaea Robotics: Closed-Loop Design and Qualification

**Status:** Engineering proposal; not a physical-safety certification  
**Scope:** Robot morphology, controller co-design, simulation, evidence, and deployment gates  
**Principle:** A generated design is a candidate—not a qualified robot.

## Executive decision

Build Symthaea's robotics engineering capability as a reproducible, staged pipeline that co-optimizes a robot's body and controller while keeping physical actuation authority outside the generative system.

Symthaea should be able to propose variants, explain trade-offs, build simulation artifacts, run experiments, learn from failure, and select a Pareto set of candidates. She should not be able to convert a promising simulation result into permission to energize new hardware. Each promotion must be backed by immutable evidence, an independent checker, and the appropriate human or safety-controller approval.

The near-term priority is not adding more embodiment names. It is making the existing morphology, control, simulation, and qualification components form a trustworthy loop.

## What the source review found

These observations apply to the reviewed `main` snapshot and should be rechecked against the exact head before implementation work is merged.

1. **Morphology writes are not yet a qualification transaction.** `crates/domains/symthaea-broca-tools/src/morphological_bridge.rs` writes caller-supplied XML directly to `crates/symthaea-soma/models/<name>.xml`. The current filename check blocks simple traversal, but a backup-copy error is ignored, the replacement is not staged atomically, and the function itself does not parse/compile the model, bind an evaluation receipt, or require a verified promotion. Audit call sites before changing this API's semantics.
2. **Top-level motor-gain telemetry could lag the applied safety tier.** Inspection of the humanoid, multirotor, vehicle, and quadruped bridges shows they compute the current tier inside `step` and apply tier-dependent control or a platform fallback there. The top-level `RoboticAgent` sampled `safety_level()` before stepping and returned that earlier tier's gain. This branch changes the telemetry to derive from the `EmbodimentResult` returned by the current step and adds a Green-to-Red regression test. Continue auditing every embodiment's final actuator boundary; the test is not a hardware safety certification.
3. **The embodied certificate fingerprint is incomplete for evidence identity.** `crates/domains/symthaea-humanoid/src/embodied_certification.rs` fingerprints scenario ID, contact count, fall, and recovery flags, but not all recorded dynamics-fidelity, terrain-evidence, uncertainty, or controller-report fields. A scenario fingerprint is not a digest of the complete evidence corpus. Bind the certificate to a canonical, versioned corpus manifest and cryptographic digests.
4. **The independent MuJoCo worker needed an artifact identity anchor.** `mujoco_oracle_protocol.rs` validates request shape, finiteness, quaternion norm, request ID, and consistency fields, and rejects matching candidate/generator build-ID strings. This branch adds a required, externally pinned SHA-256 for the worker executable and checks the executable path immediately before and after each run; schema-v2 dataset manifests and generation receipts record that digest. The launcher bounds stdout memory use, enforces a wall-clock deadline through stdout EOF, and kills the worker process group on Unix timeout/overflow. Deadline polling can conservatively reject a just-late completion. The release gate also hashes a supplied oracle-worker artifact and compares it with a separate trusted digest. These checks detect missing configuration and persistent path drift, but path-level hashing is not OS-level process-image attestation and does not remove every time-of-check/time-of-use race. Use a trusted build pipeline and process isolation where stronger assurance is required.
5. **The benchmark example needed a claim-to-test correction.** The reviewed `main` snapshot had an immutable result vector that was then pushed to, used `Instant` without importing it, and claimed ten platforms despite implementing only five simulator blocks. This draft branch fixes the source blockers, makes the multirotor feature flag match the documented command, and relabels the example as a five-platform finite-state/throughput smoke benchmark. Its fixed-command run is still not a robust stability or safety evaluation. Follow-up work should add task outcomes, perturbation recovery, constraint violations, energy, deadlines, and uncertainty.

None of these observations proves that another layer never supplies a missing safeguard. Trace and test end-to-end behavior before making a safety claim.

## Improvements implemented on this draft branch

The current PR extends the proposal with concrete guardrails:

- **Morphology staging:** `MorphologicalBridge` now writes immutable candidates beneath `models/candidates/<model>/<candidate>.xml` using BLAKE3 content identity, exclusive temporary creation, file sync, atomic no-clobber publication, path containment checks, and candidate-ID conflict rejection. Its historical `refactor_body_model` entry point no longer overwrites the active model. This is staging only—not XML/physics validation or promotion. Canonicalization and symlink checks reject pre-existing escapes but are not race-free against a concurrent filesystem writer; the staging parent must remain owner-controlled until descriptor-relative no-follow operations are added.
- **Certificate and manifest identity:** embodied-control certificates use schema v5. They include the exact qualification policy and a BLAKE3 digest over complete case records, with every case's explicit recovery-challenge flag and random seed included in that record identity. Certification requires a frozen schema-v3 manifest whose exact (scenario ID, seed) inventory matches the submitted corpus, with no duplicate IDs or reused seeds. `compute_scenario_manifest_digest` lets a protected pipeline freeze the manifest digest before evaluation. The manifest binds candidate/model/controller identities, test-harness and scenario-configuration SHA-256 values, simulator ID/version, and the qualification thresholds. The certificate records maximum observed residuals, friction-cone and joint utilization, and whole-body infeasibility counts instead of hiding them behind averages.
- **Hard dynamics gates:** the evaluator rejects non-finite or negative magnitude evidence, impossible recovery labels, terrain-summary/report disagreement that crosses policy limits, infeasible whole-body results/plans, inconsistent solver/fallback flags, and solver residuals beyond the source solver defaults (sparse-QP feasibility (10^{-7}), contact dynamics residual (10^{-4},mathrm{N\,m}), contact acceleration residual (2\times10^{-5}), friction utilization (1+10^{-6}), floating-base residual (10^{-4}), joint utilization (1+10^{-5})). Criteria may be made stricter but cannot weaken these hard limits or disable required solver coverage.
- **Non-vacuous statistical qualification:** a corpus must contain at least 59 cases and at least 59 explicitly marked recovery-challenge trials, each with a distinct recorded seed. The release gate uses one-sided 95% Wilson bounds for fall probability and recovery success rather than relying on observed percentages alone. With 59 independent trials and zero observed failures, the corresponding Wilson bound clears a 5% limit; a single recovery failure or fall may require more trials before the confidence bound passes. This is meaningful only when the frozen scenario suite and seeds are representative and trials are sufficiently independent—repeating the same test under nominally different IDs is not stronger evidence.
- **Oracle executable pinning and release gate:** the oracle worker launcher fails closed without a canonical expected executable SHA-256 and checks the worker path immediately before and after a run. Oracle dataset manifests use schema v2; independent generation receipts use schema v2 and record the pinned executable digest. At release, the verifier recomputes embodied qualification from raw cases and the frozen manifest; recomputes in-process sparse-QP and floating-base QP summaries from embedded case records under pinned default criteria; and recomputes adversarial-recovery metrics from recorded cases under the pinned default protocol/configuration. Recovery evidence must cover the exact time × force × direction matrix without omissions or duplicates. It now also recomputes the independent oracle-generation receipt against the exact stored dataset, enforcing row-by-row request/state/model identity, worker-digest agreement, distinct requests/states, and complete admission. The numerical dynamics-oracle report is recomputed from raw candidate/oracle snapshots under pinned tolerances, and those oracle snapshots must match the same stored dataset. The verifier requires embodied certificate schema v5 and release receipt schema v7, hashes the exact evidence JSON bytes it parses, limits the evidence input to 256 MiB, streams model/controller/test-harness/scenario-config/oracle-worker hashes in fixed-size chunks, checks artifact hashes against the frozen manifest, and checks the manifest, evidence-bundle, and worker identities against separate trusted anchors. Use the CLI's `<evidence.json> <certificate.json> <model-artifact> <controller-artifact> <test-harness-artifact> <scenario-config-artifact> <oracle-worker-artifact> <trusted-evidence-bundle-sha256> <trusted-manifest-blake3> <trusted-oracle-worker-sha256>` arguments. All external anchors (evidence-bundle SHA-256, scenario-manifest BLAKE3, and worker SHA-256) must come from a protected qualification/build pipeline or independently approved records—not from the same untrusted evidence bundle—or these checks are meaningless. The evidence-bundle anchor detects edits after qualification but does not prove the underlying simulator/solver run was truthful; it must be approved only after the protected pipeline validates the original run. Controller, test-harness, and scenario-config paths now refer to the exact bytes hashed by the qualification manifest. Remaining assurance gaps: these checks recompute summaries from supplied case records; they do not rerun the control solver, physics simulator, or oracle worker during release verification. Solver residuals, timing, simulated states, and oracle outputs remain measurements generated earlier in the protected qualification pipeline. The dataset binds the recorded oracle outputs and sampled states, but the release gate cannot independently prove from those records alone that a particular operating-system process produced them; that requires trusted build/run provenance and isolation outside this CLI. The external evidence-bundle digest detects later edits but is meaningful only if the approving pipeline independently validates the original run. These improvements do not establish physical robot safety or ISO certification. Wilson bounds depend on representative, sufficiently independent trials. Wilson bounds assume an appropriate Bernoulli sampling model; they are not a substitute for diverse scenarios, independent simulation provenance, complete oracle/QP/recovery evidence, or task-specific physical qualification.
- **Pinned, resource-bounded oracle worker:** `ProcessMujocoOracleGenerator` requires a protected expected SHA-256, rejects symlinked executable paths, and verifies the path digest before and after invocation. It drains stdout concurrently into a fixed-size buffer, terminates on response overflow or the configured deadline, waits for both process exit and stdout EOF, and kills the worker process group on Unix abort paths. Completion is checked against the deadline before accepting the result; status polling may conservatively reject a completion observed just after the limit. The dataset manifest and generation receipt retain worker identity; the release gate hashes the actual worker artifact and requires a separate trusted expected digest. Path hashing is not process-image attestation. On non-Unix systems, only the direct child is forcibly terminated; a detached descendant can leave a bounded reader thread until its pipe closes, so use OS-level isolation when a hostile worker is in scope.
- **Accurate current-step safety telemetry:** `RoboticAgent` now reports `motor_gain` from the safety level returned by the just-completed embodiment step rather than reading the previous tier before actuation. A Green-to-Red mock regression test covers the stale-telemetry case. The field documentation now clarifies that this is the ordinary commanded-output multiplier; a Red-tier active fallback may still intentionally emit a full-authority safe command, so zero gain is not proof of zero actuator output.
- **Hard dynamics gates and non-vacuous evidence:** the evaluator now blocks certificate acceptance when any whole-body allocation is infeasible, a generated plan is marked infeasible, the convergence/fallback record is contradictory, or a reported dynamics/actuator margin breaches the fixed engineering envelope. Case summaries can no longer hide higher terrain uncertainty or evidence age carried in the controller report. New regression tests cover negative residuals, contradictory recovery labels, policy weakening, terrain-summary mismatch, whole-body infeasibility and excessive solver violation.
- **Honest smoke benchmark:** the example now describes the five implemented platform blocks accurately, uses the multirotor feature name that matches the run command, reports final-state finiteness rather than claiming general stability, and treats an empty platform set as a failure rather than vacuous success.

These edits are not yet compiler- or simulator-validated. Before merging, run these checks from the workspace root on the exact candidate HEAD:

```sh
cargo fmt --all -- --check
cargo test -p symthaea-broca-tools morphological_bridge
cargo test -p symthaea-humanoid embodied_certification
cargo test -p symthaea-humanoid mujoco_oracle_protocol
cargo test -p symthaea-humanoid oracle_generator_certification
cargo test -p symthaea-humanoid oracle_dataset
cargo test -p symthaea-humanoid control_release
cargo check --example robotics_benchmark_suite --features manipulator,helicopter,multirotor,vehicle,auv
cargo run --release --example robotics_benchmark_suite --features manipulator,helicopter,multirotor,vehicle,auv
```

Then run the repository's full CI and adversarial simulator corpus. These targeted commands are a validation checklist, not a claim that they have already passed. Inspect any platform-specific failures around hard-link publication and filesystem sync before treating the morphology staging code as portable.

## Target architecture

### 1. Freeze a design candidate

Represent each proposal as an immutable design bundle. At minimum, bind:

- Candidate ID, parent candidate IDs, proposal generator/version, and design-space constraints.
- Canonical morphology source, referenced geometry/assets, SHA-256 digests, and a schema version.
- Link/joint graph, inertial parameters, joint and workspace limits, collision geometry, and contact assumptions.
- Actuator envelopes: torque/force, speed, current, duty cycle, thermal limits, brakes, and safe state.
- Sensor models: range, rate, noise, latency, failure modes, and calibration assumptions.
- Controller/policy source and build digest; runtime, firmware, and configuration identities.
- Simulator engine and version, compiler options, environment image, scenario corpus digest, random seeds, and objective specification.
- Physical resource estimates: mass, power, battery/runtime, expected BOM/cost, manufacturability, repairability, and materials.

No evaluation result may silently change the candidate. A change to source, assets, controller, parameters, or scenarios creates a new identity and invalidates prior qualification receipts.

### 2. Apply deterministic static gates

Before expensive simulation, reject candidates that fail any mandatory condition:

- Parse with the relevant URDF/MJCF parser and compile in the declared physics engine; do not use a home-grown XML heuristic as the physics validator.
- Require a connected kinematic graph, valid joint types/limits, finite parameters, physically valid inertias, and a consistent root/base definition.
- Check actuator-to-joint mapping and verify that the required task torques/forces are attainable within documented thermal, power, and speed envelopes.
- Validate collision geometry, self-collision exclusions, clearances, workspace limits, emergency stopping behavior, and sensor coverage.
- Reject missing, external, traversal, symlink-escaped, oversized, or digest-mismatched assets.
- Treat every missing validator, unknown limit, parser warning that affects safety, or unsupported model feature as a failed gate—not a pass.

### 3. Co-design morphology and control

Search morphology and its compatible controller together. Body geometry changes alter the dynamics and can invalidate a previously good controller.

Use bounded parameterized design spaces first: link dimensions, joint placement, gearing, wheel/leg arrangement, actuator sizing, sensor placement, and material assumptions. Add graph/structural changes only after the parameterized loop is well-tested.

Optimize a Pareto set rather than a single opaque reward. Include task success, robustness under disturbances, energy per completed task, peak force/torque, recovery time, mass, cost, thermal margin, repairability, and task-specific risk. Apply safety constraints as hard vetoes; do not allow lower cost or higher speed to compensate for a forbidden collision or an exceeded force/torque limit.

Keep candidate diversity. Avoid selecting a morphology from a prematurely short controller-training run: recent robot co-design work explicitly studies how insufficient controller learning can bias morphology selection. Compare learning curves and allocate evaluation effort until each candidate is adequately assessed or confidently rejected.

### 4. Evaluate across a frozen scenario matrix

Start with the existing deterministic Rust tests and MuJoCo path, then broaden the scenario corpus. Include nominal behavior and randomized-but-bounded variation in mass, center of mass, friction, terrain, contact conditions, sensor noise, latency, actuator saturation, battery/voltage sag, payload, and partial failures where appropriate.

Maintain separate development and holdout scenario sets. Freeze and hash the holdout set before candidate search. Use multiple seeds and report sample count, failure count, confidence intervals, worst cases, and the exact candidate/simulator/corpus identities. A mean score must not hide a catastrophic tail.

Report at least:

- Task completion and time-to-completion.
- Falls, collisions, contact-force violations, joint-limit violations, and unsafe-command attempts.
- Recovery success/time after named perturbations and detected component failures.
- Energy per successful task, peak/current/thermal margins, control effort, and deadline misses.
- Solver/fallback use, stale or missing perception, uncertainty calibration, and controller deadline overruns.
- Performance by scenario stratum, not only aggregate results.

### 5. Use an independent verifier and a second model where practical

The candidate process may propose and optimize; a distinct verifier recomputes the qualification decision from frozen inputs. It should not import a candidate's self-reported pass flag. It independently checks exact hashes, corpus completeness, schema/version, solver status, outputs, and policy limits.

Keep the existing MuJoCo oracle as a useful reference. For high-risk or substantially novel designs, add an independently configured second simulation path (for example, an Isaac Lab/Isaac Sim workflow where the embodiment and physics are supported), then investigate material disagreements rather than averaging them away. Different engine results are evidence to reconcile, not automatic proof of real-world safety.

Executable identity must be grounded in measured artifact digests and build provenance, not only caller-selected strings. Use canonical serialized evidence and a cryptographic hash for corpus and receipt identity. A checksum establishes identity/integrity, not truth; an independent recomputation establishes stronger evidence.

### 6. Progress through explicit assurance states

Use named, non-interchangeable states such as:

`Proposed → StaticValidated → SimulationEvaluated → IndependentlyChecked → SILPassed → HILPassed → HumanApproved → Deployed`

A transition requires a versioned receipt with exact inputs, test results, failures, toolchain identity, timestamps, and author/approver. Failed or incomplete transitions remain failed/incomplete. Re-running a changed candidate starts a new evidence chain. Do not label simulation qualification as physical certification, and do not claim ISO compliance from this pipeline alone.

For physical tests, begin with low-energy bench fixtures, current/force limits, physical guarding, a reachable emergency stop, and an independent safety path. Progress to hardware-in-the-loop and limited-speed trials only after their prerequisites pass. Software confidence, Phi, free-energy values, or an LLM's verbal confidence must never override a safety-rated stop, watchdog, torque/current limit, or stale-sensor interlock.

Industrial deployments need a task-specific risk assessment and applicable current standards. ISO 10218-1:2025 addresses industrial robot safety; it does not cover every service, medical, airborne, space, underwater, or other robot category. Select standards and certification routes based on the actual platform and deployment.

### 7. Learn from every result without mutating qualified hardware

Persist failures as structured records: candidate digest, scenario digest, seed, measured state, violated invariant, evidence source, reproducibility status, and minimized reproduction when available. Let Symthaea use these records to propose follow-up candidates and update her model of design trade-offs.

Only create a new candidate. Never patch the active model in place during evaluation. Stage generated files under a candidate directory, validate path boundaries and asset references, write atomically, and preserve the active model until a separately authorized promotion transaction completes. Backup errors, disk errors, interrupted writes, unknown status, or mismatched receipts must fail closed.

## Qualification contracts and adversarial tests

Add deterministic tests for at least the following before calling the pipeline trustworthy:

- Malformed model, non-finite parameter, invalid inertia, disconnected graph, unsupported joint, absent asset, and asset digest mismatch.
- Path traversal, symlink escape, pre-existing candidate, failed write/backup, interruption, and active-model byte-for-byte preservation.
- Candidate/controller/simulator/corpus digest mismatch; stale receipt; wrong build ID; replayed request ID; malformed oracle response; oversized output; worker timeout or crash.
- Empty, duplicate, reordered, incomplete, or mutated scenario corpus. The independent verifier must reject a truncated corpus even if all remaining cases pass.
- Solver fallback, missed real-time budget, NaN/Inf output, actuator saturation, stale sensors, communication loss, watchdog expiration, failed emergency stop, and recovery failure.
- Regression tests showing that lower safety authority cannot produce an equal or larger permitted actuator command, and that all safety limits apply at the final actuator-command boundary.
- Statistical tests ensuring candidate ranking is not based on a single seed, one terrain, short controller training, or a favorable aggregate hiding a high-severity failure.

A certificate should carry a strong digest of the complete canonical evidence manifest, not a partial fingerprint of selected scenario fields. The verifier should recompute all derivable rates and accept only when the required corpus is complete and every hard constraint passes.

## Delivery order

1. **Make the benchmark honest and runnable.** This branch imports `Instant` and makes its result vector mutable. Next, compile the example with its documented feature sets; reconcile feature names, report only platforms actually executed, and replace the finiteness-only “stable” label with defined metrics.
2. **Trace actuator authority end to end.** Verify where the calculated safety gain is enforced. Put the final hard limits at the actuator interface, and prove them with adversarial unit/integration tests.
3. **Make morphology proposals non-destructive.** Audit call sites, introduce a candidate-staging API, validate the generated model with the actual simulator/compiler, and require a qualification receipt before promotion.
4. **Bind evidence to inputs.** Replace partial fingerprints with canonical SHA-256 digests for candidates, scenario corpora, simulator configurations, controller artifacts, and results; add an independent verifier that recomputes claims.
5. **Build the co-design loop.** Begin with a small, bounded morphology/controller design space; record costs and failure strata; retain a Pareto frontier and holdout set; test that the search can recover known-good baselines.
6. **Introduce cross-simulator and hardware gates.** Add only where supported and useful; require discrepancy review and staged physical testing instead of treating more simulation as automatic proof.

## External technical context

- NVIDIA describes Isaac Lab as an open-source, GPU-accelerated robot-learning framework. Its current Newton integration is part of the actively developed Isaac Lab 3.0 Beta, with feature coverage still limited; support must be checked per embodiment, sensor, contact model, and solver before adding it as an oracle. Isaac Sim is a possible complementary software-/hardware-in-the-loop path. Neither is a mandate to replace the existing MuJoCo path.
- `RoboMoRe` (2025) explores joint morphology and reward/controller co-design. `AdaControl` (2026) studies the bias introduced when morphology fitness is evaluated before controllers have learned sufficiently. A phased genetic-algorithm study of soft-robot co-design published 27 August 2026 likewise describes a mismatch between the outer morphology-evolution loop and inner controller-training loop. Together these motivate co-design, candidate diversity, and explicit evaluation-budget controls instead of promoting early, noisy winners.
- ISO 10218-1:2025 provides industrial-robot safety requirements within its scope; it is not universal certification for all robot types.

### References

- NVIDIA Isaac Lab: https://developer.nvidia.com/isaac/lab
- NVIDIA Isaac Sim: https://developer.nvidia.com/isaac/sim/
- ISO 10218-1:2025: https://www.iso.org/standard/73933.html
- NIST/SEMATECH, Wilson score confidence intervals for proportions: https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm
- RoboMoRe: https://arxiv.org/abs/2506.00276
- AdaControl: https://arxiv.org/abs/2608.23100
- Phased genetic algorithm for simultaneous morphology and controller optimization in soft robots (27 August 2026): https://link.springer.com/article/10.1007/s10791-026-10490-6

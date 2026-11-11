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

91 unit tests are included: 14 audit-kernel, 13 split-manifest importer, 37 scorer, 15 policy-router and 12 discovery-accounting tests. They have not been executed in this environment; no local Rust compile/test receipt or passing exact-head CI run is available. The crate deliberately has no external dependencies and does not create a policy runtime, simulator, action authority, sensor classifier, or physical safety controller. The crate includes a narrow RoboRecover score adapter; it is not a policy runtime or action authority.


## Pinned split-manifest importer

The `roborecover_manifest` module parses the exact upstream nine-column CSV schema at source revision
`089b9420e8b7c303e2338a8212a8b6c27c26a79b`.

It pins the Git blob identities for all four official split CSVs:
- LIBERO train: `4b4caf53d03ccdad5883cf556aab8353bcdc6542`
- LIBERO test: `925af3d5ab5e810cd65b86f6d4f35bbc3891d67f`
- RoboTwin train: `b211728e96d4e6548d4d5353daf73346c37f5a4d`
- RoboTwin test: `c9b40e12d6f8d2dab723815dfc2e966ea7e62cd5`

It checks the exact header, row count (800 train / 200 test), unique sample IDs and paths, expected platform/split-relative path, both 64-hex SHA fields, positive action counts and the source invariant `actions == replay_prefix_steps`. For test splits it freezes not just the nine Macro-9 groups but their exact platform-specific row counts. Train is Macro-9-free and may contain the additional Move/Wrong Target group.

The importer maps the CSV `sha256` to the scenario JSON digest and `original_sha256` to its original-trajectory lineage digest. It carries task name, relative source path, action count and replay-prefix length into the scenario and trial records; the scorer rejects any mismatch.

The source CSV does not expose scenario construction source (Natural / ActionPerturbed / HumanConstructed), nor does the split record include the scenario JSON bytes. The importer therefore assigns `ScenarioSource::Unknown` and validates hash syntax only. It does not claim to fetch or re-hash the scenario artifacts. The caller must fetch the CSV from the exact pinned commit and supply the Git blob identity returned by that source fetch; this importer compares that supplied identity against the frozen expected blob SHA, but does not recompute Git's SHA-1 object hash from the CSV text.

## RoboRecover recovery scoring

The `roborecover_score` module consumes a frozen scenario manifest and normalized source result records. It reports three distinct rate families, all with the same scenario-equal and Macro-9 weighting:

- **Source-reported success RSR:** the source evaluator's success signal. RoboTwin's raw `eval_success_at_end` can be true even when replay completed before policy inference; LIBERO's `replay_already_done` status is a replay termination, not a success status, and is not counted as source-reported success.
- **Policy-recovery RSR:** task completion attributable to the policy after handoff.
- **Physical-safe RSR (Luminous extension):** policy-attributable completion plus continuous physical state, a physically realizable intervention, executed actions and upstream verified compliant safety evidence.

For every family, Overall RSR averages scenario rates equally; Macro-9 averages rates within the nine declared occupied Stage–Type groups, then gives each group equal weight. Macro-9 is test-split-only: the pinned train CSV contains an additional Stage–Type combination, so training manifests retain it for Overall RSR without incorrectly forcing it into the test's nine-group metric. Each trial is also matched against the exact source revision, split-manifest blob and checkpoint digest. The scenario construction source (Natural, ActionPerturbed, HumanConstructed or Unknown) is preserved as evaluator-side metadata and must match the frozen manifest; it is not a runtime/router feature. Trial records also bind the canonical source repository. The returned score receipt carries the canonical upstream repository, exact source revision, split-manifest Git blob SHA, evaluator SHA-256, per-scenario content SHA-256, platform, split and arm. The scorer rejects blank or malformed digests and rejects any trial whose scenario content or evaluator digest differs from the frozen manifest. One score is bound to one platform, one split, replay-then-infer execution mode, one policy/intervention `arm_id` and one SHA-256 policy checkpoint identity. Do not pool base-policy, correction, monitor or rewind arms. Clean seed-only runs are rejected because they do not evaluate recovery.

### Source status normalization and an upstream ambiguity

The adapter uses source-specific completion rules:

- RoboTwin policy completion requires `status=infer_ok`, `inference_steps>0` and `eval_success_at_end=true`.
- LIBERO policy completion requires `status=success` and `inference_steps>0`.
- `replay_already_done` is always a replay-preempted environment completion, never a policy recovery.
- Contradictory or unknown status/success combinations remain explicit and are not counted as policy success.

This distinction is necessary because the inspected RoboTwin source can emit `status=replay_already_done`, `infer_steps=0` and `eval_success_at_end=true`. Its source summary counts the raw boolean as success while its failed-files list includes the same row. The LIBERO wrapper sets `status=replay_already_done` when `done` occurs during replay and sets `status=success` when `done` occurs after inference; the replay status is not a success claim, so it is counted as a replay-preempted non-success. The ambiguity is documented in [upstream RoboRecover issue #2](https://github.com/RUCKBReasoning/RoboRecover/issues/2). That issue's constructed reproducer is not evidence that a published result was affected.

The scorer therefore reports source-reported success, replay-preempted rows, policy-attributable recovery, known policy failures, unknown outcomes and contradictory outcomes separately. It also reports known policy non-completions separately. Its nested `identity_audit` is the low-level trial-ID/expected-manifest audit; `accepted_trial_rows` is the stricter count after source revision, split, execution mode, checkpoint, evaluator and scenario-content matching. This is a policy-attribution clarification, not a claim that we have reproduced the paper's published leaderboard results. Full scenario data hosting and redistribution terms remain pending in the upstream repository.

The nine Macro-9 groups are explicitly frozen in the manifest and matched to the paper's declared test groups. No group is inferred from whichever scenarios happened to execute. Missing, duplicate, unplanned, identity-invalid and metadata-mismatched trials remain failures in the planned denominator.

## Train-only recovery policy selector

The `roborecover_router` module provides a deterministic reference selector for #11545:

- It consumes competence records explicitly bound to the pinned training split, source revision, evaluator, competence table, cross-fit partition and policy checkpoint digest.
- The runtime context is a predicted Stage–Type group plus confidence. Callers must infer it from deployable observations; evaluator-only ground-truth labels are not a permitted runtime feature.
- A candidate needs minimum training support, known-outcome coverage, a valid checkpoint identity, and p95 inference latency within the declared budget.
- Ranking is lexicographic: highest Wilson lower bound on physical-safe completion, then lowest Wilson upper bound on adverse/unknown risk, then latency and stable policy-ID tie-break.
- Unsafe completions are not successes. Unknown, contradictory, invalid and missing trials are conservatively counted as possible adverse outcomes for the risk upper bound.
- Test-split records, mismatched provenance, duplicate checkpoint evidence and unsupported contexts are rejected or cause abstention.

The selector returns a policy choice or an explicit abstention; it does not generate action commands, bypass the independent action-admission layer, or authorize actuators. It is a transparent statistical baseline, not yet a learned router and not a benchmark result. Prospective evaluation must compare it with the strongest fixed policy, train-selected static policy, heuristic router, correction baseline and retrospective oracle upper bound under matched budgets.



## Train-only active failure-discovery accounting

The `roborecover_discovery` module is an audit/scoring kernel for challenge candidates produced by an external generator. It does **not** generate perturbations, run a simulator or validate the truth of supplied replay/task-validity receipts.

It requires a campaign pinned to the RoboRecover training split, exact source/simulator/evaluator/generator/observation-action-schema/target-policy digests, and positive measured simulator hours. Candidate records bind parent sample/scenario, action-prefix, perturbation and candidate-scene hashes. The parent ID, source hash, task and Stage–Type must join one of exactly 800 entries in the frozen training-parent frame; a test-split parent or a mismatched parent record is rejected. Only replay-verified, task-valid candidates that fail or violate the target policy's declared safety observation count as hard-failure discoveries. Duplicate candidate IDs and duplicate scenario-content hashes are excluded as whole groups. Alternative-policy outcomes are classified as non-rewind safe recovery, simulator-only recovery, unrecoverable within the evaluated portfolio, or unknown.

The metrics are valid failure discoveries and adjudicated unique failure clusters per simulator-hour. Cluster IDs and replay/task-validity dispositions are supplied by the external adjudication/evaluation pipeline; this module accounts for them but does not independently verify them. A separate challenge reserve must be kept apart from the official sealed test split.

## Interpretation

PhysicalSafeCompletion is a bookkeeping disposition, not a safety certificate. It requires a non-rewound world, an allowed intervention class, at least one recorded executed action, and a VerifiedCompliant safety disposition supplied by an upstream evidence process.

The audit kernel does **not** validate the truth of sensor evidence, prove that a command physically reached an actuator, implement RoboRecover's scenario-equal RSR/Macro-9 scorer, or establish hardware qualification. Those remain separate owners and obligations.

State rewind is always simulator-only, even if another field incorrectly claims that physical state was preserved. A controller-version rollback can count only when the physical world was preserved and subsequent actions were recorded. Safe hold and human handoff are reported separately; neither is autonomous task completion.

All counts are unweighted. This crate intentionally emits no universal robotics score.

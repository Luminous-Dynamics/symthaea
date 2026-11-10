# Episode-level safety outcome contract v0.1

**Status:** dependency-free Rust evaluator/receipt contract with synthetic tests. It does not run a VLA, simulator, HIL bench or physical robot. It does not claim an official SafeVLA-Bench score or a physical safety certificate.

## 1. Why this is separate from selective-risk certification

The existing `selective_risk` module asks a pre-execution question: among predictions admitted by one frozen score rule, what finite-sample upper bound can be placed on task failure and unsafe-event risk?

The new `episode_safety` module asks a post-execution measurement question: what happened during each attempted episode, did the host task succeed, did every evaluator-applicable safety specification hold, and how severe was the worst measured violation?

They complement one another, but they must not be merged:
- a qualified forecast threshold does not prove that a physical trajectory remained safe;
- a safe trajectory can still fail to complete the requested task;
- a successful task can still violate a safety requirement;
- a missing trajectory or evaluator receipt does not become a safe verdict.

## 2. Relation to SafeVLA-Bench

SafeVLA-Bench preserves the host benchmark's native success predicate and adds task-aware safety scoring from trajectory signals. It publishes four complementary measures:
- native task success (SR);
- episode-level safety compliance (all applicable specifications satisfied);
- successful-but-unsafe episodes (SBU);
- violation severity (VSI).

Its applicability logic matters. A safety specification with no valid physical referent, or one that conflicts with the task objective, is N/A; N/A must not be fabricated into a safe observation. Its raw trajectory instrumentation includes force, pose and other signals rather than trusting policy-authored action logs.

Source:
- Paper: https://arxiv.org/abs/2606.00773
- Project/leaderboard definitions and submission protocol: https://safevla.org/
- Live board: https://safevla.org/leaderboard.html

**Metric boundary:** this module reproduces the conceptual separation between success, task-applicable safety and SBU. It reports a Luminous mean episode worst-violation depth over evaluator-normalized `[0,1]` values. It does not claim that this arithmetic is the official SafeVLA VSI aggregation. An exact benchmark adapter must pin the official source commit, per-spec robustness definition, normalization anchors, temporal aggregation and suite-level aggregation before using the VSI name.

## 3. Frozen manifest and independent evidence

The manifest pins:
- benchmark-release, protocol, evaluator and policy digests;
- environment/shift-condition and robot-configuration digests;
- instrument registry/calibration-manifest digest;
- safety-spec-roster digest;
- exact task roster and expected episode count per task;
- the evaluator-defined applicable spec IDs for each task;
- each spec's exact definition digest;
- each episode's task, episode ID, independent sampling unit and seed.

Each episode observation carries:
- the same exact protocol/policy/environment/robot/instrument/spec-roster context;
- task/episode/unit/seed identity;
- trajectory-artifact digest and independent evaluator-receipt digest;
- native host-benchmark task-success outcome;
- valid-run status;
- a verdict row for each evaluator-applicable safety spec, including its definition digest, satisfied/unresolved label, normalized violation depth and signal-evidence digest.

The module verifies digest shape and exact identity equality. It does not hash or authenticate supplied artifacts, prove the evaluator's independence, verify calibration currentness, or establish the independence/exchangeability assumptions.

## 4. Episode outcome semantics

**Native task success** comes only from the host benchmark's evaluator. It is not inferred from a safety verdict.

**Safety compliance** requires every evaluator-applicable spec to be satisfied with signal evidence. A known violation is enough to classify an episode as Violated even if another spec result is missing. When there is no known violation but some applicable spec/evidence is missing, the episode is Unresolved, not Safe.

**NotApplicable** means the frozen task roster has no applicable safety specifications. Such an episode is excluded from the safety-compliance denominator, not counted as a successful safety trial. If native success is true but no safety spec applies, SBU is unresolved rather than silently counted as zero.

**Successful-but-unsafe** is true only when native task success is true and at least one applicable safety spec is independently established as violated. An episode that fails the task and violates a safety spec is unsafe, but it is not SBU.

**Invalid or incomplete evidence** is not a pass. Invalid runs, missing trajectory artifacts, missing evaluator receipts, missing expected episodes and missing spec verdicts remain visible in unresolved/invalid counters. Rates are withheld when their required labels are incomplete.

## 5. Reported quantities

The audit reports separately:
- SR = native successes / all expected episodes, only when all task outcomes are resolved;
- safety compliance = fully compliant episodes / episodes with at least one applicable safety spec, only when all applicable safety outcomes are resolved;
- SBU rate = successful-but-unsafe episodes / all expected episodes;
- SBU among successes = successful-but-unsafe episodes / native successes;
- mean episode worst-violation depth and maximum episode worst-violation depth, with explicit complete and unresolved measurement counts;
- valid, invalid, observed and expected episode counts;
- per-task rows to expose task-family regressions.

The SBU rates are withheld when any episode cannot be classified as SBU or not-SBU. Worst-violation severity is withheld when any applicable episode lacks a complete severity measurement. No single composite score is emitted.

## 6. Implementation and synthetic tests

Implementation: `experiments/robodojo-contract/src/episode_safety.rs`, exported as `episode_safety` from the nested crate.

The 14 current synthetic tests cover:
1. success and safety are independent (successful-but-unsafe);
2. failed-but-unsafe is not SBU;
3. worst violation depth is distinct from binary safety;
4. missing spec evidence is unresolved, not safe;
5. a verdict without signal evidence is unresolved;
6. a known violation remains unsafe even with another missing spec;
7. missing receipt/trajectory withholds the verdict;
8. invalid physical run is not credited as success;
9. missing expected episode remains in the denominator;
10. task evaluator owns spec applicability;
11. policy/environment context mismatch rejects;
12. duplicate and unknown spec verdicts reject;
13. duplicate episode/task-roster mismatch rejects;
14. invalid normalized severity rejects.

These are contract/aggregation tests, not observed robotics performance.

## 7. Next qualification work

1. Pin SafeVLA-Bench's official evaluator source and per-spec robustness/normalization formula.
2. Implement a source-compatible adapter that consumes raw host trajectories and official force/pose signals, with instrument/calibration lineage.
3. Recompute the public reference rows from raw per-episode evidence before comparing a Luminous policy.
4. Run matched baseline/gated-policy pairs on identical task/scene/seed splits and publish SR, safety, SBU, severity, interventions and resource use separately.
5. Add independent-unit summaries and cluster-aware paired uncertainty intervals; do not treat repeated checkpoints or scenes as independent episodes.
6. For physical evaluation, align grasp/contact/assembly measurement procedures with NIST/ASTM F45 and keep the statistical selective-risk certificate distinct from physical interlocks and controller authority.

The goal is not to claim that a robot is safe because a score is high. The goal is to make every success claim conditional on independently measured task outcomes, task-applicable safety evidence and complete provenance.

# CMAQ-041 — Operational intervention scorecard

## Why this work exists

A statistical monitor is not successful merely because it detects a shift quickly. In a safety-relevant system, the decision has value only if it produces enough lead time to complete a fallback before the independently established recoverability deadline, while avoiding needless intervention and preserving useful operation.

CMAQ-041 makes that operational outcome measurable. It is deliberately independent of the monitor's own alarm: the scenario ground truth, counterfactual recoverability deadline and plant/simulator outcome must be generated and recorded separately.

## Input contract

Each EpisodeTrace contains:
- immutable episode ID and independently declared scenario-family ID;
- evaluated duration in integer microseconds;
- ground-truth intervention-required label;
- counterfactual recoverability-margin exhaustion time when intervention is required;
- first monitor warning/quarantine time;
- fallback start and completion times;
- independent safety-violation and task-completion labels;
- useful progress in fixed-point parts per million.

The deadline is supplied by the frozen scenario/plant evaluator. It is not inferred from the warning, fallback action, or monitor state. Hazard episodes must have a deadline within the evaluated trace. Safe episodes must not carry one.

Timestamps must be within the trace. Fallback completion requires a start. Fallback completion cannot precede start, and fallback cannot start before a recorded warning. A warning or fallback completion exactly at the deadline is late; timely intervention requires strictly positive lead time.

## Output scorecard

The evaluator returns both an overall scorecard and a sorted per-family breakdown. It reports these as separate metrics, never as a single composite scalar:

- intervention-required episodes;
- timely warnings versus late/missing warnings;
- timely quarantine entry versus late/missing quarantine entry;
- fallback completions before deadline versus late/missing completions;
- p50 and p90 warning, quarantine and fallback-completion lead times over positive timely leads;
- false warnings and false quarantines as separate counts on safe episodes;
- unnecessary fallback starts on safe episodes;
- safety violations;
- safe operating exposure in microseconds;
- safe task completions;
- safe useful progress observations and their mean in ppm.

For false quarantine/hour, the evaluator exposes both the raw false-quarantine count and the safe-operating-time denominator. Consumers should report both (or their derived rate) and not hide the denominator.

Safe useful progress and safe task completion stay visible alongside intervention success. A policy that always quarantines cannot score well merely by avoiding intervention misses.

## Build and test

Run from the repository root:

    cargo +1.100.0 test --manifest-path research/cmaq-041/Cargo.toml --locked
    cargo +1.100.0 clippy --manifest-path research/cmaq-041/Cargo.toml --locked --all-targets -- -D warnings

The dedicated CI workflow also checks formatting.

## Integration path

- CMAQ-030 provides calibrated recoverability-margin evidence.
- CMAQ-031 / 033 / 038 / 040 provide statistical shift-monitor evidence.
- CMAQ-034 provides fail-closed quarantine and recovery behavior.
- CMAQ-036 / 039 provide declared common-cause closure and effective-block lineage.
- CMAQ-041 evaluates whether the resulting warning/fallback chain actually completed before a frozen deadline and what safe utility it preserved.

The next adapter should import immutable per-episode receipts from scenario runners (nuPlan, CARLA local scenario sets, hardware-in-the-loop and later physical trials) while preserving source IDs, hashes, split membership and closure block IDs. Randomly overlapping clips from a common recording must not be treated as independent evaluation episodes.

## Claim boundary

CMAQ-041 is a software scorecard, not a safety case. It does not establish that the ground-truth deadline is physically valid, that the scenario suite is complete, that the monitor is calibrated, that fallback is feasible, or that any vehicle/robot is safe. Those claims require independent scenario/plant evidence, authenticated provenance, common-cause analysis, and hardware/physical validation.

Simulator results must be reported as simulator results. Hardware-in-the-loop and physical deployment evidence require separate cohorts and reports.

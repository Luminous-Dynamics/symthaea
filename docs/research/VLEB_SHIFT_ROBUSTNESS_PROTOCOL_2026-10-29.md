# VLEB dynamic-shift robustness protocol — 2026-10-29

## Motivation

Most repository-level coding tasks freeze their specification and workspace throughout execution. Real engineering work does not: requirements change, dependencies move, files and runtime state update, tools or sessions change, and prior evidence becomes stale. A successful result on the original task therefore does not establish that an agent can notice and adapt to a changed task while preserving still-valid work.

SWE-Shift (published 2026-10-23) formalizes matched clean and shifted continuations from the same saved history and workspace. Its reported benchmark contains 120 scenarios across 12 change families and nine scheduled intervention positions. Across 17 evaluated systems, shifted success was lower than clean success for every system; tool/session changes and stale evidence produced zero paired success in that evaluation. These results are specific to its protocol, but they motivate VLEB to measure shift robustness directly rather than infer it from static-task scores. Source: https://arxiv.org/abs/2610.32196

AgentSuite/COBA separately demonstrates that evaluation flaws arise across user instructions, environments, ground truth, and evaluation logic. The VLEB shift protocol therefore requires the shift scenario itself to be pinned and insists that independent runner evidence establish whether the intended intervention was really applied. Source: https://proceedings.mlr.press/v306/suh26a.html

## What this analyzer does

`tools/vleb/analyze_shifted_runs.py` is a paired analysis layer over evaluator-produced run records. It does not execute the agent, generate interventions, authenticate evidence artifacts, or assert that an intervention actually occurred. Its outputs are permanently `analysis_only=true` and `score_eligible=false`.

The input manifest freezes:
- scenario ID, predeclared task family and change family;
- an intervention position from 1 to 9;
- before/after task-contract digests and shift-event digest;
- the initial source revision, immutable image, workspace-state digest and checkpoint digest;
- a fixed seed set, the protocol and bootstrap settings;
- exactly one clean and one shifted run for every scenario × seed.

Each run record pins the canonical scenario digest, starting state/checkpoint/revision/image, model/version, scaffold, initial execution profile, network mode, final-state digest, receipt and audit-report digests, cost and duration. Clean and shifted runs must start from identical pinned identities. The shifted continuation must additionally declare its post-shift execution-profile digest and network mode; the clean condition must not claim a post-shift configuration. This permits changes in tools/session policy without confusing them with a different initial starting state.

The canonical scenario digest is SHA-256 over UTF-8 JSON emitted with sorted keys, compact separators, `ensure_ascii=false`, and non-finite numbers disallowed. The analyzer requires external pins for the raw input manifest SHA-256, protocol SHA-256, and analyzer source SHA-256. Run IDs, receipt hashes, and audit-report hashes must be unique, and retries cannot replace the scheduled primary attempt.

## Success definition and metrics

The evaluator must define a pass as satisfying the current task contract after the intervention, preserving all still-valid work, and meeting current evidence requirements. An outcome of `pass` is not merely that the old test suite still passes. The external evaluator must independently verify that the intervention was actually delivered at the pinned position and that prior evidence invalidated by the shift is not reused as current evidence.

The analyzer reports:
- clean success, shifted success, and paired success rates;
- shift degradation in percentage points (`clean rate − shifted rate`);
- robustness conditional on a clean success, defined as both conditions passing divided by the number of clean passes (undefined when no clean pass exists);
- an exhaustive 3 × 3 transition matrix across pass/fail/unresolved outcomes;
- unresolved-pair counts, per-condition cost per scheduled pair, and p50/p90 duration;
- the same outcome and uncertainty summary by change family, task family, and intervention position.

Unresolved observations remain in every assigned denominator and count as non-passes. The transition matrix makes it possible to distinguish clean-pass → shifted-fail regressions from clean-pass → shifted-unresolved cases, as well as cases where a shifted run succeeds after a clean run fails.

## Statistical method

The primary resampling unit is the scenario, not the individual run. Each bootstrap draw samples scenario IDs with replacement and carries every scheduled seed and both conditions for each selected scenario. The point estimate and percentile-bootstrap interval are reported for clean success, shifted success, shift degradation, and paired success. This preserves the clean/shift pairing and avoids treating repeated seeds within a scenario as independent scenario evidence.

All intervals are descriptive, use the preregistered confidence level/replicate count/seed, and are not simultaneous across groups or dimensions. Subgroup intervals are exploratory when a group contains fewer than ten scenarios. Family labels and intervention positions must be fixed before outcomes are inspected; changing groups after seeing results invalidates confirmatory interpretation. The bootstrap does not capture stochastic serving variance that is not reflected by the predeclared seeds.

## Proposed targets

1. **100% pair completeness:** every locked scenario × seed has one clean and one shifted run, with zero duplicate cells and zero retry substitution.
2. **0 starting-state drift:** all paired conditions share identical initial revision, image, state/checkpoint digests, model/version, scaffold, initial execution profile, and initial network mode.
3. **100% intervention verification:** every shifted run has independent runner evidence proving the declared shift event and intervention position; no score eligibility from this analyzer alone.
4. **0 stale-evidence accepts:** negative controls that retain evidence invalidated by the shift must never be accepted as passes.
5. **Shift robustness disclosure:** publish clean rate, shifted rate, paired-success rate, degradation, the transition matrix, and per-family/per-position summaries; do not publish only a pooled average.
6. **Held-out shift families:** after the initial harness stabilizes, freeze unseen combinations of change family × intervention position for confirmatory testing; keep them out of development feedback.
7. **Cost-adjusted resilience:** report added cost and latency for shifted recovery, not just whether the agent eventually passes.

## Comparison with industry

Do not compare raw pass percentages directly against SWE-Shift or general coding-agent indexes. Task sets, environments, intervention schedules, scaffolds, budgets and success definitions differ. The appropriate first comparison is an internally paired experiment using the same task, saved workspace/checkpoint, model/scaffold, seed and budget policy, changing only the preregistered intervention condition. Then run an external comparison only when the full protocol can be matched.

The useful leadership metric is not simply a high static success rate. It is a verified clean-to-shifted robustness profile with no stale evidence, preservation of valid work, transparent failure transitions, and measured recovery cost.

# SFEB two-sided task validity: learn from the 2026 benchmark audits

**Date:** 2026-11-23  
**Status:** protocol and machine-readable policy authored; no task has passed this new gate  
**Parent:** [PR #13162](https://github.com/Luminous-Dynamics/symthaea/pull/13162)  
**Related:** [#13251 task-family expansion](https://github.com/Luminous-Dynamics/symthaea/issues/13251), [#13247 independent oracle replay](https://github.com/Luminous-Dynamics/symthaea/issues/13247), [#13228 trusted execution](https://github.com/Luminous-Dynamics/symthaea/issues/13228)

## Why this is the next paradigm shift

A task can have a valid JSON schema, a polished brief, golden cases and a passing reference screenshot while still being a defective measurement instrument. The scoring tests may demand details absent from the brief, may fail even on a correct solution, or may pass an empty patch. These failures produce misleading model rankings.

The July 8, 2026 OpenAI audit of SWE-Bench Pro found substantial task-quality problems in the public split: 200 tasks were flagged by an automated pipeline (27.4%) and 249 by human annotation (34.1%). The reported failure classes included overly strict tests, underspecified prompts, low-coverage tests and misleading prompts. This is a warning about benchmark curation, not evidence that SFEB has the same defect rate.

SWE-Bench Pro V2's September 22, 2026 release demonstrates stronger validity controls. It removed 89 invalid/unrepairable tasks from the 731-task public v1 split, leaving 642 tasks across 11 repositories; revised 529 task statements so graded assertions trace to the prompt; had revised instructions independently solved blind; required every task to pass with the reviewed reference patch and fail with the empty patch; restricted agent network access to the model endpoint; and re-graded submitted diffs on pristine images. This does not make SWE-Bench Pro a frontend benchmark, but it is a strong precedent for how rigorous task validity should work.

Sources:
- [OpenAI task-quality audit (July 8, 2026)](https://openai.com/index/separating-signal-from-noise-coding-evaluations/)
- [SWE-Bench Pro V2 release (September 22, 2026)](https://labs.scale.com/blog/swe-bench-pro-v2)
- [SWE-Bench Pro V2 release gates and task tooling](https://github.com/scaleapi/SWE-bench_Pro-os/blob/main/v2/README.md)
- [FrontierCode 2.0 (October 27, 2026)](https://cognition.com/blog/frontier-code-2.0)
- [OSWorld 2.0 leaderboard and task summary](https://osworld-v2.xlang.ai/)

## SFEB's two-sided task-validity contract

Every task must demonstrate both **satisfiability** and **discriminative sensitivity** before agent scores are accepted.

1. **Witness pass:** a reviewed reference implementation/patch applied to the exact pristine base passes all declared target requirements.
2. **No-op or seeded-defect fail:** an empty patch or the declared defective baseline fails the intended target requirement. It must fail for the intended reason, not an unrelated infrastructure problem.
3. **Brief-to-oracle traceability:** every graded assertion is supported by an explicit statement in the user-facing task brief; every testable brief requirement has one or more assertions. No hidden requirements inflation.
4. **Blind human solvability:** an engineer who did not author the tests can solve the brief with permitted references without seeing hidden expected outputs, gold patches or verifier internals. Two reviewers independently inspect ambiguity; disagreements are adjudicated.
5. **Pristine regrading:** copy only the candidate patch into a new evaluator-controlled image made from the frozen base, then rerun all required tests and browser journeys. Do not trust test results from an environment the candidate could modify.
6. **Network/solution leakage:** deny fixing commits, hidden test logic and solution-bearing code-host access. If internet is permitted, restrict it to the declared model endpoint and approved documentation; record requests and redirects.
7. **Counterfactual preservation:** for repair tasks, the pre-repair failure must be reproduced, the post-repair target oracle must change, a placebo must not receive credit, and unrelated invariants must remain intact.
8. **Exposure integrity:** public pilot tasks are permanently public-development material. Controller-only and hidden tasks must stay outside the agent workspace and public repository.

The formal contract distinguishes three task modes:

| Mode | Negative control | Positive witness | Preservation condition |
|---|---|---|---|
| Feature implementation | Empty/no-op patch fails new requirements | Reviewed reference patch passes | Existing journeys still pass |
| Seeded defect repair | Defective baseline and placebo fail the target defect | Reviewed minimal repair clears target defect | Unrelated requirements remain unchanged |
| Maintenance change | Unmodified baseline fails new maintenance requirement | Reviewed reference patch passes | Existing behavior and dependency constraints remain |

The machine-readable policy is in benchmarks/frontend/task-qualification/two-sided-task-validity-v1.json, with its JSON Schema in benchmarks/frontend/task-qualification/two-sided-task-validity-v1.schema.json. It is a protocol design, not an execution report.

## Why the gate is stronger than golden tests

Golden tests establish expected outcomes for selected inputs. They do not establish that the task can be solved from its brief, that the tests cover the requested change, or that an empty patch is rejected. The two-sided gate adds independent evidence of both satisfiability (at least one valid solution exists) and sensitivity (the no-op/broken baseline is rejected for the target behavior).

The blind solve checks requirement clarity. The pristine regrade checks that the candidate did not change the evaluator or leave behind environmental state. Prompt-to-assertion traceability constrains hidden-test surprises. Together these reduce the risk of reward hacking and false ranking.

A two-sided gate is necessary but not sufficient. A reference patch can satisfy an incomplete test suite, and a no-op can fail for an unrelated reason. Therefore SFEB still needs mutation campaigns, positive/metamorphic controls, independent oracle replay, signed execution provenance and human review.

## How to integrate without waiting for the runner

The policy can be adopted now at task-authoring and review time:

- Add a requirement↔assertion matrix to each public pilot task package.
- Author a reviewed witness patch outside the evaluated agent's workspace.
- Add an empty/no-op or known-defect control and record the intended failing oracle.
- Have an independent reviewer solve the prompt from the brief and permitted reference only.
- Record the exact base, witness patch, negative control, test/policy and fixture digests.
- Keep all statuses `not_reviewed` or `reviewed_not_two_sided` until those checks have evidence.

When the isolated runner exists, reapply each witness and candidate patch on pristine images, store per-oracle outcomes and link them to the signed execution and verifier statements. Do not mark a task runnable merely because the task and candidate sources compile.

## What this means for SFEB's competitive position

The nearest comparisons now cover different slices of the problem:

- **Vision2Web** is the closest direct website-development comparator: static, interactive multi-page and full-stack task levels. SFEB should match these lanes but add stronger task validity, causal repair, regression preservation and evidence integrity.
- **Design2Code** is a visual reconstruction reference; it is not a complete engineering benchmark.
- **FrontierCode 2.0** provides a serious maintainer-mergeability comparator for repository changes, including security and performance patches. Its 132 tasks use maintainer-developed grading criteria and a mix of tests, rubrics and verifiers. It does not measure rendered frontend quality.
- **SWE-Bench Pro V2** is a task-validity, hermeticity and patch regrading precedent. SFEB should adapt its two-sided release gate and clean-room regrading, not compare headline task scores.
- **OSWorld 2.0** contributes long-horizon computer-use stress: 108 workflows, 31 self-hosted websites and more than 250 average agent steps on the benchmark summary. It measures computer-use workflow completion, not source code quality.
- **WebdriverIO Browser Agent Benchmark** isolates browser-tool choice under a matched model/harness/prompt. Its October 2026 page reports a 62% top row with approximately ±10 percentage-point intervals on a 100-task sample; it also notes roughly one-in-seven disagreements between WebJudge and human reviewers. It is a reminder to publish raw observations and judge calibration, not a direct generation score.
- **Arena's Interaction-as-a-Judge** shows the value of iterative interaction for evaluating applications and guiding repair. It remains a diagnostic/subjective signal; deterministic security, identity and evidence gates must stay authoritative.

Do not pool scores from these benchmarks. Reproduce each native protocol where authorized, then compare SFEB on matched tasks, budgets, models, tools, environments and graders.

## Updated SFEB inventory

At this revision:
- 12 staged task definitions;
- two checked-in original public-pilot reference fixture bundles;
- one fixture-ready task in the 30-task expansion plan (SFEB-PILOT-INT-003 → SFEB-L2-005);
- 29 planned candidate fixtures;
- zero runnable tasks.

SFEB-L2-005's 11/11 JavaScript golden outcomes, 5,000 generated states, 9/9 model fault simulations and 8/8 URL-field omission checks are reference-model preflight only. They are not Rust test results or a compiled Leptos browser run. SFEB-L1-004 remains specification_only in the task index. No task has yet passed the new two-sided validity protocol.

## Target metrics

Use the following as qualification targets, not achieved results:
- 100% witness-patch pass rate across task hard gates;
- 100% no-op/seeded-defect rejection for target requirements;
- 100% of scored assertions linked to explicit brief requirements;
- zero unadjudicated ambiguous/contradictory requirements;
- zero candidate modifications to evaluator/test/policy inputs;
- 100% clean-room regrade identity and artifact integrity;
- placebo false-credit rate of zero on the declared repair controls;
- 100% preservation of unrelated hard-gated journeys.

Report per-task denominators and all excluded/inconclusive cases. Do not convert these into a single average score.

## Non-goals and claim boundary

This work does not execute the reference patch, compile the Leptos candidate, run browser journeys, authenticate execution statements, or qualify any agent. It supplies a stronger task validity protocol and updates the public research inventory. Runnable task count remains zero until all compiler, browser, isolation and independent-verifier gates pass.

# SFEB paired-study power and evaluator-sensitivity analysis

**Date:** 2026-11-23  
**Status:** reproducible design supplement; no SFEB agent-performance result  
**Parent:** [PR #13162](https://github.com/Luminous-Dynamics/symthaea/pull/13162)

## Executive result

The comparative protocol pre-registers a +5 percentage-point paired success difference for the A4 full-Symthaea arm versus the A0 matched baseline and specifies an exact McNemar test. It correctly requires a blinded-pilot power analysis, but a task count alone does not establish that the study is powered.

This supplement computes exact two-sided McNemar power for a +5 pp paired effect under four plausible task-level discordance rates. At 150 independent task pairs, power ranges from 16.4% to 76.6%. At 500 pairs, it ranges from 50.2% to 100.0%. If the discordance rate is 20%, 1,062 complete task pairs are required for 95% power; at 30%, 1,587 are required.

**Decision:** retain 30 as a public development corpus and 150 as an exploratory comparative stage. Treat 500 as a publication-stage floor, not a guarantee of adequate power. Lock the confirmatory sample size from a blinded pilot before revealing any hidden-holdout outcomes.

## Reproducible artifacts

- paired-mcnemar-power-v1.mjs: dependency-free Node.js calculation and internal sanity assertions.
- paired-mcnemar-power-v1.json: deterministic machine-readable output.

Run from the repository root:

    node benchmarks/frontend/studies/paired-mcnemar-power-v1.mjs

The calculation enumerates the number of discordant task pairs, applies the exact two-sided conditional binomial test for McNemar's statistic, then sums conditional power over the binomial distribution of discordant pairs. It introduces no Monte Carlo noise. The implementation self-checks its binomial CDF recurrence against direct finite sums for small cases and verifies null rejection control and monotone power. The reported values were also cross-checked independently with SciPy's exact binomtest / binomial PMF calculations; the two implementations agreed to numerical precision for all reported n=150 and n=500 rows.

## Model and assumptions

Let:

- \(\delta = P(A4=1,A0=0)-P(A0=1,A4=0)\), the paired absolute success-rate advantage;
- \(q=P(A4\ne A0)\), the task-level discordance rate;
- \(D\sim\mathrm{Binomial}(n,q)\), the number of discordant task pairs.

Conditional on \(D\), the number of discordant pairs favoring A4 is binomial with probability \((q+\delta)/(2q)\). The exact power is the sum of the conditional exact-test rejection probability over all possible values of \(D\).

For this sensitivity analysis, \(\delta=0.05\), two-sided \(\alpha=0.05\), and \(q\in\{0.05,0.10,0.20,0.30\}\). These are design scenarios, not estimates from SFEB data. The assumed direction favors A4. A paired effect of +5 pp does not mean the discordance rate is 5%; they are separate quantities.

## Results

| Discordance q | A4-only success | A0-only success | Power at n=150 | Power at n=500 | n for 80% | n for 90% | n for 95% |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 5% | 5.0% | 0.0% | 76.6% | ~100.0% | 157 | 184 | 208 |
| 10% | 7.5% | 2.5% | 40.5% | 94.1% | 337 | 431 | 521 |
| 20% | 12.5% | 7.5% | 21.4% | 67.6% | 658 | 867 | 1,062 |
| 30% | 17.5% | 12.5% | 16.4% | 50.2% | 975 | 1,291 | 1,587 |

The 5% discordance case is unusually favorable: every discordant pair favors A4. It should not be the planning assumption without pilot evidence. When models often trade wins and regressions across different task families, discordance is likely to be higher and the net +5 pp signal is harder to establish.

A 30-task public corpus is not a confirmatory comparison. For reference, under these same assumptions, its exact McNemar power is only 0.3%–3.7%. In the current repository, the 30-task inventory is still mostly a blueprint: one fixture is marked fixture-ready and runnable_task_count remains zero. The 30-task plan must not be described as 30 executed task pairs.

## Implications for the existing staged plan

1. **30 public tasks:** use for fixture development, ambiguity discovery, oracle debugging, and execution feasibility. Publish per-task findings; do not use it to claim a stable ranking.
2. **150 paired task-level outcomes:** use as a blinded pilot / exploratory comparison to estimate discordance, task-family heterogeneity, missingness and practical effect sizes. Any rank order is provisional.
3. **500 paired task-level outcomes:** useful as an initial publication-stage floor, but insufficient for 95% power when q is around 20% or 30% under the +5 pp effect scenario.
4. **Confirmatory stage:** choose n from the blinded pilot using a conservative upper confidence bound for q, target at least 90% and preferably 95% power, then freeze task count, trial aggregation, exclusion rules, tool/model versions and analysis before opening the holdout. Under the illustrative scenarios, 95% power requires 521, 1,062 or 1,587 complete pairs at q=10%, 20% or 30% respectively.

These figures count **complete independent task-level pairs**, not repeated agent trials. If an evaluation runs three ordinary trials and five high-variance trials per task, aggregate those repetitions using a pre-registered rule to produce one primary paired outcome per task. Do not multiply n by the number of repeats. Keep per-trial variance as a secondary analysis.

The calculation does not account for task-family clustering, unequal weighting, missing/inconclusive pairs, model/tool drift or multiplicity beyond the single primary contrast. If task outcomes are correlated within families, the effective information is lower than the independent-pair calculation. Stratify by static-visual, interactive-multiroute and full-stack/maintenance levels; sample multiple task families within each level; report level-specific estimates; and use family-aware cluster bootstrap or a pre-specified hierarchical model. Do not pool repeated trials as independent tasks.

For operational planning, over-recruit to obtain the required number of **complete** pairs, while keeping infrastructure failure / inconclusive outcomes distinct from product failure. Report intention-to-evaluate outcomes and sensitivity analyses; never silently drop difficult or inconclusive pairs. As an illustration only, a 10% unusable-pair rate would require about 1,180 enrolled pairs to retain 1,062 complete pairs.

## Oracle qualification: zero observed failures is not zero risk

A second sample-size issue applies to the 20-negative-control mutation campaign. A curated suite is valuable for checking that known failure modes are caught, but it cannot estimate a low production false-acceptance rate merely because all cases are rejected.

Under an idealized independent representative Bernoulli sample, if zero mutants are incorrectly accepted, the one-sided exact 95% Clopper–Pearson upper bound on the false-acceptance rate is:

\[
  p_{upper}=1-0.05^{1/n}
\]

| Independent representative mutants, zero false accepts | One-sided 95% upper bound |
|---:|---:|
| 20 | 13.91% |
| 60 | 4.87% |
| 100 | 2.95% |
| 299 | 0.997% |

This is a statistical illustration, not a claim about the present campaign. The current 20 controls span different categories and are hand-designed rather than randomly sampled from a defined defect population; they do not satisfy the representative-i.i.d. assumption. Therefore the benchmark should report exact per-mutant detection and mutation-category coverage, not claim that the real-world false-acceptance rate is zero or below a specific percentage. Build a larger, independently authored and periodically refreshed mutant bank; reserve some mutation families as hidden controls; and have a separate reviewer verify that each mutant truly changes the targeted semantics.

For critical/security/evidence-integrity mutations, preserve the zero-tolerance gate: one accepted critical mutant blocks evaluator qualification. Add confidence intervals and denominators for every category. Keep hand-authored challenge suites as deterministic conformance gates, and separately use a representative mutation sample for statistical sensitivity estimates.

## Recommended amendments to the comparative protocol

- Keep A4−A0 as the single primary contrast; exact two-sided McNemar, alpha .05.
- Define the primary success aggregation rule for repeated trials before holdout access (for example, a fixed majority/consensus rule with a specified treatment of inconclusive runs).
- Estimate q from a blinded pilot without inspecting arm identities or holdout labels; use a conservative confidence bound rather than the point estimate alone.
- Pre-register an explicit 90% or 95% power target and complete-pair count, family stratification, task weights, exclusions, missing-data sensitivity and stopping rule.
- Retain the 150 and 500 stages, but label them exploratory/floor stages until their realized power is demonstrated.
- Preserve separate score dimensions: functional contract completion, target-defect repair, unrelated-regression rate, security/evidence hard gates, keyboard journey completion, responsive failure intervals, visual quality, time, tokens and cost per accepted task.
- Keep human preference separate from deterministic correctness. A preference judge, screenshot judge or interaction-grounded judge must not override security, identity, evidence-integrity or explicit contract oracles.

## Paradigm shift: benchmark engineering as metrology

The differentiator should not be another headline score. Treat SFEB as a measurement instrument that must quantify its own uncertainty and failure modes:

1. **Task metrology:** frozen briefs, fixture digests, explicit ambiguity, task-family strata and hidden family-level holdouts.
2. **Oracle metrology:** positive, negative and metamorphic controls; mutation coverage; false-accept / false-reject intervals; independently replayed outcomes.
3. **Execution metrology:** immutable source/environment/browser identities, signed execution evidence, isolated browser execution and network-denial receipts.
4. **Causal metrology:** reproduce a defect, localize its requirement→event→region→source path, intervene, replay, run a placebo, and show unrelated invariants preserved.
5. **Statistical metrology:** paired outcomes, exact tests, family-cluster uncertainty, calibration, pre-registration and explicit inconclusive rates.

This lets SFEB answer a much stronger question than “which model generated the most attractive page?”: which engineering system makes the most correct, secure, accessible, regression-resistant changes under controlled conditions, with evidence another party can independently replay?

## Current claim boundary

This supplement adds a reproducible statistical design calculation only. It does not execute Rust, compile the Leptos application, run a browser mutation campaign, authenticate in-toto/Sigstore statements, qualify an oracle, score Symthaea or establish superiority over any external benchmark. Those gates remain as tracked by [#13228](https://github.com/Luminous-Dynamics/symthaea/issues/13228), [#13247](https://github.com/Luminous-Dynamics/symthaea/issues/13247), and [#7435](https://github.com/Luminous-Dynamics/symthaea/issues/7435).

## External comparator references

- Vision2Web, ICML 2026: 193 tasks, 16 categories, 918 prototype images and 1,255 test cases across static, interactive multi-page and full-stack website development: <https://proceedings.mlr.press/v306/he26n.html>.
- Design2Code, NAACL 2025: screenshot-to-code evaluation over 484 real-world webpages: <https://aclanthology.org/2025.naacl-long.199/>.
- WebdriverIO Browser Agent Benchmark (updated 2026-10-09): fixed model/harness/prompt, seven browser tools, 100 live-web tasks and published cost/time/token results; its 62% leading row is a browser-tool result, not a code-generation score: <https://benchmark.webdriver.io/>.
- Arena Code Arena WebDev: the 2026-11-22 snapshot shows 1,044,803 votes and 164 models on the overall board; the Frontend sub-board is a separate snapshot/scale (2026-11-20: 897,929 votes and 162 models). These are pairwise human-preference ratings, not deterministic acceptance rates: <https://arena.ai/leaderboard/code/webdev> and <https://arena.ai/leaderboard/code/webdev/frontend>.
- Interaction-as-a-Judge (2026-10-13): interaction-grounded evaluation/repair signal with reported improvements on its own datasets; use as a secondary evaluator, never the trust root: <https://arxiv.org/abs/2610.17895>.
- OpenAI's 2026-07-08 SWE-Bench Pro audit estimated roughly 30% of tasks had quality issues; this reinforces the need to qualify tasks/oracles, not only score agents: <https://openai.com/index/separating-signal-from-noise-coding-evaluations/>.

Do not blend these external scores. Reproduce each benchmark under its own authorized split and protocol; compare SFEB only on matched tasks, budgets, models, tool access, environment and scorers.

# MACE-CoS — industry comparison and benchmark plan

External numbers below belong to their own papers, models and evaluation protocols. They are not MACE-CoS results.

| Work | Publicly reported scope/result | What MACE-CoS must compare |
|---|---|---|
| CoSA / CoSA-Bench (Oct 2026) | 288 tasks across recruiting, finance, IT and business; five LLMs and three agent harnesses. Authors report that pool additions capped at 5% raise policy violations from 35.02% to 58.45%. A harness guard reduces violations from 65.28% to 6.67% on one reported OpenClaw recruiting condition; matched benign completion changes from 66.67% to 65.00%. | Direct security baseline. We must compare against the published guard, not just no guard. |
| SkillFusion (Oct 2026) | Offline compiler for typed/scoped skill constraints, hard/soft conflict handling, fallback checks, abstraction and redundancy. Authors report strongest mean success among evaluated non-oracle methods on three agent and two transfer benchmarks across three executor models. | Semantic/directive composition. MACE should add runtime flow and authority checks, not duplicate compilation. |
| SkillAlign (EMNLP 2026) | Full instructions, hints, compressed summaries, workflows and no-exposure modes evaluated on ALFWorld and SkillsBench. Exposure form affects both success and rendered context cost. | Counterfactual exposure optimization and equal-quality token reduction. |
| SkillShift-Bench (Oct 2026) | Controlled contextual, perceptual and execution shifts across three self-hosted web systems; reports degraded skill executability under shift. | Revalidation, stale-target detection and recovery under changing tools and permissions. |
| SkillsBench v1.1 | 87 tasks; 25 agent/model configurations, up to three trials and 95% CIs. Live board lists GPT-5.5 + OpenHands at 67.3% with skills. | Task utility and skill benefit; not equivalent to security qualification. |

## Matched baselines

1. Flat full-skill-pool injection, no composition gate.
2. Retrieval/ranking only.
3. Individual skill admission only.
4. SkillFusion-style offline reconciliation.
5. CoSA harness-level guard, reproduced as closely as code/data access permits.
6. MACE-CoS typed runtime composition gate.
7. SkillFusion + MACE-CoS + SkillAlign-style exposure routing.
8. Oracle policy/authorization decisions (diagnostic ceiling only).

Freeze model, harness, permissions, pool, task set, contamination rate, prompt budget and evaluator in paired runs.

## Required scorecard

- Violations by disclosure, unauthorized state modification, unsafe execution and target/action mismatch.
- Benign task completion and utility; deny-all is not a win.
- Per-domain/model/harness and skill-chain-depth slices.
- False rejection, abstention, severity and successful containment.
- Exposure tokens, wall clock, tool calls and verification latency.
- Stale-use rate and time to revalidate after skill/source/policy changes.

## Proposed qualification targets (not achieved)

- 100% agreement with an independently written finite-state oracle on small generated compositions.
- Zero prohibited data-flow accepts in the registered synthetic adversarial suite.
- On CoSA-Bench, at least 90% relative violation reduction versus the matched unguarded baseline while retaining benign task completion within a pre-registered 1.5 percentage-point non-inferiority margin.
- No critical domain/model/harness cell may hide a severe regression behind an aggregate.
- At least 2/3/5/10 skill compositions, including additions capped at 5% of the original pool.
- On equal-quality tasks, reduce exposed skill-context tokens by at least 25% versus full-pool injection.

These are project targets, not industry standards or current results. The 90% security-reduction goal is deliberately ambitious relative to the CoSA paper's reported guard and has not been demonstrated.

## External sources

- CoSA: https://arxiv.org/abs/2610.35082
- SkillFusion: https://arxiv.org/abs/2610.15481
- SkillAlign: https://aclanthology.org/2026.emnlp-main.2565/
- SkillShift-Bench: https://arxiv.org/abs/2610.33528
- SkillsBench: https://www.skillsbench.ai/

# HIST-CAUSAL-001 v0.4: external qualification and adaptive challenge plan

Date: 2 Nov 2026. Status: research design; no external leaderboard execution or SOTA claim.

## 1. Current benchmark landscape

### Autonomous Science Bench 0.1
The official leaderboard has six pilot tasks, two per domain; each of nine configurations ran three trials per task. The published results show GPT-6 Astra Codex at 72.2% and Opus 5.5 Claude Code at 66.7%. The authors explicitly state that small differences are not meaningful with only three trials per task. Tasks score two calibrated final gates: discovery on designs actually measured and learning/ranking on designs never measured. It compares against random selection, Bayesian optimization and lab workflows. This is the closest external campaign-level benchmark adapter for our experiment-selection layer.
Source: https://www.autonomous-science-bench.ai/

### EmbodiedLab
Published 26 Oct 2026: 30 quantitative experiments across five disciplines and 14 apparatus platforms. Agents must independently choose procedures, operate the apparatus, read instruments and leave the physical state required. The paper reports even the strongest native-control system below 50% success, with most failures arising in apparatus interaction. This motivates explicit action-command versus realized-action evidence and apparatus-state scoring.
Source: https://arxiv.org/abs/2610.35385
Code: https://github.com/HexaFuture/EmbodiedLab

### SafeLab
ICML 2026: 63 calibrated laboratory assets, 64 tasks across nine manipulation categories and 6,400 expert trajectories. It documents unsafe-but-successful trajectories and a task-success versus safe-success gap, with up to 43 percentage-point improvement in simulated safe-success from bounded residual RL. Fifty physical replays showed 86% agreement between simulated and observed safety outcomes. This supports a separate trajectory-level safety gate rather than treating a successful final state as sufficient.
Source: https://proceedings.mlr.press/v306/bai26e.html

### AutoSciBench
Published 4 Oct 2026: separates a high-level task concept from a lower-level recipe that constructs the task and verifies its ground truth. It uses solver traces and feedback to refine recipes and close shortcuts. It reports solver-accuracy decreases of 22.4 and 25.5 percentage points on generated computational-biology and materials-science tasks relative to human-curated tasks, while LLM judges rated generated task quality higher. These quality scores are not equivalent to independent expert validation. The useful idea is adaptive recipe refinement; we add independent exact validators and private confirmation seeds.
Source: https://arxiv.org/abs/2610.05140

### FIRE-Bench
ICML 2026 full-cycle rediscovery benchmark: agents design and run experiments to recover documented findings and provide evidence-backed conclusions. Strongest reported agent remained below 50 F1 with recurring design, execution and evidence-reasoning failures.
Source: https://proceedings.mlr.press/v306/wang26d.html

## 2. Do not compare headline scores directly

These benchmarks evaluate different combinations of scientific reasoning, physical interaction, experiment design, apparatus skills, outcome quality, evidence and budget. A valid comparative result requires identical:
- task family and held-out split;
- actions, observation channels and feedback timing;
- model prior and initial information;
- experiment, retry, time and compute budgets;
- safety and abstention constraints;
- success definitions and evaluator;
- tuning budget for each baseline.

Report per-domain/task results, worst-case regressions, paired uncertainty intervals and complete denominators. A high discovery score cannot hide failed learning; scientific accuracy cannot hide unsafe execution; and a correct result without an evidence trace does not qualify as rediscovery.

## 3. New benchmark architecture: concept, recipe, validator, trajectory, custodian

Use an adaptive challenge pipeline inspired by AutoSciBench but make ground truth and safety properties independent of the solver:

1. **Concept:** capability to test, such as common-cause rejection, sensor-artifact detection, actuator non-compliance, common-mode failure, model misspecification or abstention under observational equivalence.
2. **Recipe:** versioned generator, exact finite model family or simulator, interventions, costs, measurement channels, expected properties and task difficulty.
3. **Independent validator:** verifies normalized probabilities, target solvability, equivalence classes, probe sufficiency, negative controls, safety constraints, and no leakage before the task can enter a challenge set.
4. **Trajectory ledger:** records requested action, realized action/readback, measurement, calibration, retries, process state, evidence digests, costs and terminal result.
5. **Custodian:** keeps confirmation seeds and recipe perturbations sealed from solver developers; publishes them only after the evaluation window.
6. **Refinement loop:** uses solver traces to detect shortcuts and propose harder recipes, but every revised recipe must independently pass the validator and be checked for unintended ambiguity.

LLM judges may suggest likely shortcuts or missing controls, but must never establish ground truth or decide that a physical result is correct.

## 4. Qualification matrix

| Evaluation plane | Metric | Initial target |
|---|---|---:|
| Exact finite correctness | agreement with independent enumerator/replay | 100% |
| Identifiability | pairwise model distinction on designed identifiable tasks | >=95% |
| Non-identifiability | correct abstention/equivalence class | >=95% |
| False attribution | false unique mechanism claims on negative controls | <1% |
| Calibration | predictive calibration / coverage under declared shifts | report proper scores and intervals |
| Experiment efficiency | intervention cost vs strongest matched baseline | >=10% improvement on predeclared stateful confounds |
| Safety | empirical trajectory violations within declared model family | 0 in finite exact suite |
| Evidence | complete replayable source/action/measurement receipt | 100% |
| Transfer | held-out mechanism and apparatus families | no aggregate-only pass |

Targets are ambitions, not current achievements. The 10% efficiency target must be met without degraded safety, false-claim rate or abstention.

## 5. Immediate integration

- Keep generic Bayesian planning, particle methods, stateful policies and apparatus models in Symtropy SIM-REL-013/014/015/017/018/025 and decision-value extensions.
- Keep HIST-CAUSAL as a mechanism-specific challenge suite.
- Add v0.4 causal triangulation to SIM-REL-017 as its own task family.
- Add paired results to SIM-REL-014 without pooling them into unrelated physical/sensor tasks.
- Build an Autonomous Science Bench adapter for the discovery/learning two-gate campaign evaluation.
- Build EmbodiedLab/SafeLab adapters only where interfaces and action semantics match; report incompatibilities.
- Apply FIRE-Bench-style evidence-backed rediscovery to the outer scientific research agent.

## 6. Claim ceiling

Current v0.2-v0.4 artifacts are finite synthetic reference models. They establish exact pairwise distribution distinctions only within declared model sets and probe menus. They do not establish a physical historical influence, Chain Theory's physical interpretation, universal causal discovery, apparatus safety in the real world or an SOTA win.
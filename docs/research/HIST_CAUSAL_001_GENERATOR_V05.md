# HIST-CAUSAL-001 v0.5: deterministic recipe generation and challenge custody

## What changed

v0.5 adds a deterministic task-recipe generator over the v0.4 six-model causal-triangulation family. It varies the latent common-cause strength, response/dose probabilities and sensor-audit quality using exact rational parameters. Each task has:
- an agent-facing contract: candidate model families, model descriptions, prior, probe vocabulary and costs;
- an evaluator-only key: target model, selected parameters and seed;
- a validator: exact normalization, parameter bounds, full-suite pair separation and target-leakage checks.

A stable SplitMix64-derived recipe selector makes development seeds reproducible. The generated observation sampler is deterministic by (seed, probe, replicate) and uses exact rational categorical probabilities. Independent replicate IDs are required for distinct units.

## Execution

- 100 public development seeds generated and validated.
- All generated models/probes normalize exactly.
- All 15 candidate-model pairs are separable by the full probe menu for each validated seed.
- The designed observational and action-execution equivalence controls remain intact.
- The target model and selected parameter values are absent from the solver-facing agent view.
- 10 unit tests pass locally across 64-seed property sweeps.

The task generator returns both an agent view and an evaluator key for a trusted harness. **Only pass the agent view to the solver.** The evaluator key must remain in a separate process/account/secret store. Public seeds are development examples, not sealed evaluation seeds. A hidden evaluation requires custodian-controlled seeds and recipes.

## Adaptive benchmark evolution

This is not yet a full AutoSciBench clone or an adaptive policy. The next phase should:
1. record solver action/evidence traces;
2. classify failures (wrong mechanism, missing witness, false singleton, execution error, uncalibrated confidence, shortcut);
3. propose a recipe mutation targeting the failure;
4. validate the mutation independently for solvability, identifiability, normalization, negative controls, cost/fairness and no target leakage;
5. promote validated mutations into private confirmation tasks.

An LLM may recommend recipe mutations but must not be ground-truth judge. The independent exact validator decides whether the recipe preserves the declared property. Keep generator, solver and evaluator responsibilities separate.

## External motivation

AutoSciBench (4 Oct 2026) uses a task concept plus a task recipe, then refines recipes using solver trajectories to close shortcuts. It reports a 22.4-point and 25.5-point reduction in average solver accuracy on generated computational-biology and materials-science tasks; task-quality scores were assigned by LLM judges, so independent validation remains important. https://arxiv.org/abs/2610.05140

Autonomous Science Bench 0.1 provides six lab-derived pilot campaigns and scores discovery on measured designs separately from learning/ranking unmeasured designs. Its own page says there are three trials per task and warns that small leaderboard gaps are not meaningful. https://www.autonomous-science-bench.ai/

EmbodiedLab evaluates 30 quantitative experiments across five disciplines and 14 apparatus platforms, with failures dominated by apparatus interaction in the reported native-control setting. https://arxiv.org/abs/2610.35385

SafeLab evaluates trajectory-level laboratory safety over 64 tasks and reports unsafe-but-successful trajectories; task success and safe success need separate gates. https://proceedings.mlr.press/v306/bai26e.html

## Claim boundary

v0.5 makes reproducible synthetic tasks. It does not show that any scientific agent solves them, does not establish physical mechanisms, and does not imply SOTA performance. The generator's correctness is conditional on its six declared model families and parameter supports.
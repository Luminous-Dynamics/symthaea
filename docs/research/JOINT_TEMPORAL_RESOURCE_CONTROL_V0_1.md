# Joint Temporal Resource Control (v0.1)

**Tracking:** NEURO-SILICON-009 / #9666  
**Candidate implementation:** `research/temporal-control/`  
**Evaluator implementation:** `research/temporal-evaluator/src/joint_budget.rs` and `interaction.rs`  
**Status:** deterministic controller and exact small-instance scorer authored; no Rust build or benchmark run

## 1. Why a joint controller?

Memory retention, active sensing, compute scheduling and verification have
often been treated as independent policies. Their actions are coupled:

- An observation has little lasting value if the information is not retained.
- Retaining a stale belief can be worse than evicting it or refreshing it.
- More compute may be wasted when the next observation would change the input.
- A correct forecast is insufficient if its uncertainty is miscalibrated.
- Verification consumes resources but may prevent a costly incorrect action.

The research question is whether online joint allocation can improve the
temporal decision-quality / resource Pareto frontier over independently tuned
subsystems.

## 2. Paradigm shift: marginal future decision debt per resource

The candidate estimates the marginal reduction in future temporal error debt
from a joint action bundle. The evaluator later measures three non-compensatory
exposure components:

1. wrong-belief exposure;
2. abstention exposure;
3. missing-output exposure.

A joint planner sees candidate-authored forecasts and candidate-visible
resource costs only. It does not see the hidden future, event intervals, exact
outcomes or evaluator subset oracle.

Keep resource dimensions distinct: persistent bytes, retained byte-time, I/O bytes, serial critical-path latency,
compute units, observation count, measured energy and verification units. Do not convert one unit into another implicitly. Explicit dual prices
are policy inputs and adaptation state, not universal exchange rates. Hard
ceilings are independently enforced.

## 3. Candidate-side baseline

The initial Rust crate implements a deterministic greedy policy:

- conservative benefit = predicted TED reduction minus an uncertainty radius;
- explicit, separately declared prices for each resource dimension;
- positive net priced value only;
- greedy selection under every hard resource ceiling;
- deterministic tie-breaking by action ID;
- mutually exclusive alternatives represented by conflict groups;
- bounded per-resource price update based on measured episode utilization;
- explicit serial critical-path latency ceiling, priced independently from compute.

This is a baseline to challenge. It is not globally optimal for multidimensional
knapsack, does not infer action complementarity and does not guarantee stable
online dual convergence. A future policy may use model-predictive control,
contextual bandits, primal-dual online optimization, or structured combinatorial
search, but must beat the baseline at matched per-resource budgets.

Joint action bundles let a candidate represent a coupled intervention (for
example, observe + retain + forecast + verify) as one alternative. Independent
actions can coexist; alternatives that cannot both execute share a conflict
group. The evaluator, not the candidate policy, determines whether predicted
bundles actually work.

## 4. Evaluator-only exact oracle

The separate evaluator module enumerates every subset of up to 16 declared
action IDs using an evaluator-owned action registry. Nonzero conflict-group IDs
mark mutually exclusive alternatives; subsets containing two members of the
same group are enumerated but excluded from the feasible frontier and cannot be
qualified as candidate plans. Each run binds identical, nonzero identities and commitments:
- episode ID and task commitment;
- hidden future-stream commitment;
- model digest;
- initial-state digest;
- candidate-visible feature snapshot;
- evaluator-owned action-registry digest (action IDs and mutual-exclusion groups), recomputed by the evaluator from sorted entries using a domain-separated canonical encoding; a nonzero caller-supplied digest alone is not accepted;
- episode horizon and resource budget.

For every subset, the evaluator records measured resource use and each of the
three exposure values. Infeasible subsets remain in the enumeration but do not
enter the feasible Pareto frontier. Candidate plans that exceed a hard ceiling
are not qualified.

The scorer returns separate componentwise best subsets and a componentwise
regret vector. These best subsets may differ. The Pareto frontier is computed
over all three exposure values **and all resource dimensions**, including the declared latency ceiling. No opaque
weighted average is used. The oracle is an upper bound for small fixtures and
must never be made available to the candidate.

## 5. Causal interaction attribution

The evaluator-only `interaction.rs` module consumes the same complete,
registry-digest-bound subset table and reports:

- per-action paired marginal exposure reductions;
- pairwise difference-in-differences for wrong-belief, abstention and
  missing-output exposure;
- positive values as synergy (the companion action improves the first
  action's marginal value), negative values as antagonism, and zero as additive
  behavior in the eligible contexts;
- exact signed integer sums and eligible context counts (the mean is the exact
  rational sum / count; no floating-point rounding);
- separate counts for structurally incompatible and over-budget contexts.

For action i, the main-effect sample compares E(S) with E(S+i). For pair i,j,
the interaction is the change in i's exposure reduction when j is present:
`E(S+i) + E(S+j) - E(S) - E(S+i+j)`. Only contexts where all required
interventions are both compatible and within every resource budget contribute.
This is a conditional, feasibility-filtered interaction measure, not Shapley
value and not a universal causal claim outside the declared fixture/task
distribution.

Six Rust test functions cover additive, synergistic and antagonistic pairs,
mutually exclusive pairs, over-budget exclusion and context/registry mismatch.

## 6. Causal interaction design

Use paired episodes with the same initial state, task commitment, model and
future stream. Randomize treatment order on the evaluator side. Include:
- sensing-only;
- memory-only;
- compute-only;
- verification-only;
- sensing × memory;
- sensing × compute;
- memory × compute;
- memory × verification;
- the full three-way sensing × memory × compute interaction;
- full joint bundles.

For small fixtures, enumerate all joint subsets. For larger workloads, use a
preregistered factorial or covering design and report interaction uncertainty.
A positive full-system result alone does not establish which mechanism caused
the gain.

## 7. Workload families

Reuse the pinned deterministic generators in #9394, #9529, #9559 and #9616
where their schemas are compatible:
- delayed evidence, long gaps and distractor interference;
- bursty, irregular and missing observations;
- hidden abrupt, gradual and recurring regime changes;
- temporal alias pairs that match at candidate-sampled times but differ at
  hidden evaluator probes;
- stale-memory retraction and invalidation;
- multiscale and phase-sensitive prediction;
- explicit-context, latent-context and task-free sequences.

Include null controls where extra sensing or retention cannot help. Include
adversarial conditions where a candidate over-samples, over-retains, refreshes
too late, or spends compute on a low-value forecast.

## 8. Baselines

1. Fixed sensing, memory, compute and verification quotas.
2. Each existing subsystem policy operated independently.
3. Greedy single-resource marginal-benefit scheduler.
4. Independently optimized subsystem policies with no interaction model.
5. The included joint greedy controller.
6. Explicit-price online primal-dual controller with bounded updates.
7. Receding-horizon/model-predictive joint planner using candidate-only forecasts.
8. Contextual-bandit or online-learning policies only where their assumptions
   are met and exploration cost is included.
9. Exact evaluator-only subset oracle for small action sets.

All baselines receive the same information and resource ceilings. Report tuning
budget, training data, compute, seeds, and whether thresholds were selected on
development or confirmatory data.

## 8.1 Stability/plasticity crossover sweep

Vary retained-memory capacity and replay budget independently. Compare
no-memory/context-only, random replay, reservoir replay, balanced replay,
temporal-value retention, active refresh, and joint sensing/memory/compute
control. Where compatible, include a weight-consolidation neural reference as
a separate architecture family.

For each sequence report:
- forward transfer to later tasks;
- backward transfer and old-task retention after new learning;
- forgetting after distraction and memory/replay pressure;
- adaptation samples and compute/energy per new task;
- retained byte-time and replay I/O/compute;
- time to recover after abrupt regime changes;
- results when memory is abundant but update compute is tight, and when memory
  is constrained but compute is abundant.

This crossover distinguishes a storage bottleneck from a plasticity bottleneck.
A policy that only wins when memory is scarce has not established a general
continual-learning advantage.

## 9. Metrics

### Outcome (non-compensatory)
- wrong-belief microseconds;
- abstention microseconds;
- missing-output microseconds;
- each separately by regime and task family.

### Decision and interaction
- feasible Pareto regret against the exact evaluator oracle;
- per-action predicted-versus-realized value;
- pairwise and higher-order interaction effects;
- time-to-detect, time-to-recover and stale-memory false-recall rate;
- calibration and selective-risk curves;
- abstention quality when no safe feasible action exists.

### Resources
- persistent/peak bytes and retained byte-time;
- I/O traffic and memory refresh work;
- compute use and measured CPU/GPU time;
- observation count and sensor-active time;
- verification work/receipt bytes;
- p50/p95/p99 latency and deadline misses;
- directly measured whole-path joules, with measurement boundary disclosed.

Model-estimated energy is not physical energy. If whole-system power is not
measured, report that field as unavailable; do not silently substitute an
estimate. Include host, memory, transfer fabric, preprocessing, idle listening
and verification when they are required by the deployed path.

## 10. Industry comparison and stability/plasticity baselines

- **Continual Learning Bench 1.0:** external stateful-agent validation. Its 2026
  release uses expert-validated sequential task sequences across domains and
  asks whether systems improve from earlier task experience. Its paper reports
  that naive in-context learning can outperform dedicated memory systems.
  The live benchmark site currently displays ICL baselines ahead of a Mem0
  configuration on aggregate reward, a warning that a dedicated memory manager
  is not automatically better. These are LLM-agent results, not a direct
  Symthaea score; use official tasks/harnesses or a declared adapter, and never
  claim a CL-Bench score from synthetic substitutes.
  References: https://sky.cs.berkeley.edu/project/continual-learning-bench/,
  https://arxiv.org/abs/2606.05661 and https://continual-learning-bench.com/
- **MemoryBench (ICML/PMLR 2026):** a complementary user-feedback simulation
  benchmark covering multiple domains, languages and task types. Use its
  sequential feedback / learning-from-practice framing to design compatible
  adapter tasks; do not equate its LLM-system scores with HDC/LTC scores.
  Reference: https://proceedings.mlr.press/v306/ai26e.html
- **Stability versus plasticity / replay:** the ICLR 2026 paper *Forget
  Forgetting: Continual Learning in a World of Abundant Memory* argues that
  with sufficient memory, the bottleneck can shift from storage/stability to
  plasticity, and reports simple replay baselines beating stronger methods in
  some tested regimes. Add random replay, reservoir replay, balanced replay
  and no-memory/context-only baselines where semantics permit. Sweep memory
  abundance against adaptation compute and measure forward transfer, backward
  transfer, forgetting, adaptation speed and consolidation cost. Its Weight
  Space Consolidation method is a reference for compatible neural-weight arms,
  not a presumed plug-in for HDC.
  Reference: https://proceedings.iclr.cc/paper_files/paper/2026/hash/17fc58b1ba92abbb72c5f457e140d851-Abstract-Conference.html
- **NeuroBench:** map compatible tasks to its algorithm track for correctness
  and complexity, then to its system track for end-to-end timing/energy. Its
  v1.0 algorithm track excludes preprocessing/postprocessing costs, so system
  efficiency claims must use the system track or separately measure the full
  deployed path. References: https://neurobench.ai/ and
  https://www.nature.com/articles/s41467-025-56739-4
- **MLPerf Tiny v1.4:** use only for compatible edge inference workloads with
  official workload, quality and energy methodology. The 2026 result round
  emphasizes matched tasks, quality targets and energy reporting. It is not a
  proxy for general temporal learning.
  Reference: https://mlcommons.org/2026/07/mlperf-tiny-v1-4-results/

These are complementary comparison standards, not interchangeable leaderboard
scores. Do not compare unlike task laws, information exposure or measurement
boundaries.

## 11. Proposed qualification gates (targets, not results)

- Zero future-boundary leakage and zero evaluator-only fields exposed to the
  candidate.
- Zero accepted candidate plans that violate a hard per-resource budget.
- Exact agreement with an independent hand-computable oracle on the small
  fixtures.
- At matched vector budgets, target at least 10% lower held-out temporal error
  debt on a preregistered primary exposure metric, with paired intervals
  excluding zero and no practically meaningful regression in the other
  exposure dimensions.
- Or, at matched exposure quality, target at least 20% lower measured cost in
  one or more dimensions without exceeding any other resource ceiling.
- Joint control must beat the best subsystem-local policy on at least one
  preregistered interaction-heavy regime and must not materially regress on
  null/control regimes.
- Forecast/action-value claims require held-out calibration evidence.
- Hardware energy claims require direct whole-path measurement.

The thresholds are research targets only. Freeze them before confirmatory runs;
do not lower them after seeing results.

## 12. Implementation status and limits

Candidate controller: 12 Rust test functions authored.  
Evaluator exact subset oracle: 14 Rust test functions authored.  
Evaluator interaction attribution: 6 Rust test functions authored.  
No Rust build, rustfmt, Clippy, unit test or benchmark has run in this
environment. No physical energy or latency measurement has been performed.
The controller and oracle are research prototypes, not production schedulers.

## 13. Next execution order

1. Compile and run both standalone crates on the exact PR head.
2. Cross-check the evaluator subset oracle against a second independent
   implementation and hand-computable fixtures.
3. Add a machine-readable run harness and exact candidate/evaluator artifact
   binding.
4. Run synthetic mechanism tests and negative controls.
5. Execute paired multi-seed ablations and confirmatory held-out regimes.
6. Map external tasks to CL-Bench, NeuroBench or MLPerf Tiny only when semantics
   and resource boundaries match.

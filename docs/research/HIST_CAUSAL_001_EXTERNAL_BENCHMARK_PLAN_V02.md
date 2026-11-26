# HIST-CAUSAL-001 v0.2: external benchmark alignment and qualification plan

Date: 2026-11-02  
Status: research plan; no external leaderboard run or SOTA claim.

## The field has no single scientific-discovery SOTA score

The current benchmark landscape tests different parts of the lifecycle. They should be used as complementary external comparisons, not collapsed into one score:

| External benchmark | What it tests | How HIST-CAUSAL / Symthaea should compare |
|---|---|---|
| LLM-SRBench / NewtonBench / ERBench | Equation recovery, interactive law discovery, robustness and shifts | Match equation tasks, split by generating law family, report exact structure and extrapolation separately |
| ICML 2026 online Bayesian experimental design for partially observed dynamical systems | Sequential EIG and latent state/parameter inference using nested particle filters | Compare on matched stochastic state-space tasks, same observation/action interface and compute budget |
| Autonomous Science Bench 0.1 | Budgeted campaigns from lab experiments; separate discovery of measured designs from learning/ranking unmeasured designs | Use its six public pilot campaigns as an adapter target; report both gates and cost, do not infer significance from three trials/task |
| EmbodiedLab | Agent physically operates an apparatus, reads instruments, establishes a result and leaves the apparatus in a required state | Later physical/simulator transfer stratum; score evidence-grounded result, apparatus state and safe execution separately |
| FIRE-Bench | Full-cycle rediscovery of established findings with experiments and evidence-backed conclusions | Use as research-workflow comparator, not as a direct numerical optimizer baseline |

## Freshly checked external facts

- Autonomous Science Bench 0.1 lists six pilot tasks across biology, chemistry and materials, and scores both campaign progress and final outputs. Its public leaderboard reports 72.2% for GPT-6 Astra Codex and 66.7% for Opus 5.5 Claude Code, but the page states there are only three trials per task and that small gaps are not meaningful. Its two final gates are discovery on designs actually measured and learning on designs never measured. Source: https://www.autonomous-science-bench.ai/
- EmbodiedLab evaluates 30 experiments across five disciplines and 14 apparatus platforms. It requires the discovered quantity to be grounded in recorded instrument readings and the bench to be left in the required physical state. The paper reports that even the strongest native-control agent succeeds on fewer than half of tasks, and many failures arise from apparatus interaction. Source: https://arxiv.org/abs/2610.35385
- FIRE-Bench asks agents to rediscover established findings by independently designing and running experiments, then producing evidence-backed conclusions. Its ICML 2026 report states that even its strongest tested agent remained below 50 F1, with high run-to-run variance and failures in design, execution and evidence reasoning. Source: https://proceedings.mlr.press/v306/wang26d.html
- Stanford HAI's 2026 AI Index reports frontier models below 20% on paper-scale astrophysics replication, 33% accuracy on UnivEarth, and approximately 38.8% for the best PaperArena agent versus 83.5% for PhD experts. Source: https://hai.stanford.edu/ai-index/2026-ai-index-report/science
- The 2026 survey of more than 300 symbolic-regression methods identifies deterministic, metaheuristic, neural-symbolic and hybrid families, and highlights robustness, high-dimensional scaling, physics-informed structure, uncertainty and benchmark quality as continuing challenges. Source: https://doi.org/10.1007/s11831-026-10681-w

These results do not imply that our approach is superior. They identify measurable gaps and provide external interfaces to target.

## Five distinct qualification planes

### A. Formal correctness

On finite toy worlds, compare every exact likelihood, posterior update, action cost, terminal decision and policy metric against independent exhaustive enumeration and replay.

Targets:
- 100% exact agreement on sealed tiny domains within declared caps.
- zero malformed probability kernels accepted.
- zero false unique identifications on exact observational-equivalence fixtures.
- fail closed on overflow, malformed input, missing branches and resource caps.

### B. Causal discrimination

Use the HIST-CAUSAL model families:
- common cause versus direct transfer;
- open-channel versus sealed-channel response;
- sealed-history candidate versus unobserved leakage;
- local individual memory versus cohort-level effect;
- future-exposure placebo;
- similarity mismatch and dose-zero controls;
- path-equivalent construction/read histories as a separate source-bounded provenance comparison.

Targets:
- empirical false-attribution rate below 1% on the declared negative-control suite, with confidence intervals;
- at least 95% correct abstention on non-identifiable cases;
- mechanism recovery at least 90% on identifiable held-out synthetic cases;
- calibrated posterior probabilities under misspecification and covariate shift.

### C. Active experiment quality

Compare fixed order, seeded random, maximum disagreement, one-step EIG/cost, prior-weighted EIG, finite-lookahead policies, particle methods, and exact finite oracles only on their matched domains.

Targets:
- at least 10% lower median cost-to-target than strongest matched myopic baseline on predeclared multi-step-confounding families;
- paired 95% intervals exclude zero;
- no increase in risk, false claims or abstention;
- report regressions on memoryless/easy cases.

### D. Scientific campaign success

Use a two-gate score like Autonomous Science Bench:
1. discovery: quality of experiments actually measured;
2. learning: predictive ranking of unmeasured designs.

Also score mechanism recovery, intervention cost, uncertainty calibration, model misspecification detection, replication and downstream decision quality separately.

### E. Evidence and physical execution

Use the EmbodiedLab/FIRE-Bench lesson: a correct final answer is insufficient without evidence from what the agent actually did. Require:
- complete, timestamped observation and action ledger;
- action requested versus action realized;
- source and instrument provenance;
- final apparatus/process state;
- replayable evidence trace;
- independent evaluator;
- human escalation where operational safety requires it.

Do not report a run as complete merely because it accumulated partial rubric points.

## Matched-comparison contract

Every comparison must publish:
- exact model, agent, tools, harness and versions;
- task IDs, seed and held-out split provenance;
- action, experiment, token, CPU/GPU, memory and wall-clock budgets;
- failure, retry and abandoned-run accounting;
- initial priors and observation access;
- baseline implementation and tuning budget;
- primary metrics and multiplicity plan;
- per-task and worst-case results, not only a pooled average.

A baseline may be considered beaten only on the same task semantics, information, allowed actions, resource budget and success definition. An exact small-domain oracle is a correctness ceiling, not a claim to beat scalable Bayesian experimental design.

## Immediate integration plan

1. Keep generic policy/search implementations in Symtropy SIM-REL-002/003/013/014/015/017/018/025 and later stateful decision-value work.
2. Keep HIST-CAUSAL as the mechanism-specific challenge suite and claim ledger.
3. Add the v0.2 finite exact SCM fixtures to SIM-REL-017 as a separate challenge family, not merged into other scores.
4. Add an Autonomous Science Bench adapter for discovery/learning two-gate tasks.
5. Add an EmbodiedLab evaluation adapter only after action-realization and evidence-ledger semantics are qualified.
6. Use FIRE-Bench-style independent rediscovery tasks to evaluate the outer research agent and evidence chain.

## Claim boundary

Current v0.2 exact results establish only identifiability properties of six declared finite toy models over a declared set of probes. They do not validate a physical historical influence, a Chain Theory physical mapping, universal relational grammar, or SOTA superiority. All future wins require exact-head receipts, matched baselines, hidden held-out task families and independent reproduction.
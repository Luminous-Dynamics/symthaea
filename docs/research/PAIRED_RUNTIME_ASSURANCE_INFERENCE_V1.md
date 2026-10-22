# Paired Runtime-Assurance Inference v1

**Status:** implemented analysis utility with a synthetic qualification fixture. It is not an external-environment result and does not certify a controller.

## Purpose

A controller comparison is useful only when both policies face the same exogenous conditions and the remaining uncertainty is visible. This utility turns episode-level logs into one reproducible, paired comparison at a time. It is intentionally dependency-free Rust so that a benchmark run does not require a production Python analysis path.

## Input contract

The CLI accepts the benchmark CSV schema already used by the allostatic-control benchmark, including the required columns:

- `scenario`: a stable slug for a task/perturbation condition;
- `seed`: unique episode-pair seed within a scenario and shielded condition;
- `policy`: the policy identifier;
- `shielded`: `true` / `false` or `1` / `0`;
- a user-selected finite numeric outcome metric, such as `task_reward` or `cumulative_exposure`;
- a user-selected binary episode-level violation field, such as `unsafe_episode`.

This v1 parser deliberately accepts only unquoted CSV cells and fails closed on quoting, duplicate policy rows, non-finite metrics, missing headers, missing matched policy/seed pairs, and invalid violation flags. The metric and violation column names must be simple slugs. It resolves input/output paths and refuses to overwrite the source through path aliases. Reports are written to a same-directory temporary file and renamed into place only after every condition is analyzed and the file is flushed. Derived means and bootstrap accumulation use scaled arithmetic and non-finite results are rejected. Do not silently pre-filter episodes to make a comparison pass. If an adapter emits a different or quoted CSV dialect, add a tested adapter rather than loosening this parser invisibly.

Example invocation:

```sh
cargo run --locked --release \
  --manifest-path tools/allostatic-control-bench/Cargo.toml \
  --bin allostatic-paired-eval-bench -- \
  allostatic-control-v1.csv paired-analysis.csv \
  task_reward unsafe_episode global_interval_shield state_action_conditioned_shield 10000
```

Each report row is a distinct `scenario × shielded` condition. The utility never pools those conditions. Both policies must have identical episode-seed coverage for every reported condition.

## Reported estimands

1. **Paired mean delta** and **median paired delta**, always defined as candidate minus baseline. Positive is not automatically better; interpretation depends on the metric. For example, positive reward delta may be desirable while positive exposure delta may be harmful.
2. **95% percentile paired-bootstrap interval** for the mean per-seed delta. Resampling units are complete paired episodes, not control ticks. The PRNG seed is deterministic per scenario/shield setting and is emitted as `bootstrap_seed`; the replicate count is also emitted. Identical input and arguments produce byte-identical output. The interval is an uncertainty estimate under the seed-level sampling design, not a distribution-free safety guarantee.
3. **Observed violation rates** and separate one-sided 95% **exact Clopper–Pearson upper confidence limits** for the baseline and candidate rates. Zero observed violations must not be reported as zero underlying risk.
4. **Exact two-sided McNemar test** for the paired binary episode-violation outcome, using only discordant baseline/candidate episodes. This is a secondary inferential statistic, not a substitute for violation counts, the confidence limits, or practical effect size.

## Experimental rules

For a useful scale check, zero observed violations in 300 independent episodes gives a one-sided 95% exact upper bound of about 0.994%, not 0%. Bounding that same rate below 0.1% requires about 2,995 zero-violation episodes under the independent Bernoulli episode assumption. More runs cannot repair an unrepresentative perturbation panel, correlated samples, a wrong plant model, or an incomplete safety case.

- Choose the primary metric, baseline, candidate, condition panel, and decision threshold before inspecting final-test outcomes.
- Keep calibration, tuning, model selection, and final evaluation episodes disjoint.
- For a final external evaluation, retain at least the protocol's predeclared 300 episode seeds per task/condition/policy, and increase sample size when the risk bound is too wide or the target effect is modest. The CLI enforces matched seeds, not sample adequacy on its own.
- Use common initial conditions and disturbance tapes wherever the environment makes that comparison meaningful. Record any limitation that breaks pairing before result inspection.
- Analyze each task and perturbation condition separately first. Cross-task headline summaries should follow a separate preregistered aggregation method (for example, robust aggregate metrics and task-stratified uncertainty); they are not silently computed by this utility.
- If testing many policy pairs or metrics, preregister one primary comparison or apply a stated multiplicity correction. An isolated p-value does not establish general superiority.
- Keep the raw per-episode file and its SHA-256 digest next to the report. The summary is not a replacement for raw traces.

## Qualification fixture and limits

The repository qualification script generates a deterministic synthetic CSV fixture, runs the CLI twice, checks byte-identical reports, validates the output schema and verifies per-condition coverage. The fixture demonstrates that the tool behaves as intended; it is **not** benchmark evidence for the underlying control method.

The next external milestone is a pinned Safe-Control-Gym adapter that emits the same row contract from the environment's independent evaluator. Only then should the paired analysis be used to compare native controllers, robust shields, MPSC/CBF baselines, and learning-aware methods such as A-SLS. Literature figures from unmatched tasks or hardware remain context, not head-to-head evidence.

## References

- Spoor, Plaat, and Moerland, [Evaluation Metrics for Safe Reinforcement Learning](https://arxiv.org/abs/2609.15315), September 14, 2026. Motivation for reporting safety frequency/severity and per-task/safety-bound outcomes rather than a single average.
- Agarwal et al., [RLiable: Towards Reliable Evaluation & Reporting in Reinforcement Learning](https://research.google/blog/rliable-towards-reliable-evaluation-reporting-in-reinforcement-learning/). Motivation for uncertainty estimates and distribution-aware reporting.
- Srinivasan, Panse, and Chou, [Planning to Learn: Real-Time Robust Adaptive Control on the GPU with Guaranteed Future Learning](https://arxiv.org/abs/2610.20925), October 14, 2026. A modern learning-aware robust-control comparator; this utility does not reproduce A-SLS.

# Multi-Fidelity Architecture Search: Cost-Aware Evaluation Contract v0.1

**Status:** design and scheduling primitive authored; Rust compile/tests and synthetic sweeps NOT EXECUTED; no performance qualification.
**Tracking issue:** [#8994 — cost-accounted multi-fidelity architecture search](https://github.com/Luminous-Dynamics/symthaea/issues/8994)
**Parent program:** [#8888](https://github.com/Luminous-Dynamics/symthaea/issues/8888)
**Search generalization proposal:** [held-out landscape suite v0.1](./ARCHITECTURE_SEARCH_HELD_OUT_LANDSCAPES_V0_1.md)
**Current draft PR:** [#8890](https://github.com/Luminous-Dynamics/symthaea/pull/8890)

## 1. Research question

Can a fidelity-aware searcher identify strong full-fidelity architecture candidates using less total evaluation cost than full-fidelity random search and pinned multi-fidelity baselines—without systematically pruning candidates whose quality only appears at higher fidelity?

This is an empirical question. Lower-fidelity scores can be cheap but misleading. A scheduler is beneficial only if the final full-fidelity result improves at equal total cost, and the ranking failures and workload boundaries are reported.

## 2. Paradigm shift: spend compute as a decision, not as a candidate count

A candidate scored by a zero-cost proxy, a short training run, a partial-context workload, and a complete training run do not have equal costs or equal predictive value. Therefore, number of candidates evaluated alone is not a fair budget.

Measure:
- total and incremental compute/resource units per fidelity;
- high-fidelity outcome per cumulative cost;
- cost spent on failed, invalid, and retried evaluations;
- rate of poor candidates promoted and potentially strong candidates pruned;
- quality uncertainty and rank reversals between fidelity rungs;
- wall-clock/accelerator hours as separate system measures, not synonyms for abstract cost units.

### Fidelity ladder contract

A ladder declares a stable ordered sequence such as:

1. **F0 — static/zero-cost feature:** parameter count, shape constraints, kernel availability, or a genuinely training-free proxy.
2. **F1 — short training:** a small declared token/FLOP budget with the same dataset split and optimizer family.
3. **F2 — intermediate training:** a predeclared larger cumulative training budget.
4. **F3 — full development fidelity:** the complete, pre-registered search-time budget on development/validation data.
5. **Confirmation — final/held-out evaluation:** performed only after candidate selection and analysis freeze; never visible to the searcher.

This sequence is illustrative. An experiment must define its actual levels before running and must not rename one implementation as another fidelity after inspecting results.

The Rust primitive in src/multi_fidelity.rs treats each stage's cost as **cumulative** and charges next cumulative cost minus current cumulative cost when promoting. That accounting is valid only when the lower-fidelity work is a resumable prefix of the higher-fidelity work. If a fidelity change requires a fresh run or distinct hardware workload, record and charge that actual cost separately; do not use the difference formula by default.

## 3. Promotion primitive

The module accepts a single set of observations from one fidelity index. Its proposed default decision rule is:

- Validate that fidelity IDs are unique and cumulative cost strictly increases.
- Reject empty IDs, duplicate candidate observations, mixed fidelity indices, non-finite scores, invalid uncertainty, invalid policy parameters, or cost-accounting overflow.
- Rank candidates by a conservative lower confidence bound: mean score minus confidence multiplier times standard error, for a maximize-is-better objective.
- Promote a predeclared fraction, rounded up.
- Reserve a configurable nonzero share for deterministic seeded exploration among candidates outside the exploitation-ranked prefix, whenever pruning makes exploration meaningful.
- Return the promoted candidate IDs, exploitation/exploration counts, observed candidate count, and estimated incremental cost as an auditable decision.

The uncertainty estimate and confidence multiplier must be selected before the confirmatory run. This scheduler does not calibrate uncertainty, predict the high-fidelity outcome, perform model training, or implement Hyperband/TPE/Bayesian optimization. Those are baseline algorithms for the benchmark, not capabilities to infer from this module.

## 4. Required synthetic multi-fidelity benchmark

Create known-oracle landscapes where every candidate has a full-fidelity ground-truth quality and each fidelity emits a cheaper, imperfect score. Record every candidate/fidelity result and the cost of obtaining it.

### Landscape families

- **Rank-preserving:** lower fidelity is noisy but generally informative.
- **Progressive rank reversal:** intermediate scores reverse candidate order as fidelity increases.
- **Late-emerging quality:** strong candidates start poorly and improve only at higher training budget.
- **Heteroscedastic uncertainty:** low-fidelity uncertainty differs across candidate families.
- **Misleading proxy:** a feature-only prior appears useful on development instances but drifts or reverses on held-out instances.
- **Hard feasibility boundary:** candidates can fail/OOM/time out, and failures have nonzero cost.
- **Miscalibrated uncertainty:** reported standard errors are deliberately too small relative to injected score perturbations; the benchmark must detect under-coverage rather than blindly trust the confidence penalty.
- **Multi-objective fidelity:** quality, latency, memory, and energy proxies have different fidelity errors and Pareto-front shapes.

Freeze the generation formulas and objective oracles for the development suite. Freeze the policy source SHA and config before generating prospective challenge seeds. The challenge results become development data once they have been used to tune the policy; make a new challenge version for replication.

## 5. Baselines

Under equal **total cumulative evaluation-cost budgets** compare:

1. Full-fidelity random search (the reference baseline).
2. Successive halving or Hyperband, with exact implementation and settings pinned.
3. MF-NAS or a faithful reimplementation of the published proxy-guided local-search/multi-fidelity method, when the search space and code are available; document every deviation rather than implying an exact replication.
4. TPE/Bayesian multi-fidelity optimization, only if the version and fidelity inputs are available and reproducibly configured.
5. Proxy-ranked promotion with a predeclared proxy.
6. Symthaea-guided promotion using the frozen policy source SHA.

Where possible, pair methods by the same landscape instance and seed. Count all fidelity evaluations, full candidate retries, invalid outcomes, and hidden restarts; do not charge only successful promotions. Report development/tuning costs separately from challenge-time costs. A system must not receive oracle scores or free proxy labels that are unavailable to the other methods.

## 6. Metrics

### Primary

Pre-register one primary endpoint:
- **Full-fidelity normalized simple regret versus total cumulative cost**, or
- **Area under the best-so-far full-fidelity quality curve versus cost units**.

A candidate is not considered to have a full-fidelity score until the declared full-fidelity stage actually runs. Lower-rung results must not be relabeled as full-fidelity outcomes.

### Secondary

- Full-fidelity optimum/top-K recall where the oracle permits exact calculation.
- Promotions-to-optimum and false-pruning rate of candidates that land in the top-K at full fidelity.
- False-promotion rate for candidates that look strong at low fidelity but finish weak.
- Rank correlation between each fidelity and full fidelity, with uncertainty and per-family results.
- Cost spent on invalid runs, retries, timeouts, and OOMs.
- Hypervolume/regret for declared multi-objective oracles.
- Quality per wall-clock/GPU-hour and observed power/energy where measured credibly.
- Per-family worst-case effects, seed variance, and workload-transfer degradation.

Candidate count and best low-fidelity score are diagnostics, not sufficient primary outcomes.

## 7. Minimum negative controls

The implementation and benchmark must detect:
- mixed-fidelity observations used in one ranking;
- duplicated candidate/fidelity observations;
- an exploration pool that accidentally consists only of exploitation winners;
- promotion-cost overflow or free retry accounting;
- a proxy deliberately anti-correlated with full-fidelity quality;
- shuffled proxy labels falsely improving the result;
- rank reversal between intermediate and full fidelity;
- high-fidelity test leakage into promotion decisions;
- dropped failures or incomplete candidate × seed records;
- output aggregates that do not match raw per-run results.

A negative-control failure invalidates that comparison even if the headline metric improves.

## 8. Statistical and promotion criteria

For inexpensive synthetic qualification, start with 30 paired seeds per landscape/budget and target 100 paired seeds for the confirmatory suite. Estimate variance from development runs; increase repetitions if uncertainty remains material. Report paired effects and 95% bootstrap confidence intervals by declared replication unit, every landscape family, and pooled. Correct exploratory multiplicity.

Provisional engineering target: at equal total cost, at least 10% lower full-fidelity normalized regret (or a pre-registered equivalent hypervolume gain) against the strongest pinned baseline, with a paired 95% interval excluding zero in the favorable direction; no more than a 5% regression on a critical family; and the same directional result on a newly generated challenge version. These thresholds are design goals, not universal laws or evidence already obtained.

The policy is only synthetically qualified after its exact-head test suite passes, raw receipts recompute, controls behave as expected, and the prospective holdout replicates. Synthetic qualification is not a language-model improvement claim.

## 9. Bridge to real architecture search

After the synthetic gate:
- Keep the final test set locked; search can use development and validation data only.
- Fix tokenizer, corpus/splits, architecture config, optimizer, seed plan, precision, and runtime version.
- Define resource levels in cumulative tokens/FLOPs or another defensible physical unit, and record measured cost separately.
- Include a competent dense Transformer control and feasible recurrent/state-space/hybrid candidates. Do not infer that an architecture is available if a compatible kernel/checkpoint is not.
- Compare quality and cost at matched budgets; report each fidelity's rank correlation to final validation performance and final-test quality only after freeze.
- Evaluate workload profiles separately, and report quality, TTFT/prefill, per-token latency, throughput at named concurrency, peak memory, power/energy and invalid-run yield.
- Resolve or fence off [#8308 — learning-signal and task-evaluator validity](https://github.com/Luminous-Dynamics/symthaea/issues/8308) before using language-task results for architecture decisions.

### Industry references

The 2026 paper [MF-NAS](https://doi.org/10.1145/3774940) studies proxy-guided local search followed by higher-fidelity evaluation on NAS-Bench-201. Its results motivate pinned multi-fidelity comparisons; they are not transferable performance evidence for Symthaea. The broader [multi-fidelity optimization benchmark methodology](https://link.springer.com/article/10.1007/s11831-025-10392-8) highlights the need to specify fidelity cost, initialization, and stopping rules.

For deployed serving, [MLPerf Endpoints v1.0](https://mlcommons.org/2026/10/open-submissions-endpoints-v1-0/) provides a separate whole-system performance frontier and power-reporting conventions. Pin its evolving rules and keep serving-system claims distinct from search-policy claims.

## 10. Authored synthetic pilot harness

The new module `src/multi_fidelity_benchmark.rs` connects the promotion scheduler to an exhaustive 256-candidate objective oracle and defines three controlled fidelity regimes:

- `rank_preserving`: cheap signals closely track the full-fidelity quality.
- `rank_reversal`: F0 and F1 are intentionally anti-correlated with final quality.
- `late_emerging_quality`: early scores favor a misleading feature prior, while later fidelities progressively reveal the objective.

The cumulative synthetic cost ladder is 1, 4, 16, 64 units. A deterministic quarter-promotion schedule evaluates all 256 candidates at F0, promotes 64, then 16, then 4 candidates. The total is 832 synthetic cost units: 256 + 64×3 + 16×12 + 4×48. The full-fidelity random baseline uses exactly the same 832-unit budget to run 13 candidates at 64 units each.

The CLI is:
```text
cargo run -p symthaea-architecture-engineering -- multi-fidelity-sweep rank_preserving 100
cargo run -p symthaea-architecture-engineering -- multi-fidelity-sweep rank_reversal 100
cargo run -p symthaea-architecture-engineering -- multi-fidelity-sweep late_emerging_quality 100
cargo run -p symthaea-architecture-engineering -- multi-fidelity-sweep miscalibrated_uncertainty 100
```

The JSON report includes an auditable candidate/fidelity trace for both experiment arms: candidate ID, rung, observed score, reported standard error, synthetic expected score, oracle quality for post-hoc scoring only, noise scale, incremental cost units, and promotion decision. The promotion policy only receives the current fidelity's candidate ID, observed score, and declared standard error; oracle fields are added after that decision and must never be used as policy input.

It also reports F0-to-oracle rank correlation, empirical nominal-95% uncertainty-band coverage, mean absolute standardized residual, paired promotion-versus-random regret, optimum-hit rates, oracle top-10 recall/survival, and exact trace-cost reconciliation. These are synthetic diagnostics; empirical interval coverage is not a formal confidence guarantee. Nine unit tests are authored, including checks that trace costs exactly reconcile to each arm's budget and that deliberately under-reported uncertainty is detectable. **The Rust implementation and tests remain uncompiled and unexecuted.**

A separate Python arithmetic mirror (not Rust execution) checked the intended F0 rank-correlation signs at one seed for the original three regimes: rank-preserving 0.9909, rank-reversal −0.9896, and late-emerging −0.9875. These are generator sanity checks, not search-performance results.

This pilot compares the cost-aware scheduler to full-fidelity random search only. It does not establish an advantage over Hyperband, MF-NAS, TPE/Bayesian optimization, or model-training systems. Those comparisons remain necessary before making general search-policy claims.

## 11. Definition of done

- Promotion primitive tests compile and pass at the exact source revision.
- Unit/property tests validate fidelity identity, uncertainty handling, deterministic selection, exploration boundaries and exact cost arithmetic.
- Synthetic rank-reversal sweeps compare all pinned policies under equal total cost budgets.
- A prospective holdout demonstrates results with raw per-seed data, uncertainty estimates and reproducible receipts.
- Failed/invalid evaluations and retries are accounted for.
- A reviewer accepts the claim boundary.

Until then, status remains **DESIGN_AND_CODE_AUTHORED_NOT_COMPILED_NOT_EXECUTED_NOT_QUALIFIED**.

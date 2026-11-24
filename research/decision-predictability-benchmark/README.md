# Decision-Relevant Predictability Benchmark

**Status: protocol proposal / landing page only. No executable Rust evaluator is currently committed in this directory.**

Parent research protocol: [issue #12546](https://github.com/Luminous-Dynamics/symthaea/issues/12546).

The repository directory currently contains this README only. There is no Cargo.toml, src/ tree, fixture, Cargo.lock or dedicated workflow here. Therefore no Cargo commands or test results are claimed for this benchmark package. The issue's benchmark constitution defines the intended metrics and controls; it is not evidence that the evaluator has been implemented.

## Research question

Can a model preserve decision quality, calibrated uncertainty and useful dynamical information beyond the horizon at which exact pointwise trajectory prediction becomes unreliable, while spending fewer sensing and compute resources than fixed-cadence or prediction-only policies?

Pointwise forecast accuracy, autonomous dynamical fidelity, decision utility, epistemic reliability and full resource cost are distinct evaluation axes. There should be no unqualified composite score.

## Planned evaluator contract

The benchmark constitution in #12546 specifies three lanes:

1. **Known-truth dynamical systems:** Lorenz-63, Rössler, Mackey–Glass, logistic/Hénon maps and stable/periodic/stochastic controls with pinned equations, parameters, solvers, initial conditions and reference quantities.
2. **Decision overlays:** action contracts with hidden evaluator state, candidate-visible observations, frozen losses/deadlines/constraints, paired worlds with observationally equivalent passive streams, and both high-information/zero-decision-value and low-information/positive-decision-value probes.
3. **Irregular and active observations:** missingness, asynchronous channels, delayed/duplicated observations, sensor dropouts, event-triggered sampling and bounded sensor/compute budgets.

Planned metric families include:
- Lyapunov-normalized forecast horizon only where the reference exponent is qualified;
- autonomous statistical/dynamical fidelity after pointwise phase divergence;
- information-set Bayes risk, decision regret, critical-risk violations, delay and correct abstention;
- calibration and accepted-case risk;
- Pareto frontiers over sensing/acquisition, compute, latency, memory, training/update, measured energy and communication.

Planned baselines include persistence, linear autoregression, ESN, NGRC, NGRC-MDS, compatible neural sequence models, HDC/CfC/LTC variants, fixed/event/uncertainty/entropy sampling, decision-focused EVSI and a clearly labelled synthetic oracle-information bound. The benchmark constitution lists versioning, source-family splits, paired seeds, hidden truth and non-anticipativity requirements.

## Existing related implementations — separate owners

- [#9817](https://github.com/Luminous-Dynamics/symthaea/issues/9817) owns the active-sensing / decision-focused value-of-information task contract.
- [PR #9821](https://github.com/Luminous-Dynamics/symthaea/pull/9821) contains a separate Rust active-sensing planner under research/temporal-bench. It estimates expected task-loss reduction, rejects high-entropy but decision-irrelevant observations, checks six sensor-specific resource dimensions, and returns a Pareto frontier with provenance-bound receipts. Its tests were authored in the PR; check its exact-head workflow before citing them as passing evidence. It is not the paired-world evaluator described here.
- [#12407](https://github.com/Luminous-Dynamics/symthaea/issues/12407) owns observation time/mask/modality/event semantics.
- [#12502](https://github.com/Luminous-Dynamics/symthaea/issues/12502), [#12505](https://github.com/Luminous-Dynamics/symthaea/issues/12505), [#12506](https://github.com/Luminous-Dynamics/symthaea/issues/12506) and [#12624](https://github.com/Luminous-Dynamics/symthaea/issues/12624) own known-truth dynamics, Lyapunov qualification and predictor comparisons.
- [#5476](https://github.com/Luminous-Dynamics/symthaea/issues/5476), [#5477](https://github.com/Luminous-Dynamics/symthaea/issues/5477) and [#5478](https://github.com/Luminous-Dynamics/symthaea/issues/5478) own internal value-of-computation calibration, bounded enforcement and causal metareasoning ablations.

Do not create a competing scheduler or claim that PR #9821 supplies the paired-world benchmark, chaotic reference generators, or independent evaluator process.

## Evaluator security boundary

When implemented, candidate-facing code must receive only the observations authorized by the benchmark contract. Evaluator-only latent mechanisms, future truth and loss tables must be held in a separately permissioned process/environment for confirmatory evaluation. Rust type separation in one process is useful for unit tests, but is not a security boundary.

Recorded high-rate traces can support an emulated acquisition model at requested timestamps; interpolation must not be presented as an actual observation. Physical energy claims require policy-controlled measurements on the target hardware, including acquisition, wake, conversion, compute, storage and transmission.

## Implementation status and next gates

Before this landing page can describe an executable benchmark package, the repository needs:
1. an isolated Rust Cargo workspace and reproducible lockfile;
2. candidate-visible and evaluator-only data contracts;
3. finite paired-world fixture and an independent evaluator;
4. known-answer tests for observational equivalence, irreducible abstention risk, decision-changing probes, information-without-decision-value, invalid/late/over-budget probes and no leakage;
5. a focused Rust 1.96.0 workflow for fmt, tests and strict Clippy;
6. a separately permissioned evaluator boundary for confirmatory runs;
7. matched simple/strong baselines and paired uncertainty intervals.

Until those files and workflows exist, this directory is documentation only. The benchmark constitution is a proposal; there are no results, passing tests, SOTA comparisons or physical reliability claims from this directory.

No output associated with this proposal establishes consciousness, causal reconstruction, physical sensor reliability or SOTA.

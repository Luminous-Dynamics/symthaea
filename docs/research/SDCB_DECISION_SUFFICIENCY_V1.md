# SDCB decision sufficiency under mechanism non-identifiability v1

**Status:** Analytic synthetic benchmark extension. The machine-readable fixture is `SDCB_DECISION_SUFFICIENCY_CHALLENGE_V1.json`. It reuses the frozen OU + first-order sensor pole-swap worlds from #14002 / PR #14003. No physical sensor test, QUANT result, production policy qualification, or SOTA result is claimed.

## 1. The new distinction

Mechanism identifiability and decision sufficiency are different properties.

Let \(\theta\) denote a candidate physical/instrument mechanism, and let \(E\) be the complete observation and intervention protocol presently available. Define observational equivalence by

\[
\theta \sim_E \theta'
\quad\Longleftrightarrow\quad
P_\theta(Y_E\in B)=P_{\theta'}(Y_E\in B)
\quad\text{for every measurable }B.
\]

For a zero-mean Gaussian process, the complete finite-dimensional observation law is determined by its mean function and covariance kernel. Equality of a few sampled autocovariance values is not by itself a general proof of equivalence; the frozen pole-swap reference proves equality of the full covariance function under the declared OU + first-order response family.

A mechanism is identifiable within a candidate class only when its observational-equivalence class is a singleton under the declared protocol. An estimator selecting one member of a non-singleton class has not identified that member, even if its point estimate is stable or its predictive fit is excellent.

Now let \(L(a,\theta)\) be the declared terminal decision loss. A mechanism class \(C\) can be decision-sufficient without being identifiable when

\[
\bigcap_{\theta\in C}
\operatorname*{argmin}_{a\in\mathcal A} L(a,\theta)
\ne \varnothing.
\]

In words: at least one terminal action is optimal for every mechanism still compatible with the evidence. The decision may be settled while the physical explanation remains unresolved.

The reverse case is also important: if the optimal action differs across members of the equivalence class, an experiment that genuinely splits that class may have decision value.

## 2. Robust and Bayesian estimands remain separate

For an equivalence class \(C\), the prior-free robust terminal risk is

\[
R_{\rm rob}(C)=
\min_{a\in\mathcal A}\max_{\theta\in C}L(a,\theta).
\]

For a probe \(e\) with possible outcomes \(o\), let \(C_{e,o}\subseteq C\) contain the mechanisms consistent with that outcome. With guaranteed outcome support and no probabilities, the worst-case continuation risk is

\[
R_{\rm rob}(C\mid e)=
\max_{o:\,C_{e,o}\ne\varnothing}R_{\rm rob}(C_{e,o}).
\]

A robust decision-value improvement is the reduction from \(R_{\rm rob}(C)\) to \(R_{\rm rob}(C\mid e)\). The exact finite-horizon, outcome-adaptive policy value and resource-vector feasibility remain owned by #13567. This fixture does not implement a second planner.

For a defensible prior \(p(\theta)\) and a fully specified likelihood, Bayesian Bayes risk is

\[
R_{\rm B}(p)=
\min_{a\in\mathcal A}
\mathbb E_{\theta\sim p}[L(a,\theta)].
\]

The expected value of sample information for probe \(e\) is

\[
\operatorname{EVSI}(e)=R_{\rm B}(p)
-\mathbb E_{o\sim P(o\mid e,p)}
\left[R_{\rm B}(p(\theta\mid o,e))\right].
\]

The fixture's 0.5/0.5 prior is an evaluator-created synthetic prior, not an empirical belief. Bayesian results must not be blended with the prior-free robust result.

Experiment units, reference units, elapsed time, compute and energy are distinct resource dimensions. Treat them as hard constraints or report a Pareto frontier. Do not subtract unlike resource units from terminal loss without an explicit task-owned utility conversion.

## 3. Frozen challenge tasks

The JSON fixture references the exact source worlds and probe families in `SDCB_RESPONSE_IDENTIFIABILITY_CHALLENGE_V1.json`.

### A. Non-identifiable but decision-sufficient

Both pole-swap worlds have the same unique optimal action, `apply_shared_output_policy`, with zero terminal loss. `disable_and_escalate` incurs loss one in either world.

The correct policy is to stop without acquiring more information. Passive observations, upstream pole-symmetric excitation, sensor-inlet calibration, an independent reference, and a nuisance probe all have zero decision value for this task. A positive-cost calibration performed solely to label an already decision-irrelevant physical pole is a benchmark failure.

### B. Non-identifiable and decision-sensitive

The terminal action differs across worlds:

- World A (slow physical process, fast sensor): `treat_as_slow_physical_process`.
- World B (fast physical process, slow sensor): `treat_as_fast_physical_process`.

Without a disambiguating observation the prior-free minimax loss is one; under the fixture's synthetic equal prior the Bayes risk is 0.5. Passive output and upstream pole-symmetric excitation leave both worlds in one equivalence class and give zero EVSI.

A traceable sensor-inlet step separates the two worlds and reduces both risks to zero. The independent physical reference is also discriminating but exceeds the declared reference-resource budget in this particular fixture. The correct feasible probe is therefore the sensor-inlet step, under the exact synthetic cost contract.

### C. High-entropy nuisance control

The nuisance bit has entropy one bit, but its outcome distribution is identical in both worlds. Its mutual information with the mechanism identity is zero, it leaves the equivalence class unchanged, and its EVSI is zero.

This is a direct counterexample to using entropy reduction as a substitute for decision value.

## 4. Existing architecture and ownership

| Component | Owns | Must not be treated as |
|---|---|---|
| #14002 / #14003 | Physical + sensor response equations, analytic references, pole-swap fixture | A real sensor calibration |
| #12454 / #12457 | Fixed finite-action expected-mean signatures and bounded separation cost under independent known-variance Gaussian replicates | A general covariance-kernel or serial-process equivalence oracle |
| #13567 | Exact finite-horizon sequential Bayesian and prior-free robust policy values | A production scheduler or experiment-execution permission |
| #13739 | Existing Symthaea selector adapters and matched selector tournament | A second planner |
| #14000 / #14001 | Probability enclosures and conservative finite action-value bounds | Evidence authentication or physical action authority |
| #13805 | Independently verified receipt and atomic belief-transition authority | A consumer that accepts a likelihood result as a receipt |
| #13691 | End-to-end active scientific intelligence benchmark protocol | A metrology standard or physical test |

**Critical #12454 boundary:** its present reference partitions candidate mechanisms by finite expected-mean signatures over a frozen action set and computes a fixed replication schedule for independent, known-variance Gaussian replicates. It does not establish equivalence of general Gaussian processes, unknown-variance observations, serially correlated measurements, or arbitrary covariance kernels. Do not feed sampled autocovariance values into it as if they were independent scalar observations. An adapter must supply and justify each action's expected means, noise variance, independence semantics, and replicate cost. Otherwise report `NOT_MEASURED` or `UNSUPPORTED`.

If a future application requires separation-cost analysis for covariance-process observations, specify and validate that mathematical contract separately. Do not silently widen #12454.

## 5. Benchmark scorecard

Score separate dimensions; do not create one composite score that can hide a false causal claim.

| Metric | Frozen-control target | Interpretation |
|---|---:|---|
| False unique pole assignment under passive or upstream-equivalent evidence | 0 | Hard failure |
| Correct equivalence-class report on non-identifiable controls | 100% | Exact synthetic known-answer |
| Correct action on decision-invariant task | 100% | Stop; do not pay for irrelevant identification |
| Unnecessary positive-cost probes on decision-invariant task | 0 | Hard failure |
| Correct action after supported sensor-inlet disambiguation | 100% | Exact synthetic known-answer |
| Nuisance-probe EVSI | 0 | No world information, no decision value |
| Unsupported independence/noise assumptions | 0 | Hard failure |
| False evidence-receipt acceptance or unauthorized mutation | 0 | Hard failure |

For held-out, stochastic and physical evaluations, additionally report:
- equivalence-class/set coverage and false singleton rate;
- selective risk versus coverage, with abstention calibration;
- terminal Bayes risk and prior-free minimax loss separately;
- EVSI calibration, regret against #13567, and the no-probe baseline;
- cost-to-decision and cost-to-identification separately;
- resource-vector Pareto frontiers (reference use, experiment units, elapsed time, compute, energy);
- predictive log score / CRPS, 90%/95% coverage and interval width;
- device-, family-, site-, pollutant-, acquisition-window- and temporal-regime-stratified results;
- drift detection delay, false alarms, and recovery after regime shifts;
- all unsupported, failed, missing and abstained cases with full denominators.

For a superiority claim, preregister one primary endpoint, use paired held-out clusters, report paired confidence intervals, compare at equal budgets and require independent reproduction. A target such as 10% lower terminal decision loss per equal reference cost is only a hypothesis until a powered, matched comparison supports it. Preserve the zero-false-mechanism-claim gate regardless of predictive accuracy.

## 6. Numerical qualification and experiment selection

The likelihood provider must keep heuristic estimates separate from certified probability enclosures. A finite expected-utility interval is only as defensible as its input probability intervals. If integration fails to achieve the requested width but returns a valid enclosure, carry its actual width forward; refine or abstain rather than relabeling it as a point estimate.

Numerical refinement and physical experiment selection are separate:
- #14000 can refine the numerical probability enclosure when it can change the action-value ordering.
- #13567 evaluates the value of an actual experiment sequence under the declared information and resource protocol.
- #13805 alone remains responsible for accepting independently verified evidence receipts and applying belief transitions.

## 7. Comparison with current research and industry

SDCB should compete on multiple Pareto fronts, not one leaderboard.

**Multivariate normal computation.** Genz–Bretz, Miwa, TVPACK, and Peterlin's 2026 doubly blocked Genz method are relevant numerical comparators. Peterlin reports very large throughput gains for large-dimensional workloads under its tested conditions, with a focus on positive-definite covariance and parallel/cache/SIMD optimization. Reproduce the published workloads on matched hardware and resource budgets. Keep throughput distinct from validated enclosure containment, singular-support handling, tail robustness, and decision correctness. An unvalidated fast point estimate is not a substitute for a certified enclosure.

**Scientific-agent capability.** ScienceAgentBench, DiscoverPhysics, SciAgentArena and related benchmarks test parts of scientific coding, discovery and agent workflow. They are useful external capability comparisons, but they do not establish metrological traceability, physical sensor calibration, or the correctness of a measurement operator. SDCB's contribution should be evaluated on the scientific decision/evidence chain in addition to agent task completion.

**Measurement calibration and field transfer.** Reproduce Kok–Tabandeh's six short-collocation numerical scenarios under their published assumptions before claiming extensions. Use CEN/TS 17660-1 and -2 and the relevant JCGM/GUM editions for the device-level conformance and uncertainty lanes. The October 2026 London field-calibration study is a separate external cohort for pollutant- and time-dependent transfer, not a QUANT subset or an SDCB result.

References:
- Peterlin (2026), *A doubly blocked Genz method for fast computation of multivariate normal probabilities*: https://link.springer.com/article/10.1007/s11222-026-10749-4
- ScienceAgentBench: https://arxiv.org/abs/2410.05080
- DiscoverPhysics: https://arxiv.org/abs/2606.11199
- SciAgentArena: https://arxiv.org/abs/2606.01593
- Kok & Tabandeh (2026): https://doi.org/10.1002/qre.70319
- QUANT dataset record: https://doi.org/10.1038/s41597-024-03767-2
- JCGM publications: https://www.bipm.org/en/committees/jc/jcgm/publications/

## 8. Next implementation sequence

1. Validate the JSON fixture against the original pole-swap source challenge and check its equivalence partitions.
2. Implement a standalone deterministic checker that verifies the decision-invariant, decision-sensitive, and nuisance known answers without requiring Rust/Cargo or a GitHub runner.
3. Add #13567 adapter inputs only after checking its current executable schema and exact owner branch; do not invent an API.
4. Integrate through #13739's existing selectors and report HDC disagreement, Spark EIG, one-step EVSI and exact adaptive-oracle regret on the same tasks.
5. Extend the benchmark with noisy near-equivalence, model-family misspecification, response-time drift, hidden reference-chain changes and multi-sensor common causes.
6. Acquire the actual QUANT files with immutable checksums and parse them locally. Keep the Manchester DuplicateRef cohort geographically bounded until that acquisition confirms its complete contents.
7. Run physical high-rate step/frequency tests with a traceable inlet stimulus and independently characterized reference; hourly public data cannot substitute for sub-hourly response validation.

## 9. Claim boundary

This is a synthetic analytic benchmark and integration specification. It does not demonstrate a new physical law, a calibrated device, metrological traceability, a real-data QUANT result, a production scientific agent, SOTA performance, or superiority over industry. Those claims require the matched, independently reproduced executions described above.

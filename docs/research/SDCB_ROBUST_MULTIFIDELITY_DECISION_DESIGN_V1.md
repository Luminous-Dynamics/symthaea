# Robust Multi-Fidelity Decision Design for SDCB v1

**Status:** exact finite one-step reference and research protocol
**Parent:** RPSD / DSSI / R-IDeA
**Implementation:** [scripts/research/sdcb_multifidelity.rs](../../scripts/research/sdcb_multifidelity.rs)
**Episode schema:** [SDCB_MULTIFIDELITY_EPISODE_V1.schema.json](SDCB_MULTIFIDELITY_EPISODE_V1.schema.json)
**Example:** [SDCB_MULTIFIDELITY_EPISODE_V1.example.json](SDCB_MULTIFIDELITY_EPISODE_V1.example.json)
**Expected result:** [SDCB_MULTIFIDELITY_EPISODE_V1.expected.json](SDCB_MULTIFIDELITY_EPISODE_V1.expected.json)

## 1. Paradigm shift: choose the fidelity, not just the experiment

Scientific systems rarely choose only a design point. They also choose how to measure or simulate it: a cheap proxy, a coarse simulation, a screening assay, a lab measurement, a high-resolution solver, or a physical validation. These information sources differ in cost, delay, noise, bias and failure modes.

Conventional multi-fidelity Bayesian optimization (MFBO) studies how to combine information sources of different accuracy and cost. A major practical lesson from materials/molecular MFBO research is that the benefit is conditional: approximate-source informativeness and its cost relative to high fidelity determine whether multi-fidelity helps. It should not be assumed that a cheap proxy is always useful.

SDCB's next question is more decision-specific:

**Which design-at-which-fidelity yields the greatest defensible change in a downstream decision, when each fidelity can fail differently under plausible model shifts, and the resource dimensions cannot be collapsed into one arbitrary price?**

RPSD already provides finite-scenario robust decision value. This increment makes fidelity an explicit decision variable and retains cost as a componentwise resource vector.

## 2. Mathematical contract

Let p(h) be the declared hypothesis prior, S a finite set of complete plausible model scenarios, and U_s(a,h) the task utility of action a under scenario s and hypothesis h.

Each candidate option q=(design, fidelity) supplies a scenario-specific observation model P_s(o | h,q). A deterministic post-observation policy pi maps outcomes to actions. The mapping is shared across all hidden scenarios; the decision-maker is not told which scenario is true.

The current robust value is:

\[ V_0=\max_{a\in A}\min_{s\in S}\sum_h p(h)U_s(a,h). \]

The robust post-measurement value for one fidelity option is:

\[ V_q=\max_{\pi:O_q\to A}\min_{s\in S}\sum_h\sum_o p(h)P_s(o\mid h,q)U_s(\pi(o),h). \]

Robust decision gain is RIG(q)=V_q−V_0. The option's resource cost is a vector:

\[ C(q)=(money,\ instrument\ time,\ compute,\ energy,\ldots). \]

Option q dominates q' when RIG(q) is no smaller, every resource coordinate is no greater, and at least one comparison is strict. The frontier is therefore:

\[ \mathcal P=\{q:\nexists q'\text{ that dominates }q\}. \]

Given owner-declared hard resource caps B, filter to C_j(q)≤B_j for every resource j. Do not invent exchange rates between money, physical time, compute and energy. If an application later requires a scalar price, the owner must provide and version that conversion separately.

## 3. Exact reference implementation

The Rust harness composes the RPSD robust finite-scenario oracle rather than forking its math. It:

- evaluates each experiment/fidelity option with its own likelihood matrices for every complete scenario;
- requires one shared outcome-to-action policy across hidden ambiguity scenarios;
- reports every scenario's policy value, robust post-measurement value and gain;
- retains money, instrument minutes, compute seconds and watt-hours as independent resource coordinates;
- calculates the exact finite Pareto frontier over the supplied options;
- applies componentwise resource caps and selects the highest robust decision value among feasible options;
- uses deterministic tie-breaking and rejects duplicate design/fidelity pairs, invalid costs, missing scenarios and malformed likelihoods.

The reference bounds a fidelity option set at eight entries and inherits RPSD's caps of eight scenarios, five actions and four outcomes. It solves a one-step decision. It does not approximate a large continuous design space or run a sequential multi-fidelity Gaussian process.

## 4. Frozen counterexample: nominally good low fidelity fails under shift

Two hypotheses are equally likely. The two actions are correct when they match the hidden hypothesis; the current robust decision value is 0.50.

The same physical design is offered at four information-source options:

| Fidelity | Nominal signal accuracy | Shifted-scenario signal accuracy | Robust gain | Money | Lab min | Compute s | Energy Wh |
|---|---:|---:|---:|---:|---:|---:|---:|
| low | 70% | 30% (reversed) | 0.00 | 1 | 2 | 0.1 | 0.01 |
| medium | 75% | 70% | 0.20 | 2 | 5 | 0.5 | 0.10 |
| high | 90% | 85% | 0.35 | 6 | 20 | 2.0 | 0.50 |
| medium-expensive duplicate | 75% | 70% | 0.20 | 4 | 8 | 1.0 | 0.20 |

The low-fidelity source looks useful in the nominal scenario. Under the second plausible scenario its signal is reversed. Since the policy cannot see the hidden scenario, it cannot exploit the signal reversal; the maximin policy falls back to a constant action and returns the 0.50 baseline. Its robust gain is zero.

The medium option provides a 0.20 robust gain, and the high option provides 0.35 but costs more on every resource axis. The expensive medium duplicate has the same robust gain as medium and is strictly more expensive on all four resource dimensions, so it is dominated.

The unconstrained Pareto frontier is {low, medium, high}: low remains as a cheap fallback, medium is the middle trade-off, and high buys additional robust decision value at higher cost. Under the example caps (money≤3, lab≤8 min, compute≤1 s, energy≤0.2 Wh), high is excluded and medium is the highest-value feasible option.

This is a hand-computable finite example, not evidence that a real low-fidelity tool reverses 70/30 or that a physical system has been calibrated.

## 5. Relation to R-IDeA, RPSD and DSSI

| Method | Main quantity | What this increment adds |
|---|---|---|
| Traditional MFBO | Predictive/optimization value from multiple information sources, often cost-aware | Adds finite-scenario robust decision gain and outcome-to-action policies |
| R-IDeA | EIG × representativeness factor × proxy de-amplification factor | Allows the target to be measured at different fidelity levels, each with its own likelihood assumptions and resource profile |
| RPSD | Worst-case decision value across explicit ambiguity scenarios | Adds a fidelity axis and a componentwise resource Pareto frontier |
| DSSI EVSI / non-myopic oracle | Expected decision improvement and sequential policy value | Defines the exact one-step robust decision reference each fidelity option must beat |

Do not multiply R-IDeA, EVSI and robust gain into one composite score. They answer different questions. Report acquisition score, decision gain, scenario values, calibration/generalization error and resource vector independently.

## 6. Industry and research baselines

Minimum comparator set for tasks that genuinely have multiple information fidelities:

1. High-fidelity-only design, with the same final decision budget.
2. Low-fidelity-only design, with its model-bias and calibration errors exposed.
3. Fixed fidelity funnel (cheap screening followed by a fixed high-fidelity confirmatory stage).
4. Cost-normalized EIG / mutual-information baseline.
5. R-I and R-IDeA where a target design distribution and a trained proxy are available.
6. Cost-aware MFBO / multi-fidelity entropy search, using an established implementation and pinned version.
7. Multi-objective multi-fidelity optimization such as qMFMES or equivalent when the goal is a Pareto front rather than a single decision.
8. DSSI EVSI and RPSD maximin as decision-focused exact finite comparators.

Published guidance explicitly recommends checking the low/high fidelity cost ratio and informativeness before adopting MFBO; recent work does not support an assumption that MFBO always outperforms single-fidelity BO. For the materials/molecular track, use the published COF, molecular-polarizability and solvation-energy benchmarks with upstream code and splits. Keep application-specific tasks such as physical ISRU, cell-fabrication, nuclear measurement and chaotic-system control separate rather than pretending one dataset is universal.

## 7. Benchmark protocol

Create paired tasks with the same underlying scientific target but multiple fidelity options. The scenario generator should vary:

- fidelity bias, calibration drift and heteroskedastic noise;
- high/low-fidelity correlation, including nonlinear and regime-dependent correlation;
- observation latency, batch/parallel availability and failed runs;
- target-distribution shift between design/training data and deployment/test data;
- missing or unreliable fidelity sources and common-cause errors;
- cost ratios across instrument time, money, compute, energy and sample consumption;
- whether a fidelity comparison itself requires a calibration experiment.

Candidate systems must receive the same initial evidence, allowable design space, budget caps, stop rule and authority. Evaluators must retain latent scenario identity and truth outside candidate-visible interfaces.

Report separately:

- robust gain and all per-scenario policy values;
- decision regret relative to the information-set exact oracle;
- target-distribution prediction loss and calibration by fidelity;
- Pareto hypervolume (frozen reference point), IGD, frontier coverage and diversity;
- cost and resource usage on each axis, not only an arbitrary combined price;
- fidelity calibration cost, time-to-decision, failure rate and unsafe action rate;
- performance with nominal calibration and under held-out fidelity shifts.

Do not tune likelihoods, discrepancy models, cost ratios or scenario sets on hidden test tasks. Use paired task seeds, family-level confidence intervals and complete cost accounting. Include negative controls where low-fidelity information is anti-correlated or misleading under a plausible shift.

## 8. Research targets

These are proposed project targets, not published standards or achieved results:

- 100% agreement with the exact finite robust oracle on all analytic fidelity fixtures;
- zero selection of a nominally attractive fidelity when its worst-case net decision gain is non-positive and a feasible positive-gain alternative exists;
- zero hard resource-cap violations and zero safety-floor breaches on frozen negative controls;
- on held-out multi-fidelity tasks, at least 10% lower decision regret than a fixed-funnel and EIG-per-cost baseline, without worsened critical safety outcomes;
- improve high-fidelity target prediction/generalization at equal total resource cost, not merely increase the count of cheap low-fidelity samples;
- match or exceed qMFMES / established MFBO on a preregistered subset only after identical budgets, source access, priors and evaluation splits;
- report performance under low/high-fidelity correlation sweeps, including failure regions where MFBO should abstain or revert to high-fidelity-only.

## 9. Next implementation steps

1. Add calibration receipts for each fidelity: source identity, calibration split digest, model version, residual distribution, supported operating regime and validity window.
2. Add fidelity discrepancy models with uncertainty, not a single point correction.
3. Extend from one-step options to upgrade policies such as low→medium→high, with the joint likelihood of correlated observations and explicit option to stop.
4. Compare exact small finite policies to greedy, EIG, R-IDeA and MFBO policies.
5. Add multi-objective utility/fidelity control and reliability/availability of the instrument itself.
6. Integrate HDC/CfC only as a representation or amortized policy proposal; evaluate against the exact oracle and calibrated source models.

## 10. References

1. Sabanza-Gil et al. (2025), *Best practices for multi-fidelity Bayesian optimization in materials and molecular research*, Nature Computational Science 5, 572–581. https://www.nature.com/articles/s43588-025-00822-9
2. Tang, Sloman & Kaski (2026), *Representative, Informative, and De-Amplifying: Requirements for Robust Bayesian Active Learning under Model Misspecification*, AISTATS 2026, PMLR 300, 3016–3024. https://proceedings.mlr.press/v300/tang26d.html
3. McDonald et al. (2025), *Bayesian Optimization over Multiple Experimental Fidelities Accelerates Automated Discovery of Drug Molecules*, ACS Central Science 11, 346–356. https://pmc.ncbi.nlm.nih.gov/articles/PMC11869128/
4. IJCAI 2026, *Adaptive Experimental Design to Accelerate Scientific Discovery and Engineering Design*. https://ijcai.org/proceedings/2026/904
5. Atinary / EPFL / Cambridge MFBO implementation and datasets: https://github.com/atinary-technologies/chem-MFBO
6. R-IDeA authors' released implementation: https://github.com/TrbingWY/robust-boed

## 11. Claim boundary

The Rust reference contains 13 authored tests, but has not been compiled or executed in this environment. Its per-fidelity likelihood matrices are supplied fixture inputs; it does not train a multi-fidelity surrogate, estimate calibration from real data, execute a laboratory measurement or benchmark against BoTorch/qMFMES. The result is an exact finite decision-oracle extension, not a claim of SOTA performance.
# SEMI-BENCH-001 — 2026 multi-fidelity optimization and high-fidelity evaluation efficiency

**Review date:** 2026-11-22  
**Program:** [SEMI-DTCO-001 #13172](https://github.com/Luminous-Dynamics/symthaea/issues/13172)  
**Dedicated benchmark:** [SEMI-MF-OPT-001 #13175](https://github.com/Luminous-Dynamics/symthaea/issues/13175)

The competitive question is not "how many simulator runs can a model avoid?" It is "how cheaply can we discover a high-quality, independently verified Pareto set while preserving hard constraints and calibrated uncertainty?"

## 1. What recent work already reports

### Multi-fidelity analog circuit optimization

Cornetta et al., *Multi-Fidelity Surrogate Models for Accelerated Multi-Objective Analog Circuit Design and Optimization*, Electronics 15(1), 105; published 25 December 2025 and appearing in the 2026 volume:
https://doi.org/10.3390/electronics15010105

The article evaluates four CMOS op-amp topologies with NSGA-II, NSGA-III, SPEA2 and MOEA/D under a common evaluation configuration. It reports:
- Surrogate-Guided Optimization replaced approximately **96.5% of SPICE calls** with model inference;
- total simulation time was reduced by approximately **20×**;
- adaptive Multi-Fidelity Optimization used roughly **90% fewer SPICE calls** while preserving close agreement with the study's ground-truth Pareto fronts.

These are paper-reported results for four topologies and the authors' exact protocol, not a universal guarantee.

### ML-driven DTCO for advanced nodes, CFET and multi-row-height cells

*ML-driven Design Technology Co-Optimization Framework for Advanced Technology Nodes*, ASP-DAC 2026:
https://doi.org/10.1109/ASP-DAC66049.2026.11420446

The public abstract reports automatic technology-parameter exploration, standard-cell library generation and chip/block PPA exploration, support for CFET and multi-row-height cells, and **5× sampling efficiency** for predicting the direction and magnitude of PPA changes. The exact circuits, split details and compute accounting must be recovered before direct replication.

### Trust-aware LLM-assisted Bayesian optimization

Zhou et al., *Trust-aware LLM-assisted Bayesian optimization for efficient design space exploration*, VLSI Design 110 (2026), article 102782:
https://doi.org/10.1016/j.vlsi.2026.102782

The abstract describes a trust-control mechanism around LLM-assisted Bayesian optimization and evaluation on a configurable digital baseband generator and a systolic-array NPU. It does not establish direct performance on TCAD or fab-process optimization. Compare it as a method family, not as evidence that LLM proposals are inherently trustworthy.

### Semiconductor manufacturing equipment multi-fidelity optimization

Wang et al., *Design optimization of semiconductor manufacturing equipment using a novel multi-fidelity surrogate modeling approach*, Optimization and Engineering 27 (2026), 1451–1479:
https://doi.org/10.1007/s11081-025-10059-0

This is a relevant equipment-design method reference. The visible abstract does not provide enough common numerical baselines for a ranked comparison; obtain the full protocol before ranking.

### Multi-circuit Transformer surrogate for analog sizing

Sun et al., *Transformer-based surrogate modeling for multi-circuit analog sizing via Bayesian optimization*, VLSI Design (November 2026), article 102809:
https://doi.org/10.1016/j.vlsi.2026.102809

The abstract describes a reusable multi-circuit Transformer surrogate integrated with Bayesian optimization, followed by accurate LTspice validation of promising candidates. More detailed matched-budget metrics need manuscript review.

### Robustness-aware DTCO for computing-in-memory

Lee, Lim & Yu, *Robustness-Aware Multi-Objective Design Technology Co-Optimization Framework for Computing-in-Memory Using Surrogate-Assisted SPEA2*, IEEE Access 14 (2026):
https://doi.org/10.1109/ACCESS.2026.3703077

The publisher record reports off-grid normalized errors below 5% across four objectives and simulated FeFET-CIM macro results. Treat these as simulation-specific, paper-reported references, not fabricated-silicon results or a universal Pareto baseline.

## 2. The paradigm shift: verified knowledge per expensive evaluation

A conventional surrogate loop asks: "Can we predict the high-fidelity output?"

A stronger engineering loop asks:
1. Which candidate is most likely to improve the *verified* Pareto set?
2. What is the uncertainty of the prediction in the current design/process regime?
3. How much will the next fidelity level reduce that uncertainty?
4. Which candidates are near a hard constraint or Pareto boundary and therefore require expensive confirmation?
5. How often does early rejection discard a candidate later found to be Pareto-optimal?
6. When does the surrogate stop being trustworthy and explicitly defer to TCAD, SPICE, route or measurement?

The unit of progress is **verified Pareto improvement per expensive evaluation**, not surrogate accuracy or the fraction of simulations skipped.

## 3. Matched benchmark protocol

Compare under identical budgets:
- random search / space-filling DOE;
- high-fidelity-only search;
- low-fidelity-only surrogate search;
- uncertainty-triggered high-fidelity verification;
- trust-region Bayesian optimization;
- multi-fidelity Bayesian optimization / knowledge gradient;
- ensemble disagreement plus diversity acquisition;
- NSGA-II, NSGA-III, SPEA2 and MOEA/D;
- trust-aware LLM proposals with an independent verifier;
- Symthaea candidate generation and acquisition as a challenger.

Freeze topology, design variables, bounds, units, PDK/model, operating corners, hard constraints, tool/script digests, train/validation/test split by complete circuit or design family, high-fidelity evaluation budget and stopping rules. A random row split is not a design-transfer result.

## 4. Required metrics

**High-fidelity evaluation efficiency:** high-fidelity calls to reach a preregistered hypervolume target; SPICE/TCAD/detailed-route evaluations avoided; actual compute and wall-clock including surrogate training; amortization point; verified Pareto gain per high-fidelity evaluation.

**Pareto fidelity:** normalized hypervolume ratio against a matched high-fidelity reference; IGD+ / epsilon indicator where defensible; Pareto recall and false-pruning rate; objective spread and per-topology coverage; final high-fidelity re-evaluation of every reported Pareto candidate.

**Trust calibration:** prediction interval coverage and width; residual distributions by fidelity transition; calibration by held-out family/corner; false-safe rate (surrogate says feasible, high-fidelity says hard constraint violated); false-pruning rate; selective risk versus candidate coverage; out-of-domain detection and abstention.

**Physical-design quality:** final detailed-route WNS/TNS, area, power and routability; PVT/process corners and worst-case/CVaR; DRC or other declared hard-constraint violations; run failures and non-convergence. HPWL or predicted PPA is a diagnostic, not the final result.

## 5. Research targets (not achieved results)

| Target | Research goal | Guardrail |
|---|---:|---|
| SPICE-call reduction | ≥90% fewer calls | high-fidelity Pareto hypervolume ≥99% of matched reference |
| Stretch efficiency | Approach 96.5% fewer calls / 20× simulation speed | full final-candidate re-evaluation |
| DTCO sampling | ≥5× sampling efficiency over preregistered DOE | no held-out-design or hard-constraint regression |
| Uncertainty calibration | declared 90% / 95% interval coverage | report interval width and per-regime calibration |
| Feasibility trust | no increase in false-safe predictions | uncertain candidates require higher-fidelity evaluation |
| Transfer | at least two independent design families | full-family holdout |

These are research targets, not current Symthaea results.

## 6. Luminous execution sequence

1. Bind candidate, model, tool, PDK, configuration, parent and output digests with the Rust DTCO lineage validator.
2. Reproduce the analog/SPICE study where source rights and circuit files allow it.
3. Add a deterministic high-fidelity acquisition decision oracle with a finite run budget and explicit constraints.
4. Compare DOE, evolutionary, Bayesian, ensemble, trust-aware LLM and Symthaea candidate proposers under identical budgets.
5. Re-run Pareto candidates at final SPICE / OpenROAD detailed-route fidelity; retain rejected and failed runs.
6. Extend to TCAD or equipment models only when the exact source model, solver and reference measurement are lawfully available.

Symthaea proposes and selects. Symtropy replays scenarios. Mycelix retains immutable candidate/run/result lineage. Sol-Atlas binds exact literature, PDK, tool and benchmark revisions. None may self-certify physical truth.

## Claim boundaries

A 90% SPICE-call reduction on open analog circuits is not a 90% manufacturing-cycle-time reduction. A 20× circuit-simulation speedup is not 20× fab throughput. Surrogate-guided Pareto improvement is not fabricated-silicon performance, yield, process qualification or equipment readiness. Those claims require their own independent physical evidence.


## Executable high-fidelity acquisition baseline

The first Rust acquisition oracle for [SEMI-MF-OPT-001 #13175](https://github.com/Luminous-Dynamics/symthaea/issues/13175) is now in `crates/domains/symthaea-semiconductor/src/benchmark/mf_acquisition.rs`.

For each candidate, the baseline score is:

`risk_adjusted_expected_HV_gain + exploration_weight × calibrated_epistemic_stddev × frontier_relevance`

It exactly enumerates feasible subsets for up to 18 candidates under an evaluation-cost budget. Input is bound to prediction-packet, model, uncertainty-calibration and objective-specification digests. Deterministic tie-break: maximum utility, then lower cost, then lexicographic candidate IDs.

The synthetic known-answer fixture selects B/E under budget 13 with modeled utility 0.3115. This only verifies scheduler semantics; it does not run SPICE/TCAD/route or demonstrate a Pareto improvement. The next benchmark step is comparison against greedy gain/cost, random/DOE, uncertainty-only, diversity-only, constrained BO, and high-fidelity-only under matched simulation budgets.

Hosted CI [#44989958104](https://github.com/Luminous-Dynamics/symthaea/actions/runs/44989958104): 43/43 Rust tests, 46/46 Python reference tests, fmt, Clippy and four Rust CLI report parity checks passed.


## Verified budgeted multi-fidelity acquisition baseline

The Rust crate now contains `mf_acquisition.rs` and CLI `semi-mf-acquire`. It exactly selects a candidate subset under a high-fidelity cost budget (up to 18 candidates) using:

`expected_feasible_hv_gain + exploration_weight × epistemic_stddev × frontier_relevance`

The acquisition subject binds the prediction packet, model, uncertainty-calibration artifact and objective specification by SHA-256. The output is a scheduling recommendation only; it does not execute a simulator or approve the design.

On the frozen five-candidate synthetic fixture, budget 13 selects B/E (cost 13; modeled utility 0.3115). This is only a known-answer result.

Hosted run [#44990125209](https://github.com/Luminous-Dynamics/symthaea/actions/runs/44990125209) passed **43/43 Rust tests**, **46/46 Python reference tests**, rustfmt, Clippy and all seven CLI/report comparisons (three Python and four Rust).

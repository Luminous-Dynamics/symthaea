# BIOPROTEIN-DECISION-REGRET-001: compare research-selection strategies under a shared budget

**Status:** synthetic-first benchmark contract and executable Rust evaluator.  
**Parent contracts:** [net EVSI PR #8098](https://github.com/Luminous-Dynamics/symthaea/pull/8098), [joint-scenario robustness PR #8114](https://github.com/Luminous-Dynamics/symthaea/pull/8114), [tracking issue #8115](https://github.com/Luminous-Dynamics/symthaea/issues/8115).

## Question

Does selecting the research experiment with the highest positive net EVSI reduce downstream decision regret compared with simple alternatives, when all methods face the same budget, scenario, candidate set and hard gates?

The ISPOR value-of-information guidance recommends representing important parameter dependencies and structural uncertainty rather than silently assuming independence or that the decision model is perfect: https://pmc.ncbi.nlm.nih.gov/articles/PMC7373630/

That is especially relevant where feedstock volatility, purification, process scale and capital requirements interact in alternative-protein systems. See the 2026 *Nature Food* perspective on the next decade of alternative proteins (https://www.nature.com/articles/s43016-026-01344-x), a 2026 precision-fermentation scalability review (https://pubmed.ncbi.nlm.nih.gov/42747475/), and the 2026 bacterial single-cell-protein feedstock review (https://doi.org/10.1016/j.coche.2026.101242).

## Strategies compared

- **Net EVSI oracle:** choose the eligible positive net-EVSI experiment with the highest net numerator, or choose no experiment if none has positive net value.
- **Yield-only:** select the highest expected-output proxy among experiments passing existing hard gates and the fixed budget.
- **Local-sensitivity-only:** select the highest local-sensitivity proxy, under the same gates and budget.
- **Equal-allocation round-robin:** allocate one experiment slot per scenario, rotating over sorted feasible candidate IDs. This is a transparent one-slot comparator, not a multi-experiment portfolio optimizer.
- **Seeded random:** choose reproducibly among feasible candidates using a deterministic seed/hash.

All selection methods share the same candidate IDs, scenario, abstract budget and non-compensatory evidence gates. A baseline cannot choose an unpermitted, scenario-mismatched or over-budget candidate just because its proxy score is high.

## Regret semantics

For each scenario, compute net EVSI with the existing qualified exact-integer evaluator. The oracle value is the largest positive net-EVSI numerator among eligible candidates, or zero when no candidate has positive net value. Each strategy's regret is:

[
\operatorname{regret}(s, m)=\operatorname{oracleValue}(s)-\operatorname{selectedValue}(s,m)
]

No selection has value zero; selecting an eligible experiment with negative net EVSI has negative selected value and therefore positive regret. A scenario family reports exact weighted regret numerators and maximum scenario regret. Scenario weights use an integer scale of 10,000, and all calculations use integer numerators to avoid floating-point drift. Utility and cost remain abstract units—not dollars or a measured biological output.

## Frozen synthetic families

Five families contain 14 scenarios:
1. Nominal prior shifts where output and sensitivity proxies disagree with EVSI.
2. A supply/energy-constrained budget excludes a high-score but unaffordable experiment.
3. All experiments have negative net EVSI; the oracle chooses none while simple heuristics can waste the budget.
4. Alternative model and cost assumptions change the best experiment.
5. A malformed scenario-weight total is a fail-closed negative control.

Baseline yield/sensitivity values, priors, likelihoods, costs, utility values and scenario weights are fabricated fixtures, not measurements.

## Targets

- 100% exact match for strategy selections, weighted regret numerators, max regret and reason codes on the frozen corpus.
- Zero hard-gate or budget violations by any comparator.
- EVSI oracle regret equals zero by construction within each declared scenario; that is a consistency property, not evidence EVSI is empirically superior.
- Seeded-random selection and all summaries are invariant to input scenario ordering.
- Report regret distribution and strategy disagreement—not only one aggregate mean.
- Future target: at least 20% lower mean decision regret than the strongest simple baseline on held-out, preregistered synthetic families, with uncertainty intervals. This is a proposed target, not an achieved result.

## Evidence ceiling and integration

Passing the Rust workflow qualifies the encoded comparator and receipt on the exact corpus and Git HEAD. It does not establish empirical research-planning superiority, calibrated probabilities, growth performance, food safety, process economics or lifecycle impact.

For a South African application, Symtropy owns the complete-cycle resource ledger, Sol Atlas supplies licensed region/time-specific scenario inputs, and Mycelix preserves permissions and evidence lineage. Each empirical parameter needs source vintage, units, uncertainty and authorization before replacing a synthetic value; do not infer correlation from tags alone.

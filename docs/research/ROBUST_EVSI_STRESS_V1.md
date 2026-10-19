# BIOPROTEIN-ROBUST-EVSI-001: decision stability under joint uncertainty

**Status:** synthetic design contract and executable stress benchmark.  
**Owner:** Symthaea decision/evidence semantics; this extends the qualified net-EVSI oracle rather than creating a new evidence system.

## Why this is the next paradigm shift

A single expected net-EVSI score can hide brittle decisions. The same evidence plan may rank first under the nominal assumptions and lose when feedstock conditions, test performance, energy cost, recovery loss, or model structure shift together. Treating such variables as independent can also create implausible combinations. The benchmark therefore evaluates explicitly named **joint scenarios** and reports decision stability, not just a single weighted mean.

This follows established value-of-information practice: uncertainty should be characterized across all decision-relevant parameters, dependencies should be represented jointly, and important structural uncertainty should be described or quantified rather than silently treated as certainty. See the ISPOR Value of Information task-force report, especially its treatment of parameter dependence, structural uncertainty, EVPI/EVPPI/EVSI and expected net benefit of sampling: https://pmc.ncbi.nlm.nih.gov/articles/PMC7373630/

The alternative-protein relevance is current: a 2026 *Nature Food* perspective notes the combined constraints of capital intensity, downstream purification, feedstock volatility and process scaling for bulk proteins; a 2026 precision-fermentation review argues for a scalability-first process design and early TEA; and a 2026 SCP feedstock review emphasizes the differences between lab substrates and industrial side streams.
- https://www.nature.com/articles/s43016-026-01344-x
- https://pubmed.ncbi.nlm.nih.gov/42747475/
- https://doi.org/10.1016/j.coche.2026.101242

## Contract

Each family is a finite weighted set of coherent scenario bundles. It does not assume that factors inside a bundle are independent. Every scenario reuses the same decision actions, utility definition and candidate experiment IDs; priors, experiment likelihoods, experiment costs, evidence gates and scenario digests may vary. Weights are integer parts-per-10,000 and must sum to exactly 10,000.

The evaluator reports:
- **Robust winner:** the same eligible, positive-net-EVSI experiment is selected in every declared scenario.
- **Scenario sensitive:** at least one scenario changes the selected experiment or turns a positive choice into no-positive-EVSI.
- **Indeterminate:** any scenario has an unresolved hard gate or unavailable eligible set.
- **No positive EVSI:** no scenario has a candidate with positive net EVSI.
- **Invalid input:** scenario weights, decision utility contract, state set, candidate set, or evidence/math inputs violate the contract.

For each candidate it also returns the exact weighted sum of net-EVSI numerators, min/max over eligible scenarios, positive/nonpositive scenario weight, and eligibility coverage. The weighted sum is divided by eligible scenario weight for its mean; candidate eligibility gaps are not silently treated as zero EVSI. Numeric outputs remain integer strings to preserve exact arithmetic.

## Frozen synthetic families

Eight families contain 22 scenario records:
1. **ROBUST-001:** stable winner under prior shifts and coupled resource tags.
2. **ROBUST-002:** recommendation flips when prior and candidate test costs shift together.
3. **ROBUST-003:** nominal winner fails under recovery degradation and cost shock.
4. **ROBUST-004:** no candidate justifies research cost in any scenario.
5. **ROBUST-005:** unknown authorization prevents a robust recommendation.
6. **ROBUST-006:** plausible alternate model structures change the winning experiment.
7. **ROBUST-007:** a low-weight distribution-shift/tail scenario removes the nominal candidate from the eligible set.
8. **ROBUST-008:** deliberately malformed scenario-weight total is rejected.

All priors, utilities, likelihoods, weights, tags, costs and scenarios are invented fixture values. The tags represent a joint assumption bundle; the oracle does not infer a real South African energy/feedstock relationship from those labels.

## Proposed benchmarks

- 100% exact agreement with frozen family statuses, winners, scenario winners and reason codes.
- Zero robust-winner labels unless the same candidate is eligible, positive-net-EVSI and selected across every declared scenario.
- Zero silent renormalization of scenario weights.
- Zero implicit independence assumptions: any scenario family states the joint-scenario semantics.
- Candidate coverage reports weight and counts where it was eligible; no missing gate is treated as zero value or permission.
- Stable output under scenario reordering and exact integer results without floating-point drift.
- On future preregistered families, report recommendation flip rate, worst-case net EVSI, weighted mean net EVSI, eligibility coverage, and decision regret vs. yield-only, local-sensitivity-only, equal-budget and random baselines.
- A proposed empirical-method target: at least 20% lower mean decision regret than the strongest simple baseline on held-out, preregistered scenario families, with uncertainty intervals. This is not claimed achieved.

## South African deployment path

Sol Atlas remains responsible for licensed, source-scoped region/time resource scenarios; Mycelix remains responsible for lineage and permission evidence; Symtropy remains responsible for the process closure ledger. A future Johannesburg/Highveld scenario may encode electricity-factor vintage, feedstock availability and competing uses, water limits, logistics, and downstream recovery together—but only from sourced, authorized values and explicit uncertainty. This corpus does not assert local quantities or calibrated probabilities.

## Validation ceiling

A successful Rust workflow proves that this implementation reproduces its synthetic contract on the exact source head and corpus digest. It does not establish calibrated probabilities, predictive accuracy, assay performance, biological growth, safety, commercial competitiveness, or lifecycle superiority.
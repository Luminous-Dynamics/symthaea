# ICCB paradigm shift: staged commitments, reversibility and real options

**Status:** Research/design protocol with a small deterministic tranche gate; no real-option valuation or live customer performance is claimed  
**Date:** 2026-10-18

## Core shift

A conventional capital review is often presented as a choice among projects using projected return, risk, liquidity, and approval limits. ICCB should also represent **the policy for how capital is committed over time**.

The question is not only “which project is best?” It is also:

- How much capital should be committed before critical uncertainties are resolved?
- Which evidence must be observed before the next tranche is released?
- Which milestones justify continuing, revising, pausing, or stopping?
- What is the cost of waiting for information versus the cost of irreversible commitment?
- Which losses become unrecoverable after each stage?
- Who is authorized to accept changed risk, and what evidence can invalidate that decision?

This creates a testable transition from one-shot project scoring to **evidence-gated, reversible commitment paths**. It is related to real-options theory, staged finance, and adaptive project governance; ICCB should not claim to have invented those disciplines. The potential contribution is a deterministic, auditable decision contract joining stage gates to physical dependencies, maintenance liabilities, provenance, and successor-readable reasons.

## Why test it now?

Family offices surveyed by UBS in 2026 reported interest in AI, power/resources, and infrastructure, while the same survey documented persistent succession gaps. J.P. Morgan's distinct survey found AI interest alongside limited infrastructure allocation. These are relevant motivations for testing real-asset decision workflows, not proof of demand for ICCB.

More immediately, recent reporting on the AI infrastructure build-out describes project delays/cancellations and increasing financing / execution concerns. The lesson for ICCB is not that all infrastructure investments are bad. It is that a benchmark should include the cost of committed capital, schedule slippage, changing assumptions, and the ability to stop before additional irreversible expenditure. See [Financial Times, “America's $10tn AI infrastructure boom is starting to fray”](https://www.ft.com/content/9ea5a37d-15c9-4d5b-94d6-719048c9cd3c) (published 18 October 2026; article is paywalled).

Two research findings sharpen the model:

- Bun, Huisman, de Kleijne and Kort's September 2026 Tilburg working paper, [*Staged Capacity Investments under Uncertainty*](https://research.tilburguniversity.edu/en/publications/staged-capacity-investments-under-uncertainty/), models multiple lumpy capacity investments. Its abstract reports that more opportunities to invest can lead a firm to invest earlier and in smaller amounts, expanding when demand increases sufficiently; this is especially profitable with a few highly price-inelastic customers. This argues against the simplistic rule “always wait” or “always stage.” Test the number of future opportunities, demand elasticity, and profit earned between stages.
- The World Bank's [*Infrastructure Foundations: From Current Assets to Future Growth*](https://www.worldbank.org/en/topic/infrastructure/publication/infrastructure-foundations-from-current-assets-to-future-growth) stresses asset stocks, financing and maintenance costs, social returns, and cross-sector complementarity. Its summary presents an Infrastructure Efficiency Ratio comparing social returns with the costs of financing and maintaining infrastructure. ICCB should therefore model complementary bundles—not just score assets as independent projects—and keep private returns distinct from wider social/productive effects.

## Decision model

Represent a capital plan as a stateful sequence:

1. **Authorize a bounded first tranche.** The authorization records amount, total commitment cap, owner, mandate, evidence snapshot, and permitted use.
2. **Observe a milestone.** The evaluator records what passed, what failed, and which evidence supports the observation.
3. **Decide whether to release the next tranche.** It must not be released unless authority, cumulative cap, milestone condition, and evidence gates pass.
4. **Recompute the decision vector.** Refresh costs, liquidity, dependencies, maintenance, capacity, forecasts, and the remaining option to stop or change direction.
5. **Record continue / revise / pause / stop.** Record the approver, dissent, reasons, changed assumptions, and evidence that would reopen the decision.

A previously spent amount is not by itself a reason to continue. The benchmark must include sunk-cost traps and make the incremental decision depend on future outcomes, not on the need to justify past spending.

## Implementation present in this branch

The crate's `staging.rs` introduces a small `TrancheRequest` contract. It evaluates:

- positive requested tranche amount;
- valid prior commitment and total cap;
- integer-overflow-safe cumulative commitment calculation;
- explicit milestone satisfaction;
- delegated authority;
- non-empty, qualified, in-scope evidence with usable observation/validity dates.

A failed condition rejects the next release. This is **a rule gate**, not an economic estimate of real-option value. The current implementation does not forecast milestone success, optimize tranche sizes, model schedule distributions, calculate discount rates, or prove that staging improves returns. The gate's milestone result and evidence state must be supplied by the caller and independently validated in future iterations.

The main evaluator also exposes:
- a cash-maximizing feasible baseline, using only options that pass hard constraints; and
- a Pareto frontier over expected cash, liquidity, productive capacity, and maintenance burden.

The candidate contract counts distinct declared dependency IDs only. It does not label any dependency as critical. Criticality scoring belongs in the separate evaluator harness, and physical critical-dependency assessments should ultimately come from a qualified Sol-Atlas result. A smaller raw count of declared links is not automatically safer and is not a Pareto advantage.

Reported valuation change remains observable but is deliberately excluded from Pareto dominance. No one metric should automatically substitute for productive capacity, and no weighted total should silently compensate for a hard-gate failure.

## Benchmark families

Add at least these cases to the frozen synthetic corpus:

1. **Successful staged deployment:** milestones pass and staging preserves the project's expected capability with manageable delay cost.
2. **Milestone failure:** the first stage reveals a fault; a fail-closed policy avoids releasing an unauthorized next tranche.
3. **Schedule slippage:** the project can still succeed, but delayed cash flows and financing costs alter the next decision.
4. **Evidence reversal:** a key initial assumption is invalidated by newer, in-scope evidence.
5. **Sunk-cost trap:** the candidate has already spent a large amount, but incremental continuation has lower prospective value than stopping.
6. **False milestone report:** a source claims a milestone passed but source scope, date, provenance or independent inspection fails validation.
7. **Irreversible versus reversible choice:** compare a large one-time commitment with staged or modular alternatives under the same hidden realized state.
8. **Liquidity shock mid-project:** a subsequent tranche would violate a predeclared liquidity floor.
9. **Dependency shock between stages:** a shared grid, water, logistics, or supplier dependency changes before release.
10. **Governance override attempt:** a high-return narrative tries to bypass the required approver or total commitment cap.
11. **Staging is not always better:** compare one-shot, wait-first, and staged policies across scenarios with different demand elasticity, future investment opportunities, financing costs, stage costs, and salvage values.
12. **Complementary infrastructure bundle:** energy capacity alone, digital connectivity alone, and a combined package produce different network value. Detect when a siloed optimum is dominated by a feasible bundle, while preserving each project's cost and risk.
13. **Waiting-cost trap:** a delay preserves flexibility but loses value through financing cost, missed revenue, scarce capacity, inflation or an expiring opportunity.
14. **Interdependent options:** a first project changes the cost, feasibility or salvage value of later projects; the evaluator must model endogenous as well as external uncertainty.

Each scenario needs at least one negative control and a simple policy baseline. The hidden evaluator knows what actually happened; candidate-facing inputs must not reveal outcome labels.

## Proposed benchmark metrics

Targets are provisional and need preregistration before final evaluation.

| Dimension | Measure | Initial target |
|---|---|---|
| Hard release gates | Unjustified tranche releases when milestone, evidence, authority or cap fails | 0 in the frozen fixture set |
| Capital bound | Cumulative commitment exceeding approved cap due to evaluation logic | 0 |
| Evidence invalidation | Detection of stale, scope-mismatched and contradicted milestone evidence | ≥95% recall on planted faults; report false positives separately |
| Sunk-cost resistance | Cases in which past spend improperly changes the incremental action absent future-value justification | 0 unjustified continuations in scripted negative controls |
| Decision regret | Difference from evaluator oracle, under a declared objective vector and scenario family | Compare to all-at-once, wait-first, staged, and cash-maximizing baselines; no preclaimed universal improvement |
| Policy choice quality | Whether the selected policy (one-shot, wait, stage, expand, pause, or abandon) matches the best policy for each frozen scenario | Score against a dynamic-programming / exhaustive-search oracle for small synthetic cases, plus regret rather than accuracy alone |
| Option preservation | Unnecessary irreversible commitment under uncertainty when a reversible path preserves value at tolerable cost | Measure on held-out scripted cases; report assumptions and regret |
| Complementarity | Difference between the best feasible project bundle and independently ranked silo choices | Include positive and negative interaction fixtures; do not assume synergies exist without model evidence |
| Time / financing cost | Delay, bridging finance, inflation and opportunity cost introduced by staging | Always report; staging does not receive free option value |
| Liquidity resilience | Violations under mid-project liquidity shocks | 0 hard-floor breaches in required negative controls |
| Decision trace | Exact approver, conditions, evidence and rationale for each release | 100% present in synthetic receipts |
| Reproducibility | Same input, configuration and version produce the same gate results and receipt | 100% equal outputs for deterministic fixture replay |

No aggregate score should hide an unauthorized tranche release or an accounting error. Report the costs of waiting as carefully as the benefits of preserving the option to stop.

## Robust capital ranking when probabilities are intervals

A point-probability EVSI or NPV calculation answers a conditional question: “What follows if this declared distribution and payoff table are suitable?” When evidence supports only lower/upper probability bounds, ICCB should also report what remains true across every probability vector allowed by those bounds.

The new `robust_decision.rs` prototype bounds the probability mass assigned to each discrete state while treating payoffs as fixed. It computes each action's exact minimum and maximum expected-value numerators on the feasible probability simplex. The algorithm starts from lower probability bounds and allocates the remaining mass to the lowest-payoff states for the minimum or highest-payoff states for the maximum; ties resolve by stable state ID. Output values use conservative floor/ceiling rounding and include the extremizing probability allocations.

The protocol labels an action a unique robust winner only when its minimum exact expectation strictly exceeds every alternative's maximum exact expectation. Overlapping payoff envelopes mean **no unique robust winner**, not permission to silently use the midpoint of the probability intervals. If the admissible probability constraints are impossible, the evaluator fails closed.

This version bounds probability ambiguity only. It does not bound uncertain payoff values, model-form discrepancy, omitted states, or out-of-domain behavior. Those require their own qualified inputs and should be kept distinct. See [ICCB-004](https://github.com/Luminous-Dynamics/symthaea/issues/8024), which aligns with Symtropy's [robust-design research #1769](https://github.com/Luminous-Dynamics/symtropy/issues/1769) and [model-discrepancy qualification #1812](https://github.com/Luminous-Dynamics/symtropy/issues/1812). The prototype also reports **worst-case expected regret** per action and the set of actions tied for the smallest worst-case regret. This can distinguish options when no robust winner exists, but it remains a decision lens rather than a mandate to execute the minimax-regret option. The robust-winner status, value envelopes, and regret ranking should all remain visible.

## Value-of-information budgeting: decide which uncertainty to buy down

Capital allocation should include a fourth decision class: **fund evidence acquisition**. A geotechnical survey, independent condition inspection, demand experiment, source confirmation, or small pilot can change the subsequent decision. The relevant question is not “would more data be useful?” but “what is the expected improvement in the decision after observing the result, and is that improvement worth the full cost and delay?”

For a decision with uncertain state \(s\), action \(a\), and payoff \(U(a,s)\):

- Baseline expected value is the value of the best action under the current state probabilities.
- Expected value of sample information (EVSI) is the expected value of selecting an action after observing the proposed noisy signal, minus the baseline value.
- Net EVSI subtracts direct survey/measurement cost, delay cost, and financing/carrying cost as separate amounts. These are validated as non-negative and summed with overflow checks. Other opportunity costs must be included explicitly in a declared component or in the payoff model; they must not disappear into an unlabelled aggregate.
- Expected value of perfect information (EVPI) is the ceiling on gross value from perfectly resolving the modeled state uncertainty. A specified signal should not exceed this ceiling under the same action/payoff model.

This is established decision-analysis methodology, not a claimed invention. See [Strong et al. (2015)](https://journals.sagepub.com/doi/10.1177/0272989X15575286), [Jackson et al. (2019)](https://www.tandfonline.com/doi/full/10.1080/01621459.2018.1562932), and the infrastructure engineering review [Value of information analysis in civil and infrastructure engineering](https://link.springer.com/article/10.1186/s43065-021-00027-0).

The Rust prototype in `value_of_information.rs` accepts explicit priors, conditional signal likelihoods, action payoffs, direct measurement cost, delay cost and financing/carrying cost. It is a deterministic finite-state calculation, not a probability-calibration system. Its parameters are assumptions unless supported by qualified external evidence and validated against outcomes. It intentionally does not make a “buy the survey” recommendation merely because the signal is informative. The direct, delay and financing components are reported separately so the net decision cannot conceal the carrying burden.

**Single-measurement benchmark cases implemented or targeted now:** (1) the measurement outcome changes the selected action and net EVSI is positive; (2) outcomes are uninformative and EVSI is zero; (3) beliefs change but the preferred action does not and EVSI is zero; (4) a survey costs more than its EVSI; (5) zero-probability outcomes are inert; (6) a hard legal/safety constraint requires evidence independent of monetary EVSI; (7) source uncertainty makes the outcome likelihoods too weakly identified to support a point estimate; and (8) permuting states, outcomes, actions or payoff rows preserves the canonical result.

**Future extension, not implemented by the current API:** compare two or more measurement actions whose errors may be correlated, where the second measurement can be chosen after observing the first. Do not add the individual EVSI estimates as if values were independent; evaluate the joint outcome space or an adaptive policy, and preserve an out-of-model-support branch when the outcome model is not qualified.

**Proposed acceptance gates:** exact hand-oracle agreement on small fixtures; probability and likelihood masses valid and complete; gross EVSI nonnegative and no greater than EVPI under the declared model; zero gross value where no signal outcome changes the action's expected-value ranking; and net EVSI negative when declared information cost exceeds gross value. Do not claim an empirically calibrated EVSI until priors, likelihoods, payoffs, costs and delay are qualified and sensitivity ranges are reported.

Cross-project boundary: Sol-Atlas already tracks value-of-information for site investigation in [issue #131](https://github.com/Luminous-Dynamics/sol-atlas-leptos/issues/131). ICCB should consume its qualified assessment rather than duplicate the spatial investigation engine. Further finite-state method validation is tracked in [ICCB-003](https://github.com/Luminous-Dynamics/symthaea/issues/8020); an evidence-bounded capital-assessment export contract is proposed in [Sol-Atlas #162](https://github.com/Luminous-Dynamics/sol-atlas-leptos/issues/162).

## Cross-project architecture boundary

Sol-Atlas already has adjacent work for the physical-resource side of this problem: [PR #116](https://github.com/Luminous-Dynamics/sol-atlas-leptos/pull/116) develops evidence-aware spatial-economic resilience analysis, while [issue #152](https://github.com/Luminous-Dynamics/sol-atlas-leptos/issues/152) covers shared-resource constrained allocation, [#153](https://github.com/Luminous-Dynamics/sol-atlas-leptos/issues/153) adds inventory conservation across time, [#155](https://github.com/Luminous-Dynamics/sol-atlas-leptos/issues/155) proposes independent sparse-allocation oracle validation, and [#131](https://github.com/Luminous-Dynamics/sol-atlas-leptos/issues/131) covers Pareto, sensitivity and value-of-information analysis.

ICCB should **consume qualified, versioned assessments from that physical-resource layer rather than duplicate its allocator, water model, or hydraulic solver**. A future adapter should preserve the scenario digest, as-of time, units, lower/upper service-capacity bounds, critical dependency IDs, evidence references, indeterminate states, distributional floors, and the explicit claim ceiling of the producing model. Unknown topology or stale evidence must widen/taint the relevant capital decision rather than becoming an optimistic default.

The intended responsibilities are complementary:
- **Sol-Atlas:** geospatial asset and service topology, resource constraints, capacity bounds, network cascades, distributional service floors, and evidence-bound physical assessments.
- **Symtropy:** dynamic scenarios and system simulation only where the modeled physics, constraints, and validation limits are explicit.
- **Symthaea / ICCB:** capital options, hard mandates, uncertainty, decision regret, staged commitment policies, Pareto outcomes, and independent decision receipts.
- **Mycelix:** optional provenance / authority / coordination facilities where they actually exist and are independently reviewed; a signed assertion is not proof of physical truth.
- **Luminous Platform:** deployment boundaries, access control, recovery, audit export and operator continuity.

**Integration status:** this is a proposed interface boundary, not a claim that these systems are integrated today. The next integration issue should freeze the adapter schema and cross-repository conformance fixtures before implementation.

## Research questions to resolve before claiming advantage

1. Which uncertainty can actually be resolved by staging, rather than merely postponed?
2. How costly is waiting in the use case (financing, schedule, missed revenue, inflation, scarce equipment or capacity)?
3. What is the appropriate baseline for the customer's existing milestone process?
4. Which milestones are observable, independently verifiable, and hard to game?
5. How should the evaluator represent correlated project risks, shared dependencies, and common financing constraints?
6. What conditions are safety / legal / mandate constraints, rather than preferences that can be weighted against return?
7. Does this protocol add value beyond a well-run stage-gate process and a transparent spreadsheet?

## References

- UBS, [Global Family Office Report 2026](https://www.ubs.com/global/en/media/display-page-ndp/en-20260528-global-family-office-report-2026.html).
- J.P. Morgan, [2026 Global Family Office Report](https://privatebank.jpmorgan.com/nam/en/insights/reports/2026-family-office-report).
- Financial Times, [“America's $10tn AI infrastructure boom is starting to fray”](https://www.ft.com/content/9ea5a37d-15c9-4d5b-94d6-719048c9cd3c), 18 October 2026.
- Bun, Huisman, de Kleijne & Kort, [*Staged Capacity Investments under Uncertainty*](https://research.tilburguniversity.edu/en/publications/staged-capacity-investments-under-uncertainty/), Tilburg CentER Discussion Paper 2026-017, published 29 September 2026.
- World Bank, [*Infrastructure Foundations: From Current Assets to Future Growth*](https://www.worldbank.org/en/topic/infrastructure/publication/infrastructure-foundations-from-current-assets-to-future-growth).
- Existing ICCB protocol and Rust prototype: [PR #7974](https://github.com/Luminous-Dynamics/symthaea/pull/7974), [PR #7987](https://github.com/Luminous-Dynamics/symthaea/pull/7987), and [issue #7976](https://github.com/Luminous-Dynamics/symthaea/issues/7976).

No real asset model, customer stage-gate workflow, live investment result, or customer commitment is represented by this document.

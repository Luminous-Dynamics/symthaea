# Appliance Decision-Value Diagnostics Protocol v0.1

**Status:** synthetic decision-analysis contract only  
**Tracker:** [APPLIANCE-DIAG-002 #7837](https://github.com/Luminous-Dynamics/symthaea/issues/7837)  
**Stacked on:** [active-diagnostics baseline PR #7834](https://github.com/Luminous-Dynamics/symthaea/pull/7834)  
**Physical status:** no real appliance priors, repair loss calibration, diagnostic benefit, safety validation, or field-cost reduction is established

## 1. Why entropy is only a baseline

v0.1 chooses a measurement using expected information gain per declared measurement cost. This is a useful, transparent diagnostic baseline, but entropy measures uncertainty reduction—not the operational value of a changed decision.

A measurement can distinguish two plausible hypotheses that lead to the same replacement part and repair procedure. It may have high information gain while saving no expected cost. Conversely, a measurement with fewer bits of information may be highly valuable if its outcome changes the appropriate repair and prevents substantial rework or downtime.

The question should evolve from:

> Which test reduces uncertainty the most?

to:

> Which safe, available measurement gives enough expected improvement in the next decision to justify its cost?

Expected value of information is an established decision-analysis concept used to determine whether new information is worth its acquisition cost. See the [PubMed-indexed decision analysis on expected value of information](https://pubmed.ncbi.nlm.nih.gov/17715257/). The synthetic appliance policy and benchmark here are proposed new work, not validated by that paper.

## 2. Decision model

Each decision case binds to one exact base trace in `fixtures/diagnostics/appliance_active_diagnostic_corpus_v0_1.json`. The corpus stores the SHA-256 of the exact baseline file bytes; if the parent diagnostic corpus changes, this decision fixture must be deliberately rebased and regenerated.

The added model declares:
- fault hypotheses and priors, inherited from the exact base trace;
- tests, outcome IDs and conditional likelihoods, also inherited from that trace;
- available repair/inspection/referral actions;
- action eligibility (`eligible`, `requires_qualified_person`, `prohibited`);
- action loss for each declared hypothesis;
- measurement acquisition cost expressed in the same synthetic loss units;
- the expected result, including prior action, prior risk, expected posterior risk, EVSI, cost, net value and the baseline entropy-policy choice.

### Bayes risk

For a declared hypothesis probability distribution (P(H)) and the set of eligible actions (A):

`R(P)= min_{a∈A} Σ_h P(h)L(a,h)`

Here `L(a,h)` is the declared loss if action `a` is taken when hypothesis `h` is the relevant state. The minimum-loss action is chosen with deterministic lexical tie-breaking.

### Expected value of sample information

For an eligible measurement (m) with outcomes (o):

`P(o|m)=Σ_h P(h)P(o|h,m)`

`P(h|o,m)=P(h)P(o|h,m)/P(o|m)`

`R_after(m)=Σ_o P(o|m)R(P(H|o,m))`

`EVSI(m)=R_before-R_after(m)`

`NetValue(m)=EVSI(m)-Cost(m)`

The action is selected after each potential outcome by the same eligible-action policy. Zero-probability outcomes contribute zero expected risk; they do not require an invented posterior.

**v0.1 chooses the available eligible test with the highest positive net value**, resolving exact ties by lexical measurement ID. If every test has non-positive net value, it reports the current best action and recommends no test. It reports the best rejected candidate and its net value so the no-test decision is auditable.

## 3. Safety and evidence precedence

Hard constraints precede utility scores. The algorithm cannot use a large EVSI to make an unsafe test or prohibited repair eligible.

- A tripped protection signal results in `abstain_and_review/protection_tripped`.
- Unknown protection results in `abstain_and_review/protection_unknown`.
- Missing, stale, conflicting or unit-invalid observations result in `abstain_and_review/observations_unreliable`.
- An incoherent action/loss matrix or invalid measurement-cost model is rejected; it is never silently repaired.
- If no eligible repair/inspection action exists, the result is `no_safe_action`.
- If actions exist but no measurement is available within the declared low-risk profile, return the best current eligible action and `no_safe_candidate` for measurement selection.
- An unavailable, qualified-person-only or prohibited measurement cannot be selected by expected value.
- The advisory result is not permission to perform a repair, operate a potentially faulty appliance, or bypass an independent protection.

The base v0.1 diagnostic evaluator remains the entropy-per-cost baseline. Decision-value policy outputs do not replace or rewrite baseline outputs.

## 4. Synthetic comparator result

The initial paired example uses the *same* priors and measurement likelihoods for both policies, but a synthetic action-loss matrix:

| Model outcome | Entropy/cost baseline | Decision-value policy |
|---|---|---|
| Highest-ranked measurement | `rpm_check` | `vibration_check` |
| Prior best repair action | — | `replace_cartridge` |
| Prior expected loss | — | 2.25 synthetic loss units |
| Expected posterior loss for selected test | — | 1.50 synthetic loss units |
| EVSI | — | 0.75 synthetic loss units |
| Test cost | Uses baseline test cost units | 0.50 synthetic loss units |
| Net expected value | Does not calculate repair decision loss | 0.25 synthetic loss units |

This is an **illustrative toy model**, not measured appliance performance. It demonstrates a policy distinction: entropy-per-cost and decision value can choose different tests even with the same sensory evidence. The loss values were not calibrated using real parts, repair time, or field costs.

A second fixture makes each test cost 2.0 synthetic units. The same best candidate has EVSI 0.75 but net value −1.25; the decision-value policy therefore recommends no test and retains the best current action. The fixture explicitly preserves the candidate's negative value rather than hiding it.

A third fixture uses two equally informative perfect binary measurements. They have equal EVSI and cost, so lexical ID tie-breaking is deterministic. Other fixtures cover tripped protection, missing observations, no available safe test, an incomplete loss matrix, and actions that all require prohibited/qualified-person handling.

## 5. Cost model caveats

A common synthetic unit allows the math to be exercised but should not hide real trade-offs. A future physical model must not combine dollars, minutes, environmental impacts and safety consequences through undocumented weights.

For real pilot work:
- measure parts, labour, tool/setup, test/calibration, waiting, downtime, rework, discarded components and recurrence separately;
- declare the analysis horizon, prices/location/date, uncertainty and data source;
- keep legally or safety-prohibited actions out of the candidate set, never as a high-loss action that the optimizer might eventually choose;
- report sensitivity/Pareto alternatives when loss weights are uncertain;
- evaluate whether a recommendation changes the actual decision and whether that change improves the held-out outcome, not just entropy.

## 6. Corpus and acceptance requirements

The v0.1 corpus is versioned as `APPLIANCE-DECISION-VALUE-001` and refers to the exact active-diagnostics baseline corpus by SHA-256. The Rust audit must verify this digest before using referenced trace IDs.

Required checks:
- exact baseline corpus digest; unique decision-case, action, hypothesis, measurement-cost and outcome identities;
- each action loss map covers every hypothesis exactly once; values are finite and non-negative;
- measurement costs cover the exact base candidate-measurement set exactly once, with finite non-negative values and a declared unit;
- posterior probabilities use the baseline likelihoods; prior and conditional likelihoods remain unchanged;
- prior and posterior risks, EVSI and net value match independent reference calculations within declared numerical tolerance;
- no ineligible action/test ever wins due to low cost or high expected value;
- mutation tests reject missing losses, negative/NaN/infinite costs, duplicate IDs, changed source digest, missing hypotheses, invalid likelihoods and altered hard-safety state;
- golden tests assert the entropy/decision-value ranking difference, no-test negative-value result, analytical perfect-information result and stable tie break;
- repeat evaluation of identical bytes yields identical report bytes and input digests.

## 7. Proposed benchmarks, not achieved results

- 100% agreement with an independently implemented golden mathematical oracle for the synthetic cases.
- Zero ineligible test/action selections, even if they have the largest nominal information gain or net value.
- Zero missing losses/costs coerced to zero.
- Exact source digest mismatch always blocks decision evaluation.
- On the declared synthetic corpus, the decision-value policy should match the minimum expected loss under its declared model; report negative net values and abstentions explicitly.
- For any later physical comparison, preregister the test-choice and repair-outcome protocol; report repair success, wrong-action loss, parts consumed, rework, restore time, downtime and recurrence. Synthetic metrics cannot be described as real diagnostic accuracy or savings.

## 8. Integration and claim ceiling

- The active-diagnostics baseline policy stays owned by [Symthaea PR #7834](https://github.com/Luminous-Dynamics/symthaea/pull/7834).
- The appliance engineering and repair-task context remains in [Symtropy PR #1806](https://github.com/Luminous-Dynamics/symtropy/pull/1806).
- Actual part consumption, repair events, post-repair verification and recurrence remain in [Mycelix issue #5156](https://github.com/Luminous-Dynamics/mycelix/issues/5156) and its existing Manufacturing/RepairProductiveLoop owners.
- Do not create another sensor ontology, repair ledger or source-artifact store.

A passing synthetic audit establishes only that the evaluator implements the declared decision model. It does not calibrate the priors, likelihoods, costs or losses to a real appliance and does not prove diagnosis accuracy, repair benefit, service life, product safety or legal conformity.

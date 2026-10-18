# Intergenerational Capability-Compounding Benchmark (ICCB)

**Status:** Research proposal; no implementation or customer-performance claim  
**Date:** 2026-10-18  
**Intended stack:** Symthaea, Sol-Atlas, Symtropy, Mycelix, Luminous Platform

## Executive thesis

Test a shift from **wealth reporting** to **capability compounding**.

Conventional wealth-management systems are important and should not be displaced merely because a broader architecture is possible. The proposed benchmark asks a different question: can a system help a long-horizon owner allocate capital across financial and productive assets while making dependencies, maintenance obligations, uncertainty, decision provenance, operational resilience, and institutional memory explicit?

The benchmark must be able to show that the proposed stack **loses**. It is not designed to prove that Luminous Dynamics, Mycelix, an alternative financial mechanism, or any one AI architecture is superior by construction.

“Capability balance sheet” in this document is a supplemental decision model, not a statutory accounting balance sheet, asset valuation, investment recommendation, or audited financial statement.

## Why investigate this now?

Recent institutional research identifies adjacent problems worth testing:

- UBS's *Global Family Office Report 2026* surveyed 307 family offices across more than 30 markets. It reports AI as a leading investment theme (65% already invested across the AI value chain), power/resources and infrastructure as themes for 37% each, and only 35% with a defined family-office succession plan. Only 27% reported a structured process to prepare heirs for future roles. These figures describe surveyed UBS clients; they do not establish demand for this project. [UBS report and methodology](https://www.ubs.com/global/en/media/display-page-ndp/en-20260528-global-family-office-report-2026.html)
- J.P. Morgan's *2026 Global Family Office Report* says 65% of its 333-office sample planned to prioritize AI, while 79% reported zero infrastructure allocation. This differs from UBS's question and sample, so the percentages should not be combined. It does highlight a potentially useful diligence gap between interest in AI and exposure to the physical systems AI depends on. [J.P. Morgan report](https://privatebank.jpmorgan.com/nam/en/insights/reports/2026-family-office-report)
- Sovereign wealth fund governance has established expectations beyond a compelling mission statement. The [Santiago Principles](https://www.ifswf.org/santiago-principles-landing/santiago-principles) emphasize governance, accountability, risk management, operational controls, and investment decisions grounded in financial risk and return.
- The [NIST AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework) provides a useful reference for organizing AI risk work, but alignment with a framework is not certification and would not by itself establish fitness for investment or infrastructure decisions.

The market evidence supports investigating the workflow. It does not establish willingness to buy ICCB, willingness to pay a premium, superior investment returns, or the readiness of the Luminous Dynamics stack for institutional deployment.

## Paradigm shifts to test

### 1. From a financial portfolio to a capability graph

Model not just holdings and returns, but the critical dependencies through which assets produce services: energy, water, logistics, people, suppliers, software, maintenance, financing, regulatory permission, and knowledge. The graph should preserve the source, scope, confidence, time validity, and owner of each edge.

**Hypothesis:** explicit dependency information improves decisions under correlated shocks compared with an asset-by-asset financial view.

### 2. From a scalar score to a mandate-conditioned outcome vector

Do not collapse return, liquidity, drawdown, operational uptime, productive capacity, maintenance backlog, concentration, resource use, and mission-specific outcomes into one opaque “impact” or “resilience” score.

Return a vector plus hard constraints. Any weighting must be supplied by the evaluator or investment mandate, versioned, and sensitivity-tested. Rights, safety and legal requirements are constraints, not tradeable terms in a weighted objective.

**Hypothesis:** vector outputs expose trade-offs and mandate conflicts that a single rank conceals.

### 3. From AI confidence to evidence contracts

Every material factual claim should be linked to source records, observation times, transformations, units, scope and uncertainty. Conflicting or stale evidence must remain visible. Unsupported claims should cause explicit abstention or a qualified answer, not confident synthesis.

Cryptographic hashes and signatures prove identity or integrity of recorded bytes under stated assumptions; they do not prove a source is true or that a physical asset matches the record.

**Hypothesis:** evidence contracts reduce silent data errors and improve independent review.

### 4. From episodic diligence to a closed learning loop

Record forecasts, recommendations, approvals, disconfirming evidence and realized outcomes. Freeze the forecast before outcomes arrive; evaluate calibration and decision usefulness later. Preserve revisions rather than rewriting history.

**Hypothesis:** post-decision evaluation catches systematically miscalibrated assumptions and makes future diligence more useful.

### 5. From succession documents to contestable institutional memory

Capture what was decided, which evidence supported it, what alternatives were rejected, which assumptions were disputed, what would change the decision, who had authority, and when the decision should be revisited. A successor must be able to challenge prior reasoning, not merely inherit an answer.

**Hypothesis:** an incoming operator can recover decision context faster and with fewer material errors than from the customer's existing documentation alone.

### 6. From platform lock-in to owner-controlled continuity

Make data export, provenance export, reproducible replay, key ownership, offline operation where required, and recovery testing part of the product. An owner should be able to leave the vendor without losing the family's institutional memory.

**Hypothesis:** verifiable portability and recovery increase resilience and trust independently of model quality.

## Scope and non-goals

ICCB evaluates decision-support workflows over synthetic, fully specified scenarios first, followed by a consented shadow pilot if a suitable partner is found.

It does **not**:
- provide a portfolio return guarantee, financial advice, fiduciary service, legal advice, or regulatory-compliance certification;
- replace accounting, custodial, portfolio-management, engineering, legal or human governance systems;
- infer a family's values from wealth or demographic attributes;
- allow AI to make or execute material investment decisions without an authorized human process;
- assume that higher asset value, higher private return, increased capital deployment, more AI, or a Luminous Dynamics component is inherently better;
- treat a hash, ZK proof, attestation or consensus result as proof that an underlying real-world statement is true.

## Benchmark structure

Use a fixed evaluator-owned synthetic corpus with the true scenario state and correct accounting identities held out from the system under test. Publish scenario generators, schemas, random seeds and scoring code after leakage review. Pin versions and hashes in a run manifest. Do not expose answer labels or hidden ground truth to the candidate.

### Scenario families

1. **Productive expansion:** several projects with distinct cash flows, useful lives, financing needs and execution risks. Include cases where the highest projected IRR is not the highest mandate-adjusted outcome.
2. **Asset-price boom:** rising collateral and reported valuations without matching growth in productive capacity or cash flow.
3. **Critical dependency cascade:** a power, water, logistics or supplier outage propagates across apparently diversified holdings.
4. **Maintenance debt:** an attractive acquisition or expansion omits lifecycle replacement, staffing and maintenance obligations.
5. **Concentration camouflage:** assets appear diversified by sector or geography but share a critical supplier, grid, insurer, lender, route or jurisdiction.
6. **Stale or conflicting evidence:** prices, forecasts, asset condition, contract terms or risk ratings have different dates and scope; one source is deliberately misleading.
7. **Transition cost:** the candidate investment is promising at steady state but implementation disrupts revenue, employment, liquidity, service continuity or legacy contracts.
8. **Succession handoff:** a new decision-maker receives the ordinary records, selected decision memos and contradictory observations, then must reconstruct the decision and identify what would change it.
9. **Governance/authority challenge:** an apparent high-confidence recommendation conflicts with the declared mandate, lacks delegated authority, or depends on undisclosed conflicts or a questionable evidence source.
10. **Tail and correlated shocks:** multiple moderate events co-occur; probability estimates and dependencies are imperfectly known.

Every family needs positive, negative, ambiguous and abstention-appropriate cases. The test must include simple cases where spreadsheets or conventional tools are sufficient; complexity should not automatically reward the proposed stack.

## Baselines and ablations

Where possible, run the same task and information budget against:

1. The customer's current documented workflow (only with permission and a recorded baseline).
2. A simple spreadsheet/rule-based baseline.
3. A standard financial model with no additional dependency graph.
4. A general-purpose LLM workflow using the same supplied evidence.
5. Symthaea-only decision support.
6. Sol-Atlas/Symtropy-enabled dependency and simulation workflow.
7. The integrated candidate stack, with Mycelix provenance/coordination and Luminous Platform deployment controls enabled where implemented.
8. An evaluator oracle that has access to the hidden state, used only to estimate the performance ceiling—not as a deployable baseline.

Run component ablations. If an integrated system provides no material improvement over a simpler baseline, report that result and remove unnecessary complexity.

## Metrics and proposed gates

All thresholds below are **provisional engineering targets**, not published achievements or guarantees. Freeze them before inspecting final test results and revise them only through a documented protocol change.

| Dimension | Metric | Initial gate proposal |
|---|---|---|
| Evidence integrity | Share of material factual claims linked to in-scope, dated evidence or explicitly marked unknown | 100% on benchmark fixtures |
| Adversarial evidence | Recall on planted stale, contradictory, scope-mismatched and manipulated evidence | ≥95%, with ≤5% false positives on clean controls |
| Hard constraints | Budget, resource, authority and stated safety constraints violated by a recommendation | 0 violations on fixtures; otherwise fail the case |
| Accounting correctness | Independent replay of cash, capital, maintenance, capacity and liability identities included in the fixture | 100% exact agreement within declared numeric tolerance |
| Decision quality | Regret versus the evaluator oracle under each declared mandate; report by scenario family and tail percentile | Better than the strongest simple baseline in pre-registered target families, with confidence intervals |
| Uncertainty | Brier/log score for binary events, interval coverage for numeric outcomes, calibration plots | No claims of calibration without enough independent trials; report sample count and uncertainty |
| Dependency analysis | Recall and precision for planted critical dependencies and cascading failure paths | ≥90% recall on the frozen synthetic set; report false alarms separately |
| Succession | Blind recovery of decision rationale, authority, dissent, assumptions and reconsideration triggers | ≥90% correct answers on a preregistered question set, without access to the original decision-maker |
| Human review burden | Time and material correction rate versus the documented baseline | Measure in shadow pilot; no universal threshold before baseline |
| Portability / continuity | Export and restore of records, evidence, access policies and decision history in a clean environment | 100% of required records restored and integrity-checked in the pilot fixture |
| Operational resilience | Restore time, stale-source detection, component failure behavior, offline/degraded operation where required | Define per customer before pilot; fail closed on loss of required evidence or authority |
| Commercial value | Customer-accepted benefit against fully burdened deployment, training, support and governance cost | Demonstrate at least one material pre-agreed improvement; do not infer general ROI from one pilot |

Do not use one aggregate score to hide a hard-gate failure. Show a dashboard of outcome vectors, per-family results, confidence intervals, failure cases and a record of tests that were not run.

## Experimental integrity

- Separate candidate and evaluator code paths; the candidate never sees hidden labels.
- Freeze the manifest, fixtures, baselines and scoring before the final run.
- Include negative controls, data leakage tests, seed replay, corrupted-input tests and deterministic accounting checks.
- Report failed, skipped and unsupported tests explicitly; queued CI is not a pass.
- Preserve run receipts with commit SHA, dataset/fixture digests, configuration, tool/model versions, environment, exit status and per-test results.
- Use synthetic data until a partner has approved data handling, security boundaries, retention, access and publication terms.
- Distinguish simulated outcomes, historical backtests and prospective live outcomes in every report.
- For real investment results, account for transaction costs, taxes where relevant, liquidity, survivorship/look-ahead bias, counterfactual uncertainty and adequate sample size. Do not claim alpha from a small case study.

## Cross-project responsibility boundaries

| Component | Candidate role in ICCB | Claim ceiling |
|---|---|---|
| Symthaea | Evidence-aware reasoning, uncertainty and decision trace | Performance must be measured on the frozen task; no general intelligence or financial-compliance claim |
| Sol-Atlas | Geospatial asset context, spatial dependencies and resource exposure | Only as complete and accurate as qualified sources and models |
| Symtropy | Scenario simulation and dynamics for modeled systems | Simulation is not a substitute for validation against observed physical outcomes |
| Mycelix | Provenance, identity/authority and coordination where implemented | Protocol integrity does not establish truth, legitimacy or fair governance by itself |
| Luminous Platform | Deployment, access, recovery and operational control | Requires independent security testing and explicit operational ownership |

A component should be marked **not implemented** when the required capability is not yet available. A proposed architecture is not evidence of integration.

## Implementation sequence

### Step 0 — Freeze the protocol
Review this proposal for overlap with Symthaea's existing capital-allocation work, then approve a versioned manifest format, mandate representation, and independent scoring rules.

### Step 1 — Build the synthetic evaluator in Rust
Implement typed structures for Scenario, Asset, Dependency, EvidenceClaim, DecisionOption, Mandate, Constraint, Forecast, Outcome, and RunReceipt. Prefer deterministic fixtures and explicit units. Keep hidden evaluator truth outside the candidate interface.

### Step 2 — Establish simple baselines
Implement spreadsheet-equivalent rule baselines and reference accounting / dependency oracles before integrating an AI model. A new architecture must outperform these baselines in specified cases to justify its complexity.

### Step 3 — Add adversarial fixtures
Introduce stale sources, conflicting evidence, missing values, false correlations, hidden maintenance burdens, contaminated provenance, unauthorized recommendations and correlated shocks. Verify that uncertainty and abstention are useful outputs.

### Step 4 — Run ablations and publish evidence
Pin exact commit and fixture digests. Compare simple baselines, individual components, and the integrated workflow. Publish limits and negative findings with positive ones.

### Step 5 — Shadow pilot with a consenting asset owner
Choose one asset class and one repeated decision workflow. Keep recommendations non-executing. Agree on data access, control group/baseline, success criteria and publication permissions before access. Measure operator time, error correction, forecast calibration and customer-validated benefit.

### Step 6 — Expand only after value is established
Only then consider additional portfolio companies, regions, or sovereign / transnational stakeholders. Institutional deployment requires a security and governance case appropriate to the customer, not just benchmark scores.

## Probability ambiguity: never manufacture a point prior

The benchmark now includes a complementary robust decision mode for cases where an evaluator can justify lower and upper probability bounds for discrete states but cannot justify one point distribution. See the implementation work in [ICCB-004](https://github.com/Luminous-Dynamics/symthaea/issues/8024).

The robust mode holds the payoff table fixed and computes, for each action, its minimum and maximum expected payoff over all probability vectors satisfying the declared bounds and summing to 10,000 basis points. It returns a unique robust winner only when one action's exact worst-case value strictly exceeds every alternative's exact best-case value. If envelopes overlap, it reports that the ranking is ambiguous rather than substituting interval midpoints or inventing a probability.

The current prototype bounds **probability uncertainty only**. It does not model uncertain payoff outputs, unknown/out-of-domain states, or model-discrepancy intervals. Those require their own validated inputs and must not be folded into this label. This track complements Symtropy's [CIV-INFRA-ROBUST-001 #1769](https://github.com/Luminous-Dynamics/symtropy/issues/1769) and [CIV-INFRA-CAL-001 #1812](https://github.com/Luminous-Dynamics/symtropy/issues/1812); ICCB should consume qualified robustness/model-discrepancy evidence from those domains rather than duplicating their calibrators or physical models.

Provisional acceptance targets: 100% agreement with a separately implemented grid oracle on frozen small fixtures; every extremal probability allocation respects per-state bounds and sums to one; infeasible interval simplexes fail closed; conservative floor/ceiling results are correct for negative fractional expectations; permutations preserve canonical outputs; and no unique winner is reported unless exact worst-case/best-case envelopes strictly separate.

## Decision rule

Proceed beyond the synthetic phase only if all hard gates pass, material limitations are explicit, and the candidate beats a simpler baseline on at least one preregistered use case without unacceptable regressions. Otherwise, narrow the scope or reject the hypothesis.

**The research objective is not to persuade dynastic families that the platform is revolutionary. It is to produce evidence that lets them independently decide whether it is useful, governable, resilient and worth buying.**

## References

1. UBS, [Global Family Office Report 2026 announcement and survey summary](https://www.ubs.com/global/en/media/display-page-ndp/en-20260528-global-family-office-report-2026.html), 2026-05-28.
2. J.P. Morgan Private Bank, [2026 Global Family Office Report](https://privatebank.jpmorgan.com/nam/en/insights/reports/2026-family-office-report), 2026.
3. International Forum of Sovereign Wealth Funds, [Santiago Principles](https://www.ifswf.org/santiago-principles-landing/santiago-principles).
4. NIST, [AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework).
5. Existing adjacent Symthaea research: [#5612 neutral capital-allocation and transition-path conformance lab](https://github.com/Luminous-Dynamics/symthaea/issues/5612), [#7036 function-level financial equivalence benchmark](https://github.com/Luminous-Dynamics/symthaea/issues/7036), and [#7186 creditism capacity, issuance, and capital-allocation stress model](https://github.com/Luminous-Dynamics/symthaea/issues/7186). ICCB should complement these models, not duplicate their mechanisms.

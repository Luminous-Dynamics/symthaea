# Systemic Extraction Stress Suite (SESS) — Research Specification v0.1

**Status:** specification only; no benchmark implementation or result is implied by this document.  
**Owner boundary:** Symthaea owns executable experiments and causal traces; Mycelix owns interoperable, evidence-bearing claims, profiles, authority and governance semantics.  
**Claim ceiling:** synthetic mechanism validation only until separately calibrated and validated against external observations.

## 1. Purpose

Determine whether a socioeconomic arrangement can sustain its apparent success by combining one or more of:

1. obligations that depend on continued new inflows or refinancing rather than reproducible operating surplus;
2. financial claims that grow faster than the productive capacity or collateral that is supposed to support them;
3. drawdown of physical/ecological stocks without adequate regeneration, repair or substitution;
4. costs displaced onto workers, households, other jurisdictions, marginalized groups or future cohorts;
5. concentrated control, coercion, opaque authority or high switching costs that prevent affected parties from refusing, correcting or leaving the arrangement.

The suite studies a **systemic extraction loop** as a causal pattern, not as a rhetorical label. A literal Ponzi scheme is a narrower financial fraud category. The suite must not classify every recession, debt-financed investment, unequal outcome, ecological impact or failing institution as “a Ponzi scheme.”

### Operational distinction

Emit separate, non-equivalent findings:

- **Financial rollover dependency:** obligations cannot be met under the declared schedule without new net claim inflows, asset sales, or refinancing; identify which source is doing the work.
- **Productive/physical overdraw:** required service or claim support is maintained by reducing productive, infrastructure, inventory, skill, or ecological stocks faster than their replenishment.
- **Externalized-cost transfer:** material burdens are causally shifted across populations, jurisdictions, domains or generations and excluded from the decision maker's own ledger.
- **Coercive persistence:** power, essential-service dependence, surveillance, retaliation or switching costs prevent meaningful refusal, challenge or exit.
- **Systemic extraction loop:** at least two of the preceding mechanisms are causally coupled over time and the coupling is material under the declared scenario. Any stricter decision rule must be versioned, explained and independently evaluated; no single scalar score may make this determination.

The label is descriptive of a declared model and evidence boundary, not a legal determination or a universal diagnosis of the real economy.

## 2. Compose existing work; do not duplicate it

This is an integration contract across existing research threads, not a competing simulation architecture:

- Symthaea [#7188](https://github.com/Luminous-Dynamics/symthaea/issues/7188): socioeconomic interoperability laboratory.
- Symthaea [#7190](https://github.com/Luminous-Dynamics/symthaea/issues/7190): power, agency, dependency and rights-floor stress.
- Symthaea [#7193](https://github.com/Luminous-Dynamics/symthaea/issues/7193): stock-flow, balance-sheet and intergenerational accounting.
- Symthaea [#7195](https://github.com/Luminous-Dynamics/symthaea/issues/7195): empirical calibration, causal identification and multiverse validation.
- Symthaea [#7196](https://github.com/Luminous-Dynamics/symthaea/issues/7196): production, input-output and critical-supply network.
- Symthaea [#7201](https://github.com/Luminous-Dynamics/symthaea/issues/7201): metric gaming / Goodhart stress.
- Mycelix [#4804](https://github.com/Luminous-Dynamics/mycelix/issues/4804): socioeconomic protocol profiles and interoperability kernel.
- Mycelix [#4806](https://github.com/Luminous-Dynamics/mycelix/issues/4806): power, rights, agency and non-domination.
- Mycelix [#4808](https://github.com/Luminous-Dynamics/mycelix/issues/4808): stock-flow and intergenerational accounting.
- Mycelix [#4813](https://github.com/Luminous-Dynamics/mycelix/issues/4813): calibration, causal identification and uncertainty.
- Mycelix [#4814](https://github.com/Luminous-Dynamics/mycelix/issues/4814): production and critical-supply network.
- Mycelix [#4816](https://github.com/Luminous-Dynamics/mycelix/issues/4816): Civilization Twin.
- Mycelix [#4825](https://github.com/Luminous-Dynamics/mycelix/issues/4825): metric gaming and objective non-optimization.

Reuse the existing scenario/model machinery wherever possible. If any required state variable or oracle is absent, add it to its owning issue rather than silently creating a parallel implementation.

## 3. Non-negotiable semantic boundaries

The suite must preserve these distinctions:

```
financial claim != productive asset != physical resource != entitlement
valuation != cash flow != settlement != solvency
access on paper != deliverable essential service
reported sustainability != physical regeneration
formal consent != meaningful exit
model output != evidence != authorization
simulation pass != empirical validity != policy legitimacy
```

No amount of currency, credit, reputation, model confidence, majority support or cryptographic consensus may replenish a physical stock, make a false observation true, or widen an authority's declared scope.

A Symthaea recommendation may identify a condition or compare interventions. It must not itself authorize coercion, seize resources, revoke rights, or alter Mycelix governance. Allocation actions require their separately declared authority and appeal path.

## 4. State and audit contract

Every run freezes and records:

- scenario ID and version;
- world-profile and institution-profile content hashes;
- financial, jurisdictional and ecological boundaries;
- initial state and time horizon;
- units and stock identities;
- agent/behavior and expectation profiles;
- topology and substitution assumptions;
- exogenous shock schedule and random-stream namespace;
- policy/intervention and declared counterfactual;
- evidence sources, uncertainty and missingness;
- implementation revision, evaluator revision and oracle revision.

Use common random numbers for treatments intended to differ only in institutional design; random draws for exogenous events must not depend on the treatment being compared. Preserve the exact seed, schedule and event lineage.

Each material state transition should be attributable to a typed event: source state, action, actor, authority, target stocks/claims, direct effect, mediated effect, external effect, uncertainty, and correction/reversal path.

## 5. Minimum scenario suite

Use the same synthetic world and equivalent exogenous shocks for paired institutional comparisons.

| ID | Scenario | Purpose and expected observation |
|---|---|---|
| S0 | Reproducible, maintained baseline | Positive control: obligations, production, maintenance and renewable stocks reconcile without unexplained growth. |
| S1 | Ordinary recession, solvent institutions | Negative control: a downturn can cause real losses and defaults without rollover dependency being misclassified as a Ponzi-like loop. |
| S2 | Rollover-dependent claims | Financial positive control: scheduled obligations exceed reproducible surplus; fresh inflows conceal the gap until inflows are withdrawn. |
| S3 | Leverage / duplicate collateral | Detect rehypothecation, duplicate claims, double-counted collateral and mark-to-model appreciation that does not add productive capacity. |
| S4 | Ecological or infrastructure drawdown | Financial indicators remain temporarily positive while maintenance, inventories, water/soil/material stocks or productive capital decline. |
| S5 | Coupled extraction loop | Combine S2/S4 with unequal cost incidence and dependencies that preserve the arrangement after damage becomes visible. |
| S6 | Externality export | Move extraction or liability outside the accounting jurisdiction; a boundary audit must reveal scope loss rather than report a real improvement. |
| S7 | Capture and suppression | Compromised authority attempts to hide liabilities, degrade evidence, silence dissent or convert temporary emergency powers into permanent control. |
| S8 | Metric gaming | Agents know the benchmark target and improve reported values while degrading independent outcomes. |
| S9 | Shock, exit and recovery | Withdraw a key bridge, supplier, financing inflow or authority; test essential-service continuity, substitutes, loss distribution and recovery. |
| S10 | Intergenerational transfer | Improve current outcomes by transferring obligations, environmental damage or lost options to later cohorts. |
| S11 | Honest alternative | Compare bounded-credit, cooperative, market, public-planning, commons and hybrid profiles on the *same world*; do not assume a preferred profile wins. |

The test harness must support mixed regimes and bridges. A bridge may translate or settle claims; it must not manufacture underlying capacity or silently transfer political authority.

## 6. Minimum metrics — a vector, never a civilizational score

### Financial and productive
- operating surplus and scheduled service by obligation class;
- net new claim inflow required for settlement;
- debt/refinancing maturity profile and liquidity gap;
- financial claims, counterparties, collateral and duplicate-claim exceptions;
- productive output and capacity utilization;
- net productive-capital stock, depreciation, maintenance backlog and replacement needs.

### Physical and ecological
- each resource stock by explicit unit, boundary and time;
- inflow, regeneration, extraction, transformation, imports and exports;
- inventory coverage, bottleneck utilization and substitution depth;
- pollution or damage stocks where modeled;
- distance from declared limits, with uncertainty and boundary caveats.

Do not compress unlike physical quantities into a single “resources remaining” figure. Planetary-boundary indicators are interdependent and apply at different scales; a local model must declare how its limits relate to regional/global constraints.

### People, power and intergenerational incidence
- essential demand, actual service delivered and unmet demand by cohort/population;
- benefits and burdens by actor group and time period;
- control of critical assets, settlement bridges, verification, agenda/veto, and information access;
- dependency asymmetry, switching cost, refusal cost and viable exit routes;
- rights-floor violations and duration;
- inherited assets, obligations, damaged stocks and feasible future choices.

### Resilience and evidence
- losses under shocks, contagion radius, recovery time and service continuity;
- oracle disagreements and false-positive / false-negative counts;
- parameter and model uncertainty; results across declared multiverse and holdout shocks;
- detection lead time, stale/missing evidence and sensitivity to boundary changes.

Report distributions and subgroup outcomes, not only averages. Preserve non-identifiability where available observations cannot distinguish mechanisms.

## 7. Independent oracles and pass/fail rules

The evaluator must not treat the system under test's own report as ground truth. Implement at least separate checks for:

1. **Stock-flow reconciliation:** for each declared stock, reconcile opening balance, inflows, outflows, transformations and exogenous changes to closing balance. Use typed units; fail closed on missing units, duplicate event identities or unexplained residuals beyond a declared tolerance.
2. **Claim/counterparty integrity:** track issuance, ownership, transfers, settlement, destruction, collateral and contingent claims without duplicate backing or missing counterparties.
3. **Physical conservation:** monetary issuance alone cannot replenish depleted physical stocks. Transformations and recycling must account for yields, energy/material inputs, losses and delays.
4. **Causal incidence:** confirm whether an observed improvement came from new production, redistribution, boundary movement, stock drawdown, suppressed demand or burden transfer.
5. **Power/exit:** verify that a reported increase in access or equality does not mask new chokepoints, retaliation, rights-floor failures or rising exit costs.
6. **Reproducibility:** same frozen inputs and revision yield the same deterministic trace where determinism is promised.

Mandatory metamorphic tests:

- Multiplying the unit scale for money by a constant must not change real resource stocks or physical service delivery.
- Adding aliases, bridges or duplicated representations must not create underlying assets, resources, or independent collateral.
- Changing the reporting boundary must change the boundary declaration and expose shifted flows; it cannot silently erase them.
- Increasing monetary issuance cannot make a depleted ecological stock pass its physical oracle.
- Reordering independent events must not change a deterministic outcome; invalid causal reorderings must be rejected.
- Removing new inflows from S2 must expose the rollover gap under its frozen parameters.
- The ordinary-recession negative control S1 must not trigger a literal Ponzi classification without the required rollover evidence.
- A high reported score with worsening independent outcomes must be flagged as metric divergence.

Every failure reports a typed reason and evidence pointers, not a generic “unsafe” label.

## 8. Adversarial campaign

At minimum attempt:

- fake, stale or selectively missing production/resource observations;
- duplicate assets and circular collateral;
- optimistic regeneration, omitted maintenance, delayed liabilities and horizon truncation;
- offshore/jurisdictional or inter-profile externality transfer;
- forged provenance, verifier collusion and compromised evaluator;
- Goodhart gaming of access, throughput, sustainability and participation metrics;
- dissent suppression, credential denial, bridge capture and exit-cost inflation;
- interventions timed just before reporting snapshots;
- a plausible alternative model producing the same aggregate fit.

The system must test both detection and harmful overreaction. A false alarm can itself cause credit contraction, service denial, surveillance or coercive intervention. Detection never automatically authorizes those actions.

## 9. Evidence ladder and claim ceiling

Label each result with one of these non-interchangeable states:

- **SPECIFIED:** requirements exist; no execution claim.
- **IMPLEMENTED:** implementation exists and is pinned to a revision.
- **EXECUTED:** a run completed with its inputs and artifacts preserved.
- **REPRODUCED:** an independent rerun matched the declared outputs within specified deterministic/numerical tolerances.
- **SYNTHETICALLY_VALIDATED:** oracles and adversarial controls passed in the declared synthetic model.
- **EMPIRICALLY_CONSTRAINED:** data provenance, population, time window, uncertainty, and identification assumptions are documented.
- **OUT-OF-SAMPLE VALIDATED:** predeclared holdout/intervention checks passed without tuning on the holdout.

A lower state must never be described using a higher-state claim. Synthetic success cannot establish the true causes of the global economy, predict a particular crisis, certify legal fraud, prove a policy morally legitimate, or show that Symthaea/Mycelix can prevent systemic violence.

## 10. Implementation sequence

**P0 — contract and controls**
1. Map each required variable/oracle to #7190, #7193, #7195, #7196, #7201 and Mycelix #4804/#4806/#4808/#4813/#4814/#4816/#4825.
2. Pin canonical units, boundary semantics, scenario IDs, stable event identities and deterministic exogenous random namespaces.
3. Implement S0/S1/S2 plus independent stock-flow and claim reconciliation before adding policy optimization.

**P1 — coupled failure**
4. Add S4/S5/S6/S10 and validate physical stocks, cost incidence and cohort outcomes independently.
5. Add capture/metric-gaming scenarios S7/S8 and demonstrate evaluator separation.
6. Add shock/exit/recovery S9 with essential-service floors and explicit transition costs.

**P2 — robustness and empirical work**
7. Run adversarial and metamorphic tests across multiple seeds, parameter sets, topologies and alternative behavioral models.
8. Freeze holdout scenarios before tuning; document non-identifiability and model disagreement.
9. Only then calibrate against external data and make appropriately scoped claims.

### Definition of done for the first vertical slice

The first slice is done only when S0, S1 and S2 have:
- frozen manifests and deterministic/replayable traces;
- independent stock-flow and claim/counterparty oracles;
- a demonstrated positive control (S2) and negative control (S1);
- at least one counterfactual removing new inflows;
- a machine-readable report of metrics, violations, uncertainties and evidence pointers;
- tests showing money-unit scaling does not change physical results;
- an honest evidence state.

CI status is useful evidence but is not the definition of done. No result is “passed” until the relevant runner and independent oracle actually execute.

## 11. Research references

- Stockholm Resilience Centre, **Planetary boundaries** (including the 2025 Planetary Health Check and the interdependence of boundaries): https://www.stockholmresilience.org/research/planetary-boundaries.html
- NIST, **AI Risk Management Framework** and its Govern / Map / Measure / Manage functions: https://www.nist.gov/itl/ai-risk-management-framework
- Veraart, **Distress and default contagion in financial networks**, *Mathematical Finance* (2020): https://doi.org/10.1111/mafi.12247
- Ecological economics and growth-independent prosperity research agenda: https://doi.org/10.1016/j.ecolecon.2019.106495

These sources motivate the benchmark; they do not validate its implementation or decide its results.

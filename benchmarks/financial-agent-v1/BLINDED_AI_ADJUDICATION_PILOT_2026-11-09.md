# Masked AI Adjudication Pilot — Synthetic Supplier/Energy Research Task

**Date:** 2026-11-09  
**Status:** evaluator-calibration artifact; not a model benchmark result, human-subject result, or SOTA claim  
**Evaluator:** one GPT-based AI adjudicator; system identities masked  
**Task:** dev.synthetic.supplier-energy-shock.0001 from the public development fixture  
**Source fixture:** dev-example.task.json and dev-example.evidence.json  
**Reference scope:** exact synthetic evidence only; no outside facts admitted

## 1. Why this pilot exists

The user authorized the assistant to serve as a blinded evaluator. This pilot records one masked AI adjudication pass. The evaluator is not a human, and this is not a substitute for a genuinely independent human panel. It tests whether the proposed rubric can distinguish:
- factual correctness from evidence-trace support;
- historical observations from current conditions;
- conditional hypotheses from causal conclusions;
- exact units/periods from invented comparisons;
- useful abstention from confident fabrication.

Candidate IDs were assigned without system/provider labels. The adjudicator saw only the response text and frozen task evidence. The candidates below are deliberately synthetic rubric-calibration examples, not outputs from separate live models. Therefore this pilot cannot establish model-to-model performance, inter-rater reliability, generalized accuracy, or human preference.

## 2. Frozen task facts

Decision frontier: 2026-10-01T12:00:00Z

### Evidence E-SUP-01

- Supplier SYN-SUPPLIER-01 has a reported shipment_schedule_disrupted status.
- The report applies from 2026-09-30T09:00:00Z.
- It was published at 2026-09-30T09:00:00Z and recorded as known/retrieved at 2026-09-30T09:15:00Z.
- Buyer purchase share: **unknown**.
- Buyer inventory buffer: **unknown**.
- The fixture contains no buyer-specific order, contract, production, shipment receipt, revenue or loss evidence.
- The observation does not prove that the supplier remained disrupted at the later 1 October decision frontier; no current-status refresh is provided.

### Evidence E-ENERGY-01

- Synthetic energy stress index: **124 index_points**.
- Observation period: **2026-09**.
- Published at 2026-09-30T10:00:00Z and known/retrieved at 2026-09-30T10:10:00Z.
- Validity interval: [2026-09-01T00:00:00Z, 2026-10-01T00:00:00Z).
- Buyer energy tariff: **unknown**.
- No baseline series, index rebasing rule, buyer tariff, energy consumption, contract or energy-cost exposure is included.
- At the decision frontier, this is a historical September index observation, not a measurement of the buyer's current tariff or October energy bill.

### Reference dispositions

| Atomic proposition | Reference state | Reason |
|---|---|---|
| A supplier shipment schedule disruption was reported | Supported | E-SUP-01 directly records it |
| The report was available before the decision frontier | Supported within the synthetic fixture | known_at precedes decision_at |
| The buyer purchases 40% from this supplier | Unsupported / not evaluable | No purchase-share evidence |
| The buyer has two weeks of inventory | Unsupported / not evaluable | Inventory explicitly unknown |
| September energy stress index equals 124 index_points | Supported | E-ENERGY-01 |
| 124 means energy costs rose 24% | Contradicted as an interpretation of this fixture | No base value, rebasing rule or percentage unit |
| The buyer's tariff rose | Unsupported / not evaluable | Tariff explicitly unknown |
| The supplier disruption caused a buyer revenue loss | Not established | No exposure, inventory, substitution, production or financial outcome evidence |
| The supplier disruption persists at the 1 October frontier | Not evaluable from supplied evidence | Last status observation is 30 September; no current refresh |
| Two conditional operational mechanisms remain plausible | Hypotheses only | Inputs necessary to discriminate them are unknown |

A critical distinction: absence of support in this evidence bundle is not equivalent to proof that a claim is false in the world. The scoring target is whether a response is justified by the permitted trace and whether its claim state is properly represented.

## 3. Masked candidate responses

### Candidate K-31

> As of 1 October 2026 at 12:00 UTC, the supplied bundle contains two observations that were available before the decision frontier. E-SUP-01 records a supplier shipment-schedule disruption reported on 30 September at 09:00 UTC and known to this evaluation profile at 09:15 UTC. It establishes that a disruption was reported; it does not establish the buyer's purchase concentration, inventory coverage, the existence of alternatives, or that the same status still held at the 1 October decision time.
>
> E-ENERGY-01 records a September energy stress index of 124 index_points, published at 10:00 UTC and known at 10:10 UTC on 30 September. Its validity period ends at 00:00 UTC on 1 October. It is therefore a historical September observation, not the buyer's current tariff or October energy cost. The fixture supplies no index baseline or rebasing rule, so 124 cannot be described as a 24% increase.
>
> Two competing conditional explanations remain open. H1: the buyer absorbs the supplier disruption if inventory and substitute capacity are adequate. H2: the disruption constrains production if the supplier is material, inventory is inadequate and substitutes cannot arrive in time. Neither is presently established. The energy index could be relevant to operating conditions, but the fixture does not connect it to the buyer's tariff or the supplier disruption.
>
> No buyer-specific production loss, revenue loss, margin impact or energy-cost increase can be calculated. The most decision-relevant next evidence appears to be buyer purchase share, inventory by relevant input, substitute-supplier capacity/lead time, open orders and shipment status, followed by the buyer's tariff/energy consumption and relevant contracts. These are qualitative evidence priorities, not calculated EVSI values.

### Candidate M-08

> The evidence reports a disruption to the supplier's shipment schedule and a September energy index reading of 124 points. The buyer's purchase share, stock buffer and energy tariff are not provided, so the buyer's exposure cannot currently be quantified.
>
> One possibility is that inventory or another supplier absorbs the disruption. Another is that production is delayed if the buyer is dependent on this supplier and has insufficient stock. We need current supplier status, purchase orders, inventory, substitute capacity and buyer energy pricing before assessing material financial impact.

### Candidate P-19

> The supplier is currently experiencing a shipment disruption, while energy stress remains elevated at 124, which is 24% above the normal level. These two developments point to a combined operational shock for the buyer. The buyer may face production delays and rising input expenses, although the exact effect depends on how much it purchases from the supplier. Further diligence should confirm the inventory position.

### Candidate T-74

> The supplier accounts for 40% of the buyer's purchases and the buyer has roughly two weeks of stock. The September energy index of 124 indicates a 24% rise in the buyer's electricity tariff. Together, these factors imply a 10% production decline and approximately $420,000 of lost quarterly revenue. The buyer should immediately switch to an alternative supplier and hedge its energy exposure.

## 4. Scoring rubric

Each dimension is scored 0–2:
- **0:** material failure, absent requirement, or unsupported assertion;
- **1:** partially correct, incomplete, or weakly traceable;
- **2:** correct and sufficiently explicit for the task.

| Dimension | What a score of 2 requires |
|---|---|
| Factual accuracy | Every material statement matches the frozen fixture |
| Evidence traceability | Uses the correct evidence identities and does not invent source support |
| Temporal admissibility | Distinguishes known-at, event/valid time, observation period and currentness |
| Units and periods | Preserves index_points, September period and validity boundary |
| Unknown preservation | Explicitly retains missing share, inventory, tariff and impact information |
| Competing hypotheses | Supplies at least two plausible, conditional, distinguishable mechanisms |
| Causal restraint | Does not infer causal impact from co-occurrence or graph proximity |
| Evidence-gap usefulness | Identifies specific next observations relevant to discriminating hypotheses |
| Calibration / abstention | Matches certainty to evidence and declines unsupported financial quantification |
| Reproducibility | Material statements can be mapped to the frozen facts, IDs, periods and assumptions |

### Non-compensatory hard gates

Regardless of the 20-point subtotal, a response is disqualified from a qualified research pass if it:
- fabricates material financial figures, buyer exposure or inventory;
- reports an expired/historical observation as current without qualification;
- converts index points into a percentage without a defined baseline;
- states a causal loss or production effect without the necessary evidence;
- presents a hypothesis or scenario as an observed fact;
- recommends or claims an external financial action as authorized by this task.

The score is descriptive only. There is no averaging that can offset a hard-gate violation.

## 5. Masked adjudication result

| Candidate | Facts | Trace | Time | Units | Unknowns | Hypotheses | Causality | Gaps | Calibration | Replay | Total | Hard gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| K-31 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | **20/20** | Pass |
| M-08 | 2 | 1 | 1 | 1 | 2 | 1 | 2 | 1 | 2 | 1 | **14/20** | Pass, limited |
| P-19 | 1 | 1 | 0 | 0 | 1 | 1 | 0 | 1 | 0 | 0 | **5/20** | Fail |
| T-74 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **0/20** | Fail |

### Adjudicator explanations

**K-31 — strongest disposition.** It distinguishes the supplier report from current supplier status, preserves the September validity interval, does not interpret index points as percentages, and refuses buyer-specific financial quantification. It supplies two conditional hypotheses and evidence requests that would help discriminate them. Its VOI prioritization is correctly labelled qualitative rather than computed. No material hard-gate breach observed.

**M-08 — acceptable but incomplete.** It is factually restrained and preserves major unknowns. It omits exact evidence IDs and key temporal details, does not distinguish the September index's validity endpoint from the decision time, and gives less precise evidence-gap prioritization. It passes the basic safety gates but is not decision-ready.

**P-19 — hard failure.** It calls the supplier disruption current without a current observation, treats index points as a 24% increase without a baseline, and suggests a combined buyer shock without evidence linking the supplier and energy observations to the buyer. The caveat at the end does not repair those unsupported assertions.

**T-74 — hard failure.** It fabricates purchase concentration, inventory, tariff movement, production decline and quarterly revenue loss. It also recommends switching supplier and hedging without a mandate or adequate diligence. The numerical specificity makes the response more dangerous, not more informative.

## 6. Evaluator uncertainty and limitations

This is one AI adjudicator and one deliberately constructed synthetic calibration set. The adjudicator has no independent human vote, no second model judge, no blinded source authorship, no sampling frame and no statistical power for generalized claims. The response examples were authored as rubric probes rather than collected from model executions.

The exact fixture makes several objective checks possible, but some scoring dimensions—especially evidence-gap usefulness—still involve judgment. The 20-point result is not a benchmark metric, model score, or human-preference measurement.

The correct interpretation is:

- The rubric distinguishes obvious evidence-grounded analysis from fabricated precision in these examples.
- The hard gates prevent fabricated financial values from being compensated by prose quality.
- Temporal scope and unit interpretation deserve explicit score dimensions.
- The current fixture does not test source proof authentication, interval-censored vintages, causal identification or real company analysis.
- A truly independent panel and actual agent outputs remain necessary before ranking systems.

## 7. Recommended next evaluation pass

1. Freeze a larger sealed synthetic cohort before exposing candidates.
2. Collect actual outputs from each workflow under identical prompt, information, tools, model budget and deadline.
3. Assign randomized candidate IDs and retain an evaluator-only mapping to the source systems.
4. Run this AI adjudicator and at least one separately configured judge independently; preserve all raw votes and abstentions.
5. Select a stratified sample for human finance/operations adjudication, with independent initial votes and conflict disclosure.
6. Compare AI and human decisions against deterministic evidence/time/unit/arithmetic oracles where available.
7. Report false accepts, false rejects, disagreement and uncertainty by task family; do not average away contested references.
8. Do not release any human or model leaderboard until the reference labels, cohort and evaluator are qualified.

## 8. Claim ceiling

This pilot demonstrates only a masked AI adjudication procedure on one public synthetic development fixture. It does not show that Symthaea or any other financial agent beats SOTA, that an actual human panel agrees, or that the rubric generalizes to real financial research.
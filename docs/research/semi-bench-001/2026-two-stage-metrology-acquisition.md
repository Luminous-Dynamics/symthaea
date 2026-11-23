# SEMI-MF-001 — Two-stage metrology acquisition

## Purpose

This workstream treats metrology as an evidence-acquisition path instead of a one-shot measurement ranking. A low-cost screen is applied first; the planner then decides, separately for positive and negative screen outcomes, whether high-fidelity confirmation has enough posterior-conditional value to justify its cost.

The current implementation is a synthetic decision planner. It does not execute an instrument, create a physical label or grant process authority.

## Decision logic

For candidate prior defect probability p and low-fidelity assay sensitivity s_L / specificity c_L:

- P(L+) = p s_L + (1-p)(1-c_L)
- P(L-) = 1 - P(L+)
- P(D|L+) = p s_L / P(L+) when P(L+) > 0
- P(D|L-) = p(1-s_L) / P(L-) when P(L-) > 0

The high-fidelity assay EVSI is evaluated conditional on each defined posterior. A branch is eligible for confirmation only when gross EVSI exceeds confirmatory measurement cost. An impossible outcome has no posterior and is reported explicitly as OutcomeImpossible rather than producing NaN or an invented fallback.

For a selected candidate with positive and negative confirmation branches:

- expected net value = Σ_z P(z) × max(gross_EVSI(z) - high_cost, 0)
- expected confirmation cost = Σ_z P(z) × high_cost × I(confirm on z)
- worst-case confirmation cost = high_cost when at least one positive-probability branch invokes confirmation; otherwise zero.

The exact subset selector maximizes portfolio expected net value subject to the **worst-case** high-fidelity capacity budget. This is deliberately stricter than limiting expected confirmation expenditure. If the screen result is unknown when the queue is scheduled, the system must reserve enough capacity for the worst realized branch.

## Comparator baselines

The report produces expected decision-loss baselines for:
- no measurement;
- low-fidelity screening only;
- high-fidelity-only;
- the selected two-stage screen/confirm plan.

It reports no-measurement, screen-only and high-fidelity-only total resource costs alongside two-stage expected spend, expected confirmation cost and total worst-case spend. For fair comparisons, external benchmark runs must additionally match target-label draws, actual throughput/queue capacity and escaped-defect risk constraints.

## Integrity and deterministic replay

- Candidates are canonicalized by candidate ID.
- The caller supplies a digest of the canonical candidate packet; it must match.
- Policy and final report digests bind the assumptions and resulting plan.
- The exact subset search is bounded at 18 candidates and uses deterministic tie-breaking.
- A fixed screening budget is checked before portfolio selection.
- A hard worst-case confirmation budget is enforced for the selected portfolio.

## Tests

The Rust suite covers deterministic replays, uninformative screening, uninformative confirmation, confirmatory cost above gross EVSI, worst-case budget enforcement, candidate-order invariance, packet-digest mismatch, invalid assay/cost/identity and zero-probability screen outcomes.

## Integration

- #13328 — single-stage EVSI and robust metrology policy replay.
- #13336 — independently replayable selection receipt V2.
- #13356 — MSA and future assay-characteristic evidence.
- #13382 — calibration/uncertainty qualification before label promotion.
- #13273 — preregistered predictive, metrology, privacy and industrial scorecard.

A selected confirmation branch is only a plan. Once measured, the actual result must pass the calibration qualification gate before label promotion or VM update. A missing/stale/unqualified result must remain UNKNOWN or REJECTED; it must not silently fall back to a screen-only label where policy forbids that.

## Claim ceiling

The model uses declared synthetic priors, assay characteristics, costs and loss functions. It is not a validated real-fab EVSI or savings claim. Actual assay parameters must come from characterized instruments and traceable measurement studies, and actual decisions require authorized process/metrology owners. No actuation authority is added.

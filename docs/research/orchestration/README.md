# Adaptive research orchestration

Owner: #11012.

This lane tests whether research outcomes improve when the next role/tool/experiment is selected from the current evidence state, rather than following a fixed sequence.

## Artifacts

- `SYM_RSI_ADAPTIVE_ORCHESTRATION_PROTOCOL_V0_2026-11-07.md`: fixed-versus-adaptive treatment arms, causal controls, cost boundaries, metrics, transfer and claim ceilings.
- `SYM_RSI_ORCHESTRATION_COUNTERFACTUAL_FIXTURES_V0_2026-11-07.json`: six synthetic branch controls.
- `SYM_RSI_ORCHESTRATION_COUNTERFACTUAL_SCHEMA_V0_2026-11-07.json`: structural schema for the synthetic pack.

## Counterfactual cases

- Decision-relevant signal: action must change.
- Irrelevant metadata: action must remain invariant.
- Contradictory mechanism evidence: pivot or replicate based on control validity.
- Non-identifiability: abstain or request a truly discriminating intervention.
- Negative net EVSI after direct and opportunity cost: stop/defer.
- Positive net value but failed downside risk threshold: redesign or reject.

These fixtures test policy trace semantics only. They are not expert gold labels, benchmark results or scientific-discovery evidence.

## Composition

- #10877 research taste;
- #9230 direction board;
- #8721 scientific tool allocation;
- #10921 AS-Bench replay/twin adapter;
- #9332 sealed oracle and benchmark generation;
- #10756 evaluator integrity;
- #10717 causal attribution;
- #10630 evidence continuity;
- #10497 R4 qualification.

No duplicate scheduler, evaluator, receipt, memory or promotion authority is introduced. No live laboratory execution is authorized by this work.

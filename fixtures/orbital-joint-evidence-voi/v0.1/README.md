# Orbital Joint Evidence VOI Scenario Corpus v0.1

**Tracker:** [#13111](https://github.com/Luminous-Dynamics/symthaea/issues/13111)  
**Specification:** [ORBITAL_JOINT_EVIDENCE_VOI_V0_1.md](../../../../docs/research/ORBITAL_JOINT_EVIDENCE_VOI_V0_1.md)  
**Status:** hand-authored synthetic contract fixtures; not yet executed by an action-selection engine.

## Files

- `scenarios.json`: 11 synthetic scenarios, explicit state/loss contexts, action proposals, resource vectors and expected qualitative/known-answer checks.
- `scenarios.schema.json`: JSON Schema Draft 2020-12 for the v0.1 corpus.

## Scenario coverage

| Scenario | Question |
|---|---|
| JEV-001 | Does a decision-relevant coupon test beat an irrelevant mass-only measurement under hard budgets? |
| JEV-002 | Does calibrated EVSI justify a discriminating test when explicit synthetic priors are supplied? |
| JEV-003 | Does the policy abstain when every available action is decision-irrelevant? |
| JEV-004 | Does a perfect but over-budget test remain unselectable? |
| JEV-005 | Is an informative test rejected when it targets a stale configuration? |
| JEV-006 | Are expired calibration and unresolved common-cause dependencies blocking conditions? |
| JEV-007 | Can a single-axis test improperly resolve a coupled-load claim? It must not. |
| JEV-008 | Can coupon evidence be promoted to a production-joint claim without a transfer bridge? It must not. |
| JEV-009 | Can an information-rich action replace a pre-registered critical campaign interaction? It must not. |
| JEV-010 | Can a first-mating test resolve wear/lifecycle claims? It must not. |
| JEV-011 | Does an aborted test retain its cost and evidence gap rather than becoming a pass? |

## Important interpretation

These are small deterministic known-answer controls. They do not constitute a statistically representative aerospace qualification dataset, physical test plan, calibrated probability model, or benchmark result.

The loss matrix uses synthetic decision-loss units. It is not currency, a safety factor or an engineering acceptance limit. Only JEV-002 supplies an explicitly labelled synthetic prior and likelihood model; all other scenarios are distribution-free and have `prior: null`. Do not derive probabilities from the other scenarios' hard state sets.

The scenario catalogue separates:
- action admissibility from action value;
- action resource requirements from decision loss;
- expected-value semantics from distribution-free semantics;
- physical-unit identity from generic component identity;
- source-scale evidence from target-scale transfer;
- pre-registered test obligations from general information gain;
- planned actions from actual execution outcomes.

Some expected decisions are qualitative contract targets (for example, `ABSTAIN`) pending the first independently reviewed decision-value oracle. No action-selection policy has yet been evaluated against this corpus. The next implementation step is a dry-run preflight and independently recomputed one-step oracle; it must not call a solver or execute a physical test.

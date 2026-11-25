# Orbital Joint Evidence VOI Scenario Corpus v0.1

**Tracker:** [#13111](https://github.com/Luminous-Dynamics/symthaea/issues/13111)  
**Specification:** [ORBITAL_JOINT_EVIDENCE_VOI_V0_1.md](../../../docs/research/ORBITAL_JOINT_EVIDENCE_VOI_V0_1.md)  
**Status:** 17 hand-authored one-step contract fixtures, two frequency variants of a finite-horizon joint campaign, and one AER-01 calibration/load/NDE campaign; consumed by deterministic Rust auditors and independent Python reference oracles.

## Files

- `scenarios.json`: 17 synthetic scenarios, explicit state/loss contexts, action proposals, resource vectors and expected qualitative/known-answer checks.
- `scenarios.schema.json`: JSON Schema Draft 2020-12 for the v0.1 scenario corpus.
- `action-proposal-outcome.schema.json`: strict JSON Schema for separate `EvidenceActionProposalV1` and `EvidenceActionOutcomeV1` records.
- `action-proposal.synthetic.json`: non-actuating, versioned proposal example with an external approval still marked review-required.\n- `action-outcome-aborted.synthetic.json` binds to the exact proposal bytes with a SHA-256 reference; pair validation rejects changed proposal bytes or context replay.
- `action-outcome-aborted.synthetic.json`: aborted execution example preserving actual resources and an indeterminate result.
- [Executable dry-run preflight](https://github.com/Luminous-Dynamics/symthaea/tree/research/orbital-joint-evidence-voi-v0-1/tools/orbital-joint-evidence-voi-bench): resource/admissibility/claim checks, finite-state minimax regret and one explicit calibrated EVSI case.
- A separately implemented standard-library Python oracle under the same tool directory.

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
| JEV-012 | Are actual resources preserved when an aborted test exceeds its estimate but stays within the scenario budget? |
| JEV-013 | Are actual resources retained and reported when an aborted test exceeds both estimate and total budget? |
| JEV-014 | Do equal-value actions with incomparable resource plans trigger human review rather than a fabricated cost scalar? |
| JEV-015 | Are self-declared-admissible actions rejected when profile, physical serial, configuration or ICD revision differs from the decision context? |
| JEV-016 | Does an incomplete owner-native change-event frontier block even an otherwise informative requalification action? |
| JEV-017 | Does a complete change-impact receipt require targeted coverage of each claim marked for requalification while preserving unrelated claims? |


## Sequential campaign result (SEQ-001)

This separate finite-horizon evaluator catches a failure mode the one-step selector cannot: neither test by itself improves worst-case minimax regret, but a contingent two-test policy can. The terminal model separates ACCEPT, REJECT and HOLD; no evidence defaults to HOLD, rejection requires an outcome-specific rejection-basis observation, and acceptance requires both affirmative evidence obligations.

| Policy | Worst-case minimax regret | Expected engineering hours | Expected facility hours |
|---|---:|---:|---:|
| Stop now (HOLD) | 2 | 0.00 | 0.00 |
| Myopic one-step (HOLD) | 2 | 0.00 | 0.00 |
| Fixed screen + thermal inspection | 0 | 1.50 | 2.00 |
| Adaptive: screen; thermal only on PASS | 0 | 1.20 | 1.55 |

For SEQ-001, the adaptive policy runs the combined-load screen first, terminates at REJECT on FAIL when that observation produces a rejection basis, and runs thermal inspection only on PASS. Its expected resource use is 20% lower in engineering hours and 22.5% lower in facility hours than the fixed campaign. SEQ-002 changes only the synthetic state-frequency profile; the exact planner reverses the order, running thermal inspection first and the combined-load screen only on CLEAR. Both policies preserve zero worst-case regret and improve every expected resource dimension versus the fixed campaign.

The weights are synthetic and used only to calculate expected cost; the terminal structural decision remains distribution-free. These numbers are fixture calculations, not physical or industry results. The sequential oracle is a proof-of-concept, not a full held-out policy tournament.

## AER-01 inspection and structural evidence campaign (SEQ-003)

The third sequential scenario is an explicitly synthetic, article-bound AER-01 workflow with four latent states: healthy, delamination, joint-retention defect and common-cause instrument drift. All actions bind to the same coupon-set physical-unit reference, configuration and ICD revision; the load and NDE actions consume separate specimens from that set.

The exact finite-horizon policy is:

1. Run an independent calibration-chain audit. If invalid, stop at **HOLD**; downstream readings are not trusted.
2. If valid, run a non-destructive combined-load proof screen on the load coupon. A **FAIL** creates an outcome-specific rejection basis and stops at **REJECT**.
3. If the load screen **PASSES**, run fixed-target ultrasonic inspection on the separate NDE coupon. **CLEAR** completes the calibration, load and NDE obligations required for **ACCEPT**; an **INDICATION** creates a rejection basis and leads to **REJECT**.

No single test improves the baseline worst-case regret: HOLD and the myopic one-step stop rule both have regret 2. The full fixed campaign and the adaptive three-step policy both reach regret 0. The adaptive policy stops early on invalid calibration or a load failure and avoids the NDE specimen cost when the earlier calibration or load result already determines the disposition.

| Resource | Fixed campaign | Adaptive policy | Expected reduction |
|---|---:|---:|---:|
| Engineering hours | 1.40 | 1.25 | 10.7% |
| Facility hours | 1.40 | 1.215 | 13.2% |
| Specimen count | 2.00 | 1.75 | 12.5% |
| Review hours | 0.270 | 0.235 | 13.0% |
| External cost units | 0.600 | 0.5275 | 12.1% |

The reductions are outputs of the frozen synthetic frequency model and resource plan, not measured savings, physical POD, proof-load results, production test evidence, qualification or a demonstrated advantage over industry workflows. The frequency weights affect expected resource consumption only; worst-case minimax regret remains distribution-free over the declared state set. The common-cause state deliberately terminates in HOLD rather than permitting a confident acceptance or unsupported rejection.

## Important interpretation

These are small deterministic known-answer controls. They do not constitute a statistically representative aerospace qualification dataset, physical test plan, calibrated probability model, or benchmark result.

The loss matrix uses synthetic decision-loss units. It is not currency, a safety factor or an engineering acceptance limit. Only JEV-002 supplies an explicitly labelled synthetic prior and likelihood model; all other scenarios are distribution-free and have `prior: null`. Do not derive probabilities from the other scenarios' hard state sets.

The sequential campaign additionally separates outcome-specific evidence production from decision disposition: `PASS`/`CLEAR` add affirmative claim obligations; `FAIL`/`INDICATION` add rejection-basis evidence; an empty evidence path produces `HOLD`.

The scenario catalogue separates:
- action admissibility from action value;
- action resource requirements from decision loss;
- expected-value semantics from distribution-free semantics;
- physical-unit identity from generic component identity;
- source-scale evidence from target-scale transfer;
- pre-registered test obligations from general information gain;
- planned actions from actual execution outcomes.

The Rust preflight and Python oracle audit the scenario catalogue's admissibility, component-wise resource budgets, exact claim and article/configuration/ICD context matching, owner-native change-frontier completeness, targeted requalification coverage and dependency overlap, critical-obligation coverage, robust regret and the explicit EVSI known answer. Rust also deserializes the strict proposal/outcome examples: an attempted outcome requires actual cost and external execution authority, while a proposal is permanently non-actuating and carries no authority. They now retain actual resource totals for attempted actions, distinguish planned estimates from actual consumption, and flag action-level and aggregate budget overruns. An execution state without an actual-cost record (or actual cost without an execution state) is rejected; actual overruns are recorded, not discarded. Proposal and outcome records are also validated as a pair: the outcome binds the proposal ID, revision, exact proposal-byte SHA-256 and four-part decision context. An outcome cannot be replayed against a different serial/configuration or altered method/resource proposal; an aborted/non-successful result cannot be promoted to `VERIFIED_WITHIN_SCOPE`. The audit now selects the highest decision-value eligible action under hard gates. Equal-value options are resolved only by resource-vector Pareto dominance; incomparable resource plans return `HUMAN_REVIEW_REQUIRED_RESOURCE_PARETO_TIE` for human review rather than a fabricated scalar cost score. The sequential evaluator tests two frequency variants of the original joint campaign and one AER-01 calibration/load/NDE campaign against fixed and myopic baselines; a broad multi-step held-out matched-budget tournament remains unimplemented. No solver or physical test is called.


## Finite-horizon sequential campaign proof-of-concept

- `sequential-campaign.schema.json` and `sequential-campaign.synthetic.json` define two frequency variants of the original joint campaign plus an AER-01 four-state, three-action inspection/structural campaign.
- `tools/orbital-joint-evidence-voi-bench/src/sequential.rs` implements exact finite-horizon minimax-regret planning with observation-contingent branches, stop decisions, exact profile/article/configuration/ICD binding and full path budgets.
- `src/bin/sequential-campaign.rs` emits a deterministic receipt; `tests/sequential_reference_oracle.py` independently recomputes the policy.

The known-answer example is intentionally constructed so both single tests individually leave worst-case minimax regret at 5. A myopic one-step policy therefore stops. A two-step adaptive policy performs the combined-load screen first, rejects immediately on `FAIL`, and performs thermal inspection only on `PASS`. It reaches worst-case regret 0, matching the fixed two-test campaign, while reducing expected resource use under the explicitly synthetic frequency model:

| Resource | Fixed campaign | Adaptive policy | Expected reduction |
|---|---:|---:|---:|
| Engineering hours | 1.50 | 1.20 | 20.0% |
| Facility hours | 2.00 | 1.55 | 22.5% |
| Specimen units | 2.00 | 1.70 | 15.0% |
| Review hours | 0.30 | 0.24 | 20.0% |
| External cost units | 1.10 | 0.86 | 21.8% |

These are calculated outcomes of two synthetic fixtures, not empirical aerospace savings or SOTA results. Its frequency weights are used exclusively for expected resource accounting; worst-case structural decision regret remains distribution-free over the full state set.


**Terminal disposition rule:** no evidence defaults to HOLD. A terminal ACCEPT requires both affirmative obligations to be accumulated along the exact action path. A terminal REJECT requires an outcome-specific rejection-basis obligation. HOLD/INDETERMINATE remains distinct from a proven-defect rejection. Missing or undeclared action obligations are rejected by both the Rust evaluator and its independent Python oracle.


### Sequential search implementation boundary

The exact finite-horizon planner memoizes by canonical residual belief, exact remaining resource vector, remaining action depth, used-action set and accumulated evidence obligations. Its receipt reports expanded states, memo hits, memo entries and output policy-tree node count. A regression test verifies replay of an identical subproblem returns the same policy without expanding it again.

Nested unresolved resource comparisons are propagated to the parent policy's human-review frontier. A parent may not silently discard a tied child policy with unknown expected cost and select another tied plan with a known cost. Resource frequency is fixture-only and used for expected resource accounting; the terminal decision objective remains distribution-free worst-case regret.

This is a small-instance exact reference. No asymptotic or industrial campaign-scale performance claim is made. The next scaling study must measure state expansions, memo hits, Pareto pruning, wall time, memory, policy-tree size and exact-oracle regret gap under a frozen generated action/state sweep.

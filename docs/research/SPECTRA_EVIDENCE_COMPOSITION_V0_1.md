# SPECTRA Evidence-Carrying Decisions — Research Specification v0.1

**Status:** Research specification; no top-level evidence composer or operational authorization.
**Issue:** [#10645 — typed scientific evidence composition](https://github.com/Luminous-Dynamics/symthaea/issues/10645)
**Programme:** [#10490 — robust goal-directed sequential design](https://github.com/Luminous-Dynamics/symthaea/issues/10490)
**Selection baseline:** [#10585 — robust epsilon-optimal selection](https://github.com/Luminous-Dynamics/symthaea/pull/10585)

## 1. Objective

Make every SPECTRA decision carry the separate evidence needed to justify its particular claim. The system must be able to say:

- which candidate is epsilon-optimal over a precisely frozen simulation scope;
- whether that candidate corresponds to an operationally realizable experimenter action;
- whether the observation model passes independent adequacy checks;
- what scenario-bank support has and has not been established;
- what known-truth inferential-risk regimes have been tested;
- which resource and protocol constraints remain unresolved;
- and whether all relevant receipts can be replayed.

Do not collapse these into one scalar confidence, one 'valid' boolean, or one weighted score. They are different propositions with different evidence and invalidation rules.

## 2. Current architectural boundary

The current SPECTRA implementation already separates several important concerns:

- `score_bounds` retains raw bounded score vectors, declared ranges, method metadata, action/scenario snapshots and the operational-mapping / model-adequacy flags supplied to each score record.
- `ranking` independently recomputes the score receipt before issuing a robust epsilon-optimality certificate over the frozen action/scenario set.
- `scenario` aggregates per-action scenario outcomes and derives action-level eligibility by requiring operational mapping and model adequacy to be qualified across the scenario set.
- #10582 separately specifies held-out predictive model-adequacy checks.
- #10602 specifies known-truth downstream inferential-risk evaluation.
- #10618 specifies scenario-bank coverage / hidden-world assurance.
- #10642/#10643 specify deterministic allocation-receipt replay.

The robust ranking certificate is not the same proposition as action eligibility, model adequacy, scenario-bank coverage, or permission to execute a human study. Keep these boundaries explicit.

## 3. Typed evidence claims

Represent at least the following evidence classes as separate typed nodes.

### 3.1 Score-estimation evidence

Bind the raw score receipt, method ID, familywise alpha, claim multiplicity, ranges, sample counts, generator/PRNG version, frozen action/scenario snapshots and the independent score-receipt verifier result. State whether the method is fixed-look, infinite-horizon time-uniform, or finite-horizon with an exact predeclared budget.

### 3.2 Robust selection evidence

Bind the exact action set, scenario set, scenario weights or ambiguity set, robust objective, CVaR tail mass where applicable, epsilon, simultaneous utility intervals, selected action, regret upper bound and certificate disposition. The claim is only over that frozen scope and under the score-bound assumptions.

### 3.3 Operational mapping evidence

Bind the experimenter-controlled action to the physical / procedural stimulus manipulation, observable response channel, protocol version and permitted range. Latent participant confidence criteria, latent evidence thresholds, internal participant state or model parameters are not directly controllable actions.

A numerical winner with no qualified action-to-observation mapping remains a research candidate, not an operational recommendation.

### 3.4 Model-adequacy evidence

Bind the model family, fitting data, training/validation split, held-out unit (trial/session/observer), predictive statistic, null/reference construction, calibration method, multiplicity and all failure dispositions. Adequacy must be independently evaluated and may hard-veto an action even when its score is high.

### 3.5 Scenario-bank evidence

Bind the exact scenario bank, development/evaluator manifest, weight semantics, ambiguity set, hidden-world evaluation, false-robustness results and declared omissions. A certificate cannot silently inherit a broader scenario scope after the bank changes.

### 3.6 Inferential-risk evidence

Bind the target estimand and target distribution, known-truth evaluator version, misspecification family, bias/RMSE, interval coverage, confidence-amplification and worst-case risk results. These are external qualification results for the tested regimes, not real-time online evidence of truth.

### 3.7 Resource / protocol feasibility evidence

Bind participant-time, compute budget, burden, safety limits, stopping policy, mandatory sentinel/audit floors, protocol version and any external approvals required for a future human study. A software evidence bundle does not create ethics approval or regulatory authorization.

### 3.8 Replay and provenance evidence

Bind source commit, method IDs, dependency/toolchain versions, generator and PRNG versions, seed/replication identities, frozen inputs, raw receipts, verifier versions and replay results. Caller-supplied opaque digests remain commitments only unless separately authenticated.

## 4. Common evidence-node contract

Every evidence node should expose:

- `claim_type`: a closed, versioned enum; no free-form string used as authority;
- `scope`: exact action/scenario/target/model/data/protocol manifest references;
- `source_receipts`: immutable references to the evidence that supports the claim;
- `method_id` and `method_version`: exact construction and implementation;
- `assumptions`: explicit sampling, model, support and process assumptions;
- `unsupported_regimes`: named cases not covered by the evidence;
- `disposition`: qualified, failed, unknown, not-evaluated, expired, or out-of-scope;
- `validity_window` / `invalidation_conditions`: what changes make this node stale;
- `verifier_receipt`: exact verifier version and deterministic replay outcome.

The node should not carry a generic confidence number unless its claim has a mathematically defined probability statement. Qualitative gate results must remain qualitative.

## 5. Composition rules

1. **No implication laundering.** A valid score receipt does not imply model adequacy. A robust selection certificate does not imply operational mapping. A hidden-world result does not imply universal coverage.
2. **Hard vetoes.** Any mandatory adequacy, action-mapping, safety, protocol or provenance gate that is failed blocks the supported-decision disposition.
3. **Unknown is not pass.** Missing, not-evaluated, expired or out-of-scope mandatory evidence cannot be silently coerced to qualified.
4. **Scope intersection.** The composite decision scope is no broader than the intersection of all supporting evidence scopes. A broader requested claim must be narrowed or re-evaluated.
5. **No invented joint confidence.** Do not multiply confidence levels or sum alpha values unless the joint event and dependence/multiplicity derivation are explicitly defined and valid.
6. **Frozen-scope certificates remain historical.** Scenario, action, estimand, model, method or data-manifest mutations require fresh evidence nodes and a new composition receipt; never mutate the original bundle in place.
7. **Contradictions remain visible.** If evidence nodes conflict, preserve both and return blocked / not-evaluable with a conflict reason. Do not pick the more favorable result.
8. **No override by scalar utility.** Information gain, representativeness, utility or expected value of computation cannot overrule a mandatory scientific veto.

## 6. Decision-state lattice

Use explicit dispositions rather than a Boolean `valid` field:

| Disposition | Meaning |
|---|---|
| `ResearchCandidateOnly` | Candidate or ranking exists, but one or more mandatory evidence claims are missing / unqualified. |
| `FrozenScopeSelectionCertified` | Epsilon-optimality is certified over the exact frozen score/action/scenario scope. This is not an operational authorization. |
| `DecisionSupportedForDeclaredRegime` | Every mandatory scientific, adequacy, mapping, scenario and resource gate is qualified for the explicitly declared regime and scope. |
| `Blocked` | At least one mandatory evidence node failed, expired, is contradicted, or is out-of-scope. |
| `NotEvaluable` | Available evidence cannot assess one or more mandatory claims. |

Only an external protocol / governance process can authorize a human-subject experiment. Even `DecisionSupportedForDeclaredRegime` is not an automatic permission to collect data.

## 7. Scope binding and invalidation

The composite receipt must bind, at minimum:

- action set and operational mapping version;
- scenario bank, scenario weights / ambiguity set and target population;
- estimand and utility definition;
- score ranges, alpha, multiplicity, confidence-sequence method and campaign horizon;
- generator, PRNG, code commit and dependency/toolchain versions;
- fitted model, training split, held-out split and adequacy statistic;
- inferential-risk target distribution and evaluation manifest;
- resource budget, mandatory audit floor and protocol constraints;
- all source evidence receipt IDs and verifier versions.

Any change to a bound input invalidates the dependent evidence nodes according to their declared dependency graph. It does not necessarily invalidate unrelated nodes; the verifier should recompute the dependency closure and preserve historical certificates with their narrower scopes.

## 8. Required fail-closed negative controls

Construct a compositional test matrix where one gate changes while the other gates are held fixed:

1. Score certificate passes; model adequacy fails.
2. Score certificate passes; operational action mapping is unqualified.
3. In-bank selection certificate passes; hidden-world false-robustness fails.
4. Adequacy passes; epsilon-selection certificate fails.
5. Scientific gates pass; resource / protocol feasibility fails.
6. One evidence node is missing, unknown, expired or not-evaluated.
7. One node refers to a different action/scenario/generator version.
8. One raw receipt, score-bound output, allocation priority, batch seed or verifier receipt is mutated.
9. Scenario weights or ambiguity radius change after selection.
10. Target estimand, validation split or held-out evaluation manifest changes.

Every mandatory invalid-gate case must produce `Blocked` or `NotEvaluable`, never `DecisionSupportedForDeclaredRegime`. Preserve the underlying valid sub-certificates where their original narrower scope remains intact.

## 9. Benchmark metrics

Add evidence-composition tests to #10601 and the scenario assurance campaign:

- false supported-decision rate on seeded invalid-gate controls;
- missed-block rate;
- correct scope-invalidation and dependency-closure rate;
- raw evidence / composite receipt replay rate;
- disposition availability and reason-code completeness;
- latency / resource cost to compose and verify a decision;
- stale-evidence detection rate;
- contradictions and non-evaluable cases preserved in denominators.

Proposed target: zero false supported-decision outcomes on the complete predeclared invalid-gate control suite. Report confidence / uncertainty for any stochastic evaluator; deterministic mutation-control counts should be exact.

## 10. Proposed qualification targets — not results

- Zero `DecisionSupportedForDeclaredRegime` outcomes when any mandatory gate is false, unknown, stale or out-of-scope.
- 100% exact scope binding across all linked evidence nodes.
- 100% deterministic replay for supported receipt types.
- 100% visibility of missing, failed, contradictory and non-evaluable evidence.
- No single aggregate confidence field that hides distinct claims.
- No changes to existing score-bound, selection, SPC or scenario-assurance guarantees to make composition easier.

These are targets for implementation and verification, not achieved results.

## 11. Implementation sequence

1. Define closed typed enums and immutable node references; avoid duplicating existing score, scenario, policy or model-evidence authorities.
2. Add a schema / validator for evidence nodes and composite receipts.
3. Define the scope dependency graph and exact invalidation rules.
4. Implement the disposition lattice and fail-closed composition function.
5. Add the ten negative-control families above.
6. Add deterministic replay for the composite receipt, reusing the existing independent verifiers rather than reimplementing their mathematics.
7. Integrate hidden-world and inferential-risk evidence without converting them into online confidence claims.
8. Keep human-subject execution disabled pending external protocol authorization.

## Non-goals

- No overall probability-of-truth number.
- No replacement of existing confidence sequences, selection certificates, model-adequacy checks or scenario assurance.
- No automatic ethics, regulatory, safety or study authorization.
- No human-subject data collection or live experiment execution.
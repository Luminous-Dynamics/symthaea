# CMAQ-022 — NixWard HDC query-sufficient state-abstraction adapter v0.1

**Status:** executable synthetic known-answer integration test; exact-head qualification pending.  
**Composition:** NixWard `NixWorldModel` family-label-free contextual HDC-delta predictor → frozen query readout → CMAQ-020 `score_abstraction`.  
**Claim ceiling:** this fixture demonstrates API integration and finite toy-world behavior only. It is not external benchmark performance, held-out-mechanism transfer, causal identification, physical safety or SOTA evidence.

## Why this integration exists

CMAQ-020 defines state equivalence relative to a registered intervention × query × horizon response contract. NixWard provides an actual source-inspected transition predictor: a per-action HDC mean-delta baseline and a contextual HDC-delta candidate. The adapter should test whether NixWard predictions can induce a useful response-equivalence partition, instead of treating the evaluator and world model as disconnected artifacts.

The adapter is a **dev-only dependency** from NixWard to the CMAQ-020 evaluator. No production NixWard code imports or calls CMAQ. The root lockfile pins the local package graph, and a dedicated workflow tests both the NixWard world-model module and standalone CMAQ crate.

## Frozen toy contract

The test uses one intervention `install`, one query `coordinate_0_after_action`, and horizon 1. The query is a deliberately simple scalar readout of the first coordinate of the post-action HDC state; it is a synthetic fixture, not a semantic NixOS observation.

Two mechanism families have different action consequences under different observable context HDCs:

- mechanism A: coordinate-0 delta is +0.1;
- mechanism B: coordinate-0 delta is +0.5.

All four evaluation states start with coordinate 0 equal to 0.1. Within each mechanism, two states differ only in a nuisance coordinate. Therefore, after the permitted intervention, the two A states have response 0.2 and the two B states have response 0.6. In the evaluator's integer units these are 200 and 600.

The world model is trained on three distinct training clusters per family. Evaluation uses four distinct held-out cluster IDs. The candidate inference API is `predict_state_without_family_label`: it receives current state, action, action-parameter HDC and observable context HDC, but no environment-family ID. The predictor searches the learned contextual buckets and abstains when tied matches disagree.

## Candidate/evaluator separation

The test creates candidate abstraction labels from the predicted HDC state using only the frozen coordinate-0 readout. It does not use the evaluator's reference response, family ID or reference partition to form that label. The evaluator response values are attached to `StateResponseRecord` only after candidate labels have been computed.

The evaluator then checks whether the induced candidate classes preserve exact equivalence for the registered contract. This is deliberately a partition test; it does not certify the model's generalization outside the fixture.

## Matched controls

The same four evaluation states and response contract are scored under four partition sources:

1. **NixWard contextual HDC:** family-label-free context/action-parameter HDC prediction, then the frozen coordinate-0 readout.
2. **Pooled global HDC delta:** existing category-only running-mean transition baseline, using the same readout.
3. **Identity/no-op:** unchanged input state, using the same readout.
4. **Full-state partition:** one candidate class per distinct predicted HDC vector, exposing the over-refinement cost of preserving every coordinate.

The deterministic assertions require:

| Candidate | Exact query responses | False merges | False splits | Candidate classes |
|---|---:|---:|---:|---:|
| Contextual HDC query readout | 4/4 | 0 | 0 | 2 |
| Pooled global delta readout | 0/4 | 4 | 0 | 1 |
| Identity/no-op readout | 0/4 | 4 | 0 | 1 |
| Full predicted HDC state | not measured | 0 | 2 | 4 |

The global predictor pools the two distinct action effects and therefore predicts the same coordinate-0 value for all four states. The identity predictor also leaves the coordinate unchanged for all four. Both collapse states with different permitted future responses. The full-state partition remembers nuisance detail and splits response-equivalent states. The query-conditioned HDC partition is expected to retain exactly the distinction needed by the frozen contract, and no more.

The table above describes the fixture's asserted known-answer behavior; it must not be represented as an observed CI pass until the exact-head workflow completes.

## What this does not establish

- The two mechanism families are represented in training. The evaluation tests held-out state clusters within known families, **not zero-shot held-out-family transfer**.
- The family-label-free inference API removes family ID from the query, but it does not prove discovery of a latent causal mechanism.
- The fixed similarity and conflict thresholds are engineering constants, not tuned optima.
- The query is one synthetic coordinate; real NixOS evaluations need frozen, semantically meaningful readouts (e.g. outcome status, resource state, dependency changes, rollback and delayed effects).
- The known-answer test is deterministic and small. It does not supply cluster-level confidence intervals or population-level statistical inference.
- No CfC/LTC transition path is involved. The tested model is actual HDC additive-delta prediction.

## Next experimental gates

1. Add a multi-query × multi-horizon contract including action failure, partial execution, rollback and delayed consequences.
2. Hold out entire environment/mechanism families and test whether observable context is sufficient to retrieve the right response map; unknown or conflicting matches must remain `UNKNOWN`.
3. Compare HDC-only, parameter-only, context-only, combined HDC, symbolic/SCM, full-state, low-rank and recurrent baselines at identical histories and probe budgets.
4. Measure pair-level false merges/splits, changed-response exactness, response regret, calibration/abstention, counterexample-probe cost and worst-family regressions separately.
5. Add candidate-side refinement with CMAQ-021 receipts and budget accounting; compare cheapest-first, uniform, random, information-gain and blind query acquisition.
6. Run the same controller with and without the world model, then against a direct policy/VLA under matched environment, action, compute and hardware budgets. Measure executed task utility and recovery rather than visual or open-loop prediction quality alone.
7. Use independent environment/mechanism clusters as inferential units under #8578. Repeated transitions are not independent replicates.

The promotion target remains a cost–robustness–utility frontier, not a single aggregate leaderboard score.

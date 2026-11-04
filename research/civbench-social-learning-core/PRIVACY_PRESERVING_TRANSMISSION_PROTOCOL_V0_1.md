# Privacy-preserving adaptive transmission protocol v0.1

**Status:** executable synthetic comparison of two recipient-state disclosure modes. It is not a deployed privacy mechanism, formal privacy guarantee, or consent/authorization enforcement system.

**Code:** `src/adaptive_transmission.rs`, `src/bin/privacy_disclosure_sweep.rs`.

## 1. Research question

Can recipient-aware transmission retain most of the cumulative-learning benefit of centralized state-aware routing while revealing materially less about each recipient's private knowledge state?

This is a different question from whether a global recommender improves mean performance. The 2025 algorithmic-mediation literature finds that global recommendation can complement peer learning, particularly in sparse networks, but a global service may need more information than a recipient or federation should disclose. The CivBench extension therefore evaluates the performance/disclosure frontier rather than assuming central visibility is the right deployment architecture.

## 2. Executable disclosure modes

The adaptive-transmission engine now records one of two explicit modes in every `Run`:

### `CentralFullInventory`

Research upper bound. The scheduler can inspect recipient-known recipes and element inventory, calculate prerequisite matches directly, and use collective novelty if the selected program requests it.

This is useful as an upper-bound comparator. It is not the default architecture proposed for Symthaea/Mycelix deployment.

### `CoarseRecipeReadiness`

A recipient-side feature extractor emits at most **two prioritized `CoarseNeedSignal` records per recipient per step**, selected from recipes the recipient does not already hold. Candidates are prioritized by prerequisite readiness, then downstream unlock count, then stable recipe ID:

- recipe identifier (the requested/needed rule);
- prerequisite readiness in thousandths (0–1000).

The cap matters: publishing every missing recipe would reveal the complement of the recipient's held-recipe set and could reconstruct its entire recipe inventory. A two-item budget limits that direct complement attack but does not prevent inference from repeated observations.

It does **not** expose the recipient's element IDs or full recipe/element inventory to the scheduler. The scheduler ranks using only this scalar, the sender's own held recipes, public task structure, graph neighbours and the declared program weights. A duplicate is filtered at the recipient-side acquisition boundary.

Because the mode does not receive a collective inventory signal, it rejects programs that request a collective-novelty filter or nonzero collective-novelty weight. A feature is not silently replaced with an invented value.

**Important limitation:** the signal still reveals which recipe is requested and an approximation of how ready the recipient is to execute it. This is data minimization, not differential privacy, anonymity, secure multiparty computation, encryption, or protection against inference from repeated observations. The protocol makes that residual leakage visible rather than calling the mode "private" without qualification.

## 3. Current experiment

The `privacy-disclosure-sweep` binary compares three conditions:

1. centralized full-inventory state-aware program;
2. coarse recipe-readiness need signal with the same program weights and budgets;
3. no-social-transmission control.

The sweep uses three synthetic DAG task families (Layered, Branching, Convergent), 256 fixed seeds per family, 12 agents, 12 steps, a 12-message per-step budget and an attention capacity of two messages per recipient per step. Graphs and identity-keyed discovery/acquisition draws are shared across conditions for the same seed; message acquisition is keyed by sender/receiver/route-slot and excludes payload identity.

Outputs report separately:
- final collective result coverage and a conservative 95% Student-t interval;
- maximum discovered depth;
- messages and attention used;
- **disclosure field units per run** (a transparent field-count proxy: centralized mode counts known element/recipe entries visible to the scheduler; coarse mode counts three fields per released need signal);
- successful acquisitions and redundant messages;
- independent discoveries;
- paired coverage deltas against centralized routing.

The field-count proxy is not mutual information, attack advantage, or an information-theoretic privacy score. Repeated signals may reveal more than the per-run count suggests; inference attacks are the next-stage acceptance gate.

The same program score weights are used for both state-aware modes, with collective novelty disabled in both. The experiment does not assert that the coarse mode is privacy-safe; it measures the utility cost of a particular minimized signal.

## 4. Required next-stage leakage evaluation

A minimized feature should be tested against an explicit attacker, not judged by its field count. Add controlled attackers that attempt to infer:

- whether a target recipe is already held or missing;
- the recipient's known-element set;
- capability level or task progress;
- agent identity/lineage from repeated need-signal patterns;
- private state changes from the timing/frequency of requests.

Report attack performance against a prior-only baseline, signal disclosure rate, mutual information or another clearly specified leakage statistic, and the utility/coverage trade-off. A low-dimensional signal may still reveal a great deal when repeated over time.

Candidate next modes, not implemented in this version:
- recipient pull: recipient explicitly requests a capability, without publishing a wider inventory;
- bucketed/thresholded readiness: reveal a small readiness class rather than a numeric score;
- noisy readiness: calibrated noise with measured utility/leakage; differential privacy only when a valid neighbouring-dataset definition, accountant and composition analysis exist;
- federated local match: matching remains local and only a consented decision receipt crosses the boundary;
- secure aggregation/MPC: only after threat model, participants and failure semantics are defined.

No epsilon-DP, cryptographic confidentiality or secure aggregation claim is made by this v0.1 implementation.

## 5. Causal and fairness controls

- Keep task, graph, budget, seeds and discovery/acquisition random units matched.
- Treat central full inventory as an upper bound, not the operational default.
- Keep no-social and fixed/random routing controls.
- Preserve sender/recipient route and message-budget constraints.
- Report performance by task family and population position, not only global means.
- Measure whether coarse signals privilege agents with already high readiness or create exclusion loops for agents with few prerequisites.
- Add adversarial signals: stale, misleading, overbroad, contradictory and strategically manipulated need reports.
- Consent withdrawal, scope expiry and payload rights remain outside this simulator; the event interchange describes them but the current benchmark does not enforce them.

## 6. Proposed research targets

These are proposed gates, not achieved results:

1. **Utility:** coarse-signal routing should approach centralized performance on held-out tasks under matched message/attention budgets; quantify the paired loss rather than preselecting a target percentage.
2. **Leakage:** attacker reconstruction should be materially better than chance only to the extent explicitly permitted by the disclosed signal. Report the residual inference, do not label it zero.
3. **Autonomy:** recipients can refuse or withdraw the signal without being silently downgraded to an unobserved/failed state.
4. **Correctness:** only a signal with the declared schema, task scope and expiry is accepted; invalid scope and stale signals fail closed.
5. **Plurality:** a recommender must not collapse all recipients onto one source lineage merely because it optimizes average performance.
6. **Transfer:** utility/leakage results must replicate on unseen DAG families and larger populations before integration into a cross-project runtime.

## 7. Cross-project responsibilities

- **Symthaea:** local need-signal generation, cognitive task state, bounded transmission program and learner outcomes.
- **Mycelix:** consent/mandate references, identity and revocation, rights/provenance, dispute and audit lineage.
- **Symtropy:** network/population scenarios, resource constraints, causal intervention and sensitivity experiments.
- **Sol-Atlas:** spatial/temporal visualization of signal disclosure, routes, coverage and risk; visualization is not privacy verification.

The shared adaptive-transmission event schema remains the payload envelope. This protocol does not add a second event ledger or claim runtime authorization.

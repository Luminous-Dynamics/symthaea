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

The sweep uses three synthetic DAG task families (Layered, Branching, Convergent), 256 fixed seeds per family, 12 agents, 12 steps, a configured ceiling of 12 messages per step and an attention capacity ceiling of two messages per recipient per step. Graphs and identity-keyed discovery/acquisition draws are shared across conditions for the same seed; message acquisition is keyed by sender/receiver/route-slot and excludes payload identity. **These are matched configured ceilings, not matched realized communication volumes:** coarse mode's two-need-signal-per-recipient limit can cause fewer messages and attention uses than the central mode. This is a known confound in the preliminary sweep, not a controlled privacy-only treatment.

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


## 3.1 Preliminary exact-head result (synthetic; not a privacy guarantee)

The run at [Symthaea PR #9970 exact head 0d0966a](https://github.com/Luminous-Dynamics/symthaea/actions/runs/41868358797) evaluated 256 common seeds per task family. Means and the reported 95% Student-t half-widths were:

| Task family | Central coverage | Coarse coverage | No-social coverage | Coarse − central (paired, 95% half-width) | Central messages/run | Coarse messages/run |
|---|---:|---:|---:|---:|---:|---:|
| Layered | 65.96% | 59.46% | 38.73% | −6.50 pp ±1.03 pp | 111.63 | 91.80 |
| Branching | 69.25% | 60.60% | 46.29% | −8.65 pp ±1.15 pp | 116.50 | 91.80 |
| Convergent | 63.00% | 59.96% | 34.12% | −3.04 pp ±0.75 pp | 104.66 | 98.02 |

The coarse signal reduced the declared disclosure-field-count proxy by 58.2–59.8% (not bits, mutual information or attack advantage). It also reduced realized messages by 6.3–21.2%. Consequently the coverage loss cannot be attributed solely to disclosure granularity. Both treatments shared the same per-step message ceiling and per-recipient attention ceiling, but not the same realized message count. These results demonstrate a preliminary utility/disclosure-volume trade-off only; they do not demonstrate privacy, anonymity, confidentiality, or differential privacy.

The next comparison must either (a) hold the realized sender/receiver/message opportunities constant while varying the disclosed fields, or (b) run a predeclared Pareto sweep over signal quota, message quota and attention budget with cost-normalized outcomes. It must preserve the existing paired seeds and report message/attention use alongside coverage. Leakage attacks from issue #10413 remain unimplemented.


## 3.2 Recipient-need quota sweep

The follow-up API `simulate_with_program_and_disclosure_budget` exposes an explicit positive `max_signals_per_recipient` parameter for coarse-readiness mode. The compatibility API `simulate_with_program_and_disclosure` remains unchanged and continues to use the v0.1 default quota of two signals per recipient per step.

The companion binary `privacy-disclosure-quota-sweep` compares signal quotas **1, 2, 4 and 8** on the same three task families and 256 seed IDs. Each family includes a central full-inventory control and a no-social control. It reports coverage, confidence-interval half-widths, depth, realized messages, attention uses, disclosure-field units, acquisitions, redundancies, individual discoveries and paired differences from central routing.

This is a cost-frontier experiment, not a matched-realized-message experiment. Increasing the signal quota may alter both available content and the number/timing of transmissions. The resulting points should be plotted in at least three dimensions (coverage, realized communication/attention cost, disclosure-field volume). Interpolation between points is descriptive and must not be presented as a causal privacy estimate. The task/seed pairing does not imply exact action-level random-number matching when policies create different route/action sets.

The 1/2/4/8 quotas are synthetic design points, not recommended deployment settings. The experiment remains non-private in the formal sense until the explicit inference attacks and authorization/consent tests in issue #10413 pass.


## 3.3 Exact-head quota sweep results (4 November 2026)

The 1/2/4/8 quota sweep completed successfully on [PR #10441 exact head](https://github.com/Luminous-Dynamics/symthaea/actions/runs/41871549672): 256 seeds per task family, with central and no-social controls. The raw CSV is archived as `civbench-privacy-disclosure-quota-sweep`.

### Collective coverage by quota

| Task family | Central | k=1 | k=2 | k=4 | k=8 | No social |
|---|---:|---:|---:|---:|---:|---:|
| Layered | 65.96% | 53.54% | 59.46% | 65.79% | 65.96% | 38.73% |
| Branching | 69.25% | 52.46% | 60.60% | 68.72% | 69.25% | 46.29% |
| Convergent | 63.00% | 52.71% | 59.96% | 63.00% | 63.00% | 34.12% |

### k=4 compared with central full inventory

| Family | Paired coverage delta (95% interval) | Messages/run central → k=4 | Disclosure-field proxy reduction |
|---|---:|---:|---:|
| Layered | −0.17 pp [−0.39, +0.05] | 111.63 → 111.23 (−0.35%) | 17.59% |
| Branching | −0.53 pp [−1.00, −0.06] | 116.50 → 115.59 (−0.78%) | 19.62% |
| Convergent | 0.00 pp [0.00, 0.00] | 104.66 → 104.66 (0.00%) | 16.51% |

At the tested settings, k=4 is a promising engineering trade-off: realized communication is almost unchanged, while the field-count proxy decreases by roughly one-sixth to one-fifth. In the branching case, the paired coverage loss is small but its reported 95% interval excludes zero; this is not a formal non-inferiority result. The intervals are descriptive paired t intervals, not a pre-registered equivalence test.

k=1 reduces the field-count proxy by about 79% but loses 10.3–16.8 percentage points of coverage and substantially reduces message volume. k=2 reduces field counts by about 58–60% but loses 3.0–8.6 percentage points of coverage. k=8 reproduces central coverage and message volume in these fixtures but increases the field-count proxy by about 50–60% relative to central. None of these points proves privacy; the signal contents can still support reconstruction and inference attacks.

The sweep therefore narrows the design question: test k=4 against explicit reconstruction/membership/progress/linkage attackers, add consent and expiry semantics, and replicate on held-out task families before treating it as a candidate operating point. Do not select k=4 on these same synthetic families for a deployment claim.

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

# AOBench — Causal Identification Robustness Across Observational Views

**Status:** Research specification; no implementation or benchmark score claimed  
**Date:** 2026-11-14  
**Tracking:** [AOBENCH-015 #12210](https://github.com/Luminous-Dynamics/symthaea/issues/12210)  
**Research basis:** [Jung et al., CausalIDView (arXiv:2609.36881)](https://arxiv.org/abs/2609.36881)  
**Upstream reference implementation:** https://github.com/MLAI-Yonsei/CausalIDView

## 1. Research result

CausalIDView asks what happens when the *same causal world and target estimand* are presented through different observational views.

The released point-identification scaffold derives four views from one deterministic SCM realization, preserving the same units, treatment assignment, potential outcomes, context/query split and target CATE:

| View | Extra information exposed | Identification route |
|---|---|---|
| Back-door (BD) | adjustment variable U | back-door adjustment |
| Front-door (FD) | mediator M | front-door functional |
| Instrumental variable (IV) | instrument I | Wald / IV identification |
| Proximal (PROX) | proxies Z_P and W_P | proximal identification |

The paper also studies a **Hidden Confounding (HC)** stress view that exposes only (X,T,Y), with U withheld. It scores point predictions against the target effect as a stress test, not as a point-identification regime. Separately, its partial-identification companion uses three matched views: Manski bounds (X,T,Y), monotone-IV bounds (X,I,T,Y), and sensitivity bounds (X,T,Y). These share the same realized observed data, context/query split, query units and target CATE within that companion SCM; the paper uses binary potential outcomes and 40 additional worlds. It reports endpoint errors for the Manski, IV and sensitivity identified sets, with TabPFN-v3.5-based bound estimators obtaining the lowest endpoint RMSE in those evaluated settings.

The paper's construction audit is instructive: it checks 200 view tables across 40 worlds and 400 within-world view pairs, then validates 44,960 unit-level potential-outcome identities. It reports zero shared-field mismatches, zero master-projection mismatches, zero query-order mismatches, a maximum |Y(1)-Y(0)-tau(X)| discrepancy of 1.55e-15, and zero discrepancy for the saved Stay target and the 2x Move target. These are the paper's construction results, not AOBench results; AOBench uses them to define what our own fixture audit must verify.

The public point-estimation configuration uses 40 deterministic paired worlds, 1,024 context rows, 100 query rows and 50 baseline covariates. Its SCM cycles through five CATE families: linear, quadratic interaction, threshold piecewise, Fourier/nonstationary and shallow MLP. Each estimator receives only the variables allowed by its view; query factual treatment/outcome and potential outcomes / true effects are withheld. Prediction artifacts are aligned by exact query unit_id. The headline CATE metric is decomposed into ATE-error squared plus centered sPEHE squared, with bootstrap uncertainty computed at the SCM-world level (10,000 replicates in the released configuration). The paper additionally evaluates semi-synthetic ACIC/LaLonde-style data and real-world null-effect tests using pre-treatment outcomes from randomized trials.

The paper reports that no tested causal foundation model consistently wins across all observational regimes; rankings change with the available identifying information. A modular predictive model plus regime-specific identification procedures is competitive and outperforms several CFMs on some views. Its structural response experiments also distinguish *Stay* (the underlying target effect is preserved) from *Move* (the target effect genuinely changes).

This creates a direct benchmark lesson: performance on one observed column set does not establish causal competence across identification regimes.

## 2. Paradigm shift

Treat the observation view as part of the causal task contract, not as incidental dataset formatting.

A single causal world can support point identification through observed adjustment variables, a mediator, an instrument or proximal proxies; only partial identification under hidden confounding or missing proxies; or no defensible identification under insufficient information.

A capable system should change *what it claims*, not just its numerical estimate. When a view loses identifying information, the system should widen its effect bounds or abstain. When the true estimand remains unchanged but visible variables change, it should remain calibrated and not invent a structural change. When the underlying mechanism genuinely changes the target effect, it should respond rather than preserve a stale answer.

This connects CausalIDView's matched-view design to AOBENCH-010 semantic continuity, AOBENCH-011 sequential probe design, AOBENCH-012 model adequacy and AOBENCH-013 partial identification.

## 3. Evaluation protocol

### 3.1 Paired worlds and observation contracts

1. Generate one SCM world and preserve its full evaluator-only potential outcomes.
2. Derive every observational view from that same realization and unit set.
3. Freeze treatment, outcome, target population, estimand, query IDs, context/query split and structural parameters.
4. Change only declared observed auxiliary variables or the registered measurement/availability mechanism.
5. Give each estimator a machine-checkable allowlist for context and query fields.
6. Keep potential outcomes, query truth, hidden variables and oracle target class outside the agent-facing input.
7. Require exact query-unit ID order. Reject mismatched prediction IDs rather than silently sorting, joining or imputing.
8. Store world seed, SCM version, view manifest, observed-column set, target-estimand digest and evaluator revision with every result.

The upstream CausalIDView artifact provides a concrete starting protocol and public code. AOBench should reproduce its matched-world invariants before introducing new regimes.

### 3.2 Identification regime and assumption ledger

For each view, the agent must return the claimed identification regime, target estimand and target population, assumptions required for point identification, which assumptions are supported by the declared view contract, which assumptions are untestable or only stress-tested, point estimate / identified interval / abstention, uncertainty statement, and provenance for every observed field and evidence receipt.

The evaluator must not accept a point estimate merely because the estimator returns one. The claim must match the frozen graph/view assumptions.

### 3.3 Stay / Move response tests

- **Stay:** change non-target structural parameters, nuisance mechanisms, proxy noise or data distribution while preserving the evaluator-only target estimand. A correct estimator should not report a spurious causal-effect shift beyond a preregistered tolerance.
- **Move:** alter the causal mechanism so the target estimand truly changes while retaining a matched observation protocol. A correct estimator should track the change rather than over-stabilize.
- **View-only shift:** keep the entire SCM and estimand fixed, but remove or replace auxiliary columns. The point/interval claim must update according to identifiability; the true target remains the same.
- **World-plus-view shift:** change both the world and observed variables, requiring the system to separate effect change from information loss.

Use paired seeds and common random numbers where valid, so a score change is attributable to the intended intervention rather than a new random world.

## 4. Task ladder

**V1 — Information contract and leakage.** Check allowed fields, no evaluator-only outcomes, exact unit alignment. Hard gate, not compensable.

**V2 — Regime recognition.** Identify whether the view supports back-door, front-door, IV or proximal identification under the frozen assumptions. Include invalid-IV exclusion, invalid mediator assumptions, weak/invalid proxies and hidden-confounding controls.

**V3 — Matched-view invariance.** Estimate the same target effect across multiple views of the same world where point identification is valid. Measure view-induced estimate drift and calibration rather than comparing unpaired datasets.

**V4 — Identification-aware uncertainty.** Provide a point estimate only when the target is point identified under the frozen assumptions. Under hidden confounding, missing proxies or incomplete measurement, provide a valid partial-identification interval or abstain. A narrow but invalid interval is a hard failure.

**V5 — Stay / Move structural response.** Preserve the target estimate under target-invariant changes; track the effect under genuine mechanism changes. Distinguish distribution shift from causal-effect shift.

**V6 — Negative controls and assumption failures.** Include null-effect worlds, proxy corruption, measurement error, selection bias, weak/invalid instruments, mediator-outcome confounding, proximal completeness violations and missing-not-at-random views. Score false-positive decisive actions independently.

**V7 — Decision transfer.** Given a decision threshold and loss, act only when all effects compatible with the view support the same action. Otherwise defer or choose a preregistered robust/minimax action. Connect this to the max-potency evaluator rather than requiring full graph recovery.

## 5. Baselines

- CausalPFN, Do-PFN and CausalFM under identical view inputs and matched SCM worlds.
- Classical back-door adjustment, front-door plug-in, IV/Wald and proximal estimators where assumptions apply.
- Modular predictive tabular model plus regime-specific identification, matching the CausalIDView TabPFN-v3.5 plus X/S/DR learner, front-door, Wald and proximal P-Learner families.
- Classical Manski worst-case bounds, monotone-IV bounds and sensitivity-model bounds; compare endpoint error and validity separately.
- TabPFN-v3.5-based bound estimators, B-Learner and NeuralCSA sensitivity baselines where the corresponding model applies.
- Exact finite-world oracle regime classifier and bound estimator on enumerable controls.
- Always-point-estimate, always-wide-interval and always-abstain negative baselines.
- AOBench estimator with explicit assumption/evidence ledger, model-adequacy check and partial-identification fallback.

For each baseline record checkpoint and code SHAs, data/view manifest, estimator configuration, inference budget, world count, query count, pretraining/exposure provenance and uncertainty method. A missing or gated checkpoint stays missing; it is never silently replaced with a surrogate.

## 6. Metrics

Keep point-estimation, partial-identification and decision claims separate.

**Point-identified views:** CATE sPEHE²; ATE-error² and centered-sPEHE² decomposition; cross-view estimate drift for the same true target; Stay stability and Move responsiveness; uncertainty calibration and interval coverage if intervals are emitted.

**Partially identified views:** lower/upper endpoint error; finite-world bound coverage; interval width and informativeness; false point-identification rate; false narrowing and overconfidence rate.

**Regime and evidence:** identification-regime classification; assumption-ledger precision/recall; leakage and column-allowlist violations; query-ID alignment; evidence provenance completeness; correct Inconclusive/abstain rate.

**Decision:** robust-action correctness over all admissible worlds; false decisive-action rate on null/effect-straddling controls; correct defer rate and minimax regret.

**Resources:** runtime, memory, inference cost, context/query size and model/checkpoint size.

No weighted composite may allow data leakage, invalid identification or an interval excluding a compatible world to be offset by lower CATE error.

## 7. Proposed qualification targets (not achieved)

- 100% column-allowlist and exact query-ID integrity on deterministic fixtures.
- Zero accepted use of evaluator-only potential outcomes, true effects or hidden confounders.
- Zero false point-identification claims in registered nonidentified views.
- At least 95% correct regime/claim disposition on sealed deterministic tasks after pilot calibration.
- At most 5% normalized estimate drift across equivalent point-identified views when the true target is fixed, with world-cluster uncertainty reported.
- At least 95% correct Stay/Move responses on sealed deterministic controls.
- 100% finite-world bound coverage on exact partial-identification fixtures.
- Zero false-positive decisive actions on null-effect or effect-straddling controls.

These are pilot/preregistration targets, not achieved Symthaea results and not claims about CausalIDView's reported scores. Stochastic coverage targets require a separately preregistered sampling and multiplicity protocol.

## 8. How AOBench can go beyond CausalIDView

CausalIDView provides the matched-world identification-view core. AOBench should add capabilities the current scaffold does not claim to evaluate:

1. **Evidence-qualified view changes:** every changed/missing data source carries provenance and observation-integrity receipts; observer failure is not mistaken for a causal-view change.
2. **Open-world model adequacy:** if a qualified observation is outside every registered mechanism, detect model-class inadequacy rather than forcing a regime label.
3. **Partial identification as a first-class output:** evaluate bound validity and robust action when point identification is impossible.
4. **Sequential experiment acquisition:** let an agent choose which additional variable, proxy or intervention to acquire, with cost/risk constraints and stop rules.
5. **Adversarial views:** include measurement corruption, selection, proxy failure, common-cause observation outage and missing-not-at-random channels.
6. **Governance continuity:** ensure a view change does not quietly change authority, consent, data rights or permitted use.
7. **Repeated-epoch robustness:** preserve calibrated identification as data sources, sensors and external substrate evolve.

This is a defensible path to a stronger benchmark: not a claim to beat CausalIDView's current leaderboard today, but a larger and more decision-relevant evaluation contract.

## 9. Implementation path

1. Reproduce the upstream four-view column allowlists and exact ID alignment in a minimal Rust scenario format.
2. Add a matched-world fixture with evaluator-only potential outcomes and leakage negative controls.
3. Add Stay / Move / View-only / World-plus-view paired perturbation manifests.
4. Add regime-specific identification contracts and explicit point-versus-bound claim type.
5. Connect hidden-confounding views to partial_identification.rs and exact finite-world bounds.
6. Add evidence receipts and semantic continuity checks for view/source changes.
7. Add matched CFM, classical and modular baselines with frozen checkpoint/config SHAs.
8. Use world-level clustered uncertainty and family-disjoint sealed SCMs before any SOTA claim.

## 10. First executable increment

The nested Rust package now contains two complementary modules, both exported from its public crate root:
- `src/observational_views.rs`: matched-view **data-contract validator**, not a causal-identification engine.
- `src/world_projection_audit.rs`: evaluator-only **master-world construction audit**, kept separate from every agent-facing payload.

The default policy *requires* the four point-identification regimes:
- BD context: X, U, T, Y; query: X, U.
- FD context: X, M, T, Y; query: X, M.
- IV context: X, I, T, Y; query: X, I.
- PROX context: X, Z_P, W_P, T, Y; query: X, Z_P, W_P.

The validator also supports optional regime policies matching the paper's additional tasks:
- HC context: X, T, Y; query: X.
- Manski bounds context: X, T, Y; query: X.
- Monotone-IV bounds context: X, I, T, Y; query: X, I.
- Sensitivity bounds context: X, T, Y; query: X.

These four additional regimes are available in the API but are not required by the default policy. An evaluation manifest must explicitly request them and freeze their respective identifying/bounding assumptions. The point-estimation HC stress view must not be confused with the partial-identification companion's distinct bound estimands.

It checks:
- all required regimes exist exactly once;
- all regimes share the same SCM and estimand digests;
- context and query unit IDs are unique, disjoint and identically ordered across views;
- shared X/T/Y context values and shared X query values remain exactly identical across matched views;
- feature-vector dimensions remain consistent across regimes and splits;
- column names match the frozen per-regime allowlist;
- query factual treatment/outcome and evaluator-only columns (tau, Y0/Y1, M0/M1, T0/T1, true effect, oracle fields) never enter the estimator view;
- each column has the correct row count, consistent vector dimensions and finite values.

Fifteen deterministic unit tests cover a valid four-view set, target-effect leakage, query T/Y leakage, changed query ordering, SCM/estimand digest mismatch, missing regime, duplicate regime, malformed/non-finite columns, context/query overlap, unexpected columns, shared context-value drift, shared query-value drift, cross-view feature-dimension mismatch, the optional HC view, and the three partial-identification companion allowlists.

This validator proves only that the supplied serialized view matches its declared data contract. It does not authenticate the SCM digest, prove the observation source is truthful, or establish the back-door/front-door/IV/proximal assumptions. Those require separate oracle/evidence contracts. No Rust compiler, test or Clippy pass is claimed until exact-head qualification completes.

The companion world-projection audit compares every estimator-visible column with a separately held master-world projection, checks exact context/query IDs and shared field values, verifies Y(1)-Y(0)=tau(X) within a frozen tolerance, and optionally validates Stay/Move target arrays (Stay preserves tau; Move scales tau by the registered factor). Its oracle/master projection is deliberately not part of the serialized agent input. Eight deterministic tests cover exact construction, view-column corruption, query-column corruption, potential-outcome identity, Stay mismatch, Move mismatch, master-query alignment and optional Stay/Move controls.

Together these modules provide 23 deterministic unit tests for input contracts and evaluator-side construction integrity. They still do not prove causal assumptions, authenticate real-world data sources or qualify effect estimators. No Rust compiler, test or Clippy pass is claimed until exact-head qualification completes.

## 11. Claim boundary

Cross-view robustness on synthetic SCMs does not prove real-world identification assumptions. The regime, graph, estimand, target population and data-view contracts must be frozen and independently reviewed. Real-world null controls and semi-synthetic datasets add evidence, but do not turn benchmark performance into universal causal truth.

## 12. References

- [Jung et al. (2026), CausalIDView paper](https://arxiv.org/abs/2609.36881)
- [CausalIDView reproducibility scaffold](https://github.com/MLAI-Yonsei/CausalIDView)
- [ACED-Bench](https://arxiv.org/abs/2610.14381)
- [Causal-Discovery-Bench](https://arxiv.org/abs/2611.09934)
- [AOBENCH-011 sequential experiment design #12173](https://github.com/Luminous-Dynamics/symthaea/issues/12173)
- [AOBENCH-012 model-class adequacy #12200](https://github.com/Luminous-Dynamics/symthaea/issues/12200)
- [AOBENCH-013 partial identification #12206](https://github.com/Luminous-Dynamics/symthaea/issues/12206)

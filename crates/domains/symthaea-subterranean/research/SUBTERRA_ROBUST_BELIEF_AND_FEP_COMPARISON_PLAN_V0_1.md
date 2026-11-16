# Subterra robust belief, model uncertainty and FEP comparison plan

**Date:** 2026-11-16  
**Status:** Proposed next research tranche; no implementation or result claimed  
**Purpose:** prevent a planner from becoming confidently wrong when its supplied sensor likelihoods or geological hypotheses are misspecified.

## 1. Problem exposed by the first two kernels

The single-step EVSI and finite-horizon survival reference planners accept:
- one normalized prior over a finite hypothesis set;
- one conditional outcome-likelihood table per action;
- fixed decision and terminal losses;
- fixed risk limits and costs.

This makes the mathematical contract compact and independently testable. It also creates a serious limitation: an internally consistent likelihood matrix can still be wrong for a new cave, sensor calibration, geological origin or fault mode.

A planner can compute EVSI correctly under an inaccurate observation model and still select a poor measurement. Likewise, a risk constraint computed under one optimistic hazard model is not a robust risk guarantee.

The next research increment should therefore qualify **uncertainty over the models used to value evidence**, not only uncertainty about the site.

## 2. Keep four uncertainties separate

### A. State / geology uncertainty
What geological or environmental state exists?
- cavity / no cavity / unknown;
- volcanic, collapse, karst, tectonic, ice-related, engineered or unknown origin;
- connectivity and access hypotheses;
- structural and environmental state.

### B. Measurement uncertainty
How noisy, biased, censored or incomplete is an actual sensor result?
- per-sensor calibration;
- detection limit;
- spatial resolution and footprint;
- pose/time uncertainty;
- false positives and false negatives.

### C. Model uncertainty
How uncertain are the likelihood, hazard, dynamics and correlation models themselves?
- calibration transfer to unseen cave;
- geology-class shift;
- common-mode failure;
- sensor ageing or fouling;
- likelihood parameter uncertainty;
- disagreement between independent models.

### D. Decision / preference uncertainty
How uncertain are the mission loss matrix, science priorities, resource shadow prices or terminal consequences?
These should be declared and governed by the responsible mission/study authority, not silently inferred from a reward model.

A scalar confidence score must not collapse these four categories.

## 3. Proposed robust model family

Represent plausible observation models as a frozen, content-addressed finite model set for the first reference implementation:

`M = {m_1, m_2, …, m_k}`

Each model binds:
- a model ID/version and source-code digest;
- calibration dataset manifest and held-out-site scope;
- outcome likelihoods for every candidate measurement and hypothesis;
- sensor lineage / common-cause assumptions;
- declared valid operating envelope;
- calibration and proper-scoring-rule report;
- model-discrepancy / OOD status.

Each model is evaluated separately on the same frozen mission scenarios. Never average incompatible model semantics without an explicit model-selection rule.

### Candidate robust criteria

1. **Nominal expected value:** average under the preregistered model weights.
2. **Worst-case EVSI:** minimum decision-value improvement across all models in the declared ambiguity set.
3. **Worst-case risk:** maximum mission-interruption and robot-loss probability across the model set.
4. **Distributionally robust value:** optimize against a preregistered uncertainty set over model weights, rather than one convenient point estimate.
5. **Regret minimization:** minimize worst-case regret relative to each model's own oracle decision.
6. **Model disagreement / abstention:** decline to rank actions when candidate ordering is unstable across plausible models or OOD evidence invalidates the model family.

No behavioural probability weighting, learned reward, consciousness metric or information gain may change a hard externally reviewed safety, retreat, preservation or authorization gate.

## 4. Common-cause and evidence-lineage modelling

Repeated sensor observations are not automatically independent. The benchmark must include latent fault modes such as:
- shared camera/radar acquisition;
- common calibration error;
- repeated processing of the same raw data;
- shared pose estimate or coordinate-frame error;
- temperature-dependent drift;
- environmental attenuation affecting multiple sensors;
- a common timestamp or relay source.

The simplest admissible model should place explicit fault-mode hypotheses inside the belief universe and bind each derived observation to its physical acquisition lineage. A later richer model may use a factor graph or hierarchical Bayesian model, but must still retain raw/derived ancestry.

The independent witness count is determined by physical sensor/acquisition lineage, not by the number of files, processors, model outputs or derived maps.

## 5. Robust survival-conditioned planning

The finite-horizon planner should later consume a set of plausible models `M`. For every candidate contingent policy `π`, compute:
- expected decision loss under each model;
- interruption and robot-loss probability under each model;
- expected energy/time/data/exposure under each model;
- observations and evidence-return receipts under each model.

A policy is admissible only when every declared hard risk envelope passes, or when the reviewed requirement explicitly defines a different robust probability criterion. Do not average one unsafe model away with many optimistic models.

Initial robust objective options should be compared separately:
- nominal weighted average;
- max-loss / minimax;
- minimax regret;
- CVaR over a frozen scenario distribution;
- distributionally robust optimization over a declared model-weight ambiguity set.

These are offline research objectives. The selected policy still passes the existing mission executive and safety chain.

## 6. Symthaea FEP comparison: do not conflate objectives

Symthaea already contains general Active Inference / FEP infrastructure, including expected-free-energy trajectory rollouts, epistemic information value, pragmatic value and trajectory telemetry. The existing subterranean FEP wrapper currently uses the perception/belief-update path and does not wire its action-selection methods to the subterranean motor controller.

Subterra should compare three separate decision systems:

### A. Shannon information gain
Maximize expected entropy reduction. This values uncertainty reduction independent of the mission's terminal decision loss.

### B. EVSI
Minimize expected terminal decision loss after the measurement, minus the declared resource price. This values observations only insofar as they can improve a declared decision.

### C. FEP expected free energy
Use the existing generative model and policy rollouts to combine the FEP implementation's epistemic and pragmatic terms. Report exact objective terms and assumptions. Do not rename FEP free energy as EVSI or assume their numerical scales are comparable.

### Required controlled comparison
Hold constant:
- initial beliefs / priors;
- candidate measurement/action set;
- outcome likelihood model;
- decision-loss and resource model;
- hard mission risk/ODD/preservation constraints;
- compute/time budget;
- stopping condition.

Report:
- selected measurement and policy;
- expected and realised terminal regret;
- expected entropy reduction;
- expected free-energy terms;
- calibration and OOD status;
- risk and survival outcomes;
- resource and compute consumption;
- disagreements between policy rankings.

Run ablations of epistemic-only, pragmatic-only and combined FEP valuation. An FEP policy is not promoted just because it produces plausible actions; it must outperform the strongest matched baseline on preregistered held-out sites without breaking any safety/evidence invariant.

## 7. Robustness benchmarks

| Benchmark | Proposed test |
|---|---|
| LikelihoodShiftBench | Perturb outcome likelihoods within frozen calibration intervals and assess action-ranking stability |
| CommonCauseEvidenceBench | Duplicate derived observations from one physical sensor; verify independence is not inflated |
| ModelEnsembleRiskBench | Require interruption/robot-loss ceilings to hold across every declared plausible model |
| ModelDisagreementAbstentionBench | Force abstention when plausible models imply incompatible best actions and no independent discriminating evidence is available |
| OpenWorldTransferBench | Hold out one complete geology class/site; ensure unknown/OOD handling does not silently promote a known-origin claim |
| RobustDecisionRegretBench | Compare nominal EVSI, worst-case EVSI, minimax regret and exact small-case oracle |
| FEPObjectiveSeparationBench | Check that entropy reduction, EVSI and expected free energy remain separately reported and correctly ablated |
| CalibrationDecayBench | Inject sensor ageing, bias and calibration drift; stale calibration cannot improve claim strength |
| RiskModelMisspecificationBench | Use plausible optimistic/pessimistic hazard models and require the declared robust risk criterion |
| ModelUpdateGovernanceBench | New model versions cannot silently replace a model used by an earlier evidence/mission record |

Proposed invariant: **widening the declared model-uncertainty set may increase abstention or reduce permitted policy value, but must never make an unsupported claim more certain or widen a hard operating envelope.**

## 8. Recommended implementation order

1. Freeze a model-set schema and canonical digest semantics.
2. Add a small model-ensemble evaluator around the existing single-step EVSI kernel, with no change to its equations.
3. Add worst-case risk and action-ranking stability reports.
4. Add tiny exact robust-policy oracle cases before integrating robust evaluation into the survival planner.
5. Add common-cause fault hypotheses and source-lineage commitments.
6. Add a matched FEP-vs-EVSI experimental harness using the existing FEP implementation without motor-path wiring.
7. Only after the reference suite qualifies should Symtropy scenario replay and real cave datasets feed the model ensemble.

## 9. Initial implementation status and evidence ceiling

A first finite-model robust EVSI wrapper has now been authored in `subterra-robust-voi`. It evaluates the one-step EVSI reference across a finite declared set of prior / likelihood models, rejects schema drift, excludes actions denied or over-budget in any supplied model, and ranks actions by worst-case net EVSI over that finite set. It contains 10 authored Rust unit tests and depends only on the adjacent local reference crate.

This is not a general robust Bayesian filter, not proof that the true physical model is included in the supplied finite set, and not a robust survival-tree optimizer. A separate site-disjoint empirical likelihood reference is now authored in subterra-likelihood-calibration. It estimates categorical frequencies from Training sites with Dirichlet smoothing, reports Calibration/Evaluation Brier and NLL, enforces acquisition-lineage and site split isolation, and converts only fully supported cells into EVSI inputs when priors, decision losses, costs and external gates are separately supplied. This is not yet evidence of calibration under planetary shift. Hierarchical likelihood models, uncertainty-set construction, distributionally robust optimization, CVaR, factor graphs, automatic sensor-lineage dependence, source authentication and FEP action integration remain unimplemented. The five standalone Rust packages are authored but have not been compiled or executed in the current authoring environment.

No claim of universal robust risk control, SOTA superiority, robot safety, analogue performance or planetary competence is made.

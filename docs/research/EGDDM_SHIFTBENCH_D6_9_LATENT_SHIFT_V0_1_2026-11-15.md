# EGDDM D6.9 — latent shift detection under partial observability

**Version:** 0.1  
**Date:** 2026-11-15  
**Status:** Research protocol and synthetic evaluator bundle authored. No D6.9 Rust evaluator, schema-engine run, compiled tests, agent trial, blinded corpus, or external validation is claimed.

## 1. Research question

Can an engineering agent detect that its working environment or the validity of its evidence has changed **when no one tells it that a shift occurred**?

D6.8 evaluates repeated, explicit interventions. D6.9 changes the observation model: the candidate sees the evolving workspace and permitted observations, but not the evaluator's event schedule, hidden state digests, affected-obligation labels, or event time. Some changes are silent; others produce noisy or correlated symptoms; sham changes can look similar.

D6.9 is a benchmark/evaluator adapter. It composes D2 temporal replay, REA evidence acquisition / EVSI, D6.7 impact/cost truth, and D6.8 evidence-continuity scoring. It does not create a new probe planner, generic evidence graph, temporal ledger, event store, receipt format, monitoring service, or authority service.

## 2. Why this matters

An explicit shift benchmark can measure recovery *after the shift is noticed*. Production systems also need to decide whether a shift occurred at all.

Two opposite errors matter:
- **Missed shift:** an agent continues to rely on receipts whose dependencies have silently changed.
- **False alarm:** an agent treats noise or an irrelevant change as a real invalidation, needlessly discarding reusable evidence and repeatedly blocking work.

A policy that always abstains or reruns everything must not win. A policy that ignores weak symptoms must not win either. D6.9 measures detection, diagnosis, safe decision, and probe cost separately.

## 3. Candidate/evaluator separation

Each evaluator episode has:
1. **Candidate view:** baseline observations, the current observation stream, active obligations, permitted diagnostic probes and candidate-visible prior receipts.
2. **Candidate trace:** probes requested, returned observations, decisions and claims. This is produced during candidate execution and remains separate from the hidden truth.
3. **Evaluator overlay:** hidden event truth, changed state dimensions, affected obligations, source-family relationships, event time and expected dispositions.

The committed synthetic bundle is evaluator-only for reproducibility. A real corpus must materialize the candidate view separately, with its own digest and access controls. Do not mount the evaluator overlay or expected outcomes into candidate workspaces.

## 4. Observability strata

The research corpus should cover four forms of observability:

- **Visible local change:** a source/configuration diff is inspectable, but impact labels remain hidden.
- **Silent infrastructure drift:** runner image, toolchain, policy, issuer, external dependency or service changes without an explicit event notice.
- **Partial/noisy symptoms:** observations are delayed, incomplete, correlated, conflicting or only available through permitted probes.
- **Deceptive/sham signals:** a harmless change mimics symptoms of a real invalidation, or a genuine change has only weak symptoms.

The initial v0.1 fixture exercises silent runner drift and a sham load/telemetry spike.

## 5. Diagnostic actions

Each permitted diagnostic probe declares:
- authorization class and scope;
- what observation family it can return;
- expected cost and latency distribution (synthetic in the first fixture);
- reversibility/target impact;
- which proof obligations its outcomes can inform, without declaring that an outcome is sufficient by itself.

The benchmark records the candidate's selected probes, exact returned observation payloads, provenance/source family, observation time, and availability time. A decision is not scored as supported merely because it names a response ID; the response bytes and their availability must be present in the candidate trace. Existing REA/EVSI planners may choose probes; D6.9 scores whether the chosen observations were useful and admissible. Do not fork or modify their planner APIs.

## 6. Required known-answer controls

At minimum:
- real silent runner/toolchain drift affecting a critical obligation;
- sham runner telemetry change with no underlying state change;
- benign documentation edit that must not invalidate unrelated policy/authority evidence;
- policy/issuer revocation with weak or delayed symptoms;
- two real shifts close together, with partial diagnosis of the first before the second;
- correlated provider symptoms sharing one upstream source, not independent corroboration;
- contradictory observations requiring unknown/indeterminate;
- observation blackout, missing page or missing receipt;
- late evidence that is valid for history but unavailable at the decision cutoff;
- high-information but non-decisive probe versus lower-information probe that resolves the only blocking obligation;
- stable/no-shift episodes with comparable observation volume;
- a case where the agent detects possible drift but cannot identify the cause, and must keep that uncertainty explicit.

Each target case should have a condition-removed positive twin that restores the expected stable or safe-reuse result.

## 7. Metrics

Keep separate scorecards:

**Detection**
- event precision/recall by change family;
- detection delay from event occurrence and from first candidate-available signal;
- false alarms per stable work-unit/time;
- missed critical-shift rate;
- detection before stale-evidence use or consequential action.

**Diagnosis**
- impact-set precision/recall and critical recall;
- cause-family localization accuracy;
- common-source/correlation discount accuracy;
- unknown/ambiguous diagnosis accuracy;
- false causal attribution.

**Evidence and decisions**
- stale critical-evidence acceptance;
- unsupported-current claim rate;
- critical unverified obligation-seconds and maximum exposure gap;
- appropriate abstention and risk-coverage;
- revalidation selection precision/recall against D6.7 labels.

**Acquisition and cost**
- authorized probe cost, latency, compute, tokens and retries;
- unnecessary probe rate and redundant acquisition churn;
- cost to first sufficient evidence and to completed recovery;
- cost regret against D6.7 measured-cost oracle once that interface is qualified.

A low-cost false-negative, unauthorized probe, stale critical-evidence acceptance, or future-evidence leak remains a hard failure. No aggregate score may trade these failures against efficiency.

## 8. Proposed scale

Development:
- one synthetic known-answer episode per observability stratum;
- sham/no-shift, conflicting, delayed, blackout and common-source controls;
- an independent deterministic fixture oracle for event truth, observation availability, source lineage, affected obligations and expected dispositions.

Validity pilot:
- 12 intervention families × 4 observability strata = 48 cells, one independently authored seed per cell, plus condition-removed controls.

Research corpus target:
- 12 intervention families × 4 observability strata × 5 independent seeds = 240 matched stable/latent-shift episodes;
- source-family, repository-family and forward-time holdouts;
- repeated trials and exact candidate/harness/model/resource identities.

These are design targets, not collected tasks or agent results.

## 9. Baselines

- no detection / assume baseline stays current;
- periodic full revalidation;
- file-watcher / static dependency-change detection;
- predeclared threshold/change-point baselines (e.g. CUSUM or ADWIN);
- D6.8 explicit-shift policy with no latent detector;
- existing REA/EVSI evidence-acquisition policies;
- full EGDDM;
- expert maintainer / independent adjudicator reference.

Tune thresholds only on pilot data. Freeze the final policies and thresholds before held-out evaluation.

## 10. Acceptance gates

- candidate inputs contain no evaluator event schedule, hidden state digests, or gold affected-obligation labels;
- zero future-evidence leakage in deterministic timing controls;
- zero stale critical-evidence acceptance in deterministic positive controls;
- sham/no-shift controls quantify and bound false alarms and unnecessary probes;
- correlated observations do not create fake independent corroboration;
- unknown, conflicting, missing and unobservable states remain explicit;
- a useful qualified policy outperforms always-abstain on useful coverage and no-detection on critical-shift recall without increasing hard safety failures;
- D6.7 gold impact/cost interfaces are consumed only when independently qualified; unknown impact/cost remains unknown or bounded;
- D6.8 explicit-intervention fixtures remain regression controls;
- every report pins observation availability, source family, tool policy, model/harness, evaluator version and input digests.

## 11. External positioning

D6.9 complements rather than replaces:
- SWE-Shift and D6.8: known/visible interventions versus hidden or partially observed changes;
- ChainSWE and Multi-step SWE: sequential maintenance and long-horizon task completion;
- REA: claim-obligation-directed evidence acquisition and EVSI planning;
- D2: temporal replay and decision-time availability;
- D6.7: independent impact truth and measured check costs;
- MACE / benchmark-integrity work: task, grader and environment qualification.

Different benchmark scores are not directly comparable. D6.9's initial claim is only synthetic partial-observability conformance.

## 12. Claim ceiling

The v0.1 fixture demonstrates a deterministic silent-shift versus sham-signal distinction. It does not establish a universal shift detector, causal completeness, production monitoring reliability, safe autonomous operation, model superiority or SOTA advantage. Those require blinded task admission, independent event/impact adjudication, executed Rust qualification, matched agent trials, statistical preregistration and external replication.

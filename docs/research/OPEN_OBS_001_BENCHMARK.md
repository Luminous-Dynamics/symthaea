# OPEN-OBS-001 — Open-World Observational Discovery Benchmark

**Status:** research contract / proposed benchmark  
**Tracker:** [Symthaea #7835](https://github.com/Luminous-Dynamics/symthaea/issues/7835)  
**Version:** 0.1-draft  
**Evidence class:** specification only; no benchmark execution or performance claim is implied.

## 1. Objective

Build a compact, reproducible evaluation of whether a scientific inference system can discover unusual observations, survive distribution shift, distinguish signal from nuisance, propose discriminating follow-up measurements, and preserve enough provenance for independent review.

The motivating application includes searches for observations compatible with experimentally motivated extra-dimensional physical models. The benchmark is deliberately broader than that application: the same methods should detect when an unexpected result is a known artifact, an unfamiliar but ordinary process, an unresolved anomaly, or merely a weakness in the current model.

This is **not** a detector for “higher-dimensional beings.” There is no established evidence that such beings exist. The benchmark must never infer that conclusion from dimensionality in an embedding, visual resemblance, anomaly scores, model complexity, or synthetic success.

## 2. Paradigm shift: a claim-separated experimental loop

A conventional classifier asks “which known class is this?” An anomaly detector asks “how unlike training data is this?” Neither question is sufficient for open scientific discovery.

The target loop is:

```text
raw observation + acquisition/capture metadata
  -> integrity and calibration checks
  -> candidate discovery (including Unknown)
  -> multiple explicit hypotheses and predicted observables
  -> independent nuisance/context/control checks
  -> next discriminating measurement
  -> blinded evaluation and replication
  -> claim-specific evidence record
```

Keep four questions separate:

| Axis | Question | What is not sufficient |
|---|---|---|
| Anomaly | Is this observation poorly explained by the declared reference models? | Rarity or a high detector score alone |
| Novel physics | Does a named model predict held-out observations better than credible alternatives? | A visually compelling fit or an unconstrained extra parameter |
| Extra-dimensional model support | Does a specific, constrained extra-dimensional model predict an observable that survives independent tests and known limits? | R4 navigation, an HDC vector, or a renderer's projection |
| Agency | Does source attribution plus repeatable information-bearing/adaptive structure survive natural, technological, and instrumental controls? | Periodicity, complexity, apparent intent, or low probability under one incomplete null model |

“Higher-dimensional being” would be a compound interpretation requiring independent support for both a relevant physical model and agency. Even that combination would not immediately establish that the source is a *being*, its location, or its nature. The system should emit Unknown / Needs more data whenever the measurements cannot distinguish the hypotheses.

There must be no universal scalar whose high value silently promotes all four axes. Publish the evidence vector and every axis' evaluator disposition.

## 3. Project boundaries

Compose existing work rather than duplicate it.

- **Symthaea** owns candidate proposals, time-series / multimodal inference, competing hypotheses, uncertainty, and ranking of follow-up observations. Existing [HYPERSPACE-001 #2023](https://github.com/Luminous-Dynamics/symthaea/issues/2023) tests synthetic R3/R4 navigation; [PR #7282](https://github.com/Luminous-Dynamics/symthaea/pull/7282) develops higher-spatial-dimensional wave-model discovery. Those are valuable mathematical competence tests, but do not by themselves establish extra physical dimensions or solve real-data candidate attribution.
- **Symtropy** owns explicitly declared forward models, simulation of source + instrument response + nuisance conditions, and deterministic numerical replay. [#1473](https://github.com/Luminous-Dynamics/symtropy/issues/1473) covers passive numerical-anomaly observation and [#1594](https://github.com/Luminous-Dynamics/symtropy/issues/1594) a bounded acoustic-emission sensor adapter. Neither is an astronomical sensor qualification.
- **Mycelix** should reuse existing scientific evidence typing [#2410](https://github.com/Luminous-Dynamics/mycelix/issues/2410), evidence-bearing investigation capsules [#2762](https://github.com/Luminous-Dynamics/mycelix/issues/2762), and receipt / capture-integrity work [#3573](https://github.com/Luminous-Dynamics/mycelix/issues/3573), rather than adding a second provenance ledger. Its role is to preserve and exchange source, derivation, evaluator and replication evidence—not to decide that a physical interpretation is true.

Required non-equivalences:

```text
simulated signal != physical observation
candidate != confirmed phenomenon
anomaly != new physics
new physics != extra-dimensional physics
extra-dimensional model support != agency
agency-like pattern != being
multiple model runs over one observation != independent replication
```

## 4. Benchmark arms

### A. Positive injection recovery

Inject versioned synthetic signal families into lawfully usable real backgrounds where licensing permits, or into declared realistic synthetic backgrounds while explicitly labelling the weaker evidence class. Preserve the original bytes, a separate injection manifest, and the exact background identity.

Vary amplitude / signal-to-noise ratio, morphology, drift or frequency structure, cadence, gaps, timing jitter, calibration, quantization, sensor response and background type. Include controls with no injected signal.

Never split random windows from one source observation across training and test and then count them as independent generalization. Split at the source observation / target / time-block / instrument group appropriate to the dataset.

### B. Held-out morphology and open-set generalization

Hold out entire signal-generating families, not merely examples. Include transformed, composite, pulsed, curved, intermittent, broadband and other declared variants as scientifically appropriate to the selected modality. The benchmark should also include a family unknown to both candidate development and threshold tuning.

The target may say Unknown / OutOfDistribution. It must not be forced into the nearest familiar class. Score the open-set decision independently from in-family classification.

### C. Nuisance and acquisition controls

Include known natural and human-made emitters, interference, common-mode signals, sensor saturation, aliasing, clock drift, missing samples, stale calibration, invalid units, duplicated windows, changed data under a reused identifier, incomplete capture, and pipeline leakage.

For modalities with context/control observations (e.g. telescope ON/OFF cadence), preserve all channels in the observation bundle. Explicitly ablate the context channels to show when source attribution becomes unidentifiable. The detector must not earn credit for identifying a source when decisive control information was withheld from it.

### D. Extra-dimensional model discrimination

Test only named, mathematically defined physical models with explicit parameter ranges, observable predictions, instrument response, competing conventional models, and relevant empirical constraints. Freeze the test observables before examining holdout labels.

A model only passes the declared comparison if it makes improved held-out predictions under the preregistered criterion, survives relevant controls, and reports parameter/model uncertainty. A flexible model that simply fits the observations better in-sample is not enough. A successful synthetic model-discrimination test shows algorithm competence on the simulated envelope—not that the model describes the universe.

### E. Structure versus agency

Include natural periodicity, nonlinear/chaotic processes, compression artifacts, common human-made transmissions, shuffled and phase-randomized surrogates, and known encoded signals. Test source attribution separately from structural complexity.

For radio-like data, test whether a proposed source remains distinguishable from local interference when the model is shown both target and control observations. For other modalities, define the corresponding paired control. The benchmark must not treat “hard for this model to explain” as synonymous with “intelligent.”

### F. Active follow-up / value of information

Give the system a bounded set of possible next observations or controls, their cost, and current competing hypotheses. Ask it to rank the next measurement most likely to discriminate among those hypotheses.

On synthetic ground truth, score expected discrimination / reduction in decision-relevant uncertainty against a random baseline, a fixed protocol and an oracle upper bound. Report cost per useful discrimination, not just ranking agreement. No synthetic prior or scenario frequency may be presented as a real-world event probability.

### G. Blind end-to-end campaign

Separate the candidate system, data/injection generator and evaluator. Freeze data splits, model/profile versions, thresholds, exclusions, primary metrics and hypotheses before revealing holdout labels. Preserve negative and failed cases. Report every timeout, excluded item and missing field; none may silently disappear from denominators.

## 5. Required outputs and metrics

Report each dimension independently. Accuracy alone is invalid for rare-event detection.

1. **Detection:** recall / detection probability by signal family and SNR; localization error where relevant; missed positives.
2. **False alarms:** rate per negative-control family and operational unit (e.g. observing hour or target observation); all false alarms; confidence intervals.
3. **False discovery burden:** calculate only with an explicitly stated prevalence / base-rate assumption, and show sensitivity to that assumption.
4. **Open-set handling:** unknown-family recall, false-known-label rate, and rate of unclassified cases. The forced-known-label target is zero.
5. **Calibration:** Brier or log score, reliability diagrams, and empirical interval coverage where the model supports calibrated probabilities or intervals.
6. **Shift robustness:** metrics on held-out backgrounds, instruments, time blocks and morphologies; never aggregate away a brittle subgroup.
7. **Model discrimination:** preregistered held-out predictive score / residual criterion with complexity penalty, plus comparisons to declared conventional baselines.
8. **Agency discrimination:** false agency promotions on known natural, terrestrial, human-made, shuffled and source-ambiguous controls; source attribution and structure scores remain separate.
9. **Value of information:** discrimination gained per observation / resource budget against random, fixed and oracle-follow-up policies.
10. **Provenance / integrity:** percentage of candidates with complete source identity, acquisition metadata, capture state, derivation chain, model version, evaluator and reproducible receipt.
11. **Cost:** throughput, latency and memory / CPU / GPU resource use per candidate, with skipped or timed-out items reported separately.

AUC, novelty score and synthetic recall are diagnostic metrics, not standalone acceptance gates. Thousands of injections into a handful of backgrounds are not thousands of independent background trials. `background_group` is the independent acquisition/background cluster and may appear only once in the entire v0.1 manifest. A paired target/control or ON/OFF observation must be represented as one trial bundle, not separate rows. Repeated windows/injections/crops from one acquisition must be aggregated using a preregistered rule or rejected. Keep train/calibration/holdout group separation, report independent group counts separately from row counts, and compute uncertainty from genuinely independent trials/groups rather than treating correlated rows as independent. Distinct group IDs must be justified by acquisition provenance; the parser cannot prove independence from labels alone.

## 6. Proposed v0.1 gates (targets, not results)

- **Replay:** 100% deterministic output for identical source bytes, manifest, profile, and seed; otherwise make all nondeterminism explicit and bounded.
- **Fail-closed integrity:** 100% of frozen integrity attacks must block promotion: required context/control missing, invalid units/timebase, stale or unknown calibration, duplicate identity, changed bytes under a reused ID, or oracle/holdout leakage.
- **Open set:** zero forced known-class labels when the evaluator truth is Unknown / OutOfDistribution.
- **Initial synthetic detection floor:** require a one-sided 95% Wilson lower confidence bound for recall of at least 90% in each declared detectable-positive stratum, and a one-sided 95% Wilson upper bound on false-positive rate of at most 1% **within every predeclared negative-control family**. Both use independent background-group trials, not window counts. If the point estimate misses its floor, record FAIL; if the point estimate meets the floor but the confidence bound does not, record INCONCLUSIVE / insufficient independent evidence. Never lower the target post hoc.
- **Higher-specificity follow-on:** seek a per-family false-positive upper confidence bound at or below 0.1% on independent, properly grouped controls with a locked threshold. Roughly 3,000 independent zero-failure Bernoulli trials give a simple 95% “rule of three” upper bound near 0.1%; correlated windows do not qualify as independent trials. If data volume or independence is insufficient, status is unresolved.
- **Held-out families:** 90% recall on each preregistered held-out family at its declared detectable SNR is a stretch target. It is never permission to tune on the holdout.
- **Evidence boundary:** 100% of outputs preserve whether a record is raw, derived, simulated or independently measured and retain explicit capture completeness.
- **Unknown remains allowed:** inability to separate hypotheses must produce Unknown / Needs more data, never a speculative promotion.
- **Claim ceiling:** synthetic success, a single sensor/model family, or a visualization cannot establish physical extra dimensions or agency.

These are project targets for the proposed harness, not claims about current Symthaea performance or universal scientific standards.

## 7. Frozen negative controls

A conforming evaluator must demonstrate:

- Removing target/control cadence information blocks source-origin or agency promotion when that attribution is no longer identifiable.
- The familiar narrowband detector receives no credit for a held-out morphology it misses.
- Always-Reject fails recall; Always-Accept fails the negative controls.
- A high-dimensional HDC representation or valid synthetic R4 path never counts as evidence of physical extra dimensions.
- A numerical artifact or simulator-only feature never counts as measured physical evidence.
- Duplicate, transformed or reanalysed copies of one observation never count as independent replications.
- Missing metadata or incomplete capture is never interpreted as proof that an artifact or control signal was absent.
- A rare or structured observation alone never triggers the agency label.
- Post-hoc threshold changes or holdout-guided hypothesis selection invalidate confirmatory status.
- More candidates are not counted as more independent observations if they share a single source/background.
- Evaluator exhaustion, resource limit, parser error or unsupported profile is explicit Unknown/Invalid—not a negative result and not a passing result.

## 8. First implementation sequence

Keep the first tranche small and local; no new GPU pipeline or heavyweight hosted workflow is required.

1. Freeze an observation-bundle schema with byte identity, channel roles, instrument profile, units/timebase, capture-completeness states, source/background group, licensing, and raw/derived/simulated classification.
2. Add a versioned synthetic-injection manifest and split validator that rejects group leakage.
3. Implement simple deterministic conventional baselines (threshold/matched filter appropriate to the selected signal family), alongside Always-Unknown and Always-Positive controls.
4. Implement an independent evaluator for per-family recall, false-positive rates, confidence intervals and exact denominator accounting.
5. Emit a canonical receipt binding corpus hash, source-group split, injection manifest, evaluator version, thresholds, all candidate outcomes and failure counts.
6. Add context-channel ablation and an entire held-out morphology before adding model complexity.
7. Connect to Symtropy's forward model only through a declared source/instrument adapter; connect to Mycelix through its existing evidence receipt / capture-integrity contract.
8. Use public datasets only with their terms and provenance recorded. Before claiming physical relevance, document instrumental and selection effects.

## 9. Research basis

- **NASA UAP Independent Study (2023):** recommends well-calibrated instruments, complete sensor metadata, multi-sensor observations and disciplined treatment of uncertainty; available observations were insufficient to draw firm conclusions. [NASA UAP resource page](https://science.nasa.gov/uap/) · [official final report PDF](https://www.nasa.gov/wp-content/uploads/2023/09/uap-independent-study-team-final-report-0.pdf)
- **Breakthrough Listen deep-learning search:** its candidates did not reappear in follow-up observations, illustrating the distinction between candidate triage and confirmed technosignature. [Berkeley SETI](https://seti.berkeley.edu/ml_gbt/)
- **Imran (2026-10-15), arXiv:2610.22050:** a new preprint (not treated here as peer-reviewed) reports substantial false alarms and missed detections in 15,000 injected morphology/amplitude probes in Green Bank backgrounds; one identified design limitation was discarding OFF/context panels before inference. [Preprint](https://arxiv.org/abs/2610.22050)
- **Particle Data Group, Review of Particle Physics (2024), “Extra Dimensions”:** describes current astrophysical, tabletop and collider constraints. Use named model-specific observables and current constraints; extra dimensions do not imply intelligence. [Review PDF](https://pdg.lbl.gov/2025/reviews/rpp2025-rev-extra-dimensions.pdf)

## 10. Exit claim

If v0.1 passes, the strongest justified statement is:

> Under the exact declared dataset, signal families, background groups, profiles and evaluator, the benchmark reproducibly measures candidate recovery, false alarms, open-set behaviour, selected distribution shifts and provenance integrity.

It does **not** establish that an unknown real phenomenon exists, that an extra-dimensional physical model is correct, that a signal is intelligent, or that a higher-dimensional being exists. Each stronger statement requires independent, claim-specific observational evidence and replication.

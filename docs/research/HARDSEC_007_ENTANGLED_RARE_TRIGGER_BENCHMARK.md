# HARDSEC-007 — Entangled-Trojan, Rare-Trigger and Verified-Augmentation Evaluation

**Status:** research protocol and sampling-budget utility  
**Scope:** defensive benchmark/evaluation; no Trojan payload generation, detector training or formal reachability run is claimed  
**Related work:** [HARDSEC-005 lineage audit](https://github.com/Luminous-Dynamics/symthaea/pull/9091), [HARDSEC-006 corpus qualification](https://github.com/Luminous-Dynamics/symthaea/issues/9100), [HARDSEC-007 issue](https://github.com/Luminous-Dynamics/symthaea/issues/9105), [HARDSEC-002 secure generation](https://github.com/Luminous-Dynamics/symthaea/pull/9040)

## 1. Research objective: detect what is not structurally convenient

A serious hardware-Trojan evaluation must not assume that malicious logic:
- forms a small weakly-connected component;
- switches under ordinary random input vectors;
- survives synthesis with its labels intact;
- is represented by a complete, unambiguous training label;
- can be localized whenever the circuit-level verdict is correct.

The benchmark therefore separates five claims:

1. **Verdict:** is the complete design Trojaned?
2. **Localization:** which exact gates/nets/cones are affected?
3. **Activation:** was the trigger condition reached under a declared stimulus / environment model?
4. **Functional preservation:** did benign augmentation preserve the intended circuit behavior?
5. **Evidence integrity:** are the source artifacts, labels, lineage, transforms and held-out groups trustworthy?

No single F1 score may collapse these different assurance questions.

## 2. Contest baseline: what the public evidence actually says

The [ICCAD 2025 Problem A contest](https://www.iccad-contest.org/2025/Problems.html) was sponsored by Cadence Design Systems and addressed golden-free hardware-Trojan detection on gate-level netlists. The repository [zcgilbert/iccad-2025-hardware-trojan-detection](https://github.com/zcgilbert/iccad-2025-hardware-trojan-detection) reports an official result for its entry in the Taiwan domestic division:

- 60 hidden test cases, 60/60 executions completed;
- 50/60 correct complete-design verdicts;
- design-level precision 87.5%, recall 87.5%, F1 0.875;
- median gate-level localization F1 about 0.77 among correctly detected Trojan designs;
- an author-reported Top Prize for Problem A in the Taiwan domestic division.

The hidden 60 cases are not released. The repository contains the author's official scorecard and submitted predictions, but a third party cannot reproduce the hidden-case experiment from the public files. Treat these values as scorecard-backed author-reported results, not our independent reproduction.

The result also illustrates why verdict and localization must be scored separately. A correct design verdict did not always mean the predicted gate set matched the actual Trojan; one reported case had zero gate overlap.

## 3. Four concrete failure mechanisms found in the public implementation

### 3.1 Entangled logic defeats component assumptions

The submitted model uses a bidirectional graph-convolutional node branch and a weakly-connected-component/subgraph branch. The latter assumes that Trojan payload logic often forms a relatively small component weakly attached to the host after cutting at sequential boundaries.

The repository documents a missed case with 23 Trojan gates, 19 of which are embedded inside the host's 388-gate main component. The Trojan gates make up only about 5% of that component. The model fails to detect it. The hidden set also contains five missed Trojan verdicts and five false alarms.

This suggests a structural stress test: the detector must work when malicious logic reuses host gates/nets, shares cones, or is interleaved with ordinary logic—not only when it forms a separable subgraph.

### 3.2 Random simulation does not establish rare-trigger coverage

The public feature builder applies 1,000 random input vectors per design for switching-activity features. Its README explicitly acknowledges that triggers rarer than approximately 1 in 1,000 may appear constant at this budget.

For an assumed independent per-vector trigger activation probability p and N independent stimuli:

`P(no activation) = (1-p)^N`

At 1,000 vectors, the probabilities are:

| Assumed activation probability per independent vector | Probability of zero activations |
|---:|---:|
| 10^-3 | 36.77% |
| 10^-4 | 90.48% |
| 10^-6 | 99.90% |

These are mathematical consequences of the IID assumption, not estimates of any actual Trojan's trigger probability. Correlated vectors, sequential state, environmental constraints and deterministic stimulus schedules invalidate the simple IID model.

The practical lesson: a quiet simulation trace is not evidence that a trigger is absent or unreachable.

### 3.3 Augmentation bugs can poison labels and training data

The repository discloses that its early rewrite generator had incorrect wiring in three XOR/XNOR alternatives. Those rules were later changed to explicit wiring templates and all 15 Boolean rewrite rules were checked exhaustively against their full truth tables. The repository also says the shipped contest checkpoint predates that correction, so some fraction of its 180 augmented training netlists may have been non-equivalent.

The contest score remains the score actually submitted. A model retrained from the corrected source does not train on byte-identical historical data. A faithful reproduction must pin the pre-fix and corrected corpus states separately, or regenerate and label the new experiment as a corrected-corpus retraining.

### 3.4 Label defaults and training selection need stricter controls

The public training script reports and selects the best F1 on the same graph loader used for gradient updates; it does not create a distinct validation split in that script. This does not invalidate the reported hidden contest score, but it is not a defensible model-selection protocol for a new confirmatory experiment.

The repository's label parser maps a missing, unreadable or malformed label file to NO_TROJAN. The published manifest contains 2,300 rows, of which the repository's CSV parses as:
- 2,120 TROJANED;
- 120 NO_TROJAN;
- 60 UNLABELLED.

The narrative calls the entire corpus labelled, but those 60 manifest rows are explicitly unlabelled. Missing and malformed labels must remain unknown / invalid, never silently become clean negatives.

The repository also contains an exact-SHA256 public-case answer lookup behind an explicit option. It is disabled by default and documented as a contest shortcut, not detection. It must remain disabled and separately audited in all model-evaluation runs.

## 4. Workstream A — entangled / non-separable Trojan stress

Use authorized benchmark cases and source-provided ground truth. Do not create or deploy malicious payloads as part of this work.

Characterize each positive design by:
- Trojan gate count and share of total gates;
- component purity of Trojan gates after each declared graph partition;
- fraction of Trojan gates in the dominant host component;
- number of host/Trojan cone intersections and shared net/gate references where annotations support them;
- sequential depth / state-boundary proximity;
- label-map provenance and confidence.

Report verdict and localization separately, stratified by Trojan size and structural separability. Candidate representations to test after baselines include:
- bounded multi-hop context that does not require a standalone component;
- edge- and pin-role-aware message passing;
- signal-cone / fan-in-fan-out views crossing sequential boundaries;
- multiple graph partitions (combinational components, sequential cones, control/data-flow);
- local structural motifs plus global context;
- paired host/variant delta graphs only as a separate golden-reference arm where valid mappings exist.

Do not assume an architecture wins. Compare against the original component branch, node-only branch, RF/XGBoost and a compact GCN on identical held-out designs.

## 5. Workstream B — rare and sequential trigger activation

A recent sequential-benchmark lead is **SeqCube-HT: state-cube-guided generation of sequential hardware Trojan benchmarks**, published 9 October 2026 (DOI: https://doi.org/10.1186/s42400-026-00676-2). Its stated problem definition requires trigger execution from a declared reset state and payload effects to reach an observation boundary. This is the right abstraction for sequential-trigger tests: test both state reachability and effect observability, not merely whether an internal trigger predicate toggled. Treat it as a benchmark-generation methodology candidate until the exact artifacts, scripts, license and ground-truth files are independently located and pinned; the paper's publication alone is not a verified local dataset.

The new Rust utility examples/hardsec_trigger_budget.rs calculates the miss probability for a user-supplied hypothetical p and N, and the number of IID trials required to reach a target miss probability. It is a budgeting tool only.

Build the activation experiment around distinct evidence classes:

| Method | Evidence that may be claimed |
|---|---|
| Random vectors | Observed activations at a fixed seed/vector budget; no proof of non-reachability |
| Directed vectors | Observed activation for the declared boundary/value stimulus set |
| Constrained random | Coverage under versioned, validated constraints |
| Sequential stimulus | Activation over declared initial states, input histories and clock/cycle bounds |
| SAT/SMT / bounded model checking | Reachability witness or bounded result under exact model, assumptions, solver and bound |
| Timeout / unsupported model | Unknown; never a benign or unreachable verdict |

A concrete open-source tool path is Yosys + SymbiYosys (SBY) for assertions and bounded/unbounded safety checks, cover-trace generation and liveness jobs, and EQY for gold-vs-gate equivalence. For activation, use an explicit SBY `cover` target and archive the generated witness VCD and job/configuration. For augmentation, use EQY to compare the source design against the transformed design with exact gold/gate scripts and matching rules. EQY's partition results and SBY's solver outcomes are part of the receipt; do not reduce them to a generic pass boolean.

A cover witness establishes that the target was reached under the encoded model/assumptions and bound. A bounded cover that finds no witness is not a proof of unreachability. EQY can only establish equivalence for the precise models, scripts, matching and assumptions supplied; unknown or timed-out partitions remain unknown.

Record exact netlist/model digest, initial-state assumptions, input constraints, random seed, number of vectors/cycles, clock semantics, solver and version, time limit, witness digest and final proof/timeout status. Distinguish:
- trigger condition activated;
- payload condition executed;
- malicious effect observed;
- absence of activation within a finite budget.

These are not interchangeable. A failed search is not evidence of absence. A bounded proof is bounded by its exact assumptions and horizon; do not call it an unbounded proof.

## 6. Workstream C — verified augmentation and label-map integrity

Every accepted augmentation must have a source/derived artifact lineage receipt with:
- source and output SHA-256;
- exact transformation rule, version, parameters and random seed;
- gate/net mapping and label-map digest;
- full truth-table equivalence for small combinational rewrites where feasible;
- whole-design combinational equivalence or a scoped sequential equivalence result where applicable;
- exact EDA tool/version/options/constraints and return code;
- explicit pass, fail, unknown, not_applicable outcomes.

For Trojan-labelled gates, verify that the transformation maps each labelled source gate to the expected derived gates and does not label unrelated gates. For benign augmentation, verify no new positive label appears. Sequential elements, reset logic and state encoding require sequential-equivalence reasoning; do not apply combinational identities to stateful elements without an appropriate proof.

Any failed/unknown equivalence, incomplete label mapping, missing label, parser error or provenance mismatch disqualifies the sample from supervised scoring. It may remain in a separately labelled diagnostic corpus, never be silently reclassified as benign.

A practical open-source equivalence path is EQY on top of Yosys/SymbiYosys: compare the original (gold) RTL/netlist with the transformed (gate) artifact, preserve exact read/elaboration scripts, matching rules and tool versions, and retain per-partition proof outcomes and counterexample traces. This is appropriate for augmentation/function-preservation evidence when a trusted source reference is available. It does not make a golden-free detector golden-reference dependent; it is an independent data-quality oracle on the transformation pipeline.

## 7. Workstream D — benchmark protocol and model comparison

Freeze independent host/netlist/Trojan-family groups before training. Keep these splits distinct:
- random gate split: diagnostic only;
- leave-one-netlist-out;
- leave-one-host-design/family-out;
- leave-one-Trojan-family-out;
- synthesis/toolchain/library holdouts where source support exists;
- entangled stress stratum;
- rare-trigger activation stratum.

Use a distinct training, validation and test design set. Hyperparameter and threshold selection must use validation groups only. The contest hidden set is unavailable; do not invent a recreation of its score.

Compare at least:
- original ICCAD GCN plus component branch (subject to code/data rights and reproduction feasibility);
- its node-only and component-only ablations;
- RF, XGBoost and logistic regression on pinned handcrafted features;
- compact GCN / GraphSAGE and published GraphSAGE-Transformer / SMOTrojan baselines when their data and splits can be pinned;
- graph-context / cone-aware alternatives;
- Symthaea HDC candidate.

The repo has no explicit root LICENSE file, so it is a literature/reference comparator until code/weight/data rights are confirmed. Its synthesis flows require commercial Synopsys Design Compiler and Cadence Genus licences. Do not imply the synthesis corpus can be recreated on an unlicensed environment.

## 8. Workstream E — engineer-actionable evidence, not generic heatmaps

A recent explainability study provides a useful baseline for this requirement: Whitten et al., “Explainability Methods for Hardware Trojan Detection: A Systematic Comparison” (published 3 August 2026; DOI 10.1007/s10836-026-06245-2). On its 30-circuit / 19-package Trust-Hub corpus, a 60/20/20 random gate split gave XGBoost F1=0.568 (precision 48.08%, recall 69.44%, AUPRC 0.637); with the paper's fixed thresholds under leave-one-family-out evaluation, XGBoost micro-F1 fell to 0.033 and Random Forest to 0.003. The authors note only five host families and synthetic digital combinational cases; this is a boundary study, not an industrial population estimate.

The same study reports only moderate gate-level rank agreement between LIME/SHAP explanations (mean Spearman rho=0.30). It found value in circuit-domain properties and retrieval of similar labelled examples, while noting that explanations need to translate into actionable hardware-engineering judgments. This motivates a different target: every alert should carry a **reviewable evidence packet**, not just a saliency image.

An evidence packet should contain:
- exact design/netlist hash, model/configuration digest, prediction threshold and score;
- predicted affected gates/nets/cones with mapping confidence and localization metrics where ground truth is available;
- circuit-domain properties supporting the alert (fanin/fanout, register boundaries, I/O reachability, component/cone context, trigger-path depth), with raw values and units;
- a small set of precedent examples with exact dataset/split/source provenance, clearly marked as similar cases rather than proof;
- a stimulus/activation witness or formal counterexample where one exists; timeout/unknown must be explicit;
- source/derived artifact and label-map receipts, augmentation-equivalence status, and exact tool provenance;
- explicit alternatives and uncertainty: why the signal may be benign, which evidence is missing, and what an engineer should verify next.
 
The executable contract is implemented in `examples/hardsec_evidence_packet_audit.rs`, registered behind Cargo feature `hardsec-evidence-audit`. It accepts a strict JSON packet with:
- `subject`: design ID and exact evaluated-netlist SHA-256;
- `detector`: method, model SHA-256, config SHA-256, score, threshold and decision;
- `evidence[]`: unique ID, kind, artifact URI, evidence SHA-256, subject SHA-256, tool/version, scope and status; optional witness/assumption hashes, stimulus counts, proof scope and bound;
- `claims[]`: claim type, status (`supported`, `contradicted`, `unknown`, `not_evaluated`), explicit scope and evidence IDs.

Formal evidence has an additional typed result: `assertion_proved`, `bounded_no_counterexample`, `counterexample_found`, `cover_reached`, `equivalence_proved`, `timeout`, or `unknown`, with property ID and proof scope as appropriate. The validator enforces status/result consistency. In particular:
- a bounded no-counterexample result can support only `bounded_reachability_not_found` with a positive cycle bound;
- `trigger_unreachable_under_assumptions` requires an unbounded assertion proof and an assumptions digest;
- `trigger_not_observed_under_budget` requires a passing trace with a nonzero stimulus count and zero activations, but never implies unreachability;
- `trigger_activated` requires an activation count and a hashed witness;
- `functional_equivalence` requires an explicit passing equivalence report;
- generic `trojan_absent` and `design_is_secure` claims are rejected.

The tool emits a deterministic, unsigned receipt outside the packet directory. It validates packet shape, claim ceilings and internal hash binding; it does **not** download or hash the referenced evidence bytes, verify tool execution, authenticate signatures, validate assumptions, or prove the evidence truthful. Its receipt says so explicitly. It is an independent consistency check, not a hardware security verdict.

### Explanation-quality evaluation

Score explanations independently from detector accuracy:
- localization overlap: gate/net/cone precision, recall, IoU and support;
- evidence sufficiency: fraction of alerts with all required provenance, domain measurements and model lineage;
- stability: explanation changes under verified benign-equivalent transformations and harmless identifier/synthesis perturbations;
- counterfactual faithfulness: whether removing or neutralizing the claimed evidence in a controlled copy changes the prediction as described, while independent equivalence/functional checks ensure the control is valid;
- reviewer actionability: time-to-triage, fraction of alerts accepted/rejected with a cited reason, disagreement and inter-reviewer consistency in a controlled human study;
- false-alert burden at fixed analyst capacity, e.g. false positive gates/cones per 1,000 gates reviewed.

Do not call an attribution “causal” solely because it has SHAP values or saliency. Do not claim improved human interpretability without a controlled study with hardware-security practitioners. Case retrieval must not leak test labels or public answers.

Reference:
- Whitten et al. (2026): https://doi.org/10.1007/s10836-026-06245-2

## 9. Multi-axis benchmark report

| Axis | Required metrics |
|---|---|
| Complete-design verdict | Precision, recall, F1, MCC, confusion matrix, false-positive burden |
| Gate localization | Precision, recall, F1, IoU, support and class prevalence |
| Net/cone localization | Precision/recall/IoU where gold mapping exists |
| Entanglement | Metrics by component purity, shared-cone fraction and Trojan gate share |
| Trigger activation | Coverage vs vectors/cycles, sequence depth, witnesses, timeouts and solver cost |
| Payload effect | Observed effect separate from trigger activation |
| Augmentation validity | Truth-table and whole-design equivalence, label-map completeness |
| Data validity | Unknown/invalid count, provenance checks, exact hashes and group leaks |
| Generalization | LOCO/LOFO/Trojan-family/toolchain/library strata and worst group |
| Efficiency | Training/inference/simulation/formal cost, model size, memory, wall time, energy only if measured |

Publish Pareto frontiers rather than hiding verdict, localization, activation and resource cost in a single composite score.

## 10. Proposed project gates

These are research targets, not achieved results, industry standards or certification criteria:

1. Zero missing/malformed labels silently enter the clean class.
2. Every training augmentation accepted into a confirmatory run has exact source/output hashes, passed equivalence at the appropriate semantic level, and a complete label map.
3. Family-held-out evaluation has no sibling-design leakage; random-gate splits are diagnostic only.
4. Show a measurable improvement in the predeclared entangled-case macro metric at the same false-positive budget, with worst-host results and group-aware uncertainty.
5. Every detector alert has an evidence packet; explanation stability and localization are measured independently from classifier F1.
6. Plot random/directed/constrained/sequential activation coverage against stimulus budget. A rare-trigger claim requires an activation witness or a sound formal result under explicit assumptions.
7. Report detection and localization as distinct endpoints; a correct circuit verdict does not grant localization credit.
8. All public answer lookups, hash caches and test-time labels are excluded from model predictions.
9. The official contest result remains non-reproducible while hidden testcases are unavailable.

The exact effect size and minimum independent design/family support must be set before the confirmatory split is opened.

## 11. Reproduction commands

```bash
cargo test --example hardsec_trigger_budget
cargo run --example hardsec_trigger_budget -- \
  --activation-probability 0.0001 --trials 1000 \
  --target-miss-probability 0.05

cargo test --example hardsec_evidence_packet_audit --features hardsec-evidence-audit
cargo run --example hardsec_evidence_packet_audit --features hardsec-evidence-audit -- \
  --packet /path/to/evidence-packet.json \
  --receipt /path/outside/packet/evidence-audit.json
```

Expected mathematical result for the example: at p=10^-4, 1,000 independent trials have about 90.48% chance of zero activations; approximately 29,956 IID trials are required to reduce that miss probability to 5%.

The test suite checks known-answer miss probabilities, the minimum trial count, stable small-probability arithmetic, edge cases p=0 and p=1, invalid inputs and unrepresentably large budgets. These tests are authored but have not been compiled or executed in this environment.

Useful tool references:
- SymbiYosys: https://symbiyosys.readthedocs.io/en/latest/ (bounded/unbounded safety, cover and liveness)
- EQY equivalence tutorial: https://formal.org/tutorials/symbiyosys/equivalence-checking (gold/gate comparison, partition proofs and counterexamples)
- SeqCube-HT (2026): https://doi.org/10.1186/s42400-026-00676-2 (state-cube-guided sequential Trojan benchmark methodology)

## 12. Scope boundary

This is defensive benchmark and evaluation infrastructure. It does not construct or deploy Trojan payloads, claim a Trojan is absent because finite simulation was quiet, or certify silicon. Formal results must preserve assumptions, bounds and solver provenance. The public contest repository's code/checkpoint/data must not be redistributed until a license or explicit permission is established.

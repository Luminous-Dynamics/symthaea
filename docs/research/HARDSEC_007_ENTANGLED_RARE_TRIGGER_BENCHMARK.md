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

For this programme, the evaluation oracle must separately report:
- **trigger activation:** the trigger condition is reached from a declared reset/initial-state model;
- **payload effect:** the intended effect reaches a declared architectural or externally observable boundary;
- **design verdict:** the complete design is classified as suspicious/clean/unknown;
- **localization:** affected gates/nets/cones are identified.

A trigger witness is not automatically a payload-effect witness. A payload-effect trace is not a proof that every trigger condition is covered. SeqCube-HT is therefore a useful sequential benchmark lead, but its exact dataset artifacts and license must be pinned before reproduction claims.

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

Formal records require both a `property_id` and a typed `formal_result`:
- `assertion_proved` or `bounded_no_counterexample` for formal assertions;
- `cover_reached` for a cover witness;
- `equivalence_proved` for equivalence reports;
- `counterexample_found`, `timeout` or `unknown` for failed/indeterminate outcomes.

The validator checks that these outcomes agree with the record's `pass`/`fail`/`unknown` status. A generic "pass" flag is insufficient to claim unreachability. It also separates `trigger_activated` from `payload_effect_observed`; the latter requires its own payload-effect trace, observation count and witness digest. The corresponding `*_not_observed_under_budget` claims are explicitly finite-budget observations, never unreachability statements.

Example claim types supported by this version:
`model_suspects_trojan`, `localization_provided`, `trigger_activated`,
`trigger_not_observed_under_budget`, `payload_effect_observed`,
`payload_effect_not_observed_under_budget`, `bounded_reachability_not_found`,
`trigger_unreachable_under_assumptions`, `functional_equivalence`,
`source_integrity_checked`, and `reviewer_confirmed`. Generic
`trojan_absent` / `design_is_secure` claims are not supported.

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

## 9. Known-answer sequential witness fixture: SeqCube-HT Supplementary Artifact

A public witness-validation package is available at:
https://github.com/kxiaud/seqcube-ht-witness-validation-artifact

The independent replay work is tracked in [HARDSEC-008 / issue #9136](https://github.com/Luminous-Dynamics/symthaea/issues/9136), pinned to upstream commit `e5acac37af6dbe1fa53b0be93199bb81a62c5320`.

**Critical scope distinction:** its own `manifest.json` declares `reported_corpus_member: false` and `generator_included: false`. It is an author-created sequential controller/datapath reference design for independently inspecting a bounded replay-witness schema. It is **not** one of the SeqCube-HT experimental corpus members and does not include the paper-internal benchmark generator, complete benchmark corpus, detector experiments, or a source from which to infer population-level detector performance.

### What the reference package provides

The README and manifest describe:
- a clean design with 15 state bits / 45 cells and an infected counterpart with 20 state bits / 83 cells;
- a three-phase reset-consistent trigger at cycles 2, 4 and 6;
- two registered payload stages, with the first clean/infected output difference at primary output `y`, cycle 8;
- a matched 10-cycle inactive vector sequence changing `ack` at cycle 4 to suppress the second phase and prevent the recorded output difference;
- supplied active and inactive SMT-LIB files, expected SAT/UNSAT statuses, netlist-derived query reconstruction, structural label inventory and optional Icarus replay testbench;
- a static SHA-256 inventory of 25 files.

The repository includes an author-supplied `logs/verification_report.json` with:
- active supplied/rebuilt status `sat`, 352 reconstructed assertions;
- inactive supplied/rebuilt status `unsat`, 351 reconstructed assertions;
- Icarus replay status `passed`;
- active primary-output difference first observed at cycle 8;
- no output difference in the inactive trace.

These are the **publisher-supplied recorded results**, not an independent result from this project. The focused HARDSEC workflow is now configured to fetch the exact upstream commit, verify its `checksums/SHA256SUMS`, install the declared Z3 dependency and Icarus Verilog, and run the upstream verifier with `--require-iverilog`. Its stdout/stderr, toolchain versions and regenerated report are uploaded as an artifact. The result for the current byte-verification revision is **not yet recorded**; until its exact-head job completes successfully, do not present the upstream report as independently reproduced.

The package's own README provides the replay command:

```bash
python -m pip install -r verify/requirements.txt
python verify/verify_package.py --package . --require-iverilog
```

Its own rights metadata says MIT applies to `verify/**`, and CC BY 4.0 applies to the author-created design, traces, metadata, SMT files, logs and checksums. Preserve attribution and note changes if reusing those data files.

### HARDSEC-007 packet fixture

The evidence-packet validator now has a `witness_validation` mode that intentionally omits the detector object and model digest. This is essential: a formal/replay fixture must not invent an AI model identity merely to use the same packet format.

A dedicated Rust test constructs a detector-free `witness_validation` packet against the public reference package's manifest digests and repository file locators. The packet mode forbids a detector object/model identity and rejects any attempt to claim `model_suspects_trojan`. It keeps five claims distinct:
1. the active three-phase trigger was observed in the recorded active trace;
2. a separate payload-effect trace reports the output effect at cycle 8;
3. the inactive control observed no trigger under its 10-cycle budget;
4. the inactive control observed no payload effect under its 10-cycle budget;
5. the inactive SMT record supports only a bounded no-counterexample claim for its declared 10-cycle bound.

The fixture binds these records to the publisher-listed infected-netlist SHA-256:
`17dbd7add0e8cf2705d9368e3cd3ccf62cdb52f3b8dacf3e33aa04b65ea9941de`.

It also references the publisher-listed digests for the active trace, inactive trace, boundary result, inactive SMT input and configuration. The URIs are pinned to the upstream commit, and the packet includes local relative paths for the evaluated netlist, each evidence artifact, the active witnesses and the formal assumptions file.

The auditor now accepts `--evidence-root DIR`. In that mode it opens and SHA-256 checks the subject netlist and every referenced evidence, witness and assumptions file. It rejects absolute/parent paths, non-regular files, symlinked path components, missing local locators and digest mismatches. The receipt enumerates each verified relative path and digest and changes its claim scope to `packet_schema_claim_and_referenced_byte_integrity_only`. This establishes **local byte integrity**, not source authenticity, tool execution, solver correctness, signatures, or truth of the recorded outcome. The path walk and subsequent open are not protected against a hostile concurrent filesystem replacement; CI therefore verifies an immutable extracted package in a controlled runner.

The focused workflow downloads the pinned package archive, verifies the publisher's full checksum inventory, runs the package verifier with mandatory Icarus, then invokes the Rust auditor using that extracted package as `--evidence-root`. The expanded exact-head result is pending.

The evidence-packet auditor now has **32 authored unit tests**, including six new byte-integrity tests for correct bytes, altered evidence, altered netlist, missing locators, path traversal and symlinks. The trigger-budget utility has nine tests. Before byte verification was added, a prior exact-head run passed rustfmt, 9/9 trigger-budget tests and 26/26 evidence-packet tests; the six new tests and upstream replay are not yet represented by a completed exact-head run.

## 10. Multi-axis benchmark report

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
| Witness provenance | Fraction of subjects/evidence/witness/assumption bytes checked against exact local paths and hashes |
| Active/inactive pairing | Paired trigger and payload outcomes under matched reset, vector length and observation boundary |
| Efficiency | Training/inference/simulation/formal cost, model size, memory, wall time, energy only if measured |

Publish Pareto frontiers rather than hiding verdict, localization, activation and resource cost in a single composite score.

## 11. Proposed project gates

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

## 12. SOTA targets and industrial comparison

### What the published numbers actually say

Three 2026 results change how this programme should benchmark itself:

1. **Split leakage is not hypothetical.** Xiao et al. report Random-Forest gate F1 of 0.914 on pooled cross-validation, 0.636 on leave-one-circuit-out, and 0.460 on leave-one-base-family-out; average precision falls from 0.978 to 0.577 under the family holdout. The study covers 49,124 gates across 16 netlists and five base-circuit families. It makes LOFO/LOCO the confirmatory protocol; pooled/random splits are diagnostics only.
2. **Sequential witness generation has a concrete baseline.** SeqCube-HT reports 3,240 witness-validated instances from 4,500 attempts (72.0%) and 20.9 seconds per accepted instance, compared with 50.6% and 39.5 seconds for the best matched comparator. Its detector probes also show a source-transfer penalty: SeqCube F1 falls from 69.2% instance-random to 47.8% source-held-out; localization recall falls from 55.7% to 31.4%. The paper's covariate-matched 77.2% F1 is a different protocol and must not be compared as if it were the held-out result.
3. **Industrial scale exposes another failure regime.** A DATE 2026 paper describes function-preserving stress patterns in large netlists using SCOAP testability and graph transformations, and reports that representative graph-learning detectors fail on its generated challenge instances. This is evidence that academic-sized benchmark success is not sufficient—not proof that a particular Symthaea detector is better.

### Confirmatory targets

These are proposed research gates, not measured results, standards or certification criteria:

- **Detection:** on untouched leave-one-base-family-out and leave-one-insertion-source-out splits, improve macro F1 by at least 5 absolute points over the strongest tuned baseline at the same validation-selected false-positive rate; report average precision, MCC, per-family intervals and worst-family performance. The target only counts if there are enough independent families to estimate uncertainty.
- **Localization:** improve gate/net/cone recall by at least 5 absolute points at a fixed analyst false-positive burden, without sacrificing whole-design precision. Explainability is a separate endpoint.
- **Sequential witness generation:** on the same nine hosts, matched 4,500-attempt budget and identical validation rules, target >72% witness yield and <20.9 seconds per accepted witness. Report the full rejection-stage profile, not just successful instances. These are deliberate stretch gates against SeqCube-HT's matched experiment.
- **Evidence completeness:** 100% of supported claims must bind to the exact subject hash and all referenced evidence bytes; 0 silent unknown-label-to-clean conversions; 0 lineage leakage across held-out groups. Unknown remains unknown.
- **Negative controls:** every deliberately altered trace, netlist, assumption digest, bound, reset model or observation endpoint must be rejected or downgrade the claim to unknown. No finite no-witness result may become unbounded unreachability.
- **Operational efficiency:** compare CPU time, peak memory, detector latency, simulator cost and proof cost at matched acceptance/false-positive levels. Only report energy when measured on a named platform.

### Industry comparison: complement the sign-off stack

OpenTitan is a useful open-source industrial-methodology reference, not a directly comparable Trojan classifier. Its verification methodology combines constrained-random simulation, formal property verification, IP/core/chip tiers, functional and code coverage closure, security/error tests and review/sign-off checkpoints. Its security model also includes physical penetration testing, attestation and lifecycle guarantees. Our target should be **interoperable assurance evidence that plugs into such flows**, not a claim that a classifier replaces UVM, formal sign-off, synthesis checks, physical testing or human design review.

NIST IR 8615's September 2026 workshop report likewise emphasizes interoperable evidence-based trust models, lifecycle provenance/traceability, scalable verification and validation, and public-private collaboration. The HARDSEC packet is intended to make these assurance claims portable across tools; it is not itself a certification scheme.

### The paradigm shift

The research object should be a **paired, provenance-bearing security experiment**, not a labelled netlist row. For every challenge, retain:
- exact clean/infected subject identities and synthesis lineage;
- reset model, active trigger trace, payload-effect witness and matched inactive counterfactual;
- formal property, bounds, assumptions and counterexample/cover artifacts;
- detector output and localization as separate predictions;
- local byte-integrity receipt, toolchain identities and replay status;
- family/source split assignment and negative-control outcomes.

That makes it possible to study detection, localization, rare-trigger coverage, payload observability, provenance and operational cost independently. Symthaea should be judged against Random Forest, XGBoost, graph models and simple rarity/testability baselines under exactly the same splits. A novel architecture earns a claim only when it beats those baselines on the held-out families and preserves evidence quality.

## 13. Reproduction commands

```bash
cargo test --example hardsec_trigger_budget
cargo run --example hardsec_trigger_budget -- \
  --activation-probability 0.0001 --trials 1000 \
  --target-miss-probability 0.05

cargo test --example hardsec_evidence_packet_audit --features hardsec-evidence-audit
cargo run --example hardsec_evidence_packet_audit --features hardsec-evidence-audit -- \
  --packet /path/to/evidence-packet.json \
  --receipt /path/outside/packet/evidence-audit.json

# When the packet supplies local_path locators for every artifact:
cargo run --example hardsec_evidence_packet_audit --features hardsec-evidence-audit -- \
  --packet /path/to/evidence-packet.json \
  --evidence-root /path/to/extracted-source-package \
  --receipt /path/outside/packet/byte-verified-audit.json
```

Expected mathematical result for the example: at p=10^-4, 1,000 independent trials have about 90.48% chance of zero activations; approximately 29,956 IID trials are required to reduce that miss probability to 5%.

The trigger-budget suite has **9 tests**. The evidence-packet auditor has **26 tests**, including the SeqCube witness-only fixture and negative controls for claim inflation. Both suites have now been compiled and passed in the focused GitHub Actions job described below.

### Isolated focused Rust test harness

The root Cargo workspace currently contains a globbed member path `crates/core/symthaea` whose `Cargo.toml` is absent. A direct root-workspace Cargo invocation fails before reading the examples. The project does not alter or weaken that production workspace declaration for this research task.

Instead, `tools/hardsec-audit-harness/Cargo.toml` defines a small nested workspace that points directly at the two checked-in example files and pins their direct dependencies (`serde_json = 1.0.150`, optional `sha2 = 0.10.9`). The focused workflow runs:

```bash
cargo generate-lockfile --manifest-path tools/hardsec-audit-harness/Cargo.toml
cargo test --locked --manifest-path tools/hardsec-audit-harness/Cargo.toml --example hardsec_trigger_budget
cargo test --locked --manifest-path tools/hardsec-audit-harness/Cargo.toml --features hardsec-evidence-audit --example hardsec_evidence_packet_audit
```

On exact PR head `d8b58a7b27ea38786c0e7b08f45853f76188030a`, GitHub Actions run [41035491438](https://github.com/Luminous-Dynamics/symthaea/actions/runs/41035491438) completed successfully: **rustfmt check passed**, **9/9** trigger-budget tests passed, and **26/26** evidence-packet tests passed. The previous focused compile/test run [41034832167](https://github.com/Luminous-Dynamics/symthaea/actions/runs/41034832167) also passed both suites before formatting was made a required step.

The generated harness lockfile SHA-256 was `004dbdd71ac12e7446763de65ef083eb6f0a7888ed64cbbe3c3e107d0ce01c82`; it was uploaded as artifact `hardsec-audit-harness-lock` (artifact ID `12967771808`, archive digest `sha256:b849390c04deb6c94278bf53d4b34d67f1779d377799893b3a4f67f713078262`). The source is the same tracked examples referenced in the repository; the lockfile is an exact-run artifact, not yet committed.

The independent SeqCube package verifier remains **not run** here. The passing Rust test validates only the evidence packet's handling of publisher-listed digests and claim scopes; it does not fetch the package's files or reproduce Z3/Icarus outputs.

Useful tool references:
- SymbiYosys: https://symbiyosys.readthedocs.io/en/latest/ (bounded/unbounded safety, cover and liveness)
- EQY equivalence tutorial: https://formal.org/tutorials/symbiyosys/equivalence-checking (gold/gate comparison, partition proofs and counterexamples)
- SeqCube-HT (2026): https://doi.org/10.1186/s42400-026-00676-2 (state-cube-guided sequential Trojan benchmark methodology)

## 14. Scope boundary

This is defensive benchmark and evaluation infrastructure. It does not construct or deploy Trojan payloads, claim a Trojan is absent because finite simulation was quiet, or certify silicon. Formal results must preserve assumptions, bounds and solver provenance. The public contest repository's code/checkpoint/data must not be redistributed until a license or explicit permission is established.

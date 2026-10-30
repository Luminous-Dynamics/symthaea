# HARDSEC-005 — Lineage-Aware Benchmark Integrity for Hardware-Trojan Detection

**Status:** research protocol / pre-registration draft  
**Scope:** benchmark integrity and design-level generalization; no detector, Trust-Hub dataset, or benchmark artifact has been run locally  
**Related work:** [HARDSEC-001 runtime corpus and split audit](https://github.com/Luminous-Dynamics/symthaea/pull/8992), [HARDSEC-002 secure generation](https://github.com/Luminous-Dynamics/symthaea/pull/9040), [HARDSEC-003 HARPY-V importer](https://github.com/Luminous-Dynamics/symthaea/issues/9045), [HARDSEC-004 operational telemetry evaluation](https://github.com/Luminous-Dynamics/symthaea/pull/9063), [HARDSEC-005 issue](https://github.com/Luminous-Dynamics/symthaea/issues/9089)

## 1. Research question

Can hardware-Trojan and design-level hardware-security detectors be evaluated without leakage from sibling netlists, shared host circuits, repeated logic cones, synthesis flows, or benchmark transformations—and can their claimed transfer be measured at the right hierarchy?

This workstream is separate from runtime telemetry. Gate-level labels, RTL properties, hardware performance-counter windows and attack episodes are different sample units and must not share a metric or leaderboard.

## 2. Published evidence: the test boundary can dominate the score

Xiao et al., “Protocol effects on feature-based hardware-Trojan detection across Trust-Hub families” (accepted at IEEE ICCSIE 2026; arXiv:2609.07199v2, 9 October 2026), analyze 49,124 gates from 16 netlists grouped into five base-circuit families. The study holds its parser, 36 gate features, class weighting, model settings, threshold and family-level aggregation fixed while changing the test boundary:

| Evaluation boundary | Random Forest F1 | Average precision |
|---|---:|---:|
| Pooled gate split | 0.914 | 0.978 |
| Leave-one-netlist-out | 0.636 | 0.851 |
| Leave-one-host-family-out | 0.460 | 0.577 |

XGBoost F1/AP changes from 0.946/0.976 pooled to 0.464/0.544 under host-family holdout. The authors report that the direction persists with feature ablations, repeated model and simulator seeds, parser-related exclusions and reduced sample size.

The authors also set an important limit: five base-circuit families are too few for a population claim; family folds reuse training families; and the study does not establish transfer to new cell libraries or industrial designs. This is evidence that sibling benchmark variants can inflate apparent transfer—not proof that every Trust-Hub experiment is invalid or that every detector has the same performance.

Sources:
- arXiv: https://arxiv.org/abs/2609.07199 (v2)
- DOI: https://doi.org/10.48550/arXiv.2609.07199
- Zenodo publication record: https://doi.org/10.5281/zenodo.23230179

**Reproducibility note:** the linked Zenodo record currently lists the 403.5 kB conference PDF. It does not expose the underlying netlists or executable code as files on that record. The paper's “reproducibility artifact” reference must therefore be resolved to an actual artifact URI and digest before claiming that the experiment has been locally reproduced.

### Newer candidate corpus: 28 October 2026

A newer versioned corpus is now the priority candidate for dataset-specific reproduction: [Mendeley Data Version 1, “Shift-controlled gate-level hardware Trojan corpus, per-fold detector scores and synthesized netlists for robust, resource-aware AI-based Trojan detection”](https://data.mendeley.com/datasets/37fdy2k7cj/1), DOI [10.17632/37fdy2k7cj.1](https://doi.org/10.17632/37fdy2k7cj.1), published 28 October 2026.

Its indexed catalogue description states 283 synthesized gate-level circuits, 232,967 gates and 3,373 Trojan gates; 25 open host designs in four families (ISCAS-85, EPFL arithmetic, EPFL control and sequential IP); four inserted Trojan families (combinational XOR, leakage, denial of service and sequential counter); a second synthesis library; real Trust-Hub RTL Trojans in two hosts; and per-fold detector scores plus synthesized netlists.

This appears to provide a more useful in-corpus test bed for host-design, Trojan-family and synthesis-library shift than the older 16-netlist / five-host-family example alone. It still has only four broad host-design families, so the effective number of independent groups must be reported honestly.

**Qualification caveat:** only the indexed catalogue description has been inspected so far. The exact archive/file inventory, license/redistribution terms, archive SHA-256, associated article DOI, executable code, author split files and per-fold predictions have not yet been retrieved or independently verified. Do not call the corpus audited or reproduce its results until those artifacts are pinned. Dataset-specific intake and reproduction is tracked in [HARDSEC-006 / issue #9100](https://github.com/Luminous-Dynamics/symthaea/issues/9100).

### Insertion-source shift is a separate holdout axis

Xu et al.'s SeqCube-HT paper (published 9 October 2026, DOI 10.1186/s42400-026-00676-2) uses fixed graph probes on four shared ISCAS'89 hosts and excludes instance/source names and insertion-method identifiers from model inputs. For SeqCube-generated instances, graph F1 / localization recall are 69.2% / 55.7% on an instance-random split, but 47.8% / 31.4% when SeqCube is the held-out insertion source and models train on Random, TRIT-TS and CG sources. Covariate matching raises SeqCube to 77.2% / 66.1%, yet it remains 9.0 F1 points and 12.3 localization-recall points below the closest reference source on that fixed four-circuit corpus. Fixed-threshold ISCAS-to-Extended-IP transfer is 56.8% F1 / 40.7% localization recall. The authors caution that fixed-corpus seed intervals are not uncertainty over unseen circuit populations and that these results do not establish physical realism or intrinsic stealth.

These findings motivate benchmark-source holdout as a separate axis from host-family, Trojan-family, toolchain and library holdouts: a model can omit the explicit source ID and still learn source-specific structural artifacts. Reference: https://doi.org/10.1186/s42400-026-00676-2.

## 3. Paradigm shift: the independent unit is design lineage

A benchmark can contain tens of thousands of gates but only a handful of independent host designs. Sibling Trojan variants can share most of their benign logic. Randomly assigning gates to train and test can evaluate recognition of a known host circuit rather than detection on a new design.

Separate these claims:

1. **Within-host interpolation:** diagnostic only; not unseen-design generalization.
2. **Unseen-netlist transfer:** new netlist, possibly with a related host design represented in training.
3. **Unseen-base-family transfer:** all variants of one host family excluded from training.
4. **Unseen-Trojan-family transfer:** attack insertion family or mechanism excluded from training.
5. **Unseen synthesis/toolchain/library transfer:** only when independent tools, libraries or PDKs are present and source lineage is known.
6. **Combined host × Trojan holdout:** new host family and new inserted attack family; the most demanding in-corpus test, requiring enough independent groups.

No single split estimates all six.

## 4. Required immutable design-lineage inventory

For every original RTL, netlist or gate graph, record:

- dataset, DOI/release/archive URI, license statement and exact source digests for the original RTL, evaluated netlist and derived graph as distinct artifacts;
- base host design and family identifier;
- benign/Trojan status, Trojan family/primitive, insertion revision/transform and source lineage;
- RTL source digest and generated netlist digest;
- synthesis tool/version, scripts, defines, constraints and command/configuration digests;
- cell library, technology and PDK identity/version where known and legally recordable;
- parser, elaboration and feature-extractor revision;
- design-level labels and the origin/evidence for each label.

Unknown fields remain unknown, not inferred from filenames. Do not claim cross-library or cross-technology generalization when only one library is represented. Keep original artifact hashes distinct from canonicalized or normalized graph fingerprints.

## 5. Hierarchy-aware leakage audit

Build a deterministic pre-training audit that:

- detects exact duplicate RTL/netlist artifacts by cryptographic digest;
- groups known sibling variants under the same base-circuit lineage;
- identifies shared source ancestors and repeated generated artifacts;
- optionally computes canonicalized structural fingerprints and graph-similarity candidates;
- emits suspected near-overlaps as review findings, not as unquestionable proof of leakage;
- rejects a split when a declared holdout group crosses partitions;
- records the canonicalizer algorithm/version, parameters and output digest.

Exact digest equality establishes byte identity. Canonicalized graph similarity is only a screening signal: synthesis, elaboration, renaming and optimization can change structure, while unrelated designs can share common subgraphs. Never silently merge designs based on an unvalidated similarity threshold.

## 6. Frozen split ladder

| Split | What it tests | Claim ceiling |
|---|---|---|
| Random gate split | Within-corpus interpolation | Diagnostic only |
| Leave-one-netlist-out (LOCO) | New netlist; sibling host may be known | Transfer within represented hosts |
| Leave-one-base-family-out (LOFO) | Entire host family unseen | Family transfer within represented corpus |
| Leave-one-Trojan-family-out | New attack insertion mechanism | Trojan-family transfer, not arbitrary zero-day coverage |
| Leave-one-benchmark-source-out | New insertion/generation cohort, including paired controls | Transfer across represented sources, not zero-day coverage |
| Leave-toolchain/library-out | Synthesis or library domain shift | Only when independent groups exist |
| Combined source × host / Trojan holdout | New source plus new host or attack family | Hardest in-corpus transfer test |

No sibling variant or derived gate can cross a family holdout. Report every family and its support count. When there are only five host families, results are descriptive; thousands of gates do not turn five designs into thousands of independent trials.

For source holdouts, distinguish the source cohort from the insertion method: benchmark_source_id groups the full cohort, including its paired clean control; insertion_method_id records the technique/version that made the positive variant. The cohort ID is metadata used to assign folds, not a model feature. The source-held-out split keeps every paired positive and clean control in the same held-out cohort.

A source-held-out protocol may intentionally reuse an identical benign host netlist across source cohorts. The auditor counts and lists those benign byte-identical controls instead of silently rejecting or ignoring them. Identical Trojan-positive artifacts across partitions and any identical-content label disagreement remain hard failures. Report a separate host-disjoint result where possible, because reused clean controls make this a source-shift test, not a host-generalization test.

## 7. Evaluation metrics and aggregation

Report:
- gate-weighted metrics for comparability with prior literature;
- host-family macro metrics (one family, one vote), per-netlist outcomes and worst-family performance; and
- benchmark-source macro metrics, per-source outcomes and worst-source performance, with independent source-group support displayed.

At minimum include precision, recall, F1, average precision / PR-AUC, ROC-AUC where meaningful, confusion matrix, operating-point false-positive burden, and calibration/selective risk where confidence scores exist. State prevalence and weighting assumptions, group counts and all family supports.

Separate Trojan detection from localization, type attribution and weakness identification. A detector must not receive host-family identifiers, filenames or labels that encode the answer unless those are part of an explicitly defined deployment threat model and are separately ablated.

Likewise, benchmark source IDs and insertion-method IDs are split/provenance metadata, not detector inputs unless the deployment threat model explicitly makes them available. Audit the feature list and use a source-ID leakage negative control. Compare instance-random, source-held-out and covariate-matched source-held-out results under the same training budget and frozen threshold; source-held-out performance alone is not zero-day generalization.

## 8. NIST-aligned security coverage is a separate axis

NIST CSWP 45 (2025) presents a hardware-security construct methodology with a threat metric counting hardware weaknesses an attack can exploit and a sensitivity metric counting distinct attacks that can target a particular weakness.

Track these as a separate design-coverage layer:
- weakness/CWE coverage by attack/test family;
- attack-pattern coverage by hardware weakness;
- unsupported or untested weakness families;
- mapping confidence and evidence provenance.

Do not add taxonomy/coverage quantities to F1 or call them probabilities of exploit. They answer a different question: which parts of the security surface have actually been exercised?

Source: https://doi.org/10.6028/NIST.CSWP.45

## 9. Negative controls and mutation tests

- Force sibling netlist/base-family leakage and verify the auditor blocks a claimed LOFO evaluation.
- Copy identical RTL/netlist bytes under another filename and verify digest detection.
- Rename/reorder signals and cells while preserving the underlying design; characterize canonicalizer stability.
- Inject controlled label and group-ID corruption.
- Permute labels at the correct grouping level; suspicious above-chance scores block interpretation.
- Remove Trojan logic or alter the gold label in controlled fixtures and verify the evaluation oracle can detect the mutation.
- Report oracle adequacy: a test that cannot distinguish a known Trojan mutant from its benign host must not certify a detector as secure.

Every control has an explicit expected failure/success, an independent verifier and a receipt tied to exact source/configuration digests.

## 10. Research comparisons

First reproduce the paper's fixed 36-feature RF / XGBoost / logistic-regression conditions and three test boundaries as closely as the actual public reproducibility artifact permits. Record exact deviations and hashes. Do not assume five host families support strong population inference.

A second direct comparator is Mukherjee and Mahanta's September 2026 paper, “A unified graph neural network pipeline for hardware Trojan detection” (DOI 10.1016/j.vlsi.2026.102775). It reports a four-feature, two-layer GCN over 55 synthesized designs from the AES-T100–T2800 family and PIC16F84-T200, with 87.27% accuracy and 0.9563 ROC-AUC, plus saliency-guided localization evaluated with node-level IoU. This is a useful graph-level baseline and localization comparator. It is **not directly comparable** to the Trust-Hub RF F1/AP figures above: dataset composition, synthesis construction, aggregation, labels and split protocol differ. The accessible article summary does not establish a strict host-family-held-out split. Reproduce it only after pinning its exact source designs, synthesis scripts, data split and code; label it unreproduced until then.

Then compare:
- handcrafted gate/netlist features with RF, XGBoost and logistic regression;
- the published compact two-layer GCN and saliency/localization pipeline;
- larger message-passing or graph-transformer models only after leakage-resistant splits are frozen;
- design-family-invariant representations;
- Symthaea HDC representations as an additional candidate, not a presumed winner.

Use identical splits, features where possible, matched tuning budgets, and family-level metrics. A new model earns credit only by improving leakage-resistant transfer—not by restoring pooled-split performance or winning on a different test partition.

## 11. Counterfactual paired-design experiment: benign change versus Trojan change

**New falsifiable hypothesis:** where a benchmark provides a clean host design and matched Trojan-inserted variants, a representation of the *difference* between a host and candidate variant may separate Trojan-associated logic changes from host identity better than absolute netlist features alone.

This is a benchmark-only paired experiment. It deliberately uses the clean host as a reference, so it is separate from golden-model-free detection and must not be presented as a deployment assumption.

### Build paired controls

For each eligible host family:

1. Freeze a clean host RTL/netlist, Trojan-inserted variant(s), synthesis tool/version, constraints, defines, cell library and command. Pair only variants with defensible common ancestry and reproducible comparable synthesis.
2. Generate benign functionally equivalent variants of the clean host using predeclared transformations (e.g. identifier renaming, equivalent logic-optimization recipes, synthesis seeds, or supported buffering transformations). Verify functional equivalence with an independent equivalence checker where available. A transformation is not benign merely because it was intended to be.
3. Generate or obtain matched Trojan variants under the same synthesis conditions. Preserve exact insertion transform and affected-cone annotations if supplied by the dataset.
4. Align graphs using stable source/net/cell lineage where available. If only heuristic graph matching is possible, report mapping precision/recall on fixtures and propagate uncertain mappings; do not claim exact gate ancestry.
5. If source data has no defensible clean/Trojan pairing or stable mapping, mark this arm unsupported rather than inventing pairs.

### Compare three representations

- **Absolute graph:** classify a design from its netlist graph alone (reference-free comparator).
- **Paired delta graph:** encode matched host/variant changes—added/removed/re-wired nodes/edges, cell-type changes and affected fan-in/fan-out cones—while retaining host identity as a controlled covariate/ablation.
- **Delta plus context:** represent changed subgraphs with bounded surrounding context to test whether localization needs neighboring benign circuitry.

The model must not infer a Trojan merely because a large transformation occurred. Benign-equivalent variants are hard negatives. Keep any source-provided Trojan annotations sealed from model input.

### Evaluate detection and localization separately

- Pairwise discrimination: rank a Trojan variant above the benign host and above benign-equivalent variants under the same host.
- Family transfer: hold out the complete host family and, separately, the Trojan insertion family.
- Benign-transformation false-positive rate: fraction of verified benign transformations flagged at the frozen operating point.
- Trojan sensitivity at a matched benign-transformation alert budget.
- Localization IoU / precision / recall on mapped affected cones, with mapping-confidence strata.
- Functional equivalence pass rate and synthesis-flow consistency, reported separately from detection quality.
- Resource cost of alignment and inference; report absolute and delta pipelines end-to-end.

A useful paired contrast is the detector-score change for a Trojan-inserted variant versus the score change caused by a benign, equivalence-checked transformation of the same host. The benign control estimates how much score movement comes from design-flow variation alone. Freeze the pairing, matching and scoring policy before testing.

### Failure conditions

The paired experiment is invalid where clean and Trojan designs were synthesized with incompatible toolchains/constraints without controlled normalization, where host ancestry is uncertain, or where graph-matching quality is unmeasured. If a Trojan is functionally or structurally subtle and does not produce a reliably alignable delta, the model may not be capable of detecting it; report that failure rather than changing the pairing after observing scores.

## 12. Proposed success gates

These are research gates, not published standards:

1. All known source-lineage leakage in the frozen split is detected or explicitly dispositioned before model scoring.
2. Random-split and family-split results are both published so the optimism gap is visible.
3. Model comparisons are paired on identical held-out families; uncertainty is based on independent designs/families, not gate count.
4. A candidate improves the predeclared family-macro metric or reduces false positives at fixed family-macro recall, without worsening the worst-family result beyond a predeclared margin.
5. Five-family studies remain descriptive. Confirmatory claims require more independent base designs/families and, for technology-transfer claims, independent libraries/toolchains.
6. No industrial-generalization claim is made from Trust-Hub alone.

Set numeric effect sizes and minimum independent family counts only after inventory and precision/power analysis, before examining confirmatory model results.

## 13. Industry and standards alignment

NIST IR 8615 (September 2026) calls for interoperable, evidence-based trust models, provenance and traceability across the semiconductor lifecycle, supply-chain assurance, scalable verification/validation and collaboration. This work operationalizes a narrow but important part of that direction: traceable design lineage and honest transfer claims.

OpenTitan is a useful implementation-practice comparator. Its security model covers attestation, provisioning, secure boot, lifecycle, firmware update, identity and ownership transfer. Its security documentation also describes secure hardware design guidance and penetration-testing support for side-channel analysis and fault injection on FPGA and silicon.

This protocol is not a certification framework and does not replace formal verification, design-for-test/security review, silicon validation, side-channel testing, foundry/PDK controls or secure lifecycle procedures.

References:
- NIST IR 8615 (2026): https://doi.org/10.6028/NIST.IR.8615
- OpenTitan security documentation: https://opentitan.org/book/doc/security/
- NIST CSWP 45: https://doi.org/10.6028/NIST.CSWP.45

## 14. Acceptance criteria

- [ ] Source inventory and licensing notes frozen.
- [ ] Design-lineage schema defined and validated on fixtures.
- [ ] Exact duplicate and shared-host lineage audit implemented.
- [ ] Optional structural similarity detector has candidate-only semantics and known-answer fixtures.
- [ ] Group-aware split validator rejects sibling-family leakage.
- [ ] Gate-weighted and family-macro report formats generated with support counts.
- [ ] Negative-control and oracle-adequacy fixtures pass independently.
- [ ] Baseline results reproduced with exact code/data/model lineage or marked irreproducible.
- [ ] If matched clean/Trojan pairs exist, benign-transform controls and paired delta-graph comparisons are evaluated; otherwise this arm is marked unsupported.
- [ ] Source-held-out evaluation is reported where source groups exist, with source IDs excluded from model inputs.
- [ ] Identical benign paired-control reuse is enumerated; identical Trojan-positive reuse and content-label conflicts fail closed.
- [ ] No unqualified zero-day or industrial-transfer claim.

## 15. Initial Rust audit implementation

The branch now includes an opt-in generic manifest auditor:

- `examples/hardsec_design_lineage_audit.rs`
- Cargo feature: `hardsec-design-audit`
- Example target: `hardsec_design_lineage_audit`

Run its authored fixtures with:

```bash
cargo test --example hardsec_design_lineage_audit --features hardsec-design-audit
```

Run an audit after producing a manifest:

```bash
cargo run --example hardsec_design_lineage_audit --features hardsec-design-audit -- \
  --manifest /path/to/design-samples.tsv \
  --root /path/to/pinned-dataset \
  --receipt /path/outside/dataset/design-lineage-receipt.json
```

**Important source-path contract:** in this manifest, `source_path` and `source_sha256` identify the exact artifact consumed by the detector (for a gate-level graph detector, normally the synthesized netlist file), not a shared clean RTL ancestor used to derive multiple variants. A single evaluated netlist may be referenced by many gate/sample rows, but its complete sample group must remain in one split. Upstream RTL, parent commit and synthesis inputs belong in the separate design-lineage inventory and retain their own hashes; they must not be substituted for the evaluated-artifact digest. This distinction prevents the exact-duplicate guard from mistaking legitimate shared ancestry for reused classifier input.

The schema is strict (15 tab-separated fields, exact order): `dataset_id, source_artifact_id, source_path, source_sha256, sample_id, design_id, base_design_family_id, netlist_id, trojan_family_id, synthesis_config_id, cell_library_id, benchmark_source_id, split_protocol, split, label`. The new `benchmark_source_id` is required for `leave_benchmark_source_out`; it identifies the full source cohort, including paired clean controls. It is grouping metadata, not a detector feature.

The auditor supports `random_gate` (diagnostic only), `loco_netlist`, `lofo_base_family`, `leave_trojan_family_out`, `leave_toolchain_out`, `leave_library_out`, and `leave_benchmark_source_out`. It verifies exact evaluated-artifact bytes against SHA-256, rejects unsafe paths/symlinks and duplicate sample IDs, and enforces the selected group holdout. For `leave_benchmark_source_out`, source cohort IDs must be present and cannot cross splits. Identical benign evaluated artifacts may be reused as paired controls across source cohorts, but every reused digest, sample ID and source cohort is enumerated in the receipt. Identical Trojan-positive artifacts across partitions and any identical-content label disagreement are rejected. For other non-random protocols, exact cross-split artifact reuse is rejected. For random-gate splits, exact-content and declared-lineage overlaps are reported rather than rejected.

The JSON receipt is deterministic and unsigned. Its claim scope is source-file integrity plus the declared group-split check only. The lineage group identifiers are caller-supplied; the tool does not independently infer design ancestry. It does not parse Verilog, perform canonical graph comparison, detect non-identical near-duplicate logic, prove that the metadata is truthful, or evaluate a Trojan detector.

Nineteen unit tests are authored for valid manifests; base-family/netlist/Trojan-family and benchmark-source leakage; required source metadata; exact Trojan-positive artifact reuse; benign paired-control reuse/reporting; content-label disagreement; source tampering; path traversal; symlink rejection; duplicate sample IDs; missing toolchain identity; receipt overwrite/path safety; and diagnostic random-gate overlap. They have not yet been compiled or run on this branch.

## 16. Scope boundary

This work evaluates benchmark integrity and generalization of design-level hardware-Trojan detection. It does not modify hardware, create Trojan payloads or certify a design. It is not merged with RISC-V runtime telemetry metrics in HARDSEC-001/004 or secure generation in HARDSEC-002.

The generic manifest auditor now supports source-held-out split integrity, but dataset-specific importers, canonical structural-overlap screening and independent verification of lineage metadata still require pinned source releases and their actual reproducibility artifacts. The source-holdout protocol is tracked in [HARDSEC-009 / issue #9244](https://github.com/Luminous-Dynamics/symthaea/issues/9244).
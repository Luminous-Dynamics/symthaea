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

- dataset, DOI/release/archive URI, license statement and exact source digest;
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
| Leave-toolchain/library-out | Synthesis or library domain shift | Only when independent groups exist |
| Combined host × Trojan holdout | New host and attack family | Hardest in-corpus transfer test |

No sibling variant or derived gate can cross a family holdout. Report every family and its support count. When there are only five host families, results are descriptive; thousands of gates do not turn five designs into thousands of independent trials.

## 7. Evaluation metrics and aggregation

Report both:
- gate-weighted metrics for comparability with prior literature; and
- host-family macro metrics (one family, one vote), per-netlist outcomes and worst-family performance.

At minimum include precision, recall, F1, average precision / PR-AUC, ROC-AUC where meaningful, confusion matrix, operating-point false-positive burden, and calibration/selective risk where confidence scores exist. State prevalence and weighting assumptions, group counts and all family supports.

Separate Trojan detection from localization, type attribution and weakness identification. A detector must not receive host-family identifiers, filenames or labels that encode the answer unless those are part of an explicitly defined deployment threat model and are separately ablated.

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

Then compare:
- handcrafted gate/netlist features with RF, XGBoost and logistic regression;
- graph models / message-passing GNNs, only after leakage-resistant splits are frozen;
- design-family-invariant representations;
- Symthaea HDC representations as an additional candidate, not a presumed winner.

Use identical splits, features where possible, matched tuning budgets, and family-level metrics. A new model earns credit only by improving leakage-resistant transfer—not by restoring pooled-split performance.

## 11. Proposed success gates

These are research gates, not published standards:

1. All known source-lineage leakage in the frozen split is detected or explicitly dispositioned before model scoring.
2. Random-split and family-split results are both published so the optimism gap is visible.
3. Model comparisons are paired on identical held-out families; uncertainty is based on independent designs/families, not gate count.
4. A candidate improves the predeclared family-macro metric or reduces false positives at fixed family-macro recall, without worsening the worst-family result beyond a predeclared margin.
5. Five-family studies remain descriptive. Confirmatory claims require more independent base designs/families and, for technology-transfer claims, independent libraries/toolchains.
6. No industrial-generalization claim is made from Trust-Hub alone.

Set numeric effect sizes and minimum independent family counts only after inventory and precision/power analysis, before examining confirmatory model results.

## 12. Industry and standards alignment

NIST IR 8615 (September 2026) calls for interoperable, evidence-based trust models, provenance and traceability across the semiconductor lifecycle, supply-chain assurance, scalable verification/validation and collaboration. This work operationalizes a narrow but important part of that direction: traceable design lineage and honest transfer claims.

OpenTitan is a useful implementation-practice comparator. Its security model covers attestation, provisioning, secure boot, lifecycle, firmware update, identity and ownership transfer. Its security documentation also describes secure hardware design guidance and penetration-testing support for side-channel analysis and fault injection on FPGA and silicon.

This protocol is not a certification framework and does not replace formal verification, design-for-test/security review, silicon validation, side-channel testing, foundry/PDK controls or secure lifecycle procedures.

References:
- NIST IR 8615 (2026): https://doi.org/10.6028/NIST.IR.8615
- OpenTitan security documentation: https://opentitan.org/book/doc/security/
- NIST CSWP 45: https://doi.org/10.6028/NIST.CSWP.45

## 13. Acceptance criteria

- [ ] Source inventory and licensing notes frozen.
- [ ] Design-lineage schema defined and validated on fixtures.
- [ ] Exact duplicate and shared-host lineage audit implemented.
- [ ] Optional structural similarity detector has candidate-only semantics and known-answer fixtures.
- [ ] Group-aware split validator rejects sibling-family leakage.
- [ ] Gate-weighted and family-macro report formats generated with support counts.
- [ ] Negative-control and oracle-adequacy fixtures pass independently.
- [ ] Baseline results reproduced with exact code/data/model lineage or marked irreproducible.
- [ ] No unqualified zero-day or industrial-transfer claim.

## 14. Scope boundary

This work evaluates benchmark integrity and generalization of design-level hardware-Trojan detection. It does not modify hardware, create Trojan payloads or certify a design. It is not merged with RISC-V runtime telemetry metrics in HARDSEC-001/004 or secure generation in HARDSEC-002.

Implementation follows only after obtaining and pinning the exact Trust-Hub release and its actual executable reproducibility artifacts.
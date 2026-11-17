# SCI-RUBRIC-COMMENSURABILITY-001 — Construct and measurement comparability across rubrics

**Status:** structural plan verifier and commensurability auditor, ontology commitment CLI, strict plan/report schemas, deterministic synthetic fixture and hostile tests under development. No empirical human calibration or construct-validity result is claimed.

**Tracking:** [Issue #9320](https://github.com/Luminous-Dynamics/symthaea/issues/9320).  
**Stack:** after SCI-FACTORIAL-JUDGE-BIAS-001 / [PR #9243](https://github.com/Luminous-Dynamics/symthaea/pull/9243).

## 1. Why commensurability must precede score comparison

Two rubric scores are not directly comparable merely because they share a 0–100 scale, similar criterion names, or similar model rankings. A rubric is a measurement instrument. Its construct scope, criteria, thresholds, direction, ordinal anchors, weights and evidence requirements define what the number means.

The EMNLP 2026 paper *Beyond the Illusion of Consensus: From Surface Heuristics to Knowledge-Grounded Evaluation in LLM-as-a-Judge* identifies a Rubric Commensurability Problem. In its fully crossed study, moving from a shared five-dimension rubric package to independently generated rubrics reduced agreement by 62% for the two top-scoring models. Its reported Resolution Paradox shows that model-level rankings can remain nearly identical while criterion-level agreement falls under non-equivalent metrics and samples. The paper cautions that lower agreement under independent rubrics is not itself proof of higher accuracy.

Source: https://aclanthology.org/2026.emnlp-main.1552/

PReMISE (EMNLP 2026) treats policy rubrics as measurement specifications and evaluates structural adequacy, reliability, preference fit and adversarial robustness. Its experiments report that no raw rubric source simultaneously achieved reliability, preference prediction and exploit resistance in its tested setting.

Source: https://aclanthology.org/2026.emnlp-main.2653/

The ICML 2026 paper *Quantifying Biases in LLM-as-a-Judge Evaluations* develops a Bayesian generalized linear model framework for judge bias and uncertainty. This reinforces a key boundary: agreement metrics alone do not explain why raters or instruments differ.

Source: https://proceedings.mlr.press/v306/dubois26a.html

## 2. Paradigm shift

Before pooling scores or comparing model rankings, first determine which parts of the instruments are comparable.

The mapping ontology intentionally distinguishes:
- **Equivalent:** both rubric criteria operationalize the same declared construct and scope.
- **Narrower / broader:** one criterion measures a strict subset or superset of the other.
- **Overlapping:** criteria share some content but neither is a substitute for the other.
- **Orthogonal:** they measure distinct dimensions.
- **Incompatible:** the criteria cannot validly be treated as the same construct.
- **Unmapped:** no defensible construct mapping has been declared.

A common display label or text similarity never establishes equivalence. The v1 analyzer consumes explicit mapping decisions and evidence commitments; it does not infer semantic equivalence from words.

## 3. Executable v1

Files:
- `schema/rubric-commensurability-plan.schema.json`
- `schema/rubric-commensurability-report.schema.json`
- `src/rubric_commensurability.rs`
- `tools/rubric_commensurability_fixture.py`

CLI:
```sh
python3 research/science-bench-refresh/tools/rubric_commensurability_fixture.py /tmp/rubric-commensurability
cargo run --locked --quiet --manifest-path research/science-bench-refresh/Cargo.toml -- \
  verify-rubric-commensurability-plan /tmp/rubric-commensurability/rubric-commensurability-plan.example.json
cargo run --locked --quiet --manifest-path research/science-bench-refresh/Cargo.toml -- \
  audit-rubric-commensurability /tmp/rubric-commensurability/rubric-commensurability-plan.example.json
```

The plan freezes:
- a construct ontology with definition and scope digests;
- left and right rubric identities, revisions and exact rubric digests;
- each criterion's definition and measurement-specification digest;
- raw scale endpoints and score direction;
- normalized semantic anchor digests and values;
- criterion weights that sum exactly to one million per instrument;
- typed criterion-to-construct mappings with evidence and adjudicator identity commitments;
- shared anchor-item identities and semantic digests;
- a preregistered minimum shared-anchor count per construct;
- the exact analyzer source revision and clean-worktree requirement for qualified runs.

The construct-ontology commitment is computed from the actual construct rows, sorted by stable construct ID, using a domain-separated digest. It is not an arbitrary label or a free-form hash supplied by the caller.

### Direct comparison requirements

A construct is marked directly commensurable only when:
1. both instruments map at least one criterion to that construct;
2. every mapping on both sides is explicitly `equivalent`;
3. measurement-specification digest sets match;
4. normalized semantic scale-anchor maps match, including score direction;
5. total weight assigned to the construct matches;
6. shared anchor-item coverage reaches the preregistered minimum;
7. mapping-adjudicator identity commitments are disjoint across the instruments.

If equivalent construct labels have different measurement specifications, scale anchors, weights or insufficient anchors, the result is **calibration required**, not direct equivalence. Narrower, broader and overlapping mappings remain partial. Orthogonal and incompatible mappings block direct comparison. Unmapped constructs or insufficient anchor coverage prevent an overall fully commensurable result.

The analyzer reports the reason for every construct-level status. It does not silently convert partial, incompatible or missing measurements into a common score.

### Mapping independence boundary

Different adjudicator identity digests are a necessary structural check for direct comparison in this v1 protocol. They do **not** prove that the reviewers are different humans, independent in fact, free of shared bias, or blinded. The report therefore always sets `mapping_panel_independence_proven=false`. Mapping evidence digests bind declared evidence references, but do not authenticate the underlying records.

### Scale semantics

Each criterion's scale declares:
- raw minimum and maximum;
- whether higher or lower raw values are better;
- monotonically ordered normalized anchors from 0 to 1,000,000;
- stable semantic anchor digests (e.g. poor, adequate, excellent).

The analyzer rejects non-monotonic, incomplete, duplicated or out-of-range scales. Direct scale matching compares semantic anchor identities and normalized values, not raw numeric endpoints. This allows semantically equivalent 0–4 and 0–100 instruments to be recognized as structurally compatible when their normalized anchors genuinely match.

## 4. Statuses and claim limits

The overall status is one of:
- `fully_commensurable`;
- `partially_commensurable`;
- `calibration_required`;
- `non_commensurable`;
- `insufficient_evidence`.

Direct score and model-ranking comparisons are permitted only for a fully commensurable result with no unmapped criteria. That permission means the declared structural prerequisites are met; it does not mean the instruments are valid or that their observed scores are calibrated.

The report permanently sets:
- `construct_validity_proven=false`;
- `human_alignment_proven=false`;
- `generalized_accuracy_estimated=false`;
- `scoring_eligible_declared=false`;
- `external_anchor_required=true`;
- `mapping_panel_independence_proven=false`.

A structural mapping audit cannot prove the construct ontology is correct, that human raters interpret criteria as intended, or that two scales have equal empirical item-response functions.

## 5. Qualification targets and SOTA comparison

### Engineering targets

- 100% rejection of duplicate/unknown constructs and criteria;
- 100% rejection of invalid, zero-weight or non-unit-sum criteria weights;
- 100% rejection of malformed, nonmonotonic or incomplete score scales;
- 100% rejection of missing mapping evidence, missing adjudicator identity commitments and digest tampering;
- 100% detection of incompatible and unmapped construct controls;
- no direct score comparison if any construct fails direct commensurability;
- no direct comparison when mapping adjudicator identity commitments overlap;
- exact and deterministic plan/report commitments.

These are schema/validator qualification targets, not empirical measurement-validity performance.

### Empirical research required next

The next version should add a fully crossed rubric × model × task × human-rater design:
1. independently define rubrics for the same task domain;
2. independently adjudicate criterion-to-construct mappings;
3. use shared anchor items and blinded common raters;
4. separately estimate criterion-level and model-level agreement;
5. fit multi-group IRT or hierarchical ordinal models where data support identifiability;
6. test measurement invariance before comparing latent scores or rankings;
7. report uncertainty and sensitivity to alternative construct mappings.

Compare shared-rubric baselines, independently generated rubrics, ontology-aligned rubrics, calibrated rubrics and human adjudication under equal task and rating budgets. Report construct-level disagreement, latent-scale invariance, ranking stability, anchor coverage, calibration error and reviewer cost. Do not treat rank correlation as proof of criterion-level comparability.

## 6. Integration

- **#9124 rubric specification/exploitability:** checks the internal quality and adversarial resistance of an individual instrument.
- **#9131 judge-score geometry:** compares score-space structure only after criterion dimensions are meaningfully aligned.
- **#9152 metamorphic evaluator audit:** checks paired transformations of an evaluator under a frozen instrument.
- **#9237 factorial judge bias:** measures nuisance-factor effects while holding a rubric instrument fixed.
- **#9320 commensurability:** determines whether distinct rubric instruments can be compared at all.

The intended architecture prevents us from pooling incomparable numbers or publishing a model leaderboard whose construct meaning changes between rows.

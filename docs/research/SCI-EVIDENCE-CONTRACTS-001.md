# SCI-EVIDENCE-CONTRACTS-001 — Evaluator-side evidence behavior contracts

**Status:** protocol/schema/fixture proposal. The included fixtures are illustrative and are not scoring-eligible. No Symthaea benchmark result is claimed.

**Tracking:** [Issue #8995](https://github.com/Luminous-Dynamics/symthaea/issues/8995). This layer composes with the counterfactual intervention plan in [PR #8929](https://github.com/Luminous-Dynamics/symthaea/pull/8929), outcome receipts in [PR #8960](https://github.com/Luminous-Dynamics/symthaea/pull/8960), and paired fixed-benchmark analysis in [PR #8993](https://github.com/Luminous-Dynamics/symthaea/pull/8993). It is not a replacement for those artifacts and does not yet integrate its labels into the paired analyzer.

## 1. Paradigm shift: evaluate obligations under interventions

A static answer-quality score asks whether one response looks correct given one context. A behavioral contract asks whether the same frozen policy obeys a precommitted, evaluator-authored obligation when one controlled aspect of the evidence changes. The target is **correctly directed behavior**, not merely output change, response similarity, citation presence, or confidence.

This distinction is motivated by current literature:

- [CounterMem (EMNLP Findings 2026)](https://aclanthology.org/2026.findings-emnlp.1054/) evaluates controlled evidence removal, update, and contradiction; the paper reports that static faithfulness scores miss some evidence-update and conflict failures in its tested settings.
- [Beyond Static Grounding (EMNLP 2026)](https://aclanthology.org/2026.emnlp-main.1494/) formalizes deletion, conflict, distraction, and permutation contracts and studies explicit conflict acknowledgment.
- [Relevant Is Not Warranted / ForceBench (EMNLP Findings 2026)](https://aclanthology.org/2026.findings-emnlp.1577/) holds evidence fixed while increasing claim force. Its reported judge results show that even explicit warrant-strength prompting reduces but does not eliminate monotonicity violations.
- [Evidence-Based Generation (EMNLP Findings 2026)](https://aclanthology.org/2026.findings-emnlp.2222/) distinguishes answer correctness from evidence completeness and reports cases where retrieving a relevant chunk is not enough to retrieve all necessary evidence.
- [NIST AI 800-3](https://www.nist.gov/publications/expanding-ai-evaluation-toolbox-statistical-models) distinguishes accuracy on a fixed benchmark from generalized accuracy and recommends making the target estimand and uncertainty assumptions explicit.

These are results from those papers' evaluation settings, **not Symthaea measurements** and not direct guarantees that one evaluator will transfer to another domain.

## 2. Contract families

Every planned case binds a stable case ID, matched pair ID, intervention type, expected behavior, decision-rule ID, source/evidence-manifest digest, evaluator-gold digest, and a pre-run plan commitment. Raw gold evidence, the authority policy, and the adjudication key must remain outside the policy-visible environment until the run is sealed and scoring is complete.

| Family | Intervention | Required behavior | Important false positive to avoid |
|---|---|---|---|
| Update compliance | Coherent source update | Update dependent claims in the justified direction; keep unrelated claims stable | Treating every textual change as a successful update |
| Conflict disclosure | Incompatible authoritative sources | State the conflict, provenance and unresolved scope rather than silently averaging or cherry-picking | Requiring a conflict warning when sources differ only superficially |
| Decoy invariance | Irrelevant distractor added | Preserve materially supported conclusions and calibrated confidence | Penalizing a justified update because the distractor actually contains relevant evidence |
| Deletion calibration | Necessary evidence removed | Reduce claim strength, qualify, abstain or retain only what remaining evidence warrants | Automatically requiring abstention when remaining evidence is sufficient |
| Permutation invariance | Evidence order/format changed without semantic change | Preserve the justified conclusion | Comparing exact response hashes or phrasing instead of semantic obligations |
| Evidence-force monotonicity | Evidence warrants less specificity/strength | Never make a stronger unsupported claim from weaker evidence | Treating a less fluent or shorter answer as necessarily better calibrated |
| Claim/evidence completeness | Full response mapped to required evidence spans | Every material claim is warranted at the required granularity, and required evidence is accounted for | Giving full credit for one relevant citation while other material claims are unsupported |

For force-calibration cases, the changed axis is recorded separately: relation, modality, scope, temporal validity, or numeric specificity. The plan's family-to-behavior mapping is normative; the future validator must reject incompatible mappings rather than silently coerce them.

## 3. Artifact contracts

The initial machine-readable artifacts are:

- **schema/evidence-behavior-contract-plan.schema.json** — strict pre-run case ontology and evaluator-side commitments.
- **schema/evidence-behavior-contract-report.schema.json** — strict terminal adjudication records and report binding.
- **fixtures/evidence-behavior-contract-plan.example.json** — seven-family illustrative plan.
- **fixtures/evidence-behavior-contract-report.example.json** — illustrative results, including passes, violations and a not-evaluable case.

Both fixtures deliberately use zero digests and the NOT_IMPLEMENTED_YET revision marker. They are examples for schema and ontology review only. They must fail any future qualified-run admission check. They are not real commitments, evidence, adjudications, or benchmark results.

The planned Rust validator must enforce all of the following, not merely validate JSON shape:

1. The exact plan bytes, typed plan commitment, source-authority policy, evaluator build revision and decision-rule revision are bound consistently.
2. Case IDs are unique; every planned case receives exactly one terminal adjudication; duplicates, omissions and unplanned records fail closed.
3. Every record has digest references for the baseline response, intervention response and adjudication artifact when those artifacts exist. Missing means unknown, not zero.
4. pass, violation, and not_evaluable are different terminal states. A violation requires one or more violation codes; not-evaluable requires an explicit reason; it is never silently dropped from the assigned denominator.
5. The contract family and expected behavior match; decision rules and source-authority policy cannot be revised after outcomes are seen.
6. The verified report emits per-family counts and eligible denominators, with violation rate reported only when the eligible denominator is nonzero. It keeps runner_execution_proven=false and external_anchor_required=true.
7. A digest proves byte identity under its declared construction; it does not authenticate who adjudicated, whether a runner applied the intervention, or whether a label is true. A qualified run needs an independent pre-run anchor, runner receipt and adjudication audit.

Response hashes are provenance links only. Equal hashes do not by themselves prove invariance, and different hashes do not prove a valid update. Semantic contract adjudication must decide whether the required obligation was met.

## 4. Evaluation and competitive plan

### Core metrics (publish separately by family)

- **Update compliance:** proportion of eligible cases where dependent claims move in the required direction; separately report unrelated-claim drift.
- **Conflict disclosure:** conflict-recognition recall, explicit acknowledgment precision/recall, provenance completeness, and unsupported winner-selection rate.
- **Decoy invariance:** rate of unjustified material-claim changes and confidence movement under truly irrelevant distractors.
- **Deletion calibration:** proportion of claims whose force is calibrated to the evidence remaining; report both overclaim and unnecessary-abstention rates.
- **Permutation invariance:** semantic-equivalence pass rate and order-induced violation rate.
- **Evidence-force monotonicity:** monotonicity violation rate (the stronger claim is judged at least as warranted despite weaker evidence), stratified by the five force axes.
- **Claim/evidence completeness:** material-claim support precision/recall and required-evidence-span recall; never substitute citation presence for evidence support.
- **Measurement integrity:** assigned, observed, incomplete, contaminated and not-evaluable counts; record-level reasons; cost/tool/time denominators; blind-adjudication coverage; judge-vs-human agreement and uncertainty.

An aggregate Intervention Compliance Score may be reported as a secondary summary only if its weighting is frozen before trials. Never let a high score on permutation invariance compensate for failure to disclose a safety-critical conflict. Release decisions should retain per-family gates.

### Target ladder

| Stage | Qualification target | What it would establish |
|---|---|---|
| Contract parser/validator | Reject 100% of deterministic malformed, duplicate, omitted, incompatible-family and post-hoc mutation controls in the committed fixture suite | Correctness of the bookkeeping implementation only |
| Synthetic behavioral mutants | Detect 100% of the explicitly enumerated deterministic violations and accept all clean controls in the test suite | Unit-test coverage of known constructions, not open-world detector accuracy |
| Blind adjudication pilot | Publish per-family inter-rater agreement and confidence intervals; proposed release-gate threshold of agreement ≥ 0.80 on high-consequence labels, with disagreements adjudicated and logged | Whether human labels are sufficiently reproducible for the stated use |
| Automated evaluator audit | Freeze judge model/version/prompt; compare against blinded adjudicated examples; publish per-family false-positive/false-negative rates and calibrated intervals | How much the judge can be trusted for each contract family |
| External held-out comparison | Use sealed, independently authored cases and matched policy/model/tool budgets; report all planned cases and costs | Comparative evidence about the declared task distribution, not universal truth or generalization |
| Generalization analysis | Predeclare task-family and domain sampling; use a justified hierarchical/GLMM or other design-appropriate model and diagnostics | An inference about the declared target population, subject to sampling and model assumptions |

The 100% targets apply only to enumerated deterministic fixture controls. They are not a target for unseen real-world cases, not an achieved result, and not a SOTA claim. The proposed 0.80 agreement threshold is a governance target to validate, not a universal statistical law.

### Baselines and fair comparison

Run the same sealed cases, source snapshot, system policy, intervention pairs and scoring rules against: (1) a conventional retrieval-augmented generation pipeline; (2) a long-context/no-retrieval baseline; (3) an eligible tool-using agent; (4) Symthaea's evaluated configuration; and (5) an oracle-evidence diagnostic where appropriate. The oracle-evidence condition diagnoses whether failures arise before or after evidence access; it is not a competitor and must not be mixed into headline ranking.

Freeze model/version, prompts, tool list, retrieval index, source date, decoding settings, budgets, retry policy and evaluator before trials. Match budgets and report quality-versus-cost frontiers rather than comparing unconstrained outputs. Keep public development, immutable anchors, sealed refreshes and independent external tasks as separate lanes. For broad generalization claims, do not treat repeated questions from one task family as independent samples; publish hierarchical assumptions and diagnostics.

## 5. Failure taxonomy and audit

Every violation should carry a family-local code and the supporting adjudication-artifact digest. At minimum distinguish: ignored coherent update; unjustified unrelated-claim drift; silent conflict; unsupported source preference; decoy-induced overclaim; evidence deletion with unchanged overclaim; unnecessary abstention; order-induced semantic drift; claim-force increase under weaker warrant; unsupported material claim; omitted required evidence; evaluator disagreement; missing evidence; and execution/receipt failure.

Automated judges are not gold by default. Use blinded human adjudication on a stratified sample that over-samples disagreement, conflict cases, and high-impact claims. Publish agreement and audit-selection rules. Human reviewers must see the relevant sources and the frozen rubric, not the system identity or other treatment's score before recording an initial judgment. Preserve raw adjudication artifacts and revisions with privacy controls.

## 6. Claim boundary and next acceptance sequence

This initial artifact set defines the ontology and schemas; it does not yet contain the Rust enforcement layer. Next, implement and test typed commitments, exact one-record-per-case accounting, invalid-plan/duplicate/missing-record negative controls, per-family scorecards, deterministic report commitments, and schema/CLI parity. Only after those pass should the contract report be joined to the paired analyzer in #8993. A later independent pilot can compare baseline systems and estimate effects.

Contract compliance is conditional on the declared case construction, evidence-authority policy and adjudication rubric. It is not, by itself, proof of causal reasoning, truth, safety, or generalized intelligence. No measured Symthaea improvement or state-of-the-art claim follows from these schemas or fixtures.

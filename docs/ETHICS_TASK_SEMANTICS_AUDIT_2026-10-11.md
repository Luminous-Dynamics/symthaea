# ETHICS Benchmark Task-Semantics Audit

**Date:** 2026-10-11  
**Status:** audit finding; corrected task-matched harness added for commonsense only. Full ETHICS-suite qualification remains open.  
**Scope:** repository-source inspection and published task schemas. No Rust benchmark was executed in the authoring environment.

## Executive finding

The previously reported **56.2% aggregate must not currently be presented as a qualified ETHICS moral-reasoning accuracy result**. This is not just because the score is low. The current live-path pilot applies a generic “is this morally wrong?” score to multiple ETHICS sub-tasks whose labels answer different questions.

The earlier 91.1% / 94.5% / 84.7% headline family remains retracted as leakage-inflated. This audit does not revive those figures or replace them with a new result.

## Exact source findings

At the inspected current main revision, examples/agw_ethics_holdout.rs:

1. Generates one scalar wrongness score: -0.8 for a parsed consent violation; otherwise 0.6 * (good_similarity - bad_similarity) + 0.4 * deontological_score, clamped to [-1, 1].
2. Predicts “wrong” iff the score is negative;
3. Interprets label == 1 as “wrong” for every domain;
4. Concatenates deontology scenario and excuse into one string;
5. Loads only the first 200 rows in each of four domains;
6. Excludes utilitarianism and test_hard;
7. Emits a combined cross-domain accuracy.

That mapping is task-valid for commonsense wrongness, but not for all the other sub-tasks:

- **Commonsense:** “Is this action wrong?” This is the one task that directly matches the current scalar target. In the source dataset, label 1 means wrong and label 0 means not wrong.
- **Deontology:** asks whether an excuse is reasonable for a stated duty/request. This requires reasoning over the duty and the excuse relationship, not simply classifying the concatenated text as wrong.
- **Justice:** asks whether a scenario/claim is reasonable or justified under the task's justice construction. A general moral-valence score is not a semantic substitute.
- **Virtue:** asks whether a particular trait is exhibited by the character in the scenario. A trait mention or global wrongness score does not answer whether the scenario actually demonstrates that trait. The scenario and queried trait must be evaluated jointly.
- **Utilitarianism:** compares two activities/outcomes and requires a pairwise preference/ranking target rather than single-sentence wrongness.

ETHICS also includes contrast/group structure for some tasks (groups of four for justice/deontology and five for virtue in the common task loaders). Per-row accuracy is not automatically comparable with a task-level exact-match metric. Any deviation must be explicit.

## Sample-count and provenance mismatch

The current source pilot sets N_PER_DOMAIN = 200, across four domains: **800 rows**, not 2,000. The README/research pages have also described the result as “2K samples”. The documentation and source runner therefore do not identify the same experiment as currently written, unless a separate run artifact proves the count and revision. The result needs a manifest tying together:

- exact code commit and command;
- dataset source revision and hashes of every CSV;
- task, split, number of rows and row-selection rule;
- label semantics and input transformation per task;
- scoring threshold, grouping rule, and metric;
- per-row predictions or a privacy-safe/evidence-appropriate output artifact.

The first-N selection is deterministic but is not representative sampling. Public availability of benchmark labels also means an official test split is not a secret holdout against iterative development. Keep test-hard and newly authored counterfactual examples for robustness checks, without tuning thresholds on them.

## Distinct legacy model pipelines and additional validity defects

The live Stage-1 pilot, older HDC classifier, composite benchmark, and ablation script are different experiments. Their headline percentages must not be compared as if they measured the same model under the same evaluation contract.

- `examples/benchmark_ablation_ethics.rs` loads `ethics.json` into the evaluation set and passes the same file to `train_per_cat`. That function trains per-category classifiers from those examples, which are then scored again. Results for configurations using those classifiers are **in-sample**, not held-out accuracy. The historical per-category ablation gain must not be reported as evidence of generalization.
- `examples/benchmark_ethics_hdc.rs` has separate train/test file paths, but its legacy `load_labeled_csv` extracts data by splitting each raw line at the first comma rather than using a CSV parser. Quoted commas/fields and extra columns can therefore become part of the model input. It also flattens deontology's scenario and excuse instead of preserving their typed relation. Further, it selects a per-category “best” method using `max(ngram_retrain_acc, moral_acc)` on the **test** scores; that selected “best” aggregate is test-set-selected and is not a clean unbiased final estimate. Predeclared method scores should be reported separately.
- `examples/benchmark_moral_unified.rs` is a legacy composite across several datasets and model components with multiple evaluation paths. Its exact output needs a run manifest identifying dataset revision, split, task adapter, model configuration and the scoring function; the existence of train/test-looking fields does not prove every component followed the same split contract.
- The live 56.2% pilot and historical 92.9% trained/composite figures are not direct before/after measurements: they use materially different paths and their underlying artifacts are not yet reconciled. Do not conclude either that the architecture regressed from 92.9% to 56.2% or that the trained 92.9% configuration generalizes.

These source-level findings are enough to invalidate the *unqualified claims*, but not enough to assign a single corrected full-suite score. Preserve each legacy program for investigation; qualify its output and build a new task-matched harness rather than merely adjusting a reported percentage.

## Additional runtime semantic risk: unspecified consent defaults to implied consent

A separate source inspection found a runtime risk beyond benchmark scoring:

- `ParsedMoralScenario::default()` initializes consent to `ConsentState::Implied`.
- `MoralParser::detect_consent()` also returns `ConsentState::Implied` when neither an absent-consent phrase nor a given-consent phrase is detected.
- The enum documents `Implied` as “implicit/assumed consent”, while `Absent` means “no consent requested”. Therefore, “the text did not tell us whether consent exists” is currently collapsed into “consent is implied”.

That is an unsafe evidence collapse for consent-sensitive decisions. **Unknown is not granted, denied, or necessarily absent consent.** The follow-on design should add an explicit `Unspecified`/`Unknown` state, preserve `Absent` and `Implied` as different claims, and attach provenance/confidence to the parser's determination. For actions whose domain requires consent, unknown consent should block automatic authorization or trigger abstention/clarification—not be silently treated as permission. Conversely, not every action requires consent, so unknown consent should not mechanically label all actions immoral.

Acceptance tests should cover no mention of consent, explicit grant, explicit denial, not requested, contextual implication, negation, quoted speech, and adversarial keyword placement. Use property-based paraphrase/scope tests and a fail-closed authorization test for consent-required actions before integrating this into the live engine. This code change is intentionally a separate follow-on; it is not implemented by the benchmark-audit PR.

## Corrected immediate scope

A new examples/ethics_commonsense_holdout.rs evaluates **only the commonsense task**, on the full standard test CSV and, when present, the full hard-test CSV. It uses the existing live-path scalar scorer with the task's actual binary target and a fixed zero threshold. It reports confusion counts, accuracy, balanced accuracy, precision, recall, specificity, majority-class baseline, row count, split path, and BLAKE3 digest of the exact source file.

This is a task-matched diagnostic, not proof of moral competence. The example has not been executed in this authoring session; its output must be generated and archived at an exact commit before any result is quoted.

## Next implementation gates

1. **Commonsense:** execute the new full-split harness; independently rerun; preserve data digest and per-example errors. Compare with majority and published baselines using exactly described splits.
2. **Deontology:** represent DutyScenario and Excuse as separate typed inputs; evaluate excuse reasonableness and appropriate group metrics.
3. **Justice:** define its exact question and contrast group before selecting a justice-specific evaluator.
4. **Virtue:** represent Scenario and TargetTrait separately; score whether the trait is evidenced in context, including vice and virtue traits, negation, counterexamples, and trait-swap controls.
5. **Utilitarianism:** compare paired activities with order randomization pinned per pair; use pairwise accuracy and tie/uncertainty policy.
6. **Leakage resistance:** add paraphrase/counterfactual transformations and a privately authored or newly generated locked challenge set. Public official tests are regression data, not a final contamination-resistant holdout.
7. **Model ablations:** compare keyword parser only, HDC prototype only, deontological rules only, and the live combination on the same examples. Report category-wise error clusters, not one grand number.
8. **Human/ethical disagreement:** label whether each example has a task-defined answer, depends on missing facts, or legitimately varies by declared normative premises. Do not automatically convert disagreement into model error or fake consensus.

## Why this is the right first move

Improving model behavior against a mis-specified target risks optimizing the wrong thing. Establishing a valid measurement contract comes first. Only then can we distinguish model limitations from dataset artifacts, bad input representation, threshold problems, and failures of the ethical framework itself.

## References

- Hendrycks et al., Aligning AI With Shared Human Values, ICLR 2021: https://arxiv.org/abs/2008.02275
- Original ETHICS repository and published leaderboard/splits: https://github.com/hendrycks/ethics
- Dataset/task schema with duty/excuse, scenario/trait, and paired utilitarian tasks: https://huggingface.co/datasets/EleutherAI/hendrycks_ethics
- 2026 position paper arguing that moral-value prediction and context-sensitive moral-norm application are distinct evaluation problems: https://arxiv.org/abs/2608.14566

# Contamination Panel Benchmark

Standalone Rust crate for deterministic T1–T4 challenge-pair generation and blinded human adjudication validation.

## Why it exists

The contamination-panel workstream needs controlled positive and negative examples before comparing detectors. The generator creates *planted controls* whose construction mechanism is known to the evaluator. The reviewer-facing packet deliberately excludes tier, control label, transformation family, source lineage and recipe lineage.

This makes the tool useful for measuring detector behavior under controlled transformations. It does not establish that an external model has or has not seen any item in pretraining, instruction tuning, retrieval caches or post-training.

## Build and test

From this directory:

~~~sh
cargo fmt --all -- --check
cargo test --locked
cargo clippy --locked --all-targets -- -D warnings
~~~

The crate is standalone and is not a workspace member of the main Symthaea Cargo workspace.

## Generate a challenge packet

Use at least 32 bytes of fresh entropy from an OS-backed cryptographic random source. Keep the seed and sealed key in evaluator-only storage. Example:

~~~sh
seed_hex="$(openssl rand -hex 32)"

cargo run --locked -- generate \
  contamination-panel-2026-10-a \
  25 \
  "$seed_hex" \
  /reviewer-release/public-packet.json \
  /secure/evaluator/sealed-key.json \
  /secure/evaluator/manifest.json
~~~

The generator emits eight challenge pairs per requested pair-count unit: one planted positive and one planted negative control for each of T1, T2, T3 and T4. With 25 pairs per tier, it emits 200 pairs total.

The files have different access requirements:

| Artifact | Intended access | Contents |
|---|---|---|
| public-packet.json | Reviewers and detector teams | Opaque pair/item IDs and two task texts |
| sealed-key.json | Independent evaluator only | Tier, target mechanism, planted control label, transformation family and hidden source/recipe lineage |
| manifest.json | Evaluator; a redacted copy may be published | Artifact digests, seed commitment, generation config and claim boundary |

Do not distribute the seed or sealed key to reviewers, detector authors, or the model under test. The CLI writes each artifact to the exact path supplied, so choose separate access-controlled locations deliberately. The manifest contains a sealed-key digest commitment; it does not expose the key itself.

## What the tiers actually mean

The sealed key stores all planted mechanisms present, not only a single positive/negative bit. A T1-positive exact duplicate also necessarily has derivative, semantic-equivalence and task-recipe relationships. A T1-negative pair may still share a task recipe while lacking the exact-instance mechanism. This avoids mislabelling every non-identical pair as completely unrelated.

| Tier | Positive control | Hard negative control |
|---|---|---|
| T1 exact | Byte-identical task instance | Same task recipe but changed parameters / distinct instance |
| T2 syntactic | Same instance with casing, spacing and punctuation transformation | Lexically close parameter change, distinct instance |
| T3 semantic | Authored semantically equivalent reformulation | Same domain with inverse/different target |
| T4 task-level | Same abstract recipe, distinct instances and answers | Same broad algebra domain, different task recipe |

T3 paraphrases are authored fixtures, not expert-validated semantic gold. T4 positives are shared-recipe controls, not proof that seeing one item contaminates a later benchmark. The current generator uses one small algebra/linear-mapping domain and should be treated as a software fixture and protocol test—not a representative contamination benchmark or a meaningful detector leaderboard. Build additional domains and task families before any cross-domain or SOTA claim. Both T3 and T4 require blinded expert validation before their labels are treated as human-grounded reference truth. Keep synthetic construction truth separate from expert construct-validity adjudication.

## Blind adjudication

Reviewers receive only public-packet.json. They must not receive the sealed key, tier labels, planted controls, detector scores, or any per-pair hypothesis from the generator.

Each reviewer supplies one JSON object per line in adjudications.jsonl with this strict shape:

~~~json
{
  "schema": "symthaea.contamination_panel_adjudication.v1",
  "pair_id": "<opaque pair id>",
  "pair_digest_sha256": "<digest from public packet>",
  "reviewer_pseudonym": "reviewer-opaque-01",
  "independence_group": "institution-or-team-opaque-01",
  "blind_protocol_version": "blind-pair-review-v1",
  "reviewer_aware_of_tier": false,
  "reviewer_aware_of_planted_control": false,
  "decision": "same_construct_different_instance",
  "confidence_permille": 900,
  "rationale_sha256": "<digest of separately stored rationale>",
  "evidence_bundle_sha256": "<digest of separately stored evidence>"
}
~~~

The allowed decisions are same_target_same_instance, same_construct_different_instance, shared_recipe_different_instance, same_domain_different_target, shared_domain_different_recipe, different_construct, and indeterminate. This keeps same-domain negative controls distinguishable from semantically equivalent targets and shared task recipes. The implementation requires exactly three adjudicators per pair, with distinct reviewer pseudonyms and declared independence groups within that pair. It reports total unique reviewers/groups and reviewer/group reuse across pairs; reuse is expected in a repeated panel and must be treated as a rater-clustering factor in downstream inference. It does not authenticate identities or institutions. The blinding fields are reviewer self-attestations, not proof that reviewers could not infer the task type from its content.

Run the validator:

~~~sh
cargo run --locked -- audit-adjudications \
  /reviewer-release/public-packet.json \
  /secure/evaluator/adjudications.jsonl \
  /secure/evaluator/adjudication-report.json
~~~

A 2–1 split is disputed, never a majority-vote truth label. Three indeterminate reviews yield insufficient evidence. Only a unanimous non-indeterminate panel receives a unanimous status, and even that means the panel was structurally consistent—not that the judgement is scientifically true.

## Required downstream detector scorecard

Compare exact/canonical matching, normalized string and n-gram methods, MinHash/SimHash, task-appropriate structural matching, embeddings, cross-encoders / LLM-assisted review, and any authorized model-probing methods. Include FTD (ACL 2026) as a mandatory statistical baseline when model access supports its likelihood-based detectors.

For each tier and transformation family report:

- precision, recall and PR-AUC;
- recall at fixed FPR (including 1% and 0.1% only where support permits);
- confidence intervals with source/task-family cluster-aware resampling;
- calibration and abstention;
- manual review burden and expert minutes;
- latency, compute, model/API calls and cost;
- family-macro and worst-family results;
- PPV under plausible real-world prevalence grids, not only balanced-sample accuracy.

Thresholds are chosen on development/calibration data, not sealed qualification examples. Report insufficient support as not_estimable, not as a pass.

## Claim boundary

This tool validates deterministic generation, packet/key separation, artifact commitments, exact pair binding, review-panel structure, self-reported blinding and conservative disagreement handling. It does not prove:

- that a model encountered or did not encounter a task in training;
- that an authored T3 pair is semantically equivalent without expert review;
- that a T4 shared recipe constitutes harmful contamination;
- that reviewer identity or independence is authentic;
- that the seed was generated securely or that the evaluator kept the key secret;
- that the benchmark is representative of a broader task population;
- that any detector is complete across all contamination mechanisms.

A clean detector report remains evidence about the detector and evidence supplied—not evidence of universal absence.


## Built-in detector baselines

The first built-in baselines are deliberately simple and independently reproducible:

- exact_text_equality: byte-for-byte equality of the two task strings;
- normalized_text_equality: ASCII case-folding plus punctuation/whitespace normalization. It is explicitly **not** full Unicode normalization;
- token_ngram_jaccard: Jaccard similarity over normalized token n-grams (n from 2 through 8).

These establish lower bounds and fixture sanity checks. They are not the final competitive panel. MinHash/SimHash, language-appropriate structural/AST matching, embeddings, cross-encoders, LLM-assisted detectors, and FTD (ACL 2026) remain separate comparators to implement and reproduce. FTD requires a compatible model-access regime (likelihood/log-probability scores); its paper's FDR results must not be assumed to transfer.

Create a detector specification such as:

~~~json
{
  "schema": "symthaea.contamination_panel_detector_spec.v1",
  "detector_id": "exact-text-v1",
  "detector_version": "1.0.0",
  "kind": "exact_text_equality",
  "threshold_permille": 1000,
  "token_ngram_size": 0
}
~~~

For an n-gram detector, use kind "token_ngram_jaccard" and token_ngram_size between 2 and 8. The risk_permille output is a ranking/similarity score from 0 to 1000. It is **not** a probability and must not be used for Brier score, calibration or PPV without a separate calibration procedure.

Run the detector against the public packet only:

~~~sh
cargo run --locked -- score \
  /reviewer-release/public-packet.json \
  /detector-workspace/exact-text-spec.json \
  /detector-workspace/exact-text-run.json
~~~

Only after the score file is frozen should the evaluator join it to the sealed key and manifest:

~~~sh
cargo run --locked -- evaluate \
  /reviewer-release/public-packet.json \
  /secure/evaluator/sealed-key.json \
  /secure/evaluator/manifest.json \
  /detector-workspace/exact-text-run.json \
  /secure/evaluator/exact-text-report.json
~~~

The evaluator verifies the public packet digest, sealed-key digest, seed commitment, generator/version and expected pair count against the challenge manifest. Publish a redacted report and retain the sealed key in evaluator-only storage.

The report distinguishes precision, recall, specificity, false-positive rate and false-discovery rate for each tier. Empty class denominators produce null/not-estimable metrics, not zero. Current reports are descriptive counts for synthetic planted controls; cluster-aware confidence intervals, prevalence-adjusted PPV, threshold curves and temporal transfer are not yet implemented by this crate.

## Detector development boundary

Detector execution must never receive the sealed key. A detector adapter should consume the public packet and emit one score per opaque pair ID, bound to the exact packet digest. Evaluator-only code checks coverage and joins scores to the sealed key. For a model/API detector, additionally bind its exact model/provider version, prompt, runtime, model-access mode, token/call budget and any retrieval/network exposure through the separate score-validity disclosure workstream (#9493).



## FTD-style statistical comparison kernel

The crate also provides an FTD-style p-value/fusion kernel in src/ftd.rs. It is independent of model-specific score extraction: a PPL, Min-K, Min-K++, gradient, or other authorized adapter supplies per-item scores and a contaminated-reference score sample for each feature.

Important paper-to-code audit:

- ACL 2026 paper Eq. (4) prints the lower-tail reference probability, but the paper also states that low scores indicate contamination, defines contamination as the null, and selects BH rejections as clean. Those statements imply the upper-tail p-value for a detector whose high score is clean-like.
- The official liano3/FTD implementation supports both score directions and uses the greater direction in its listed experiments: it counts reference scores greater than or equal to the observed test score. This makes high test scores yield small p-values and resolves the intended direction in code.
- The paper's Eq. (7) describes normalized per-detector BH rejection counts. The official repository further applies temperature-scaled softmax sharpening (temperature 0.5 in its experiment path). These are implemented as distinct weighting policies.

Sources:
- ACL paper: https://aclanthology.org/2026.acl-long.1390/
- Official reference implementation, pinned source: https://github.com/liano3/FTD/blob/412e6dc597e65b14ac5db3257265ed04102c2a89/src/FTD.py
- Official clean-set selection stage: https://github.com/liano3/FTD/blob/412e6dc597e65b14ac5db3257265ed04102c2a89/src/clean_data.py

The Rust kernel provides:
- deterministic randomized-tie empirical reference p-values;
- explicit greater/less tail direction;
- single-feature BH rejection counts;
- paper normalized-count weights and official softmax-sharpened weights;
- weighted Cauchy p-value fusion and final BH selection.

The statistical report does **not** claim that these calculations automatically guarantee FDR control. The reference sample must be representative of the contaminated null, the resulting p-values must be super-uniform under that null, and cross-hypothesis dependence must meet the relevant multiple-testing assumptions. This crate does not establish those conditions. It also does not yet implement PPL, Min-K, Min-K++ or model-serving adapters, so it is an aggregation/reproduction kernel rather than a full FTD reproduction.



Run the empirical reference step for each detector feature. Reference scores are the chosen contaminated-null sample; test scores are per opaque pair ID:

~~~sh
cargo run --locked -- ftd-pvalues \
  ppl \
  greater \
  /secure/evaluator/reference-ppl-scores.json \
  /detector-workspace/test-ppl-scores.json \
  "$seed_hex" \
  /detector-workspace/ppl-feature-pvalues.json
~~~

Use greater when high scores are evidence that an item is unlike the contaminated reference (the official FTD code direction for its reported experiments). Use less only when the detector's score semantics are reversed. Do not infer direction from a feature name; predeclare and validate it.

Fuse feature p-values and apply final BH:

~~~sh
cargo run --locked -- ftd-fuse \
  /detector-workspace/feature-pvalues.json \
  0.05 \
  paper \
  none \
  /secure/evaluator/ftd-paper-weights-report.json

cargo run --locked -- ftd-fuse \
  /detector-workspace/feature-pvalues.json \
  0.05 \
  official \
  500 \
  /secure/evaluator/ftd-official-weights-report.json
~~~

The first command uses paper Eq. (7) normalized BH rejection-count weights. The second adds the official implementation's softmax sharpening at temperature 0.5. Compare both; do not silently label the extra sharpening as part of the printed equation.

The reported p-value direction in the ACL paper's Eq. (4) appears to be a lower-tail sign typo given the stated score direction and clean-selection rule. The official implementation uses the greater-tail direction in its listed experiments. This crate records both directions so the choice is explicit and testable.

This is not a full FTD reproduction: model scoring for PPL, Min-K and Min-K++ is not implemented here. Nor does running BH or a Cauchy combination alone establish an FDR guarantee. The null reference must be representative, p-values must be super-uniform under contamination, and the multiple-hypothesis dependence assumptions must be justified on the actual evaluation population.



## DVD reproduction and two-state mixture diagnostics

The module src/dvd.rs implements the EMNLP 2026 DVD statistic and an additional explicit state-separation analysis.

Paper reference:
- Liang et al., DVD: A Robust Method for Detecting Variant Contamination in Large Language Model Evaluation, EMNLP 2026, pp. 29366–29384.
- https://aclanthology.org/2026.emnlp-main.1289/

### Faithful DVD baseline

For each of N sampled responses:
1. Obtain the base model's conditional log-probability for every generated response token.
2. Select the K least-probable tokens (smallest / most negative log-probability values).
3. Average their log-probabilities to get that response's synthetic difficulty.
4. Compute the population variance of the N synthetic difficulties, dividing by N.

The paper's baseline configuration is N=50 and K=20. The implementation reports population variance separately from unbiased sample variance; only population variance is the paper DVD score.

Important: the temperature parameter controls sampling, but the score requires base-model conditional token log-probabilities—not probabilities from the temperature-scaled sampling distribution. Providers that do not expose per-token output log-probabilities cannot run this faithful baseline. Do not substitute text similarity or token entropy and still label it DVD.

### Added state-separation diagnostic

The paper motivates DVD by a two-state mixture: memory adherence and perturbation drift. Its scalar variance score detects fluctuation but does not explicitly test whether the observations exhibit two separated modes.

The extension fits:
- a one-Gaussian model;
- a two-Gaussian mixture using bounded deterministic EM and several quantile initializations;
- BIC for each model and delta-BIC = BIC(one component) - BIC(two components);
- mixture weights, means, variances, standardized mean separation and mean posterior entropy;
- median absolute deviation and 10th/90th percentile winsorized variance.

Positive delta-BIC favors the two-component model after its extra-parameter penalty. This is a model-comparison statistic, not a calibrated contamination probability. It can also respond to non-contamination causes of multimodality; threshold selection and calibration require clean and contaminated held-out controls. The report explicitly leaves calibration status unset.

### Input and CLI

The strict input JSON Schema is at schema/dvd-generation-input.schema.json. Example input using the paper's N=50/K=20 setting:

~~~sh
cargo run --locked -- score-dvd \
  research/contamination-panel-001/fixtures/dvd-two-state-input.json \
  /secure/evaluator/dvd-two-state-report.json
~~~

The fixture is synthetic and demonstrates the two-state numerical path. It is not model output and must not be used to claim detector accuracy.

The input binds:
- model/provider/version/access mode;
- exact harness, prompt, decoding configuration and environment digests;
- sampling temperature and K/N;
- a generation seed-schedule commitment and a distinct seed commitment per response;
- exact response token count and every response token's base-model log-probability.

A report hash binds the entire input, including the token log-probability arrays. It reports sample count, total generated-token count and the digest fields needed to reconstruct the run's declared identity.

### Required comparison and negative controls

Reproduce DVD's N=50/K=20 baseline and compare it with the added mixture statistic under identical model outputs. Do not claim improvement until the paired comparison is complete.

Negative controls must include:
- clean tasks with naturally high response variability;
- temperature and prompt sensitivity without contamination;
- high-variance unimodal output distributions;
- heavy-tailed but single-state output distributions;
- long/short response and tokenizer effects;
- reasoning tasks with multiple valid solutions;
- model architecture/size and domain shifts.

Positive controls must separately manipulate exact, syntactic, semantic-variant and answer-augmented exposure at pretraining-like, instruction-tuning and post-training stages. Keep source families and task recipes held out across train/development/qualification splits.

Report AUROC and AUPRC, recall at fixed false-positive rates, per-family and worst-family results, calibration/PPV under prevalence grids, and generation/token/cost budgets. The current module produces per-item scores only; cohort-level calibration, ROC/PR evaluation and cluster-aware confidence intervals remain separate work.


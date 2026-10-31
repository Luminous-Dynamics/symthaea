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



## Finite-sample FTD calibration stress harness

The module src/ftd_calibration.rs runs deterministic Monte Carlo comparisons over known-answer synthetic contaminated/clean score populations. It exposes:

- shared per-feature reference banks reused across all test items;
- independent per-hypothesis reference banks;
- block-correlated items (common latent scores within task/source clusters);
- positively or negatively dependent detector features;
- deliberately shifted reference distributions;
- upper- and lower-tail score directions;
- paper normalized-rejection weights versus official softmax-sharpened weights.

The dependence generator uses mixtures of common and independent uniform latent variables so each one-dimensional score marginal remains uniform under the matched-null design. Clean items are shifted toward higher scores with a monotone square-root transform. Contaminated items follow the null score distribution. Positive reference shifts use a power transform that pushes the reference scores lower; negative shifts push them higher.

Example:

~~~sh
seed_hex="$(openssl rand -hex 32)"
cargo run --locked -- ftd-calibrate \
  research/contamination-panel-001/fixtures/ftd-calibration-shared-reference.json \
  "$seed_hex" \
  /secure/evaluator/ftd-calibration-report.json
~~~

Additional fixtures:
- fixtures/ftd-calibration-null-only.json: all items follow the contaminated null. Clean-sample power is not estimable.
- fixtures/ftd-calibration-shifted-reference.json: intentionally misspecified low-shifted reference plus block and feature dependence.
- fixtures/ftd-calibration-shared-reference.json: mixed clean/contaminated population with a common finite reference bank.

The simulator also compares ordinary Benjamini–Hochberg with Benjamini–Yekutieli final selection. BY is more conservative and is included as a dependence-robust multiple-testing comparator, conditional on valid input p-values. It does not repair a misspecified contaminated-reference distribution.

The simulator also compares ordinary Benjamini–Hochberg with Benjamini–Yekutieli final selection. BY is a more conservative dependence-robust multiple-testing comparator, conditional on valid input p-values; it does not repair a misspecified contaminated-reference distribution.

The report separately estimates:
- mean false-discovery proportion across runs (the Monte Carlo analogue of FDR);
- frequency of individual runs whose FDP exceeds the target alpha;
- clean-sample retention power;
- selected fraction and counts;
- fixed-grid null p-value ECDF excess, Kolmogorov–Smirnov and Cramér–von Mises descriptive statistics per detector and after fusion;
- 95% Hoeffding intervals for bounded Monte Carlo mean outcomes, plus empirical Monte Carlo standard errors.

The run-level exceedance rate is **not** the FDR definition. A high exceedance rate is a useful instability diagnostic; the mean FDP is the Monte Carlo estimate of expected FDP. All intervals describe Monte Carlo uncertainty over the simulator repetitions, not statistical uncertainty about real-world model contamination.

### Important theorem/formula audit

The paper's Section 4.2 Eq. (4) and Theorem 1 both print a lower-tail p-value. However, the paper's Section 4.1 convention says low scores indicate contamination, while Theorem 2's power premise requires clean items to have small p-values. For high-score-clean detectors, that direction requires the upper tail. The official FTD implementation uses the upper-tail greater direction in its reported experiment path.

The lower-tail expression appears in both Eq. (4) and Theorem 1, while Theorem 2's power premise requires clean items to have small p-values. A lower-tail empirical p-value can still be uniform under the contaminated null when test and reference scores are exchangeable and continuous, so the printed lower-tail direction does not by itself disprove the null-uniformity/FDR argument. It does, however, reverse which alternative is powerful under the stated score semantics. The reproduction record therefore treats the official code direction as the operational comparator, preserves the lower-tail option for audit, and tests the power reversal explicitly.

The simulation is a falsification tool, not a proof. A passing synthetic grid does not establish reference exchangeability, real-data null super-uniformity, cross-item dependence assumptions, semantic validity, or any universal FDR guarantee. Any scenario with reference shift or task clustering that exceeds the target blocks an FDR guarantee claim for that scope.


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



## Randomized causal exposure interventions

The module src/intervention.rs adds a preregistered randomized assignment layer for estimating causal contamination effects. The randomization unit is an indivisible source/task cluster, not a question. All items from one source-family digest remain together; duplicate source-family digests across nominal clusters are rejected to prevent source leakage disguised as separate IDs.

### Design and exact budget conservation

A design binds:
- experiment and replicate IDs;
- benchmark/version and task-set digest;
- base model checkpoint digest;
- training stage and exposure mechanism;
- treatment and placebo training-corpus digests;
- optimizer/hyperparameter and schedule digests;
- common/shared training corpus digest and token count;
- total training token budget and training-step count;
- per-source-cluster item membership, exposure-variant digest/token count and matched placebo-record digest/token count.

The same design fixes the base checkpoint, common corpus, stage, optimizer schedule, hyperparameters, training steps and total budget across treatment and placebo arms. For assigned clusters, benchmark-derived treatment records are replaced by unrelated placebo records with equal token counts.

The validator enforces:
- per-cluster exposure and placebo token counts are exactly equal and positive;
- treatment and placebo records have distinct, non-reused digests;
- all clusters within a task-family / cluster-size stratum use the same exposure token count, making the assigned token total invariant to which cluster IDs are selected;
- shared_training_token_count + exposed_training_token_count equals total_training_token_budget;
- shared_training_token_count + matched_placebo_token_count equals total_training_token_budget;
- all item IDs are globally unique and no source family is split across clusters;
- each interior-dose task-family / cluster-size stratum has enough clusters to include treatment and placebo;
- source-cluster allocation is deterministic and replayable from the committed seed and replicate ID.

### Randomization and replay

The allocator builds strata from task family and cluster-size bucket, computes HMAC-derived cluster priorities using experiment and replicate identity, and assigns the predeclared fraction within each stratum. Interior doses require at least two clusters per stratum and retain at least one cluster in both treatment and placebo groups, even for extreme 1‰/999‰ targets. Token counts are homogeneous within each stratum so the fixed total training budget does not depend on which random clusters are selected. The manifest records exact assignments, treatment/placebo token totals, strata, assignment digest, design digest and seed commitment.

The reviewer/evaluator retains the seed and assignment key. Model/detector teams receive only the information approved by the preregistered protocol; do not disclose hidden assignment labels before the confirmatory run is locked.

Example:

~~~sh
seed_hex="$(openssl rand -hex 32)"

cargo run --locked -- randomize-intervention \
  research/contamination-panel-001/fixtures/intervention-semantic-variant-design.json \
  "$seed_hex" \
  /secure/evaluator/intervention-manifest.json \
  /secure/evaluator/intervention-audit.json
~~~

Replay a supplied manifest independently:

~~~sh
cargo run --locked -- audit-intervention \
  research/contamination-panel-001/fixtures/intervention-semantic-variant-design.json \
  /secure/evaluator/intervention-manifest.json \
  "$seed_hex" \
  /secure/evaluator/intervention-replay-report.json
~~~

Strict input/output schemas:
- schema/randomized-intervention-design.schema.json
- schema/randomized-intervention-manifest.schema.json

The included fixtures/intervention-semantic-variant-design.json has 32 source clusters (8 each for math, code, science and logic), 3 items per cluster, 50% treatment assignment, 8,000 exposure tokens, 8,000 matched placebo tokens, 92,000 shared-corpus tokens, and a 100,000-token total budget. It is a synthetic allocation fixture, not an executed training experiment.

### Causal estimands and required training receipts

The manifest validates assignment intent, not actual model training. A future trainer adapter must append independently captured receipts for:
- treatment and placebo checkpoint digests;
- exact corpus shard and post-tokenization digests;
- observed token counts and optimizer steps;
- job/container/runtime identity;
- training code and hyperparameter digest;
- checkpoint parent lineage and artifact custody;
- evidence that treatment clusters entered only the intended treatment arm and placebo records only the placebo arm.

Primary estimands should include intent-to-treat score inflation, dose-response, exposure-mechanism effects, training-stage interactions, detector sensitivity on treatment and false positives on matched placebo, and family-held-out transport. Analyze at the randomized source-cluster level using randomization inference and cluster-aware uncertainty; individual items within a cluster are not independent experimental units.

The current code does not train models or attest checkpoint provenance. It is the deterministic randomization, budget-matching and replay-validation layer. A valid allocation manifest is not evidence that the declared corpus was actually consumed by a trainer.



## Paired treatment/placebo training execution receipts

The module src/training_receipt.rs validates a paired treatment/placebo execution receipt against the exact intervention design and sealed randomization manifest.

### What each arm must attest

For each training run, the receipt binds:
- arm identity, run ID and completed/partial/failed status;
- base and parent checkpoint digests, plus output checkpoint digest;
- exact treatment or placebo corpus digest and the shared corpus digest;
- intervention manifest and assignment digests;
- canonical digest of the selected treatment payload or matched placebo payload;
- declared and observed shared/intervention/total token counts;
- requested and observed training steps;
- training code, runtime image, tokenization pipeline, hyperparameters and optimizer schedule;
- corpus shard manifest and post-tokenization manifest;
- trainer-job execution receipt, checkpoint-custody receipt and runner-attestation receipt digests;
- UTC start/completion timestamps and compute/wall-time counters.

The validator recomputes the intervention design digest using the same canonical serialization as the randomization module. It replays the assignment, derives treatment and placebo payload digests from the selected cluster records, and verifies both arm receipts against those exact values.

### Paired-run invariants

Both arms must use the same base/parent checkpoint, shared corpus, training code, runtime image, tokenizer, optimizer, hyperparameters, requested steps and total token budget. Treatment and placebo runs must use their respective preregistered corpus digests and exact matched intervention payloads. The output checkpoint digests and run IDs must be distinct.

Completed runs must report the exact declared token and step budgets. Partial/failed runs can be represented, but are not execution-consistent or ready for confirmatory analysis.

### CLI validation

~~~sh
cargo run --locked -- audit-training-receipt \
  /secure/evaluator/intervention-design.json \
  /secure/evaluator/sealed-intervention-manifest.json \
  "$seed_hex" \
  /secure/evaluator/paired-training-receipt.json \
  /secure/evaluator/paired-training-audit.json
~~~

Schemas:
- schema/paired-training-execution-receipt.schema.json
- schema/paired-training-execution-audit.schema.json

### The crucial authentication boundary

The validator checks SHA-256 digest formats and exact cross-artifact bindings. It does **not** cryptographically verify signatures, authenticate runner identities, authenticate checkpoint-custody issuers, prove a trainer actually read the declared data, or verify the hardware/TEE environment.

The report therefore always sets independently_verified=false and labels external attestation as digest-present-but-not-cryptographically-verified. It can report ready_for_independent_attestation_verification when all structural and cross-arm checks pass, but a separate trust-root/signature verifier must be built before receipts can be treated as independently authenticated.

A structurally valid paired receipt is evidence that the supplied claims are consistent with the preregistered design—not proof that the claims are true.



## Paired training execution receipt hardening

The paired receipt validator also checks calendar-valid UTC timestamps, not only timestamp string shape. For example, an impossible date such as February 30 or an hour value of 24 is rejected.

Runner-attestation and checkpoint-custody receipt digests are explicitly nullable. This is deliberate: missing external attestation evidence must remain a visible negative condition, not make the entire structurally valid run record disappear. The validator can therefore return:
- execution consistent;
- external runner/custody attestation missing;
- not ready for independent verification;
- independently verified = false.

When those external digests are present, this still proves only their format and binding—not their signature validity or issuer identity. Missing evidence blocks readiness; present-but-unverified evidence does not become trusted automatically.



## Replicated model-pair causal analysis

The module src/causal_analysis.rs analyzes independently trained treatment/placebo model pairs. The **model pair is the independent training replicate**. The randomized source/task clusters are assignment units, and benchmark questions are nested measurements. Repeated decoding from one checkpoint or hundreds of questions from one model pair do not create hundreds of independent training replicates.

### Paired training randomness

Each intervention design carries a training RNG seed commitment. Treatment and placebo runs inside one pair must use that same committed RNG schedule (paired initialization/dropout/randomness), while each independently trained model-pair replicate must have a unique training RNG commitment. The commitment is bound into the design digest, manifest and both arm receipts.

This reduces avoidable variance from different initialization/random streams. It does not remove all training stochasticity or prove the training backend honored the seed.

### Outcomes and evaluation evidence

Each replicate contains:
- the exact design, randomized assignment manifest and original randomization seed;
- the paired training execution audit and its content digest;
- distinct base/treatment/placebo evaluation receipt digests;
- common checkpoint, oracle/adjudication, metric, evaluation harness and prompt/decoding bindings;
- one cluster outcome for every source cluster, with base, treatment and placebo success counts on the exact same item set.

The validator replays the original assignment, checks the training audit digest, checks output checkpoint bindings and requires complete, identical evaluation denominators. Missing clusters, incomplete item coverage, receipt reuse, duplicate output checkpoints or altered assignments fail closed.

### Primary estimand

For each independently trained model pair, compute the cluster-weighted mean of:

treatment-checkpoint score minus placebo-checkpoint score

over the full benchmark cluster universe. The overall estimate is the mean of those pair-level effects. Its standard deviation, standard error and 95% Student-t interval are calculated across independent model-pair replicates—not across individual questions or clusters.

The report separately shows:
- treatment minus common base-checkpoint score;
- placebo minus common base-checkpoint score;
- cluster-weighted and item-weighted treatment/placebo effects;
- average paired advantage on exposed clusters;
- average paired advantage on unexposed clusters (spillover diagnostic);
- a task-family-stratified exposed-minus-unexposed localization contrast;
- family-level means and every replicate-level effect.

This decomposition matters: a positive treatment-minus-placebo difference can reflect treatment gains, placebo degradation, or both. The base model reference shows which occurred.

### Interference and randomization inference

Training changes global model parameters. Exposure to one source cluster can influence scores on other clusters. Therefore the exposed-minus-unexposed localization contrast is a diagnostic, not automatically a pure direct causal effect.

The Monte Carlo randomization test reassigns cluster exposure within the original task-family / size strata while preserving each stratum's exposed count. It tests the Fisher sharp null that assignment has no effect on any cluster's paired-advantage outcome, including spillover. It is not a confidence interval for a direct effect under arbitrary global interference.

The t interval's independent unit is still the model-pair replicate. A minimum of 20 pairs is set as an initial operational floor for confirmatory planning, not as a universal power guarantee. The true required count must be chosen by preregistered power simulations that include model-pair stochasticity, source-family dependence, size imbalance, spillover, heterogeneity and missing runs.

### Exploratory-only trust boundary

The current paired execution audit validates digest/field consistency and can be marked ready for external attestation verification, but it does not cryptographically authenticate signatures or trust roots. This analysis module rejects a self-asserted independently_verified=true flag and always sets confirmatory_eligible=false. Results remain exploratory until a separate pinned trust-root verifier authenticates trainer and checkpoint attestations.

### CLI

~~~sh
cargo run --locked -- analyze-causal-study \
  /secure/evaluator/replicated-causal-study.json \
  /secure/evaluator/replicated-causal-report.json
~~~

Schemas:
- schema/replicated-causal-study.schema.json
- schema/replicated-causal-report.schema.json

The Rust tests include a synthetic two-pair end-to-end fixture, but no real training has been performed and no causal effect or benchmark improvement is claimed by that fixture.



The analysis report includes a SHA-256 digest of the exact serialized study input and a commitment to the Monte Carlo randomization-test seed. Keep the original study JSON, evaluator-only seeds, paired training audit records, manifests and final report together as the reproducibility bundle. The report's study digest must match byte-for-byte canonical serde JSON serialization under the declared hash domain.



The intervention design also binds a per-replicate training RNG commitment: treatment and placebo use the same committed RNG schedule within a pair, while independent model-pair replicates must use distinct training RNG commitments. This is checked by the paired receipt validator and repeated-replicate analysis.

## Causal exposure-to-competence transfer ladder (SCI-EXPOSURE-TRANSFER-001)

Tracking: [Issue #9726](https://github.com/Luminous-Dynamics/symthaea/issues/9726).

The randomized-intervention and paired-model analysis layers estimate whether a declared training exposure changes model scores. This extension asks *what kind of capability the measured change represents*. It partitions the evaluation set by the relationship between each task and the treatment exposure, then estimates the treatment-minus-placebo effect in every relation stratum.

This is deliberately not another contamination detector. It is a causal outcome decomposition attached to the existing randomized-training experiment. Detector scores (FTD, DVD, Min-K%++, likelihood/gradient, lexical, structural and embedding methods) remain separate comparators.

### Six evaluator-only relation strata

| Relation | Operational definition | Interpretation ceiling |
| --- | --- | --- |
| exact_replay | Same source item and target instance that appeared in the treatment exposure payload. | Narrow exact-instance retention; not proof of rote memorization by itself. |
| surface_variant | Same underlying item and target, transformed wording, formatting or presentation. | Surface robustness; can remain item-specific. |
| semantic_equivalent | Independently authored reformulation preserving assumptions, target and required reasoning, confirmed by blinded experts. | Semantic retention beyond lexical overlap; still related to the exposed target. |
| fresh_same_recipe | New post-freeze parameters/data generated using the same solution recipe. | Instance transfer within a learned recipe. |
| novel_recipe_same_skill | Different recipe and target instance requiring the same predeclared skill/capability. | Stronger evidence of transfer across task recipes, conditional on the skill mapping. |
| unrelated_control | Different target skill/construct, matched on format and externally calibrated difficulty. | Negative control for generic training drift and unrelated score changes. |

Relation labels, item IDs, task/answer payloads, source and recipe lineage, and evaluator oracles are not model-visible. Semantic-equivalence and same-skill assignments require blinded expert adjudication. Disputed or indeterminate labels are not silently promoted into positive strata.

### Experimental and statistical invariants

1. Reuse the existing source-cluster randomization, matched-token placebo, paired training receipts and independent model-pair replication. Do not re-randomize by question.
2. Each task family × difficulty band must contain all six relation strata with matched item and cluster counts.
3. The exact item-set, source-lineage, recipe-lineage and adjudication commitments must match across all independent model-pair replicates.
4. Base, treatment and placebo checkpoints must be scored on identical item IDs, metric/oracle, harness, prompt/decoding and resource budgets. The transfer study binds the existing causal-study model-pair IDs, evaluation-evidence digests and training-receipt audit digests.
5. Cross-stratum item-ID duplication, partial relation matrices, denominator mismatch, stale lineage bindings or mismatched pair receipts fail closed.
6. Within a task family, difficulty bands are equally weighted; task families are then equally weighted. The reported family-macro effect therefore does not let a large or easy task family dominate the headline.
7. All confidence intervals are calculated across independent model-pair training replicates. Items, task clusters, repeated generations and tool steps are nested measurements—not independent training replicates.
8. The primary effect in relation r is the family-macro average of treatment accuracy minus placebo accuracy in r. Treatment-minus-base and placebo-minus-base are also reported, to separate exposure benefit from placebo degradation or generic training drift.

The preregistered relation contrasts are:
- exact memorization premium: exact_replay minus novel_recipe_same_skill;
- surface retention premium: surface_variant minus novel_recipe_same_skill;
- semantic retention premium: semantic_equivalent minus novel_recipe_same_skill;
- same-recipe transfer: fresh_same_recipe minus unrelated_control;
- cross-recipe skill transfer: novel_recipe_same_skill minus unrelated_control;
- exact-to-same-recipe gap: exact_replay minus fresh_same_recipe.

These contrasts are descriptive 95% paired-replicate intervals in v1. No multiplicity adjustment or confirmatory p-value is emitted. They are not pure direct effects under unrestricted interference: model parameters are global, so exposure may affect every relation stratum.

### Rust interface

The new module is src/exposure_transfer.rs, exported from lib.rs.

    cargo run --locked --manifest-path research/contamination-panel-001/Cargo.toml -- \
      analyze-exposure-transfer \
      replicated-causal-study.json \
      exposure-transfer-study.json \
      exposure-transfer-report.json

The CLI first revalidates the existing replicated causal study. It then checks the transfer-study model-pair identities, exact evaluation-evidence digest and training-receipt digest, matched relation matrix, item-set commitments and lineage/adjudication bindings. The report contains six relation effects, per-family effects, pair-level effects and six predeclared contrasts.

The package-wide formatting gate is the same command used by the dedicated qualification workflow:

    cargo fmt --manifest-path research/contamination-panel-001/Cargo.toml -- --check

The evaluator-side study input is defined by:
- schema/exposure-transfer-study.schema.json
- schema/exposure-transfer-report.schema.json
- fixtures/exposure-transfer-relation-registry.v0.1.json

The registry fixture specifies the relation ontology, task families, required controls, estimands, comparator set and claim ceiling. The executable analyzer requires at least two task families and at least two independent model pairs for uncertainty. The base causal protocol's 20-pair minimum remains a planning floor, not a power guarantee. Determine the confirmatory sample size from a blinded pilot and preregistered power simulation using pair-level variance, the smallest effect of interest, family heterogeneity, global spillover and missing-run rates.

### Multi-domain benchmark and comparators

The target task-family registry is mathematics, code, science, factual QA, tool use and multimodal evaluation. For each family, construct exact/surface/semantic relation cases, same-recipe fresh instances, novel-recipe same-skill tasks and unrelated controls. Difficulty must be calibrated independently of the post-treatment checkpoints.

Detector comparisons should include FTD (with paper and released-code configurations labeled separately), DVD, Min-K%++, likelihood/perplexity, gradient-based features where model access permits, exact/normalized/n-gram and structural/AST matching, MinHash/SimHash, embeddings and cross-encoders. Report AUROC and AUPRC by contamination tier and relation, recall at a fixed false-positive rate (initially 1%), null p-value calibration, selective risk, abstention and coverage. Do not let a good detector AUROC stand in for evidence of transferable competence.

Reuse established evaluation runners such as Inspect and Lighteval through adapters that attach the transfer-study, score-validity and exposure-evidence artifacts. Do not rebuild their generic dataset, solver, scorer or multi-backend execution layers.

### Initial qualification targets (not achieved results)

- 100% item/checkpoint/metric coverage across the base, treatment and placebo models in every estimable relation cell.
- Zero accepted duplicate item IDs across relation/family/difficulty cells, stale lineage digests or adjudication/oracle mismatches.
- Six relation-stratum reports plus family-macro and worst-family results; no pooled accuracy-only headline.
- Initial detector operating point: at most 1% FPR, with 95% recall as a T1/T2 aspiration only after fresh multi-domain holdouts. T3/T4 must be reported separately at fixed FPR with confidence intervals and abstention.
- FTD FDR claims remain blocked until contaminated-null representativeness, per-feature p-value super-uniformity, feature dependence and multiple-testing assumptions are qualified on realistic held-out cohorts.
- Causal-transfer claims remain exploratory until actual paired training runs, independent training/checkpoint receipt authentication and blinded cross-domain adjudication are complete.

### Claim ceiling

A passing schema, synthetic fixture or content-digest audit establishes internal consistency only. It does not prove that a training job consumed the declared exposure, that the task-relation labels are semantically true, that a score change was caused by a particular retrieved item, that exposure was absent from pretraining, that global spillover is absent, or that the result generalizes outside the declared benchmark. This module hard-sets independent authentication and confirmatory eligibility to false. No causal effect, FDR guarantee or SOTA improvement is claimed by the implementation alone.

### Research references and industry alignment

- [Are LLM Benchmarks Already Contaminated? A Systematic Review of Contamination Detection Methods (GEM 2026)](https://erfan-nourbakhsh.github.io/publications/contamination-review/) reviews 55 studies through late 2025 and organizes evidence into T1–T4 contamination tiers and five detector families. This is the reason the panel reports tier-specific failure modes rather than a universal detector score.
- [Controllable Contamination Detection for Reliable LLM Evaluation with Statistical Guarantees (ACL 2026)](https://aclanthology.org/2026.acl-long.1390/) is the FTD statistical comparator. Its theorem-level FDR claim must be tested separately from detector power and null-reference validity; the reported paper formula/code-tail discrepancy is tracked in the FTD workstream.
- [DVD: A Robust Method for Detecting Variant Contamination in Large Language Model Evaluation (EMNLP 2026)](https://aclanthology.org/2026.emnlp-main.1289/) is the repeated-generation variance comparator, with reported experiments on Omni-MATH and SuperGPQA. The benchmark must test whether its result transfers beyond those task/model/training configurations.
- [Syntax-Aware Detection of Training Data Contamination in Code LLMs (Findings of EMNLP 2026)](https://aclanthology.org/2026.findings-emnlp.1010/) is an additional code-specific comparator. Its authors report absolute AUC improvements over uniform token treatment under their evaluated code-model, language and transformation settings; those claims should be reproduced in the code stratum, not generalized to other domains.
- [Search-Time Contamination in Deep Research Agents (Findings of EMNLP 2026)](https://aclanthology.org/2026.findings-emnlp.760/) identifies benchmark-metadata, question-context and explicit-answer leakage, reporting inflation up to 4% across seven public benchmarks. This is why run-time retrieval exposure is a separate validity dimension from training contamination.
- [Inspect](https://inspect.aisi.org.uk/) already provides composable datasets, solvers, scorers, agent/tool execution and evaluation logs; its scoring documentation includes grouped metrics and clustered standard errors. [Lighteval](https://huggingface.co/docs/lighteval/index) already spans multiple inference backends, a large task library and custom metrics. Symthaea's opportunity is the randomized-exposure, relation-adjudication and independently bound evidence layer—not a competing generic runner.

Comparator interpretation: published headline metrics are not directly rankable unless model checkpoint, exposure mechanism, training stage, task family, evaluation access, detector score direction, operating FPR and split protocol match. Report paper-reported numbers, official-code reproductions and Symthaea's fresh-holdout results as separate columns.

### Relation adjudication qualification

Every relation × task-family × difficulty cell carries an evaluator-only adjudication report digest, at least three distinct blinded reviewer pseudonyms, a final accepted / indeterminate / rejected decision and the minimum observed per-item agreement in permille. The implementation requires at least 800‰ agreement for a cell to be scoreable. The semantic_equivalent and novel_recipe_same_skill strata require 1000‰ agreement on every included item; indeterminate/rejected labels fail closed. Reviewer pseudonyms and report digests must remain identical across model-pair replicates.

This is a conservative protocol choice to avoid relabeling ambiguous semantics as transferable skill. It does not prove reviewer independence or identity: these fields are self-declared until a pinned external reviewer-identity and custody verifier is added. A digest proves only that the declared report content is unchanged.


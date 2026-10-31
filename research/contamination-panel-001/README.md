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


## Secret-seeded inference-time fresh benchmark instances

The module src/fresh_instance.rs adds an independent benchmark lane whose exact instances are generated after the evaluated checkpoint freeze time.

This is complementary to contamination detection:
- the static lane supports stable public regression and construct-validity comparisons;
- the fresh lane tests whether a model can apply the task-generating rule to post-freeze instances it could not have memorized as exact benchmark items.

A fresh item is not proof of task-recipe novelty. A model may have learned the general underlying method. Freshness also does not prove that the model has never encountered related training data.

### Current executable fixture: exact 2x2 integer linear systems

The initial generator:
- samples bounded integer 2x2 matrix coefficients;
- rejects matrices whose absolute determinant falls below the configured minimum;
- samples a hidden integer solution vector;
- constructs the right-hand side with checked exact integer arithmetic;
- exposes only the resulting equations and response format to the evaluated model;
- stores matrix parameters and exact solution in an evaluator-only sealed oracle packet.

This is a deterministic protocol fixture, not a broad reasoning benchmark. Future task generators should cover mathematics, code, logic, science, tools and multi-step procedural tasks.

### Evidence and artifact split

The strict config binds:
- benchmark/task family and version;
- checkpoint digest and checkpoint-freeze receipt digest;
- freeze timestamp and challenge-generation timestamp;
- generator source, binary and runtime image digests;
- exact numeric generation bounds and item count;
- a canonical generator-config digest.

The generator-config digest is calculated over the config after clearing its own digest field. The validator recomputes this digest, so changing a parameter range or item count without updating the commitment fails closed.

Three output artifacts are produced:

1. Public challenge packet — model-facing task text, item IDs, hashes and generator/checkpoint metadata only.
2. Sealed oracle packet — evaluator-only random seed, hidden matrix parameters, exact answer and per-item parameter/answer digests.
3. Manifest — commitments to the checkpoint-freeze receipt, generator/config, seed, public packet, sealed oracle and exact item set.

Never distribute the sealed oracle packet or generation seed to model-serving, prompting or scoring agents before evaluation is complete. Store them separately with evaluator-only access controls.

### Generate and independently regenerate

Use a cryptographically random evaluator-held seed of at least 256 bits:

~~~sh
seed_hex="$(openssl rand -hex 32)"

cargo run --locked -- generate-fresh \
  research/contamination-panel-001/fixtures/fresh-instance-linear-system-config.json \
  "$seed_hex" \
  /secure/evaluator/fresh-public-packet.json \
  /secure/evaluator/fresh-sealed-oracle.json \
  /secure/evaluator/fresh-manifest.json
~~~

Audit by regenerating all artifacts from the original config and evaluator-held seed:

~~~sh
cargo run --locked -- audit-fresh \
  research/contamination-panel-001/fixtures/fresh-instance-linear-system-config.json \
  "$seed_hex" \
  /secure/evaluator/fresh-public-packet.json \
  /secure/evaluator/fresh-sealed-oracle.json \
  /secure/evaluator/fresh-manifest.json \
  /secure/evaluator/fresh-audit.json
~~~

The auditor verifies exact deterministic regeneration, manifest/artifact digests, unique item/text digests, nonsingular matrices, exact answer recomputation, and the public/oracle field boundary.

Schemas:
- schema/fresh-instance-config.schema.json
- schema/fresh-instance-public-packet.schema.json
- schema/fresh-instance-sealed-oracle.schema.json
- schema/fresh-instance-manifest.schema.json
- schema/fresh-instance-audit.schema.json

Fixture:
- fixtures/fresh-instance-linear-system-config.json (32 tasks, 2x2 systems, fixed checkpoint/freeze and generation times).

### Trust boundary

The module does not cryptographically authenticate the checkpoint-freeze receipt, prove the checkpoint was truly frozen when declared, authenticate the generator binary publisher, or prevent a compromised evaluator from leaking the sealed oracle. The audit always returns independently_verified_checkpoint_freeze=false.

Trusted checkpoint-time signatures, key rotation/revocation, evaluator custody and independent generation execution receipts are separate requirements. A manifest with valid digests is not proof of a trusted timestamp or recipe novelty.



## Exact scoring for fresh inference-time instances

The module src/fresh_scoring.rs scores model outputs against the evaluator-only sealed oracle. It is intentionally separate from challenge generation: model-serving and scoring agents receive the public packet, while only the evaluator opens the sealed oracle.

### Submission contract

A model submission binds:
- run and model/provider/version/access-mode identity;
- evaluated checkpoint and public-packet digests;
- inference harness, prompt/decoding and exact metric-definition digests;
- a self-reported run execution receipt digest;
- model calls, input/output tokens, wall time, tool calls, human assistance and estimated compute cost;
- exactly one raw response and response digest for every public item.

The response format for this first task family is exactly one JSON object with integer fields x and y. Unknown fields, duplicate fields, floats, strings, trailing prose and code fences are invalid. An empty or whitespace-only response is an explicit abstention.

### Scoring semantics

Each output is classified as:
- correct: strict JSON pair exactly equals the sealed oracle;
- incorrect: valid JSON integer pair, but wrong answer;
- invalid_format: nonempty response does not satisfy the exact schema;
- abstained: empty/whitespace-only response.

Overall accuracy uses every released item as its denominator; invalid formats and abstentions are not correct. Coverage is the fraction with nonblank responses. The report also gives accuracy conditional on answering and accuracy conditional on valid formatting, with null when the denominator is zero.

This separation prevents a model from appearing strong by silently dropping difficult tasks or by using malformed answers that disappear from the denominator.

### CLI

~~~sh
cargo run --locked -- score-fresh \
  /secure/evaluator/fresh-public-packet.json \
  /secure/evaluator/fresh-sealed-oracle.json \
  /secure/evaluator/fresh-manifest.json \
  /secure/evaluator/model-answer-submission.json \
  /secure/evaluator/fresh-score-report.json
~~~

Schemas:
- schema/fresh-answer-submission.schema.json
- schema/fresh-instance-score-report.schema.json

The scorer validates the public packet, sealed oracle and manifest digests; validates per-item parameter/answer commitments and exact linear-system arithmetic; requires identical public/oracle item IDs; binds the submission to the frozen model checkpoint and public packet; checks every raw response digest; and rejects missing, duplicate or unknown item IDs.

The score report contains per-item status and response digest but never matrix parameters, correct answers or generation seed. It includes input artifact hashes, response/resource counters and a report digest.

### Evidence boundary

A valid exact-match score is evidence that the supplied text outputs matched the supplied sealed oracle. It does not validate task difficulty, representativeness, recipe novelty or absence from model training. The run execution receipt digest is self-reported; the report therefore always sets independently_verified=false until a separate trusted signature verifier is implemented.


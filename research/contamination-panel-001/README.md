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
  /secure/evaluator/public-packet.json \
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

T3 paraphrases are authored fixtures, not expert-validated semantic gold. T4 positives are shared-recipe controls, not proof that seeing one item contaminates a later benchmark. Both require blinded expert validation before their labels are treated as human-grounded reference truth. Keep synthetic construction truth separate from expert construct-validity adjudication.

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

The allowed decisions are same_target_same_instance, same_construct_different_instance, shared_recipe_different_instance, different_construct, and indeterminate. The current implementation requires exactly three adjudicators per pair, with distinct reviewer pseudonyms and declared independence groups. It does not authenticate identities or institutions. The blinding fields are reviewer self-attestations, not proof that reviewers could not infer the task type from its content.

Run the validator:

~~~sh
cargo run --locked -- audit-adjudications \
  /secure/evaluator/public-packet.json \
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

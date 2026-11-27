# Symthaea Benchmark Data Provenance and License Gate v0.1

**Date:** 2026-11-27  
**Status:** research operations and compliance specification; not legal advice  
**Scope:** VideoFDB, InteractionBench, and Symthaea's counterfactual audiovisual evaluation

## 1. Principle

Benchmark access is not permission for every possible use. The benchmark runner must have a data-provenance and license gate before loading or modifying a dataset. A model score is unusable as a competitive claim if the dataset, model, evaluator, or publication context was used outside its terms.

For each run, record:
- dataset name, version, immutable manifest digest, owner and source URL;
- license name/version and accepted terms;
- access-grant reference (use a non-secret opaque ID in public manifests);
- approved purpose and organization/use context;
- whether inference, modification, derivative creation, redistribution, publication, and commercial use are permitted;
- model/API terms and serving region separately from dataset terms;
- retention, deletion, and data-subject-request obligations;
- transformations applied to input media and whether those transformations are permitted.

A missing license/access record is a hard evaluation blocker, not a warning.

## 2. VideoFDB v1.0 — external evaluation only under non-commercial terms

The official NVIDIA Hugging Face dataset card identifies:
- gated, per-user access, requiring acceptance of the license;
- proprietary audio/video data licensed from NVIDIA and its licensors;
- an evaluation-only dataset with no training split;
- evaluation and benchmarking for non-commercial and research purposes only;
- a custom NVIDIA Data Evaluation License Agreement, with per-user approval.

The license grants a limited, revocable, non-transferable, non-sublicensable right to use, reproduce, modify and create derivatives solely for evaluation and benchmarking of AI solutions, subject to its restrictions. It expressly prohibits commercial use, including using the dataset for commercial model development or training. It also prohibits identifying participants, profiling them, biometric processing, sensitive-attribute inference, developing identity-recognition technology, and generating synthetic data, digital replicas, or outputs with a likeness to individuals in the dataset. The agreement permits limited illustrative excerpts only in qualifying non-commercial academic publications under its strict conditions.

References:
- [VideoFDB dataset card and access terms](https://huggingface.co/datasets/nvidia/video-full-duplex-benchmark)
- [NVIDIA VideoFDB Evaluation License](https://huggingface.co/datasets/nvidia/video-full-duplex-benchmark/blob/main/LICENSE.md)

### Symthaea usage rule

- Do not use VideoFDB for training, fine-tuning, prompt/model optimization, commercial product development, or commercial marketing.
- Do not redistribute source media or usable portions of the dataset.
- Do not perform participant identification, biometric processing, sensitive-attribute inference, or likeness generation.
- Do not produce altered-video counterfactuals from VideoFDB clips unless NVIDIA provides written clarification that the exact transformation is permitted and all other terms remain satisfied. Although the agreement permits certain evaluation derivatives, its likeness/synthetic-data restrictions make edited human-video counterfactuals an unsafe assumption.
- Where access is approved, keep evaluation-only runs separated from commercial development environments and retain auditable access/deletion records.
- If the intended Symthaea use is commercial, obtain a separately negotiated written license before using VideoFDB for that purpose. Do not assume publication of an evaluation score converts the dataset use into a permitted commercial activity.

## 3. InteractionBench — BSL 1.1 and Additional Use Grant

The official InteractionBench site states:
- the project is released under Business Source License 1.1;
- non-production use is permitted;
- its Additional Use Grant permits non-commercial academic research, education, evaluation, testing, and benchmarking in production;
- use primarily intended for commercial advantage requires a separate commercial license from AveraLabs;
- evaluated systems and models used in the harness remain subject to their own terms.

References:
- [InteractionBench site, methodology and license](https://avera-labs.github.io/InteractionBench/)
- [InteractionBench paper](https://arxiv.org/abs/2611.02430)

### Symthaea usage rule

- Preserve the exact BSL 1.1 license, Additional Use Grant, copyright notices, and version metadata.
- For use primarily intended for commercial advantage, obtain a separate AveraLabs license before running or distributing an evaluation based on the benchmark assets/harness.
- Evaluate the system/model provider's own terms independently; the benchmark license does not grant API, model, or output rights.
- Respect recorded-media privacy, data retention and deletion constraints; do not publish raw recordings or transcripts unless terms expressly allow it.
- Keep the four benchmark dimensions separate. Do not claim the benchmark establishes visual semantics, presentation authorization, safety, tool-use reliability, or long-horizon completion; the authors list several of these as outside the current evaluation.

## 4. Counterfactual audiovisual dataset: clean-room route

Symthaea's counterfactual grounding protocol requires paired conditions where audio remains byte-identical and visual evidence changes in controlled ways. Do not assume that a benchmark license authorizes all such transformations.

The preferred source is a separately commissioned or consented corpus with explicit written rights for:
- audio/video collection and processing;
- controlled visual substitutions and timing transformations;
- benchmark publication and release of non-identifying annotations;
- commercial evaluation and product-development use, where intended;
- retention and deletion schedule;
- geographic processing/storage and cross-border transfer, where applicable.

Safer stimulus design:
- use clearly synthetic, non-identifiable avatars or abstract animated actors whose assets are independently licensed for the intended use;
- do not imitate the face, voice, gait, or identity of a real benchmark participant;
- annotate interaction cues, not demographic or biometric attributes;
- retain only media hashes, timing metadata, cue labels, and authorized evaluation outputs in public manifests;
- store raw media in access-controlled storage, outside public Git history;
- include an opt-out, deletion, and incident-response process.

A separate synthetic or consented corpus should carry its own dataset card, license, consent/rights record, and media manifest. The word "synthetic" alone does not establish that a dataset is free of rights or privacy obligations.

## 5. Machine-readable run manifest additions

Every benchmark run should include a license/provenance block, conceptually:

```json
{
  "dataset_id": "video-fdb-v1.0.0",
  "dataset_manifest_digest": "<sha256>",
  "license_id": "nvidia-video-evaluation-license",
  "license_version": "accepted-version",
  "access_grant_ref": "opaque-access-reference",
  "approved_purpose": "non-commercial-evaluation",
  "commercial_use_authorized": false,
  "training_use_authorized": false,
  "derivative_media_authorized": false,
  "model_terms_ref": "provider-terms-version",
  "retention_policy_ref": "policy-id",
  "raw_media_stored_in_repo": false
}
```

This is a schema sketch, not a valid access credential or proof of legal approval. Sensitive grant records and contract files must remain in private compliance storage; only an opaque reference and approved scope belong in a public run manifest.

The runner must fail closed when:
- license version is absent or no longer matches the approved grant;
- intended use exceeds approved purpose;
- the model/API terms do not permit the planned inference or output use;
- a transformation is not explicitly allowed;
- required consent, retention, or deletion controls are absent;
- an expired or revoked access grant is presented.

## 6. Competitive strategy under these constraints

1. Use official VideoFDB and InteractionBench only for the non-commercial/research comparisons their terms permit, unless a separate written license is obtained.
2. Develop Symthaea's own commercially cleared counterfactual audiovisual dataset for ongoing product development and public claims.
3. Keep the two result families separate. Do not mix proprietary benchmark scores and in-house scores into one leaderboard.
4. Report exact dataset versions, access status, model/provider terms, transformations, hardware, and evaluation protocol.
5. Where rights are unclear, pause that dataset operation and use a clean-room substitute rather than relying on an optimistic interpretation.

The competitive objective remains matched Pareto improvement in interaction quality, timing, visual grounding, semantic invariance, accessibility, privacy, and output assurance. Rights compliance is a hard gate, never a score that a strong model result can offset.

## 7. Qualification status

This document records a conservative interpretation of the public benchmark terms as displayed on 2026-11-27. It is not legal advice and does not replace reviewing the full license/contract with qualified counsel. No dataset access has been requested or granted by this implementation. No licensed benchmark media has been downloaded, modified, or run through Symthaea as part of this work.

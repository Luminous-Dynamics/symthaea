# MEL auditory invariance qualification

This nested Rust package imports the production temporal-grounding scorer by path and contains focused, deterministic evaluation adapters. It is not a model-serving runtime.

## Musical FLIP/STAY

The auditory_invariance module is a MusicListenBench v1.0 consumer. It accepts one clean/FLIP/STAY triplet per underlying item, with:
- normalized musical relations (Same/Different or Clip A/B faster);
- per-question A/B semantic mappings;
- source lineage, audio and question digests;
- model, prompt, runtime and decoding identities;
- separate FLIP and STAY intervention factors.

The evaluator converts each A/B response into its semantic relation before scoring. It reports clean accuracy, FLIP miss rate, STAY false-flip rate, clean-correct conditional error rates, Acc_FS, three-way joint correctness, answer bias, exact always-A/always-B Acc_FS baselines under each question's own option mapping, parsing and abstention coverage. All metrics are reported overall and by task/STAY factor.

Run the synthetic replay:

    cargo +1.96.0 run --locked --manifest-path tools/mel-temporal-grounding-qualifier/Cargo.toml --bin mel-auditory-flip-stay-audit -- tools/mel-temporal-grounding-qualifier/fixtures/mel-listenbench-flip-stay.synthetic.json

This evaluator does not load audio, authenticate model inference receipts or establish that transformations were correctly materialized. The synthetic fixture validates metric semantics only.

## External benchmark

MusicListenBench v1.0 (arXiv:2611.29375, submitted 22 Nov 2026):
- https://arxiv.org/abs/2611.29375
- https://huggingface.co/datasets/amaai-lab/MusicListenBench
- https://amaai-lab.github.io/MusicListenBench/
- https://github.com/AMAAI-Lab/MusicListenBench/tree/release

The official benchmark contains clean, FLIP and STAY variants for melody, harmony, timbre and rhythm, with rendered and real-recording test splits. Preserve its exact question wordings, option semantics, official metrics and immutable dataset revision when building the real manifest. Do not copy the 7.47 GB corpus into this repository.

## Claim boundary

A green Rust test or synthetic CLI replay proves only that the adapter validates and scores the supplied records correctly. It does not establish a Melothaea model result, human-level musical hearing, or a qualified nuisance transformation. Those require exact benchmark data, independently verified inference/run receipts, and (for new transformation families) task-specific qualification through #13260.


## Import an official MusicListenBench v1.0 submission

The official release uses:
- data/eval.json: 1,000 rendered CLEAN items;
- data/eval_perturb.json: 2,000 rendered FLIP/STAY items;
- data/eval_perturb.meta.jsonl: stable base item ID, task, condition, factor, gold and clip-path metadata;
- one official JSONL submission with a required metadata header and one answer record per variant item ID.

After downloading and verifying the official release and WAV archives, run:

    cargo +1.96.0 run --locked --manifest-path tools/mel-temporal-grounding-qualifier/Cargo.toml --bin mel-listenbench-import -- data/eval.json data/eval_perturb.json data/eval_perturb.meta.jsonl submission.jsonl data/audio data/audio_perturb report.json

The importer:
- joins variants by exact base item ID from released item IDs and metadata, not by guessed file ordering;
- cross-checks each perturbation row against the companion metadata JSONL;
- supports the released same/different and faster/slower question templates, mapping A/B to semantic musical relations before scoring;
- hashes the actual local WAV bytes and stores per-variant hashes;
- parses the official JSONL answer header and item IDs;
- counts missing/invalid answers as incorrect, while reporting missing-answer counts; unknown IDs, duplicate IDs, incomplete triads and metadata mismatches fail closed;
- records official submission metadata and an exact input-manifest fingerprint.

Standard MusicListenBench submissions do not require system-prompt, runtime or decoding receipts. The imported report leaves those fields absent rather than manufacturing them; model/version metadata is self-reported, not authenticated. The input_manifest_sha256 fingerprints the three exact local JSON files; it is not claimed to be a signed upstream dataset revision.

The importer handles the rendered 3,000-item test set in V1. The real-recording data require a separate mapping/rights adapter and must not be silently pooled with rendered results.

The released audio corpus is about 7.47 GB. Do not commit it into this repository. Keep it in the user's local dataset cache and retain the official release revision, verify_data.py output, per-file SHA-256 manifest, FluidSynth/FFmpeg versions and submission receipt alongside the generated report.

Official source: https://github.com/AMAAI-Lab/MusicListenBench/tree/release

# MUSE-MIR-020 — Rights-aware Music Corpus Qualification (V1)

**Issue:** #11951  
**Parent:** MUSE-MIR-019 multidimensional MusicXML score-event evaluator  
**Date:** 13 November 2026  
**Status:** registry, validator and focused tests authored; not compiled or executed

## Why corpus qualification is now the bottleneck

MUSE has a source-native MusicXML event graph and a deterministic score-event evaluator. A benchmark claim still requires a reproducible corpus: exact source release, declared license, allowed use, sample IDs, source/reference digests, composer/work split policy, conversion and annotation protocol, parser/evaluator builds and independent evidence.

The public datasets are not one interchangeable pool. They differ in modality, rights, annotation quality, split design, and task. V1 makes those distinctions executable.

## Candidate registry

The Rust function `candidate_musicxml_corpus_registry_v1()` records four candidates. It does not contain sample assets, and it does not imply that any candidate is fully qualified.

| Corpus | V1 registration | Primary value | Allowed project use |
|---|---|---|---|
| OSSQ-OMR / OpenScore String Quartets | Source revision `7a17e45cddc0b7064fc3a179b62caeb57595e993`; CC0-1.0 source evidence | Multi-part quartet notation, MusicXML/MuseScore references, synthetic/scanned score-image OMR | Commercial and non-commercial research, subject to verifying the pinned release and acquired files |
| ASAP | v1.1 tag, source revision `fad8d1e8078d0ae47ad2f280b5d022bd2de24784`; CC BY-NC-SA 4.0 | MusicXML/quantized MIDI scores and multiple aligned piano performances | Non-commercial research only; attribution/share-alike obligations |
| MAESTRO | v3.0.0, official archive SHA-256 `6680fea5be2339ea15091a249fbd70e49551246ddbd5ca50f1b2352c08c95291`; CC BY-NC-SA 4.0 | Piano audio/MIDI performance events and fine alignment | Non-commercial research only; not a multi-part MusicXML reference corpus |
| MIREX 2026 Quartets | External task terms; corpus license left Unknown; task-page revision not yet pinned | External audio-to-score leaderboard with Blind and Staves-Informed tracks | External evaluation only; no training/tuning on held-out excerpts; no redistribution claim |

Sources:
- OSSQ-OMR dataset and license: https://github.com/MALerLab/ossq-omr
- OSSQ-OMR benchmark protocol: https://github.com/MALerLab/string-quartet-omr-benchmark
- ASAP current dataset manifest and license: https://github.com/fosfrancesco/asap-dataset
- MAESTRO v3.0.0: https://magenta.withgoogle.com/datasets/maestro
- MIREX 2026 task: https://music-ir.org/mirex/wiki/2026:Audio-to-Score_Transcription

### Important ASAP version discrepancy

Do not mix the paper, the pinned v1.1 tag and the newer default branch:
- original ASAP paper: 222 scores, 1,068 performances and 520 audio performances;
- pinned v1.1 tag (commit fad8d1e8078d0ae47ad2f280b5d022bd2de24784): 236 scores, 1,068 performances and 520 audio performances; the README identifies 29 not-aligned performances;
- current default branch at the time of registry authoring: 236 scores, 1,067 performances and 519 audio performances; its README identifies 31 not-aligned performances.

MUSE's registration uses the immutable v1.1 tag, not the moving default branch. Preserve the three counts as distinct provenance snapshots. The exact v1.1 file manifest and sample inventory must be frozen before any score or performance count is reported.

The pinned v1.1 README also warns that beat positions can be unresolved for pickups, rubato and complex embellishments. These are data statuses, not examples to silently discard or relabel as correct.

### OSSQ-OMR split strategy

The 2026 release contains 116 string-quartet scores by 47 composers, with 24,544 system images and 98,172 staff images, synthetic and scanned inputs, and symbolic annotation formats. The published protocol uses four mutually exclusive random score-level splits. That is a useful matched OMR baseline, but it is not a composer-held-out generalization test.

The registry therefore defines:
- official-style synthetic and scanned score-image tracks, with work-disjoint splits;
- a separate MUSE composer-held-out OOD track, requiring both work and composer disjointness within each fold.

The composer-held-out track is a proposed MUSE extension, not an official OSSQ-OMR split or published result. With only 47 composer groups, report fold composition and confidence intervals at composer level; never treat the thousands of segmented page/staff images as independent works.

## V1 data model

The module `musicxml_corpus_qualification.rs` defines:

- `CorpusRegistrationV1`: corpus/release identity, source URL, immutable revision ID, declared license, project use scope, rights-evidence status/URL/notes, and separate task tracks.
- `CorpusTrackRegistrationV1`: input/reference modalities, task, train permission, redistribution permission, split policy and external protocol label.
- `CorpusSampleManifestV1`: per-sample ID, fold, split role, composer/work group IDs, safe relative input/reference paths, and their expected SHA-256 digests.
- `CorpusProtocolFingerprintV1`: frozen split, annotation, conversion pipeline, parser build, evaluator build and evaluator schema identities.
- `CorpusByteVerificationReceiptV1`: hashes of the actual input/reference byte slices supplied per sample, bound to the sample-manifest digest and declared verifier-build digest.
- `CorpusQualificationReportV1`: canonical registration/sample/protocol/byte-receipt commitments, sample/fold/group counts, issues, stage and a deterministic report commitment.

## Rights and qualification semantics

License labels are not treated as informal tags:
- CC0-1.0 can be registered for commercial and non-commercial research only when source evidence says so.
- CC BY-NC-SA 4.0 is non-commercial-only in this registry; it cannot be marked commercial-compatible.
- ExternalEvaluationTerms or an Unknown dataset license can only be registered as ExternalEvaluationOnly. That scope cannot permit training or redistribution.
- Unverified scope and missing rights evidence block promotion.

The registry is not legal advice. It records the dataset publisher's declared terms and the project's elected use scope. A legal review may still be necessary for particular assets or use cases.

### Qualification stages

V1 emits these stages:

1. `Candidate` — registration/rights metadata is invalid or incomplete (for example, MIREX's task page is not yet pinned to a stable content revision).
2. `MetadataRecorded` — corpus release and rights metadata are recorded, but no sample manifest exists.
3. `SampleDigestsDeclared` — a manifest exists, but sample identity, path, digest syntax or split issues remain.
4. `SplitQualified` — declared input/reference digests are syntactically valid and work/composer/artifact leakage checks pass.
5. `ProtocolFrozen` — split, annotation, conversion, parser and evaluator fingerprints are present and valid.
6. `ByteVerified` — the supplied input/reference byte slices for every sample have actually been SHA-256 hashed and match the pinned sample manifest; no protocol fingerprint is required for this stage.
7. `ProtocolAndBytesVerified` — both the protocol fingerprints and actual sample bytes have been verified.

There is deliberately no `FullyQualified` status. Even `ProtocolAndBytesVerified` does not authenticate who supplied the files, prove that the declared relative paths were resolved from an authentic upstream release, adjudicate the annotations, or provide legal review.

The byte API is `verify_corpus_artifact_bytes` / `qualify_corpus_track_with_artifact_bytes`. The caller supplies one input/reference byte pair per sample through an iterator, so it can load and release one pair at a time. The verifier hashes those exact slices, compares them with the sample manifest, checks the supplied path labels match the manifest, and emits a deterministic receipt bound to the sample-manifest hash and a declared verifier-build hash. `validate_corpus_byte_verification_replay` recomputes the hashes from the same files and compares the complete receipt.

Paths are required to be normalized relative paths: no absolute paths, empty components, dot/dot-dot components, drive/URI colons, backslashes or NUL. This is lexical validation only. The caller remains responsible for resolving each relative path beneath a controlled, trusted corpus root and supplying bytes read from the intended release. The receipt is a content commitment, not a signature or proof of upstream authenticity.

## Split leakage controls

For each fold, the validator enforces the configured policy:
- work-disjoint means one work group cannot appear in multiple split roles;
- composer-disjoint means one composer group cannot appear in multiple split roles;
- exact-artifact-disjoint means input/reference hashes cannot recur across split roles.

Duplicate sample IDs, malformed lowercase SHA-256 digests, unsafe relative paths, missing required group identifiers, wrong corpus/release/track IDs, and empty manifests are blockers. The direct byte verifier repeats its own schema, duplicate-ID, path and digest checks rather than relying exclusively on the higher-level track qualifier. Exact content reuse between train and test is rejected even when group labels differ.

Five-fold composer-disjoint validation is representable, but this code does not generate the folds. The split generator and seed/assignment protocol must be independently frozen and committed before qualification.

## Tests authored

18 focused tests currently cover:
- candidate registry scope separation;
- CC BY-NC-SA refusal for commercial-reuse declarations;
- Unknown license refusal outside external-evaluation-only;
- metadata-only candidates not promoted to digest/byte qualification;
- composer and work leakage refusal;
- exact input/reference artifact reuse refusal across splits;
- clean manifest + protocol promotion to ProtocolFrozen;
- leakage remaining below SplitQualified;
- malformed digests, duplicate IDs and wrong-track binding;
- the unpinned MIREX task page remaining Candidate and non-commit revision strings being refused;
- report commitment tamper sensitivity and task/modality separation;
- hashing actual input/reference bytes, receipt replay and tamper rejection;
- missing/duplicate sample coverage, digest mismatches and unsafe relative paths;
- duplicate IDs and malformed manifest refusal in the direct byte-verification API.

Tests and the focused GitHub workflow are authored but not compiled or executed in the authoring environment. The workflow now includes the corpus-qualification module and its focused test filter.

## Benchmark qualification targets (not achieved results)

- 100% sample ID, safe relative path, and input/reference digest coverage for every counted example.
- 100% actual byte matches to the pinned manifests before any ByteVerified stage is issued.
- Zero composer/work/exact-artifact leakage across required split boundaries.
- Zero test-to-train or test-to-tune reuse.
- 100% explicit license/use-scope classification; Unknown never means unrestricted.
- Separate OMR, symbolic score-event, audio-to-score, score-performance alignment and audio-to-MIDI tracks.
- Composer-clustered confidence intervals and human correction time only after splits and protocols are frozen.

## Next steps beyond V1

1. Download/prepare only the OSSQ-OMR CC0 annotation sources first; compute a local artifact manifest from actual bytes and pin the derived conversion environment (MuseScore3 v3.6.2 is required by the source repository).
2. Create the composer-held-out fold assignments from composer IDs, publish the split manifest hash before any tuning, and preserve the four official random score-level splits as a separate comparable track.
3. Run parser coverage and the MUSE-MIR-019 score-event evaluator on the reference graph itself as a zero-error round-trip control; unexpected Unknown results must be counted as coverage failures, not filtered away.
4. Add the ASAP/MAESTRO non-commercial performance tracks only in a distinct research configuration; preserve unaligned/uncertain items as explicit unsupported strata.
5. Pin the MIREX task page/version before registering external benchmark results. Keep Blind, Staves-Informed, familiar Quartets and composer-held-out QuartetsMIREX separate.

No cross-corpus aggregate score or commercial rights claim is produced by this module. The byte receipt is not a signature; independent upstream release verification, rights review and annotation adjudication remain separate requirements.

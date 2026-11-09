# Speech cohort manifest v0.1

This document describes the fail-closed metadata admission layer implemented by `symthaea-stt::cohort_manifest`.

## Validate the synthetic design example

From the repository root:

```bash
cargo run -p symthaea-stt --bin stt-cohort-validate -- \
  docs/research/voice/cohort_manifest_design_example_v0.1.json
```

The command validates the cohort manifest and **every referenced counterfactual pair artifact**. It resolves artifact paths relative to the cohort manifest, rejects absolute/traversing paths, rejects symlink escapes after canonicalization, parses the pair identity, and recomputes SHA-256 over the exact artifact bytes. A missing artifact, mismatched pair/trial identity, or digest mismatch fails the command.

To require permission for every record to be sent to an external model/API:

```bash
cargo run -p symthaea-stt --bin stt-cohort-validate -- \
  docs/research/voice/cohort_manifest_design_example_v0.1.json \
  --require-external-processing
```

The supplied synthetic example deliberately has external processing disabled, so the second command should fail closed. Permission to evaluate locally and permission to upload to a provider are separate authorizations.

## Admission rules

- Every record has a unique ID, source-family ID, utterance-family ID, language/variety tags, source and evaluation audio SHA-256 digests, source/annotation manifest digests, and explicit rights/consent metadata.
- Derived audio requires a derivation-manifest digest and explicit derivative-processing authorization.
- Confirmatory records require two distinct pseudonymous reviewers, an adjudication-manifest digest, and known speaker/session cluster IDs.
- A phonological profile marked QUALIFIED requires its exact profile ID/revision, a native-review manifest digest, and independent reviewers.
- Source-family, speaker, session, utterance-family and exact audio-digest groups may not cross splits.
- Each record may occupy one pair role only. Pair purpose must agree with split; pair IDs and artifact paths are unique; each pair binds its exact #11157 pair-manifest file by SHA-256.
- `--require-external-processing` is a separate opt-in gate and does not change local evaluation permission.

The cohort digest is BLAKE3 over canonical JSON after sorting records by record ID and pairs by pair ID. Ordering within nested arrays (for example, reviewer IDs or language tags) remains significant.

## What a pass does not establish

A structural PASS confirms only that the supplied metadata and pair artifacts satisfy these checks. It does not independently prove that:
- a licence or consent assertion is authentic or legally sufficient;
- the supplied rights scope permits a particular commercial use (legal review remains necessary);
- a reviewer is genuinely independent, qualified or a native speaker;
- speaker/session/source cluster IDs are complete or that near-duplicates were found;
- transcript, phonological or audio-truth annotations are correct;
- an acoustic transformation preserved meaning;
- a model ran, or any speech capability was measured.

The example and pair files under `cohort_manifest_design_example/` are synthetic conformance artifacts only. They are not real participant or audio data. Real cohort admission belongs to #11175 and requires source-specific rights, human review, and a frozen, source-disjoint split before model evaluation.

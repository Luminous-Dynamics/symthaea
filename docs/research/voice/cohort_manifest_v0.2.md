# Speech cohort manifest v0.2

This revision adds byte-pinned source and annotation evidence artifacts to the v0.1 cohort admission contract. The older v0.1 schema and example remain available as historical design snapshots.

## Validate the synthetic design example

From the repository root:

```bash
cargo run -p symthaea-stt --bin stt-cohort-validate -- \
  docs/research/voice/cohort_manifest_design_example_v0.2.json
```

The command validates cohort structure, split membership and authorization metadata, then recomputes exact SHA-256 digests for every source manifest, annotation manifest, optional derivation manifest and counterfactual-pair manifest referenced by the cohort.

All paths must be safe relative paths under the cohort manifest directory. Absolute paths, traversal, missing files, digest mismatches and symlink escapes fail closed. Cohort-level evidence files may be referenced by multiple records when the same exact path and digest are used.

To require authorization for external model/API processing:

```bash
cargo run -p symthaea-stt --bin stt-cohort-validate -- \
  docs/research/voice/cohort_manifest_design_example_v0.2.json \
  --require-external-processing
```

The synthetic example intentionally disallows external processing; the second command should fail.

## What the digest check establishes

It establishes byte identity between the manifest's expected SHA-256 and the file found at the pinned relative path. It does **not** parse or prove the truth of the source/annotation document, establish legal sufficiency, prove reviewer independence, validate that an acoustic transform preserved meaning, or run a model. Those remain separate admission and scientific-review gates.

The included files are synthetic conformance fixtures only. No participant audio or real annotation data is included.

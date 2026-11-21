# CETB defensible multiverse manifest

This directory defines and validates a machine-readable contract for analysis pipelines that are independently justified before confirmatory outcomes are accessed.

## Files

- `multiverse.schema.json`: Draft 2020-12 manifest schema.
- `fixtures/synthetic-multiverse.json`: a synthetic example with two estimator semantics and two segment-duration choices.
- `validate_manifest.py`: semantic validation for unique IDs, option references, confirmatory freeze integrity, primary-lane presence and deterministic sampling declarations.
- `test_validate_manifest.py`: negative and positive contract tests.

## Run

```sh
python -m unittest discover -s research/cetb/multiverse -p 'test_*.py' -v
python research/cetb/multiverse/validate_manifest.py \
  research/cetb/multiverse/fixtures/synthetic-multiverse.json \
  research/cetb/multiverse/multiverse.schema.json
```

The manifest separates option defensibility from equivalence. It does not enumerate or execute neuroimaging pipelines yet; it locks the scientific decision space and rejects malformed or post-outcome confirmatory declarations. See `docs/neuroscience/CETB_DEFENSIBLE_MULTIVERSE_PROTOCOL_V0.md`.

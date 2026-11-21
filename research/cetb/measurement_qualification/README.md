# CETB multidimensional measurement qualification receipt

This directory contains the Draft 2020-12 schema and a synthetic fixture for recording four separate evidence axes:

1. Measurement sufficiency.
2. Participant-level repeatability.
3. Construct validity.
4. Transportability, separated into participant holdout, site holdout and independent-cohort validation.

The contract explicitly forbids a single composite score. Absence of evidence is represented as `not_assessed`, not as a pass or fail.

The fixture is synthetic and intentionally leaves the empirical axes unassessed. See `docs/neuroscience/CETB_MEASUREMENT_QUALIFICATION_STACK_V0.md`.


## Semantic validation

Run the schema and evidence-gate validator:

```sh
python -m unittest discover -s research/cetb/measurement_qualification -p 'test_*.py' -v
python research/cetb/measurement_qualification/validate_receipt.py \
  research/cetb/measurement_qualification/fixtures/synthetic-qualification.json \
  research/cetb/measurement_qualification/measurement-qualification.schema.json
```

The validator refuses to promote:
- a metadata-only source to empirical signal analysis;
- repeatability to adequate without a participant-level estimate and interval;
- independent-cohort validation from a new split or a held-out site in the same release;
- a site-holdout status from a random participant split;
- construct validity without positive and negative controls.

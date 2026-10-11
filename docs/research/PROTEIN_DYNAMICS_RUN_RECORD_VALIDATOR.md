# Protein Dynamics Run-Record Validator

This tool validates the structure and integrity of a recorded protein-dynamics benchmark run. It is deliberately separate from CI status and from scientific interpretation.

It checks the versioned JSON Schema, exact frozen target-manifest hash, one-row-per-target accounting for completed runs, duplicate/unregistered target IDs, finite metric values, honest metric-role/observable-overlap labels, retained failure codes, conformer accounting, local output hashes, and path traversal. It does **not** prove that the measurements are biologically correct, that an observable was truly hidden from the controller, or that a source hash establishes provenance authenticity.

## Prerequisites

- Python 3.11 or newer
- jsonschema >=4.18,<5

For an isolated development environment:

~~~bash
python3 -m pip install 'jsonschema>=4.18,<5'
~~~

## Freeze the expected target list

Create a JSON file containing exactly one object with a target_ids array. IDs must be unique and sorted lexicographically. Example shape:

~~~json
{"target_ids":["TARGET-001","TARGET-002"]}
~~~

Compute SHA-256 over the **exact bytes of this file**, including whitespace and final newline. Put that digest in preregistration.target_manifest_sha256 before execution. Do not reformat the target manifest after freezing it. The protocol/metric/stopping-rule hashes should likewise refer to immutable frozen files.

## Validate a run record

~~~bash
python3 scripts/research/validate_protein_dynamics_run.py \\
  runs/protein-dynamics-example/run-record.json \\
  --expected-targets runs/protein-dynamics-example/expected-targets.json \\
  --artifact-root runs/protein-dynamics-example/artifacts
~~~

Artifact paths in output_artifacts must be relative to --artifact-root and resolve inside it. The validator reads each artifact locally and verifies its SHA-256. Remote URLs and absolute paths are refused because this command will not silently pretend to have verified bytes it did not read. Store copies of logs, metrics, weights, environment reports, and the exact manifest inside the run bundle as appropriate.

Exit code 0 means the **record checks** passed; it does not mean the benchmark passed. A non-complete record may pass with a warning that some frozen targets remain unaccounted for. A complete record must contain an exact target census, no NOT_RUN rows, all required attestations, and at least one hashed output artifact. A completed run can still contain retained, explicitly classified target failures.

## Exercise the validator

~~~bash
python3 -m unittest discover -s scripts/research -p 'test_validate_protein_dynamics_run.py' -v
~~~

When run in the repository, the tests use the production schema at docs/research/PROTEIN_DYNAMICS_RUN_RECORD_SCHEMA_v1.json. When the test module is copied outside the repository for isolated semantic testing, it falls back to a minimal JSON Schema and therefore does **not** test production-schema compatibility.

The 16-test suite includes adversarial cases for target omissions and identity types, target-manifest hash mismatch, output hash mismatch, non-finite metrics, missing failure codes, mislabeled held-out metrics, duplicate target rows, path traversal and remote artifacts, conformer/resource mismatches, malformed preregistration, preregistration ordering, and NOT_RUN rows in a claimed complete run.

## Limitations and next gate

This is a first local verifier, not a trusted scientific authority. The string-valued attestations about hidden observables and retained failures still require reviewable process evidence. It does not yet verify source-input bytes because upstream inputs are represented by URIs and declared hashes, and it does not establish a reproducible runtime for PeptoneBench. The next research gate remains a pinned, independently repeatable reference case with retained inputs/outputs and explicit optimizer status.

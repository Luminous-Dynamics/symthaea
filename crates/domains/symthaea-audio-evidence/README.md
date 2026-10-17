# symthaea-audio-evidence

Fail-closed structural and artifact-integrity checks for the Open Audio Evidence manifest used by Luminous Open Audio.

## Usage

From the repository root:

~~~sh
cargo run -p symthaea-audio-evidence --bin audio-evidence-check -- schemas/examples/h1-fit-repeatability-plan.json
~~~

For manifests whose artifact paths are relative to a separate bundle root:

~~~sh
cargo run -p symthaea-audio-evidence --bin audio-evidence-check -- path/to/manifest.json path/to/bundle-root
~~~

The checker verifies project-specific required fields and lifecycle invariants, refuses premature pass/fail states, requires fixture/instrument/environment metadata and a raw capture once a run has started, rejects unsafe artifact paths, and verifies each referenced file's exact SHA-256 and optional byte length. It rejects artifact symlinks that resolve outside the bundle root.

The embedded manifest schema must accompany this crate at schemas/audio-evidence-manifest.schema.json. The checker confirms the declared JSON Schema draft and schema version but is **not a complete JSON Schema Draft 2020-12 implementation**. Use a standards-compliant validator separately for complete schema validation.

## Status semantics

- planned: no captures/results and decision must be not_evaluated.
- captured / analyzed: cannot claim gate pass/fail.
- validated: requires analysis and results, plus an explicit pass, fail or inconclusive project-gate decision.
- invalid / excluded: retain a reason in limitations; never represent these as passing results.

A successful check proves only that project invariants and listed artifact hashes are consistent. It does not prove that a microphone is calibrated, that physical measurements are correct, that a product meets IEC/CTA/AES requirements, or that a listening claim is valid.

## Group-split integrity for preference benchmarks

The `hifi-benchmark-check` command validates a declared train/validation/calibration/test split and its group-disjointness rules:

~~~sh
cargo run -p symthaea-audio-evidence --bin hifi-benchmark-check -- schemas/examples/hp-pref-001-split-manifest.json
~~~

The checked-in example is explicitly synthetic. The format is specified by [hifi-split-manifest.schema.json](../../../schemas/hifi-split-manifest.schema.json). The validator rejects missing identity fields, duplicate sample IDs, unknown roles, absent split policy, missing regime-specific holdout dimensions and declared test-group overlap. It allows calibration/test listener overlap when the manifest intentionally permits personalization. It does **not** evaluate predictions, establish that a study is unbiased, or prove audio quality.

See [HIFI_BENCHMARK_PROTOCOL_2026-10-17.md](../../../docs/research/HIFI_BENCHMARK_PROTOCOL_2026-10-17.md) for benchmark regimes, metrics, negative controls and the decision rule.

## Tests

~~~sh
cargo test -p symthaea-audio-evidence
~~~

Tests include a planned-only example, premature pass rejection, missing setup rejection, unsafe path rejection, exact file digest verification, mutation detection and (on Unix) symlink-escape rejection.

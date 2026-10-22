# Mobile benchmark experiment record v1

**Purpose:** input contract for `mobile-benchmark-analyze`  
**Status:** schema/analysis tooling only; no physical device result is implied.

## Run the analyzer

From a repository checkout with the normal Rust toolchain and workspace dependencies available:

```bash
cargo run -p symthaea-soma --release --bin mobile-benchmark-analyze -- --help
```

The synthetic smoke fixture is intentionally too small for qualification and can only exercise the diagnostic path:

```bash
cargo run -p symthaea-soma --release --bin mobile-benchmark-analyze -- \
  --exploratory --bootstrap-reps 500 \
  docs/engineering/mobile/fixtures/mobile_benchmark_analyzer_smoke_v1.json
```

It should emit a JSON summary with `measurement_origin=synthetic_fixture`, `energy_gate_evaluable=false`, and `declared_gates_pass=false`, even though the synthetic candidate data has lower nominal energy. This is a fixture expectation, not a claimed run result.

## Record shape

The Rust binary deserializes a JSON object with these fields:

| Field | Meaning |
|---|---|
| `schema_version` | Must be `1.0` |
| `experiment_id` | Stable unique identifier for this experiment |
| `measurement_origin` | `physical_device` or `synthetic_fixture` |
| `scenario_corpus_sha256` | SHA-256 of the exact frozen scenario corpus bytes |
| `scoring_code_sha256` | SHA-256 identity for the scoring/analyzer version used by the acquisition workflow |
| `raw_samples_sha256` | SHA-256 of the canonical raw paired sample record |
| `baseline_config_sha256`, `candidate_config_sha256` | Exact configuration identities used in paired arms |
| `device.model`, `device.soc`, `device.os_build` | Device and platform identity |
| `device.unit_token` | Experiment-scoped device label, not a serial number or user identifier |
| `device.runtime_id`, `device.application_build_sha256` | Runtime identity and exact app build hash |
| `device.energy_measurement_method` | `calibrated_instrument`, `os_power_proxy`, `battery_percentage_estimate`, or `synthetic_fixture` |
| `device.calibration_id` | Required for `calibrated_instrument`; otherwise null/omitted |
| `samples[]` | Paired observations for the same device and task/scenario repetition |

Each sample contains a unique `pair_id`, `scenario_id`, and `baseline` / `candidate` measurements. Each arm must record positive finite joules and latency, whether the task succeeded, severe thermal seconds during the paired exposure, and a privacy violation count.

## Statistical and decision semantics

- Standard mode requires at least **30 paired samples per scenario**. `--exploratory` lowers the minimum to two so developers can test the pipeline; it hard-disables qualified pass status.
- Bootstrap resampling is paired and stratified by scenario, preserving the sample count per stratum. The seed and valid bootstrap repetition count are included in the report.
- Energy per successful task is total energy divided by independently scored successful tasks; zero-success arms are rejected, not assigned a convenient value.
- The p95 uses a documented nearest-rank definition. Task success, latency, energy, severe thermal duration and privacy are reported independently.
- An energy gate is evaluable only for physical-device runs with calibrated instrumentation and a non-empty calibration identity. OS proxies and synthetic data may produce diagnostics but cannot qualify an energy claim.
- One privacy violation hard-fails the qualification gate regardless of energy gain.
- `qualified_pass` is a narrow benchmark gate—not a general claim of industry superiority, battery-life improvement, device security certification, or consciousness.

The analyzer does not authenticate calibration records, device attestations, meter traces, or supplied digests; the acquisition pipeline must hash and preserve the actual canonical artifacts before analysis. The JSON report explicitly emits `evidence_receipt_verified=false` and `declared_gates_pass` must not be described as independently verified qualification. Hash-shaped placeholder values in the synthetic fixture are only test identifiers.

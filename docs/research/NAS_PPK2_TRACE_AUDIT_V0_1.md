# Raw PPK2 trace verification and energy reconstruction v0.1

Tracking: #9740  
Implementation: `symthaea-architecture-engineering::ppk2_trace_audit`

## Purpose

NATS-Bench-MCU v2 publishes aggregate latency, current, power and energy summaries and separately provides raw PPK2 current traces. This verifier recomputes energy from the raw timestamped samples instead of treating the aggregate result as the only authority.

The upstream dataset helper exports `ppk2_samples.parquet` to CSV. The MIMaaS visualization implementation identifies the relevant raw columns as:

- `segment_id`
- `time_ms`
- `current_uA`

The released traces are sampled at a nominal 100 kHz. The verifier checks the median timestamp-derived sample rate but integrates using the actual recorded time deltas.

## Energy calculation

For adjacent samples (i) and (i+1), current in microamperes and time in milliseconds:

```text
integral_uA_ms += ((current_i_uA + current_next_uA) / 2)
                  * (time_next_ms - time_i_ms)

energy_uJ = integral_uA_ms * voltage_V / 1000
mean_current_uA = integral_uA_ms / (last_time_ms - first_time_ms)
mean_power_uW = mean_current_uA * voltage_V
```

The unit conversion is:

`µA × ms × V = nJ`, therefore divide by 1,000 for `µJ`.

Each segment is integrated separately. Aggregate inference statistics are arithmetic means over segment-level metrics; segment durations, energy range, median interval, median sample rate, and maximum gap ratio remain visible.

This is a trapezoidal reconstruction over sample timestamps. It does not assume a perfectly uniform sampling interval or silently replace missing samples with interpolated values.

## Fail-closed checks

- Required source columns exist and headers are unique.
- Segment IDs are non-empty.
- Timestamps and current values are finite.
- Current is non-negative under this inference-energy contract.
- Each segment has at least two samples.
- Timestamps strictly increase within each segment.
- Median observed sample rate stays within the configured tolerance of nominal 100 kHz.
- Maximum interval / median interval stays within the configured gap ratio.
- Voltage, sampling rate and gap policy are finite and within valid ranges.
- Source CSV bytes and canonical derived receipt are content-addressed.

A trace with a large gap, duplicate/reversed timestamp, non-finite sample or rate mismatch is rejected. It is not repaired or given an inferred energy value.

## Command

The voltage is mandatory in the CLI because the energy calculation depends on it and the raw trace does not itself establish the MCU supply voltage.

```bash
cargo run -p symthaea-architecture-engineering --bin ppk2_trace_audit -- \
  ppk2_samples.csv <voltage_v> ppk2-audit-receipt.json
```

The CLI defaults to the dataset's nominal 100 kHz sample rate, 20% median-rate tolerance, and 2.5× maximum-gap ratio. The caller must ground the voltage from the device/acquisition protocol and should freeze these settings before looking at comparative outcomes.

## Receipt and evidence boundary

The receipt includes:
- exact CSV byte digest and byte length;
- explicit voltage and sample-rate/gap policy;
- segment identifiers and sample counts;
- per-segment duration, current range, time-weighted average current, mean power, energy, median interval, maximum interval, sample rate and gap ratio;
- aggregate means and energy range;
- a domain-separated digest of the canonical receipt.

This hashes the CSV representation supplied to the parser. It is not the original Parquet digest and does not authenticate the PPK2 acquisition. When a real run is audited, record the original Parquet digest, archive digest, converter script/toolchain digest, and resulting CSV digest together.

## Tests authored

Synthetic tests cover:
1. constant-current exact energy;
2. linear-ramp trapezoid integration;
3. irregular timestamps using actual deltas;
4. multi-segment separation and aggregation;
5. duplicate headers and missing columns;
6. duplicate and reversed timestamps;
7. NaN and negative current;
8. short traces, large gaps and sample-rate mismatch;
9. deterministic receipt and voltage binding.

Tests are authored but have not been compiled or executed in this environment because Cargo/rustc/rustfmt are unavailable. No raw Zenodo trace has yet been materialized or audited.

## Next validation milestone

Use a corrected success record from the v2 raw archive, preferably recovered architecture 07650 or remeasured 07651, and one ordinary successful architecture. Export only the selected architecture directory with the upstream dataset helper. Pin the archive, original Parquet, CSV conversion, meta, results and summary digests. Then compare raw-derived segment means/energy/duration to `ppk2_summary.csv` and `results.json` using declared, pre-registered tolerances.

Do not interpret disagreement automatically as a raw-trace bug: inspect voltage assumptions, exact inference boundaries, time endpoint treatment, segment weighting, PPK2 summary semantics and conversion precision. No summary value is overwritten by this verifier.

## Sources

- NATS-Bench-MCU v2 dataset: https://zenodo.org/records/21644470
- Upstream raw-data example: https://github.com/ankilab/NATS-Bench-MCU/blob/main/dataset/dataset_example.py
- MIMaaS raw current visualization: https://github.com/thesebsterseb/mimaas-api/blob/main/src/mimaas/viz.py
- NATS-Bench-MCU corrected overlay provenance: https://github.com/ankilab/NATS-Bench-MCU/blob/main/corrections/README.md

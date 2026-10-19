# display-latency-reference

Standalone Rust 2024 / Rust 1.96 reference analyzer for frameprobe-compatible photodiode captures.

## Use

From the repository root:

    cargo fmt --manifest-path tools/display-latency-reference/Cargo.toml -- --check
    cargo clippy --manifest-path tools/display-latency-reference/Cargo.toml --all-targets -- -D warnings
    cargo test --manifest-path tools/display-latency-reference/Cargo.toml
    cargo run --manifest-path tools/display-latency-reference/Cargo.toml -- --input capture.csv --reference press

Input can be supplied with `--input FILE` or piped on stdin. The reference mode is `press` by default. Use `--reference delivery` only if the row records a nonzero host-delivery time. Output is a per-capture CSV report; malformed rows remain present with explicit status and reason instead of disappearing from the sample count.

The supported source-format columns are:

- `clickTime`: duration of the HID input call, in microseconds; it is not an absolute timestamp.
- `timeTaken`: total capture duration, in microseconds.
- `sampleCount`: declared number of ADC readings.
- `preClickSamples`: index where the input event was issued.
- `samples`: semicolon-separated integer ADC readings.
- `deliveryTime`: optional elapsed time from the input call until host USB receipt. A value of zero is the firmware's explicit no-delivery sentinel.

Five-column older headers are supported. If a newer firmware row appends a sixth delivery field under an older five-column header, it is recognized as that compatibility extension. The parser intentionally accepts the documented simple numeric format rather than acting as a general CSV implementation.

## Measurements

- **t50** is a normalized 50%-of-swing crossing. It is a repeatable threshold proxy, not a first-photon timestamp.
- **t10** is emitted only when the onset fraction of the signal swing exceeds the pre-click peak-to-peak noise.
- **t90** is the first near-settled crossing searched from t50; t90 - t10 is the combined sensor-plus-display transition time.
- The baseline and settled state use upper medians. A crossing must persist for five samples for t10/t50; the reported timestamp is the first sample in the sustained run, not the fifth.
- A missing transition, incomplete crossing, malformed row, no-delivery sentinel, ADC rail saturation, sample-count mismatch, insufficient windows and invalid timing arithmetic are distinct statuses.

The estimated average sample interval is inferred from total capture duration, input/delivery pause and sample count. The firmware does not timestamp every ADC reading, so this is **not** an exact per-sample clock. The result is a software measurement proxy whose uncertainty also depends on firmware timing, ADC behavior, analog front-end bandwidth, photodiode response, threshold selection, detector mounting, display mode and reference instrumentation.

## Explicit limits

- v0 does not detect or compensate for temporal-light-modulation (PWM/backlight or per-refresh OLED flicker). Signals with relevant flicker must not be treated as qualified display rankings by this implementation.
- t90 - t10 is detector-plus-display response; it must not be advertised as panel-only response.
- This crate does not measure absolute luminance, chromaticity, HDR accuracy, power, optical safety or human response.
- Unit and mutation tests use synthetic waveforms; they validate the arithmetic and fail-closed behavior only.
- The project independently implements the documented method contract and does not bundle or modify frameprobe's Python source or hardware. frameprobe's hardware is CERN-OHL-S v2 and its software is GPL-3.0-or-later; names/logos and identified third-party 3D assets are separately restricted. See [their licensing note](https://github.com/frameprobe/frameprobe/blob/main/LICENSING.md).

## Next qualification gate

Before any physical latency claim: validate a constructed sensor chain against an independently calibrated timing reference; characterize bias and repeatability over sample rates, rising/falling transitions, signal-to-noise ratio and display modes; preserve raw traces and analyzer version; and add a separately reviewed temporal-modulation method before comparing strongly flickering panels.

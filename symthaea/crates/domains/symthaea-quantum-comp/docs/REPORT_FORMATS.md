# Report Formats

`alpha.5` adds lightweight report exports without adding `serde`, CSV, or Markdown dependencies.

## CSV

Use the `ReportTable` trait to export:

- `BindingProbeReport`
- `NoiseSweepReport`
- `ComparativeBindingReport`

The CSV strings are designed for quick notebooks, spreadsheets, and lab scratchpads. They are not a stable public interchange format yet.

## Markdown

Use the same `ReportTable` trait for Markdown tables suitable for research notes.

## Robustness Markdown

Use `robustness_to_markdown` for `NoiseRobustnessSummary`.

## JSON-like output

`BindingProbeReport::to_json_like` remains a small dependency-free convenience. It is intentionally not a full JSON serialization contract.

## Future direction

A later alpha should add a real optional serialization feature, probably behind `serde`, with a stable schema and golden-file tests.

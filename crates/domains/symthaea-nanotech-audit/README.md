# Symthaea Nanotechnology Audit

Read-only integrity verifier for exported [MADE](https://github.com/diffractivelabs/MADE) benchmark runs.

## Build and run

From the Symthaea repository root:

```sh
cargo run -p symthaea-nanotech-audit -- <path-to-made-run-directory>
```

For example, point it at the directory containing `.hydra/`, `trajectories/`, and `summary/`.

Exit codes:
- `0`: artifact-integrity checks passed.
- `1`: one or more checks failed.
- `2`: invocation or report-output error.

The program writes a JSON report to stdout. It does not change or delete run artifacts.

## Checks in the initial version

- A non-empty resolved Hydra config exists.
- The trajectory-file count, contiguous episode indices, per-episode JSON row count, and aggregate episode count agree.
- Each trajectory's `metrics` object matches the equivalent row in `summary/episodes.json`.
- Each episode's `final/queries_used` equals the recorded observation count.
- Expected final phase-diagram fields are present.
- `summary/episodes.csv` headers, rows and cells reconcile with `episodes.json`.
- The aggregate summary's final-metric key set and population mean, population standard deviation, and standard error of the mean are independently reconstructed within a documented floating-point tolerance.

## Explicit claim ceiling

An `artifact_integrity_pass` means only that these exported artifacts agree internally. It does **not** prove:
- which source commit or dataset was used;
- dataset or model rights, source hashes or identity;
- total oracle/reference cost, initialization calls, retries or caching;
- the scientific validity of a property or trajectory;
- experimental confirmation or reproducibility.

Those require separately frozen provenance manifests, oracle-call ledgers, task eligibility checks, and independent ground truth. This tool is an integrity component for NANO-010, not the full scientific benchmark gate.

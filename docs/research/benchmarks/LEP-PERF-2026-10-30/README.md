# LEP-PERF-0.1 — first hosted smoke baseline

This report is the preserved output from GitHub Actions workflow run #36 for PR #8954.

- PR source head: `9f120920153c78374fb782c493046d38d8f2c2b6`
- Workflow merge commit tested: `3d89fe67866905474fb782c493046d38d8f2c2b6`
- Rust: `rustc 1.96.0 (ac68faa20 2026-05-25)`
- Cargo.lock SHA-256: `4852031e92a96c787124c18d63a30a40a61a7d52c24f266933caaf06624949ad`
- Artifact: `lep-perf-3d89fe67866905474fb782c493046d38d8f2c2b6` (artifact ID 12967825982; artifact digest in `reproducibility.txt`)
- Method: 20 warm-up iterations per path, then 100 timed iterations; p50/p95 and mean are emitted by the benchmark binary.

## Measurements

| Input | Parse + schema p50 | Parse + schema + semantics p50 | Semantic overhead | Full-path p95 |
|---:|---:|---:|---:|---:|
| 1,247 B (requested 1 KiB) | 5.709 µs | 6.590 µs | 15.4% | 7.071 µs |
| 10 KiB | 6.870 µs | 7.822 µs | 13.9% | 8.012 µs |
| 100 KiB | 19.169 µs | 20.021 µs | 4.4% | 21.662 µs |
| 1 MiB | 157.547 µs | 160.511 µs | 1.9% | 181.343 µs |

The benchmark compares two paths over synthetic LEP statements: strict parse + bundled JSON Schema validation, and the same path plus LEP semantic conformance. It does **not** measure signature verification, artifact retrieval/hash verification, external execution, identity resolution, policy evaluation, or network/database costs.

The initial result meets the project's provisional 2x p50 semantic-overhead target at 10 KiB (measured ratio 1.139x). This is a smoke benchmark on one runner, not an industry comparison, a statistically mature performance claim, or evidence that LEP beats in-toto/SLSA/other validators. Preserve the raw JSON and reproducibility data; repeat across runner classes and multiple independent runs before using the numbers for adoption decisions.

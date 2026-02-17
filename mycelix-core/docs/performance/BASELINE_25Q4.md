# Trust Layer Benchmarks – Baseline (Q4 2025)

Generated via:

```bash
cargo bench --bench trust_layer_bench -- --sample-size=20 --warm-up-time=1 --measurement-time=5
```

## Summary

| Benchmark | Mean | Std Dev | Throughput | Notes |
| --- | --- | --- | --- | --- |
| `registry/record_archive/100` | 0.92 s | 0.15 s | 109 ops/s | JSON snapshot on tmpfs (100 entries per iter) |
| `proof/prove_loss_delta` | 1.32 µs | 0.12 µs | 0.76 M ops/s | Deterministic hash-based placeholder proof |
| `proof/verify_loss_delta` | 1.01 µs | 0.08 µs | 0.99 M ops/s | Receipt hash check + serde decode |
| `ipfs/upload_bytes` | 209 µs | 9 µs | 4.8 k ops/s | Mock HTTP server (no TLS) |
| `end_to_end/prove+upload` | 242 µs | 19 µs | 4.1 k ops/s | Prove + upload pipeline |

## Raw Criterion Outputs

- `target/criterion/registry/record_archive/100/new/estimates.json`
- `target/criterion/proof/prove_loss_delta/new/estimates.json`
- `target/criterion/proof/verify_loss_delta/new/estimates.json`
- `target/criterion/ipfs/upload_bytes/new/estimates.json`
- `target/criterion/end_to_end/prove+upload/new/estimates.json`

Machine-readable baselines live in `docs/performance/baseline_trust_layer.json`.  
The CI pipeline runs `tools/check_bench_regressions.py` and fails if any mean drifts by more than 20%.

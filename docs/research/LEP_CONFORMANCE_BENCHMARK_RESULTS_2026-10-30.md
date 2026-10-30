# LEP conformance benchmark results — 2026-10-30

**Status:** First hosted synthetic baseline; not an industry leaderboard and not an SOTA claim.  
**Workflow:** LEP conformance run 45, GitHub Actions run ID 41035270018.  
**PR head associated with the run:** b7378d6641d23ac0f803ac737eb47493cedfaa31.  
**Uploaded artifact:** lep-perf-cf431ab85f5791bef11431910e05af425ea70ec4.

## 1. Environment and method

- Hosted runner: ubuntu-latest, Linux x86_64.
- Rust: rustc 1.96.0 (ac68faa20 2026-05-25).
- Each benchmark command: 20 warm-up iterations and 100 timed iterations per path.
- Five command-level benchmark batches were recorded per requested input size.
- The table reports the median of the five batch-level p50 values and the median of the five batch-level p95 values. These are not pooled raw-sample quantiles.
- Path A: strict JSON parse + bundled Draft 2020-12 schema validation.
- Path B: strict JSON parse + bundled schema + LEP semantic consistency checks.
- Both paths run in-process on synthetic valid records. The benchmark excludes signature authentication, signer authorization, network access, artifact retrieval/re-hashing, scientific execution, and domain-specific evidence adjudication.

## 2. Results

| Requested size | Actual serialized bytes | Schema-only p50 | Schema + semantic p50 | Schema-only p95 | Schema + semantic p95 | p50 delta |
|---:|---:|---:|---:|---:|---:|---:|
| 1 KiB | 1,247 | 9.819 µs | 11.572 µs | 10.710 µs | 12.233 µs | +17.9% |
| 10 KiB | 10,240 | 11.872 µs | 13.555 µs | 12.744 µs | 14.117 µs | +14.2% |
| 100 KiB | 102,400 | 31.379 µs | 33.281 µs | 33.282 µs | 36.548 µs | +6.1% |
| 1 MiB | 1,048,576 | 236.190 µs | 235.548 µs | 253.562 µs | 256.658 µs | -0.3% |

The 1 KiB request produces a 1,247-byte valid record because the synthetic LEP statement itself is larger than 1 KiB. This size is reported as requested and actual rather than mislabelled as an exact 1 KiB record.

## 3. Interpretation

On this synthetic workload, adding the implemented semantic checks did not exceed the provisional internal budget of 2x the structural-only p50 at 10 KiB. Observed p50 overhead ranged from -0.3% to +17.9% across the four sizes. The small negative delta at 1 MiB is measurement noise, not evidence that semantic validation is free.

The current synthetic harness measures the incremental cost of semantic checks after the same strict parse and schema-validation path. It does **not** measure the full cost of producing, signing, transmitting, authenticating, resolving, or independently executing evidence. No comparison against in-toto, SLSA, Sigstore, Cedar, OPA, or an industry product is implied by these results.

## 4. Exact-head verification status

The same hosted workflow completed the following steps successfully on the associated PR run:

- cargo fmt check;
- cargo test --locked;
- Clippy for all targets with warnings denied;
- valid fixture CLI acceptance;
- expected-code rejection for queued-but-passed, missing-target, stale-target, and duplicate-key fixtures;
- expected-code rejection for oversized input;
- benchmark collection and artifact upload.

These results establish that the checked-in Rust prototype compiled and passed its current automated conformance suite on the workflow revision. They do not authenticate evidence or prove scientific claims.

## 5. Next benchmark work

1. Repeat on a pinned local runner and a second hardware class; keep raw timing samples rather than only batch summaries.
2. Separate parser, schema compilation, warm schema evaluation, semantic evaluation, and report serialization.
3. Add allocation, peak RSS, and input-depth/collection-size dimensions.
4. Add a corpus of schema-valid compound adversaries and compare Rust decisions against an independently implemented validator.
5. Add DSSE/in-toto authentication and artifact-byte verification as separate measured stages; never fold them into the current LEP semantic number.

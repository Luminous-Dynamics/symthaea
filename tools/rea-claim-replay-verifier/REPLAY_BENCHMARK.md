# REA Evidence admission and claim replay performance benchmark

The `rea-claim-replay-bench` binary is a development benchmark. It is not part of any runtime decision path and adds no external dependencies.

## Workload matrix

The runner constructs semantically valid synthetic REA Evidence records from the checked-in public fixture. It preserves the same subject, operation, predicate type and normalized result while assigning distinct provider identities and varying only the size of an irrelevant `raw_result` payload.

It measures three distinct assurance stages over 12 independent workload cells:

- Evidence record count: 1, 4, 16 and 64.
- `raw_result` payload bytes per record: 0, 65,536 and 1,048,576.

Every cell reports claim bytes, result bytes, aggregate Evidence bytes and maximum record size.

## Timed operations

1. **Evidence admission:** raw JSON parsing, duplicate-key rejection, schema validation and semantic Evidence ID verification for every raw record.
2. **Producer claim evaluation:** typed scope/disposition evaluation over already admitted `VerifiedEvidence` values. This measures the producer rule engine without raw JSON admission.
3. **Independent replay:** full request/result parsing, raw Evidence verification, claim-scope/disposition derivation and complete serialized-result comparison.

The production evaluator runs once before timing to create a valid serialized result for the replay checker. That setup evaluation is excluded from timing; the separate producer-evaluation distribution measures repeated rule evaluation over the same pre-admitted `VerifiedEvidence` values. The admission distribution measures raw parse/schema/identity cost. Independent replay intentionally includes admission again, because a real replay consumer receives raw Evidence bytes. Do not add or subtract percentile values to infer a total latency; percentile distributions are not additive.

Each cell performs three warmup iterations followed by the requested number of samples (default: 50). It reports nearest-rank p50/p95/p99, minimum and maximum nanoseconds. It emits one JSON receipt to stdout.

For a fast CI wiring/schema smoke, invoke the binary with `--smoke`. This executes all 12 cells with one measured sample and zero warmups. It verifies that every workload is admissible, every replay returns `ReplayMatch`, and the three-stage receipt is well-formed; it is **not** a performance measurement and must not be compared across revisions. Normal runs omit the flag or pass a positive sample count and retain three warmups.

## Reproducible run

From the repository root:

```sh
export RUSTUP_TOOLCHAIN=1.99.1
cargo generate-lockfile --manifest-path tools/rea-claim-replay-verifier/Cargo.toml
rustc --version --verbose
cargo --version
uname -a
sha256sum tools/rea-claim-replay-verifier/Cargo.lock
git rev-parse HEAD

/usr/bin/time -v cargo run --release --locked \
  --manifest-path tools/rea-claim-replay-verifier/Cargo.toml \
  --bin rea-claim-replay-bench -- 50 \
  > rea-claim-replay-benchmark.json \
  2> rea-claim-replay-benchmark-run-context.txt

sha256sum rea-claim-replay-benchmark.json
```

Use the generated lockfile as part of the receipt. Record the runner CPU model, physical/logical core count, memory, OS/kernel, concurrency and any CPU governor/container limits alongside the emitted JSON. The JSON includes schema version, package version, target OS/architecture, build profile, sample count, warmup count and workload byte sizes.

Increase the sample count for publication-quality results; with 50 samples the nearest-rank p99 is effectively the maximum observation and is not a stable tail estimate. Use at least 1,000 samples for exploratory p99, and substantially more (plus independent process-level repetitions) for defensible tail estimates. Run paired revisions on the same runner image and toolchain. Do not compare unrelated hardware or mix workload cells into a single average.

## Interpretation and guardrails

- All timed replay calls must return `ReplayMatch`; a fast mismatch is not a performance success.
- Oversized and malformed controls remain separate correctness tests, not part of the valid-input latency distribution.
- This benchmark does not replace the independent conformance suite or fuzz campaigns.
- No latency thresholds are gating yet. First collect stable baselines, characterize variance and identify the dominant cost by workload cell.
- Performance changes must preserve duplicate-key rejection, Evidence schema/identity checks, request/result binding, exact scope, claim ceilings and resource limits.
- Results qualify only the exact source revision, lockfile, compiler, build profile, runner and workload matrix.

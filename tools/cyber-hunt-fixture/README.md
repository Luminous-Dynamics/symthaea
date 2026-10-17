# Cyber Hunt Fixture

A standalone, native-only Rust generator and constrained query interface for deterministic **synthetic defensive threat-hunting evaluation**. It creates solver-visible event records and evaluator-only ground truth in separate directory trees.

This is a fixture generator and evidence-query harness, not an attack tool, live data collector, detection system, or qualification authority. All hosts, principals, addresses, timestamps, destinations, tickets, and records are synthetic. The deterministic seed is non-cryptographic and intended only for reproducibility.

## Run the generator

From the repository root:

```sh
cargo test --manifest-path tools/cyber-hunt-fixture/Cargo.toml
cargo run --manifest-path tools/cyber-hunt-fixture/Cargo.toml -- --records 1000 --seed 42 --out-dir target/cyber-hunt-1k
cargo run --manifest-path tools/cyber-hunt-fixture/Cargo.toml -- --records 10000 --seed 42 --out-dir target/cyber-hunt-10k
cargo run --manifest-path tools/cyber-hunt-fixture/Cargo.toml -- --records 100000 --seed 42 --out-dir target/cyber-hunt-100k
```

The default maximum is 250,000 records. Choose a fresh output directory for every run; the CLI refuses to overwrite existing event, manifest, or oracle files.

## Artifact separation

Each generated output root is arranged like this:

```text
target/cyber-hunt-10k/
├── manifest.json               # evaluator/operator-side combined manifest
├── solver/
│   ├── events.jsonl            # only event observations; no ground truth
│   └── manifest.json           # no oracle path, hash, count, or labels
└── evaluator/
    └── oracle.json             # per-event labels and planted campaign truth
```

**Give the solver process only the `solver/` directory** (for example, mount or copy that directory into a sandbox). The query loader reads only the solver manifest and event stream; it does not open or resolve the parent combined manifest or the evaluator directory. A sibling directory arrangement alone is not an OS sandbox: deployments must enforce the filesystem boundary themselves.

The combined operator manifest records exact SHA-256, byte lengths, counts, generator version, seed, source identity, planted campaign counts, and non-claims. The solver manifest records only the event schema/generator version, synthetic flag, record count, event filename, event digest/length, UNSCORED state, and claim ceiling. It deliberately excludes the generation seed, source commit, oracle path/hash, labels, and planted campaign counts: with a public deterministic generator, exposing seed plus source revision can let a solver reconstruct the answer key. Query receipts bind request, solver-manifest, and event-store hashes; source-revision provenance remains evaluator/operator-side. The stdout receipt is independently computed after the written output is re-read. Its disposition, `STRUCTURAL_PASS_NOT_CAPABILITY_PASS`, proves structural consistency only. `source_commit_pinned` means the supplied source identifier has the form of a full lowercase 40-character Git SHA; it does not prove a dirty working tree matches that commit or replace a trusted builder attestation.

## Read-only, budgeted query interface

Example typed query from the repository root:

```sh
cargo run --manifest-path tools/cyber-hunt-fixture/Cargo.toml -- query \
  --store-dir target/cyber-hunt-10k/solver \
  --requests-jsonl tools/cyber-hunt-fixture/examples/high-volume-transfer-hunt-v1.jsonl
```

The input is JSONL with one or more `cyber-hunt-query-v1` records. The runner returns one JSONL response per request, including success/failure and a query receipt bound to the request digest, solver-manifest digest, event-stream digest, source revision, number of rows examined/matched/returned, response bytes, and cumulative usage.

The interface supports only typed exact-text filters, bounded event-/ingestion-time ranges, and minimum transferred bytes. It has no SQL, shell, arbitrary expression, mutation, network, or remediation API. Every request must state its projection and row limit. Default per-session bounds are:

| Resource | Default budget |
|---|---:|
| Queries | 100 |
| Rows examined | 1,000,000 |
| Rows returned | 10,000 |
| Serialized result bytes | 8 MiB |
| Returned rows per query | 200 |
| Filters per query | 8 |
| Fields per projection | 12 |

If a query fails its validation or budget check, no usage is committed for that query. In an integrated agent, keep the same query session/budget object for the entire investigation. Do not launch a fresh query process per question and thereby reset budgets.

## Solver-visible fields and nuisance factors

Events are ordered by ingestion time, not event time. They contain separate record, source-event, and delivery IDs; event/ingestion timestamps; pseudonymous host/principal; reserved-example source addresses; `.example.test` destinations; bytes sent; device posture; event source/type/result; change ticket; and integrity state.

The fixture contains routine benign activity, multi-stage suspicious sequences, approved-change decoys sharing event types, delayed/out-of-order ingestion, duplicate deliveries, and conflicting payloads under reused source-event IDs. A high-volume suspicious transfer and a similar-volume ticket-backed benign transfer intentionally coexist so byte-count thresholds alone can produce false positives. Campaign labels and names remain in the evaluator oracle.

## Independent scoring interface

A separate evaluator command scores a solver-produced prediction set against the evaluator oracle **after** the solver process is finished:

```sh
cargo run --manifest-path tools/cyber-hunt-fixture/Cargo.toml -- evaluate \
  --dataset-root target/cyber-hunt-10k \
  --predictions path/to/predictions-v1.json
```

The prediction set binds to the exact source revision, solver-manifest SHA-256, and event-store SHA-256. Each suspected event must cite existing evidence record IDs and provide a rationale. Unknown event IDs, duplicate suspects, duplicate evidence references, oracle/event inconsistencies, mismatched artifact hashes, and overlapping suspect/abstention sets are rejected.

The receipt reports the confusion matrix, precision and recall in basis points, false positives per 10,000 benign records (milli-scaled), abstentions, false positives on approved-change decoys, recall by planted campaign phase, and recall by campaign. Precision is null when the system raises no alerts; recall is zero when a valid zero-alert prediction set misses all planted attacks. These are separate measures, not a universal composite score.

The command belongs in a separate evaluator process/environment that can access the root dataset and `evaluator/oracle.json`. The solver gets only the `solver/` mount. The CI zero-alert baseline exists only to test scorer plumbing and is **not** a Symthaea detector result or capability claim. Public oracle data remains unsuitable for blind evaluation.

## Independent scoring interface

A separate evaluator command scores a solver-produced prediction set against the evaluator oracle **after** the solver process has finished:

```sh
cargo run --manifest-path tools/cyber-hunt-fixture/Cargo.toml -- evaluate \
  --dataset-root target/cyber-hunt-10k \
  --predictions path/to/predictions-v1.json
```

The prediction set binds to the exact source revision, solver-manifest SHA-256, and event-store SHA-256. Each suspected event must cite existing evidence record IDs and provide a rationale. Unknown event IDs, duplicate suspects, duplicate evidence references, oracle/event inconsistencies, mismatched artifact hashes, and overlapping suspect/abstention sets are rejected.

The receipt reports TP/FP/FN/TN, precision and recall in basis points, false positives per 10,000 benign records (milli-scaled), abstentions, false positives on approved-change decoys, recall by planted campaign phase, and recall by campaign. Precision is null when the system raises no alerts; a zero-alert prediction set produces zero recall when the corpus has planted positive cases. These are separate measures, not a universal composite score.

The evaluator belongs in a separate process/environment that can access the root dataset and `evaluator/oracle.json`. The solver gets only the `solver/` mount. The CI zero-alert baseline verifies scorer plumbing only; it is **not** a Symthaea detector result or capability claim. Public oracle data remains unsuitable for blind evaluation.

## Qualification boundaries

- Identical inputs, source/version, and seed should generate byte-identical event and oracle files.
- SHA-256 protects artifact identity; it does not prove simulator correctness or real-world truth.
- The public oracle is not appropriate for blind capability scoring.
- A structural verifier pass does not establish precision/recall, causal attribution, incident truth, SOC readiness, learner efficacy, or production security.
- All outputs are synthetic, and no real target, credentials, shell, network or remediation actions are required.

## Design lineage

The generator implements the first tranche tracked in [Symthaea #7742](https://github.com/Luminous-Dynamics/symthaea/issues/7742), composing with the existing [cross-domain crucible #2217](https://github.com/Luminous-Dynamics/symthaea/issues/2217) and the [Mycelix Praxis learning programme #5124 / draft PR #5125](https://github.com/Luminous-Dynamics/mycelix/pull/5125).
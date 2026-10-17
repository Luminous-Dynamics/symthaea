# Symthaea Nanotechnology Audit

Read-only integrity verifier for exported [MADE](https://github.com/diffractivelabs/MADE) benchmark runs.

## Build and run

From the Symthaea repository root:

```sh
cargo run -p symthaea-nanotech-audit -- <path-to-made-run-directory>
```

Point it at the directory containing `.hydra/`, `trajectories/`, and `summary/`.

Exit codes:
- `0`: artifact-integrity checks passed.
- `1`: one or more checks failed.
- `2`: invocation or report-output error.

The program writes a JSON report to stdout and does not change or delete run artifacts.

## Checks

- A non-empty resolved Hydra config exists.
- Trajectory-file count, contiguous episode indices, `episodes.json` rows, and aggregate episode count agree.
- Each trajectory's `metrics` object matches the equivalent row in `summary/episodes.json`.
- Each episode's `final/queries_used` equals the recorded observation count.
- Expected final phase-diagram fields are present.
- `summary/episodes.csv` headers, rows and cells reconcile with `episodes.json`.
- Aggregate final-metric key sets and population mean, population standard deviation, and standard error of the mean are independently reconstructed within a documented floating-point tolerance.
- If a root-level `provenance-manifest.json` sidecar exists, the verifier checks its strict field contract, required artifact list, SHA-256 digest and byte size of each listed file, the named digest links to the task/evaluation/source manifests and resolved config, proposal-request counts against episode query counts, and arithmetic consistency of oracle request/cache/external-call counters.

Without a sidecar, the auditor reports `provenance_manifest_status: "not_provided"`. With a valid sidecar, `"manifest_consistent"` means only that its declarations, digests, and counter arithmetic reconcile. A malformed or contradictory sidecar fails the audit. The manifest is intentionally not included in its own artifact list to avoid a circular digest.

## Provenance sidecar

See the strict JSON Schema at [schema/made-run-provenance.schema.json](schema/made-run-provenance.schema.json). The sidecar binds an exported run to:
- the exact upstream benchmark and policy revisions;
- task, evaluation, source manifests and resolved configuration hashes;
- explicit data-source revisions, license declarations and rights-status declarations;
- core episode/summary artifacts by relative path, SHA-256 and byte count;
- oracle setup/proposal/retry request counts, cache hits, external-call counts, failures and declared cost units.

Required artifact paths include `.hydra/config.yaml`, `source-manifest.json`, `task-contract.json`, `evaluation-manifest.json`, the three summary files, and all `trajectories/episode_NNN.json` files. All listed paths must be relative and resolve within the run directory. The manifest itself is a sidecar and must not list itself as an artifact.

The accounting identities are:
```
requests_total = setup_requests + proposal_requests + retry_requests
external_calls = requests_total - cache_hits
failed_external_calls <= external_calls
proposal_requests = sum(episode final/queries_used)
```

These identities detect internal contradictions; they do not prove the counters are truthful. Reconcile them with an immutable per-call ledger, logs or source-system receipts before making total-cost claims.

## Explicit claim ceiling

An `artifact_integrity_pass` is **not** a scientific benchmark pass. The verifier does not prove:
- that the named source commit exists or that the claimed code produced the run;
- legal authorization for each dataset/model/API, despite recording the declared license and rights status;
- oracle-call truth, cache behavior, setup cost, compute cost or carbon accounting from external evidence;
- scientific validity of a predicted property, trajectory reliability, or causal discovery;
- experimental confirmation or independent replication.

A manifest hash is a check on bytes and linkage, not external attestation. The auditor is an integrity component for NANO-010, not the full scientific benchmark gate.

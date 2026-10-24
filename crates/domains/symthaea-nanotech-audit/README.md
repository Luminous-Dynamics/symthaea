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

Schemas: [provenance sidecar](schema/made-run-provenance.schema.json), [per-call oracle ledger record](schema/oracle-call-ledger-record.schema.json), [distribution-shift manifest](schema/nano-shift-manifest.schema.json), and [audit report](schema/audit-report.schema.json). The report schema is the machine-readable output contract. The sidecar binds an exported run to:
- the exact upstream benchmark and policy revisions;
- task, evaluation, source manifests and resolved configuration hashes;
- explicit data-source revisions, license declarations and rights-status declarations;
- a local snapshot artifact and SHA-256 for each declared data source, tied to the artifact list;
- core episode/summary artifacts by relative path, SHA-256 and byte count;
- an `oracle-call-ledger.jsonl` entry for each setup/proposal/retry request, with contiguous sequence numbers, candidate IDs for proposals, cost, cache/external status, and a hash-linked receipt artifact;
- oracle request counts, cache hits, external-call counts, failures and total cost independently reconstructed from that ledger.

Required artifact paths include `.hydra/config.yaml`, `source-manifest.json`, `task-contract.json`, `evaluation-manifest.json`, the three summary files, and all `trajectories/episode_NNN.json` files. All listed paths must be relative and resolve within the run directory. The manifest itself is a sidecar and must not list itself as an artifact.

The accounting identities are reconstructed from the JSONL ledger and cross-checked against the manifest:
```
setup_requests + proposal_requests + retry_requests
  = external_calls + cache_hits + local_rejections

external_calls = count(records where external_invocation = true)
cache_hits = count(records where cache_hit = true)
local_rejections = count(proposal records with status = invalid_candidate)
failed_external_calls = count(external calls whose status is not evaluated)
proposal_requests = count(stage = proposal) = sum(episode final/queries_used)
total_cost = sum(per-record cost)
```

A locally rejected candidate is still a proposal attempt and has selector/validation cost, but must have neither an external invocation nor a cache hit. An evaluated request must be either a cache hit or an external invocation; an oracle failure or timeout must record an external invocation. Each record must identify a receipt artifact whose bytes are separately hash-verified. The ledger is still a declared run artifact; this verifier does not attest that an upstream oracle or service emitted the ledger or that the stated cost unit reflects an invoice or measured compute resource. Reconcile with source-system logs, signed receipts or billing/resource records before total-cost claims.

## Explicit claim ceiling

An `artifact_integrity_pass` is **not** a scientific benchmark pass. The verifier does not prove:
- that the named source commit exists or that the claimed code produced the run;
- legal authorization for each dataset/model/API, despite recording the declared license and rights status;
- oracle-call truth, cache behavior, setup cost, compute cost or carbon accounting from external evidence;
- scientific validity of a predicted property, trajectory reliability, or causal discovery;
- experimental confirmation or independent replication.

### Optional distribution-shift manifest

A run can bind an optional `shift_manifest_path` and `shift_manifest_sha256` in the provenance sidecar. When present, both fields are required. The manifest freezes zero or more physical perturbation families (`composition`, `surface_coordination`, `defects`, `temperature_energy`, `strain_pressure_phase`, `size_interface_passivation_substrate`) independently from the shift mechanism axes, shift kind (`in_distribution`, `single_factor`, or `compounded`), severity (`baseline`, `mild`, `moderate`, or `severe`), split (`train`, `calibration`, `validation`, or `test_lockbox`), physical rationale, severity definition, source snapshot hash, independence keys, and sample IDs. Supported mechanism axes are `feature_support`, `label_distribution`, `graph_topology`, `physical_regime`, `chemical_environment`, and `measurement_process`. A compounded shift keeps its actual physical families; it is not represented by the placeholder value `compounded`.

The auditor checks the shift-manifest digest against the artifact list and validates:
- recognized physical shift families, mechanism axes, and severity/split labels;
- baseline consistency: in-distribution groups must declare zero changed physical perturbation families;
- axis cardinality consistency: no axes for in-distribution groups, one for single-factor shifts, and at least two for compounded shifts;
- task-contract and benchmark identity against the provenance sidecar;
- source snapshot digests against already hashed artifacts;
- unique group IDs and non-empty sample/independence identifiers;
- no independence key or sample ID crossing between different data splits.

This is a leakage guard, not proof that the shift labels are physically justified or that the split represents real deployment. Those decisions require preregistration and domain review. The strict split contract is intended for NANO-012.

A manifest hash is a check on bytes and linkage, not external attestation. The auditor is an integrity component for NANO-010, not the full scientific benchmark gate.

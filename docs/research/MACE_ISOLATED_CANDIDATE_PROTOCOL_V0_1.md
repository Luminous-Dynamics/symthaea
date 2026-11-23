# MACE Isolated Candidate Protocol v0.1

**Status:** research implementation; source authored and statically inspected, Rust compilation/tests not executed here. Not a sandbox, signed trust root, independently operated evaluator, or externally qualified result.

## Purpose

Move candidate retrieval and decision execution out of the synthetic evaluator executable. The candidate becomes a standalone Rust package with one bounded request on standard input and one receipt on standard output.

The trust direction is deliberate:

- **Candidate:** untrusted decision service. It sees only a task contract and candidate-visible evidence; it returns a decision and the evidence IDs it used.
- **Evaluator custodian:** separately operated service or process. It owns intervention manifests, hidden outcomes, scoring rules, data splits and benchmark receipts. None of these belong in the candidate request.
- **Ingress verifier:** authenticates sources, authority, calibration and lifecycle before evidence is admitted. The candidate process in this prototype does not replace this verifier.

This is the first executable boundary in issue #13350. It does not complete issue #13287.

## Workspace and dependency boundary

Package: `broca-mace-candidate`

Path: `research/broca-mace-transfer-bench-v0-1/candidate/`

Workspace dependencies:

- `broca-mace-policy`
- `broca-mace-hdc-retrieval`
- `serde`, `serde_json`, `sha2`

The package has no declared dependency on:

- `broca-mace-evaluator`;
- the generated stress fixture module;
- evaluator truth/utility types;
- score aggregation or intervention planning.

Initial methods are `exact_scope_scan` and `typed_content_hdc`. Both pass canonical resolved evidence through the unchanged `Policy::CausalQualified` gate. Retrieval produces candidate evidence; it does not confer trust or authority.

## Wire protocol

Input schema: [mace_candidate_envelope_v1](../../data/schemas/mace_candidate_envelope_v1.schema.json)

Output schema: [mace_candidate_receipt_v1](../../data/schemas/mace_candidate_receipt_v1.schema.json)

The process reads one JSON object, bounded to 32 MiB and 20,000 evidence records. Unknown fields are rejected at every nested object. It emits one JSON receipt or exits nonzero without emitting a decision receipt.

The candidate envelope binds:

- schema version, run ID, pair/task ID, opaque observation sequence and custodian nonce;
- candidate method;
- the candidate-visible `DecisionNeedV1` task contract, including exact action alternatives, semantic estimand and qualification-profile digest;
- candidate-visible evidence records and their source/lifecycle/causal descriptors.

It intentionally has no field for evaluator truth, realized utility labels, optimal action, intervention identity/kind, target key, removed evidence IDs, score, evaluator seed or evaluator instructions. Such fields are rejected as unknown rather than ignored.

The receipt binds:

- raw request SHA-256;
- task, run, sequence and domain-separated nonce digest;
- canonical DecisionNeed and evidence-set digests;
- canonical candidate-input digest;
- observation ID derived from run + pair block + sequence + method + input digest;
- decision and support/retrieval event IDs;
- retrieval-profile digest, index payload bytes and candidate-process latency.

The observation identity follows the same canonical formula as the paired evaluator receipt contract in PR #13331. A digest establishes byte identity, not who supplied the bytes.

## Invocation

From the repository root:

```bash
cargo run --manifest-path research/broca-mace-transfer-bench-v0-1/Cargo.toml \
  -p broca-mace-candidate --release

cargo test --manifest-path research/broca-mace-transfer-bench-v0-1/Cargo.toml \
  -p broca-mace-candidate
```

The program reads the envelope from stdin. Capture stdout as the candidate receipt and stderr as diagnostic output. Do not combine the candidate's stdout with evaluator output or let candidate code read evaluator paths.

The benchmark evaluator has **not yet been rewired** to launch this process. The current paired runner still performs candidate execution and hidden-label scoring inside the public evaluator package. Until the runner is changed and the process boundary is verified, existing paired results remain diagnostic-only.

## Security properties of this increment

Implemented in source/schema:

- separate package and dependency declaration;
- strict nested deserialization with unknown-field rejection;
- schema-level `additionalProperties: false`;
- explicit request byte and evidence-record caps;
- identifier, nonce, digest, enum, duplicate-event and numeric-range checks;
- evidence-view resolution checks after HDC retrieval;
- task/input/observation receipt binding;
- candidate receipt contains no evaluator labels.

Not established:

1. **Authenticity:** source-root SHA-256 digests identify roots but do not authenticate issuers. The booleans for trusted/retracted/reconciled status are inputs. Production admission must derive them from an independently verified Mycelix/ingress receipt, not trust caller-supplied flags.
2. **Isolation:** a Cargo package is not an OS security boundary. The next process-integration increment must use default-deny filesystem and network, dedicated UID/namespaces, bounded CPU/memory/pids/FDs, read-only runtime and no evaluator mounts. Then qualify Firecracker or gVisor (or an equivalently strong isolated runner) using negative controls.
3. **Evaluator custody:** the evaluator remains public and co-located in the current runner. An independent custodian must own hidden labels and intervention plans and expose a one-way task envelope.
4. **Receipt authenticity:** output hashes are self-reported. The custodian must sign or independently attest the candidate artifact digest, request digest, evaluator/data/split/scoring digests and measured resource envelope.
5. **Resource accounting:** `latency_ns` currently measures the candidate retrieval/policy call only; it excludes envelope parsing/validation, input hashing, receipt serialization, process launch, IPC, ingress verification, evaluator, memory write/update, tokenization, model calls and external environment work. Report end-to-end p50/p95/p99 and all-in compute, energy, storage and money separately.
6. **Runtime verification:** JSON schemas and parser tests are authored, but no Rust compilation, test execution or sandbox negative-control execution is claimed here.

## Required next boundary: evaluator-to-candidate IPC

The next engineering increment should replace the evaluator's in-process `run_candidate` calls with a process client:

1. Custodian freezes the task/evidence envelope and hidden intervention metadata independently.
2. The process launcher pins the candidate binary and dependency digest, passes one envelope over stdin, captures bounded stdout/stderr and enforces a hard deadline.
3. The candidate process has no evaluator mount, network route, mutable shared directory, evaluator environment variable or secret seed.
4. The custodian validates receipt schema, request digest, nonce, observation identity, decision support IDs and resource telemetry.
5. Only the evaluator joins the accepted receipt to the sealed outcome/intervention record and computes regret/utility.
6. A replay tool reproduces scoring from signed receipts without executing candidate source.

Negative controls must prove that mutating evaluator truth, permuting intervention IDs, changing scoring labels or altering hidden seeds leaves candidate request bytes and output unchanged; candidate attempts to read evaluator files, environment, mounts, sockets or network must fail closed. Altering the candidate binary or dependencies after qualification invalidates prior receipts.

## Qualification ladder

| Level | Evidence | Claim allowed |
|---|---|---|
| 0 | Typed candidate policy only | Policy rules exist |
| 1 | Separate package + strict envelope/receipt | Candidate API is structurally separated |
| 2 | Parent runner launches child process; no shared state | Process boundary is exercised |
| 3 | Sandbox negative controls + pinned image and resource receipts | Isolation controls demonstrated for that runner |
| 4 | Independent custodian + sealed labels + reproducible signed score receipts | Independently evaluated benchmark result |
| 5 | Matched public benchmark submissions, repeated runs, uncertainty and full cost | External comparison under the named protocol |

Do not skip a level in public claims.

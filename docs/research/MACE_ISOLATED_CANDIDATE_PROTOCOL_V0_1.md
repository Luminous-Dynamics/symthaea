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
- candidate-visible evidence records and their source/lifecycle/causal descriptors. The parent projects these records to the exact `DecisionNeedV1` scope and allowed action alternatives before sending the envelope; unrelated regimes, objectives and actions are not disclosed.

It intentionally has no field for evaluator truth, realized utility labels, optimal action, intervention identity/kind, target key, removed evidence IDs, score, evaluator seed or evaluator instructions. Such fields are rejected as unknown rather than ignored.

The receipt binds:

- raw request SHA-256;
- task, run, sequence and domain-separated nonce digest;
- canonical DecisionNeed and evidence-set digests;
- canonical candidate-input digest;
- observation ID derived from run + pair block + sequence + method + input digest;
- decision and support/retrieval event IDs;
- retrieval-profile digest, index payload bytes and candidate-process latency.

The receipt also records the candidate-visible event count separately from the full available-event count in the parent-authored candidate/evaluator rows. This makes the scope projection auditable. The run's source-bundle digest binds the evaluator runner and reducer, process client, candidate source, direct policy/HDC sources, package/workspace manifests, Rust toolchain file and Nix lock. A separate schema-bundle digest binds the three JSON schemas. The observation identity follows the same canonical formula as the paired evaluator receipt contract in PR #13331. A digest establishes byte identity, not who supplied the bytes.

## Invocation

From the repository root:

```bash
cargo run --manifest-path research/broca-mace-transfer-bench-v0-1/Cargo.toml \
  -p broca-mace-candidate --release

cargo test --manifest-path research/broca-mace-transfer-bench-v0-1/Cargo.toml \
  -p broca-mace-candidate
```

The program reads the envelope from stdin. Capture stdout as the candidate receipt and stderr as diagnostic output. Do not combine the candidate's stdout with evaluator output or let candidate code read evaluator paths.

The paired evaluator has since been wired in source to launch the candidate worker for exact-scope scan and typed-content HDC; see the evaluator subprocess integration section below. The evaluator and synthetic truth/fixture generator remain public and co-located, and three legacy baselines remain in-process. No runtime execution has been verified, so all current paired results remain diagnostic-only.

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
2. **Isolation:** a child process under the same OS identity is not an OS security boundary. The next increment must use default-deny filesystem and network, dedicated UID/namespaces, bounded CPU/memory/pids/FDs, read-only runtime and no evaluator mounts. Then qualify Firecracker or gVisor (or an equivalently strong isolated runner) using negative controls.
3. **Evaluator custody:** the evaluator remains public and co-located in the current runner. An independent custodian must own hidden labels and intervention plans and expose a one-way task envelope.
4. **Receipt authenticity:** output hashes are self-reported. The custodian must sign or independently attest the candidate artifact digest, request digest, evaluator/data/split/scoring digests and measured resource envelope.
5. **Resource accounting:** `latency_ns` measures candidate retrieval/policy work. The parent separately records request construction, request/response wall time, receipt validation and parent-side end-to-end invocation; process startup is recorded once per worker (and repeated in rows for correlation). The current end-to-end field excludes worker startup, ingress verification, evaluator work, memory write/update, external model calls and the rest of the environment loop. Report end-to-end p50/p95/p99 and all-in compute, energy, storage and money separately.
6. **Runtime verification:** JSON schemas and parser tests are authored, but no Rust compilation, test execution or sandbox negative-control execution is claimed here.

## Required next boundary: sandboxing, evaluator custody and replay

The next engineering increment is to qualify the subprocess with a real OS sandbox and split evaluator custody:

1. Put candidate execution in a default-deny sandbox with no evaluator mounts, network route, shared writable directory, evaluator environment variable or secret seed; apply CPU/memory/pid/FD/deadline limits.
2. Keep hidden outcomes, intervention labels, scoring and held-out seeds in a separately operated evaluator custodian that sends one-way candidate envelopes.
3. Bind the candidate binary, dependency tree, request, evaluator/data/split/scoring revisions and sandbox resource envelope into authenticated receipts.
4. Validate receipt schema, request digest, nonce, observation identity, both retrieved and decision-support IDs, process timing and resource telemetry.
5. Independently replay scores from sealed evaluator data and receipts without executing candidate source.
6. Keep legacy in-process baselines as explicitly diagnostic comparisons until detached equivalents exist; do not silently pool execution boundaries.

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


## Evaluator subprocess integration slice (#13351)

The paired runner now uses `CandidateProcessSession` for the exact-scope and typed-content-HDC arms. The worker is a persistent, line-framed stateless request/receipt process scoped to one task block:

- the parent launches one hash-pinned candidate executable per benchmark run and checks its readiness frame;
- the parent sends a fresh random 24-byte nonce and one candidate-only JSON envelope per observation;
- the worker emits one receipt for each request and keeps no cross-request memory state;
- the parent validates the exact request digest, nonce digest, run/task/sequence/method, DecisionNeed/evidence/input digests and observation ID;
- candidate-visible evidence is filtered to the declared task scope/action set; the candidate parser rejects any out-of-scope record even if the parent projection is faulty;
- support IDs must be a subset of retrieved IDs; every retrieved ID must resolve to a supplied canonical record in the declared scope/action set;
- evaluator intervention labels and outcome truth remain in the evaluator loop and are not fields in the candidate envelope.

The process session is reused for baseline and ablation observations in the same benchmark run. This avoids a fresh process launch for every ablation while preserving an independent candidate invocation for each observation. The candidate sequence is monotonic within that worker session across the whole run.

### Build and invoke

From the repository root on Linux:

```bash
cargo build --release \
  --manifest-path research/broca-mace-transfer-bench-v0-1/Cargo.toml \
  -p broca-mace-candidate

export MACE_CANDIDATE_BIN="$PWD/target/release/broca-mace-candidate"
export MACE_CANDIDATE_BIN_SHA256="$(sha256sum "$MACE_CANDIDATE_BIN" | awk '{print $1}')"

MACE_SOURCE_REVISION="$(git rev-parse HEAD)" \
cargo run --release \
  --manifest-path research/broca-mace-transfer-bench-v0-1/Cargo.toml \
  -p broca-mace-evaluator \
  --bin mace_counterfactual_evidence \
  -- 1 /tmp/mace-candidate.jsonl /tmp/mace-evaluator.jsonl
```

The evaluator requires both `MACE_CANDIDATE_BIN` and `MACE_CANDIDATE_BIN_SHA256`; missing or mismatched values fail closed. The child process runs with a cleared environment and working directory `/`, but remains under the parent user's OS identity. The digest pins the candidate executable file bytes only; it does not pin dynamic shared libraries, the host kernel, CPU microcode, runtime image or sandbox configuration. Those must be included in a later container/attestation manifest. The nested MACE workspace still lacks its own Cargo.lock; dependency-lock reproducibility is tracked in [#13364](https://github.com/Luminous-Dynamics/symthaea/issues/13364).

### Timing contract

Candidate and evaluator records now carry:

- `candidate_process_startup_ns`: worker spawn/readiness time; repeated on each detached observation from the same run-scoped worker. Aggregate once per distinct `(run_id, candidate_artifact_sha256)`, not once per row.
- `candidate_request_build_ns`: parent nonce generation, evidence ordering, wire construction/serialization and expected-digest preparation.
- `candidate_process_wall_ns`: from writing/flushing the request until the response line is received. It includes candidate execution and IPC/scheduling; it is **not** pure network/IPC transport.
- `candidate_latency_ns` / score-row `latency_ns`: candidate-internal retrieval/policy timing reported by the child after request parse/validation.
- `candidate_receipt_validation_ns`: parent JSON receipt decoding and semantic/identity validation.
- `candidate_end_to_end_ns`: parent-side request-build through receipt validation. It excludes worker startup, which is recorded separately.

Three remaining baseline arms—event-ID KV top-k, static HDC and HDC-LTC zero-input—are still explicitly labelled `legacy_in_process`; they have null child-process metadata. The detached exact-scope arm returns all eligible in-scope evidence as a ceiling, whereas typed-content HDC returns its top-k candidate set. Do not describe all five arms as process-isolated or compare them without accounting for their different execution boundaries.

### Updated validation boundary

Static source/schema inspection now covers the child-process launcher, executable digest pinning, readiness and request/response framing, strict receipt validation, support-ID reconciliation, explicit detached/legacy labels and the timing fields. Regression tests are authored in the reducer. **No Rust build, test, runtime child process, sandbox or benchmark run has been executed in this environment.**

This process integration still does not provide OS sandboxing, evaluator custody, candidate source authenticity or sealed truth. The parent evaluator and fixture generator remain public and in the same executable. It is progress toward #13287, not completion of #13287.

# MACE Evaluator Qualification Harness v0.1

**Status:** executable reference prototype; not an independent benchmark certification.

This package implements a small, dependency-free Go harness for qualifying benchmark graders and result adapters. It complements the MACE benchmark-integrity protocol in the parent branch.

## Why this exists

A benchmark grader must be tested as an instrument. A reference solution passing is insufficient: a grader that returns PASS for every run will accept both a reference solution and a no-op. Qualification therefore requires positive controls, negative controls, and condition-removed twins that demonstrate sensitivity to the intended defect.

MACE reports three independent dimensions:

- task_result: did the candidate satisfy the benchmark task?
- effect_assurance_result: were consequential external effects authorized and evidenced?
- benchmark_integrity_result: did the evaluator and its evidence pipeline remain valid?

These dimensions are not collapsed into a weighted score. A payment can satisfy the requested task while being executed by the wrong principal; task utility and effect assurance must preserve that distinction.

## Contents

- model.go: versioned inputs/reports and exact-byte SHA-256 helpers.
- RESEARCH_PROGRAMME.md: benchmark comparison, accountability crosswalk, proposed targets and execution roadmap.
- MACE_EFFECT_FENCE_BENCHMARK.md: destination-side effect-or-fence semantics, fault matrix, metrics and qualification targets.
- BENCHMARK_AUDIT_LANDSCAPE.md: 2026 evaluator/oracle/trajectory-auditing research and MACE differentiation plan.
- qualify.go: fail-closed corpus, run-manifest, observation and metamorphic-relation checks.
- cmd/mace-evalqual: offline CLI.
- fixtures/: synthetic demonstration corpus, pinned manifest and corresponding observations.
- qualify_test.go: deterministic adversarial and statistical-semantics tests.
- qualify_grader_mutation_test.go: grader-integrity negative-control and condition-removal tests.
- schemas/: Draft 2020-12 structural schemas; cross-record invariants are enforced by the Go validator.
- verify.sh: format, tests, race detector, vet, build and demo CLI checks.
- oraclelab/: snapshot/event based non-interference, repeatability, evaluation-order and Linux filesystem observer checks for graders.

## Run

Requires Go 1.23+ and uses only the standard library.

    bash verify.sh

    go run ./cmd/mace-evalqual validate -corpus fixtures/corpus.json -manifest fixtures/manifest.json

    go run ./cmd/mace-evalqual score -corpus fixtures/corpus.json -manifest fixtures/manifest.json -observations fixtures/observations.json -out report.json

The score command exits successfully only when every registered fixture matches its expected three-dimensional result and all paired relations pass. Missing, duplicate, unregistered or mismatched observations reject before scoring.


## Deterministic effect-observer laboratory

The observerlab package adds a separate simulator event ledger and observer-stream comparator. Its synthetic trace covers authorization, dispatch, external effect and reconciliation, linked by explicit parent event IDs.

Run the dropped-event demonstration:

    go run ./cmd/mace-observerlab

The command deliberately omits the external-effect event. Its report must identify the missing event and set accepted=false. The test suite additionally injects fabricated events, duplicates/replays, principal and payload substitutions, parent substitutions, source-sequence substitutions and transport reordering.

The comparator distinguishes delivery sequence from source sequence. A reordered transport stream is recorded as a delivery diagnostic but is not rejected solely for arrival order when exact source sequence and parent references remain correct. This prevents a clock/order heuristic from replacing explicit causal evidence.

This remains a simulator laboratory: ground truth is authored by the fixture and is not an independent OS or service oracle. The next step is to replace the fixture's event source with independently instrumented filesystem/process/network/API boundaries while retaining this comparator as a conformance oracle.

## Destination effect-or-fence experiment

The `effectbridge` package is a stateful destination fixture, not a broker or authority issuer.

A normal `GET /v1/reconcile/{key}` result of `not_found_unfenced` remains indeterminate: a delayed request may still arrive after the lookup. The destination's `PUT /v1/fence/{key}` serializes against `POST /v1/effects` under the same journal lock and durably commits an exact-intent tombstone. After a fence commits, later arrivals for that key are rejected. The two terminal states are mutually exclusive: effect committed or fence committed.

Run the local service:

    go run ./cmd/mace-effectsvc -listen 127.0.0.1:8087 -journal ./effects.jsonl

The service has no production authentication/TLS. Keep it on loopback and use disposable test state. The modeled business effect is the journal commit itself; a real destination must atomically commit its business mutation and idempotency/fence state, or retain indeterminate.

This composes with #11647 (durable broker/IPC/grant accounting) and #9490 (action authority). It does not mint permits or claim distributed exactly-once behavior.


## Corpus contract

Each corpus pins benchmark name, exact version/revision, split and task-set digest. Each fixture has a unique ID, task, family, kind and expected result vector.

Supported kinds:

- positive: known-good/reference state; all dimensions must pass.
- negative: registered fault/attack; its target dimension must fail or remain indeterminate.
- condition_removed: removes only the registered negative condition and must restore the positive baseline.
- representation_twin: varies an irrelevant representation while preserving semantic outcome.

A negative case names a positive baseline and a condition-removed twin. Links are reciprocal, and all three share the same task and family. Every family must include at least one positive, negative and condition-removed case. These relations make it harder for a reject-everything grader to qualify or for a negative control to change several conditions at once.

The ten-fixture synthetic demonstration exercises all three failure dimensions: a no-op task failure, wrong-principal effect failure, and grader-test tampering that fails benchmark integrity while task utility remains successful. Each negative has a condition-removed twin.

The corpus declares expected outcomes; it does not execute candidates. A benchmark-specific adapter must run the reference, no-op, mutated and condition-removed cases, query its native grader and independent state/event oracle, and produce observations in the declared format.

## Run-manifest and evidence binding

The manifest pins benchmark revision/split/task set; model/provider; agent scaffold; prompts/policy; memory; tool schemas; retry and network policies; environment image; grader; independent oracle; reporter; host/runtime; seed; and hard budgets.

Observations bind the exact raw bytes of the corpus and manifest with SHA-256. Reformatting those files requires regenerating the observation bindings. This v0.1 implementation uses exact-byte hashing; it does not claim RFC 8785/JCS, cross-language canonicalization, signatures, or producer authentication.

## Scoring and uncertainty

The report separately counts per-dimension expected/observed outcomes, false accepts, false rejects, unexpected indeterminates and mismatches. It also counts condition-removed recoveries and representation-invariance pairs.

- A false accept is an expected fail or indeterminate observed as pass.
- A false reject is an expected pass observed as fail.
- Missing evidence remains a validation error; indeterminate is not converted to pass/fail.
- not_applicable is currently permitted only for effect assurance when a fixture has no external effect to qualify.

A fixed mutation corpus is a conformance challenge, not a random production sample. Zero accepted mutations means zero observed accepts in that registered corpus; it is not a universal population error rate.

For a declared IID random sample only, the harness can emit the exact one-sided 95% zero-error upper bound 1 - 0.05^(1/n), assuming truly independent draws from the pinned frame. It cannot verify the sampling declaration or correct for clustering. For example, 300 independent IID negative trials yield an upper bound of approximately 0.994%; 300 hand-authored mutations, repeated seeds of one task, or correlated members of 30 task families do not justify that inference.

## Limits

This is an offline result-bundle checker. It does not execute or sandbox benchmark agents/graders, inspect OS/process/filesystem/network activity, prove observer completeness, authenticate artifact producers, or establish administrative independence. SHA-256 identifies bytes; it does not prove their truth.

The demonstration fixtures are synthetic plumbing examples. They are not measured external-agent results, a real tool integration, a production broker test, independent review, benchmark certification or SOTA result. The next gate is one deterministic effect-origin simulator whose independently maintained ground-truth event log can be compared against runtime observer and gateway receipts, including intentional observer gaps.


## Observer independence profile

The proposed E0–E4 observer taxonomy and trust-domain vector are documented in [OBSERVER_INDEPENDENCE.md](OBSERVER_INDEPENDENCE.md). The current separate-process SQLite audit companion is E2: useful for database-snapshot reconciliation, but not an external-service or independent-host witness. The public branch contains the profile and scope contract; the executable SQLite observer remains a separate local companion pending promotion and review.


## Formal protocol model

A bounded TLA+ model and pinned TLC runner are available in [formal/](formal/README.md). The specification covers delayed dispatch, destination serialization, staged effect/fence transactions, rollback and reconciliation. The Go state explorer has passed its declared finite test; the new TLA+ module has **not yet been run through TLC** in this environment. The runner records the stable tool pin and refuses an unverified JAR.


## Detached checkpoint chain

The proposed Ed25519 checkpoint wire contract, monotonic consumer-state rules and cross-language test vector are documented in [CHECKPOINT_CHAIN.md](CHECKPOINT_CHAIN.md). The public branch publishes the JSON test vector and schema. A local companion Go verifier passes its signature/rollback/fork tests, but the signer, independently administered witness and external trust-state store are not deployed; the Go implementation itself is not part of this public PR's source diff.


## Detached checkpoint chain

The proposed Ed25519 checkpoint wire contract, monotonic consumer-state rules and cross-language test vector are documented in [CHECKPOINT_CHAIN.md](CHECKPOINT_CHAIN.md). The public branch now includes the executable standard-library Go verifier under `checkpoints/`, 16 focused tests and the public test vector. The signer, independently administered witness service, protected external trust-state store and key custody are not deployed. The verifier does not turn this into an E3/E4 witness or certification.

The public checkpoint verifier also exposes `AcceptObserved`, which joins the signature to an observer-supplied database ID, journal sequence and head digest from one coherent snapshot before advancing the external trusted cursor.

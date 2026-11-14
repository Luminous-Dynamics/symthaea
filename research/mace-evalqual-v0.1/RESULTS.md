# MACE Evaluator Qualification Harness v0.1 — verification record

**Status:** local prototype verification only. This is not a GitHub Actions result, production integration, external benchmark run, independent review or certification.

## Source and environment

- Source directory: research/mace-evalqual-v0.1/
- Language/runtime: Go 1.23.2
- Host: Linux/amd64, kernel 6.18.57
- Implementation and tests: Go standard library only
- Public source branch: research/mace-evalqual-v0.1 (draft PR #11965, stacked on PR #11921)

## Local checks

| Check | Result |
|---|---|
| gofmt | Pass; no unformatted Go files |
| go test -count=1 ./... | Pass; 16 test functions |
| go test -count=1 -race ./... | Pass |
| go vet ./... | Pass |
| CLI build | Pass |
| CLI validate on demo corpus/manifest | Pass |
| CLI score on complete demo observations | Pass; exit 0 |
| Draft 2020-12 schema validation | Pass for all three schema/fixture pairs with jsonschema 4.26.0 |
| Core package statement coverage | 73.7% |

The CLI package has no isolated Go unit tests; verify.sh exercises both CLI subcommands end-to-end on synthetic fixtures.

## Demonstrator output

- Fixtures: 10
- Exact expected/observed three-axis matches: 10/10
- Positive fixtures: 3
- Negative fixtures: 3
- Condition-removed pairs: 3/3
- Representation-invariance pairs: 1/1
- False accepts: 0 across the three fixed negative fixtures
- False rejects: 0
- Unexpected indeterminates: 0
- Population inference: not permitted; the corpus is fixed and synthetic
- Example report digest: a5ae6242b29d6ef4f33c904babffe4c3ea9f2a27dd33e04c07a4fbac4a7b439c

The examples include a no-op task failure, wrong-principal effect failure while task utility passes, and grader-tampering benchmark-integrity failure while task utility passes. These are synthetic expected/observed fixture vectors, not real external-agent results or live payment/grader tests.


## Deterministic effect-observer laboratory

- Nine observerlab tests pass under the standard test and race-detector runs.
- Clean four-event authorization → dispatch → external-effect → reconciliation trace: accepted.
- Deliberately dropped external-effect event: rejected; one missing event reported.
- Fabricated event: detected as extra.
- Duplicate event replay: detected.
- Principal and payload substitutions: detected independently.
- Parent-receipt substitution: detected; causal graph marked invalid.
- Source-sequence substitution: detected.
- Transport delivery reordering with unchanged source sequences and parent links: accepted as causally valid and counted as a delivery reorder.
- Unknown/non-causal ground-truth parent: rejected before comparison.
- Observerlab statement coverage: 87.9%.

The observer CLI is a self-test demonstration that intentionally supplies an incomplete observer stream; its expected behavior is to report rejection. This simulator-authored event ledger is not an independent production observer and does not establish real-world coverage.

## Input digests

- Corpus: 9aa3da88418e0e44511f2da57efc38057c66546d75cc7620b0fdffa0318de4be
- Run manifest: 5fe2ff681e99d919516ed0ffdaebe8d585a9f7093dfa40354e6ae6b7c59495d6
- Observations: d66742e7ad677cd5e7d333f66ca4f188a1be028cb2877249cbc67b2cd00ac66f


## Oracle non-interference laboratory

A new `oraclelab` package adds snapshot + observer-event checks for the evaluator itself.

- **19 test functions pass** in the local standalone Go package (Go 1.23.2/Linux amd64).
- `go test -count=1 ./...`: pass.
- `go test -count=1 -race ./...`: pass.
- `go vet ./...`: pass.
- Statement coverage: **70.9%**.
- Persistent state mutation: rejected.
- Transient write-then-restore: rejected from the revision/event evidence even when the final content digest equals the initial digest.
- Missing event, incomplete observer coverage, uninstrumented path, scope mismatch and invocation-attribution mismatch: indeterminate; the evaluator is not run where coverage is already known to be incomplete.
- Repeated-query output instability: detected with a shared evaluator instance and exact pristine-state reset between invocations.
- A→B versus B→A evaluation-order dependence: detected.
- Grader errors and panics: grade remains indeterminate; state mutation after error still fails non-interference.
- Pure read-only oracle: passes the non-interference check.

**Trust boundary:** the lab's in-memory world and event recorder are synthetic. It assumes the adapter's independently operated event source honestly and completely covers the declared scope. Final-state snapshots alone cannot detect transient writes that are restored; those require event monitoring. A dishonest or incomplete event source can still hide activity. Production qualification therefore requires a domain-native snapshot/transaction-log source plus an observer outside the evaluator's trust domain.

This package qualifies neither a production sandbox nor the real observer, and does not prove that the oracle's expected labels are correct. It is a reusable test contract for the next adapter.


### Linux filesystem observer experiment

A real local Linux filesystem observer adapter was added alongside the in-memory oracle harness.

- `ObserveDirectory` installs recursive inotify watches on a pinned existing directory tree, captures pre/post content digests, and keeps the mutation event stream as a separate observation.
- Root device/inode is checked before and after evaluation. Nested paths are compared with the most-specific mount ID from `/proc/self/mountinfo`.
- The parser uses the 16-byte Linux inotify wire header explicitly. An initial implementation incorrectly used Go's 20-byte `syscall.InotifyEvent` memory size; it corrupted event boundaries and failed transient-write tests. That was corrected before publication.
- 12 focused filesystem observer tests, three root/mount hardening tests, and two cross-process mutation tests pass.
- Cases include read-only, persistent writes, transient write/restore with unchanged final content, transient create/delete, nested file changes, symlinks, dynamic subdirectories, callback panic, root replacement, mountpoint selection and live mount-table parsing.
- The combined local `oraclelab` package has 36 test functions (19 in-memory non-interference, 12 filesystem observer, 3 root/mount hardening, 2 child-process mutation tests). Go 1.23.2 / Linux amd64:
  - `go test -count=1 ./...`: pass.
  - `go test -count=1 -race ./...`: pass.
  - `go vet ./...`: pass.
  - `go test -cover ./oraclelab`: 70.0% statement coverage.

The filesystem observer records its limitations in the coverage result: it trusts the host kernel and mount namespace, requires all child writers to be joined before callback return, excludes symlinks/special files/new subdirectories, and does not observe remote service state or privileged host activity outside scope. It is a local OS-event prototype, not a protected independent production witness.

These checks are still standalone package evidence. No combined exact-PR-checkout replay, GitHub Actions run, disposable remote-service adapter, actual database transaction-log comparison or broker experiment is claimed.


## Destination-side effect-or-fence experiment

The new `effectbridge` Go package and `mace-effectsvc` command provide a local destination-side protocol fixture.

- The service uses a single-writer append-only JSONL journal, hash-chained records, exact intent digests, Linux non-blocking `flock`, parent-directory sync on open, and `fsync` per terminal record.
- `POST /v1/effects` durably commits an effect or returns the same receipt for an exact duplicate; a changed intent under the same key conflicts.
- `GET /v1/reconcile/{key}` returns an existing effect/fence, `not_found_unfenced`, or journal-indeterminate. An ordinary absence never proves non-execution.
- `PUT /v1/fence/{key}` linearizes with effect dispatch and durably commits a tombstone that rejects later requests for the exact idempotency key/intent.
- The client maps a transport failure to unknown/indeterminate and does not auto-retry a non-idempotent dispatch. A lost fence acknowledgement can be resolved by repeating the same idempotent fence.
- Local expanded tests include eight independent client processes, process restart after remote success with lost acknowledgement, process ownership exclusion, dispatch/fence concurrency, fence-ack loss and recovery, and 10,000 seeded protocol operations compared with a separate finite-state reference model.

**The 10,000 figure is generated sequential protocol operations, not 10,000 independent crash/restart schedules.** The service's modeled business effect is its journal commit; no separate business database is integrated. Process kill/restart is not a power-loss/device-durability test.

Local expanded-source verification (Go 1.23.2 / Linux amd64):
- Root evaluator package: 16 tests.
- observerlab: 9 tests.
- effectbridge: 24 tests.
- Total across this local assembled snapshot: 49 named test functions.
- `go test -count=1 ./...`, race detector, `go vet`, three CLI builds and `verify.sh`: pass.
- effectbridge statement coverage: 62.1%.
- Draft 2020-12 effectbridge schema: metaschema check and 11 semantic fixtures pass. These exercise request, effect receipt, fence receipt, found, fenced, ordinary absence and indeterminate states, plus invalid status/receipt combinations.

The current public PR branch carries the reference service, client, locking implementation, CLI, schema and a focused conformance test file. The expanded local test source is larger than that focused public test file. No hosted workflow or exact combined PR-head replay is claimed. The portable source archive is a local snapshot, not a substitute for public commit history.

## Claim boundary

This implementation validates a benchmark adapter's declared fixture catalogue, exact input-byte bindings, typed three-axis outcomes and paired metamorphic relations. It does not launch or sandbox tasks; verify grader behavior in a real benchmark; prove task/oracle semantics; authenticate producers; establish oracle independence; observe all OS/API/filesystem/network events; or qualify the durable one-shot broker.

No benchmark leaderboard, SOTA, deployment-safety or production-readiness result is claimed.


## 14 November 2026 — concurrent history linearizability

Added `effectbridge/linearizability_test.go` as a dependency-free executable sequential specification and backtracking history checker. Each of 128 concurrent histories launches four overlapping operations against one idempotency key: append intent A, fence intent A, reconcile the key, and append changed intent B. The checker searches all candidate sequential orders while enforcing real-time precedence (an operation that returned before another was invoked must linearize first). It verifies the observed receipt kind, duplicate flag, conflict/fenced result and reconciliation status against the model, then closes and reopens the journal and verifies that the single terminal decision is unchanged.

Local run against the assembled MACE v0.2 source snapshot (Go 1.23.2, Linux/amd64): the new test passed; `go test -race -count=3 ./effectbridge -run TestConcurrentEffectFenceLookupHistoriesAreLinearizable` passed; `go vet ./...` passed; and `go test -count=1 ./...` passed. This test was run against the assembled local snapshot, not a clean replay of the exact public PR checkout. The public branch now contains the test; hosted CI status must be checked separately.

Claim boundary: this is a bounded history checker for the single-process JSONL journal interface. It is not a formal proof, not exhaustive over all possible schedules, and not a test of network partitions, multi-host consensus, SQLite's transaction implementation, or hardware power loss. The next stronger layer is to export captured histories into an independent linearizability checker and model the protocol in TLA+/PlusCal, then run fault-injected service histories against a real destination.


## 14 November 2026 — finite-state safety exploration

Added `effectbridge/modelcheck_test.go`, a dependency-free breadth-first explorer for a bounded destination protocol model. The finite model includes two intent classes, delayed requests, delivery, an exclusive effect/fence transaction boundary, staged receipt and business writes, atomic commit, rollback/process crash before commit, and derived reconciliation states. It checks invariants at every reachable state: an effect is visible only with both its receipt and business row; a fence has a receipt but no business row; no transaction overlaps a committed terminal state; and `not_found_unfenced` is reported only while no terminal decision exists.

Observed local exploration: **68 unique reachable states, 200 transitions, maximum shortest trace depth 7**, with effect, fence, late-arrival-after-fence and rollback-with-pending-request cases all reached. Local verification on the assembled v0.2 source snapshot (Go 1.23.2/Linux amd64): both new tests passed under three consecutive race-detector runs; `go vet ./...`, default `go test -count=1 ./...`, and optional SQLite-tagged `go test -count=1 -tags mace_sqlite ./...` passed.

This is exhaustive only for the small declared finite abstraction. It is not TLC/TLA+ verification, not a proof of the Go implementation, and not a model of arbitrary message multiplicity, multiple hosts, storage firmware, power failure or Byzantine destinations. The next formal step is to encode the same invariants in TLA+/PlusCal and independently check captured implementation histories with a separate linearizability checker.


## 14 November 2026 — generated multi-key concurrent histories

Added `effectbridge/linearizability_schedules_test.go`. With a fixed reproducible seed, it generates 256 histories of six overlapping operations over two idempotency keys and two possible intents. Each operation is an append, fence or lookup. A legal sequential history must match every returned result and respect real-time precedence; each key's terminal lookup is also compared before and after journal restart.

Local execution on the assembled companion snapshot (Go 1.23.2/Linux amd64): generated-history test passed; the three linearizability/model tests passed three consecutive race-detector runs; default tests, SQLite-tagged tests, `go vet`, and `CGO_ENABLED=0` tests passed. These 256 fixed-seed histories are a bounded conformance corpus, not IID trials or a population reliability estimate. The local companion snapshot has 52 named test functions by default and 67 with the SQLite tag. SQLite source remains outside this public PR diff.


## 14 November 2026 — observer trust-domain qualification profile

Added public OBSERVER_INDEPENDENCE.md to formalize a proposed E0–E4 observer classification and the full trust-domain vector (producer, process, host/kernel/storage, credentials/admin, network, clock/sequence, verifier and retention). The profile explicitly classifies the separate read-only SQLite audit companion as E2 only. It states that the companion verifies a database snapshot but cannot prove downstream payment/actuation, cannot independently attest the writer's connection-scoped synchronous pragma, and shares the same host/storage/admin failure domain.

The E2 implementation is a local companion source artifact, not part of this public branch as of this document revision. Publicly committed are the profile, test protocol and next E3/E4 research gates. The companion's local tests and benchmarks remain reported separately in the handoff archive and must not be treated as hosted CI or independent certification.

## 14 November 2026 — local E2 observer checkpoint boundary

The local companion now includes an expected-head comparison that illustrates a further boundary: a hash chain proves internal consistency but does not authenticate an entire database against a privileged full rewrite. A test rewrites a receipt and matching business row, recomputes the intent/record hashes, and shows that the unanchored audit accepts the internally consistent replacement while the original out-of-band head rejects it. The caller-supplied checkpoint API and test remain local companion source, not this public PR diff; this public document records the trust model and E3/E4 follow-up. No remote checkpoint service, signature or key custody is implemented.


## 14 November 2026 — bounded TLA+ specification authored (TLC pending)

Added formal/MACEEffectFence.tla, formal/MACEEffectFence.cfg and formal/run-tlc.sh. The model contains two idempotency keys and two intents, delayed pending dispatch, globally serialized effect/fence transactions, receipt/business staging, atomic commit, rollback/process crash and derived reconciliation outcomes. Invariants check type correctness, atomic visibility, transaction coherence, intent binding, reconciliation soundness and that a fenced key never exposes a business effect.

The runner pins the stable TLA+ tools v1.7.4 release using the upstream-published SHA-1 for tla2tools.jar and runs a single TLC worker. TLC was **not run** in this environment: Java 21 is available but no TLA+ JAR is installed, and outbound artifact retrieval is unavailable. The Go finite-state model remains a separate, executed test (68 states / 200 transitions); it is not a substitute for TLC. Do not count the formal tier as passed until the pinned runner succeeds and records its state/transition counts and tool version.


## 14 November 2026 — detached signed checkpoint wire contract

Added public CHECKPOINT_CHAIN.md, a Draft 2020-12 JSON schema and a synthetic cross-language test vector publishing exact canonical signing bytes, public key, signature hex/base64 and envelope digest. The v1 contract uses Ed25519 over a fixed domain-separated binary encoding, links each checkpoint to the previous signed envelope and requires an external monotonic consumer cursor to reject rollback/forks.

The Go verifier and its 16 test functions are in the separate local companion package, not the public PR diff. It validates signature, database/key identity, sequence continuity, predecessor, journal-head progress, issue-time monotonicity, exact replay and malformed trust state. It does not run a witness service, protect private keys, persist the trusted cursor or solve key rotation/quorum/split-view detection. No E3/E4 deployment claim is made.


## 14 November 2026 — signed checkpoint verifier promoted to public branch

The standard-library Go Ed25519 checkpoint verifier is now committed under `checkpoints/`, along with 16 focused tests, README and a package-local copy of the published synthetic interoperability vector. It implements the fixed domain-separated binary signing representation, detached signature verification, envelope digest, bootstrap identity binding and monotonic `AcceptNext` rules. The public root `verify.sh` now runs the package under default/race/vet checks, builds all three public CLIs, and conditionally invokes TLC only when a pinned v1.7.4 JAR is supplied.

Local verification of the assembled v0.8 companion source after aligning the public package contents: focused checkpoint tests, focused checkpoint race test and vet passed; full `bash verify.sh` passed, including default and SQLite-tagged suites/race/vet/build legs. The local companion has 68 named test functions in the default build and 90 with the optional SQLite tag. Those aggregate counts and SQLite-tagged results describe the local companion, not the exact public checkout; the public PR now includes the 16-test checkpoint package and the updated public verifier script.

The public verifier is a building block, not a deployed witness. It does not provision or custody production keys, protect its own trusted cursor, operate a quorum/transparency log, provide split-view gossip, or prove downstream effect occurrence. TLC remains pending unless the pinned runner is executed and its state counts/results are archived.


## 14 November 2026 — independent Ed25519 implementation cross-check

Added `checkpoints/openssl_interop_test.go`. Where OpenSSL CLI is installed, the test reconstructs the public test vector's exact domain-separated signing bytes, packages the public Ed25519 key as RFC 8410 SubjectPublicKeyInfo DER and asks OpenSSL `pkeyutl` to verify the detached signature. Local test passed with OpenSSL 3.5.5. If OpenSSL is absent, the test skips with an explicit reason; the Go standard-library vector verification remains required. This checks the signing bytes/signature across two crypto providers, not the full monotonic checkpoint state machine in an independent implementation.


## 14 November 2026 — checkpoint-to-observed-head join

Added `ObservedHead` and `AcceptObserved` to the public checkpoint verifier. It checks that the observed database identity, journal sequence and journal-head digest match the detached signed checkpoint before applying monotonic sequence/predecessor rules. Four tests cover the valid join and rejection of a mismatched database, journal sequence and digest. The observer must produce the three values from one coherent snapshot; this API prevents an omitted comparison but cannot establish observer independence or downstream effect coverage.

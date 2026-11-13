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

## Claim boundary

This implementation validates a benchmark adapter's declared fixture catalogue, exact input-byte bindings, typed three-axis outcomes and paired metamorphic relations. It does not launch or sandbox tasks; verify grader behavior in a real benchmark; prove task/oracle semantics; authenticate producers; establish oracle independence; observe all OS/API/filesystem/network events; or qualify the durable one-shot broker.

No benchmark leaderboard, SOTA, deployment-safety or production-readiness result is claimed.

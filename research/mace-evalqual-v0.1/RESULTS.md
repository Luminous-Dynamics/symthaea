# MACE Evaluator Qualification Harness v0.1 — verification record

**Status:** local prototype verification only. This is not a GitHub Actions result, production integration, external benchmark run, independent review or certification.

## Source and environment

- Source directory: research/mace-evalqual-v0.1/
- Language/runtime: Go 1.23.2
- Host: Linux/amd64, kernel 6.18.57
- Implementation and tests: Go standard library only
- Public source branch: research/mace-evalqual-v0.1 (stacked on the benchmark-integrity protocol branch)

## Local checks

| Check | Result |
|---|---|
| gofmt | Pass; no unformatted Go files |
| go test -count=1 ./... | Pass; 14 test functions |
| go test -count=1 -race ./... | Pass |
| go vet ./... | Pass |
| CLI build | Pass |
| CLI validate on demo corpus/manifest | Pass |
| CLI score on complete demo observations | Pass; exit 0 |
| Draft 2020-12 schema validation | Pass for local schema/fixture pairs with jsonschema 4.26.0 |
| Core package statement coverage | 73.3% |

The CLI package has no isolated Go unit tests; verify.sh exercises both CLI subcommands end-to-end on synthetic fixtures.

## Demonstrator output

- Fixtures: 7
- Exact expected/observed three-axis matches: 7/7
- Positive fixtures: 2
- Negative fixtures: 2
- Condition-removed pairs: 2/2
- Representation-invariance pairs: 1/1
- False accepts: 0 across the two fixed negative fixtures
- False rejects: 0
- Unexpected indeterminates: 0
- Population inference: not permitted; the corpus is fixed and synthetic

The examples include a no-op that must fail task utility and a principal-substitution mutation that must fail effect assurance while task utility remains successful. These are declared synthetic expected outcomes, not observations from an external agent or real payment system.

## Claim boundary

This implementation validates a benchmark adapter's declared fixture catalogue, exact input-byte bindings, typed three-axis outcomes and paired metamorphic relations. It does not launch/sandbox tasks; verify grader behavior in an actual benchmark; prove task/oracle semantics; authenticate producers; prove source independence; observe all OS/API/filesystem/network events; or qualify the durable one-shot broker.

No benchmark leaderboard, SOTA, deployment-safety or production-readiness result is claimed.

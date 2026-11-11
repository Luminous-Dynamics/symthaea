# MACE-CoS paired-world noninterference oracle v0.1 — results

Date: 2026-11-11.

## Local reproducibility

Environment: Go 1.23.2, linux/amd64. Module uses the Go standard library only.

| Check | Result |
|---|---:|
| `go test -count=1 ./...` | PASS |
| `go test -race -count=1 ./...` | PASS |
| `go vet ./...` | PASS |
| `gofmt` cleanliness | PASS |
| Named test functions | 16 |
| Generated protected-lineage chain cases | 363 / 363 rejected as expected |
| Statement coverage | 75.7% |
| Third-party Go dependencies | 0 |

The test suite covers direct flows, aliases/copies, hashing, merge lineage, exact and expired declassification grants, purpose/audience rebinding, equal-secret negative controls, public-input matching, irrelevant protected mutations, output-value hash omission, and program/policy digest binding.

The generated chain test exhaustively enumerates all 3^d operator sequences for depths d=1..5 over copy, SHA-256 and concat, for 363 total cases. Every generated path has a protected source ancestor and must be rejected at the low sink without a release grant.

## Claim boundary

These are local results for this finite, deterministic Go DSL. They do not establish arbitrary LLM or tool-runtime noninterference, hidden-state/timing/network side-channel security, external-action authorization, issuer authenticity, or benchmark superiority. The report digest binds program and policy definitions but does not publish or unsalted-hash protected world values. Exact pair fixture custody and any secret commitment belong to the evaluator-side harness and must not be exposed to the agent.

This package is a companion research oracle for #11601 and MACE-CoS #11594, not a production reference monitor.

# CPDC synthetic preflight result 1.1

Environment: Go 1.23.2, linux/amd64.

Commands completed locally:
- `go test -race -count=5 ./...` — pass.
- `go vet ./...` — pass.
- `go run . cases-v1.json` — emitted the committed machine-readable report.
- `go test -count=1 -cover ./...` — 59.7% statement coverage.
- seven named test functions.

Synthetic results over 15 deterministic fixtures:
- citation-only: 4/15 exact; 8 false-current, 4 missed reauthorizations, 11 unauthorized allows, 1 missed current-risk review.
- snapshot-only: 10/15 exact; 4 missed reauthorizations, 1 missed current-risk review.
- lineage-aware: 15/15 exact; zero counted safety errors.

The v1.1 corpus adds a temporal anti-retroactivity case: a pending action queried at a historical time remains evaluated against the source state known then, even when a later correction has since superseded it. The implementation separately computes current risk review from current source status.

Results are synthetic, policy-baseline comparisons only. The lineage-aware implementation scores 15/15 because it encodes the expected rule set; this is not an independently measured system, external-product comparison, profile-conformance result, or SOTA claim.

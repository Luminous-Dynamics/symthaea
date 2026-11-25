# CPDC synthetic preflight result 1.0

Environment: Go 1.23.2, linux/amd64.

Commands completed locally:
- `go test -race -count=5 ./...` — pass.
- `go vet ./...` — pass.
- `go run . cases-v1.json` — emitted the committed machine-readable report.
- `go test -count=1 -cover ./...` — 58.0% statement coverage.
- ZIP archive integrity check — pass.

Results are synthetic, policy-baseline comparisons only. The lineage-aware implementation scores 14/14 because it encodes the expected rule set; this is not an independently measured system, external-product comparison, profile-conformance result, or SOTA claim.

See `results-v1.json` for per-metric counts and `README.md` for metric definitions and limitations.

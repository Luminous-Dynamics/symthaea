# MON-029 — Verified Artifact Consumption

MON-029 closes the gap between verifying a custody bundle and consuming the evidence inside it.

The previous MON-028 scoring CLI correctly verified the complete MON-026 bundle, but then reopened selected JSON artifacts through a plain path-based read. MON-029 adds a small Go reader module that checks the selected file's exact signed size and SHA-256 on the bytes it will parse.

## What changed

- Revalidates each selected suite, candidate-result and optional baseline artifact against its unique entry in the already custody-verified manifest.
- Rejects unsafe relative paths, symlinks, non-regular files and path substitution detected between inspection and open.
- Reads at most 256 MiB for structured JSON artifacts.
- Hashes the in-memory byte sequence and parses that same byte sequence.
- Rejects unknown JSON fields and trailing/multiple JSON documents.
- Adds adversarial tests for same-size substitution after custody verification, size mismatch, duplicate/unlisted paths, traversal, symlinks, oversized input and invalid JSON.

## Use and validation

The MON-028 CLI consumes the reader as a local module dependency. A caller must still verify the exact MON-026 manifest with the proper external trust policy before invoking the reader; the reader does not authenticate signatures.

Run the MON-029 module tests:

`sh
cd benchmarks/frontend/monitoring/sfeb-mon029/reader
go vet ./...
go test ./...
go test -race ./...
`

Run the combined MON-028 and MON-029 checks:

`sh
bash benchmarks/frontend/monitoring/sfeb-mon028/CHECKS.sh
`

## Status

Implementation and test source are committed on the MON-029 research branch. Exact-head Go execution, race detection and integration execution have not yet been run in this authoring environment. No live database recovery, signer integration or production qualification is claimed.

# MACE-PI-001 results — 2026-11-10

## Environment

- Go 1.23.2
- linux/amd64
- standard library only
- Rust/Cargo unavailable in the execution environment

## Local verification

| Check | Result |
|---|---:|
| Top-level tests | 13/13 passed |
| Generated cohorts vs exhaustive oracle | 256/256 passed, for every allowed MTR violation budget |
| Race detector | PASS |
| go vet | PASS |
| gofmt | clean |
| Statement coverage | 77.4% |
| Third-party dependencies | 0 |

Commands:
```sh
gofmt -w *.go
go test -count=1 ./...
go test -race -count=1 ./...
go vet ./...
go test -cover ./...
go run . --input fixtures/example.json
go run . --input fixtures/example.json --mtr
```

## Canonical fixture

Observed treated-minus-control risk difference: `2/3`.

Without MTR restriction:
- ATE lower bound: `-1/4`.
- ATE upper bound: `3/4`.
- At threshold `1/10`: `NOT_IDENTIFIED`.

With individual MTR:
- ATE lower bound: `0/1`.
- ATE upper bound: `3/4`.
- At threshold `1/10`: `NOT_IDENTIFIED`.

Allowing one MTR violation restores the lower bound to `-1/4`. The sensitivity curve makes the effect of the assumption explicit.

## Interpretation and boundaries

The observed association is not the ATE. The interval is sharp for the fully enumerated finite cohort under the declared consistency, no-interference and well-defined binary treatment assumptions. It has no sampling uncertainty and does not generalize to a population or a different regime.

Assumptions are explicit declarations, not authenticated facts. The dataset digest is byte identity, not proof of source issuer authenticity, random assignment or intervention delivery.

This is not an implementation of the Jung & Kang information-theoretic causal bounds estimator or the NeurIPS 2025 data-fusion estimator. No external benchmark, model, randomized trial, population causal estimate, Rust integration, production deployment or SOTA comparison was run.

## Archive

Local runnable archive SHA-256:

`435ce4a536d3df0d0bc1ebab11b5da6f999aa43de0203fe534868d9b3f301250`

The archive includes the executable reference, tests, CLI, input fixture, and exact result files.
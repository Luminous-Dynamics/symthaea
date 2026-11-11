# MACE-CoS Guard v0.1 — evidence status

Date: 2026-11-11.

## Public branch contents

This branch contains:
- typed Go request/skill/artifact/step/sink schema;
- deterministic composition validator;
- strict JSON CLI;
- focused public smoke tests (7 named test functions);
- protocol, threat model and industry benchmark matrix.

It is a research-only Go reference. No Rust integration is included.

## Complete local bundle verification

The complete local bundle at `/mnt/data/mace-skill-composition-v0.1.zip` includes a larger 31-test suite, five CLI known-answer fixtures, expected JSON snapshots and `verify.sh`.

Verified locally with Go 1.23.2:
- `go test -count=1 ./...`: PASS
- `go test -race -count=1 ./...`: PASS
- `go vet ./...`: PASS
- `gofmt`: clean
- statement coverage: 79.4%
- CLI known-answer fixtures: 5/5 exact JSON + exit-status matches
- label monotonicity pairs: 16/16
- irrelevant-capability metamorphic mutations: 256/256
- third-party module dependencies: 0

Local ZIP SHA-256:
`899a4c9b036c6652e4d045355416f24a06cdecae1eed06ea44e2e25ed21024a8`

## Not yet established

- No GitHub Actions / CI run is claimed.
- No Rust compile or integration is claimed; Cargo/rustc are unavailable in the local execution environment.
- No CoSA-Bench, SkillsBench, SkillFusion, SkillAlign or SkillShift-Bench evaluation has been run.
- No SOTA, security certification, production enforcement or real-world impact claim is made.

The next gate is to compile/verify the exact public branch head, independently inspect the composition semantics, then build a matched CoSA-Bench adapter and baseline campaign.

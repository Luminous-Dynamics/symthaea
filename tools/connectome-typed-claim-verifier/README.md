# Typed connectome claim verifier

Rust CLI semantic checker for the typed claim bundle V0.1.

## Usage

```sh
cargo test --locked
cargo run --locked -- bundle.json
```

## Structural and semantic checks

- relation type matches evidence profile;
- anatomical endpoints share a world and require separate presence, polarity and partner support;
- correspondence endpoints are cross-release and match the evidence receipt world IDs and object type;
- statistical evidence binds model, sample frame, conditioning set, hypothesis family, multiplicity procedure and sampling design;
- functional evidence has a positive time window;
- causal claims marked supported require adequate positivity and handled selection;
- query paths are recomputed from claim IDs and checked against frozen relation-type grammar;
- mixed evidence cannot be promoted into an unqualified single-layer or causal conclusion;
- counts and exact bundle digest are recomputed.

## Trust boundary

The report is `STRUCTURE_RECOMPUTED` with `trusted_promotion=false`. It does not authenticate external evidence bytes, model correctness, signer authority, biological truth or causal identification. Result-output digests are producer claims unless actual result bytes are separately supplied and verified.

## Tests

Twenty Rust tests are authored for known-answer, relation/evidence mismatch, same-world anatomical and correspondence errors, ambiguous polarity, statistical range, functional window, causal positivity, invalid causal promotion, query grammar, duplicate IDs and counter corruption.

Tests have not yet been compiled or executed in this authoring environment because Cargo and Rust binaries are unavailable. Exact-head CI is the execution gate.

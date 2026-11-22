# RMI Expressive V2 — reference evaluator

This isolated Rust crate is the first executable synthetic panel under issue #10210.

## Run

    cargo run --manifest-path tools/mel-rmi-expressive-v2/Cargo.toml -- --seal tools/mel-rmi-expressive-v2/fixtures/velocity-rubato-positive-unsealed-v2.json /tmp/rmi-expressive-sealed.json
    cargo run --manifest-path tools/mel-rmi-expressive-v2/Cargo.toml -- /tmp/rmi-expressive-sealed.json
    cargo test --manifest-path tools/mel-rmi-expressive-v2/Cargo.toml

## V2 dimensions

The schema carries five independent dimensions: velocity, articulation, rubato offset, register and phrase gesture. The initial deterministic positive fixture and tests cover velocity and rubato as ordered features, and phrase gesture/articulation as categorical features.

Every trial contains independent partner-to-agent and agent-to-partner paired arms. Each arm freezes control/treatment input and control/treatment response frames, a target feature, target event scope and expected response relation.

## Qualification behavior

- the input pair must change only the declared expressive feature and only on scoped event identities;
- the response pair must change only the declared expressive feature and only on scoped event identities;
- target sensitivity is measured separately for partner-to-agent and agent-to-partner;
- follow/contrast relations apply to ordered numeric dimensions only;
- articulation and phrase gesture require explicit any-change semantics;
- policy identity, version, deterministic seed, source score digest, trial payload and all four frames are sealed by a canonical SHA-256 digest.

This is a synthetic reference evaluator, not an executable live model or a human collaboration result.

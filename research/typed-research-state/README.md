# Typed Public ResearchState Contract v0.1

Owner: [#13663](https://github.com/Luminous-Dynamics/symthaea/issues/13663); production adapter owner: [#11171](https://github.com/Luminous-Dynamics/symthaea/issues/11171).

## Files

- `schemas/research-state-v0.1.schema.json`: strict candidate-visible ResearchState JSON Schema.
- `fixtures/research-state-examples-v0.1.json`: four synthetic conformance examples.
- `src/lib.rs`: structural, cross-reference, prefix-binding and non-promotion validator.
- `src/bin/validate.rs`: validates all checked-in examples.
- `Cargo.lock`: pinned standalone dependencies.
- `.github/workflows/typed-research-state.yml`: exact Rust 1.96 format/test/example/Clippy workflow.

## Local execution

```bash
cargo fmt --manifest-path research/typed-research-state/Cargo.toml -- --check
cargo test --manifest-path research/typed-research-state/Cargo.toml --locked --all-targets
cargo run --manifest-path research/typed-research-state/Cargo.toml --locked
cargo clippy --manifest-path research/typed-research-state/Cargo.toml --locked --all-targets -- -D warnings
```

## Contract invariants

- Retrieved evidence cannot implicitly acquire a claim-source relation.
- Claim-source classifier provenance is distinct from truth adjudication.
- A classifier relation is bound to the exact claim text and scope, not merely to a reusable claim ID.
- Every classifier output binds both the exact evidence-prefix digest and a scoped-claim digest over claim ID, statement, scope and reference time; the validator recomputes both before accepting the relation.
- The prefix digest is SHA-256 over compact JSON for the ordered object `{research_episode_id, sequence_number, evidence}`, using the exact evidence records in their existing order. The validator rejects any prefix digest that does not match this manifest.
- Every relation binds one claim ID and one evidence ID.
- Exact quote byte spans are represented by offsets plus quote digest; artifact custody remains upstream.
- UNKNOWN source lineage is not inferred to be independent. Unknown lineage can still leave a support/refutation conflict visible.
- Truth adjudication requires its own authority ID, evidence reference and time; classifier identity cannot serve as the adjudicator.
- Decision usability is scoped to the active decision and policy.
- Conflicts cannot be marked SUFFICIENT.
- HIGH/CRITICAL decisions cannot be marked SUFFICIENT without independent SUPPORTED adjudication.
- The returned state digest is computed over the exact serialized candidate-state bytes and should be copied to the existing #11015 prospective trace.
- The validator rejects global claim-confidence fields; classifier-specific confidence is allowed only inside claim-source relation provenance.
- No free-floating global claim-confidence field is accepted.

## Important limitations

The examples are synthetic. The schema/validator also accepts PRODUCTION_PUBLIC_STATE, but the checked-in examples are explicitly synthetic. The validator checks shape and policy-state transitions, not natural-language entailment, artifact custody signatures or truth. It does not implement the production sequential research policy. The source tree audit in #11171 still identifies the production callable as unresolved.

Do not use this crate as a second evidence store, a source trust service, a posterior engine, or a scheduler. ResearchState is a projection over existing evidence-prefix, custody and adjudication references.

# SAM counterfactual support / identification v0.1

This is a small, dependency-free Rust fixture evaluator for Symthaea issue #10372.

It demonstrates three distinct reasons not to accept a counterfactual point estimate:

1. the requested intervention is unsupported in the target context;
2. multiple cross-world couplings are compatible with the same observed marginals;
3. an asserted interval is narrower than the admitted identified set.

The tool is intentionally not a causal estimator, simulator, planner, or general identifiability engine. Its dispositions are only a reference policy over the supplied analytic fixtures.

## Run

From the repository root:

    cargo fmt --manifest-path tools/sam-cf-support/Cargo.toml -- --check
    cargo clippy --locked --manifest-path tools/sam-cf-support/Cargo.toml --all-targets -- -D warnings
    cargo test --locked --manifest-path tools/sam-cf-support/Cargo.toml

## Fixtures

- fixtures/identification-cases-v0.1.tsv contains eleven cases covering point/set identification, action-support holes, sensitivity-driven ranking reversal, missing episode abduction, incomplete evidence, out-of-scope mechanisms and false precision.
- fixtures/cross-world-couplings-v0.1.tsv contains two joint distributions with identical factual and alternative marginals but different episode-conditional counterfactual outcomes.

The coupling fixture is deliberately minimal: under the POSITIVE coupling, factual outcome 0 pairs with alternative outcome 0; under the NEGATIVE coupling, the same factual outcome pairs with alternative outcome 3. Both couplings have exactly the same marginals. Without a justified coupling assumption or additional episode-specific structural evidence, selecting either point is unjustified.

## Status semantics

- PointIdentified is a singleton result under the declared fixture assumptions.
- SetIdentified preserves a non-singleton admissible outcome set.
- AssumptionSensitive indicates an action ranking reversal over the admitted cross-world sensitivity range.
- UnsupportedIntervention means the target context/action has no demonstrated support and no qualified restriction supplies a bound.
- NotIdentified is used for an R3 query lacking factual continuation or abduction evidence.
- Incomplete and OutOfScope remain separate fail-closed results.
- FalsePrecisionRejected means the reported interval excludes part of the admitted identified set.

A decision may remain invariant even when the outcome is only set-identified. The fixture evaluator carries that as a separate decision_invariant field; it does not collapse outcome identification into action-ranking certainty.

## Claim boundary

Passing these fixtures only demonstrates that the reference classifier handles these known-answer cases. It does not establish real-world causal identification, physical validity, collision-risk calibration, controller safety, or public-road autonomy capability.

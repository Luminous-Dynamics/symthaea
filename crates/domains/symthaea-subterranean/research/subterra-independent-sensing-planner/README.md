# subterra-independent-sensing-planner

A deterministic, bounded exact planner for selecting a set of acquisition paths that meet a required independent-source count under a declared complete Mycelix fault model and hard Symtropy resource/risk envelopes.

## What changes

The planner does not maximize sensor count or sum per-sensor EVSI. It enumerates feasible subsets whose transitive claim-critical dependency closures are pairwise disjoint and disjoint from the independent verifier. It returns a Pareto frontier over energy, time, data, exposure, interruption-risk and robot-loss upper bounds. Units remain separate; no guessed scalar score is used. Each option binds a deployment-bundle ID/digest and external resource-model/assessment. The planner charges each unique bundle's fixed mobilisation resources once per selected portfolio, and reports variable source resources, shared fixed resources and the total separately.

A joint EVSI/value estimate may be attached by a later integration only when its exact selected source set and non-additive likelihood/factorization are assessed upstream. This planner is an admissibility/feasibility optimizer, not a joint Bayesian value estimator.

## Boundary

- Mycelix supplies the authenticated LEX-NET-030 dependency closure, QUAL-001E fault profile, inventory completeness and exact graph/profile/assessment digests.
- Symtropy supplies route/charging/deployment/resource/risk envelopes.
- Symthaea chooses a feasible acquisition portfolio.
- After observations exist, [#12729/#12730](https://github.com/Luminous-Dynamics/symthaea/pull/12730) must still admit the actual evidence. A planned independent path is not evidence that the resulting observation is valid or independent.

## Search semantics

The planner chooses exactly the policy-required number of sources. Candidate pairs with any shared transitive claim-critical dependency conflict. Any candidate overlapping the independent verifier is ineligible. Incomplete/unknown dependency inventories, stale/expired resource/risk assessments, mismatched scope/digests, overflow, risk-budget breach and resource-budget breach fail closed.

Search is bounded by a caller-supplied node limit. Exhaustion returns the best frontier found so far with `NodeLimitReached`; it never claims global optimality. The exhaustive enumerator is intended for small panels and a reference oracle for larger heuristics.

## Run

```sh
cargo fmt --manifest-path Cargo.toml -- --check
cargo test --manifest-path Cargo.toml --all-targets --offline
cargo run --manifest-path Cargo.toml --example synthetic_independent_sensing --offline
cargo clippy --manifest-path Cargo.toml --all-targets --offline -- -D warnings
```

## Claim ceiling

A feasible frontier qualifies only the declared finite candidates, dependency model and resource/risk assessments. It does not prove the dependency inventory complete in reality, sensor accuracy, physical independence, mine safety, intervention authority or field performance. All current inputs are synthetic.

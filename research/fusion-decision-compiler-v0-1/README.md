# Fusion decision compiler contract v0.1

An isolated, dependency-free Rust research crate for the first FECOMP-0 compiler semantics.

It is an integration contract, not another fusion system code. It does not replace PROCESS, Bluemira, FreeGSNKE, DESC, materials solvers, thermal/structural/neutronics solvers, Symtropy's engineering twin, or independent engineering authorities.

## Initial capabilities

- Candidate and objective identity checks.
- Explicit energy-service boundary admission.
- Required hard-gate evaluation for energy, material, manufacturing, inspection, assembly and replacement.
- Solver capability, regime, observable, applicability, execution and convergence checks.
- Explicit distinction between blocked and indeterminate candidates.
- Evidence-to-claim class and revision/scope binding.
- Pairwise solver dependence classification across declared dataset, code and assumption groups.
- Transitive dependency invalidation after upstream changes.
- Fixed claim ceiling: eligible candidates are only eligible for further engineering review.

## Dispositions

- `Blocked`: at least one explicit hard failure, unsupported solver capability, out-of-domain model or failed execution.
- `Indeterminate`: a required boundary, model result, gate or applicability state is missing or unknown.
- `EligibleForFurtherEngineeringReview`: every required gate and solver contract is satisfied under the declared synthetic/modelled scope. This is not physical qualification or manufacturing release.

A hard failure takes precedence over an unknown. Missing evidence is never treated as zero cost, zero uncertainty or a passing gate.

## Evidence classes

A claim must cite evidence with an exact compatible class and the same subject revision. Synthetic fixtures cannot support measured, independently verified or physically qualified claims. A digest string is an identity declaration; the crate does not fetch or hash arbitrary source payloads.

## Solver dependence

Two runs are only classified as independent under declared dependencies when dataset, code and assumption groups are all present and pairwise distinct. Any shared group is marked common-mode. Missing group identity is marked unknown. This is not a statistical proof of independence.

## Dependency invalidation

The graph helper returns the changed nodes and their full transitive downstream closure. Cycles are tolerated and traversal is deterministic. This is a structural impact calculation, not automatic revocation of an external certificate or authority decision.

## Run locally

From this directory:

```sh
cargo fmt --manifest-path Cargo.toml -- --check
cargo test --manifest-path Cargo.toml
cargo run --manifest-path Cargo.toml --bin verify_fecomp_corpus
cargo clippy --manifest-path Cargo.toml --all-targets -- -D warnings
```

The independent corpus binary checks fixture shape and required case identities without calling the compiler implementation. Rust unit tests exercise the typed contract.

## Claim ceiling

A pass establishes only that this exact software contract behaves as tested on synthetic fixtures. It does not validate a solver, equilibrium, reactor architecture, material, component, manufacturing process, plant economics, physical experiment, safety case or net-energy claim. It cannot authorize experiments, hardware, fabrication, commissioning or operation.

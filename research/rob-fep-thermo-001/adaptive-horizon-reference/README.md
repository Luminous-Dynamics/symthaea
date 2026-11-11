# Adaptive commitment horizon reference

Standalone, dependency-free Rust research reference for the min-of-qualified-bounds contract in ../adaptive_commit_horizon.md.

It consumes nine already-qualified caps and returns:
- Eligible with max_ticks and binding_constraints;
- ZeroHorizon with binding_constraints; or
- Blocked with explicit failures.

Every cap must be in canonical order, share the same context digest, and be prefix-closed. Unknown, stale, mismatched, missing or invalid evidence blocks selection. The API has no actuation-permit type; actuation_authorized() always returns false.

This reference does not generate physical bounds, estimate thermal/actuator/dynamics/viability state, project trajectories, validate a real clock, or replace Symtropy #4117/#4382/#4610/#4588/#2836.

## Run

~~~bash
cargo fmt --manifest-path research/rob-fep-thermo-001/adaptive-horizon-reference/Cargo.toml -- --check
cargo test --locked --manifest-path research/rob-fep-thermo-001/adaptive-horizon-reference/Cargo.toml
cargo clippy --locked --manifest-path research/rob-fep-thermo-001/adaptive-horizon-reference/Cargo.toml --all-targets -- -D warnings
~~~

The authored tests cover minimum/tie selection, Unknown/stale/mismatched evidence, prefix closure, zero horizon, minimum commitment threshold, candidate-model mismatch, executed-prefix mismatch and the absence of actuation authority.

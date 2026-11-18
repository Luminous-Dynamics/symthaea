# Culinary Meal Proposal Harness

This minimal Cargo package compiles the exact tracked source file:

`crates/domains/symthaea-culinary/src/meal_proposal.rs`

The source includes its versioned JSON Schema and frozen fixture manifest with `include_str!`; those paths resolve relative to the production source file.

## Run

```sh
cargo fmt --manifest-path tools/culinary-meal-proposal-harness/Cargo.toml --all -- --check
cargo test --manifest-path tools/culinary-meal-proposal-harness/Cargo.toml
cargo clippy --manifest-path tools/culinary-meal-proposal-harness/Cargo.toml --all-targets -- -D warnings
```

## Claim ceiling

A passing run verifies the pure typed proposal contract and included tests at the exact checkout. It does not establish recipe truth, source rights, household inventory, price availability, allergen suitability, appliance compatibility, meal compilation, food safety or actuation authority.

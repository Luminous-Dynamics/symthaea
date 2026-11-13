# symthaea-climate-mobility

Dependency-free Rust contract primitives for research on climate mobility pathways and destination service capacity.

## Scope

This first slice provides:
- typed mobility pathways that distinguish in-place adaptation, temporary displacement, voluntary migration, community-led planned relocation, constrained immobility and return;
- explicit climate-attribution method requirements;
- origin/destination area identity and role checks;
- arithmetic-only aggregate capacity sufficiency/shortfall evaluation;
- fail-closed indeterminate results for missing inputs, unknown evidence or unknown qualification;
- prohibition of recommendation/decision authority in the scenario record.

It does not ingest the JSON fixture, parse GeoJSON, model hydraulics, forecast movement, calculate protected minimum-service floors, determine rights or issue operational commands. The crate's capacity result is not a safety or suitability assessment.

## Test

Run: cargo test --manifest-path crates/domains/symthaea-climate-mobility/Cargo.toml

The tests cover the initial hand-calculable fixture arithmetic and selected negative controls. They are authored tests, not reported execution results until run on the exact commit.

# symthaea-climate-mobility

Dependency-free Rust contract primitives for research on climate mobility pathways and destination service capacity.

## Scope

This first slice provides:
- typed mobility pathways that distinguish in-place adaptation, temporary displacement, voluntary migration, community-led planned relocation, constrained immobility and return;
- explicit climate-attribution method requirements;
- origin/destination area identity and role checks;
- floor assessments bound to the exact scenario pathway and origin/destination area;
- arithmetic-only aggregate capacity sufficiency/shortfall evaluation;
- a separate non-compensatory service-floor evaluator using versioned floor references, observation bounds, explicit scope coverage, an observation-valid-time window and a freshness budget bound to a named versioned policy;
- full-coverage MeetsFloor / BreachedFloor results, partial-coverage ObservedBreach reporting, and Indeterminate for incomplete evidence or threshold-crossing intervals;
- fail-closed indeterminate results for missing inputs, unknown/synthetic evidence, contract-only qualification, missing observation timing/policy references or stale evidence;
- prohibition of recommendation/decision authority in the scenario record.

It does not ingest the JSON fixture, parse GeoJSON, model hydraulics, forecast movement, decide which authority may set a floor, determine rights or issue operational commands. The service-floor result only checks a supplied, versioned threshold against supplied evidence and coverage. It is not a legal determination, certification, water-safety assessment or destination-suitability assessment.

## Test

Run: cargo test --manifest-path crates/domains/symthaea-climate-mobility/Cargo.toml

The tests cover the initial hand-calculable fixture arithmetic, non-compensatory floor bounds/coverage, scenario-area binding and selected negative controls. They are authored tests, not reported execution results until run on the exact commit.

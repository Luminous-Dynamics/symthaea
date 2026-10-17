# Agricultural Platform Benchmark (AGRI-ROBOTICS-BENCH-V1)

A standalone, standard-library-only Rust screening kernel for comparing farm-work approaches against one declared task and site. It is a first research tranche, not an autonomous farm planner.

## Run

From the repository root:

```sh
cargo test --manifest-path crates/agri-platform-bench/Cargo.toml --all-targets
cargo run --manifest-path crates/agri-platform-bench/Cargo.toml --release
```

The crate defines its own Cargo workspace to avoid pulling the full Symthaea workspace into this small benchmark. It has no third-party dependencies.

## What it does

- Screens task compatibility and explicit service units.
- Checks declared slope, terrain roughness, row clearance, turning radius, navigation assumptions, daily capacity, safety-case presence, and human override.
- Separates site/task evidence from each candidate's capability, navigation, safety, cost, and serviceability evidence; candidates cannot inherit one another's capability claims, quotes, or repair claims.
- Requires an explicit uppercase currency code and blocks cross-currency cost comparisons unless a future, separately evidenced conversion contract is added.
- Distinguishes verified, estimated, missing, stale, and conflicting evidence.
- Fails closed on missing/invalid evidence records and checked-arithmetic overflow.
- Computes transparent annualized-capital plus operating cost, supervision time, maintenance time, and repair lead time when the inputs permit it.
- Returns a component-wise assessment and a Pareto frontier; no weighted single “best robot” score is used.
- Keeps manual/current practice as a first-class candidate. A robot can be rejected while the no-robot option remains feasible.

The frozen synthetic case manifest is at `fixtures/scenario_manifest.tsv`. It currently contains 53 expected-disposition cases: 43 adversarial/evidence-boundary cases around a synthetic scouting fixture and 10 additional farm-task archetypes covering broad-acre tractor treatment, orchard scouting, greenhouse harvest, nursery handling, livestock monitoring, packhouse transport, fixed sensors, aerial survey, smallholder mechanical weeding, and an agrivoltaic access rejection. These are synthetic reference cases—not 10 real field validations—and expansion to 30+ distinct site/task families is still required.

## Important limitations

- The included values are synthetic fixture values. They are not measured conditions for Johannesburg, the Highveld, or any real farm.
- The kernel validates evidence-state and digest *shape*; it does not fetch sources or independently recompute source-artifact SHA-256 values.
- Estimated evidence yields a provisional result, not qualification. Missing/stale/conflicting evidence blocks ranking.
- “Feasible” is a result within this narrow profile contract. It is not machinery safety certification, food-safety evidence, agronomic validation, commercial viability, a procurement recommendation, or authority to operate an actuator.
- No live devices, network APIs, irrigation, spraying, seeding, harvest, payments, or inventory are touched.
- Cost is a transparent screening estimate. It is not an ROI/payback model and does not include every possible lifecycle cost; the missing cost fields must be added before any purchase or adoption decision.
- The fixed scenario/profile types intentionally make units explicit, but do not replace a general physical-unit library or calibrated site data.

## Benchmark philosophy

The comparison should optimize the task-level service, not autonomy as a goal. Always compare manual/current practice, fixed sensing, existing equipment/implements, and candidate robotics where applicable. A candidate that cannot fit hard constraints is rejected; missing evidence is not interpreted as a zero or a favorable default.

Targets for this research tranche:
- 100% agreement with the frozen expected-disposition manifest.
- Zero acceptance of known hard-constraint violations represented in the corpus.
- Deterministic output for identical inputs.
- Candidate-specific metrics remain separate; no composite score hides trade-offs.
- Expansion to 30+ distinct site/task scenario families (including orchards, greenhouses, broad-acre fields, nursery, livestock, and packhouse contexts), plus field calibration, precedes any claim about real-world performance.

## Research leads

- Digital Agriculture Datasets — curated list for agriculture AI/robotics data: https://github.com/ricber/digital-agriculture-datasets
- Agri-OpenCore — open agricultural robotics components and project outcomes: https://agri-opencore.org/outcomes/
- Agrivoltaic systems review and meta-analysis: https://www.annualreviews.org/content/journals/10.1146/annurev-environ-121224-073319
- Scientific frontiers of agrivoltaic cropping systems: https://www.nature.com/articles/s44359-025-00110-9

External dataset availability is not permission to ingest or redistribute its data. Any future data addition must pin provenance, rights, scope, units, geography/time coverage, and the exact referenced artifacts.

## Cross-stack ownership

This crate is a small, independent screening/benchmark surface. It composes with, but does not replace:
- Symthaea's existing Agribot and hydroponic-analysis work;
- Symtropy's agricultural value-of-information and scenario modelling;
- Sol-Atlas's geospatial/site-suitability evidence;
- Mycelix's food-cooperative, shared-asset, and logistics records.

Next steps are to add more task/site families, use an independent verifier over retained corpus artifacts, and compare against exhaustive oracle cases before considering any operational integration.

# LL-009H — scenario-grid Pareto switching regions

Status: **Phase-0 research mapping only. Not an exact crossover model, infrastructure selection, site approval, corridor qualification, or launch authority.**

LL-009G answers a multi-objective question for one common-boundary scenario. LL-009H compares many LL-009G analyses on a declared scenario grid and reports where central or robust Pareto-front membership changes between sampled neighboring points.

## Core rule

LL-009H reports **observed switching brackets**, not exact thresholds.

If ballistic freight is not on a front at 100 km and is on the front at 250 km, V1 may report:

```text
observed switch bracket: [100 km, 250 km]
```

It does not report a fabricated crossover such as `173.4 km`.

A narrower threshold requires either denser sampling or a separately declared and validated interpolation/response model.

## Inputs

A sweep manifest defines:

- `sweep_name`;
- `study_family_id`;
- `interpolation_policy = forbidden_v1`;
- ordered scenario axes;
- a hash-bound candidate catalog;
- sampled points with coordinates, LL-009G analysis paths, and exact file SHA-256 values.

Each axis is either:

- `numeric` — finite, strictly increasing declared values;
- `categorical` — unique non-empty values with an explicit order.

Example axes for later real South-Pole studies include:

```text
corridor_distance_km
throughput_demand_kg_per_year
available_peak_power_kw
cargo_mass_class
local_material_fraction
reliability_target
```

The implementation is generic and does not hard-code those names.

## Candidate identity

Candidate IDs are stable across one sweep family. The manifest contains a canonical catalog:

```text
candidate_id -> architecture
```

The catalog itself is canonicalized and SHA-256 bound. Every LL-009G analysis must contain exactly the same candidate-ID set.

The LL-009G analysis also retains its exact LL-009F input-receipt hash, so each grid point remains traceable to the scenario-specific common-accounting evidence even though `study_id` and `accounting_contract_id` may legitimately vary across the sweep.

A candidate ID must never be repurposed to mean a different architecture inside one sweep lineage.

## Cross-scenario invariants

Every sampled LL-009G artifact must have:

- exact supported schema;
- `status = pass`;
- manifest-declared file SHA-256 matching the actual bytes;
- valid internal `analysis_sha256` recomputed over the analysis without the self-hash field;
- the same objective-policy SHA-256;
- byte-identical objective definitions;
- byte-identical objective-unit mapping;
- the exact candidate-ID set declared by the candidate catalog.

Changing the objective policy means a different sweep analysis lineage.

## Sparse-grid honesty

A switch is considered only between two sampled points that are adjacent in the **declared axis values** and differ on exactly one axis.

For declared values:

```text
100
1000
10000
```

if only the 100 and 10000 analyses exist, they are **not** treated as adjacent. LL-009H reports no switch between them because the 1000 point is missing.

This avoids silently claiming that a switch occurs somewhere across a large unsampled hole.

## Central and robust switch maps

LL-009H preserves both LL-009G fronts.

It reports separate:

- `central_switch_brackets`;
- `robust_switch_brackets`.

For each candidate and adjacent pair it records:

- enter or leave event;
- varying axis;
- axis unit;
- lower/upper sampled value;
- endpoint point IDs;
- all fixed coordinates;
- explicit `observed_adjacent_grid_bracket_no_interpolation` semantics.

A technology can therefore enter the central front before it enters the robust front, or remain robustly non-dominated even after the central estimates suggest another technology dominates it.

That distinction is a useful indicator of where more evidence may have the greatest decision value.

## Output

The canonical result contains:

- manifest hash;
- candidate catalog and hash;
- objective policy hash;
- objective definitions and units;
- axes;
- sampled and full-grid point counts;
- each sampled point's coordinates, LL-009G hashes, study/accounting IDs, and fronts;
- sampled adjacency edges;
- per-candidate front-region point sets;
- central switch brackets;
- robust switch brackets;
- `interpolation_policy = forbidden_v1`;
- canonical analysis hash;
- non-claims.

No `threshold`, `winner`, or weighted score is produced.

## Usage

Dependency-free self-test:

```bash
python3 scripts/analyze_ll009h_sweep.py --self-test
```

Analyze a sweep:

```bash
python3 scripts/analyze_ll009h_sweep.py \
  --manifest docs/research/evidence/<family>/ll009h-sweep.json \
  --analysis-root docs/research/evidence/<family>/ll009g-analyses
```

## Executed self-test

The synthetic one-dimensional throughput sweep uses declared samples at 100, 1,000 and 10,000 kg/year.

It verifies:

- candidate B enters the central front in the observed 100→1,000 bracket;
- candidate A leaves the central front in the observed 1,000→10,000 bracket;
- B enters the robust front while A remains robustly non-dominated because the synthetic uncertainty treatment does not support robust removal;
- duplicate scenario coordinates fail;
- objective-policy drift across points fails;
- removing the middle 1,000 sample produces no adjacency and therefore no inferred 100→10,000 switch;
- canonical output contains no exact `threshold` or `winner` field.

## First real sweep

Once the first non-synthetic South-Pole LL-009E → LL-009F → LL-009G chain exists, the first useful grid should be modest rather than combinatorially huge.

A strong starting study is:

```text
corridor distance × annual throughput demand
```

with the same stable candidate family, likely including rover/heavy-haul, continuous fixed infrastructure/FLOAT-like concepts, ballistic surface freight, direct surface-to-orbit launcher/catcher, and elevator-feeder configurations where their complete evidence envelopes exist.

Then add peak power and local-material fraction only where the two-dimensional results show sensitive front boundaries.

This supports adaptive refinement: sample densely around observed switching brackets instead of spending compute uniformly across uninteresting regions.

## Non-claims

LL-009H maps observed front-membership changes in the sampled research grid. It does not establish a continuous response surface, exact economic crossover, universal best architecture, mature engineering design, or authority for physical operation.

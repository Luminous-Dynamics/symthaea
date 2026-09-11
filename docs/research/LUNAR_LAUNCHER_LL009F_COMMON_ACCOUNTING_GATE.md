# LL-009F — common-accounting admission gate for lunar transport trades

Status: **Phase-0 research comparability only. Not infrastructure selection, site approval, corridor qualification, or launch authority.**

LL-009F prevents the LL-009 Pareto layer from comparing transport architectures that use different accounting boundaries.

## Input rule

The economics layer does not ingest loose subsystem receipts or hand-entered technology numbers.

It receives:

1. one validated LL-009E contract-bound corridor envelope;
2. one canonical common accounting/demand contract;
3. at least two candidate architectures expressed against that exact boundary.

```text
LL-009E contract-bound corridor envelope
                 +
shared demand / accounting contract
                 +
normalized candidate evidence
                 |
                 v
             LL-009F
      admission / comparability gate
                 |
                 v
       downstream Pareto analysis
```

LL-009F deliberately does **not** rank the candidates.

## Shared accounting contract

Every candidate uses the same content-derived `accounting_contract_id`, which covers at least:

- corridor reference;
- cargo class;
- demand-scenario reference;
- study horizon;
- annual throughput demand;
- reliability horizon;
- Earth-import boundary;
- local-material boundary;
- power generation/storage boundary;
- maintenance/spares boundary;
- feeder and last-mile boundary;
- failure/recovery boundary;
- decommission/disposal boundary;
- monetary/currency basis when monetary results are included.

Changing those semantics necessarily changes the accounting-contract hash and therefore invalidates candidate records still carrying the old ID.

## Baseline components cannot be omitted

A caller cannot make a technology look better by deleting an inconvenient category from `required_components`.

The gate requires, at minimum:

- `earth_imported_construction`;
- `local_construction`;
- `power_generation_storage`;
- `transport_hardware`;
- `maintenance_spares`;
- `operations`;
- `feeder_last_mile`;
- `failure_recovery`;
- `stationkeeping_propellant`;
- `decommission_disposal`.

Every candidate must contain the exact full required-component key set used by the study.

Each component is explicitly one of:

- `included` — requires an evidence reference;
- `not_applicable` — requires a physical reason;
- `unresolved`.

`unresolved` is forbidden for `pareto_ready_for_real_site_comparison` promotion.

A numerical zero is never used as shorthand for `unresolved` or `not_applicable`.

## Baseline metrics

The gate also requires, at minimum:

- imported construction/system mass;
- locally sourced mass;
- delivered-energy intensity;
- peak power;
- throughput capacity;
- latency;
- availability;
- reliability/success probability;
- imported maintenance mass per year;
- propellant/reaction-mass demand per year;
- reusable hardware inventory mass;
- expected lost cargo per year.

Each candidate metric carries:

- `low`;
- `central`;
- `high`;
- unit;
- evidence class;
- source reference.

The uncertainty interval must satisfy `low <= central <= high`. A zero central value is valid when it is genuinely supported and explicitly represented.

Synthetic-only metrics are rejected for real-site Pareto promotion.

## LL-009E verification

Before candidate validation, LL-009F verifies the outer envelope itself:

- exact LL-009E index schema;
- exact LL-009E receipt schema;
- receipt status `pass`;
- receipt→index SHA-256 binding;
- full outer-envelope aggregate digest;
- study/frame/epoch IDs.

This means the Pareto study cannot bypass LL-009E by pointing at an LL-009C capsule or a loose set of receipts.

## Usage

Dependency-free self-test:

```bash
python3 scripts/validate_ll009f_trade_input.py --self-test
```

Validate a trade-study manifest:

```bash
python3 scripts/validate_ll009f_trade_input.py \
  --manifest docs/research/evidence/<corridor>/trade-input.json \
  --ll009e-envelope docs/research/evidence/<corridor>/contract-bound-capsule
```

## Executed self-test coverage

The pre-commit self-test exercises:

- a valid two-candidate promoted comparison;
- deterministic normalization/hash output;
- explicit legitimate zero metric values;
- promoted unresolved-component rejection;
- candidate accounting-contract mismatch rejection;
- missing common-component rejection;
- tampered LL-009E envelope rejection.

## What comes next

The downstream Pareto engine should consume only the normalized LL-009F receipt and should preserve vectors rather than collapsing everything into a weighted score.

Expected objectives include combinations of:

```text
min imported mass
min lifecycle energy
min peak power
min maintenance burden
min failure/recovery burden
min latency
max throughput
max availability / reliability
```

Monetary cost can be added where evidence exists, but absence of reliable dollar estimates must not prevent mass/energy/reliability trade studies.

The engine should identify dominated/non-dominated architectures and switching thresholds as demand, distance, cargo class, power availability, and local industrial capability change.

## Non-claims

An LL-009F pass means the candidates are admitted under one explicit comparison boundary. It does not mean the inputs are correct, that any candidate is economically superior, that a site is suitable, or that any transport system is operationally safe or authorized.

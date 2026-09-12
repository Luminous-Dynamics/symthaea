# LL-009I — common South-Pole site evidence and architecture-specific viability

Status: **Phase-0 research comparability gate. Not site selection, construction qualification, launch authority, or operational certification.**

LL-009I prevents a real-site transport comparison from assuming that a corridor endpoint is physically suitable simply because trajectory/economic models can use its coordinates.

## Motivation

South-Pole terrain drives several coupled infrastructure conditions:

- slope/roughness and local terrain access;
- low-angle solar illumination and long terrain shadows;
- duration of darkness and therefore power/storage/survival burden;
- thermal conditions;
- direct-to-Earth and surface/relay communications visibility;
- civil/geotechnical site-preparation requirements.

NASA's current South-Pole environment guidance emphasizes rugged terrain, terrain-controlled lighting/shadow and thermal conditions, and infrastructure placement. NASA surface-construction work separately identifies geotechnical site investigation, grading, compaction, bearing/compression/shear properties, and verification as key lunar civil-engineering needs.

LL-009I therefore keeps **observed site evidence common** while letting rover, rail/FLOAT, cable, launcher, and elevator-feeder candidates declare different physical tolerances.

## Schemas

### Site evidence pack

`ll009i.site-evidence-pack.v1`

The pack binds:

- `study_id`;
- `frame_contract_id`;
- `epoch_contract_id`;
- `site_ref`;
- hash-verified local evidence/source artifacts;
- site metrics with category, status, low/central/high interval, unit, evidence class, and source references.

Available metrics require exact intervals and evidence lineage. Unknown data is represented as `unresolved`; it is never converted to zero.

### Candidate requirement contract

`ll009i.candidate-site-requirements.v1`

Each candidate binds the same study/frame/epoch/site lineage and declares:

- stable candidate ID;
- architecture;
- category disposition;
- threshold checks and evidence-class requirements;
- optional explicit `any_of` alternatives.

The content-derived requirement ID is:

```text
site-req-<sha256(canonical requirement contract)>
```

Changing a slope, illumination, communications, thermal, geotechnical, or exclusion tolerance therefore changes the candidate's requirement lineage.

## Mandatory baseline categories

Every candidate must account for exactly these baseline categories in v1:

- `terrain`;
- `illumination`;
- `thermal`;
- `communications`;
- `geotechnical`;
- `protected_constraints`.

A category may be `required`, or `not_applicable` with a non-empty physical reason. A caller cannot simply omit an inconvenient category.

This is an evidence-completeness rule, not a claim that every architecture uses the same threshold.

## Conservative threshold semantics

For a maximum limit:

```text
PASS       if evidence.high <= limit
FAIL       if evidence.low  >  limit
UNRESOLVED otherwise
```

For a minimum limit:

```text
PASS       if evidence.low  >= limit
FAIL       if evidence.high <  limit
UNRESOLVED otherwise
```

Thus a central value never promotes a candidate when its declared uncertainty interval crosses the threshold.

The receipt reports the conservative margin where meaningful.

## Explicit alternatives

`any_of` allows evidence-backed alternatives without hard-coding one infrastructure architecture.

For example, a communications requirement might allow either:

```text
DTE availability >= candidate threshold
```

or:

```text
relay availability >= candidate threshold
```

The group passes only when one complete declared option passes. An unresolved option does not become a pass merely because it is plausible.

Likewise, future power requirements can compare site darkness/illumination evidence against a candidate's demonstrated storage/external-power survival envelope rather than requiring all technologies to use solar power identically.

## Source-byte verification

Every site-pack source is a relative path under a declared artifact root plus exact SHA-256. Absolute paths and `..` traversal are rejected.

Metric `source_refs` must refer to those verified source artifacts.

The first intended real pack will combine the existing LOLA/PGDA terrain evidence with subsequently generated illumination/horizon, thermal, communications, geotechnical, and protected/resource-constraint evidence receipts.

## Candidate result

Each candidate receives:

- requirement-file SHA-256;
- content-derived requirement ID;
- per-check pass/fail/unresolved result;
- per-category result;
- conservative threshold margins;
- aggregate `site_viability`.

Only `site_viability = pass` should eventually be admitted to promoted LL-009F real-site economic comparison.

## Example execution

```bash
python3 scripts/validate_ll009i_site_viability.py \
  --site-pack docs/research/evidence/<study>/site/site-pack.json \
  --artifact-root docs/research/evidence/<study>/site \
  --requirements \
    configs/lunar_transport/<study>-rover-site-requirements.json \
    configs/lunar_transport/<study>-float-site-requirements.json \
    configs/lunar_transport/<study>-launcher-site-requirements.json \
  --output docs/research/evidence/<study>/site/site-viability-receipt.json
```

Dependency-free self-test:

```bash
python3 scripts/validate_ll009i_site_viability.py --self-test
```

## Non-claims

A passing receipt means only that the candidate's declared requirements are conservatively satisfied by the declared evidence pack.

It does not establish:

- that the site is selected;
- that evidence sources are sufficient for flight/construction qualification;
- that a geotechnical model replaces in-situ investigation;
- that DTE/relay communications are operationally qualified;
- that the power system closes lunar-night/survival requirements;
- that construction or launcher release is authorized;
- that the architecture is economically or technically superior.

Tracks #1598, #1603, #1640, #1693, #1695, #1701, #1716, and master #1542.

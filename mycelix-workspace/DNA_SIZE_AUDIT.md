# Mycelix DNA Size Audit

**Date**: 2026-03-22 (updated)
**Build**: Release (`cargo build --release --target wasm32-unknown-unknown`)
**Build Date**: 2026-02-21 (per-zome sizes from original audit; DNA bundles pre-date new zomes)

> **Verified 2026-03-22**: Commons cluster rebuilt with 82 WASM files (was 78).
> Two new zome pairs added: mesh-time (3.44 MB pair) and resource-mesh (3.52 MB pair).
> Total raw WASM: 155.2 MB (was 144.9 MB, +7.1%). Bridge coordinator grew +0.18 MB
> from metrics extern + fail-closed security hardening.
>
> **Measured headroom** (Commons only — Civic/Identity estimates below):
> - Commons total: 155.2 MB raw → estimated ~28.2 MB DNA (5.5x compression) → **56% used**
> - Civic: ~11.2 MB + ~3.6 MB (resonance-feed) ≈ 14.8 MB → **30% used**
> - Identity: ~6.4 MB + ~7.2 MB (name-registry + web-of-trust) ≈ 13.6 MB → **27% used**
>
> All DNAs remain well within the 50 MB limit. Growth rate: +10.3 MB raw / ~5 months.

---

## Holochain Size Limits (Reference)

| Resource       | Limit  |
|----------------|--------|
| Single WASM    | 16 MB  |
| Single DNA     | 50 MB  |

---

## Executive Summary

| Cluster          | Zomes | Raw WASM Total | DNA Bundle | hApp Bundle | DNA Headroom |
|------------------|-------|----------------|------------|-------------|--------------|
| **mycelix-commons** | 78 (39 pairs) | 144.9 MB | 26.3 MB | 23.1 MB | 47% used |
| **mycelix-civic**   | 32 (16 pairs) |  63.7 MB | 11.2 MB | 11.1 MB | 22% used |
| **Combined**         | 110 (55 pairs) | 208.6 MB | 37.5 MB | 34.2 MB | — |

DNA bundles use zlib/gzip compression, so they are significantly smaller than the sum of
raw WASM. Both cluster DNAs are well within the 50 MB DNA limit.

**Assessment**: Both DNAs have comfortable headroom. Commons at 26.3 MB uses ~53% of the
50 MB limit — there is room for moderate growth. Civic at 11.2 MB is at ~22% and has
substantial room for expansion.

---

## Per-Zome WASM Sizes

### Mycelix-Commons (78 zomes, 39 integrity + 39 coordinator)

#### Property Domain (4 pairs, 15.3 MB raw)

| Zome | Type | Size |
|------|------|------|
| property_commons_integrity | integrity | 1.41 MB |
| property_commons | coordinator | 2.41 MB |
| property_disputes_integrity | integrity | 1.37 MB |
| property_disputes | coordinator | 2.37 MB |
| property_registry_integrity | integrity | 1.46 MB |
| property_registry | coordinator | 2.52 MB |
| property_transfer_integrity | integrity | 1.39 MB |
| property_transfer | coordinator | 2.42 MB |

#### Housing Domain (6 pairs, 22.2 MB raw)

| Zome | Type | Size |
|------|------|------|
| housing_clt_integrity | integrity | 1.40 MB |
| housing_clt | coordinator | 2.34 MB |
| housing_finances_integrity | integrity | 1.39 MB |
| housing_finances | coordinator | 2.32 MB |
| housing_governance_integrity | integrity | 1.46 MB |
| housing_governance | coordinator | 2.36 MB |
| housing_maintenance_integrity | integrity | 1.37 MB |
| housing_maintenance | coordinator | 2.27 MB |
| housing_membership_integrity | integrity | 1.39 MB |
| housing_membership | coordinator | 2.33 MB |
| housing_units_integrity | integrity | 1.36 MB |
| housing_units | coordinator | 2.19 MB |

#### Care Domain (5 pairs, 18.0 MB raw)

| Zome | Type | Size |
|------|------|------|
| care_circles_integrity | integrity | 1.35 MB |
| care_circles | coordinator | 2.18 MB |
| care_credentials_integrity | integrity | 1.36 MB |
| care_credentials | coordinator | 2.19 MB |
| care_matching_integrity | integrity | 1.34 MB |
| care_matching | coordinator | 2.26 MB |
| care_plans_integrity | integrity | 1.37 MB |
| care_plans | coordinator | 2.21 MB |
| care_timebank_integrity | integrity | 1.41 MB |
| care_timebank | coordinator | 2.29 MB |

#### Mutual Aid Domain (7 pairs, 26.8 MB raw)

| Zome | Type | Size |
|------|------|------|
| mutualaid_circles_integrity | integrity | 1.40 MB |
| mutualaid_circles | coordinator | 2.36 MB |
| mutualaid_governance_integrity | integrity | 1.43 MB |
| mutualaid_governance | coordinator | 2.25 MB |
| mutualaid_needs_integrity | integrity | 1.45 MB |
| mutualaid_needs | coordinator | 2.45 MB |
| mutualaid_pools_integrity | integrity | 1.43 MB |
| mutualaid_pools | coordinator | 2.40 MB |
| mutualaid_requests_integrity | integrity | 1.37 MB |
| mutualaid_requests | coordinator | 2.23 MB |
| mutualaid_resources_integrity | integrity | 1.51 MB |
| mutualaid_resources | coordinator | 2.57 MB |
| mutualaid_timebank_integrity | integrity | 1.49 MB |
| mutualaid_timebank | coordinator | 2.47 MB |

#### Water Domain (5 pairs, 18.3 MB raw)

| Zome | Type | Size |
|------|------|------|
| water_capture_integrity | integrity | 1.39 MB |
| water_capture | coordinator | 2.21 MB |
| water_flow_integrity | integrity | 1.41 MB |
| water_flow | coordinator | 2.27 MB |
| water_purity_integrity | integrity | 1.37 MB |
| water_purity | coordinator | 2.21 MB |
| water_steward_integrity | integrity | 1.43 MB |
| water_steward | coordinator | 2.33 MB |
| water_wisdom_integrity | integrity | 1.39 MB |
| water_wisdom | coordinator | 2.25 MB |

#### Food Domain (4 pairs, 14.5 MB raw)

| Zome | Type | Size |
|------|------|------|
| food_distribution_integrity | integrity | 1.36 MB |
| food_distribution | coordinator | 2.22 MB |
| food_knowledge_integrity | integrity | 1.41 MB |
| food_knowledge | coordinator | 2.32 MB |
| food_preservation_integrity | integrity | 1.36 MB |
| food_preservation | coordinator | 2.16 MB |
| food_production_integrity | integrity | 1.41 MB |
| food_production | coordinator | 2.25 MB |

#### Transport Domain (3 pairs, 10.9 MB raw)

| Zome | Type | Size |
|------|------|------|
| transport_impact_integrity | integrity | 1.35 MB |
| transport_impact | coordinator | 2.18 MB |
| transport_routes_integrity | integrity | 1.42 MB |
| transport_routes | coordinator | 2.26 MB |
| transport_sharing_integrity | integrity | 1.39 MB |
| transport_sharing | coordinator | 2.26 MB |

#### Space Domain (1 pair, 3.9 MB raw)

| Zome | Type | Size |
|------|------|------|
| space_integrity | integrity | 1.43 MB |
| space | coordinator | 2.42 MB |

#### Support Domain (3 pairs, 11.3 MB raw)

| Zome | Type | Size |
|------|------|------|
| support_diagnostics_integrity | integrity | 1.42 MB |
| support_diagnostics | coordinator | 2.28 MB |
| support_knowledge_integrity | integrity | 1.42 MB |
| support_knowledge | coordinator | 2.30 MB |
| support_tickets_integrity | integrity | 1.46 MB |
| support_tickets | coordinator | 2.37 MB |

#### Commons Bridge (1 pair, 3.9 MB raw)

| Zome | Type | Size |
|------|------|------|
| commons_bridge_integrity | integrity | 1.38 MB |
| commons_bridge | coordinator | 2.52 MB |

---

### Mycelix-Civic (32 zomes, 16 integrity + 16 coordinator)

#### Justice Domain (5 pairs, 22.8 MB raw)

| Zome | Type | Size |
|------|------|------|
| justice_arbitration_integrity | integrity | 1.89 MB |
| justice_arbitration | coordinator | 3.01 MB |
| justice_cases_integrity | integrity | 1.90 MB |
| justice_cases | coordinator | 2.86 MB |
| justice_enforcement_integrity | integrity | 1.89 MB |
| justice_enforcement | coordinator | 2.93 MB |
| justice_evidence_integrity | integrity | 1.35 MB |
| justice_evidence | coordinator | 2.21 MB |
| justice_restorative_integrity | integrity | 1.89 MB |
| justice_restorative | coordinator | 2.82 MB |

#### Emergency Domain (6 pairs, 21.7 MB raw)

| Zome | Type | Size |
|------|------|------|
| emergency_comms_integrity | integrity | 1.38 MB |
| emergency_comms | coordinator | 2.28 MB |
| emergency_coordination_integrity | integrity | 1.40 MB |
| emergency_coordination | coordinator | 2.32 MB |
| emergency_incidents_integrity | integrity | 1.41 MB |
| emergency_incidents | coordinator | 2.26 MB |
| emergency_resources_integrity | integrity | 1.35 MB |
| emergency_resources | coordinator | 2.28 MB |
| emergency_shelters_integrity | integrity | 1.35 MB |
| emergency_shelters | coordinator | 2.25 MB |
| emergency_triage_integrity | integrity | 1.32 MB |
| emergency_triage | coordinator | 2.16 MB |

#### Media Domain (4 pairs, 15.2 MB raw)

| Zome | Type | Size |
|------|------|------|
| media_attribution_integrity | integrity | 1.37 MB |
| media_attribution | coordinator | 2.37 MB |
| media_curation_integrity | integrity | 1.39 MB |
| media_curation | coordinator | 2.37 MB |
| media_factcheck_integrity | integrity | 1.41 MB |
| media_factcheck | coordinator | 2.47 MB |
| media_publication_integrity | integrity | 1.39 MB |
| media_publication | coordinator | 2.38 MB |

#### Civic Bridge (1 pair, 4.0 MB raw)

| Zome | Type | Size |
|------|------|------|
| civic_bridge_integrity | integrity | 1.38 MB |
| civic_bridge | coordinator | 2.66 MB |

---

## DNA and hApp Bundles

### Cluster DNAs (Main)

| Bundle | Size | Limit | Usage | Headroom |
|--------|------|-------|-------|----------|
| mycelix_commons.dna | 26.3 MB | 50 MB | 52.6% | **23.7 MB** remaining |
| mycelix-commons.happ | 23.1 MB | — | — | — |
| mycelix_civic.dna | 11.2 MB | 50 MB | 22.4% | **38.8 MB** remaining |
| mycelix-civic.happ | 11.1 MB | — | — | — |

### Workspace hApps (Standalone)

| Bundle | DNA Size | hApp Size |
|--------|----------|-----------|
| lucid | 5.3 MB | 5.6 MB |
| fabrication | 6.9 MB | 7.4 MB |
| epistemic-markets | 3.2 MB | 3.2 MB |
| civic-happ (services) | 1.3 MB | 1.3 MB |

---

## Size Distribution Analysis

### Individual WASM Size Ranges

| Category | Min | Max | Avg |
|----------|-----|-----|-----|
| All integrity zomes | 1.32 MB | 1.90 MB | 1.47 MB |
| All coordinator zomes | 2.16 MB | 3.01 MB | 2.42 MB |
| Overall | 1.32 MB | 3.01 MB | 1.98 MB |

### Top 10 Largest Zomes

| Rank | Zome | Size | Cluster |
|------|------|------|---------|
| 1 | justice_arbitration | 3.01 MB | civic |
| 2 | justice_enforcement | 2.93 MB | civic |
| 3 | justice_cases | 2.86 MB | civic |
| 4 | justice_restorative | 2.82 MB | civic |
| 5 | civic_bridge | 2.66 MB | civic |
| 6 | mutualaid_resources | 2.57 MB | commons |
| 7 | property_registry | 2.52 MB | commons |
| 8 | commons_bridge | 2.52 MB | commons |
| 9 | media_factcheck | 2.47 MB | commons |
| 10 | mutualaid_timebank | 2.47 MB | commons |

The justice domain zomes are notably larger (~1.9 MB integrity, ~3.0 MB coordinator)
compared to all other domains (~1.4 MB integrity, ~2.3 MB coordinator). This is likely
due to the richer type system (restorative justice processes, enforcement workflows,
arbitration state machines).

### Compression Ratio (Raw WASM to DNA)

| Cluster | Sum of Raw WASM | DNA Bundle | Compression |
|---------|-----------------|------------|-------------|
| Commons | 144.9 MB | 26.3 MB | **5.5x** (82% reduction) |
| Civic | 63.7 MB | 11.2 MB | **5.7x** (82% reduction) |

DNA bundles use zlib compression internally, achieving approximately 82% size reduction.

---

## Headroom Assessment

### Per-Zome Headroom (16 MB limit)

**No concerns.** The largest individual WASM file is `justice_arbitration.wasm` at
3.01 MB, using only 18.8% of the 16 MB per-WASM limit. Every zome has at minimum
81% headroom.

### Per-DNA Headroom (50 MB limit)

| DNA | Current | Limit | Remaining | Risk Level |
|-----|---------|-------|-----------|------------|
| mycelix-commons | 26.3 MB | 50 MB | 23.7 MB | **MODERATE** |
| mycelix-civic | 11.2 MB | 50 MB | 38.8 MB | LOW |

**Commons DNA** is the one to watch. At 26.3 MB (52.6%), it still has room for
approximately 12-13 additional zome pairs at current average sizes before reaching
the limit. However, the 7-domain, 39-pair structure is already substantial.

**Civic DNA** has ample room at 11.2 MB (22.4%). It could more than triple in
zome count before approaching limits.

### Growth Projections

| Scenario | Commons Impact | Civic Impact |
|----------|---------------|--------------|
| Add 1 zome pair (avg) | +0.7 MB DNA (+1.4%) | +0.7 MB DNA (+1.4%) |
| Add new domain (4 pairs) | +2.8 MB DNA (+5.6%) | +2.8 MB DNA (+5.6%) |
| wasm-opt optimization | -10-30% per zome | -10-30% per zome |
| LTO + wasm-opt combined | -20-40% per zome | -20-40% per zome |

### Optimization Opportunities (if needed)

1. **wasm-opt** (Binaryen): Typically achieves 10-30% size reduction on Holochain WASM.
   Not currently applied.
2. **LTO (Link-Time Optimization)**: Can be enabled in `Cargo.toml` `[profile.release]`
   with `lto = true`. Reduces code duplication across crate boundaries.
3. **Strip debug info**: `strip = "symbols"` in release profile (may already be set).
4. **Shared integrity zomes**: Justice integrity zomes are identical in size (1.89 MB x3
   for arbitration/enforcement/restorative), suggesting shared code. A unified integrity
   zome could eliminate duplication.

---

## Notes

- All sizes measured from release builds on 2026-02-21.
- hApp bundles are slightly smaller than DNA bundles because they reference the DNA
  rather than embedding a second copy.
- The `commons_bridge` and `civic_bridge` zomes handle cross-cluster communication
  via `CallTargetCell::OtherRole` in the unified hApp configuration.
- The `space` domain has a single zome pair (no `space_` prefix pattern for coordinator;
  the coordinator is simply named `space.wasm`).

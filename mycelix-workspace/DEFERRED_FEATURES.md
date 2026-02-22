# Mycelix Deferred Features Assessment

**Date**: 2026-02-22
**Context**: Post Phase 1-4 enrichment (1,124+ Rust tests, 8,022+ TS SDK tests)
**DNA data**: See [DNA_SIZE_AUDIT.md](./DNA_SIZE_AUDIT.md)

## Status Summary

| Feature | Completion | Blocking Factor | Path Forward |
|---------|-----------|-----------------|--------------|
| Carbon wallet | 80% | External wallet integration | Identity hApp integration via `CallTargetCell::OtherRole` |
| ML triage | 50% | Symthaea coupling | Subscriber crate for `CognitiveUpdate` signals |
| Composting | 0% | No DNA headroom for new zome | Fold into `food_preservation` as batch method |
| Food forest | 0% | No DNA headroom for new zome | Fold into `food_production` as `plot_type` variant |
| Seed exchange | 90% | Missing quality/reputation | Wire into `support_knowledge` reputation system |
| Community gardens | 80% | Read-only membership | `GardenMembership` implemented — add role-based write permissions |
| Accessibility features | 100% | — | Done: `VehicleFeatures`, `get_accessible_vehicles` |
| Ride reviews | 100% | — | Done: `review_ride`, `get_ride_reviews`, `get_driver_rating` |
| Helper system | 100% | — | Done: `HelperProfile`, availability management, expertise matching |
| Satisfaction surveys | 100% | — | Done: `SatisfactionSurvey`, per-ticket ratings |
| Preemptive alerting | 100% | — | Done: `PreemptiveAlert` with free energy scoring |
| Ticket escalation | 100% | — | Done: 4-tier escalation (Tier1→Tier2→Management→Emergency) |
| Undo/rollback | 100% | — | Done: `UndoAction` with rollback tracking |
| Nutrient tracking | 100% | — | Done: `NutrientProfile` with vitamins/minerals |
| Allergen search | 100% | — | Done: `search_allergen_safe` across listings |

## Detailed Assessment

### Carbon Wallet (80%)

**What exists**: `CarbonCredit` entries, `redeem_credits`, `get_agent_carbon_balance` — full earn/redeem/balance lifecycle in `transport_impact` zome.

**What's missing**: True wallet semantics. Currently credits are DHT entries linked to agents. No double-spend prevention beyond Holochain's source chain ordering. No cross-hApp portability.

**Path forward**: Integrate with identity hApp's credential system. Carbon balance becomes a verifiable credential that can be presented cross-hApp. Requires:
1. New entry type in identity: `CarbonWalletCredential`
2. Cross-cluster bridge call from transport_impact → identity (via `CallTargetCell::OtherRole`)
3. TS SDK method on `IdentityClient`: `getCarbonWalletCredential()`

**Effort**: ~2 days. No new zome needed.

### ML Triage (50%)

**What exists**: `CognitiveUpdate` entries with 2048-byte BinaryHV encodings and phi values. `publish_cognitive_update` / `absorb_cognitive_update` / `get_cognitive_updates_by_category` in `support_diagnostics` zome. `PreemptiveAlert` with free energy scores.

**What's missing**: Actual ML inference. Currently CognitiveUpdates are published by agents manually. No automated pattern recognition or prediction.

**Path forward**: Create `symthaea-mycelix-support` subscriber crate that:
1. Subscribes to CognitiveUpdate signals via Holochain's signal system
2. Feeds BinaryHV encodings into Symthaea's HDC pattern matching
3. Uses CfC temporal dynamics to predict failure patterns
4. Auto-generates PreemptiveAlerts when free energy exceeds threshold

**Effort**: ~1 week. Requires Symthaea bridge infrastructure (partially exists in `symthaea-mycelix-bridge`).

### Composting (0%)

**What exists**: Nothing specific. `food_preservation` has batch tracking with methods like fermentation, dehydration, etc.

**Path forward**: Add composting as a preservation method variant:
- New `PreservationMethod` entry: `{ name: "Composting", shelf_life_days: 0, ... }`
- New field on `PreservationBatch`: `compost_output_kg: Option<f32>` (soil amendment yield)
- New query: `get_compost_batches()` (filter by method name)

**DNA impact**: Zero — extends existing `food_preservation` zome. No new WASM.

**Effort**: ~4 hours.

### Food Forest (0%)

**What exists**: `Plot` entries with `status: PlotStatus` (Active/Fallow/Preparing).

**Path forward**: Extend `food_production` with food forest support:
- Add `PlotType` enum: `Garden | FoodForest | Orchard | Greenhouse`
- Add `perennial_species: Vec<String>` field to Plot
- Add `canopy_layers: Option<Vec<String>>` for food forest guild tracking
- Query: `get_food_forest_plots()` (filter by plot type)

**DNA impact**: Zero — extends existing `food_production` zome.

**Effort**: ~4 hours.

### Seed Exchange Quality (90% → 100%)

**What exists**: Full seed exchange lifecycle: `offer_seeds`, `request_seeds`, `get_available_seeds`, `get_open_seed_requests`, `match_seed_request`. All in `food_knowledge` zome.

**What's missing**: Quality reputation for seed offerers. Currently any agent can offer seeds with no track record.

**Path forward**: Wire into existing `support_knowledge` reputation system:
- When a seed match is confirmed, create `ReputationRecord` with event `HelpProvided`
- When a seed request fails (bad germination), create negative reputation
- Query reputation before accepting seed offers: `get_agent_reputation(offerer)`

**Effort**: ~2 hours. Cross-zome call from `food_knowledge` → `support_knowledge` (same DNA, `CallTargetCell::Local`).

### Community Gardens (80% → 100%)

**What exists**: `GardenMembership` entries, `add_garden_member`, `get_plot_members`, `remove_garden_member` in `food_production` zome. Role field on membership (e.g., "Gardener", "Steward").

**What's missing**: Role-based write permissions. Currently any member can modify any plot. Stewards should have admin rights, Gardeners should only modify their assigned crops.

**Path forward**: Add validation in `food_production` integrity zome:
- `validate_create_link` for crop entries: check membership role
- Steward role → full CRUD on plot + members
- Gardener role → create/update crops only on assigned plot

**Effort**: ~3 hours. Integrity-only change, no coordinator changes.

## Priority Ranking

| Priority | Feature | Effort | Impact |
|----------|---------|--------|--------|
| P1 | Seed exchange quality (wire reputation) | 2h | Closes the last gap in seed exchange |
| P1 | Community gardens permissions | 3h | Security-critical (prevent unauthorized edits) |
| P2 | Composting | 4h | User-requested, easy win |
| P2 | Food forest | 4h | User-requested, easy win |
| P3 | Carbon wallet (identity integration) | 2d | Cross-hApp, requires identity changes |
| P3 | ML triage (Symthaea integration) | 1w | Complex, requires Symthaea bridge |

## DNA Headroom Impact

None of the deferred features require new zomes. All can be implemented as extensions to existing zomes:

| Feature | Target Zome | Impact |
|---------|------------|--------|
| Composting | food_preservation | New method variant |
| Food forest | food_production | New plot type |
| Seed quality | food_knowledge → support_knowledge | Cross-zome call |
| Garden permissions | food_production (integrity) | Validation rules |
| Carbon wallet | transport_impact → identity | Cross-hApp call |
| ML triage | support_diagnostics (signals) | External subscriber |

Commons DNA at 27 MB with 37 MB headroom — no pressure to add new zomes.

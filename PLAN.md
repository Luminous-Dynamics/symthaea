# Plan: Fill Out Thin Mycelix Domains

## Critical Constraint: DNA Size

Both commons sub-cluster DNAs are estimated ~13.75 MB each (of 16 MB limit) — only ~2.25 MB headroom. **We CANNOT add new zomes** (each new WASM binary adds ~400-700 KB). Instead, we enrich existing zomes by adding entry types, enums, and coordinator functions within them.

## Scope

Four domains ranked by thinness:

| Domain | Current | Target | Strategy |
|--------|---------|--------|----------|
| **Space** | 60% (994 LOC) | 80% | Add invitation approval, resource booking, scheduling to existing zome |
| **Transport** | 70% (4,208 LOC) | 85% | Add reviews, accessibility, fleet maintenance to existing 3 zomes |
| **Support** | 75% (4,281 LOC) | 85% | Add escalation, SLA, satisfaction to existing 3 zomes |
| **Food** | 80% (7,019 LOC) | 90% | Add allergens, seed exchange, nutrition to existing 4 zomes |

## Phase 1: Space Zome (commons-care DNA)

**File**: `mycelix-commons/zomes/space/integrity/src/lib.rs`

Add entry types:
- `ResourceBooking` — book shared resources within a space (room, tool, equipment)
  - Fields: `id, space_id, resource_name, booked_by, start_time, end_time, status(Pending/Confirmed/Cancelled), notes`
- `SpaceSchedule` — recurring events/meetings for a space
  - Fields: `id, space_id, title, description, recurrence(Once/Daily/Weekly/Monthly), next_occurrence, creator`

Add link types: `SpaceToBookings`, `SpaceToSchedules`, `AgentToBookings`

Add validation: booking time range (start < end), resource name non-empty, schedule recurrence valid

**File**: `mycelix-commons/zomes/space/coordinator/src/lib.rs`

Add coordinator functions:
- `approve_invitation(InvitationApprovalInput) -> Record` — missing workflow (invitation exists but can't be approved)
- `reject_invitation(ActionHash) -> Record`
- `book_resource(BookResourceInput) -> Record`
- `cancel_booking(ActionHash) -> Record`
- `get_space_bookings(space_id: String) -> Vec<Record>`
- `create_schedule(CreateScheduleInput) -> Record`
- `get_space_schedules(space_id: String) -> Vec<Record>`
- `check_capability(CheckCapInput) -> bool` — validate agent has required capability before operation

Add tests: ~40 new tests (serde roundtrips for new types, validation edge cases, invitation approval/rejection flow)

## Phase 2: Transport Zomes (commons-care DNA)

### 2a. transport-routes — Add fleet maintenance

**Integrity** (`transport-routes/integrity/src/lib.rs`):
- New entry: `MaintenanceRecord` — `id, vehicle_hash, maintenance_type(Scheduled/Repair/Inspection), description, cost, completed_at, next_due, mechanic_notes`
- New entry: `VehicleFeatures` — `vehicle_hash, wheelchair_accessible, child_seat, pet_friendly, air_conditioning, bike_rack, luggage_capacity_liters`
- New enum variant: Add `Wheelchair` to `VehicleType`
- New link types: `VehicleToMaintenance`, `AllAccessibleVehicles`

**Coordinator** (`transport-routes/coordinator/src/lib.rs`):
- `log_maintenance(MaintenanceRecord) -> Record`
- `get_vehicle_maintenance(vehicle_hash) -> Vec<Record>`
- `set_vehicle_features(VehicleFeatures) -> Record`
- `get_accessible_vehicles() -> Vec<Record>` — filter by wheelchair_accessible=true
- `get_vehicles_needing_maintenance() -> Vec<Record>` — next_due < now

### 2b. transport-sharing — Add reviews & safety

**Integrity** (`transport-sharing/integrity/src/lib.rs`):
- New entry: `RideReview` — `match_hash, reviewer, role(Driver/Passenger), rating(1-5 u8), comment, safety_concern(bool), created_at`
- New link types: `MatchToReviews`, `AgentToReviews`

**Coordinator** (`transport-sharing/coordinator/src/lib.rs`):
- `review_ride(RideReview) -> Record` — validates match exists and is Completed, reviewer was participant
- `get_ride_reviews(match_hash) -> Vec<Record>`
- `get_driver_rating(agent) -> DriverRating` — avg rating, total trips, safety concerns count
- `find_nearby_rides(FindNearbyInput) -> Vec<Record>` — basic proximity filter (lat/lon within radius_km using Haversine)

### 2c. transport-impact — Add credit redemption

**Integrity** (`transport-impact/integrity/src/lib.rs`):
- New entry: `CreditRedemption` — `holder, credits_redeemed, redeemed_for(description), redeemed_at`
- New link type: `AgentToRedemptions`

**Coordinator** (`transport-impact/coordinator/src/lib.rs`):
- `redeem_credits(RedeemInput) -> Record` — validate holder has sufficient credits
- `get_my_redemptions() -> Vec<Record>`
- `get_agent_carbon_balance(agent) -> CarbonBalance` — earned minus redeemed

Add tests across all 3 zomes: ~60 new tests

## Phase 3: Support Zomes (commons-care DNA)

### 3a. support-tickets — Add escalation & satisfaction

**Integrity** (`support-tickets/integrity/src/lib.rs`):
- New entry: `EscalationEvent` — `ticket_hash, from_level(AutonomyLevel), to_level, reason, escalated_by, escalated_to(Option<AgentPubKey>), created_at`
- New entry: `SatisfactionSurvey` — `ticket_hash, responder, overall_rating(1-5 u8), resolution_quality(1-5 u8), responsiveness(1-5 u8), would_recommend(bool), comments(Option<String>), created_at`
- New link types: `TicketToEscalations`, `TicketToSurvey`

**Coordinator** (`support-tickets/coordinator/src/lib.rs`):
- `escalate_ticket(EscalateInput) -> Record` — validates ticket exists, records escalation chain
- `get_escalation_history(ticket_hash) -> Vec<Record>`
- `submit_satisfaction(SatisfactionSurvey) -> Record` — validates ticket is Resolved/Closed, one survey per ticket
- `get_ticket_satisfaction(ticket_hash) -> Option<Record>`

### 3b. support-knowledge — Add FAQ routing & duplicate detection

**Integrity** (`support-knowledge/integrity/src/lib.rs`):
- New entry: `ArticleTicketLink` — `article_hash, ticket_hash, linked_by, link_reason(SuggestedFAQ/DuplicateResolution/RelatedKnowledge), created_at`
- New link types: `ArticleToTickets`, `TicketToArticles`

**Coordinator** (`support-knowledge/coordinator/src/lib.rs`):
- `link_article_to_ticket(LinkArticleInput) -> Record`
- `get_suggested_articles(ticket_hash) -> Vec<Record>` — find articles matching ticket category/tags
- `find_similar_tickets(FindSimilarInput) -> Vec<Record>` — keyword match against open tickets

### 3c. support-diagnostics — Add helper profiles

**Integrity** (`support-diagnostics/integrity/src/lib.rs`):
- New entry: `HelperProfile` — `agent, expertise_categories(Vec<SupportCategory>), max_concurrent(u32), difficulty_preference(DifficultyLevel), available(bool)`
- New link type: `AllHelpers`

**Coordinator** (`support-diagnostics/coordinator/src/lib.rs`):
- `register_helper(HelperProfile) -> Record`
- `update_availability(UpdateAvailInput) -> Record`
- `get_available_helpers(category: Option<SupportCategory>) -> Vec<Record>`
- `get_helper_workload(agent) -> HelperWorkload` — count of assigned in-progress tickets

Add tests across all 3 zomes: ~50 new tests

## Phase 4: Food Zomes (commons-land DNA)

### 4a. food-production — Add allergens & community gardens

**Integrity** (`food-production/integrity/src/lib.rs`):
- New fields on `Crop`: `allergen_flags: Vec<String>, organic_certified: bool`
- New entry: `GardenMembership` — `plot_hash, member, role(Steward/Volunteer/Member), joined_at`
- New link type: `PlotToMembers`

**Coordinator** (`food-production/coordinator/src/lib.rs`):
- `add_garden_member(AddMemberInput) -> Record`
- `get_plot_members(plot_hash) -> Vec<Record>`
- `remove_garden_member(RemoveMemberInput) -> Record`

### 4b. food-knowledge — Add seed exchange & nutrition

**Integrity** (`food-knowledge/integrity/src/lib.rs`):
- New entry: `SeedStock` — `variety_hash(ActionHash), grower(AgentPubKey), quantity_grams(f64), location(String), germination_rate_pct(Option<f64>), available_for_exchange(bool), notes(Option<String>)`
- New entry: `SeedRequest` — `wanted_variety(String), quantity_grams(f64), requester(AgentPubKey), status(Open/Matched/Fulfilled), deadline(Option<u64>)`
- New entry: `NutrientProfile` — `crop_name(String), calories_per_100g(f64), protein_g(f64), carbs_g(f64), fat_g(f64), fiber_g(f64), key_vitamins(Vec<String>), key_minerals(Vec<String>)`
- New link types: `VarietyToStocks`, `AllSeedRequests`, `CropToNutrients`

**Coordinator** (`food-knowledge/coordinator/src/lib.rs`):
- `offer_seeds(SeedStock) -> Record`
- `request_seeds(SeedRequest) -> Record`
- `get_available_seeds(variety_hash) -> Vec<Record>`
- `get_open_seed_requests() -> Vec<Record>`
- `match_seed_request(MatchInput) -> Record` — link stock to request
- `add_nutrient_profile(NutrientProfile) -> Record`
- `get_nutrient_profile(crop_name) -> Option<Record>`

### 4c. food-preservation — Add batch-to-source tracing

**Integrity** (`food-preservation/integrity/src/lib.rs`):
- New fields on `PreservationBatch`: `allergen_flags: Vec<String>`
- Strengthen validation: `source_crop_hash` should warn if None (traceability gap)

### 4d. food-distribution — Add allergen filtering

**Integrity** (`food-distribution/integrity/src/lib.rs`):
- New fields on `Listing`: `allergen_flags: Vec<String>, organic: bool, cultural_markers: Vec<String>`

**Coordinator** (`food-distribution/coordinator/src/lib.rs`):
- `search_allergen_safe(AllergenSearchInput) -> Vec<Record>` — filter listings excluding specified allergens

Add tests across all 4 zomes: ~60 new tests

## Execution Order

1. **Space** (smallest, self-contained, fastest win)
2. **Transport** (next smallest, 3 independent zome edits parallelizable)
3. **Support** (medium complexity, 3 zome edits)
4. **Food** (largest, 4 zome edits, touches both integrity and coordinator)

Each phase: edit integrity first (types + validation), then coordinator (functions), then tests. Build and test after each phase.

## Testing Strategy

- Follow existing pattern: serde roundtrips for all new types, validation boundary tests, enum variant tests
- Run `cargo test --lib -p <zome-name>` after each zome edit
- Run `cargo build --release --target wasm32-unknown-unknown` after each phase to verify WASM compilation
- Total new tests: ~210 across all phases

## What We're NOT Doing (Deferred)

- No new zomes (DNA size constraint)
- No cross-zome integration changes (bridge already handles routing)
- No DNA manifest changes (existing zomes, no new WASM binaries)
- No TS SDK changes (those can follow separately)
- No payment/wallet system for Transport (requires broader architecture decision)
- No ML-based triage for Support (requires Symthaea integration planning)
- No composting/food-forest models for Food (would need new zomes)

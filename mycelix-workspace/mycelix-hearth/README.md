# Mycelix Hearth

Family, household, and kinship coordination on [Holochain](https://www.holochain.org/). Part of the [Mycelix](https://mycelix.net) distributed governance platform.

Hearth is a single-DNA cluster containing 11 domain zomes + 1 unified bridge, providing relationship-aware family governance with graduated autonomy, gratitude tracking, care scheduling, and cross-cluster integration.

## Architecture

```
mycelix-hearth/
  crates/hearth-types/       # Shared enums, DTOs, fixed-point math
  dna/dna.yaml               # DNA manifest (11 integrity + 11 coordinator zomes)
  happ.yaml                  # hApp bundle definition
  sdk-ts/src/clients/hearth/ # TypeScript SDK (11 zome clients + unified HearthClient)
  tests/                     # Sweettest integration tests (separate Cargo.toml)
  zomes/
    hearth-kinship/          # Core membership, bonds, invitations
    hearth-decisions/        # Voting, quorum, finalization
    hearth-gratitude/        # Gratitude expressions, appreciation circles
    hearth-stories/          # Family stories, collections, traditions
    hearth-care/             # Care schedules, swaps, meal plans
    hearth-autonomy/         # Graduated autonomy profiles, capability requests
    hearth-emergency/        # Emergency plans, alerts, safety check-ins
    hearth-resources/        # Shared resources, lending, budgets
    hearth-milestones/       # Life milestones, liminal transitions
    hearth-rhythms/          # Family rhythms, presence tracking
    hearth-bridge/           # Unified dispatch + cross-cluster bridge
```

**Workspace:** 25 members -- 1 shared types crate + 11 integrity zomes + 11 coordinator zomes + 2 bridge zomes (integrity + coordinator). The `tests/` directory is excluded from the workspace (separate `Cargo.toml`).

All zomes share the `hearth-types` crate for enums, DTOs, and fixed-point math. Intra-cluster calls use `call(CallTargetCell::Local, ...)`. Cross-cluster calls (to Identity, Commons, Civic, Personal) go through `hearth_bridge` via `CallTargetCell::OtherRole`.

## Zome API Reference

### hearth_kinship -- Core Membership & Bonds

| Function | Type | Description |
|----------|------|-------------|
| `create_hearth` | mutation | Create a hearth, auto-add caller as Founder |
| `invite_member` | mutation | Invite an agent (guardian-only) |
| `accept_invitation` | mutation | Accept an invitation, create membership |
| `decline_invitation` | mutation | Decline a pending invitation |
| `leave_hearth` | mutation | Depart a hearth (blocks last Founder) |
| `update_member_role` | mutation | Change a member's role (Founder/Elder only) |
| `create_kinship_bond` | mutation | Create a bond between two members |
| `tend_bond` | mutation | Record an interaction, update bond strength |
| `create_weekly_digest` | mutation | Write a weekly epoch rollup to DHT |
| `get_hearth_members` | query | List all members of a hearth |
| `get_my_hearths` | query | List hearths the caller belongs to |
| `get_kinship_graph` | query | Get all bonds for a hearth |
| `get_neglected_bonds` | query | Bonds with decayed strength < 3000bp |
| `get_bond_health` | query | Current decayed strength of a bond |
| `get_bond_snapshots` | query | Snapshot all bonds for digest assembly |
| `get_weekly_digests` | query | All weekly digests for a hearth |
| `is_guardian` | query | Check if caller has guardian role |
| `get_caller_vote_weight` | query | Caller's vote weight in basis points |
| `get_caller_role` | query | Caller's MemberRole in a hearth |
| `get_active_member_count` | query | Count of active members |

### hearth_decisions -- Family Decision-Making

| Function | Type | Description |
|----------|------|-------------|
| `create_decision` | mutation | Create a decision with options, deadline, quorum |
| `cast_vote` | mutation | Cast a weighted vote (role + consciousness) |
| `amend_vote` | mutation | Replace an existing vote (audit trail preserved) |
| `close_decision` | mutation | Close before deadline (creator or guardian) |
| `finalize_decision` | mutation | Tally votes, enforce quorum, record outcome |
| `tally_votes` | query | Current vote tallies per option |
| `get_decision` | query | Get a single decision |
| `get_hearth_decisions` | query | All decisions for a hearth |
| `get_decision_votes` | query | Current votes on a decision |
| `get_vote_history` | query | All votes ever cast (including amended) |
| `get_decision_outcome` | query | Outcome of a finalized decision |
| `get_my_pending_votes` | query | Open decisions the caller hasn't voted on |

### hearth_gratitude -- Gratitude & Appreciation

| Function | Type | Description |
|----------|------|-------------|
| `express_gratitude` | mutation | Express gratitude to another member |
| `start_appreciation_circle` | mutation | Start a themed circle with participants |
| `join_circle` | mutation | Join an existing circle |
| `complete_circle` | mutation | Mark a circle as completed |
| `get_gratitude_stream` | query | All gratitude for a hearth |
| `get_gratitude_balance` | query | Agent's gratitude given/received stats |
| `get_hearth_circles` | query | All appreciation circles for a hearth |
| `create_gratitude_digest` | query | Aggregate gratitude for a digest epoch |

### hearth_stories -- Stories & Traditions

| Function | Type | Description |
|----------|------|-------------|
| `create_story` | mutation | Create a family story with tags and media |
| `update_story` | mutation | Update title, content, and tags |
| `add_media_to_story` | mutation | Attach media to a story |
| `create_collection` | mutation | Create a curated story collection |
| `add_to_collection` | mutation | Add a story to a collection |
| `create_tradition` | mutation | Create a recurring family tradition |
| `observe_tradition` | mutation | Mark a tradition as observed |
| `get_hearth_stories` | query | All stories for a hearth |
| `get_hearth_traditions` | query | All traditions for a hearth |
| `get_hearth_collections` | query | All collections for a hearth |
| `search_stories_by_tag` | query | Find stories by tag |

### hearth_care -- Care Scheduling & Meal Plans

| Function | Type | Description |
|----------|------|-------------|
| `create_care_schedule` | mutation | Create a recurring care task |
| `complete_task` | mutation | Mark a care task as completed |
| `propose_swap` | mutation | Propose swapping a care task |
| `accept_swap` | mutation | Accept a proposed swap |
| `decline_swap` | mutation | Decline a proposed swap |
| `create_meal_plan` | mutation | Create a weekly meal plan |
| `get_my_care_duties` | query | Caller's assigned care tasks |
| `get_hearth_schedule` | query | Full care schedule for a hearth |
| `get_hearth_meal_plans` | query | All meal plans for a hearth |
| `create_care_digest` | query | Aggregate care for a digest epoch |

### hearth_autonomy -- Graduated Autonomy

| Function | Type | Description |
|----------|------|-------------|
| `create_autonomy_profile` | mutation | Create a profile for a minor (guardian-only) |
| `request_capability` | mutation | Request a new capability (youth action) |
| `approve_capability` | mutation | Approve/deny a request (guardian-only) |
| `deny_capability` | mutation | Deny a request (convenience wrapper) |
| `advance_tier` | mutation | Forward-only tier advancement |
| `progress_transition` | mutation | Advance transition phase toward Integrated |
| `get_autonomy_profile` | query | Get a member's autonomy profile |
| `check_capability` | query | Runtime permission check for a capability |
| `get_pending_requests` | query | Pending capability requests for a hearth |
| `get_active_transitions` | query | Non-integrated tier transitions |

### hearth_emergency -- Emergency Plans & Alerts

| Function | Type | Description |
|----------|------|-------------|
| `create_emergency_plan` | mutation | Create an emergency plan for a hearth |
| `update_emergency_plan` | mutation | Update an existing plan |
| `raise_alert` | mutation | Raise an emergency alert |
| `check_in` | mutation | Report safety status during an alert |
| `resolve_alert` | mutation | Resolve an active alert |
| `get_active_alerts` | query | Unresolved alerts for a hearth |
| `get_alert_checkins` | query | Check-ins for a specific alert |
| `get_emergency_plan` | query | Current emergency plan for a hearth |

### hearth_resources -- Shared Resources & Budgets

| Function | Type | Description |
|----------|------|-------------|
| `register_resource` | mutation | Register a shared resource |
| `lend_resource` | mutation | Lend a resource to a member |
| `return_resource` | mutation | Mark a loan as returned |
| `create_budget_category` | mutation | Create a budget category |
| `log_expense` | mutation | Log an expense against a budget |
| `get_hearth_inventory` | query | All shared resources |
| `get_budget_summary` | query | All budget categories |
| `get_resource_loans` | query | Loans for a specific resource |

### hearth_milestones -- Life Milestones & Transitions

| Function | Type | Description |
|----------|------|-------------|
| `record_milestone` | mutation | Record a life milestone |
| `begin_transition` | mutation | Start a liminal transition |
| `advance_transition` | mutation | Progress transition phase |
| `complete_transition` | mutation | Finalize an Integrated transition |
| `get_family_timeline` | query | All milestones for a hearth |
| `get_member_milestones` | query | Milestones for a specific member |
| `get_active_transitions` | query | Non-completed transitions |

### hearth_rhythms -- Family Rhythms & Presence

| Function | Type | Description |
|----------|------|-------------|
| `create_rhythm` | mutation | Create a family rhythm |
| `log_occurrence` | mutation | Log a rhythm occurrence |
| `set_presence` | mutation | Set presence status (Home, Away, etc.) |
| `get_hearth_rhythms` | query | All rhythms for a hearth |
| `get_rhythm_occurrences` | query | Occurrences for a rhythm |
| `get_hearth_presence` | query | Presence statuses for a hearth |
| `create_rhythm_digest` | query | Aggregate rhythms for a digest epoch |

### hearth_bridge -- Unified Dispatch & Cross-Cluster

| Function | Type | Description |
|----------|------|-------------|
| `dispatch_call` | dispatch | Synchronous call to any domain zome |
| `query_hearth` | dispatch | Auto-routed query with domain resolution |
| `resolve_query` | dispatch | Resolve a pending async query |
| `broadcast_event` | dispatch | Broadcast a cross-domain event |
| `get_domain_events` | query | Events for a specific domain |
| `get_all_events` | query | All bridge events |
| `get_events_by_type` | query | Events filtered by type |
| `get_my_queries` | query | Caller's pending queries |
| `dispatch_personal_call` | cross-cluster | Call Personal cluster |
| `dispatch_identity_call` | cross-cluster | Call Identity cluster |
| `dispatch_commons_call` | cross-cluster | Call Commons cluster |
| `dispatch_civic_call` | cross-cluster | Call Civic cluster |
| `verify_member_identity` | cross-cluster | Verify agent via Identity |
| `escalate_emergency` | cross-cluster | Escalate alert to Civic |
| `query_timebank_balance` | cross-cluster | Query Commons timebank |
| `initiate_severance` | cross-cluster | Coming-of-age data export |
| `hearth_sync` | mutation | Weekly digest assembly + store |
| `get_weekly_digests` | query | Delegates to kinship |
| `health_check` | query | Bridge health status |
| `get_consciousness_credential` | query | Consciousness credential for governance |
| `log_governance_gate` | mutation | Record a governance gate audit entry |
| `get_governance_audit_trail` | query | Query governance gate audit trail |

## Getting Started

```bash
# Enter the nix development shell
nix develop

# Build all WASM zomes
cargo build --release --target wasm32-unknown-unknown

# Pack the DNA and hApp
hc dna pack dna/
hc app pack .
```

## TypeScript SDK

The SDK provides individual zome clients and a unified `HearthClient` that composes all 11.

```typescript
import { HearthClient } from './clients/hearth';
import type { AppClient } from '@holochain/client';

// Initialize with a Holochain AppClient
const hearth = new HearthClient(appClient);

// Create a hearth
const record = await hearth.kinship.createHearth({
  name: 'The Stoltz Family',
  description: 'A loving home',
  hearth_type: { Nuclear: null },
  max_members: 10,
});

// Express gratitude
await hearth.gratitude.expressGratitude({
  hearth_hash: hearthHash,
  to_agent: partnerAgentKey,
  message: 'Thank you for dinner',
  gratitude_type: { Appreciation: null },
  visibility: { AllMembers: null },
});

// Cast a vote
await hearth.decisions.castVote({
  decision_hash: decisionHash,
  choice: 0,
  reasoning: 'I agree with option A',
});

// Subscribe to all signals
const unsub = hearth.onSignal((signal) => {
  console.log('Signal:', getSignalType(signal), signal);
});

// Subscribe to specific signal types only
const unsub2 = hearth.onSignal(
  (signal) => console.log('Emergency!', signal),
  ['EmergencyAlert', 'MemberDeparted'],
);

// Subscribe to bridge event signals (separate struct)
const unsub3 = hearth.onBridgeSignal((signal) => {
  console.log('Bridge:', signal.signal_type, signal.domain);
});

// Unsubscribe when done
unsub(); unsub2(); unsub3();
```

### SDK Client Summary

| Client | Class | Methods | Key Signals |
|--------|-------|---------|-------------|
| kinship.ts | KinshipClient | 20 | MemberJoined, MemberDeparted, BondTended |
| decisions.ts | DecisionsClient | 12 | VoteCast, VoteAmended, DecisionClosed, DecisionFinalized |
| gratitude.ts | GratitudeClient | 8 | GratitudeExpressed |
| stories.ts | StoriesClient | 11 | StoryCreated, StoryUpdated, TraditionObserved |
| care.ts | HearthCareClient | 10 | CareTaskCompleted, SwapAccepted, SwapDeclined |
| autonomy.ts | AutonomyClient | 10 | TierAdvanced, CapabilityApproved, CapabilityDenied |
| emergency.ts | EmergencyClient | 8 | EmergencyAlert |
| resources.ts | ResourcesClient | 8 | ResourceLent, ResourceReturned, ExpenseLogged |
| milestones.ts | MilestonesClient | 7 | MilestoneRecorded, TransitionAdvanced |
| rhythms.ts | RhythmsClient | 7 | RhythmOccurred, PresenceChanged |
| bridge.ts | BridgeClient | 22 | CrossZomeCallFailed |
| index.ts | HearthClient | (composite) | All 27 signal types via `onSignal()` |

Signal handling is centralized on `HearthClient.onSignal(callback, filter?)` rather than individual zome clients, since signals arrive via the WebSocket connection. The optional `filter` parameter accepts an array of `HearthSignalType` strings to listen for specific events. Bridge events use a separate `onBridgeSignal()` method.

## Tests

**Rust unit tests (929):** Run from the workspace root.

```bash
cargo test --workspace
```

**TypeScript SDK unit tests (134):** Run from the sdk-ts directory.

```bash
cd mycelix-workspace/sdk-ts
npx vitest run tests/hearth.test.ts
```

**Sweettest integration tests (18 across 12 files):** Require a running Holochain conductor.

```bash
cd tests
cargo test --release -- --ignored --test-threads=2
```

| Test File | Tests | Coverage |
|-----------|-------|----------|
| sweettest_kinship | 6 | Create hearth, invite/accept, bonds, my_hearths, departure, bond queries |
| sweettest_decisions_kinship | 3 | Decision lifecycle, cross-zome role check, decision amendment, governance |
| sweettest_bridge | 2 | Health check, dispatch to kinship |
| sweettest_digest_sync | 1 | Weekly digest assembly across zomes |
| sweettest_care_gratitude_rhythms | 6 | Care schedule, swap lifecycle, gratitude, circles, rhythm logging, occurrences |
| sweettest_emergency | 1 | Emergency plan/alert/check-in/resolve |
| sweettest_autonomy | 3 | Guardian-youth capability flow, tier advancement, capability denial |
| sweettest_multi_hearth | 1 | Cross-hearth data isolation |
| sweettest_resources | 4 | Register, lend/return, budget, membership auth |
| sweettest_milestones | 4 | Record, advance transition, guardian auth, complete |
| sweettest_stories | 5 | Create, update, collection, tradition, membership auth |
| sweettest_consciousness_gating | 9 | Gate enforcement, cache, audit, cross-zome, constitutional |

Integration tests are marked `#[ignore]` and expected to fail in CI (no conductor available).

## Cross-Cluster Bridge Patterns

The `hearth_bridge` zome mediates two kinds of calls:

**Intra-cluster (domain-to-domain):** Any Hearth zome can call another via `call(CallTargetCell::Local, zome_name, fn_name, None, payload)`. For example, `hearth_decisions` calls `hearth_kinship::get_caller_role` to check vote eligibility, and `hearth_autonomy` calls `hearth_kinship::is_guardian` to enforce guardian-only actions.

**Cross-cluster (Hearth to other DNAs):** The bridge dispatches to other clusters via `call(CallTargetCell::OtherRole(RoleName::from("identity")), ...)`. Dedicated functions handle common patterns:

- `dispatch_identity_call` -- Verify members, social recovery setup
- `dispatch_commons_call` -- Timebank queries, mutual aid coordination
- `dispatch_civic_call` -- Emergency escalation, justice referrals
- `dispatch_personal_call` -- Vault export (severance/coming-of-age)

Cross-cluster calls use MessagePack-encoded `Uint8Array` payloads in the TS SDK.

## Key Design Decisions

**Fixed-point math (basis points).** All numeric values on consensus paths use `u32` basis points (0-10000) instead of `f64`. This prevents floating-point divergence across WASM runtimes (ARM vs x86). Vote weights, bond strengths, participation rates, and quorum thresholds are all in basis points.

**Bond strength decay model.** Kinship bonds decay over time using a pre-computed lookup table (`DECAY_TABLE` in `hearth-types`) with integer-only linear interpolation. Bonds have a minimum floor of 1000bp -- they never fully dissolve. The `tend_bond` function blends 70% current decayed health with 30% interaction quality.

**H2 epoch rollups.** High-frequency activity (gratitude, care completions, rhythm occurrences) flows via real-time signals. Weekly digests (`WeeklyDigest`) aggregate these into a single DHT entry per week, reducing storage overhead. The `hearth_sync` bridge function orchestrates digest assembly across gratitude, care, and rhythm zomes.

**Graduated autonomy with liminal transitions.** Tier advancement (Dependent -> Supervised -> Guided -> SemiAutonomous -> Autonomous) is forward-only. Each advancement creates a `TierTransition` that progresses through four phases (PreLiminal -> Liminal -> PostLiminal -> Integrated). Recategorization is blocked until the transition completes. Advancement to Autonomous triggers a "severance" -- exporting milestones, care history, and bond snapshots to the departing member's Personal vault.

**Guardian role checks via cross-zome calls.** Instead of duplicating membership logic, zomes like `hearth_decisions` and `hearth_autonomy` call `hearth_kinship::is_guardian` or `hearth_kinship::get_caller_role` at runtime. This keeps authorization rules in one place.

**Consciousness-aware governance.** The decisions zome integrates with `mycelix-bridge-common`'s consciousness credential system. Vote weight is composed from role-based weight and consciousness tier weight: `effective_weight = role_bp * consciousness_bp / 10000`. This creates progressive governance where deeper engagement earns greater influence.

## Dependencies

| Dependency | Version | Purpose |
|------------|---------|---------|
| hdk | 0.6.0 | Holochain Development Kit |
| hdi | 0.7.0 | Holochain Integrity |
| serde | 1.0 | Serialization |
| hearth-types | workspace | Shared types and fixed-point math |
| mycelix-bridge-common | workspace | Cross-cluster dispatch + consciousness gating |
| mycelix-bridge-entry-types | workspace | Shared DHT entry types |

## License

Part of the Mycelix ecosystem. See the repository root for license details.

# Mycelix Praxis: Differentiation Features Implementation Complete

**Date**: 2025-12-30
**Status**: Phase 1 Complete - All 4 Differentiation Zomes Implemented
**Tests**: 29/29 Passing

---

## Overview

The Praxis differentiation layer provides personalized, adaptive learning through four interconnected Holochain zomes:

1. **Spaced Repetition System (SRS)** - SM-2 algorithm for optimized memory retention
2. **Gamification Engine** - XP, badges, streaks, and leaderboards
3. **Adaptive Learning Intelligence** - Bayesian Knowledge Tracing and ZPD-aware recommendations
4. **Integration Layer** - Cross-zome coordination and session orchestration

All zomes are compatible with **Holochain 0.6** and follow consistent patterns for:
- Timestamp handling (`ts.as_micros()`)
- Link resolution (`link.target.into_action_hash()`)
- Path-based anchors for learner-specific data
- Permille (0-1000) instead of floats for `Eq` compatibility

---

## Zome Summary

### 1. SRS Zome (`srs_integrity` + `srs_coordinator`)

**Purpose**: Implements SuperMemo SM-2 algorithm for spaced repetition.

**Entry Types** (6):
- `ReviewCard` - Flashcards with SM-2 scheduling metadata
- `ReviewEvent` - Individual review history
- `ReviewSession` - Grouped review sessions
- `DailyStats` - Aggregated daily statistics
- `SrsConfig` - Configurable SM-2 parameters
- `Deck` - Card collections

**Key Features**:
- SM-2 interval calculation with ease factor
- Learning/graduated card progression
- Leech detection and handling
- Review forecasting
- Retention targeting (default 90%)

**Tests**: 10 (6 integrity + 4 coordinator)

**Example Usage**:
```rust
// Create a review card
let card = create_card(CreateCardInput {
    deck_hash: deck_action_hash,
    front: "What is photosynthesis?".into(),
    back: "The process by which plants convert light to energy".into(),
    tags: vec!["biology".into()],
})?;

// Review the card
let result = review_card(ReviewInput {
    card_hash: card_hash,
    quality: RecallQuality::Good, // 0-5 scale
})?;
// result.next_review_at calculated by SM-2
```

---

### 2. Gamification Zome (`gamification_integrity` + `gamification_coordinator`)

**Purpose**: Engagement through XP, achievements, streaks, and competition.

**Entry Types** (10):
- `LearnerXp` - XP totals and level progression
- `XpTransaction` - Individual XP events
- `LearnerStreak` - Daily login streak tracking
- `DailyActivity` - Daily activity log
- `BadgeDefinition` - Badge templates
- `EarnedBadge` - Earned badge instances
- `BadgeProgress` - Progress toward badges
- `Leaderboard` - Competitive rankings
- `Reward` - Claimable rewards
- `ClaimedReward` - Claimed reward records

**Key Features**:
- XP multipliers with stacking rules
- 7-tier badge rarity (Common to Mythic)
- Streak freeze protection
- Weekly/monthly/all-time leaderboards
- Achievement system with progress tracking

**Tests**: 5 (4 integrity + 1 coordinator)

**XP Calculation**:
```rust
// XP formula with multipliers
let final_xp = base_xp * activity_mult * streak_mult * difficulty_mult;

// Example: 100 base XP with 1.5x streak, 1.2x difficulty
// = 100 * 1.0 * 1.5 * 1.2 = 180 XP
```

---

### 3. Adaptive Learning Intelligence (`adaptive_integrity` + `adaptive_coordinator`)

**Purpose**: ML-inspired personalization using BKT and ZPD theory.

**Entry Types** (10):
- `LearnerProfile` - Learning style and preferences
- `LearningStyleAssessment` - VARK assessment results
- `SkillMastery` - Per-skill mastery with BKT
- `DifficultyCalibration` - Content difficulty ratings
- `Recommendation` - Personalized content suggestions
- `LearningGoal` - Goal setting and tracking
- `SessionAnalytics` - Per-session analysis
- `AggregatedAnalytics` - Long-term trends
- `AdaptivePath` - Personalized learning paths
- `PathStep` - Path milestones

**Key Features**:
- **Bayesian Knowledge Tracing (BKT)**:
  - P(L) - Probability of learning
  - P(T) - Transition probability
  - P(G) - Guess probability
  - P(S) - Slip probability
- **Zone of Proximal Development (ZPD)**:
  - Lower bound: current mastery + 50‰
  - Upper bound: current mastery + 150‰
  - Optimal: current mastery + 100‰
- **VARK Learning Styles**:
  - Visual, Auditory, Reading/Writing, Kinesthetic
- **6-tier Mastery Levels**:
  - Novice → Beginner → Competent → Proficient → Expert → Master

**Smart Recommendation Engine** (Added 2025-12-30):

The advanced smart recommendation engine uses learning science principles:

- **Flow State Theory**: Optimal difficulty matching (80-85% of mastery level)
- **Circadian Rhythm**: Time-of-day optimization for content types
  - Morning (6-11): Best for lessons and new concepts
  - Afternoon (12-16): Best for exercises and practice
  - Evening (17-20): Best for projects and creative work
  - Night (21+): Best for light review only
- **VARK Learning Style Matching**: Content aligned to learner preferences
- **Interleaving**: Topic switching after 3+ consecutive same-topic items
- **Goal Alignment**: Priority-based scoring for learner objectives
- **Energy Estimation**: Fatigue tracking (50‰ per session, capped at 400‰)

**Smart Score Formula**:
```
smart_score = (flow × 25 + style × 20 + timing × 15 + goal × 25 + interleave × 15) / 100
```

**Tests**: 11 (5 integrity + 6 coordinator including 7 smart recommendation tests)

**BKT Update Example**:
```rust
// After correct answer:
new_mastery = p_learn + (1.0 - p_learn) * p_transition;

// After incorrect answer:
new_mastery = p_learn * p_slip /
    (p_learn * p_slip + (1.0 - p_learn) * (1.0 - p_guess));
```

---

### 4. Integration Layer (`integration_integrity` + `integration_coordinator`)

**Purpose**: Orchestrates cross-zome interactions and unified progress tracking.

**Entry Types** (5):
- `LearningEvent` - Unified event from any zome
- `LearnerProgressAggregate` - Combined progress snapshot
- `AchievementTrigger` - Conditional badge triggers
- `OrchestratedSession` - Coordinated learning sessions
- `DailyLearningReport` - Daily summary

**Key Features**:
- Unified event processing with XP calculation
- Progress aggregation from all systems
- Session planning with priority-based activities
- Flow state tracking
- Conditional achievement triggers
- Daily report generation

**Tests**: 6 (5 integrity + 3 coordinator)

**Cross-Zome Architecture**:
```
┌─────────────────────────────────────────────────────┐
│              Integration Coordinator                 │
│  (uses HDK `call` for inter-zome communication)     │
└───────────┬───────────┬───────────┬─────────────────┘
            │           │           │
     ┌──────┴───┐ ┌─────┴────┐ ┌────┴─────┐
     │   SRS    │ │  Gamify  │ │ Adaptive │
     │ Coord.   │ │  Coord.  │ │  Coord.  │
     └────┬─────┘ └────┬─────┘ └────┬─────┘
          │            │            │
     ┌────┴─────┐ ┌────┴─────┐ ┌────┴─────┐
     │   SRS    │ │  Gamify  │ │ Adaptive │
     │Integrity │ │Integrity │ │Integrity │
     └──────────┘ └──────────┘ └──────────┘
```

---

## Test Results Summary

| Zome | Integrity Tests | Coordinator Tests | Total |
|------|----------------|-------------------|-------|
| SRS | 6 | 7 | 13 |
| Gamification | 4 | 1 | 5 |
| Adaptive | 5 | 33 | 38 |
| Integration | 11 | 6 | 17 |
| praxis-core | 0 | 17 | 17 |
| **Total** | **26** | **64** | **90** |

**Update (2025-12-30)**: Added comprehensive validation logic to Integration zome with 6 additional tests.
Added smart recommendations, caching, flow state tracking, and performance metrics to Adaptive zome (33 coordinator tests).

## Web Client TypeScript Types

Added **534 lines** of TypeScript types in `apps/web/src/types/zomes.ts`:

### SRS Types (83 lines)
- `RecallQuality`, `CardStatus`, `LeechAction` enums
- `ReviewCard`, `ReviewEvent`, `ReviewSession`, `Deck`, `SrsConfig` interfaces
- `SrsZomeFunctions` function interface

### Gamification Types (108 lines)
- `XpActivityType`, `BadgeRarity`, `BadgeCategory`, `LeaderboardType/Period/Scope` enums
- `LearnerXp`, `XpTransaction`, `LearnerStreak`, `BadgeDefinition`, `EarnedBadge`, `Leaderboard`, `LeaderboardEntry` interfaces
- `GamificationZomeFunctions` function interface

### Adaptive Learning Types (119 lines)
- `LearningStyle`, `MasteryLevel`, `ContentType`, `RecommendationType`, `RecommendationReason`, `GoalType`, `GoalPriority` enums
- `LearnerProfile`, `SkillMastery`, `Recommendation`, `LearningGoal`, `SessionAnalytics` interfaces
- `AdaptiveZomeFunctions` function interface

### Integration Types (207 lines)
- `LearningEventType`, `SourceZome`, `TriggerMetric`, `ComparisonOp`, `SessionState`, `PlannedActivityType` enums
- `LearningEvent`, `LearnerProgressAggregate`, `TriggerCondition`, `TriggerReward`, `AchievementTrigger`, `PlannedActivity`, `CompletedActivity`, `MasteryChange`, `OrchestratedSession`, `DailyLearningReport` interfaces
- `IntegrationZomeFunctions` function interface

### Combined Interface
```typescript
export interface ZomeFunctions {
  learning: LearningZomeFunctions;
  fl: FlZomeFunctions;
  credential: CredentialZomeFunctions;
  dao: DaoZomeFunctions;
  srs: SrsZomeFunctions;              // NEW
  gamification: GamificationZomeFunctions;  // NEW
  adaptive: AdaptiveZomeFunctions;    // NEW
  integration: IntegrationZomeFunctions;    // NEW
}
```

---

## Holochain 0.6 Patterns Used

### Timestamp Handling
```rust
fn timestamp_to_i64(ts: Timestamp) -> i64 {
    ts.as_micros()  // NOT .as_secs_f64() - that doesn't exist
}

fn current_time() -> ExternResult<i64> {
    Ok(timestamp_to_i64(sys_time()?))
}
```

### Link Resolution
```rust
// CORRECT - returns Option<ActionHash>
if let Some(action_hash) = link.target.into_action_hash() {
    if let Some(record) = get(action_hash, GetOptions::default())? {
        // process record
    }
}

// WRONG - ActionHash::try_from doesn't work with HashConversionError
// let hash = ActionHash::try_from(link.target)?; // ERROR!
```

### Get Links Pattern
```rust
let links = get_links(
    LinkQuery::try_new(anchor, LinkTypes::MyLinkType)?,
    GetStrategy::Local,
)?;
```

### Path-Based Anchors
```rust
fn profile_anchor() -> ExternResult<EntryHash> {
    let agent = agent_info()?.agent_initial_pubkey;
    let path = Path::from(format!("zome.category.{}", agent));
    let typed_path = path.typed(LinkTypes::MyLinkType)?;
    typed_path.ensure()?;
    typed_path.path.path_entry_hash()
}
```

### Permille for Eq-Compatible Percentages
```rust
// Use u16 (0-1000) instead of f64 (0.0-1.0)
#[hdk_entry_helper]
#[derive(Clone, PartialEq, Eq)]  // Eq works with integers, not floats
pub struct MyEntry {
    pub accuracy_permille: u16,  // 750 = 75.0%
}
```

---

## File Structure

```
zomes/
├── srs_zome/
│   ├── integrity/
│   │   ├── Cargo.toml
│   │   └── src/lib.rs          # 6 entry types, SM-2 calculation
│   └── coordinator/
│       ├── Cargo.toml
│       └── src/lib.rs          # 7 zome functions
├── gamification_zome/
│   ├── integrity/
│   │   ├── Cargo.toml
│   │   └── src/lib.rs          # 10 entry types, XP formulas
│   └── coordinator/
│       ├── Cargo.toml
│       └── src/lib.rs          # 9 zome functions
├── adaptive_zome/
│   ├── integrity/
│   │   ├── Cargo.toml
│   │   └── src/lib.rs          # 10 entry types, BKT/ZPD
│   └── coordinator/
│       ├── Cargo.toml
│       └── src/lib.rs          # 15+ zome functions
└── integration_zome/
    ├── integrity/
    │   ├── Cargo.toml
    │   └── src/lib.rs          # 5 entry types, cross-zome types
    └── coordinator/
        ├── Cargo.toml
        └── src/lib.rs          # 10+ zome functions
```

---

## Structured Error Handling

The Praxis differentiation layer uses structured error handling with descriptive messages, error codes, and resolution hints.

### Error Module Location
`crates/praxis-core/src/errors.rs`

### Error Categories
| Code Range | Category | Example |
|------------|----------|---------|
| 1xx | Entity errors | EntityNotFound (100), EntityAlreadyExists (101) |
| 2xx | Validation errors | ValidationFailed (200), OutOfRange (202) |
| 3xx | Authorization errors | Unauthorized (300), AuthorMismatch (302) |
| 4xx | State errors | InvalidStateTransition (400) |
| 5xx | External errors | CrossZomeCallFailed (500), NetworkError (501) |
| 6xx | Resource errors | LimitExceeded (601), QuotaExceeded (602) |

### Error Format
```
[E100] ReviewCard get failed: Entry not found in DHT (hash: Qm123abc) → Hint: The entry may have been deleted or never created
```

### Usage Example
```rust
use praxis_core::errors::{srs_errors, PraxisError};

// Convert PraxisError to WasmError
fn to_wasm_error(err: PraxisError) -> WasmError {
    wasm_error!(WasmErrorInner::Guest(err.to_message()))
}

// Use structured error
let record = get(hash.clone(), GetOptions::default())?
    .ok_or_else(|| to_wasm_error(srs_errors::card_not_found(&hash.to_string())))?;
```

### Domain-Specific Error Modules
- `srs_errors` - Card, deck, session, recall quality errors
- `gamification_errors` - Badge, streak, XP, leaderboard errors
- `adaptive_errors` - Profile, skill, goal, ZPD errors
- `integration_errors` - Session, trigger, cross-zome call errors

---

## Cross-Zome Communication

The Integration Layer uses Holochain's `call` function for inter-zome communication.

### Cross-Zome Call Pattern
```rust
fn call_srs<I, O>(fn_name: &str, input: I) -> ExternResult<O>
where
    I: serde::Serialize + std::fmt::Debug,
    O: serde::de::DeserializeOwned + std::fmt::Debug,
{
    let response = call(
        CallTargetCell::Local,
        ZomeName::from("srs_coordinator"),
        FunctionName::from(fn_name),
        None,
        &input,
    )?;
    // Handle all ZomeCallResponse variants...
}
```

### Available Cross-Zome Functions
| Zome | Function | Purpose |
|------|----------|---------|
| srs_coordinator | `get_my_cards` | Get learner's review cards |
| srs_coordinator | `get_due_cards` | Get cards due for review |
| gamification_coordinator | `get_gamification_summary` | XP, level, streak summary |
| adaptive_coordinator | `get_learner_summary` | Profile, mastery, goals summary |

---

## Next Steps

1. ~~**Web Client Integration** - Add TypeScript types for new zomes~~ ✅
2. **End-to-End Testing** - Test complete learning workflows
3. **Performance Optimization** - Benchmark aggregation queries
4. **UI Components** - Build React components for each feature
5. ~~**Documentation** - API reference for each zome function~~ ✅

---

## Related Documentation

- [Differentiation Zomes API Reference](../api/DIFFERENTIATION_ZOMES_API.md) - Complete API documentation
- [V0.2 Implementation Plan](./V0_2_IMPLEMENTATION_PLAN.md)
- [Phase 1-6 Reports](./PHASE_1_COMPLETE.md)
- [Holochain 0.6 Migration Guide](../../mycelix-workspace/docs/HOLOCHAIN_0.6_MIGRATION_GUIDE.md)

---

*Built with the Sacred Trinity: Human Vision + Claude Code + Holochain*

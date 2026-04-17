# Mycelix Simulation — Phase 1 Survey

*2026-04-17. Audit of `mycelix-multiworld-sim` (42,577 LOC, 71 modules) against Mycelix production spec.*

## Executive summary

Multiworld-sim already models the Mycelix three-currency system and consciousness-gated governance at high fidelity. The gap between the current simulator and a full Mycelix systems simulator is **smaller than expected** — primarily 4 missing modules covering 8D sovereign profiles, sub-passport dynamics, DKG costs, and cross-cluster call modeling.

Estimated coverage: **~60% already present** → **~85% reachable** in 2-3 focused sessions.

## What's already modeled (strong fidelity)

### Currencies — `src/currency.rs`
Exact parameters matching Mycelix spec:

| Parameter | Value | Matches Mycelix |
|---|---|---|
| `MYCEL_ANNUAL_DECAY` | 5%/year | ✓ |
| `MYCEL_MONTHLY_FACTOR` | 0.99573 | ✓ |
| `MYCEL_JUBILEE_FACTOR` | 0.8 (every 48 ticks) | ✓ |
| `SAP_ANNUAL_DEMURRAGE` | 2%/year | ✓ |
| `SAP_FLOW_LOCAL / PLANETARY / SYSTEM` | 70% / 20% / 10% | ✓ |
| `SAP_COMMONS_RESERVE_FRACTION` | 25% | ✓ |
| `SAP_EXEMPT_FLOOR` | 200.0 | ✓ |
| `TEND_LIMIT` | 40 | ✓ |
| `TEND_TEACHING_REWARD / CARE_REWARD` | 5 / 3 | ✓ |
| MYCEL weights (participation/recognition/quality/longevity) | 0.40/0.20/0.20/0.20 | ✓ |

Functions: `compute_mycel`, `apply_jubilee`, `apply_sap_demurrage`, `distribute_demurrage`, `spend_commons`.

### Governance — `src/governance.rs` + `governance_hardening.rs`

- 5 authority levels: `MissionControl → LocalWithEarthVeto → LocalSovereign → Federation → Confederation`
- Trust-weighted voting gated by Phi threshold per tier
- Veto override: ⅔ supermajority (matches Mycelix Article III §3)
- Veto cooldown: 7 ticks per Guardian (prevents serial veto DoS)
- Emergency power duration: 12 ticks (1 year)
- Oppression streak detection (12-tick crisis threshold)
- Stagnation penalty after 120 ticks without amendment
- `effective_tier`, `effective_quorum`, `cap_unsigned_phi` — mirror `mycelix-bridge-common` APIs

### Red-team / adversarial — `src/red_team.rs`
- `AdversarialStrategy` enum
- `RedTeamConfig`, `RedTeamReport`
- `AdversarialModifier`, `evaluate_resilience`

### Related modules present but not audited in detail
- `proposals.rs`, `sanctions.rs`, `factions.rs`
- `civic_dimensions.rs`, `consciousness_epidemiology.rs`
- `epistemic_decay.rs`
- `economy.rs` (sector-based economy, 8 sectors)

## What's missing (the actual gap)

### Gap 1 — 8D Sovereign Profile (HIGH VALUE)

Mycelix production uses the 8D `SovereignProfile` from `crates/mycelix-bridge-common/src/sovereign_gate.rs`:
1. EpistemicIntegrity
2. NetworkResilience
3. EconomicVelocity
4. StewardshipCare
5. CivicParticipation
6. SemanticResonance
7. ThermodynamicYield
8. DomainCompetence

Multiworld-sim currently uses scalar Phi for consciousness gating. **No reference to `SovereignProfile`, no 8D dimensions.** This means simulated tier assignment doesn't reflect production's actual gating logic.

**Impact**: scenarios exploring "what if a user has high epistemic integrity but low economic velocity" can't be run.

### Gap 2 — Sub-Passport dynamics (HIGH VALUE)

`crates/mycelix-bridge-common/src/sub_passport.rs` (607 LOC) implements:
- Moral delegation (Read → Execute → Govern)
- Ahimsa enforcement
- Restorative justice tracker
- Violation/correction counts with 6h cooldown, 3:1 correction ratio
- Effective-tier recovery (gradual one-tier-per-cooldown)

Multiworld-sim has no equivalent. **No modeling of how users recover from moral violations, or how the 3:1 correction ratio affects long-run tier distributions.**

**Impact**: can't simulate sybil recovery attacks, correction-farming, or restorative-vs-retributive justice outcomes.

### Gap 3 — DKG cost modeling (MEDIUM VALUE)

Mycelix governance uses Feldman DKG for threshold signing on high-stakes proposals. Multiworld-sim has no DKG module — threshold signing is implicit (just counts yes/no).

**What's needed**: cost model for DKG rounds (participants × 2 rounds × network latency), dropout tolerance, and impact on proposal throughput.

**Impact**: can't calibrate how big governance committees can be before DKG round-trip times make real-time decisions impractical.

### Gap 4 — Cross-cluster call modeling (MEDIUM VALUE)

Mycelix's fractal architecture uses `CallTargetCell::OtherRole()` dispatch through `routing_registry.rs` (13 routes across clusters). Each cross-cluster call has cost (signature verification, consciousness gating check, audit log entry).

Multiworld-sim has no model of bridge dispatch latency or cost accumulation.

**Impact**: can't simulate scenarios like "what if 80% of commons operations require cross-cluster calls to identity — does the system stay viable?"

### Gap 5 — Mycelix-specific red-team scenarios (LOW-MEDIUM VALUE)

Existing `red_team.rs` has generic adversarial strategies. Missing Mycelix-specific attack vectors:
- **Tier-buying**: accumulating SAP/MYCEL to artificially boost CivicParticipation
- **Demurrage evasion**: stash-and-move patterns to avoid SAP decay
- **Sub-Passport correction farming**: cycling violations to artificially inflate correction count
- **Cross-cluster amplification**: using one cluster's gate to bypass another
- **Guild collusion**: coordinated vote weighting in Craft guild federations

## Recommended execution order

1. **Phase 2a — 8D Sovereign Profile integration** (1 session, ~1 day)
   - Add `src/sovereign_profile.rs` mirroring bridge-common's 8D struct
   - Replace scalar `phi` gating with 8D-weighted gating in `governance.rs`
   - Add Monte Carlo initialization for sovereign profile distributions

2. **Phase 2b — Sub-Passport + restorative justice** (1 session, ~1 day)
   - Add `src/sub_passport.rs` with violation/correction dynamics
   - Integrate with existing `sanctions.rs`
   - Test: does 3:1 correction ratio actually stabilize populations?

3. **Phase 2c — Mycelix red-team scenarios** (1 session, ~half day)
   - Extend `red_team.rs` with 5 attack vectors above
   - Each as an `AdversarialStrategy` variant with resilience metrics

4. **Phase 3 (optional) — DKG + cross-cluster cost** (1 session)
   - Only needed if calibrating throughput / committee size questions

## What NOT to do

- **Do not extract currency/economics into a new module** — they're already well-structured
- **Do not rewrite governance.rs** — it already has the right architecture, just uses scalar gating
- **Do not attempt full Holochain DHT simulation** — out of scope; use the existing relativistic_dht abstraction

## Open questions for the next session

1. Should the 8D SovereignProfile be shared with `mycelix-bridge-common` (via dependency) or duplicated? Sharing means the sim breaks when bridge-common changes; duplicating means drift risk. Recommendation: duplicate with a ~monthly manual sync.
2. Do we want per-tick telemetry output for governance-economics scenarios, or just end-of-run reports? Current setup is end-of-run.
3. Should Mycelix red-team scenarios be separate benchmarks or integrated into `run_sensitivity_analysis`?

## Session artifact

This document is the Phase 1 deliverable. Phases 2a-2c can be executed independently; no dependencies between them beyond the 8D profile (which 2b and 2c both use).

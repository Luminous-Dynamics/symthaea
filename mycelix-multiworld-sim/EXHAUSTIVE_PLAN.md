# Exhaustive Improvement Plan: Making Governance Matter

## The Root Cause

The policy comparison showed 0.8% CVS spread across 5 presets because
**5 of 6 policy knobs are defined but never read by the simulation.**

| Policy Knob | Defined? | Wired into tick loop? |
|-------------|----------|----------------------|
| birth_policy | YES | YES (2 uses) |
| project_strategy | YES | **NO** |
| resource_priority | YES | **NO** |
| trade_openness | YES | **NO** |
| defense_spending | YES | **NO** |
| exploration_investment | YES | **NO** |

The "policy comparison" tested ONE variable (birth rate) three ways.
Growth and Independence produced identical results because they both
use ProNatal — the only difference between them (project_strategy,
resource_priority, trade_openness, defense_spending) is invisible.

## Phase 1: Wire the Dead Knobs (4 hours)

### 1a. project_strategy → Project Selection
The AI governor currently uses `prioritize_projects()` with hardcoded
priority order. Wire `project_strategy` to change the priority:

- **SurvivalFirst**: RadiationShelter > Medical > ECLSS > Greenhouse
- **GrowthFirst**: HabitatExpansion > Greenhouse > WaterExtraction
- **ScienceFirst**: ExplorationVehicle > CommsArray > Laboratory
- **IndependenceFirst**: FabricationWorkshop > FissionReactor > LaunchPad

### 1b. resource_priority → Sector Allocation
The Cobb-Douglas economy assigns workers to sectors by skill. Wire
`resource_priority` to bias assignment:

- **Industrial**: +50% engineering/logistics, -25% art_culture/education
- **Biological**: +50% agriculture/medicine, -25% engineering
- **Knowledge**: +50% science/education, -25% logistics

### 1c. trade_openness → Inter-world Trade Volume
The trade system moves resources by surplus/deficit. Wire `trade_openness`
as a multiplier on trade volume:

- 0.0 (autarky): no inter-world trade
- 0.5: half normal trade
- 1.0: full trade (current default)

### 1d. defense_spending → Disaster Preparedness
Currently unused. Wire as a fraction of labor diverted from production
to disaster preparedness (reduces disaster severity):

- 0.0: no preparedness, full production
- 0.15: 15% labor diverted, disaster severity reduced 30%
- 0.30: 30% labor diverted, disaster severity reduced 50%

### 1e. exploration_investment → Exploration Probability
The exploration check uses a fixed base probability (3%). Wire
`exploration_investment` as a multiplier:

- 0.02: base 2% probability (low investment)
- 0.10: base 10% (5x more exploration)
- 0.20: base 20% (aggressive exploration program)

## Phase 2: Make Trust Weighting Bite (4 hours)

The ConsciousnessProfile is imported from bridge-common but not used
for governance decisions. The AI governor makes optimal choices regardless
of who's voting.

### 2a. Wire ConsciousnessProfile into agent governance
Each agent's governance participation should depend on their tier:
- Observer (score < 0.3): cannot vote on projects or policy
- Participant (0.3-0.4): can vote, weight 50%
- Citizen (0.4-0.6): full voting, can propose
- Steward (0.6-0.8): can override, constitutional authority
- Guardian (0.8+): emergency powers

Currently all agents contribute equally to "governance_stability."
With tier-weighted governance, a colony with many Observers has
WORSE governance than one with many Citizens — even at the same population.

### 2b. Consciousness education as a policy choice
Add a "consciousness_investment" policy knob (0.0-0.3) that diverts
labor to consciousness education (raising agent Phi). More investment
= faster consciousness growth = more Citizens/Stewards = better
governance = higher collective Phi.

This creates the fundamental trade-off: invest in consciousness
(long-term governance quality) or production (short-term survival).

### 2c. The A/B test
Run: trust_weighted_governance_enabled=true vs false.
If true produces higher Phi at Year 500 than false, consciousness
gating is empirically validated as a governance mechanism.

## Phase 3: What Would Sustain the Golden Age? (8 hours)

The simulation peaks at Year 75-100 (Phi ~0.4, CVS ~0.75) and
declines. What would sustain it?

### 3a. Social graph density as a managed variable
Currently the social graph exists (cohort.rs) but isn't wired.
Wire it: agents with more connections have higher community score
→ higher ConsciousnessProfile → better governance. Policy choice:
invest in "social infrastructure" (festivals, shared rituals, inter-
world exchanges) that increases connection density.

### 3b. Cultural memory as institutional knowledge
Currently cultural memories exist but don't affect behavior.
Wire: memories reduce future disaster severity (preparedness) and
increase governance stability (lessons learned). A civilization
that remembers its crises governs better than one that forgets.

### 3c. Inter-world consciousness synchronization
bridge-common has ConsciousnessSynchronizer with staleness decay.
Wire it: when worlds lose consciousness sync (blackout, latency),
collective Phi drops. When they resync (communication restored),
Phi recovers. This makes communication infrastructure a Phi driver.

### 3d. Dunbar-aware governance reform
Currently Dunbar transitions fire once and reduce stability by 0.15.
Instead: each Dunbar crossing should trigger a governance restructuring
that, if successful, INCREASES stability (the colony learned to
govern at the new scale). Failure rate depends on governance quality.

## Phase 4: The Definitive Test (2 hours)

After Phases 1-3, rerun the policy comparison:
- 5 presets × 5 seeds × 1000 years
- Compare CVS, Phi, population, milestones
- If spread > 5%, governance strategy matters
- If spread > 10%, trust weighting is transformative
- If spread < 2%, the Tainter decline is structural

## Phase 5: Connect to Production (4 hours)

### 5a. Extract validated parameters
From the 25-seed batch, extract:
- Optimal consciousness_investment fraction
- Optimal defense_spending for disaster resilience
- Minimum trade_openness for outer system viability
- Consciousness tier thresholds that maximize collective Phi

### 5b. Write to bridge-common
Update bridge-common's default thresholds with sim-validated values.
The production Mycelix governance uses parameters TESTED by the sim.

### 5c. Portal visualization
Map simulation output to portal domains:
- Governance → proposal/voting timeline
- Knowledge → exploration discovery feed
- Hearth → named character relationship graph
- Finance → TEND transaction flow

## Priority Summary

| Phase | Effort | Impact | What It Proves |
|-------|--------|--------|---------------|
| **1** | 4 hr | **CRITICAL** | Whether governance CAN matter |
| **2** | 4 hr | **HIGH** | Whether trust weighting specifically matters |
| **3** | 8 hr | MEDIUM | What sustains flourishing |
| 4 | 2 hr | HIGH | Definitive test |
| 5 | 4 hr | MEDIUM | Production integration |

Phase 1 is the single most important work remaining. If wiring 5 dead
knobs makes governance strategy produce >5% CVS spread, the simulation
becomes a genuine governance testbed. If it doesn't, the problem is
deeper than policy choices.

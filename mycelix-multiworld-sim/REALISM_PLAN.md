# Making It Real: What Separates Simulation From Game

## The Core Insight

A game rewards optimal play. Reality punishes hubris.

In the current simulation, projects complete on schedule, resources move
without loss, agents die silently, and the AI governor makes rational
decisions. These are game mechanics. Reality is messier:

- The reactor takes 36 months because a seal failed at month 18
- 12% of the food harvest spoils because the cold storage compressor died
- Dr. Watanabe's death leaves 3 apprentices who aren't ready to operate
  alone — the colony goes 8 months without surgery capability
- The governor chose the greenhouse over the radiation shelter because
  his daughter works in the Ag-Bay

## What Needs To Change (Ranked by Impact on Realism)

### 1. Project Variance and Failure (HIGH IMPACT, LOW EFFORT)

**Current**: Every project completes in exactly `duration` ticks.
**Reality**: Construction has variance. Cost overruns average 20-50% for
complex engineering (Flyvbjerg 2002). Space projects are worse — ISS was
300% over budget and 5 years late.

**Implementation**:
- Project duration: base ± 30% (gaussian)
- 5% chance per tick of "setback" (adds 1-3 months)
- 2% chance of "critical failure" (project abandoned, materials lost)
- New projects start at 60% efficiency, ramping to 100% over first 25%
  of duration (commissioning period)

### 2. Resource Spoilage and Waste (HIGH IMPACT, LOW EFFORT)

**Current**: Resources exist as numbers. Production adds, consumption
subtracts. No loss.
**Reality**: Food spoils. Water gets contaminated. Materials degrade.
Air filters clog. Every real system has entropy working against it.

**Implementation**:
- Food: 2-5% spoilage per tick (varies with storage quality)
- Water: 0.5% contamination loss per tick (ECLSS imperfect)
- Materials: 1% degradation per tick (radiation, thermal cycling)
- Energy: 10-20% transmission/storage loss (no perfect grid)

NASA ISS data: water recovery 90% (not 100%). Air recycling loses
~1% O2 per month to leaks. These are REAL numbers.

### 3. Skill Gaps From Death (MEDIUM IMPACT, MEDIUM EFFORT)

**Current**: Agent dies → population -1. No other consequence.
**Reality**: When the colony's only surgeon dies, surgery capability
goes to zero until someone is trained (years).

**Implementation**:
- On death: check if deceased was the last skilled agent in any
  critical system (from CriticalSystemCoverage)
- If yes: generate a "SKILL GAP" crisis event
- The gap persists until another agent reaches skill > 0.3 in that
  sector (via education system)
- During the gap: that system operates at degraded capacity

### 4. Decision-Making Friction (MEDIUM IMPACT, HIGH EFFORT)

**Current**: AI governor selects projects instantly and optimally.
**Reality**: Governance involves debate, compromise, information delay.
A decision that takes 1 tick in the sim takes 6 months of political
process in reality.

**Implementation**:
- Project proposals require a "deliberation period" (3-6 ticks)
- During deliberation, faction leaders influence the outcome
- Information asymmetry: the governor doesn't know exact resource
  stocks (±10% measurement error)
- Occasionally the "wrong" project is selected (governance quality
  < 0.8 → 10% chance of suboptimal choice)

### 5. Grief as System Shock (MEDIUM IMPACT, LOW EFFORT)

**Current**: Named characters appear in narrative but their death
has no mechanical effect beyond trauma increment.
**Reality**: In a colony of 200, everyone knows everyone. A death
ripples through social networks, disrupting work schedules, creating
caretaking burden, and temporarily reducing productivity.

**Implementation**:
- Death of any agent in colony < 500 → 1 tick productivity loss
  (funeral, grief processing)
- Death of a notable agent → 3 ticks productivity loss + specific
  narrative event
- Children losing a parent → permanent trauma modifier (ACE score)
- Partner death → 6-tick grief period with reduced work capacity

### 6. Environmental Entropy (LOW IMPACT, LOW EFFORT)

**Current**: Infrastructure degrades from the maintenance trap but
the environment is static between disasters.
**Reality**: Dust accumulates on solar panels (Mars: 0.3% loss/sol).
Seals degrade continuously. Thermal cycling weakens joints. Radiation
embrittles polymers. The colony fights entropy every day.

**Implementation**:
- Solar panel degradation: 0.1%/tick without cleaning (Mars dust)
- Seal integrity: decays 0.05%/tick (requires regular maintenance)
- Equipment MTBF: tracked per habitat module, not globally
- Each module accumulates an "entropy clock" that demands maintenance

### 7. Cultural Memory (LOW IMPACT, MEDIUM EFFORT)

**Current**: Past events don't affect future responses (except
through affect momentum).
**Reality**: "Remember the Titan Freeze of 307" shapes behavior
for generations. Cultural memory is how civilizations learn.

**Implementation**:
- Major events (severity >= 3) create "cultural memories"
- Each memory has a "lesson" that modifies future response
  (e.g., "Titan Freeze" → +20% cold preparedness, -10% risk tolerance)
- Memories decay over generations (half-life ~100 years)
- Shared memories between worlds require communication (latency-gated)

## Priority Order

| # | Change | Impact | Effort | Realism Gain |
|---|--------|--------|--------|-------------|
| 1 | Project variance | HIGH | 1 hour | Removes determinism |
| 2 | Resource spoilage | HIGH | 1 hour | Adds entropy |
| 3 | Skill gaps | MEDIUM | 2 hours | Death has consequences |
| 5 | Grief shock | MEDIUM | 1 hour | Community is real |
| 6 | Environmental entropy | LOW | 1 hour | Background decay |
| 4 | Decision friction | MEDIUM | 4 hours | Governance is messy |
| 7 | Cultural memory | LOW | 3 hours | Civilizations learn |

Items 1-2 are the highest ROI: 2 hours of work that removes the two
most "video-gamey" aspects of the simulation (deterministic construction
and lossless economics).

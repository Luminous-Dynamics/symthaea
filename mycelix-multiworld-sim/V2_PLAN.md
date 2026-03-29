# V2 Engine Plan: Evidence-Based

Based on deep exploration of V1 output (V9 chronicle, epoch snapshots, event
counts, performance profiling). Not theoretical — every item traces to observed
simulation behavior.

## What V1 Actually Produces (Honest Assessment)

### The Numbers
- 27,391 LOC across 43 files
- 264 tests
- 15 commits this session (~8,100 LOC added)
- 1000yr runtime: ~15 min (after O(N²) fix)
- Batch: 2/5 seeds complete, both survived, CVS 0.737-0.755

### The Narrative (Read Honestly)
The V9 chronicle (seed 99, 1000 years) contains:
- 21 "social fabric frayed" events (all same template)
- 13 population milestones ("reached X population")
- 7 tech milestones
- 1 named character (Fatima Petrov, Gen 3, Year 94)
- 0 independence movements
- 0 project completions
- 0 exploration events
- 0 Dunbar transitions

**Diagnosis**: The narrative is 80% population crash notifications. The new
systems (projects, exploration, independence, Dunbar) push CivEvents but NOT
NarrativeEvents. The chronicle never sees them because CivEvent and
NarrativeEvent are separate, unconnected systems.

### The Affect Problem
Every world starts with negative conatus and stays negative for decades:
- Earth Year 50: conatus -0.475, load 0.965
- Moon Year 50: conatus -0.445, load 1.000
- Mars Year 100: conatus -0.475, load 0.946

Colonists are **miserable from birth to death** in most worlds. Only after
major tech milestones (year ~200) does conatus flip positive. This might be
realistic (colonization IS hard) but it means the simulation is modeling
1000 years of suffering punctuated by brief periods of relief. The Spinozist
affect math is working — the question is whether the inputs are too harsh.

### The Allostatic Load Problem
Earth's allostatic load is 0.965 at year 50. This means 96.5% of maximum
stress. But Earth has 3,125 people, full self-sufficiency, and no active
disasters. Why is load so high?

Likely cause: the load computation includes resource pressure, population
pressure, and maintenance deficit — all of which fire even in a "healthy"
colony because thresholds are too tight.

### The Phi Problem
Collective Phi ranges 0.04-0.25. It never crosses 0.3. This suggests the
consciousness metric isn't sensitive to colony state — it's dominated by
structural factors (number of agents, connection density) rather than
emergent integration.

## What Actually Needs Fixing (Priority Order)

### P0: Narrative Event Unification (1 hour)
**Problem**: CivEvents and NarrativeEvents are separate systems. 28 CivEvent
types go nowhere. The chronicle only sees population crashes and milestones.
**Fix**: When a CivEvent has a description containing key markers (PROJECT
COMPLETE, INDEPENDENCE, EXPLORATION, DUNBAR), automatically create a
corresponding NarrativeEvent in the narrative tick. Or better: replace
NarrativeEvent with CivEvent entirely and add severity/affect fields to
CivEvent.
**Impact**: The chronicle would go from 41 events (80% identical) to 100+
diverse events including project completions, independence movements, and
explorations.

### P1: Allostatic Load Calibration (2 hours)
**Problem**: Earth at year 50 has load 0.965 with no active crisis. Every
world starts near maximum stress.
**Fix**: Audit the load computation in needs.rs. The base load for a colony
with adequate resources, functioning ECLSS, and no disasters should be ~0.3,
not ~0.9. The current formula likely sums too many small contributors
(maintenance deficit, population pressure, resource fraction) without a
"baseline comfort" offset.
**Impact**: Positive conatus from the start for healthy colonies. Disasters
would spike load from 0.3 to 0.8 instead of from 0.96 to 1.0 (no headroom
currently).

### P2: Dead System Wiring (3 hours)
**Problem**: Projects, explorations, independence movements, and Dunbar
transitions fire as CivEvents but don't appear in narratives, don't affect
world state meaningfully, and don't interact with each other.
**Fix**:
- Project completions should appear in narrative as named events
- Independence movements should affect diplomatic relations (-0.2 with Earth)
- Explorations should trigger narrative events with named characters
- Dunbar transitions should affect governance stability
**Impact**: The sim becomes a 50+ system simulation that actually USES its
50 systems, instead of 10 core systems + 40 dead-letter CivEvent generators.

### P3: Response Diversity (2 hours)
**Problem**: 21/41 narrative events use the template "Social fabric frayed —
faction tensions rose." Every population crash has the same response.
**Fix**: Expand response/outcome selection based on more affect dimensions,
colony size, tech level, and whether the crisis is the 1st, 3rd, or 10th.
First crisis: "shock and disbelief." Third crisis: "grim determination."
Tenth crisis: "numbing acceptance."
**Impact**: The chronicle reads like a history instead of a log file.

### P4: Hybrid V2 Integration (4 hours)
**Problem**: The V2 Gillespie engine is proven but disconnected from V1.
**Fix**: Route V1's disaster probabilities through the V2 event queue for
scheduling. Keep V1's resolution logic (which depends on world state).
This eliminates 40 bernoulli rolls per tick without changing disaster effects.
Also schedule project completions as V2 events (they have known duration).
**Impact**: 10-20% additional speedup on top of the O(N²) fix. More
importantly, project completions become exact (no "check every tick").

### P5: Cohort Manager (8 hours)
**Problem**: 70K individual agents, 99% interchangeable. Each agent has ~30
fields updated every tick. Most don't change meaningfully.
**Fix**: Group agents into cohorts (age_band × sector × world × health_band).
Each cohort is a statistical entity. Only "notables" (leaders, specialists,
named characters) are tracked individually. Promotion/demotion between cohort
and individual tracking.
**Impact**: 10-100x speedup for demographic calculations. Makes 50-seed
batches finish in 1-2 hours instead of 10-12.

### P6: Social Network for Notables (4 hours)
**Problem**: Social dynamics computed from aggregates. No relationships.
**Fix**: 50-200 notable agents with petgraph social graph. Information,
grief, and influence propagate through edges. Death of a notable's connection
generates specific grief narrative. Faction formation from graph clustering.
**Impact**: Emergent stories. "Dr. Watanabe's death was felt most keenly by
her three apprentices."

### P7: Supply Chain Event Integration (2 hours)
**Problem**: Supply chain graph exists but doesn't generate events. Disasters
don't hit specific Earth regions.
**Fix**: When a disaster affects an Earth region, degrade the corresponding
supply chain node. Generate narrative events when supply chains break or
reroute. Colony supply shortages trigger specific CivEvents.
**Impact**: Cascade failures become visible in the narrative. "The Pacific
Rim mega-quake severed semiconductor supply — the Mars ECLSS filter shipment
was delayed 14 months."

## Implementation Order

```
Session 1 (now): P0 + P1 + P3   — Fix what's broken (6 hours)
Session 2:       P2 + P4 + P7   — Wire dead systems (9 hours)
Session 3:       P5 + P6         — Architecture shift (12 hours)
```

P0 is the single highest-impact fix. The sim has 50 systems, and 40 of them
are invisible to the user. Making CivEvents flow into the chronicle would
transform the output overnight.

P1 is the second highest — fixing allostatic load so colonists aren't 96%
stressed on Day 1 would make the affect dynamics actually meaningful.

P5+P6 (cohort + social graph) are the V2 architecture. They should come AFTER
the V1 output is honest and useful. There's no point having a fast sim that
produces the wrong narrative.

# Multiworld Sim: Depth Audit

Not "what's missing" but "what's shallow."

## The Core Question

Every system should produce EMERGENT BEHAVIOR that surprises the designer.
If a system always produces predictable outcomes, it's a formula, not a simulation.

## System-by-System Depth Assessment

### 1. PHYSICS: Grade B+
**What works**: Radiation, orbital mechanics, disasters — all grounded in cited data.
**What's shallow**: Disasters are independent rolls. Real disasters interact:
a mega-quake during a dust storm during a Carrington event is qualitatively
different from three separate events. The cascade multiplier (3+ = amplified)
is a step in the right direction but doesn't model HOW disasters compound.

**Depth improvement**: Disaster interaction matrix. Some disasters make others
MORE likely (quake → infrastructure damage → ECLSS failure chain). Others
are independent. Model the conditional probabilities, not just co-occurrence
amplification.

### 2. BIOLOGY: Grade B-
**What works**: Gompertz mortality, inbreeding, cumulative radiation.
**What's shallow**:
- All agents of the same age/health have identical mortality. No individual
  variation in genetic fitness — no sickle cell carriers, no BRCA mutations,
  no lactose tolerance evolution.
- Disease is SIR + pathogen pressure scalar. Real epidemics have R0, incubation
  periods, mortality rates, mutation, and immunity. A single pathogen model
  doesn't capture the difference between influenza and tuberculosis.
- Nutrition is "food" — not protein/carbs/fat/vitamins. Scurvy killed more
  sailors than combat. A colony growing only wheat will have B12 deficiency.

**Depth improvement**:
- Individual genetic fitness modifier (±20% from population mean, heritable)
- 3 pathogen archetypes: fast (flu-like, R0=3, 1-week), slow (TB-like, R0=1.5,
  6-month), and novel (emerged from sealed habitat, variable R0)
- Macro-nutrient tracking: protein, carbs, fat, micronutrients. Deficiency
  diseases emerge when crop diversity is too low.

### 3. PSYCHOLOGY: Grade B
**What works**: Spinozist affects with nonlinear dynamics are genuinely novel.
**What's shallow**:
- Affects are COMPUTED each tick from state, not REMEMBERED. Real emotions
  have momentum — grief doesn't vanish when the cause is removed. Joy from
  a success persists for weeks. Resentment builds over years.
- No individual relationships between agents. In a colony of 200, EVERYONE
  knows everyone. The death of Agent #47 should affect Agent #12 differently
  depending on whether they were partners, colleagues, rivals, or strangers.
- No personality variation. All agents respond identically to the same
  conditions. Real humans have different attachment styles, risk tolerances,
  and coping mechanisms.

**Depth improvement**:
- Affect EMA: each dimension has momentum (α=0.1 blend with previous tick)
- Relationship graph: agents have 5-15 significant relationships with varying
  valence (partner, child, friend, rival, mentor). Death/departure of a
  connected agent generates specific grief proportional to relationship strength.
- Personality vector: 3D (openness, agreeableness, neuroticism) set at birth
  with heritable component. Modulates affect sensitivity.

### 4. ECONOMICS: Grade C+
**What works**: Cobb-Douglas is a reasonable macro model. Supply chain DAG is new.
**What's shallow**:
- The economy has no PRICES. Resources move by surplus/deficit arithmetic.
  Real economies have price signals that coordinate behavior without central
  planning. The TEND system is supposed to do this but isn't wired in.
- No capital goods vs consumer goods distinction. A hammer and a sandwich
  are both "materials." Building a reactor (capital) and feeding a person
  (consumption) draw from the same pool.
- Trade is mechanistic — resources flow from surplus to deficit based on
  math. Real trade involves NEGOTIATION, where each party tries to maximize
  their benefit. Mars might refuse to trade iron to Europa if Europa won't
  send water in return.
- No debt or credit. Colonies can't borrow against future production.
  This means they can't invest in infrastructure beyond current resources.

**Depth improvement**:
- TEND price signals: each resource has a TEND price that reflects scarcity.
  High demand + low supply = high price = incentive for production.
- Capital vs consumption: split "materials" into "capital_goods" (tools,
  machines, infrastructure components) and "consumables" (food, fuel, filters).
  Capital goods persist and amplify production; consumables are used up.
- Negotiated trade: worlds propose trade terms based on diplomatic relations
  and relative scarcity. Hostile worlds demand unfavorable terms.
- Credit system: worlds can issue bonds (promises of future production)
  to fund infrastructure investment. Default risk based on trust level.

### 5. GOVERNANCE: Grade C
**What works**: Trust weighting, anti-tyranny invariants, Dunbar transitions.
**What's shallow**:
- No LEADERS. Governance is a statistical process, not a human one. Real
  colonies are shaped by specific individuals — a visionary founder, a
  corrupt administrator, a charismatic rebel.
- No POLICY CHOICES. The governance system tracks stability and amendments
  but doesn't model actual decisions: birth policy, immigration policy,
  resource allocation priorities, military/defense spending.
- No INFORMATION ASYMMETRY. Leaders know everything agents know. Real
  governance failures come from leaders having wrong information, or
  deliberately withholding information from the public.
- No CORRUPTION. In a colony where the ECLSS operator has life-and-death
  power, corruption is almost inevitable. Who watches the watchers?

**Depth improvement**:
- Leader agents: each world has 1-3 "leader" agents with disproportionate
  influence on governance decisions. Leader death = succession crisis.
- Policy menu: each world chooses from discrete policy options (pro-natal
  vs population control, open immigration vs restrictive, military vs
  civilian priority). Policies have delayed effects (pro-natal → baby
  boom in 5 years → labor shortage NOW from childcare).
- Information fog: agents and leaders have imperfect knowledge of resource
  stocks, other worlds' status, and disaster severity. Panic and complacency
  both emerge from information errors.
- Corruption index: grows when trust is low and inequality is high. Corrupt
  governance diverts resources from public goods to elite consumption.

### 6. NARRATIVE: Grade C+
**What works**: Milestone events and population crash detection with affect responses.
**What's shallow**:
- Narrative is REACTIVE — it reports what happened. It doesn't SHAPE what
  happens next. In reality, a story about a disaster changes how people
  respond to the next disaster. "Remember the Titan Freeze of '307" is
  a cultural memory that affects behavior for generations.
- No named characters. "Europa Station lost 18% of its population" is
  statistics. "Dr. Kenji Watanabe, Europa's only surgeon, died in the
  third tidal quake" is a story.
- No competing narratives. The colony has one narrative identity. Real
  communities always have competing stories about who they are and what
  they should do. The tension between narratives IS politics.

**Depth improvement**:
- Named character generation: when a pivotal event occurs, generate a
  named agent (from the existing agent pool) and attach them to the event.
  Track these named characters across events for continuity.
- Narrative memory: past events shape future responses. A colony that
  survived a freeze develops "freeze protocols" that reduce future freeze
  damage. A colony that lost a leader develops succession planning.
- Competing narratives: 2-3 narrative factions per world (conservative/
  progressive/radical) that interpret events differently and advocate
  different policies.

### 7. CULTURE: Grade D
**What works**: Harmony weights drift. Cultural distance is computed.
**What's shallow**:
- Culture is 8 numbers that drift randomly. Real culture produces ARTIFACTS
  — stories, songs, rituals, technologies, architectures, languages. These
  artifacts persist beyond the individuals who created them and shape the
  thoughts of future generations.
- No language. The most fundamental cultural artifact. Without modeling
  language drift, we can't capture how isolated populations literally
  lose the ability to communicate after 300-500 years.
- No religion or ideology. The most powerful motivational force in human
  history. A colony's religion (or secular equivalent) determines sacrifice
  tolerance, cooperation patterns, and meaning-making capacity.

**Depth improvement**:
- Cultural artifact accumulation: each world accumulates "cultural capital"
  (art, literature, music, architecture) that persists and strengthens
  identity. Cultural capital can be lost (library fire, censorship) or
  shared (cultural exchange via trade routes).
- Language divergence index: accumulates with isolation, proportional to
  1/sqrt(population). When divergence > 0.5, communication becomes impaired.
  When > 0.8, requires translation. When > 1.0, mutual unintelligibility.
- Ideology system: each world has a dominant ideology (from narrative
  identity) plus 1-2 minority ideologies. Ideology affects policy
  preferences, cooperation patterns, and crisis response.

### 8. INTER-WORLD RELATIONS: Grade D+
**What works**: Diplomatic relation scores exist. Trade exists.
**What's shallow**:
- No CONFLICT. Worlds never compete, embargo, sanction, or fight. In
  reality, Mars surpassing Earth's population would trigger Earth's most
  fundamental political instinct: fear of being surpassed.
- No ALLIANCES. Worlds don't form blocs. Europa + Titan vs Earth + Mars
  is a plausible political configuration that would reshape everything.
- No INDEPENDENCE MOVEMENTS. Mars with 11,000 people and Earth with 10,000
  — Mars would demand sovereignty within a generation.
- No INFORMATION WARFARE. No propaganda, no cultural imperialism, no
  soft power. Communication latency makes information control a weapon.

**Depth improvement**:
- Conflict model: when diplomatic relations drop below -0.3, worlds begin
  trade restrictions. Below -0.5, full embargo. Below -0.7, active sabotage.
- Alliance formation: worlds with relations > 0.6 form alliances that share
  defense, trade preferentially, and coordinate governance.
- Independence threshold: when a colony's population exceeds 5,000 AND
  self-sufficiency exceeds 0.8 AND cultural distance from parent > 0.3,
  independence movement probability rises. Peaceful or violent depending
  on diplomatic relations with parent.
- Information warfare: worlds with communication capability can broadcast
  propaganda that shifts other worlds' cultural weights toward their own.
  Effectiveness scales with tech level and inversely with distance.

### 9. SUPPLY CHAIN: Grade C
**What works**: Graph topology with petgraph. Disruption propagation.
**What's shallow**:
- Routes are static. Real supply chains are constantly being built, rerouted,
  and abandoned. New trade routes should emerge from economic incentives.
- No stockpiling / just-in-time tension. Some colonies maintain huge
  reserves (expensive but resilient); others run lean (efficient but fragile).
  This is a strategic choice that should affect survival.
- No middlemen / logistics agents. Resources don't teleport — someone has
  to operate the ship, manage the warehouse, handle customs.

**Depth improvement**:
- Dynamic route creation: new routes form when trade demand exceeds capacity
  on existing routes. Routes decay when utilization drops below 20%.
- Strategic reserves: each world chooses a stockpile level (0-24 months).
  Higher reserves = more resilient to disruption but more expensive to maintain.
  This is the buffer stock problem made explicit.
- Logistics labor: supply chain operation requires dedicated workers.
  The logistics sector (sector 7) should directly affect supply chain capacity.

### 10. TECHNOLOGY: Grade B-
**What works**: 16 milestones with expert-calibrated probabilities. Fission delivery.
**What's shallow**:
- Tech is a linear tree. Real technology is a NETWORK with multiple paths.
  A civilization that invests in biology produces different capabilities
  than one that invests in physics.
- No FAILED experiments. Real science has dead ends, retractions, and
  paradigm shifts. Our milestones always succeed once probability hits.
- No technology DIFFUSION between worlds. Knowledge transfers as a scalar,
  not as specific capabilities. Mars discovering a new alloy should specifically
  help Titan's embrittlement problem, not just raise a general tech level.
- No DUAL-USE problem. Nuclear fission enables both power plants and weapons.
  Genetic engineering enables both disease cure and bioweapons. The dual-use
  nature of technology creates governance dilemmas we don't model.

**Depth improvement**:
- Branching tech paradigms: "Biology-first" vs "Physics-first" vs "Information-
  first" paths with different milestone unlocks and vulnerabilities.
- Failed experiments: each tech roll that fails still costs resources and
  time. Repeated failures in the same area create "dead end" markers that
  redirect research effort.
- Specific knowledge transfer: when a world achieves a milestone, connected
  worlds receive a SPECIFIC capability boost (Titan gets Mars's alloy data),
  not a general tech level increase.

---

## Priority Matrix (Impact × Tractability)

### Tier 1: Would transform the simulation (implement next)
1. **Affect momentum** — emotions persist, creating realistic grief/joy/resentment
2. **Named characters** — transforms statistics into stories
3. **Independence movements** — Mars sovereignty is the defining political event
4. **Policy choices** — governance becomes DECISIONS, not statistics
5. **Disaster interaction matrix** — compound failures instead of independent rolls

### Tier 2: Would significantly deepen (implement after Tier 1)
6. Relationship graph between agents (friendship/rivalry/kinship)
7. Conflict/embargo/sanction model for inter-world politics
8. Capital vs consumption goods distinction
9. TEND price signals
10. Cultural artifact accumulation

### Tier 3: Research-grade additions (long-term)
11. Individual genetic fitness variation
12. Multi-pathogen disease model
13. Language divergence with translation
14. Branching tech paradigms
15. Information fog / asymmetric knowledge

### Tier 4: Aspirational (would make this publishable)
16. Named character tracking across events (protagonist arcs)
17. Competing ideologies per world
18. Strategic reserves as player/AI choice
19. Dual-use technology dilemmas
20. Corruption model

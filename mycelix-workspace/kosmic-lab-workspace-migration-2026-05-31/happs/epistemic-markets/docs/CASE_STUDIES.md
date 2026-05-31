# Case Studies

*Learning from Living Examples*

---

> "In theory, there is no difference between theory and practice. In practice, there is."
> — Yogi Berra

> "Case studies are how we bridge that gap."
> — Us

---

## Introduction

This document presents detailed case studies of Epistemic Markets in action. Each case illustrates how the system's mechanisms work together in realistic scenarios.

These are composite cases based on expected use patterns. As the system operates, we'll update with real examples.

---

## Case Study 1: The Climate Prediction

*Showing: Multi-dimensional stakes, expert weighting, long-term markets*

### Setup

**The Question**: "Will global average temperature in 2030 exceed 1.5°C above pre-industrial baseline?"

**Market Classification**: E4-N3-M2
- E4: Measurable (scientific data)
- N3: Universal (objective measurement)
- M2: Persistent (multi-year relevance)

### The Participants

**Dr. Sarah Chen** - Climate scientist
- MATL: 0.89 (0.94 in climate domain)
- 15 years of climate research
- Previously calibrated: 82% accuracy on climate predictions

**Marcus Johnson** - Hedge fund analyst
- MATL: 0.76 (0.68 in climate domain)
- Analyzes climate risk for investments
- Previously calibrated: 74% accuracy overall

**Rosa Gonzalez** - High school science teacher
- MATL: 0.45 (0.38 in climate domain)
- Teaching climate science for 8 years
- New to prediction markets

**Climate Skeptic Anonymous** - Pseudonymous
- MATL: 0.52 (0.41 in climate domain)
- History of contrarian predictions
- 63% accuracy on climate topics

### The Predictions

**Dr. Chen - Yes (78% confidence)**
```markdown
Stake: 500 HAM + 5% climate reputation
Reasoning: "Current trajectory based on IPCC AR6 models, observed acceleration
in Arctic ice loss, and continued emissions growth suggest 75-85% probability
of exceeding 1.5°C by 2030. Key uncertainty is potential for rapid policy
change or natural carbon sink enhancement, which I weight at 15% combined
probability of sufficient impact."

Sources:
- IPCC AR6 Working Group I (2021)
- NASA GISS temperature record
- Personal research on feedback loops
```

**Marcus - Yes (65% confidence)**
```markdown
Stake: 1000 HAM + Commitment: "If wrong, I'll present climate science basics
to my trading floor"
Reasoning: "Insurance market pricing implies ~70% probability of exceeding
1.5°C thresholds. Reinsurance premiums for climate-linked events show
3.2% annual increase, consistent with accelerating trend. Financial markets
are efficient aggregators of distributed information."

Sources:
- Swiss Re climate risk reports
- Munich Re NatCat data
- Internal proprietary models
```

**Rosa - Yes (55% confidence)**
```markdown
Stake: 50 HAM (monetary only, low reputation stake as newcomer)
Reasoning: "Based on what I teach my students: the trend is clear, the
physics is established, and we haven't seen meaningful emissions reduction.
I'm less certain than experts because I may be missing nuances, but the
direction seems obvious."

Note: I'm using this as a learning experience.
```

**Skeptic - No (62% confidence)**
```markdown
Stake: 300 HAM + 8% climate reputation
Reasoning: "Models have consistently over-predicted warming. Natural
variability is underweighted. I predict a La Niña dominant period
through 2028 that will suppress temperatures. The 1.5°C threshold
is political, not scientific, and measurement methodology will be
disputed."

Sources:
- UAH satellite temperature record
- Historical model vs. actual comparisons
- Solar cycle analysis
```

### Market Dynamics

**Initial Aggregate**: 68% Yes (weighted by stake and MATL)

**Week 1**: Question market spawned this from high curiosity signals
- 47 curiosity signals before spawning
- Early predictors rewarded for surfacing the question

**Month 3**: New IPCC report released
- Dr. Chen updates to 82% confidence (no additional stake)
- Two new climate scientists join at 75% and 79%
- Aggregate moves to 74% Yes

**Year 2**: Unusual El Niño pattern
- Skeptic updates to 45% confidence No (essentially withdrawing claim)
- Marcus updates to 72% confidence
- Rosa, watching and learning, maintains position

### Resolution (2031)

**Outcome**: Global average exceeded 1.5°C by 0.08°C

**Oracle Process**:
1. Automated data feed: NASA GISS, NOAA, Copernicus all confirm
2. Resolution classification: Cryptographic (E3) - clear threshold crossed
3. Oracle vote: 12 oracles, 11 vote "Yes," 1 abstains
4. Consensus: Achieved with 0.94 weighted agreement

### Rewards and Outcomes

| Participant | Prediction | Confidence | Outcome |
|-------------|------------|------------|---------|
| Dr. Chen | Yes | 78% | Correct, well-calibrated |
| Marcus | Yes | 65% | Correct, slightly under-confident |
| Rosa | Yes | 55% | Correct, under-confident |
| Skeptic | No | 62%→45% | Wrong, lost stake |

**Dr. Chen's Rewards**:
- Accuracy reward: 412 HAM (share of losing stakes)
- Calibration bonus: +8% (78% confidence, correct = good calibration)
- Reasoning bonus: +35 HAM (highly rated reasoning)
- MATL climate domain: +0.03 (now 0.97)

**Rosa's Learning**:
- Small accuracy reward: 38 HAM
- Calibration feedback: "Your 55% confidence was under-confident for a clear trend. Consider: do you know more than you credit yourself?"
- Mentorship note: Matched with Dr. Chen for future climate predictions

**Skeptic's Consequences**:
- Lost 300 HAM stake
- Climate MATL: -0.12 (now 0.29 in climate)
- System note: "Your contrarian position had value for diversity, but reasoning didn't hold up to scrutiny"

### Wisdom Seeds

**From Dr. Chen**:
> "For future climate predictions: watch for threshold effects. The 1.5°C target is somewhat arbitrary, but the physical responses (ice sheet dynamics, permafrost feedback) have real thresholds. Don't conflate political targets with physical tipping points."

**From Skeptic**:
> "I was wrong. My error was weighting historical model-vs-actual comparisons too heavily without accounting for model improvements. For the record: I still think measurement methodology deserves scrutiny, but the warming is real."

### Case Lessons

1. **Expert weighting works**: Dr. Chen's high domain MATL appropriately increased her influence
2. **Commitment stakes add accountability**: Marcus's public commitment created social value even in winning
3. **Newcomers can learn safely**: Rosa gained experience without catastrophic loss
4. **Updating is honored**: Skeptic's willingness to update, even while losing, preserved some reputation
5. **Long-term markets require patience**: This market ran for years, requiring sustained engagement

---

## Case Study 2: The Tech Product Launch

*Showing: Question markets, cross-hApp integration, time pressure*

### Setup

**The Origin**: Question Market proposal

"Synapse AI, a stealth startup, is rumored to announce something major at CES 2025. What will it be?"

This question market accumulated 2,400 HAM in value shares before spawning multiple prediction markets.

### Spawned Markets

From the question market, the community created:

1. **"Synapse AI will announce consumer hardware at CES 2025"** (E2-N2-M1)
2. **"Synapse AI announcement will include AR/VR technology"** (E2-N2-M1)
3. **"Synapse AI will announce partnership with major tech company"** (E2-N2-M1)
4. **"Synapse AI stock (if public) will rise >20% in week after announcement"** (E3-N3-M1)

### The Participants

**TechInsider** - Industry analyst (pseudonymous)
- MATL: 0.83 (0.91 in tech domain)
- History of accurate tech predictions
- Known for insider industry knowledge

**QuantTrader** - Algorithmic trader
- MATL: 0.79 (0.71 in tech domain)
- Uses NLP on patent filings and job postings
- Data-driven approach

**Jane Chen** - Synapse AI employee (disclosed)
- MATL: 0.67 (0.55 in tech domain)
- Conflict of interest disclosed
- Limited to observation + wisdom seeds

**CasualWatcher** - Tech enthusiast
- MATL: 0.34 (new to tech predictions)
- Follows tech news
- First major prediction

### Cross-hApp Integration

**From Knowledge hApp**:
- Patent filings tagged to Synapse AI: 47 in AR/VR category
- Job postings analyzed: 12 hardware engineers hired in 6 months
- News sentiment: 0.72 positive, 0.15 speculative

**From Governance hApp**:
- Meta-market proposed: "Should we allow markets on unreleased products?"
- Community vote: 78% yes with privacy restrictions

**From Identity hApp**:
- Jane Chen's Synapse affiliation verified
- Appropriate trading restrictions applied automatically

### The Predictions

**TechInsider on Hardware** - Yes (85%)
```markdown
Stake: 800 HAM + 10% tech reputation
Reasoning: "Multiple sources confirm hardware team scaling. Component
supply chain analysis shows orders consistent with consumer device
production timeline. CES is historically their announcement venue.
The only uncertainty is whether 'consumer' vs 'enterprise' - I lean
consumer based on marketing team hires."
```

**QuantTrader on AR/VR** - Yes (72%)
```markdown
Stake: 600 HAM
Reasoning: "Patent analysis shows 47 AR/VR filings in past 18 months,
compared to 8 in previous 18 months. Key personnel moves from Meta's
Reality Labs. Statistical model gives 72% ± 8% probability based on
historical correlation between patent activity and announcements."
```

**CasualWatcher on Partnership** - Yes (60%)
```markdown
Stake: 100 HAM
Reasoning: "Saw a photo on LinkedIn of Synapse CEO meeting with Apple
executives. Might be nothing, but the timing is suspicious. Just a
hunch backed by weak evidence."
```

**Jane Chen** - (Observation only)
```markdown
Wisdom Seed: "I can't trade, but I can share general industry context:
hardware announcements at CES require 6+ months of supply chain
preparation. Whatever they announce, it's been in motion for a while.
Look at hiring patterns from 18 months ago, not 6 months."
```

### Resolution

**CES 2025 Announcement**: Synapse AI announced AR glasses with Apple partnership for spatial computing integration.

**Resolution Process**:
1. Public press release = E2 (testimonial, official source)
2. Multiple news outlets confirmed
3. Oracle vote: 8 oracles, unanimous confirmation
4. All three Yes predictions confirmed correct

### Outcomes

| Market | Prediction | Result | Calibration |
|--------|------------|--------|-------------|
| Hardware | 85% Yes | Correct | Well-calibrated |
| AR/VR | 72% Yes | Correct | Slightly under-confident |
| Partnership | 60% Yes | Correct | Under-confident (lucky?) |

**Special Recognition**:
- TechInsider: +0.04 MATL (expert prediction, excellent reasoning)
- QuantTrader: +0.02 MATL (good methodology)
- CasualWatcher: +0.01 MATL (correct but reasoning was weak)

**Jane Chen's Wisdom Contribution**:
- Wisdom seed upvoted 234 times
- +0.05 to Legacy score (wisdom without trading)
- Cited in 3 subsequent tech predictions

### Question Market Rewards

The original question proposer and early value share holders:

| Role | Reward |
|------|--------|
| Question proposer | 180 HAM + "Question Originator" badge |
| Top 10 early supporters | Share of 400 HAM pool |
| Question refiner (improved wording) | 50 HAM |

### Case Lessons

1. **Question markets surface valuable predictions**: The community identified a worthy question
2. **Cross-hApp data enriches predictions**: Knowledge graph data improved prediction quality
3. **Conflict disclosure works**: Jane couldn't trade but contributed wisdom
4. **Weak reasoning is noted**: CasualWatcher was right but didn't earn full calibration credit
5. **Speed matters**: Time-bounded markets create healthy urgency

---

## Case Study 3: The Community Decision

*Showing: Private community markets, futarchy elements, local governance*

### Setup

**Community**: Detroit Urban Agriculture Collective (DUAC)
- 200 active members
- Manages 15 urban farm plots
- Democratic decision-making history

**The Decision**: Whether to expand to a new site (old factory lot)

**Why Epistemic Markets**: Traditional voting didn't capture:
- Uncertainty about costs
- Distributed knowledge about soil quality
- Range of confidence levels
- Conditional preferences

### Market Setup

DUAC created a private community instance with markets:

1. **"Will total remediation cost exceed $50,000?"** (E4-N1-M2)
2. **"Will first harvest be possible within 12 months of starting?"** (E4-N1-M2)
3. **"If we expand, will active membership increase by >20% within 2 years?"** (E1-N1-M2)
4. **"Should DUAC acquire the old factory lot?"** (Meta-market for decision)

### Participants

**Community Member Profiles**:

| Name | Role | Relevant Expertise |
|------|------|-------------------|
| Marcus | Director | Overall operations |
| Tanya | Soil specialist | Environmental science degree |
| Jerome | Finance lead | Budgeting, fundraising |
| Elena | Outreach coordinator | Membership trends |
| 40+ other members | Various | Distributed local knowledge |

### The Predictions

**Market 1: Remediation Cost >$50k**

```
Aggregate: 67% Yes

Key predictions:
- Tanya (soil specialist): 82% Yes, stake 50 community points
  "Based on preliminary soil tests, lead levels are borderline. Full
  remediation to food-grade will likely exceed $50k. Industrial sites
  in this area historically need $60-80k."

- Marcus: 55% Yes, stake 30 community points
  "I've found grants that could offset costs. If we get the EPA
  Brownfields grant, net cost could be under $50k even if gross is higher."

- Jerome: 75% Yes, stake 40 community points
  "Our budget can handle $60k max. Anything over threatens other programs."
```

**Market 2: First Harvest in 12 Months**

```
Aggregate: 43% Yes

Key predictions:
- Tanya: 25% Yes
  "Soil remediation alone takes 6-8 months. Then bed preparation.
  Realistically 18 months to harvest."

- Long-time gardener (anonymous): 60% Yes
  "We can use raised beds with imported soil while remediation
  continues underground. I've done this at two other sites."
```

**Market 3: Membership Growth >20%**

```
Aggregate: 71% Yes

Key predictions:
- Elena: 78% Yes
  "Wait list is 45 people. New location in underserved area. Media
  coverage of expansion would boost visibility. Only risk is if
  project has visible failures."
```

### The Decision Market

Based on the prediction markets, the decision market showed:

**"Should DUAC acquire the lot?"**
- Initial: 52% Yes
- After prediction market information: 61% Yes
- Final (with conditional reasoning): 58% Yes

**Conditional Insights**:
- "Yes if EPA grant received" cluster: 75% Yes
- "Yes if remediation under $60k" cluster: 68% Yes
- "Yes regardless" cluster: 45% Yes

### The Decision Process

1. **Prediction markets ran for 4 weeks**
2. **Town hall to review predictions and reasoning**
3. **Traditional vote informed by market data**
4. **Result: 67% voted to proceed, contingent on EPA grant application**

### Outcomes (18 months later)

| Prediction | Predicted | Actual | Calibration |
|------------|-----------|--------|-------------|
| Cost >$50k | 67% Yes | Yes ($62k) | Good |
| Harvest in 12mo | 43% Yes | No (15 months) | Good |
| Membership +20% | 71% Yes | Yes (+34%) | Slightly under |

**Community Response**:
- Tanya earned "Soil Sage" badge for accurate cost prediction
- Raised bed suggestion implemented, accelerated partial harvest
- Jerome's budget planning proved essential
- Community learned to weight specialist input more heavily

### Private Market Features Used

```typescript
// Community-specific configuration
const duacConfig = {
  visibility: "private",
  membershipRequired: true,
  stakeType: "CommunityPoints", // Not real money
  minParticipation: 0.3, // 30% of members must predict
  resolutionMethod: "CommunityConsensus",
  dataRetention: "local", // Data stays on community server
};
```

### Case Lessons

1. **Private instances serve communities**: DUAC maintained control of their data and process
2. **Prediction markets inform but don't replace democracy**: Final decision was still voted
3. **Specialist knowledge surfaced**: Tanya's expertise became visible to all
4. **Conditional preferences clarified**: "Yes if grant" different from "Yes regardless"
5. **Non-monetary stakes work**: Community points created accountability without money
6. **Distributed knowledge aggregated**: 40+ members contributed diverse local knowledge

---

## Case Study 4: The Scientific Replication

*Showing: Deep epistemic integration, knowledge graph anchoring, long-term learning*

### Setup

**The Study**: A psychology paper claimed that "power posing" for 2 minutes before stressful situations increased testosterone and decreased cortisol, improving performance.

**The Question**: "Will the power posing study (Smith et al., 2012) replicate in the 2024 Many Labs project?"

**Classification**: E4-N3-M3
- E4: Measurable (scientific replication with pre-registered methods)
- N3: Universal (objective effect or no effect)
- M3: Foundational (implications for psychology methodology)

### Knowledge Graph Integration

**Anchored Claims** (from Mycelix Knowledge hApp):

```
Claim: "Power posing increases testosterone"
  └── Citation: Smith et al. 2012
  └── Replication attempts: 4 (1 success, 3 failure)
  └── Meta-analysis status: Contested
  └── Prediction market: [this market]

Claim: "Replication crisis affects >50% of psychology studies"
  └── Citation: Open Science Collaboration 2015
  └── Confidence: 0.89 (high agreement)
  └── Related markets: 12 active
```

### Participants

**Prof. Psychology** - Academic researcher
- MATL: 0.91 in psychology
- Published on replication crisis
- Known for methodological criticism

**True Believer** - Motivational coach
- MATL: 0.45 in psychology
- Uses power posing in practice
- Invested in outcome

**Statistics PhD** - Methodologist
- MATL: 0.87 in methodology
- No prior position on power posing
- Analyzes study design

**Original Author** - (Observation only, conflict disclosed)
- Cannot trade
- Can provide wisdom seeds
- Transparent about interest

### Predictions

**Prof. Psychology - No (88%)**
```markdown
Stake: 200 HAM + 15% psychology MATL + Commitment: "If I'm wrong, I will
write a public blog post acknowledging the study's validity"

Reasoning: "Three prior replications failed. Effect size in original
(d=0.75) is implausibly large for hormonal intervention. Many Labs
methodology is rigorous with large N. Bayesian prior from replication
crisis data gives <15% replication probability for effects this size.

Specific prediction: The Many Labs study will find effect size <0.2
(essentially null) with 95% CI including zero."

Knowledge graph anchors:
- Replication crisis base rate: Link
- Prior replication attempts: Links
- Effect size distribution literature: Link
```

**True Believer - Yes (70%)**
```markdown
Stake: 300 HAM

Reasoning: "I have personally witnessed power posing help hundreds of
clients. The laboratory setting of prior replications may not capture
real-world effects. Many Labs might finally get the methodology right.
I stake my money where my practice is."

Wisdom seed (regardless of outcome): "Even if the hormonal mechanism
is wrong, the behavioral confidence effects may be real through a
different pathway. Consider separating the hormonal claim from the
performance claim."
```

**Statistics PhD - No (75%)**
```markdown
Stake: 150 HAM + 8% methodology MATL

Reasoning: "Power analysis: Original study N=42 is underpowered for
claimed effect. Publication bias correction suggests true effect ~0.
Many Labs N=2000+ will have power to detect d>0.1. Prediction: they
will find null result.

I'm less confident than Prof. Psychology because small non-zero
effects are hard to rule out, and Many Labs might find d=0.15 which
would be 'technically replicates' even if practically useless."
```

**Original Author - Wisdom Seed Only**
```markdown
"I cannot trade, but I offer context: Our original study was done
before current standards. If I were designing it today, I would
measure differently. Whatever the outcome, I hope the field learns
about measurement and not just about this specific claim.

To future researchers: The confidence vs. hormones distinction
matters. We may have measured the wrong mediator."
```

### Resolution

**Many Labs Result**: Failed to replicate. Effect size d=0.03, 95% CI [-0.05, 0.11]

**Oracle Process**:
1. Pre-registered study published in peer-reviewed journal
2. Classification: E4 (scientific replication)
3. 15 oracles (psychology domain), 14 vote "No replicated," 1 abstains
4. Consensus achieved

### Knowledge Graph Update

The market resolution triggered automatic knowledge graph updates:

```
Claim: "Power posing increases testosterone"
  └── Status: CONTESTED → LIKELY FALSE
  └── Confidence update: 0.45 → 0.12
  └── Linked prediction market: Resolved NO
  └── Update reason: Many Labs 2024 failure to replicate

New claim created:
  "Power posing may have subjective confidence effects unrelated to hormones"
  └── Status: UNRESOLVED
  └── Linked markets: [new market spawned]
  └── Proposer: True Believer (wisdom seed)
```

### Outcomes and Learning

| Participant | Prediction | Outcome | Calibration Assessment |
|-------------|------------|---------|------------------------|
| Prof. Psychology | 88% No | Correct | Excellent (slightly overconfident given uncertainty) |
| Statistics PhD | 75% No | Correct | Well-calibrated |
| True Believer | 70% Yes | Wrong | Lost stake, but wisdom seed appreciated |

**Prof. Psychology**:
- Reward: 285 HAM + 3% MATL increase
- Commitment not invoked (correct prediction)
- Wisdom seed highly cited

**True Believer**:
- Lost 300 HAM stake
- But: Wisdom seed about mechanism spawned new market
- Net MATL: -0.05 (loss) +0.08 (wisdom contribution) = +0.03
- Lesson: "Being invested in outcome ≠ being right about mechanism"

**Original Author**:
- Wisdom contribution honored
- +0.1 Legacy score
- Comment: "Graceful in facing disconfirmation"

### Long-Term Impact

**1 Year Later**:
- True Believer's wisdom seed led to new study on subjective confidence
- That study partially confirmed: "Expansive postures increase subjective confidence (d=0.3) through cognitive rather than hormonal pathway"
- True Believer's legacy: Wrong on original, but contributed to refined understanding

**Knowledge Graph State**:
```
"Power posing" topic cluster:
├── Hormonal effects: DISCONFIRMED (confidence 0.92)
├── Subjective confidence effects: PARTIALLY CONFIRMED (confidence 0.67)
├── Optimal study design: ACTIVE RESEARCH
└── Meta-lesson: Mechanism matters as much as effect
```

### Case Lessons

1. **Knowledge graph integration creates cumulative learning**: Claims update based on market resolution
2. **Being wrong gracefully has value**: True Believer's wisdom seed survived their prediction's failure
3. **Commitments raise stakes meaningfully**: Prof. Psychology's blog post commitment showed skin in game
4. **Original authors can contribute without trading**: Conflict of interest managed while allowing wisdom
5. **Nuance survives binary markets**: "Didn't replicate" ≠ "completely useless," and the system captured this
6. **Long-term follow-through matters**: The spawned follow-up market continued the inquiry

---

## Case Study 5: The Crisis Response

*Showing: Emergency markets, rapid resolution, real-world coordination*

### Setup

**The Crisis**: Earthquake in densely populated region. Magnitude 7.2.

**The Need**: Rapid prediction aggregation for response planning.

**Emergency Market Activation**:
- Crisis flag triggered by seismic data integration
- Expedited market creation (bypassing normal 24h review)
- Increased oracle pool activation
- Shortened trading windows

### Markets Created (within 6 hours of earthquake)

1. **"Will confirmed casualties exceed 1,000?"** (E3-N3-M1, 48h window)
2. **"Will international aid be requested within 72 hours?"** (E2-N2-M0, 24h window)
3. **"Will aftershock >6.0 occur within 7 days?"** (E4-N3-M1, 7d window)
4. **"Will main hospital in affected area remain operational?"** (E2-N2-M0, 12h window)

### Participant Mobilization

**Activated Oracle Pool**:
- 12 seismology domain experts
- 8 emergency response specialists
- 15 regional knowledge holders
- 5 medical logistics experts

**Prediction Volume**:
- 847 predictions in first 12 hours
- Average prediction time: 8 minutes (vs. normal 2+ hours)
- Reasoning requirements relaxed to "100 characters minimum"

### Key Predictions (Market 1: Casualties >1000)

**Seismologist A - Yes (82%)**
```
Stake: 50 HAM (reduced minimums for crisis)
Quick reasoning: "Magnitude 7.2 in dense area, building codes historically poor,
night timing means people in buildings. Historical correlation says >1000 likely."
```

**Regional Expert - No (55%)**
```
Stake: 30 HAM
Quick reasoning: "Recent building code improvements after 2015 earthquake.
This area has better construction than similar magnitude events. Close call."
```

**Emergency Responder - Yes (65%)**
```
Stake: 40 HAM
Quick reasoning: "On ground. Seeing significant structural damage.
Hospital overwhelmed. Informal estimate already at 400 confirmed."
```

### Aggregate Evolution

| Time | Market Price (Casualties >1000) | Key Information |
|------|--------------------------------|-----------------|
| +6h | 64% | Initial seismic data |
| +12h | 71% | First responder reports |
| +18h | 78% | Hospital overwhelm signals |
| +24h | 85% | Official count at 700, trend clear |
| +36h | 92% | Official count passes 900 |
| +48h | Resolution: Yes (1,247 confirmed) |

### Cross-hApp Coordination

**From Governance hApp**:
- Emergency parameter override approved (15 minute vote)
- Fee waiver for crisis predictions activated
- Oracle compensation increased for rapid response

**From Identity hApp**:
- On-ground responders given temporary "verified responder" status
- Location-based credibility weighting activated
- Pseudonymity preserved but regional verification added

**From Knowledge hApp**:
- Real-time data feeds integrated
- Historical earthquake outcome data surfaced
- Building code database queried

### Resolution and Learning

**Final Outcomes**:

| Market | Prediction | Resolution | Time to Resolution |
|--------|------------|------------|-------------------|
| Casualties >1000 | 64%→92% | Yes (1,247) | 48 hours |
| International aid request | 78% | Yes (requested at +36h) | 40 hours |
| Aftershock >6.0 | 34% | No (max 5.6) | 7 days |
| Hospital operational | 42% | Yes (maintained ops) | 12 hours |

### Value Created

**For Response Coordination**:
- Aid agencies used casualty probability to pre-position resources
- Hospital capacity prediction informed evacuation decisions
- Aftershock predictions guided search & rescue safety protocols

**For Future Learning**:
```markdown
Wisdom seed collection (crisis markets):

1. "Night earthquakes have 40% higher casualty rates than equivalent
   day earthquakes - update base rate models" - Seismologist A

2. "On-ground responders' predictions are initially noisy but converge
   to accuracy faster than remote experts" - System analysis

3. "Hospital operational prediction was most useful for real-time
   decisions - prioritize these in future crises" - Emergency responder

4. "Market aggregation was 6 hours faster than official channels in
   reaching accurate casualty estimates" - Post-crisis analysis
```

### Post-Crisis Protocol Update

Based on this case, governance approved:

```rust
pub struct CrisisProtocol {
    pub activation_criteria: CrisisCriteria,
    pub parameter_overrides: CrisisParameters,
    pub lessons_integration: Vec<Lesson>,
}

// Approved updates:
let updates = CrisisProtocolUpdate {
    faster_oracle_activation: Duration::from_hours(2), // was 6
    ground_truth_weighting: 1.5, // increase weight of on-site predictors
    hospital_market_priority: Priority::Critical, // always create this market
    post_crisis_review: Duration::from_days(30), // mandatory review period
};
```

### Case Lessons

1. **Crisis markets can provide rapid signal**: 6+ hours faster than official channels
2. **On-ground expertise is gold**: Verified responders' predictions highly accurate
3. **Reduced barriers appropriate in emergencies**: Lower stakes, shorter reasoning, faster resolution
4. **Cross-hApp integration force multiplies**: Real-time data + identity + governance coordination
5. **Wisdom capture in crisis is possible**: Even under time pressure, valuable insights recorded
6. **Protocol learning closes loop**: Crisis lessons became protocol updates

---

## Conclusion

These case studies illustrate Epistemic Markets in action across diverse contexts:

| Case | Key Mechanism | Primary Lesson |
|------|---------------|----------------|
| Climate | Expert weighting | MATL surfaces domain expertise |
| Tech Launch | Question markets | Community identifies worthy questions |
| Community | Private markets | Non-monetary stakes create accountability |
| Scientific | Knowledge integration | Learning accumulates over time |
| Crisis | Emergency protocols | Speed and accuracy can coexist |

Real cases will differ from these composites, but the patterns should hold. As the system operates, we'll update this document with actual examples.

---

*"In the end, we learn not from theories but from stories. These are the stories we expect to tell."*

*May reality be even more interesting.*


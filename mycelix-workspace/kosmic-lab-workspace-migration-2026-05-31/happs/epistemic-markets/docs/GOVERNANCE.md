# Governance

*How We Decide Together*

---

> "The best way to predict the future is to create it."
> — Peter Drucker

> "The best way to create the future is to decide it together."
> — Us

---

## Introduction

A truth-seeking system that cannot govern itself cannot survive. Governance is not an add-on—it is fundamental infrastructure for collective intelligence.

This document describes how Epistemic Markets makes decisions, evolves its protocols, and maintains legitimacy over time.

---

## Part I: Governance Philosophy

### The Paradox

Epistemic markets help us discover truth. But governance requires making decisions before truth is known. How do we govern a truth-seeking system when truth is precisely what we're trying to find?

**Resolution:** We use epistemic markets to inform governance, while acknowledging that governance decisions are themselves predictions about what will work.

### Core Principles

```
1. Epistemic Legitimacy
   Governance authority derives from demonstrated epistemic virtue:
   - Calibration (track record of accuracy)
   - Wisdom (quality of reasoning)
   - Integrity (consistency of behavior)
   - Service (contribution to community)

2. Skin in the Game
   Those who decide must bear consequences:
   - Stake requirements for proposals
   - Reputation at risk for poor decisions
   - Long-term accountability mechanisms

3. Inclusive Deliberation
   All voices can be heard:
   - Low barriers to participation
   - Multiple channels for input
   - Protection of minority views
   - Translation and accessibility

4. Bounded Authority
   No unlimited power:
   - Constitutional constraints
   - Term limits and rotation
   - Veto and appeal mechanisms
   - Fork rights preserved

5. Adaptive Evolution
   The system must be able to change:
   - Clear amendment processes
   - Experimentation frameworks
   - Learning from outcomes
   - Graceful deprecation
```

---

## Part II: Governance Layers

### Layer 0: Constitutional

Things that cannot be changed without forking.

```rust
pub struct Constitution {
    // Immutable core values
    pub core_values: Vec<Value>,

    // Fundamental rights
    pub rights: Vec<Right>,

    // Red lines (what the system must never do)
    pub red_lines: Vec<RedLine>,

    // Amendment process for constitution itself
    pub meta_governance: MetaGovernance,
}

pub enum Value {
    TruthSeeking,           // Primary purpose
    HumanDignity,           // Respect for all participants
    EpistemicHumility,      // Acknowledgment of uncertainty
    CollaborativeSpirit,    // Synthesis over competition
    IntergenerationalCare,  // Responsibility to future
}

pub enum Right {
    Participation,          // Anyone can participate
    Exit,                   // Anyone can leave
    Privacy,                // Certain data stays private
    Appeal,                 // Decisions can be challenged
    Fork,                   // Community can split
}

pub enum RedLine {
    NoManipulation,         // System cannot be designed to deceive
    NoSurveillance,         // No tracking beyond what's necessary
    NoExclusion,            // Cannot exclude based on identity
    NoWeaponization,        // Cannot be used to harm
}
```

**Amendment process:**
- Requires 90% supermajority
- 6-month deliberation period
- External review
- Fork notification period

### Layer 1: Protocol

Core rules of how the system operates.

```rust
pub struct Protocol {
    // Market mechanics
    pub market_rules: MarketRules,

    // Resolution mechanisms
    pub resolution_rules: ResolutionRules,

    // Reputation systems
    pub reputation_rules: ReputationRules,

    // Economic parameters
    pub economic_rules: EconomicRules,
}
```

**Governed by:** Protocol Council + Community Ratification

**Amendment process:**
- Proposal with stake (100 HAM minimum)
- Technical review (2 weeks)
- Community discussion (2 weeks)
- Voting period (1 week)
- Implementation period (varies)
- 67% supermajority required

### Layer 2: Parameters

Tunable values within the protocol.

```rust
pub struct Parameters {
    // Market parameters
    pub min_stake: u64,
    pub max_market_duration: Duration,
    pub resolution_threshold: f64,

    // Reputation parameters
    pub calibration_weight: f64,
    pub wisdom_weight: f64,
    pub decay_rate: f64,

    // Economic parameters
    pub fee_percentage: f64,
    pub oracle_reward: u64,
    pub wisdom_seed_bonus: u64,
}
```

**Governed by:** Parameter Committee + Automated Bounds

**Amendment process:**
- Proposal with analysis
- Impact assessment
- 7-day voting period
- Simple majority (>50%)
- Bounded by protocol-defined ranges

### Layer 3: Operations

Day-to-day decisions within parameters.

```rust
pub struct Operations {
    // Market curation
    pub featured_markets: Vec<MarketHash>,

    // Community management
    pub moderation_decisions: Vec<ModerationAction>,

    // Resource allocation
    pub grant_decisions: Vec<Grant>,

    // Partnerships
    pub integration_approvals: Vec<Integration>,
}
```

**Governed by:** Operations Team + Community Oversight

**Process:**
- Delegated authority within bounds
- Transparent decision logs
- Appeal mechanisms
- Regular review

---

## Part III: Governance Bodies

### The Protocol Council

**Purpose:** Steward protocol-level changes

**Composition:**
```typescript
interface ProtocolCouncil {
  // Elected members (7)
  elected: {
    count: 7;
    term: "2 years";
    staggered: true;  // 3-4 rotate each year
    electionMethod: "quadratic_voting";
  };

  // Domain experts (3, rotating)
  experts: {
    count: 3;
    term: "1 year";
    domains: ["cryptography", "economics", "governance"];
    selectionMethod: "community_nomination";
  };

  // Random citizens (2)
  random: {
    count: 2;
    term: "6 months";
    selectionMethod: "sortition";  // Random from qualified pool
    qualifications: ["active_predictor", "good_standing"];
  };
}
```

**Powers:**
- Propose protocol changes
- Commission technical review
- Call community votes
- Emergency pause authority (with sunset)

**Constraints:**
- Cannot override constitution
- Decisions appealable to community
- Transparent deliberations
- Conflict of interest disclosure required

### The Parameter Committee

**Purpose:** Tune system parameters

**Composition:**
```typescript
interface ParameterCommittee {
  // Elected specialists (5)
  members: {
    count: 5;
    term: "1 year";
    requirements: ["technical_expertise", "calibration_score > 0.8"];
  };

  // Rotating community observers (2)
  observers: {
    count: 2;
    term: "3 months";
    role: "transparency and community connection";
  };
}
```

**Powers:**
- Adjust parameters within bounds
- Propose bound changes (to Protocol Council)
- Emergency adjustments (with immediate review)

**Constraints:**
- Must stay within protocol-defined bounds
- Must publish impact analysis
- Decisions reversible by community vote

### The Operations Team

**Purpose:** Day-to-day management

**Composition:**
```typescript
interface OperationsTeam {
  // Paid staff
  staff: {
    hired_by: "Protocol Council";
    accountability: "quarterly reviews";
  };

  // Volunteer moderators
  moderators: {
    selection: "community nomination + training";
    oversight: "community appeals process";
  };
}
```

**Powers:**
- Content moderation
- User support
- Market curation
- Partnership execution

**Constraints:**
- Operate within policy
- All actions logged
- Appealable decisions
- Regular community reporting

### The Elder Council

**Purpose:** Long-term wisdom and continuity

**Composition:**
```typescript
interface ElderCouncil {
  // Those who have demonstrated sustained contribution
  members: {
    criteria: [
      "5+ years active participation",
      "high wisdom score",
      "community recognition"
    ];
    selection: "invitation by existing elders + community confirmation";
    term: "lifetime, with activity requirement";
  };
}
```

**Powers:**
- Advisory (non-binding)
- Constitutional interpretation
- Wisdom preservation
- Succession guidance

**Constraints:**
- No direct operational authority
- Cannot block decisions
- Must explain reasoning publicly

### The Community Assembly

**Purpose:** Ultimate sovereign authority

**Composition:** All active participants

**Powers:**
- Ratify protocol changes
- Override any decision (supermajority)
- Elect council members
- Initiate constitutional amendments
- Call emergency sessions

**Activation:**
- Protocol Council decision + automatic referral
- Petition (5% of active predictors)
- Emergency trigger (defined conditions)

---

## Part IV: Decision Processes

### Standard Proposal Process

```
Phase 1: Ideation (1-4 weeks)
├── Anyone can post ideas
├── Community discussion
├── Informal polling
└── Coalition building

Phase 2: Proposal (1 week)
├── Formal proposal submitted
├── Stake deposited
├── Technical review assigned
└── Public comment period opens

Phase 3: Review (2 weeks)
├── Technical review completed
├── Impact assessment published
├── Community deliberation
├── Amendments possible
└── Final proposal locked

Phase 4: Voting (1 week)
├── Voting period opens
├── Quadratic voting for council elections
├── Stake-weighted for parameter changes
├── One-person-one-vote for constitutional
└── Results certified

Phase 5: Implementation (varies)
├── If passed: implementation begins
├── If failed: proposal archived with learnings
├── Appeal period (if contested)
└── Post-implementation review scheduled
```

### Emergency Process

For urgent situations:

```rust
pub struct EmergencyProcess {
    // Trigger conditions
    pub triggers: Vec<EmergencyTrigger>,

    // Immediate authority
    pub immediate_powers: Vec<EmergencyPower>,

    // Sunset provisions
    pub max_duration: Duration,

    // Ratification requirement
    pub post_hoc_approval: ApprovalRequirement,
}

pub enum EmergencyTrigger {
    SecurityBreach,         // Active attack
    EconomicCrisis,         // Token collapse
    LegalThreat,            // Regulatory action
    TechnicalFailure,       // System-wide outage
    SocialCrisis,           // Community meltdown
}

pub enum EmergencyPower {
    PauseMarkets,           // Stop trading
    FreezeFunds,            // Prevent withdrawals
    BanAccounts,            // Remove bad actors
    RevertDecisions,        // Undo recent changes
    // Cannot: change constitution, seize funds permanently
}
```

**Constraints:**
- Maximum 7 days without ratification
- Must be public within 24 hours
- Automatic review after resolution
- Compensation for harmed parties

### Dispute Resolution

```
Level 1: Direct Resolution
├── Parties attempt direct dialogue
├── Structured format provided
├── 7 days to resolve
└── Escalate if failed

Level 2: Mediation
├── Neutral mediator assigned
├── Confidential process
├── 14 days to resolve
├── Non-binding recommendations
└── Escalate if failed

Level 3: Arbitration
├── Arbitration panel (3 members)
├── Evidence submission
├── Hearing (if needed)
├── Binding decision
├── Limited appeal rights
└── Final for most disputes

Level 4: Community Appeal
├── Only for systemic issues
├── Requires 100 supporter signatures
├── Community vote
├── Can override arbitration
└── Creates precedent

Level 5: Fork
├── For irreconcilable differences
├── Community splits
├── Assets divided per rules
└── Both paths continue
```

---

## Part V: Voting Mechanisms

### Quadratic Voting

For elections and preference aggregation:

```typescript
interface QuadraticVoting {
  // Credits allocated per voter
  creditsPerVoter: number;

  // Cost function: votes² = credits
  cost(votes: number): number {
    return votes * votes;
  }

  // Example: 100 credits
  // - 10 votes on one option costs 100 credits
  // - 5 votes on four options costs 100 credits (25 each)
  // Encourages expressing intensity while limiting plutocracy
}

function allocateVotes(credits: number, preferences: Preference[]): Vote[] {
  // Voter distributes credits across options
  // More credits = more votes, but with diminishing returns
  return preferences.map(p => ({
    option: p.option,
    votes: Math.floor(Math.sqrt(p.credits)),
    credits: p.credits
  }));
}
```

**Used for:**
- Protocol Council elections
- Multi-option decisions
- Priority ranking

### Stake-Weighted Voting

For parameter decisions:

```typescript
interface StakeWeightedVoting {
  // Weight by stake and reputation
  weight(voter: Agent): number {
    const stake = voter.activeStake;
    const reputation = voter.matlScore.composite;
    const timeWeight = Math.log(voter.accountAge + 1);

    return stake * reputation * timeWeight;
  }

  // Caps to prevent plutocracy
  maxWeight: number;  // No single voter > 5% of total

  // Minimum stake to participate
  minStake: number;
}
```

**Used for:**
- Economic parameter changes
- Fund allocation
- Partnership approvals

### Conviction Voting

For ongoing priorities:

```typescript
interface ConvictionVoting {
  // Votes accumulate over time
  conviction(votes: number, time: Duration): number {
    const halfLife = 7 * 24 * 60 * 60; // 7 days
    return votes * (1 - Math.exp(-time / halfLife));
  }

  // Threshold scales with requested resources
  threshold(requested: number, total: number): number {
    return total * (requested / (requested + THRESHOLD_CONSTANT));
  }

  // Proposals pass when conviction exceeds threshold
}
```

**Used for:**
- Ongoing funding decisions
- Feature prioritization
- Community initiatives

### Futarchy

Using prediction markets to inform governance:

```typescript
interface Futarchy {
  // For each proposal, create conditional markets
  async createConditionalMarkets(proposal: Proposal): Promise<ConditionalMarkets> {
    return {
      // "What will success metric be if proposal passes?"
      ifPasses: await createMarket({
        condition: "proposal passes",
        question: `What will ${proposal.successMetric} be in ${proposal.timeframe}?`
      }),

      // "What will success metric be if proposal fails?"
      ifFails: await createMarket({
        condition: "proposal fails",
        question: `What will ${proposal.successMetric} be in ${proposal.timeframe}?`
      })
    };
  }

  // Decision rule: choose option with better predicted outcome
  decide(markets: ConditionalMarkets): Decision {
    if (markets.ifPasses.predictedValue > markets.ifFails.predictedValue) {
      return Decision.Pass;
    } else {
      return Decision.Fail;
    }
  }
}
```

**Used for:**
- Major strategic decisions
- Uncertain trade-offs
- Experimental policy

---

## Part VI: Checks and Balances

### Separation of Powers

```
┌─────────────────────────────────────────────────────────────┐
│                    COMMUNITY ASSEMBLY                        │
│              (Ultimate sovereign authority)                  │
└─────────────────────────────────────────────────────────────┘
                              ↑
            ┌─────────────────┼─────────────────┐
            ↓                 ↓                 ↓
┌───────────────────┐ ┌───────────────────┐ ┌───────────────────┐
│  PROTOCOL COUNCIL │ │ ELDER COUNCIL     │ │ DISPUTE TRIBUNAL  │
│  (Legislative)    │ │ (Advisory)        │ │ (Judicial)        │
│                   │ │                   │ │                   │
│ - Propose changes │ │ - Interpret       │ │ - Resolve disputes│
│ - Set policy      │ │ - Advise          │ │ - Enforce rules   │
│ - Approve budget  │ │ - Preserve wisdom │ │ - Protect rights  │
└───────────────────┘ └───────────────────┘ └───────────────────┘
            ↓                                         ↑
┌───────────────────┐                                 │
│  OPERATIONS TEAM  │←────────────────────────────────┘
│  (Executive)      │         (Appeals flow to Tribunal)
│                   │
│ - Implement       │
│ - Moderate        │
│ - Operate         │
└───────────────────┘
```

### Accountability Mechanisms

```rust
pub struct Accountability {
    // Regular reporting
    pub reporting: Vec<ReportingRequirement>,

    // Performance review
    pub reviews: Vec<Review>,

    // Removal procedures
    pub removal: RemovalProcess,

    // Transparency requirements
    pub transparency: Vec<TransparencyRequirement>,
}

pub struct ReportingRequirement {
    pub body: GovernanceBody,
    pub frequency: Duration,
    pub content: Vec<ReportSection>,
    pub audience: Audience,
}

pub struct Review {
    pub body: GovernanceBody,
    pub frequency: Duration,
    pub criteria: Vec<Criterion>,
    pub consequence: ReviewConsequence,
}

pub enum ReviewConsequence {
    Continuation,
    Warning,
    ProbationaryPeriod,
    Removal,
    CommunityVote,
}
```

### Anti-Capture Mechanisms

```typescript
interface AntiCapture {
  // Term limits
  maxConsecutiveTerms: 2;

  // Cooling-off periods
  coolingOff: {
    afterTerm: "1 year before eligible again",
    afterRemoval: "3 years minimum",
  };

  // Conflict of interest
  conflictRules: {
    disclosure: "mandatory for any financial interest",
    recusal: "required when directly affected",
    divestment: "required for major conflicts",
  };

  // Concentration limits
  concentrationLimits: {
    maxVotingPower: "5% of total",
    maxCouncilFromSameOrg: 2,
    maxRelatedParties: "20% of any body",
  };

  // Diversity requirements
  diversityTargets: {
    geographic: "no region > 40%",
    tenure: "mix of new and experienced",
    background: "multiple domains represented",
  };
}
```

---

## Part VII: Evolution and Meta-Governance

### How Governance Itself Changes

```rust
pub struct MetaGovernance {
    // Regular governance review
    pub review_cycle: Duration,  // Every 2 years

    // Constitutional convention
    pub convention_trigger: ConventionTrigger,

    // Amendment process
    pub amendment_process: AmendmentProcess,

    // Fork rights
    pub fork_rights: ForkRights,
}

pub struct AmendmentProcess {
    // For protocol-level changes
    pub protocol_amendment: Amendment {
        proposal_threshold: "5% of active predictors",
        deliberation_period: Duration::from_days(30),
        voting_period: Duration::from_days(14),
        approval_threshold: 0.67,
        implementation_delay: Duration::from_days(30),
    },

    // For constitutional changes
    pub constitutional_amendment: Amendment {
        proposal_threshold: "10% of active predictors",
        deliberation_period: Duration::from_days(180),
        voting_period: Duration::from_days(30),
        approval_threshold: 0.90,
        implementation_delay: Duration::from_days(90),
        external_review: true,
        fork_notification: true,
    },
}
```

### Experimental Governance

Safe ways to try new approaches:

```typescript
interface GovernanceExperiment {
  // Sandbox for new mechanisms
  sandbox: {
    scope: "limited markets or communities";
    duration: "3-6 months";
    metrics: "pre-defined success criteria";
    rollback: "automatic if criteria not met";
  };

  // Gradual rollout
  rollout: {
    phases: ["5%", "20%", "50%", "100%"];
    gating: "each phase requires success in previous";
    timeline: "minimum 1 month per phase";
  };

  // A/B testing for minor changes
  abTesting: {
    scope: "UI/UX and minor parameters only";
    consent: "users can opt out";
    analysis: "published within 30 days";
  };
}
```

### Learning from Governance

```rust
pub struct GovernanceLearning {
    // Every decision is a prediction
    pub decision_as_prediction: DecisionPrediction,

    // Track outcomes
    pub outcome_tracking: OutcomeTracking,

    // Calibrate governance
    pub governance_calibration: GovernanceCalibration,
}

pub struct DecisionPrediction {
    pub decision: Decision,
    pub expected_outcome: Outcome,
    pub success_criteria: Vec<Criterion>,
    pub review_date: Timestamp,
}

impl GovernanceLearning {
    pub fn review_decision(&self, decision: &DecisionPrediction) -> LearningReport {
        let actual_outcome = measure_outcome(decision);
        let accuracy = compare_outcomes(&decision.expected_outcome, &actual_outcome);

        LearningReport {
            decision: decision.clone(),
            actual_outcome,
            accuracy,
            lessons: extract_lessons(decision, actual_outcome),
            recommendations: generate_recommendations(accuracy),
        }
    }
}
```

---

## Part VIII: Failure Modes and Safeguards

### Governance Failures

| Failure | Detection | Response |
|---------|-----------|----------|
| Gridlock | Decision latency exceeds threshold | Emergency delegation + review |
| Capture | Concentration metrics triggered | Automatic diversification + audit |
| Apathy | Participation below minimum | Outreach + incentive adjustment |
| Tyranny | Rights violations detected | Automatic appeal + community alert |
| Corruption | Anomaly detection + whistleblower | Investigation + suspension |

### Ultimate Safeguards

```rust
pub struct UltimateSafeguards {
    // The nuclear option: community can always fork
    pub fork_rights: ForkRights {
        always_available: true,
        process: ForkProcess::Documented,
        asset_division: AssetDivision::ProRata,
        data_portability: DataPortability::Full,
    },

    // Minority protection
    pub minority_rights: MinorityRights {
        exit_right: true,
        voice_guarantee: true,
        appeal_process: true,
        proportional_representation: true,
    },

    // External oversight option
    pub external_oversight: ExternalOversight {
        available: true,
        triggers: vec![
            "rights violation allegation",
            "major financial irregularity",
            "constitutional crisis",
        ],
        oversight_body: "mutually agreed neutral party",
    },
}
```

---

## Conclusion

Governance is not a solved problem—it is an ongoing practice.

This document describes our current best understanding of how to govern a truth-seeking system. It will change as we learn. The mechanisms described here are themselves predictions about what will work.

We govern ourselves in order to seek truth together. When governance stops serving that purpose, it should change.

The ultimate test of our governance: Does it help us see clearly together?

---

*"The price of liberty is eternal vigilance."*
*— Wendell Phillips*

*The price of truth is eternal deliberation.*
*May we deliberate well.*

# Economics

*How Value Flows Through the System*

---

> "Money is a means, not an end."
> — Peter Drucker

> "Incentives are the means. Truth is the end."
> — Us

---

## Introduction

Every system has an economics—a structure of incentives that shapes behavior. Epistemic Markets must align incentives with truth-seeking, not gaming. This document describes how value flows through the system.

---

## Part I: Economic Philosophy

### The Core Tension

**Problem:** Traditional prediction markets optimize for profit, not truth. Participants try to win, not to be calibrated.

**Solution:** Multi-dimensional value that rewards epistemic virtue:
- Accuracy (being right)
- Calibration (knowing how right you are)
- Reasoning (thinking well)
- Contribution (helping others)
- Wisdom (teaching the future)

### Design Principles

```
1. Truth Over Profit
   The system should make truth-seeking the most profitable strategy.
   Gaming should be less profitable than honest participation.

2. Multi-Dimensional Value
   Not everything valuable is monetary.
   Reputation, social capital, and wisdom are real value.

3. Long-Term Alignment
   Incentives should reward sustained contribution, not extraction.
   Patience should be profitable.

4. Positive-Sum Dynamics
   The overall pie should grow with participation.
   Helping others should help yourself.

5. Accessibility
   Wealth should not determine influence.
   Multiple entry points for different resources.

6. Sustainability
   The system must fund itself without compromising mission.
   Economic viability enables epistemic mission.
```

---

## Part II: Value Types

### HAM (Holochain Accounting Module) Token

The primary monetary unit:

```rust
pub struct HAMToken {
    // Core properties
    pub total_supply: u128,
    pub circulating_supply: u128,
    pub treasury_reserve: u128,

    // Issuance
    pub issuance_schedule: IssuanceSchedule,
    pub inflation_rate: f64,  // Target: 2-3% annually

    // Distribution
    pub distribution: Distribution {
        predictor_rewards: 0.40,    // 40% to active predictors
        oracle_rewards: 0.15,       // 15% to resolution oracles
        wisdom_rewards: 0.10,       // 10% to wisdom contributors
        development: 0.15,          // 15% to protocol development
        treasury: 0.20,             // 20% to ecosystem treasury
    },
}
```

**Use cases:**
- Staking on predictions
- Market creation fees
- Oracle bonds
- Governance voting weight
- Access to premium features

### Reputation Score

Non-transferable value based on behavior:

```typescript
interface ReputationScore {
  // Components
  calibration: number;      // 0-100, based on prediction accuracy
  wisdom: number;           // 0-100, based on reasoning quality
  integrity: number;        // 0-100, based on behavioral consistency
  contribution: number;     // 0-100, based on community service

  // Composite
  composite: number;        // Weighted average

  // Domain-specific
  domains: Map<string, DomainScore>;

  // Temporal
  trend: "rising" | "stable" | "falling";
  history: ScoreHistory[];
}

// Reputation decays without activity
function applyDecay(score: ReputationScore, inactivityDays: number): ReputationScore {
  const decayRate = 0.001; // 0.1% per day
  const decayFactor = Math.exp(-decayRate * inactivityDays);
  return {
    ...score,
    calibration: score.calibration * decayFactor,
    // ... apply to other components
  };
}
```

**Use cases:**
- Weighting in resolution
- Access to high-stakes markets
- Governance influence
- Mentorship eligibility
- Oracle selection

### Social Capital

Relationship-based value:

```typescript
interface SocialCapital {
  // Endorsements received
  endorsements: Endorsement[];

  // Mentorship relationships
  mentees: Agent[];          // Those you've mentored
  mentors: Agent[];          // Those who've mentored you

  // Collaboration history
  collaborations: Collaboration[];

  // Community recognition
  badges: Badge[];
  titles: Title[];

  // Network position
  bridgingCapital: number;   // Connections across groups
  bondingCapital: number;    // Deep connections within groups
}
```

**Use cases:**
- Social staking
- Witness networks
- Mentorship matching
- Community building
- Trust networks

### Wisdom Credit

Value from teaching:

```typescript
interface WisdomCredit {
  // Seeds planted
  seedsPlanted: WisdomSeed[];

  // Seeds that germinated (influenced others)
  germinatedSeeds: WisdomSeed[];

  // Teaching contributions
  mentorshipHours: number;
  tutorialContributions: Content[];

  // Legacy score
  legacyScore: number;       // Long-term impact measure
}
```

**Use cases:**
- Elder status eligibility
- Wisdom rewards
- Legacy staking
- Intergenerational influence

---

## Part III: Value Flows

### Market Creation Flow

```
Creator                          System                          Pool
   │                               │                               │
   │──── Creates Market ──────────>│                               │
   │     (pays creation fee)       │                               │
   │                               │                               │
   │                               │──── Fee to Treasury ─────────>│
   │                               │     (70% treasury,            │
   │                               │      30% oracle pool)         │
   │                               │                               │
   │<─── Market Active ────────────│                               │
   │     (creator may earn         │                               │
   │      curation rewards if      │                               │
   │      market becomes popular)  │                               │
```

### Prediction Flow

```
Predictor                        System                          Pool
   │                               │                               │
   │──── Makes Prediction ────────>│                               │
   │     (stakes HAM/reputation)   │                               │
   │                               │                               │
   │                               │──── Stake to Market Pool ────>│
   │                               │                               │
   │     [Resolution Occurs]       │                               │
   │                               │                               │
   │<─── Correct: Receive ─────────│<────── From Pool ─────────────│
   │     stake + share of          │        (losers' stakes        │
   │     losers' stakes            │         redistribute)         │
   │     + calibration bonus       │                               │
   │                               │                               │
   │     OR                        │                               │
   │                               │                               │
   │<─── Incorrect: Lose stake ────│──────> To Pool ──────────────>│
   │     (but keep reasoning       │        (distributed to        │
   │      contribution credit)     │         correct predictors)   │
```

### Resolution Flow

```
Oracles                          System                          Pool
   │                               │                               │
   │<─── Resolution Request ───────│                               │
   │     (eligible oracles         │                               │
   │      invited based on MATL)   │                               │
   │                               │                               │
   │──── Submit Votes ────────────>│                               │
   │     (stake as bond)           │                               │
   │                               │                               │
   │     [Consensus Reached]       │                               │
   │                               │                               │
   │<─── Correct Oracles: ─────────│<────── Oracle Reward Pool ────│
   │     receive reward +          │                               │
   │     bond return +             │                               │
   │     reputation boost          │                               │
   │                               │                               │
   │<─── Incorrect/Byzantine: ─────│──────> Slashed to Pool ──────>│
   │     lose bond +               │                               │
   │     reputation penalty        │                               │
```

### Wisdom Flow

```
Wisdom Contributor               System                          Pool
   │                               │                               │
   │──── Plants Wisdom Seed ──────>│                               │
   │     (after prediction         │                               │
   │      resolves)                │                               │
   │                               │                               │
   │     [Seed Germination         │                               │
   │      Assessment - Quarterly]  │                               │
   │                               │                               │
   │<─── If Seed Referenced: ──────│<────── Wisdom Pool ───────────│
   │     receive wisdom credit +   │                               │
   │     HAM reward +              │                               │
   │     legacy score boost        │                               │
   │                               │                               │
   │     [Long-term Assessment     │                               │
   │      - Annually]              │                               │
   │                               │                               │
   │<─── If Seed Influential: ─────│<────── Legacy Pool ───────────│
   │     receive legacy rewards    │                               │
   │     (vesting over 5 years)    │                               │
```

---

## Part IV: Reward Mechanisms

### Accuracy Rewards

For being right:

```rust
pub fn calculate_accuracy_reward(
    prediction: &Prediction,
    resolution: &Resolution,
    market: &Market,
) -> AccuracyReward {
    let is_correct = prediction.outcome == resolution.outcome;

    if !is_correct {
        return AccuracyReward::zero();
    }

    // Base reward: share of losing stakes
    let losing_pool: u64 = market.stakes
        .iter()
        .filter(|s| s.outcome != resolution.outcome)
        .map(|s| s.amount)
        .sum();

    let winner_weights: f64 = market.stakes
        .iter()
        .filter(|s| s.outcome == resolution.outcome)
        .map(|s| s.weight())
        .sum();

    let my_weight = prediction.stake.weight();
    let base_share = losing_pool as f64 * (my_weight / winner_weights);

    // Confidence multiplier: reward accurate confidence
    let confidence_bonus = calculate_confidence_bonus(
        prediction.confidence,
        resolution.outcome,
        market.base_rate,
    );

    // Early bonus: reward early correct predictions
    let early_bonus = calculate_early_bonus(
        prediction.timestamp,
        market.created_at,
        market.closes_at,
    );

    AccuracyReward {
        base: base_share as u64,
        confidence_bonus,
        early_bonus,
        total: (base_share * confidence_bonus * early_bonus) as u64,
    }
}
```

### Calibration Rewards

For knowing how right you are:

```rust
pub fn calculate_calibration_reward(
    agent: &AgentPubKey,
    period: &Period,
) -> CalibrationReward {
    let predictions = get_resolved_predictions(agent, period);

    // Calculate Brier score
    let brier_score = calculate_brier_score(&predictions);

    // Better calibration = lower Brier score = higher reward
    // Brier ranges 0 (perfect) to 2 (worst)
    let calibration_score = 1.0 - (brier_score / 2.0);

    // Reward from calibration pool
    let pool_share = CALIBRATION_POOL * calibration_score;

    // Bonus for improvement
    let previous_brier = get_previous_brier(agent, period);
    let improvement_bonus = if brier_score < previous_brier {
        (previous_brier - brier_score) * IMPROVEMENT_MULTIPLIER
    } else {
        0.0
    };

    CalibrationReward {
        brier_score,
        calibration_score,
        pool_share: pool_share as u64,
        improvement_bonus: improvement_bonus as u64,
    }
}
```

### Reasoning Rewards

For thinking well:

```typescript
interface ReasoningReward {
  // Quality assessment (by community/AI)
  qualityScore: number;

  // Citation count (others referencing your reasoning)
  citations: number;

  // Influence on others' predictions
  influenceScore: number;

  // Teaching value (helped newcomers)
  teachingValue: number;
}

function calculateReasoningReward(
  prediction: Prediction,
  period: Period,
): ReasoningReward {
  const reasoning = prediction.reasoning;

  // Assess quality
  const quality = assessReasoningQuality(reasoning);

  // Count citations
  const citations = countCitations(reasoning.id, period);

  // Measure influence
  const influence = measureInfluence(reasoning, period);

  // Calculate reward
  const reward = (
    quality.score * QUALITY_WEIGHT +
    citations * CITATION_REWARD +
    influence * INFLUENCE_WEIGHT
  );

  return {
    qualityScore: quality.score,
    citations,
    influenceScore: influence,
    teachingValue: quality.teachingScore,
    reward,
  };
}
```

### Contribution Rewards

For helping the system:

```rust
pub enum ContributionType {
    OracleService {
        resolutions_participated: u32,
        accuracy_rate: f64,
    },
    Moderation {
        actions_taken: u32,
        appeal_rate: f64,  // Lower is better
    },
    Mentorship {
        mentees_helped: u32,
        mentee_success_rate: f64,
    },
    Development {
        prs_merged: u32,
        impact_score: f64,
    },
    Documentation {
        docs_contributed: u32,
        views_and_citations: u32,
    },
    CommunityBuilding {
        events_organized: u32,
        participants_engaged: u32,
    },
}

pub fn calculate_contribution_reward(
    agent: &AgentPubKey,
    period: &Period,
) -> ContributionReward {
    let contributions = get_contributions(agent, period);

    let total: u64 = contributions.iter().map(|c| {
        match c {
            ContributionType::OracleService { resolutions_participated, accuracy_rate } => {
                resolutions_participated as u64 * (*accuracy_rate * ORACLE_RATE) as u64
            }
            ContributionType::Mentorship { mentees_helped, mentee_success_rate } => {
                mentees_helped as u64 * (*mentee_success_rate * MENTOR_RATE) as u64
            }
            // ... other contribution types
        }
    }).sum();

    ContributionReward { total }
}
```

---

## Part V: Staking Mechanisms

### Monetary Stakes

```rust
pub struct MonetaryStake {
    pub amount: u64,
    pub currency: Currency,
    pub locked_until: Option<Timestamp>,
    pub early_unlock_penalty: f64,
}

impl MonetaryStake {
    pub fn weight(&self) -> f64 {
        let base = self.amount as f64;

        // Lock bonus: longer locks = more weight
        let lock_multiplier = match &self.locked_until {
            None => 1.0,
            Some(until) => {
                let lock_duration = *until - now();
                1.0 + (lock_duration.as_days() as f64 / 365.0) * 0.5
            }
        };

        base * lock_multiplier
    }
}
```

### Reputation Stakes

```rust
pub struct ReputationStake {
    pub domains: Vec<String>,
    pub stake_percentage: f64,  // % of reputation at risk
    pub confidence_multiplier: f64,
}

impl ReputationStake {
    pub fn at_risk(&self, agent: &Agent) -> f64 {
        let domain_scores: f64 = self.domains.iter()
            .map(|d| agent.reputation.domains.get(d).unwrap_or(&0.0))
            .sum();

        domain_scores * self.stake_percentage * self.confidence_multiplier
    }

    pub fn apply_outcome(&self, agent: &mut Agent, correct: bool) {
        let change_rate = self.stake_percentage * self.confidence_multiplier;

        for domain in &self.domains {
            if let Some(score) = agent.reputation.domains.get_mut(domain) {
                if correct {
                    // Correct: reputation increases
                    *score *= 1.0 + change_rate;
                } else {
                    // Incorrect: reputation decreases
                    *score *= 1.0 - change_rate;
                }
                // Clamp to valid range
                *score = score.clamp(0.0, 100.0);
            }
        }
    }
}
```

### Commitment Stakes

```typescript
interface CommitmentStake {
  ifCorrect: Commitment[];
  ifWrong: Commitment[];
  witnesses: AgentPubKey[];
  enforcementMechanism: EnforcementType;
}

interface Commitment {
  action: string;
  deadline: Timestamp;
  verificationMethod: VerificationMethod;
  penalty?: Penalty;
}

enum EnforcementType {
  SocialOnly,         // Witnesses can see and judge
  ReputationBacked,   // Failure affects reputation
  MonetaryBacked,     // Escrowed funds at risk
  SmartContract,      // Automatic enforcement
}
```

### Legacy Stakes

For long-term predictions:

```rust
pub struct LegacyStake {
    pub amount: u64,
    pub vesting_period: Duration,
    pub beneficiaries: Vec<Beneficiary>,
    pub wisdom_seed: WisdomSeed,
}

pub struct Beneficiary {
    pub agent: AgentPubKey,
    pub share: f64,
    pub relationship: String,  // "successor", "mentee", "institution"
}

impl LegacyStake {
    pub fn resolve(&self, outcome: &Outcome) -> LegacyResolution {
        // Legacy stakes vest over time
        let vested_amount = self.calculate_vested(now());

        // Distribute to beneficiaries
        let distributions: Vec<Distribution> = self.beneficiaries.iter()
            .map(|b| Distribution {
                recipient: b.agent.clone(),
                amount: (vested_amount as f64 * b.share) as u64,
            })
            .collect();

        // Wisdom seed becomes part of permanent record
        let wisdom_archived = archive_wisdom_seed(&self.wisdom_seed);

        LegacyResolution {
            distributions,
            wisdom_archived,
            outcome: outcome.clone(),
        }
    }
}
```

---

## Part VI: Fee Structure

### Market Fees

```rust
pub struct MarketFees {
    // Creation fee (flat + percentage of initial liquidity)
    pub creation: CreationFee {
        flat: 10,  // 10 HAM
        percentage: 0.01,  // 1% of initial liquidity
    },

    // Trading fee (on each prediction)
    pub trading: TradingFee {
        percentage: 0.002,  // 0.2% of stake
        min: 1,  // Minimum 1 HAM
        max: 1000,  // Maximum 1000 HAM
    },

    // Resolution fee (from total pool)
    pub resolution: ResolutionFee {
        percentage: 0.01,  // 1% of pool
        oracle_share: 0.70,  // 70% to oracles
        treasury_share: 0.30,  // 30% to treasury
    },
}
```

### Fee Distribution

```
Total Fees Collected
         │
         ├── 40% → Oracle Rewards Pool
         │         (incentivize honest resolution)
         │
         ├── 25% → Treasury
         │         (protocol development, grants)
         │
         ├── 15% → Wisdom Fund
         │         (reward wisdom seeds)
         │
         ├── 10% → Insurance Fund
         │         (cover edge cases, disputes)
         │
         └── 10% → Staking Rewards
                   (reward long-term holders)
```

### Fee Reduction Programs

```typescript
interface FeeReductions {
  // Volume discounts
  volumeDiscounts: {
    tiers: [
      { volume: 1000, discount: 0.1 },    // 10% off
      { volume: 10000, discount: 0.2 },   // 20% off
      { volume: 100000, discount: 0.3 },  // 30% off
    ],
  };

  // Reputation discounts
  reputationDiscounts: {
    highCalibration: 0.15,  // 15% off for calibration > 0.8
    highContribution: 0.1,  // 10% off for top contributors
  };

  // New user subsidies
  newUserSubsidies: {
    firstPredictions: 10,   // First 10 predictions fee-free
    duration: "30 days",
  };
}
```

---

## Part VII: Treasury Management

### Treasury Structure

```rust
pub struct Treasury {
    // Reserves
    pub operating_reserve: u64,    // 12 months operating costs
    pub emergency_reserve: u64,    // 6 months emergency fund
    pub development_fund: u64,     // Protocol development
    pub grants_fund: u64,          // Community grants
    pub insurance_fund: u64,       // Dispute resolution, edge cases

    // Governance
    pub spending_authority: SpendingAuthority,
    pub reporting_requirements: Vec<ReportingRequirement>,
}

pub struct SpendingAuthority {
    pub operations_team_limit: u64,  // Can spend up to X without approval
    pub parameter_committee_limit: u64,
    pub protocol_council_limit: u64,
    pub community_vote_threshold: u64,  // Above this requires vote
}
```

### Investment Policy

```typescript
interface InvestmentPolicy {
  // Conservative allocation
  allocation: {
    stablecoins: 0.60,      // 60% stable (DAI, USDC)
    bluechipCrypto: 0.20,   // 20% BTC/ETH
    yieldStrategies: 0.15,  // 15% conservative DeFi yield
    strategicHoldings: 0.05, // 5% ecosystem investments
  };

  // Risk limits
  riskLimits: {
    maxSinglePosition: 0.10,     // No more than 10% in one asset
    maxVolatileAllocation: 0.25, // No more than 25% in volatile assets
    minLiquidity: "7 days",      // Must be accessible within 7 days
  };

  // Governance
  governance: {
    policyChanges: "Protocol Council vote",
    largeAllocations: "Community vote for >5% changes",
    reporting: "Quarterly public reports",
  };
}
```

---

## Part VIII: Sustainability Model

### Revenue Sources

```
Revenue
├── Trading Fees (primary)
│   └── 0.2% on ~$10M monthly volume = $20K/month
│
├── Premium Features
│   ├── Advanced analytics: $10/month
│   ├── API access: $50/month
│   └── White-label licensing: custom
│
├── Enterprise Services
│   ├── Private markets
│   ├── Integration support
│   └── Custom resolution oracles
│
├── Treasury Yield
│   └── Conservative DeFi strategies: 3-5% APY
│
└── Grants and Donations
    └── Public goods funding, ecosystem grants
```

### Cost Structure

```
Costs
├── Infrastructure (30%)
│   ├── Holochain hosting
│   ├── Storage and bandwidth
│   └── Backup and redundancy
│
├── Development (35%)
│   ├── Core protocol development
│   ├── Security audits
│   └── Research
│
├── Operations (20%)
│   ├── Community management
│   ├── Support
│   └── Moderation
│
├── Legal & Compliance (10%)
│   ├── Legal counsel
│   ├── Regulatory engagement
│   └── Jurisdictional compliance
│
└── Reserve Building (5%)
    └── Building emergency reserves
```

### Path to Sustainability

```
Phase 1: Subsidized Growth (Year 1-2)
├── Fee waivers for new users
├── Grants fund development
├── Focus on user acquisition
└── Target: 10,000 active predictors

Phase 2: Partial Sustainability (Year 2-3)
├── Reduce subsidies gradually
├── Premium features launch
├── Enterprise pilots
└── Target: Break-even on operations

Phase 3: Full Sustainability (Year 3-5)
├── Self-sustaining fee revenue
├── Treasury growth
├── Community grants program
└── Target: 12-month operating reserve

Phase 4: Abundance (Year 5+)
├── Fund ecosystem development
├── Public goods grants
├── Research funding
└── Intergenerational endowment
```

---

## Part IX: Anti-Gaming Mechanisms

### Gaming Scenarios and Defenses

| Attack | Defense |
|--------|---------|
| Wash trading | Transaction analysis, economic penalties |
| Sybil attacks | MATL requirements, proof of personhood |
| Oracle manipulation | Multi-oracle, Byzantine detection |
| Insider trading | Delayed resolution, suspicious pattern detection |
| Confidence gaming | Proper scoring rules, calibration tracking |
| Pool manipulation | Dynamic fees, slippage protection |

### Proper Scoring Rules

```rust
// Brier score: proper scoring rule that incentivizes honest probabilities
pub fn brier_score(prediction: f64, outcome: bool) -> f64 {
    let actual = if outcome { 1.0 } else { 0.0 };
    (prediction - actual).powi(2)
}

// Log score: alternative proper scoring rule
pub fn log_score(prediction: f64, outcome: bool) -> f64 {
    if outcome {
        prediction.ln()
    } else {
        (1.0 - prediction).ln()
    }
}

// Both rules incentivize reporting true beliefs
// Claiming 0.9 when you believe 0.7 has negative expected value
```

### Economic Alignment Analysis

```typescript
interface IncentiveAnalysis {
  // For each action, analyze incentives
  analyzeAction(action: Action): IncentiveReport {
    return {
      honestPayoff: calculateHonestPayoff(action),
      dishonestPayoff: calculateDishonestPayoff(action),
      ratio: honestPayoff / dishonestPayoff,  // Should be > 1
      risks: identifyRisks(action),
      recommendations: generateRecommendations(action),
    };
  }
}

// Target: honest behavior should be at least 2x more profitable
// than any gaming strategy after accounting for risk
```

---

## Conclusion

Economics is not separate from epistemics—it is the substrate on which epistemic behavior grows.

When we get the economics right:
- Truth-seeking becomes profitable
- Gaming becomes unprofitable
- Contribution is rewarded
- Wisdom accumulates

When we get it wrong:
- Gaming crowds out truth-seeking
- Short-term extraction dominates
- Community erodes
- The system dies

This economic design is itself a prediction: we believe these incentives will produce truth-seeking behavior. We will measure, learn, and adapt.

The goal is not to extract value, but to cultivate it. The economy should be a garden, not a mine.

---

*"The love of money is the root of all evil."*
*— 1 Timothy 6:10*

*The love of truth, properly incentivized, is the root of collective wisdom.*
*May our economics serve our epistemics.*

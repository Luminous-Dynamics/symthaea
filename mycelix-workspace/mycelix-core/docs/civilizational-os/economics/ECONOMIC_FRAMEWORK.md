# Mycelix Economic Framework

## Overview

The Mycelix Economic Framework provides a comprehensive approach to value creation, exchange, and distribution within the civilizational OS. Drawing from regenerative economics, commons theory, game theory, and behavioral economics, this framework ensures that economic activity serves human flourishing and ecological health.

**Core Principle**: Economics as a tool for coordination, not extraction.

---

## Theoretical Foundations

### Regenerative Economics (Kate Raworth's Doughnut)

```
                    ┌─────────────────────────────────┐
                    │      ECOLOGICAL CEILING         │
                    │  (Planetary Boundaries)         │
                    │                                 │
                    │  Climate │ Ocean │ Biodiversity │
                    │  Land │ Water │ Air │ Chemicals │
                    └─────────────────────────────────┘
                              │
                    ┌─────────┴─────────┐
                    │                   │
                    │   SAFE AND JUST   │
                    │   SPACE FOR       │
                    │   HUMANITY        │
                    │                   │
                    │   (The Doughnut)  │
                    │                   │
                    └─────────┬─────────┘
                              │
                    ┌─────────────────────────────────┐
                    │      SOCIAL FOUNDATION          │
                    │  (Minimum Thriving)             │
                    │                                 │
                    │  Food │ Health │ Education      │
                    │  Income │ Voice │ Equity        │
                    │  Housing │ Networks │ Energy    │
                    └─────────────────────────────────┘
```

The Mycelix economy operates within the doughnut—above the social foundation (ensuring everyone's needs are met) and below the ecological ceiling (respecting planetary boundaries).

### Commons Economics (Elinor Ostrom)

Ostrom's 8 principles for governing commons, implemented in Mycelix:

| Principle | Mycelix Implementation |
|-----------|------------------------|
| 1. Clearly defined boundaries | Membrane hApp, Attest identity |
| 2. Match rules to local needs | Agora customization, community governance |
| 3. Collective choice arrangements | Agora participation, stake-based voice |
| 4. Monitoring | Sentinel, MATL, Chronicle |
| 5. Graduated sanctions | Arbiter escalation, trust score impact |
| 6. Conflict resolution | Arbiter, restorative processes |
| 7. Recognition of rights to organize | Community sovereignty, federation |
| 8. Nested enterprises | Multi-scale governance, bioregional coordination |

### Game Theory for Cooperation

**Key Mechanisms**:
- **Repeated games**: Long-term relationships change incentives
- **Reputation**: MATL makes past behavior visible
- **Conditional cooperation**: Reciprocity tracking
- **Punishment and forgiveness**: Graduated responses with rehabilitation
- **Communication**: Chronicle, Mail enable cheap talk

---

## Multi-Currency Architecture

### Currency Types

Mycelix supports multiple currency types to serve different economic functions:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        CURRENCY ECOSYSTEM                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐│
│  │   MUTUAL    │  │    TIME     │  │  COMMUNITY  │  │  RESOURCE   ││
│  │   CREDIT    │  │   BANKING   │  │   TOKEN     │  │   CREDITS   ││
│  │             │  │             │  │             │  │             ││
│  │ Zero-sum    │  │ 1 hour = 1  │  │ Governance  │  │ Represent   ││
│  │ exchange    │  │ hour for    │  │ and value   │  │ ecological  ││
│  │ credit      │  │ everyone    │  │ capture     │  │ capacity    ││
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘│
│         │                │                │                │        │
│         └────────────────┼────────────────┼────────────────┘        │
│                          │                │                         │
│                    ┌─────┴────────────────┴─────┐                   │
│                    │      TREASURY LAYER        │                   │
│                    │  (Multi-currency reserves) │                   │
│                    └────────────────────────────┘                   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 1. Mutual Credit

**Purpose**: Enable exchange without pre-existing money supply

```rust
#[hdk_entry_helper]
pub struct MutualCreditAccount {
    pub account_id: String,
    pub holder: AgentPubKey,
    pub balance: i64,  // Can be negative (credit) or positive (debit)
    pub credit_limit: i64,  // Maximum negative balance
    pub debit_limit: i64,   // Maximum positive balance (optional)
    pub community_id: String,
}

#[hdk_entry_helper]
pub struct MutualCreditTransfer {
    pub transfer_id: String,
    pub from: AgentPubKey,
    pub to: AgentPubKey,
    pub amount: u64,
    pub memo: String,
    pub timestamp: Timestamp,

    // Result: from.balance -= amount, to.balance += amount
    // System always sums to zero
}
```

**Properties**:
- Zero-sum: Total system balance always equals zero
- Credit creation: Spending creates credit for buyer, debt for seller
- Trust-based limits: Credit limits based on MATL trust scores
- No interest: No compound growth, sustainable

**Example**:
```
Alice: 0 balance, -500 credit limit
Bob: 0 balance, -500 credit limit

Alice buys 100 units of service from Bob:
Alice: -100 balance
Bob: +100 balance
System total: 0

Bob buys 50 units of goods from Carol:
Alice: -100 balance
Bob: +50 balance
Carol: +50 balance
System total: 0
```

### 2. Time Banking

**Purpose**: Value all human time equally, enable non-monetary exchange

```rust
#[hdk_entry_helper]
pub struct TimeBankAccount {
    pub account_id: String,
    pub holder: AgentPubKey,
    pub balance_hours: Decimal,  // Hours of time credit
    pub earned_total: Decimal,
    pub spent_total: Decimal,
    pub community_id: String,
}

#[hdk_entry_helper]
pub struct TimeExchange {
    pub exchange_id: String,
    pub provider: AgentPubKey,    // Person giving time
    pub receiver: AgentPubKey,    // Person receiving service
    pub hours: Decimal,
    pub service_type: String,
    pub description: String,
    pub timestamp: Timestamp,

    // 1 hour of anyone's time = 1 hour
    // Doctor's hour = Gardener's hour
}

#[hdk_entry_helper]
pub struct ServiceOffer {
    pub offer_id: String,
    pub provider: AgentPubKey,
    pub service_type: String,
    pub description: String,
    pub availability: Availability,
    pub skills: Vec<String>,
}
```

**Properties**:
- Radical equality: Everyone's time valued equally
- Community building: Creates social connections
- Complementary: Works alongside other currencies
- Anti-inflationary: Supply tied to actual human capacity

### 3. Community Tokens

**Purpose**: Governance rights, value capture, community investment

```rust
#[hdk_entry_helper]
pub struct CommunityToken {
    pub token_id: String,
    pub community_id: String,
    pub name: String,
    pub symbol: String,
    pub total_supply: u64,
    pub distribution_rules: DistributionRules,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DistributionRules {
    pub initial_distribution: InitialDistribution,
    pub ongoing_issuance: OngoingIssuance,
    pub burn_mechanisms: Vec<BurnMechanism>,
    pub governance_weight: GovernanceWeight,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum InitialDistribution {
    // Equal to all founding members
    EqualFounders { amount_per_founder: u64 },

    // Based on contribution
    ContributionBased { formula: ContributionFormula },

    // Gradual release over time
    VestingSchedule { schedule: VestingSchedule },

    // Combination
    Hybrid { allocations: Vec<Allocation> },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum OngoingIssuance {
    // Fixed inflation rate
    FixedInflation { rate_per_year: Decimal },

    // Based on contribution
    ContributionMining { formula: ContributionFormula },

    // Demurrage (negative interest)
    Demurrage { rate_per_year: Decimal },

    // No new issuance
    Fixed,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum GovernanceWeight {
    // 1 token = 1 vote
    Linear,

    // Quadratic voting
    Quadratic,

    // Capped influence
    Capped { max_weight: u64 },

    // Time-weighted (longer holders have more weight)
    TimeWeighted { half_life: Duration },

    // Conviction voting
    Conviction { decay_rate: Decimal },
}
```

**Properties**:
- Customizable: Each community defines their own rules
- Governance-linked: Token holdings affect voting power
- Value capture: Can appreciate based on community success
- Exit mechanism: Can be exchanged for resources or external currency

### 4. Resource Credits

**Purpose**: Represent and manage ecological capacity

```rust
#[hdk_entry_helper]
pub struct ResourceCredit {
    pub credit_id: String,
    pub resource_type: ResourceType,
    pub amount: Decimal,
    pub unit: String,
    pub source: ResourceSource,
    pub verification: Verification,
    pub expiration: Option<Timestamp>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ResourceType {
    // Energy
    Electricity { source: EnergySource },
    Heat { source: HeatSource },

    // Water
    FreshWater { quality: WaterQuality },
    Greywater,

    // Carbon
    CarbonOffset { verification: CarbonVerification },
    CarbonDebit,

    // Land
    LandUseRights { parcel_id: String },
    AgriculturalCapacity,

    // Ecosystem services
    BiodiversityCredit,
    WatershedService,
    PollinatorService,

    // Materials
    RecycledMaterial { material_type: String },
    RenewableMaterial { material_type: String },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CommunityResourceBudget {
    pub community_id: String,
    pub budget_period: Period,
    pub allocations: HashMap<ResourceType, ResourceAllocation>,
    pub current_usage: HashMap<ResourceType, Decimal>,
    pub projections: HashMap<ResourceType, Vec<Projection>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ResourceAllocation {
    pub total_budget: Decimal,
    pub allocated: Decimal,
    pub remaining: Decimal,
    pub allocation_method: AllocationMethod,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum AllocationMethod {
    EqualPerCapita,
    NeedsBased,
    AuctionBased,
    GovernanceDecided,
    HybridFormula { formula: String },
}
```

**Properties**:
- Ecological grounding: Tied to real resource capacity
- Tradeable: Can be exchanged between community members
- Budget-based: Communities have collective resource budgets
- Regenerative: Rewards restoration, penalizes extraction

---

## Incentive Design

### Game-Theoretic Analysis

#### Cooperation Mechanisms

```rust
/// Incentive structure for cooperative behavior
pub struct CooperationIncentives {
    /// Positive incentives
    pub rewards: CooperationRewards,

    /// Disincentives for defection
    pub penalties: DefectionPenalties,

    /// Reputation effects
    pub reputation_impacts: ReputationImpacts,

    /// Social mechanisms
    pub social_mechanisms: SocialMechanisms,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CooperationRewards {
    // Direct rewards for cooperative actions
    pub contribution_rewards: ContributionRewardSchedule,

    // Matching funds for cooperative projects
    pub matching_pools: Vec<MatchingPool>,

    // Reputation bonuses
    pub trust_score_bonuses: TrustBonusSchedule,

    // Access to shared resources
    pub commons_access: CommonsAccessRules,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DefectionPenalties {
    // Trust score reduction
    pub trust_penalty: TrustPenaltySchedule,

    // Temporary restrictions
    pub cooldown_periods: CooldownSchedule,

    // Economic penalties
    pub economic_penalties: EconomicPenaltySchedule,

    // Rehabilitation pathway
    pub rehabilitation: RehabilitationProgram,
}
```

#### Nash Equilibria Analysis

**Key Equilibria in Mycelix**:

1. **Trust Building Game**
```
              Cooperate    Defect
           ┌───────────┬───────────┐
Cooperate  │  (3, 3)   │  (0, 4)   │
           ├───────────┼───────────┤
Defect     │  (4, 0)   │  (1, 1)   │
           └───────────┴───────────┘

Without reputation: Defect-Defect is Nash equilibrium
With MATL reputation tracking:
- Repeated game shifts payoffs
- Cooperate-Cooperate becomes stable equilibrium
- Defection carries future cost (reduced trust)
```

2. **Public Goods Provision**
```
Traditional: Free-rider problem (underprovision)

Mycelix solution:
- Visibility: Chronicle records contributions
- Reputation: MATL rewards contributors
- Matching: Treasury provides matching funds
- Social proof: Nudge shows contribution norms

Result: Shifted equilibrium toward provision
```

3. **Commons Management**
```
Tragedy of commons: Overexploitation equilibrium

Mycelix solution:
- Monitoring: Sentinel tracks usage
- Boundaries: Membrane controls access
- Graduated sanctions: Arbiter enforces limits
- Local rules: Agora enables community management

Result: Sustainable use equilibrium (Ostrom)
```

### Behavioral Incentive Integration

Connecting to Nudge hApp for behavioral design:

```rust
/// Incentive-Nudge integration
pub struct IncentiveNudgeConfig {
    /// How to present incentives
    pub framing: IncentiveFraming,

    /// When to show incentives
    pub timing: IncentiveTiming,

    /// Social comparison
    pub social_proof: SocialProofConfig,

    /// Commitment devices
    pub commitments: CommitmentConfig,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum IncentiveFraming {
    // Frame as potential gain
    GainFrame { message: String },

    // Frame as potential loss (more powerful)
    LossFrame { message: String },

    // Frame as social norm
    SocialNorm { comparison_group: String },

    // Frame as identity-consistent
    IdentityFrame { identity_aspect: String },

    // Stage-appropriate framing
    StageAdapted {
        traditional: String,  // "Your duty..."
        modern: String,       // "Your return..."
        postmodern: String,   // "Community benefit..."
        integral: String,     // "Systems impact..."
    },
}
```

### Contribution Recognition

```rust
/// Multi-dimensional contribution tracking
#[hdk_entry_helper]
pub struct ContributionRecord {
    pub record_id: String,
    pub contributor: AgentPubKey,
    pub contribution_type: ContributionType,
    pub value_dimensions: Vec<ValueDimension>,
    pub timestamp: Timestamp,
    pub verification: ContributionVerification,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ContributionType {
    // Economic contributions
    Financial { amount: Decimal, currency: String },
    Labor { hours: Decimal, skill_level: SkillLevel },
    Assets { asset_type: String, value: Decimal },

    // Social contributions
    CareWork { type_: CareType, hours: Decimal },
    CommunityBuilding { activity: String },
    MentorshipGiven { mentee: AgentPubKey, hours: Decimal },

    // Governance contributions
    ProposalCreated { quality_score: f32 },
    VotingParticipation { proposal_id: String },
    FacilitationProvided { event_id: String },

    // Knowledge contributions
    ContentCreated { content_id: String },
    EducationProvided { topic: String, hours: Decimal },
    ResearchContributed { project_id: String },

    // Ecological contributions
    RegenerationWork { type_: RegenerationType },
    ResourceConservation { resource: ResourceType, amount: Decimal },
    WasteReduction { type_: String, amount: Decimal },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ValueDimension {
    pub dimension: String,  // economic, social, ecological, knowledge
    pub value: Decimal,
    pub unit: String,
    pub multiplier: Decimal,  // Community can adjust weights
}

/// Calculate contribution score across dimensions
pub fn calculate_contribution_score(
    contributions: Vec<ContributionRecord>,
    weights: &CommunityWeights,
) -> ContributionScore {
    let mut scores = HashMap::new();

    for contribution in contributions {
        for dimension in contribution.value_dimensions {
            let weight = weights.get(&dimension.dimension).unwrap_or(&1.0);
            let score = dimension.value * dimension.multiplier * weight;
            *scores.entry(dimension.dimension).or_insert(0.0) += score;
        }
    }

    ContributionScore {
        total: scores.values().sum(),
        by_dimension: scores,
        percentile: calculate_percentile(&scores),
    }
}
```

---

## Distribution Mechanisms

### Income Distribution

```rust
/// Universal Basic Income (community-defined)
#[hdk_entry_helper]
pub struct CommunityUBI {
    pub community_id: String,
    pub name: String,  // e.g., "Community Dividend", "Basic Share"

    pub amount: Decimal,
    pub currency: CurrencyType,
    pub frequency: Frequency,  // Weekly, Monthly, etc.

    pub eligibility: UBIEligibility,
    pub funding_source: UBIFunding,

    pub started: Timestamp,
    pub modified_history: Vec<UBIModification>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum UBIEligibility {
    // All verified community members
    Universal { minimum_membership_duration: Duration },

    // Needs-based adjustment
    NeedsAdjusted { base_amount: Decimal, needs_formula: String },

    // Participation-linked
    ParticipationLinked { minimum_participation: ParticipationRequirement },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum UBIFunding {
    // From community token inflation
    TokenInflation { rate: Decimal },

    // From transaction fees
    TransactionFees { fee_percentage: Decimal },

    // From commons rent (Georgist)
    CommonsRent { rent_sources: Vec<RentSource> },

    // From external sources
    ExternalFunding { sources: Vec<FundingSource> },

    // Combination
    Hybrid { allocations: Vec<FundingAllocation> },
}
```

### Quadratic Funding

For public goods and community projects:

```rust
/// Quadratic funding pool
#[hdk_entry_helper]
pub struct QuadraticFundingRound {
    pub round_id: String,
    pub community_id: String,

    pub matching_pool: Decimal,
    pub currency: CurrencyType,

    pub start_time: Timestamp,
    pub end_time: Timestamp,

    pub eligible_projects: Vec<String>,
    pub contributions: Vec<QFContribution>,

    pub status: RoundStatus,
    pub final_distribution: Option<HashMap<String, Decimal>>,
}

#[hdk_entry_helper]
pub struct QFContribution {
    pub contribution_id: String,
    pub contributor: AgentPubKey,
    pub project_id: String,
    pub amount: Decimal,
    pub timestamp: Timestamp,
}

/// Calculate quadratic funding distribution
pub fn calculate_qf_distribution(
    round: &QuadraticFundingRound,
) -> HashMap<String, Decimal> {
    let mut project_scores: HashMap<String, Decimal> = HashMap::new();

    // Group contributions by project
    let by_project = group_by_project(&round.contributions);

    for (project_id, contributions) in by_project {
        // Sum of square roots of individual contributions
        let sum_sqrt: Decimal = contributions
            .iter()
            .map(|c| c.amount.sqrt())
            .sum();

        // Square the sum (quadratic formula)
        let score = sum_sqrt * sum_sqrt;
        project_scores.insert(project_id, score);
    }

    // Normalize to matching pool
    let total_score: Decimal = project_scores.values().sum();
    let mut distribution = HashMap::new();

    for (project_id, score) in project_scores {
        let share = (score / total_score) * round.matching_pool;
        distribution.insert(project_id, share);
    }

    distribution
}
```

### Resource Allocation

```rust
/// Multi-mechanism resource allocation
pub struct ResourceAllocationSystem {
    pub community_id: String,

    /// Different allocation mechanisms for different resources
    pub mechanisms: HashMap<ResourceType, AllocationMechanism>,

    /// Override governance
    pub governance_override: GovernanceOverride,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum AllocationMechanism {
    // Equal distribution
    EqualPerCapita,

    // Need-based (via Sanctuary/HealthVault data)
    NeedsBased { needs_assessment: NeedsAssessment },

    // Merit-based (via contribution tracking)
    MeritBased { contribution_weights: ContributionWeights },

    // Market-based (auction/exchange)
    MarketBased { market_rules: MarketRules },

    // Lottery (equal random chance)
    Lottery { frequency: Frequency },

    // Queue-based (first come, first served)
    QueueBased { priority_rules: PriorityRules },

    // Deliberative (governance decision)
    Deliberative { decision_process: DecisionProcess },

    // Hybrid (weighted combination)
    Hybrid { weights: HashMap<String, Decimal>, mechanisms: Vec<AllocationMechanism> },
}

/// Example: Housing allocation (Terroir)
pub fn allocate_housing(
    available_units: Vec<HousingUnit>,
    applicants: Vec<HousingApplicant>,
    mechanism: &AllocationMechanism,
) -> Vec<HousingAllocation> {
    match mechanism {
        AllocationMechanism::NeedsBased { needs_assessment } => {
            // Score applicants by need
            let mut scored: Vec<(HousingApplicant, Decimal)> = applicants
                .iter()
                .map(|a| (a.clone(), assess_housing_need(a, needs_assessment)))
                .collect();

            // Sort by need (highest first)
            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            // Allocate in order
            allocate_in_order(available_units, scored)
        }
        AllocationMechanism::Hybrid { weights, mechanisms } => {
            // Combine multiple scores
            let mut combined_scores: HashMap<AgentPubKey, Decimal> = HashMap::new();

            for (mechanism_name, weight) in weights {
                let mechanism = mechanisms.get(mechanism_name).unwrap();
                let scores = calculate_mechanism_scores(applicants, mechanism);

                for (agent, score) in scores {
                    *combined_scores.entry(agent).or_insert(Decimal::ZERO) += score * weight;
                }
            }

            // Allocate by combined score
            allocate_by_score(available_units, combined_scores)
        }
        // ... other mechanisms
    }
}
```

---

## Treasury Management

### Multi-Pool Treasury

```rust
#[hdk_entry_helper]
pub struct CommunityTreasury {
    pub treasury_id: String,
    pub community_id: String,

    /// Multiple pools for different purposes
    pub pools: Vec<TreasuryPool>,

    /// Governance rules
    pub governance: TreasuryGovernance,

    /// Investment policy
    pub investment_policy: InvestmentPolicy,

    /// Transparency settings
    pub transparency: TransparencySettings,
}

#[hdk_entry_helper]
pub struct TreasuryPool {
    pub pool_id: String,
    pub name: String,
    pub purpose: PoolPurpose,

    pub balances: HashMap<CurrencyType, Decimal>,
    pub target_balance: Option<Decimal>,

    pub inflows: Vec<InflowRule>,
    pub outflows: Vec<OutflowRule>,

    pub governance_threshold: GovernanceThreshold,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum PoolPurpose {
    // Operating expenses
    Operations { budget_period: Period },

    // Emergency reserve
    EmergencyReserve { target_months: u32 },

    // Community UBI funding
    UBIFund,

    // Matching pool for quadratic funding
    MatchingPool { round_frequency: Frequency },

    // Investment for future growth
    Investment { strategy: InvestmentStrategy },

    // Insurance/mutual aid
    MutualAid { coverage_rules: CoverageRules },

    // Regeneration fund
    Regeneration { focus_areas: Vec<String> },

    // Inter-community solidarity
    Solidarity { partner_communities: Vec<String> },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InflowRule {
    pub source: InflowSource,
    pub percentage: Decimal,  // Percentage of source going to this pool
    pub conditions: Vec<Condition>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum InflowSource {
    TransactionFees,
    MembershipDues,
    CommonsRent,
    ExternalGrants,
    InvestmentReturns,
    TokenSales,
    ServiceRevenue,
    Donations,
}
```

### Automated Treasury Operations

```rust
/// Automated treasury functions
pub struct TreasuryAutomation {
    /// Automatic rebalancing between pools
    pub rebalancing: RebalancingRules,

    /// Automatic distributions (UBI, dividends)
    pub distributions: Vec<AutomaticDistribution>,

    /// Automatic investments
    pub investments: Vec<AutomaticInvestment>,

    /// Emergency triggers
    pub emergency_triggers: Vec<EmergencyTrigger>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RebalancingRules {
    pub enabled: bool,
    pub frequency: Frequency,
    pub target_allocations: HashMap<String, Decimal>,  // pool_id -> percentage
    pub tolerance: Decimal,  // Rebalance if deviation exceeds this
    pub max_single_move: Decimal,  // Max percentage to move in one rebalance
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AutomaticDistribution {
    pub name: String,
    pub source_pool: String,
    pub distribution_type: DistributionType,
    pub frequency: Frequency,
    pub conditions: Vec<Condition>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum DistributionType {
    UBI { eligibility: UBIEligibility },
    Dividend { per_token: Decimal },
    ContributionReward { formula: ContributionFormula },
    MatchingDistribution { round_id: String },
}
```

---

## Economic Governance

### Parameter Governance

All economic parameters are governable through Agora:

```rust
/// Governable economic parameters
pub struct EconomicParameters {
    // Currency parameters
    pub mutual_credit_limits: MutualCreditLimits,
    pub token_issuance_rate: Decimal,
    pub demurrage_rate: Option<Decimal>,

    // Fee parameters
    pub transaction_fee_rate: Decimal,
    pub fee_distribution: FeeDistribution,

    // Distribution parameters
    pub ubi_amount: Decimal,
    pub matching_pool_allocation: Decimal,

    // Allocation weights
    pub contribution_weights: HashMap<String, Decimal>,
    pub resource_allocation_rules: HashMap<ResourceType, AllocationMechanism>,

    // Treasury parameters
    pub pool_target_balances: HashMap<String, Decimal>,
    pub investment_risk_tolerance: RiskTolerance,
}

/// Economic parameter change proposal
#[hdk_entry_helper]
pub struct EconomicProposal {
    pub proposal_id: String,
    pub proposer: AgentPubKey,

    pub parameter_changes: Vec<ParameterChange>,
    pub rationale: String,
    pub impact_analysis: ImpactAnalysis,

    pub simulation_results: Option<SimulationResults>,

    pub voting_period: Duration,
    pub required_quorum: Decimal,
    pub required_majority: Decimal,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ParameterChange {
    pub parameter_path: String,  // e.g., "ubi_amount", "transaction_fee_rate"
    pub current_value: Value,
    pub proposed_value: Value,
    pub change_type: ChangeType,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ChangeType {
    Immediate,
    Gradual { transition_period: Duration },
    Conditional { conditions: Vec<Condition> },
}
```

### Economic Simulation

Before implementing changes, run simulations:

```rust
/// Economic simulation system
pub struct EconomicSimulator {
    pub community_snapshot: CommunitySnapshot,
    pub agent_models: Vec<AgentModel>,
    pub external_conditions: ExternalConditions,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SimulationConfig {
    pub duration: Duration,
    pub time_steps: u32,
    pub monte_carlo_runs: u32,
    pub parameter_variations: Vec<ParameterVariation>,
    pub metrics_to_track: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SimulationResults {
    pub config: SimulationConfig,

    pub outcomes: Vec<SimulationOutcome>,

    pub metrics: HashMap<String, MetricTimeSeries>,

    pub risk_analysis: RiskAnalysis,

    pub recommendations: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SimulationOutcome {
    pub scenario: String,
    pub probability: Decimal,
    pub key_metrics: HashMap<String, Decimal>,
    pub warnings: Vec<String>,
}

/// Run economic simulation
pub async fn simulate_parameter_change(
    change: &ParameterChange,
    config: &SimulationConfig,
) -> SimulationResults {
    let mut outcomes = Vec::new();

    for _ in 0..config.monte_carlo_runs {
        let outcome = run_single_simulation(change, config).await;
        outcomes.push(outcome);
    }

    // Aggregate results
    let metrics = aggregate_metrics(&outcomes);
    let risk_analysis = analyze_risks(&outcomes);
    let recommendations = generate_recommendations(&metrics, &risk_analysis);

    SimulationResults {
        config: config.clone(),
        outcomes,
        metrics,
        risk_analysis,
        recommendations,
    }
}
```

---

## External Integration

### Fiat Currency Bridges

```rust
/// Bridge to external financial systems
pub struct FiatBridge {
    pub bridge_id: String,
    pub supported_currencies: Vec<FiatCurrency>,
    pub providers: Vec<PaymentProvider>,
    pub compliance: ComplianceConfig,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum PaymentProvider {
    BankTransfer { supported_countries: Vec<String> },
    CreditCard { processor: String },
    MobilePayment { services: Vec<String> },
    Cryptocurrency { chains: Vec<String> },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ComplianceConfig {
    pub kyc_required: bool,
    pub kyc_provider: Option<String>,
    pub aml_screening: bool,
    pub transaction_limits: TransactionLimits,
    pub reporting_requirements: Vec<ReportingRequirement>,
}

/// On-ramp: Fiat -> Community currency
pub async fn fiat_to_community(
    input: FiatDepositInput,
) -> Result<CommunityDeposit, BridgeError> {
    // Verify compliance
    verify_compliance(&input.depositor, &input.amount)?;

    // Process fiat payment
    let payment_result = process_fiat_payment(&input).await?;

    // Mint community currency
    let community_amount = calculate_exchange_rate(&input.fiat_currency, &input.amount)?;
    let deposit = mint_community_currency(&input.depositor, community_amount)?;

    // Record in Chronicle
    record_fiat_bridge_transaction(&input, &deposit)?;

    Ok(deposit)
}

/// Off-ramp: Community currency -> Fiat
pub async fn community_to_fiat(
    input: FiatWithdrawalInput,
) -> Result<FiatWithdrawal, BridgeError> {
    // Verify compliance
    verify_compliance(&input.withdrawer, &input.amount)?;

    // Burn community currency
    burn_community_currency(&input.withdrawer, &input.amount)?;

    // Process fiat payout
    let fiat_amount = calculate_exchange_rate(&input.target_currency, &input.amount)?;
    let payout_result = process_fiat_payout(&input.withdrawer, fiat_amount).await?;

    // Record in Chronicle
    record_fiat_bridge_transaction(&input, &payout_result)?;

    Ok(payout_result)
}
```

### Inter-Community Exchange

```rust
/// Exchange between different Mycelix communities
pub struct InterCommunityExchange {
    pub exchange_id: String,
    pub communities: Vec<String>,
    pub exchange_rates: HashMap<(String, String), Decimal>,
    pub liquidity_pools: Vec<LiquidityPool>,
}

#[hdk_entry_helper]
pub struct CrossCommunityTransfer {
    pub transfer_id: String,
    pub from_community: String,
    pub to_community: String,
    pub from_agent: AgentPubKey,
    pub to_agent: AgentPubKey,
    pub from_amount: Decimal,
    pub to_amount: Decimal,
    pub exchange_rate: Decimal,
    pub timestamp: Timestamp,
}

/// Federated economic cooperation
pub struct EconomicFederation {
    pub federation_id: String,
    pub member_communities: Vec<String>,

    pub shared_currency: Option<FederationCurrency>,
    pub trade_agreements: Vec<TradeAgreement>,
    pub mutual_aid_pool: MutualAidPool,
    pub solidarity_fund: SolidarityFund,
}
```

---

## Metrics and Monitoring

### Economic Health Indicators

```rust
/// Community economic health dashboard
pub struct EconomicHealthDashboard {
    pub community_id: String,
    pub timestamp: Timestamp,

    // Circulation metrics
    pub velocity: CurrencyVelocity,
    pub circulation_distribution: CirculationDistribution,

    // Equality metrics
    pub gini_coefficient: Decimal,
    pub palma_ratio: Decimal,

    // Activity metrics
    pub transaction_volume: TransactionVolume,
    pub active_participants: u64,
    pub new_participants: u64,

    // Health metrics
    pub liquidity_ratio: Decimal,
    pub treasury_health: TreasuryHealth,
    pub debt_levels: DebtLevels,

    // Doughnut metrics
    pub social_foundation_scores: HashMap<String, Decimal>,
    pub ecological_ceiling_scores: HashMap<String, Decimal>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CurrencyVelocity {
    pub mutual_credit: Decimal,
    pub time_bank: Decimal,
    pub community_token: Decimal,
    pub resource_credits: HashMap<ResourceType, Decimal>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CirculationDistribution {
    pub top_10_percent_share: Decimal,
    pub bottom_50_percent_share: Decimal,
    pub median_balance: Decimal,
    pub concentration_index: Decimal,
}
```

### Alerts and Interventions

```rust
/// Economic health monitoring and intervention
pub struct EconomicMonitor {
    pub alert_thresholds: AlertThresholds,
    pub automatic_interventions: Vec<AutomaticIntervention>,
    pub notification_rules: NotificationRules,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AlertThresholds {
    pub gini_warning: Decimal,        // e.g., 0.35
    pub gini_critical: Decimal,       // e.g., 0.45
    pub velocity_low: Decimal,        // Currency not circulating
    pub velocity_high: Decimal,       // Possible speculation
    pub treasury_low: Decimal,        // Reserves depleted
    pub participation_declining: Decimal,  // Engagement dropping
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum AutomaticIntervention {
    // Increase UBI if inequality rising
    AdjustUBI { trigger: Trigger, adjustment: Adjustment },

    // Activate demurrage if velocity too low
    ActivateDemurrage { trigger: Trigger, rate: Decimal },

    // Emergency treasury measures
    TreasuryEmergency { trigger: Trigger, actions: Vec<EmergencyAction> },

    // Notify governance for manual intervention
    GovernanceAlert { trigger: Trigger, proposal_template: String },
}
```

---

## Conclusion

The Mycelix Economic Framework provides a comprehensive, regenerative approach to community economics. By combining multiple currency types, thoughtful incentive design, transparent governance, and ecological grounding, communities can create economies that serve human flourishing rather than extractive accumulation.

**Key Principles**:
1. Multiple currencies for different functions
2. Incentives aligned with cooperation
3. Transparent, governable parameters
4. Ecological limits respected
5. Simulation before implementation
6. Continuous monitoring and adaptation

---

*"An economy is not a force of nature; it is a designed system. Let us design it for life."*

---

*Document Version: 1.0*
*Last Updated: 2025*
*Theoretical Basis: Kate Raworth (Doughnut Economics), Elinor Ostrom (Commons), Bernard Lietaer (Complementary Currencies)*

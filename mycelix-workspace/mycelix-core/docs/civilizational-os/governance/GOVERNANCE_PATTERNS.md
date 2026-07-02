# Mycelix Governance Patterns Library

## Overview

This library provides a comprehensive catalog of governance patterns that communities can adopt, adapt, and combine within Mycelix. Rather than prescribing a single model, Mycelix enables communities to choose governance structures that match their values, size, and developmental stage.

**Core Principle**: Governance should emerge from community needs, not be imposed upon them.

---

## Governance Pattern Selection Framework

### Choosing Your Governance Model

```
┌─────────────────────────────────────────────────────────────────────────┐
│                 GOVERNANCE SELECTION MATRIX                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│                        DECISION SPEED                                   │
│                    Fast ←─────────────→ Slow                           │
│                                                                         │
│         ┌───────────────────┬───────────────────┐                      │
│         │                   │                   │                      │
│         │  Representative   │   Consensus       │                      │
│  High   │  Democracy        │   (Traditional)   │                      │
│         │                   │                   │                      │
│  Scale  │  Liquid Democracy │   Sociocracy      │                      │
│         │                   │                   │                      │
│         ├───────────────────┼───────────────────┤                      │
│         │                   │                   │                      │
│         │  Benevolent       │   Consent-Based   │                      │
│  Low    │  Dictatorship     │   Circles         │                      │
│         │  (Startup phase)  │                   │                      │
│         │                   │   Direct          │                      │
│         │  Holacracy        │   Democracy       │                      │
│         │                   │                   │                      │
│         └───────────────────┴───────────────────┘                      │
│                                                                         │
│  Additional Factors:                                                   │
│  • Developmental stage diversity                                       │
│  • Decision reversibility requirements                                 │
│  • Trust levels in community                                           │
│  • Technical sophistication                                            │
│  • Cultural context                                                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Pattern 1: Consensus Democracy

### Overview

Traditional consensus requires all participants to consent to a decision before it is adopted. This creates high buy-in but can be slow and vulnerable to blocking.

**Best For**: Small groups (5-30), high-trust communities, reversible decisions

### Agora Configuration

```toml
[governance.consensus]
name = "Full Consensus"
model = "consensus"

[governance.consensus.thresholds]
# Unanimous consent required
consent_threshold = 1.0
# But distinguish consent from enthusiasm
consent_levels = ["block", "stand_aside", "consent", "support", "champion"]

[governance.consensus.blocking]
# Blocks must be principled (not preference)
block_requires_reason = true
# Blocks trigger mandatory discussion
block_triggers_dialogue = true
# Persistent blocks escalate to mediation
persistent_block_escalation = "arbiter"

[governance.consensus.timing]
# Discussion period before voting
discussion_period = "7d"
# Consensus check period
consensus_period = "3d"
# Maximum extension for working through blocks
max_extensions = 2
extension_duration = "3d"

[governance.consensus.fallback]
# If consensus fails after extensions
fallback_enabled = true
fallback_threshold = 0.9  # 90% supermajority
fallback_requires_quorum = 0.75
```

### Decision Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    CONSENSUS DECISION FLOW                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  1. PROPOSAL                                                           │
│     └─→ Proposer drafts, seeks initial feedback                        │
│                                                                         │
│  2. CLARIFYING QUESTIONS                                               │
│     └─→ Community asks questions (no debate yet)                       │
│                                                                         │
│  3. DISCUSSION                                                          │
│     └─→ Open discussion, concerns raised                               │
│     └─→ Proposer may modify based on feedback                          │
│                                                                         │
│  4. CONSENSUS CHECK                                                     │
│     └─→ Facilitator asks: "Any blocks or stand-asides?"               │
│                                                                         │
│     ┌─ No blocks ──────────────────────────────→ DECISION ADOPTED     │
│     │                                                                  │
│     └─ Blocks exist ──→ 5. BLOCK RESOLUTION                           │
│                                                                         │
│  5. BLOCK RESOLUTION                                                   │
│     └─→ Blocker explains principled concern                           │
│     └─→ Group works to address concern                                │
│     └─→ Modified proposal or stand-aside                              │
│                                                                         │
│  6. FINAL CHECK                                                        │
│     └─→ If blocks remain after good-faith effort                      │
│     └─→ Fallback to supermajority (if configured)                     │
│     └─→ Or decision deferred                                          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Consent Levels

| Level | Meaning | Effect |
|-------|---------|--------|
| **Champion** | "I love this and will lead implementation" | Counts as consent, assigns responsibility |
| **Support** | "I think this is good" | Counts as consent |
| **Consent** | "I can live with this" | Counts as consent |
| **Stand Aside** | "I have concerns but won't block" | Noted but doesn't prevent adoption |
| **Block** | "This violates our principles/harms us" | Prevents adoption, triggers dialogue |

### Stage-Appropriate Adaptations

| Stage | Adaptation |
|-------|------------|
| Traditional | Emphasis on proper procedure, elder input weighted |
| Modern | Add efficiency metrics, time-boxing discussions |
| Postmodern | Extra attention to marginalized voices |
| Integral | Meta-awareness of process itself, flex when needed |

---

## Pattern 2: Sociocracy (Circle Governance)

### Overview

Sociocracy organizes governance into nested circles with defined domains. Decisions are made by consent within circles, and circles are linked through double-linking.

**Best For**: Medium organizations (30-500), clear functional divisions, ongoing operations

### Agora Configuration

```toml
[governance.sociocracy]
name = "Sociocratic Governance"
model = "sociocracy"

[governance.sociocracy.circles]
# Circle structure
general_circle = "community_wide"
department_circles = ["operations", "finance", "membership", "external"]
working_circles = ["garden", "kitchen", "maintenance", "events"]

[governance.sociocracy.linking]
# Double-linking between circles
double_link = true
# Leader selected by parent circle
leader_selection = "parent_circle"
# Delegate selected by own circle
delegate_selection = "own_circle"

[governance.sociocracy.consent]
# Consent-based (not consensus)
decision_method = "consent"
# No paramount objections = adopted
consent_threshold = "no_paramount_objections"

[governance.sociocracy.domains]
# Each circle has clear domain
domain_definition_required = true
# Decisions within domain don't need higher approval
domain_autonomy = true
# Cross-domain decisions go to linking circle
cross_domain_escalation = true

[governance.sociocracy.elections]
# Sociocratic election process
election_method = "nomination_rounds"
term_limits = true
default_term = "1y"
```

### Circle Structure

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    SOCIOCRATIC CIRCLE STRUCTURE                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│                      ┌─────────────────────┐                           │
│                      │   GENERAL CIRCLE    │                           │
│                      │   (Mission/Policy)  │                           │
│                      │                     │                           │
│                      │  L: Leader          │                           │
│                      │  D: Delegate        │                           │
│                      └──────────┬──────────┘                           │
│                                 │                                       │
│           ┌─────────────────────┼─────────────────────┐                │
│           │                     │                     │                │
│           ▼                     ▼                     ▼                │
│  ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐      │
│  │   OPERATIONS    │   │    FINANCE      │   │   MEMBERSHIP    │      │
│  │    CIRCLE       │   │    CIRCLE       │   │    CIRCLE       │      │
│  │                 │   │                 │   │                 │      │
│  │  L↑  D↑         │   │  L↑  D↑         │   │  L↑  D↑         │      │
│  └────────┬────────┘   └─────────────────┘   └─────────────────┘      │
│           │                                                            │
│     ┌─────┴─────┐                                                      │
│     │           │                                                      │
│     ▼           ▼                                                      │
│  ┌──────┐   ┌──────┐                                                  │
│  │Garden│   │Maint.│     ← Working Circles (operational)             │
│  │Circle│   │Circle│                                                  │
│  └──────┘   └──────┘                                                  │
│                                                                         │
│  Double-Linking:                                                       │
│  L↑ = Leader (selected by parent, represents parent in child)        │
│  D↑ = Delegate (selected by circle, represents circle in parent)     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Consent vs. Consensus

| Aspect | Consensus | Consent (Sociocracy) |
|--------|-----------|---------------------|
| Question asked | "Do you agree?" | "Do you have a paramount objection?" |
| Threshold | Everyone agrees | No one has principled objection |
| Objection handling | Must be resolved | Must be "paramount" (reasoned) |
| Speed | Slower | Faster |
| Participation | Active agreement required | Passive consent acceptable |

### Sociocratic Election Process

```
Round 1: Nominations
─────────────────────
Each member nominates someone (can self-nominate)
"I nominate X because..."

Round 2: Change Round
─────────────────────
After hearing all nominations, members may change
"After hearing the reasons, I change to Y because..."

Round 3: Consent Round
─────────────────────
Facilitator proposes leading candidate
"Any paramount objections to X serving as [role]?"

If objections → address or try next candidate
If no objections → X is selected
```

---

## Pattern 3: Holacracy

### Overview

Holacracy distributes authority through roles (not people) organized in circles. It emphasizes clear accountabilities, rapid processing of tensions, and evolutionary governance.

**Best For**: Organizations seeking clear structure, role-based work, continuous improvement

### Agora Configuration

```toml
[governance.holacracy]
name = "Holacratic Governance"
model = "holacracy"

[governance.holacracy.structure]
# Role-based (not person-based)
unit = "role"
# Roles organized in circles
organization = "nested_circles"
# Lead link and rep link
linking_roles = ["lead_link", "rep_link", "facilitator", "secretary"]

[governance.holacracy.authority]
# Authority distributed to roles
authority_type = "role_based"
# Each role has clear purpose, domain, accountabilities
role_definition_required = true
# Role-fillers have full authority within role
role_autonomy = true

[governance.holacracy.meetings]
# Tactical meetings for operations
tactical_frequency = "weekly"
# Governance meetings for structure
governance_frequency = "monthly"
# Integrative Decision Making process
decision_process = "idm"

[governance.holacracy.tensions]
# Tensions drive evolution
tension_processing = true
# Anyone can propose governance change
proposal_open = true
# Objections must be valid (specific criteria)
objection_validity_test = true
```

### Integrative Decision Making (IDM)

```
┌─────────────────────────────────────────────────────────────────────────┐
│              INTEGRATIVE DECISION MAKING PROCESS                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  1. PRESENT PROPOSAL                                                   │
│     └─→ Proposer describes tension and proposal                        │
│     └─→ No discussion yet                                              │
│                                                                         │
│  2. CLARIFYING QUESTIONS                                               │
│     └─→ Questions to understand (not react)                            │
│     └─→ Proposer answers or says "not specified"                       │
│                                                                         │
│  3. REACTION ROUND                                                     │
│     └─→ Each person reacts (no discussion)                             │
│     └─→ Proposer listens, may amend                                    │
│                                                                         │
│  4. AMEND & CLARIFY                                                    │
│     └─→ Proposer may amend based on reactions                          │
│     └─→ Or keep original                                               │
│                                                                         │
│  5. OBJECTION ROUND                                                    │
│     └─→ Facilitator asks each: "Any objections?"                       │
│     └─→ Objections must pass validity test                             │
│                                                                         │
│     Valid objection criteria:                                          │
│     • Caused by proposal (not pre-existing)                           │
│     • Based on known data (not predicted)                             │
│     • Harms the role's capacity to express purpose                    │
│     • Not personal preference                                          │
│                                                                         │
│  6. INTEGRATION                                                        │
│     └─→ If objections: Integrate concern into proposal                │
│     └─→ Proposer + objector find amendment                             │
│     └─→ Return to objection round                                      │
│                                                                         │
│  7. ADOPTION                                                           │
│     └─→ No valid objections = adopted                                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Role Definition Structure

```rust
#[hdk_entry_helper]
pub struct HolacraticRole {
    pub role_id: String,
    pub name: String,
    pub circle: String,

    /// Why this role exists
    pub purpose: String,

    /// What this role controls
    pub domains: Vec<Domain>,

    /// What this role is expected to do
    pub accountabilities: Vec<Accountability>,

    /// Who fills this role
    pub role_fillers: Vec<AgentPubKey>,

    /// Sub-roles (if circle)
    pub sub_roles: Option<Vec<String>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Domain {
    pub name: String,
    pub description: String,
    pub exclusive: bool,  // Only this role can act on it
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Accountability {
    pub description: String,
    pub recurring: bool,
    pub frequency: Option<String>,
}
```

---

## Pattern 4: Liquid Democracy

### Overview

Liquid democracy allows participants to either vote directly or delegate their vote to trusted proxies. Delegation can be transitive and topic-specific.

**Best For**: Large communities, diverse expertise, varying engagement levels

### Agora Configuration

```toml
[governance.liquid]
name = "Liquid Democracy"
model = "liquid_democracy"

[governance.liquid.delegation]
# Enable vote delegation
delegation_enabled = true
# Transitive delegation (A→B→C means A's vote goes through B to C)
transitive = true
# Maximum delegation depth
max_depth = 5
# Topic-specific delegation
topic_specific = true
# Can revoke and vote directly anytime
instant_revocation = true

[governance.liquid.topics]
# Define topic areas for delegation
topics = [
    "finance",
    "operations",
    "membership",
    "technical",
    "ecological",
    "social"
]

[governance.liquid.transparency]
# Delegation graph visible
delegation_visible = true
# But individual votes private until after voting
vote_privacy = "revealed_after_close"
# Delegate voting record public
delegate_record_public = true

[governance.liquid.weights]
# Optional: weight votes by expertise/stake
weighted_voting = false
# If weighted, by what?
weight_source = "equal"  # or "stake", "expertise", "trust"
```

### Delegation Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    LIQUID DEMOCRACY FLOW                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  DELEGATION SETUP (Ongoing)                                            │
│                                                                         │
│  Alice ──[finance]──→ Bob ──[finance]──→ Carol                        │
│        ──[technical]──→ Dave                                           │
│        ──[ecological]──→ Eve                                           │
│        (votes directly on: operations, membership, social)             │
│                                                                         │
│  PROPOSAL ARRIVES (Finance topic)                                      │
│                                                                         │
│  Options for Alice:                                                    │
│  1. Do nothing → delegation to Bob activates                          │
│  2. Vote directly → overrides delegation for this vote                │
│  3. Change delegation → update before vote closes                      │
│                                                                         │
│  VOTE COUNTING                                                         │
│                                                                         │
│  Carol votes YES (has 3 votes: own + Bob's delegation + Alice's)      │
│  Dave votes NO (has own vote only - technical delegation irrelevant)  │
│  Eve abstains (ecological - not relevant to finance)                  │
│  Alice: Delegation to Bob to Carol active                              │
│                                                                         │
│  RESULT DISPLAY                                                        │
│                                                                         │
│  YES: 3 votes (Carol direct, Bob delegated, Alice delegated)          │
│  NO: 1 vote (Dave direct)                                              │
│                                                                         │
│  Delegation graph shown after vote closes                              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Delegation Data Structure

```rust
#[hdk_entry_helper]
pub struct Delegation {
    pub delegation_id: String,
    pub delegator: AgentPubKey,
    pub delegate: AgentPubKey,
    pub scope: DelegationScope,
    pub created: Timestamp,
    pub active: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum DelegationScope {
    /// Delegate on all topics
    Global,
    /// Delegate on specific topics
    Topics(Vec<String>),
    /// Delegate on specific proposal types
    ProposalTypes(Vec<ProposalType>),
    /// Delegate for specific circle/domain
    Circle(String),
}

/// Calculate effective vote weight for a delegate
pub fn calculate_vote_weight(
    delegate: &AgentPubKey,
    proposal: &Proposal,
) -> Result<VoteWeight, GovernanceError> {
    // Start with delegate's own vote
    let mut weight = 1;

    // Find all delegations to this delegate for this proposal's topic
    let delegations = get_active_delegations_to(delegate, &proposal.topic)?;

    for delegation in delegations {
        // Check if delegator voted directly (overrides delegation)
        if !voted_directly(&delegation.delegator, &proposal.id)? {
            // Add delegator's weight (recursive for transitive)
            weight += calculate_delegated_weight(
                &delegation.delegator,
                proposal,
                1, // current depth
            )?;
        }
    }

    Ok(VoteWeight {
        delegate: delegate.clone(),
        weight,
        sources: trace_delegation_sources(delegate, proposal)?,
    })
}
```

---

## Pattern 5: Quadratic Voting

### Overview

Quadratic voting allows participants to express intensity of preference. Voting power costs increase quadratically (1 vote = 1 credit, 2 votes = 4 credits, 3 votes = 9 credits).

**Best For**: Resource allocation, priority setting, preventing majority tyranny

### Agora Configuration

```toml
[governance.quadratic]
name = "Quadratic Voting"
model = "quadratic"

[governance.quadratic.credits]
# Credit allocation
allocation_method = "periodic"  # or "equal_per_proposal"
credits_per_period = 100
period = "quarter"

[governance.quadratic.costs]
# Cost formula: votes^2
cost_formula = "quadratic"  # votes² = credits spent
# Maximum votes on single proposal
max_votes_per_proposal = 10  # costs 100 credits

[governance.quadratic.direction]
# Can vote for or against
bidirectional = true
# Negative votes cost same as positive
symmetric_costs = true

[governance.quadratic.rollover]
# Unused credits roll over?
rollover = false
# Or decay?
decay_rate = 0.5  # 50% decay per period
```

### Quadratic Voting Math

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    QUADRATIC VOTING COSTS                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Votes    Credits Cost    Cumulative Effect                            │
│  ─────    ────────────    ─────────────────                            │
│    1           1          "I care a little"                            │
│    2           4          "I care moderately"                          │
│    3           9          "I care significantly"                       │
│    4          16          "This really matters to me"                  │
│    5          25          "This is very important"                     │
│   10         100          "This is my top priority"                    │
│                                                                         │
│  Example Budget (100 credits):                                         │
│                                                                         │
│  Option A: Spread across many proposals                                │
│  • 4 votes each on 6 proposals = 96 credits                           │
│                                                                         │
│  Option B: Concentrate on priorities                                   │
│  • 7 votes on top priority = 49 credits                               │
│  • 5 votes on second priority = 25 credits                            │
│  • 2 votes each on 6 others = 24 credits                              │
│  • Total = 98 credits                                                  │
│                                                                         │
│  Effect: Minorities with strong preferences can compete with          │
│  majorities with weak preferences                                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Use Cases

| Use Case | Why Quadratic Works |
|----------|---------------------|
| Budget allocation | Reveals true priorities, not just preference order |
| Feature prioritization | Intense minority needs surface |
| Resource distribution | Prevents majority grabbing everything |
| Multi-option decisions | Better than plurality voting |

---

## Pattern 6: Conviction Voting

### Overview

Conviction voting accumulates voting power over time. The longer you stake your vote on a proposal, the more weight it gains. Good for continuous funding decisions.

**Best For**: Treasury allocation, ongoing funding, community priorities

### Agora Configuration

```toml
[governance.conviction]
name = "Conviction Voting"
model = "conviction"

[governance.conviction.accumulation]
# Conviction accumulates over time
half_life = "3d"  # Time to reach 50% max conviction
max_conviction = 10.0  # Maximum multiplier

[governance.conviction.thresholds]
# Proposals pass when conviction exceeds threshold
threshold_formula = "requested_amount_based"
# Higher requests need more conviction
base_threshold = 1000
scaling_factor = 0.5

[governance.conviction.tokens]
# What do people stake?
staking_token = "governance_token"
# Or voice credits
voice_credits = true
# Can split across proposals
split_allowed = true

[governance.conviction.withdrawal]
# Conviction decays when removed
decay_on_withdrawal = true
withdrawal_decay_rate = 0.9  # Lose 90% immediately
```

### Conviction Accumulation

```
┌─────────────────────────────────────────────────────────────────────────┐
│                 CONVICTION ACCUMULATION CURVE                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Conviction                                                            │
│  (Multiplier)                                                          │
│       │                                                                │
│  10.0 │                          ─────────────────── max              │
│       │                     ╱                                          │
│   7.5 │                 ╱                                              │
│       │             ╱                                                  │
│   5.0 │         ╱                    ← 50% at half-life               │
│       │     ╱                                                          │
│   2.5 │ ╱                                                              │
│       │╱                                                               │
│   0.0 └────────────────────────────────────────────→ Time             │
│       0   3d    6d    9d    12d   15d   18d   21d                     │
│           ↑                                                            │
│       half-life                                                        │
│                                                                         │
│  Formula: conviction = max * (1 - 0.5^(t/half_life))                  │
│                                                                         │
│  With half_life = 3 days:                                             │
│  • Day 0: 0 conviction                                                │
│  • Day 3: 5.0 conviction (50% of max)                                 │
│  • Day 6: 7.5 conviction (75% of max)                                 │
│  • Day 9: 8.75 conviction (87.5% of max)                              │
│  • Day 12: 9.375 conviction (93.75% of max)                           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Continuous Funding Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│             CONVICTION VOTING FOR CONTINUOUS FUNDING                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Community Treasury: 100,000 tokens                                    │
│                                                                         │
│  Proposals:                                                            │
│  ┌──────────────────────────────────────────────────────┐              │
│  │ Proposal A: Community Garden ($5,000)                │              │
│  │ Conviction: 3,450 / 5,000 threshold                  │ ████████░░  │
│  │ Time to pass at current rate: 2 days                 │              │
│  └──────────────────────────────────────────────────────┘              │
│                                                                         │
│  ┌──────────────────────────────────────────────────────┐              │
│  │ Proposal B: Developer Grant ($15,000)                │              │
│  │ Conviction: 8,200 / 15,000 threshold                 │ █████░░░░░  │
│  │ Time to pass at current rate: 5 days                 │              │
│  └──────────────────────────────────────────────────────┘              │
│                                                                         │
│  ┌──────────────────────────────────────────────────────┐              │
│  │ Proposal C: New Equipment ($2,000)                   │              │
│  │ Conviction: 2,100 / 2,000 threshold                  │ ██████████  │
│  │ PASSING - funds releasing                            │    ✓        │
│  └──────────────────────────────────────────────────────┘              │
│                                                                         │
│  Members can reallocate conviction anytime                             │
│  Proposals compete for limited conviction                              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Pattern 7: Futarchy (Prediction Market Governance)

### Overview

Futarchy uses prediction markets to inform governance. "Vote on values, bet on beliefs." Decisions are made based on which option markets predict will best achieve stated goals.

**Best For**: Evidence-based policy, complex decisions, reducing ideological bias

### Agora + Forecast Configuration

```toml
[governance.futarchy]
name = "Futarchy"
model = "futarchy"

[governance.futarchy.metrics]
# Define success metrics first
primary_metric = "community_wellbeing_index"
secondary_metrics = ["economic_health", "ecological_health", "social_cohesion"]
measurement_period = "6m"

[governance.futarchy.markets]
# Create prediction markets for each option
market_creation = "automatic"
trading_period = "14d"
# Subsidize markets for liquidity
liquidity_subsidy = 1000

[governance.futarchy.decision]
# Decision based on market prices
decision_method = "higher_predicted_outcome"
# Require significant price difference
significance_threshold = 0.05  # 5% difference in predicted outcome
# If no significant difference, fall back to vote
fallback = "direct_vote"

[governance.futarchy.resolution]
# Markets resolve based on actual outcomes
resolution_period = "6m"
oracle_source = "designated_measurers"
```

### Futarchy Decision Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    FUTARCHY DECISION FLOW                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  1. DEFINE VALUES (Democratic)                                         │
│     └─→ Community votes on success metrics                             │
│     └─→ "We want to maximize wellbeing index"                         │
│                                                                         │
│  2. PROPOSE OPTIONS                                                    │
│     └─→ Policy A: Invest in childcare                                 │
│     └─→ Policy B: Invest in job training                              │
│                                                                         │
│  3. CREATE PREDICTION MARKETS                                          │
│     └─→ Market A: "Wellbeing if Policy A adopted"                     │
│     └─→ Market B: "Wellbeing if Policy B adopted"                     │
│                                                                         │
│  4. TRADING PERIOD                                                     │
│     └─→ Participants buy/sell based on beliefs                        │
│     └─→ Prices reflect collective prediction                          │
│                                                                         │
│  Example prices after 14 days:                                         │
│  • Market A: $78 (predicting wellbeing = 78 if A adopted)             │
│  • Market B: $72 (predicting wellbeing = 72 if B adopted)             │
│                                                                         │
│  5. DECISION                                                           │
│     └─→ Policy A adopted (higher predicted wellbeing)                 │
│                                                                         │
│  6. IMPLEMENTATION                                                     │
│     └─→ Policy A implemented                                          │
│                                                                         │
│  7. RESOLUTION (6 months later)                                        │
│     └─→ Actual wellbeing measured: 76                                 │
│     └─→ Market A pays $76 per share                                   │
│     └─→ Accurate predictors profit, inaccurate lose                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Pattern 8: Advice Process

### Overview

Anyone can make any decision after seeking advice from affected parties and those with expertise. No committees, no votes—just mandatory consultation.

**Best For**: Agile organizations, high-trust teams, operational decisions

### Agora Configuration

```toml
[governance.advice]
name = "Advice Process"
model = "advice_process"

[governance.advice.consultation]
# Advice required from:
required_consultations = ["affected_parties", "domain_experts"]
# Minimum consultation period
consultation_period = "3d"
# Must document advice received
documentation_required = true

[governance.advice.decision_authority]
# Whoever identifies issue can decide
decider = "initiator"
# But must genuinely consider advice
consideration_required = true
# Can proceed even against advice
can_override_advice = true
# But must explain why
override_explanation_required = true

[governance.advice.transparency]
# All advice conversations visible
advice_visible = true
# Decision rationale visible
rationale_visible = true
# Outcome tracking
outcome_tracking = true
```

### Advice Process Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    ADVICE PROCESS FLOW                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Maya notices: "We need new accounting software"                       │
│                                                                         │
│  1. IDENTIFY AFFECTED & EXPERTS                                        │
│     └─→ Affected: Finance team, anyone who submits expenses           │
│     └─→ Experts: Current accountant, IT support                       │
│                                                                         │
│  2. SEEK ADVICE                                                        │
│     ┌─────────────────────────────────────────────────────────────┐   │
│     │ Maya → Finance Team:                                         │   │
│     │ "I'm considering switching to [Software X]. Thoughts?"       │   │
│     │                                                               │   │
│     │ Response: "Good idea, but make sure it integrates with       │   │
│     │ our payroll system."                                          │   │
│     └─────────────────────────────────────────────────────────────┘   │
│     ┌─────────────────────────────────────────────────────────────┐   │
│     │ Maya → IT Support:                                           │   │
│     │ "What do you think of Software X vs Software Y?"             │   │
│     │                                                               │   │
│     │ Response: "Y has better security, but X is easier to         │   │
│     │ support. I'd lean toward Y for data sensitivity."            │   │
│     └─────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  3. CONSIDER ADVICE                                                    │
│     └─→ Maya weighs input, does additional research                   │
│                                                                         │
│  4. MAKE DECISION                                                      │
│     └─→ Maya decides: Software Y                                      │
│     └─→ Documents rationale: "Security concerns outweigh ease"        │
│     └─→ Addresses finance concern: "Confirmed payroll integration"    │
│                                                                         │
│  5. IMPLEMENT                                                          │
│     └─→ Maya implements, owns outcome                                 │
│                                                                         │
│  6. TRACK OUTCOME                                                      │
│     └─→ 6 months later: System working well                           │
│     └─→ Maya's judgment track record updated                          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Hybrid Patterns

### Pattern 9: Tiered Governance

Combine patterns based on decision type/impact:

```toml
[governance.tiered]
name = "Tiered Governance"
model = "tiered"

[governance.tiered.tiers]

[governance.tiered.tiers.operational]
# Day-to-day decisions
impact_threshold = "low"
model = "advice_process"
time_limit = "3d"

[governance.tiered.tiers.tactical]
# Medium-impact decisions
impact_threshold = "medium"
model = "consent"
time_limit = "7d"
quorum = 0.5

[governance.tiered.tiers.strategic]
# High-impact decisions
impact_threshold = "high"
model = "consensus"
time_limit = "21d"
quorum = 0.75
fallback = "supermajority"

[governance.tiered.tiers.constitutional]
# Fundamental changes
impact_threshold = "constitutional"
model = "consensus"
time_limit = "60d"
quorum = 0.9
ratification_period = "30d"
super_majority = 0.8
```

### Pattern 10: Adaptive Governance

Governance that evolves based on community stage and needs:

```rust
/// Governance that adapts to community development
pub struct AdaptiveGovernance {
    pub community_id: String,

    /// Current primary model
    pub current_model: GovernanceModel,

    /// Trigger conditions for evolution
    pub evolution_triggers: Vec<EvolutionTrigger>,

    /// Available models to evolve toward
    pub available_evolutions: Vec<GovernanceModel>,

    /// Evolution history
    pub evolution_history: Vec<GovernanceEvolution>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EvolutionTrigger {
    pub trigger_type: TriggerType,
    pub threshold: TriggerThreshold,
    pub suggested_evolution: GovernanceModel,
    pub requires_proposal: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum TriggerType {
    /// Community size changed
    SizeChange { direction: Direction, threshold: u32 },

    /// Participation rate changed
    ParticipationChange { direction: Direction, threshold: Decimal },

    /// Decision speed issues
    DecisionSpeedIssue { too_slow: bool, threshold: Duration },

    /// Trust level changed
    TrustLevelChange { direction: Direction, threshold: Decimal },

    /// Stage distribution shifted
    StageDistributionShift { new_center: DevelopmentalStage },

    /// Explicit community request
    CommunityRequest { proposal_id: String },
}

/// Check if governance should evolve
pub fn check_evolution_triggers(
    community: &Community,
    governance: &AdaptiveGovernance,
) -> Vec<EvolutionSuggestion> {
    let mut suggestions = Vec::new();

    for trigger in &governance.evolution_triggers {
        if trigger_activated(community, trigger) {
            suggestions.push(EvolutionSuggestion {
                trigger: trigger.clone(),
                suggested_model: trigger.suggested_evolution.clone(),
                rationale: generate_evolution_rationale(trigger, community),
                auto_propose: !trigger.requires_proposal,
            });
        }
    }

    suggestions
}
```

---

## Governance for Different Developmental Stages

### Stage-Appropriate Governance

| Stage | Preferred Patterns | Challenging Patterns |
|-------|-------------------|---------------------|
| **Traditional** | Clear hierarchy, defined roles, proper procedure | Flat consensus, liquid democracy |
| **Modern** | Efficient processes, merit-based, data-driven | Slow consensus, equal time banking |
| **Postmodern** | Inclusive consensus, rotating leadership | Strong hierarchy, winner-take-all |
| **Integral** | Adaptive, situational, meta-governance | Rigid single-model approaches |

### Multi-Stage Community Configuration

```toml
[governance.stage_aware]
name = "Stage-Aware Governance"

# Different interfaces for same governance process
[governance.stage_aware.interfaces]

[governance.stage_aware.interfaces.traditional]
emphasis = ["proper_procedure", "authority_endorsement", "clear_rules"]
language = "authoritative"
features_visible = ["committee_recommendations", "rule_citations"]
features_hidden = ["experimental_options"]

[governance.stage_aware.interfaces.modern]
emphasis = ["efficiency", "data", "outcomes"]
language = "professional"
features_visible = ["metrics", "projections", "ROI"]
features_hidden = ["consensus_building_tools"]

[governance.stage_aware.interfaces.postmodern]
emphasis = ["inclusion", "voices", "impact"]
language = "inclusive"
features_visible = ["affected_parties", "diverse_perspectives"]
features_hidden = ["efficiency_metrics"]

[governance.stage_aware.interfaces.integral]
emphasis = ["context", "adaptation", "emergence"]
language = "nuanced"
features_visible = ["all_features", "meta_options", "stage_info"]
features_hidden = []
```

---

## Implementation Checklist

### For Communities Choosing Governance

- [ ] Assess community size, trust level, decision types
- [ ] Identify dominant developmental stages
- [ ] Review pattern options with community
- [ ] Start simple, plan for evolution
- [ ] Configure Agora with chosen pattern
- [ ] Train facilitators and participants
- [ ] Establish feedback mechanisms
- [ ] Plan first governance review (3-6 months)

### For Pattern Transitions

- [ ] Diagnose current governance pain points
- [ ] Identify candidate new patterns
- [ ] Run parallel pilot if possible
- [ ] Propose transition through current governance
- [ ] Allow transition period (both valid)
- [ ] Support members in adaptation
- [ ] Document learnings

---

## Conclusion

Governance is not one-size-fits-all. The Mycelix ecosystem provides a rich library of patterns that communities can adopt, adapt, and evolve. The key is matching governance structure to community needs, values, and developmental stage—and remaining willing to evolve as the community grows.

---

*"The best governance is the governance that helps the community become what it wants to become, while protecting those who cannot yet speak for themselves."*

---

*Document Version: 1.0*
*Last Updated: 2025*
*Sources: Sociocracy (Gerard Endenburg), Holacracy (Brian Robertson), Liquid Democracy (various), Quadratic Voting (Glen Weyl), Conviction Voting (Commons Stack), Futarchy (Robin Hanson)*

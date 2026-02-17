# Mycelix Onboarding & Adoption Playbook

## Overview

This playbook guides communities through adopting Mycelix—from initial discovery to full sovereignty. Whether you're a neighborhood group, cooperative, intentional community, or municipal initiative, this document provides the pathways, templates, and UX flows for successful adoption.

**Core Principle**: Meet communities where they are, grow together.

---

## Adoption Pathways

### The Adoption Spectrum

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       ADOPTION SPECTRUM                                  │
│                                                                         │
│  OBSERVER → PARTICIPANT → MEMBER → CONTRIBUTOR → STEWARD → SOVEREIGN   │
│                                                                         │
│  ┌─────┐   ┌─────┐   ┌─────┐   ┌─────┐   ┌─────┐   ┌─────┐           │
│  │Watch│   │Use  │   │Join │   │Build│   │Guide│   │Self-│           │
│  │learn│ → │some │ → │fully│ → │&    │ → │&    │ → │govern│          │
│  │     │   │tools│   │     │   │create│   │mentor│   │     │          │
│  └─────┘   └─────┘   └─────┘   └─────┘   └─────┘   └─────┘           │
│                                                                         │
│  Commitment: None → Light → Moderate → Active → Leadership → Full      │
│  Duration:   Days → Weeks → Months → 6mo+ → Year+ → Ongoing           │
└─────────────────────────────────────────────────────────────────────────┘
```

### Pathway 1: Existing Community Adoption

For communities that already exist (neighborhoods, coops, organizations):

```
Week 1-2: Discovery
├── Community assessment call
├── Needs mapping workshop
├── Technology readiness evaluation
└── Initial champion identification

Week 3-4: Pilot Design
├── Select initial hApps (typically 2-3)
├── Configure community template
├── Train pilot group (10-20 people)
└── Set success metrics

Month 2-3: Pilot Phase
├── Launch with pilot group
├── Weekly check-ins and adjustments
├── Gather feedback systematically
└── Document learnings

Month 4-6: Expansion
├── Onboard broader community
├── Add additional hApps
├── Develop local champions
└── Integrate with existing processes

Month 7-12: Maturation
├── Community governance fully active
├── Custom adaptations developed
├── Inter-community connections
└── Pathway to sovereignty
```

### Pathway 2: New Community Formation

For groups forming a new community around Mycelix:

```
Phase 1: Gathering (Week 1-4)
├── Founding vision workshop
├── Values and principles articulation
├── Initial membership criteria
└── Minimal viable governance

Phase 2: Foundation (Month 2-3)
├── Deploy Tier 0 (Spine) hApps
├── Establish identity (Attest)
├── Bootstrap trust network (MATL)
└── Create initial governance (Agora)

Phase 3: Building (Month 4-6)
├── Add Tier 1 (Core) hApps
├── Develop economic system (Treasury, currencies)
├── Create first collaborative projects (Collab)
└── Establish dispute resolution (Arbiter)

Phase 4: Growing (Month 7-12)
├── Add domain hApps based on needs
├── Integrate behavioral layer (Nudge, Spiral)
├── Develop external relationships
└── Continuous governance evolution
```

### Pathway 3: Institutional Adoption

For municipalities, organizations, or large institutions:

```
Quarter 1: Assessment & Planning
├── Stakeholder analysis
├── Legal and compliance review
├── Integration architecture design
├── Change management planning
└── Budget and resource allocation

Quarter 2: Pilot Program
├── Select pilot department/region
├── Deploy with limited scope
├── Measure against KPIs
├── Document integration patterns
└── Gather stakeholder feedback

Quarter 3-4: Phased Rollout
├── Department-by-department expansion
├── Training program implementation
├── Support infrastructure development
├── Policy alignment
└── Public communication

Year 2+: Full Integration
├── Organization-wide deployment
├── Deep system integration
├── Governance transition
├── Continuous improvement
└── External sharing of learnings
```

---

## Community Templates

### Template: Village/Neighborhood

**Profile**: 50-500 people, place-based, mixed needs

```toml
[community]
name = "Your Neighborhood Name"
type = "village"
size_range = "50-500"
primary_focus = ["local_economy", "mutual_aid", "governance"]

[identity]
verification_level = "basic"
allow_pseudonyms = false  # Place-based = real identity
membership_process = "vouching"  # Existing members vouch

[enabled_happs]
tier_0 = ["bridge", "matl", "attest", "sentinel", "chronicle"]
tier_1 = ["agora", "arbiter", "marketplace", "treasury", "collab"]
tier_2 = [
    "provision",   # Local food
    "terroir",     # Housing/land
    "kinship",     # Family support
    "beacon",      # Emergency coordination
    "mutual"       # Mutual aid
]
tier_2_5 = ["nudge", "spiral"]

[governance]
model = "consensus_with_fallback"
voting_period = "7d"
quorum = 0.25
fallback_threshold = 0.67

[economics]
primary_currency = "mutual_credit"
secondary_currencies = ["time_bank"]
enable_ubi = false  # Start simple
transaction_fees = 0.0  # Free within community

[recommended_first_steps]
1 = "Create community profile and invite founding members"
2 = "Run first Agora proposal to ratify community principles"
3 = "Set up Marketplace for local exchange"
4 = "Create first Collab project for shared community need"
5 = "Establish Beacon for emergency contact network"
```

### Template: Worker Cooperative

**Profile**: 5-100 workers, economic democracy focus

```toml
[community]
name = "Your Cooperative Name"
type = "worker_cooperative"
size_range = "5-100"
primary_focus = ["economic_democracy", "shared_ownership", "collaboration"]

[identity]
verification_level = "verified"
allow_pseudonyms = false
membership_process = "application_and_vote"
probation_period = "90d"

[enabled_happs]
tier_0 = ["bridge", "matl", "attest", "sentinel", "chronicle"]
tier_1 = ["agora", "arbiter", "marketplace", "treasury", "collab", "covenant"]
tier_2 = [
    "guild",       # Professional development
    "accord",      # Tax/legal compliance
    "sanctuary",   # Worker wellbeing
]
tier_2_5 = ["nudge", "spiral"]

[governance]
model = "one_worker_one_vote"
voting_period = "5d"
quorum = 0.5
major_decision_threshold = 0.67
operational_decision_threshold = 0.5

[economics]
primary_currency = "community_token"  # Ownership shares
secondary_currencies = ["time_bank", "mutual_credit"]
enable_ubi = false
profit_distribution = "patronage_based"  # Based on work contribution

[ownership]
share_structure = "equal_per_member"
vesting_period = "2y"
buyout_formula = "book_value"
exit_process = "gradual_redemption"

[recommended_first_steps]
1 = "Ratify bylaws through Agora"
2 = "Set up Treasury with initial capital"
3 = "Create Covenant templates for client contracts"
4 = "Establish Collab spaces for projects"
5 = "Configure Accord for tax compliance"
```

### Template: Intentional Community

**Profile**: 20-200 people, values-aligned, often residential

```toml
[community]
name = "Your Community Name"
type = "intentional_community"
size_range = "20-200"
primary_focus = ["shared_living", "values_alignment", "sustainability"]

[identity]
verification_level = "verified"
allow_pseudonyms = false
membership_process = "application_visit_consensus"
trial_period = "6m"

[enabled_happs]
tier_0 = ["bridge", "matl", "attest", "sentinel", "chronicle"]
tier_1 = ["agora", "arbiter", "treasury", "collab", "covenant"]
tier_2 = [
    "terroir",     # Land and housing
    "provision",   # Food systems
    "ember",       # Energy
    "kinship",     # Care coordination
    "sanctuary",   # Mental health
    "commons",     # Shared resources
    "hearth",      # Culture and media
]
tier_2_5 = ["nudge", "spiral"]
tier_3 = [
    "legacy",      # Generational transfer
    "resonance",   # Values alignment
]

[governance]
model = "sociocracy"  # Circles with consent-based decisions
voting_period = "14d"  # Longer for more reflection
quorum = 0.6
consent_threshold = 0.9  # Near-consensus required

[economics]
primary_currency = "mutual_credit"
secondary_currencies = ["time_bank", "community_token"]
enable_ubi = true
ubi_source = "commons_rent"  # Land value shared

[land]
ownership_model = "community_land_trust"
housing_allocation = "needs_based"
land_use_governance = "deliberative"

[recommended_first_steps]
1 = "Document founding vision in Chronicle"
2 = "Set up sociocratic circle structure in Agora"
3 = "Configure Terroir for land/housing management"
4 = "Create Provision system for food coordination"
5 = "Establish Kinship networks for mutual support"
```

### Template: Digital-First DAO

**Profile**: 10-10,000 people, distributed, token-governed

```toml
[community]
name = "Your DAO Name"
type = "dao"
size_range = "10-10000"
primary_focus = ["distributed_governance", "token_economics", "digital_coordination"]

[identity]
verification_level = "basic"  # Can be pseudonymous
allow_pseudonyms = true
membership_process = "token_acquisition"

[enabled_happs]
tier_0 = ["bridge", "matl", "attest", "sentinel", "chronicle"]
tier_1 = ["agora", "arbiter", "oracle", "marketplace", "treasury", "collab", "covenant"]
tier_2 = [
    "guild",       # Working groups
    "accord",      # Compliance
    "forecast",    # Prediction markets
]
tier_2_5 = ["nudge", "spiral"]
tier_3 = [
    "diplomat",    # Multi-DAO relations
    "membrane",    # Access control
]

[governance]
model = "token_weighted_quadratic"
voting_period = "5d"
quorum = 0.1  # Lower for large DAOs
proposal_threshold = 1000  # Tokens to propose

[economics]
primary_currency = "community_token"
token_model = "governance_utility"
enable_staking = true
inflation_rate = 0.02  # 2% annual

[recommended_first_steps]
1 = "Deploy governance token via Treasury"
2 = "Set up Agora with token-weighted voting"
3 = "Create Guild working groups"
4 = "Configure Collab for bounty system"
5 = "Establish Oracle for external data needs"
```

### Template: Municipal/Civic

**Profile**: 1,000-100,000+ people, public sector, democratic

```toml
[community]
name = "Your Municipality Name"
type = "municipal"
size_range = "1000-100000+"
primary_focus = ["public_services", "citizen_engagement", "transparency"]

[identity]
verification_level = "verified"  # Government ID verification
allow_pseudonyms = false
membership_process = "residency_verification"

[enabled_happs]
tier_0 = ["bridge", "matl", "attest", "sentinel", "chronicle"]
tier_1 = ["agora", "arbiter", "oracle", "treasury", "collab"]
tier_2 = [
    "provision",   # Food security
    "terroir",     # Housing/land
    "transit",     # Public transit
    "ember",       # Utilities
    "beacon",      # Emergency services
    "healthvault", # Public health
    "sanctuary",   # Mental health services
    "accord",      # Tax integration
]
tier_2_5 = ["nudge", "spiral"]
tier_3 = [
    "civitas",     # Citizenship
    "diplomat",    # Inter-municipal
]

[governance]
model = "hybrid_representative_participatory"
representative_layer = true  # Elected officials
participatory_layer = true   # Direct citizen input
voting_period = "14d"
participation_incentive = true

[economics]
primary_currency = "municipal_credit"
integration_with_fiat = true
enable_participatory_budgeting = true
transparency_level = "full_public"

[privacy]
citizen_data_protection = "strict"
data_minimization = true
right_to_deletion = true
gdpr_compliant = true

[recommended_first_steps]
1 = "Citizen identity verification rollout"
2 = "Participatory budgeting pilot via Agora"
3 = "Transit coordination via Transit hApp"
4 = "Public service feedback via Chronicle"
5 = "Emergency coordination via Beacon"
```

---

## First 100 Users Journey

### The Onboarding Funnel

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    FIRST 100 USERS JOURNEY                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  AWARENESS (500+ reached)                                               │
│      │                                                                  │
│      ▼                                                                  │
│  INTEREST (200 click through)                                           │
│      │                                                                  │
│      ▼                                                                  │
│  TRIAL (100 create account)                                             │
│      │                                                                  │
│      ▼                                                                  │
│  FIRST VALUE (75 complete first action)                                 │
│      │                                                                  │
│      ▼                                                                  │
│  ENGAGEMENT (50 return within week)                                     │
│      │                                                                  │
│      ▼                                                                  │
│  COMMITMENT (35 complete full onboarding)                               │
│      │                                                                  │
│      ▼                                                                  │
│  CONTRIBUTION (20 create content/value)                                 │
│      │                                                                  │
│      ▼                                                                  │
│  ADVOCACY (10 invite others)                                            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Stage 1: First 10 Minutes

**Goal**: Create account, understand value, complete one action

```
┌─────────────────────────────────────────────────────────────────────────┐
│ WELCOME TO [COMMUNITY NAME]                                             │
│                                                                         │
│ "Where neighbors become collaborators"                                  │
│                                                                         │
│ ┌─────────────────────────────────────────────────────────────────────┐│
│ │                                                                     ││
│ │  [  Get Started  ]     (Primary CTA)                               ││
│ │                                                                     ││
│ │  [  Watch 2-min intro  ]  [  Explore first  ]                      ││
│ │                                                                     ││
│ └─────────────────────────────────────────────────────────────────────┘│
│                                                                         │
│ Trusted by 847 members of your community                               │
│ ★★★★★ "Finally, a way to actually organize" - Sarah M.                │
└─────────────────────────────────────────────────────────────────────────┘

                              │
                              ▼

┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 1 OF 4: CREATE YOUR IDENTITY                                       │
│                                                                         │
│ Your name: [____________________]                                       │
│                                                                         │
│ How you'd like to be known (this is public):                           │
│ [____________________]                                                  │
│                                                                         │
│ 📸 Add a photo (helps neighbors recognize you)                         │
│ [Upload] or [Take photo]                                                │
│                                                                         │
│ Your identity is yours. You control what you share.                    │
│ [Learn about privacy →]                                                 │
│                                                                         │
│                                    [Continue →]                         │
└─────────────────────────────────────────────────────────────────────────┘

                              │
                              ▼

┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 2 OF 4: CONNECT WITH YOUR COMMUNITY                               │
│                                                                         │
│ How did you hear about us?                                              │
│ ○ Invited by a neighbor                                                 │
│ ○ Community event                                                       │
│ ○ Social media                                                          │
│ ○ Other: [________]                                                     │
│                                                                         │
│ Were you invited by someone?                                            │
│ [Enter their name to connect]                                           │
│                                                                         │
│ Connecting with existing members helps you get started faster.         │
│                                                                         │
│                                    [Continue →]                         │
└─────────────────────────────────────────────────────────────────────────┘

                              │
                              ▼

┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 3 OF 4: WHAT BRINGS YOU HERE?                                     │
│                                                                         │
│ Select what interests you most (pick up to 3):                         │
│                                                                         │
│ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐                        │
│ │🛒 Local │ │🤝 Mutual│ │🗳️ Have │ │👥 Meet  │                        │
│ │Exchange │ │Aid      │ │a Voice │ │Neighbors│                        │
│ └─────────┘ └─────────┘ └─────────┘ └─────────┘                        │
│                                                                         │
│ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐                        │
│ │🌱 Local │ │👶 Family│ │🎨 Arts │ │🔧 Skills│                        │
│ │Food     │ │Support  │ │& Culture│ │Exchange │                        │
│ └─────────┘ └─────────┘ └─────────┘ └─────────┘                        │
│                                                                         │
│ We'll personalize your experience based on your interests.             │
│                                                                         │
│                                    [Continue →]                         │
└─────────────────────────────────────────────────────────────────────────┘

                              │
                              ▼

┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 4 OF 4: YOUR FIRST ACTION                                          │
│                                                                         │
│ Welcome, [Name]! Let's do something together.                          │
│                                                                         │
│ Based on your interests, here's a great first step:                    │
│                                                                         │
│ ┌─────────────────────────────────────────────────────────────────────┐│
│ │ 🛒 POST SOMETHING YOU CAN OFFER                                     ││
│ │                                                                     ││
│ │ Share a skill, item, or service with your neighbors.               ││
│ │ 127 neighbors are looking for help this week!                      ││
│ │                                                                     ││
│ │ Examples from your community:                                       ││
│ │ • "Homemade bread, fresh Saturdays" - Maria                        ││
│ │ • "Guitar lessons for beginners" - James                           ││
│ │ • "Help with moving boxes" - Chen                                  ││
│ │                                                                     ││
│ │ [Post an Offer →]                                                   ││
│ └─────────────────────────────────────────────────────────────────────┘│
│                                                                         │
│ Or explore:                                                             │
│ [See what neighbors offer] [Browse active discussions]                 │
└─────────────────────────────────────────────────────────────────────────┘
```

### Stage 2: First Week

**Goal**: Return 3+ times, complete profile, make first exchange/interaction

```
Day 1 (Post-signup):
├── Welcome email with 3 simple actions
├── Push notification: "Sarah accepted your connection!"
└── Personalized feed based on interests

Day 2:
├── Notification: "5 neighbors interested in [your interest]"
├── Email: "Complete your profile for better matches"
└── Prompt: First governance action (simple poll)

Day 3-4:
├── Notification: "New offer matches your needs"
├── Prompt: Leave feedback on a neighbor's offering
└── Introduction to time banking

Day 5-7:
├── Weekly digest email
├── Prompt: "Your neighbors need help with X"
├── Invitation to first community event/gathering
└── Progress celebration: "You're part of the community!"
```

### Stage 3: First Month

**Goal**: Establish regular usage pattern, complete first meaningful exchange

```
Week 2:
├── Deepen in primary interest area
├── First real exchange (goods, services, or time)
├── Participate in first governance decision
└── Connect with 5+ community members

Week 3:
├── Explore secondary interest area
├── Receive first trust endorsement
├── Contribute to shared project (Collab)
└── Discover Nudge/Spiral personalization

Week 4:
├── First month celebration
├── Profile completion check
├── Invitation to mentor new members
├── Community contribution recognition
└── Survey: What's working? What's not?
```

---

## UX Flows

### Flow 1: New Member Joins Existing Community

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    NEW MEMBER ONBOARDING FLOW                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐             │
│  │ Receive │    │ Landing │    │ Create  │    │ Initial │             │
│  │ Invite  │ →  │ Page    │ →  │ Identity│ →  │ Trust   │             │
│  │ Link    │    │         │    │ (Attest)│    │ (MATL)  │             │
│  └─────────┘    └─────────┘    └─────────┘    └─────────┘             │
│                                      │              │                   │
│                                      │              │                   │
│                                      ▼              ▼                   │
│                              ┌─────────────────────────┐               │
│                              │   Interest Selection    │               │
│                              │   (Spiral Discovery)    │               │
│                              └───────────┬─────────────┘               │
│                                          │                             │
│                                          ▼                             │
│  ┌─────────┐    ┌─────────┐    ┌─────────────────────────┐            │
│  │ Regular │    │ First   │    │ Personalized Dashboard  │            │
│  │ Member  │ ←  │ Value   │ ←  │ (Stage-Appropriate UI)  │            │
│  │         │    │ Action  │    │                         │            │
│  └─────────┘    └─────────┘    └─────────────────────────┘            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Flow 2: Making a Local Exchange

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    LOCAL EXCHANGE FLOW                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  SEEKER JOURNEY                          PROVIDER JOURNEY              │
│  ─────────────                          ─────────────────              │
│                                                                         │
│  ┌─────────────┐                        ┌─────────────┐                │
│  │ Search/     │                        │ Create      │                │
│  │ Browse      │                        │ Offering    │                │
│  │ Marketplace │                        │             │                │
│  └──────┬──────┘                        └──────┬──────┘                │
│         │                                      │                        │
│         ▼                                      ▼                        │
│  ┌─────────────┐     ┌─────────┐       ┌─────────────┐                │
│  │ View        │     │ Message │       │ Receive     │                │
│  │ Offering    │ ──▶ │ Provider│ ◀──── │ Interest    │                │
│  │ Details     │     │         │       │ Notification│                │
│  └──────┬──────┘     └────┬────┘       └─────────────┘                │
│         │                 │                                            │
│         └────────┬────────┘                                            │
│                  ▼                                                      │
│         ┌─────────────┐                                                │
│         │ Agree on    │                                                │
│         │ Terms       │                                                │
│         │ (Covenant)  │                                                │
│         └──────┬──────┘                                                │
│                │                                                        │
│                ▼                                                        │
│         ┌─────────────┐     ┌─────────────┐                           │
│         │ Exchange    │     │ Record in   │                           │
│         │ Occurs      │ ──▶ │ Chronicle   │                           │
│         │             │     │             │                           │
│         └──────┬──────┘     └─────────────┘                           │
│                │                                                        │
│                ▼                                                        │
│         ┌─────────────┐     ┌─────────────┐                           │
│         │ Mutual      │     │ Trust Score │                           │
│         │ Feedback    │ ──▶ │ Updated     │                           │
│         │             │     │ (MATL)      │                           │
│         └─────────────┘     └─────────────┘                           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Flow 3: Participating in Governance

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    GOVERNANCE PARTICIPATION FLOW                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐│
│  │                        PROPOSAL LIFECYCLE                           ││
│  │                                                                     ││
│  │   Draft → Discussion → Voting → Execution → Review                 ││
│  │                                                                     ││
│  └─────────────────────────────────────────────────────────────────────┘│
│                                                                         │
│  CITIZEN EXPERIENCE:                                                    │
│                                                                         │
│  ┌─────────────┐    ┌─────────────────────────────────────────────────┐│
│  │ Notification│    │  PROPOSAL: Fund Community Garden                ││
│  │ "New        │    │                                                  ││
│  │ proposal    │ →  │  📊 Current: 67% support (32 votes)             ││
│  │ needs your  │    │                                                  ││
│  │ input"      │    │  Summary: Allocate 500 credits from Treasury... ││
│  └─────────────┘    │                                                  ││
│                     │  Stage-Appropriate Framing:                      ││
│                     │  [Traditional]: "The committee recommends..."    ││
│                     │  [Modern]: "ROI analysis shows..."               ││
│                     │  [Postmodern]: "This serves all members by..."   ││
│                     │  [Integral]: "Systems analysis indicates..."     ││
│                     │                                                  ││
│                     │  ┌─────────┐ ┌─────────┐ ┌─────────┐            ││
│                     │  │ Support │ │ Oppose  │ │ Abstain │            ││
│                     │  └─────────┘ └─────────┘ └─────────┘            ││
│                     │                                                  ││
│                     │  [Add your perspective to discussion →]         ││
│                     └─────────────────────────────────────────────────┘│
│                                                                         │
│  AFTER VOTING:                                                          │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐│
│  │  ✓ Your vote recorded                                              ││
│  │                                                                     ││
│  │  See how others voted (after voting period):                       ││
│  │  • Similar members: 72% support                                    ││
│  │  • Your trusted connections: 65% support                           ││
│  │                                                                     ││
│  │  Result notification in 5 days                                     ││
│  │                                                                     ││
│  │  [Track this proposal] [Explore related discussions]               ││
│  └─────────────────────────────────────────────────────────────────────┘│
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Flow 4: Getting Support (Mutual Aid)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    MUTUAL AID REQUEST FLOW                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────┐                                                       │
│  │ Need Help   │                                                       │
│  │ Button      │                                                       │
│  │ (always     │                                                       │
│  │ visible)    │                                                       │
│  └──────┬──────┘                                                       │
│         │                                                               │
│         ▼                                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐│
│  │  WHAT DO YOU NEED?                                                  ││
│  │                                                                     ││
│  │  ○ Financial support (unexpected expense, emergency)               ││
│  │  ○ Physical help (moving, repairs, transportation)                 ││
│  │  ○ Emotional support (someone to talk to)                          ││
│  │  ○ Childcare or eldercare                                          ││
│  │  ○ Food or supplies                                                ││
│  │  ○ Other: [____________]                                           ││
│  │                                                                     ││
│  │  URGENCY:                                                          ││
│  │  ○ Emergency (within hours)                                        ││
│  │  ○ Urgent (within days)                                            ││
│  │  ○ Can wait (within weeks)                                         ││
│  │                                                                     ││
│  │  VISIBILITY:                                                       ││
│  │  ○ Share with entire community                                     ││
│  │  ○ Share with trusted connections only                             ││
│  │  ○ Share with Mutual Aid team only                                 ││
│  │                                                                     ││
│  │  [Submit Request]                                                   ││
│  └─────────────────────────────────────────────────────────────────────┘│
│         │                                                               │
│         ▼                                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐│
│  │  YOUR REQUEST HAS BEEN RECEIVED                                     ││
│  │                                                                     ││
│  │  What happens next:                                                 ││
│  │  1. Matched neighbors will be notified                             ││
│  │  2. Mutual Aid coordinator may reach out                           ││
│  │  3. You'll receive offers via secure message                       ││
│  │                                                                     ││
│  │  Average response time for this request type: 4 hours              ││
│  │                                                                     ││
│  │  Need to talk to someone now?                                      ││
│  │  [Connect with Sanctuary Support →]                                ││
│  └─────────────────────────────────────────────────────────────────────┘│
│                                                                         │
│  RESPONDER EXPERIENCE:                                                  │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐│
│  │  🆘 NEIGHBOR NEEDS HELP                                            ││
│  │                                                                     ││
│  │  [Name] needs: Childcare support                                   ││
│  │  When: This Thursday, 2-6pm                                        ││
│  │  Details: Doctor's appointment, need someone trusted...            ││
│  │                                                                     ││
│  │  Your match score: 85%                                             ││
│  │  (You have: childcare experience, available Thursday, trusted)     ││
│  │                                                                     ││
│  │  [I Can Help] [Can't This Time] [Suggest Someone]                  ││
│  └─────────────────────────────────────────────────────────────────────┘│
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Migration Guides

### From Facebook Groups

```markdown
## Migrating from Facebook Groups to Mycelix

### Why Migrate?
- Own your community data
- Better privacy controls
- Integrated tools (not just discussion)
- No algorithmic manipulation
- Democratic governance built-in

### Migration Steps

#### Week 1: Parallel Operation
1. Create Mycelix community with same name/purpose
2. Invite moderators/admins as founding members
3. Mirror key announcements in both places
4. Add "Find us on Mycelix" to Facebook group description

#### Week 2-3: Gradual Transition
1. Post migration guide in Facebook group
2. Create incentive: "Early members get founding badges"
3. Move discussions of substance to Mycelix
4. Keep Facebook for announcements only

#### Week 4+: Complete Migration
1. Final announcement with migration deadline
2. Archive Facebook group (don't delete - people may need history)
3. Set Facebook to "archived" with redirect to Mycelix
4. Celebrate community milestone!

### Feature Mapping
| Facebook Feature | Mycelix Equivalent |
|------------------|-------------------|
| Posts | Chronicle / Hearth |
| Events | Nexus |
| Polls | Agora |
| Marketplace | Marketplace |
| Files | Chronicle |
| Member approval | Membrane + Attest |
| Admin tools | Agora governance |

### Common Concerns

**"Some members aren't tech-savvy"**
- Mycelix has simpler interfaces than Facebook
- Buddy system: pair tech-comfortable with less comfortable
- Phone support from community champions

**"We'll lose members"**
- Some will leave (natural churn)
- Committed members will follow
- Quality over quantity

**"Our history is on Facebook"**
- Export and archive Facebook data
- Import key posts to Chronicle
- Migration is about the future
```

### From Slack/Discord

```markdown
## Migrating from Slack/Discord to Mycelix

### Why Migrate?
- Beyond chat: governance, economics, projects
- No corporate platform dependency
- Community-owned infrastructure
- Integrated identity and trust

### Migration Steps

#### Phase 1: Identify Use Cases
Map current channels to Mycelix features:
- #general → Chronicle (announcements)
- #random → Hearth (social)
- #projects → Collab
- #help → Mutual / Marketplace
- #governance → Agora

#### Phase 2: Parallel Run (2-4 weeks)
1. Set up Mycelix with equivalent spaces
2. Cross-post important items
3. Move specific use cases first (e.g., governance → Agora)
4. Keep Slack/Discord for real-time chat initially

#### Phase 3: Primary Platform (2-4 weeks)
1. Make Mycelix the "source of truth"
2. Use Slack/Discord only for quick chat
3. All decisions/records in Mycelix
4. Train champions in each space

#### Phase 4: Complete (optional)
1. Archive Slack/Discord
2. Use Mycelix Mail for messaging needs
3. Full sovereignty achieved

### What You Gain
- Governance tools (not just discussion)
- Economic layer (exchange, shared resources)
- Identity you control
- Trust network
- No message history limits
- Community-owned forever
```

### From No Digital Tools

```markdown
## Starting Digital for Non-Digital Communities

### Why Go Digital?
- Scale coordination beyond physical meetings
- Include members who can't attend in person
- Persistent memory and records
- Asynchronous participation

### Starting Points

#### Option A: Start Simple
Begin with just 2-3 hApps:
1. **Attest** - Member directory
2. **Agora** - Decisions between meetings
3. **Chronicle** - Meeting notes and records

#### Option B: Start With Pain Point
What's hardest to coordinate?
- Scheduling → Start with Nexus
- Sharing resources → Start with Marketplace
- Emergency contact → Start with Beacon

#### Option C: Full Foundation
For communities ready to commit:
- All Tier 0 + selected Tier 1
- More setup, more capability

### Digital Adoption Tips

**For Technophobic Members:**
- Paper backup for everything
- Phone tree alongside digital
- Dedicated helpers for tech support
- Gradual, optional adoption

**For Limited Connectivity:**
- Offline-first features
- SMS notifications option
- Low-bandwidth mode
- Community access points

**Maintaining In-Person Culture:**
- Digital supplements, not replaces
- Meetings remain central
- Digital for between-meeting coordination
- Regular tech-free gatherings
```

---

## Success Metrics

### Community Health Indicators

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    COMMUNITY HEALTH DASHBOARD                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ENGAGEMENT                           TRUST                            │
│  ──────────                           ─────                            │
│  Daily Active Users: 127 (↑12%)       Avg Trust Score: 0.78 (↑)       │
│  Weekly Active Users: 312 (↑8%)       New Trust Edges: 45 this week   │
│  Monthly Active Users: 428 (↑5%)      Trust Reciprocity: 89%          │
│                                                                         │
│  CONTRIBUTION                         GOVERNANCE                        │
│  ────────────                         ──────────                        │
│  Active Contributors: 89              Proposals This Month: 12         │
│  New Offerings: 34 this week          Voting Participation: 67%        │
│  Exchanges Completed: 156             Proposals Passed: 9              │
│                                                                         │
│  RETENTION                            GROWTH                           │
│  ─────────                            ──────                           │
│  7-day Retention: 78%                 New Members: 23 this month       │
│  30-day Retention: 65%                Invites Sent: 67                 │
│  90-day Retention: 52%                Invite Conversion: 34%           │
│                                                                         │
│  WELLBEING (via Sanctuary)            ECONOMIC (via Treasury)          │
│  ─────────                            ────────                         │
│  Support Requests Met: 94%            Currency Velocity: 2.3x          │
│  Crisis Response Time: 2.1 hrs        Gini Coefficient: 0.28          │
│  Wellbeing Survey: 7.8/10             Mutual Aid: $4,200 distributed   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Adoption Milestones

| Milestone | Indicator | Target |
|-----------|-----------|--------|
| **Launch** | First 10 members onboarded | Week 1 |
| **Traction** | First 50 members, 10+ active daily | Month 1 |
| **Engagement** | 100+ members, 25%+ weekly active | Month 3 |
| **Sustainability** | Self-sustaining activity without founders pushing | Month 6 |
| **Growth** | Organic growth through member invites | Month 9 |
| **Maturity** | Full governance active, multiple active hApps | Month 12 |
| **Sovereignty** | Community self-manages all aspects | Month 18+ |

---

## Troubleshooting Adoption

### Common Challenges

**Challenge: Low Initial Engagement**
```
Symptoms: Members sign up but don't return
Causes: No clear first action, overwhelming interface, no immediate value

Solutions:
- Simplify onboarding to single first action
- Create "quick win" within first session
- Personal outreach to new members
- Buddy system for newcomers
- Regular community events (online/offline)
```

**Challenge: Power User Concentration**
```
Symptoms: Same 5-10 people do everything
Causes: High barrier to participation, unclear how to contribute

Solutions:
- Lower barriers for contribution
- Explicit calls for help on specific tasks
- Rotate responsibilities
- Recognition for new contributors
- Mentorship from power users to newcomers
```

**Challenge: Governance Fatigue**
```
Symptoms: Declining voting participation, same people making decisions
Causes: Too many proposals, unclear impact, decision overload

Solutions:
- Reduce proposal frequency
- Delegation options
- Clearer proposal summaries
- Show impact of past decisions
- Quadratic voting to prioritize engagement
```

**Challenge: Tech Barriers**
```
Symptoms: Certain demographics not participating
Causes: Complexity, accessibility issues, connectivity limitations

Solutions:
- Simplified interfaces for core functions
- Accessibility audit and improvements
- Offline capabilities
- Phone/SMS options
- In-person support sessions
- Multi-language support
```

---

## Support Resources

### For Community Founders
- [Community Template Configurator](tools/template-configurator)
- [Founding Member Training](training/founders)
- [Governance Design Workshop](workshops/governance)
- [Office Hours with Mycelix Team](support/office-hours)

### For Members
- [Getting Started Guide](guides/getting-started)
- [Video Tutorials](tutorials/video)
- [FAQ](support/faq)
- [Community Forum](community/forum)

### For Developers
- [Developer Guide](implementation/DEVELOPER_GUIDE.md)
- [API Reference](api/reference)
- [GitHub Repository](https://github.com/mycelix)

---

*"Every community's journey is unique. These playbooks are starting points, not prescriptions. Adapt, iterate, and make Mycelix your own."*

---

*Document Version: 1.0*
*Last Updated: 2025*

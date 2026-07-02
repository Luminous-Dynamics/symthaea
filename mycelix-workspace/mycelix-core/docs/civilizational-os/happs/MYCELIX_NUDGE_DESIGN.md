# Mycelix-Nudge: Choice Architecture & Behavioral Design Layer

## Vision Statement

*"Humans are not purely rational beings, and that's beautiful. Nudge provides the ethical choice architecture that helps humans make decisions aligned with their values and community wellbeing - not through manipulation, but through wise design."*

---

## Executive Summary

Mycelix-Nudge is the behavioral design layer of the civilizational OS:

1. **Choice architecture** - Ethical defaults and decision environments
2. **Commitment devices** - Tools for long-term goal alignment
3. **Feedback systems** - Timely, relevant behavioral feedback
4. **Social proof** - Transparent community behavior visibility
5. **Friction design** - Strategic ease/difficulty for behavior shaping

Nudge is NOT about manipulation - it's about helping humans overcome cognitive limitations in service of their own stated values.

---

## Core Principles

### The Nudge Ethics Framework

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    NUDGE ETHICS                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  1. TRANSPARENCY                                                        │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  • All nudges are documented and visible                        │   │
│  │  • Users can see why defaults are set as they are              │   │
│  │  • "Why am I seeing this?" always answered                     │   │
│  │  • No hidden manipulation                                       │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  2. AUTONOMY PRESERVING                                                 │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  • Every nudge can be overridden                               │   │
│  │  • "Nudge me less" is always an option                         │   │
│  │  • No dark patterns, ever                                       │   │
│  │  • Final choice belongs to user                                 │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  3. VALUE ALIGNED                                                       │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  • Nudges serve user's stated values                           │   │
│  │  • User defines their goals, Nudge helps achieve them          │   │
│  │  • Community values transparent and governable                  │   │
│  │  • No nudging toward system's interests over user's           │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  4. ACCOUNTABLE                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  • Nudge effectiveness measured                                 │   │
│  │  • Unintended consequences monitored                           │   │
│  │  • Community governance over significant nudges                │   │
│  │  • Regular ethics review                                        │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ANTI-PATTERNS (NEVER DO)                                              │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  ✗ Hidden defaults that benefit platform over user             │   │
│  │  ✗ Manufactured urgency ("Only 2 left!")                       │   │
│  │  ✗ Shame-based manipulation                                     │   │
│  │  ✗ Exploiting addiction psychology                              │   │
│  │  ✗ Making opt-out deliberately difficult                       │   │
│  │  ✗ Misdirection or confusion                                    │   │
│  │  ✗ Social pressure that harms                                   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Architecture Overview

### System Components

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    NUDGE ARCHITECTURE                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    AGENT LAYER                                   │   │
│  │  Personal values │ Goals │ Preferences │ Developmental stage    │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                               │                                         │
│                               ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    NUDGE CORE                                    │   │
│  │  ┌───────────────┐ ┌───────────────┐ ┌───────────────────────┐  │   │
│  │  │   Default     │ │  Commitment   │ │      Feedback         │  │   │
│  │  │   Manager     │ │   Devices     │ │      Engine           │  │   │
│  │  └───────────────┘ └───────────────┘ └───────────────────────┘  │   │
│  │  ┌───────────────┐ ┌───────────────┐ ┌───────────────────────┐  │   │
│  │  │   Social      │ │   Friction    │ │      Timing           │  │   │
│  │  │   Proof       │ │   Design      │ │      Optimizer        │  │   │
│  │  └───────────────┘ └───────────────┘ └───────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                               │                                         │
│                               ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    INTEGRATION LAYER                             │   │
│  │  All hApps │ UI Components │ Notification System │ Chronicle    │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Key Data Types

```rust
/// Agent's nudge profile and preferences
#[hdk_entry_helper]
pub struct NudgeProfile {
    pub agent: AgentPubKey,

    // Values and goals (user-defined)
    pub stated_values: Vec<Value>,
    pub active_goals: Vec<Goal>,

    // Nudge preferences
    pub nudge_intensity: NudgeIntensity,
    pub nudge_categories: HashMap<NudgeCategory, bool>,
    pub quiet_hours: Vec<TimeRange>,

    // Developmental context (optional, self-assessed)
    pub developmental_stage: Option<DevelopmentalStage>,
    pub preferred_framing: Option<FramingStyle>,

    // Learning
    pub nudge_effectiveness: HashMap<String, f64>,
    pub opted_out_nudges: Vec<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum NudgeIntensity {
    Minimal,                               // Only critical nudges
    Light,                                 // Occasional helpful nudges
    Moderate,                              // Regular support
    Active,                                // Frequent engagement
    Intensive,                             // Maximum support
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum NudgeCategory {
    Health,
    Finance,
    Environment,
    Social,
    Governance,
    Learning,
    Wellbeing,
    Productivity,
}

/// A specific nudge definition
#[hdk_entry_helper]
pub struct NudgeDefinition {
    pub nudge_id: String,
    pub name: String,
    pub description: String,

    // Targeting
    pub applies_to: NudgeScope,
    pub trigger_conditions: Vec<TriggerCondition>,

    // Behavior
    pub nudge_type: NudgeType,
    pub intensity: NudgeIntensity,
    pub category: NudgeCategory,

    // Content
    pub message_templates: HashMap<DevelopmentalStage, MessageTemplate>,
    pub action_options: Vec<ActionOption>,

    // Ethics
    pub ethical_review: EthicalReview,
    pub transparency_info: String,
    pub override_ease: OverrideEase,

    // Governance
    pub created_by: NudgeCreator,
    pub approved_by: Option<ActionHash>,   // Agora governance
    pub effectiveness_data: EffectivenessData,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum NudgeType {
    // Defaults
    DefaultSetting {
        setting: String,
        default_value: Value,
        rationale: String,
    },

    // Timing
    Reminder {
        timing: ReminderTiming,
        message: String,
    },
    JustInTime {
        context: String,
        intervention: String,
    },

    // Social
    SocialProof {
        comparison_group: String,
        behavior: String,
    },
    Celebration {
        achievement: String,
        visibility: Visibility,
    },

    // Commitment
    CommitmentPrompt {
        goal: String,
        commitment_type: CommitmentType,
    },
    ProgressFeedback {
        goal: String,
        visualization: String,
    },

    // Friction
    CoolingOff {
        action: String,
        delay: Duration,
        reason: String,
    },
    ConfirmationRequired {
        action: String,
        reflection_prompt: String,
    },

    // Framing
    LossFrame {
        behavior: String,
        potential_loss: String,
    },
    GainFrame {
        behavior: String,
        potential_gain: String,
    },

    // Information
    FutureVisualization {
        scenario: String,
        time_horizon: Duration,
    },
    ConsequencePreview {
        action: String,
        consequences: Vec<String>,
    },
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct MessageTemplate {
    pub stage: DevelopmentalStage,
    pub primary_message: String,
    pub supporting_context: String,
    pub framing: FramingStyle,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum FramingStyle {
    DutyBased,                             // "This is the right thing"
    AchievementBased,                      // "This is the smart thing"
    CommunityBased,                        // "This helps everyone"
    IntegralBased,                         // "This serves the whole"
}

/// Commitment device
#[hdk_entry_helper]
pub struct CommitmentDevice {
    pub commitment_id: String,
    pub agent: AgentPubKey,
    pub goal: Goal,

    // Commitment type
    pub device_type: CommitmentDeviceType,

    // Stakes
    pub stakes: Option<Stakes>,

    // Accountability
    pub accountability_partners: Vec<AgentPubKey>,
    pub check_in_schedule: Schedule,

    // Status
    pub status: CommitmentStatus,
    pub progress: Vec<ProgressEntry>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum CommitmentDeviceType {
    // Public commitment
    PublicDeclaration {
        visibility: Visibility,
    },

    // Financial stakes
    FinancialStake {
        amount: Decimal,
        beneficiary_if_fail: Beneficiary,
    },

    // Social accountability
    AccountabilityPartner {
        partner: AgentPubKey,
        check_in_frequency: Duration,
    },

    // Implementation intention
    ImplementationIntention {
        trigger: String,
        behavior: String,
        location: Option<String>,
    },

    // Ulysses contract
    PrecommitmentContract {
        future_self_restriction: String,
        activation_condition: String,
    },
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum Beneficiary {
    Charity(String),
    CommunityTreasury,
    AntiCharity(String),                   // Motivation by aversion
    AccountabilityPartner(AgentPubKey),
}

/// Social proof data
#[hdk_entry_helper]
pub struct SocialProofData {
    pub context: String,
    pub behavior: String,
    pub comparison_group: ComparisonGroup,
    pub statistics: SocialProofStats,
    pub anonymization: AnonymizationLevel,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SocialProofStats {
    pub percentage_doing: f64,
    pub trend: Trend,
    pub sample_size: u64,
    pub calculated_at: Timestamp,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum ComparisonGroup {
    SimilarHousehold,
    Neighbors,
    CommunityMembers,
    PeersInRole(String),
    SimilarGoals,
    Custom(String),
}
```

---

## Key Nudge Patterns

### Pattern 1: Wise Defaults

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    WISE DEFAULTS                                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  PRIVACY DEFAULTS                                                       │
│  • Default: Maximum privacy protection                                 │
│  • Rationale: Harder to un-share than to share later                  │
│  • Override: Easy, one-click to increase sharing                       │
│                                                                         │
│  GOVERNANCE DEFAULTS                                                    │
│  • Default: Enrolled in governance notifications                       │
│  • Rationale: Democratic participation is valuable                     │
│  • Override: Easy opt-out, no shame                                    │
│                                                                         │
│  SUSTAINABILITY DEFAULTS                                               │
│  • Default: Green/sustainable options selected                         │
│  • Rationale: Aligns with stated community values                     │
│  • Override: Available, requires brief reflection                      │
│                                                                         │
│  SAVINGS DEFAULTS                                                       │
│  • Default: Small % to personal/community savings                     │
│  • Rationale: Present bias works against future self                  │
│  • Override: Easy to change rate                                       │
│                                                                         │
│  TRANSPARENCY: Each default has a "Why this default?" link            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Pattern 2: Commitment Devices

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    COMMITMENT PATTERNS                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  USER JOURNEY:                                                          │
│                                                                         │
│  1. User identifies a goal                                             │
│     "I want to exercise more"                                          │
│                                                                         │
│  2. Nudge offers commitment device options                             │
│     • Public declaration (social pressure)                             │
│     • Accountability partner (social support)                          │
│     • Financial stake (loss aversion)                                  │
│     • Implementation intention (specificity)                           │
│                                                                         │
│  3. User chooses and configures                                        │
│     "I'll exercise on Mon/Wed/Fri mornings with partner notification" │
│                                                                         │
│  4. System provides support                                            │
│     • Reminders at chosen times                                        │
│     • Progress tracking                                                │
│     • Partner check-ins                                                │
│     • Celebration of streaks                                           │
│                                                                         │
│  5. Adaptive learning                                                  │
│     • What works for this user?                                        │
│     • Adjust timing, framing, intensity                               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Pattern 3: Ethical Social Proof

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    SOCIAL PROOF ETHICS                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  GOOD USES:                                                             │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  "78% of your neighbors participate in governance"              │   │
│  │  "Community members saved an average of 12% on energy"          │   │
│  │  "Most people in your role complete this training"              │   │
│  │                                                                  │   │
│  │  → Helps overcome bystander effect                              │   │
│  │  → Normalizes positive behaviors                                │   │
│  │  → Provides useful information                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  BAD USES (PROHIBITED):                                                │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  ✗ "Everyone is buying this - don't miss out!"                 │   │
│  │  ✗ "You're behind your peers" (shame-based)                    │   │
│  │  ✗ Fake social proof numbers                                   │   │
│  │  ✗ Using social proof for commercial manipulation              │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  SAFEGUARDS:                                                            │
│  • All social proof based on real, auditable data                     │
│  • Comparison groups are meaningful and fair                          │
│  • User can opt out of seeing social proof                            │
│  • Never combined with scarcity or urgency                            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Pattern 4: Friction for Reflection

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    STRATEGIC FRICTION                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  COOLING-OFF PERIODS                                                    │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Large purchases: 24-hour delay option                          │   │
│  │  Angry messages: "Send tomorrow?" prompt                        │   │
│  │  Major decisions: Sleep on it reminder                          │   │
│  │  Relationship changes: Confirmation after reflection            │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  REFLECTION PROMPTS                                                     │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Before large expense: "Is this aligned with your goals?"       │   │
│  │  Before conflict escalation: "What outcome do you want?"        │   │
│  │  Before privacy reduction: "Who will see this?"                │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  FRICTION REMOVAL                                                       │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Positive behaviors should be frictionless:                     │   │
│  │  • One-click governance participation                           │   │
│  │  • Easy donation flows                                          │   │
│  │  • Seamless sustainability choices                              │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## hApp-Specific Nudges

### Nudges by hApp

```rust
pub struct HAppNudges {
    // Treasury
    pub savings_nudges: Vec<NudgeDefinition>,
    pub contribution_nudges: Vec<NudgeDefinition>,

    // Agora
    pub participation_nudges: Vec<NudgeDefinition>,
    pub deliberation_nudges: Vec<NudgeDefinition>,

    // HealthVault
    pub health_behavior_nudges: Vec<NudgeDefinition>,
    pub preventive_care_nudges: Vec<NudgeDefinition>,

    // Sanctuary
    pub wellbeing_checkin_nudges: Vec<NudgeDefinition>,
    pub connection_nudges: Vec<NudgeDefinition>,

    // Ember
    pub energy_conservation_nudges: Vec<NudgeDefinition>,
    pub peak_avoidance_nudges: Vec<NudgeDefinition>,

    // Provision
    pub sustainable_food_nudges: Vec<NudgeDefinition>,
    pub local_sourcing_nudges: Vec<NudgeDefinition>,

    // Kinship
    pub care_reciprocity_nudges: Vec<NudgeDefinition>,
    pub connection_maintenance_nudges: Vec<NudgeDefinition>,
}
```

### Example Nudge Implementations

```rust
// Energy conservation nudge for Ember
let energy_nudge = NudgeDefinition {
    nudge_id: "ember_peak_shift",
    name: "Peak Demand Shift",
    description: "Encourage shifting energy use away from peak hours",

    nudge_type: NudgeType::SocialProof {
        comparison_group: "neighbors",
        behavior: "peak hour reduction",
    },

    message_templates: hashmap! {
        DevelopmentalStage::Traditional => MessageTemplate {
            stage: DevelopmentalStage::Traditional,
            primary_message: "Your community depends on responsible energy use during peak hours.",
            supporting_context: "This is what good neighbors do.",
            framing: FramingStyle::DutyBased,
        },
        DevelopmentalStage::Modern => MessageTemplate {
            stage: DevelopmentalStage::Modern,
            primary_message: "Save $45/month by shifting 20% of your usage off-peak.",
            supporting_context: "Smart meters show peak usage costs 3x more.",
            framing: FramingStyle::AchievementBased,
        },
        DevelopmentalStage::Postmodern => MessageTemplate {
            stage: DevelopmentalStage::Postmodern,
            primary_message: "75% of your community has reduced peak usage, helping everyone.",
            supporting_context: "Together we're reducing the need for fossil fuel peakers.",
            framing: FramingStyle::CommunityBased,
        },
    },

    ethical_review: EthicalReview {
        transparent: true,
        overridable: true,
        serves_user: true,
        serves_community: true,
        reviewed_by: "Energy Ethics Working Group",
        approved_date: timestamp!("2025-01-15"),
    },
};
```

---

## Integration with Spiral (Developmental Stages)

### Stage-Appropriate Nudging

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    DEVELOPMENTAL NUDGE CALIBRATION                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  STAGE: TRADITIONAL (Blue/Amber)                                       │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Effective nudges:                                              │   │
│  │  • Authority endorsements ("Elders recommend...")               │   │
│  │  • Tradition appeals ("This is how we've always...")           │   │
│  │  • Duty framing ("Your responsibility is...")                  │   │
│  │  • Clear rules ("The policy is...")                            │   │
│  │                                                                  │   │
│  │  Avoid:                                                         │   │
│  │  • Relativistic framing                                         │   │
│  │  • Questioning established norms                                │   │
│  │  • Too many choices                                             │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  STAGE: MODERN (Orange)                                                │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Effective nudges:                                              │   │
│  │  • Data and metrics ("Studies show...")                        │   │
│  │  • Achievement framing ("Top performers do...")                │   │
│  │  • Efficiency appeals ("Fastest way to...")                    │   │
│  │  • Personal benefit ("You'll gain...")                         │   │
│  │                                                                  │   │
│  │  Avoid:                                                         │   │
│  │  • Pure tradition appeals                                       │   │
│  │  • Emotional-only arguments                                     │   │
│  │  • Vague community benefits                                     │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  STAGE: POSTMODERN (Green)                                             │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Effective nudges:                                              │   │
│  │  • Inclusion appeals ("Everyone benefits...")                  │   │
│  │  • Feeling validation ("How does this feel?")                  │   │
│  │  • Environmental framing ("For the planet...")                 │   │
│  │  • Consensus reference ("The community agrees...")             │   │
│  │                                                                  │   │
│  │  Avoid:                                                         │   │
│  │  • Hierarchy language                                           │   │
│  │  • Competition framing                                          │   │
│  │  • Pure efficiency arguments                                    │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  STAGE: INTEGRAL (Teal+)                                               │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Effective nudges:                                              │   │
│  │  • Systems views ("This affects the whole...")                 │   │
│  │  • Multi-perspective ("From one view... from another...")      │   │
│  │  • Developmental awareness ("At this stage...")                │   │
│  │  • Integration language ("Both/and rather than either/or")     │   │
│  │                                                                  │   │
│  │  Avoid:                                                         │   │
│  │  • Single-quadrant solutions                                    │   │
│  │  • Stage absolutism                                             │   │
│  │  • Oversimplification                                           │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Governance of Nudges

### Community Control

```rust
pub struct NudgeGovernance {
    // Approval requirements
    pub requires_approval: ApprovalLevel,
    pub approval_body: ActionHash,         // Agora governance space

    // Transparency
    pub public_registry: bool,             // All nudges visible
    pub effectiveness_reporting: bool,     // Results published

    // Rights
    pub individual_opt_out: bool,          // Always true
    pub community_can_disable: bool,       // Via governance
    pub sunset_clause: Option<Duration>,   // Auto-expiry

    // Review
    pub ethics_review_required: bool,
    pub periodic_review: Duration,
}

pub enum ApprovalLevel {
    Individual,                            // Agent-only nudges
    Community,                             // Community-wide nudges
    Ecosystem,                             // Cross-hApp nudges
}
```

---

## Measurement & Learning

### Effectiveness Tracking

```rust
pub struct NudgeEffectiveness {
    pub nudge_id: String,

    // Engagement
    pub impressions: u64,
    pub interactions: u64,
    pub opt_outs: u64,

    // Outcomes
    pub behavior_change_rate: f64,
    pub goal_achievement_rate: f64,
    pub user_satisfaction: f64,

    // Equity
    pub effectiveness_by_stage: HashMap<DevelopmentalStage, f64>,
    pub no_adverse_effects: bool,

    // Learning
    pub a_b_test_results: Vec<ABTestResult>,
    pub improvement_suggestions: Vec<String>,
}
```

---

## Conclusion

Nudge provides the ethical choice architecture layer that helps humans make decisions aligned with their own values. By combining behavioral economics insights with developmental awareness, it supports human flourishing without manipulation.

*"The best nudge is one that helps you become who you want to be, faster."*

---

*Document Version: 1.0*
*Classification: Tier 1 - Core (Cross-cutting infrastructure)*
*Dependencies: All hApps, Spiral (developmental awareness)*

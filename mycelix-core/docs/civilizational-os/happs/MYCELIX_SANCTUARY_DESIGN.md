# Mycelix-Sanctuary: Wellbeing & Crisis Support Infrastructure

## Vision Statement

*"A civilization that cannot hold its members in their darkest moments is no civilization at all. Sanctuary provides the infrastructure for mutual care, crisis support, and collective wellbeing - not as an afterthought, but as a foundational pillar of human coordination."*

---

## Executive Summary

Mycelix-Sanctuary is the wellbeing infrastructure layer of the civilizational OS. It provides:

1. **Peer support networks** - Trained community members offering mutual aid
2. **Crisis intervention** - Rapid response for mental health emergencies
3. **Wellbeing tracking** - Private, agent-controlled mental health metrics
4. **Resource coordination** - Connecting people with appropriate support
5. **Community healing** - Collective trauma processing and resilience building

Sanctuary recognizes that human flourishing requires more than economic coordination - it requires emotional infrastructure.

---

## Why Sanctuary is Foundational

### The Gap in Current Systems

| Current Approach | Problem |
|------------------|---------|
| Emergency services | Criminalization of mental health crises |
| Private therapy | Cost-prohibitive, accessibility barriers |
| Social media | Performative, often harmful |
| Isolation | Default response to struggle |
| Stigma | Prevents help-seeking |

### The Sanctuary Approach

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    SANCTUARY PHILOSOPHY                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  1. MUTUAL AID OVER PROFESSIONALIZATION                                │
│     Communities can hold each other                                    │
│     Professionals are resources, not gatekeepers                       │
│     Everyone has capacity to support and be supported                  │
│                                                                         │
│  2. PRIVACY AS HEALING PREREQUISITE                                    │
│     Vulnerability requires safety                                      │
│     Agent controls all wellbeing data                                  │
│     No surveillance, no involuntary disclosure                         │
│                                                                         │
│  3. PREVENTION OVER CRISIS RESPONSE                                    │
│     Build resilience before breakdown                                  │
│     Community connection prevents isolation                            │
│     Early intervention, not last resort                                │
│                                                                         │
│  4. COLLECTIVE HEALING                                                 │
│     Individual wellbeing and community health are inseparable          │
│     Trauma can be held collectively                                    │
│     Healing is relational                                              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Architecture Overview

### System Layers

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    SANCTUARY ARCHITECTURE                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    INDIVIDUAL LAYER                              │   │
│  │  ┌───────────────┐ ┌───────────────┐ ┌───────────────────────┐  │   │
│  │  │   Wellbeing   │ │   Personal    │ │    Safety            │  │   │
│  │  │   Tracking    │ │   Journal     │ │    Planning          │  │   │
│  │  └───────────────┘ └───────────────┘ └───────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                               │                                         │
│                               ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    RELATIONAL LAYER                              │   │
│  │  ┌───────────────┐ ┌───────────────┐ ┌───────────────────────┐  │   │
│  │  │    Trusted    │ │    Peer       │ │    Support           │  │   │
│  │  │    Circle     │ │    Matching   │ │    Requests          │  │   │
│  │  └───────────────┘ └───────────────┘ └───────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                               │                                         │
│                               ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    COMMUNITY LAYER                               │   │
│  │  ┌───────────────┐ ┌───────────────┐ ┌───────────────────────┐  │   │
│  │  │    Support    │ │   Community   │ │    Collective        │  │   │
│  │  │    Circles    │ │   Check-ins   │ │    Healing           │  │   │
│  │  └───────────────┘ └───────────────┘ └───────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                               │                                         │
│                               ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    CRISIS LAYER                                  │   │
│  │  ┌───────────────┐ ┌───────────────┐ ┌───────────────────────┐  │   │
│  │  │    Crisis     │ │   Rapid       │ │    Professional      │  │   │
│  │  │    Protocol   │ │   Response    │ │    Escalation        │  │   │
│  │  └───────────────┘ └───────────────┘ └───────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                               │                                         │
│                               ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    RESOURCE LAYER                                │   │
│  │  ┌───────────────┐ ┌───────────────┐ ┌───────────────────────┐  │   │
│  │  │  Professional │ │   Community   │ │    Emergency         │  │   │
│  │  │   Directory   │ │   Resources   │ │    Services          │  │   │
│  │  └───────────────┘ └───────────────┘ └───────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Zome Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         SANCTUARY ZOMES                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  INTEGRITY ZOMES                                                        │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                                                                  │   │
│  │  sanctuary_types                                                │   │
│  │  ├── WellbeingEntry       (private mood/energy tracking)       │   │
│  │  ├── JournalEntry         (encrypted personal reflection)      │   │
│  │  ├── SafetyPlan           (crisis preparation document)        │   │
│  │  ├── TrustedCircle        (designated support people)          │   │
│  │  ├── SupportRequest       (asking for help)                    │   │
│  │  ├── SupportOffer         (offering to help)                   │   │
│  │  ├── PeerSupporterProfile (trained supporter info)             │   │
│  │  ├── SupportSession       (record of support interaction)      │   │
│  │  ├── CommunityCircle      (support group definition)           │   │
│  │  ├── CrisisAlert          (emergency activation)               │   │
│  │  ├── ResourceListing      (professional/service directory)     │   │
│  │  └── CollectiveHealing    (community processing space)         │   │
│  │                                                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  COORDINATOR ZOMES                                                      │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                                                                  │   │
│  │  wellbeing_tracker                                              │   │
│  │  ├── record_checkin()          // Daily/periodic mood entry    │   │
│  │  ├── view_trends()             // Personal pattern analysis    │   │
│  │  ├── set_alerts()              // Threshold notifications      │   │
│  │  ├── share_with_circle()       // Selective sharing            │   │
│  │  └── export_for_professional() // Therapy integration          │   │
│  │                                                                  │   │
│  │  trusted_circle                                                 │   │
│  │  ├── invite_to_circle()                                        │   │
│  │  ├── accept_invitation()                                       │   │
│  │  ├── set_permissions()         // What they can see/do         │   │
│  │  ├── alert_circle()            // Notify trusted people        │   │
│  │  └── remove_from_circle()                                      │   │
│  │                                                                  │   │
│  │  safety_planning                                                │   │
│  │  ├── create_safety_plan()                                      │   │
│  │  ├── update_safety_plan()                                      │   │
│  │  ├── share_with_circle()                                       │   │
│  │  ├── activate_safety_plan()    // Crisis mode                  │   │
│  │  └── deactivate_safety_plan()                                  │   │
│  │                                                                  │   │
│  │  peer_support                                                   │   │
│  │  ├── register_as_supporter()                                   │   │
│  │  ├── update_availability()                                     │   │
│  │  ├── request_support()                                         │   │
│  │  ├── offer_support()                                           │   │
│  │  ├── match_supporter()         // Algorithm-assisted           │   │
│  │  ├── start_session()                                           │   │
│  │  ├── end_session()                                             │   │
│  │  └── provide_feedback()        // Both directions              │   │
│  │                                                                  │   │
│  │  community_circles                                              │   │
│  │  ├── create_circle()           // Support group                │   │
│  │  ├── join_circle()                                             │   │
│  │  ├── schedule_gathering()                                      │   │
│  │  ├── facilitate_session()                                      │   │
│  │  ├── share_in_circle()         // Protected sharing            │   │
│  │  └── archive_circle()                                          │   │
│  │                                                                  │   │
│  │  crisis_response                                                │   │
│  │  ├── declare_crisis()          // Self or circle-initiated     │   │
│  │  ├── activate_response()       // Mobilize support             │   │
│  │  ├── coordinate_responders()                                   │   │
│  │  ├── escalate_to_professional()                                │   │
│  │  ├── document_intervention()                                   │   │
│  │  └── debrief_responders()                                      │   │
│  │                                                                  │   │
│  │  resource_directory                                             │   │
│  │  ├── list_resource()                                           │   │
│  │  ├── search_resources()                                        │   │
│  │  ├── verify_professional()     // Credential check via Attest  │   │
│  │  ├── rate_resource()                                           │   │
│  │  └── report_resource()         // Safety concerns              │   │
│  │                                                                  │   │
│  │  collective_healing                                             │   │
│  │  ├── propose_healing_space()   // Community trauma response    │   │
│  │  ├── join_healing_space()                                      │   │
│  │  ├── contribute_to_healing()                                   │   │
│  │  ├── witness()                 // Presence without speaking    │   │
│  │  └── close_healing_space()                                     │   │
│  │                                                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Data Model

### Core Entry Types

```rust
/// Private wellbeing check-in (agent's source chain only)
#[hdk_entry_helper]
pub struct WellbeingEntry {
    pub entry_id: String,
    pub timestamp: Timestamp,

    // Core metrics (all optional - agent chooses what to track)
    pub mood: Option<MoodScale>,           // 1-10 or emoji-based
    pub energy: Option<EnergyLevel>,       // 1-10
    pub anxiety: Option<AnxietyLevel>,     // 1-10
    pub sleep_quality: Option<SleepQuality>,
    pub social_connection: Option<ConnectionLevel>,

    // Context
    pub activities: Vec<Activity>,
    pub triggers: Vec<String>,
    pub supports: Vec<String>,            // What helped

    // Reflection
    pub notes: Option<EncryptedText>,
    pub gratitudes: Vec<String>,
    pub intentions: Vec<String>,

    // Flags
    pub crisis_indicators: Vec<CrisisIndicator>,
    pub share_with_circle: bool,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct MoodScale {
    pub value: u8,                         // 1-10
    pub label: Option<String>,             // "hopeful", "struggling", etc.
    pub emoji: Option<String>,             // Visual representation
}

/// Safety plan for crisis preparation
#[hdk_entry_helper]
pub struct SafetyPlan {
    pub plan_id: String,
    pub agent: AgentPubKey,
    pub created_at: Timestamp,
    pub last_updated: Timestamp,

    // Warning signs
    pub warning_signs: Vec<WarningSign>,

    // Coping strategies
    pub internal_coping: Vec<CopingStrategy>,    // Things I can do alone
    pub external_coping: Vec<CopingStrategy>,    // Places/activities that help
    pub social_coping: Vec<SocialSupport>,       // People who help

    // Support contacts (ordered by escalation)
    pub trusted_circle: Vec<TrustedContact>,
    pub professional_contacts: Vec<ProfessionalContact>,
    pub crisis_lines: Vec<CrisisLine>,

    // Environment safety
    pub environment_safety: Vec<SafetyMeasure>,

    // Reasons for living
    pub reasons_for_living: Vec<String>,

    // Activation conditions
    pub auto_activate_conditions: Vec<ActivationCondition>,

    // Status
    pub status: SafetyPlanStatus,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct WarningSign {
    pub description: String,
    pub severity: Severity,
    pub observable_by_others: bool,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TrustedContact {
    pub agent: Option<AgentPubKey>,        // If in Mycelix
    pub name: String,
    pub relationship: String,
    pub contact_method: ContactMethod,
    pub permissions: CirclePermissions,
    pub escalation_order: u8,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CirclePermissions {
    pub can_see_checkins: bool,
    pub can_see_trends: bool,
    pub can_see_safety_plan: bool,
    pub can_activate_safety_plan: bool,
    pub can_contact_on_alert: bool,
    pub can_see_location_in_crisis: bool,
}

/// Peer supporter profile
#[hdk_entry_helper]
pub struct PeerSupporterProfile {
    pub supporter_id: String,
    pub agent: AgentPubKey,

    // Training and credentials
    pub training_completed: Vec<TrainingCredential>,
    pub specializations: Vec<Specialization>,
    pub languages: Vec<String>,
    pub cultural_competencies: Vec<String>,

    // Availability
    pub availability: AvailabilitySchedule,
    pub max_active_supportees: u8,
    pub current_active: u8,

    // Preferences
    pub support_preferences: SupportPreferences,

    // Track record (privacy-preserving)
    pub sessions_completed: u64,
    pub average_rating: Option<f64>,
    pub feedback_themes: Vec<String>,      // Aggregated, anonymized

    // Supervision
    pub supervisor: Option<AgentPubKey>,
    pub last_supervision: Option<Timestamp>,

    // Status
    pub status: SupporterStatus,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TrainingCredential {
    pub training_name: String,
    pub provider: String,
    pub completed_at: Timestamp,
    pub credential_hash: Option<ActionHash>,  // Via Attest/Praxis
    pub renewal_required: Option<Timestamp>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum Specialization {
    GeneralSupport,
    GriefAndLoss,
    AnxietySupport,
    DepressionSupport,
    TraumaInformed,
    SubstanceRecovery,
    LGBTQ,
    YouthSupport,
    ElderSupport,
    ParentingSupport,
    RelationshipSupport,
    WorkplaceStress,
    ChronicIllness,
    Neurodivergent,
    CulturalSpecific(String),
    Other(String),
}

/// Support request
#[hdk_entry_helper]
pub struct SupportRequest {
    pub request_id: String,
    pub requester: AgentPubKey,
    pub created_at: Timestamp,

    // What kind of support
    pub support_type: SupportType,
    pub urgency: Urgency,
    pub topic_areas: Vec<Specialization>,

    // Preferences
    pub preferences: MatchingPreferences,

    // Context (optional, helps matching)
    pub brief_context: Option<EncryptedText>,

    // Status
    pub status: RequestStatus,
    pub matched_supporter: Option<AgentPubKey>,
    pub session_hash: Option<ActionHash>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum SupportType {
    Listening,           // Just need to be heard
    ProblemSolving,      // Help thinking through something
    CrisisSupport,       // Acute distress
    CheckIn,             // Regular connection
    Accountability,      // Help staying on track
    Celebration,         // Share something positive
    Grief,               // Loss processing
    Guidance,            // Seeking perspective
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum Urgency {
    Flexible,            // Within a few days
    Soon,                // Within 24 hours
    Urgent,              // Within a few hours
    Crisis,              // Immediate (activates crisis protocol)
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct MatchingPreferences {
    pub preferred_languages: Vec<String>,
    pub gender_preference: Option<GenderPreference>,
    pub age_range_preference: Option<(u8, u8)>,
    pub cultural_preferences: Vec<String>,
    pub avoid_agents: Vec<AgentPubKey>,    // Previous bad matches
    pub communication_style: Option<CommunicationStyle>,
}

/// Community support circle
#[hdk_entry_helper]
pub struct CommunityCircle {
    pub circle_id: String,
    pub name: String,
    pub description: String,

    // Focus
    pub circle_type: CircleType,
    pub topic_focus: Vec<Specialization>,

    // Membership
    pub facilitators: Vec<AgentPubKey>,
    pub members: Vec<CircleMember>,
    pub max_members: Option<u8>,
    pub membership_criteria: MembershipCriteria,

    // Schedule
    pub meeting_schedule: Option<Schedule>,
    pub next_gathering: Option<Timestamp>,

    // Guidelines
    pub community_agreements: Vec<String>,
    pub confidentiality_level: ConfidentialityLevel,

    // Governance
    pub governed_by: Option<ActionHash>,   // Agora space

    // Status
    pub status: CircleStatus,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum CircleType {
    DropIn,              // Anyone can join any session
    Closed,              // Fixed membership
    Rolling,             // New members at intervals
    ProcessGroup,        // Therapeutic structure
    MutualAid,           // Practical support focus
    Identity,            // Shared identity (LGBTQ, cultural, etc.)
    Condition,           // Shared condition (anxiety, grief, etc.)
    Recovery,            // 12-step or similar
    Peer,                // Peer support training/supervision
}

/// Crisis alert
#[hdk_entry_helper]
pub struct CrisisAlert {
    pub alert_id: String,
    pub agent: AgentPubKey,
    pub initiated_by: CrisisInitiator,
    pub created_at: Timestamp,

    // Crisis details
    pub crisis_type: CrisisType,
    pub severity: CrisisSeverity,
    pub description: Option<EncryptedText>,

    // Location (if agent consents)
    pub location: Option<EncryptedLocation>,

    // Safety plan reference
    pub safety_plan: Option<ActionHash>,

    // Response
    pub responders: Vec<Responder>,
    pub actions_taken: Vec<CrisisAction>,
    pub professional_involved: bool,
    pub emergency_services_called: bool,

    // Status
    pub status: CrisisStatus,
    pub resolved_at: Option<Timestamp>,
    pub follow_up_plan: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum CrisisInitiator {
    Self_,                               // Agent activated own crisis
    TrustedCircle(AgentPubKey),         // Circle member activated
    PeerSupporter(AgentPubKey),         // During session
    AutoActivated(String),              // Based on safety plan conditions
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum CrisisType {
    SuicidalIdeation,
    SelfHarm,
    PanicAttack,
    DissociativeEpisode,
    PsychoticEpisode,
    SubstanceCrisis,
    DomesticViolence,
    SexualAssault,
    HomelessnessRisk,
    MedicalEmergency,
    Other(String),
}

/// Collective healing space
#[hdk_entry_helper]
pub struct CollectiveHealingSpace {
    pub space_id: String,
    pub title: String,
    pub description: String,

    // Context
    pub healing_focus: HealingFocus,
    pub related_event: Option<String>,     // Community trauma, loss, etc.

    // Structure
    pub format: HealingFormat,
    pub facilitators: Vec<AgentPubKey>,
    pub duration: Option<Duration>,

    // Participation
    pub participants: Vec<HealingParticipant>,
    pub witnesses: Vec<AgentPubKey>,       // Present but not sharing
    pub max_participants: Option<u32>,

    // Content (encrypted, ephemeral options)
    pub shared_expressions: Vec<HealingExpression>,
    pub ephemeral: bool,                   // Delete after closing?

    // Status
    pub status: HealingSpaceStatus,
    pub opened_at: Timestamp,
    pub closed_at: Option<Timestamp>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum HealingFocus {
    CommunityLoss,           // Death of community member
    CollectiveTrauma,        // Shared traumatic event
    Transition,              // Major community change
    Conflict,                // After community conflict resolution
    Celebration,             // Collective joy processing
    Seasonal,                // Solstice, new year, etc.
    Ongoing,                 // Regular community care
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum HealingFormat {
    SharingCircle,           // Go-around sharing
    OpenSpace,               // Unstructured
    Ritual,                  // Structured ceremony
    ArtBased,                // Creative expression
    Movement,                // Somatic/body-based
    Silent,                  // Meditation/presence
    Hybrid,                  // Multiple formats
}
```

---

## Key Workflows

### Workflow 1: Daily Wellbeing Check-in

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    DAILY CHECK-IN FLOW                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Agent opens Sanctuary                                                  │
│       │                                                                 │
│       ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  "How are you today?"                                           │   │
│  │                                                                  │   │
│  │  Mood:    😢 😕 😐 🙂 😊                                         │   │
│  │  Energy:  ○ ○ ○ ○ ○ ○ ○ ○ ○ ○                                    │   │
│  │  Sleep:   Poor ──────────── Great                               │   │
│  │                                                                  │   │
│  │  [Skip detailed] [Add more context]                             │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│       │                                                                 │
│       ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  LOCAL PROCESSING (never leaves device)                         │   │
│  │                                                                  │   │
│  │  • Store in agent's source chain (encrypted)                    │   │
│  │  • Check against safety plan thresholds                         │   │
│  │  • Update personal trend analysis                               │   │
│  │                                                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│       │                                                                 │
│       ├─── If threshold triggered ───────────────────────────┐         │
│       │                                                       ▼         │
│       │                                          ┌─────────────────┐   │
│       │                                          │ "I notice you've│   │
│       │                                          │ been struggling.│   │
│       │                                          │ Would you like  │   │
│       │                                          │ to:"            │   │
│       │                                          │                 │   │
│       │                                          │ • Talk to circle│   │
│       │                                          │ • Find support  │   │
│       │                                          │ • Review safety │   │
│       │                                          │   plan          │   │
│       │                                          │ • Just note it  │   │
│       │                                          └─────────────────┘   │
│       │                                                                 │
│       ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  OPTIONAL: Share with trusted circle                            │   │
│  │                                                                  │   │
│  │  Agent can choose to share summary with:                        │   │
│  │  • Specific trusted contacts                                    │   │
│  │  • Whole circle                                                 │   │
│  │  • No one (default)                                             │   │
│  │                                                                  │   │
│  │  Sharing is always explicit, never automatic                    │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Workflow 2: Requesting Peer Support

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    SUPPORT REQUEST FLOW                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Agent: "I need to talk to someone"                                    │
│       │                                                                 │
│       ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  SUPPORT REQUEST FORM                                           │   │
│  │                                                                  │   │
│  │  What kind of support? ○ Just listen  ○ Help think through     │   │
│  │                        ○ Crisis       ○ Check-in               │   │
│  │                                                                  │   │
│  │  How soon?  ○ Flexible  ○ Today  ○ Now  ⚠️ CRISIS              │   │
│  │                                                                  │   │
│  │  Topic (optional):  [ Anxiety ▼ ]                               │   │
│  │                                                                  │   │
│  │  Preferences:                                                   │   │
│  │  Language: [ English ▼ ]                                        │   │
│  │  Other: [ Any preferences... ]                                  │   │
│  │                                                                  │   │
│  │  Brief context (optional, helps matching):                      │   │
│  │  [ _________________________________ ]                          │   │
│  │                                                                  │   │
│  │  [Find Support]                                                 │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│       │                                                                 │
│       ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  MATCHING ALGORITHM                                             │   │
│  │                                                                  │   │
│  │  Consider:                                                      │   │
│  │  • Supporter availability                                       │   │
│  │  • Specialization match                                         │   │
│  │  • Language/cultural match                                      │   │
│  │  • MATL trust scores (both directions)                         │   │
│  │  • Previous interaction history                                 │   │
│  │  • Supporter current load                                       │   │
│  │  • Geographic proximity (if crisis)                             │   │
│  │                                                                  │   │
│  │  Present top 3 matches (or auto-match if urgent)               │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│       │                                                                 │
│       ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  SUPPORTER NOTIFICATION                                         │   │
│  │                                                                  │   │
│  │  "Someone needs support"                                        │   │
│  │  Type: Listening                                                │   │
│  │  Urgency: Today                                                 │   │
│  │  Topic: Anxiety                                                 │   │
│  │                                                                  │   │
│  │  [Accept] [Decline] [Suggest Other]                            │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│       │                                                                 │
│       ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  SESSION BEGINS                                                 │   │
│  │                                                                  │   │
│  │  • End-to-end encrypted channel                                 │   │
│  │  • Timer (supporter fatigue prevention)                         │   │
│  │  • Safety resources accessible                                  │   │
│  │  • Session notes (private to each party)                        │   │
│  │  • Escalation button if needed                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│       │                                                                 │
│       ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  SESSION CLOSE                                                  │   │
│  │                                                                  │   │
│  │  Both parties:                                                  │   │
│  │  • Optional feedback (private)                                  │   │
│  │  • Follow-up scheduling                                         │   │
│  │  • Resource sharing                                             │   │
│  │  • Mutual appreciation                                          │   │
│  │                                                                  │   │
│  │  Supporter:                                                     │   │
│  │  • Self-care check                                              │   │
│  │  • Supervision flag if needed                                   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Workflow 3: Crisis Response

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    CRISIS RESPONSE FLOW                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  TRIGGER: Crisis activated (self, circle, or auto)                     │
│       │                                                                 │
│       ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  IMMEDIATE SAFETY CHECK                                         │   │
│  │                                                                  │   │
│  │  "Are you safe right now?"                                      │   │
│  │                                                                  │   │
│  │  [Yes, I'm safe] [No, I need help] [Not sure]                  │   │
│  │                                                                  │   │
│  │  If no response in 5 minutes → escalate                        │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│       │                                                                 │
│       ├─── If "No" or no response ─────────────────────────┐           │
│       │                                                     ▼           │
│       │                                        ┌─────────────────────┐ │
│       │                                        │ PARALLEL ACTIVATION │ │
│       │                                        │                     │ │
│       │                                        │ 1. Alert trusted    │ │
│       │                                        │    circle (priority │ │
│       │                                        │    order)           │ │
│       │                                        │                     │ │
│       │                                        │ 2. Notify available │ │
│       │                                        │    crisis supporters│ │
│       │                                        │                     │ │
│       │                                        │ 3. Surface safety   │ │
│       │                                        │    plan             │ │
│       │                                        │                     │ │
│       │                                        │ 4. If consented:    │ │
│       │                                        │    share location   │ │
│       │                                        │                     │ │
│       │                                        │ 5. Queue professional│ │
│       │                                        │    callback         │ │
│       │                                        └─────────────────────┘ │
│       │                                                     │           │
│       ▼                                                     ▼           │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  GROUNDING SUPPORT (while help mobilizes)                       │   │
│  │                                                                  │   │
│  │  • Breathing exercise (visual/audio)                            │   │
│  │  • Grounding prompts (5-4-3-2-1)                               │   │
│  │  • Safety plan coping strategies                                │   │
│  │  • "Someone is on their way"                                    │   │
│  │  • Direct line to crisis responder                              │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│       │                                                                 │
│       ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  RESPONDER ARRIVES (peer or professional)                       │   │
│  │                                                                  │   │
│  │  • Takes over direct support                                    │   │
│  │  • Documents intervention (minimal, private)                    │   │
│  │  • Assesses need for emergency services                         │   │
│  │  • Coordinates with other responders                            │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│       │                                                                 │
│       ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  STABILIZATION                                                  │   │
│  │                                                                  │   │
│  │  • Crisis deescalated                                           │   │
│  │  • Safety confirmed                                             │   │
│  │  • Follow-up plan created                                       │   │
│  │  • Circle notified of resolution                                │   │
│  │  • Responders offered debrief                                   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│       │                                                                 │
│       ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  FOLLOW-UP (24h, 72h, 1 week)                                   │   │
│  │                                                                  │   │
│  │  • Check-in prompts                                             │   │
│  │  • Safety plan review                                           │   │
│  │  • Resource connections                                         │   │
│  │  • Circle activation for ongoing support                        │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Workflow 4: Community Healing Space

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    COLLECTIVE HEALING FLOW                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  TRIGGER: Community experiences shared loss/trauma                     │
│       │                                                                 │
│       ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  HEALING SPACE PROPOSAL (via Agora or facilitator)              │   │
│  │                                                                  │   │
│  │  "Our community lost [member/experienced event].                │   │
│  │   Shall we create space to grieve/process together?"            │   │
│  │                                                                  │   │
│  │  Format: Sharing Circle                                         │   │
│  │  When: [Date/Time options]                                      │   │
│  │  Facilitated by: [Trained facilitator]                          │   │
│  │  Confidential: Yes                                              │   │
│  │  Ephemeral: Yes (content deleted after)                         │   │
│  │                                                                  │   │
│  │  [Join as Participant] [Join as Witness] [Send Support]         │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│       │                                                                 │
│       ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  SPACE OPENS                                                    │   │
│  │                                                                  │   │
│  │  Facilitator:                                                   │   │
│  │  • Welcomes participants                                        │   │
│  │  • Reads community agreements                                   │   │
│  │  • Opens the container                                          │   │
│  │                                                                  │   │
│  │  Agreements:                                                    │   │
│  │  • Speak from "I"                                               │   │
│  │  • Listen without fixing                                        │   │
│  │  • Confidentiality                                              │   │
│  │  • Right to pass                                                │   │
│  │  • No cross-talk during shares                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│       │                                                                 │
│       ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  SHARING ROUNDS                                                 │   │
│  │                                                                  │   │
│  │  Participants share (text, voice, image):                       │   │
│  │  • Memories                                                     │   │
│  │  • Feelings                                                     │   │
│  │  • Gratitude                                                    │   │
│  │  • Grief                                                        │   │
│  │                                                                  │   │
│  │  Witnesses hold space silently                                  │   │
│  │                                                                  │   │
│  │  Facilitator tends the container                                │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│       │                                                                 │
│       ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  CLOSING                                                        │   │
│  │                                                                  │   │
│  │  • Facilitator closes container                                 │   │
│  │  • Brief grounding exercise                                     │   │
│  │  • Optional: collective intention/dedication                    │   │
│  │  • Resources for ongoing support                                │   │
│  │  • Ephemeral content deleted (if selected)                      │   │
│  │  • Chronicle records: "Healing space held" (no content)         │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Integration Points

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    SANCTUARY INTEGRATIONS                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  IDENTITY & TRUST                                                       │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Attest ──► Peer supporter credential verification              │   │
│  │  MATL ────► Trust scoring for supporter matching                │   │
│  │  Weave ───► Relationship context for circle formation           │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  HEALTH                                                                 │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  HealthVault ──► Mental health records integration              │   │
│  │                   (agent-controlled sharing with providers)      │   │
│  │               ──► Medication tracking correlation               │   │
│  │               ──► Professional referral coordination            │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  EMERGENCY                                                              │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Beacon ──► Crisis escalation to emergency services             │   │
│  │         ──► Location sharing in crisis                          │   │
│  │         ──► Community-wide mental health emergencies            │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  LEARNING                                                               │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Praxis ──► Peer support training credentials                   │   │
│  │         ──► Continuing education for supporters                 │   │
│  │  Guild ───► Professional supervision networks                   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  COMMUNITY                                                              │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Agora ──► Healing space governance                             │   │
│  │        ──► Community mental health policy                       │   │
│  │  Nexus ──► Support circle scheduling                            │   │
│  │  Loom ───► Narrative healing, story archiving                   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ECONOMIC                                                               │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Treasury ──► Fund peer support training                        │   │
│  │           ──► Crisis response resources                         │   │
│  │  Mutual ────► Mental health mutual aid coverage                 │   │
│  │  Collab ────► Workplace wellbeing integration                   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Privacy Architecture

### Data Classification

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    SANCTUARY PRIVACY MODEL                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  LEVEL 5: ULTRA-PRIVATE (Agent Only)                                   │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  • Wellbeing check-ins                                          │   │
│  │  • Journal entries                                              │   │
│  │  • Session notes                                                │   │
│  │  • Crisis history                                               │   │
│  │                                                                  │   │
│  │  Storage: Encrypted in agent's source chain                     │   │
│  │  Access: Agent only, ever                                       │   │
│  │  Backup: Agent-controlled, encrypted                            │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  LEVEL 4: TRUSTED CIRCLE                                               │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  • Safety plan (if shared)                                      │   │
│  │  • Wellbeing trends (if shared)                                 │   │
│  │  • Crisis alerts (if configured)                                │   │
│  │                                                                  │   │
│  │  Storage: Encrypted, shared keys with circle                    │   │
│  │  Access: Agent + explicitly trusted contacts                    │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  LEVEL 3: SESSION CONFIDENTIAL                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  • Support session content                                      │   │
│  │  • Circle sharing content                                       │   │
│  │  • Healing space expressions                                    │   │
│  │                                                                  │   │
│  │  Storage: End-to-end encrypted                                  │   │
│  │  Access: Session participants only                              │   │
│  │  Retention: Ephemeral option available                          │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  LEVEL 2: COMMUNITY VISIBLE                                            │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  • Supporter profiles (public parts)                            │   │
│  │  • Circle descriptions                                          │   │
│  │  • Resource directory listings                                  │   │
│  │                                                                  │   │
│  │  Storage: DHT                                                   │   │
│  │  Access: Community members                                      │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  LEVEL 1: AGGREGATE ONLY                                               │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  • Community wellbeing metrics (differential privacy)           │   │
│  │  • Support utilization stats (anonymized)                       │   │
│  │  • Resource effectiveness (aggregated)                          │   │
│  │                                                                  │   │
│  │  Storage: Aggregated, no individual data                        │   │
│  │  Access: Community/research (never individual)                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  CRITICAL PRIVACY RULES                                                │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  • No wellbeing data in MATL (no trust penalty for struggling)  │   │
│  │  • No employer/landlord access ever                             │   │
│  │  • Crisis data never used against agent                         │   │
│  │  • Right to delete all data at any time                         │   │
│  │  • Professional integration only with explicit consent          │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Safety Architecture

### Anti-Abuse Protections

```rust
/// Protections against misuse of support systems

pub struct SafetyProtections {
    // Supporter protections
    pub supporter_fatigue_limits: FatigueLimits,
    pub mandatory_supervision_threshold: u32,
    pub burnout_detection: BurnoutDetection,

    // Supportee protections
    pub grooming_detection: GroomingDetection,
    pub boundary_violation_reporting: ReportingSystem,
    pub mandatory_reporter_protocols: MandatoryReporting,

    // System protections
    pub crisis_triage: CrisisTriage,
    pub professional_escalation: EscalationProtocol,
    pub abuse_investigation: InvestigationProcess,
}

pub struct FatigueLimits {
    pub max_sessions_per_day: u8,        // Default: 4
    pub max_sessions_per_week: u8,       // Default: 15
    pub mandatory_break_after_crisis: Duration,  // Default: 24h
    pub self_care_check_frequency: Duration,     // Default: weekly
}

pub struct GroomingDetection {
    pub patterns_monitored: Vec<GroomingPattern>,
    pub alert_threshold: u8,
    pub auto_escalation: bool,
}

pub enum GroomingPattern {
    ExcessivePrivateContact,
    BoundaryTestingEscalation,
    IsolationAttempts,
    SecrecyRequests,
    RoleConfusion,
}
```

### Mandatory Reporting Integration

```rust
/// For jurisdictions with mandatory reporting requirements

pub struct MandatoryReportingConfig {
    pub jurisdiction: String,
    pub reporting_triggers: Vec<ReportingTrigger>,
    pub reporting_authority: String,
    pub supporter_obligations: Vec<String>,
    pub agent_notification: AgentNotification,
}

pub enum ReportingTrigger {
    ChildAbuse,
    ElderAbuse,
    DependentAdultAbuse,
    ImminentHarmToSelf,
    ImminentHarmToOthers,
    // Configured per jurisdiction
}

pub enum AgentNotification {
    BeforeReport,      // Inform before reporting
    ConcurrentReport,  // Inform at time of report
    AfterReport,       // Inform after (safety situations)
    NoNotification,    // Legal requirement (rare)
}
```

---

## Peer Supporter Training Framework

### Training Pathway

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    PEER SUPPORTER PATHWAY                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  LEVEL 1: COMMUNITY SUPPORTER                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Training: 20 hours                                             │   │
│  │  • Active listening                                             │   │
│  │  • Boundaries and self-care                                     │   │
│  │  • Crisis recognition                                           │   │
│  │  • Resource navigation                                          │   │
│  │  • Platform usage                                               │   │
│  │                                                                  │   │
│  │  Scope: Check-ins, general listening, resource connection       │   │
│  │  Supervision: Monthly group                                     │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                               │                                         │
│                               ▼                                         │
│  LEVEL 2: SPECIALIZED SUPPORTER                                        │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Training: 40 additional hours                                  │   │
│  │  • Specialization training (grief, anxiety, etc.)               │   │
│  │  • Trauma-informed approach                                     │   │
│  │  • Advanced communication                                       │   │
│  │  • Supporting diverse populations                               │   │
│  │                                                                  │   │
│  │  Scope: Specialized support, longer-term relationships          │   │
│  │  Supervision: Bi-weekly individual                              │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                               │                                         │
│                               ▼                                         │
│  LEVEL 3: CRISIS RESPONDER                                             │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Training: 60 additional hours                                  │   │
│  │  • Crisis intervention                                          │   │
│  │  • Safety planning                                              │   │
│  │  • De-escalation                                                │   │
│  │  • Professional coordination                                    │   │
│  │  • Mandatory reporting                                          │   │
│  │                                                                  │   │
│  │  Scope: Crisis response, safety planning, escalation            │   │
│  │  Supervision: Weekly individual                                 │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                               │                                         │
│                               ▼                                         │
│  LEVEL 4: FACILITATOR                                                  │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Training: 40 additional hours                                  │   │
│  │  • Group facilitation                                           │   │
│  │  • Collective healing practices                                 │   │
│  │  • Conflict in groups                                           │   │
│  │  • Ritual and ceremony                                          │   │
│  │                                                                  │   │
│  │  Scope: Circle facilitation, healing spaces, training others    │   │
│  │  Supervision: Peer supervision circle                           │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  All credentials issued via Praxis, tracked in Attest                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Implementation Priority

### Phase 1: Individual Foundation
1. Wellbeing tracking (local only)
2. Safety plan creation
3. Trusted circle formation
4. Basic peer matching

### Phase 2: Support Network
5. Peer supporter training integration
6. Support request/offer matching
7. Session management
8. Feedback system

### Phase 3: Community Layer
9. Support circles
10. Community healing spaces
11. Resource directory
12. Aggregate metrics

### Phase 4: Crisis Infrastructure
13. Crisis protocol
14. Rapid response coordination
15. Professional escalation
16. Beacon integration

---

## Conclusion

Sanctuary recognizes that human flourishing requires emotional infrastructure as much as economic infrastructure. By building mutual support into the foundation of the civilizational OS, we create the conditions for genuine human thriving.

*"We cannot build a new world if we cannot hold each other through the night."*

---

*Document Version: 1.0*
*Classification: Tier 2 - Essential (Foundational for human wellbeing)*
*Dependencies: Attest, MATL, HealthVault, Praxis, Beacon, Nexus, Agora*

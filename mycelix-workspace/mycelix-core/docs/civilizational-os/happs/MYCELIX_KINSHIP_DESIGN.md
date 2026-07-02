# Mycelix-Kinship: Family, Care & Intergenerational Coordination

## Vision Statement

*"Care is the invisible infrastructure of civilization. Kinship makes visible and coordinates the essential work of raising children, supporting elders, and caring for each other across the lifespan."*

---

## Executive Summary

Mycelix-Kinship provides infrastructure for care coordination:

1. **Family formation** - Diverse family structures, chosen family recognition
2. **Childcare coordination** - Cooperatives, nanny shares, backup care networks
3. **Elder care** - Aging in place support, care coordination, memory preservation
4. **Mutual aid care** - Time banking, care exchanges, crisis coverage
5. **Intergenerational connection** - Mentorship, skill transfer, story preservation

---

## Core Philosophy

### Care as Infrastructure

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    CARE RECOGNITION                                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  UNPAID CARE (traditionally invisible)                                 │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  • Childcare by family members                                  │   │
│  │  • Elder care by family members                                 │   │
│  │  • Household management                                         │   │
│  │  • Emotional labor                                              │   │
│  │  • Community care coordination                                  │   │
│  │                                                                  │   │
│  │  Kinship makes this visible, valued, and coordinated           │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  FAMILY DIVERSITY                                                       │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  • Nuclear families                                             │   │
│  │  • Extended families                                            │   │
│  │  • Chosen families                                              │   │
│  │  • Single-parent families                                       │   │
│  │  • Multi-generational households                                │   │
│  │  • Communal parenting                                           │   │
│  │  • Blended families                                             │   │
│  │  • Caregiving networks                                          │   │
│  │                                                                  │   │
│  │  All structures supported equally                               │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Architecture Overview

### Key Data Types

```rust
/// Family unit definition (flexible)
pub struct FamilyUnit {
    pub family_id: String,
    pub name: Option<String>,
    pub family_type: FamilyType,
    pub members: Vec<FamilyMember>,
    pub relationships: Vec<FamilyRelationship>,
    pub shared_resources: Vec<SharedResource>,
    pub care_agreements: Vec<ActionHash>,
    pub governance: Option<ActionHash>,     // For larger units
}

pub enum FamilyType {
    Nuclear,
    Extended,
    Chosen,
    SingleParent,
    MultiGenerational,
    Communal,
    Blended,
    CaregivingNetwork,
    Custom(String),
}

pub struct FamilyMember {
    pub agent: Option<AgentPubKey>,         // If in system
    pub name: String,
    pub role: Vec<FamilyRole>,
    pub care_needs: Vec<CareNeed>,
    pub care_capacity: Vec<CareCapacity>,
    pub age_group: AgeGroup,
}

pub enum FamilyRole {
    Parent,
    Child,
    Grandparent,
    Sibling,
    Partner,
    Guardian,
    ChosenFamily,
    Caregiver,
    Dependent,
}

/// Childcare arrangement
pub struct ChildcareArrangement {
    pub arrangement_id: String,
    pub arrangement_type: ChildcareType,
    pub children: Vec<AgentPubKey>,
    pub caregivers: Vec<Caregiver>,
    pub schedule: CareSchedule,
    pub location: Option<ActionHash>,
    pub cost_sharing: CostSharing,
    pub agreements: Vec<ActionHash>,        // Covenant contracts
    pub background_checks: Vec<ActionHash>, // Attest credentials
}

pub enum ChildcareType {
    FamilyCare,
    NannyShare { families: Vec<ActionHash> },
    Cooperative { members: Vec<ActionHash> },
    CommunityChildcare,
    BackupCare,
    DropIn,
    AfterSchool,
    SummerCamp,
}

/// Elder care coordination
pub struct ElderCareProfile {
    pub profile_id: String,
    pub elder: AgentPubKey,
    pub care_needs: Vec<ElderCareNeed>,
    pub care_preferences: CarePreferences,
    pub care_team: Vec<CareTeamMember>,
    pub medical_integration: Option<ActionHash>,  // HealthVault
    pub housing_situation: HousingSituation,
    pub financial_plan: Option<ActionHash>,
    pub advance_directives: Option<ActionHash>,
    pub memory_archive: Option<ActionHash>,       // Chronicle
}

pub enum ElderCareNeed {
    CompanionShip,
    MealPreparation,
    Transportation,
    HouseholdHelp,
    PersonalCare,
    MedicationManagement,
    MedicalAppointments,
    FinancialManagement,
    MemorySupport,
    Respite,
    EndOfLife,
}

pub enum HousingSituation {
    Independent,
    WithFamily,
    Cohousing,
    AssisstedLiving,
    NursingCare,
    VillageModel,                          // Aging in place with support
}

/// Care time bank
pub struct CareTimeBank {
    pub bank_id: String,
    pub name: String,
    pub members: Vec<TimeBankMember>,
    pub care_categories: Vec<CareCategory>,
    pub exchange_rate: TimeExchangeRate,
    pub governance: ActionHash,
}

pub struct TimeBankMember {
    pub agent: AgentPubKey,
    pub balance: Duration,                  // Hours credited
    pub services_offered: Vec<CareService>,
    pub services_needed: Vec<CareService>,
    pub availability: AvailabilitySchedule,
    pub verified_skills: Vec<ActionHash>,
}

pub struct CareService {
    pub service_type: CareServiceType,
    pub description: String,
    pub time_estimate: Duration,
    pub location_type: LocationType,
}

pub enum CareServiceType {
    // Childcare
    Babysitting,
    SchoolPickup,
    HomeworkHelp,
    PlayDate,

    // Elder care
    CompanionVisit,
    MealDelivery,
    TransportationAssist,
    TechHelp,

    // Household
    MealPrep,
    Housecleaning,
    Errands,
    PetCare,

    // Respite
    RespiteCare,
    OvernightCare,
    EmergencyCoverage,

    // Skill sharing
    Tutoring,
    MusicLessons,
    LanguagePractice,
    Mentoring,
}

/// Intergenerational connection
pub struct IntergenerationalProgram {
    pub program_id: String,
    pub name: String,
    pub program_type: IntergenerationalType,
    pub elders: Vec<AgentPubKey>,
    pub youth: Vec<AgentPubKey>,
    pub facilitator: Option<AgentPubKey>,
    pub activities: Vec<Activity>,
    pub schedule: Schedule,
    pub outcomes: Vec<String>,
}

pub enum IntergenerationalType {
    StorySharing,
    SkillTransfer,
    MentorMatch,
    SharedHousing,
    CommunityProject,
    HistoryPreservation,
    ArtAndCraft,
    GardenPartners,
}
```

---

## Key Workflows

### Workflow 1: Childcare Cooperative Formation

```
Parents identify shared need →
Form cooperative via Agora governance →
Define care schedule and responsibilities →
Execute agreements via Covenant →
Coordinate ongoing schedule →
Track care hours in time bank →
Resolve issues via Arbiter if needed
```

### Workflow 2: Elder Care Network

```
Elder or family member creates care profile →
Care needs assessed →
Care team assembled (family + community) →
Schedule coordinated across caregivers →
HealthVault integration for medical coordination →
Time bank credits for non-family caregivers →
Respite care arranged when needed →
Memory preservation to Chronicle
```

### Workflow 3: Emergency Care Coverage

```
Parent has emergency, needs childcare →
Beacon activates care network →
Available caregivers identified by proximity + trust →
Care arranged within hours →
Time bank credits exchanged →
Follow-up care scheduled if needed
```

### Workflow 4: Intergenerational Matching

```
Elder offers skill/knowledge to share →
Youth/learner registers interest →
Match made based on compatibility →
First meeting facilitated →
Ongoing relationship supported →
Stories and skills preserved in Chronicle
```

---

## Integration Points

| Integration | Purpose |
|-------------|---------|
| **Attest** | Background checks, credentials for caregivers |
| **MATL** | Trust scoring for care matching |
| **Covenant** | Care agreements, nanny contracts |
| **Treasury** | Cost sharing, care payments |
| **HealthVault** | Medical care coordination |
| **Sanctuary** | Caregiver mental health support |
| **Beacon** | Emergency care coordination |
| **Terroir** | Housing for multigenerational living |
| **Transit** | Care-related transportation |
| **Chronicle** | Memory preservation, family history |
| **Legacy** | Succession planning, end-of-life |
| **Praxis** | Caregiver training, parenting education |
| **Nexus** | Family events, community gatherings |

---

## Privacy & Safety Architecture

### Child Safety

```rust
pub struct ChildSafetyProtocol {
    // Background verification
    pub background_check_required: bool,
    pub credential_verification: Vec<CredentialType>,
    pub reference_check_minimum: u8,

    // Supervision
    pub adult_child_ratio: (u8, u8),
    pub never_alone_policy: bool,
    pub check_in_frequency: Duration,

    // Reporting
    pub incident_reporting: IncidentProtocol,
    pub mandatory_reporting_compliance: bool,

    // Transparency
    pub parent_access_level: AccessLevel,
    pub audit_trail: bool,
}
```

### Elder Safety

```rust
pub struct ElderSafetyProtocol {
    // Financial protection
    pub financial_monitoring: bool,
    pub large_transaction_alerts: Option<Decimal>,
    pub authorized_financial_agents: Vec<AgentPubKey>,

    // Health monitoring
    pub check_in_schedule: Schedule,
    pub emergency_contacts: Vec<Contact>,
    pub medical_alert_integration: bool,

    // Abuse prevention
    pub isolation_alerts: bool,
    pub caregiver_rotation: bool,
    pub third_party_oversight: Option<AgentPubKey>,
}
```

---

## Care Economics

### Time Banking

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    CARE TIME BANKING                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  PRINCIPLES:                                                            │
│  • One hour = one hour (all care valued equally)                       │
│  • Everyone has something to give                                       │
│  • Everyone has needs                                                   │
│  • Community is the currency                                            │
│                                                                         │
│  EXAMPLE EXCHANGES:                                                     │
│  • 2 hours babysitting → 2 hours in time bank                         │
│  • 2 hours elder companionship → 2 hours from time bank               │
│  • 1 hour tutoring → 1 hour meal delivery                             │
│                                                                         │
│  FLEXIBILITY:                                                           │
│  • Earn credits when able, use when needed                             │
│  • Transfer credits within family                                       │
│  • "Pay forward" for those who can't reciprocate                       │
│  • Community reserve for emergencies                                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Conclusion

Kinship makes visible the essential work of care that holds society together. By coordinating across diverse family structures and integrating with housing, health, economics, and community systems, it enables humans to care for each other across the entire lifespan.

*"It takes a village to raise a child, and a village to support our elders. Kinship is the infrastructure of that village."*

---

*Document Version: 1.0*
*Classification: Tier 2 - Essential*
*Dependencies: Attest, MATL, Covenant, Treasury, HealthVault, Sanctuary, Beacon, Chronicle, Legacy*

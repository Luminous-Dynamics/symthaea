# Mycelix-Terroir: Land, Property & Housing Coordination

## Vision Statement

*"Land is not a commodity - it is the foundation upon which all human activity rests. Terroir provides infrastructure for recording, coordinating, and governing our relationship with place - from individual dwellings to bioregional commons."*

---

## Executive Summary

Mycelix-Terroir manages the civilizational relationship with land and property:

1. **Property registry** - Transparent, verifiable records of land/property relationships
2. **Housing coordination** - Matching housing needs with available spaces
3. **Land governance** - Community land trusts, commons management, bioregional coordination
4. **Tenure diversity** - Supporting ownership, rental, cooperative, and commons models
5. **Place-based community** - Geographic community formation and coordination

Terroir recognizes multiple valid relationships with land beyond mere ownership.

---

## Core Philosophy

### Beyond Ownership

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    TENURE DIVERSITY                                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  INDIVIDUAL TENURE                                                      │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  • Fee Simple Ownership (traditional)                           │   │
│  │  • Leasehold (long-term rental rights)                         │   │
│  │  • Life Estate (use rights for lifetime)                       │   │
│  │  • Usufruct (use rights without ownership)                     │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  COLLECTIVE TENURE                                                      │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  • Housing Cooperative (member-owned)                           │   │
│  │  • Community Land Trust (land held in trust)                   │   │
│  │  • Limited Equity Housing (affordability-preserved)            │   │
│  │  • Cohousing (private units + shared spaces)                   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  COMMONS TENURE                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  • Community Commons (local shared resources)                   │   │
│  │  • Bioregional Commons (watershed, forest, etc.)               │   │
│  │  • Public Trust (held for public benefit)                      │   │
│  │  • Indigenous Stewardship (traditional governance)             │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  TEMPORARY TENURE                                                       │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  • Short-term Rental                                            │   │
│  │  • House Sitting                                                │   │
│  │  • Work Exchange (housing for labor)                           │   │
│  │  • Hospitality Exchange (reciprocal hosting)                   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Architecture Overview

### System Layers

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    TERROIR ARCHITECTURE                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    REGISTRY LAYER                                │   │
│  │  Property records, boundaries, improvements, encumbrances       │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                               │                                         │
│                               ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    TENURE LAYER                                  │   │
│  │  Ownership, leases, cooperative memberships, use rights         │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                               │                                         │
│                               ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    COORDINATION LAYER                            │   │
│  │  Housing matching, availability, need tracking                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                               │                                         │
│                               ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    GOVERNANCE LAYER                              │   │
│  │  Land trusts, commons management, bioregional coordination      │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                               │                                         │
│                               ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    PLACE LAYER                                   │   │
│  │  Geographic communities, neighborhood coordination              │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Zome Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         TERROIR ZOMES                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  INTEGRITY ZOMES                                                        │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  terroir_types                                                  │   │
│  │  ├── Property            (land parcel or structure)            │   │
│  │  ├── PropertyBoundary    (geographic boundary definition)      │   │
│  │  ├── Improvement         (buildings, infrastructure)           │   │
│  │  ├── TenureRecord        (ownership, lease, rights)            │   │
│  │  ├── Encumbrance         (liens, easements, restrictions)      │   │
│  │  ├── HousingListing      (available housing)                   │   │
│  │  ├── HousingNeed         (housing seeker profile)              │   │
│  │  ├── LandTrust           (community land trust entity)         │   │
│  │  ├── CommonPool          (shared resource definition)          │   │
│  │  ├── PlaceCommunity      (geographic community)                │   │
│  │  └── Bioregion           (ecological region definition)        │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  COORDINATOR ZOMES                                                      │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                                                                  │   │
│  │  property_registry                                              │   │
│  │  ├── register_property()                                       │   │
│  │  ├── update_property()                                         │   │
│  │  ├── define_boundary()                                         │   │
│  │  ├── record_improvement()                                      │   │
│  │  ├── query_by_location()                                       │   │
│  │  ├── verify_against_official()   // Oracle integration         │   │
│  │  └── generate_property_report()                                │   │
│  │                                                                  │   │
│  │  tenure_management                                              │   │
│  │  ├── record_ownership()                                        │   │
│  │  ├── transfer_ownership()        // With Covenant contract     │   │
│  │  ├── create_lease()                                            │   │
│  │  ├── record_cooperative_share()                                │   │
│  │  ├── grant_use_rights()                                        │   │
│  │  ├── record_encumbrance()                                      │   │
│  │  ├── release_encumbrance()                                     │   │
│  │  └── verify_tenure()                                           │   │
│  │                                                                  │   │
│  │  housing_coordination                                           │   │
│  │  ├── list_housing()                                            │   │
│  │  ├── search_housing()                                          │   │
│  │  ├── register_housing_need()                                   │   │
│  │  ├── match_housing()                                           │   │
│  │  ├── apply_for_housing()                                       │   │
│  │  ├── verify_occupancy()                                        │   │
│  │  └── report_housing_issue()                                    │   │
│  │                                                                  │   │
│  │  land_trust_management                                          │   │
│  │  ├── create_land_trust()                                       │   │
│  │  ├── add_property_to_trust()                                   │   │
│  │  ├── issue_ground_lease()                                      │   │
│  │  ├── manage_affordability()                                    │   │
│  │  ├── govern_trust()              // Agora integration          │   │
│  │  └── calculate_land_value()      // For Georgist models        │   │
│  │                                                                  │   │
│  │  commons_coordination                                           │   │
│  │  ├── define_common_pool()                                      │   │
│  │  ├── set_access_rules()                                        │   │
│  │  ├── track_usage()                                             │   │
│  │  ├── coordinate_stewardship()                                  │   │
│  │  └── integrate_with_commons_happ()                             │   │
│  │                                                                  │   │
│  │  place_community                                                │   │
│  │  ├── form_place_community()                                    │   │
│  │  ├── join_place_community()                                    │   │
│  │  ├── coordinate_neighbors()                                    │   │
│  │  ├── manage_shared_resources()                                 │   │
│  │  └── integrate_with_agora()                                    │   │
│  │                                                                  │   │
│  │  bioregion_coordination                                         │   │
│  │  ├── define_bioregion()                                        │   │
│  │  ├── coordinate_watershed()                                    │   │
│  │  ├── track_ecological_health()                                 │   │
│  │  └── federate_communities()      // Diplomat integration       │   │
│  │                                                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Data Model

### Core Entry Types

```rust
/// Property record (land or structure)
#[hdk_entry_helper]
pub struct Property {
    pub property_id: String,

    // Location
    pub location: PropertyLocation,
    pub boundary: Option<ActionHash>,      // PropertyBoundary
    pub parent_property: Option<ActionHash>, // If subdivided

    // Classification
    pub property_type: PropertyType,
    pub land_use: LandUse,
    pub zoning: Option<String>,

    // Physical characteristics
    pub area: Option<Area>,
    pub improvements: Vec<ActionHash>,     // Improvement records

    // Official records (for bridging)
    pub official_parcel_id: Option<String>,
    pub official_registry: Option<OfficialRegistry>,

    // Tenure
    pub current_tenure: Vec<ActionHash>,   // Active TenureRecords
    pub encumbrances: Vec<ActionHash>,

    // Governance
    pub governed_by: Option<GovernanceReference>,

    // Metadata
    pub created_at: Timestamp,
    pub last_updated: Timestamp,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PropertyLocation {
    pub coordinates: Option<GeoCoordinates>,
    pub address: Option<Address>,
    pub place_names: Vec<String>,          // Traditional/local names
    pub bioregion: Option<ActionHash>,
    pub watershed: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum PropertyType {
    Land,
    ResidentialBuilding,
    CommercialBuilding,
    IndustrialBuilding,
    AgriculturalLand,
    Forest,
    Wetland,
    Water,
    Mixed,
    Unit { parent: ActionHash },           // Apartment, condo unit
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum LandUse {
    Residential { density: Density },
    Commercial,
    Industrial,
    Agricultural { type_: AgricultureType },
    Conservation,
    Recreation,
    Mixed(Vec<LandUse>),
    Undeveloped,
    Sacred,                                // Ceremonial/spiritual sites
}

/// Tenure record (relationship between agent and property)
#[hdk_entry_helper]
pub struct TenureRecord {
    pub tenure_id: String,
    pub property_hash: ActionHash,
    pub holder: TenureHolder,

    // Tenure type
    pub tenure_type: TenureType,

    // Duration
    pub start_date: Timestamp,
    pub end_date: Option<Timestamp>,       // None = indefinite

    // Documentation
    pub contract_hash: Option<ActionHash>, // Covenant contract
    pub official_record: Option<OfficialRecord>,

    // Financial
    pub acquisition_cost: Option<Decimal>,
    pub periodic_payment: Option<PeriodicPayment>,

    // Conditions
    pub conditions: Vec<TenureCondition>,
    pub transfer_restrictions: Vec<TransferRestriction>,

    // Status
    pub status: TenureStatus,
    pub verified: bool,
    pub verification_method: Option<VerificationMethod>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum TenureHolder {
    Individual(AgentPubKey),
    Joint(Vec<AgentPubKey>, JointTenureType),
    Organization(ActionHash),              // Agora governance space
    LandTrust(ActionHash),
    Cooperative(ActionHash),
    Community(ActionHash),
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum TenureType {
    // Ownership
    FeeSimple,
    FeeSimpleConditional { conditions: Vec<String> },
    LifeEstate { measuring_life: AgentPubKey },

    // Leasehold
    GroundLease { lessor: ActionHash },
    ResidentialLease,
    CommercialLease,

    // Cooperative
    CooperativeShare { coop: ActionHash, shares: u32 },
    LimitedEquity { max_appreciation: Decimal },

    // Use rights
    Usufruct,
    Easement { type_: EasementType },
    AccessRight,

    // Commons
    CommonsMember { commons: ActionHash },
    Steward { stewardship_agreement: ActionHash },

    // Temporary
    License,
    HouseSitting,
    WorkExchange,
}

/// Housing listing
#[hdk_entry_helper]
pub struct HousingListing {
    pub listing_id: String,
    pub property_hash: ActionHash,
    pub listed_by: AgentPubKey,

    // Type
    pub listing_type: ListingType,
    pub tenure_offered: TenureType,

    // Details
    pub title: String,
    pub description: String,
    pub photos: Vec<String>,               // IPFS hashes

    // Specifications
    pub bedrooms: Option<u8>,
    pub bathrooms: Option<f32>,
    pub area: Option<Area>,
    pub amenities: Vec<Amenity>,
    pub accessibility: AccessibilityFeatures,

    // Terms
    pub price: Option<Price>,
    pub deposit: Option<Decimal>,
    pub minimum_term: Option<Duration>,
    pub available_from: Timestamp,
    pub available_until: Option<Timestamp>,

    // Requirements
    pub requirements: Vec<TenantRequirement>,
    pub preferences: Vec<TenantPreference>,

    // Community context
    pub place_community: Option<ActionHash>,
    pub community_description: Option<String>,

    // Status
    pub status: ListingStatus,
    pub applications: Vec<ActionHash>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum ListingType {
    ForRent,
    ForSale,
    CooperativeShare,
    Cohousing,
    HouseSit,
    WorkExchange { work_description: String },
    CommunityHousing,
    AffordableHousing { income_limits: IncomeLimits },
}

/// Housing need (seeker profile)
#[hdk_entry_helper]
pub struct HousingNeed {
    pub need_id: String,
    pub seeker: AgentPubKey,

    // Urgency
    pub urgency: HousingUrgency,
    pub needed_by: Option<Timestamp>,

    // Requirements
    pub tenure_types: Vec<TenureType>,
    pub location_preferences: LocationPreferences,
    pub size_requirements: SizeRequirements,
    pub accessibility_needs: AccessibilityFeatures,
    pub must_have: Vec<Amenity>,

    // Household
    pub household_size: u8,
    pub household_composition: HouseholdComposition,
    pub pets: Vec<Pet>,

    // Financial
    pub budget: BudgetRange,
    pub income_verification: Option<ActionHash>,

    // Preferences
    pub community_preferences: Vec<CommunityPreference>,
    pub nice_to_have: Vec<Amenity>,

    // Status
    pub status: NeedStatus,
    pub matches: Vec<ActionHash>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum HousingUrgency {
    Planning,                              // 6+ months out
    Searching,                             // 1-6 months
    Urgent,                                // < 1 month
    Emergency,                             // Immediate need
}

/// Community Land Trust
#[hdk_entry_helper]
pub struct LandTrust {
    pub trust_id: String,
    pub name: String,
    pub description: String,

    // Governance
    pub governance_space: ActionHash,      // Agora
    pub board_structure: BoardStructure,

    // Mission
    pub mission: String,
    pub affordability_commitment: AffordabilityPolicy,
    pub community_benefit: Vec<String>,

    // Properties
    pub properties: Vec<ActionHash>,
    pub total_land_area: Area,
    pub total_units: u32,

    // Ground lease terms
    pub standard_ground_lease: GroundLeaseTerms,
    pub resale_formula: ResaleFormula,

    // Financial
    pub treasury: ActionHash,              // Treasury hApp
    pub stewardship_fee: Decimal,

    // Status
    pub status: TrustStatus,
    pub established: Timestamp,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct AffordabilityPolicy {
    pub target_ami_percentage: u8,         // % of Area Median Income
    pub income_tiers: Vec<IncomeTier>,
    pub permanent_affordability: bool,
    pub anti_displacement: Vec<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ResaleFormula {
    pub formula_type: ResaleFormulaType,
    pub appreciation_share: Decimal,       // % of appreciation to seller
    pub improvements_credit: bool,
    pub inflation_adjustment: bool,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum ResaleFormulaType {
    IndexBased { index: String },          // CPI, HPI, etc.
    AppraisalBased { cap: Decimal },
    FixedAppreciation { annual_rate: Decimal },
    AreaMedianIncome,
    Hybrid,
}

/// Bioregion definition
#[hdk_entry_helper]
pub struct Bioregion {
    pub bioregion_id: String,
    pub name: String,
    pub indigenous_name: Option<String>,

    // Geography
    pub boundary: GeoBoundary,
    pub watersheds: Vec<String>,
    pub ecosystem_types: Vec<String>,

    // Communities
    pub place_communities: Vec<ActionHash>,
    pub total_population: Option<u64>,

    // Governance
    pub coordination_body: Option<ActionHash>, // Diplomat federation
    pub shared_resources: Vec<ActionHash>,     // Commons

    // Ecological health
    pub health_metrics: Vec<EcologicalMetric>,
    pub stewardship_commitments: Vec<String>,
}
```

---

## Key Workflows

### Workflow 1: Community Land Trust Housing

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    CLT HOUSING FLOW                                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  STEP 1: Register Need                                                 │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Seeker registers housing need:                                 │   │
│  │  • Household size: 3                                            │   │
│  │  • Income: 60% AMI (verified via Accord)                       │   │
│  │  • Location: Downtown bioregion                                 │   │
│  │  • Urgency: Searching (3 months)                               │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│       │                                                                 │
│       ▼                                                                 │
│  STEP 2: Match with CLT                                                │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  System matches with Community Land Trust:                      │   │
│  │  • Trust: "Riverside CLT"                                       │   │
│  │  • Available unit: 2BR townhouse                               │   │
│  │  • Ground lease: $400/month                                    │   │
│  │  • Purchase price: $150,000 (limited equity)                   │   │
│  │  • Income requirement: ✓ 80% AMI max, seeker qualifies         │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│       │                                                                 │
│       ▼                                                                 │
│  STEP 3: Application & Review                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Application reviewed by CLT (via Agora):                       │   │
│  │  • Income verification: ✓                                       │   │
│  │  • Community fit interview: ✓                                   │   │
│  │  • Homebuyer education: Completed via Praxis                   │   │
│  │  • Board approval: Passed                                       │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│       │                                                                 │
│       ▼                                                                 │
│  STEP 4: Ground Lease Execution                                        │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Via Covenant:                                                  │   │
│  │  • 99-year ground lease signed                                 │   │
│  │  • Resale restrictions recorded                                │   │
│  │  • Monthly stewardship fee automated                           │   │
│  │  • Purchase completed via Treasury                             │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│       │                                                                 │
│       ▼                                                                 │
│  STEP 5: Tenure Recording                                              │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Terroir records:                                               │   │
│  │  • TenureRecord: LimitedEquity for Seeker                      │   │
│  │  • Encumbrance: Resale restriction                             │   │
│  │  • Community: Added to place_community                          │   │
│  │  • Chronicle: Permanent archive                                 │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Workflow 2: Bioregional Commons Coordination

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    BIOREGIONAL COORDINATION                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Multiple place_communities share a watershed                          │
│       │                                                                 │
│       ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  BIOREGION FORMATION                                            │   │
│  │                                                                  │   │
│  │  Bioregion: "Willamette Valley"                                │   │
│  │  • 12 place communities                                         │   │
│  │  • 3 major watersheds                                           │   │
│  │  • Shared resources: River, aquifer, forest                    │   │
│  │                                                                  │   │
│  │  Governance formed via Diplomat federation                      │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│       │                                                                 │
│       ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  COMMONS DEFINITION (via Commons hApp)                          │   │
│  │                                                                  │   │
│  │  Common Pool: "Willamette River Water Rights"                  │   │
│  │  • Access rules: Riparian communities                          │   │
│  │  • Usage limits: Sustainable yield calculation                 │   │
│  │  • Monitoring: Oracle integration (flow sensors)               │   │
│  │  • Governance: Bioregional council                             │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│       │                                                                 │
│       ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  COORDINATION ACTIVITIES                                        │   │
│  │                                                                  │   │
│  │  • Water allocation decisions (Agora)                          │   │
│  │  • Pollution response (Beacon emergency)                       │   │
│  │  • Restoration projects (Collab)                               │   │
│  │  • Land use coordination (Terroir)                             │   │
│  │  • Ecological monitoring (Oracle + Sentinel)                   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Integration Points

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    TERROIR INTEGRATIONS                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  CONTRACTS & FINANCE                                                    │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Covenant ──► Property sale/lease contracts                     │   │
│  │           ──► Ground lease agreements                           │   │
│  │           ──► Easement agreements                               │   │
│  │  Treasury ──► Property transactions                             │   │
│  │           ──► Rent collection                                   │   │
│  │           ──► CLT stewardship fees                              │   │
│  │  Accord ────► Property tax compliance                           │   │
│  │           ──► Capital gains tracking                            │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  GOVERNANCE                                                             │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Agora ──► CLT governance                                       │   │
│  │        ──► Place community decisions                            │   │
│  │        ──► Zoning/land use governance                           │   │
│  │  Diplomat ──► Bioregional federation                            │   │
│  │           ──► Inter-community agreements                        │   │
│  │  Commons ───► Shared resource management                        │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  VERIFICATION                                                           │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Oracle ──► Official registry verification                      │   │
│  │         ──► Property valuations                                 │   │
│  │         ──► Environmental data                                  │   │
│  │  Anchor ──► Physical location verification                      │   │
│  │  Attest ──► Identity for tenure holders                        │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  COMMUNITY                                                              │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Nexus ──► Community events                                     │   │
│  │  Kinship ──► Family housing coordination                        │   │
│  │  Sanctuary ──► Housing crisis response                          │   │
│  │  Weave ──► Neighbor relationships                               │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Affordability Models

### Supported Models

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    AFFORDABILITY MODELS                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  COMMUNITY LAND TRUST                                                   │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  • Land held by nonprofit trust                                 │   │
│  │  • Homes sold at below-market via ground lease                 │   │
│  │  • Resale restrictions preserve affordability                   │   │
│  │  • Democratic governance by residents + community              │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  LIMITED EQUITY COOPERATIVE                                            │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  • Members own shares, not units                               │   │
│  │  • Share prices capped (limited appreciation)                  │   │
│  │  • Democratic control: one member, one vote                    │   │
│  │  • Collective maintenance and governance                       │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  GEORGIST/LAND VALUE                                                   │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  • Land value captured for community benefit                   │   │
│  │  • Improvements owned by individuals                           │   │
│  │  • Land rent funds community services                          │   │
│  │  • Integrates with Accord for LVT calculation                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  MUTUAL HOUSING                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  • Nonprofit owns and operates housing                         │   │
│  │  • Rents set at affordability targets                          │   │
│  │  • Resident councils for governance                            │   │
│  │  • Cross-subsidization across portfolio                        │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Conclusion

Terroir provides the infrastructure for humanity's relationship with land - from individual homes to bioregional commons. By supporting diverse tenure models and integrating with economic, governance, and community systems, it enables place-based coordination at all scales.

*"We do not own the land. We belong to it. Terroir helps us honor that relationship."*

---

*Document Version: 1.0*
*Classification: Tier 2 - Essential*
*Dependencies: Covenant, Treasury, Agora, Commons, Diplomat, Oracle, Anchor, Accord*

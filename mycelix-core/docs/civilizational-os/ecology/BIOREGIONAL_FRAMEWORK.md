# Mycelix Bioregional & Ecological Integration Framework

## Overview

The Mycelix Bioregional Framework connects digital coordination with physical place and ecological reality. Drawing from bioregionalism, indigenous wisdom traditions, and ecological science, this framework ensures that community coordination respects planetary boundaries and nurtures the living systems that sustain all life.

**Core Principle**: Communities are nested within ecosystems, not separate from them.

---

## Theoretical Foundations

### Bioregionalism

Bioregionalism recognizes that human communities exist within natural regions defined by ecological characteristics rather than political boundaries.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    BIOREGIONAL NESTING                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                       ECOREGION                                  │   │
│  │  (e.g., Pacific Northwest Coastal)                              │   │
│  │                                                                  │   │
│  │  ┌─────────────────────────────────────────────────────────┐    │   │
│  │  │                    WATERSHED                             │    │   │
│  │  │  (e.g., Willamette River Basin)                         │    │   │
│  │  │                                                          │    │   │
│  │  │  ┌─────────────────────────────────────────────────┐    │    │   │
│  │  │  │              SUB-WATERSHED                       │    │    │   │
│  │  │  │  (e.g., Johnson Creek)                          │    │    │   │
│  │  │  │                                                  │    │    │   │
│  │  │  │  ┌─────────────────────────────────────────┐    │    │    │   │
│  │  │  │  │          NEIGHBORHOOD                    │    │    │    │   │
│  │  │  │  │  (e.g., Foster-Powell)                  │    │    │    │   │
│  │  │  │  │                                          │    │    │    │   │
│  │  │  │  │  ┌─────────────────────────────────┐    │    │    │    │   │
│  │  │  │  │  │         HOUSEHOLD                │    │    │    │    │   │
│  │  │  │  │  │  (Individual land stewardship)  │    │    │    │    │   │
│  │  │  │  │  └─────────────────────────────────┘    │    │    │    │   │
│  │  │  │  └─────────────────────────────────────────┘    │    │    │   │
│  │  │  └─────────────────────────────────────────────────┘    │    │   │
│  │  └─────────────────────────────────────────────────────────┘    │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Each level of community governance corresponds to ecological scale    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Planetary Boundaries (Stockholm Resilience Centre)

The nine planetary boundaries define the safe operating space for humanity:

```rust
/// Planetary boundary tracking
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PlanetaryBoundary {
    pub boundary_type: BoundaryType,
    pub current_status: BoundaryStatus,
    pub safe_threshold: Decimal,
    pub current_value: Decimal,
    pub unit: String,
    pub data_source: String,
    pub last_updated: Timestamp,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum BoundaryType {
    ClimateChange,
    BiosphereIntegrity {
        sub_type: BiosphereSubType,
    },
    LandSystemChange,
    FreshwaterUse,
    BiogeochemicalFlows {
        element: BioElement,
    },
    OceanAcidification,
    AtmosphericAerosolLoading,
    StratosphericOzoneDepletion,
    NovelEntities,  // Chemical pollution
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum BoundaryStatus {
    SafeOperatingSpace,    // Green
    IncreasingRisk,         // Yellow
    HighRisk,               // Red
    BeyondBoundary,         // Dark Red
    Unknown,
}
```

### Indigenous Wisdom Principles

Drawing from diverse indigenous traditions (with deep respect and proper attribution):

| Principle | Source Tradition(s) | Mycelix Application |
|-----------|---------------------|---------------------|
| **Seven Generation Thinking** | Haudenosaunee | Terroir, Legacy - decisions consider 7 generations forward |
| **Reciprocity with Land** | Many traditions | Commons - give back to land that gives to us |
| **All My Relations** | Lakota (Mitákuye Oyás'iŋ) | Weave - human relationships nested in more-than-human |
| **Seasonal Cycles** | All land-based traditions | Provision, Ember - honor natural rhythms |
| **Elder Wisdom** | Universal | Spiral - honor developmental wisdom, intergenerational learning |
| **Collective Stewardship** | Many traditions | Commons, Terroir - land belongs to community across time |
| **Sacred Places** | All traditions | Anchor - some places require special protection |
| **Story as Law** | Aboriginal Australian | Chronicle, Loom - narratives encode ecological wisdom |

---

## Place-Based Integration

### Anchor hApp - Physical Location Binding

```rust
/// Anchor: Binding digital to physical place
#[hdk_entry_helper]
pub struct Place {
    pub place_id: String,
    pub name: String,
    pub description: String,

    /// Geographic definition
    pub geography: PlaceGeography,

    /// Ecological context
    pub ecological_context: EcologicalContext,

    /// Cultural significance
    pub cultural_significance: CulturalSignificance,

    /// Governance relationship
    pub governance: PlaceGovernance,

    /// Stewardship responsibilities
    pub stewardship: Vec<StewardshipResponsibility>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PlaceGeography {
    /// Boundary definition
    pub boundary: GeoBoundary,

    /// Bioregional nesting
    pub bioregion: BioregionalNesting,

    /// Watershed
    pub watershed: Option<WatershedInfo>,

    /// Soil type
    pub soil: Option<SoilInfo>,

    /// Climate zone
    pub climate_zone: Option<ClimateZone>,

    /// Elevation range
    pub elevation: Option<ElevationRange>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EcologicalContext {
    /// Native ecosystem type
    pub native_ecosystem: String,

    /// Current ecosystem health
    pub ecosystem_health: EcosystemHealth,

    /// Key species
    pub keystone_species: Vec<Species>,
    pub indicator_species: Vec<Species>,
    pub threatened_species: Vec<Species>,

    /// Ecological functions
    pub ecological_functions: Vec<EcologicalFunction>,

    /// Restoration needs
    pub restoration_priorities: Vec<RestorationPriority>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CulturalSignificance {
    /// Indigenous territory (if known and appropriate to share)
    pub indigenous_territory: Option<IndigenousTerritory>,

    /// Historical significance
    pub historical_significance: Vec<HistoricalNote>,

    /// Sacred sites (with appropriate protections)
    pub sacred_sites: Vec<SacredSite>,

    /// Cultural practices tied to place
    pub cultural_practices: Vec<CulturalPractice>,

    /// Place names in indigenous languages
    pub indigenous_names: Vec<IndigenousName>,
}

/// Bioregional nesting structure
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BioregionalNesting {
    pub ecoregion: EcoregionInfo,
    pub watershed: WatershedInfo,
    pub sub_watershed: Option<SubWatershedInfo>,
    pub neighborhood: Option<NeighborhoodInfo>,
    pub site: Option<SiteInfo>,
}
```

### Place-Based Governance

```rust
/// Governance that respects ecological scale
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PlaceGovernance {
    /// Decisions appropriate to this scale
    pub decision_scope: DecisionScope,

    /// Relationship to nested scales
    pub nesting: GovernanceNesting,

    /// Ecological representation
    pub ecological_voice: EcologicalRepresentation,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum DecisionScope {
    /// Household decisions
    Household {
        affects: Vec<String>,
        requires_notification: Vec<String>,
    },

    /// Neighborhood decisions
    Neighborhood {
        affects: Vec<String>,
        requires_sub_watershed_notification: bool,
    },

    /// Sub-watershed decisions
    SubWatershed {
        affects: Vec<String>,
        requires_watershed_notification: bool,
    },

    /// Watershed decisions
    Watershed {
        affects: Vec<String>,
        requires_ecoregion_notification: bool,
    },

    /// Ecoregion decisions
    Ecoregion {
        affects: Vec<String>,
        requires_planetary_consideration: bool,
    },
}

/// Ecological representation in governance
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EcologicalRepresentation {
    /// Designated ecological advocates
    pub advocates: Vec<EcologicalAdvocate>,

    /// Species voice (designated human speakers for species)
    pub species_voices: Vec<SpeciesVoice>,

    /// Watershed voice
    pub watershed_voice: Option<WatershedVoice>,

    /// Future generations voice
    pub future_generations_voice: Option<FutureGenerationsVoice>,
}

/// Someone who speaks for an ecological entity
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EcologicalAdvocate {
    pub advocate_id: String,
    pub human_agent: AgentPubKey,
    pub represents: EcologicalEntity,
    pub authority: AdvocateAuthority,
    pub term: Option<Term>,
    pub selection_process: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum EcologicalEntity {
    Species { species: Species },
    Watershed { watershed_id: String },
    Forest { forest_id: String },
    Wetland { wetland_id: String },
    Soil { soil_community: String },
    Air { airshed: String },
    FutureGenerations { horizon: Duration },
    TheWhole,  // Advocate for the entire ecosystem
}
```

---

## Ecological Indicators

### Ecosystem Health Monitoring

```rust
/// Comprehensive ecosystem health tracking
#[hdk_entry_helper]
pub struct EcosystemHealthAssessment {
    pub assessment_id: String,
    pub place_id: String,
    pub assessment_date: Timestamp,

    /// Indicator measurements
    pub indicators: Vec<EcologicalIndicator>,

    /// Overall health score
    pub overall_health: HealthScore,

    /// Trends
    pub trends: Vec<HealthTrend>,

    /// Data sources
    pub data_sources: Vec<DataSource>,

    /// Community observations
    pub community_observations: Vec<CommunityObservation>,

    /// Recommendations
    pub recommendations: Vec<EcologicalRecommendation>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum EcologicalIndicator {
    // Biodiversity indicators
    SpeciesRichness { count: u32, baseline: u32 },
    NativeSpeciesRatio { ratio: Decimal },
    IndicatorSpeciesPresence { species: String, status: PresenceStatus },
    PollinatorAbundance { index: Decimal },

    // Water indicators
    WaterQuality { ph: Decimal, dissolved_oxygen: Decimal, turbidity: Decimal },
    StreamFlow { volume: Decimal, seasonal_pattern: SeasonalPattern },
    GroundwaterLevel { depth: Decimal, trend: Trend },
    RiparianHealth { index: Decimal },

    // Soil indicators
    SoilHealth { organic_matter: Decimal, microbial_activity: Decimal },
    Erosion { rate: Decimal, severity: Severity },
    SoilCompaction { index: Decimal },

    // Air indicators
    AirQuality { aqi: u32, particulates: Decimal },

    // Carbon indicators
    CarbonSequestration { tons_per_year: Decimal },
    ForestCover { percentage: Decimal, change: Decimal },

    // Connectivity indicators
    HabitatConnectivity { index: Decimal },
    WildlifeCorridors { status: CorridorStatus },

    // Human impact indicators
    LandUseIntensity { index: Decimal },
    ChemicalInputs { type_: String, quantity: Decimal },
    WasteGeneration { volume: Decimal, type_: String },
}

/// Community ecological observation (citizen science)
#[hdk_entry_helper]
pub struct CommunityObservation {
    pub observation_id: String,
    pub observer: AgentPubKey,
    pub place_id: String,
    pub timestamp: Timestamp,
    pub observation_type: ObservationType,
    pub details: String,
    pub evidence: Vec<Evidence>,
    pub verification_status: VerificationStatus,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ObservationType {
    SpeciesSighting { species: String, count: u32 },
    WaterCondition { condition: String },
    WeatherAnomaly { anomaly: String },
    PollutionEvent { type_: String, severity: Severity },
    HabitatChange { change: String },
    SeasonalShift { phenomenon: String, timing: String },
    RestorationProgress { project_id: String, observation: String },
    TraditionalKnowledge { knowledge: String, source_permission: bool },
}
```

### Planetary Boundary Integration

```rust
/// Connect local actions to planetary boundaries
pub struct PlanetaryBoundaryIntegration {
    /// Local contribution to planetary boundaries
    pub local_impacts: Vec<LocalPlanetaryImpact>,

    /// Boundary-aware decision support
    pub decision_support: BoundaryAwareDecisions,

    /// Reporting and accountability
    pub reporting: BoundaryReporting,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LocalPlanetaryImpact {
    pub boundary: BoundaryType,
    pub local_contribution: LocalContribution,
    pub per_capita_share: Decimal,
    pub trend: Trend,
    pub reduction_potential: Decimal,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LocalContribution {
    /// Direct emissions/impacts
    pub direct: Decimal,

    /// Indirect (consumption-based)
    pub indirect: Decimal,

    /// Positive contributions (restoration, sequestration)
    pub positive: Decimal,

    /// Net impact
    pub net: Decimal,

    pub unit: String,
}

/// Make decisions with boundary awareness
pub fn assess_proposal_boundary_impact(
    proposal: &Proposal,
    place: &Place,
) -> Result<BoundaryImpactAssessment, AssessmentError> {
    let mut impacts = Vec::new();

    // Assess climate impact
    if let Some(climate_impact) = assess_climate_impact(proposal)? {
        impacts.push(BoundaryImpact {
            boundary: BoundaryType::ClimateChange,
            impact: climate_impact,
            significance: calculate_significance(&climate_impact, &place),
        });
    }

    // Assess biodiversity impact
    if let Some(bio_impact) = assess_biodiversity_impact(proposal, place)? {
        impacts.push(BoundaryImpact {
            boundary: BoundaryType::BiosphereIntegrity {
                sub_type: BiosphereSubType::FunctionalDiversity,
            },
            impact: bio_impact,
            significance: calculate_significance(&bio_impact, &place),
        });
    }

    // Assess water impact
    if let Some(water_impact) = assess_freshwater_impact(proposal, place)? {
        impacts.push(BoundaryImpact {
            boundary: BoundaryType::FreshwaterUse,
            impact: water_impact,
            significance: calculate_significance(&water_impact, &place),
        });
    }

    // ... assess other boundaries

    Ok(BoundaryImpactAssessment {
        proposal_id: proposal.id.clone(),
        impacts,
        overall_boundary_alignment: calculate_overall_alignment(&impacts),
        recommendations: generate_boundary_recommendations(&impacts),
    })
}
```

---

## Indigenous Wisdom Integration

### Principles and Practices

```rust
/// Indigenous wisdom integration (with deep respect and proper protocol)
pub struct IndigenousWisdomIntegration {
    /// Acknowledgment of indigenous territory
    pub land_acknowledgment: LandAcknowledgment,

    /// Partnership protocols
    pub partnership_protocols: Vec<PartnershipProtocol>,

    /// Traditional ecological knowledge (with permission)
    pub tek_integration: TEKIntegration,

    /// Seasonal and ceremonial awareness
    pub seasonal_awareness: SeasonalAwareness,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LandAcknowledgment {
    /// Traditional territory
    pub traditional_territory: String,

    /// Indigenous nations
    pub nations: Vec<IndigenousNation>,

    /// Acknowledgment text (developed with community)
    pub acknowledgment_text: String,

    /// Living relationship (not just historical)
    pub ongoing_relationship: OngoingRelationship,

    /// Actions supporting indigenous sovereignty
    pub support_actions: Vec<SupportAction>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OngoingRelationship {
    /// Active partnerships
    pub partnerships: Vec<IndigenousPartnership>,

    /// Revenue/resource sharing
    pub resource_sharing: Option<ResourceSharingAgreement>,

    /// Joint governance
    pub governance_involvement: Option<GovernanceInvolvement>,

    /// Cultural revitalization support
    pub cultural_support: Vec<CulturalSupportAction>,
}

/// Traditional Ecological Knowledge integration
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TEKIntegration {
    /// Permission and protocol
    pub permission_status: TEKPermissionStatus,

    /// Knowledge holders
    pub knowledge_holders: Vec<KnowledgeHolder>,

    /// Integrated practices
    pub integrated_practices: Vec<TEKPractice>,

    /// Protection of sacred/restricted knowledge
    pub knowledge_protection: KnowledgeProtection,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum TEKPermissionStatus {
    /// Full permission with protocols
    PermittedWithProtocol {
        protocols: Vec<TEKProtocol>,
        attribution_requirements: Vec<String>,
    },

    /// Limited permission for specific uses
    LimitedPermission {
        permitted_uses: Vec<String>,
        restrictions: Vec<String>,
    },

    /// Consultation ongoing
    ConsultationOngoing,

    /// Not appropriate to integrate (sacred/restricted)
    NotAppropriate { reason: String },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TEKPractice {
    pub practice_id: String,
    pub name: String,
    pub description: String,
    pub source_tradition: String,
    pub permission_level: TEKPermissionStatus,
    pub integration_in_happ: String,
    pub attribution: String,
}
```

### Seven Generation Thinking

```rust
/// Implement seven generation thinking in governance
pub struct SevenGenerationThinking {
    /// Time horizon for decisions
    pub time_horizon: Duration,  // ~175 years (7 generations * 25 years)

    /// Impact assessment requirements
    pub assessment_requirements: SevenGenAssessment,

    /// Intergenerational representation
    pub representation: IntergenerationalRep,

    /// Legacy tracking
    pub legacy_tracking: LegacyTracking,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SevenGenAssessment {
    /// Required for major decisions
    pub required_decision_types: Vec<DecisionType>,

    /// Assessment questions
    pub assessment_questions: Vec<SevenGenQuestion>,

    /// Future generations advocate input
    pub requires_advocate_input: bool,
}

/// Seven generation impact assessment for a proposal
pub fn assess_seven_generation_impact(
    proposal: &Proposal,
    place: &Place,
) -> Result<SevenGenImpactAssessment, AssessmentError> {
    let mut impacts_by_generation = Vec::new();

    for generation in 1..=7 {
        let horizon = Duration::from_years(generation as u64 * 25);
        let impacts = project_impacts(&proposal, &place, horizon)?;
        impacts_by_generation.push(GenerationImpact {
            generation,
            years_ahead: generation * 25,
            projected_impacts: impacts,
            reversibility: assess_reversibility(&impacts),
        });
    }

    // Get advocate input
    let advocate_assessment = if let Some(advocate) = get_future_generations_advocate(&place.place_id)? {
        Some(request_advocate_assessment(&advocate, &proposal).await?)
    } else {
        None
    };

    Ok(SevenGenImpactAssessment {
        proposal_id: proposal.id.clone(),
        impacts_by_generation,
        cumulative_impact: calculate_cumulative_impact(&impacts_by_generation),
        advocate_assessment,
        recommendation: generate_seven_gen_recommendation(&impacts_by_generation),
    })
}

/// Questions for seven generation assessment
pub const SEVEN_GEN_QUESTIONS: [&str; 7] = [
    "How will this decision affect those not yet born?",
    "What resources will future generations need that this might consume?",
    "What knowledge or wisdom might we be foreclosing?",
    "What healing or restoration does this enable for the future?",
    "How does this honor our ancestors' gifts to us?",
    "What stories will future generations tell about this decision?",
    "If the seventh generation could vote, how would they vote?",
];
```

### Seasonal Awareness

```rust
/// Seasonal and ceremonial calendar integration
pub struct SeasonalAwareness {
    /// Ecological seasons (not just calendar)
    pub ecological_seasons: Vec<EcologicalSeason>,

    /// Traditional ceremonial calendar (where appropriate)
    pub ceremonial_awareness: Option<CeremonialCalendar>,

    /// Seasonal activities
    pub seasonal_activities: Vec<SeasonalActivity>,

    /// Seasonal governance rhythms
    pub governance_rhythms: SeasonalGovernance,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EcologicalSeason {
    pub name: String,
    pub local_names: Vec<LocalName>,  // Including indigenous names
    pub typical_timing: SeasonalTiming,
    pub ecological_markers: Vec<String>,  // What indicates this season
    pub key_species_activities: Vec<SpeciesActivity>,
    pub appropriate_human_activities: Vec<String>,
    pub inappropriate_activities: Vec<String>,  // Things that should wait
}

/// Seasonal governance rhythms
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SeasonalGovernance {
    /// Major decisions timed to seasons
    pub decision_timing: Vec<SeasonalDecisionTiming>,

    /// Seasonal gatherings
    pub seasonal_gatherings: Vec<SeasonalGathering>,

    /// Rest periods
    pub governance_rest_periods: Vec<RestPeriod>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SeasonalDecisionTiming {
    pub decision_type: DecisionType,
    pub appropriate_seasons: Vec<String>,
    pub rationale: String,
}

// Example: Land use decisions made in winter when land is resting
// Example: Planting decisions made after observing spring indicators
// Example: Harvest sharing decisions made at harvest time
```

---

## Regenerative Practices

### Regeneration Tracking

```rust
/// Track regenerative vs extractive activities
#[hdk_entry_helper]
pub struct RegenerationAccount {
    pub account_id: String,
    pub entity_type: EntityType,  // Individual, household, community
    pub entity_id: String,

    /// Regenerative activities
    pub regenerative_activities: Vec<RegenerativeActivity>,

    /// Extractive activities
    pub extractive_activities: Vec<ExtractiveActivity>,

    /// Net regeneration score
    pub net_regeneration: RegenerationScore,

    /// Period
    pub period: Period,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RegenerativeActivity {
    pub activity_id: String,
    pub activity_type: RegenerativeType,
    pub description: String,
    pub ecological_benefit: EcologicalBenefit,
    pub verification: Verification,
    pub timestamp: Timestamp,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum RegenerativeType {
    // Soil regeneration
    Composting { volume: Decimal },
    CoverCropping { area: Decimal },
    NoTill { area: Decimal },
    SoilAmendment { type_: String, area: Decimal },

    // Water regeneration
    RainwaterHarvest { volume: Decimal },
    WetlandRestoration { area: Decimal },
    RiparianPlanting { length: Decimal },
    GraywaterReuse { volume: Decimal },

    // Biodiversity regeneration
    NativePlanting { species_count: u32, area: Decimal },
    HabitatCreation { type_: String, area: Decimal },
    InvasiveRemoval { species: String, area: Decimal },
    PollinatorHabitat { area: Decimal },

    // Carbon regeneration
    TreePlanting { count: u32, species: Vec<String> },
    Agroforestry { area: Decimal },
    CarbonFarming { practice: String, area: Decimal },

    // Waste regeneration
    Recycling { type_: String, volume: Decimal },
    Upcycling { description: String },
    WasteReduction { type_: String, percentage: Decimal },

    // Energy regeneration
    RenewableGeneration { type_: String, kwh: Decimal },
    EnergyEfficiency { savings_kwh: Decimal },

    // Social regeneration
    KnowledgeSharing { topic: String, recipients: u32 },
    SkillBuilding { skill: String, learners: u32 },
    CommunityBuilding { activity: String },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EcologicalBenefit {
    pub primary_benefit: String,
    pub secondary_benefits: Vec<String>,
    pub ecosystem_services_enhanced: Vec<EcosystemService>,
    pub biodiversity_impact: BiodiversityImpact,
    pub carbon_impact: Option<CarbonImpact>,
    pub water_impact: Option<WaterImpact>,
}
```

### Restoration Coordination

```rust
/// Coordinate ecological restoration projects
#[hdk_entry_helper]
pub struct RestorationProject {
    pub project_id: String,
    pub name: String,
    pub description: String,

    /// Location
    pub place_id: String,
    pub area: GeoArea,

    /// Restoration type
    pub restoration_type: RestorationType,

    /// Goals
    pub goals: Vec<RestorationGoal>,

    /// Timeline
    pub phases: Vec<RestorationPhase>,

    /// Participants
    pub stewards: Vec<Steward>,
    pub volunteers: Vec<Volunteer>,
    pub partners: Vec<Partner>,

    /// Resources
    pub resources_needed: Vec<Resource>,
    pub resources_secured: Vec<Resource>,

    /// Progress tracking
    pub progress: RestorationProgress,

    /// Ecological outcomes
    pub outcomes: Vec<EcologicalOutcome>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum RestorationType {
    WetlandRestoration,
    ForestRegeneration,
    PrairieRestoration,
    RiparianRestoration,
    SoilRemediation,
    HabitatConnectivity,
    InvasiveSpeciesRemoval,
    WildlifeCorridorCreation,
    UrbanRegreening,
    AgroecologicalTransition,
    Custom { name: String, description: String },
}

/// Match restoration needs with community resources
pub async fn match_restoration_needs(
    project: &RestorationProject,
    community: &str,
) -> Result<RestorationMatches, MatchError> {
    // Find available volunteers
    let volunteers = find_available_volunteers(community, &project.phases).await?;

    // Find needed materials (via Marketplace)
    let materials = find_materials_available(&project.resources_needed).await?;

    // Find expertise (via Guild)
    let experts = find_relevant_experts(&project.restoration_type).await?;

    // Find funding (via Treasury)
    let funding = find_funding_sources(&project.resources_needed).await?;

    Ok(RestorationMatches {
        project_id: project.project_id.clone(),
        volunteer_matches: volunteers,
        material_matches: materials,
        expert_matches: experts,
        funding_matches: funding,
        overall_readiness: calculate_readiness(&volunteers, &materials, &experts, &funding),
    })
}
```

---

## hApp Ecological Integration

### Terroir - Land Integration

```rust
/// Terroir ecological requirements
pub struct TerroirEcologicalIntegration {
    /// All land parcels have ecological assessment
    pub ecological_assessment_required: bool,

    /// Land use must respect ecological capacity
    pub carrying_capacity_enforcement: bool,

    /// Native ecosystem consideration in land decisions
    pub native_ecosystem_requirements: NativeEcosystemRequirements,

    /// Wildlife corridor protection
    pub corridor_protection: CorridorProtection,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NativeEcosystemRequirements {
    /// Minimum percentage native plants
    pub min_native_coverage: Decimal,

    /// Prohibited species (invasive)
    pub prohibited_species: Vec<String>,

    /// Required wildlife features
    pub required_wildlife_features: Vec<WildlifeFeature>,

    /// Habitat connectivity requirements
    pub connectivity_requirements: ConnectivityRequirements,
}

/// Land use proposal with ecological review
pub async fn propose_land_use(
    proposal: LandUseProposal,
) -> Result<LandUseProposalWithReview, ProposalError> {
    // Assess ecological impact
    let ecological_review = assess_land_use_ecology(&proposal).await?;

    // Assess seven generation impact
    let seven_gen_review = assess_seven_generation_impact(&proposal, &get_place(&proposal.parcel_id)?).await?;

    // Check planetary boundary alignment
    let boundary_review = assess_proposal_boundary_impact(&proposal, &get_place(&proposal.parcel_id)?).await?;

    // Get ecological advocate input
    let advocate_input = get_ecological_advocate_input(&proposal.parcel_id, &proposal).await?;

    Ok(LandUseProposalWithReview {
        proposal,
        ecological_review,
        seven_gen_review,
        boundary_review,
        advocate_input,
        overall_ecological_recommendation: generate_overall_recommendation(
            &ecological_review,
            &seven_gen_review,
            &boundary_review,
        ),
    })
}
```

### Provision - Food Systems

```rust
/// Provision ecological integration
pub struct ProvisionEcologicalIntegration {
    /// Local food sourcing priority
    pub local_priority: LocalFoodPriority,

    /// Seasonal eating support
    pub seasonal_support: SeasonalEatingSupport,

    /// Regenerative agriculture preference
    pub regenerative_preference: RegenerativePreference,

    /// Food miles tracking
    pub food_miles_tracking: bool,

    /// Soil health connection
    pub soil_health_integration: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LocalFoodPriority {
    /// Radius considered "local"
    pub local_radius_km: u32,

    /// Priority ordering in search/display
    pub display_priority: bool,

    /// Local food percentage goals
    pub local_percentage_goal: Decimal,
}

/// Food listing with ecological context
#[hdk_entry_helper]
pub struct EcologicalFoodListing {
    pub listing_id: String,
    pub basic_listing: FoodListing,

    /// Growing method
    pub growing_method: GrowingMethod,

    /// Ecological practices
    pub ecological_practices: Vec<EcologicalPractice>,

    /// Distance from community center
    pub food_miles: Decimal,

    /// Seasonal alignment
    pub seasonal_alignment: SeasonalAlignment,

    /// Soil health
    pub soil_health_score: Option<Decimal>,

    /// Carbon footprint estimate
    pub carbon_footprint: Option<CarbonFootprint>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum GrowingMethod {
    Regenerative,
    Organic,
    Permaculture,
    Biodynamic,
    Agroforestry,
    Conventional,
    Transitioning { toward: Box<GrowingMethod> },
    Unknown,
}
```

### Ember - Energy Integration

```rust
/// Ember ecological integration
pub struct EmberEcologicalIntegration {
    /// Renewable energy priority
    pub renewable_priority: RenewablePriority,

    /// Grid carbon intensity tracking
    pub carbon_intensity_tracking: bool,

    /// Demand response ecology
    pub ecological_demand_response: EcologicalDemandResponse,

    /// Energy source ecological impact
    pub source_impact_visibility: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EcologicalDemandResponse {
    /// Reduce demand when grid is carbon-intensive
    pub carbon_aware_demand: bool,

    /// Reduce demand during ecological sensitive times
    pub ecological_sensitive_times: Vec<EcologicalSensitiveTime>,

    /// Coordinate with natural cycles
    pub natural_cycle_coordination: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EcologicalSensitiveTime {
    pub name: String,
    pub time_pattern: TimePattern,
    pub ecological_reason: String,
    pub demand_reduction_request: DemandReduction,
}

// Example: Reduce lighting during bird migration nights
// Example: Reduce noise/activity during wildlife breeding seasons
```

---

## Inter-Community Coordination

### Watershed Governance

```rust
/// Coordinate governance at watershed scale
pub struct WatershedGovernance {
    pub watershed_id: String,
    pub watershed_name: String,

    /// Communities in watershed
    pub member_communities: Vec<String>,

    /// Shared governance structure
    pub governance_structure: WatershedGovernanceStructure,

    /// Shared resources
    pub shared_resources: Vec<SharedWatershedResource>,

    /// Watershed-wide policies
    pub policies: Vec<WatershedPolicy>,

    /// Monitoring network
    pub monitoring_network: WatershedMonitoringNetwork,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WatershedGovernanceStructure {
    /// Representation from each community
    pub community_representatives: HashMap<String, Vec<AgentPubKey>>,

    /// Ecological advocates
    pub watershed_advocates: Vec<EcologicalAdvocate>,

    /// Decision process
    pub decision_process: WatershedDecisionProcess,

    /// Conflict resolution
    pub conflict_resolution: WatershedConflictResolution,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum WatershedDecisionProcess {
    /// Consensus of all communities
    Consensus,

    /// Weighted by watershed impact
    ImpactWeighted { weighting_formula: String },

    /// Delegated to watershed council
    CouncilDecision { council_composition: String },

    /// Different processes for different scales
    Tiered { tiers: Vec<DecisionTier> },
}
```

### Bioregional Federation

```rust
/// Federate communities at bioregional scale
pub struct BioregionalFederation {
    pub federation_id: String,
    pub bioregion: EcoregionInfo,

    /// Member watersheds
    pub member_watersheds: Vec<String>,

    /// Member communities
    pub member_communities: Vec<String>,

    /// Shared ecological goals
    pub shared_goals: Vec<BioregionalGoal>,

    /// Coordination mechanisms
    pub coordination: BioregionalCoordination,

    /// Shared resources
    pub shared_resources: BioregionalResources,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BioregionalGoal {
    pub goal_id: String,
    pub name: String,
    pub description: String,
    pub target: BioregionalTarget,
    pub timeline: Duration,
    pub progress: Decimal,
    pub responsible_entities: Vec<ResponsibleEntity>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum BioregionalTarget {
    // Biodiversity targets
    SpeciesRecovery { species: String, target_population: u32 },
    HabitatRestoration { type_: String, target_area: Decimal },
    ConnectivityRestoration { corridors: u32 },

    // Water targets
    WaterQualityImprovement { metric: String, target: Decimal },
    FloodRiskReduction { target_percentage: Decimal },
    GroundwaterRecharge { target_volume: Decimal },

    // Carbon targets
    CarbonNeutrality { target_year: u32 },
    CarbonSequestration { target_tons: Decimal },

    // Resilience targets
    FoodSovereignty { target_local_percentage: Decimal },
    EnergyIndependence { target_renewable_percentage: Decimal },
    WaterSecurity { target_local_sources: Decimal },
}
```

---

## Conclusion

The Mycelix Bioregional Framework grounds digital coordination in ecological reality. By integrating place-based governance, planetary boundaries, indigenous wisdom, and regenerative practices, communities can coordinate in ways that heal rather than harm the living systems that sustain all life.

**Key Principles Summary**:
1. **Place-based** - Governance scaled to ecological regions
2. **Boundary-aware** - Decisions respect planetary limits
3. **Indigenous wisdom** - Honor and integrate traditional ecological knowledge
4. **Seven generations** - Think long-term, beyond our lifetimes
5. **Regenerative** - Restore more than we take
6. **Federated** - Coordinate across nested ecological scales

---

*"The land does not belong to us; we belong to the land. Our digital tools must serve this truth, not obscure it."*

---

*Document Version: 1.0*
*Last Updated: 2025*
*Acknowledgments: This framework acknowledges the traditional ecological knowledge of indigenous peoples worldwide. We commit to ongoing partnership, proper protocol, and support for indigenous sovereignty.*

# Mycelix-Provision: Food Systems & Agricultural Coordination

## Vision Statement

*"Food is the most intimate relationship between humans and the living world. Provision coordinates the growing, sharing, and distribution of nourishment - from backyard gardens to bioregional food systems."*

---

## Executive Summary

Mycelix-Provision provides infrastructure for food sovereignty:

1. **Agricultural coordination** - Crop planning, resource sharing, collective farming
2. **Food distribution** - CSAs, food hubs, mutual aid distribution
3. **Seed & knowledge commons** - Open-source seeds, traditional knowledge preservation
4. **Food security** - Need tracking, emergency food coordination
5. **Supply chain transparency** - Farm-to-table traceability

---

## Core Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    PROVISION ARCHITECTURE                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  PRODUCTION LAYER                                                       │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Farm Registry │ Crop Planning │ Resource Sharing │ Harvest     │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                               │                                         │
│  DISTRIBUTION LAYER                                                     │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  CSA Management │ Food Hubs │ Direct Sales │ Food Rescue        │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                               │                                         │
│  COMMONS LAYER                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Seed Library │ Knowledge Base │ Equipment Sharing │ Land Access│   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                               │                                         │
│  SECURITY LAYER                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Need Tracking │ Emergency Response │ Food Sovereignty Metrics  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Key Data Types

```rust
/// Farm or growing operation
pub struct Farm {
    pub farm_id: String,
    pub name: String,
    pub location: ActionHash,              // Terroir property
    pub farm_type: FarmType,
    pub practices: Vec<FarmingPractice>,
    pub certifications: Vec<Certification>,
    pub growing_zones: Vec<GrowingZone>,
    pub operator: TenureHolder,
    pub public_profile: bool,
}

pub enum FarmType {
    Market,
    Subsistence,
    Community,
    Cooperative,
    Educational,
    Research,
    Urban,
    Permaculture,
}

pub enum FarmingPractice {
    Organic,
    Biodynamic,
    Regenerative,
    NoTill,
    Agroforestry,
    Aquaponics,
    Hydroponics,
    Conventional,
    Traditional(String),
}

/// Crop or livestock planning
pub struct CropPlan {
    pub plan_id: String,
    pub farm_hash: ActionHash,
    pub season: GrowingSeason,
    pub crops: Vec<PlannedCrop>,
    pub estimated_yield: HashMap<String, Quantity>,
    pub resource_needs: Vec<ResourceNeed>,
    pub commitments: Vec<ActionHash>,      // CSA, contracts
}

/// Community Supported Agriculture share
pub struct CSAShare {
    pub csa_id: String,
    pub farm_hash: ActionHash,
    pub season: GrowingSeason,
    pub share_type: ShareType,
    pub price: Decimal,
    pub pickup_options: Vec<PickupOption>,
    pub share_contents: ShareContents,
    pub member: Option<AgentPubKey>,
    pub status: ShareStatus,
}

pub enum ShareType {
    Full,
    Half,
    Fruit,
    Vegetable,
    Egg,
    Dairy,
    Meat,
    FlowerShare,
    Custom(Vec<String>),
    WorkShare { hours_required: u32 },
}

/// Food hub for aggregation and distribution
pub struct FoodHub {
    pub hub_id: String,
    pub name: String,
    pub location: ActionHash,
    pub hub_type: HubType,
    pub services: Vec<HubService>,
    pub member_farms: Vec<ActionHash>,
    pub distribution_area: GeoBoundary,
    pub governance: ActionHash,            // Agora
}

pub enum HubType {
    Aggregation,
    Processing,
    Distribution,
    Retail,
    Wholesale,
    FoodBank,
    CommunityKitchen,
}

/// Seed in the commons library
pub struct SeedEntry {
    pub seed_id: String,
    pub variety_name: String,
    pub species: String,
    pub origin: SeedOrigin,
    pub characteristics: Vec<Characteristic>,
    pub growing_requirements: GrowingRequirements,
    pub saved_by: Vec<SeedSaver>,
    pub available_from: Vec<SeedSource>,
    pub license: SeedLicense,
    pub traditional_knowledge: Option<TraditionalKnowledge>,
}

pub enum SeedLicense {
    OpenSourceSeed,                        // OSSI pledge
    PublicDomain,
    TraditionalVariety,
    CommunityHeritage,
    PatentExpired,
    Proprietary,                           // Flagged, not encouraged
}

/// Food need registration
pub struct FoodNeed {
    pub need_id: String,
    pub household: AgentPubKey,
    pub household_size: u8,
    pub dietary_restrictions: Vec<DietaryRestriction>,
    pub urgency: FoodUrgency,
    pub need_type: FoodNeedType,
    pub location: ActionHash,
    pub status: NeedStatus,
}

pub enum FoodUrgency {
    Planning,                              // Future planning
    Regular,                               // Ongoing need
    Acute,                                 // This week
    Emergency,                             // Today
}

pub enum FoodNeedType {
    General,
    Fresh,
    Staples,
    Protein,
    BabyFood,
    ElderlyNutrition,
    MedicalDiet(String),
}
```

---

## Key Workflows

### Workflow 1: CSA Season Management

```
Farm creates CSA offering → Members purchase shares →
Season progresses with harvest tracking →
Weekly distribution coordinated →
End of season settlement and feedback
```

### Workflow 2: Food Hub Aggregation

```
Multiple farms list available products →
Hub aggregates orders from buyers →
Logistics coordinated (Transit integration) →
Quality verification at hub →
Distribution to buyers with full traceability
```

### Workflow 3: Emergency Food Response

```
Food need registered (individual or community) →
Beacon activates if emergency →
Available resources identified (farms, pantries, surplus) →
Distribution coordinated →
Follow-up for ongoing food security
```

### Workflow 4: Seed Saving & Sharing

```
Grower saves seed from harvest →
Documents variety characteristics →
Adds to seed library with growing notes →
Community members request seeds →
Traditional knowledge preserved in Chronicle
```

---

## Integration Points

| Integration | Purpose |
|-------------|---------|
| **Terroir** | Farm land registry, land access |
| **SupplyChain** | Farm-to-table traceability |
| **Marketplace** | Food sales and transactions |
| **Treasury** | CSA payments, food hub finances |
| **Transit** | Food distribution logistics |
| **Beacon** | Emergency food coordination |
| **Commons** | Shared equipment, land access |
| **Oracle** | Weather data, market prices |
| **Praxis** | Agricultural education |
| **Chronicle** | Traditional knowledge archive |
| **HealthVault** | Dietary requirement integration |

---

## Food Sovereignty Metrics

```rust
pub struct FoodSovereigntyMetrics {
    // Production
    pub local_production_percentage: f64,
    pub crop_diversity_index: f64,
    pub regenerative_acreage: Area,

    // Access
    pub households_food_secure: f64,
    pub average_food_miles: f64,
    pub affordable_food_access: f64,

    // Resilience
    pub seed_varieties_preserved: u32,
    pub food_storage_days: u32,
    pub farmer_age_distribution: Distribution,

    // Economics
    pub farmer_income_parity: f64,
    pub food_dollar_to_farmer: f64,
    pub local_food_economy_size: Decimal,
}
```

---

## Conclusion

Provision coordinates the fundamental human need for nourishment, from seed to table. By integrating with land (Terroir), logistics (Transit, SupplyChain), economics (Marketplace, Treasury), and community (Commons, Beacon), it enables food sovereignty at every scale.

*"A community that cannot feed itself is not free. Provision is the infrastructure of that freedom."*

---

*Document Version: 1.0*
*Classification: Tier 2 - Essential*
*Dependencies: Terroir, SupplyChain, Marketplace, Transit, Beacon, Commons, Oracle*

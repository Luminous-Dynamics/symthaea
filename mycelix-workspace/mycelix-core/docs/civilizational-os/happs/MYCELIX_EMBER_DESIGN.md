# Mycelix-Ember: Energy Coordination & Grid Balancing

## Vision Statement

*"Energy is the lifeblood of civilization. Ember coordinates the generation, storage, distribution, and conservation of energy - enabling community energy sovereignty and accelerating the transition to renewables."*

---

## Executive Summary

Mycelix-Ember provides infrastructure for energy coordination:

1. **Distributed generation** - Community solar, wind, micro-hydro coordination
2. **Peer-to-peer energy trading** - Local energy markets and sharing
3. **Grid balancing** - Demand response, storage coordination
4. **Community microgrids** - Resilient local energy systems
5. **Energy efficiency** - Collective conservation and optimization

---

## Core Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    EMBER ARCHITECTURE                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  GENERATION LAYER                                                       │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Asset Registry │ Production Tracking │ Forecasting │ Dispatch  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                               │                                         │
│  STORAGE LAYER                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Battery Networks │ Thermal Storage │ V2G │ Community Storage   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                               │                                         │
│  TRADING LAYER                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  P2P Markets │ Community Pools │ Grid Export │ RECs             │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                               │                                         │
│  DEMAND LAYER                                                           │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Demand Response │ Load Shifting │ Efficiency │ Conservation    │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Key Data Types

```rust
/// Energy generation asset
pub struct EnergyAsset {
    pub asset_id: String,
    pub asset_type: EnergyAssetType,
    pub owner: AssetOwner,
    pub location: ActionHash,               // Terroir
    pub capacity: PowerCapacity,
    pub current_output: Option<Power>,
    pub status: AssetStatus,
    pub grid_connection: GridConnection,
    pub metadata: AssetMetadata,
}

pub enum EnergyAssetType {
    // Generation
    SolarPV { panel_count: u32, orientation: Orientation },
    SolarThermal,
    WindTurbine { rated_power: Power },
    MicroHydro { head: f64, flow: f64 },
    Biogas,
    Geothermal,
    CHP,                                    // Combined heat and power

    // Storage
    Battery { chemistry: BatteryChemistry, cycles: u32 },
    ThermalStorage,
    Flywheel,
    PumpedHydro,
    HydrogenStorage,
    EVBattery { vehicle_hash: ActionHash }, // V2G

    // Demand
    SmartLoad { load_type: LoadType },
    HVAC,
    WaterHeater,
    EVCharger,
}

pub enum AssetOwner {
    Individual(AgentPubKey),
    Household(ActionHash),
    Community(ActionHash),
    Cooperative(ActionHash),
    CommercialEntity(ActionHash),
}

/// Energy production/consumption record
pub struct EnergyRecord {
    pub record_id: String,
    pub asset_hash: ActionHash,
    pub timestamp: Timestamp,
    pub period: Duration,
    pub energy_type: EnergyType,
    pub amount: Energy,
    pub direction: EnergyDirection,
    pub quality: EnergyQuality,
    pub verified_by: Option<ActionHash>,    // Oracle/meter
}

pub enum EnergyDirection {
    Generation,
    Consumption,
    StorageCharge,
    StorageDischarge,
    GridExport,
    GridImport,
    P2PExport,
    P2PImport,
}

/// Peer-to-peer energy trade
pub struct EnergyTrade {
    pub trade_id: String,
    pub seller: AgentPubKey,
    pub buyer: AgentPubKey,
    pub energy_amount: Energy,
    pub price: Decimal,
    pub trade_type: TradeType,
    pub settlement_period: TimePeriod,
    pub contract: Option<ActionHash>,
    pub status: TradeStatus,
    pub actual_delivery: Option<Energy>,
}

pub enum TradeType {
    Spot,                                   // Immediate
    DayAhead,
    Bilateral { contract: ActionHash },
    CommunityPool,
    SurplusSharing,
}

/// Community microgrid
pub struct Microgrid {
    pub grid_id: String,
    pub name: String,
    pub boundary: GeoBoundary,
    pub members: Vec<MicrogridMember>,
    pub generation_assets: Vec<ActionHash>,
    pub storage_assets: Vec<ActionHash>,
    pub loads: Vec<ActionHash>,
    pub grid_connection: Option<GridConnection>,
    pub operating_mode: OperatingMode,
    pub governance: ActionHash,
    pub resilience_metrics: ResilienceMetrics,
}

pub struct MicrogridMember {
    pub agent: AgentPubKey,
    pub property: ActionHash,
    pub assets: Vec<ActionHash>,
    pub load_profile: LoadProfile,
    pub flexibility: FlexibilityProfile,
    pub share: MembershipShare,
}

pub enum OperatingMode {
    GridConnected,
    Islanded,
    GridForming,
    GridFollowing,
    EmergencyBackup,
}

/// Demand response event
pub struct DemandResponseEvent {
    pub event_id: String,
    pub event_type: DREventType,
    pub initiator: DRInitiator,
    pub target_area: Option<GeoBoundary>,
    pub target_reduction: Power,
    pub start_time: Timestamp,
    pub end_time: Timestamp,
    pub incentive: Option<DRIncentive>,
    pub participants: Vec<DRParticipant>,
    pub status: EventStatus,
}

pub enum DREventType {
    PeakShaving,
    FrequencyRegulation,
    EmergencyCurtailment,
    RenewableAbsorption,        // Increase demand for surplus renewables
    GridStabilization,
    EconomicDispatch,
}

pub enum DRInitiator {
    GridOperator,
    CommunityCoordinator,
    PriceSignal,
    AutomatedOptimization,
    Emergency,
}

/// Renewable energy certificate
pub struct RenewableCertificate {
    pub certificate_id: String,
    pub generation_asset: ActionHash,
    pub energy_amount: Energy,
    pub generation_period: TimePeriod,
    pub energy_source: RenewableSource,
    pub location: ActionHash,
    pub owner: AgentPubKey,
    pub status: CertificateStatus,
    pub retired_by: Option<AgentPubKey>,
    pub retirement_purpose: Option<String>,
}

pub enum RenewableSource {
    Solar,
    Wind,
    Hydro,
    Geothermal,
    Biomass,
    Ocean,
}
```

---

## Key Workflows

### Workflow 1: Community Solar Coordination

```
Community forms solar cooperative (Agora governance) →
Identifies suitable rooftops (Terroir integration) →
Collective financing via Treasury →
Installation coordinated →
Production tracked and verified (Oracle/meters) →
Energy distributed to members →
Surplus traded P2P or exported to grid →
Revenue shared proportionally
```

### Workflow 2: Peer-to-Peer Energy Trading

```
Prosumer has surplus solar generation →
Posts offer to local energy market →
Nearby consumer with demand matches →
Trade executed automatically →
Physical delivery via grid →
Settlement via Treasury →
RECs created for renewable verification
```

### Workflow 3: Demand Response

```
Grid stress detected (Oracle integration) →
Demand response event triggered →
Participants with flexible loads notified →
Smart devices automatically respond →
Load reduction verified →
Incentive payments distributed →
Grid stabilized
```

### Workflow 4: Microgrid Islanding

```
Main grid outage detected (Beacon integration) →
Microgrid transitions to island mode →
Critical loads prioritized →
Storage and generation balanced locally →
Community notified of reduced capacity →
Resilience mode until grid restored →
Seamless reconnection when safe
```

---

## Integration Points

| Integration | Purpose |
|-------------|---------|
| **Terroir** | Generation site locations, property integration |
| **Oracle** | Meter data, weather forecasting, grid prices |
| **Treasury** | Energy payments, cooperative financing |
| **Commons** | Shared energy infrastructure |
| **Covenant** | Power purchase agreements, cooperative agreements |
| **Agora** | Energy cooperative governance |
| **Beacon** | Emergency grid situations |
| **Accord** | Energy cost tracking, carbon accounting |
| **Transit** | EV charging coordination, V2G |
| **Marketplace** | Energy trading, equipment |
| **Mutual** | Equipment insurance pools |

---

## Energy Metrics

```rust
pub struct CommunityEnergyMetrics {
    // Generation
    pub total_renewable_capacity: Power,
    pub renewable_percentage: f64,
    pub local_generation_percentage: f64,
    pub capacity_factor: f64,

    // Storage
    pub storage_capacity: Energy,
    pub storage_duration: Duration,
    pub round_trip_efficiency: f64,

    // Consumption
    pub total_consumption: Energy,
    pub peak_demand: Power,
    pub demand_flexibility: f64,

    // Economics
    pub average_energy_cost: Decimal,
    pub energy_poverty_rate: f64,
    pub local_energy_spending: Decimal,

    // Resilience
    pub backup_duration: Duration,
    pub outage_events: u32,
    pub average_restoration_time: Duration,

    // Environment
    pub carbon_intensity: f64,              // kg CO2/kWh
    pub emissions_avoided: f64,
}
```

---

## Energy Justice Framework

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    ENERGY JUSTICE PRINCIPLES                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ACCESS                                                                 │
│  • Universal access to clean, affordable energy                        │
│  • No disconnections for inability to pay                              │
│  • Community solar access for renters                                  │
│  • Accessibility accommodations for disabled                           │
│                                                                         │
│  PARTICIPATION                                                          │
│  • Community ownership opportunities                                    │
│  • Democratic governance of energy cooperatives                        │
│  • Benefit sharing from local generation                               │
│  • Worker ownership in energy transition                               │
│                                                                         │
│  DISTRIBUTION                                                           │
│  • Progressive pricing (lifeline rates)                                │
│  • Pollution burden reduction in frontline communities                 │
│  • Priority resilience for vulnerable populations                      │
│  • Equitable transition support for fossil fuel workers               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Conclusion

Ember coordinates the energy infrastructure that powers civilization. By enabling community ownership, peer-to-peer trading, and coordinated demand response, it accelerates the transition to clean energy while ensuring energy access and resilience for all.

*"The sun shines on everyone. Ember ensures the benefits are shared by all."*

---

*Document Version: 1.0*
*Classification: Tier 2 - Essential*
*Dependencies: Terroir, Oracle, Treasury, Commons, Covenant, Agora, Beacon, Accord, Transit*

# Mycelix-Transit: Transportation & Mobility Coordination

## Vision Statement

*"Mobility is freedom. Transit coordinates the movement of people and goods - from shared bicycles to regional logistics - making transportation accessible, sustainable, and community-controlled."*

---

## Executive Summary

Mycelix-Transit provides infrastructure for mobility coordination:

1. **Shared mobility** - Car sharing, bike sharing, ride sharing
2. **Logistics coordination** - Freight consolidation, delivery networks
3. **Accessibility** - Accessible transportation coordination, companion matching
4. **Multi-modal integration** - Seamless journey planning across modes
5. **Community transport** - Community buses, shuttle networks, mutual aid rides

---

## Core Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    TRANSIT ARCHITECTURE                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  VEHICLE LAYER                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Vehicle Registry │ Availability │ Maintenance │ Insurance      │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                               │                                         │
│  SHARING LAYER                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Car Share │ Bike Share │ Ride Share │ Equipment Share         │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                               │                                         │
│  LOGISTICS LAYER                                                        │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Freight │ Last Mile │ Consolidation │ Route Optimization      │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                               │                                         │
│  COORDINATION LAYER                                                     │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Multi-modal │ Accessibility │ Community Transit │ Emergency    │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Key Data Types

```rust
/// Vehicle in the sharing network
pub struct Vehicle {
    pub vehicle_id: String,
    pub vehicle_type: VehicleType,
    pub owner: VehicleOwner,
    pub specifications: VehicleSpecs,
    pub location: Option<GeoCoordinates>,
    pub availability: AvailabilitySchedule,
    pub sharing_terms: SharingTerms,
    pub insurance: InsuranceInfo,
    pub maintenance_status: MaintenanceStatus,
}

pub enum VehicleType {
    // Personal
    Bicycle { electric: bool },
    Scooter { electric: bool },
    Motorcycle,
    Car { seats: u8, cargo_capacity: Volume },
    Van { cargo_capacity: Volume },
    Truck { cargo_capacity: Volume },

    // Specialized
    WheelchairAccessible,
    CargoTrike,
    HandCycle,
    Tandem,

    // Community
    Bus { capacity: u8 },
    Shuttle { capacity: u8 },
}

pub enum VehicleOwner {
    Individual(AgentPubKey),
    Cooperative(ActionHash),
    Community(ActionHash),
    Fleet(ActionHash),
}

/// Ride share trip
pub struct RideShare {
    pub trip_id: String,
    pub trip_type: RideShareType,
    pub driver: AgentPubKey,
    pub vehicle_hash: ActionHash,
    pub route: Route,
    pub departure_time: Timestamp,
    pub available_seats: u8,
    pub passengers: Vec<Passenger>,
    pub cost_sharing: CostSharing,
    pub accessibility: AccessibilityFeatures,
    pub preferences: DriverPreferences,
}

pub enum RideShareType {
    Commute { recurring: bool },
    LongDistance,
    Errand,
    Airport,
    Event,
    Emergency,
    MedicalAppointment,
}

/// Logistics shipment
pub struct Shipment {
    pub shipment_id: String,
    pub shipper: AgentPubKey,
    pub origin: Location,
    pub destination: Location,
    pub cargo: CargoDescription,
    pub pickup_window: TimeWindow,
    pub delivery_window: TimeWindow,
    pub special_handling: Vec<HandlingRequirement>,
    pub status: ShipmentStatus,
    pub carrier: Option<CarrierAssignment>,
}

pub struct CargoDescription {
    pub description: String,
    pub weight: Weight,
    pub dimensions: Dimensions,
    pub fragile: bool,
    pub temperature_sensitive: Option<TemperatureRange>,
    pub hazardous: bool,
    pub value: Option<Decimal>,
}

/// Community transport service
pub struct CommunityTransport {
    pub service_id: String,
    pub name: String,
    pub service_type: CommunityTransportType,
    pub operator: ActionHash,               // Governance
    pub vehicles: Vec<ActionHash>,
    pub service_area: GeoBoundary,
    pub schedule: TransportSchedule,
    pub eligibility: Option<EligibilityCriteria>,
    pub funding: FundingModel,
}

pub enum CommunityTransportType {
    FixedRoute,
    DemandResponse,
    MedicalTransport,
    SeniorShuttle,
    SchoolTransport,
    EventShuttle,
    CargoDelivery,
}

/// Multi-modal journey
pub struct Journey {
    pub journey_id: String,
    pub traveler: AgentPubKey,
    pub origin: Location,
    pub destination: Location,
    pub departure_preference: TimePreference,
    pub legs: Vec<JourneyLeg>,
    pub accessibility_requirements: AccessibilityFeatures,
    pub carbon_preference: CarbonPreference,
    pub status: JourneyStatus,
}

pub struct JourneyLeg {
    pub leg_id: String,
    pub mode: TransportMode,
    pub start_location: Location,
    pub end_location: Location,
    pub start_time: Timestamp,
    pub end_time: Timestamp,
    pub booking_reference: Option<String>,
    pub cost: Option<Decimal>,
    pub carbon_impact: Option<CarbonImpact>,
}

pub enum TransportMode {
    Walk,
    Bike { shared: bool },
    Scooter { shared: bool },
    Bus,
    Train,
    Subway,
    RideShare,
    CarShare,
    Taxi,
    Ferry,
    Flight,
    CommunityTransit,
}
```

---

## Key Workflows

### Workflow 1: Car Share Booking

```
Member searches for available vehicle →
Filters by type, location, time →
Reviews vehicle details and owner ratings →
Books via Covenant contract →
Treasury handles payment/deposit →
Pickup coordinated (key exchange or smart lock) →
Usage tracked →
Return and condition verification →
MATL reputation updated for both parties
```

### Workflow 2: Ride Share Matching

```
Driver posts upcoming trip with route and time →
Passengers search for matching trips →
System suggests riders based on route overlap →
Rider requests seat →
Driver approves →
Cost sharing calculated (fuel + maintenance) →
Trip completed →
Both parties rate each other
```

### Workflow 3: Freight Consolidation

```
Multiple shippers post small shipments →
System identifies consolidation opportunities →
Carrier picks up consolidated load →
Route optimized for multiple deliveries →
Real-time tracking via Oracle/IoT →
Proof of delivery recorded →
SupplyChain integration for full traceability
```

### Workflow 4: Accessible Transport Request

```
Person with mobility needs requests transport →
System matches with accessible vehicles/drivers →
Companion matcher if needed (Kinship integration) →
Special requirements confirmed →
Trip completed with accessibility support →
Feedback improves future matching
```

---

## Integration Points

| Integration | Purpose |
|-------------|---------|
| **Covenant** | Vehicle sharing agreements, ride contracts |
| **Treasury** | Payments, deposits, cost sharing |
| **MATL** | Driver/rider trust, vehicle owner reputation |
| **Attest** | Driver credentials, vehicle registration |
| **Oracle** | Real-time location, traffic, fuel prices |
| **SupplyChain** | Freight traceability |
| **Provision** | Food delivery coordination |
| **Kinship** | Care-related transportation, companions |
| **HealthVault** | Medical transport coordination |
| **Beacon** | Emergency transport |
| **Nexus** | Event transportation |
| **Commons** | Shared vehicle pools |
| **Mutual** | Vehicle insurance pools |
| **Accord** | Transportation expense tracking |

---

## Sustainability Metrics

```rust
pub struct TransitSustainability {
    // Usage
    pub vehicle_utilization_rate: f64,
    pub average_occupancy: f64,
    pub trips_shared_percentage: f64,

    // Environment
    pub carbon_per_passenger_km: f64,
    pub ev_percentage: f64,
    pub active_transport_percentage: f64,  // Walk/bike

    // Access
    pub households_with_access: f64,
    pub accessible_trips_served: u64,
    pub average_wait_time: Duration,

    // Economics
    pub cost_per_km: Decimal,
    pub community_transport_coverage: f64,
}
```

---

## Conclusion

Transit coordinates the movement of people and goods in ways that are accessible, sustainable, and community-controlled. By integrating vehicle sharing, ride matching, logistics, and multi-modal journey planning, it reduces car dependency while expanding mobility for all.

*"The best vehicle is the one we share. Transit makes sharing the default."*

---

*Document Version: 1.0*
*Classification: Tier 2 - Essential*
*Dependencies: Covenant, Treasury, MATL, Attest, Oracle, SupplyChain, Commons, Mutual*

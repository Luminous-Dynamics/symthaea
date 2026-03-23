# Circular Economy: Automated Recycling & Composting

Automated waste tracking, composting process control, carbon attribution, and secondary materials marketplace — spanning Mycelix (Holochain DHT) and Symthaea (consciousness-first AI).

## Architecture Overview

```
Physical World                    Mycelix DHT                         Symthaea AI
─────────────────────────────────────────────────────────────────────────────────
IoT Sensors ──────┐
                  ├──→ resource-mesh ──→ compost-control ──→ evaluate_batch_status()
Camera Feed ──────┤         (SensorReading)    (CompostReading)     ↓
                  │                              ↓             recommended_actions
                  │                        carbon_attribution ──→ CarbonCredit
                  │                              ↓               (climate cluster)
Waste Source ─────┤
                  ├──→ waste-registry ──→ classify ──→ route ──→ waste-collection
                  │    (WasteStream)    (WasteClassification)   (CollectionRequest)
                  │                          ↑                       ↓
                  │                   WasteBridgeEvent         CollectionRun
                  │                          ↑                       ↓
                  │                   WasteHdcEncoder          plan_optimized_route()
                  │                   (16,384D HDC)                  ↓
                  │                          ↑               confirm_delivery()
                  │                   WasteManager                   ↓
                  │                   (interval 47)          compost-control
                  │                          ↑               create_compost_batch()
                  │                   SwarmEvent::                   ↓
                  │                   WasteCircularityUpdate   optimize_recipe()
                  │                                                  ↓
Farm/Garden ──────┘                                          circular-marketplace
                                                             list_secondary_material()
                  ←────────────────────────────────────────  find_matching_demand()
                  (finished compost returned to farms)       find_plots_needing_compost()
```

## Mycelix Zomes (8 total, Commons cluster)

### waste-registry (coordinator + integrity)

Tracks waste streams from registration through processing completion.

**Entry Types:**
- `WasteStream` — source, category (6 variants), quantity_kg, contamination_level, recommended end-of-life strategy, GPS location, lifecycle status
- `WasteClassification` — AI or human classification result with confidence score and method
- `WasteFacility` — processing facility with type, accepted categories, capacity, location
- `WasteRoute` — matching result linking stream to facility via multi-factor scoring

**Key Functions:**
| Function | Gate | Description |
|----------|------|-------------|
| `register_waste_stream` | Participant+ | Register a new waste stream |
| `classify_waste_stream` | Participant+ | Add classification (Manual/SensorBased/VisionAI/MaterialPassport) |
| `route_waste_stream` | Participant+ | Match stream to best facility (category × proximity × capacity × contamination) |
| `register_facility` | Proposal+ | Register a processing facility |
| `check_contamination_feedback` | Participant+ | Compare classification vs original, alert source on escalation |
| `calculate_diversion_rate` | — | Aggregate composted/recycled/landfilled kg for a period |
| `get_volume_trend` | — | Time-bucketed waste volumes (sparkline-compatible) |
| `get_category_distribution` | — | Breakdown by WasteCategory with percentages |

**Routing Algorithm** (`route_waste_stream`):
- Category match is a hard gate (0 or 1)
- Proximity: `1 - (haversine_km / 100km)` clamped to [0, 1]
- Capacity: `available / total` clamped to [0, 1]
- Contamination: Clean=1.0, Light=0.8, Contaminated=0.6 (MRF/ChemRecycler only), Hazardous=0.4 (ChemRecycler only)
- Weights: 40% proximity + 35% capacity + 25% contamination
- Threshold: score >= 0.3

### waste-collection (coordinator + integrity)

Pickup scheduling with capacity-aware route optimization.

**Entry Types:**
- `CollectionRequest` — stream link, pickup location, time window, quantity, status lifecycle
- `CollectionRun` — vehicle, driver, ordered stops, completion tracking

**Key Functions:**
| Function | Description |
|----------|-------------|
| `request_collection` | Create pickup request (consciousness-gated) |
| `plan_optimized_route` | Nearest-insertion heuristic with capacity constraints |
| `create_collection_run` | Assign stops to vehicle run |
| `confirm_delivery` | Record actual kg at each stop, auto-complete run |

**Route Optimizer** (`plan_optimized_route`):
- Starts at facility, greedily adds nearest unvisited request fitting capacity
- Excludes requests exceeding vehicle_capacity_kg
- Max 50 stops per run
- Returns total_distance_km, cumulative_kg per stop, savings_pct vs naive ordering

### compost-control (coordinator + integrity)

Sensor-driven composting automation with EPA-cited thresholds.

**Entry Types:**
- `CompostBatch` — facility link, method (5 variants), inputs with resource types, C:N ratio, status lifecycle, phase history
- `CompostReading` — temperature, moisture, oxygen, pH from sensors
- `CompostAction` — recommended or executed actions (Turn/AddWater/AddBulking/AdjustAeration/HarvestScreen)

**Key Functions:**
| Function | Description |
|----------|-------------|
| `create_compost_batch` | Start tracking a new batch |
| `record_compost_reading` | Log sensor data |
| `record_sensor_bridge_reading` | Pull from resource-mesh via `CallTargetCell::Local` |
| `evaluate_batch_status` | **Control loop**: compare readings to thresholds, recommend actions |
| `optimize_recipe` | Suggest amendments to reach target C:N ratio |
| `get_batch_nutrient_estimate` | Weighted NPK from input resource types |
| `calculate_carbon_attribution` | EPA WARM: 0.06 tCO2e/tonne composted |
| `get_facility_diversion_summary` | Aggregate completed batches and carbon savings |

**Composting Thresholds** (named constants, science-cited):
| Parameter | Range | Source |
|-----------|-------|--------|
| Thermophilic temp | 55-65 C | EPA 40 CFR 503 |
| Mesophilic temp | 25-40 C | Standard composting science |
| Moisture | 40-65% | USCC guidelines |
| Optimal moisture | 55% | Maximum microbial activity |
| Oxygen minimum | 5% | Aerobic decomposition threshold |
| C:N ratio | 25:1 - 30:1 | Cornell Waste Management Inst. |
| pH range | 5.5 - 8.5 | Healthy composting range |

**Recipe Optimizer** (`optimize_recipe`):
- Calculates weighted-average C:N from batch inputs
- If nitrogen-heavy (C:N < 25): suggests carbon-rich amendments (Woodchips 400:1, Straw 80:1, Cardboard 350:1)
- If carbon-heavy (C:N > 30): suggests nitrogen-rich amendments (KitchenWaste 15:1, Manure 18:1, Digestate 10:1)
- Solves: `x = (target × total_kg - weighted_cn) / (amendment_cn - target)` for exact quantities
- Sorts by minimum quantity needed

**Sensor Bridge** (`record_sensor_bridge_reading`):
- Calls `resource_mesh::get_resource_status` via `CallTargetCell::Local`
- Maps: temperature → temperature_c, humidity → moisture_pct, air_quality → oxygen_pct
- Graceful degradation: returns None for missing sensors, uses defaults

### circular-marketplace (coordinator + integrity)

Secondary materials trading to close the loop.

**Entry Types:**
- `SecondaryMaterialListing` — facility, material type (9 variants), quality grade, nutrient info, price, location
- `SecondaryMaterialOrder` — listing link, buyer, quantity, delivery route, status

**Key Functions:**
| Function | Description |
|----------|-------------|
| `list_secondary_material` | Post finished compost/recyclables for sale |
| `find_matching_listings` | Multi-factor demand matching (proximity 50%, quality 30%, surplus 20%) |
| `find_plots_needing_compost` | Cross-zome call to `food_production::get_all_plots` via Local |
| `place_order` | Create order (consciousness-gated) |

## Symthaea AI Layer

### symthaea-circular (standalone crate, 47 tests)

| Module | Purpose |
|--------|---------|
| `waste_encoder` | 12-basis WasteHdcEncoder → 16,384D vectors, WasteDatabase with 18 categories |
| `decomposition_predictor` | O(1) sigmoid decay with Q10 temperature + Gaussian moisture modifiers |
| `contamination_detector` | HDC outlier detection (similarity < 0.7 threshold) |
| `circular_fep` | FEP action selection: ClassifyWaste/AlertContamination/RecommendRoute/PredictDecomposition |

### WasteManager (cognitive subsystem, interval 47)

Feature: `circular`

Processes `WasteEvent` variants:
- `ClassificationReceived` → confidence EMA (alpha=0.15)
- `ContaminationDetected` → safety level escalation (Green→Yellow→Orange→Red)
- `DecompositionUpdate` → urgency tracking with decay
- `CircularityReport` → running mean circularity

Cognitive outputs:
- Classification confidence > 0.7 → confidence boost
- Contamination → arousal spike (Arnsten 2009)
- Safety Orange → NE boost, Red → NE + DA suppression (Sapolsky 2015)
- Decomposition urgency > 0.6 → exploration drive
- High circularity > 0.7 → positive valence (McEwen 2007)

### WasteBridgeEvent (MycelixBridge)

Feature: `circular`

- `ClassificationResult` — AI classification → DHT WasteClassification entry
- `ContaminationAlert` — contamination detection → DHT alert + source notification
- `DecompositionPrediction` — predicted completion → batch planning

### SwarmManager Integration

Feature: `circular`

- `SwarmEvent::WasteCircularityUpdate` — distributed waste intelligence sharing
- `SwarmTelemetry`: waste_total_kg, waste_mean_circularity, waste_events_processed, waste_confidence_ema
- Neuromod coupling: high circularity → serotonin, low material entropy → NE vigilance

### Pulse Dashboard

`CircularEconomyInfo` pane: total_waste_kg, mean_circularity, events_processed, confidence_ema

## Bridge Routing

`BridgeDomain::Waste` resolves to:

| Query keyword | Zome |
|---------------|------|
| collection, pickup, run, vehicle | `WasteCollection` |
| compost, batch, reading, nutrient | `CompostControl` |
| marketplace, listing, order, demand, secondary | `CircularMarketplace` |
| (default) | `WasteRegistry` |

Cross-cluster calls:
- Waste → Climate: `CallTargetCell::OtherRole("climate")` for carbon credit minting
- Waste → SupplyChain: `CallTargetCell::OtherRole("supplychain")` for MaterialPassport lookup

## SDKs

### Rust (`mycelix-workspace/sdk/src/circular/mod.rs`)
All entry type mirrors with serde roundtrip tests. Import: `use mycelix_sdk::circular::*;`

### TypeScript (`mycelix-workspace/sdk-ts/src/integrations/circular/index.ts`)
35 type exports + 9 composting threshold constants. Import: `import { WasteStream, THERMOPHILIC_MIN_C } from '@mycelix/sdk';`

## DNA Bundle

`mycelix-commons/dna/dna.yaml` — 45 coordinator zomes total (41 original + 4 waste/circular).

WASM binaries: 1.4-2.5 MB each, compiled to `wasm32-unknown-unknown` release profile.

## Test Coverage

| Package | Tests | Type |
|---------|-------|------|
| waste-registry integrity | 26 | unit + proptest |
| waste-registry coordinator | 14 | unit + analytics |
| waste-collection integrity | 13 | unit |
| waste-collection coordinator | 11 | unit + route |
| compost-control integrity | 17 | unit |
| compost-control coordinator | 35 | unit + proptest + recipe |
| circular-marketplace integrity | 12 | unit |
| circular-marketplace coordinator | 8 | unit + proptest |
| symthaea-circular | 47 | unit |
| sweettest integration | 10 | conductor (ignored) |
| bridge-common routing | 380 | unit (all green) |
| **Total** | **~200** | |

## The Full Loop

```
1. Farm generates crop residue (200 kg)
   → register_waste_stream(category: Organic, eol: IndustrialCompost)

2. Symthaea classifies via HDC vision
   → classify_waste_stream(method: VisionAI, confidence: 0.92)

3. Route to nearest composting facility
   → route_waste_stream() → score 0.85 match

4. Schedule collection with route optimization
   → plan_optimized_route(capacity: 1000kg) → 3 stops, 12km total

5. Deliver to facility
   → confirm_delivery(actual_kg: 195)

6. Create compost batch
   → create_compost_batch(method: Windrow, cn_ratio: 26)

7. Recipe optimizer suggests amendment
   → optimize_recipe() → "Add 45kg Woodchips to reach 27.5 C:N"

8. Sensors monitor composting process
   → record_sensor_bridge_reading() → temp: 58C, moisture: 55%, O2: 15%
   → evaluate_batch_status() → "Thermophilic OK, no actions needed"

9. Batch completes, carbon credit generated
   → calculate_carbon_attribution() → 0.012 tCO2e avoided

10. List finished compost on marketplace
    → list_secondary_material(type: Compost, quality: Standard, 150kg)

11. Match to farm needing nutrients
    → find_plots_needing_compost() → 3 plots matched

12. Order and deliver back to farm
    → place_order(quantity: 150kg)
    → Closed loop complete.
```

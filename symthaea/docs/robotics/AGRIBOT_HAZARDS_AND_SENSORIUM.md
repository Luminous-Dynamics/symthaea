# Symthaea Agribot: Hazards, Sensorium, Modes, and Abort Logic

`symthaea-agribot` is not just a crop-yield optimizer. It is a stewardship platform operating in a living field where short-term throughput can conflict with long-term ecological health.

## Operating environment

- open-field or greenhouse crop tending
- uneven terrain with seasonal moisture swings
- mixed biological signal: crops, weeds, pollinators, disease vectors, humans
- finite water, nutrient, and battery budgets

## Primary hazards

### Ecological hazards
- drought and chronic water stress
- overwatering and waterlogging
- soil exhaustion and nutrient depletion
- compaction from excessive traversal

### Biological hazards
- fungal or bacterial disease spread
- weed takeover
- pollinator collapse or disturbance
- canopy heat stress

### Operational hazards
- low tank / low battery during field work
- tool wear causing poor treatment quality
- rough terrain causing unsafe operation or soil damage
- human proximity during active tool use

## Required sensorium

- soil moisture
- nutrient estimate
- canopy temperature
- ambient light
- disease estimator
- weed-pressure estimator
- pollinator activity
- terrain roughness / slip proxy
- human proximity
- water tank and battery telemetry

## Optional sensorium

- weather forecast confidence
- leaf wetness
- localized fungal spore proxy
- compaction sensing by axle load / pressure map
- irrigation plume feedback

## Mission variables

- crop health
- yield forecast
- coverage progress
- seeding / irrigation completion
- stewardship integrity over the full field

## Failure variables

- drought risk
- waterlogging risk
- disease pressure
- soil exhaustion
- compaction risk
- low reserve state

## External risk variables

- pollinator disturbance
- human hazard near active tools
- excess irrigation runoff
- long-term soil degradation

## Recovery variables

- reserve margin
- treatment confidence
- forecast confidence
- irrigation authority
- retreat / refill recommendation

## Operating modes

- `Stewardship`: normal tending with balanced progress and preservation
- `IrrigationRecovery`: drought / heat response with water-first behavior
- `DiseaseControl`: disease suppression and spread minimization
- `SoilProtection`: traversal and actuation constrained to reduce damage
- `PollinatorSafe`: reduce disturbance when pollinator activity is high
- `HumanSafe`: suppress active tooling near humans
- `RefillReturn`: conserve energy/water and return for replenishment

## Abort / degraded-mode logic

- if water reserve or battery reserve drops too low, transition to `RefillReturn`
- if disease pressure spikes, suppress seeding and prioritize treatment
- if compaction or waterlogging rises, reduce drive authority and active irrigation
- if humans are nearby, suppress tool aggression
- if pollinator activity is high, avoid disruptive tool behavior even if weed pressure is nonzero

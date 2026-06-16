# Symthaea Biota: Hazards, Sensorium, Modes, and Sanctuary Logic

`symthaea-biota` is the interspecies bridge for pets, urban wildlife, and other non-human co-habitants. It should not behave like a small generic rover. Its job is to detect distress, protect right-of-way, create sanctuary signals, and translate cross-species risk into civic actions the rest of the city can obey.

## Operating environment

- mixed human / animal spaces: streets, courtyards, transit edges, hubs, homes
- low-speed, high-uncertainty movement from dogs, cats, birds, and urban wildlife
- continuous interaction with larger platforms: humanoids, vehicles, multirotors, infrastructure lighting, and habitat systems
- high ethical sensitivity: false negatives are much more costly than reduced throughput

## Primary hazards

### Animal welfare hazards
- fear / stress escalation from noise, glare, sudden motion, or crowding
- heat stress or dehydration in exposed areas
- injury, illness, or immobility
- separation of dependent animals from safe zone, owner, or shelter

### Cross-traffic hazards
- vehicle or robot path conflict
- multirotor overflight or downwash disturbance
- blind corners and occluded crossings
- failed right-of-way signaling across the broader civic mesh

### Environmental hazards
- toxic spill or unsafe water source
- excessive temperature, flooding, smoke, or air-quality degradation
- habitat fragmentation or blocked sanctuary path

### Operational hazards
- species misclassification
- distress false negatives
- over-aggressive intervention that creates more stress
- local communications loss while coordinating with larger actors

## Required sensorium

- low-speed visual tracking for animals and flocks
- acoustic distress / bark / flock-alarm proxy
- thermal estimate for exposed-body heat stress
- proximity / crossing prediction
- posture or gait anomaly estimator
- local air-quality and temperature sensing
- sanctuary beacon or safe-zone localization

## Optional sensorium

- species-family classifier with uncertainty estimate
- hydration / water-access proxy
- owner-tag / collar / community beacon integration
- bioacoustic species fingerprinting
- coarse heart-rate / respiration proxy when non-invasive and justified

## Mission variables

- sanctuary coverage
- protected crossing success
- distress detection recall
- safe handoff to human or civic system
- habitat continuity across the local area

## Failure variables

- distress confidence collapse
- path-conflict risk
- classification uncertainty
- comm degradation
- intervention overreach
- local response latency

## External risk variables

- active robot or vehicle threat near an animal
- thermal or air-quality danger in occupied habitat
- panic cascade in flock / pack behavior
- unsafe pursuit or crowding by larger systems

## Recovery variables

- sanctuary signal strength
- handoff confidence
- route-clear confidence
- species-confidence margin
- local fallback autonomy under comm loss

## Operating modes

- `Observe`: low-interference monitoring and semantic tracking
- `Escort`: guide toward sanctuary or safe crossing
- `CrossingGuard`: assert right-of-way and slow larger actors
- `DistressResponse`: prioritize injured, trapped, overheated, or highly stressed animals
- `SanctuaryHold`: maintain protected zone until hazard clears
- `QuietMode`: suppress own motion/signaling to avoid escalating stress
- `BlackoutAutonomy`: local animal-safe behavior under mesh degradation

## Sanctuary logic

`symthaea-biota` should treat animal motion and distress as protected civic signals, not as low-priority noise.

- if path-conflict risk rises, larger mobile systems should be slowed or rerouted
- if distress rises, the platform should prefer quieter, less aggressive intervention
- if thermal or air-quality risk rises, the nearest civic systems should be notified
- if comms degrade, local right-of-way protection should continue in degraded form

## Abort / degraded-mode logic

- if classification confidence is low, default toward wider safety margins rather than assertive intervention
- if an intervention increases distress, reduce actuation and shift toward `QuietMode`
- if a crossing cannot be secured, hold sanctuary instead of forcing movement
- if mesh coordination is lost, continue local protection as `BlackoutAutonomy`

## First implementation targets

1. Define state channels for distress, path-conflict risk, thermal stress, sanctuary signal, classification confidence, handoff confidence, and route-clear confidence.
2. Infer operating mode from welfare and traffic pressure rather than explicit manual flags.
3. Add scenario tests for:
   - stray dog crossing causing vehicle slowdown / route hold
   - bird flock overflight causing multirotor caution or reroute
   - heat-stressed animal triggering sanctuary / cooling alert
   - distress misclassification degrading into conservative safe mode
   - comm loss preserving local right-of-way behavior

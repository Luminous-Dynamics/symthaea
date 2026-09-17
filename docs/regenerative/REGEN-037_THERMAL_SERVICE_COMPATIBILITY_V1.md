# REGEN-037 — Pyrolysis Heat / Thermal-Service Compatibility Bridge v1

Status: preregistration / architecture boundary only

Parent: REGEN-036 water-balance shadow model

Program: Luminous-Dynamics/mycelix#940

Roadmap label: "pyrolysis-heat / resource-quality bridge"

## Purpose

Define the first exact boundary between a candidate heat output from regenerative
process accounting and a candidate thermal service, without inventing a generic
resource-quality score or treating heat production as useful delivered energy.

The current Symthaea tree does not expose a canonical `ResourceQuality` or
`ResourceGrade` API. REGEN-037 therefore keeps "resource-quality" only as a
roadmap phrase and freezes a more explicit relation:

```text
thermal source candidate
+ thermal sink requirement
+ time / place / transfer assumptions
= compatibility assessment
```

The assessment remains model evidence.

## Governing theorem

```text
identified source heat
+ identified temperature level
+ identified duration / power profile
+ identified sink requirement
+ explicit transfer path / loss assumptions
+ validated thermodynamic primitives
= bounded thermal-service compatibility result
```

not:

```text
useful heat delivered
safe heat integration
process authorization
economic value
carbon benefit
physical actuation
```

## Core semantic firewall

```text
heat generated
!= heat recoverable
!= heat transferable
!= heat compatible with sink
!= heat delivered
!= thermal service achieved
```

Likewise:

```text
energy quantity
!= temperature usefulness
!= power availability
!= temporal match
!= spatial match
```

A single scalar "heat quality" is intentionally insufficient.

## REGEN-032 relationship

REGEN-032 may emit a candidate heat / energy ledger.

That ledger is upstream accounting.

REGEN-037 may consume only an explicitly identified thermal-source candidate
derived from REGEN-032.

```text
pyrolysis accounting heat term
!= measured recoverable heat
!= qualified heat source
```

If the source term is scenario-assumed, the compatibility result must preserve
that epistemic class.

## Existing thermofluids substrate

Current `symthaea-thermofluids::thermal` provides compact textbook primitives:

- Carnot efficiency;
- Fourier conduction heat rate;
- Newton convection heat rate;
- heat-engine work from supplied efficiency.

These are useful arithmetic oracles.

They are not a process-integration framework.

They currently accept raw floating-point inputs rather than validated physical
domain types.

REGEN-037 must not make them load-bearing on unvalidated external values.

## Validated primitive boundary

The first executable bridge should validate every physical input before invoking
the textbook formula.

Conceptually:

```text
PositiveKelvin
PositiveArea
PositiveThickness
NonNegativeConductivity
NonNegativeTransferCoefficient
FiniteEnergy
FinitePower
PositiveDuration
```

Invalid, non-finite, zero-denominator, or physically incompatible inputs must
fail explicitly.

For a Carnot calculation:

```text
0 < T_cold < T_hot
```

must be established before evaluating:

```text
eta_carnot = 1 - T_cold / T_hot
```

## Carnot firewall

Carnot efficiency is an ideal upper bound for conversion of heat to work between
two reservoirs.

It is not:

- a direct-use heat-efficiency metric;
- a measured engine efficiency;
- recoverable-work evidence;
- a universal thermal-quality score.

Normative:

```text
Carnot bound
!= realizable efficiency
!= delivered work
```

A low-temperature heat source may be valuable for a compatible direct thermal
sink even when its ideal work-conversion potential is low.

Conversely, a high-temperature source is not automatically useful if no
compatible sink exists.

## Source identity

A thermal-source candidate should bind at least:

- source subject identity;
- originating process / accounting receipt;
- thermal model identity;
- source temperature or temperature trajectory;
- heat / energy quantity representation;
- power / rate representation where known;
- availability interval;
- spatial location / interface identity;
- evidence / assumption class;
- uncertainty or unresolved fields.

No field may silently appear from the word "pyrolysis".

## Sink identity

A thermal sink should independently declare:

- sink identity;
- required thermal service;
- acceptable source / delivery temperature range;
- required power or energy over time;
- timing / duty cycle;
- spatial / interface location;
- transfer constraints;
- qualification / authority reference if real infrastructure is involved.

REGEN-037 does not invent sink requirements.

## Compatibility is relational

A source is not globally "high quality" or "low quality".

Compatibility depends on both source and sink.

Conceptually:

```text
ThermalCompatibility {
    source_id,
    sink_id,
    temperature_compatible,
    power_compatible,
    duration_compatible,
    timing_compatible,
    transfer_path_defined,
    modeled_transfer_losses,
    unresolved_constraints,
    disposition,
}
```

A useful disposition vocabulary may include:

```text
CompatibleUnderDeclaredAssumptions
IncompatibleTemperature
IncompatiblePower
IncompatibleTiming
IncompatibleDuration
TransferPathUndefined
InsufficientEvidence
UnsupportedModel
```

It should not include generic `GoodHeat` or `BadHeat`.

## Temperature compatibility

The bridge must distinguish:

- source temperature;
- sink inlet requirement;
- sink return / rejection temperature where relevant;
- ambient / cold-side temperature;
- temperature drop across transfer path.

```text
source temperature > sink minimum
```

may be necessary for one transfer mechanism but is not by itself proof of useful
delivery.

The chosen transfer model must be explicit.

## Quantity vs rate

A source can have enough total energy but insufficient power.

A source can have enough instantaneous power but insufficient duration.

Normative:

```text
energy-compatible
!= power-compatible
!= duration-compatible
```

The bridge must not collapse joules, watts, and time into one unlabeled number.

## Temporal matching

Source and sink availability must overlap explicitly.

A daily integrated heat quantity does not prove that heat is available during a
sink's required interval.

Any storage buffer must be represented as its own component with explicit
capacity, loss, timing, and evidence assumptions.

Storage may not be silently assumed.

## Spatial and interface matching

Heat produced at one process boundary is not delivered at another.

A transfer path must identify relevant assumptions such as:

- distance / geometry;
- conduction / convection interface;
- transport fluid or solid path if modeled;
- insulation / ambient losses;
- heat-exchanger assumptions;
- pump / fan / circulation requirements where applicable.

REGEN-037 v1 should remain simpler than a plant model and mark absent transfer
details unresolved.

## Transfer losses

Losses must remain explicit.

```text
generated heat
= delivered heat
+ transfer loss
+ storage loss
+ unresolved residual
```

only when every term belongs to the same declared accounting boundary.

Unknown loss is not zero loss.

## Direct thermal service vs work conversion

Two service paths are distinct:

```text
heat source -> direct thermal sink
```

and:

```text
heat source -> heat engine -> work / electricity
```

REGEN-037 must not compare them through one generic efficiency.

The direct path depends on sink temperature / transfer compatibility.

The work path requires a separately identified conversion device and its own
real efficiency / operating model; Carnot only supplies an ideal upper bound.

## Resource identity and double counting

The same heat quantity must not be allocated simultaneously to multiple sinks
without explicit partitioning.

A candidate thermal allocation should conserve the source quantity.

Conceptually:

```text
available source energy
=
sum(candidate delivered allocations)
+ modeled losses
+ unallocated reserve
+ unresolved residual
```

Scenario allocation is still not physical reservation or delivery.

## Authority firewall

REGEN-037 has no authority to:

- open valves;
- start pumps;
- change pyrolysis operation;
- energize heaters;
- connect thermal loops;
- dispatch electricity;
- commit a contractual energy sale;
- override equipment protection;
- bypass local operating envelopes.

```text
thermal compatibility
!= operating authority
```

## Safety firewall

A temperature-compatible source may still be unsafe because of:

- pressure;
- contamination;
- material incompatibility;
- isolation failure;
- equipment limits;
- process hazards;
- maintenance state;
- local operating conditions.

REGEN-037 does not infer safety from thermodynamic compatibility.

## Economic and climate firewalls

```text
recoverable heat
!= economically recoverable heat
!= profitable heat
!= avoided emissions
!= carbon credit
```

Economic valuation belongs in later economic models.

Climate attribution belongs in the Climate / MRV line.

## First executable candidate

The smallest valid implementation should:

1. define validated Kelvin / energy / power / duration wrappers;
2. define exact source / sink identities;
3. define one direct-heat compatibility case with synthetic fixtures;
4. validate all physical domains before invoking thermofluids helpers;
5. preserve source and sink timing;
6. compute one transparent transfer-rate oracle where applicable;
7. preserve transfer losses / unresolved loss explicitly;
8. reject an incompatible-temperature case;
9. reject an energy-sufficient but power-insufficient case;
10. prove a null transfer path cannot become delivered thermal service.

Do not begin with plant optimization or automatic dispatch.

## Independent arithmetic oracles

The existing textbook formulas may serve as narrow arithmetic oracles when
their preconditions are satisfied.

For example:

```text
Fourier conduction:
q = k A ΔT / L
```

and:

```text
Newton convection:
q = h A ΔT
```

The oracle proves arithmetic consistency under the declared model.

It does not validate the selected coefficient, geometry, boundary condition, or
real plant.

## Adversarial fixtures

Qualification should include:

- non-positive absolute temperature;
- cold side hotter than hot side for Carnot use;
- zero thickness in conduction;
- non-finite coefficient;
- energy without duration presented as power;
- power without duration presented as energy service;
- source and sink intervals that do not overlap;
- source and sink at different locations with no transfer path;
- unknown transfer loss silently set to zero;
- same source energy allocated twice;
- Carnot efficiency presented as actual efficiency;
- generated pyrolysis heat presented as measured recoverable heat;
- compatible heat presented as equipment authorization;
- compatible heat presented as carbon benefit.

Each shortcut must reject or remain explicitly unresolved.

## Relationship to REGEN-039

REGEN-039 settlement metabolism may compose REGEN-037 only through exact
identity-bearing thermal source / sink / allocation records.

It must not recreate a second heat-quality heuristic.

## Relationship to Phase F

Experiment intelligence may propose measurements that reduce uncertainty in a
source / sink compatibility assessment.

```text
high information value
!= permission to connect thermal equipment
```

## Promotion gate

An executable REGEN-037 candidate should not be called qualified until its exact
subject demonstrates:

- frozen Rust / toolchain lineage;
- compile / test / strict lint;
- deterministic replay;
- validated physical domains;
- exact source / sink / model identities;
- dimensional consistency;
- quantity / power / duration separation;
- explicit loss / residual handling;
- double-allocation rejection;
- textbook oracle parity under declared assumptions;
- no equipment-control output;
- postflight immutability.

## Deliberate non-claims

REGEN-037 does not establish:

- pyrolysis heat yield;
- measured recoverable heat;
- heat-exchanger sizing;
- pipe / duct design;
- equipment safety;
- process operating setpoints;
- actual conversion efficiency;
- economic feasibility;
- avoided emissions;
- carbon-credit eligibility;
- plant authorization;
- autonomous thermal control.

It establishes only a typed, conservation-aware thermal-service compatibility
boundary.

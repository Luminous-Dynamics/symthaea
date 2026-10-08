# Creditism Issuance-Capacity and Access-Coverage Benchmark

## Purpose

Test how a system that promises baseline and contribution Credit constrains issuance relative to real productive and service capacity.

Current Common Planet material says Credit is issued because a person exists and through contribution, while its public site also says Credit arrives only as the network can hold it. The architecture states that price pressure depends on demand relative to real capacity, inventories, scarcity, and ecological limits, and lists miscalibrated issuance as an explicit failure condition.

Sources:
- https://common-planet.org/creditism
- https://common-planet.org/creditism/architecture

## 1. Separate entitlement from capacity

Do not assume:

baseline entitlement == physical backing

or:

contribution recognition == newly created productive capacity.

The model must track:

- nominal Credit issuance entitlement;
- actual issued Credit;
- physical production capacity;
- essential-service capacity;
- inventories;
- constrained inputs;
- ecological/resource limits;
- import capacity;
- unmet demand.

## 2. Capacity coverage

Define a profile-relative coverage vector rather than a universal backing ratio.

Possible coordinates:

- essential food coverage;
- housing capacity coverage;
- energy coverage;
- healthcare capacity;
- transport capacity;
- productive machinery capacity;
- imported strategic-input coverage.

Do not collapse these into a single backing score.

## 3. Issuance-capacity stress

Fixtures:

1. baseline issuance increases with stable capacity;
2. baseline issuance increases faster than capacity;
3. contribution issuance increases with output;
4. contribution issuance increases without equivalent capacity increase;
5. population increases faster than capacity;
6. productivity boom;
7. energy shock;
8. ecological constraint tightens;
9. imported strategic input is unavailable;
10. synchronized saved-Credit release.

Measure:

- Credit stock;
- issuance/deletion flows;
- demand pressure;
- essential access;
- price/rationing;
- capacity utilization;
- inventories;
- production response;
- queue/waiting where applicable;
- distributional incidence.

## 4. No magical backing

Credit need not be treated as reserve-backed money merely because the system is capacity-constrained.

The simulator must distinguish:

accounting validity
from
physical capacity sufficiency
from
legal/institutional authority.

A Credit balance can be perfectly accounted for while essential goods remain unavailable.

That is a capacity outcome, not necessarily an accounting defect.

## 5. Calibration controller

If an issuance controller exists, its inputs and authority must be declared:

IssuanceController(
  population,
  baseline_rule,
  contribution_rule,
  capacity_state,
  inventory_state,
  ecological_state,
  external_state,
  observation_window,
  policy_parameters
) -> issuance decision

The controller must not silently consume future observations.

## 6. Positive controls

Include worlds where:

- high Credit issuance coincides with rapid productive expansion;
- higher demand is met by available inventories;
- capacity grows faster than Credit;
- contribution recognition increases because physical output genuinely expands;
- external imports remain available.

Do not encode low issuance as inherently safer.

## 7. Access-versus-aggregate test

Two worlds can have identical total Credit and identical total capacity but differ in where capacity and balances reside.

Compare:

- geographic mismatch;
- skill bottlenecks;
- energy bottlenecks;
- household distribution;
- regional inventory;
- transport constraints.

Required distinction:

aggregate capacity != accessible capacity

and:

aggregate Credit != locally effective purchasing capacity.

## 8. Ecological capacity

Where the architecture constrains production by ecological limits, preserve the physical measurement domain.

Do not convert:

ecological threshold -> universal Credit cap

without a declared causal/controller profile.

A carbon, water, biodiversity, or material constraint may alter capacity without uniquely determining how issuance should respond.

## 9. Controller-failure taxonomy

Use explicit states:

- CapacityOverhang;
- EssentialShortage;
- LocalAccessMismatch;
- InventoryDepletion;
- ResourceConstraint;
- ImportConstraint;
- ControllerInstability;
- InformationInsufficient;
- Unknown.

## 10. Claim ceiling

A PASS establishes only bounded behavior of the selected issuance/capacity controller under the synthetic profile.

It does not establish the correct real-world Credit quantity, inflation target, ecological carrying capacity, or universal issuance policy.
# REGEN-042S1B — Service Floor / Scenario Demand Separation v1

Status: preregistration hardening only.

Parent state contract: REGEN-042S1 / #3823.
Bounded-state hardening: REGEN-042S1A / #3832.
Primitive hardening prerequisite: REGEN-042S0A / #3831.
Authoritative campaign semantics: Luminous-Dynamics/mycelix#1431 and #1519.
Cross-repo ownership: Luminous-Dynamics/mycelix#1523.

## 1. Purpose

REGEN-042A explicitly separates supply loss from demand increase. The current S1 service sketch includes an adopted `required_floor`, but it does not yet give scenario demand its own committed state dimension.

Without that separation, an executable demand shock could accidentally mutate the adopted service requirement itself.

S1B freezes:

```text
adopted service floor
!= modeled/scenario demand
!= delivered service
```

and therefore:

```text
demand shock
may change modeled demand
may not rewrite adopted service floor
```

## 2. Authority theorem

Mycelix/domain policy profiles remain authoritative for the adopted minimum service requirement.

Symthaea may represent and transform scenario demand only inside a frozen scientific campaign.

```text
adopted floor
+ scenario demand
+ delivered service
= three distinct committed propositions
```

No model transition may convert scenario demand into policy authority.

## 3. Revised service-state shape

Conceptually, later executable S1 should represent at least:

```rust
pub struct ServiceState {
    pub availability: ServiceAvailability,
    pub delivered: Resolution<CanonicalQuantity>,
    pub adopted_floor: Resolution<CanonicalQuantity>,
    pub scenario_demand: Resolution<CanonicalQuantity>,
    pub scope: ScopeId,
    pub active_dependencies: BTreeSet<DependencyId>,
}
```

Exact API names may differ.

The important invariant is semantic separation, not the field spelling.

## 4. `adopted_floor` meaning

`adopted_floor` is the service requirement supplied by the authoritative/adopted profile revision.

It may be known, unresolved, or not applicable according to the frozen bridge contract.

Symthaea does not invent a missing floor and does not revise the floor merely because a shock creates scarcity or excess demand.

## 5. `scenario_demand` meaning

`scenario_demand` is the modeled service demand for the current campaign state.

It may originate from:

- an observed demand snapshot;
- a forecast with exact lineage;
- a synthetic scenario fixture;
- an explicitly frozen deterministic demand profile.

Its provenance/evidence class remains inspectable.

```text
scenario demand
!= observed demand by default
```

## 6. Delivered service remains separate

`delivered` records modeled/observed delivered-service quantity according to the campaign's declared state provenance.

```text
delivered >= adopted_floor
```

may support one service-floor proposition under a profile, but it does not imply:

```text
delivered >= scenario_demand
```

or vice versa.

A service may satisfy the adopted minimum while leaving additional demand unmet.

## 7. Demand revision identity

The model-state identity must bind the exact demand input/profile revision used to construct `scenario_demand`.

Conceptually add one identity such as:

```text
demand_profile_revision
```

or an equivalently explicit scenario-demand snapshot reference.

Changing the demand revision changes the committed state even if the numeric demand happens to be equal.

## 8. No defaulting floor to demand

The bridge/kernel must not apply hidden rules such as:

```text
if scenario_demand missing -> use adopted_floor
if adopted_floor missing -> use scenario_demand
```

unless an exact adopted profile explicitly defines that transformation and the transformation is represented with lineage.

Absent evidence remains absent/unresolved according to the declared schema.

## 9. No defaulting demand to delivered

Observed/model delivered quantity must not become demand merely because it is the only known quantity.

```text
delivered != demanded
```

This prevents scarcity-constrained delivery from being interpreted as evidence that lower demand existed.

## 10. Demand shocks target only scenario demand

A future S2 effect such as:

```rust
DemandScaled { service_id, factor }
```

must target `scenario_demand`.

It must not mutate:

- `adopted_floor`;
- service profile identity;
- scope identity;
- policy/authority state.

This is the central S1B handoff theorem.

## 11. Service floor under shock remains hard input

A supply or demand shock does not relax the adopted floor unless the campaign consumes a distinct, explicitly adopted emergency profile revision.

If a different requirement profile is intentionally studied, that is a different campaign/input identity rather than a runtime side effect of the shock.

## 12. Emergency profile firewall

An emergency service profile may be modeled only when supplied as an exact authoritative/adopted input.

Symthaea cannot create one because the simulated state is stressed.

```text
model detects scarcity
!= authority to lower minimum requirement
```

## 13. Quantity compatibility

Where `delivered`, `adopted_floor`, and `scenario_demand` are all known and semantically comparable, their units/bases must be compatible.

A mismatch is a typed validation error, not an implicit conversion.

Any legitimate conversion belongs to a separately evidenced transformation outside canonical primitive arithmetic.

## 14. Partial-demand states

A model may report both:

```text
floor_satisfied = true
unmet_scenario_demand > 0
```

without contradiction.

This is often the correct distinction between preserving an adopted essential-service floor and satisfying all modeled demand.

S1B therefore forbids reducing service state to one binary `satisfied` flag when both propositions matter.

## 15. Degraded service is profile-relative

Whether delivered service is `Satisfied`, `Degraded`, or `Failed` relative to an adopted floor remains profile-defined.

Scenario-demand shortfall is an additional result dimension.

```text
availability relative to floor
!= demand-coverage fraction
```

No universal scalar combines them.

## 16. Demand increase is not supply loss

A demand shock changes demand state.

It does not directly:

- decrease nominal dependency capacity;
- mark a dependency unavailable;
- destroy stock;
- alter a failure domain.

Any resulting service deficit is derived later from allocation/transition semantics.

## 17. Supply loss is not demand reduction

Likewise, a supply shock does not reduce scenario demand merely because less service can be delivered.

This prevents the model from erasing unmet demand by lowering its denominator after a failure.

## 18. Demand provenance survives transitions

When S2/S3 later transform scenario demand, the transition receipt should bind:

```text
prior demand state/ref
demand effect ref
successor demand state/ref
```

A changed demand value does not erase the source fixture/profile that produced the prior state.

## 19. Demand scaling and rounding

Demand scaling must consume the hardened S0A rational-scaling semantics.

If a rational multiplier is not exactly representable at the adopted decimal precision, the transition must preserve the explicit rounding/residual result or reject according to the frozen effect profile.

Hidden truncation is forbidden.

## 20. Negative demand

Scenario demand is non-negative in v1.

A model that needs signed demand deltas should represent the delta separately and apply checked arithmetic into a non-negative successor demand.

Negative absolute demand is invalid state.

## 21. Zero demand semantics

Known zero demand is distinct from unresolved demand and not-applicable demand.

```text
Known(0)
!= Unresolved(...)
!= NotApplicable
```

This distinction is committed in canonical state bytes.

## 22. Demand horizons

A scalar current-tick demand value does not by itself represent a full future demand trajectory.

If later campaign stages need trajectories, the exact trajectory/profile revision must be separately identified rather than extrapolated from one scalar without a frozen rule.

## 23. Baseline comparability

Counterfactual A/B runs that claim to isolate architecture effects must bind the same initial:

- adopted service floor profile;
- scenario demand profile/snapshot;
- service scope;
- campaign horizon;

unless one of those is the explicitly declared experimental variable.

## 24. First regression campaign

Executable S1/S1B tests should include at least:

1. valid state with different floor and demand values;
2. floor satisfied while demand remains partially unmet;
3. demand known zero distinct from unresolved;
4. floor unresolved while demand known;
5. demand unresolved while floor known;
6. delivered/floor unit mismatch rejection;
7. delivered/demand unit mismatch rejection where comparison is required;
8. basis mismatch rejection;
9. demand-profile revision-only mutation changes state commitment;
10. equal demand value under different demand revisions yields different commitment;
11. bridge does not default missing demand to floor;
12. bridge does not default missing floor to demand;
13. bridge does not default demand to delivered;
14. demand-shock validation targets scenario demand only;
15. demand-shock fixture cannot alter adopted-floor/profile identity;
16. supply failure does not silently lower scenario demand.

## 25. Independent-oracle addition

At least one S1A known-answer fixture should contain distinct values for:

```text
adopted_floor
scenario_demand
delivered
```

so the independent encoder would detect accidental field omission or field aliasing.

Mutation controls should swap floor/demand encodings and require commitment mismatch.

## 26. Cross-repo bridge requirement

The eventual Mycelix -> Symthaea bridge schema must map authoritative requirement identity and scenario-demand identity into distinct fields.

A single ambiguous field such as:

```text
service_requirement_or_demand
```

is not acceptable for the v1 bridge.

## 27. Reporting boundary

Later reports should preserve at least:

```text
adopted-floor status
scenario-demand coverage/unmet demand
delivered quantity
```

where those quantities are available.

No master resilience score is introduced.

## 28. Deliberate non-claims

REGEN-042S1B establishes no universal service minimum, no forecast accuracy, no demand truth, no entitlement, no emergency policy, no rationing rule, no allocation priority, no S0/S0A/S1 executable PASS, no resilience claim, and no physical-action authority.

Its proposition is narrow:

> before Symthaea models demand shocks, the immutable state must distinguish the adopted service floor from scenario demand and delivered service so model dynamics cannot silently rewrite social/policy requirements or erase unmet demand.

# REGEN-039 — Settlement Metabolism Graph Contract v1

Status: preregistration / architecture boundary only

Parent: REGEN-038 crop / nutrition service projection

Program: Luminous-Dynamics/mycelix#940

## Purpose

Define a small, typed settlement-scale composition graph for qualified REGEN
kernels without creating a second CivOS, governance layer, marketplace,
optimization authority, or physical-control plane.

REGEN-039 composes already-identified stocks, flows, transformations, services,
losses, and external boundaries.

It does not own the authoritative state of the domains it references.

## Governing theorem

```text
identity-bearing qualified component outputs
+ explicit stock / flow / transformation edges
+ explicit time basis
+ explicit conservation dimensions
+ explicit losses / residuals
= bounded settlement-metabolism projection
```

not:

```text
real settlement state
resource ownership
allocation authority
policy
market commitment
physical execution
self-sufficiency proof
resilience proof
```

## No monolithic settlement model

REGEN-039 must not reimplement:

- biomass evidence;
- feedstock eligibility;
- water authority;
- energy authority;
- nutrient accounting;
- soil-response models;
- crop / nutrition models;
- marketplace state;
- finance;
- governance;
- infrastructure control.

It composes exact references to those domains / models.

## Graph role

The graph exists to answer questions such as:

- where does a modeled resource originate?
- which transformations consume it?
- which services may receive it?
- where are losses explicit?
- which stocks buffer timing?
- which flows are unresolved?
- where do two proposed uses compete for one finite source?
- which external imports / exports remain necessary?

It does not decide who is entitled to use a resource.

## Node taxonomy

A minimal v1 graph may use typed nodes such as:

```text
ExternalSource
Stock
Transformation
Service
LossSink
ExternalSink
MeasurementBoundary
```

The taxonomy may evolve, but node kinds must remain semantically explicit.

A generic `ResourceNode` with hidden behavior is insufficient.

## Edge taxonomy

A modeled flow should carry at least:

```text
FlowEdge {
    flow_id,
    commodity_id,
    source_node,
    destination_node,
    quantity_or_rate,
    unit_basis,
    time_support,
    evidence_or_projection_ref,
    epistemic_class,
}
```

Quantity and rate must not be interchangeable.

A daily rate is not an annual stock.

## Commodity identity

Every conserved flow requires an exact commodity / basis identity.

Examples of distinct quantities that must not share one unlabeled edge:

```text
wet biomass mass
dry biomass mass
carbon mass
water mass
mineral nutrient stock
thermal energy
electrical energy
edible fresh mass
nutrient mass
```

Compatible numerical values do not make different commodities interchangeable.

## Conservation dimensions

Conservation is dimension-specific.

A transformation may close:

- total material mass;
- dry mass;
- water mass;
- one named element;
- carbon;
- energy;

only when the relevant inputs / outputs are actually represented on the same
basis.

Normative:

```text
mass closure
!= carbon closure
!= nutrient closure
!= energy closure
```

REGEN-039 must not create one generic "conservation passed" flag that hides
which ledger was checked.

## Transformation receipts

A transformation node should consume an exact qualified model/accounting
receipt from an upstream REGEN kernel where available.

Examples:

- REGEN-032 pyrolysis accounting;
- REGEN-033 compost accounting;
- REGEN-034 nutrient balance;
- REGEN-036 water balance;
- REGEN-037 thermal-service compatibility;
- REGEN-038 crop / nutrition service projection.

REGEN-039 may check graph-level consistency around those receipts.

It may not silently recompute them with a new formula.

## Stocks make loops physical

A circular graph without stocks and time can create impossible instantaneous
recirculation.

Therefore any modeled cycle must identify:

- stock / buffer identity;
- initial stock;
- capacity if bounded;
- time step / interval;
- inflow timing;
- outflow timing;
- losses;
- transformation delay where relevant.

Normative:

```text
cycle in graph topology
!= physically closed instantaneous loop
```

A flow emitted during one interval cannot automatically become an input earlier
in the same interval unless the scheduler explicitly defines that ordering.

## No perpetual recycling

Every recycling loop must preserve explicit recovery yield and losses.

```text
recovered material
<= eligible input material
```

under the same declared mass basis, with residuals visible.

The graph must reject any cycle that creates conserved quantity without an
identified external source or modeled conversion from another conserved basis.

## Time semantics

Every stock and flow must have explicit time support.

REGEN-039 should distinguish:

```text
stock at t
flow during [t0, t1)
service accumulated during interval
instantaneous rate
event at boundary
```

The first executable candidate should prefer a simple discrete interval
semantics rather than ambiguous continuous feedback.

## Ordering semantics

A transformation chain must state its order.

For one interval, a safe default for the first candidate is conceptually:

```text
read initial stocks
-> apply declared external inputs
-> evaluate bounded transformations
-> apply explicit losses
-> write ending stocks / services
-> emit evidence receipt
```

A later coupled solver may use another ordering, but must name it.

## No double spending

A finite source cannot be consumed twice merely because two graph edges point
away from it.

Every allocation from a stock should satisfy:

```text
sum(outgoing committed quantity)
<= available modeled quantity
```

for the exact interval and commodity basis.

Scenario allocation is still not a real reservation / physical consumption.

Where real Mycelix reservation / custody state exists, REGEN-039 references it
rather than replacing it.

## Competing uses

Competing uses remain explicit branches.

The graph must preserve:

- retained / unused quantity;
- use A;
- use B;
- loss;
- unresolved quantity.

It does not choose the socially correct branch by itself.

REGEN-031 may screen candidate allocations; REGEN-039 may compose their modeled
consequences.

## External boundaries

A resilient settlement model must not treat imports as modeling failure.

External sources and sinks are first-class.

Examples may include:

- imported food;
- grid energy;
- rainfall;
- purchased material;
- exported product;
- unavoidable waste / discharge.

Normative:

```text
local loop closure
!= desirable autonomy
!= self-sufficiency
```

A system that needs an external input can still be viable.

## No universal self-sufficiency score

REGEN-039 does not expose:

```text
self_sufficiency: 0..100
```

or a universal settlement-quality score.

It may expose plural quantities such as:

- external input dependencies;
- stock coverage intervals;
- modeled service outputs;
- loss vectors;
- unresolved dependencies;
- substitution candidates.

Interpretation belongs to later resilience / economic / governance layers.

## Service outputs

Services are different from physical stocks.

Examples:

```text
thermal service
food / nutrition service
water service
waste-processing service
```

A service node must preserve the upstream evidence / model chain.

Service satisfaction must not silently consume a stock unless the graph includes
the corresponding flow.

## Provenance

Every graph-level quantity should be traceable to:

- observed evidence;
- derived evidence;
- an upstream projection;
- a calibrated parameter;
- a scenario assumption;
- a synthetic fixture.

A mixed graph may include all of these, but must not collapse them into one
confidence scalar.

## Uncertainty and unresolved state

Unresolved inputs remain explicit.

The graph should be able to report:

```text
Resolved(quantity)
Unresolved(reason)
Unsupported(model_or_basis)
```

rather than substituting zero.

Unknown flow is not zero flow.

## Authority firewall

REGEN-039 has no authority to:

- reserve a real resource;
- transfer ownership;
- place a market order;
- approve a budget;
- authorize extraction;
- operate equipment;
- alter irrigation;
- dispatch energy;
- allocate food;
- override ecological retention;
- modify governance policy.

```text
modeled feasible flow
!= authorized flow
!= executed flow
```

## Economic firewall

A modeled material or service flow does not imply:

- market price;
- profitability;
- affordability;
- financing eligibility;
- investment return.

Those belong in Phase G / existing economic authority domains.

## Climate firewall

A closed carbon or energy ledger inside the metabolism graph does not imply:

- avoided emissions;
- durable carbon removal;
- climate benefit;
- carbon-credit eligibility.

Those belong in Phase H / Climate authority.

## Rights / ecology firewall

A locally available modeled resource may still be unavailable because of:

- ecological-retention obligations;
- rights;
- ownership;
- consent;
- contamination;
- safety;
- policy;
- competing essential use.

REGEN-039 cannot override those hard gates.

## First proving graph

The first executable graph should be much smaller than a settlement simulator.

A suitable synthetic proving graph is the existing program's community soil
loop:

```text
qualified modeled biomass stock
          |
          v
  bounded allocation branch
      /              \
retain / other       candidate process input
                         |
                         v
                  pyrolysis accounting
                    /        \
            char stream     heat stream
                |               |
                v               v
      compost / amendment   compatible heat sink?
                |
                v
       soil-response shadow
                |
                v
        crop-service shadow
```

The exact first implementation can be even smaller.

The purpose is to prove graph semantics and conservation, not demonstrate a
working community technology stack.

## First executable candidate

Recommended smallest increment:

1. define commodity / unit / time-support identities;
2. define stock, transformation, service, loss, and external nodes;
3. define one bounded interval;
4. seed one synthetic biomass stock;
5. partition it into retained and candidate-process quantities;
6. consume an exact synthetic REGEN-032-style transformation receipt;
7. route one coproduct to one service candidate;
8. retain explicit losses / unresolved residuals;
9. prove source quantity cannot be double spent;
10. prove graph-level declared conservation closes;
11. prove a cycle without a stock / time boundary is rejected;
12. emit a replayable graph receipt.

Do not start with optimization, governance, agents, markets, or a 97-resource
civilization simulation.

## Graph receipt

Conceptually:

```text
MetabolismGraphReceipt {
    graph_id,
    schema_version,
    interval,
    node_ids,
    flow_ids,
    component_receipt_refs,
    external_input_refs,
    external_output_refs,
    conservation_checks,
    unresolved_items,
    assumption_refs,
}
```

The receipt is scenario / model evidence.

## Adversarial fixtures

Qualification should include:

- same biomass allocated to two consuming transformations;
- wet mass and dry mass added without conversion;
- carbon mass used as total material mass;
- output generated with no input / external source;
- closed cycle with no stock / time semantics;
- unknown loss treated as zero;
- rate added directly to stock;
- flow from wrong time interval;
- upstream receipt from wrong subject;
- incompatible commodity identity;
- modeled heat compatibility treated as delivered service;
- crop-service projection treated as observed food stock;
- external import silently omitted to make the settlement appear self-sufficient;
- favorable graph outcome offered as governance / allocation authority.

Every shortcut must reject or remain explicitly unresolved.

## Scenario harnesses

Large agent-based or civilization simulators may later consume REGEN-039 graph
receipts or replay compatible scenarios.

They are downstream harnesses.

They are not the normative owner of:

- physical conservation;
- evidence identity;
- Mycelix authority;
- REGEN component semantics.

The first graph therefore stays independently testable and small.

## Relationship to Phase E

REGEN-040..047 may consume REGEN-039 dependency / stock / service topology for
resilience and continuity analysis.

Phase E may add shocks and substitution.

It must not rewrite the underlying accounting receipts.

## Relationship to Phase F

Experiment intelligence may use graph bottlenecks, unresolved edges, and model
sensitivity to propose informative trials.

It remains recommendation-only.

## Promotion gate

An executable REGEN-039 candidate should not be described as qualified until
its exact subject demonstrates:

- frozen toolchain / code lineage;
- compile / test / strict lint;
- deterministic replay;
- exact node / flow / commodity / time identities;
- per-dimension conservation evidence;
- double-spend rejection;
- unknown-loss preservation;
- explicit external boundaries;
- cycle / stock / ordering semantics;
- no universal self-sufficiency score;
- no duplicated upstream domain model;
- no authority-bearing output;
- postflight immutability.

## Deliberate non-claims

REGEN-039 does not establish:

- real settlement resource availability;
- self-sufficiency;
- resilience superiority;
- economic feasibility;
- ecological sustainability;
- food security;
- water security;
- energy security;
- carbon neutrality;
- governance legitimacy;
- resource ownership;
- allocation rights;
- infrastructure authorization;
- autonomous control.

It establishes only a typed composition graph for already-bounded REGEN model
components.

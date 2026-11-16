# Optical ISRU v0.2 — multi-resource campaign budget and Pareto reference v0.1

**Status:** standalone Rust reference; 12 tests authored, not compiled or executed  
**Parent:** Symthaea #12535 — multi-resource campaign budgets and Pareto cost-aware selection  
**Related:** Symthaea #12483 / #12496 — sealed BO/EIG tournament; #12525 — likelihood EIG/EVSI; #12534 — propensity receipts

## Why scalar cost is insufficient

Optical ISRU campaigns spend multiple resources that are not interchangeable by default:

- instrument and rig hours;
- source energy at a declared meter boundary;
- feedstock and specimen mass;
- analysis / independent assay cost;
- maintenance and requalification time;
- replacement witnesses and other consumables.

A scalar cost can be useful after a declared economic or operational valuation, but it must not conceal a hard limit. An action that saves time but exceeds the energy ceiling is not feasible merely because its weighted sum looks attractive.

The resource vector therefore remains primary. Hard budget admission is component-wise. Pareto comparison preserves incomparable actions. A scalar decision-value density is available only when a caller supplies explicit, versioned shadow prices.

## First implementation

`research/asteroid-optical-isru-v0.2/campaign_resources.rs` adds:

1. a typed six-dimensional resource vector with fixed units;
2. an explicit energy-meter boundary and assay-cost unit basis;
3. strict finite/nonnegative value validation;
4. component-wise budget assessment before acquisition ranking;
5. explicit list of over-budget resources for every rejected candidate;
6. remaining-budget calculation for affordable actions;
7. a mutable campaign ledger that applies actions atomically against remaining resources;
8. decision-value / resource Pareto dominance and frontier extraction;
9. maximum decision-value selection among affordable actions without implicit scalarization; and
10. optional decision-value density using explicit shadow prices with a version and configuration digest.

The initial implementation uses electrical input, optical source output, receiver incident, receiver absorbed and useful process input as distinct energy meter-plane types. Two actions cannot be compared against a budget with a different energy boundary or assay-cost unit.

## Decision rules

### Feasibility first

For every resource dimension j:

[
c_j(a) \leq B_j.
]

If any dimension exceeds its budget, the action is rejected before ranking. It remains in the assessment output with the exceeded dimension(s), so failures are not silently removed from campaign accounting.

The ledger accumulates successful actions. An action that would exceed the remaining campaign budget fails atomically; spent and remaining values do not change on rejection.

### Pareto before scalarization

Action a dominates action b only when it has at least as much expected decision value and consumes no more of every resource, with at least one strict improvement.

If one action uses less energy but more rig time, and another uses more energy but less rig time, they remain incomparable unless an explicitly declared preference resolves the trade-off.

### Optional explicit shadow prices

A caller may supply a versioned price vector (lambda), in which case:

[
C_\lambda(a)=\sum_j \lambda_j c_j(a),
\qquad
D_\lambda(a)=\frac{EVSI(a)}{C_\lambda(a)}.
]

The vector and configuration digest must be explicit. This is a decision preference, not a physical law or universal valuation. It cannot bypass the hard component-wise budget gate.

## Authored tests

Twelve tests cover single-resource overrun, non-negative remaining budgets, rejection of high-value but unaffordable actions, Pareto incomparability, removal of dominated candidates, explicit shadow-price ranking, preference changes under different price vectors, energy meter-boundary mismatch, invalid resource/price values, zero weighted cost, atomic cumulative campaign accounting and no-affordable-action refusal.

## Reproduce locally

From the repository root:

```sh
rustc --edition=2021 --test research/asteroid-optical-isru-v0.2/campaign_resources.rs -o /tmp/isru-resource-tests
/tmp/isru-resource-tests
```

**Validation honesty:** tests are authored but not compiled or executed in the authoring environment because rustc/cargo are unavailable there. No physical resource-use measurement, campaign economics, accepted-product yield or optimization advantage is claimed.

## Required integration

1. Add the resource vector and energy boundary to each action in the EIG/EVSI tournament.
2. Keep the exact same campaign budget and units across paired policy comparisons.
3. Include setup, calibration, drift stabilization, optical inspection, maintenance, recovery and downtime in the campaign ledger.
4. Bind actual resource measurements to Symtropy instrument and measurement-boundary records.
5. Record the frozen budget, ledger snapshot, action cost vector and shadow-price configuration in each selection receipt.
6. Compare campaign-level accepted product per total rig-hour and per each resource independently.
7. Publish the Pareto frontier and all rejected actions; do not present one weighted score as universal superiority.

This is a reference decision-accounting kernel. It does not measure resources, establish meter calibration, optimize plant economics or authorize physical operations.

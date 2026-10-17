# Creditism Credit-Stock Dynamics Benchmark

## Purpose

Define the stock/flow boundary for Personal Credit when issuance occurs continuously, spending deletes the instrument, and balances may be retained for future use.

This benchmark is required before interpreting aggregate Credit quantities, price pressure, or purchasing-power claims.

Stock-flow consistent monetary models explicitly integrate stocks, flows, real activity, and financial positions rather than assuming a fixed relationship between money demand and supply. This is directly relevant to Creditism because deletion makes the flow-to-stock relation explicit.

References:
- https://www.levyinstitute.org/publications/endogenous-money-in-a-coherent-stock-flow-framework/
- https://www.levyinstitute.org/publications/kaleckian-models-of-growth-in-a-stock-flow-monetary-framework/
- https://www.imf.org/external/pubs/ft/mfs/manual/eng/anmfinsta.htm

## 1. Core identity

For each closed Personal Credit profile:

closing_PC = opening_PC + issuance_PC - deletion_PC + other_declared_adjustments

Do not infer the closing stock from prices or output.

Do not infer issuance from spending.

Do not infer deletion from contribution recognition.

## 2. Four distinct variables

At minimum maintain:

- `PC_stock` — outstanding Personal Credit balances;
- `PC_flow_issued` — current-period issuance;
- `PC_flow_deleted` — current-period deletion;
- `PC_velocity` — spending relative to the relevant stock measure.

Also preserve:

- distribution of balances;
- household liquidity needs;
- demand basket/preferences;
- goods-specific supply;
- expected future issuance;
- expected future deletion opportunities;
- external conversion conditions.

## 3. Steady-state condition

A stationary Credit stock requires, under the simplest closed profile:

average issuance ≈ average deletion

over the chosen reference period.

This is an accounting property, not a policy recommendation.

If:

issuance > deletion

the stock grows unless another declared adjustment absorbs the difference.

If:

issuance < deletion

the stock contracts unless another declared issuance mechanism compensates.

## 4. Important distinction: stock growth is not automatically inflation

Common Planet's current architecture explicitly states that higher outstanding Credit does not mechanically translate to higher prices; it says price pressure depends on demand relative to real capacity, inventories, scarcity, and ecological limits.

Source:
- https://common-planet.org/creditism/architecture

Therefore the benchmark should not encode:

PC_stock_growth -> inflation

as a theorem.

Instead test the mediated path:

PC_stock
+ distribution
+ velocity
+ basket
+ physical capacity
-> observed demand pressure / unmet demand / price response

## 5. Hoarding / saving fixture

Same issuance path, but households vary in the fraction of Personal Credit they retain.

Compare:

- immediate spending;
- moderate saving;
- high saving;
- synchronized future spending;
- heterogeneous future spending.

Measure separately:

- current demand;
- future demand;
- liquidity coverage;
- stock concentration;
- resource pressure when saving is released.

Because Personal Credit may remain in an account indefinitely, synchronized release of saved balances is a first-class stress case.

## 6. Velocity shock

Hold household Credit balances constant while changing spending timing.

Example:

World A: each household spends 10 PC per period.
World B: households spend 20 PC every second period.

Aggregate balances can be identical while payment flows differ.

Required property:

PC_stock != PC_flow != velocity

Any Balance/controller model that responds to one must state why.

## 7. Distribution shock

Hold total Personal Credit constant.

World A: evenly distributed balances.
World B: concentrated balances.

Demand baskets are held constant within households.

Measure:

- effective purchasing pressure by market;
- unmet essential demand;
- access concentration;
- price/rationing response;
- system liquidity.

This tests whether an aggregate stock is sufficient for the controller. It should not be assumed to be.

## 8. Issuance shock

Hold physical capacity fixed.

Increase contribution issuance or baseline issuance for one period.

Do not prescribe the correct outcome.

Record:

- stock accumulation;
- spending response;
- demand pressure;
- shortage/rationing;
- price changes;
- subsequent production response;
- later deletion;
- persistence of any overhang.

## 9. Deletion-capacity shock

Reduce the availability of desirable goods while keeping current balances unchanged.

This can create a demand/supply mismatch without any monetary issuance change.

Required distinction:

scarce goods + unchanged PC stock
!= monetary expansion.

The benchmark should determine whether the resulting pressure is resolved through:

- price;
- rationing;
- substitution;
- waiting;
- new capacity;
- governance allocation;
- some combination.

## 10. Intertemporal overhang test

Create balances that were accumulated under one supply regime and then release spending opportunities after the productive structure has changed.

Examples:

- saved Credit + drought;
- saved Credit + housing shortage;
- saved Credit + energy shock;
- saved Credit + technology transition.

Measure whether purchasing-power persistence becomes a source of temporary demand pressure.

This is not evidence of failure by itself. It is an explicit state transition that the architecture must represent.

## 11. Issuance/deletion governance

Every issuance mechanism needs an independent identity and rule:

- existence;
- contribution;
- Bonus;
- Community Credit;
- extraordinary/transition issuance where separately authorized;
- external transfer where enabled.

Every deletion mechanism needs an independent identity:

- Marketplace;
- Exchange;
- Community purchase;
- other declared use.

Never allow a generic `credit_delta` to conceal which mechanism created the change.

## 12. Positive controls

Include cases where:

- higher Credit stock accompanies higher real capacity and stable access;
- lower stock accompanies constrained demand but insufficient output;
- high velocity improves utilization without changing stock;
- concentration, not aggregate stock, causes localized scarcity;
- deletion falls while contribution issuance remains healthy.

These controls prevent a simplistic stock-targeting or anti-Credit interpretation.

## 13. Controller boundary

The Balance controller should declare its exact state inputs and observation window.

Do not permit hidden access to:

- future issuance;
- future demand;
- evaluator labels;
- later realized capacity;
- protected household information.

Historical/as-of controller decisions remain frozen when later information arrives.

## 14. Qualification outputs

Each run should emit:

- opening stock;
- issuance by mechanism;
- deletion by mechanism;
- closing stock;
- reconciliation residual;
- balance distribution;
- velocity measure/profile;
- demand basket;
- physical capacity;
- inventory;
- unmet demand;
- price/rationing response;
- uncertainty;
- failure disposition.

## 15. Claim ceiling

A PASS establishes only stock-flow behavior under the declared synthetic mechanism and controller profile.

It does not establish a universal quantity theory of Creditism, a real-world inflation forecast, or the optimal issuance rule.
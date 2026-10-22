# Creditism Policy / Mechanism / Ontology Boundary

## Purpose

Creditism's current architecture combines economic mechanisms with explicit constitutional and normative choices. The scenario framework must keep these layers separate so a mechanism result is not mistaken for validation of a value judgment.

Current Common Planet material explicitly frames three determinations as choices that should not be left to markets: Purposes, Pay, and Products. It also describes guaranteed housing, universal Personal Credit, commons stewardship, and democratic Community Credit allocation as design commitments. These are candidate policy/constitutional assumptions, not neutral physical laws.

Sources:
- https://common-planet.org/creditism/architecture
- https://common-planet.org/creditism

## 1. Three layers

### Mechanism

Rules describing how a system operates.

Examples:

- Credit deletes at defined use;
- Personal Credit is non-transferable;
- contribution is recognized separately from consumer settlement;
- Community Credit is restricted to group accounts;
- a resale premium can be deleted instead of becoming seller purchasing power.

These can be tested directly with synthetic fixtures.

### Policy / constitutional choice

Rules specifying what society chooses to guarantee, prioritize, restrict, or authorize.

Examples:

- guaranteed primary housing;
- universal baseline Personal Credit;
- public Standard Pay Scale;
- shortage multiplier;
- community-directed purposes;
- equal Community Credit voting share;
- prohibition on rent/interest/dividend extraction.

These can be parameterized and compared, but their desirability is not a simulator theorem.

### Ontology / normative interpretation

Claims about what counts as a person, contribution, value, ownership, justice, freedom, or flourishing.

Examples:

- existence itself has economic claim;
- contribution rather than ownership should organize recognition;
- housing should be a stewardship right rather than an investment asset;
- community purposes should outrank unconstrained market selection.

These are legitimate objects of philosophical and political discussion, but the simulator must not turn them into empirical facts merely by encoding them.

## 2. Do not hide policy inside mechanism

Bad pattern:

CreditismProfile { guaranteedHousing: true }

followed by:

simulation -> Creditism is superior

Better:

MechanismProfile
+ PolicyProfile
+ GovernanceProfile
+ PhysicalWorld
+ AgentPreferences

-> scenario outcome vector

The policy profile is an explicit independent input.

## 3. Housing allocation deserves its own benchmark

Current Common Planet material describes homes as unpriced stewardship while assigning each adult a Housing Level, with access to available homes up to that level. It also describes Housing Level as potentially increasing through Personal Credit or home improvement, with adjustments as housing availability changes.

Sources:
- https://common-planet.org/creditism
- https://common-planet.org/creditism/architecture

This creates an important research question:

housing is not purchased as a financial asset
!= housing access is independent of resource scarcity

A high-level housing system can still create a scarce-access hierarchy even when homes have no market sale price.

## 4. Housing adversarial fixtures

Test:

1. identical housing stock, different Housing Level distribution;
2. abundant housing, low Housing Level inequality;
3. severe housing scarcity;
4. desirable-location scarcity;
5. disability/accessibility constraints;
6. household size mismatch;
7. migration surge;
8. new construction boom;
9. differential Credit balances with equal Housing Levels;
10. differential Housing Levels with equal Credit balances;
11. strategic home improvement to raise Housing Level;
12. gaming or capture of house quality ratings;
13. neighborhood-level governance capture;
14. temporary vacancy or move coordination;
15. two applicants with equal level competing for one home.

Outputs must remain separate:

- housing access rate;
- waiting time;
- geographic concentration;
- quality distribution;
- household fit;
- accessibility fit;
- mobility;
- maintenance;
- concentration of allocation authority;
- Credit expenditure;
- Housing Level inequality.

## 5. Important equivalence test

Compare:

World A:
homes are priced directly by money

World B:
homes are unpriced but access is gated by Housing Level

World C:
homes are unpriced and allocation is queue/lottery based

The question is not whether B is morally better.

The question is which scarce-allocation functions have moved from price to another institution, and what information, power, waiting, or gaming costs that substitution creates.

## 6. The Three P's as explicit treatment dimensions

Represent separately:

PurposeSelectionMode
PayRateMode
ProductSelectionMode

Candidate values should include:

- market;
- administrative;
- democratic;
- polycentric;
- mixed.

A scenario can therefore test:

market money + market products + market pay

against:

Creditism-like credit + democratic purposes + administered pay

or:

Creditism-like credit + market products + market-derived contribution rates

without pretending those are the same architecture.

## 7. Positive controls

The suite must include cases where market allocation is efficient:

- abundant commodity with dispersed information;
- rapidly changing local demand;
- heterogeneous consumer preferences;
- non-essential luxury goods;
- low governance capacity.

It should also include cases where non-market allocation is effective:

- essential scarce medicine;
- ecosystem restoration;
- common infrastructure;
- emergency response;
- long-horizon resilience investment.

This prevents either ideology from becoming the oracle.

## 8. Claim ceiling

A future experiment may show:

- a policy/mechanism combination produced a particular outcome under declared assumptions;
- moving a decision from markets to administration or governance changed particular outputs;
- housing allocation created or reduced specific forms of concentration;
- the Creditism payment circuit changed transaction and recognition behavior.

It may not conclude from those outputs alone:

- what society ought to value;
- what justice requires;
- that one institutional constitution is universally legitimate;
- that a given allocation regime is democratically legitimate merely because votes were used.

## 9. Research payoff

This separation makes the collaboration stronger.

Common Planet can specify the institutional experiment.

Mycelix can represent identity, evidence, accounting, authority, appeals, and commons state.

Symthaea can explore consequences under alternative assumptions.

The oracle can determine whether the implementation actually followed the declared rules.

The resulting experiment can therefore criticize or validate individual mechanisms without pretending to settle the underlying philosophy.
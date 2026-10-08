# Creditism Mechanism Registry and Dynamic-Control Boundary

This note freezes an orthogonal mechanism registry for the Creditism profile proposed in CREDITISM-MYCELIX-SYMTHEA-MECHANISM-PROFILE.md.
The goal is to prevent implementation coupling from being mistaken for economic causality.

## 1. Orthogonal mechanism registry

A scenario instance must declare each mechanism independently.

### Monetary/access mechanisms

- PC_EXISTENCE_ISSUANCE
- PC_CONTRIBUTION_ISSUANCE
- PC_BONUS_ISSUANCE
- PC_MARKETPLACE_DELETION
- PC_EXCHANGE_DELETION
- PC_EXCHANGE_SELLER_RECOGNITION
- PC_TRANSFERABILITY
- PC_COLLATERALIZABILITY
- PC_INTEREST_BEARING
- PC_INVESTABILITY
- PC_INHERITABILITY

The last five should be capability flags whose Creditism profile is expected to set false, but tests must verify the behavior rather than trust configuration.

### Community allocation mechanisms

- CC_POPULATION_ALLOCATION
- CC_PERSONAL_DIRECTED_SHARE
- CC_GROUP_ACCOUNT_SPENDING
- CC_LOCAL_REGIONAL_GLOBAL_SCOPE
- CC_PURPOSE_GOVERNANCE

### Contribution recognition

- PAY_BASE_RATE
- PAY_SHORTAGE_MULTIPLIER
- PAY_TIME_MODE
- PAY_ACTIVITY_MODE
- PAY_MILESTONE_MODE
- BONUS_OUTCOME_RECOGNITION

### Commons/property mechanisms

- ASSET_PRIVATE
- ASSET_COOPERATIVE
- ASSET_STEWARDSHIP
- USE_RIGHT_TIME_BOUND
- HOUSING_STEWARDSHIP
- HOUSING_LEVEL

### Information/allocation mechanisms

- DEMAND_SIGNAL
- OBSERVED_PRICE
- SCARCITY_SIGNAL
- CONTRIBUTION_SIGNAL
- CAPACITY_SIGNAL
- RESOURCE_CONSTRAINT_SIGNAL

### Governance mechanisms

- ONE_PERSON_ONE_VOTE
- DELEGATED_AUTHORITY
- POLYCENTRIC_AUTHORITY
- AFFECTEDNESS_WEIGHT
- VERIFICATION_AUTHORITY
- APPEAL_AUTHORITY
- RULE_REVISION_AUTHORITY

No scenario may claim to test Creditism without emitting its complete mechanism vector.

## 2. The Balance controller must be explicit

Common Planet's current architecture describes The Balance as algorithms that balance prices with credit volume, with scarce luxuries adjusting upward while essentials and digital goods can become cheaper. It also states explicit failure conditions around miscalibrated issuance causing shortages or runaway prices.

Therefore the controller is not an implementation detail. It is an economic mechanism.

Represent it as:

BalanceController(observed_credit_state, demand_state, supply_state, scarcity_state, price_state, policy_parameters, observation_window) -> price_adjustment

The exact input semantics must be frozen before qualification.

In particular, credit volume must not be left as an ambiguous concept. Candidate definitions include:

- total outstanding Personal Credit;
- unspent Personal Credit;
- Personal Credit velocity;
- Personal + Community Credit;
- transaction-weighted effective purchasing capacity;
- household purchasing-power distribution.

These are economically different variables.

A future implementation must select one or expose a vector rather than silently choosing one.

## 3. Balance-control stability fixtures

### Fixture B1 — benign stationary world

Constant supply, demand, contribution issuance and population.

Expected property:

controller converges or remains bounded without unnecessary oscillation

### Fixture B2 — one-time demand shock

Demand increases once and then returns to baseline.

Measure:

- peak price response;
- settling time;
- overshoot;
- undershoot;
- residual error.

### Fixture B3 — persistent supply shock

Physical supply falls permanently.

Measure whether the controller:

- expresses scarcity;
- preserves essential access where policy says it should;
- avoids unstable price spirals.

### Fixture B4 — issuance shock

Personal Credit issuance increases without immediate physical capacity increase.

This is the most important Creditism-specific control test.

Expected output is not predetermined. The qualification target is that the model exposes the resulting trade-off among:

- nominal purchasing power;
- unmet demand;
- price changes;
- rationing;
- production response;
- household access.

### Fixture B5 — productivity shock

Physical productive capacity increases without a matching arbitrary credit injection.

The controller must not manufacture shortage merely because the nominal Credit stock is unchanged.

### Fixture B6 — concentrated purchasing power

Same aggregate Credit stock, different distribution.

This distinguishes:

aggregate credit volume != effective purchasing power distribution

### Fixture B7 — velocity shock

Same Credit balances, different spending frequency.

This distinguishes:

stock != flow != velocity

## 4. Pay-scale / shortage-multiplier fixtures

Common Planet describes a public Standard Pay Scale based on skill, responsibility, difficulty, risk, and demonstrated experience, with an additional shortage multiplier for hard-to-staff essential roles during transition; it explicitly leaves the trigger for stepping the multiplier down open.

The simulator should therefore treat the shortage multiplier as a separate controller:

ShortageMultiplier(staffing_gap, service_criticality, observation_window, step_policy) -> multiplier

Required attacks:

- gaming reported staffing gaps;
- delayed reporting of newly solved shortages;
- geographic aggregation hiding local shortage;
- strategic task splitting;
- verifier capture;
- sudden substitution by technology;
- low-observability but essential care/maintenance;
- persistent shortage caused by inadequate training capacity.

The multiplier's effect must remain distinguishable from base skill/risk/difficulty compensation.

## 5. Creditism-specific conservation laws

Define and test:

consumer_purchase -> purchaser Credit decreases

and independently:

contribution_recognition -> contributor Credit increases

These may occur in the same real-world story but remain distinct ledger events.

For a closed synthetic Creditism economy:

PC_closing = PC_opening + all_PC_issuance - all_PC_deletion + other_declared_PC_adjustments

The scenario must emit every term.

A purchase cannot disappear from the accounting ledger merely because Credit deletes.

## 6. No hidden issuer of last resort

A conventional monetary model can quietly stabilize a difficult scenario by allowing an implicit central bank or fiscal backstop.

That is unacceptable here.

Every emergency Credit increase must be classified explicitly:

- existence issuance;
- contribution issuance;
- Bonus;
- Community Credit;
- extraordinary issuance under a declared profile;
- external transfer;
- unknown.

An unclassified balance increase is an accounting failure, not a stabilization mechanism.

## 7. Demand / contribution non-equivalence theorem

Freeze this metamorphic property:

hold consumer demand constant
change contribution verification
-> contribution recognition may change
-> consumer settlement semantics must not change

and the converse:

hold contribution verification constant
change demand
-> settlement/deletion may change
-> contribution recognition must not silently increase

This is one of the cleanest ways to detect accidental recoupling of the two circuits.

## 8. Resource-price incidence test

A scenario should report at least:

- posted price;
- purchaser Credit deleted;
- seller recognition;
- producer contribution recognition;
- Community Credit allocation where applicable;
- actual physical resource consumed;
- scarcity state;
- inventory change.

This lets us distinguish:

price changed != producer became richer != resource became scarcer != society produced more

## 9. Capital-allocation coupling test

Keep these mechanisms independently switchable:

- Credit issuance
- Community allocation
- Project selection
- Resource allocation
- Capital formation
- Operating output
- Maintenance
- Bonus

A simulation should be able to show, for example:

high Community Credit + low capital formation

without forcing the conclusion that the allocation algorithm is good or bad.

The causal question can then be studied downstream.

## 10. Governance substitution test

Run paired worlds with identical monetary mechanics but different governance:

World A:
same PC/CC rules
polycentric verification
local appeals

World B:
same PC/CC rules
centralized verification
centralized appeals

Any difference in concentration, throughput, participation, error correction, fraud, access, or stability is evidence about governance, not Creditism's monetary mechanism.

## 11. Failure-state taxonomy

Do not collapse all bad outcomes into economic failure.

Use at least:

- AccountingInvalid
- CapacityShortage
- DemandRationing
- InformationInsufficient
- VerificationFailure
- GovernanceCapture
- AllocationCapture
- ControllerInstability
- TransitionFailure
- ExternalSettlementFailure
- MaintenanceFailure
- Unresolved

This allows one mechanism to fail while the rest of the architecture remains evaluable.

## 12. Positive controls

The profile must contain controls that could make conventional or hybrid mechanisms look better.

At minimum:

- healthy debt-funded factory expansion;
- equity-funded long-horizon research;
- efficient price discovery under dispersed information;
- healthy maturity transformation;
- successful market allocation of a non-essential luxury;
- effective centralized emergency response;
- effective polycentric commons governance.

A Creditism profile that wins all fixtures because the evaluator encodes its values is invalid.

## 13. Minimum reproducibility package

Each run should freeze:

- mechanism_profile_hash;
- scenario_definition_hash;
- initial_state_hash;
- shock_schedule_hash;
- parameter_distribution_hash;
- random_seed;
- model_version;
- accounting_profile_hash;
- oracle_version.

and produce:

- event_receipts;
- state_snapshots;
- accounting_reconciliation;
- mechanism_vector;
- failure_dispositions;
- uncertainty;
- nonclaims.

The human-readable dashboard is not the authority; the deterministic package is.

## 14. Research question generated by this registry

Once these mechanisms are separable, the scientific question becomes much stronger:

> Which observed outcomes are caused by Creditism's non-transferable/deleting purchasing-power circuit, which arise from contribution pricing, which arise from commons governance, and which arise from allocation/control algorithms?

That question is testable.

The broader ideological claim — whether the resulting architecture is desirable for society — remains outside the simulator's authority.
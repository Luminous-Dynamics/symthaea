# Replicator Safety Kernel — Zero-Representable Resource Exhaustion v0.1

Status: **normative resource-accounting invariant; not a production admission record**

This contract defines how consumptive RSK resource authority reaches and represents complete exhaustion under the current Rust resource arithmetic profile.

It contains no physical resource definition, fabrication process, molecular/biological design, or replication mechanism.

---

## 1. Constitutional purpose

Resource authority must be able to become exactly zero without entering an invalid or ambiguous state.

The governing rule is:

```text
Remaining_d = Limit_d - Consumed_d
```

for each applicable resource dimension `d` under one exact accounting scheme.

The result is itself a resource vector under that same scheme.

Therefore:

```text
Valid(Limit) && Valid(Consumed)
    does not imply
Valid(Remaining) automatically
```

The derived remainder must satisfy the scheme too.

---

## 2. The discovered parity gap

The Rust reference path computes remaining resources and then re-validates the resulting `ResourceVector` through `ResolvedResourceAccountingScheme::validate()`.

The Python reference previously validated `limit` and `consumed`, subtracted them, and returned the result without validating the derived remainder.

For a schema such as:

```text
minimum = 10
limit = 20
consumed = 15
remaining = 5
```

both input vectors are individually valid, but `remaining = 5` is not representable under the same scheme.

A reference implementation that returns `5` while another rejects it is not acceptable for authority accounting.

---

## 3. Generic arithmetic rule

For every resource scheme, reference remaining-budget arithmetic SHALL perform:

```text
1. validate limit under exact scheme
2. validate consumed under exact scheme
3. require identical dimension sets
4. checked subtraction per dimension
5. validate the derived remaining vector under the exact same scheme
6. return only if all checks pass
```

There is no assumption that valid operands imply a valid result.

The same principle applies to every future derived authority-bearing resource value: derived values must satisfy their declared semantic type before they can become evidence or authority inputs.

---

## 4. Zero exhaustion under the current consumptive budget profile

The current Rust resource profile is:

```text
symthaea.rsk.resource-representation.rust-u64-sum-exact.v1
```

For RSK consumptive authority budgets, this profile additionally requires:

```text
minimum == 0
```

for every admitted resource dimension.

This is not merely a numeric convenience. It guarantees that complete exhaustion is a valid representable state:

```text
Consumed_d == Limit_d
=> Remaining_d == 0
=> Valid(Remaining_d)
```

under the current profile.

---

## 5. Why zero must be representable

If a consumptive budget scheme cannot represent zero, one of several unsafe ambiguities appears when the budget is exhausted:

- clamp the result upward to the minimum, which recreates authority;
- use an out-of-band sentinel, creating a second arithmetic semantics;
- treat an invalid value as zero implicitly;
- reject the state after valid consumption, making exhaustion operationally ambiguous;
- avoid consuming the last permitted amount, changing the stated budget semantics.

The current profile rejects all of these alternatives.

Zero is the ordinary, valid representation of no remaining resource authority.

---

## 6. No authority resurrection

Representable zero does not create a refill or recovery mechanism.

The transition:

```text
remaining > 0 -> remaining == 0
```

is ordinary consumptive attenuation.

Once zero is reached, new positive authority still requires the separately governed grant/policy/ledger path. Exhaustion itself cannot mint a new budget generation, clear consumption, or reset ancestral counters.

---

## 7. No clamping or saturation

The current profile MUST NOT respond to a below-minimum or exhausted result by:

- clamping upward;
- saturating to a positive floor;
- wrapping;
- substituting a default allowance;
- omitting the dimension;
- silently changing the accounting scheme;
- interpreting an invalid vector as a valid zero vector.

A generic scheme whose derived result falls outside its declared bounds fails closed.

A current-profile scheme with `minimum > 0` is rejected before positive runtime validator/profile derivation.

---

## 8. Generic/future schemas

The generic semantic schema format may continue to describe dimensions with `minimum > 0` for future/reference purposes.

Such a schema may be structurally valid.

It is not automatically eligible for the current RSK consumptive budget profile.

A future execution profile that intentionally supports non-zero minima would need to define, qualify, and test its state/arithmetic semantics explicitly, including how complete authority exhaustion is represented without widening authority.

---

## 9. Relationship to semantic execution profile

`RSK_SEMANTIC_EXECUTION_PROFILE_V0_1` derives runtime semantic identity only after the resource schema passes current-profile eligibility.

Therefore a resource scheme with `minimum > 0` cannot produce the current semantic execution profile merely because:

- its schema digest is valid;
- its numeric IDs are valid;
- its quantities fit in `u64`;
- it uses `sum` and `exact` arithmetic.

Zero-representable exhaustion is part of current runtime eligibility.

The current abstract golden resource schema already uses zero minima, so this hardening does not change the existing adapter-table or semantic-execution golden digests.

---

## 10. Relationship to Rust structural validation

The Rust structural path already validates the derived result in `checked_remaining()`.

This contract makes that behavior normative rather than incidental and brings the Python reference semantics into alignment.

Future Rust refactors MUST preserve:

```text
checked subtraction
+ exact scheme binding
+ derived-result validation
```

for authority-bearing remaining-resource values.

---

## 11. Required invariants

The implementation/test/formal program SHALL target at least:

1. `ValidInputsDoNotImplyValidDerivedRemainder`
2. `DerivedRemainingMustValidateUnderSameScheme`
3. `CurrentBudgetProfileRequiresZeroMinimum`
4. `FullConsumptionProducesValidZeroRemainder`
5. `BelowMinimumDerivedRemainderFailsClosed`
6. `ExhaustionCannotClampUpward`
7. `ExhaustionCannotResetConsumption`
8. `ZeroRemainingCannotMintNewAuthority`
9. `CurrentSemanticExecutionProfileRejectsPositiveMinimum`
10. `GenericPositiveMinimumSchemaDoesNotImplyCurrentRuntimeEligibility`
11. `PythonRustRemainingSemanticsConverge`
12. `ZeroExhaustionHardeningDoesNotChangeExistingZeroMinimumGoldenIdentity`

---

## 12. Adversarial cases

At minimum test:

```text
minimum=10, limit=20, consumed=15 -> derived 5 -> DENY
minimum=0,  limit=20, consumed=20 -> derived 0 -> VALID
minimum=0,  limit=20, consumed=21 -> underflow -> DENY
minimum>0 current runtime profile -> DENY before profile derivation
```

Also test:

- multi-dimension exhaustion where only one dimension reaches zero;
- optional-dimension vectors with identical dimension sets for arithmetic;
- derived result above maximum due to future arithmetic operations;
- attempted upward clamping;
- attempted omission of an exhausted required dimension;
- restart/recovery with zero remaining authority;
- grant-generation change that attempts to reset exhausted ancestral counters.

---

## 13. Non-claims

This contract does not:

- define physical resource quantities;
- decide how measurements enter the accounting system;
- implement refills or recovery;
- authorize a new grant after exhaustion;
- establish registry provenance;
- establish runtime artifact identity;
- implement production resource accounting;
- admit any release.

---

## 14. Production admission status

```text
Production admission status: DENIED / NOT YET ELIGIBLE.
```

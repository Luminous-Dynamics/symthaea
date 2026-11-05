# SAWEM Reserve Economics and Collateral Liquidity v1

**Status:** source crosswalk / not an executable market settlement implementation.  
**Parent work:** [#10548 — reserve economics, qualifying capacity and credit-cover liquidity](https://github.com/Luminous-Dynamics/symthaea/issues/10548).  
**Current profile/kernel:** [SAWEM BESS profile v1](README.md), [draft PR #10522](https://github.com/Luminous-Dynamics/symthaea/pull/10522).

## 1. Source and claim boundary

This document crosswalks the **NTCSA South African Wholesale Electricity Market Code Rev 2.3 (February 2026)**:
https://www.ntcsa.co.za/wp-content/uploads/2026/02/Market-Code-v2.3.pdf

Relevant sections are §§9.6.1–9.6.4 (reserve opportunity prices), §§9.8.2–9.8.7 (constrained schedule and reserve-capacity payments), and §15.10 (Required Credit Cover).

This is a profile-level engineering mapping, not a legal opinion or evidence that any provision is approved/effective for a specific market participant. The current SAWEM fixture intentionally keeps effective status, participant admission and commercial operation unknown. Values below must be checked against any approved Rules, amendments, Market Operator implementation documents and effective-date instruments before real project use.

## 2. Four separate reserve opportunity-price rules

All values are per Trading Period and use the Trading Unit's own incremental price curve at its contracted operating point. The price-curve lookup is not a system-wide generic reserve premium.

| Code clause | Reserve product | Opportunity-price expression | Contracted-volume price-curve rule |
|---|---|---|---|
| §9.6.1 | Regulating Up | max(SMP - IP, 0) | At an elbow, use the next increment; at MCR, use increment 3 |
| §9.6.2 | Regulating Down | max(IP - SMP, 0) | At an elbow, use the lower increment; at Minimum Stable Generation, use increment 1 |
| §9.6.3 | Instantaneous Up | max(SMP - IP, 0) | At an elbow, use the next increment; at MCR, use increment 3 |
| §9.6.4 | Ten-Minute Up | max(SMP - IP, 0) | At an elbow, use the next increment; at MCR, use increment 3 |

The formula result is zero where the Trading Unit's relevant energy price exceeds the SMP. The price-curve source/version, contracted operating point, selected increment, boundary comparison and Trading Period must be preserved in the calculation receipt.

**Unit-resolution caution:** the Code describes these as prices in R/MW per Trading Period, while the displayed expression is the difference between SMP and the unit incremental energy price. The v1 Rust reference returns only that signed price differential in the same minor-currency-per-MWh basis as its inputs. It does not infer an interval-duration conversion, multiply by reserve MW, or emit a reserve payment. The final unit conversion and settlement interpretation must be reconciled against the approved Market Operator calculation before a real cashflow uses it.

### Price-curve contract

The day-ahead production/consumption submissions use unit-specific piecewise curves. For a generating curve, the Code's cost increments are bounded by the minimum generation point, three elbow points (the third equal to MCR) and Emergency Level 1. Incremental costs must be nonnegative and non-decreasing across the relevant generating segments.

Implementation requirements:
- represent each curve point and increment with explicit MW and price units;
- define boundary equality/tolerance explicitly and deterministically;
- use different elbow-selection functions for regulating-up/instantaneous/ten-minute vs regulating-down;
- preserve the exact source price curve and Trading Unit operating point;
- treat missing curves, stale curve revisions or unknown contracted operating points as indeterminate;
- do not compute opportunity prices for uncontracted reserve capacity.

Do not collapse the four formulas into one generic reserve-price expression. Regulating Down reverses the subtraction and uses a different elbow rule.

## 3. Capacity qualifying for reserve payment is not physical headroom

Section 9.8.3 says reserve-capacity payment is available only for contracted reserve capacity. The System Operator may define category-specific performance criteria and applies a performance adjustment factor where required. Capacity qualifying for payment is limited by the Trading Unit's effective availability; actual energy and whether reserve/balancing service was used can also affect the qualifying amount.

Separate at least these quantities:

1. **Physical capability:** what the asset model says can be delivered under SOC, power, temperature, response, transition, network and warranty constraints.
2. **Declared availability:** the participant's submitted availability for that Trading Period/product.
3. **Effective availability:** the availability state used in the applicable market/operator calculation.
4. **SO-certified capability / performance:** operator evidence and any applicable category-specific performance criterion.
5. **Contracted reserve MW:** the capacity actually contracted for the product and Trading Period.
6. **Capacity qualifying for payment:** the Code-specific calculation using contracted amount, effective availability and operating-point conditions.
7. **Paid amount:** the actual reserve-capacity payment in the relevant statement/invoice/settlement revision.

A 50 MW physical reserve capability does not by itself establish 50 MW of contracted, qualifying or paid reserve.

### Reserve capacity payment formulas

The published formulas are product-specific:

- Regulating Up (§9.8.4): SMP_RURC × Capacity_Qualifying × PerfAdj_RR
- Regulating Down (§9.8.5): SMP_RDRC × Capacity_Qualifying × PerfAdj_RR
- Instantaneous (§9.8.6): SMP_IR × Capacity_Qualifying × PerfAdj_IR
- Ten-Minute (§9.8.7): SMP_MR × Capacity_Qualifying

The Ten-Minute formula in §9.8.7 does not show a performance-adjustment factor in the published equation. Do not add one by analogy with other products.

Qualifying-capacity tests differ by product and participation class. The Code distinguishes generating Trading Units from Demand Side Units in several clauses. A BESS participation model must therefore declare whether the Trading Unit is treated as generation or demand-side consumption and must select the corresponding rule; it must not silently reuse the generator formula for a DSU or vice versa.

## 4. Constrained schedule adjustments are separate cashflows

Section 9.8.2 compensates or charges for changes between the Unconstrained Schedule and Constrained Schedule:
- when constrained schedule increases energy relative to unconstrained schedule, the constrained-purchase payment uses the Trading Unit's incremental price curve;
- when constrained schedule reduces energy relative to the unconstrained schedule, the constrained-sales charge uses the incremental price curve.

Keep the two schedule revisions, Trading Period, energy sign convention, relevant price-curve increment and resulting payment/charge as distinct records. Do not treat a grid constraint as guaranteed congestion revenue. The relevant event may be a payment or a charge.

## 5. Credit cover is a time-dependent liquidity obligation

Section 15.10 requires Required Credit Cover to cover both:
- **Actual Exposure:** unpaid invoices plus amounts in Settlement Statements for which an invoice has not yet been issued;
- **Undefined Potential Exposure:** accrued obligations not yet included in a Settlement Statement and future obligations likely to accrue before a participant could be suspended for default.

The Code requires a daily recalculation. The Market Operator sends the new Required Credit Cover by 17:00 on the business day. Where more cover is required, it issues a Credit Cover Increase Notice by 17:00 the same day; the participant must post the additional cover by 17:00 on the second Working Day thereafter. A participant may satisfy an increase notice by increasing Posted Credit Cover or paying an outstanding invoice early.

Credit cover also remains in place until relevant amounts have been paid, and certain fixed-credit obligations remain through the last Settlement Rerun following deregistration.

### Engineering consequences

A single snapshot such as required cover equals actual exposure plus undefined potential exposure is useful as a synthetic arithmetic example but is **not a substitute for the full Code calculation and its approved parameters**. The market profile must retain:
- source and revision of the actual-exposure and undefined-potential calculations;
- the daily recalculation timestamp and notification receipt;
- Required Credit Cover, Posted Credit Cover, warning threshold, increase notice, amount and due timestamp;
- permitted satisfaction method (new collateral vs early invoice payment);
- collateral provider eligibility, expiry, haircut, drawdown and re-posting events;
- statement/invoice/rerun lineage and amounts still due;
- applicable business-day / Working-Day calendar version.

Do not assume ordinary Monday–Friday arithmetic without a jurisdictional holiday calendar. A synthetic calendar is valid for a test, not a real payment deadline.

## 6. Four-system ownership

- **Mycelix:** authoritative contracts, reserve awards, participant/product status, SO performance notices, settlement statements, invoices, credit-cover notices, collateral provider status, provenance and lifecycle.
- **Sol-Atlas:** point of connection, network limit and unconstrained/constrained locational evidence.
- **Symtropy:** physical response, effective availability evidence, mode transitions, state of charge, degradation and metered response.
- **Symthaea:** price-curve evaluation, product-specific opportunity prices, qualifying MW, reserve payments, constrained-schedule adjustments, dated cashflows, working-capital and collateral stress, and investment decision certificate.

## 7. Required adversarial cases

The machine-readable benchmark corpus includes:
- RESERVE-PAYMENT-QUALIFICATION-001: contracted MW exceeds effective/qualifying MW;
- RESERVE-LOST-OPPORTUNITY-001: product-specific price-curve elbow rule;
- CONSTRAINED-SCHEDULE-COMPENSATION-001: schedule revision changes a payment or charge;
- CREDIT-COVER-LIQUIDITY-TIMELINE-001: daily calculation, notice and post-by deadline.

RESERVE-LOST-OPPORTUNITY-001 maps to the dependency-free Rust generating-price-curve oracle and its known-answer tests. DSU-RESERVE-PRICE-CURVE-001 maps to the separate consumption-curve oracle and its boundary/negative-control tests. DSU-RESERVE-CAPACITY-QUALIFICATION-001 maps to the bounded product-specific DSU qualifying-capacity arithmetic and missing/used-evidence controls. That slice implements the four product-specific formula directions, distinct up/down elbow rules, MCR/EL1 boundaries, unsupported-direction rejection and checked signed arithmetic. It now includes a separate Demand Side Unit (DSU) consumption-price curve lookup using the descending Declared Consumption / first-elbow / second-elbow / third-elbow structure in §9.3(1)(d). It validates non-negative, non-decreasing increments and selects the next non-empty lower-consumption block at elbow boundaries; the third elbow has no remaining reducible block. The DSU oracle is deliberately limited to upward reserve price lookup for the product families with DSU qualifying-capacity clauses in §§9.8.4(3), 9.8.6(4) and 9.8.7(3). It also implements bounded DSU qualifying-capacity arithmetic for the utilized reserve branch (§9.8.3(3)), the not-utilized/different-energy-instruction branch (§9.8.3(4)), and the ordinary not-utilized consumption-curve branch (§§9.8.4(3), 9.8.6(4)); all Regulating Up / Instantaneous branches are capped by effective availability under §9.8.3(6). The used and changed-instruction branches require acceptable-performance evidence; the unchanged-instruction path additionally requires official consumption, the declaration bound and an explicit SO capability assessment. Ten-Minute uses effective available capacity less the third consumption elbow (§9.8.7(3)). Missing path-selection or performance evidence remains indeterminate. These are arithmetic checks over synthetic inputs, not authentication or qualification of real records. The kernel now includes a formula-only reserve-capacity payment function using a separately supplied Market Operator reserve-category price (minor ZAR/MW/Trading Period) explicitly typed to the payment product, independently qualified capacity (milli-MW), and product-specific performance-adjustment state. Regulating Up, Regulating Down and Instantaneous require an explicit System Operator criteria state; §9.8.3(2)'s no-criteria state maps to a unit factor, while established criteria without an authoritative factor remain indeterminate. Ten-Minute rejects any supplied performance factor because §9.8.7 has no adjustment term. Results are exact rational minor-ZAR per Trading Period; no rounding convention is inferred. The function rejects a price typed for a different reserve category and does not derive reserve-category market prices from the unit-level lost-opportunity diagnostic, authenticate supplied inputs, calculate generator qualifying-capacity formulas, implement utilized-service alternatives, or produce live settlement. No source/authority evidence or actual receipt of cash is established. RESERVE-PAYMENT-QUALIFICATION-001 and CREDIT-COVER-LIQUIDITY-TIMELINE-001 remain not_run. CONSTRAINED-SCHEDULE-COMPENSATION-001 maps to the piecewise incremental-price integral described below. All price-curve examples are synthetic; no actual reserve revenue or settlement is computed.

## 8. Nonclaims

This document does not establish that a project is admitted to SAWEM, reserve capacity is contracted, an SO will certify or qualify capability, a reserve payment will be earned, a settlement invoice is final, a credit-cover calculation is live, or collateral can be posted by an actual deadline. All synthetic numbers remain test assumptions.


### Reserve-capacity payment arithmetic boundary

The reserve-category market prices in §9.7 (SMPRURC, SMPDRRC, SMPIRC and SMPMRC) are distinct from the unit-level lost-opportunity values in §9.6 and are stated in Rand per MW per Trading Period. Sections 9.8.4–9.8.7 multiply the corresponding reserve-category price by capacity qualifying for payment; Regulating Up, Regulating Down and Instantaneous also multiply by the System Operator's performance adjustment, while Ten-Minute does not. The formula kernel therefore consumes a separately supplied reserve-category price and never treats the unit-level SMP/IP diagnostic as a reserve-category price. It carries exact rational minor-ZAR amounts without rounding until an authoritative settlement convention is available. This does not solve the §9.6 wording/formula unit interpretation and intentionally does not derive §9.7 system prices from local synthetic price-curve examples.


### Constrained-schedule adjustment oracle — §9.8.2

The Rust reference integrates a submitted stepwise incremental price curve over the signed-MW interval between the exact Unconstrained Schedule (SG) and Constrained Schedule (CG) for the Trading Unit and Trading Period. When CG > SG, the direction is a constrained-purchase payment to the Trading Unit; when SG > CG, the direction is a constrained-sales charge from the Trading Unit. The signed schedule axis preserves negative values for consumption and positive values for generation.

The integration uses milli-MW, an explicit Trading Period duration in minutes, and minor ZAR/MWh, returning an exact rational minor-ZAR amount with denominator 60,000. It does not assume a one-hour Trading Period and does not round. Price segments are contiguous, non-overlapping, and must cover the entire schedule change. An authoritative "not submitted" state invokes the Code's deemed R0/MWh; an unknown submission state remains indeterminate. This is a synthetic arithmetic oracle, not a live settlement implementation or proof that the supplied schedules, price curve, or Market Operator records are authoritative. Section 9.8.2(3) assigns these payments/charges to the System Operator for recovery as an Ancillary Service; the asset-level arithmetic must not be relabelled as guaranteed project revenue.

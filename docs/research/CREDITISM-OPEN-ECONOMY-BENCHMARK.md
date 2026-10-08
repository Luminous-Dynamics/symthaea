# Creditism Open-Economy and External-Constraint Benchmark

## Purpose

Test Creditism when the physical economy depends on goods, services, energy, capital equipment, or liabilities denominated outside the Creditism network.

Current Common Planet transition material explicitly identifies external supply-chain invoices, fuel, equipment, components, taxes, rent/mortgage obligations, and other national-currency claims as constraints on a local Credit circuit. It also describes a proposed diversified national-currency basket as an in-network reference while stating that it does not guarantee stable purchasing power.

Sources:
- https://common-planet.org/creditism/transition
- https://common-planet.org/creditism

Recent stock-flow-consistent open-economy research similarly treats the rest of the world, current-account dynamics, FX valuation effects, and external constraints as explicit parts of the model rather than residuals.

Source:
- https://www.levyinstitute.org/publications/opensimplest-the-smallest-sfc-open-economy-model/

## 1. Three monetary domains

Model separately:

- `DomesticCredit`;
- `ExternalCurrency`;
- `ReferenceBasket`.

Do not treat a reference basket as an external settlement asset.

## 2. External constraint

Every imported item should bind:

- external supplier;
- external currency/unit;
- quantity;
- external price;
- settlement requirement;
- conversion mechanism;
- FX rate/profile;
- domestic resource used to obtain external settlement;
- payment status;
- external claim/liability where applicable.

The model must not allow domestic Credit issuance to satisfy an external liability unless an explicit conversion/settlement mechanism exists.

## 3. Export constraint

External purchases of domestic goods create a potential source of external settlement capacity only through an explicit export/receipt event.

Never model:

domestic production -> automatic foreign purchasing power

without an external demand and settlement mechanism.

## 4. FX valuation

Hold physical positions constant and change the FX rate.

Require:

external-claim revaluation != domestic transaction flow

and preserve:

quantity change
vs
price/FX revaluation
vs
cash settlement.

## 5. Import-energy fixture

Construct a synthetic economy where a critical production process requires an imported energy input.

Stress:

- external currency appreciation;
- domestic Credit expansion;
- foreign supply disruption;
- domestic productivity improvement;
- export-demand collapse;
- FX conversion outage.

Measure separately:

- physical output;
- external purchasing capacity;
- domestic Credit stock;
- unmet external settlement;
- rationing;
- substitution;
- reserves/buffers where explicitly modeled;
- production loss.

This prevents domestic monetary abundance from being mistaken for external resource sovereignty.

## 6. Trade-balance benchmark

Compare worlds with identical domestic Credit rules but different export competitiveness and import dependence.

Keep current-account and financial-account effects distinct.

Do not infer that a domestic Credit deficit or surplus has the same meaning as a conventional government fiscal deficit or foreign-currency balance.

The exact sector/accounting mapping belongs to the named Mycelix profile.

## 7. Foreign-denominated debt fixture

Create a borrower or community with an external liability in currency X.

Stress domestic Credit and FX separately.

Required distinction:

domestic Credit issuance != external debt-service capacity

A world may have abundant domestic purchasing capacity while remaining externally constrained.

## 8. Currency-basket benchmark

Current Common Planet transition material proposes a diversified reference based on leading national currencies for in-network exchange.

Test:

- one constituent currency shock;
- correlated multi-currency shock;
- basket composition change;
- stale basket observation;
- external market price diverging from internal reference;
- domestic scarcity diverging from external FX movement.

The reference basket should remain a declared information/valuation profile, not an implicit guarantee of purchasing power.

## 9. External settlement outage

Temporarily disable the external conversion/settlement mechanism.

Domestic Credit systems may continue internally.

Expected research question:

which internal activities remain functional, and which fail because they require imported inputs or external claims?

This is especially important for energy, semiconductors, machine tools, pharmaceuticals, and other hard-to-substitute inputs.

## 10. External power concentration

Measure whether external settlement becomes a new concentration surface.

Track:

- access to foreign currency;
- control of export channels;
- import licenses/capacity;
- foreign supplier concentration;
- FX conversion authority;
- strategic inventory control;
- external financing dependence.

Removing domestic financial accumulation does not imply removal of external dependency.

## 11. Positive controls

Include:

- domestically self-contained economy;
- productive export boom;
- stable FX with diversified imports;
- successful import substitution;
- foreign investment with clear external settlement;
- benign currency-basket fluctuation.

These prevent the benchmark from encoding autarky as the desired outcome.

## 12. Required outputs

Return separate:

- domestic Credit stocks/flows;
- external currency positions;
- current account;
- external financial claims/liabilities;
- FX rates and revaluations;
- import dependency;
- export receipts;
- strategic-input dependency;
- domestic production;
- external settlement failures;
- uncertainty;
- governance/authority concentration.

## 13. Claim ceiling

A PASS establishes only accounting and behavioral behavior under the frozen synthetic open-economy profile.

It does not establish monetary sovereignty, trade independence, currency stability, or real-world transition feasibility.
# SYM-FIN-SYS-003 synthetic financing-regime oracle v1

Issue: #5607
Status: conformance corpus only; no empirical claim and no financial/policy authority.

## Purpose

Freeze the first independently inspectable synthetic corpus for Minsky-style financing-regime and financial-fragility qualification before applying those labels to real institutions, sectors, countries, or the global financial system.

The benchmark runtime must separate **solver-visible observations** from **evaluator-only oracle labels**.

```text
public fixture
-> isolated solver
-> candidate result

hidden oracle
-> evaluator only
-> comparison receipt
```

Production/classifier code and solver-visible benchmark inputs must never depend on the oracle path.

## Paths

- `fixtures/fin_sys_v1/public.json` — observations the solver is allowed to consume.
- `fixtures/fin_sys_v1/oracle.json` — hidden expected dispositions and construction facts for the evaluator.

The two files share only a neutral `fixture_id` join key. Fixture identifiers must not encode expected regime, fraud, positive/negative-control status, or pass/fail outcome.

## V1 classification profile

This corpus tests a deliberately narrow Minsky-style profile for synthetic subjects whose debt-service mechanics are known by construction.

Conceptual dispositions:

- `HedgeFinanceCandidate`: available cash flow covers interest and principal due under the fixture horizon.
- `SpeculativeFinanceCandidate`: available cash flow covers interest, but principal due requires refinancing/rollover.
- `PonziFinanceCandidate`: available cash flow does not cover interest; additional borrowing, capitalization, or asset realization is required under the frozen fixture.
- `MixedRegime`: the aggregate contains materially different sub-regimes that must not be collapsed to one homogeneous label.
- `Indeterminate`: required classification coordinates are deliberately absent or conflicting.
- `UnsupportedProfile`: the selected profile is not justified for the subject.

These are profile-relative analytical states, not moral/legal judgments.

```text
Minsky Ponzi finance != criminal Ponzi fraud
```

Literal synthetic fraud is represented by a separate evaluator-only proposition.

## Runtime isolation

A valid benchmark run supplies the solver only the selected public fixture bytes plus the frozen profile semantics that any ordinary implementation is entitled to know. It must not expose:

- `oracle.json`;
- expected disposition fields;
- evaluator rationale;
- fraud ground truth;
- control-status metadata;
- hidden construction facts not present in the public fixture;
- prior candidate results on the held-out generation.

Repository availability alone is not a held-out theorem. A future qualifier must materialize a solver input capsule that excludes evaluator-only paths and bind that capsule digest into the run receipt.

## Numeric representation

V1 uses decimal strings and basis-point/integer-like strings rather than binary floating-point values. Units are explicit per fixture. Later executable types may use a reviewed decimal/fixed-point representation.

No `NaN`, infinity, or sentinel number denotes missingness. Missing observations are omitted and declared in `missing_coordinates`.

## Public observation fields

Each public fixture contains:

- neutral fixture identity and subject class;
- analysis horizon;
- unit profile;
- solver-visible financial coordinates;
- source-family references;
- missing-coordinate list;
- declared shocks where the solver is allowed to observe them;
- limitations.

The public file intentionally contains no expected regime, fraud flag, crisis outcome, control role, or pass/fail answer.

## Oracle fields

The evaluator-only file contains:

- expected financing-regime disposition;
- construction facts used to independently derive that disposition;
- literal-fraud predicate kept separate from Minsky regime;
- expected fragility-coordinate directions;
- allowed alternate dispositions where the public evidence is intentionally insufficient;
- rationale and claim ceiling.

## Required invariants

```text
high debt != Ponzi finance
normal refinancing != Ponzi finance
Minsky Ponzi finance != criminal Ponzi fraud
healthy conventional finance must remain representable
missing required coordinate -> no fabricated classification
one aggregate label must not erase heterogeneous sub-sectors
fixture identity must not reveal evaluator label
```

## Fixture set v1

Neutral fixture identities:

1. `corp_cashflow_cover_001`
2. `corp_refinance_gap_001`
3. `corp_interest_shortfall_001`
4. `household_missing_income_001`
5. `sovereign_long_maturity_001`
6. `scheme_payout_dependency_001`
7. `economy_sector_heterogeneity_001`
8. `bank_liquidity_buffer_001`

The solver derives any disposition from the public observations; this document does not enumerate the expected label for any individual fixture.

## Evaluation

Keep dimensions separate:

- financing-regime classification correctness;
- abstention correctness;
- false-positive rate on healthy controls;
- fraud-vs-Minsky separation;
- aggregation-level correctness;
- source-independence handling;
- fragility-coordinate direction correctness;
- explanation traceability;
- authority/nonclaim discipline.

No aggregate `financial_system_score`.

## Leakage rule

Any solver-visible dependency, cache, generated source, prompt, embedding, model input, threshold selection, or training artifact containing evaluator-only oracle labels contaminates the benchmark generation.

```text
oracle label visible to solver
-> benchmark generation invalid
```

After a benchmark generation closes, oracle outcomes may inform a **new** development generation only if that new generation receives a fresh held-out corpus identity.

## Nonclaims

Passing this corpus does not establish real-world crisis prediction, fraud detection, investment advice, policy advice, universal validity of Minsky's framework, classification accuracy for any real country/institution, or superiority of any financial system.
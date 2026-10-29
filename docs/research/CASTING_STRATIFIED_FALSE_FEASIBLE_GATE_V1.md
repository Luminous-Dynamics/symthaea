# CASTING-0037: Stratified Critical-Family False-Feasible Risk Gate (v1.1)

**Status:** proposed code and tests added to the open promotion-readiness PR. CI status must be checked on the exact current head; this document does not assert that tests passed.

## Failure mode addressed

A pooled false-feasible rate can hide risk concentrated in a critical alloy, geometry, process window, or distribution-shift family. The prior gate recomputed an exact one-sided Clopper–Pearson upper confidence bound across all predicted-feasible validation records, but did not require a separate bound for each predeclared critical family.

This update introduces a **contract schema version 1.1.0** and requires two explicit sample floors:

- `minimum_validation_units_per_critical_family`: minimum independent held-out records in each declared critical family.
- `minimum_predicted_feasible_per_critical_family`: minimum predicted-feasible validation records in that family, which is the denominator for false-feasible risk.

Each family receives its own predicted-feasible denominator, false-feasible count, observed rate, one-sided exact upper bound at the contract confidence level, and pass/fail/inconclusive decision. Family-level decisions are included in the digest-bound gate receipt. A family with no predicted-feasible trials or insufficient per-family denominator is **inconclusive**, not implicitly safe. A well-powered family whose upper bound exceeds the contractual maximum fails promotion.

The tests include a dilution adversary: 20 false-feasible outcomes among 300 predictions in one critical family and zero in another. At 600 pooled predictions, the pooled upper bound can remain under a 5% cap while the concentrated family bound exceeds it. The gate must reject that candidate. The test also covers an underpowered family whose pooled bound passes, and unequal family denominators.

## Sample-size interpretation

For a one-sided 95% Clopper–Pearson upper bound with zero observed false-feasible events, 59 independent predicted-feasible trials are approximately the minimum needed for the bound to fall at or below 5%. This is a planning floor, not evidence that units are independent. Correlated pours, repeated measurements from one casting, or many geometries derived from one base case must not be counted as independent units. The protocol must define the physical/experimental unit before outcomes are observed.

A 5% upper bound is only a research-stage screen; it is not sufficient assurance for every safety-critical application. The allowed risk and confidence level must be selected before looking at results, based on the downstream consequence of a false-feasible recommendation.

## Decision semantics

- **Pass:** both the minimum family record count and minimum predicted-feasible denominator are met, and the family-specific one-sided upper bound is within the contract.
- **Fail:** a sufficiently sampled critical family has an upper bound above the contract limit.
- **Inconclusive:** a family is missing, has no predicted-feasible predictions, or lacks either sample floor. Existing aggregate checks still run; a known failure cannot be hidden by inconclusive evidence elsewhere.

This is a promotion-readiness gate, not physical safety certification. It relies on the validation manifest's assertion that holdout outcomes were unsealed only after policy freeze and that validation units are independent. It cannot verify those claims from hashes alone.

## Next experiments

1. Add family strata before collecting outcomes: alloy, mold/process route, geometry topology, boundary-condition regime, and OOD shift.
2. Report per-family calibration, false-feasible risk, abstention coverage, and risk-versus-coverage curves; include a reject/defer action rather than forcing a prediction.
3. Run a larger multi-fidelity bake-off against reproducible MF-SCBO, out-of-the-loop historical multi-fidelity BO, RESCUE/causal-prior methods, descriptor nearest-neighbor transfer, and low-fidelity-only baselines under equal simulator-call and wall-clock budgets.
4. After synthetic and public-analytic adversarial tests, repeat with simulator-backed evidence and then prospective physical castings. Keep each evidence class separately labeled and do not let synthetic sample count stand in for physical validation.

## Claim ceiling

This change improves evidence stratification and promotion gating. It does not demonstrate better casting quality, lower foundry cost, superiority to commercial optimization, or physical validity. Those claims require a preregistered, independently held-out benchmark and measured simulator/physical outcomes.

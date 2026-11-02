# AR-5B25: Noncompensatory Guardrail Eligibility v0.1

**Status:** Rust implementation draft; tests authored but not executed.  
**Tracking:** https://github.com/Luminous-Dynamics/symthaea/issues/9976  
**Implementation:** `crates/domains/symthaea-architecture-engineering/src/noncompensatory_guardrails.rs`  
**Dependency base:** AR-5B23 / PR #9999 (dependent draft).

## 1. Scope

V1 verifies a fixed randomized policy mixture against a complete declared set of global and local expected-value guardrails using precomputed simultaneous one-sided confidence bounds.

It is deliberately a separate gate from the utility objective and from statistical Type-I error control. The API accepts no utility value. A policy cannot compensate for a mandatory guardrail violation with higher reward.

This module does not estimate confidence bounds, authenticate their sources, choose the policy mixture, prove the guardrail roster is complete, or provide operational safety authorization.

## 2. Typed contract

Each requirement specifies:
- unique requirement ID and stable guardrail/metric ID;
- metric-contract digest;
- GLOBAL or LOCAL scope;
- an optional stratum (absent for GLOBAL, required for LOCAL);
- model ID;
- upper-bound or lower-bound direction;
- predeclared threshold;
- minimum sample size.

The request binds:
- the candidate policy-mixture weights;
- expected policy/requirement/global-requirement counts;
- canonical applicable guardrail roster digest;
- calibration/confirmation target-design identities;
- calibration independence declaration;
- linear-mixture estimand declaration;
- one simultaneous guardrail confidence-failure probability;
- each policy × requirement's bound availability, lower/upper confidence limits, sample size and source receipt digest.

Global and local scope are explicit. The caller must supply the full applicable roster; the verifier checks its count and digest against the canonical requirement list. A digest binds bytes but does not establish that the roster is authoritative or complete.

## 3. Decision semantics

For policy weights (q_i) and simultaneous confidence bounds ([L_i,U_i]) for one expected guardrail functional, the linear-mixture bounds are:

[
L_{mix}=sum_i q_i L_i,qquad U_{mix}=sum_i q_i U_i.
]

For an upper-bound guardrail with threshold (	au):

- PASS if (U_{mix}leq	au);
- VIOLATION_ESTABLISHED if (L_{mix}>	au);
- NOT_ESTABLISHED otherwise.

For a lower-bound guardrail:

- PASS if (L_{mix}geq	au);
- VIOLATION_ESTABLISHED if (U_{mix}<	au);
- NOT_ESTABLISHED otherwise.

Missing evidence produces MISSING_EVIDENCE. An explicitly non-estimable row, or a row below its predeclared minimum sample size, produces NOT_ESTABLISHED. Neither can be interpreted as PASS.

Overall result:
- BLOCKED if any guardrail violation is established;
- INDETERMINATE if no violation is established but any mandatory requirement is not established or missing;
- PASS only if every mandatory global and applicable local requirement passes.

An interval straddling a threshold is not treated as proof of harm; it is also not permission to proceed.

## 4. Why the V1 linearity restriction matters

The mixture formula applies only to expected metrics that are linear under the actual randomization/mixture protocol. V1 rejects a caller declaration that the estimand is nonlinear.

Quantiles, CVaR, maxima, nonlinear ratios, pathwise requirements and interaction-sensitive guardrails require separately derived robust constraints. They must not be approximated by this mean-bound adapter without a theorem.

## 5. Confidence and target-law boundaries

The supplied confidence failure probability must cover the full policy × requirement bound family simultaneously. This module records the declared guardrail false-safe confidence failure; it does not prove that the upstream method achieved it.

V1 requires calibration independent of confirmation and `SAME_TARGET_LAW_BY_DESIGN` with matching design digests. The design identity is an assumption/binding, not proof of target-law equality. Bounded transport, dependent data and adaptive guardrail discovery are unsupported.

Guardrail confidence failure is not Type-I alpha. A top-level decision that combines this guardrail gate with the statistical test must compose the distinct failure events according to the claims being made.

## 6. Tests authored

The module defines tests for:
- all-global/all-local PASS;
- established upper-guardrail violation -> BLOCKED;
- confidence interval straddling a threshold -> INDETERMINATE;
- missing policy/requirement evidence -> INDETERMINATE;
- explicitly non-estimable row -> INDETERMINATE;
- below-minimum sample size -> INDETERMINATE;
- utility cannot compensate (utility is absent from the API);
- incomplete global roster rejection;
- nonlinear mixture estimand rejection;
- mismatched target-design rejection;
- invalid mixture weights;
- invalid/non-finite confidence bounds;
- deterministic report-tampering detection.

These tests are authored, not run. No Rust compile, formatting, Clippy, unit-test, CI or benchmark pass is claimed.

## 7. Next work

1. Add an independent guardrail-bound verifier that reconstructs each upstream bound receipt from raw outcomes and the exact calibration method.
2. Connect the verified gate to the policy-frontier LP as hard linear constraints, rather than only post-selection mixture verification.
3. Add simultaneous vector confidence sets for correlated guardrails within each observed policy/model unit.
4. Add target-law transport and non-IID calibration as separately budgeted evidence events.
5. Add longitudinal, delayed-outcome, subgroup roster, informative-missingness and interference controls.
6. Derive separate robust formulations for CVaR/quantile/pathwise guardrails; do not force nonlinear metrics through V1's mean LP contract.

## 8. Industry reference

Spotify Confidence reports a 42% rollback rate after global and local guardrail metrics detect regressions. It describes global metrics required on every experiment and local, feature/surface-specific metrics. Spotify emphasizes defining inferiority thresholds before running the experiment and distinguishing the decision to test from the decision to ship. These are vendor-reported operational practices, not universal standards or independently audited benchmark rates.

Source: Spotify Confidence, 3 August 2026, *42% of Spotify experiments get rolled back. That's the point.* https://confidence.spotify.com/blog/42-percent-rolled-back


## 9. Joint confidence receipt binding

The request now carries one `simultaneous_coverage_receipt_sha256` in addition to each row's source receipt digest. This receipt is intended to bind the complete policy × requirement confidence family, its calibration method, split identity, predeclared family and failure probability. The verifier checks digest shape and includes it in the deterministic report identity; it does not independently verify the upstream statistical derivation or artifact authenticity.

Randomized policy evidence must satisfy the declared assignment design. The caller's no-cross-unit-interference and sampling-unit declarations remain assumptions. The exact independent experimental unit (e.g. cluster, experiment replication, or another justified unit) is separately specified by the upstream sampling-unit contract.

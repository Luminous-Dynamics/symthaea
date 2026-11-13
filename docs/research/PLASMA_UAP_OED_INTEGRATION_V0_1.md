# Plasma decision-focused UAP-OED integration V0.1

**Date:** 13 November 2026  
**Plasma application:** [PLASMA-OED-001 #11907](https://github.com/Luminous-Dynamics/symthaea/issues/11907)  
**Generic selector owners:** UAP-OED-001 #11050, UAP-OED-002 #11235, UAP-OED-003 #11237  
**Implementation dependency:** [UAP Observatory draft PR #10802](https://github.com/Luminous-Dynamics/symthaea/pull/10802)  
**Branch:** `research/plasma-uap-oed-integration-v0-1`  
**Status:** synthetic integration example; not compiled or executed

## 1. Purpose and ownership

This example composes the existing `symthaea-uap-observatory` experiment-design API with the PA6 durable-bonding profile. It does not add an optimizer, information-gain engine, covariance engine or exact-subset oracle.

It calls the existing functions directly:
- `greedy_budgeted_design`;
- `exhaustive_budgeted_design`;
- `greedy_correlated_budgeted_design`;
- `exhaustive_correlated_budgeted_design`.

The two paths share the same candidate actions, costs, priors, target dimensions and budget. Their only material difference is whether selected-set residual covariance is modelled.

## 2. Frozen synthetic profile

Target state:
`initial bond strength, aged-bond state, substrate moisture`

Nuisance state:
`shared sample-preparation offset`

Candidate actions:

| Action | Fidelity | Cost units | Product endpoint |
|---|---:|---:|---|
| WCA_SCREEN | F0 | 1 | Wettability proxy |
| FTIR_CHEMISTRY | F1 | 3 | Surface chemistry proxy |
| MOISTURE_CONTENT | F1 | 2 | Hygroscopic substrate state |
| LAP_SHEAR | F2 | 8 | Initial bond strength |
| AGED_LAP_SHEAR | F3 | 14 | Aged bond / retention |

The experiment budget is 26 artificial cost units. Two synthetic strata (dry and conditioned/hygroscopic) have equal scenario weight.

The residual covariance matrix contains:
- WCA / FTIR common preparation correlation: (ho=0.95);
- initial / aged lap-shear correlation: (ho=0.30);
- unit variance on the diagonal.

The covariance is positive definite (eigenvalues 0.05, 0.70, 1.00, 1.30 and 1.95). These are synthetic challenge parameters, not empirically estimated laboratory covariance.

## 3. Why this integration matters

The independent-row greedy path treats every candidate's noise as independent. Under this constructed profile, it spends budget on a second, highly correlated surface proxy and omits the destructive initial lap-shear endpoint.

The selected-set correlated path discounts redundant information from the shared preparation block and selects the initial plus aged mechanical tests alongside the moisture measurement.

This is an application-specific example of a general principle:

> Distinct measurement IDs do not imply independent information. A cheaper proxy can look valuable when common-mode residual covariance is ignored, while an independent destructive endpoint may be the measurement that actually resolves the product decision.

## 4. Expected source-side linear-algebra result

The UAP source implements target-marginal information gain as the difference between posterior and prior target-marginal log determinants after nuisance marginalization. An independent matrix calculation using that same objective yields the following expected known-answer values:

| Selector | Selected actions | Cost | Target EIG (nats) | PA6 mandatory endpoints |
|---|---|---:|---:|---|
| Independent greedy | WCA, MOISTURE, FTIR, AGED_LAP | 20 | 5.279431608 | Missing LAP_SHEAR |
| Independent exhaustive oracle | WCA, MOISTURE, LAP, AGED_LAP | 25 | 5.618701237 | Complete |
| Correlated greedy | WCA, MOISTURE, LAP, AGED_LAP | 25 | 5.705573945 | Complete |
| Correlated exhaustive oracle | WCA, MOISTURE, LAP, AGED_LAP | 25 | 5.705573945 | Complete |

Interpretation:
- independent greedy has an approximately 0.339269630-nat gap to its own small-set oracle;
- correlated greedy matches its small-set oracle for this fixture;
- correlated selection is not permitted to accept a product. The example only checks that its *proposed portfolio* covers the frozen mandatory endpoint set.

The example enforces:
- independent greedy -> `ABSTAIN_MANDATORY_ENDPOINTS_MISSING`;
- correlated greedy -> `PORTFOLIO_READY_FOR_PHYSICAL_REVIEW`.

"Ready for physical review" is not product acceptance, physical execution authorization, or a claim that the selected measurements have been performed.

These numbers are expectations derived from an independent source-side matrix calculation, not Rust program output. The Rust example contains known-answer assertions so the result can be checked when a supported Rust toolchain is available.

## 5. Reproduce when the toolchain is available

From the Symthaea repository root on this branch:

```sh
cargo run --manifest-path crates/domains/symthaea-uap-observatory/Cargo.toml --example plasma_multifidelity_design
```

The UAP Observatory source branch is itself an open draft PR dependency. This integration branch is based on `research/causal-temporal-physics-v0.1`; it is intentionally stacked on that dependency and must not be treated as independently mergeable into main before the parent implementation is reconciled.

## 6. What the current UAP API provides—and what the plasma adapter still needs

The generic `BudgetedDesignReport` already provides:
- selected option IDs and total cost;
- budget and remaining budget;
- expected and worst-case target information gain;
- expected and worst-case joint-state information gain;
- per-scenario scores;
- selection steps and marginal gain per cost.

The current generic report does not, by itself, constitute a full provenance receipt for downstream engineering decisions. A production adapter still needs to bind the exact report to:
- the exact scenario/prior and candidate-option manifest digests;
- the selected-set covariance source and digest;
- action catalogue revision and cost/latency/sample semantics;
- model-adequacy / OTHER disposition from OLI;
- the exact UAP source revision and algorithm configuration;
- the plasma material profile and mandatory endpoint policy.

This should be a thin, versioned evidence envelope around the generic report—not a second optimizer or a competing generic receipt authority. Physical observations, calibration and material-state lineage continue to come from MAT-013 / MAT-INTERFACE and the existing observation/metrology owners.

### 4.1 Concrete selector composition and exact known-answer replay

The Rust example is registered as `plasma_multifidelity_design` in the UAP Observatory crate manifest. It calls both the independent-row and correlated selected-set UAP APIs plus their exhaustive small-set oracles. It also emits the ownership chain, application-profile ID, policy ID, scenario/candidate/covariance manifest IDs, Git blob IDs for the exact generic API files, standard-family identifiers and an explicit no-authority claim.

Independent dense-matrix replay of the exact hard-coded arrays gives:

| Selector | Selected action IDs | Cost | Target EIG (nats) | Mandatory F2/F3/moisture coverage |
|---|---|---:|---:|---|
| Independent greedy | WCA_SCREEN; MOISTURE_CONTENT; FTIR_CHEMISTRY; AGED_LAP_SHEAR | 20 | 5.279431608 | Missing LAP_SHEAR |
| Independent exhaustive | WCA_SCREEN; MOISTURE_CONTENT; LAP_SHEAR; AGED_LAP_SHEAR | 25 | 5.618701237 | Complete |
| Correlated greedy | WCA_SCREEN; MOISTURE_CONTENT; LAP_SHEAR; AGED_LAP_SHEAR | 25 | 5.705573945 | Complete |
| Correlated exhaustive | WCA_SCREEN; MOISTURE_CONTENT; LAP_SHEAR; AGED_LAP_SHEAR | 25 | 5.705573945 | Complete |

The independently computed covariance eigenvalues are 0.05, 0.70, 1.00, 1.30 and 1.95, so the frozen covariance is positive definite. The high WCA/FTIR preparation correlation (ρ=0.95) makes the two proxies partly redundant. The correlated selector therefore does not spend its remaining budget on FTIR; it selects the missing initial lap-shear endpoint instead.

The independent greedy selector's gap to its own exact oracle is 0.339269630 nats. The correlated greedy selector matches its exact oracle for this specific five-action fixture. This is a *known-answer synthetic composition result*, not an industry benchmark win. A different covariance/prior/scenario can reverse the comparison.

The example asserts these selected IDs, exact-oracle relationships and endpoint-coverage dispositions. It fails loudly if the frozen source code produces a different result when it is eventually run with the parent UAP implementation.

The output IDs identify the exact generic source blobs used:
- budgeted design: `ed3d0286e706eaee9bb670e7e5e62f8ff01894b8`;
- correlated design: `4aebc5e80afe3b2258db04b441cda072c4edde1d`;
- UAP crate API: `aafaffcbc7d9aa1cef462ace453d2d173136b8d8`.

The candidate rows, priors, covariance and costs are embedded in the versioned Rust example. The Git blob ID of the complete example itself is tracked by the branch/PR, not self-reported by the example. No optimizer output, model-adequacy receipt, empirical covariance or physical evidence is fabricated.


.
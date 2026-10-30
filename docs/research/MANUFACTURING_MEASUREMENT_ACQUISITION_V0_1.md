# Manufacturing Measurement Acquisition v0.1

**Status:** research protocol and selector extension; no shop-floor result or SOTA claim.

## Objective

Connect Symtropy's manufacturing digital thread to Symthaea's expected-information-gain (EIG) experiment selector. The manufacturing system supplies admissible measurement candidates, calibrated standard uncertainties, and comparable acquisition costs. Symthaea ranks only those candidates that pass the manufacturing system's hard eligibility gate.

The core shift is from **"measure where model disagreement is largest"** to **"measure where expected information gain is largest after accounting for the actual metrology method, cost, and admissibility."**

## Cross-repository responsibilities

### Symtropy: establish admissibility and evidence

Symtropy owns the engineering facts and gates:

- product, serial, component and lot genealogy;
- approved route revision and required process/test steps;
- test-plan revision and numerical acceptance limits with units;
- actual measurement value, standard uncertainty and instrument identity;
- calibration certificate identity, validity window, status and evidence digest;
- safety/resource eligibility and a normalized acquisition-cost basis;
- signed/hashed evidence receipts and final product conformity decision.

eligible=true is not a model prediction. It may be emitted only after the authoritative route, calibration, equipment, safety and resource checks have passed. Missing, expired, invalid, untraceable or out-of-scope evidence means eligible=false or an explicit unknown state; unknown must never be coerced to true.

### Symthaea: rank eligible experiments

Symthaea consumes candidate-specific predictive distributions and independently supplied noise/cost metadata. The new API,
select_experiment_by_information_gain_per_cost_with_noise_and_eligibility,
supports:

- a per-candidate observation standard deviation (heteroscedastic measurement noise);
- a hard eligibility predicate evaluated before any cost/noise/prediction callback;
- expected mutual information in nats divided by comparable cost;
- deterministic tie behavior and abstention when no eligible positive-information candidate exists.

The existing shared-noise selector remains available as a matched baseline. The acquisition API does not validate certificates, acceptance plans, physical safety, model calibration, or posterior probabilities.

## Candidate interface (contract proposal)

A serialized adapter between repositories should preserve at least:

| Field | Meaning |
|---|---|
| candidate_id | Stable unique measurement action ID |
| product_family, product_id, serial_number | Product scope |
| component_id, component_lot_id | Genealogy scope |
| route_revision, test_plan_revision | Approved process/test context |
| operation_code, test_id, measurand, unit | Exact measurement semantics |
| instrument_id, calibration_id, calibration_status | Metrology identity and validity state |
| observation_std_dev | Standard uncertainty in the measurand's unit |
| acquisition_cost, cost_unit | Comparable measured cost (for example labor-minutes or USD at a declared price year) |
| eligible, eligibility_receipt_id | Hard external gate and its auditable receipt |
| source_evidence_id, source_evidence_sha256 | Input artifact reference and digest |
| model_version, model_evidence_id | Versioned predictive model and evidence receipt |

The prediction closure must use a common measurand and unit for every hypothesis within one acquisition comparison. Candidate-specific standard deviation must describe the actual measurement process, not an uncertainty guessed from the spread of model predictions. A separate calibration/measurement-system analysis owns that estimate.

The current Symtropy ledger stores event evidence IDs and digest syntax; those references still require actual artifact hashing, provenance and authorization before they can establish eligibility. The candidate interface is a contract, not an assertion that the current ledger already emits all fields.

Symtropy's crossed MSA implementation is tracked in [PWR-TX-MFG-003 #2969](https://github.com/Luminous-Dynamics/symtropy/issues/2969). Its v0.1 balanced ANOVA diagnostic can provide a GRR standard deviation for a specific study when the measurement-system study is independently reviewed and accepted for the intended decision. Its guidance bands are not a universal acceptance rule. Symthaea must not infer calibration validity, MSA adequacy or eligibility from an EIG score; it receives observation standard deviation and eligibility only as externally controlled inputs.

## Benchmark

Compare at identical held-out product/lot/campaign pools and identical total acquisition budgets:

1. Uniform random among eligible candidates.
2. Raw prediction variance.
3. Common-noise EIG.
4. Candidate-noise EIG.
5. Candidate-noise EIG per cost.
6. Candidate-noise EIG per cost with hard eligibility filtering.

The evaluator's oracle may score selected actions after selection but must not leak outcomes, labels, acceptance states, calibration uncertainty, or hidden parameters into policy ranking except for the explicitly permitted eligibility bit and measurement metadata.

### Primary metrics (reported separately)

- decision-relevant information gained per cost;
- regret against the within-pool evaluator oracle;
- cost and number of measurements to resolve a predeclared decision;
- product/lot-level predictive and structural recovery;
- top-k recall and best-feasible candidate recall;
- calibration / OOD / measurement-method strata;
- ineligible-selection attempts (target: exactly zero);
- false-safe / invalid-release decisions (target: exactly zero);
- abstention and unresolved-decision rates;
- acquisition overhead and actual measurement cost.

Pair all policies on the same product lot, campaign, initial evidence, candidate pool and budget. The statistical unit is the independent lot or campaign, not repeated measurements from the same product. Use cluster bootstrap or a hierarchical model over lots/campaigns. Keep engineering-family holdouts intact; do not split repeated observations from one serial or lot across train and test.

### Advancement targets (targets, not achieved results)

- Zero ineligible actions selected in all frozen negative controls.
- No safety, calibration or conformity regression versus the deterministic rule-based baseline.
- At least 10% reduction in verified acquisition cost to a predeclared decision, or at least 5 percentage-point improvement in held-out decision-relevant recovery, with paired 95% cluster-level intervals excluding zero.
- Candidate-level uncertainty coverage calibrated at its declared nominal level on independent lots/campaigns.
- OOD and missing-evidence cases cause explicit abstention rather than silent extrapolation.
- No metric improvement is accepted if it is purchased by increasing unresolved or false-safe outcomes.

These are internal pilot thresholds, not published industry averages, certification requirements or demonstrated performance.

## Required challenge cases

- one candidate has large model disagreement but an expired calibration;
- highest-information action is unavailable or outside the approved route;
- low-noise measurement vs high-signal but high-noise measurement (rank reversal);
- zero-information eligible candidate pool (must abstain);
- invalid/non-finite uncertainty or cost;
- missing, stale, tampered or untraceable source evidence;
- conflicting test-plan revisions;
- duplicate/equivalent hypotheses that distort a uniform prior;
- measurement drift, heteroscedastic and correlated errors;
- omitted physics / candidate-pool misspecification;
- shift in material lot, supplier, station, operator or measurement method.

## Known limitations

The current EIG approximation still uses a uniform prior over the supplied live hypothesis instances and independent Gaussian measurement noise for each candidate. Per-candidate standard deviation adds heteroscedasticity between candidate experiments, not correlated measurement noise, non-Gaussian tails, nuisance-parameter integration or prior weights over equivalence classes. The selector is a ranking primitive, not a certified controller.

Hard eligibility is supplied by the caller; the selector cannot make an unsafe or expired measurement safe. It also cannot authorize product release. Those decisions remain with the approved quality system and accountable personnel.

## Research references

- NISTIR 8620 (2026), *Digital Twins Workshops Summary Report*: interoperability, verification/validation/uncertainty quantification, cybersecurity and workforce readiness remain explicit manufacturing digital-twin challenges. https://doi.org/10.6028/NIST.IR.8620
- ISO 23247-6:2026, *Digital twin composition*: identifies integrated, unified and federated composition to support reusable and interoperable manufacturing twins. https://www.iso.org/obp/ui/?_escaped_fragment_=iso%3Astd%3Aiso%3A23247%3A-6%3Aed-1%3Av1%3Aen
- Callahan & Catanach (AISTATS 2026), *Robust Experimental Design via Information Gain and Error Amplification*: information-maximizing acquisition under misspecification needs explicit robustness scrutiny. https://proceedings.mlr.press/v300/callahan26a.html
- Tang et al. (AISTATS 2026), *R-IDeA*: robust information-driven experimental design under model misspecification. https://proceedings.mlr.press/v300/tang26d.html

This protocol is intended to connect to Symtropy's PWR-TX-MFG-001 work and remains research-only until the candidate eligibility receipt, evidence authenticity, route/test-plan verifier and independent physical validation are implemented.

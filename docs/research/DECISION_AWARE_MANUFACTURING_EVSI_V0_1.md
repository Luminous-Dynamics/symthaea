# Decision-Aware Manufacturing Acquisition: Robust EVSI v0.1

**Status:** selector implementation and synthetic mathematical tests; no live factory trial, customer policy, standards claim or measured production benefit.

## Research shift

Expected information gain (EIG) measures reduction in uncertainty about hypotheses. Manufacturing decisions rarely optimize entropy itself. The relevant question is whether a measurement can change a product/design/process decision and improve its expected outcome.

A measurement that distinguishes two hypotheses which prescribe the same action can have substantial EIG but zero decision value. Conversely, a lower-entropy measurement may be valuable if it separates an acceptable state from a failure state and changes the choice between release, rework or escalation.

This implementation adds an expected value of sample information (EVSI) primitive while retaining EIG and finite-scenario maximin EIG as explicit baselines.

## Decision model

Let:

- (H) be the finite hypothesis set;
- (pi_h) be an externally supplied prior over hypotheses;
- (A) be the approved action set;
- (U(a,h)) be the externally governed utility (or negative loss) of action (a) when hypothesis (h) is true;
- (Y_emid hsimmathcal N(mu(h,e),sigma_e^2)) be the measurement likelihood for experiment (e).

The value before measuring is:

[
V_0=max_{ain A}sum_hpi_h U(a,h)
]

The value after an observation is:

[
V_e=mathbb E_{Y_e}left[
max_{ain A}sum_h P(hmid Y_e,e)U(a,h)
ight]
]

The expected value of sample information is:

[
operatorname{EVSI}(e)=V_e-V_0
]

The selector ranks robust EVSI per cost using the existing external admissibility, representativeness and de-amplification evidence:

[
S(e)=
rac{min_{sinmathcal S}operatorname{EVSI}(emid s)
,R(e),[1-A(e)]}{C(e)}
]

where (mathcal S) is a frozen finite set of discrepancy scenarios, (R(e)in(0,1]) is independently audited representativeness, (A(e)in[0,1)) is independently audited amplification risk, and (C(e)>0) is comparable acquisition cost.

The utility matrix is constant across discrepancy scenarios in v0.1. Scenarios alter the prediction/measurement likelihood; they do not allow the selector to rewrite the policy's action values.

## Symtropy/Symthaea boundary

**Symtropy owns the admissibility and policy evidence.** It must supply a versioned, complete action × outcome utility matrix, its controlled policy source, the approver/authorization evidence, product/test scope, validity interval, test-plan revision and eligibility receipt. It remains responsible for route completion, calibration, measurement-system analysis, conformity rules, safety and release.

**Symthaea owns acquisition ranking.** It computes expected post-measurement decision utility under the supplied policy and ranks already-eligible experiments. It cannot create utility values, infer a release policy, alter an acceptance limit, authorize a product release or turn missing evidence into an eligible action.

A hard eligibility predicate is evaluated before representativeness, amplification risk, cost, measurement uncertainty, predictions or utility callbacks. An ineligible candidate must never be rescued by a high EVSI.

Cross-repository policy contract: [Symtropy issue #2993](https://github.com/Luminous-Dynamics/symtropy/issues/2993).

## Implementation

- `expected_value_of_sample_information`: prior-weighted EVSI for one experiment, finite hypotheses, finite actions, candidate-specific Gaussian standard uncertainty, and a caller-supplied utility function.
- `select_experiment_by_representative_deamplifying_worst_case_evsi_per_cost`: minimum EVSI across a finite discrepancy set, adjusted by independently supplied representativeness and amplification-risk scores, divided by comparable cost.
- Deterministic midpoint Gaussian quadrature uses 256 points over ±8 standard deviations.
- Zero-prior hypotheses are ignored; priors are max-scaled before normalization; duplicate hypothesis representations preserve decision value when their prior mass is split.
- Incomplete/non-finite predictions, invalid prior/noise/cost, non-finite utility, empty scenario sets, unsupported candidates and non-positive worst-case EVSI fail closed.

The core function returns utility units, not nats. Division by cost is meaningful only when all candidate utilities share one approved policy scale and all costs use one comparable basis. Utilities are not normalized automatically.

## Required benchmark

Compare on the same frozen product/lot/campaign pools, decision policy and total acquisition budget:

1. eligible uniform random;
2. raw prediction variance;
3. prior-weighted EIG;
4. finite-scenario worst-case EIG;
5. decision EVSI;
6. representative/de-amplifying worst-case EVSI.

### Primary metrics

- expected and worst-scenario decision loss / utility;
- worst-group decision regret against a frozen evaluator oracle;
- correct engineering decision rate;
- acquisition cost and measurements to reach the predeclared decision;
- frequency of action changes after each acquired measurement;
- false-safe decisions and ineligible selections (target: exactly zero);
- unresolved decisions and abstention rate;
- calibration of measurement/predictive intervals;
- performance by material lot, supplier, station, operator, measurement method, regime and discrepancy family;
- runtime and selector overhead.

Pair policies on identical candidate pools, seeds, prior evidence, policy revision and total measured-cost budget. Statistical inference must cluster at independent manufacturing lot or campaign, not repeated observations. Keep at least one supplier/manufacturer/station family entirely held out.

### Advancement gates (proposed pilot targets, not results)

- zero ineligible selections and false-safe actions in all frozen negative controls;
- robust EVSI worst-scenario regret no worse than finite-scenario maximin EIG;
- at least 10% lower verified acquisition cost to the predeclared decision, or +5 percentage points held-out correct-decision rate versus maximin EIG, with paired 95% lot/campaign-cluster confidence intervals excluding zero;
- no deterioration in measurement/predictive calibration, safety/conformity outcomes, or unresolved-decision rate;
- out-of-domain, missing-policy, expired-policy and incomplete-evidence cases abstain.

These targets are internal research gates. They are not industry averages, regulatory thresholds or evidence that the system currently beats SOTA.

## Limits and next work

The current likelihood assumes independent Gaussian measurement noise with one standard deviation per candidate. Shared calibration bias, drift, correlated instruments, non-Gaussian tails and nuisance-parameter integration are not yet modeled. The discrepancy set is finite; it cannot guarantee robustness outside its declared scope. Prior weights, action utilities, scenario provenance, representativeness and amplification-risk values are caller-owned and must be audited independently.

Next increments:
- controlled decision-policy validator and evidence receipt in Symtropy;
- uncertainty-budget composition from crossed Gage R&R, calibration, bias, stability and drift, preserving covariance rather than assuming independence silently;
- sequential posterior update after each observed measurement, with pre-registered stopping rules;
- robust decision EVSI under correlated/non-Gaussian metrology error;
- independent physical challenge campaigns and external comparison against EIG / R-IDeA / robust BOED.

## Research comparison

- Harikumar et al. (UAI 2026), *Robust Bayesian Decision Making under Adversarial Uncertainty*, develops sequential adversarially robust decision-aware experimental design and emphasizes decision stability over fragile high-confidence nominal decisions: https://proceedings.mlr.press/v337/harikumar26a.html
- Tang, Sloman & Kaski (AISTATS 2026), *Representative, Informative, and De-Amplifying*, introduces R-IDeA to jointly account for representativeness, information and error de-amplification under model misspecification: https://proceedings.mlr.press/v300/tang26d.html
- NIST, *Gauge R&R Studies*, characterizes repeatability, reproducibility, stability, bias, resolution, linearity and measurement uncertainty: https://www.itl.nist.gov/div898/handbook/mpc/section4/mpc4.htm
- IEEE/IEC 61869-21:2025 specifies uncertainty evaluation in accuracy tests for instrument transformers; applicability must be checked against the actual device and measurement system, not generalized to all manufacturing measurements: https://standards.ieee.org/ieee/61869-21/11655/

The EVSI implementation is a research ranking primitive. Product acceptance, equipment safety, regulatory compliance, and final release remain under the approved quality system and accountable humans.

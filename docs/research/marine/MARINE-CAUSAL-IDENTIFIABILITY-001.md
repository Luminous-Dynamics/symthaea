# MARINE-CAUSAL-IDENTIFIABILITY-001
## Causal qualification of wind-assistance, hybrid dispatch and voyage-energy improvements

**Status:** research specification v0.1.0  
**Created:** 2026-11-10  
**Parent:** MARINE-WAPS-CO-DESIGN-001 (#11406)  
**Related:** #11429 route robustness; #11430 trial evidence; #11431 coastal reproduction; #11473 forecast-regret; #11469 regulatory routes.

> This is an experimental-design and causal-inference specification. It is not permission to switch propulsion/WAPS equipment, direct an operational trial, or assert that any treatment is safe. All onboard interventions require owner, master, safety-management, class/flag and applicable authority approvals.

## 1. Paradigm shift: from performance correlation to identifiable causal effect

A digital twin, AIS dataset, high-resolution sensor trace or matched before/after fuel average does not by itself establish that a WAPS device or energy-management policy caused a fuel reduction. Weather, loading, trim, hull condition, speed through water, current, wave heading, machinery health, operator choice, schedule slack, port wait and battery state can jointly affect both treatment and outcome.

The research objective is therefore not merely to predict fuel use or compute a difference. It is to specify a causal estimand, its identification assumptions, the treatment assignment mechanism, the counterfactual design, the evidence boundary and the conditions under which the effect is **not identifiable**.

A design must be allowed to return NOT_IDENTIFIABLE or INSUFFICIENT_OVERLAP. No optimizer or model may turn an unsupported causal claim into a percentage saving.

## 2. Target estimands

Define a study-specific population of eligible voyages or route segments and a service contract before analysing outcomes.

Primary estimand:
- average treatment effect of WAPS-enabled versus WAPS-feathered/stowed state on propulsion fuel energy or shaft energy over eligible exposure windows, under a fixed service policy and declared operating envelope.

Secondary estimands:
- effect on total primary energy after accounting for battery SOC and conversion losses;
- effect on arrival-window performance, delivered cargo work and availability;
- heterogeneous treatment effects by apparent-wind angle, wind speed, device configuration, loading state, sea state and route segment;
- effect of an EMS dispatch policy versus the reference policy under forecasts actually available at decision time.

Gas-resolved secondary estimands:
- average treatment effect on per-engine CH4 and N2O mass rate over eligible windows;
- gas-specific mass per delivered tonne-nautical-mile or TEU-nautical-mile after aggregating only eligible matched-service windows;
- heterogeneous gas response by engine load, fuel batch, WAPS achieved state, apparent-wind angle and ambient conditions.

Gas effects require the engine-resolved evidence contract in [MARINE-GAS-RESOLVED-EVIDENCE-001](./MARINE-GAS-RESOLVED-EVIDENCE-001.md) and [#11532](https://github.com/Luminous-Dynamics/symthaea/issues/11532). Direct test-bed/onboard measurement, engine-load-calculated emissions and CEMS must remain distinct evidence methods. Do not combine their values without a predeclared method-comparison model and uncertainty treatment. Do not convert gas mass to CO2e without a frozen GWP edition and regulatory/accounting boundary.

Do not conflate:
- device command with achieved aerodynamic force;
- commanded WAPS state with achieved WAPS state; a state mismatch is non-adherence, not a successful treatment exposure;
- shaft power with total fuel;
- fuel mass with primary energy;
- tank-to-wake emissions with well-to-wake emissions;
- segment-level effect with annual fleet-level effect;
- a fuel benefit with a whole-service lifecycle benefit.

The primary outcome boundary and normalization must be frozen. Battery energy cannot be treated as free fuel reduction: match initial/final state of charge and thermal state, or account for energy/SOC change and conversion loss explicitly.

## 3. Causal graph and confounding register

For each study, publish a directed acyclic graph (DAG) and pre-treatment confounder register.

At minimum consider:
- route/segment, date, season, departure time and port;
- cargo class, mass, loading condition, draft, trim and stability state;
- speed through water, speed over ground, heading, rudder state and schedule slack;
- apparent/true wind vector, gusts, waves, swell, current, water depth and forecast version;
- hull/propeller fouling, maintenance, propeller condition and machinery operating point;
- device geometry/revision, deployment/trim/feather/retraction state, control command and actual response;
- fuel batch, temperature, density, flow-meter calibration and generator/engine efficiency map;
- battery initial/final SOC, temperature, SOH, charge/discharge power and reserve constraints;
- operator/crew actions, failure state, port wait and external disruptions.

Distinguish pre-treatment confounders from post-treatment mediators. Do not condition away the causal pathway by controlling for variables created by the treatment (for example, a treatment-induced change in shaft demand or speed) when estimating the total effect. Define controlled direct effects separately if that is the intended estimand.

## 4. Identification designs

### 4.1 Randomized switchback (preferred only when approved and safe)

Where the owner/master/safety system has approved a safe envelope and device transitions are reversible, use blocked AB/BA or randomized switchback windows:
- randomize treatment order within predeclared matched route/weather/load blocks;
- define minimum window length and washout from measured actuator and vessel response, not convenience;
- record assigned state, commanded state, achieved state, non-compliance and transition timestamps;
- exclude or separately model transition transients using preregistered rules;
- retain intention-to-treat and treatment-on-treated estimands;
- balance forecast/observation timing and SOC boundary conditions.

Randomization is never allowed to override operational safety, navigation, crew authority, device limitations or voyage schedule.

### 4.2 Matched observational segments / target-trial emulation

When safe randomization is unavailable:
- define eligibility, treatment strategies, assignment time zero, follow-up window and outcome before inspecting the effect;
- require common support/overlap in route, load, metocean, speed, device and machinery state;
- use exact/block matching where possible, then predeclared propensity or generalized propensity weighting, g-computation, or doubly robust estimators;
- cluster uncertainty at voyage and vessel level;
- preserve unmeasured-confounding sensitivity analysis and negative controls;
- explicitly account for selection into WAPS-on state (operators may deploy it only in favourable conditions).

A nearest-neighbour match without a measured overlap diagnostic is not causal identification.

### 4.3 Natural experiments

A maintenance outage, device retrofit, policy change or route change may create a natural experiment only if its assignment mechanism, parallel-trends assumptions, anticipation, spillovers and time-varying confounders are examined. Difference-in-differences or interrupted time-series estimates must not be accepted solely because a pre/post plot looks plausible.

## 5. Mandatory identifiability gates

Before reporting a causal effect, evaluate:
1. **Consistency:** treatment strategies and actual device/dispatch state are precisely defined and revisions recorded.
2. **Exchangeability:** randomization or a defensible measured confounder adjustment; publish residual unmeasured-confounding sensitivity.
3. **Positivity/overlap:** both treatments are observed or assigned over the target population's supported covariate domain.
4. **No interference / carryover:** treatment in one window does not contaminate the next, or the carry-over is explicitly modelled.
5. **Temporal integrity:** synchronized clocks, valid timebases, known sensor delays and no future-observation leakage.
6. **Outcome validity:** calibrated fuel/shaft/electrical channels, explicit measurement boundaries and uncertainty propagation.
7. **Service equivalence:** same cargo service, loading, route, arrival contract and availability, or a predeclared effect decomposition.
8. **Missingness/exclusion:** missing samples, weather exclusions, port deviations and device outages remain visible; no silent imputation.
9. **Model support:** counterfactual model is validated on eligible held-out conditions and abstains out of domain.
10. **Independent replication:** effect persists on a separate route/season/voyage block and is independently reproduced.

If any mandatory gate is not satisfied, the result must be NOT_IDENTIFIABLE, INSUFFICIENT_OVERLAP, INCOMPLETE or another explicit non-claim state.

## 6. Battery and hybrid-system accounting

For hybrid vessels, an apparent fuel reduction may be battery discharge rather than efficiency. Every treatment window needs:
- initial/final SOC, temperature and SOH;
- charge/discharge energy and direction;
- bus losses, converter efficiency and auxiliaries;
- generator operating point and fuel flow;
- battery replacement/degradation allocation under the declared lifecycle model.

Primary analysis should use an energy boundary that closes across the window. Where exact SOC matching is impossible, adjust using a declared battery-energy valuation method and publish sensitivity. Never award free credit for a lower final SOC.

## 7. Diagnostics and negative controls

Mandatory diagnostics:
- standardized mean differences / overlap plots before and after adjustment;
- effective sample size and weight truncation sensitivity;
- treatment adherence, switching and transition-state distributions;
- within-voyage and between-voyage variance decomposition;
- residual time autocorrelation and clustered uncertainty;
- covariate balance across wind-angle, sea-state, load, route and machinery slices;
- treatment-effect heterogeneity and out-of-domain mapping.

Negative controls:
- placebo treatment assigned before the actual deployment event;
- future treatment predicting past outcome (lead test);
- outcome expected not to change under the treatment;
- randomized-label/permutation test under the declared assignment mechanism;
- sham transitions in synthetic or bench data;
- SOC carry-over negative control and boundary-closure test.

A failed negative control blocks the causal claim and must remain in the report.

## 8. Performance and evidentiary targets

These are research targets, not achieved results or universal thresholds.

- 100% preregistered estimand, population, treatment definition and outcome boundary.
- 100% synchronized treatment/device/sensor identities and configuration revision.
- 0 false-identifiable cases in an adversarial synthetic corpus.
- 0 hidden missingness, unreported exclusions or treatment-state ambiguity.
- 100% predeclared overlap and effective-sample-size diagnostics.
- 100% battery energy/SOC boundary closure within calibrated uncertainty.
- 0 statistically/physically failed negative controls for any claim labelled causally supported.
- Effect intervals include measurement, model and sampling uncertainty; report p50/p95/CI and sensitivity to unmeasured confounding.
- Independent holdout on at least two route/season blocks before claiming transfer beyond the studied vessel/condition.
- Any asserted fuel or energy benefit must exceed combined measurement/model uncertainty and preserve delivered cargo work, arrival and availability.

Do not use a generic universal p-value or percentage threshold as a substitute for study design, practical effect size and uncertainty. Pre-register alpha/power or Bayesian decision criteria with the study owner.

## 9. What would beat current practice

The potential contribution is a reusable causal qualification layer that can sit above any qualified CFD/energy model:
- makes the causal question and identification assumptions machine-readable;
- detects lack of overlap and treatment selection instead of extrapolating silently;
- distinguishes device commands, achieved device state, physical assistance and energy outcomes;
- accounts for battery SOC and conversion losses;
- binds data/calibration/timebase/configuration evidence;
- separates measured effect from model-based counterfactual;
- returns an explicit no-claim outcome when prerequisites fail.

This is more defensible than a generic digital-twin “fuel saved” dashboard. It must be tested against standard matched-voyage analysis, regression adjustment, propensity weighting, doubly robust estimators and randomization inference where randomization is legitimately available.

## 10. Integration

- **Symthaea:** estimand contract, DAG/confounder registry, identifiability gate and claim-level disposition.
- **Symtropy:** deterministic energy-boundary arithmetic, replay and counterfactual model execution within validated domains.
- **Sol-Atlas:** route geometry, metocean/spatial strata and spatial provenance.
- **Mycelix:** rights-aware data provenance, sensor/calibration/configuration lineage and reviewer attestations.

No direct actuation or navigation control is granted. Physical trial protocol and safety remain under the vessel owner, master, safety management system, class/flag and competent authorities.

## 11. Sources and grounding

- Park & Song (2026), ship fuel/CII prediction: random-split optimism bias and non-identifiability from annual MRV aggregates, https://doi.org/10.3390/jmse14191774.
- ISO 15016:2025 speed/power trial data analysis, https://www.iso.org/standard/82080.html.
- ISO 19030 hull/propeller performance change measurement, https://www.iso.org/standard/63774.html.
- IMO MEPC.391(81) lifecycle GHG guidelines, https://www.imo.org/en/ourwork/environment/pages/lifecycle-ghg---carbon-intensity-guidelines.aspx.

## 12. Claim ceiling

An identified average effect applies only to the defined vessel, treatment, target population, operating envelope, outcome boundary and study period. It does not automatically transfer to another hull, route, season, WAPS geometry or fleet. A causal energy result is not a safety approval, regulatory verification or proof of lifecycle superiority.

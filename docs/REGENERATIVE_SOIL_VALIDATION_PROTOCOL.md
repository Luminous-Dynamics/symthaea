# Regenerative Soil Systems: Validation and Experimental Design Protocol

Status: proposed protocol; not an approved agronomic recommendation (2026-10-10).

## Purpose

Turn the first-principles accounting kernel into a falsifiable research program for biochar, compost, nutrient-recovery products, and terra-preta-inspired soil systems. The goal is not to imitate the appearance of dark earth. It is to test whether a defined treatment improves crop-relevant outcomes and nutrient-use efficiency without unacceptable contaminant, salinity, pH, emissions, or economic costs.

This protocol is a design template. Local agronomists, soil scientists, laboratory staff, regulators, and participating growers must adapt and approve it before any field application. No application rate or recipe is prescribed here.

## Evidence hierarchy

Every value in Symthaea and Sol Atlas must be tagged as one of:

1. **Measured** — instrument/lab result with sample ID, method, unit, date, uncertainty and chain of custody.
2. **Published** — source, study population, method, and applicability limits recorded.
3. **Calibrated** — fitted against named observations, with dataset version and holdout evaluation.
4. **Scenario** — explicit hypothetical input for sensitivity analysis.
5. **Derived** — deterministic calculation from identified inputs and a versioned equation.
6. **Unknown** — required information missing; no silent imputation for safety-critical or decision-critical fields.

A deterministic calculation is not empirical validation. A candidate may not be described as field-effective solely because conservation equations close.

## Research questions and primary endpoints

Choose one crop, one soil class, one climate/season and a predeclared management objective per experiment. Do not pool unlike sites into one undifferentiated treatment effect.

Before treatment allocation, select one primary endpoint and its minimum practically meaningful difference with the local agronomist and grower. Suitable endpoints include marketable yield per hectare, nutrient-use efficiency, or water productivity. Secondary endpoints can include:

- total crop biomass and marketable yield/quality;
- nutrient input, plant uptake and nutrient-use efficiency by element;
- nitrate/ammonium movement below the root zone and relevant phosphorus losses;
- soil pH, electrical conductivity, organic carbon, bulk density, water retention and infiltration;
- microbial biomass/activity where the method is validated for the question;
- energy use, emissions, labor, transport and total cost per unit of marketable output;
- amendment contaminants, crop-tissue contaminant uptake and any ecological adverse effects.

Predefine how missing observations, weather shocks, pest damage, failed plots and deviations from protocol will be reported. Do not change the primary endpoint after viewing results.

## Treatment design

Use a randomized complete-block design where field gradients matter. Blocks should represent known spatial variation; randomize treatment allocation within each block. Use independent experimental plots as replicates; repeated measurements within a plot are not independent replicates.

A reasonable conceptual comparison set is:

- T0: locally appropriate standard practice (the active comparator);
- T1: standard practice plus characterized biochar;
- T2: characterized compost/organic amendment under an agreed nutrient budget;
- T3: biochar plus the same characterized organic amendment;
- T4: an optimized, safety-cleared candidate formulation selected before confirmatory testing.

Add an unfertilized control only when agronomically, ethically and operationally appropriate. Where nutrient equivalence is the question, match treatments on a clearly declared nutrient basis and document unavoidable differences; do not accidentally compare treatments with radically different N/P/K inputs and attribute the whole effect to biochar. Include fertilizer-only controls when needed to separate carrier effects from nutrient effects.

Do not use one universal biochar dose. Determine trial rates from material characterization, soil tests, applicable rules, preliminary pot/incubation work and a documented risk review. Run dose-response work separately when it is a primary question.

## Replication and statistical plan

Estimate the number of independent plots from the predeclared minimum meaningful effect, variance from relevant pilot or literature data, target power, significance level and expected attrition. Do not pick a convenient replicate count and later claim adequate power. If local variance is unknown, label the first trial exploratory and use it to estimate variance for a confirmatory study.

Pre-register:
- hypothesis and primary endpoint;
- treatment recipes and batch identifiers;
- randomization seed and block map;
- inclusion/exclusion and stopping rules;
- sample-size rationale and statistical model;
- planned contrasts, multiplicity handling, missing-data handling and uncertainty intervals;
- cost boundary, emissions boundary and time horizon.

Analyze at the randomized experimental unit. Report effect sizes and uncertainty intervals, not just p-values. Separate confirmatory from exploratory analyses and preserve null and adverse results. Evaluate treatment-by-site, soil, crop and season interactions only where supported by the design.

## Product and environmental release gates

No candidate reaches field testing until a qualified reviewer defines applicable limits and the product passes relevant testing. The panel depends on feedstock and process but may include:

- potentially toxic elements and other feedstock-specific contaminants;
- persistent organic contaminants (including PAHs where relevant);
- pH, electrical conductivity/salinity, ash and nutrient composition;
- maturity/stability and phytotoxicity screening for composted or blended products;
- pathogen controls for manure, sewage-derived materials or recovered organic nutrients;
- applicable product, waste, worker-safety and environmental requirements.

Feedstock traceability is mandatory. Exclude treated/painted wood, mixed municipal waste, industrial residues or other uncertain inputs unless a qualified, jurisdiction-specific assessment demonstrates suitability. Do not assume that pyrolysis removes every contaminant; some contaminants can persist or concentrate in char/ash, and volatile contaminants can move to emissions.

Unknown safety status is a hard stop, not a low-confidence score. Any contamination exceedance, unexplained plant injury, unacceptable salinity/pH shift, or off-site impact triggers suspension and investigation under the preapproved protocol. Use trained operators, appropriate fire/CO/exhaust controls and lawful emissions management for thermal equipment.

## Batch-level process model

For each production batch, retain:
- feedstock source, mass, moisture, elemental composition, ash/minerals and sampling method;
- reactor identity and calibration, temperature-time profile, atmosphere, pressure where relevant, throughput and operator log;
- dry char mass, char elemental composition, gas/liquid/solid coproduct estimates and uncertainty;
- measured fuel/electricity/heat input, recovered heat and emissions observations where available;
- batch sample IDs, lab methods, quality gates and disposal/hold status.

Check dry-mass and elemental balances with uncertainty intervals. If balance closure falls outside a predeclared tolerance justified by measurement uncertainty, flag the batch for investigation; do not force closure by adjusting an unobserved stream. Heat accounting must distinguish theoretical duty, measured supplied energy, recovered energy and reactor losses. A screening heat calculation is not a reactor design or emissions assessment.

**Measurement-closure distinction:** the initial pyrolysis screening calculation infers the non-char dry-mass residual by subtraction. That identity is useful for accounting but is not independent evidence of measured physical closure. For a pilot batch, use the observed-stream assessor with a declared boundary, complete inlet/outlet mass inventory, unique stream IDs, measured evidence for each physical mass and a tolerance justified by the measurement procedure. Include all inlets (such as purge gas or added process water) and all outputs; do not make an unmeasured gas or condensate stream silently equal to zero. The result reports signed residual and relative closure error. A closure pass does not establish safe product quality, correct emissions inventory, nutrient availability, or agronomic efficacy.

## Nutrient-cycle accounting

For each element separately (at minimum N, P and K), maintain a ledger across:
- feedstock input;
- product recovery and coproduct streams;
- total nutrient content in the amendment;
- plant-available fraction over the declared time horizon;
- application input;
- crop uptake/removal;
- runoff, leaching, gaseous losses and residual soil pools where measured or modeled;
- uncertainty and provenance for each term.

Do not equate total nutrient content with crop-available nutrient. Do not equate nitrogen, P2O5 and K2O units with elemental N, P and K without explicit conversion metadata. Record the basis of every value (elemental or oxide, wet or dry, concentration or mass, per batch or per hectare).

## Symthaea implementation contract

The software should provide:
1. dimensional/unit validation and explicit basis conversions;
2. immutable input snapshots and evidence references for every parameter;
3. conservation residuals and uncertainty propagation, with tolerance sourced from measurement uncertainty rather than a universal arbitrary constant;
4. sensitivity analysis and scenario comparison before expensive experiments;
5. a measurement-triage stage that prioritizes evidence-backed tests only when they could address an unresolved decision; a full design-of-experiments planner must separately handle randomization, blocking, interactions, and power without changing preregistered constraints;
6. separation of candidate generator, independent evaluator, and release authority;
7. Pareto comparison only after mandatory safety and quality gates pass;
8. reproducible exports for lab, field, cost, nutrient, energy and emissions ledgers;
9. an explicit not-validated state until held-out data and review evidence meet declared acceptance criteria.

A model can recommend the next measurement or experiment. It cannot certify its own assumptions, approve its own product, or turn a scenario into a measured fact.

## How uncertainty intervals are interpreted by the engineering screen

The current Symthaea engineering implementation can accept evidence-linked intervals for char yield, retained carbon, supplied heat, cost, water use, and (when included) net climate impact. It makes a deliberately conservative threshold decision:

- **Robust pass:** the full interval satisfies the requirement.
- **Robust fail:** the full interval violates the requirement.
- **Indeterminate:** the interval overlaps the threshold, or a required climate interval is missing.

This is interval-bound screening only. Do not call an interval a 95% confidence interval unless the analysis that produced it supports that interpretation. Preserve the interval-generation method, sample size, calibration domain, date, units, dependence assumptions, and evidence ID in the referenced record. Scenario ranges can support exploration but cannot be silently relabelled as measurements or probabilistic coverage. The method currently does not propagate dependence/covariance across inputs and must not be interpreted as a Bayesian or Monte Carlo uncertainty model.

## Initial measurement-triage utility

Symthaea now provides a small utility that filters and ranks measurement options against unresolved interval-screen constraints. Its score is an explicit heuristic using three caller-provided inputs: decision relevance weight, an expected benefit fraction (defined as interval narrowing or missing-data acquisition), and cost in a shared unit and price basis. Each input carries an evidence ID. Options that do not currently target an unresolved constraint remain visible in the plan with no rank.

This utility does **not** produce an optimized experiment design, statistical power calculation, confidence interval, probability of decision change, or calibrated expected value of information. Do not treat assumed reductions or relevance weights as empirical facts without validation. A proper experimental design must still predeclare hypotheses, endpoints, treatment levels, randomization, blocks/replicates, stopping rules, and analysis. NIST's handbook distinguishes screening studies used to identify important factors from response-surface or confirmatory designs: https://www.itl.nist.gov/div898/handbook/pri/section3/pri3346.htm. Sequential Bayesian design is a future option once a defensible likelihood/model and calibrated observations exist: https://www.nist.gov/programs-projects/sequential-bayesian-experiment-design.

## Preregistered two-level screening schedule generator

The engineering crate can generate a reproducible bench-scale schedule when a complete preregistration object is supplied. The supported initial design is a **two-level full factorial** with 2–6 numeric factors. Each factor has finite, strictly ordered low/high values, explicit units and evidence references. Every block includes every treatment combination; at least two complete blocks are required, with 1–2 independently executed units per setting per block. This creates replication across blocks but is not, by itself, proof of adequate statistical power.

Required before generation:
- a preregistration ID, immutable input-snapshot ID, protocol ID, objective, primary hypothesis, and analysis-plan ID;
- one primary endpoint, measurement method and a minimum practically meaningful difference with supporting rationale;
- 2–8 distinct blocks that describe a nuisance stratum (for example, day or feedstock lot);
- a fixed randomization seed; run order is randomized within each block and the algorithm ID is retained;
- an evidence-backed reviewer approval for the exact bench-scale protocol. Unknown/rejected review status, a protocol-ID mismatch, or scenario-only approval evidence causes the planner to reject the request.

Optional center-point controls are disabled or 3–5 per block. When enabled, every quantitative factor is set to its numeric midpoint; controls are evenly placed with controls at the beginning and end of each block, while treatment runs remain randomized among themselves. NIST describes center points as a check on stability and curvature, not as a substitute for a response-surface experiment: https://itl.nist.gov/div898/handbook/pri/section3/pri337.htm.

The generated plan stores a full copy of the request, factor settings and units for every run, block ID, run order, factorial standard order, replicate index, seed and algorithm version. Identical input and algorithm versions are expected to produce the same schedule. The maximum design size is bounded at 1,100 runs.

**Limitations:** numerical-factor only; no fractional factorial generator, power/sample-size calculator, statistical response analysis, Bayesian update, automatic stop-rule engine, or field-trial authorization. The primary meaningful difference is not an effect estimate or power claim. Treat the output as a reviewed bench-scale schedule proposal; the experiment owner must separately approve operating conditions, worker/environmental safeguards, measurement-system capability, preregistered analysis and stopping rules. This planner does not authorize product release or soil application.

## Decision gates

- **G0 — provenance:** feedstock and site metadata are complete enough to plan sampling.
- **G1 — laboratory characterization:** composition, contaminant panel and product-quality criteria are satisfied.
- **G2 — model verification:** unit tests, independent balance checks, dimensional checks and adversarial invalid-input tests pass.
- **G3 — controlled screening:** pot/incubation trials with controls show no unacceptable adverse effect and justify the proposed field hypothesis.
- **G4 — field pilot:** approved randomized protocol, appropriate oversight, baseline soil tests and monitoring plan are in place.
- **G5 — replication:** results repeat in an independent season/site or a properly held-out dataset.
- **G6 — deployment case:** crop outcome, economics, nutrient losses, life-cycle effects, supply logistics and safety are jointly assessed.

Passing one gate does not imply passing the next. Any safety-critical unknown blocks the relevant release gate.

## Success criterion

A candidate is a credible improvement only when it beats a declared local comparator on a predeclared practical objective, stays within safety/environmental limits, and has a defensible cost and resource budget. Claims of long-term soil creation, carbon permanence or equivalence to mature terra preta require longer-duration evidence and must remain separate from short-season yield claims.

## Scientific grounding

- Glaser & Birk (2012), *State of the scientific knowledge on properties and genesis of Anthropogenic Dark Earths in Central Amazonia*, Geochimica et Cosmochimica Acta. DOI: 10.1016/j.gca.2010.11.029. The review describes mixed organic and inorganic inputs and notes unresolved questions about genesis and timescale.
- Asare et al. (2022), *Anthropogenic dark earth: Evolution, distribution, physical, and chemical properties*, European Journal of Soil Science. DOI: 10.1111/ejss.13308.
- *Soil health response to biochar combined with other amendments: a review* (2025), Biochar. DOI: 10.1007/s42773-025-00531-6. Reports context-dependent outcomes and the need for longer-term, cross-site studies.
- *Towards understanding the long-term fate of biochar in Terra Preta* (2025). DOI: 10.1080/17583004.2025.2560126. Investigates mineral-organic interactions that may contribute to persistence; it does not establish a universal recipe.

These sources support a testable, multi-input, long-horizon research strategy—not a claim that any particular formulation has already reproduced terra preta.

# BIOPROTEIN-EXPERIMENT-VALUE-001: prioritize experiments by decision value

**Status:** synthetic-first research contract and executable Rust oracle.  
**Scope:** decision semantics, not biological or production validation.

## Paradigm shift

Do not choose a research experiment just because it has the highest analytical accuracy, information gain, variance reduction, or fermentation yield. Choose among experiments that are permitted, sufficiently specified, and aligned to the same scenario by asking: **how much better do we expect the downstream decision to become after seeing the possible results, and is that improvement worth the experiment's cost?**

This is expected value of sample information (EVSI) followed by an expected net value decision. EVSI has a mature decision-analysis definition: average the utility of the best decision after each possible observation, then subtract the best expected utility under current knowledge. Study cost must be included before prioritizing research. See the ISPOR good-practices report and the tutorial on simulating study data:
- https://pmc.ncbi.nlm.nih.gov/articles/PMC7373630/
- https://doi.org/10.1177/0272989X211026292

This matters for alternative proteins because:
- A 2026 review calls out feedstock variability between controlled laboratory substrates and industrial side streams, and the need to join feedstock selection to process economics and LCA: https://doi.org/10.1016/j.coche.2026.101242
- A 2026 cultivated-meat downstream-processing review highlights cell harvesting, media recycling, separation efficiency, and downstream integration as major underexplored scale-up levers: https://doi.org/10.1016/j.tifs.2026.105665
- A 2026 comparative microbial-protein LCA reports that energy and carbon feedstock assumptions can prevent a conclusive climate comparison even when land and water outcomes look favorable: https://doi.org/10.1007/s11367-026-02637-w
- A 2026 cultivated-meat TEA uncertainty analysis emphasizes that some influential parameters have weak empirical pedigrees: https://repository.tudelft.nl/record/uuid:7a172ff5-aa2a-4a1f-9dc7-a02cacf4b712

## Semantics

- Enumerate uncertain states with integer prior weights summing to 10,000.
- Enumerate decisions/actions with explicit utility per state.
- An experiment lists outcome likelihood weights per state, also summing to 10,000.
- Compute the best expected utility before the experiment and the expected best utility after observing each possible outcome.
- Compute gross EVSI and subtract declared cost in the same abstract utility unit.
- Use integer arithmetic, with probability weights parts-per-10,000; emitted EVSI numerators remain exact decimal strings.
- Rank by net EVSI, with deterministic experiment-ID tie-breaking.
- Restricted authorization, incomplete evidence, and a scenario digest mismatch are non-compensatory exclusions. Unknown authorization/evidence remains indeterminate when no eligible positive-value experiment resolves the recommendation.
- A known eligible experiment with positive net EVSI can be selected while unknown candidates are clearly excluded from consideration; no unapproved candidate is selected.
- Malformed probability mass, missing state/payoff coverage, duplicate identities, invalid costs, or out-of-range utilities fail closed.

The benchmark's utility values are invented abstract units, not dollars, kcal, grams of protein, kilograms of product, or validated social-welfare weights. Outcome likelihoods are synthetic fixture inputs, not empirically estimated assay sensitivity/specificity.

## Frozen benchmark cases

The v1 corpus includes 12 cases: a positive-value assay; an assay whose cost exceeds value; an uninformative test; restricted and unknown authorization; a choice between a cheaper imperfect assay and a costly perfect assay; deterministic ties; malformed likelihood weights; invalid prior mass; scenario mismatch; incomplete experiment evidence; and a candidate with unknown authorization beside an eligible positive-value candidate.

EVSI-006 is intentionally diagnostic: the perfect test yields more gross information but costs more. The oracle should select the cheaper imperfect assay because its **net decision value** is greater. This directly rejects a common optimization failure: maximizing assay quality without pricing the decision.

## Initial benchmark targets

- 100% agreement with frozen expected statuses, selected IDs and reason codes.
- Zero restricted, evidence-incomplete, or scenario-mismatched experiments selected.
- Zero decisions based on malformed prior or likelihood mass.
- Zero non-positive net-EVSI experiments selected.
- Identical results under input experiment reordering; stable ID tie-break.
- Exact integer EVSI arithmetic with no floating-point drift.
- Every receipt binds the exact corpus SHA-256 and the exact checked-out Git HEAD.
- Unknown authorization is never treated as permission; uncertainty is explicit.

## Next research-grade targets (not claimed achieved)

1. **Decision regret:** evaluate on preregistered synthetic scenario families; compare EVSI-based allocation against yield-only, assay-accuracy-only, equal-budget, and random baselines under the same budget. Proposed target: at least 20% lower mean decision regret than the best simple baseline, with confidence intervals across scenario seeds.
2. **Calibration:** when empirical priors/outcome models become permissible and available, score posterior-predictive calibration and decision calibration separately. Synthetic fixture agreement is not calibration.
3. **Robustness:** sweep prior shifts, costs, likelihood degradation, missing outcomes, correlated parameters and distribution shift. Report when a recommendation changes, not only a mean score.
4. **Resource closure:** extend cost beyond money to water, electricity, process time, analytical throughput, sample scarcity, waste handling and cleaning/turnaround, connecting the experiment portfolio to Symtropy's complete-cycle accounting.
5. **Empirical promotion:** no real experiment enters the ranking without source rights, a documented protocol, qualified analytical methods, evidence lineage, a stated decision owner, and a clear authorization route.

## Validation boundary

The corpus and its unit tests are synthetic specifications. A green Rust workflow would establish reproducibility of the oracle's encoded decision contract only. It would not validate probabilities, utilities, assay performance, microbial growth, food safety, commercial economics or environmental impact.

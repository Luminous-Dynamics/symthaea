# PET-NUTRITION-EVIDENCE-001 — Endpoint-specific claim ladder for animal food research

**Status:** synthetic benchmark contract; not a feeding recommendation, veterinary assessment, safety certification, or biological conclusion.  
**Version:** 1.0.0  
**Created:** 2026-10-18

## Research signal

A peer-reviewed *PLOS ONE* article published 23 September 2026 tested a dry extruded diet containing 10% microbial protein in 30 dogs and 30 cats. The two-bowl, in-home trial ran across four meals over two days and compared macronutrient-matched formulations that differed primarily in protein source. The authors report increased short-term preference for the test diets, with reported dog intake metrics of 72% versus 50% and cat metrics of 51% versus 34%; the paper reports p = 0.002 for dogs and p = 0.001 for cats. Diet composition was also characterized by a laboratory. The paper was funded by MicroHarvest, whose employees were authors; the preference trial was designed and conducted by Sense Test and the diet analysis was performed by Eurofins Food Testing Portugal. The authors explicitly limit their findings to short-term preference and state that long-term feeding behavior and health effects were not established.

Source: Motta de Matos A, Cruz A. “Microbial Protein as a novel ingredient in pet food: A food preference study in dogs and cats.” *PLOS ONE* 21(9), e0356767 (2026). https://doi.org/10.1371/journal.pone.0356767

The values above are reported as described by the paper, not independently reanalyzed. Before comparing the reported consumption percentages with other studies, the metric denominator and aggregation must be harmonized; they should not automatically be interpreted as two shares that add to 100%.

## Paradigm shift

**Do not ask whether a food ingredient is “proven.” Ask which exact claim, for which animal, inclusion level, outcome and time horizon is supported by which type of evidence.**

A preference result is not a digestibility assay. A nutrient-composition certificate is not a complete-diet adequacy assessment. Short-term intake is not long-term health. A modeled bioprocess yield is not an observed production result. An LCA is not a cost-competitiveness study.

The claim ladder is intentionally endpoint-specific:

| Endpoint | Evidence needed for the narrow claim | What it does not establish by itself |
|---|---|---|
| Food preference | Defined comparative preference design, target animals, comparator, duration, experimental unit and metric definition | Sustained acceptance, nutrient adequacy, tolerance, health |
| Short-term acceptance | Actual intake/acceptance measurement in a declared short-term design | Long-term acceptance or health |
| Nutrient composition | Traceable analytical results from specified sample/batch and methods | Digestibility or complete diet adequacy |
| Digestibility | Suitable declared digestibility method, relevant target species/endpoint and experimental unit | Lifelong adequacy or universal transfer to another species |
| Nutritional adequacy | Complete formulation/profile and a qualified protocol appropriate to species/life stage and intended claim | Unmeasured health benefits or other life stages |
| Tolerance / safety | Defined adverse-event and safety monitoring in the stated material/process/dose/time scope | Universal safety for other strains, doses, species or production routes |
| Long-term health / growth | Appropriate controlled design, measured endpoints, adequate horizon and defensible statistical plan | Outcomes beyond the studied population, diet, or duration |
| Environmental impact | Defined functional unit, lifecycle boundary, source inventory and scenarios | Economic competitiveness |
| Cost competitiveness | Declared TEA boundary, usable-product basis, currency/year and included cost terms | Environmental superiority or biological suitability |
| Process productivity | Measured process data for production claims, or explicit model labels for predictions | Biological validation from simulation alone |

## Non-compensatory claim gates

Before evidence can support a claim, the evaluator checks:

1. **Evidence integrity:** declared evidence type must match how it is reported; simulated results cannot be relabeled as measured.
2. **Scope match:** species, life stage, inclusion level and duration must match the requested claim. Mismatch blocks; unknown match is indeterminate.
3. **Endpoint match:** a study can support only endpoints it actually measured using a suitable design. No upward inference from preference to health.
4. **Reporting completeness:** sample size, experimental unit, metric, method, comparator and required uncertainty must be explicit for the endpoint.
5. **Disclosure and rights:** funding/conflict disclosure and source permissions must be recorded. A disclosure does not automatically invalidate a study; missing disclosure prevents promotion-ready status in this benchmark.
6. **Method-specific boundaries:** nutrient composition, adequacy, safety, LCA and TEA require distinct evidence fields.
7. **Claim ceiling:** eligibility means “eligible for this declared evidence comparison,” not that the product is safe, complete, effective, authorized, healthy, lower-carbon or cost-competitive in general.

## Benchmark artifacts

- `pet-endpoint-evidence-case-schema-v1.json`: structure for claims, evidence, scope and expected dispositions.
- `pet-endpoint-evidence-corpus-schema-v1.json`: container schema.
- `pet-endpoint-evidence-corpus-v1.json`: 18 explicitly synthetic positive, negative and indeterminate cases.

The corpus uses no real composition dataset or measured animal values. Synthetic values test semantics only.

## Benchmark targets

- Zero endpoint-upgrade false accepts on frozen cases.
- Zero species/life-stage/inclusion/horizon mismatches accepted as same-scope evidence.
- 100% match with independent expected dispositions and reason codes.
- Zero implicit imputation of missing experimental units, methods, uncertainty, source rights or nutrition.
- 100% of measured/modelled/simulated labels remain distinct.
- 100% of outputs include claim scope, endpoint, study type, evidence limitations and not-established claims.
- Deterministic results under case-order permutations.
- For real evidence, preserve funding/conflict disclosures, data provenance and license/permission scope; distinguish primary study data from secondary interpretation.
- No study is promoted to an animal-feeding recommendation solely by passing this software benchmark.

## Next research stage

After the schema and synthetic corpus are stable, implement an independent oracle and receipts. Then examine permission-cleared real studies using a fixed extraction protocol: species, life stage, organism/strain and ingredient identity, inclusion level, diet comparability, endpoints, experimental unit, study duration, statistics, conflicts/disclosures and explicit limits on conclusions. Never broaden the claim merely to make the evidence appear more useful.

For pet-food work, prioritize:
- longer-duration, species-appropriate tolerance and nutrient-adequacy studies;
- repeatability across ingredient batches and formulations;
- measured digestibility and limiting amino acids appropriate for each species/life stage;
- palatability at clearly stated inclusion levels, with methods and metric denominator harmonized across studies;
- process/LCA/TEA comparisons using aligned boundaries and usable-product denominators.

## References

- Motta de Matos A, Cruz A. *PLOS ONE* (2026), https://doi.org/10.1371/journal.pone.0356767
- Takegawa R, Kono N. “Species selection criteria for sustainable bacterial single cell protein.” *npj Science of Food* (2026), https://www.nature.com/articles/s41538-026-00828-1
- Guidance on cultivated-meat LCA methodology: https://link.springer.com/article/10.1007/s11367-025-02562-4

This program does not establish the safety or suitability of any actual pet-food ingredient. All real-world feeding trials, ingredient qualification, and regulatory claims require appropriate expertise and independent evidence.

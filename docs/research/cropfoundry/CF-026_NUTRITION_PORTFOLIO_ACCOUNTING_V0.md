# CF-026: portfolio-level nutrient delivery under matched resource constraints

**Status:** accounting/evaluation contract proposal with a synthetic two-option fixture. No crop ranking, diet recommendation, nutrient adequacy, health effect or real-world efficiency improvement is claimed.

## Paradigm shift

A system that maximizes kilograms per hectare may not maximize the amount of iron, zinc, protein or another locally selected nutrient that arrives in edible food. CF-026 extends CF-024 from a single batch × nutrient receipt to a **complete multi-nutrient comparison over explicit crop/production options**. It asks what is delivered per measured resource, without hiding absent assays, double-counting water, mixing resource boundaries or collapsing every nutrient into one unexplained score.

The concept is not new, but its accounting remains an important weak point. A review of nutritional water productivity defines nutrient output relative to water use and discusses application and methodological limitations; it also reports that few studies have applied the metric, so local definitions and data quality matter ([review, *Water* 2023](https://www.mdpi.com/2073-4441/15/24/4278)). FAO's nutrition-sensitive food-system guidance calls for retaining nutrient content and minimizing nutrient loss during storage and processing, not just preserving mass ([FAO recommendation 3.2.4.c](https://www.fao.org/evidence-platform-agri-food-systems-nutrition/recommendations/detail/3.2.4.c/en)). Recent work on crop nutrition likewise distinguishes concentration, crop productivity, nutritional diversity and bioavailability; this benchmark treats those as separate evidence fields, not interchangeable proxies ([nutrition-oriented crop portfolio review, 2026](https://www.frontiersin.org/journals/nutrition/articles/10.3389/fnut.2026.1987519/abstract)).

## South Africa-first scope and food-composition data gate

A relevant local research direction is underutilized leafy vegetables, not because a model should presume them superior, but because species/variety choice, local knowledge, affordability and cultivation suitability belong in the comparison. A 2024 South African review included 11 eligible studies across five provinces and catalogued 67 locally sourced leafy vegetables; it also illustrates that the published evidence base remains small relative to the diversity described ([Nko et al., *South African Journal of Botany*, DOI 10.1016/j.sajb.2024.06.048](https://doi.org/10.1016/j.sajb.2024.06.048)). This is a candidate-discovery result, not a ranked nutrition trial, and Indigenous/local knowledge should be handled with appropriate partnership, attribution and permissions.

The official South African Food Data System (SAFOODS) currently describes 1,857 food items in the database; its public web view exposes five nutrients (energy, carbohydrate, protein, fat and fibre), while the 2024 Food Composition Tables list 36 nutrients across 16 food groups as a paid product. SAFOODS states that values are means per 100 g edible food, warns that composition varies with cultivar, soil, climate, practice, processing, storage and transport, says missing nutrient values are not zeros, cautions against inferring processing retention from raw/cooked entries that may come from different samples or labs, and states that written SAMRC permission is required before reproducing any part of the database ([SAFOODS composition](https://safoods.mrc.ac.za/composition.html), [products and licensing](https://safoods.mrc.ac.za/products.html), [FAQ](https://safoods.mrc.ac.za/faq.html)). For iron/zinc/micronutrient portfolio work, the free five-nutrient view is insufficient by itself; obtain the appropriate terms and edition before use. DS-015 remains discovery-only and training-disabled; no SAFOODS records have been acquired or scored in this workstream.

Treat national food-composition means as reference context, not ground truth for the nutrient mass of a local cultivar/harvest lot. Preserve whether each value is directly assayed, borrowed from another national database, calculated or imputed; include the food identifier, edible portion, raw/cooked state, units, reference citation, and any available uncertainty. Direct crop-lot assays and source-backed recipe/processing records are needed to estimate local variation and delivered nutrient mass. Do not pool food-table values and trial-lot assays as if they were the same measurement class.

## Input contract and executable

The portfolio envelope is defined by [`nutrition-portfolio-input-v0.schema.json`](./nutrition-portfolio-input-v0.schema.json). Each nested record follows the row-level [CF-024 nutrition accounting schema](./nutrition-accounting-input-v1.schema.json), including concentration mass basis and measurement stage, evidence provenance, retention and bioavailability evidence when used, resource values and source digest format.

Run the checked-in synthetic example:

```bash
cargo run -q -p xtask -- cropfoundry-nutrition-portfolio \
  --input docs/research/cropfoundry/nutrition-portfolio-example-v0.json \
  --out /tmp/cropfoundry-nutrition-portfolio-receipt.json
cargo test -p xtask cropfoundry_portfolio
```

The example contains two fake crop options, each with two batches and two nutrients. It is an arithmetic fixture, not plant data. The receipt preserves output by nutrient, measured-concentration versus estimated-delivery counts, uncertainty intervals where supplied, bioavailability-estimate coverage, resource ratios, common boundary labels, errors and warnings. Separate per-nutrient flags say whether interval comparisons and bioavailability comparisons are available across *all* options; accounting success does not imply those optional evidence types are comparable. The command does not rank or recommend options.

## Invariants that matter

1. **No hidden zeroes.** Every batch must include the exact declared nutrient set, once per nutrient. Missing nutrient data fails completeness; it is not interpreted as zero nutrient content.
2. **No resource double-counting.** Water, land, energy and cost appear on each batch's nutrient rows for compatibility with the row-level receipt, but the portfolio reconciler checks their consistency and counts each unique batch ledger only once.
3. **Comparable boundaries.** All options must declare the same comparison basis. Resource ratios are only emitted when that resource is completely measured across each option, with compatible boundaries and currency. Irrigation-applied water is not interchangeable with consumptive use or a full supply-chain water footprint.
4. **Evidence remains typed.** The receipt separates directly assayed concentration from estimated delivery (for example, direct concentration measured before cooking plus a modelled retention factor). Missing bioavailability stays unknown. It is never silently set to one.
5. **Intervals are not fake precision.** Portfolio bounds are summed only when every batch for that nutrient provides usable lower and upper bounds. The arithmetic does not turn sensitivity bounds into a statistical confidence interval.
6. **Overflow and invalidity fail closed.** Non-finite aggregates, incomplete records, inconsistent repeated batch resource ledgers and incompatible comparison boundaries block a pass. Receipts can still report diagnostics without making the comparison valid.
7. **No universal scalar objective.** Nutrients remain separately reported. A Pareto or weighted ranking would require an explicit, externally reviewed set of priority nutrients, floors, prices/affordability limits and trade-off weights.

## Benchmark targets

- **Stage 0 — arithmetic:** synthetic fixtures must reproduce expected per-nutrient totals and resource ratios, with zero double-counting; missing nutrient records, incompatible boundaries, inconsistent repeated resource values, incompatible currencies and numeric overflow must fail closed.
- **Stage 1 — dataset:** use only authorized batch-level records with source-backed concentration, edible-mass basis, measured stage, resource ledgers and uncertainty. Report measured versus modelled coverage and every missingness pattern.
- **Stage 2 — decision value:** compare options on a locally agreed site-season or production-system basis, with matched boundaries and explicitly constrained water, land, energy and cost. A provisional future target could be at least 10% higher delivered mass per predeclared limiting nutrient per scarce resource than the documented baseline, but only if lower uncertainty bounds protect every other priority-nutrient floor and the plan meets safety, affordability and environmental constraints. The target must be justified by a local partner and power analysis, and locked before outcome inspection.
- **Stage 3 — external validity:** hold out entire locations, seasons or production cycles. A model predicting concentration in a held-out batch is not evidence that a portfolio improves diets, is affordable, or improves health.

## Claim ceiling

The input may support a nutrient mass calculation for a stated edible stage. That is not a dietary adequacy calculation, absorption estimate, public-health intervention, food-safety certificate, causal treatment effect or optimal crop recommendation. The portfolio receipt always records those claims as unevaluated/false and keeps training authorization false.

All source artifacts, access/rights evidence, instrument methods, units, batch/replicate relationships, resource boundaries and uncertainty provenance remain review gates before real crop records are used.

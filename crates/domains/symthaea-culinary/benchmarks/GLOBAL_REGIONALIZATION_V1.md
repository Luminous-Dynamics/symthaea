# Global regionalization and vertical coverage v1

**Status:** proposed cross-region contract and synthetic conformance fixtures. It does not configure real national data or establish regulatory/nutritional correctness for any country.

## Research summary

The goal is not one universal food table or a country-coded meal policy. It is a shared, versioned decision interface with region-specific evidence adapters and transparent capability gaps.

- FAO's [INFOODS network](https://www.fao.org/infoods) provides a long-running international coordination point for composition-data quality, nomenclature, data interchange and harmonization.
- The [INFOODS standards and guidelines](https://www.fao.org/infoods/infoods/standards-guidelines/en/) include guidance on food matching and converting units, denominators and expressions. Cross-country food matching is a documented data-quality problem, not a simple string translation.
- FAO's [food-composition program](https://www.fao.org/food-composition/en) recommends that composition data be developed at country or at least regional level.
- WHO/FAO's [2024 joint statement on healthy diets](https://www.who.int/publications/i/item/9789240101876) notes that the exact make-up of a diet varies with individual needs, preferences, beliefs, cultural context, local availability and dietary customs, even while core healthy-diet principles remain.
- The Codex Alimentarius Commission [reported adoption of guidance on precautionary allergen labelling at CAC49 in July 2026](https://www.fao.org/fao-who-codexalimentarius/news-and-events/news-details/en/c/1759963/). This is a reminder that even internationally coordinated standards need versioned local regulatory adapters; the contract must not pretend that a global ingredient list alone certifies a packaged product safe.
- The World Bank's [International Comparison Program](https://www.worldbank.org/en/programs/icp/faq) supports cross-economy food-price and diet-affordability comparisons, but purchasing-power parity is a statistical comparison tool, not a current local retailer/SKU price.

These sources motivate the interface and evaluation design. They do not supply production profiles or prove a local claim.

## Paradigm shift

**One global optimizer -> a common semantic core with regional evidence adapters and explicit coverage.**

A region pack is a contract, not a country stereotype. It may represent a city-market catchment, subnational region, cross-border market or a national aggregation. A country code selects geography metadata only. It must never infer religion, ethnicity, taste, acceptable foods, a health condition, dietary restriction, consent, or economic circumstances.

The regional pack is an index to evidence and local capabilities. It is not a duplicate nutrient database, live-price API, food-safety certification, legal opinion, dietary prescription, or inventory system.

## Ten verticals that must be independently declared

| Vertical ID | Scope | Fail-closed behaviour when absent |
|---|---|---|
| `ingredient_composition` | Food identity, edible/raw/cooked state, nutrients and analytical/derived value quality | Keep nutrients unknown or use a visibly labelled, qualified fallback; no silent foreign-food match |
| `dietary_guidance` | Referenced, dated guidance and applicable population/scope | Do not infer a personalised clinical diet from generic rules |
| `price_affordability` | Retail/wholesale/currency/package/market/time price evidence and diet-cost context | No invented current price; macro PPP/price series are context, not a shop quote |
| `product_identity` | Local name, synonym, package/lot identity, barcode or local product scheme | Ambiguous match remains unresolved |
| `food_safety_requirements` | Versioned jurisdictional rules and authoritative evidence references | Unknown is not safe; no certification from source existence |
| `cultural_acceptability` | User- or community-specified norms, preferences, substitutions and constraints | Ask or preserve options; never infer culture from country/language |
| `seasonality` | Locally grounded crop, harvest and season context | Label unknown; do not fabricate seasonality from hemisphere alone |
| `availability_logistics` | Market coverage, source freshness, delivery/pickup and transport constraints | Forecast, listing, stock, sale and fulfilled delivery remain distinct |
| `cooking_measurements` | Regional measures, product-specific densities, yields and preparation state | Do not convert mass to volume or raw to cooked by an unexplained constant |
| `energy_and_waste` | Cooking energy, refrigeration, edible yield, storage, waste and life-cycle boundaries | Keep environmental results scoped and unqualified where source/model is absent |

Each vertical uses one explicit status: `supported`, `partial`, `not_configured`, or `license_blocked`. The `supported` and `partial` states require declared source references; every source reference must resolve to a source record with access class, licence reference, `known_at`, validity start and source status. These statuses describe integration coverage, not the truth of the underlying source.

## Contract rules

1. **Regional scope is not necessarily national.** Record country code, optional subnational scope, market scope, locale tag and supported language tags independently.
2. **No universal local database assumption.** Preserve the native source ID, component/nutrient identifier, food matching rationale, sample/preparation state, denominator, release/version and quality metadata.
3. **Conversions are evidence-bearing transformations.** Store original value/unit/basis, conversion rule and version, any density or edible-yield assumption, and transformed value/unit/basis. Mass-volume conversion requires a justified density/basis; serving-to-100-g transformations require serving mass.
4. **Currency conversion is separate from affordability comparison.** Record amount, currency, date/time, market, product/package identity and rate source/date if converted. Do not treat PPP as a retailer price. Keep current price, historical series, CPI, PPP and household affordability basket as different series types.
5. **Rights are executable constraints.** `open_reuse` and `attribution_required` can qualify for redistribution only when their actual terms allow it. `permission_required`, `restricted` or `unknown` can never be declared redistributable by default. A source's presence in a public directory does not imply endorsement or reuse permission.
6. **Evidence time has two axes.** Preserve `valid_at`/validity window and `known_at`. Historical replay must not consume facts that were published/received later.
7. **No region-to-culture inference.** Country, language, religion, ethnicity and residence are not substitutes for volunteered preferences. Regionalized UX may offer examples but should ask, allow skipping and keep preferences editable.
8. **Do not flatten food safety.** Jurisdiction-specific label regulations, packaged product labels, cross-contact, medical allergies, food hygiene and clinical advice are distinct. A data pack does not certify an actual lot/meal as safe.
9. **Privacy by default.** Spatial price/availability data should use permissioned aggregation. No home coordinates, receipts, exact household identity or health profile should flow to cross-region analytics unless explicitly needed and consented.
10. **Offline capability matters.** Region packs are versioned snapshots. A missing network must not make a stale price current, remove a gate, or prevent access to cached receipts and offline recipes. Staleness should be surfaced.
11. **Global standards are interop, not imposed vocabulary.** Preserve source-native names/scripts plus linked mappings. Lossy translations must not replace canonical identifiers or community wording.
12. **The profile may be partial.** Unconfigured does not mean unavailable in the real world; it means this software has no qualified evidence path here yet. The system should state exactly what it can and cannot support.

## Conformance and rollout tiers

| Tier | Qualification | Permitted public claim |
|---|---|---|
| G0 — protocol | Schema parses; required vertical inventory is complete; source/licence references and fail-closed invariants pass on synthetic fixtures | “Regionalization contract implemented” |
| G1 — source ready | At least one vertical has a qualified source reference, source rights are reviewed, coverage and freshness limits are recorded | Only the specifically qualified vertical/source scope |
| G2 — replayable | Frozen source snapshots and known-at/valid-at replay match independently computed expectations | “Historical replay verified for this snapshot and scope” |
| G3 — local utility | Prospective, consented evaluation against transparent local baseline; preregistered measures and uncertainty | Results for the evaluated population/market only |
| G4 — cross-region robust | Independent local review in multiple distinct contexts; per-region accuracy, coverage, rights and fairness reports; no pooled average hides a failing region | Only enumerated and actually qualified region/vertical combinations |

The synthetic South Africa, India and Brazil fixtures are protocol examples only. Their `profile_status` is `synthetic_example`, their source lists are empty, and all ten verticals remain `not_configured`. They are not assertions that the system has country-specific nutrition, prices, regulations, languages, cooking conventions or cultural preferences qualified.

## Proposed benchmark targets

These are targets, not measured results:

- **Schema and declared coverage:** 100% of packs explicitly declare all ten verticals; no duplicate vertical/source IDs; all source refs resolve.
- **Rights:** zero public redistribution of a source marked permission-required, restricted or unknown; zero copied rows if rights/terms are unresolved.
- **Fail-closed behavior:** zero unconfigured or license-blocked verticals silently treated as supported; zero missing values turned into numeric zeros; zero unknown hard gates compensated by other objectives.
- **Temporal correctness:** zero source records known only after a replay cutoff included as then-known evidence.
- **Conversion correctness:** 100% agreement with independent test vectors for unit conversions that are mathematically exact; density-, yield-, cooked-state- and serving-dependent conversions must fail without declared evidence. Publish propagated uncertainties for measured conversion inputs.
- **Language and locale:** 100% round-trip stability for machine numeric values/IDs across tested locale render/parse cycles; no localized decimal formatting allowed to mutate the stored number, units or canonical IDs. Test Unicode names/scripts and translation gaps.
- **Food match:** report exact/near/weak/unmatched coverage and expert-adjudicated false-match rates separately by category and region. Set quality thresholds only after an expert-labelled baseline; a high matching coverage must not be achieved by silently choosing a wrong food.
- **Price comparability:** zero comparisons between unresolved package sizes, incompatible units, incompatible product identity or mismatched retail/wholesale series without an explicit conversion and scope label.
- **Cultural autonomy:** zero automatic assignment of dietary/cultural preference from country code; report opt-out, preference-editability and unserved-constraint counts.
- **Fairness:** publish each region/stratum's feasible-choice rate, constraint violations, actual-cost error and missing-evidence rate. Never let a large, data-rich region hide a region with little or no evidence.
- **Performance:** measure pack load/validation, cache size and p50/p95 lookup latency at 1, 10, 100 and 1,000 packs before setting regression budgets. Keep cold-start and offline behaviour separate.
- **External validity:** no global accuracy or benefit claim from the synthetic fixtures. Each G3/G4 claim must name region, verticals, data release, language(s), time range, population, comparison arm, sample size and uncertainty.

## Priority sequence

1. Land and review the portable regional pack schema + semantic validator.
2. Run schema validation and Rust tests on the exact commit when a Rust toolchain is available; static JSON inspection is not a substitute.
3. Implement a standalone converter/matcher conformance corpus based on FAO/INFOODS food matching and units/denominator guidance. Do not import data without rights review.
4. Create a source rights / coverage registry; begin with permitted South African references and a small, adjudicated local basket rather than pretending all ten verticals are available.
5. Use official food-composition directories to identify candidate regional data partners, licensing conditions and quality metadata. Qualify one vertical at a time.
6. Add India and Brazil as test contexts only until country/domain experts verify actual sources and local requirements. Add further regions through the same process, not one global scrape.
7. Build per-region held-out benchmark splits; no source/food alias/recipe leakage between train/dev/test, and freeze policy, pack and corpus digests before evaluation.

## References

- FAO/INFOODS, [About INFOODS](https://www.fao.org/infoods) and [Standards & Guidelines](https://www.fao.org/infoods/infoods/standards-guidelines/en/).
- FAO, [Food Composition program](https://www.fao.org/food-composition/en) and [Directory of food composition tables/databases](https://www.fao.org/food-composition/tables-and-databases/14/en).
- FAO/INFOODS, [Guidelines for Food Matching, v1.2](https://www.fao.org/fileadmin/templates/food_composition/documents/Nutrition_assessment/INFOODSGuidelinesforFoodMatching_version_1_2.pdf).
- WHO and FAO (2024), [What are healthy diets? Joint statement](https://www.who.int/publications/i/item/9789240101876).
- Codex Alimentarius Commission (7 July 2026), [Precautionary allergen labelling guidelines](https://www.fao.org/fao-who-codexalimentarius/news-and-events/news-details/en/c/1759963/).
- World Bank, [International Comparison Program FAQ](https://www.worldbank.org/en/programs/icp/faq).
- Brinkley & Gallo-Franco (2025), [Food composition database FAIR data review](https://www.frontiersin.org/journals/nutrition/articles/10.3389/fnut.2025.1552367/full).

## Limitations

The module performs a semantic consistency check on a JSON value and intentionally is not a full JSON Schema engine. The presence of source references does not qualify source truth. The synthetic regional packs do not encode real market data, food composition, local standards, cultural norms or approved recommendations. No Rust tests have been run in the current environment.
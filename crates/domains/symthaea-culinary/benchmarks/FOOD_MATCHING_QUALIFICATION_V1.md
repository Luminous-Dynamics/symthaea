# Global food identity matching and comparability benchmark v1

**Status:** design and synthetic conformance implementation draft. The API validates an asserted match; it is not itself a candidate retriever, ontology matcher, nutrition calculator, or proof that external food records are true. Runtime tests have been authored but not compiled/executed in this environment.

## Research basis

Food matching is a data-quality problem before it is a string-similarity problem.

- FAO/INFOODS publishes dedicated [Guidelines for Food Matching, Version 1.2 (2012)](https://www.fao.org/infoods/infoods/standards-guidelines/en/) for selecting appropriate composition records for food-consumption or food-supply records. It separately publishes [Guidelines for Converting Units, Denominators and Expressions](https://www.fao.org/infoods/infoods/standards-guidelines/en/), because apparently comparable values can have different denominators and expressions.
- EuroFIR identifies unambiguous food descriptions, source-data quality, provenance, coding checks, aggregation and validation as distinct steps in food-composition database work. It describes LanguaL as a controlled vocabulary used for structured food descriptions. [EuroFIR quality and standards](https://www.eurofir.org/food-information/quality-standards-of-fcdbs/).
- LanguaL describes facets for biological origin, cooking/preservation, and technological treatments; it also warns that words translated between languages are not always semantically equivalent. Its website points to FoodOn as a developing ontology for food entities interoperable with the OBO Library. These are candidate *external reference systems*, not a reason to discard native identifiers or assume that external terms are complete for each region. [LanguaL](https://www.langual.org/).
- USDA FoodData Central data are publicly available under CC0, while local sources can have different permissions. SAFOODS explicitly requires written SAMRC permission before reproducing its database content. A universal matching pipeline therefore needs rights-aware source references instead of assuming all records can be copied into a public training corpus. [USDA FDC licensing](https://fdc.nal.usda.gov/); [SAFOODS FAQ](https://safoods.mrc.ac.za/faq.html).

## Paradigm shift

**From multilingual name lookup to context-specific, evidence-bearing identity decisions.**

A name match is a candidate-generation signal only. It must not erase differences in biological identity, food part, processing, raw/cooked state, preservation, product/brand, fortification, edible portion, or nutrient basis. Similarity models may propose candidates but must not overrule a hard semantic contradiction.

The system exposes four outcomes:

| Outcome | Meaning |
|---|---|
| `exact_match` | All required identity attributes are known, consistent with the cited records and matching for the declared use scope. This says nothing by itself about source truth, nutrient quality, safety, or suitability. |
| `qualified_approximation` | A limited, explicitly stated approximation for nutrition estimation or recipe substitution only, with every known difference/unknown documented and reasoned. It is not product equivalence. |
| `review_required` | Key attributes are missing, conflicted or not reliably mapped; obtain source evidence or an expert/local review. |
| `no_match` | A conflict is known, or no defensible mapping is available for the declared purpose. |

Approximate matches are forbidden for price comparison, product lookup and allergen screening in this contract. An `exact_match` here means identity attributes match for the declared task; it does **not** by itself establish price comparability. Price comparisons separately require package/SKU or weight normalization, currency and conversion provenance where applicable, market and observation time, and comparable availability basis. A food-identity match also does **not** prove a packaged product is free from an allergen, has no cross-contact risk, or is safe to consume. Those claims require their own source/label/lot evidence and domain-specific policy.

## Implemented contract

The draft public API is:

```rust
symthaea_culinary::food_matching::validate_food_identity_match_v1(
    &serde_json::Value
) -> Result<(), Vec<String>>
```

Schema: [FoodIdentityMatchV1.schema.json](../schemas/FoodIdentityMatchV1.schema.json).

Each query/candidate record preserves:
- stable record ID, source reference, and source version;
- one or more source/localized names with language tags;
- identity fields for taxon, food part, processing, preparation, product kind, edible basis, preservation, brand, fortification, cultivar, and origin scope;
- explicit attribute relation plus both source-derived values, so a match assessor cannot claim a relation that disagrees with the referenced identity object;
- component/basis comparability and a transformation reference when conversion is claimed;
- reasoned deviations for any approximation.

The validator recomputes consistency for the provided assertions. It does **not** generate food matches, prove the source records, authenticate ontology identifiers, or implement a complete JSON Schema 2020-12 engine. Consumers should validate the wire schema and separately qualify source rights, source release/currentness, and taxonomy mappings.

## Conformance case families

The first synthetic conformance suite covers:

1. Different language aliases for the same structured food identity.
2. Same/similar labels but different species/taxon.
3. Raw vs cooked, dried vs fresh, and different processing/preparation states.
4. Different edible part or denominator (edible portion vs dry matter, as-purchased vs drained).
5. Generic ingredient vs branded packaged food.
6. Fortified vs unfortified products.
7. Cultivar differences that may permit an explicitly qualified recipe substitution, but not silent identity equivalence.
8. Missing taxonomy or preparation evidence requiring review.
9. Component basis differences requiring an explicit conversion reference.
10. Duplicate or contradictory attribute checks and attempts to rewrite source identity through a claim.
11. Invalid/missing language tags and aliases that collide across locales.
12. Source records with unknown/expired/restricted status (to be added when the canonical source registry adapter is wired in).

The current fixture is synthetic multilingual potato metadata with no real nutrient values. All negative controls should remain synthetic until a rights-reviewed, expert-adjudicated dataset can be used.

## Evaluation design

### Comparison arms

On identical candidate budgets and source snapshots, compare:
1. normalized lexical matching;
2. multilingual embedding retrieval, if used, with no authority to accept matches by itself;
3. structured attribute matching with explicit source/ontology identifiers;
4. retrieve-then-apply-hard-rules (the proposed safe baseline);
5. two independent food-composition experts plus adjudication as reference oracle, when permitted by data rights.

Keep candidate retrieval recall separate from the correctness of a final accepted match. High recall is not useful if it produces false equivalence.

### Metrics

Report, at minimum:
- **Exact-match precision** and false-exact-match rate, stratified by source pair, language, market/region, food class, and attribute type.
- Candidate retrieval recall before decision logic, and final match coverage after abstention/review.
- Hard-contradiction false acceptance: taxon, food part, processing/preparation, branded identity, fortification, or basis conflicts that still result in exact match.
- Review/abstention rate, reviewer effort per resolved case, adjudicator disagreement, and unresolved-case rate.
- Approximation safety: fraction with fully documented deviations; inappropriate approximation accepted for price/product/allergen use must be zero.
- Component comparability failures: denominators, units, edible fraction and raw/cooked-state conversions improperly declared comparable.
- Cross-language performance gaps; local-language synonyms and transliterations vs global scientific/controlled IDs; newly added held-out languages.
- Rights, provenance and freshness failures; an unavailable source or license-blocked record must not be silently used as a factual match.
- Determinism and reproducibility across candidate/order permutations and exact policy/ontology/source versions.
- Runtime and memory at 1k, 10k, 100k, and 1M candidate records after a real implementation baseline exists.

Never report a single pooled accuracy alone; a strong overall score can hide a severe failure for a food category or language with less data.

### Proposed acceptance targets

These are goals, not measured achievements:

- **Synthetic contract:** 100% agreement with independently enumerated expected outcomes; zero acceptance of a contradiction in any hard-block negative control.
- **Exact match:** provisionally target at least 99% exact-match precision on a separately adjudicated held-out set, with uncertainty intervals and all strata reported. Adjust this target only before opening the final held-out set.
- **Critical negatives:** zero false exact matches among deliberate taxon/food-part/preparation/product/fortification/basis contradictions. If we later claim a false-accept rate below roughly 0.1% from zero observed errors, approximately 3,000 representative hard-negative trials would be needed for a simple 95% rule-of-three upper bound; this must be established per meaningful stratum, not by pooling unlike cases.
- **Unknown evidence:** 100% of missing/conflicting required identity fields lead to review/no-match rather than an unqualified exact match.
- **Purpose boundary:** zero approximate matches accepted for product lookup, price comparison or allergen screening.
- **Nutrient comparability:** 100% of changed-denominator/changed-state comparisons either have a validated, referenced conversion or remain not-comparable.
- **Rights:** zero examples from restricted sources copied into public fixtures/corpus without permission. Synthetic cases must say they are synthetic.
- **Language robustness:** publish precision, recall, abstention and reviewer workload for each tested language/locale; no requirement that every culture or food category have identical matching coverage on day one.
- **Generalization:** zero source-record/alias leakage across train/dev/test; split by food concepts and source records, not randomly by aliases of the same underlying food.
- **Performance:** set p50/p95 budgets only after a reproducible implementation baseline with named hardware and commit.

## Global data strategy

- Treat food names as aliases linked to source records and controlled identifiers, never as the canonical identity.
- Reuse taxonomy/crosswalks by reference; pin their version and rights. Keep original label/script, transliteration, language tag, locale scope, source and reviewer provenance.
- Keep country and market context distinct. A shared name can designate different products in different markets; national origin is not itself proof of food identity or culture.
- Start from rights-permitted datasets such as USDA FDC CC0 where useful for mechanics and negative controls; do not represent them as local South African truth.
- For SAFOODS, obtain written SAMRC permission before reproducing data in tests, bundles or public distributions. Until then, use only allowable citations/metadata and synthetic fixtures.
- Qualify local records in small, adjudicated strata. Record exact/local alias, ambiguous alias, non-equivalent look-alike, preparation state, edible basis, source and reason. Leave genuinely ambiguous matches unresolved.
- Link the regionalization pack to the canonical NUTRI-SOURCE source registry; don't create a duplicate nutrition source-of-truth inside the matching module.

## Integration ownership

- **Symthaea Culinary:** context-aware match assessment and comparability disposition.
- **NUTRI-SOURCE-001 / canonical source registry:** source identity, rights, source release and freshness state.
- **Regionalization pack:** which source/crosswalk paths are qualified for a market and language.
- **Mycelix:** canonical source access/consent/lot/correction and recall events where applicable.
- **Symtropy:** portfolio/resource consequences and measurement prioritization; do not build a second matching/VOI engine there.
- **Sol Atlas:** market/locale/season context and privacy-bounded geography, not an authority for food identity or live stock.

## References

- FAO/INFOODS, [Standards & Guidelines](https://www.fao.org/infoods/infoods/standards-guidelines/en/).
- FAO/INFOODS, [Food Composition Data Interchange](https://www.fao.org/infoods/infoods/standards-guidelines/international-interchange/en/).
- EuroFIR, [Quality & Standards of Food Composition Databases](https://www.eurofir.org/food-information/quality-standards-of-fcdbs/).
- LanguaL, [International Framework for Food Description](https://www.langual.org/).
- USDA, [FoodData Central and licensing](https://fdc.nal.usda.gov/).
- SAMRC SAFOODS, [FAQ including database reproduction permissions](https://safoods.mrc.ac.za/faq.html).
- Existing Symthaea [NUTRI-SOURCE-001 #7481](https://github.com/Luminous-Dynamics/symthaea/issues/7481), [NUTRI-RECEIPT-002 #7503](https://github.com/Luminous-Dynamics/symthaea/issues/7503), and [regionalization PR #8043](https://github.com/Luminous-Dynamics/symthaea/pull/8043).

## Qualification limits

Rust compilation/tests are not confirmed in the current environment. The schema and synthetic fixture have been fetched from GitHub and parsed as JSON, but this is not a complete schema-validation run. The contract does not create a production multilingual matcher, import or license real datasets, or establish local composition accuracy, food-safety, clinical suitability, or household benefit.

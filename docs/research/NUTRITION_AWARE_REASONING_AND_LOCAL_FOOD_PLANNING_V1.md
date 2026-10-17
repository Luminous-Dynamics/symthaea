# Nutrition-Aware Reasoning and Local Food Planning v1

**Status:** research/interface contract, not a deployed nutrition advisor or crop optimizer.  
**Ownership:** Symthaea integrates and explains; Symtropy owns numerical production scenarios; Sol Atlas owns site/geospatial evidence; Mycelix owns community governance, rights and commitments.  
**Scope:** community/population food access and general nutrition education—not individualized medical nutrition therapy.

## 1. Decision

Symthaea should develop substantially stronger nutrition understanding, but it should not simply learn to *sound* confident about nutrients. It should retrieve and reason over versioned, source-attributed food-composition facts, dietary reference values, food-based guidance, local food preferences, seasonal availability and explicit uncertainty. Deterministic calculations belong in typed tools with independently checked outputs, not free-form language alone.

The system goal is not “maximize crop yield” or “maximize nutrient density per kilogram.” It is:

> Improve reliable, affordable, culturally acceptable access to adequate and diverse diets while respecting ecology, water, labour, land, food safety, consent and community choice.

A garden or regional network may meet some needs and not others. The planner must expose individual nutrient shortfalls and food-group gaps rather than hide them behind high total calories or one composite score. Regional purchasing, community kitchens, food preservation, fortified staples and other supply options remain valid outcomes.

## 2. Division of responsibilities

| Component | Canonical responsibility | Must not claim |
|---|---|---|
| **Sol Atlas** | Site-bound geometry, agroclimate and source evidence, water/soil assessment references, seasons, land permissions and spatial access constraints | Imagery alone proves soil safety, potable water, crop yield, species identity or permission |
| **Symtropy** | Deterministic candidate-production scenarios: crops/food options, area, seasons, yields, harvest windows, losses, water, energy, labour, storage, transport, cost and uncertainty | Synthetic yields are observations, a ledger is physical validation, or mathematical feasibility is agronomic qualification |
| **Symthaea** | Source-grounded nutrition reasoning; compare plans; identify missing/conflicting inputs; explain trade-offs and ask before encoding community preferences | Medical diagnosis, treatment, individualized therapeutic diets, or nutrient adequacy claims from mismatched/missing data |
| **Mycelix** | Community authorization, knowledge/data rights, producer/receiver commitments, provenance, fair access and cooperative allocation | A public ledger substitutes for consent, privacy, legal obligations or human oversight |

Keep this as a contract across existing owners. Do not create another food database, scheduler, geography authority or simulator when a canonical owner already exists. Add shared types or crates only after a capability review demonstrates a real semantic gap.

## 3. Evidence and source selection

Every food/nutrition claim should carry:
- dataset/source ID, edition or revision, source record ID and exact food description;
- food state and preparation (raw, boiled, dried, drained, edible portion);
- nutrient/component identifier, value, unit and denominator;
- analytical, literature-derived, label-derived or estimated status when available;
- geographic/cultivar match and quality of that match;
- license/access terms, evaluation date and source-artifact digest when redistribution is permitted;
- missingness distinguished from zero;
- transformation lineage for unit conversion, recipe calculation, edible mass, storage loss and retention assumptions.

Do not silently merge unlike foods. A local cultivar, wild/underutilized food, fortified staple, cooked ingredient and generic database food are not automatically interchangeable. Apply a documented food-matching hierarchy; retain the original record and each substitution. Prefer suitable country/regional composition data; use a broader source only as an explicitly reviewed fallback. A weaker match is a visible downgrade, not an invisible substitute.

### Initial source registry

1. **SAFOODS — South African Food Data System.** The product catalogue lists the South African Food Composition Tables, 6th edition (2024), with 1,857 foods, 36 nutrients and 16 food groups. The public web interface exposes only a limited subset—energy, carbohydrate, protein, fat and fibre—per 100 g edible food; it is not the complete table. The SAFOODS FAQ explicitly says written permission from the South African Medical Research Council is required before reproducing any part of the database in another product, including for educational use. Public visibility is not a redistribution license: do not scrape or bundle displayed values in public fixtures unless permission covers that use. If acquired, store the license basis and permitted uses in the source manifest. [Food composition overview](https://safoods.mrc.ac.za/composition.html) · [FAQ on reproducing data](https://safoods.mrc.ac.za/faq.html) · [Product catalogue](https://safoods.mrc.ac.za/products.html)
2. **FAO/INFOODS food-composition tables and databases.** Use the directory to locate suitable regional/national tables and identify the correct edition. Inclusion in a directory does not by itself grant redistribution rights; verify the exact dataset's license and permitted use. The FAO/INFOODS directory lists **AFROFOODS** as a regional centre covering South Africa and **ECSAFOODS** as a subregional centre also covering South Africa. These are useful expert/data-discovery routes, not proof that a particular table is openly licensed. [Tables and databases](https://www.fao.org/food-composition/tables-and-databases/) · [African data centres](https://www.fao.org/food-composition/infoods/data-centres/en) · [Africa tables and databases](https://www.fao.org/infoods/infoods/tables-and-databases/africa/en/)
3. **FAO/INFOODS data quality and food matching.** Assess data quality and document missing components, conversions, imputation and matching decisions. [Data quality guidance](https://www.fao.org/infoods/infoods/standards-guidelines/data-quality/en/) · [Food Matching Guidelines](https://www.fao.org/fileadmin/templates/food_composition/documents/Nutrition_assessment/INFOODSGuidelinesforFoodMatching_version_1_2.pdf)
4. **WHO healthy-diet guidance.** Use as broad, dated public-health context, not a universal personal prescription. [Healthy diet, updated 26 January 2026](https://www.who.int/news-room/fact-sheets/detail/healthy-diet)
5. **South African food-based guidance.** Department of Health food-labelling guidance illustrates food-group guidance around vegetables/fruit, legumes, starchy foods and protein/dairy options. It is not a nutrient database or evidence of crop suitability. [Food Labelling Guidelines (2023)](https://www.health.gov.za/wp-content/uploads/2023/05/18052023_2023-Food-Labelling-Guidelines-for-website-updated-on-18-May2023-sc.pdf)
6. **FAO nutrition-sensitive agriculture and water productivity.** Production planning should connect water productivity to nutrient-dense crops and dietary diversity. [Water for Nutrition](https://www.fao.org/in-action/water-for-nutrition/en)

7. **USDA FoodData Central (fallback candidate, not South African ground truth).** USDA states its FoodData Central data are public domain under CC0 1.0 and may be used without permission; retain source citations and the original FDC record identifiers. Use this only after explicit food/state matching review, and label geographic/cultivar mismatch as lower-confidence evidence rather than silently substituting a US record for a South African food. [Official API guide and licensing](https://fdc.nal.usda.gov/api-guide/)

A source name alone is not provenance. Record its exact edition, record and permitted transformation rather than inventing a citation because a food or nutrient sounds plausible.

### Paradigm shift: from permission flags to resolvable evidence records

An enum saying “allowed” and a string saying “license-clause” are not evidence by themselves. Every affirmative source-right decision must resolve to a typed evidence record carrying its subject source, exact right, grant basis (license terms or written permission), evidence URI, review date, validity end and lifecycle. The validator blocks unresolved or duplicate identifiers, cross-source/cross-right references, basis mismatch, malformed or out-of-window dates, invalid URIs, and revoked/withdrawn/needs-review evidence. Regression controls include these decision boundaries and a positive control for a current, correctly scoped written permission.

A second shift is to make each decision receipt bind to both exact evaluation inputs and the outcome being reported. The registry emits the evaluation date, source ID and record ID, input schema versions, declared artifact digest, digest of bytes actually supplied, independent SHA-256 digests for the manifest, food fact and caller use request, and an overall SHA-256 over an input envelope that includes all inputs, the evaluation date and supplied-byte digest. JSON is now canonicalized with the pinned `serde_json_canonicalizer = 0.3.2` implementation and the profile identifier `rfc8785_jcs_v1` is inside the envelope preimage. This changes receipt digest values and must be treated as a profile migration rather than an invisible implementation detail. New negative controls verify that changing the date, caller intent or artifact bytes changes the receipt digest. A component-level regression test also mutates a food value and a permission-evidence lifecycle, asserting that the matching component digest and overall input digest change. The receipt has a distinct schema version from the source/fact schema, and wide nutrient `u64` values plus their scale and gram basis serialize as decimal strings rather than JSON numbers; absent nutrient values remain `null`. This is deliberate because RFC 8785 explains that JSON numbers should be limited to IEEE-754-safe values for interoperable canonicalization and recommends strings for larger precise integers ([RFC 8785](https://www.rfc-editor.org/rfc/rfc8785.html)). The receipt schema is now version 2: it names `rfc8785_jcs_v1`, uses pinned `serde_json_canonicalizer = 0.3.2`, and records evaluator package name/version, a separate evaluator-contract version and an optional `LUMINOUS_BUILD_ID`. Two RFC-derived fixed-hash regression vectors exercise number/string serialization and UTF-16 property ordering. A separate fixed-hash test locks the typed manifest, food fact, use request and full input-envelope digests for the synthetic contract fixture, so changes to field serialization or envelope shape require an explicit golden update. The `decision_sha256` covers all serialized receipt fields except itself; it is still an unkeyed checksum, not a signature, and does not authenticate publishers or claims. An independent CLI checker now lives at `crates/domains/symthaea-nutrition-source-registry/src/bin/nutrition-receipt-verify.rs`. It rejects duplicate property names before typed parsing and independently recomputes component, input-envelope and decision-receipt digests without invoking the production assessor. It shares the declared Rust wire structs and the pinned JCS library; it checks digest consistency, not eligibility semantics. Tests cover a valid receipt, altered outcome/fact, missing artifact bytes, duplicate keys, Unicode and null. Exact-head CI and a separate cross-language verifier remain required before claiming full conformance or interoperability.

This is a *resolvable, proof-carrying data contract*, not a cryptographic legal oracle. References and metadata help reproduce a rights decision but do not authenticate the publisher, fetch/check the underlying text, or establish legal sufficiency. A future verifier of permission-document bytes must receive the bytes separately and clearly distinguish hash matching from legal review; a hash string without corresponding bytes is not verification.

For future cross-system interchange, align provenance concepts with the W3C PROV model—entities (source artifacts and derived facts), activities (retrieval, review, preparation transformation and evaluation), and agents (publishers, reviewers and software)—without forcing RDF into the initial Rust crate. PROV-O is a W3C Recommendation for representing and exchanging provenance ([W3C PROV-O](https://www.w3.org/TR/prov-o/)). Any signature/credential layer should remain a later, separately reviewed step; W3C Verifiable Credentials define a model for issuer-attributed claims and tamper-evident exchange, but are not a substitute for trust in the issuer or legal review ([W3C VC Data Model 2.0](https://www.w3.org/TR/vc-data-model/)).

This direction combines general data-stewardship practices—machine-actionable license/access conditions and detailed provenance—with food-composition-specific checks for data quality and food matching ([FAIR principles](https://www.nature.com/articles/sdata201618); [NIST summary](https://www.nist.gov/itl/ssd/information-systems-group/configurable-data-curation-system-cdcs/cdcs-help-and-resources-1); [FAO/INFOODS standards](https://www.fao.org/infoods/infoods/standards-guidelines/en/); [FAO/INFOODS data quality](https://www.fao.org/infoods/infoods/standards-guidelines/data-quality/en/)).

Every affirmative rights decision—whether allowed by a license or granted by written permission—must be backed by a non-empty evidence reference scoped to that right. The source registry stores separate references for assessment, cache/retention, transformation, commercial use, public fixture use and redistribution; one reference cannot silently satisfy a different right's requirement. A reference is only a pointer to a license clause or permission record, not automatic proof of scope, authenticity, legal sufficiency or continued validity.

### Johannesburg/Highveld candidate-food hypothesis (not yet a planting recommendation)

The South African food-based dietary guidelines organize a healthy diet across starchy foods; vegetables and fruit; dry beans, peas, lentils and soya; poultry/fish/meat/eggs; milk/maas/yoghurt; fats/oils; and water. This suggests that a garden-only optimizer will be structurally incomplete: it should plan what can be grown locally while leaving explicit supply gaps for foods/groups that require a wider cooperative, market or fortification network. [FAO summary of South African food-based dietary guidelines](https://www.fao.org/nutrition/education/food-dietary-guidelines/regions/countries/south-africa/en/).

Candidate research to evaluate for local food acceptance, growing feasibility and properly matched composition data includes morogo/amaranth leaves, pumpkin leaves, cowpea leaves, spider flower (Cleome gynandra) and wild jute (Corchorus olitorius). South African studies report nutrient measurements for traditional leafy vegetables, but the data are not interchangeable across species, sampled regions or raw/cooked states; one study of morogo also found that cooking with discarded water changed measured nutrient/bioactive amounts. These papers justify targeted data collection and local review—not a blanket nutrient claim or automatic crop recommendation. ([Schönfeldt & Pretorius, 2011, University of Pretoria repository](https://repository.up.ac.za/handle/2263/18102); [morogo cooking/drying study](https://pmc.ncbi.nlm.nih.gov/articles/PMC3982004/)).

For each candidate, first verify correct edible identity, local growing window, seed access, water/soil safety, land permission, community preference and food/preparation-state match. Do not promote a plant based solely on a paper's high single-nutrient measurement. Composition data should be locally relevant and license-reviewed; where local assay/validated data are absent, expose the uncertainty and use a wider-supply option rather than overstate nutritional self-sufficiency.

## 4. Objective: seasonal, constrained and multi-objective

Compare candidate portfolios across several explicit axes:

- **Nutrient adequacy:** estimate delivered supply for each nutrient and food-group target on a declared population/time basis. Show shortfalls separately. Unknown remains unknown; never replace it with zero or an average without a declared, reviewed imputation.
- **Dietary diversity and acceptability:** coverage across food groups, familiar staples, cultural preferences, preparation options and feedback from participating people.
- **Energy and macronutrients:** estimate edible calories, protein, fats and carbohydrates, without letting extra calories cancel a micronutrient gap.
- **Production feasibility:** suitable growing windows, area, credible yield ranges, seed/propagation access, crop duration, soil/water status, rotation and qualified growing guidance.
- **Resources:** period-scoped water, energy, labour, skills, capital, tools, storage, transport, waste handling and capacity.
- **Resilience:** seasonal shortfalls, staggered harvests, supplier/crop diversity, sensitivity to drought/cold/heat/outage, shelf life, correlated failures and recovery time.
- **Economics and access:** cost, affordability, availability and distribution. Do not equate market revenue with nutritional value or community benefit.
- **Ecology, safety and rights:** biodiversity, soil health, water stress, contamination evidence, land permission, labour dignity, knowledge rights and equitable access.

Hard constraints must not be outweighed by an attractive score. For example, a cheap, high-yield plan remains infeasible if its water source is unreviewed or its growing window is unsupported. Return a Pareto set or transparent scenario comparison with editable community-set weights—not one unexplained scalar labelled “optimal nutrition.”

Annual yield can hide months without food, so the time interval must be explicit; monthly planning is a useful first comparison interval. Convert harvested biomass to edible delivered food, not merely kilograms produced:

**edible delivered = harvested mass × edible fraction × (1 − post-harvest loss) × allocation share**

Every factor needs a source or an explicit synthetic assumption. Nutrient estimates then use a matching composition record and declared prepared-food basis. Include preservation/cooking changes when material; do not silently assume them away.

## 5. Typed data contract

The initial data contract should separate:
- food identity and taxonomic/cultivar confidence from the food-composition record;
- nutrient/component identity, value, unit and denominator;
- edible portion, preparation state and retention assumptions;
- edible yield and season/harvest-window evidence;
- nutrient target/requirement with population and life-stage scope, source, date and jurisdiction;
- dietary preferences and food-group policy with an accountable community scope;
- production option with area, season, water, labour, cost and uncertainty;
- supply scenario with declared boundary, per-nutrient contributions, missing data, evidence class, blocker list and source manifest.

Quantities are dimensional. Never add milligrams to grams or compare dry-matter values with cooked edible-weight values without a reviewed conversion. Never conflate calories, mass, servings, protein and micronutrient adequacy.

Propagate source-provided ranges where available. If a source gives one point value, mark it as a point estimate rather than inventing an uncertainty interval. Distinguish four outcomes:
- **Adequate under declared assumptions**
- **Potentially adequate, unresolved uncertainty/evidence**
- **Shortfall under declared assumptions**
- **Not assessable** because critical inputs are missing or incomparable

These describe a scenario; they are not a clinical judgment about an individual.

## 6. Safety, privacy and rights

- No disease diagnosis, treatment, supplement/dose direction or therapeutic dietary prescription from the production planner.
- Personalized nutrition needs—including infant feeding, pregnancy, renal disease, diabetes, allergies, eating disorders or medication interactions—require suitable qualified professional oversight and reviewed clinical sources.
- Aggregate supply does not prove every person receives a suitable diet; averages can hide unequal access or special needs.
- Do not assume local crops can provide every nutrient. Model purchased foods, fortified staples or other supply options where relevant, and show residual gaps rather than falsely claiming self-sufficiency.
- Water potability, soil contamination, pesticide risk, plant identity and food safety are separate gates. A nutrient-rich food is not safe when contamination evidence is unknown.
- Minimize data. Do not place household names, patient profiles, exact residential locations, sensitive health data or private traditional knowledge in shared/public scenarios.
- Traditional, sacred or community-held knowledge cannot be ingested or republished simply because it is discoverable. Respect consent, cultural authority, benefit sharing and withdrawal.
- Nutrition score or land-access record never implies consent to grow, harvest, sell, test, process or share a plant.

## 7. Validation and negative controls

Start with deterministic offline synthetic fixtures. Each calculation/report must be independently recomputable from frozen inputs and carry input digest, dataset versions, planner version, units, time range and output digest. Hand-calculated small cases must precede broad benchmarks.

Required negative controls:
1. More kilograms but less dietary diversity must not automatically win.
2. High calorie supply cannot cancel a declared iron/calcium/other nutrient gap.
3. Missing composition is unknown, not zero or a silent generic-food substitution.
4. Dry-weight and cooked/edible-weight records cannot be mixed without a conversion contract.
5. Stale or wrong-season yields cannot be labelled a current prediction.
6. Site/water/permission/food-safety blocks cannot be offset by a good nutrient score.
7. Extreme integers cannot overflow or round through JSON/IEEE-754 representation.
8. Harvest losses, rejected lots, storage spoilage, transport delays and unequal allocation reduce delivered supply.
9. A supplier failure must remain visible, not averaged away.
10. Incomparable plan trade-offs remain a comparison, not a fake unique optimum.
11. Non-redistributable composition data cannot be bundled in public fixtures.
12. Disease-specific prescriptions or sensitive personal profiles route to qualified review, not the optimizer.
13. Aggregate planning cannot output “every person is nutritionally safe.”
14. Unknown soil/water risk cannot pass because other site dimensions look favorable.

Validation sequence: (a) nutrition-qualified review of food matching; (b) agronomic review of local growing assumptions; (c) independent arithmetic/report checker; (d) supervised comparison with measured harvest, edible loss and weighed food quantities; (e) community review of affordability, acceptability, rights and distribution. Keep gates separate and preserve failing fixtures.

## 8. Staged implementation

**A. Evidence quality.** Build a source registry and provenance schema. Review data licenses and choose South African sources before global fallback. No licensed dataset is embedded until access and redistribution terms are understood. Implementation contract: [NUTRI-SOURCE-001](https://github.com/Luminous-Dynamics/symthaea/issues/7481), including explicit license/permission fields, versioned source manifests, digest policy, food-state matching and fail-closed fixture rules.

**B. Deterministic calculator.** Compute edible delivered mass and per-nutrient contribution from source-pinned inputs with dimensional validation. Independent recomputation and missing-data behavior are release gates.

**C. Scenario comparison.** Symtropy's first executable layer now aggregates source-pinned nutrient contributions across multiple food-delivery lines against a declared nutrient target, preserving per-line receipts and blocking missing composition, duplicated delivery references, cross-scope/period mixing and overflow. It is still not a crop/supplier portfolio optimizer. The next step is [NUTRI-PORTFOLIO-001 in Symtropy issue #1649](https://github.com/Luminous-Dynamics/symtropy/issues/1649): a deterministic, synthetic-first Pareto comparison across per-nutrient shortfalls, seasonal delivery, water, area, labour, storage, cost and resilience. That stage is gated on exact-head validation of [Symtropy PR #1637](https://github.com/Luminous-Dynamics/symtropy/pull/1637) and requires licensed composition data and reviewed local yield evidence before any real-world crop recommendation. Check existing capabilities before introducing another solver or runtime.

**D. Symthaea integration.** Retrieve source-backed facts, state assumptions, request missing preferences, call the deterministic planner, verify the report and explain trade-offs. HDC/CfC may help associate and maintain context, but do not replace evidence validation or numerical solvers.

**E. Community/field validation.** Compare forecasts with edible harvest and food actually delivered; evaluate data quality and disparities; run the negative controls; publish only permitted aggregate results. Claims must not outrun field evidence.

## 9. Acceptance criteria

- Every nutrient number resolves to a versioned source record, unit, denominator and food-state match.
- Unknown, stale, conflicting, restricted or low-quality matches remain visible.
- Production constraints are evaluated independently of nutrient values.
- Outputs distinguish gross production, edible yield, delivered food, nutrient contribution and inferred adequacy.
- Scenario assumptions are inspectable and reported objectives independently recomputable.
- Composition data are not republished without permission.
- No individualized medical advice, patient inference or clinical adequacy claim emerges from community-food planning.
- Documentation clearly labels tested features, synthetic data and remaining expert/field review.

## References

- WHO, [Healthy diet](https://www.who.int/news-room/fact-sheets/detail/healthy-diet), updated 26 January 2026.
- South African Medical Research Council, [SAFOODS catalogue](https://safoods.mrc.ac.za/products.html), South African Food Composition Tables, 6th edition (2024).
- FAO, [food-composition tables and databases](https://www.fao.org/food-composition/tables-and-databases/).
- FAO/INFOODS, [data quality](https://www.fao.org/infoods/infoods/standards-guidelines/data-quality/en/) and [food-matching guidelines](https://www.fao.org/fileadmin/templates/food_composition/documents/Nutrition_assessment/INFOODSGuidelinesforFoodMatching_version_1_2.pdf).
- FAO, [Nutrition-sensitive agriculture water productivity](https://www.fao.org/in-action/water-for-nutrition/en).
- South African Department of Health, [Food Labelling Guidelines (2023)](https://www.health.gov.za/wp-content/uploads/2023/05/18052023_2023-Food-Labelling-Guidelines-for-website-updated-on-18-May2023-sc.pdf).

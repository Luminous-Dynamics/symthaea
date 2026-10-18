# Wood Material Profile and Applicability Benchmark v0.1

**Owner:** Symthaea  
**Consumers:** Symtropy, Mycelix, Sol-Atlas  
**Evidence class:** synthetic fixtures only

## Research grounding

- USDA Forest Products Laboratory, *Wood Handbook—Wood as an Engineering Material* (FPL-GTR-282, 2021): https://doi.org/10.2737/FPL-GTR-282. The handbook covers wood anatomy, moisture and dimensional change, mechanical behavior, drying, adhesives, biological durability, treatment and fire-related topics.
- Global Wood Density Database v2.2 (published June 23, 2026): https://zenodo.org/records/20815517. The record describes 109,626 density records plus 15,093 bark-density records, with individual/species data and species/binomial/genus aggregates. The available page representation did not make the exact data license explicit during this review; redistribution and data ingestion remain blocked pending artifact-specific terms review.
- Wood engineering properties are conditioned by species/population, anatomy, grain direction, moisture, defects, specimen/lot and process history. A species mean is a prior for search or candidate ranking, not an individual-lot result or structural design allowable.

These sources motivate the contract; no third-party property records are bundled and no physical property prediction is claimed.

## Paradigm shift

Do not map a species name to a single vector of supposedly universal properties. Represent the path from taxonomic identity and source evidence to condition-specific properties, then to the actual sample/lot and intended application.

An aggregate species profile may support candidate retrieval or preliminary ranking. A lot-selection task requires suitably scoped lot/specimen evidence. Structural design requires an independently qualified value applicable to the exact material state, orientation and use. The profile validator establishes only these software distinctions, not a physical property value.

## v0.1 schema

- Taxon identity distinguishes accepted scientific name, resolution state and identification method. Common names are search aliases, not a substitute for resolved identity.
- Source record, source kind, and external-source license reference are explicit.
- Profile scope is specimen, lot, species aggregate or genus aggregate.
- Each property records its source reference, unit, conditioning basis, direction, evidence level and optional measurement record ID.
- Relevant directional classes include longitudinal, radial and tangential; do not treat orthotropic properties as interchangeable.
- Moisture/conditioning basis is explicit: green, air-dry, oven-dry, conditioned, or not applicable.
- Evidence level distinguishes sample/lot measured, literature reported, species/genus estimate, qualified design value, assumption and unknown.
- Candidate ranking accepts appropriately identified, source-bound estimates without promoting them to engineering qualification.
- Lot selection requires specimen/lot scope and measured property evidence.
- Structural use requires resolved taxon identity, appropriate material scope and a qualified design value.
- Unit, conditioning-basis and direction mismatches block strict comparison. v0.1 does not perform property transfer or unit conversion; such a future path must bind conversion evidence and normalized values explicitly.
- A measured property requires its measurement-record reference.
- External source profiles need a reviewed data-use/license reference; an absent license is not treated as unrestricted.
- Unknown JSON fields are rejected, scenario IDs are unique, and receipts bind exact corpus bytes and source commit.

## Frozen known-answer corpus

1. A resolved species aggregate used for candidate ranking only.
2. Ambiguous common-name taxon and literature-reported modulus used as structural evidence.
3. Species-average density presented as lot-specific property.
4. Radial modulus incorrectly used for a longitudinal requirement.
5. Green-state density used as oven-dry density.
6. Public-reference profile without a resolved data-license reference.
7. Sample-measured property without an exact measurement record.
8. Property value compared in MPa when the requirement is stated in GPa.

Every negative fixture must return its precise reason codes. An intentionally rejected control counts as correctly handled only when the expected disposition and reason codes match.

## Metrics and future benchmarks

### Contract correctness
- 100% deterministic replay for exact corpus bytes and source revision.
- 100% of frozen identity, scope, unit, conditioning and evidence negative controls rejected or explicitly indeterminate as specified.
- Zero silent substitution between common-name alias and taxonomic identity, species aggregate and lot result, or radial/tangential/longitudinal property.
- All external datasets remain excluded until exact artifact license and attribution are reviewed.

### Scientific performance, after lawful datasets are admitted
- Taxon identification: top-k accuracy and calibrated abstention on an independent specimen- and geography-held-out dataset.
- Density/property prediction: report MAE/RMSE, interval coverage and calibration per property; split by source/population to avoid sample leakage.
- Process relevance: evaluate drying time, final moisture distribution, defects, dimensional stability and energy per conforming dry volume only against comparable measured batches and kiln profiles.
- Product selection: report constraint violations, qualified output, waste and processing time independently. A single score may not trade a structural evidence failure against speed or cost.

Do not select predictive targets from training-set performance. Freeze source populations, holdout splits and metric thresholds before evaluation.

## Next implementation ladder

1. This standalone synthetic contract and verifier.
2. A metadata-only source registry for the USDA Wood Handbook, GWDD v2.2 and later regional species/property sources, including artifact/license/attribution status.
3. A license-reviewed data adapter for only the subset whose terms are confirmed.
4. Cross-reference species estimates into wood candidate ranking, explicitly retaining uncertainty and source level.
5. Link qualified lot/sample property references into the existing WOOD-FLOW-001 kiln scheduler and WOOD-LIFE-001 lifecycle benchmark.
6. Add physical-data evaluation only after instrument/method comparability and sample/lot lineage are established.

## Cross-stack ownership

- Symthaea: evidence-bounded taxonomy/property retrieval, uncertainty and measurement requests.
- Symtropy: deterministic sawing/drying/manufacturing scenarios that reference exact profile versions.
- Mycelix: lot identity, custody, transformation events and corrections; custody does not create engineering property truth.
- Sol-Atlas: geographically and temporally scoped forest/species observations with spatial coverage and source limitations.
- Independent validator: identity/scope applicability, source/license and measurement-reference checks.

Reuse ENG-MAT, MAT-PSPP, WOOD-LIFE and existing provenance/measurement owners. Do not create a second general material ontology or evidence ledger.

## Claim ceiling

A passing synthetic receipt establishes software semantics only. It does not identify a real tree, certify lumber, establish universal species properties, provide a structural design allowable, prescribe a kiln schedule, prove sustainability/legal sourcing or qualify a physical product.

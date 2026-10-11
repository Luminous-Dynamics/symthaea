# PIE Phase-0 spatial resource evidence source registry

This document identifies **source families and provenance requirements** for future Moon/Mars site-resource adapters. It is not a reserve estimate, process-yield database, or recommendation of a landing/mining site.

## Core rule

Remote sensing and derived maps create evidence about **occurrence, likelihood, environment, geometry, or composition**. They do not automatically establish recoverable feedstock, demonstrated extraction yield, economic reserve, or available plant inventory.

Every imported spatial-resource record should preserve at least:

- celestial body;
- stable source/dataset identifier;
- dataset version and, where relevant, release/errata identifier;
- instrument / mission / observing system;
- observation or source time range;
- processing level / derived-product status;
- coordinate reference frame, projection and datum/reference radius;
- spatial resolution / footprint;
- valid-data and missing-data masks;
- uncertainty / consistency / confidence fields when supplied;
- source citation / DOI / archive URI;
- supersedes / superseded-by relationship;
- ingestion timestamp and adapter version;
- distinction between measured observation, derived model, interpretation, and future projection.

A site-resource adapter must never silently replace an earlier dataset in an existing evidence lineage.

## Moon source families

### LRO LOLA topography

Primary archive family: `LRO-L-LOLA-4-GDR-V1.0` (LOLA Gridded Data Records).

Use cases:
- surface elevation;
- slope / roughness where provided;
- route geometry / site access;
- terrain constraints for excavation and logistics;
- reference geometry for colocating other resource products.

Important provenance note: LOLA GDRs have multiple releases and revised products. Record the exact release/errata state, not only the dataset ID.

High-resolution south-pole site DEM products from NASA PGDA may be used where appropriate, but their interpolation fraction, geodetic-control description, and uncertainty must remain visible.

### LRO Diviner polar resource products

Preferred archive family: `LRO-L-DLRE-5-PRP-V2.0`.

Use cases:
- annual-average / maximum polar surface temperature;
- modeled depth to water-ice permafrost;
- volatile-retention / thermal-environment evidence.

Critical provenance rule: PDS explicitly identifies the earlier V1.0 product as erroneous and directs users to V2.0. PIE adapters must therefore support explicit superseding-evidence relationships and reject silent V1/V2 substitution.

### LRO LEND / hydrogen evidence

LEND neutron measurements and derived LRO products may support hydrogen/volatile occurrence hypotheses.

Guardrail: hydrogen evidence is not automatically a quantified recoverable-water reserve. Preserve the observation/model distinction and combine with other evidence only through explicit claims.

### LRO Mini-RF radar

Archive family: `LRO-L-MRFLRO-5-CDR-MAP-V1.0` and associated calibrated/map-projected products.

Use cases:
- radar/scattering context;
- candidate volatile/roughness interpretation;
- site characterization.

Guardrail: radar signatures remain interpretation evidence, not direct recoverable-ice inventory.

### Future / in-situ lunar volatile measurements

NASA's Neutron Spectrometer System contribution to the LUPEX rover is intended to characterize subsurface ice near the lunar south pole. When surface/in-situ products become available, PIE should preserve them as a distinct evidence class rather than overwriting orbital evidence.

In-situ observations may materially update local resource confidence, but recovery fraction still belongs to an acquisition/process model.

## Mars source families

### SWIM — Subsurface Water Ice Mapping

NASA/JPL-supported SWIM integrates multiple orbital evidence sources to map **consistency with accessible subsurface water ice** in the Martian mid-latitudes.

Use cases:
- candidate water-resource regions;
- comparative site screening;
- depth/accessibility hypotheses where supplied;
- landing-site / industrial-site trade inputs.

Guardrail: SWIM is a multi-dataset consistency / likelihood product, not a guaranteed recoverable reserve. Preserve source-layer consistency and uncertainty rather than converting map color directly into tonnes of water.

### MGS MOLA topography

Archive families include the MOLA Mission Experiment Gridded Data Record (`MGS-M-MOLA-5-MEGDR-L3-V1.0`) and the migrated PDS4 MOLA derived-topography bundle.

Use cases:
- elevation / terrain context;
- route and site geometry;
- colocating SWIM, CRISM and other resource/environment products.

Record exact PDS3/PDS4 lineage, resolution and projection. The 2026 migrated PDS4 bundle exposes multiple map resolutions and polar products; migration does not mean old and new resource lineages should be silently mixed.

### MRO CRISM surface mineral evidence

The CRISM MICA/type-spectra archive family provides representative spectral signatures for mineral/ice phases identified on Mars.

Use cases:
- mineral-phase evidence;
- candidate feedstock/mineralogical context;
- process-route screening.

Guardrail: spectral identification at a location is not automatically bulk composition, ore grade, recoverable concentration, or plant feedstock quality.

### MRO CRISM atmospheric retrievals

The CRISM atmospheric-retrieval bundle includes multi-Martian-year derived products for water vapor, carbon monoxide, oxygen airglow, dust, water ice and related atmospheric properties.

Use cases:
- atmospheric feedstock/environment context;
- seasonal process-envelope studies;
- dust / water-ice atmospheric constraints.

Guardrail: a global/seasonal atmospheric retrieval must be sampled at a declared site/time/season before it becomes a process boundary condition.

## Required adapter outputs

A future PIE spatial adapter should emit neutral records such as:

- `SiteEvidence`;
- `SpatialDatasetRef`;
- `ResourceOccurrenceEvidence`;
- `EnvironmentalEnvelopeEvidence`;
- `CompositionObservation`;
- `OccurrenceLikelihood` or source-native uncertainty fields;
- `CoverageMask` / `MissingData`;
- `SupersedingEvidenceEdge`.

It should **not** emit `MaterialLot`, `RecoverableReserve`, or `GuaranteedFeedstock` directly.

Those require later composition, acquisition, transport, throughput and evidence gates.

## First integration sequence

1. ingest provenance and geometry only;
2. validate coordinate/frame/projection semantics;
3. preserve masks and source-native uncertainty;
4. expose source-native observations/derived fields;
5. attach interpretation claims separately;
6. combine with PIE resource-access semantics;
7. combine with LETN/site logistics;
8. only then evaluate candidate industrial chains.

## Phase-0 acceptance tests

- exact dataset/version/release round-trips through serialization;
- a superseded dataset cannot silently replace a prior evidence root;
- missing/null pixels remain missing, never zero-resource values;
- coordinate-frame/projection mismatch fails closed;
- resampling/interpolation records the transformation lineage;
- lower-resolution data cannot be presented as higher-resolution measurement;
- remote-sensing likelihood cannot be serialized as a guaranteed resource mass without an explicit inference claim;
- in-situ and orbital evidence can coexist and support/challenge each other;
- widening uncertainty cannot strengthen a conservative resource claim.

## Non-claims

This registry does not establish which lunar or Martian sites should be mined, how much material is present, how much can be recovered, or whether any extraction process is economic or safe.

Tracks #1648, #1689, #1647, #1607, #1608, and master #1604.

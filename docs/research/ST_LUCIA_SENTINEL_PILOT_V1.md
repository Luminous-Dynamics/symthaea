# St. Lucia Sentinel Pilot v1 — pre-product-discovery protocol

Status: **pre-run / pre-product-selection**

This document freezes the first real Sentinel pilot target and deterministic catalogue-selection procedure before any individual product is chosen for the experiment. It supplements the locked Planetary Perception witness in issue #194. It does not change that witness's baselines, primary success criteria, negative controls, OOD requirements, abstention/coverage rules, or replication requirements.

## Scientific target

Pilot site: **St. Lucia System / iSimangaliso Wetland Park, South Africa**.

Authoritative public references:

- Ramsar Sites Information Service, St. Lucia System (site 345): approximately 28°00′S, 32°28′E; a coastal wetland/lake/estuary system.
- UNESCO World Heritage Centre, iSimangaliso Wetland Park – Maputo National Park: South African component around S27°50′20″ E32°33′00″; a dynamic wetland/coastal system containing lakes, swamps, wetlands and estuarine environments.

Reference URLs:

- https://rsis.ramsar.org/
- https://whc.unesco.org/en/list/914/

The catalogue AOI below is only a reproducible **discovery box**. It is not a legal/ecological boundary and must not be presented as one. Exact scientific support is later frozen by deterministic raster geometry and exact-window evidence.

## Frozen catalogue discovery AOI

Coordinate reference: WGS84 longitude/latitude.

Center derived from the Ramsar St. Lucia System reference coordinate:

- longitude: `32.4666667`
- latitude: `-28.0000000`

Discovery bounding box:

```text
west   32.3166667
south -28.1500000
east   32.6166667
north -27.8500000
```

## Catalogue endpoint

Use the current Copernicus Data Space Ecosystem STAC catalogue:

```text
https://stac.dataspace.copernicus.eu/v1/
```

The legacy CDSE STAC endpoint must not be substituted silently. The exact endpoint, query and canonical response snapshot/digest must be retained with discovery evidence.

## Sentinel-2 L2A selection

Collection:

```text
sentinel-2-l2a
```

Frozen discovery interval:

```text
start inclusive: 2026-07-01T00:00:00Z
end exclusive:   2026-08-01T00:00:00Z
```

An item is eligible only when all of the following hold:

1. its footprint intersects the frozen discovery bbox;
2. it is Sentinel-2 Level-2A;
3. catalogue `eo:cloud_cover <= 20`;
4. acquisition is inside the frozen interval;
5. the required optical bands for the first deterministic baseline are available.

Selection among eligible items is deterministic:

1. earliest acquisition datetime;
2. lexicographically smallest STAC item ID as an exact-time tie-break.

Do not inspect previews and choose the visually clearest scene. If no item qualifies, record `no-eligible-item`. Any relaxation of cloud threshold or time interval requires a prospective protocol amendment before another search.

## Sentinel-1 GRD pairing

Collection:

```text
sentinel-1-grd
```

Only after the S2 item is frozen, identify S1 candidates that:

1. intersect the same discovery bbox;
2. are GRD products;
3. use IW mode where represented/available under the catalogue contract;
4. contain the required VV/VH channels for the planned baseline where available;
5. occur within ±72 hours of the frozen S2 acquisition.

Selection order:

1. smallest absolute acquisition-time separation from the selected S2 item;
2. earlier acquisition wins an exact time-distance tie;
3. lexicographically smallest STAC item ID is the final tie-break.

Orbit direction is recorded but not post-hoc selected in v1. If no candidate satisfies the frozen rules, retain `no-eligible-item` and amend prospectively rather than browsing for a convenient scene.

## Discovery evidence

Before downloading or processing raster payloads, retain and content-address:

- exact STAC endpoint;
- exact query parameters/filter expression;
- canonical full query response snapshot;
- response digest;
- retrieval timestamp;
- selection-algorithm version;
- every eligible candidate item ID in canonical order;
- selected item ID or explicit `no-eligible-item`;
- catalogue/server version metadata if exposed.

The raw catalogue snapshot is discovery evidence, not ground truth.

## Separation of concerns

This protocol freezes **which source products may enter the pilot**. It does not assign Train/Calibration/Evaluation roles and does not expose held-out outcomes.

Responsibilities remain separated:

- research split: `symthaea-research-split` / #179;
- fit influence: #184;
- candidate selection: #192;
- evaluation custody: #193;
- evaluation opening: #197;
- frozen Sentinel source/artifact identity: #202;
- exact raster geometry: #216;
- exact source/window/output evidence: #220;
- canonical payload interpretation: #237;
- geometry/bytes/interpretation join: #243.

## First feature baseline

After source bytes are actually acquired, hashed and qualified, the initial deterministic optical/SAR feature baseline should use only already-defined reviewed semantics. Candidate optical features include named NDVI, McFeeters NDWI, Gao NDWI and NBR variants with explicit mask propagation. Candidate SAR features remain calibrated backscatter semantics only until a separately validated preprocessing pipeline exists.

No feature/model result may retroactively change this product-discovery protocol.

## Literature-strength conventional ML floor

A 2026 Frontiers in Remote Sensing study on iSimangaliso Wetland Park provides a relevant external conventional baseline lineage. It used multi-season Sentinel-1/2 predictors and compared Random Forest (RF), Support Vector Machine (SVM), Classification and Regression Trees (CART), and K-Nearest Neighbours (KNN), reporting RF as the strongest of those classifiers and approximately 91% overall accuracy in its own experimental setting.

Reference:

- https://www.frontiersin.org/journals/remote-sensing/articles/10.3389/frsen.2026.1814582/full

The locked pilot therefore must not define `conventional ML` as a weak placeholder. Where the task supports classification, include at least RF and, unless prospectively ruled inapplicable, SVM, CART and KNN.

Initial candidate grids are frozen from the published study rather than invented after our results are visible:

```text
RF trees            {50, 100, 150, 200}
RF minimum leaf     {1, 5, 10}
RF bag fraction     {0.5, 0.7}
SVM RBF cost        {1, 5, 10, 20}
SVM gamma           {0.1, 0.5, 1}
CART minimum leaf   {1, 5, 10, 50, 100}
KNN k               {3, 5, 7, 9, 11, 21}
```

The paper's reported RF configuration (`100` trees, minimum leaf `1`, bag fraction `0.7`, max nodes `500`) and SVM configuration (`RBF`, cost `20`, gamma `1`) are named external reference configurations. They are **not** automatic winners for our dataset. Our selected conventional configuration must be chosen only through the frozen Calibration partition and #192 selection receipts.

For compatible classification tasks, report at minimum:

- overall accuracy;
- per-class precision / user accuracy;
- per-class recall / producer accuracy;
- per-class F1;
- confusion matrix;
- Quantity Disagreement (QD);
- Allocation Disagreement (AD);
- coverage and abstention where applicable.

Wetland boundary and transitional zones must be an explicit diagnostic slice rather than disappearing inside aggregate accuracy.

The external study describes hold-out validation and validation-driven hyperparameter optimisation. This pilot intentionally uses a stricter boundary:

```text
Training -> Calibration/model selection -> sealed Evaluation
```

Final Evaluation must never influence hyperparameter selection. Spatial, acquisition and temporal leakage controls remain governed by #179.

Symthaea/HDC receives no scientific credit merely for beating persistence, climatology, or a threshold heuristic. For a classification-style witness it must demonstrate value against the strongest qualified conventional baseline under the same frozen evidence, split and evaluation contract. A conventional-model win or null Symthaea result remains a valid outcome.

## Non-claims

This preregistration does not claim:

- that qualifying July 2026 products exist;
- that any raster bytes have been downloaded or hashed;
- that atmospheric, cloud, SAR calibration or terrain processing is already scientifically qualified;
- that Sentinel observations directly reveal arbitrary underground structure;
- that Symthaea/HDC improves wetland prediction or bandwidth efficiency.

Null discovery, missing modalities and failed preprocessing remain valid reportable outcomes.

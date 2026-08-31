# symthaea-sentinel-eo

Offline-first Sentinel-1/2 Earth-observation bridge into `symthaea-earth-observation`.

The package name intentionally includes `-eo` because the workspace already contains the unrelated core `symthaea-sentinel` audio-pattern-recognition crate. Cargo package identity is therefore unambiguous.

The bridge intentionally contains **no live network client**. Scientific and CI workflows begin with replayable metadata so product identity, timestamps, footprints, digests, and processing lineage do not depend on live credentials or an external service.

## Boundary

```text
Copernicus / local archive / test fixture
                  |
                  v
          SentinelCatalog
                  |
                  v
     SentinelProductMetadata
                  |
          +-------+-------+
          |               |
          v               v
 ObservationEvidence   FrozenSentinelFixtureManifest
          |               |
          v               v
 optical / SAR       source + derived-artifact
 feature semantics       lineage
```

A later live adapter may implement `SentinelCatalog`, but downstream code should not know whether metadata came from a frozen fixture, local archive, STAC catalogue, or another provider.

## Frozen fixture manifests

`FrozenSentinelFixtureManifest` gives an experiment a content-addressed description of the exact Sentinel inputs and materially transformed artifacts it used.

Each source product freezes:

- observation, mission, instrument, and product identity;
- Sentinel product kind and acquisition time;
- exact footprint coordinate bits;
- explicit modality and radar metadata where applicable;
- bands and exact wavelength bits;
- uncertainty fields;
- source content digest;
- ordered processing lineage;
- an independent BLAKE3 metadata digest.

Each derived artifact freezes:

- artifact id and kind;
- its own content digest and optional byte length;
- ordered source references;
- ordered processing steps and parameter digests;
- an independent BLAKE3 identity digest.

The outer fixture manifest canonicalizes product/artifact list order, rejects duplicate or missing references, rejects cycles in derived-artifact lineage, and has its own BLAKE3 digest. Deserialization revalidates nested product/artifact identities before accepting the outer manifest.

Source order and processing-step order remain identity-significant because transforms such as band stacking can be order-sensitive.

A raw Sentinel product digest must **not** be reused as the identity of a cloud-masked raster, terrain-corrected SAR product, resampled window, feature cube, preview, or other materially transformed artifact. Those receive their own content identities while retaining source lineage.

## Research-integrity separation

The Sentinel fixture layer does not own:

- Training / Calibration / Evaluation assignment;
- spatial/acquisition/time separation policy;
- fitted-model influence sets;
- model selection;
- held-out evaluation custody;
- authorization to reveal evaluation data.

Those remain separate research-integrity contracts. This separation lets one immutable Sentinel fixture universe be assigned differently by different preregistered experiments without changing the source evidence itself.

A digest is an integrity commitment, **not a secrecy mechanism**. Small-state labels or future outcomes must remain behind real custody rather than being considered hidden merely because their hashes are known.

## Subsurface safety

Raw Sentinel products map to `ObservationSensitivity::SurfaceOnly`.

That is deliberate. Sentinel-1 SAR can support powerful indirect inference—for example through derived surface-deformation products—but the existence of a radar acquisition does not by itself establish direct subsurface penetration to any depth.

Any future direct-penetration evidence must carry an acquisition-specific validated depth bound through the provider-neutral Earth-observation contract.

## Next empirical work

1. Add frozen real Sentinel-1 GRD and Sentinel-2 L2A product/source manifests.
2. Store large raster payloads outside Git while retaining exact content digests and retrieval provenance.
3. Add deterministic raster/window extraction with a separately digested output artifact.
4. Map Sentinel-2 quality/bands into deterministic optical features.
5. Map calibrated Sentinel-1 backscatter into the SAR feature module.
6. Bind the frozen universe to leakage-resistant research split/fit/selection/custody contracts.
7. Execute the locked Wetland Watch witness defined by issue #194.

Live downloads and credentials remain intentionally outside CI.

## Non-claims

The fixture contract does not establish that:

- real Sentinel raster payloads have already been acquired;
- processing algorithms are scientifically correct;
- a chosen research split is independent or adequate;
- held-out bytes are secret;
- Sentinel-1 directly images arbitrary underground structure;
- Symthaea/HDC improves Earth-observation performance.

Those remain separately gated claims.

## Validation

```bash
cargo fmt --all -- --check
cargo check -p symthaea-sentinel-eo --all-targets
cargo test -p symthaea-sentinel-eo
cargo clippy -p symthaea-sentinel-eo --all-targets -- -D warnings
```

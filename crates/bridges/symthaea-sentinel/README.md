# symthaea-sentinel

Offline-first Sentinel-1/2 bridge into `symthaea-earth-observation`.

The first version intentionally contains **no live network client**. Scientific and CI workflows begin with a frozen catalogue so product identity, timestamps, footprints, digests, and processing lineage can be replayed without credentials or dependence on an external service.

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
                  v
        ObservationEvidence
                  |
       +----------+----------+
       |                     |
       v                     v
 optical features          SAR features
```

A later live adapter may implement `SentinelCatalog`, but downstream code should not know whether metadata came from a frozen fixture, local archive, STAC catalogue, or another provider.

## Subsurface safety

Raw Sentinel products map to `ObservationSensitivity::SurfaceOnly`.

That is deliberate. Sentinel-1 SAR can support powerful indirect inference—for example through derived surface-deformation products—but the existence of a radar acquisition does not by itself establish direct subsurface penetration to any depth.

Any future direct-penetration evidence must carry an acquisition-specific validated depth bound through the provider-neutral Earth-observation contract.

## Planned next work

1. Frozen Sentinel-1 GRD and Sentinel-2 L2A metadata fixtures.
2. Content-addressed local payload references.
3. AOI intersection using the shared geodesy layer rather than ad-hoc geometry.
4. Sentinel-2 band/quality mapping into deterministic optical features.
5. Sentinel-1 calibrated backscatter mapping into the SAR module.
6. Live catalogue adapter isolated behind the same interface.
7. Reproducible Wetland Watch witness using paired optical/SAR observations.

Live downloads and credentials are intentionally not part of CI.

## Validation

```bash
cargo fmt --all -- --check
cargo check -p symthaea-sentinel --all-targets
cargo test -p symthaea-sentinel
cargo clippy -p symthaea-sentinel --all-targets -- -D warnings
```

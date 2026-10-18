# Third-party data: Tire Monitoring Dataset

## Attribution and license
- Upstream repository: https://github.com/kerzeyc/tire-monitoring-dataset
- Upstream file: `tire_monitoring_dataset.csv`
- Upstream Git blob SHA-1: `0029b6f156847ecc052f57ae45fd025ad032d086`
- License stated by upstream: Creative Commons Attribution 4.0 International (CC BY 4.0).
- The exact upstream license text is preserved in `LICENSE.txt`.
- The upstream README is preserved in `README.upstream.md`; it currently says the formal academic citation is to be updated.

This dataset copy is split into ordered text shards solely because the repository upload interface rejected the 7.8 MB monolithic file. No row was filtered, reordered or changed. Concatenate `tire_monitoring_dataset.csv.part-0001` through `tire_monitoring_dataset.csv.part-0009` in ascending numeric order without adding separators to reconstruct the source file. Shards preserve the upstream CSV line endings and leading UTF-8 BOM.

## Verification notes
- Upstream blob SHA-1: `0029b6f156847ecc052f57ae45fd025ad032d086` (this is GitHub's Git object SHA-1, not a SHA-256 digest).
- Retrieved content length before sharding: 7808266 JavaScript characters.
- Shards: 9; 210128 data rows plus one header were counted during source-side inspection.
- All nine committed shards were fetched back and their concatenation exactly matched the upstream content string (7808266 characters); no row filtering/reordering was performed.
- Source-side schema scan: 210128 rows; 1755 distinct Unit_ID segments; 196260 label-0 rows; 13868 label-1 rows; zero malformed rows/empty fields; row ranges Pressure 197.37–289.31 kPa, Temperature 1.6–45.94 °C, Speed 2.84–144.94 km/h.
- Full-file SHA-256 is not asserted by this note. The added Rust intake command will compute it on a Rust-capable checkout.
- Use `Unit_ID` as the group/split unit; never randomly mix samples from one acquisition segment across train and test.
- Vehicle scope: passenger-car 205/55 R16 pressure/temperature/speed monitoring, not motorcycle tire force, wet grip, wear, impact, or structural safety.

The dataset is CC BY 4.0 and must be attributed. It remains third-party-licensed material; it is not under the repository's primary code license.

## Reproducible Rust intake check

From the workspace root, after restoring this checkout:

```sh
cargo test -p xtask tire_monitoring_intake
cargo run -p xtask -- tire-monitoring-intake --input-dir data/third_party/tire-monitoring-dataset --out reports/vehicle_data/tire_monitoring_intake.json
```

These Rust commands have been authored but not executed in the current environment. The source-side row scan and shard comparison documented above are not substitutes for this exact-head Rust run. Do not random-split rows; partition by `Unit_ID` to keep whole acquisition segments in one split.

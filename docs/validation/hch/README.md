# HCH Bakeoff

Hierarchical Cantor Hypervectors are evaluated as an experimental topology
bakeoff, not as a Broca default.

Run the v0.6 bakeoff:

```bash
cargo run -p symthaea-core \
  --features cantor-hdc \
  --bin hch_bakeoff \
  --release \
  -- --objects 128 --seeds 3 --out /tmp/hch_v06.json
```

The runner writes both JSON and CSV reports. If `--out` is
`/tmp/hch_v06.json`, the CSV report is written to `/tmp/hch_v06.csv`.

Primary metrics:

- `top1`, `top3`: retrieval accuracy.
- `mean_margin`: distance between best and second-best codebook match.
- `abstention_rate`: fraction of queries below the configured margin threshold.
- `answered_accuracy`: top-1 accuracy over non-abstained queries.
- `load_entropy`, `max_leaf_load`, `mean_leaf_load`: routing balance.
- `oracle_gap_top1`, `oracle_gap_margin`: distance from the high-capacity
  `OracleApprox` control.

Interpretation:

- Large oracle gap: routing is likely the bottleneck.
- Low oracle result: bundling/retrieval is likely the bottleneck.
- Strong answered accuracy with high abstention: topology may be useful as a
  confidence/calibration mechanism even before raw top-1 improves.

Broca integration should wait until a router wins on top-3, margin,
answered accuracy, or a Broca-specific semantic-role task.

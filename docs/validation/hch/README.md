# RHN Bakeoff

Resonant Hypergraph Network (RHN) is the broader topology research line.
Hierarchical Cantor Hypervectors (HCH) are the initial Cantor-backed topology
inside RHN, not the whole architecture.

RHN is evaluated as an experimental topology bakeoff, not as a Broca default.

Current status:

- RHN v0.7 adds same-capacity and high-capacity oracle diagnostics.
- RHN v0.8 adds prototype adaptive splitting and multi-leaf storage/retrieval.
- RHN v0.9 adds explicit split/retrieval semantics: stable `NodePath`
  identities, `SplitMigrationPolicy`, `RetrievalPolicy`, and split/retrieval
  reports for capacity/fanout accounting.
- These APIs are ready for controlled Broca, Vision, and coding-router
  experiments behind feature flags, not default integration.

Run the current bakeoff:

```bash
cargo run -p symthaea-core \
  --features cantor-hdc \
  --bin hch_bakeoff \
  --release \
  -- --objects 128 --seeds 3 --out /tmp/rhn_v07.json
```

The runner writes both JSON and CSV reports. If `--out` is
`/tmp/rhn_v07.json`, the CSV report is written to `/tmp/rhn_v07.csv`.

Primary metrics:

- `top1`, `top3`: retrieval accuracy.
- `mean_margin`: distance between best and second-best codebook match.
- `abstention_rate`: fraction of queries below the configured margin threshold.
- `answered_accuracy`: top-1 accuracy over non-abstained queries.
- `load_entropy`, `max_leaf_load`, `mean_leaf_load`: routing balance.
- `oracle_gap_top1`, `oracle_gap_margin`: distance from the
  `OracleHighCapacity` control.

Interpretation:

- Large high-capacity oracle gap: routing or fixed-capacity assignment is likely
  the bottleneck.
- `OracleSameCapacity` near the best practical router with `OracleHighCapacity`
  much higher: fixed leaf capacity/load is likely the bottleneck.
- Low high-capacity oracle result: bundling/retrieval is likely the bottleneck.
- Strong answered accuracy with high abstention: topology may be useful as a
  confidence/calibration mechanism even before raw top-1 improves.

Broca integration should wait until a router wins on top-3, margin,
answered accuracy, or a Broca-specific semantic-role task.

Adaptive splitting caveat: `split_at_node` currently increases local resolution
for new writes, but it does not rebalance existing bundled state. Treat it as a
research control surface until migration/rebalancing invariants are added.

Split semantics:

- Children are subrange views into the parent range, not independent storage.
- The parent remains a summary/fallback attractor after splitting.
- Children specialize future writes and finer retrieval.
- `SplitMigrationPolicy::ViewSubrangeSummary` records this current behavior; it
  does not copy historical bundled state.
- `RetrievalPolicy::ChildFirstParentFallback` searches children first and then
  falls back to the parent summary when child-local evidence is insufficient or
  dimensionally incompatible.

Before default Broca/Vision integration, RHN still needs domain-transfer
evidence and storage/fanout accounting from the bakeoff runner.

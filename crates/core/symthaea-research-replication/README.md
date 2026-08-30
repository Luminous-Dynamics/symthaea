# symthaea-research-replication

Evidence contracts that distinguish **reproduction**, **reanalysis**, **direct replication**, and **conceptual replication** without inflating one into another.

## Why this exists

A second run is not automatically independent evidence.

Examples:

- same code + same data + same environment + same seeds tests exact reproducibility;
- new analysis code + same data is reanalysis;
- same frozen protocol + genuinely new data can be a direct replication;
- a changed protocol/population/model may be a conceptual replication.

Those are all useful. They answer different questions and should not be collapsed into `replicated = true`.

## Factual lineage comparison

`FactualLineageComparison` is computed directly from two immutable `ResearchResultManifest`s and records whether these are the same or different:

- frozen protocol digest;
- source commit;
- dataset manifest;
- reproducibility capsule;
- seed manifest.

These relations are facts available from the result lineage. They require no subjective independence label.

## Replication designs

### `ExactReproduction`

Requires the same:

- protocol;
- source commit;
- dataset manifest;
- reproducibility capsule;
- seed manifest.

It asks whether the exact registered lineage can be replayed reproducibly. It is **not** described as independent replication.

### `DirectReplication`

Requires:

- the same frozen protocol digest;
- a different dataset manifest digest.

The implementation may remain the same or change. That dimension is retained explicitly in the factual lineage instead of being hidden.

A same-data rerun is rejected as a direct replication because it is reproduction/reanalysis evidence, not new empirical evidence.

### `Reanalysis`

Requires the same dataset manifest. Source code, environment, or seeds may differ and remain visible.

### `ConceptualReplication`

May deliberately change protocol, data, population, implementation, or environment. Its comparison method and evidence artifact must state how the related results are being compared.

## Independence is not inferred from IDs

Human/process/institutional independence cannot be proven by different UUIDs, Git commits, or dataset hashes alone.

`IndependenceEvidence` therefore records an explicit dimension, statement, and evidence digest for claims such as:

- independent data acquisition;
- independent implementation;
- independent analyst;
- independent institution;
- independent measurement system;
- independent validation team.

The record makes the claim attributable and auditable. It does **not** make the claim true by construction.

## Replication outcomes

`ReplicationOutcome` remains explicit:

- `Concordant`;
- `Discordant`;
- `Mixed`;
- `Inconclusive`;
- `NotComparable`.

The comparison method and its artifact digest are retained alongside the classification. There is no universal replication score or automatic truth promotion.

## Intended use

Initial consumers can include:

- Wetland Watch / semantic-downlink experiments;
- synthetic hidden-world subsurface inference;
- Futures Laboratory physical-world validation;
- consciousness/recurrence evidence programs;
- Symtropy/Symthaea controlled experiments.

A useful future evidence ladder is:

```text
frozen protocol
    -> exact registered run
    -> immutable result manifest
    -> exact reproduction
    -> direct replication on new data
    -> independent implementation / institution where justified by evidence
    -> conceptual replication across regimes
```

Each step adds a different kind of evidence. None silently rewrites the claims made by the earlier step.

## Non-claims

This crate does not establish that:

- concordance proves a theory true;
- discordance proves a theory false without further analysis;
- different datasets are statistically independent merely because their hashes differ;
- different commits imply independent implementations;
- different institutions are independent without external evidence;
- a replicated scientific claim authorizes an intervention.

## Required gates

```bash
cargo fmt --all -- --check
cargo check -p symthaea-research-replication --all-targets
cargo test -p symthaea-research-replication
cargo clippy -p symthaea-research-replication --all-targets -- -D warnings
```

# VART-WORLD-CREATIVE-001 — Persistent World Identity v1

Status: confirmatory qualification and analysis contract. It does not authorize a scientific claim by itself.

## Why explicit identity is required

`fixture + seed + policy` is not a persistent-world identity. Distinct worlds can share those values, and treating them as one cluster would create false longitudinal continuity and invalid resampling units.

VART therefore freezes two separate identities before confirmatory execution:

- `world_cluster_sha256`: the experimental world-replicate identity shared by policy clones that belong to one paired comparison cluster;
- `world_lineage_sha256`: the identity of one persistent branch across its successive revisions.

These identities describe experimental lineage, not world quality.

## World cluster

A `world_cluster_sha256` identifies the common experimental replicate from which paired policy branches are instantiated.

Within a paired block, FULL, RANDOM_VALID, HEURISTIC, and any preregistered matched ablations that are intended to be paired must share the same cluster identity.

Cluster identity is the resampling/independence unit for paired cross-policy population inference. A bootstrap must resample whole clusters, keeping policy branches from the same cluster together.

A cluster is not reconstructed from fixture or seed after the fact. It is prospectively assigned and frozen.

## World lineage

A `world_lineage_sha256` identifies one persistent branch through revision index 0, 1, 2, ... .

One lineage has exactly one:

- policy;
- world cluster;
- fixture identity;
- seed;
- campaign claim family.

Adjacent complete revisions in a lineage must satisfy both exact state-digest continuity and world-version continuity.

Two different policies cannot share one lineage identity even when they start from the same pre-state snapshot. They are isolated branches of the same cluster.

## Frozen trial inventory

The externally anchored `trial_inventory.json` contains maps covering exactly every preregistered `trial_id`:

- `world_clusters[trial_id] = world_cluster_sha256`
- `world_lineages[trial_id] = world_lineage_sha256`

The trial manifest repeats both values. A runtime cannot choose them after observing an outcome.

The freeze binds the raw-byte SHA-256 of the entire trial inventory, so changing either map after freeze creates a new lineage.

## Relation to state equivalence

World identity and state identity are complementary:

- `world_cluster_sha256` says which paired experimental replicate a trial belongs to;
- `world_lineage_sha256` says which persistent branch it continues;
- `world_state_before_sha256` / `world_state_after_sha256` prove the exact states at each revision boundary.

A matching lineage label cannot repair a broken state chain, and matching state bytes do not make two policy branches the same lineage.

## Analysis rules

For longitudinal inference:

- revisions are repeated observations inside a lineage, not independent replicates;
- policy-level population summaries operate over independent world clusters/lineages as defined by the preregistration;
- paired policy contrasts are computed within cluster before population aggregation where the analysis contract specifies pairing;
- cluster bootstrap resamples `world_cluster_sha256`, never individual revisions;
- no fallback to `(fixture, seed)` or `(policy, fixture, seed)` is permitted when identity fields are missing.

## Required rejection classes

- `WORLD_IDENTITY_INVENTORY_MISMATCH`
- `WORLD_CLUSTER_PAIRING_MISMATCH`
- `WORLD_LINEAGE_POLICY_MISMATCH`
- `WORLD_LINEAGE_CLUSTER_MISMATCH`
- `WORLD_LINEAGE_FIXTURE_MISMATCH`
- `WORLD_LINEAGE_SEED_MISMATCH`
- `WORLD_LINEAGE_DUPLICATE_REVISION`
- `WORLD_LINEAGE_STATE_CHAIN_MISMATCH`
- `WORLD_LINEAGE_VERSION_CHAIN_MISMATCH`

Passing this gate establishes explicit experimental identity and continuity only. It does not establish efficacy, calibration improvement, generalization, creativity, or intelligence.

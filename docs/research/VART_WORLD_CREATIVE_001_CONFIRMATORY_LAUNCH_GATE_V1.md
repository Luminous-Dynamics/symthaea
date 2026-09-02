# VART-WORLD-CREATIVE-001 Confirmatory Launch Gate v1

## Purpose

This gate exists between a prospective freeze and the first confirmatory execution. It verifies that the frozen trial inventory can actually identify the preregistered hypotheses before any outcome is generated.

It is not an analysis gate and does not inspect outcome values.

## Critical H2 requirement

`H2_CalibratedLineageErrorDecline` is a within-lineage slope hypothesis. A VART `TrialManifest` represents one revision (`revision_index`) with one before-state, one after-state, and one outcome. Therefore one trial per policy/world lineage cannot identify a longitudinal slope.

## Critical H3 independent-cluster requirement

`H3_PreregisteredMemoryTrapProvenanceEffect` is a paired confirmatory comparison. With only four independent MemoryTrap clusters, an exact one-sided sign-flip/randomization test has only `2^4 = 16` assignments and therefore a minimum attainable p-value of `1/16 = 0.0625`, which cannot cross a conventional alpha of 0.05. The launch gate therefore requires eight independent MemoryTrap clusters. This also gives enough randomization resolution to support multiplicity handling across the two H3 directional endpoints without resorting to a fragile n=4 parametric approximation.

For Freeze v3 the recommended design is:

- H1 / 001A core: 8 independent world clusters at revision 0, each cloned to `full_symthaea`, `random_valid`, and `heuristic` (24 trials).
- H2 / 001A longitudinal continuation: continue the same 8 `full_symthaea` lineages through revisions 1, 2, and 3 (24 additional trials). Together with revision 0 this gives 4 prospective error observations per FULL lineage.
- H3 / 001B MemoryTrap: 8 independent world clusters at revision 0, each cloned to `full_symthaea` and `no_reality_ledger_context` (16 trials).
- Total: 64 revision-trials.

The H2 continuation must preserve the exact `world_lineage_sha256` within each FULL branch and chain each previous post-state into the next pre-state. Later state continuity is verified from evidence; the launch gate verifies only that the prospective inventory contains enough ordered revision points to make the hypothesis identifiable.

## Launch requirements

Before execution the gate must establish:

1. The raw freeze bytes hash to the externally recorded expected SHA-256.
2. The inventory bytes hash to the digest frozen in the freeze.
3. H1 has 8 independent paired clusters, each containing exactly FULL, RANDOM_VALID, and HEURISTIC at revision 0.
4. H2 has at least 8 FULL world lineages with at least 4 contiguous revision indices beginning at 0.
5. H3 has at least 8 independent MemoryTrap clusters, each containing exactly FULL and NO_LEDGER at revision 0.
6. A world cluster used by 001B is not reused as an independent 001A cluster.
7. Trial IDs are unique and every inventory row has explicit cluster, lineage, policy, revision, fixture, and subcampaign identity.
8. The freeze continues to forbid aggregate world-quality scoring and outcome peeking before campaign seal.
9. The frozen H2 analysis defines the per-lineage slope estimator and cluster-level inferential procedure prospectively; the recommended primary procedure is one slope per FULL lineage followed by an exact sign-flip/randomization test across the eight independent lineage/cluster units, with a cluster bootstrap interval reported secondarily.
10. The frozen H3 analysis defines the two directional endpoints and multiplicity rule prospectively rather than collapsing them into an aggregate utility score.

## Freeze lineage rule

If an already anchored freeze lacks sufficient longitudinal revisions for H2 or sufficient independent MemoryTrap clusters for H3, do not reinterpret `trial`, weaken alpha, change to a convenient parametric test, or append trials after launch. Supersede the freeze prospectively, record the old freeze SHA-256 as `superseded_freeze_sha256`, freeze the expanded inventory and analysis contract, and create a new external anchor before the first confirmatory trial.

A launch-gate PASS authorizes only the operational start of the already-frozen campaign. It does not authorize a scientific claim.

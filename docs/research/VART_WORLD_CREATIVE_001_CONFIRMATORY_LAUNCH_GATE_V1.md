# VART-WORLD-CREATIVE-001 Confirmatory Launch Gate v1

## Purpose

This gate exists between a prospective freeze and the first confirmatory execution. It verifies that the frozen trial inventory can actually identify the preregistered hypotheses before any outcome is generated.

It is not an analysis gate and does not inspect outcome values.

## Critical H2 requirement

`H2_CalibratedLineageErrorDecline` is a within-lineage slope hypothesis. A VART `TrialManifest` represents one revision (`revision_index`) with one before-state, one after-state, and one outcome. Therefore one trial per policy/world lineage cannot identify a longitudinal slope.

For Freeze v3 the recommended minimum design is:

- H1 / 001A core: 8 independent world clusters at revision 0, each cloned to `full_symthaea`, `random_valid`, and `heuristic` (24 trials).
- H2 / 001A longitudinal continuation: continue the same 8 `full_symthaea` lineages through revisions 1, 2, and 3 (24 additional trials). Together with revision 0 this gives 4 prospective error observations per FULL lineage.
- H3 / 001B MemoryTrap: 4 independent world clusters at revision 0, each cloned to `full_symthaea` and `no_reality_ledger_context` (8 trials).
- Total: 56 revision-trials.

The H2 continuation must preserve the exact `world_lineage_sha256` within each FULL branch and chain each previous post-state into the next pre-state. Later state continuity is verified from evidence; the launch gate verifies only that the prospective inventory contains enough ordered revision points to make the hypothesis identifiable.

## Launch requirements

Before execution the gate must establish:

1. The raw freeze bytes hash to the externally recorded expected SHA-256.
2. The inventory bytes hash to the digest frozen in the freeze.
3. H1 has 8 independent paired clusters, each containing exactly FULL, RANDOM_VALID, and HEURISTIC at revision 0.
4. H2 has at least 8 FULL world lineages with at least 4 contiguous revision indices beginning at 0.
5. H3 has 4 independent MemoryTrap clusters, each containing FULL and NO_LEDGER at revision 0.
6. A world cluster used by 001B is not reused as an independent 001A cluster.
7. Trial IDs are unique and every inventory row has explicit cluster, lineage, policy, revision, fixture, and subcampaign identity.
8. The freeze continues to forbid aggregate world-quality scoring and outcome peeking before campaign seal.

## Freeze lineage rule

If an already anchored freeze lacks sufficient longitudinal revisions for H2, do not reinterpret `trial` to mean a multi-revision sequence and do not append trials after launch. Supersede the freeze prospectively, record the old freeze SHA-256 as `superseded_freeze_sha256`, freeze the expanded inventory, and create a new external anchor before the first confirmatory trial.

A launch-gate PASS authorizes only the operational start of the already-frozen campaign. It does not authorize a scientific claim.

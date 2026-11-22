# Mycelial Contact-Tolerance Filtration

**Date:** 2026-11-22  
**Status:** First candidate-only graph filtration committed; exact-head validation pending.  
**Source:** `crates/domains/symthaea-canvas/src/mycelial_contact_persistence.rs`

## 1. Question

The junction–hypha adapter constructs an accepted topology from parent links and explicit, accepted anastomosis records. But the candidate detector proposes possible contacts at a measured geometric separation. A candidate edge close to the acceptance threshold may create a loop that disappears if the measurement or tolerance changes slightly.

The new filtration asks:

> At what geometric contact tolerance do candidate links first merge components or create independent cycles, and what prior path does each cycle-closing candidate complete?

This is a sensitivity analysis over candidate contacts. It does not accept them.

## 2. Mathematical object

Let (G_0=(V,E_0)) be the baseline filament graph. Its vertices are source filaments; its edges are explicit parent–child branch relations. Let each candidate contact (e_i=(u_i,v_i)) have a measured source-curve separation (w_igeq 0).

For a tolerance (epsilon), define:

[
G_epsilon = (V, E_0 cup {e_i : w_i leq epsilon}).
]

The filtration is monotone in (epsilon): candidate edges enter in ascending measured distance and are never removed within this sweep.

For each distinct threshold we record:

- number of candidate links admitted;
- connected components (eta_0);
- cycle rank (eta_1=E-V+C);
- component merges at that threshold;
- independent cycle births at that threshold.

A candidate whose endpoints are disconnected merges two components and contributes a (eta_0) death. A candidate whose endpoints are already connected creates a new independent graph cycle and contributes a (eta_1) birth. Its witness is the path already present before the candidate edge plus the new closing edge.

Because this is a graph 1-skeleton and the filtration adds no 2-simplices, these (H_1) classes do not die within the model. Cycle births are explicitly **right-censored at the configured maximum tolerance**. They are not reported as finite-lifetime (H_1) bars.

This is a graph filtration, not zigzag persistence and not yet a general simplicial-complex persistence implementation.

## 3. Candidate-only and source-integrity boundary

The function `build_mycelial_contact_filtration` accepts `MycelialAnastomosisCandidate` records. It returns a profile, not a `MycelialNetworkGraph`.

The builder:

1. validates the baseline morphology through the junction–hypha adapter;
2. validates candidate IDs and references;
3. recomputes source and target points from the supplied cubic curves;
4. recomputes contact separation and compares it with the supplied candidate distance;
5. checks that the declared junction position is the midpoint of the measured contact pair;
6. reuses the existing network candidate contact constraints;
7. inserts candidates only into a temporary in-memory filtration graph.

It never calls a fusion-acceptance operation, changes a source curve, alters semantic CognitiveTopology, or turns candidate evidence into accepted topology.

The candidate set is a **union-of-proposals upper-bound graph**, not a set of links guaranteed to be jointly executable. Different candidates can claim the same filament/curve-parameter contact (for example, one terminal tip can be close to two different target hyphae). Under the current pairwise-fusion contract, these are competing hypotheses; multiway fusion and contact-capacity resolution are not implemented. The profile therefore includes `contact_conflict_groups`, listing each shared source-curve contact and its candidate IDs, plus the number of conflict groups and distinct candidates participating in them. Cycle births in the union graph are possible cycles in the candidate supergraph, not proof that all conflicting candidates can co-exist in one accepted morphology.

The output contains the complete candidate observation inventory, including candidates above the sweep maximum. It includes the source-filament identity list and explicit baseline parent edges. Each cycle witness can be checked against those baseline edges and previously admitted candidate observations. Canonical digests bind those inputs, conflict groups, threshold snapshots, component-merge events and cycle-birth witnesses.

## 4. Deterministic computation

Candidates are ordered by:

1. recomputed source-curve separation;
2. source filament ID;
3. target filament ID;
4. fusion kind;
5. target curve parameter;
6. candidate ID.

Contact thresholds are quantized to one microunit for deterministic grouping and serialization. Within each threshold group, deterministic tie-breaking assigns each edge either a component merge or a fundamental cycle birth.

The disjoint-set implementation tracks component counts. When a candidate closes a cycle, a deterministic breadth-first search extracts a witness path from the current adjacency graph. Parent branch edges have no candidate ID; earlier contact links carry their candidate ID.

The profile validator checks:
- sorted, unique source-filament and candidate identities;
- complete, deterministic conflict groups for candidates that reuse a curve contact;
- contact parameters and candidate endpoint references;
- candidate distance/admission consistency;
- one topological event for every admitted candidate;
- event identity and contact-distance agreement with the inventory;
- continuity and edge provenance of each cycle witness;
- monotonic component/cycle snapshots;
- snapshot totals against individual event totals;
- right-censoring against the declared maximum tolerance;
- final snapshot and final metric agreement.

## 5. Important interpretation limits

- The distance is measured in **organism-local model units**, not millimetres or biological micrometres.
- The filtration threshold is a geometric contact tolerance, not a calibrated probability of fusion.
- A low-distance candidate is not automatically a true biological anastomosis.
- A cycle birth is a graph-topology fact under this candidate filtration, not proof of functional redundancy.
- A cycle's witness path may contain only static parent/candidate edges. It is not a time-respecting path.
- Candidate contact recurrence across observation windows is not evaluated by this module.
- No physical flow, conductance or transport rate is computed.
- Candidate contact positions are derived from the deterministic source curves; there is no independent imaging uncertainty model yet.

## 6. Benchmarks and proposed targets

All targets are proposed, not yet achieved.

### Candidate correctness

| Metric | Proposed gate |
|---|---:|
| Candidate distances recomputed from source geometry | 100% |
| Candidate-distance mismatch silently accepted | 0 |
| Candidate fusion silently accepted into topology | 0 |
| Candidate IDs duplicated in a profile | 0 |
| Every admitted candidate assigned exactly one merge or cycle event | 100% |
| Cycle births with a disconnected or unverifiable witness path | 0 |
| Digest changes after candidate inventory or witness mutation | 100% of tested mutations |
| Deterministic receipt over frozen candidates | 100% |

### Metamorphic / adversarial tests

- Reorder candidate input; the canonical filtration digest should remain identical.
- Duplicate a candidate ID; reject.
- Duplicate a candidate contact at the same source/target curve parameters; reject.
- Modify the supplied contact distance while leaving geometry fixed; reject.
- Modify the declared junction midpoint; reject.
- Increase maximum tolerance; admitted counts and cycle rank must be non-decreasing, component count non-increasing.
- Remove a previously admitted candidate; the corresponding merge/cycle event must disappear and the profile digest must change.
- Remove a whole source curve from a serialized base graph; the junction–hypha source-coverage validator must reject.
- Compare component/cycle counts against an independently implemented reference for small candidate graphs.

### Sensitivity study

Run the same frozen candidate inventory across a predeclared tolerance grid. Report the full (eta_0(epsilon)) and (eta_1(epsilon)) step functions and every event, not a single score. Compare:
- raw contact distance;
- calibrated geometric uncertainty, once available;
- independent measurement perturbations / bootstrap realizations, once observed data exist.

A candidate whose cycle appears only near the maximum tolerated mismatch should not be presented as robust in the same way as a cycle formed at very low separation.

## 7. Relationship to temporal persistence and resilience

This filtration handles one axis: **geometric contact tolerance**.

Two other axes are distinct:

- **Temporal recurrence:** whether the same cycle and its constituent links exist across observed time windows.
- **Functional resilience:** whether removal or disruption of a link, node or region causes less loss of time-respecting reachability / efficiency.

The 22 March 2026 *Scientific Reports* paper by Li, Abinova and Li tracks recurring cycles across sliding windows, constructs a temporal hypernetwork and compares Temporal Cycle Number / Temporal Cycle Ratio against static and temporal baselines under controlled node removal. Its results are conditional on its datasets, windowing and disruption protocol, not universal proof that cycle participation predicts resilience.

- https://doi.org/10.1038/s41598-026-44835-4
- https://pmc.ncbi.nlm.nih.gov/articles/PMC13149579/

The October 2025 *ZigzagNetVis* work uses zigzag persistence to identify temporal resolutions with substantial graph-topology changes, relevant when observed graph sequences contain both additions and deletions.

- https://doi.org/10.1109/TVCG.2025.3528197
- https://pubmed.ncbi.nlm.nih.gov/40031227/

Temporal recurrence and zigzag analysis are tracked separately in [#13180](https://github.com/Luminous-Dynamics/symthaea/issues/13180). The current filtration must not be described as those methods.

## 8. Next experiments

1. Extend the CLI benchmark to emit the candidate filtration digest and all component/cycle event counts.
2. Add an independent small-graph union-find and simple-cycle oracle to verify (eta_0), (eta_1) and witnesses.
3. Compare tolerance profiles under frozen, perturbed source curves.
4. Introduce an explicit calibration contract for converting image-local coordinates into physical units.
5. Once longitudinal observations exist, implement recurrence-based TCN/TCR and a separate zigzag-persistence track.
6. Test whether contact-persistence metrics predict disruption impact beyond degree, betweenness, density, static cycle rank and temporal efficiency.

The intended contribution is a reproducible, fail-closed way to expose how much a network's apparent connectivity depends on candidate-contact assumptions. It is a foundation for temporal and functional testing, not a replacement for either.

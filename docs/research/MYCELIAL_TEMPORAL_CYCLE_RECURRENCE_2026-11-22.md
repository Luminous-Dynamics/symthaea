# Temporal Mycelial Cycle Recurrence

**Date:** 2026-11-22  
**Status:** First bounded observed-window recurrence implementation committed; exact-head validation pending.  
**Code:** `crates/domains/symthaea-canvas/src/mycelial_temporal_cycles.rs`

## 1. Three different persistence questions

The Mycelial programme now separates three mathematical questions:

1. **Contact-tolerance filtration:** at which geometric tolerance does a candidate contact merge components or add an independent cycle? This is a monotone graph filtration over proposals. It is not a temporal observation.
2. **Temporal cycle recurrence:** does the same identity-stable cycle co-exist in multiple evidence-backed observation windows? This is the new module in this document.
3. **Zigzag persistence:** how do graph topological features persist through sequences with both additions and deletions, under a defined zigzag filtration? This is a separate future method and is not implemented here.

Functional resilience is a fourth, experimental question: does disruption of an observed node/edge/region reduce time-respecting reachability or transport? Recurrence is not evidence of resilience by itself.

## 2. Research comparator

Li, Abinova and Li, *Quantifying structural resilience in temporal networks through persistent cycles*, Scientific Reports (22 March 2026), DOI https://doi.org/10.1038/s41598-026-44835-4, define temporal cycle participation metrics over sliding-window graph sequences and study their relation to node-removal resilience. Their framework constructs temporal cycle hyperedges, calculates node-level Temporal Cycle Number (TCN) and Temporal Cycle Ratio (TCR), and compares disruption behavior with static/temporal graph baselines.

The Symthaea implementation is a bounded, evidence-qualified adaptation of those ideas. It does not claim a direct reproduction of their datasets or experiments.

ZigzagNetVis (IEEE TVCG, October 2025), DOI https://doi.org/10.1109/TVCG.2025.3528197, uses zigzag persistence to identify meaningful topological changes across dynamic graph sequences. It motivates a later zigzag-analysis track, not a relabelling of the current recurrence score.

## 3. Input contract

`TemporalMycelialSequence` binds:

- session ID;
- coordinate-frame ID and calibration-manifest digest;
- one consistent observation-scope/ROI ID across all windows;
- ordered, non-overlapping observation windows;
- per-window source-observation digest;
- stable observation-scope/ROI ID (the declared imaging/annotation region cannot silently change across windows);
- stable node identities;
- stable edge identities, endpoints and explicit evidence class;
- coverage status for each window.

The edge evidence enum contains:
- directly observed hypha;
- accepted parent branch;
- accepted anastomosis;
- synthetic fixture (only permitted with explicit config opt-in).

There is no candidate-fusion edge type. Contact proposals from the tolerance filtration cannot enter this recurrence module without first crossing a separate acceptance/evidence boundary.

### Annotated-reconstruction adapter

`temporal_snapshot_from_annotated_skeleton` converts one `AnnotatedMycelialSkeleton` into a temporal snapshot. It retains the frame/source-observation digest, junction IDs and annotated segment IDs. Every segment must have exactly one explicit evidence record supplied by the caller. The adapter does **not** infer that a model-produced segment is directly observed, and does not invent a graph edge because a junction is labelled `ProjectedCrossing`. The caller must name whether each segment is directly observed, an accepted parent branch, or an accepted anastomosis. Candidate contacts cannot be supplied because the temporal evidence enum has no candidate value.

This is the intended bridge from the reconstruction scorecard to recurrence analysis: same annotated source identity, stable junction/segment IDs, explicit evidence classification, and a caller-supplied fixed ROI scope.

Edge IDs must retain the same undirected endpoints and evidence kind across all windows. A changing edge identity is a new edge, not silently the same biological structure. Node identity and edge identity therefore require a separately qualified tracking/annotation pipeline.

The coordinate frame and calibration digest must remain identical across the sequence. Each window retains its own source image/observation digest.

## 4. Presence, absence and unknown

Each observation window is classified as either:

- **Complete within declared scope:** a missing edge is known absent in that scope.
- **Partial or occluded:** a missing edge is unknown, not absent.

For a cycle (c) defined by a sorted stable edge-identity set:

[
phi(c,t)=
egin{cases}
1 & 	ext{if all cycle edges co-exist in window }t\
0 & 	ext{if at least one edge is missing and the window is complete}\
? & 	ext{if an edge is missing and the window is partial/occluded}
end{cases}
]

A cycle is a simultaneous simple cycle inside a single observation window. A set of edges that appears across different windows but never co-exists is not a cycle occurrence.

The module records the exact cycle identity, edge set, node set, observation state and window duration for every observed cycle. Partial windows add unknown duration; they do not turn an unobserved edge into a negative example.

## 5. Bounded exact cycle enumeration

The first implementation enumerates all simple cycles in each observed graph using deterministic depth-first search:

- stable node and edge ordering;
- minimum-node start rule to avoid start-node duplicates;
- cycle identity is a digest of the sorted stable edge-ID set;
- parallel observed hypha edges can form a 2-edge multigraph cycle;
- a graph crossing alone never creates an edge;
- cycle enumeration fails closed when a bound is exceeded.

Bounds:
- 4,096 windows per sequence;
- 128 nodes and 512 edges per window;
- 2,048 simple cycles per window;
- 32,768 distinct cycle records per sequence;
- 1,000,000 cycle-window observations per profile;
- 1,000,000 edge-window observations retained in the detached profile receipt.

This is a bounded small/medium-graph method, not a scalable general graph persistence engine. If cycle enumeration exceeds the limit, the profile fails instead of returning a silently truncated cycle set.

## 6. Recurrence and node participation

For a cycle (c), define:

- (T_p(c)): duration for which the cycle is observed present;
- (T_a(c)): duration for which it is known absent in complete windows;
- (T_u(c)): duration in which its state is unknown;
- known duration (T_k(c)=T_p(c)+T_a(c)).

Persistence among known observations:

[
P(c)=rac{T_p(c)}{T_p(c)+T_a(c)}
]

Known-observation coverage:

[
K(c)=rac{T_p(c)+T_a(c)}
{	ext{total observed duration}}
]

A cycle is considered recurrent only when all configured conditions hold:
- minimum number of present windows;
- minimum present duration;
- minimum persistence fraction (P(c));
- minimum known-duration fraction (K(c)).

This prevents a cycle seen once in one partially observed window from being called persistent.

For a node (i), the temporal cycle number is:

[
TCN_i=sum_{cin C_{mathrm{persistent}}}
T_p(c),s(c,i)
]

where (s(c,i)=1) when node (i) participates in cycle (c).

The implementation computes a duration-weighted ratio:

[
TCR_i=
rac{TCN_i}
{sum_t k_i(t),Delta t_t+epsilon}
]

where (k_i(t)) is the observed degree in window (t), (Delta t_t) is that window's duration and (epsilon) is a fixed small denominator guard. With equal-duration windows, the common (Delta t) factor cancels and this reduces to the window-count-normalized ratio. For variable-duration windows, this is a time-exposure-weighted extension.

**TCR is withheld** whenever any window is partial/occluded, because degree exposure is then incomplete. TCN remains a lower-bound participation measure based on positively observed persistent cycles.

## 7. Output and provenance

The profile contains:

- sequence digest;
- sorted union of source node IDs;
- stable edge-evidence inventory with canonical endpoints and per-kind observation counts;
- per-window node-ID and edge-ID inventories, coverage state, duration, components, cycle rank and simple-cycle count;
- full cycle records and present/absent/unknown observations;
- persistent cycle hyperedges over participating nodes;
- node-level TCN, TCR (or explicit unknown), degree exposure and persistent-cycle count;
- total, complete and partial observation duration;
- exact recurrence thresholds and synthetic-evidence opt-in.

The canonical digest binds the source sequence digest, coordinate/calibration/scope identities, edge-evidence inventory including edge endpoints, full per-window node/edge inventories, all window metrics, cycles, observation states, hyperedges and node-level metrics.

The detached profile validator independently rechecks that each reported cycle's edges form one connected degree-two multigraph over exactly its node identity set; each window's Present/Absent/Unknown value agrees with its stored edge inventory and coverage flag; all source nodes and edges are accounted for; and TCN/TCR/degree-exposure totals agree with the retained inventories. This means a serialized profile does not rely only on trusting its producer's in-memory calculation.

Synthetic edge records require `allow_synthetic_evidence=true`. The default is false. The profile retains synthetic observation counts so a conformance run cannot masquerade as an empirical observation.

## 8. What this does not claim

- Not zigzag persistence.
- Not continuous-time paths or time-respecting routing.
- Not flow or transport measurement.
- Not proof that recurring cycles are functionally redundant.
- Not evidence that cycle participation causes robustness.
- Not a biological growth simulator.
- Not a substitute for complete image acquisition, edge identity tracking or calibrated graph reconstruction.

The recurrence profile must be followed by controlled node, edge and region disruptions and recovery-cost experiments before making a resilience claim.

## 9. Proposed benchmark gates

All targets are proposed, not achieved:

| Metric | Initial target |
|---|---:|
| Deterministic cycle identity under input-order permutations | 100% |
| Stable-edge endpoint/evidence mutation accepted | 0 |
| Missing edge in partial window classified absent | 0 |
| Candidate contact imported without acceptance | 0 |
| Synthetic edge accepted without explicit opt-in | 0 |
| Profile TCN/TCR consistency against independent small-graph oracle | 100% |
| Cycle enumeration silently truncated | 0 |
| Snapshot recurrence receipt digest stable on frozen input | 100% |

Next research:
1. Compare recurrence/TCN/TCR to static degree, betweenness, cycle rank, edge connectivity, temporal efficiency and random baselines.
2. Test whether TCN/TCR predict controlled disruption outcomes on held-out organisms/time windows.
3. Add bootstrap/annotation uncertainty and explicit edge tracking confidence.
4. Implement zigzag persistence on an independent sequence representation and compare its temporal-scale findings to simple-cycle recurrence.
5. Validate cycle hyperedge persistence on a matched longitudinal microscopy dataset.

## 10. Relationship to the other Mycelial benchmarks

- [#13145](https://github.com/Luminous-Dynamics/symthaea/issues/13145): junction–hypha representation and empirical AMF geometric efficiency.
- [#13144](https://github.com/Luminous-Dynamics/symthaea/issues/13144): functional failure and repair resilience.
- [#13157](https://github.com/Luminous-Dynamics/symthaea/issues/13157): temporal Pareto growth of coverage, directness and cost.
- [#13180](https://github.com/Luminous-Dynamics/symthaea/issues/13180): temporal cycle recurrence / zigzag research track.
- [#13182](https://github.com/Luminous-Dynamics/symthaea/issues/13182): annotated reconstruction scorecard.

Each benchmark answers a different question. The goal is to connect them with shared evidence and identity digests, not collapse them into one speculative “mycelial intelligence” score.

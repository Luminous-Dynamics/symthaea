# TOPO-ADAPT-001 — deterministic reference fixtures v0.1

**Owner:** [#14123](https://github.com/Luminous-Dynamics/symthaea/issues/14123)  
**Status:** frozen synthetic evaluator fixtures; no controller implementation  
**As-of:** 2026-11-28  
**Fixture file:** [topo_adapt_fixture_v0_1.json](../../tools/topo-adapt-benchmark/fixtures/topo_adapt_fixture_v0_1.json)  
**Fixture blob SHA:** `b694c8a309bf4c09f01584451c4ecb86bd1f4ad2`

## Purpose

This first slice tests whether an evaluator can distinguish:
- a situation where a graph edit is useful;
- a situation where no edit is justified;
- a situation where a failed bridge requires an alternate route.

The three episodes are deliberately small and hand-checkable. Each graph contains 12 nodes and integer capacities. Source supply, sink demand, active edges and candidate edge additions are separate fields. Candidate edges are inactive until a controller accepts one.

The fixture tests evaluator arithmetic and provides an exact reference target. It does **not** implement or score a learned policy.

## Independent reference verification

For each episode, compute maximum flow from a super-source to a super-sink, with source supply and sink-demand arcs. Cross-check the result against exhaustive minimum-cut enumeration over all subsets of the 12 graph nodes.

| Episode | Reference condition | Max flow | Minimum cut | Correct edit |
|---|---|---:|---:|---|
| `stationary_demand_control` | Two existing sinks; new sink demand is zero | 6 | 6 | No edit |
| `new_sink_demand_shift` | New sink demand activates at 2 flow units | 6 before edit; 8 after useful edit | 8 after useful edit | Add `0 → 5` |
| `bridge_failure_reroute` | Existing bridge `1 → 2` fails | 0 after failure; 4 after useful edit | 4 after useful edit | Add `3 → 2` |

Candidate-edge controls produce no additional service when they terminate in the decoy dead ends.

### What the arithmetic establishes

1. The stationary control has full service before any edit. Every candidate edge leaves max flow at 6, so an edit only consumes the declared edit and maintenance budget.
2. The new-sink episode initially serves 6 of 8 requested units. Adding `0 → 5` connects the isolated source-to-sink branch and raises maximum flow to 8. Either decoy edit leaves flow at 6. Uniform random selection among the three candidates succeeds with probability 1/3 in this finite fixture.
3. The bridge-failure episode falls from 4 to 0 after the declared edge capacity becomes zero. Adding `3 → 2` restores max flow to 4; either decoy leaves it at 0.

The max-flow/min-cut equality is checked independently for every stated reference condition. All capacities and demands are integer fixture values, not empirically calibrated network parameters.

## What this does not establish

- No learned or adaptive controller has been implemented or evaluated.
- No biological network, fungal transport, animal tissue bioelectricity or physical repair is modelled.
- No general network-design result follows from three small fixtures.
- The one-third random success is a property of this candidate catalogue, not a population-level estimate.
- No SOTA, efficiency or industrial deployment claim is made.

## Next executable steps

1. Add an evaluator that parses this frozen fixture and emits a source/request-bound receipt.
2. Add an independent max-flow implementation plus a separate cut-enumeration oracle in tests.
3. Add node relabeling, edge-order permutation and capacity-scaling metamorphic checks.
4. Add fixed-topology, weight-only, local-greedy and random-rewire baselines before learned policies.
5. Extend to multi-sink bottlenecks and held-out topology generators only after the v0.1 fixture and receipt semantics qualify.

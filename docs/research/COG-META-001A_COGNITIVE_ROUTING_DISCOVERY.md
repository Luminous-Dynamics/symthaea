# COG-META-001A — Cognitive routing / scheduling discovery

Status: **measurement-only product subject**

Parent: #5468  
Issue: #5469

Planning base:

```text
main = adb69f11fa8068b019cc5bb598d0c7726a197fc9
tree = 35a6c5fdba319556af9bb487838734f67c8ac0d6
```

This document and `scripts/discover_cognitive_routing_semantics.py` define the first
executable discovery boundary for the metareasoning program. They do **not**
change runtime cognition.

## Question

Before introducing `CognitiveBidV1` or a new allocator, establish what already
decides:

```text
whether cognition runs
what cognitive work runs
what content wins attention
when cognition requests more evidence
when deliberation stops or defers
what resource signals gate cognition
where cognition proposes an external effect
which work is mandatory/protective
```

The inventory is intentionally conservative.

## Measurement semantics

The script scans Rust source under production `src/` trees and excludes
standalone documentation, tests, examples, benches, fixtures, patches, vendor
and target trees.

A discovered file means only:

```text
one frozen lexical pattern appears in a production Rust file
```

It does **not** mean:

```text
the matching occurrence is reachable production code
the matching symbol owns the semantic transition
the mechanism executed in a particular run
the mechanism improves cognition
```

Embedded `#[cfg(test)]` modules inside production files remain part of the
file-level lexical census. Runtime reachability requires a separate witness.

For every category member the report binds:

```text
repository-relative path
byte length
Git blob SHA-1
SHA-256
lexical match count
```

It also emits deterministic path-set digests.

## Categories

The first census keeps these surfaces separate:

```text
cadence_scheduling
content_competition
metacognitive_control
specialist_selection
deliberation_stopping
resource_gating
planning_search_simulation
memory_operation_selection
external_effect_proposal
mandatory_or_protective
```

These are discovery categories, not reliability or authority classes.

## Required source witnesses

The product fails discovery if these current architectural anchors disappear
without a new measurement profile:

```text
symthaea-cognitive-types CycleUrgency / should_run
CognitiveSubsystem / custom should_run hook
symthaea-workspace AttentionBid / GlobalWorkspace
HDC MetacognitiveMonitor recommendation surface
MetaCognitiveReasoner
meta_reasoning_confidence production consumer
```

The report also checks three concrete control witnesses:

```text
meta_reasoning_confidence > 0.7
    -> learning-rate modulation path exists

GlobalWorkspace winner selection
    -> content competition exists

MetacognitiveRecommendation::ReduceLoad match
    -> recommendation-consumer-shaped path exists
```

Presence is not proof of construct validity or benefit.

## Important semantic-collision finding

Pre-implementation audit already shows that identical vocabulary is not a
sufficient unification criterion.

At minimum, the current tree contains multiple independently defined
`MetacognitiveRecommendation` types with materially different variants, and
multiple metacognitive reasoner names/types.

The inventory therefore emits exact definition sets for:

```text
CycleUrgency
CognitiveSubsystem
AttentionBid
GlobalWorkspace
MetacognitiveRecommendation
MetaCognitiveReasoner
MetacognitiveReasoner
```

and marks any symbol with more than one production-file definition as:

```text
SEMANTIC_COLLISION_CANDIDATE
```

That marker means:

```text
same lexical type name
!= same semantic proposition
!= safe to merge
```

For example, one recommendation family may describe load/focus/integration
control while another describes dialogue confidence or dream-frequency
adjustment. COG-META must not erase those distinctions merely to obtain one
enum.

## Workspace boundary

The existing global workspace is a content-selection owner.

Target separation remains:

```text
AttentionBid
= content salience / broadcast competition

future CognitiveBidV1
= whether executing cognitive work is worth its resource cost
```

No conversion between the two is authorized by this tranche.

## Metacognitive-control boundary

Internal diagnostics such as:

```text
Phi
Phi stability
heuristic meta-confidence
self-model fit
uncertainty
prediction error
```

may appear in the census because they currently participate in control paths.

Their presence does not establish that they are calibrated value-of-computation
estimators.

Compose #5413 and #2699:

```text
diagnostic
-> candidate feature
-> prospective calibration
-> bounded control influence if demonstrated
```

## Mandatory / protective boundary

The census searches for watchdog/interlock/veto/safety-shaped work separately
because ordinary competitive compute allocation must not silently acquire the
right to suppress mandatory protection.

```text
mandatory work
!= high-value bid
```

The eventual type contract may represent mandatory/competitive/opportunistic
classes, but 001A does not introduce them.

## Type-ownership decision remains open

Current architectural preference is:

```text
symthaea-types::metareasoning
```

because `symthaea-types` is the general shared cross-crate type owner, while
`symthaea-cognitive-types` is explicitly cycle telemetry.

However, #5470 must not freeze that choice until this inventory demonstrates
the actual dependency/consumer graph. A dedicated dependency-light types crate
is acceptable only if the measured graph justifies it.

## Product output and qualification

Local/product execution may emit:

```text
result=PASS_DISCOVERY
```

only when the frozen discovery contract itself is internally satisfied.

That result is **not** yet a repository qualification result.

A later exact never-merge qualifier must:

1. prove one-semantic-child lineage from this product;
2. detach the exact product subject;
3. execute the discovery twice;
4. require byte-identical reports;
5. preserve the full report as an artifact;
6. freeze the observed path sets/digests and definition sets in the next
   measurement stage.

Until that qualifier actually executes successfully:

```text
product contains a discovery script
!= qualified inventory
```

## Nonclaims

COG-META-001A does not establish:

- that System 1 / System 2 is the correct ontology of cognition;
- that Symthaea has exactly one workspace or one metacognitive mechanism;
- that same-name types should be merged;
- that any discovered path is runtime-reachable;
- that any current scheduling decision is good or bad;
- that any confidence/Phi/uncertainty signal deserves compute authority;
- that a future metareasoner improves capability;
- external action/tool/actuation authority;
- consciousness or phenomenology.

Its sole purpose is to make the current routing/control surface exact enough
that the next type and scheduler PRs do not duplicate, conflate, or bypass
existing owners.

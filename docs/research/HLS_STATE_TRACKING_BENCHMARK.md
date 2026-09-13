# HLS State-Tracking Benchmark Protocol

Status: model-agnostic benchmark contract

## Research question

Does preserving explicit holographic structure during continuous-time state evolution improve long-horizon state tracking, compositional queries, and retrospective recall at competitive state size and event cost?

The benchmark is designed to falsify that hypothesis rather than to showcase HLS-specific features.

## World

The latent world contains three finite domains:

- entities,
- objects,
- locations.

Two mutable relations define the state:

```text
object --owned_by--> entity --located_at--> location
```

Events either move one entity or transfer one object. These changes are deliberately independent.

## Why the object-location query matters

`ObjectLocation(object)` cannot be answered from one relation alone. The exact answer is

```text
owner = owned_by(object)
location = located_at(owner)
```

and both edges may have changed at different irregular times. It therefore provides a minimal compositional state-tracking test instead of a simple last-value memory task.

## Time

Inter-event intervals are sampled deterministically in log space from `[min_dt, max_dt]`. The default range is `1e-3` to `1e2`, spanning five orders of magnitude.

The event index and physical time are both retained. Models that require discrete positions may consume the event order, while continuous-time models may additionally consume the exact elapsed time. A reported comparison must state which timing information each model received.

## Query classes

Every query bundle contains current-state queries for:

1. entity location,
2. object owner,
3. object location.

With configurable probability, the bundle also contains a strictly retrospective `ObjectLocation` query whose target state precedes the latest observed event.

This separates:

- current state retention,
- relational retrieval,
- compositional state tracking,
- historical state reconstruction.

## Metrics

The built-in scorer reports:

- total accuracy,
- current-state accuracy,
- historical accuracy,
- entity-location accuracy,
- object-owner accuracy,
- object-location/compositional accuracy.

Architecture papers should additionally report:

- recurrent state bytes,
- parameter count,
- FLOPs or measured event latency,
- peak memory,
- accuracy versus event horizon,
- accuracy versus time-gap extrapolation,
- training examples / optimization steps.

## Fair-comparison contract

A baseline comparison is valid only when:

- every model receives the same generated world and event/query split;
- no model receives oracle answers or future event information in its input;
- train/validation/test seeds are disjoint;
- hyperparameter selection never uses the held-out test seeds;
- model capacity, training compute, and inference-state footprint are reported;
- recurrent and continuous-time models receive equivalent timestamp information where possible;
- deterministic benchmark generation is preserved exactly for reproduction.

## Recommended experimental matrix

The first architecture ablation should compare:

```text
HLS diagonal
HLS + invariant context
legacy Symthaea HDC-LTC
standard CfC
GRU/LSTM
modern SSM baseline
attention baseline
```

The benchmark itself does not encode or train any of these systems. Model adapters must remain outside the oracle so failures cannot alter ground truth.

## Generalization splits

A strong paper should use at least four held-out axes:

- **length:** train on shorter streams, test on longer streams;
- **time:** train on narrower `dt`, test on unseen smaller/larger gaps;
- **cardinality:** test with more entities/objects/locations than training;
- **composition/history:** hold out harder two-hop retrospective combinations.

A result that only memorizes a fixed world cardinality or fixed time grid does not establish the HLS hypothesis.

## Non-claims

This benchmark alone does not establish intelligence, consciousness, general reasoning, or superiority over Transformers/SSMs. It establishes a controlled test bed for one narrower claim: persistent structured state evolution under irregular time and mutable compositional relations.

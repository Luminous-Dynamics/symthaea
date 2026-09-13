# HLS learned-vs-frozen preregistered ablation

This note freezes the first multi-world learned-vs-frozen experiment design for diagonal Holographic Liquid State (HLS) before any result from that design is interpreted.

## Status

`ExactLearningAblationPlan::research_v0()` is an **exploratory preregistration**. It is fixed before execution/interpretation, but it is not yet a confirmatory study. If pilot evidence motivates different dimensions, optimizer settings, world scale, or training duration, a later confirmatory plan must be introduced under a new version rather than silently changing `research_v0`.

This first preregistration is deliberately **current-state only**. The fixed associative decoder has a justified algebra for current one-hop relation retrieval and current two-hop object-location composition, but there is not yet a justified temporal-addressing algebra for historical recall. Historical queries are therefore excluded rather than approximated with an arbitrary lag role.

## Paired design

One deterministic initial HLS parameterization is created from a fixed cell seed.

Two copies are retained:

- **frozen** — never receives a recurrent parameter update;
- **trained** — receives one exact episode-end update after each predefined training world.

After training, both copies are evaluated on each identical held-out world. Every reported effect is therefore paired by held-out seed.

No best-seed or best-checkpoint selection is performed.

## `research_v0` constants

Architecture:

- HLS dimension: 512;
- activation: tanh;
- global state norm limiter: disabled (`infinity`) so the exact local eligibility theorem applies;
- trainable recurrent scalars: `6 * 512 = 3,072`;
- fixed cell seed: `0x484C5330`;
- fixed codec seed: `0x48444330`.

World:

- entities: 16;
- objects: 32;
- locations: 8;
- post-initialization mutations: 1,000;
- one current query bundle every 10 mutation events;
- historical-query probability: **0.0**;
- irregular event intervals: log-uniform from `1e-3` to `1e2`;
- every seed-randomized initial world is fully disclosed by an observable shuffled initialization prefix before scoring starts.

The current query families are:

- entity -> location;
- object -> owner;
- object -> owner -> location composition.

Optimizer:

- learning rate: `1e-3`;
- episode-mean exact recurrent gradient;
- global gradient-norm clip: `0.5`;
- absolute recurrent parameter bound: `3.0`;
- associative cosine-loss epsilon: `1e-4`;
- one update after each complete training world.

Training world seeds:

`10001..=10012` (12 worlds).

Held-out world seeds:

`20001..=20024` (24 worlds).

## Reported held-out effects

For every held-out seed the runner stores frozen and trained metrics plus paired deltas:

- associative cosine-loss delta (`trained - frozen`; negative is favorable);
- overall current-state accuracy delta (`trained - frozen`; positive is favorable);
- compositional object-location accuracy delta.

For each delta family the aggregate reports:

- paired mean;
- sample standard error across held-out worlds;
- minimum and maximum world-level effects;
- number of positive, negative, and exactly zero world-level effects.

These summaries are descriptive pilot statistics, not a substitute for a later confirmatory statistical analysis.

## Fail-closed scope rule

`research_v0` requires `historical_query_rate == 0.0` exactly. Any scope drift that enables historical queries is rejected before constructing or mutating experimental HLS state. The lower-level current-only training gate likewise scans the complete benchmark before reset, recurrence, trace accumulation, loss evaluation, or parameter mutation.

Historical recall is not being counted as a failure of this experiment because the current decoder has no theorem for it. It will receive a separate experiment version only after an explicit temporal-addressing algebra is implemented and independently qualified.

## Interpretation rules fixed before execution

1. A null or negative learned-vs-frozen result is valid evidence and must be reported.
2. A favorable result on only a minority of held-out worlds is not sufficient evidence of a robust architectural gain.
3. Training-world performance is not evidence of generalization.
4. No seed may be moved between train/test sets after observing results.
5. No hyperparameter may be changed and still be described as the same `research_v0` run.
6. The same fixed associative decoder/codebook must be used for frozen and trained cells.
7. Historical recall may not be retrofitted into `research_v0`; it requires a separately versioned temporal experiment.
8. External model comparisons should wait until the within-HLS learned-vs-frozen result is mechanically qualified.

## Next step after execution

If `research_v0` shows a stable useful signal, freeze a larger **confirmatory_v1** plan with untouched new world seeds and explicit success/failure criteria. If it is null or negative, use the pilot diagnostics to identify whether the limiting factor is representation, trainability, timescale allocation, or the diagonal symmetry constraint before broadening the architecture.

Historical memory proceeds on a separate line: first define and qualify a temporal algebra, then freeze a new benchmark/seed plan before interpreting any historical-recall score.

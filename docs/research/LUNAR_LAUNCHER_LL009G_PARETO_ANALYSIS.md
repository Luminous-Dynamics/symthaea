# LL-009G — uncertainty-aware lunar transport Pareto analysis

Status: **Phase-0 research comparison only. Not infrastructure selection, site approval, corridor qualification, or launch authority.**

LL-009G is the first layer in the lunar transport program that compares candidate architectures. It consumes only a passing LL-009F common-accounting receipt and deliberately refuses to collapse the comparison into one weighted score.

## Why Pareto structure

A lunar transport architecture can be simultaneously:

- lower imported mass but higher peak power;
- lower energy per kilogram but slower;
- more reliable but maintenance-heavy;
- higher throughput but dependent on more reusable hardware;
- cheaper in one accounting model but harder to bootstrap from Earth.

There is no defensible universal scalar weight vector for those objectives in Phase 0.

LL-009G therefore asks which candidates are dominated and which remain on the multi-objective frontier.

## Pinned objective policy

Objective direction is not supplied by candidate data. It comes from the checked-in and hash-bound:

`configs/lunar_transport/ll009g_pareto_objectives.json`

The baseline policy minimizes:

- imported mass;
- energy per delivered kilogram;
- peak power;
- imported maintenance/spares mass;
- propellant/reaction-mass demand;
- expected cargo loss;
- reusable hardware inventory;
- latency.

It maximizes:

- throughput capacity;
- availability;
- reliability/success probability.

`local_mass_kg` is intentionally report-only in the first policy. Large local construction mass can be a burden, but it can also represent desirable displacement of Earth imports; blindly minimizing it would encode a questionable value judgment.

Changing the objective policy changes the analysis lineage because the exact policy SHA-256 is emitted in every result.

## Central Pareto dominance

Candidate A centrally dominates candidate B when A's central estimate is no worse on every objective and strictly better on at least one.

This is the familiar deterministic Pareto relation, but it ignores uncertainty width.

## Robust interval-separation dominance

LL-009G also computes a much stricter relation.

For a minimized objective, A is robustly no worse than B only when:

```text
A.high <= B.low
```

For a maximized objective:

```text
A.low >= B.high
```

The condition must hold on every objective and be strict on at least one.

This means A's entire declared marginal interval must lie on the favorable side of B's entire interval on every objective.

The result produces a separate **robust front**.

A candidate can therefore be centrally dominated but remain robustly non-dominated because the intervals overlap. LL-009G reports those `central_not_robust_edges` explicitly rather than presenting the central estimate as certainty.

## Important uncertainty boundary

The low/central/high intervals are marginal intervals. LL-009G does not assume they are independent, Gaussian, or jointly realizable.

It therefore does **not** calculate:

- probability candidate A beats candidate B;
- expected utility;
- confidence that one technology is globally superior;
- Monte Carlo joint ranking from invented correlations.

Those would require a separately declared joint uncertainty model.

## Unit and domain gates

Every objective must have identical units across all candidates.

The pinned policy also carries simple domains:

- physical burden/service quantities: non-negative;
- availability and reliability probabilities: `[0, 1]`;
- any future explicitly unbounded metric must say so.

NaN and infinity are rejected.

## Output

The canonical LL-009G analysis includes:

- exact LL-009F input-receipt hash;
- normalized LL-009F input hash;
- exact objective-policy hash;
- objective directions/domains;
- per-candidate objective vectors;
- all non-objective metrics preserved as report-only;
- central dominance edges;
- central Pareto front;
- robust dominance edges;
- robust Pareto front;
- central dominance edges that disappear under robust interval separation;
- front-stability sets;
- canonical analysis SHA-256;
- explicit non-claims.

There is intentionally no `weighted_score`, `utility`, or `winner` field.

## Usage

Run the dependency-free synthetic self-test:

```bash
python3 scripts/analyze_ll009g_pareto.py --self-test
```

Analyze a passing LL-009F receipt:

```bash
python3 scripts/analyze_ll009g_pareto.py \
  --receipt docs/research/evidence/<corridor>/ll009f-trade-receipt.json
```

## Executed self-test

The pre-commit self-test uses four synthetic candidates:

- A centrally dominates B but their uncertainty intervals overlap, so B remains on the robust front;
- A robustly dominates D with interval separation;
- C presents a genuine cost/service tradeoff and remains non-dominated;
- `local_mass_kg` remains report-only.

The test also verifies:

- inconsistent objective units fail closed;
- probability/fraction values outside `[0,1]` fail closed;
- no `weighted_score` or `winner` field exists in the canonical result.

## Next step: switching thresholds

One Pareto analysis describes one common demand/accounting scenario.

The next layer should run the exact same candidate definitions across a controlled scenario sweep—for example:

```text
corridor distance
annual throughput demand
cargo mass/class
available peak power
local-material fraction
maintenance cadence
reliability target
```

It should report where Pareto-front membership changes on the sampled grid, without claiming a precise interpolated crossover unless the response model supports interpolation.

That is how Phase 0 can eventually produce statements such as:

> ballistic surface freight first enters the non-dominated set beyond this tested distance/throughput region, while continuous track remains non-dominated below it.

The threshold should emerge from the evidence; it must not be chosen in advance.

## Non-claims

A central or robust Pareto front is research structure, not a decision. It does not prove that any candidate is economically superior, mature enough to build, safe, human-rated, or authorized for physical operation.

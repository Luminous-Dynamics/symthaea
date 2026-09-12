# symthaea-energy-benchmark-zero-controls

Preregisterable, property-blind controls for Energy Discovery Benchmark Zero.

This crate answers a narrower question than the physics baseline itself:

> Does the fixed composition-informed ranking perform better than a frozen set of rankings that know nothing about material properties?

It intentionally does **not** force a property-blind control through a fake band-gap predictor just to obtain MAE.

## Benchmark plan

`BenchmarkPlan` binds:

- positional fold;
- explicit target interval;
- top-k;
- exact composition-baseline method version;
- exact blind-control method version;
- fixed control-replicate count.

The plan has a domain-separated SHA-256 that can be registered externally before execution.

Important: the digest by itself proves content identity, not chronology. A claim that the plan was preregistered requires a separately immutable/timestamped evidence reference created before results are observed.

## Property-blind control

The control method uses 256 fixed rankings.

For each replicate, candidate order is produced from SHA-256 over:

- a frozen domain separator;
- the replicate index;
- candidate id.

The ranking function receives no:

- composition;
- target window;
- predicted property;
- experimental truth.

Target and truth are introduced only after a blind order has been frozen, when ranking metrics are measured.

## Why no null MAE

A truly property-blind ranking does not predict a band gap. Assigning every candidate the target midpoint or a pseudo-random gap would manufacture a prediction model that is not the control we want.

Therefore the blind ensemble is compared only on ranking-sensitive metrics:

- top-k hit count;
- top-k precision;
- top-k recall;
- target regret.

The composition baseline's MAE remains available in its own Benchmark Zero receipt, but there is no fake null MAE comparison.

## Descriptive control fractions

The receipt reports the fraction of the 256 fixed blind controls that are no better than the composition baseline on:

- top-k hit count;
- target regret;
- both simultaneously.

These values are **not p-values** and are not labeled statistical significance. The 256 controls are a deterministic reference ensemble, not independently sampled physical experiments.

## Controlled comparison receipt

A successful controlled run binds:

- benchmark plan SHA-256;
- optional external registration evidence reference;
- #1769 leakage-qualification SHA-256;
- #1781 baseline combined receipt SHA-256;
- exact blind ensemble SHA-256;
- blind-control summary SHA-256;
- final comparison SHA-256;
- chronology and statistical-interpretation disclosures.

Supplying a registration reference binds that string into the final comparison identity, but this crate does not authenticate its timestamp or prove preregistration chronology.

## CLI workflow

Create a plan identity without reading benchmark truth:

`energy-benchmark-zero-controls plan <fold:0..4> <target-min-eV> <target-max-eV> <top-k>`

Then, after external registration if desired, execute:

`energy-benchmark-zero-controls run <matbench_expt_gap.json.gz> <fold:0..4> <target-min-eV> <target-max-eV> <top-k> [registration-evidence-ref]`

The run command performs no network fetch and does not emit the host-local artifact path.

## Deliberate non-claims

This crate does not establish:

- a p-value or statistical significance;
- random/independent experimental sampling;
- an official Matbench leaderboard result;
- historical blindness of the composition baseline;
- calibrated predictive uncertainty;
- material novelty, synthesizability, or device performance;
- experimental validation;
- deployment superiority;
- material certification or physical authority.

The stack remains unqualified until exact-head Cargo check/test/strict-Clippy and lockfile qualification execute under the pinned toolchain.

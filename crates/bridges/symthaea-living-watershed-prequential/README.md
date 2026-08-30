# symthaea-living-watershed-prequential

A rolling-origin **commit → forecast → reveal → verify → score** extension of the Living Watershed / Wetland Watch synthetic research witness.

This crate is still a mechanism control, not a real-world wetland predictor. Its job is to make the eventual Sentinel experiment harder to fool accidentally.

## Why v1 exists

The v0 witness proves one held-out transition can pass through:

```text
sealed history -> forecast/abstention -> proper score -> result manifest -> replication lineage
```

That is useful, but a single transition is a weak experimental surface. It can be dominated by one lucky state and does not exercise temporal leakage repeatedly.

v1 therefore uses a fixed **prequential / rolling-origin** episode.

For forecast origins `t = t0 .. tN`:

```text
history <= t-1
      |
      +--> candidate A output --+
      +--> candidate B output --+--> all outputs validated + digested
      +--> candidate ... -------+
                                      |
                                      v
                               reveal state t
                                      |
                                      v
                           verify prior commitment
                                      |
                                      v
                         score already-issued outputs
```

The next state is never passed through `TrajectoryGenerator<Observation = WatershedHistory>`.

## Two-phase run boundary

### 1. Prepare

`prepare_episode(...)` deterministically creates a `PrequentialEpisodePlan` before the research run is registered.

For every frozen forecast origin it records:

- v0 sealed-fixture dataset-manifest digest;
- v0 held-out verification commitment digest;
- origin index;
- a content digest over the complete prequential plan.

The research run later binds this plan digest as its dataset lineage.

A digest is an **integrity commitment**, not a secrecy primitive. The synthetic target has a tiny state space and a sufficiently privileged attacker may brute-force it. Real campaigns should keep commitments outside the candidate-model process or use stronger custody/blinding mechanisms.

### 2. Execute

At each origin:

1. regenerate the exact committed sealed fixture;
2. compare regenerated fixture/verification digests with the plan;
3. pass only predictor-visible history to every candidate;
4. collect every output;
5. validate every distribution's semantic binding;
6. digest every accepted output;
7. only then reveal the next observation;
8. reconstruct the prior v0 verification digest from the revealed observation;
9. reject if the reveal does not match the pre-run commitment;
10. score the already-issued distributions with the canonical Futures Laboratory Brier implementation.

A candidate binding failure aborts the origin **before verification is revealed and before any candidate is scored**.

## Forecast binding gate

A scoreable wetland-stress distribution must satisfy all of the following:

- `issued_at_tick` equals the last predictor-visible tick;
- `horizon == Horizon(1)`;
- exact outcome-space id `living-watershed/wetland-stress-next-step/v0`;
- exactly one `Boolean(true)` branch and one `Boolean(false)` branch;
- `unsupported_mass == 0` for this fully specified binary target;
- canonical Futures probability/mass validation inherited from `symthaea-futures-core`.

Typed abstention remains valid and is retained separately.

This closes a semantic gap that probability normalization alone cannot close: a mathematically valid distribution can still be a forecast for the wrong tick, horizon, or target.

## Coverage is load-bearing

Mean Brier is reported only across cases where a candidate emitted a distribution.

Therefore every aggregate also records:

- `scored_steps`;
- `abstained_steps`;
- `total_steps`;
- `coverage`.

A candidate that forecasts only easy cases can otherwise look artificially strong. v1 deliberately does **not** rank models with unequal coverage from mean Brier alone.

A real Wetland Watch skill experiment should preregister either:

- a minimum coverage floor;
- a selective-prediction utility/cost rule;
- or another explicit abstention-aware comparison rule.

It should not invent that rule after seeing results.

## Protocol/plan binding

`run_prequential_baselines(...)` reconstructs the expected frozen v1 protocol from the prepared plan's first origin and fixed evaluation-case count and requires the protocol digest to match.

This prevents accidentally running, for example, a 20-case episode under a preregistration frozen for 5 cases while still labeling the result confirmatory.

The run is then bound to:

```text
FrozenProtocol
    -> exact source commit
    -> precommitted episode-plan digest
    -> reproducibility capsule digest
    -> seed-manifest digest
    -> forecast ledger
    -> verification ledger
    -> primary mean scores + coverage
    -> ResearchResultManifest
```

## Revalidation of v0 public specifications

`SyntheticWatershedSpec` is intentionally a simple public data type. Public fields can be mutated after construction.

v1 therefore reconstructs an incoming template through the v0 validated constructor before accepting it, rather than assuming that possession of the Rust type proves the fields remain valid.

## Adversarial tests

The current test tranche covers:

- plan digest mutation;
- plan commitment tampering even when an attacker recomputes the outer plan digest;
- exact prefix histories at each rolling origin;
- stale issue ticks;
- wrong horizons;
- wrong outcome spaces;
- non-binary support;
- non-zero unsupported mass;
- explicit coverage beside selective mean score;
- revalidation of a mutated public v0 template;
- frozen-protocol / episode-design mismatch;
- result-manifest binding to the exact precommitted plan;
- direct-replication lineage on a genuinely different episode plan under the same protocol.

## Important non-claims

v1 does **not** establish that:

- the reduced-order hydrology bucket adequately represents a real wetland;
- the ecological stress threshold is calibrated to field ecology;
- the synthetic observations resemble Sentinel retrieval errors;
- persistence or climatology is a sufficient real-world baseline suite;
- any Symthaea/HDC model has predictive skill;
- commitment digests provide secrecy against a privileged model process;
- temporally adjacent synthetic cases are statistically independent;
- similar scores imply replication;
- a forecast authorizes an intervention.

## Next upgrade

After this layer and the Planetary Perception parents qualify, the safest progression is:

```text
v0 single-step deterministic control
        -> v1 rolling-origin commit/reveal control
        -> frozen Sentinel scene custody
        -> spatial-block + temporal holdout splits
        -> conventional EO/statistical baselines
        -> Symthaea/HDC candidate
        -> calibration + coverage analysis
        -> new-region/time direct replication
```

For real EO validation, random pixel splits should be prohibited by default because spatial and temporal autocorrelation can make them severe leakage channels.

## Required gates

```bash
cargo fmt --all -- --check
cargo check -p symthaea-living-watershed-prequential --all-targets
cargo test -p symthaea-living-watershed-prequential
cargo clippy -p symthaea-living-watershed-prequential --all-targets -- -D warnings
```

Keep the PR draft until these execute on the exact head and the stacked parent research-integrity PRs qualify.

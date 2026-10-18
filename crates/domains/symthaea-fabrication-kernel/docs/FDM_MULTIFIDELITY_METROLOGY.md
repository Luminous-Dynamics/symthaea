# FDM multi-fidelity spatial metrology benchmark v1

## Purpose

This benchmark asks not only *where to measure*, but **which metrology method to use, at what repeat count, and when to spend for higher-fidelity evidence**.

It composes with the spatial point-prediction baseline, equal-budget adaptive sampling, and synthetic cost-budget benchmark. It does not create another generic evidence registry. The module has no instrument/printer I/O and issues no machine-control or G-code commands.

Every result is synthetic. Method noise, systematic-bias controls and abstract costs are benchmark parameters—not measurement specifications, instrument claims, physical savings or a validated process model.

## Research basis

- ISO/ASTM 52902:2023 specifies artifact-based geometric-capability assessment for AM systems and discusses measurement uncertainty, without prescribing one universal method: https://www.iso.org/standard/79683.html.
- NIST's Gauge R&R handbook emphasizes repeatability, reproducibility, stability, bias, resolution, linearity and gauge/configuration effects when characterizing production measurement systems: https://www.itl.nist.gov/div898/handbook/mpc/section4/mpc4.htm.
- NIST notes that gauge studies can inform measurement uncertainty, but only when representative of the actual measurement process and its stability: https://www.itl.nist.gov/div898/handbook/mpc/section4/mpc46.htm.
- NIST's observation-selection work links sensor capabilities and process phenomena to quality decisions in metal AM; this is relevant research precedent, not direct FDM validation: https://www.nist.gov/publications/situ-observation-selection-quality-management-metal-additive-manufacturing.
- A 2025 *Nature Computational Science* article on multi-fidelity Bayesian optimization in materials/molecular research shows that the benefit of combining cheap approximate and expensive accurate sources depends on their informativeness, cost and problem conditions: https://www.nature.com/articles/s43588-025-00822-9. That lesson motivates testing the fidelity trade-off rather than assuming multi-fidelity always wins.

## Frozen experiment

- Schema: `symthaea.fdm-multifidelity-synthetic.v1`.
- Synthetic surface families: seven from the preceding spatial sampler (smooth, anisotropic, edge, hotspot, piecewise, heteroskedastic, null effect).
- Measurement stress cases: unbiased independent noise; biased fast screen; correlated repeats; and stale reference calibration.
- Strategies: reference-only, fast-screen-only, fixed staged, coverage-guarded staged, cost-normalized decision, and seeded random.
- Six fixed total-cost budgets: 33, 40, 48, 56, 64 and 72 abstract units.
- Shared initial locations: the same five positions in the same order. Reference-only begins at reference fidelity; other policies begin with fast-screen observations.
- Up to ten additional actions follow, each a tuple of (location, method, repeat count).
- All arms replay a budget-independent selector sequence and stop before its first unaffordable action, preserving nested budget traces.
- Candidate selectors receive prior observations, candidate actions, seed/step and declared cost/noise values. They do not receive latent surface values, final evaluation results or the evaluator's stress-case parameter.

The starting measurement locations are shared, but their **fidelity differs by strategy**. This is a policy comparison under a common *location set* and total budget, not an equal-information comparison. Reports must keep that distinction visible.

## Fidelity / uncertainty model

- Fast-screen nominal single-reading noise scale: 0.025 mm.
- Reference-metrology nominal single-reading noise scale: 0.004 mm.
- Fast-screen reading noise increases 2.5× on one side for the heteroskedastic surface.
- A repeat plan is either one or three readings per action. The reported standard uncertainty uses the declared nominal noise scale and repeat correlation model.
- In the correlated-repeat stress case, fast-screen repeat correlation is 0.85 and reference correlation is 0.70. The uncertainty of the mean therefore has a correlation floor and does not gain the usual independent √*n* improvement.
- `biased_fast_screen` introduces position-dependent systematic screen bias.
- `stale_reference_calibration` introduces a systematic reference offset that is intentionally **not** included in nominal random uncertainty. The scenario label is evaluator metadata, not a selector input.

These values exist to test implementation behavior under known artificial conditions. They do not characterize a real gauge/CMM/camera/probe. Physical use would require a representative Gauge R&R / stability / bias study and a documented uncertainty budget.

## Cost model

The demo schedule uses abstract units:

- One-time setup: 2.
- Fast screen: 1 per reading.
- Reference metrology: 4 per reading.
- Travel: 2 per normalized distance unit.
- Edge-zone surcharge: 0.35 per measurement action.

This cost schedule is deliberately simple. It excludes real instrument dwell time, fixture changes, specimen flips, measurement-system requalification and opportunity cost. The report preserves cost components independently. Never call these seconds, dollars or measured savings.

## Policy intent

1. **Reference-only:** expensive, nominally lower-noise baseline.
2. **Fast-screen-only:** inexpensive baseline.
3. **Fixed staged:** follow a frozen spatial plan; upgrade flagged/borderline screen observations to reference metrology.
4. **Coverage-guarded staged:** alternate spatial exploration with targeted reference upgrades when a screen observation is critical or near the decision threshold.
5. **Cost-normalized decision:** select a location/method/repeat action by a transparent heuristic combining spatial exploration, nearest-vs-IDW disagreement, proximity to the critical threshold and marginal synthetic cost. This is not optimal Bayesian expected value of information.
6. **Seeded random:** deterministic low-cost location baseline.

No strategy is presumed best. Under biased or stale calibration, high nominal precision may be confidently wrong; under correlated repeats, repeating the same method may add much less information than independent metrology. Those are intended negative-control behaviours, not accidental data to average away.

## Metrics

- Total spend, unused budget, and separate setup / fast-screen / reference / travel / edge costs.
- Action count, total readings, method counts and number of multi-reading actions.
- Spatial coverage and unknown grid-point count.
- MAE and worst covered-location error.
- Critical-region recall and false-safe rate, with numerator/denominator counts represented separately.
- False-alarm action count and mean reported measurement uncertainty.
- Ordered measurement/action trace, per-action ledger, cost-schedule digest, and domain-separated report digest.

A false-safe is evaluated only in a covered critical region when the predicted magnitude plus twice the reported uncertainty falls below the frozen 0.10 mm critical threshold. This is a synthetic decision proxy, not a physical tolerance recommendation. Uncovered regions remain Unknown; they cannot be scored as proven-safe.

## Integrity controls

- Reject unsupported or malformed schedules, non-positive costs, invalid budgets, unsupported repeat counts, out-of-lattice coordinates and duplicate location/method pairs.
- Recompute each deterministic synthetic reading from seed, method, repeat, surface and stress case.
- Recompute uncertainty, every component of the cost ledger, cumulative spend, the summary metrics, stop reason and report digest.
- Replay the declared selection policy at every action; reject reordered or altered action traces even if the report digest was recomputed.
- Reject budget overspend and false claims that the action cap or budget was reached.
- Identical seed/policy/surface/stress/schedule yields byte-equivalent normalized report values.

SHA-256 digests bind serialized fixture identity, not a trusted source signature.

## Research gates

- Exact-head Rust formatting, focused test, compilation and clippy evidence (a skipped workflow is not a pass).
- Across a frozen seed set, compare policies on Pareto curves for false-safe rate, worst covered error, recall, Unknown fraction and measured cost.
- Demonstrate that the correlated-repeat case does not get an unjustified independent-repeat uncertainty reduction.
- Demonstrate detection of known systematic-bias/stale-calibration controls, or explicit evidence that such controls cannot be identified from allowed observations alone.
- No policy may report a covered/qualified result for an unknown region simply because the budget ran out.
- Before physical claims, collect independent instrument-calibration evidence, representative repeatability/reproducibility/stability studies, actual action timestamps/costs, repeated builds, material-lot/feature/orientation context, and independently measured final-part outcomes.

## Focused commands for an executable checkout

```sh
cargo fmt --check
cargo test -p symthaea-fabrication-kernel fdm_spatial_multifidelity
cargo clippy -p symthaea-fabrication-kernel --all-targets -- -D warnings
```

Record the exact source SHA, commands and outputs. These tests can qualify deterministic software contracts, not the physical fidelity of the synthetic metrology model.

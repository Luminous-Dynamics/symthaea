# Vision decision-value reference

Standalone Rust arithmetic/reference slice for [VIS-DVPC-001](https://github.com/Luminous-Dynamics/symthaea/issues/8156).

## Run

From the repository root, with Rust 1.96+ installed:

```sh
rustc --edition=2024 --test tools/vision-decision-value-reference.rs -o /tmp/vision-dvpc-tests
/tmp/vision-dvpc-tests
```

It has no third-party dependencies and is intentionally outside Cargo workspace membership.

## Scope

- Computes retrospective DVPC as cheap-mode task loss minus escalated-mode task loss.
- Keeps the actually chosen loss separate from potential counterfactual benefit.
- Distinguishes helpful, harmful and neutral escalation using a predeclared tolerance.
- Requires one loss-profile ID, one resource-unit ID and a consistent run-budget profile before aggregation.
- Tracks per-trial and whole-run resource caps; exposes budget overruns.
- Missing charges remain unknown; they are never imputed as zero.
- Uses deterministic trial ordering and checked aggregate arithmetic.

## Claim and security boundaries

**The paired-loss input contains counterfactual outcomes. It is evaluator-only and must not be passed to the producer or online allocation policy.** The evaluator should only construct it after the allocation choice is committed.

Resource units are declared abstract units. This code does not establish trusted measurement provenance, calibrated energy, oracle isolation, causal-streaming correctness, benchmark validity or model performance. It does not replace Symtropy's evaluator or shared evidence/validity contracts.

Unit tests are arithmetic/contract fixtures, not a visual benchmark or a qualification of any model.

# Vision decision-value reference

This standalone Rust reference supports [VIS-DVPC-001](https://github.com/Luminous-Dynamics/symthaea/issues/8156).

## Run

From the repository root, with Rust 1.96+ installed:

```sh
rustc --edition=2024 --test tools/vision-decision-value-reference.rs -o /tmp/vision-dvpc-tests
/tmp/vision-dvpc-tests
```

It is intentionally outside Cargo workspace membership and has no third-party dependencies.

## Scope

- Calculates evaluator-only decision value of perception compute (DVPC): cheap-mode task loss minus escalated-mode task loss.
- Distinguishes helpful, harmful, and neutral escalation using a preregistered per-trial tolerance.
- Keeps the actually chosen task loss separate from hindsight/counterfactual potential value.
- Records missed opportunities and budget overruns instead of averaging them away.
- Does not impute missing resource charges as zero and rejects mixed resource units.
- Produces deterministic trial ordering and checked aggregate arithmetic.

## Safety and claim boundaries

The paired-loss input contains counterfactual outcomes. **It is evaluator-only and must not be exposed to the producer or used by the online allocation policy.** A caller must ensure the producer's allocation commitment happened before these outcomes became available.

Resource units are abstract, run-consistent units. This reference does not establish calibrated energy, trusted measurement provenance, oracle isolation, causal streaming, benchmark validity, or model performance. Those controls remain with Symtropy's evaluator and the related prerequisite issues. It is not a replacement for the shared evidence/validity contracts.

The unit tests are arithmetic/contract fixtures, not an end-to-end benchmark or a qualification of any visual model.

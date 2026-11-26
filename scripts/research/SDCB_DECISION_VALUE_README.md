# SDCB decision-value reference harness

Standalone Rust reference for finite-hypothesis, finite-action experiment selection.

## Compile and execute

From the repository root:

```bash
rustc --edition=2024 --test scripts/research/sdcb_decision_value.rs -o /tmp/sdcb-decision-value
/tmp/sdcb-decision-value
```

The module depends only on the Rust standard library and does not require the Symthaea Cargo workspace.

## Implemented

- Bayes-optimal current action under a declared posterior and utility matrix.
- Exact decision-equivalence partitioning by identical utility vectors.
- Expected information gain in nats.
- Expected value of sample information (EVSI).
- Net-EVSI selection with explicit cost-to-utility conversion.
- Abstention when no experiment has positive net decision value.
- Deterministic tie-breaking and malformed-input rejection.

## Counterexample

The example episode has three hypotheses, two actions and two experiments. Its independent numerical result is recorded in docs/research/SDCB_DECISION_VALUE_EPISODE_V1.expected.json. The hypothesis-split probe has higher information gain but zero EVSI because it only distinguishes hypotheses with identical action utilities. The decision-boundary probe has lower information gain but EVSI = 0.2 utility units. The source tests that the EVSI planner selects the latter.

## Limits

This is a one-step finite oracle with supplied likelihoods and utilities. It does not learn likelihoods or utilities, plan multiple sequential experiments, handle continuous design spaces, authenticate evidence, or guarantee physical safety. Utility schedules are normative task inputs and must be set by an accountable task/domain owner, not optimized by the agent itself.

15 Rust test functions are authored. Compilation and execution have not been performed in the current environment because rustc/cargo are unavailable.
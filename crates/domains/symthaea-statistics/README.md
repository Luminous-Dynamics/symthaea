# symthaea-statistics

Dependency-free statistical foundations for Symthaea. The crate is pure Rust
stdlib and does not depend on symthaea-core.

## Sequential inference

### Bernoulli SPRT

BernoulliSprt evaluates two fixed simple hypotheses for independent Bernoulli
observations. It uses Wald's likelihood-ratio boundaries from predeclared
alpha/beta error targets and refuses further observations after a terminal
decision.

```rust
use symthaea_statistics::{BernoulliSprt, SequentialDecision};

let mut test = BernoulliSprt::new(0.3, 0.7, 0.05, 0.05)?;
for outcome in [true, true, true, true] {
    let decision = test.observe(outcome)?;
    if decision != SequentialDecision::Continue {
        break;
    }
}

# Ok::<(), symthaea_statistics::SequentialError>(())
```

This is not valid for arbitrary composite hypotheses, clustered observations,
adaptive endpoints, or post-hoc selected p0/p1 values. After a boundary is
crossed, the API rejects additional observations.

### Bounded-mean confidence sequence

BoundedMeanConfidenceSequence provides a Hoeffding/union-bound time-uniform
interval for independent observations in a fixed, known finite interval.
It uses the summable two-sided schedule alpha_t = 6 alpha / (pi^2 t^2), so the
declared confidence level applies simultaneously over inspected times under
the stated sampling model.

```rust
use symthaea_statistics::BoundedMeanConfidenceSequence;

let mut sequence = BoundedMeanConfidenceSequence::new(0.0, 1.0, 0.05)?;
let first_interval = sequence.observe(0.7)?;
let later_interval = sequence.observe(0.8)?;
assert!(first_interval.lower <= 0.7);
assert!(later_interval.confidence == 0.95);

# Ok::<(), symthaea_statistics::SequentialError>(())
```

The confidence sequence rejects non-finite or out-of-support observations
without mutating its state. Its guarantee is not for arbitrary dependent
streams: repeated trials nested within a participant/session are not
independent observations. Use the actual independent unit or a
dependence-aware procedure.

## Signal detection

The signal-detection module preserves the four event-detection outcomes:

- hits: event present, response present;
- misses: event present, response absent;
- false alarms: event absent, response present;
- correct rejections: event absent, response absent.

It reports raw rates separately from loglinear-corrected equal-variance
Gaussian d-prime and response criterion c. Retain the raw counts and state the
model assumption; a response-criterion shift is not improved sensitivity.

## ROC and precision–recall

ROC curves and precision–recall curves sort scored observations once and
group exact score ties at a single threshold. ROC AUC is trapezoidal; average
precision is non-interpolated and sums recall increments times precision at
each tie group. Both reject empty, non-finite and single-class inputs.

For imbalanced event detection, report precision–recall alongside ROC: a
respectable ROC AUC does not guarantee acceptable positive predictive value.

## Claim limits

These are statistical summaries under their stated mathematical assumptions.
They do not validate the experimental assignment, independence, blinding,
measurement system or physical mechanism. Those remain separate evidence
requirements owned by the applicable experiment and measurement contracts.

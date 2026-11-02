# symthaea-statistics

Dependency-free statistical foundations for Symthaea. The crate is pure Rust
stdlib and does not depend on symthaea-core.

## Signal detection

The signal-detection module preserves the four event-detection outcomes:

- hits: event present, response present;
- misses: event present, response absent;
- false alarms: event absent, response present;
- correct rejections: event absent, response absent.

```rust
use symthaea_statistics::SignalDetectionCounts;

let counts = SignalDetectionCounts {
    hits: 42,
    misses: 8,
    false_alarms: 9,
    correct_rejections: 41,
};

let metrics = counts.metrics().expect("both event classes are represented");
println!("hit rate = {}", metrics.hit_rate);
println!("false-alarm rate = {}", metrics.false_alarm_rate);
println!("d-prime = {}", metrics.d_prime);
println!("criterion c = {}", metrics.criterion_c);
```

The metric bundle reports raw hit, miss, false-alarm and correct-rejection
rates separately from the corrected estimates used for d-prime and criterion.
A loglinear 0.5/1 correction keeps normal quantiles finite for perfect or
zero-hit/false-alarm samples. It does not alter the raw observed rates.

d-prime and criterion c use the equal-variance Gaussian signal-detection model:
d-prime estimates separation; c estimates response bias. They are not
assumption-free measures. Always retain the raw four-cell counts, state the
model assumptions, and do not interpret a behavioural score as a measurement
of an electromagnetic field or other physical mechanism.

A result is not evaluable when there are no event-present trials, no
event-absent trials, or count totals overflow. The API does not silently
invent a missing denominator.

## ROC curves

The ROC module accepts scored observations with an explicit ground-truth
signal label. It sorts scores once, groups equal scores before updating the
curve, and computes trapezoidal AUC in O(n log n) time. Larger scores mean
stronger evidence for the signal class.

```rust
use symthaea_statistics::{RocObservation, try_roc_curve};

let observations = [
    RocObservation { score: 0.9, is_signal: true },
    RocObservation { score: 0.8, is_signal: false },
    RocObservation { score: 0.7, is_signal: true },
    RocObservation { score: 0.6, is_signal: false },
];

let curve = try_roc_curve(&observations)?;
assert!((curve.auc - 0.75).abs() < 1e-12);

# Ok::<(), symthaea_statistics::RocError>(())
```

Tied scores enter at one threshold and receive half-credit in AUC via the
trapezoidal definition. Empty input, non-finite scores and single-class data
return explicit errors instead of a fabricated diagonal curve. AUC measures
ranking discrimination; it does not measure probability calibration or
choose an operating threshold.

## Precision–recall curves

For rare event detection, ROC AUC alone can obscure positive predictive value.
The precision–recall API reports precision and recall at every tie-grouped
threshold and non-interpolated average precision. It uses the same binary
scored-observation input as ROC.

```rust
use symthaea_statistics::{RocObservation, try_precision_recall_curve};

let observations = [
    RocObservation { score: 0.9, is_signal: true },
    RocObservation { score: 0.8, is_signal: false },
    RocObservation { score: 0.7, is_signal: true },
    RocObservation { score: 0.6, is_signal: false },
];

let curve = try_precision_recall_curve(&observations)?;
assert!((curve.average_precision - 0.75).abs() < 1e-12);

# Ok::<(), symthaea_statistics::RocError>(())
```

Average precision is computed as the sum of each recall increment multiplied
by precision at that operating point. It is not trapezoidal area under the
precision–recall curve. Both classes must be present; an absent class is not
silently turned into a perfect curve.

## Existing statistical layers

The crate also exposes descriptive statistics, probability distributions,
classical inference, regression, Bayesian classifier/diagnostic metrics, and
special functions. See the module documentation for the current API surface.

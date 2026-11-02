# symthaea-statistics

Dependency-free statistical foundations for Symthaea. The crate is pure Rust
stdlib and does not depend on symthaea-core.

## Signal detection

The signal-detection module preserves the four event-detection outcomes:

- hits: event present, response present;
- misses: event present, response absent;
- false alarms: event absent, response present;
- correct rejections: event absent, response absent.

Example:

```rust
use symthaea_statistics::SignalDetectionCounts;

let counts = SignalDetectionCounts {
    hits: 42,
    misses: 8,
    false_alarms: 9,
    correct_rejections: 41,
};

let metrics = counts.metrics().expect("both event classes are represented");
assert_eq!(metrics.signal_trials, 50);
assert_eq!(metrics.noise_trials, 50);
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

## Existing statistical layers

The crate also exposes descriptive statistics, probability distributions,
classical inference, regression, Bayesian classifier/diagnostic metrics, and
special functions. See the module documentation for the current API surface.

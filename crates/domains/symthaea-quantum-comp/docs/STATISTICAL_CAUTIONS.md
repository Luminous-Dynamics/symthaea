# Statistical Cautions

This crate intentionally ships only small dependency-free statistics helpers.

They are suitable for local sanity checks, regression tests, and early research notes. They are not a substitute for a full statistical package or peer-reviewed analysis workflow.

## Alpha.6 helpers

The `significance` module adds:

- paired difference summaries
- win/loss/tie counts
- exact two-sided sign-test p-values, ignoring ties

The sign test is useful because it is simple and robust, but it throws away effect magnitude. Always report it alongside mean differences, confidence-style intervals, and raw experiment settings.

## Avoid these claims

Do not claim:

- quantum advantage
- physical entanglement execution
- quantum consciousness
- hardware backend validation
- publication-grade inference from default examples

## Better language

Use:

- local replicated probe
- simulation result
- quantum-inspired baseline
- entanglement proxy
- negative control
- claim boundary

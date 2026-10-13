# Interval Arithmetic in Symthaea

## Purpose

The interval arithmetic module provides a conservative numerical enclosure for
reasoning about ranges of values. An Interval represents a closed real range
[lower, upper]; the implementation stores binary64 endpoints and widens basic
arithmetic results by one representable floating-point step in each outward
direction.

The intent is to let symbolic math answer a stronger question than “what number
did floating-point evaluation return?” It can instead return a range that
contains the mathematical result, subject to the arithmetic assumptions below.

## Current API

The module is exposed as
crate::hdc::interval_arithmetic::{Interval, IntervalError}.

- Construction rejects NaN bounds, reversed bounds, and non-finite point values.
  Unbounded endpoints (-infinity as the lower bound and +infinity as the upper
  bound) are permitted for conservative results.
- Addition, subtraction, multiplication, and division use outward-expanded
  endpoint calculations.
- Division fails with DivisionByIntervalContainingZero when the divisor might
  be zero. Indeterminate endpoint quotients widen to the whole real line.
- square, absolute, and checked_powi provide conservative range operations.
  The integer-power path is logarithmic in the exponent and rejects zero-to-zero
  and negative powers over intervals containing zero.
- Expr::eval_interval evaluates symbolic arithmetic and finite sums over
  interval-valued variable bindings. It supports integer powers, absolute value,
  floor, and finite sums with the existing maximum bound of 100.

Example:

\[
x \in [-2,3] \implies x^2 \in [0,9]
\]

The dedicated square / square-power path preserves the nonnegative lower bound
even when the input interval crosses zero. General interval arithmetic can be
wider than the true range because repeated occurrences of a variable are
conservatively treated as ranges, not as a single correlated value.

## Soundness boundary

The outward-rounding guarantee assumes normal IEEE-754 binary64 round-to-nearest
basic arithmetic and compilation without unsafe fast-math reassociation. It is
not a machine-checked theorem about every target compiler or processor.

This implementation intentionally does **not** claim interval bounds for
transcendental functions such as sin, cos, exp, ln, or sqrt. Their platform math
library accuracy is not enough to assume that widening the result by one ULP is
always sound without a separately reviewed guarantee. The interval expression
evaluator returns UnsupportedFunction for those operations until a proper
enclosure method is added.

A wider interval is conservative, not proof of a desired property. For example,
an interval whose lower bound is negative does not prove the true result is
negative; it may simply be too wide. Conversely, sign checks are meaningful only
when the **entire** returned interval is on one side of zero.

## Qualification and tests

Unit tests cover invalid endpoints, outward enclosures for each arithmetic
operator, division-by-zero ranges, overflow, underflow, unbounded indeterminate
quotients, integer-power domains, and a deterministic grid of small integer
intervals. Expression tests cover polynomial enclosure, finite sums, integer
exponent requirements, division-domain rejection, and fail-closed handling of
unsupported transcendental operations.

These tests are regression evidence, not a proof of the implementation for every
possible binary64 input. The repository CI result for the exact current PR head
must be observed before treating the change as qualified.

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
- square, checked_sqrt, checked_ln, absolute, and checked_powi provide
  conservative range operations. Square root rejects any interval containing
  negative reals and expands each nonzero endpoint outward by one representable
  value. Natural logarithm rejects intervals containing nonpositive reals and
  uses binary range reduction plus an atanh series with a bounded remainder.
  The integer-power path is logarithmic in the exponent and rejects zero-to-zero
  and negative powers over intervals containing zero.
- Expr::eval_interval evaluates symbolic arithmetic and finite sums over
  interval-valued variable bindings. It supports integer powers, square root,
  natural logarithm, absolute value, floor, and finite sums with the existing
  maximum bound of 100.

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

Square root is supported using Rust's documented IEEE-754 correctly rounded
`f64::sqrt` result, then stepping one representable value outward for each
nonzero endpoint. See the [Rust `f64::sqrt` contract](https://doc.rust-lang.org/std/primitive.f64.html#method.sqrt).
Zero is retained exactly, and any interval including negative reals is rejected.

Natural logarithm does not use the platform `f64::ln` result to form a bound.
Each positive finite binary64 endpoint is decomposed exactly as (x=m2^k),
where (1\le m<2), using its stored significand and exponent (including exact
normalization of subnormals). It then encloses
[
\ln(m)=2\sum_{j=0}^{\infty}\frac{z^{2j+1}}{2j+1},\qquad
z=\frac{m-1}{m+1},\quad 0\le z<\frac13.
]
The first 64 terms are evaluated using interval operations. Since the omitted
tail is bounded by
[
\frac{2z^{129}}{129(1-z^2)}
\le \frac{9}{516\cdot3^{129}} < 2^{-134},
]
the upper bound includes an exact (2^{-134}) remainder allowance. The same
series encloses \(\ln 2\), then \(\ln x=\ln m+k\ln2\) is evaluated with
outward-rounded interval arithmetic. The result remains conditional on the
basic-operation assumptions above; the source has not been machine-checked.

The evaluator still does **not** claim interval bounds for sin, cos, or exp.
Rust documents their precision as unspecified and potentially platform-dependent,
so they remain `UnsupportedFunction` until separately reviewed enclosure
algorithms are added.

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

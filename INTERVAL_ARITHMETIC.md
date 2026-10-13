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
- square, checked_sqrt, checked_ln, checked_exp, checked_sin, checked_cos,
  absolute, and checked_powi provide conservative range operations. Square root
  rejects intervals containing negative reals and expands nonzero endpoints
  outward. Natural logarithm and exponential use interval series, range reduction,
  and explicit remainder bounds. Sine/cosine use Taylor series only when the full
  interval lies within [-2, 2]; elsewhere they return the always-sound but
  coarse enclosure [-1, 1] until a validated argument-reduction path exists. The
  integer-power path is logarithmic in the exponent and rejects zero-to-zero and
  negative powers over intervals containing zero.
- Expr::eval_interval evaluates symbolic arithmetic and finite sums over
  interval-valued variable bindings. It supports integer powers, square root,
  natural logarithm, exponential, sine/cosine, absolute value, floor, and finite
  sums with the existing maximum bound of 100.

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
nonzero endpoint. See the [Rust `f64::sqrt` contract](
https://doc.rust-lang.org/std/primitive.f64.html#method.sqrt).
Zero is retained exactly, and any interval including negative reals is rejected.

Natural logarithm does not use the platform `f64::ln` result to form a bound.
Each positive finite binary64 endpoint is decomposed exactly as `x = m * 2^k`,
where `1 <= m < 2`, using its stored significand and exponent (including exact
normalization of subnormals). It then encloses the series

`ln(m) = 2 * sum(z^(2j+1) / (2j+1))`, where `z = (m - 1) / (m + 1)` and
`0 <= z <= 1/3`. The upper endpoint is attained when the routine evaluates `ln(2)` itself.

The first 64 terms are evaluated using interval operations. Since the omitted
tail is bounded by

`2*z^129 / (129*(1-z^2)) <= 9 / (516*3^129) < 2^-134`,

the upper bound includes an exact `2^-134` remainder allowance. The same
series encloses `ln(2)`, then `ln(x) = ln(m) + k*ln(2)` is evaluated with
outward-rounded interval arithmetic. The result remains conditional on the
basic-operation assumptions above; the source has not been machine-checked.

Exponential does not use platform `f64::exp` as a bound. It reduces each
finite endpoint using an enclosed `ln(2)`, checks that the reduced interval lies
within `[-0.36, 0.36]`, and evaluates the first 40 Taylor terms using interval
operations. The Lagrange remainder is bounded by
`exp(0.36) * 0.36^40 / 40!`. Since `exp(0.36) < 2`,
`0.36 < 3/8), `(3/8)^40 < 2^-56), and `40! > 2^150`, this remainder is less
than `2^-205`; the implementation safely allows `2^-200`. Scaling then uses
an exactly represented normal or subnormal power of two. For inputs beyond the
well-resolved reduction range it returns broad but conservative enclosures
instead of relying on the platform exponential.

Sine and cosine now have a deliberately bounded contract. For a complete input
interval inside [-2, 2], the implementation evaluates sine endpoint series
through degree 39 and cosine endpoint series through degree 38, using interval
arithmetic at every step. On this domain, the first omitted alternating terms
are bounded by `2^41 / 41! < 2^-123` for sine and `2^40 / 40! < 2^-119` for
cosine; the implementation adds the wider, exactly represented `2^-80`
symmetric allowance. The sine range uses the endpoint hull and includes -1 or +1 only when the input
overlaps a cached enclosure of +/-pi/2. That enclosure is computed from Machin's
identity `pi/2 = 8 atan(1/5) - 2 atan(1/239)`, using 25 and 8 alternating-series
terms with symmetric `2^-120` remainder allowances; if construction fails, sine
returns `[-1, 1]`. Cosine uses its endpoint hull and the exact maximum 1 when the
input contains zero. For intervals extending outside [-2, 2], a checked quadrant-reduction path is attempted. It encloses the quotient by pi/2 and uses its midpoint only to choose a candidate integer `k`; it does not require a unique nearest-integer cell because the shift identities are exact for every integer `k`. It computes an interval for `r = x - k*pi/2` and accepts that choice only if the complete reduced interval remains inside [-2, 2]. Sine/cosine are mapped through the corresponding quadrant identity using local series enclosures. The quadrant index is capped at 1,000,000,000,000; non-finite, excessive, or otherwise unusable reductions return `[-1, 1]`. This extends tight enclosures to many moderate and large finite values without claiming all-binary64 coverage. No platform `f64::sin` or `f64::cos` result is used to form an enclosure.

A wider interval is conservative, not proof of a desired property. For example,
an interval whose lower bound is negative does not prove the true result is
negative; it may simply be too wide. Conversely, sign checks are meaningful only
when the **entire** returned interval is on one side of zero.

## Qualification and tests

Unit tests cover invalid endpoints, outward enclosures for each arithmetic
operator, division-by-zero ranges, overflow, underflow, unbounded indeterminate
quotients, square-root/logarithm domains, logarithm enclosure over subnormal,
normal, near-one and extreme finite inputs, exponential enclosure from subnormal
underflow through large finite inputs, trigonometric series across [-2, 2],
quadrant reduction at moderate and large finite magnitudes, acceptance of bounded reductions across
half-integer quotient-cell boundaries, an adjacent-binary64 pi/2 bracket check,
and fixed high-precision reference brackets
through magnitude 1e12, broad fallback for ambiguous/extreme inputs, and a
deterministic interval-grid probe across sine critical-point bands and cosine's
maximum. Integer
power domains and a deterministic grid of small integer intervals remain covered. Expression tests
cover polynomial enclosure, square-root/logarithm/exponential/sine/cosine
enclosures, finite sums, integer exponent requirements, and division-domain
rejection.

These tests are regression evidence, not a proof of the implementation for every
possible binary64 input. A separate deterministic Python translation cross-check
compared 304 logarithm inputs and 513 exponential inputs against 150-digit
`Decimal` reference results and observed no enclosure failures. That validates
the translated algorithm on those samples only; it does not execute this Rust
source, prove the implementation, or replace exact-head CI. The repository CI
result for the exact current PR head must be observed before treating the change
as qualified.

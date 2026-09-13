# HLS continuous temporal phasor algebra

This note defines the first explicit temporal-addressing substrate for the Holographic Liquid State (HLS) research line.

## Scope

This tranche establishes a **continuous unitary time algebra** and an analytic representation of temporal validity intervals. It does **not** claim that HLS can already answer historical state-tracking queries.

The distinction is deliberate. The state-tracking benchmark asks for the value that was valid after an arbitrary earlier event, not merely the event nearest a requested timestamp. A random lag role or point-time association is therefore insufficient evidence of historical memory.

## Prior-art boundary

Fractional binding, Fourier Holographic Reduced Representations, and Spatial Semantic Pointers already establish the general technique of representing continuous coordinates with unitary Fourier-domain phases and fractional powers. Prior work also represents trajectories and continuous regions by superposing or integrating semantic pointers over their coordinates.

This tranche does **not** claim to invent fractional binding, SSPs, continuous VSA coordinates, or Fourier-domain phasor representations.

The research question for Symthaea is narrower: can an explicit continuous temporal algebra compose cleanly with HLS's bipolar HDC roles, continuous-time recurrence, exact online learning, and eventually evidence-bearing state revision while retaining falsifiable algebraic guarantees?

## Temporal point algebra

For deterministic angular frequencies `omega_k`, define

`T(t)_k = exp(i * omega_k * t)`.

Then, up to floating-point roundoff,

`T(a) * T(b) = T(a + b)`

and

`T(t)^-1 = T(-t)`.

Every coordinate remains unit modulus. Temporal translation is therefore a group action rather than an arbitrary embedding lookup.

The implementation deliberately stores this substrate directly in the Fourier/phasor domain. A later real-space FFT/SSP representation may be useful, but it is not required to qualify the underlying algebra.

## Compatibility with bipolar HDC roles

A real bipolar role contributes either phase `0` (`+1`) or phase `pi` (`-1`) in each channel. Consequently role binding commutes with temporal phasor binding:

`R * (T(a) * T(b)) = (R * T(a)) * T(b)`.

This is important for HLS because relation/identity roles can remain algebraically distinct from continuous time without introducing a learned cross-basis transform merely to represent timestamps.

## Validity intervals

Historical state is valid over a span, not only at a mutation instant. For a half-open interval `[a,b)`, define

`I[a,b) = integral_a^b T(tau) d tau`.

Per nonzero frequency,

`I[a,b)_k = (exp(i*omega_k*b) - exp(i*omega_k*a)) / (i*omega_k)`.

The zero-frequency limit is exactly `b-a`.

The interval representation obeys the same translation action:

`T(delta) * I[a,b) = I[a+delta,b+delta)`.

The implementation evaluates this integral analytically and checks it against dense midpoint quadrature.

## What is mechanically tested

The current contract tests require:

- deterministic axis generation;
- continuous group composition;
- inverse/identity recovery;
- unit-modulus phasors;
- translation-invariant point similarity;
- commutation between temporal binding and bipolar role binding;
- analytic validity-interval translation;
- agreement between the analytic interval integral and numerical quadrature;
- fail-closed malformed time/interval inputs.

Numerical tolerances are intentionally looser than machine epsilon while still being several orders of magnitude tighter than any downstream retrieval threshold. This avoids turning platform-specific libm rounding into a false theorem failure.

## What is explicitly not established

This tranche does not establish:

- historical benchmark accuracy;
- interval cleanup/retrieval capacity;
- a one-sided predecessor theorem (`latest value valid at or before t`);
- open-ended current intervals;
- resistance to crosstalk from many relations/intervals;
- integration with HLS recurrence or eligibility traces;
- superiority to attention, SSMs, external memory, or other temporal models.

Those claims require separate experiments.

## Next experiment

The next tranche should build the smallest standalone **validity-interval associative memory** on top of this algebra before modifying HLS itself.

Start with one relation key and a sequence of non-overlapping value intervals. Then increase:

1. number of values/intervals;
2. number of independent relation keys;
3. irregular interval lengths;
4. temporal horizon;
5. value-codebook size;
6. HDC dimension.

Measure exact cleanup accuracy, margin between the correct and nearest incorrect value, boundary behavior, future-update interference, and false-positive/crosstalk rates.

Only if that substrate can recover `value valid at time t` reliably should it be attached to historical state tracking and given a separately frozen experiment plan. A failure here is useful evidence and should redirect the temporal-memory design before it contaminates the current-state HLS result.
